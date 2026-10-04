# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler workarounds scoped exclusively to the M192 attention JIT.

Each compile owns a temporary directory. The installed backend/compile functions
are never replaced. The workaround and compiler sources enter the cache key.
"""
import hashlib
import inspect
import json
import os
import re
import subprocess
import tempfile
import textwrap
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path
from types import FunctionType

import triton.backends.mthreads.compiler as compiler
import triton.compiler.compiler as compile_module
from triton._C.libtriton import ir
from triton.runtime.jit import create_function_from_signature

_compile_root = ContextVar("mthreads_triplet_compile_root")
_OPT = Path(os.environ.get("MUSA_HOME", "/usr/local/musa")) / "bin" / "opt"


def build_phase_counter(original):
    enabled = True

    def make_llir(src, metadata, options, arch):
        text = original(src, metadata, options, arch)
        if "@attention(" not in text:
            return text
        root = _compile_root.get()
        tag = f'counter-{metadata.get("num_warps")}-{len(text)}'
        (root / f"{tag}-before.llir").write_text(text)
        pattern = r"(?m)^(\s*)(%[0-9]+) = (?:tail )?call i32 @llvm.musa.async.arrive\(i32 (\d+)\)(,.*)?$"
        matches = list(re.finditer(pattern, text))
        ids = sorted({int(m[3]) for m in matches})
        assert ids and matches
        changes = []
        if enabled:
            for bid in ids:
                assert re.search(
                    rf"async.init.arrival\(i32 {bid}, i32 \d+, i32 0\)", text
                )
            allocas = "".join(
                f"  %ws_phase_{bid} = alloca i32, align 4\n"
                f"  store i32 0, ptr %ws_phase_{bid}, align 4\n"
                for bid in ids
            )
            definition = re.search(
                r"(?m)^define mtgpu_kernel void @attention\(.*\{\n", text
            )
            assert definition is not None
            text = text[: definition.end()] + allocas + text[definition.end() :]

            def replace_arrive(m):
                pad, value, bid, suffix = m.groups()
                suffix = suffix or ""
                changes.append(dict(ssa=value, barrier=int(bid)))
                return (
                    f"{pad}{value} = load i32, ptr %ws_phase_{bid}, align 4{suffix}\n"
                    f"{pad}call void @llvm.musa.async.arrive.none.phaseid(i32 {bid}){suffix}"
                )

            text = re.sub(pattern, replace_arrive, text)
            for change in changes:
                value, bid = change["ssa"], change["barrier"]
                wait = (
                    rf"(?m)^(\s*)(?:tail )?call void @llvm.musa.async.wait\(i32 {bid}, "
                    rf"i32 {re.escape(value)}\)(,.*)?$"
                )

                def replace_wait(m):
                    pad, suffix = m.groups()
                    suffix = suffix or ""
                    return (
                        f"{pad}call void @llvm.musa.async.wait(i32 {bid}, i32 {value}){suffix}\n"
                        f"{pad}%ws_next_{value[1:]} = xor i32 {value}, 1\n"
                        f"{pad}store i32 %ws_next_{value[1:]}, ptr %ws_phase_{bid}, align 4"
                    )

                text, count = re.subn(wait, replace_wait, text)
                assert count == 1, (change, count)
            assert len(changes) == len(matches)
        (root / f"{tag}-rewritten.llir").write_text(text)
        # The installed libtriton has no text-to-LLVM parser. Use the existing
        # MUSA LLVM 14 compatibility conversion and promote only these stack
        # counters with mem2reg; avoid a second general O3 optimization pass.
        if compiler._detect_llvm_major_version(str(_OPT)) != 14:
            raise NotImplementedError(
                "M192 prefill requires the validated MUSA LLVM 14 opt"
            )
        compatible = compiler._drop_unsupported_attrs(text, 14)
        compatible = compiler._llvm_compat(compatible)
        compatible = compiler._rewrite_llvm_scmp_ucmp_to_icmp(compatible)
        infile, outfile = root / f"{tag}-compatible.ll", root / f"{tag}-mem2reg.ll"
        infile.write_text(compatible)
        command = [
            str(_OPT),
            "--opaque-pointers",
            "-mem2reg",
            "-S",
            str(infile),
            "-o",
            str(outfile),
        ]
        operation = subprocess.run(command, text=True, capture_output=True, timeout=60)
        (root / f"{tag}-opt-stderr.txt").write_text(operation.stderr)
        assert operation.returncode == 0, operation.stderr
        result = outfile.read_text()
        if enabled:
            assert "alloca i32" not in result, "counter was not promoted to registers"
            assert not re.search(r"call i32 @llvm.musa.async.arrive\(i32", result)
        (root / f"{tag}-after.llir").write_text(result)
        (root / f"{tag}-changes.json").write_text(
            json.dumps(
                dict(
                    enabled=enabled,
                    hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    barrier_ids=ids,
                    changes=changes,
                ),
                indent=2,
            )
        )
        return result

    return make_llir


serial = 1


def after_allocate(mod):
    text = str(mod)
    if "ttg.warp_specialize(" not in text:
        return mod
    root = _compile_root.get()
    (root / f"static-{serial}-before.ttgir").write_text(text)
    # Each consumer owns one explicit shared-memory P-to-PV rendezvous.
    assert text.count("ttg.barrier all ") == 3, text.count("ttg.barrier all ")
    text = text.replace("ttg.barrier all ", "ttg.barrier local ")
    sizes = dict(ks=32768, vs=32768)
    dest = dict(ks=0, vs=32768)
    for i in range(3):
        sizes.update({f"q{i}": 32768, f"k{i}": 32768, f"v{i}": 32768, f"p{i}": 8192})
        dest.update(
            {
                f"q{i}": 65536 + i * 32768,
                f"k{i}": 0,
                f"v{i}": 32768,
                f"p{i}": 163840 + i * 8192,
            }
        )
    original, layouts = {}, {}
    for name in sizes:
        matches = re.findall(
            rf"^\s*%{name} = ttg.local_alloc .*allocation.offset = (\d+) : i32.* -> !ttg.memdesc<([^>]+)>.*$",
            text,
            re.M,
        )
        assert len(matches) == 1, (name, matches)
        original[name] = int(matches[0][0])
        layouts[name] = matches[0][1]
    for i in range(3):
        assert layouts["ks"] == layouts[f"k{i}"], layouts
        assert layouts["vs"] == layouts[f"v{i}"], layouts
        assert layouts[f"q{i}"].startswith("64x256xbf16,")
        assert layouts[f"p{i}"].startswith("64x64xbf16,")
    assert layouts["ks"].startswith("1x256x64xbf16,")
    assert layouts["vs"].startswith("1x64x256xbf16,")
    spans = sorted((base, base + sizes[name], name) for name, base in original.items())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans
    old_end = max(end for _, end, _ in spans)
    before = int(re.findall(r"ttg.shared = (\d+) : i32", text)[0])
    assert before >= old_end
    mapping = {}
    for offset in set(map(int, re.findall(r"allocation.offset = (\d+) : i32", text))):
        if offset >= old_end:
            mapping[offset] = 188416 + offset - old_end
        else:
            matches = [(lo, name) for lo, hi, name in spans if lo <= offset < hi]
            assert len(matches) == 1, (offset, spans)
            lo, name = matches[0]
            mapping[offset] = dest[name] + offset - lo
    after = 188416 + before - old_end
    text = re.sub(
        r"allocation.offset = (\d+) : i32",
        lambda m: f"allocation.offset = {mapping[int(m[1])]} : i32",
        text,
    )
    text = text.replace(f"ttg.shared = {before} : i32", f"ttg.shared = {after} : i32")
    path = root / f"static-{serial}-mapped.ttgir"
    path.write_text(text)
    (root / f"static-{serial}-mapping.json").write_text(
        json.dumps(
            dict(
                before=original,
                dest=dest,
                layouts=layouts,
                mapping=mapping,
                shared_before=before,
                shared_after=after,
            ),
            indent=2,
        )
    )
    replacement = ir.parse_mlir_module(str(path), mod.context)
    replacement.context = mod.context
    return replacement


source = textwrap.dedent(inspect.getsource(compiler.MUSABackend.make_llir)).replace(
    "@staticmethod\n", ""
)
needle = "    mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, capability)"
assert source.count(needle) == 1
source = source.replace(
    needle,
    """    pm.run(mod, 'ws_static_allocate')
    mod = after_allocate(mod)
    src = mod
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, capability)""",
)
scope = dict(compiler.__dict__)
scope["after_allocate"] = after_allocate
exec(source, scope)
allocate_llir = scope["make_llir"]


def make_llir(src, metadata, options, arch):
    text = allocate_llir(src, metadata, options, arch)
    if "@attention(" not in text:
        return text
    root = _compile_root.get()
    (root / f"init-{serial}-before.llir").write_text(text)
    tid = re.search(r"(%[0-9]+) = .*call i32 @llvm.musa.read.ptx.sreg.tid.x", text)[1]
    masks = re.findall(
        r"(%[0-9]+) = and i32 " + re.escape(tid) + r", (?:127|255|511|1023),", text
    )
    changes = 0
    for mask in masks:
        pattern = r"(%[0-9]+ = icmp eq i32 )" + re.escape(mask) + r"(, 0,)"
        text, count = re.subn(pattern, lambda m: m[1] + tid + m[2], text)
        changes += count
    assert changes == 1, (changes, masks)
    text, exits = re.subn(
        r"(?m)^  ret void,",
        "  call void @llvm.musa.syncthreads.lm()\n  ret void,",
        text,
    )
    assert exits == 1
    (root / f"init-{serial}-after.llir").write_text(text)
    return text


_make_with_counter = build_phase_counter(make_llir)


def _make_scoped_llir(src, metadata, options, arch):
    with tempfile.TemporaryDirectory(prefix="mthreads-triplet-") as folder:
        token = _compile_root.set(Path(folder))
        try:
            return _make_with_counter(src, metadata, options, arch)
        finally:
            _compile_root.reset(token)


def bind_kernel(kernel):
    make_llir = _make_scoped_llir
    source_paths = [compiler.__file__, compile_module.__file__]
    original_llir = compiler.MUSABackend.make_llir
    original_compile = compile_module.compile
    original_binder = kernel.create_binder
    assert not kernel.device_caches, "attach only to a fresh JIT instance"
    digest = hashlib.sha256()
    for path in [__file__, *source_paths]:
        digest.update(Path(path).read_bytes())
    digest.update(original_compile.__code__.co_code)
    digest.update(str(compiler._tool_version_signature(str(_OPT))).encode())
    fingerprint = digest.hexdigest()

    class TripletBackend(compiler.MUSABackend):
        def hash(self):
            return super().hash() + "-triplet-scoped-" + fingerprint

    TripletBackend.make_llir = staticmethod(make_llir)
    scope = dict(original_compile.__globals__)
    scope["make_backend"] = lambda target: TripletBackend(target)
    scoped_compile = FunctionType(
        original_compile.__code__,
        scope,
        "compile_triplet_scoped",
        original_compile.__defaults__,
        original_compile.__closure__,
    )
    scoped_compile.__kwdefaults__ = original_compile.__kwdefaults__

    def compile_checked(*args, **kw):
        assert compiler.MUSABackend.make_llir is original_llir
        assert compile_module.compile is original_compile
        result = scoped_compile(*args, **kw)
        assert compiler.MUSABackend.make_llir is original_llir
        assert compile_module.compile is original_compile
        return result

    def create_binder():
        cache, keys, target, _, _ = original_binder()
        backend = TripletBackend(target)
        kernel.compile = compile_checked
        binder = create_function_from_signature(
            kernel.signature, kernel.params, backend
        )
        return cache, keys, target, backend, binder

    kernel.device_caches = defaultdict(create_binder)
