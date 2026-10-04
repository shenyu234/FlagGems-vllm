# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent paired short prefill: independent numerics and mutable Graph."""
import math

import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case, reference

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


def short_case(qlen=1035, klen=1025, maxq=None, heads=(16, 2), causal=True):
    args = make_case(torch.bfloat16, True, [qlen], [1088], heads=heads, dim=256)
    args.update(
        causal=causal, max_seqlen_q=maxq or qlen, out=torch.empty_like(args["q"])
    )
    args["seqused_k"].fill_(klen)
    return args


def observe(monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import (
        short_pair_prefill as impl,
    )

    monkeypatch.delenv("MTHREADS_FA_DISABLE_WS_PREFILL", raising=False)
    original_launch = impl.launch_short_pair
    calls, shifts = [], []

    def launch(*args, **kwargs):
        kernel = original_launch(*args, **kwargs)
        calls.append(kernel)
        return kernel

    def guard(original, is_classify):
        def run(*args, **kwargs):
            # Every specialization passes the resource gate before launch.
            kernel = original(*args, **dict(kwargs, warmup=True))
            kernel._init_handles()
            assert kernel.n_spills < 40
            if is_classify:
                shifts.append(args[2])
            return original(*args, **kwargs)

        return run

    monkeypatch.setattr(impl, "launch_short_pair", launch)
    for kernel in (
        impl.gather_norm_kernel,
        impl.classify192,
        impl.attention,
        impl.online_fallback,
    ):
        monkeypatch.setattr(
            kernel, "run", guard(kernel.run, kernel is impl.classify192)
        )
    return calls, shifts


def check(output, args, length):
    expected, expected_lse = reference(args, [args["q"].shape[0]], [length])
    assert output[0] is args["out"]
    torch.testing.assert_close(output[0].cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(output[1].cpu(), expected_lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("qlen", [769, 960, 1035, 1345, 2047])
@pytest.mark.parametrize("mode", ["fast", "online", "mixed_first"])
def test_v10_odd_even_groups(monkeypatch, qlen, mode):
    import triton.backends.mthreads.compiler as compiler

    before = compiler.MUSABackend.make_llir
    calls, shifts = observe(monkeypatch)
    args = short_case(qlen)
    if mode == "online":
        args["q"].mul_(8)
    elif mode == "mixed_first":
        args["q"][:192].mul_(8)
    output = flaggems_vllm.flash_attn_varlen_func(**args)
    check(output, args, 1025)
    assert calls and shifts
    assert compiler.MUSABackend.make_llir is before
    fast = torch.isfinite(shifts[-1][: math.ceil(qlen / 192) * 3]).cpu()
    if mode == "fast":
        assert fast.all()
    elif mode == "online":
        assert not fast.any()
    else:
        assert fast.any() and not fast.all()


@pytest.mark.parametrize("klen", [0, 17, 1025])
def test_v10_conservative_maxq(monkeypatch, klen):
    calls, _ = observe(monkeypatch)
    args = short_case(577, klen, maxq=1035)
    output = flaggems_vllm.flash_attn_varlen_func(**args)
    check(output, args, klen)
    assert calls


def test_v10_mutable_graph(monkeypatch):
    calls, shifts = observe(monkeypatch)
    args = short_case()
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(2):
            flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        output = flaggems_vllm.flash_attn_varlen_func(**args)
    assert calls and shifts
    for mode, length in [
        ("online", 1025),
        ("first", 1025),
        ("last", 1025),
        ("pages", 1023),
        ("empty", 0),
        ("zero", 1025),
    ]:
        if mode == "online":
            args["q"].mul_(8)
        elif mode == "first":
            args["q"].div_(8)
            args["q"][:192].mul_(8)
        elif mode == "last":
            args["q"][:192].div_(8)
            args["q"][-96:].mul_(8)
        elif mode == "pages":
            args["k"].neg_()
            args["v"].mul_(-0.75)
            args["block_table"].copy_(args["block_table"].flip(1))
        elif mode == "zero":
            args["q"].zero_()
        args["seqused_k"].fill_(length)
        for _ in range(3):
            graph.replay()
            torch.musa.synchronize()
            check(output, args, length)
        fast = torch.isfinite(shifts[-1]).cpu()
        if mode == "online":
            assert not fast.any()
        elif mode in ("first", "last"):
            assert fast.any() and not fast.all()


@pytest.mark.parametrize("kind", ["disabled", "tp4", "noncausal"])
def test_v10_original_routes(monkeypatch, kind):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import (
        short_pair_prefill as impl,
    )

    args = short_case(
        heads=(4, 1) if kind == "tp4" else (16, 2), causal=kind != "noncausal"
    )
    if kind == "disabled":
        monkeypatch.setenv("MTHREADS_FA_DISABLE_WS_PREFILL", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("short pair route was selected outside its region")

    monkeypatch.setattr(impl, "launch_short_pair", forbidden)
    output = flaggems_vllm.flash_attn_varlen_func(**args)
    check(output, args, 1025)
