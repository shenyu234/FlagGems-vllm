# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent O/LSE and mutable-Graph coverage for the M192 cached route."""
import math

import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case, reference

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


def routed_case(causal):
    args = make_case(torch.bfloat16, True, [577], [16384], heads=(4, 1), dim=256)
    # A legal conservative Q maximum exercises inactive CTAs and the 192/64
    # tail, while the runtime K length is changed independently of capacity.
    args.update(causal=causal, max_seqlen_q=4096, out=torch.empty_like(args["q"]))
    args["seqused_k"].fill_(1025)
    return args


def observe(monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import (
        triplet_prefill,
    )

    monkeypatch.delenv("MTHREADS_FA_DISABLE_WS_PREFILL", raising=False)
    original_launch = triplet_prefill.launch_triplet
    original_classify = triplet_prefill.classify192.run
    calls, classifications = [], []

    def launch(*args, **kwargs):
        kernel = original_launch(*args, **kwargs)
        calls.append(kernel)
        assert kernel.n_spills < 40
        return kernel

    def classify(*args, **kwargs):
        classifications.append(args[2])
        return original_classify(*args, **kwargs)

    monkeypatch.setattr(triplet_prefill, "launch_triplet", launch)
    monkeypatch.setattr(triplet_prefill.classify192, "run", classify)
    return calls, classifications


def check(actual, lse, args, length):
    expected, expected_lse = reference(args, [577], [length])
    assert actual is args["out"]
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mode", ["fast", "online", "mixed", "empty_kv", "short_kv"])
def test_v9_tail_and_online(monkeypatch, causal, mode):
    import triton.backends.mthreads.compiler as compiler

    before = compiler.MUSABackend.make_llir
    calls, classifications = observe(monkeypatch)
    args = routed_case(causal)
    length = 1025
    if mode == "online":
        args["q"].mul_(8)
    elif mode == "mixed":
        args["q"][:192].mul_(8)
    elif mode in ("empty_kv", "short_kv"):
        length = 0 if mode == "empty_kv" else 17
        args["seqused_k"].fill_(length)
    actual, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    check(actual, lse, args, length)
    assert calls and classifications
    assert compiler.MUSABackend.make_llir is before
    active = classifications[-1][: math.ceil(args["q"].shape[0] / 192) * 3]
    fast = torch.isfinite(active).cpu()
    if mode == "fast":
        assert fast.all()
    elif mode == "online":
        assert not fast.any()
    elif mode == "mixed":
        assert fast.any() and not fast.all()


def test_v9_mutable_graph(monkeypatch):
    calls, classifications = observe(monkeypatch)
    args = routed_case(True)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(2):
            flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        actual, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    assert calls and classifications
    for mode, length in [
        ("online", 1025),
        ("mixed", 1025),
        ("pages", 1023),
        ("empty", 0),
    ]:
        if mode == "online":
            args["q"].mul_(8)
        elif mode == "mixed":
            args["q"].div_(8)
            args["q"][:192].mul_(8)
        elif mode == "pages":
            args["k"].neg_()
            args["v"].mul_(-0.75)
            args["block_table"].copy_(args["block_table"].flip(1))
        args["seqused_k"].fill_(length)
        expected, expected_lse = reference(args, [577], [length])
        for _ in range(3):
            graph.replay()
            torch.musa.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


def test_v9_disable_preserves_original_route(monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import (
        triplet_prefill,
    )

    args = routed_case(True)
    monkeypatch.setenv("MTHREADS_FA_DISABLE_WS_PREFILL", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled M192 route was selected")

    monkeypatch.setattr(triplet_prefill, "launch_triplet", forbidden)
    actual, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    check(actual, lse, args, 1025)
