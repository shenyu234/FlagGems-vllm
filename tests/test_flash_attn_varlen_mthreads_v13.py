# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-KV decode: independent O/LSE, metadata replay and nonfinite scores."""
import itertools

import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case
from tests.test_flash_attn_varlen_mthreads_full import compile_resources
from tests.test_flash_attn_varlen_mthreads_v7 import check

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


@pytest.mark.parametrize("heads", [(4, 1), (16, 2)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.0625])
def test_singlekv_mutable_graph(heads, causal, scale, monkeypatch):
    qlens, klens = [1, 0, 1, 1], [1, 1, 0, 1]
    args = make_case(torch.bfloat16, True, qlens, klens, heads=heads, dim=256)
    # Exercise independent outer strides for the KV tensors and the page table.
    args["k"] = torch.stack([args["k"], args["k"]], dim=1)[:, 1]
    args["v"] = torch.stack([args["v"]] * 3, dim=1)[:, 2]
    args["block_table"] = torch.cat([args["block_table"]] * 2, dim=1)[:, :1]
    args.update(causal=causal, softmax_scale=scale, out=torch.empty_like(args["q"]))
    resources = compile_resources([], args, monkeypatch)
    assert len(resources) == 1 and resources[0]["name"] == "single_kv_kernel"
    assert resources[0]["spills"] < 40
    check(args, qlens, klens)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        flaggems_vllm.flash_attn_varlen_func(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        got = flaggems_vllm.flash_attn_varlen_func(**args)
    for qnew, knew in [(qlens, klens), ([0, 1, 1, 1], [0, 1, 1, 0])]:
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, *itertools.accumulate(qnew)], dtype=torch.int32, device="musa"
            )
        )
        args["seqused_k"].copy_(torch.tensor(knew, dtype=torch.int32, device="musa"))
        args["block_table"].copy_(args["block_table"].flip(0))
        for _ in range(2):
            got[0].fill_(float("nan"))
            got[1].fill_(float("nan"))
            graph.replay()
            check(args, qnew, knew, got)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_singlekv_nonfinite_matches_existing_decode(value, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.decode_tile import (
        launch_decode_tile,
    )

    args = make_case(torch.bfloat16, True, [1, 1], [1, 0], heads=(4, 1), dim=256)
    args.update(out=torch.empty_like(args["q"]), softmax_scale=0.0625)
    args["q"].fill_(value)
    args["k"].fill_(1)
    resources = compile_resources([], args, monkeypatch)
    assert all(r["spills"] < 40 for r in resources)
    got = flaggems_vllm.flash_attn_varlen_func(**args)
    expected = torch.empty_like(got[0]), torch.empty_like(got[1])
    launch_decode_tile(
        args["q"],
        args["k"],
        args["v"],
        *expected,
        args["cu_seqlens_q"],
        args["seqused_k"],
        args["block_table"],
        0.0625,
    )
    torch.musa.synchronize()
    for actual, ref in zip(got, expected):
        torch.testing.assert_close(actual, ref, equal_nan=True, atol=0.002, rtol=0.002)
    assert (got[0][1] == 0).all() and torch.isposinf(got[1][:, 1]).all()


def test_singlekv_maxk_boundary(monkeypatch):
    args = make_case(torch.bfloat16, True, [1, 0, 1], [2, 1, 0], heads=(4, 1), dim=256)
    args.update(out=torch.empty_like(args["q"]))
    resources = compile_resources([], args, monkeypatch)
    assert all(r["name"] != "single_kv_kernel" for r in resources)
    assert all(r["spills"] < 40 for r in resources)
    check(args, [1, 0, 1], [2, 1, 0])
