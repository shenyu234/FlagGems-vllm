# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Qwen3.6 TP1/TP4 trace cases shared by tests and benchmarks.

The original benchmark derives host maxima and page-table width from lengths
and uses num_splits=0. Original trace hints remain in trace_metadata.
"""
import json
import os
import zlib
from pathlib import Path


def load_cases(shape_set="full", case_ids=None):
    if shape_set not in ("full", "tp1-full", "tp4-full"):
        raise ValueError(f"Unknown MThreads shape set: {shape_set}")
    root = Path(__file__).with_name("shape_data")
    cases = []
    for tp, count in (("tp1", 275), ("tp4", 287)):
        rows = json.loads((root / f"qwen36_{tp}_flash_attn_cases.json").read_text())
        if len(rows) != count:
            raise ValueError(f"Incomplete {tp} shape data")
        cases.extend(rows)
    if len({case["case_id"] for case in cases}) != 562:
        raise ValueError("Duplicate case IDs")
    if shape_set != "full":
        cases = [case for case in cases if case["tp"].lower() == shape_set[:3]]
    if case_ids is None:
        case_ids = os.environ.get("MTHREADS_FA_CASE_IDS", "")
    if case_ids:
        requested = set(case_ids.split(","))
        unknown = requested - {case["case_id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown or out-of-set case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["case_id"] in requested]
    return cases


def case_config(case):
    return (
        tuple(case["cu_seqlens_q"]),
        tuple(case["seqused_k"]),
        case["q_shape"][1],
        case["k_shape"][2],
        case["q_shape"][2],
        case["k_shape"][1],
        case["k_shape"][0],
        False,
        None,
    )


def case_seed(case):
    return 20261008 + zlib.crc32(case["case_id"].encode())
