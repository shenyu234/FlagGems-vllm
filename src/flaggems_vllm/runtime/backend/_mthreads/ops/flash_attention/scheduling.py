# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionPlan:
    family: str
    block_m: int
    block_n: int
    num_warps: int = 4
    num_stages: int = 1


def select_plan(max_seqlen_q, head_groups):
    """Static S5000 v1 policy; no tuning or device synchronization in capture.

    MThreads-only offline tuning is intentional. NVIDIA's libtuner policy and
    the MetaX C550 launch configurations do not apply to this kernel.
    """
    if max_seqlen_q == 1 and head_groups <= 16:
        return AttentionPlan("packed_decode", 16, 32)
    return AttentionPlan("direct", 32, 32)
