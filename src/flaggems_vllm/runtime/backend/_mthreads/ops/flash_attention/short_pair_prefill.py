# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Short causal prefill with persistent first/last M192 group pairs.

Each CTA balances two causal prefixes. K/V barrier slots and phases advance
continuously across groups; each group has fresh softmax state and Q-copy phase.
"""
import os

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.experimental.tle.language.gpu.types import swizzled_shared_layout
from triton.tools.tensor_descriptor import TensorDescriptor

from ._triplet_compiler import bind_kernel
from .bounded_prefill import _AFFINE_LIB, _MATH_LIB, gather_norm_kernel, update_softmax
from .tle_prefill import rescale_o
from .triplet_prefill import online_fallback

PRODUCER_K_LAYOUT = tl.constexpr(
    swizzled_shared_layout(
        16, 2, 8, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0]
    )
)
PRODUCER_V_LAYOUT = tl.constexpr(
    swizzled_shared_layout(
        32, 1, 4, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0]
    )
)


@triton.jit
def produce(
    K,
    V,
    ks,
    vs,
    kb,
    ke,
    vb,
    ve,
    kh,
    iterations,
    NK: tl.constexpr,
    BN: tl.constexpr,
    SHIFTS,
    first_tile,
    qlen,
    klen,
    head,
    HQ: tl.constexpr,
    CAUSAL: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # All four producer warps participate in the compiler's local barrier;
    # TME itself is issued by the elected partition leader.
    step_base = 0
    for part in range(2):
        group = tl.where(part == 0, first_tile, GROUPS - 1 - first_tile)
        row_base = group * 192
        shift = tl.load(SHIFTS + group * 3 * HQ + head)
        enabled = (shift != -float("inf")) & ((part == 0) | (group != first_tile))
        high = klen
        if CAUSAL:
            high = tl.minimum(high, tl.minimum(row_base + 192, qlen) + klen - qlen)
        iterations = tl.cdiv(
            tl.maximum(tl.where((row_base < qlen) & enabled, high, 0), 0), 64
        )
        for i in range(iterations):
            previous = step_base + i - 1
            tle.gpu.barrier_wait(ke[previous & 1], phaseIdx=(previous >> 1) & 1)
            tle.gpu.copy(
                K,
                ks.slot(0),
                (256, BN),
                (kh * 256, i * BN),
                barrier=kb[(step_base + i) & 1],
            )
            tle.gpu.barrier_wait(ve[previous & 1], phaseIdx=(previous >> 1) & 1)
            tle.gpu.copy(
                V,
                vs.slot(0),
                (BN, 256),
                (kh * NK + i * BN, 0),
                barrier=vb[(step_base + i) & 1],
            )

        step_base += iterations


@triton.jit
def consume0(
    Q,
    qbc,
    kb,
    ke,
    vb,
    ve,
    OUT,
    LSE,
    qstart,
    qlen,
    klen,
    row_base,
    head,
    bounded: tl.constexpr,
    enabled,
    shift,
    iterations,
    HQ: tl.constexpr,
    TQ: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SHIFTS,
    GROUPS: tl.constexpr,
):
    q0 = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    k0 = tle.gpu.alloc((1, 256, BN), OUT.dtype.element_ty)
    v0 = tle.gpu.alloc((1, BN, 256), OUT.dtype.element_ty)
    p0 = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    first_tile = row_base // 192
    step_base = 0
    for part in range(2):
        group = tl.where(part == 0, first_tile, GROUPS - 1 - first_tile)
        group_row_base = group * 192
        row_base = group_row_base + 0
        shift = tl.load(SHIFTS + group * 3 * HQ + head)
        enabled = (shift != -float("inf")) & ((part == 0) | (group != first_tile))
        high = klen
        if CAUSAL:
            high = tl.minimum(
                high, tl.minimum(group_row_base + 192, qlen) + klen - qlen
            )
        iterations = tl.cdiv(
            tl.maximum(tl.where((group_row_base < qlen) & enabled, high, 0), 0), 64
        )
        tle.gpu.copy(Q, q0, (BM, 256), (qstart + row_base, head * 256), barrier=qbc)
        tle.gpu.barrier_wait(qbc, phaseIdx=part & 1)
        rows = row_base + tl.arange(0, BM)
        dims = tl.arange(0, 256)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
        denominator = tl.zeros((BM, 8), tl.float32)
        acc = tl.zeros((BM, 256), tl.float32)
        consumer_high = klen
        if CAUSAL:
            consumer_high = tl.minimum(
                klen, tl.minimum(row_base + BM, qlen) + klen - qlen
            )
        for i in range(iterations):
            tle.gpu.barrier_wait(
                kb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
            )
            if i * BN < consumer_high:
                score = tle.gpu.wgmma(q0, k0.slot(0))
                score = tle.gpu.wgmma_wait(0, score)
                tle.gpu.barrier_arrive(
                    ke[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
                p, alpha, maximum, denominator = update_softmax(
                    score,
                    maximum,
                    denominator,
                    rows,
                    row_base,
                    qlen,
                    klen,
                    i,
                    SCALE,
                    CAUSAL,
                    bounded,
                )
                if not bounded:
                    acc = rescale_o(acc, alpha, False)
                p = p.to(OUT.dtype.element_ty)
                p = tl.reshape(
                    tl.permute(tl.reshape(p, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN)
                )
                tl.store(tle.gpu.local_ptr(p0), p)
                tl.debug_barrier()
                tle.gpu.barrier_wait(
                    vb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
                )
                acc = tle.gpu.wgmma(p0, v0.slot(0), acc)
                acc = tle.gpu.wgmma_wait(0, acc)
                tle.gpu.barrier_arrive(
                    ve[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
            else:
                # Participate in the complete producer protocol even for blocks
                # whose logits are all masked for this consumer.
                tle.gpu.barrier_arrive(
                    ke[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
                tle.gpu.barrier_wait(
                    vb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
                )
                tle.gpu.barrier_arrive(
                    ve[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
        denominator = tl.sum(denominator, 1)
        maximum *= SCALE * 1.4426950408889634
        lse = tl.where(
            denominator > 0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(LSE + head * TQ + qstart + rows, lse, (rows < qlen) & enabled)
        output = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(
            OUT
            + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256
            + dims[None, :],
            output,
            (rows[:, None] < qlen) & enabled,
        )

        step_base += iterations


@triton.jit
def consume1(
    Q,
    qbc,
    kb,
    ke,
    vb,
    ve,
    OUT,
    LSE,
    qstart,
    qlen,
    klen,
    row_base,
    head,
    bounded: tl.constexpr,
    enabled,
    shift,
    iterations,
    HQ: tl.constexpr,
    TQ: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SHIFTS,
    GROUPS: tl.constexpr,
):
    q1 = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    k1 = tle.gpu.alloc((1, 256, BN), OUT.dtype.element_ty)
    v1 = tle.gpu.alloc((1, BN, 256), OUT.dtype.element_ty)
    p1 = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    first_tile = row_base // 192
    step_base = 0
    for part in range(2):
        group = tl.where(part == 0, first_tile, GROUPS - 1 - first_tile)
        group_row_base = group * 192
        row_base = group_row_base + 64
        shift = tl.load(SHIFTS + group * 3 * HQ + head)
        enabled = (shift != -float("inf")) & ((part == 0) | (group != first_tile))
        high = klen
        if CAUSAL:
            high = tl.minimum(
                high, tl.minimum(group_row_base + 192, qlen) + klen - qlen
            )
        iterations = tl.cdiv(
            tl.maximum(tl.where((group_row_base < qlen) & enabled, high, 0), 0), 64
        )
        tle.gpu.copy(Q, q1, (BM, 256), (qstart + row_base, head * 256), barrier=qbc)
        tle.gpu.barrier_wait(qbc, phaseIdx=part & 1)
        rows = row_base + tl.arange(0, BM)
        dims = tl.arange(0, 256)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
        denominator = tl.zeros((BM, 8), tl.float32)
        acc = tl.zeros((BM, 256), tl.float32)
        consumer_high = klen
        if CAUSAL:
            consumer_high = tl.minimum(
                klen, tl.minimum(row_base + BM, qlen) + klen - qlen
            )
        for i in range(iterations):
            tle.gpu.barrier_wait(
                kb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
            )
            if i * BN < consumer_high:
                score = tle.gpu.wgmma(q1, k1.slot(0))
                score = tle.gpu.wgmma_wait(0, score)
                tle.gpu.barrier_arrive(
                    ke[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
                p, alpha, maximum, denominator = update_softmax(
                    score,
                    maximum,
                    denominator,
                    rows,
                    row_base,
                    qlen,
                    klen,
                    i,
                    SCALE,
                    CAUSAL,
                    bounded,
                )
                if not bounded:
                    acc = rescale_o(acc, alpha, False)
                p = p.to(OUT.dtype.element_ty)
                p = tl.reshape(
                    tl.permute(tl.reshape(p, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN)
                )
                tl.store(tle.gpu.local_ptr(p1), p)
                tl.debug_barrier()
                tle.gpu.barrier_wait(
                    vb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
                )
                acc = tle.gpu.wgmma(p1, v1.slot(0), acc)
                acc = tle.gpu.wgmma_wait(0, acc)
                tle.gpu.barrier_arrive(
                    ve[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
            else:
                # Participate in the complete producer protocol even for blocks
                # whose logits are all masked for this consumer.
                tle.gpu.barrier_arrive(
                    ke[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
                tle.gpu.barrier_wait(
                    vb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
                )
                tle.gpu.barrier_arrive(
                    ve[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
        denominator = tl.sum(denominator, 1)
        maximum *= SCALE * 1.4426950408889634
        lse = tl.where(
            denominator > 0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(LSE + head * TQ + qstart + rows, lse, (rows < qlen) & enabled)
        output = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(
            OUT
            + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256
            + dims[None, :],
            output,
            (rows[:, None] < qlen) & enabled,
        )

        step_base += iterations


@triton.jit
def consume2(
    Q,
    qbc,
    kb,
    ke,
    vb,
    ve,
    OUT,
    LSE,
    qstart,
    qlen,
    klen,
    row_base,
    head,
    bounded: tl.constexpr,
    enabled,
    shift,
    iterations,
    HQ: tl.constexpr,
    TQ: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SHIFTS,
    GROUPS: tl.constexpr,
):
    q2 = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    k2 = tle.gpu.alloc((1, 256, BN), OUT.dtype.element_ty)
    v2 = tle.gpu.alloc((1, BN, 256), OUT.dtype.element_ty)
    p2 = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    first_tile = row_base // 192
    step_base = 0
    for part in range(2):
        group = tl.where(part == 0, first_tile, GROUPS - 1 - first_tile)
        group_row_base = group * 192
        row_base = group_row_base + 128
        shift = tl.load(SHIFTS + group * 3 * HQ + head)
        enabled = (shift != -float("inf")) & ((part == 0) | (group != first_tile))
        high = klen
        if CAUSAL:
            high = tl.minimum(
                high, tl.minimum(group_row_base + 192, qlen) + klen - qlen
            )
        iterations = tl.cdiv(
            tl.maximum(tl.where((group_row_base < qlen) & enabled, high, 0), 0), 64
        )
        tle.gpu.copy(Q, q2, (BM, 256), (qstart + row_base, head * 256), barrier=qbc)
        tle.gpu.barrier_wait(qbc, phaseIdx=part & 1)
        rows = row_base + tl.arange(0, BM)
        dims = tl.arange(0, 256)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
        denominator = tl.zeros((BM, 8), tl.float32)
        acc = tl.zeros((BM, 256), tl.float32)
        consumer_high = klen
        if CAUSAL:
            consumer_high = tl.minimum(
                klen, tl.minimum(row_base + BM, qlen) + klen - qlen
            )
        for i in range(iterations):
            tle.gpu.barrier_wait(
                kb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
            )
            if i * BN < consumer_high:
                score = tle.gpu.wgmma(q2, k2.slot(0))
                score = tle.gpu.wgmma_wait(0, score)
                tle.gpu.barrier_arrive(
                    ke[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
                p, alpha, maximum, denominator = update_softmax(
                    score,
                    maximum,
                    denominator,
                    rows,
                    row_base,
                    qlen,
                    klen,
                    i,
                    SCALE,
                    CAUSAL,
                    bounded,
                )
                if not bounded:
                    acc = rescale_o(acc, alpha, False)
                p = p.to(OUT.dtype.element_ty)
                p = tl.reshape(
                    tl.permute(tl.reshape(p, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN)
                )
                tl.store(tle.gpu.local_ptr(p2), p)
                tl.debug_barrier()
                tle.gpu.barrier_wait(
                    vb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
                )
                acc = tle.gpu.wgmma(p2, v2.slot(0), acc)
                acc = tle.gpu.wgmma_wait(0, acc)
                tle.gpu.barrier_arrive(
                    ve[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
            else:
                # Participate in the complete producer protocol even for blocks
                # whose logits are all masked for this consumer.
                tle.gpu.barrier_arrive(
                    ke[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
                tle.gpu.barrier_wait(
                    vb[(step_base + i) & 1], phaseIdx=((step_base + i) >> 1) & 1
                )
                tle.gpu.barrier_arrive(
                    ve[(step_base + i) & 1], arrive_count=1, phaseIdx=0
                )
        denominator = tl.sum(denominator, 1)
        maximum *= SCALE * 1.4426950408889634
        lse = tl.where(
            denominator > 0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            float("inf"),
        )
        tl.store(LSE + head * TQ + qstart + rows, lse, (rows < qlen) & enabled)
        output = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(
            OUT
            + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256
            + dims[None, :],
            output,
            (rows[:, None] < qlen) & enabled,
        )

        step_base += iterations


@triton.jit
def attention(
    Q,
    SHIFTS,
    K,
    V,
    OUT,
    LSE,
    CUQ,
    USEDK,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NK: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    PKLAYOUT: tl.constexpr,
    PVLAYOUT: tl.constexpr,
    GROUPS: tl.constexpr,
):
    kb = tle.gpu.alloc_barriers(2, expect_bytes=64 * 256 * 2)
    vb = tle.gpu.alloc_barriers(2, expect_bytes=64 * 256 * 2)
    ke = tle.gpu.alloc_barriers(2, arrive_count=12)
    ve = tle.gpu.alloc_barriers(2, arrive_count=12)
    qb0 = tle.gpu.alloc_barrier(expect_bytes=64 * 256 * 2)
    qb1 = tle.gpu.alloc_barrier(expect_bytes=64 * 256 * 2)
    qb2 = tle.gpu.alloc_barrier(expect_bytes=64 * 256 * 2)
    tile, head = tl.program_id(0), tl.program_id(1)
    qstart = tl.load(CUQ)
    qlen = tl.load(CUQ + 1) - qstart
    klen = tl.load(USEDK)
    row_base = tile * 192
    kh = head // (HQ // HK)
    shift = tl.load(SHIFTS + tile * 3 * HQ + head)
    enabled = shift != -float("inf")
    high = klen
    if CAUSAL:
        high = tl.minimum(high, tl.minimum(row_base + 192, qlen) + klen - qlen)
    iterations = tl.cdiv(
        tl.maximum(tl.where((row_base < qlen) & enabled, high, 0), 0), 64
    )
    ks = tle.gpu.alloc((1, 256, 64), OUT.dtype.element_ty, layout=PKLAYOUT)
    vs = tle.gpu.alloc((1, 64, 256), OUT.dtype.element_ty, layout=PVLAYOUT)
    tle.gpu.warp_specialize(
        [
            (
                consume0,
                (
                    Q,
                    qb0,
                    kb,
                    ke,
                    vb,
                    ve,
                    OUT,
                    LSE,
                    qstart,
                    qlen,
                    klen,
                    row_base,
                    head,
                    True,
                    enabled,
                    shift,
                    iterations,
                    HQ,
                    TQ,
                    SCALE,
                    CAUSAL,
                    64,
                    64,
                    SHIFTS,
                    GROUPS,
                ),
            ),
            (
                consume1,
                (
                    Q,
                    qb1,
                    kb,
                    ke,
                    vb,
                    ve,
                    OUT,
                    LSE,
                    qstart,
                    qlen,
                    klen,
                    row_base,
                    head,
                    True,
                    enabled,
                    shift,
                    iterations,
                    HQ,
                    TQ,
                    SCALE,
                    CAUSAL,
                    64,
                    64,
                    SHIFTS,
                    GROUPS,
                ),
            ),
            (
                consume2,
                (
                    Q,
                    qb2,
                    kb,
                    ke,
                    vb,
                    ve,
                    OUT,
                    LSE,
                    qstart,
                    qlen,
                    klen,
                    row_base,
                    head,
                    True,
                    enabled,
                    shift,
                    iterations,
                    HQ,
                    TQ,
                    SCALE,
                    CAUSAL,
                    64,
                    64,
                    SHIFTS,
                    GROUPS,
                ),
            ),
            (
                produce,
                (
                    K,
                    V,
                    ks,
                    vs,
                    kb,
                    ke,
                    vb,
                    ve,
                    kh,
                    iterations,
                    NK,
                    64,
                    SHIFTS,
                    tile,
                    qlen,
                    klen,
                    head,
                    HQ,
                    CAUSAL,
                    GROUPS,
                ),
            ),
        ],
        [4, 4, 4],
        [240, 240, 32],
    )


@triton.jit
def classify192(
    QRAW,
    NORM,
    SHIFTS,
    CUQ,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NK: tl.constexpr,
    SCALE: tl.constexpr,
    GN: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    qstart = tl.load(CUQ)
    qlen = tl.load(CUQ + 1) - qstart
    local = tl.arange(0, 256)
    rows = tile * 192 + local
    dims = tl.arange(0, 256)
    q = tl.load(
        QRAW
        + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256
        + dims[None, :],
        (local[:, None] < 192) & (rows[:, None] < qlen),
        0,
    ).to(tl.float32)
    qnorm = tl.max(tl.sum(q * q, 1), 0)
    kh = head // (HQ // HK)
    ni = tl.arange(0, triton.next_power_of_2(triton.cdiv(NK, GN)))
    knorm = tl.max(
        tl.load(NORM + kh * tl.cdiv(NK, GN) + ni, ni < tl.cdiv(NK, GN), 0), 0
    )
    bound = tl.sqrt(qnorm * knorm) * tl.abs(SCALE)
    eligible = (bound >= 0.0) & (bound <= 30.0)
    shift = tl.where(eligible, tl.ceil(bound * 1.4426950408889634) + 2.0, -float("inf"))
    for part in tl.static_range(3):
        tl.store(SHIFTS + (tile * 3 + part) * HQ + head, shift)


def supports_short_pair(q, k, table, maxq, causal):
    """Pair only short, causal, multi-wave prefill with enough query heads."""
    return (
        os.environ.get("MTHREADS_FA_DISABLE_WS_PREFILL") != "1"
        and table.shape[0] == 1
        and q.dtype == torch.bfloat16
        and q.shape[2] == 256
        and causal
        and 512 <= maxq <= 2048
        and 0 < table.shape[1] * k.shape[1] <= 2048
        and q.shape[1] >= 16
        and triton.cdiv(maxq, 192) * q.shape[1] > 60
    )


def launch_short_pair(q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal):
    gather_tile, gather_warps = 16, 8
    hk, d = k.shape[2], q.shape[2]
    nk = triton.cdiv(table.shape[1] * k.shape[1], 64) * 64
    kc = torch.empty((hk * d, nk), device=q.device, dtype=q.dtype)
    vc = torch.empty((hk * nk, d), device=q.device, dtype=q.dtype)
    norms = torch.empty(
        (hk, triton.cdiv(nk, gather_tile)), device=q.device, dtype=torch.float32
    )
    groups = triton.cdiv(maxq, 192)
    shifts = torch.empty((groups * 3, q.shape[1]), device=q.device, dtype=torch.float32)
    qd = TensorDescriptor(q, (q.shape[0], q.shape[1] * d), (q.shape[1] * d, 1), (64, d))
    kd = TensorDescriptor(kc, (hk * d, nk), (nk, 1), (d, 64))
    vd = TensorDescriptor(vc, (hk * nk, d), (d, 1), (64, d))
    gather_norm_kernel[triton.cdiv(nk, gather_tile), hk, 1](
        k,
        v,
        kc,
        vc,
        norms,
        table,
        usedk,
        hk,
        d,
        k.shape[1],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        nk,
        True,
        64,
        BN=gather_tile,
        BD=256,
        num_warps=gather_warps,
        num_stages=1,
    )
    classify192[groups, q.shape[1]](
        q,
        norms,
        shifts,
        cuq,
        q.shape[1],
        hk,
        nk,
        scale,
        gather_tile,
        num_warps=16,
        num_stages=1,
    )
    common = (
        qd,
        shifts,
        kd,
        vd,
        out,
        lse,
        cuq,
        usedk,
        q.shape[0],
        q.shape[1],
        hk,
        nk,
        scale,
        causal,
    )
    compiled = attention[triton.cdiv(groups, 2), q.shape[1]](
        *common,
        GROUPS=groups,
        PKLAYOUT=PRODUCER_K_LAYOUT,
        PVLAYOUT=PRODUCER_V_LAYOUT,
        num_warps=4,
        num_stages=1,
        enable_backend_opt=True,
        maxnreg=240,
        extern_libs={"fa_math4": _MATH_LIB, "fa_affine4": _AFFINE_LIB},
    )
    online_fallback[triton.cdiv(maxq, 64), 1, q.shape[1]](
        *common,
        BATCH=1,
        COMPACT=False,
        FAST=False,
        DOUBLE_V=False,
        num_warps=16,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"fa_math4": _MATH_LIB, "fa_affine4": _AFFINE_LIB},
    )
    return compiled


bind_kernel(attention)
