# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""M192 S5000 prefill with explicit alternating K/V barriers."""
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
    K, V, ks, vs, kb, ke, vb, ve, kh, iterations, NK: tl.constexpr, BN: tl.constexpr
):
    # All four producer warps participate in the compiler's local barrier;
    # TME itself is issued by the elected partition leader.
    for i in range(iterations):
        previous = i - 1
        tle.gpu.barrier_wait(ke[previous & 1], phaseIdx=(previous >> 1) & 1)
        tle.gpu.copy(K, ks.slot(0), (256, BN), (kh * 256, i * BN), barrier=kb[i & 1])
        tle.gpu.barrier_wait(ve[previous & 1], phaseIdx=(previous >> 1) & 1)
        tle.gpu.copy(V, vs.slot(0), (BN, 256), (kh * NK + i * BN, 0), barrier=vb[i & 1])


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
):
    q0 = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    k0 = tle.gpu.alloc((1, 256, BN), OUT.dtype.element_ty)
    v0 = tle.gpu.alloc((1, BN, 256), OUT.dtype.element_ty)
    tle.gpu.copy(Q, q0, (BM, 256), (qstart + row_base, head * 256), barrier=qbc)
    tle.gpu.barrier_wait(qbc, phaseIdx=0)
    rows = row_base + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
    denominator = tl.zeros((BM, 8), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
    p0 = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    for i in range(iterations):
        tle.gpu.barrier_wait(kb[i & 1], phaseIdx=(i >> 1) & 1)
        score = tle.gpu.wgmma(q0, k0.slot(0))
        score = tle.gpu.wgmma_wait(0, score)
        tle.gpu.barrier_arrive(ke[i & 1], arrive_count=1, phaseIdx=0)
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
        p = tl.reshape(tl.permute(tl.reshape(p, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN))
        tl.store(tle.gpu.local_ptr(p0), p)
        tl.debug_barrier()
        tle.gpu.barrier_wait(vb[i & 1], phaseIdx=(i >> 1) & 1)
        acc = tle.gpu.wgmma(p0, v0.slot(0), acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        tle.gpu.barrier_arrive(ve[i & 1], arrive_count=1, phaseIdx=0)
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
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        output,
        (rows[:, None] < qlen) & enabled,
    )


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
):
    q1 = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    k1 = tle.gpu.alloc((1, 256, BN), OUT.dtype.element_ty)
    v1 = tle.gpu.alloc((1, BN, 256), OUT.dtype.element_ty)
    tle.gpu.copy(Q, q1, (BM, 256), (qstart + row_base, head * 256), barrier=qbc)
    tle.gpu.barrier_wait(qbc, phaseIdx=0)
    rows = row_base + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
    denominator = tl.zeros((BM, 8), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
    p1 = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    for i in range(iterations):
        tle.gpu.barrier_wait(kb[i & 1], phaseIdx=(i >> 1) & 1)
        score = tle.gpu.wgmma(q1, k1.slot(0))
        score = tle.gpu.wgmma_wait(0, score)
        tle.gpu.barrier_arrive(ke[i & 1], arrive_count=1, phaseIdx=0)
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
        p = tl.reshape(tl.permute(tl.reshape(p, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN))
        tl.store(tle.gpu.local_ptr(p1), p)
        tl.debug_barrier()
        tle.gpu.barrier_wait(vb[i & 1], phaseIdx=(i >> 1) & 1)
        acc = tle.gpu.wgmma(p1, v1.slot(0), acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        tle.gpu.barrier_arrive(ve[i & 1], arrive_count=1, phaseIdx=0)
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
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        output,
        (rows[:, None] < qlen) & enabled,
    )


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
):
    q2 = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    k2 = tle.gpu.alloc((1, 256, BN), OUT.dtype.element_ty)
    v2 = tle.gpu.alloc((1, BN, 256), OUT.dtype.element_ty)
    tle.gpu.copy(Q, q2, (BM, 256), (qstart + row_base, head * 256), barrier=qbc)
    tle.gpu.barrier_wait(qbc, phaseIdx=0)
    rows = row_base + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
    denominator = tl.zeros((BM, 8), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
    p2 = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    for i in range(iterations):
        tle.gpu.barrier_wait(kb[i & 1], phaseIdx=(i >> 1) & 1)
        score = tle.gpu.wgmma(q2, k2.slot(0))
        score = tle.gpu.wgmma_wait(0, score)
        tle.gpu.barrier_arrive(ke[i & 1], arrive_count=1, phaseIdx=0)
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
        p = tl.reshape(tl.permute(tl.reshape(p, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN))
        tl.store(tle.gpu.local_ptr(p2), p)
        tl.debug_barrier()
        tle.gpu.barrier_wait(vb[i & 1], phaseIdx=(i >> 1) & 1)
        acc = tle.gpu.wgmma(p2, v2.slot(0), acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        tle.gpu.barrier_arrive(ve[i & 1], arrive_count=1, phaseIdx=0)
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
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        output,
        (rows[:, None] < qlen) & enabled,
    )


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
                    row_base + 64,
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
                    row_base + 128,
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
                ),
            ),
            (produce, (K, V, ks, vs, kb, ke, vb, ve, kh, iterations, NK, 64)),
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
    ni = tl.arange(0, triton.next_power_of_2(triton.cdiv(NK, 32)))
    knorm = tl.max(
        tl.load(NORM + kh * tl.cdiv(NK, 32) + ni, ni < tl.cdiv(NK, 32), 0), 0
    )
    bound = tl.sqrt(qnorm * knorm) * tl.abs(SCALE)
    eligible = (bound >= 0.0) & (bound <= 30.0)
    shift = tl.where(eligible, tl.ceil(bound * 1.4426950408889634) + 2.0, -float("inf"))
    for part in tl.static_range(3):
        tl.store(SHIFTS + (tile * 3 + part) * HQ + head, shift)


@triton.jit
def _online_body(
    Q,
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
    ONLY_BOUNDED: tl.constexpr,
    DOUBLE_V: tl.constexpr,
    SHIFT,
    tile,
    batch,
    head,
):
    qs = tle.gpu.alloc((64, 256), OUT.dtype.element_ty)
    qb = tle.gpu.alloc_barrier(expect_bytes=32768)
    kb = tle.gpu.alloc_barriers(4, expect_bytes=32768)
    vb = tle.gpu.alloc_barriers(4, expect_bytes=32768)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    row_base = tile * 64
    if row_base >= qlen:
        return
    rows = row_base + tl.arange(0, 64)
    dims = tl.arange(0, 256)
    kh = batch * HK + head // (HQ // HK)
    high = klen
    if CAUSAL:
        high = tl.minimum(high, tl.minimum(row_base + 64, qlen) + klen - qlen)
    high = tl.maximum(high, 0)
    tle.gpu.copy(Q, qs, (64, 256), (qstart + row_base, head * 256), barrier=qb)
    tle.gpu.barrier_wait(qb, phaseIdx=0)
    shift = SHIFT
    bounded: tl.constexpr = ONLY_BOUNDED
    maximum = tl.full((64,), -float("inf"), tl.float32)
    maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
    denominator = tl.zeros((64, 8), tl.float32)
    acc = tl.zeros((64, 256), tl.float32)
    ps = tle.gpu.alloc((2, 64, 64), OUT.dtype.element_ty)
    ks = tle.gpu.alloc((256, 64), OUT.dtype.element_ty)
    vs = tle.gpu.alloc((2 if DOUBLE_V else 1, 64, 256), OUT.dtype.element_ty)
    iterations = tl.cdiv(high, 64)
    if iterations > 0:
        tle.gpu.copy(K, ks, (256, 64), (kh * 256, 0), barrier=kb[0])
        tle.gpu.barrier_wait(kb[0], phaseIdx=0)
        score = tle.gpu.wgmma(qs, ks)
        score = tle.gpu.wgmma_wait(0, score)
        if iterations > 1:
            tle.gpu.copy(K, ks, (256, 64), (kh * 256, 64), barrier=kb[1])
        (p, alpha, maximum, denominator) = update_softmax(
            score,
            maximum,
            denominator,
            rows,
            row_base,
            qlen,
            klen,
            0,
            SCALE,
            CAUSAL,
            bounded,
        )
        p_store = p.to(OUT.dtype.element_ty)
        p_store = tl.reshape(
            tl.permute(tl.reshape(p_store, (64, 8, 8)), (0, 2, 1)), (64, 64)
        )
        tl.store(tle.gpu.local_ptr(ps.slot(0)), p_store)
        if DOUBLE_V and iterations > 1:
            tle.gpu.copy(V, vs.slot(0), (64, 256), (kh * NK, 0), barrier=vb[0])
        for i in range(1, iterations):
            if not DOUBLE_V:
                tle.gpu.copy(
                    V,
                    vs.slot(0),
                    (64, 256),
                    (kh * NK + (i - 1) * 64, 0),
                    barrier=vb[(i - 1) % 4],
                )
            tle.gpu.barrier_wait(kb[i % 4], phaseIdx=i // 4 & 1)
            score = tle.gpu.wgmma(qs, ks)
            if ONLY_BOUNDED:
                tle.gpu.barrier_wait(vb[(i - 1) % 4], phaseIdx=(i - 1) // 4 & 1)
                pending_acc = tle.gpu.wgmma(
                    ps.slot((i - 1) % 2), vs.slot((i - 1) % 2 if DOUBLE_V else 0), acc
                )
            score = tle.gpu.wgmma_wait(0, score)
            if DOUBLE_V and i + 1 < iterations:
                tle.gpu.copy(
                    V,
                    vs.slot(i % 2),
                    (64, 256),
                    (kh * NK + i * 64, 0),
                    barrier=vb[i % 4],
                )
            if i + 1 < iterations:
                tle.gpu.copy(
                    K, ks, (256, 64), (kh * 256, (i + 1) * 64), barrier=kb[(i + 1) % 4]
                )
            if not ONLY_BOUNDED:
                tle.gpu.barrier_wait(vb[(i - 1) % 4], phaseIdx=(i - 1) // 4 & 1)
                pending_acc = tle.gpu.wgmma(
                    ps.slot((i - 1) % 2), vs.slot((i - 1) % 2 if DOUBLE_V else 0), acc
                )
            (p, alpha, maximum, denominator) = update_softmax(
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
            acc = tle.gpu.wgmma_wait(0, pending_acc)
            if bounded:
                pass
            else:
                acc = rescale_o(acc, alpha, False)
            p_store = p.to(OUT.dtype.element_ty)
            p_store = tl.reshape(
                tl.permute(tl.reshape(p_store, (64, 8, 8)), (0, 2, 1)), (64, 64)
            )
            tl.store(tle.gpu.local_ptr(ps.slot(i % 2)), p_store)
        last = iterations - 1
        tle.gpu.copy(
            V,
            vs.slot(last % 2 if DOUBLE_V else 0),
            (64, 256),
            (kh * NK + last * 64, 0),
            barrier=vb[last % 4],
        )
        tle.gpu.barrier_wait(vb[last % 4], phaseIdx=last // 4 & 1)
        acc = tle.gpu.wgmma(
            ps.slot(last % 2), vs.slot(last % 2 if DOUBLE_V else 0), acc
        )
        acc = tle.gpu.wgmma_wait(0, acc)
    denominator = tl.sum(denominator, 1)
    maximum *= SCALE * 1.4426950408889634
    lse = tl.where(
        denominator > 0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        float("inf"),
    )
    tl.store(LSE + head * TQ + qstart + rows, lse, rows < qlen)
    output = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        output,
        rows[:, None] < qlen,
    )


@triton.jit
def online_fallback(
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
    BATCH: tl.constexpr,
    COMPACT: tl.constexpr,
    FAST: tl.constexpr,
    DOUBLE_V: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    if tile * 64 >= qlen:
        return
    shift = tl.load(SHIFTS + tl.program_id(0) * HQ + head)
    eligible = shift != -float("inf")
    if eligible == FAST:
        _online_body(
            Q,
            K,
            V,
            OUT,
            LSE,
            CUQ,
            USEDK,
            TQ,
            HQ,
            HK,
            NK,
            SCALE,
            CAUSAL,
            FAST,
            DOUBLE_V,
            shift if FAST else 0.0,
            tile,
            batch,
            head,
        )


def supports_triplet(q, k, table, maxq):
    """Long-query prefill, including the medium-cache region."""
    return (
        os.environ.get("MTHREADS_FA_DISABLE_WS_PREFILL") != "1"
        and table.shape[0] == 1
        and q.dtype == torch.bfloat16
        and q.shape[2] == 256
        and maxq >= 4096
        and table.shape[1] * k.shape[1] >= 4096
    )


def launch_triplet(q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal):
    hk, d = k.shape[2], q.shape[2]
    nk = triton.cdiv(table.shape[1] * k.shape[1], 64) * 64
    kc = torch.empty((hk * d, nk), device=q.device, dtype=q.dtype)
    vc = torch.empty((hk * nk, d), device=q.device, dtype=q.dtype)
    norms = torch.empty((hk, triton.cdiv(nk, 32)), device=q.device, dtype=torch.float32)
    groups = triton.cdiv(maxq, 192)
    shifts = torch.empty((groups * 3, q.shape[1]), device=q.device, dtype=torch.float32)
    qd = TensorDescriptor(q, (q.shape[0], q.shape[1] * d), (q.shape[1] * d, 1), (64, d))
    kd = TensorDescriptor(kc, (hk * d, nk), (nk, 1), (d, 64))
    vd = TensorDescriptor(vc, (hk * nk, d), (d, 1), (64, d))
    gather_norm_kernel[triton.cdiv(nk, 32), hk, 1](
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
        BD=256,
        num_warps=4,
        num_stages=1,
    )
    classify192[groups, q.shape[1]](
        q, norms, shifts, cuq, q.shape[1], hk, nk, scale, num_warps=16, num_stages=1
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
    compiled = attention[groups, q.shape[1]](
        *common,
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
