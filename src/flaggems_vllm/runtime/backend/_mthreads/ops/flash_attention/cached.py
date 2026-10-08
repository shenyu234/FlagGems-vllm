# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton KV compaction and a descriptor-based S5000 prefill path."""

import torch
import triton
import triton.language as tl

try:
    from triton.tools.experimental_descriptor import create_2d_tma_descriptor
except ImportError:
    create_2d_tma_descriptor = None

# TLE is optional so the original Triton 3.2 baseline environment can import us.
try:
    from .tle_prefill import launch_prefill
except ImportError:
    launch_prefill = None

# Bound K/V workspace without falling off the cached route just beyond 64K.
# The S5000 full-shape workload needs up to 520 MiB for the compacted K/V.
_MAX_CACHE_BYTES = 1024 * 1024 * 1024


def supports_cached(q, k, v, table, maxq, window, softcap, alibi, plan):
    if (create_2d_tma_descriptor is None and launch_prefill is None) or table is None:
        return False
    if q.shape[-1] != 256 or q.shape[0] < 512 or maxq <= 1:
        return False
    if window != (-1, -1) or softcap or alibi is not None or plan.num_splits != 1:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    nk = triton.cdiv(table.shape[1] * k.shape[1], 64) * 64
    return 0 < nk and table.shape[0] * k.shape[2] * 256 * nk * 4 <= _MAX_CACHE_BYTES


@triton.jit
def gather_cache_kernel(
    K,
    V,
    KC,
    VC,
    TABLE,
    USEDK,
    HK: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    NK: tl.constexpr,
    BN: tl.constexpr = 32,
    BD: tl.constexpr = 64,
):
    block, bh, db = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh // HK, bh % HK
    cols = block * BN + tl.arange(0, BN)
    dims = db * BD + tl.arange(0, BD)
    length = tl.load(USEDK + batch)
    page = tl.load(TABLE + batch * TS + cols // PAGE, cols < length, 0)
    ko = page.to(tl.int64) * KP + (cols % PAGE) * KS + head * KH
    vo = page.to(tl.int64) * VP + (cols % PAGE) * VS + head * VH
    mask = (cols[:, None] < length) & (dims[None, :] < D)
    key = tl.load(K + ko[:, None] + dims[None, :], mask, 0)
    value = tl.load(V + vo[:, None] + dims[None, :], mask, 0)
    output_mask = (cols[:, None] < NK) & (dims[None, :] < D)
    tl.store(
        KC + (bh * D + dims[None, :]).to(tl.int64) * NK + cols[:, None],
        key,
        output_mask,
    )
    tl.store(
        VC + (bh * NK + cols[:, None]).to(tl.int64) * D + dims[None, :],
        value,
        output_mask,
    )


@triton.jit
def cached_attention_kernel(
    QDESC,
    KDESC,
    VDESC,
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
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    row_base = tile * BM
    if row_base >= qlen:
        return
    rows = row_base + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    kh = head // (HQ // HK)
    q = tl._experimental_descriptor_load(
        QDESC, [qstart + row_base, head * 256], [BM, 256], OUT.dtype.element_ty
    )
    high = klen
    if CAUSAL:
        high = tl.minimum(high, tl.minimum(row_base + BM, qlen) + klen - qlen)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
    for start in range(0, high, BN):
        key = tl._experimental_descriptor_load(
            KDESC, [(batch * HK + kh) * 256, start], [256, BN], OUT.dtype.element_ty
        )
        scores = tl.dot(q, key) * (SCALE * 1.4426950408889634)
        fully_visible = start + BN <= klen
        if CAUSAL:
            fully_visible &= start + BN <= row_base + klen - qlen + 1
        if not fully_visible:
            cols = start + tl.arange(0, BN)
            mask = (rows[:, None] < qlen) & (cols[None, :] < klen)
            if CAUSAL:
                mask &= cols[None, :] <= rows[:, None] + klen - qlen
            scores = tl.where(mask, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        probs = tl.exp2(scores - safe_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probs, 1)
        acc *= alpha[:, None]
        value = tl._experimental_descriptor_load(
            VDESC, [(batch * HK + kh) * NK + start, 0], [BN, 256], OUT.dtype.element_ty
        )
        acc = tl.dot(probs.to(value.dtype), value, acc)
        maximum = next_maximum
    result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    lse = tl.where(
        denominator > 0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        float("inf"),
    )
    tl.store(
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        result,
        rows[:, None] < qlen,
    )
    tl.store(LSE + head * TQ + qstart + rows, lse, rows < qlen)


def launch_cached(q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal):
    if launch_prefill is not None:
        return launch_prefill(q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal)
    batch, hk, dim = table.shape[0], k.shape[2], q.shape[2]
    nk = triton.cdiv(table.shape[1] * k.shape[1], 64) * 64
    kc = torch.empty((batch * hk * dim, nk), device=q.device, dtype=q.dtype)
    vc = torch.empty((batch * hk * nk, dim), device=q.device, dtype=q.dtype)
    gather_cache_kernel[(triton.cdiv(nk, 32), batch * hk, 4)](
        k,
        v,
        kc,
        vc,
        table,
        usedk,
        hk,
        dim,
        k.shape[1],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        nk,
        num_warps=4,
        num_stages=1,
    )
    qdesc = create_2d_tma_descriptor(
        q.data_ptr(), q.shape[0], q.shape[1] * dim, 32, dim, q.element_size()
    )
    kdesc = create_2d_tma_descriptor(
        kc.data_ptr(), kc.shape[0], nk, dim, 64, q.element_size()
    )
    vdesc = create_2d_tma_descriptor(
        vc.data_ptr(), vc.shape[0], dim, 64, dim, q.element_size()
    )
    return cached_attention_kernel[(triton.cdiv(maxq, 32), batch, q.shape[1])](
        qdesc,
        kdesc,
        vdesc,
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
        num_warps=4,
        num_stages=1,
        en_backend_opt=True,
    )
