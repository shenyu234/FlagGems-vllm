# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USEDK,
    TABLE,
    ALIBI,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    SOFTCAP: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    if PAGED:
        kstart = 0
        klen = tl.load(USEDK + batch)
    else:
        kstart = tl.load(CUK + batch)
        klen = tl.load(CUK + batch + 1) - kstart
    groups: tl.constexpr = HQ // HK
    rows = tl.arange(0, BM)
    if PACKED:
        qi = tl.full((BM,), 0, tl.int32)
        qh = head * groups + rows
        kh = head
        valid_q = (rows < groups) & (qlen > 0)
        row_base = 0
        row_end = 1
    else:
        qi = tile * BM + rows
        qh = tl.full((BM,), head, tl.int32)
        kh = head // groups
        valid_q = qi < qlen
        row_base = tile * BM
        row_end = tl.minimum((tile + 1) * BM, qlen)
    if row_base >= qlen:
        return
    dims = tl.arange(0, BD)
    q = tl.load(
        Q + (qstart + qi[:, None]).to(tl.int64) * QS + qh[:, None] * QH + dims[None, :],
        valid_q[:, None] & (dims[None, :] < D),
        0,
    )
    low = 0
    high = klen
    if WL >= 0:
        low = tl.maximum(0, row_base + klen - qlen - WL)
        low = low // BN * BN
    if CAUSAL:
        high = tl.minimum(high, row_end + klen - qlen)
    if WR >= 0:
        high = tl.minimum(high, row_end + klen - qlen + WR)
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    if HAS_ALIBI:
        slope = tl.load(ALIBI + batch * AS + qh, valid_q, 0)
    for start in range(low, high, BN):
        cols = start + tl.arange(0, BN)
        if PAGED:
            page = tl.load(TABLE + batch * TS + cols // PAGE, cols < klen, 0)
            # Cache tensors in the active workload exceed signed 32-bit offsets.
            ko = page.to(tl.int64) * KP + (cols % PAGE) * KS
            vo = page.to(tl.int64) * VP + (cols % PAGE) * VS
        else:
            ko = (kstart + cols).to(tl.int64) * KS
            vo = (kstart + cols).to(tl.int64) * VS
        k = tl.load(
            K + ko[None, :] + kh * KH + dims[:, None],
            (cols[None, :] < klen) & (dims[:, None] < D),
            0,
        )
        scores = tl.dot(q, k) * SCALE
        if SOFTCAP > 0:
            z = scores / SOFTCAP
            scores = SOFTCAP * (2.0 / (1.0 + tl.exp(2.0 * -z)) - 1.0)
        pos = qi[:, None] + klen - qlen
        if HAS_ALIBI:
            scores -= slope[:, None] * tl.abs(pos - cols[None, :])
        mask = valid_q[:, None] & (cols[None, :] < klen)
        if CAUSAL:
            mask &= cols[None, :] <= pos
        if WL >= 0:
            mask &= cols[None, :] >= pos - WL
        if WR >= 0:
            mask &= cols[None, :] <= pos + WR
        scores = tl.where(mask, scores * 1.4426950408889634, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        row_sum = row_sum * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v = tl.load(
            V + vo[:, None] + kh * VH + dims[None, :],
            (cols[:, None] < klen) & (dims[None, :] < D),
            0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m = new_m
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    tl.store(
        O + (qstart + qi[:, None]).to(tl.int64) * OS + qh[:, None] * OH + dims[None, :],
        result,
        valid_q[:, None] & (dims[None, :] < D),
    )
    tl.store(LSE + qh * TQ + qstart + qi, lse, valid_q)


def launch_direct(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
):
    paged = table is not None
    packed = plan.family == "packed_decode"
    hkv = k.shape[-2]
    grid = (
        1 if packed else triton.cdiv(maxq, plan.block_m),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    return attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        # Keep both MMA output dimensions in the validated S5000 range.
        max(64, triton.next_power_of_2(q.shape[2])),
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )
