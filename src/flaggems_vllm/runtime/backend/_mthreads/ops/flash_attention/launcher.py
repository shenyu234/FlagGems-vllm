# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from .cached import launch_cached, supports_cached
from .common import validate_inputs
from .decode_tile import launch_decode_tile, supports_decode_tile
from .direct import launch_direct
from .scheduling import select_plan
from .single_kv import launch_single_kv, supports_single_kv
from .splitkv import launch_splitkv


def launch_attention(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits=0,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    fa_version=2,
):
    """Inference-only MUSA frontend; no data conversion or torch compute path."""
    if dropout_p != 0 or return_attn_probs:
        raise NotImplementedError("Dropout and attention probabilities are unsupported")
    if any(
        x is not None
        for x in (
            q_v,
            scheduler_metadata,
            q_descale,
            k_descale,
            v_descale,
            s_aux,
            cp_tot_seqused_k,
        )
    ):
        raise NotImplementedError("FA3 auxiliary inputs are unsupported")
    if num_splits != 0 or cp_world_size != 1 or cp_rank != 0 or fa_version != 2:
        raise NotImplementedError(
            "Explicit splits, context parallel and FA3 are unsupported"
        )
    validate_inputs(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table,
        out,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
    )
    window = (-1, -1) if window_size is None else tuple(window_size)
    if len(window) != 2 or any(not isinstance(x, int) or x < -1 for x in window):
        raise ValueError("window_size must contain two integers >= -1")
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    softcap = float(softcap)
    if not math.isfinite(scale) or not math.isfinite(softcap) or softcap < 0:
        raise ValueError("Scale must be finite and softcap finite/nonnegative")
    plan = select_plan(
        max_seqlen_q,
        q.shape[1] // k.shape[-2],
        max_seqlen_k=max_seqlen_k,
        total_q=q.shape[0],
        num_heads=q.shape[1],
        batch_size=cu_seqlens_q.numel() - 1,
        head_size=q.shape[2],
    )
    with torch.musa.device(q.device):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
        )
        if q.shape[0]:
            if supports_single_kv(
                q,
                k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                window,
                softcap,
                alibi_slopes,
            ):
                launch_single_kv(
                    q, k, v, out, lse, cu_seqlens_q, seqused_k, block_table, scale
                )
                return (out, lse) if return_softmax_lse else out
            if supports_decode_tile(
                q, k, block_table, max_seqlen_q, window, softcap, alibi_slopes, plan
            ):
                launch_decode_tile(
                    q, k, v, out, lse, cu_seqlens_q, seqused_k, block_table, scale
                )
                return (out, lse) if return_softmax_lse else out
            if supports_cached(
                q, k, v, block_table, max_seqlen_q, window, softcap, alibi_slopes, plan
            ):
                launch_cached(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    scale,
                    causal,
                )
                return (out, lse) if return_softmax_lse else out
            launch = launch_splitkv if plan.num_splits > 1 else launch_direct
            launch(
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                scale,
                causal,
                window,
                softcap,
                plan,
            )
    return (out, lse) if return_softmax_lse else out
