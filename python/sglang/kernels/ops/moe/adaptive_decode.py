"""Experimental GPU-only per-expert SIMT/tensor-core decode selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.direct_decode import _gate_up, direct_decode_supported
from sglang.kernels.ops.moe.grouped_decode import (
    _expert_rows,
    _grouped_down,
    _grouped_gate_up,
    _grouped_sum,
)


@triton.jit
def _use_grouped(
    IDS,
    M: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BT: tl.constexpr,
    MIN_REUSE: tl.constexpr,
):
    if M >= MIN_REUSE:
        expert, leader, rows, first = _expert_rows(IDS, M, E, TOPK, 16, BT)
        # Count distinct tokens, not duplicate slots in the same token.
        return tl.sum(((rows < M) & (first < TOPK)).to(tl.int32), 0) >= MIN_REUSE
    else:
        return False


@triton.jit
def _adaptive_gate_up(
    X,
    W,
    B,
    IDS,
    ACT,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
    MIN_REUSE: tl.constexpr,
):
    if _use_grouped(IDS, M, E, TOPK, BT, MIN_REUSE):
        if tl.program_id(1) < tl.cdiv(I, GN):
            _grouped_gate_up(
                X,
                W,
                B,
                IDS,
                ACT,
                M,
                H,
                I,
                E,
                TOPK,
                HAS_BIAS,
                16,
                BT,
                GN,
                GK,
            )
    else:
        if tl.program_id(1) < tl.cdiv(I, SN):
            _gate_up(X, W, B, IDS, ACT, H, I, E, TOPK, HAS_BIAS, SN, SK)


@triton.jit
def _adaptive_down(
    ACT,
    W,
    B,
    IDS,
    WEIGHTS,
    ROUTES,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
    MIN_REUSE: tl.constexpr,
):
    if _use_grouped(IDS, M, E, TOPK, BT, MIN_REUSE):
        if tl.program_id(1) < tl.cdiv(H, GN):
            _grouped_down(
                ACT,
                W,
                B,
                IDS,
                WEIGHTS,
                ROUTES,
                M,
                H,
                I,
                E,
                TOPK,
                HAS_BIAS,
                16,
                BT,
                GN,
                GK,
            )
    else:
        route = tl.program_id(0)
        expert = tl.load(IDS + route).to(tl.int64)
        if (expert >= 0) & (expert < E) & (tl.program_id(1) < tl.cdiv(H, SN)):
            n = tl.program_id(1) * SN + tl.arange(0, SN)
            k = tl.arange(0, SK)
            a = tl.load(ACT + route * I + k, mask=k < I, other=0).to(tl.float32)
            w = tl.load(
                W + expert * (H * I) + n[:, None] * I + k[None, :],
                mask=(n[:, None] < H) & (k[None, :] < I),
                other=0,
            ).to(tl.float32)
            down = tl.sum(w * a[None, :], 1)
            if HAS_BIAS:
                down += tl.load(B + expert * H + n, mask=n < H, other=0)
            weight = tl.load(WEIGHTS + route)
            tl.store(ROUTES + route * H + n, down * weight, mask=n < H)


def adaptive_decode(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    scale: float = 1.0,
    inplace: bool = False,
    min_reuse: int = 4,
    simt_n: int = 16,
    group_n: int = 32,
    group_k: int = 256,
) -> torch.Tensor:
    """Three launches, no host routing decision, no atomic output accumulation.

    Experts shared by min_reuse distinct tokens use the grouped implementation.
    Others use route-local SIMT. Both materialize weighted route outputs before
    the same final sum. Threshold 9 forces SIMT for the supported M1..8 range.
    """
    if not direct_decode_supported(x, w1, w2, ids, weights, b1, b2):
        raise ValueError("Unsupported adaptive MoE decode metadata")
    if (
        min_reuse not in range(1, 10)
        or simt_n not in (4, 8, 16)
        or group_n not in (16, 32, 64)
        or group_k not in (32, 64, 128, 256)
    ):
        raise ValueError("Unsupported adaptive MoE launch configuration")
    m, h = x.shape
    e, _, i = w2.shape
    topk = ids.shape[1]
    act = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    routes = torch.empty((m * topk, h), device=x.device, dtype=x.dtype)
    out = x if inplace else torch.empty_like(x)
    options = dict(
        BT=triton.next_power_of_2(topk),
        SN=simt_n,
        GN=group_n,
        GK=group_k,
        MIN_REUSE=min_reuse,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    with torch.cuda.device(x.device):
        _adaptive_gate_up[(m * topk, triton.cdiv(i, min(simt_n, group_n)))](
            x,
            w1,
            b1,
            ids,
            act,
            m,
            h,
            i,
            e,
            topk,
            b1 is not None,
            SK=triton.next_power_of_2(h),
            **options,
        )
        _adaptive_down[(m * topk, triton.cdiv(h, min(simt_n, group_n)))](
            act,
            w2,
            b2,
            ids,
            weights,
            routes,
            m,
            h,
            i,
            e,
            topk,
            b2 is not None,
            SK=triton.next_power_of_2(i),
            **options,
        )
        _grouped_sum[(m, triton.cdiv(h, 256))](
            routes,
            ids,
            out,
            h,
            e,
            topk,
            scale,
            BT=triton.next_power_of_2(topk),
            BN=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
