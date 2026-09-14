"""Experimental output-owned SIMT decode with bounded live route state."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.direct_decode import (
    _down_sum,
    _gate_up,
    direct_decode_supported,
)


@triton.jit
def _streamed_down_sum(
    ACT,
    W,
    B,
    IDS,
    WEIGHTS,
    OUT,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    result = tl.full((BN,), 0, tl.float32)
    # Keep one route's weight tile live, rather than a padded top-k dimension.
    for slot in tl.range(0, TOPK, loop_unroll_factor=UNROLL):
        route = token * TOPK + slot
        expert = tl.load(IDS + route).to(tl.int64)
        valid = (expert >= 0) & (expert < E)
        a = tl.load(ACT + route * I + k, mask=valid & (k < I), other=0).to(tl.float32)
        w = tl.load(
            W + expert * (H * I) + n[:, None] * I + k[None, :],
            mask=valid & (n[:, None] < H) & (k[None, :] < I),
            other=0,
        ).to(tl.float32)
        down = tl.sum(w * a[None, :], 1)
        if HAS_BIAS:
            down += tl.load(B + expert * H + n, mask=valid & (n < H), other=0)
        weight = tl.load(WEIGHTS + route, mask=valid, other=0)
        result += (down * weight).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + token * H + n, result * SCALE, mask=n < H)


def streamed_decode(
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
    up_n: int = 4,
    down_n: int = 16,
    up_warps: int = 4,
    down_warps: int = 4,
    unroll: int = 1,
    vectorized_down: bool = False,
) -> torch.Tensor:
    """Two launches, call-private activation, no routing readback or atomics.

    The vectorized variant reuses the previous kernel for tiling ablation.
    FP16/BF16 materialization boundaries remain; reduction order can differ.
    No runner selects this experiment automatically.
    """
    if not direct_decode_supported(x, w1, w2, ids, weights, b1, b2):
        raise ValueError("Unsupported streamed MoE decode metadata")
    if (
        up_n not in (2, 4, 8, 16)
        or down_n not in (2, 4, 8, 16, 32, 64)
        or up_warps not in (4, 8)
        or down_warps not in (4, 8)
        or unroll not in (1, 2, 4, 8)
    ):
        raise ValueError("Unsupported streamed MoE launch configuration")
    m, h = x.shape
    e, _, i = w2.shape
    topk = ids.shape[1]
    act = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    out = x if inplace else torch.empty_like(x)
    with torch.cuda.device(x.device):
        _gate_up[(m * topk, triton.cdiv(i, up_n))](
            x,
            w1,
            b1,
            ids,
            act,
            h,
            i,
            e,
            topk,
            b1 is not None,
            BN=up_n,
            BK=triton.next_power_of_2(h),
            num_warps=up_warps,
            enable_fp_fusion=False,
        )
        kernel = _down_sum if vectorized_down else _streamed_down_sum
        options = (
            dict(BT=triton.next_power_of_2(topk))
            if vectorized_down
            else dict(UNROLL=unroll)
        )
        kernel[(m, triton.cdiv(h, down_n))](
            act,
            w2,
            b2,
            ids,
            weights,
            out,
            h,
            i,
            e,
            topk,
            scale,
            b2 is not None,
            BN=down_n,
            BK=triton.next_power_of_2(i),
            num_warps=down_warps,
            enable_fp_fusion=False,
            **options,
        )
    return out
