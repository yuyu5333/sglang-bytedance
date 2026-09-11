"""Experimental two-launch, unsorted MoE for small unquantized decode batches."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(
    X,
    W,
    B,
    IDS,
    ACT,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    route = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    expert = tl.load(IDS + route).to(tl.int64)
    valid = (expert >= 0) & (expert < E)
    x = tl.load(X + (route // TOPK) * H + k, mask=k < H, other=0).to(tl.float32)
    offset = expert * (2 * I * H) + n[:, None] * H + k[None, :]
    mask = valid & (n[:, None] < I) & (k[None, :] < H)
    gate = tl.sum(tl.load(W + offset, mask=mask, other=0).to(tl.float32) * x, 1)
    up = tl.sum(tl.load(W + offset + I * H, mask=mask, other=0).to(tl.float32) * x, 1)
    if HAS_BIAS:
        gate += tl.load(B + expert * (2 * I) + n, mask=valid & (n < I), other=0)
        up += tl.load(B + expert * (2 * I) + I + n, mask=valid & (n < I), other=0)
    # Preserve the materialized GEMM1 rounding boundary before the activation.
    gate = gate.to(ACT.dtype.element_ty).to(tl.float32)
    up = up.to(ACT.dtype.element_ty).to(tl.float32)
    value = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(ACT + route * I + n, value, mask=n < I)


@triton.jit
def _down_sum(
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
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    t = tl.arange(0, BT)
    routes = token * TOPK + t
    expert = tl.load(IDS + routes, mask=t < TOPK, other=-1).to(tl.int64)
    valid = (t < TOPK) & (expert >= 0) & (expert < E)
    a = tl.load(
        ACT + routes[:, None] * I + k[None, :],
        mask=(t[:, None] < TOPK) & (k[None, :] < I),
        other=0,
    ).to(tl.float32)
    offset = expert[:, None, None] * (H * I) + n[None, :, None] * I + k[None, None, :]
    w = tl.load(
        W + offset,
        mask=valid[:, None, None] & (n[None, :, None] < H) & (k[None, None, :] < I),
        other=0,
    ).to(tl.float32)
    down = tl.sum(w * a[:, None, :], 2)
    if HAS_BIAS:
        down += tl.load(
            B + expert[:, None] * H + n[None, :],
            mask=valid[:, None] & (n[None, :] < H),
            other=0,
        )
    weight = tl.load(WEIGHTS + routes, mask=valid, other=0)
    # Round each weighted route as GEMM2 would, then reduce in FP32.
    down = (down * weight[:, None]).to(OUT.dtype.element_ty).to(tl.float32)
    result = tl.sum(down, 0) * SCALE
    tl.store(OUT + token * H + n, result, mask=n < H)


def direct_decode_supported(x, w1, w2, ids, weights, b1=None, b2=None):
    """Metadata-only capability check; routing values never leave the GPU."""
    if x.device.type != "cuda" or torch.version.hip is not None:
        return False
    if (
        x.ndim != 2
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not 1 <= x.shape[0] <= 8
        or not 128 <= x.shape[1] <= 8192
        or w1.ndim != 3
        or w2.ndim != 3
        or ids.ndim != 2
        or ids.shape != weights.shape
        or ids.shape[0] != x.shape[0]
        or not 1 <= ids.shape[1] <= 8
        or ids.dtype not in (torch.int32, torch.int64)
        or weights.dtype != torch.float32
    ):
        return False
    e, h, i = w2.shape
    if (
        e < ids.shape[1]
        or h != x.shape[1]
        or not 128 <= i <= 4096
        or w1.shape != (e, 2 * i, h)
        or w1.dtype != x.dtype
        or w2.dtype != x.dtype
    ):
        return False
    if any(
        t.device != x.device or not t.is_contiguous() for t in (x, w1, w2, ids, weights)
    ):
        return False
    for b, shape in ((b1, (e, 2 * i)), (b2, (e, h))):
        if b is not None and (
            b.shape != shape
            or b.device != x.device
            or b.dtype != x.dtype
            or not b.is_contiguous()
        ):
            return False
    return True


def direct_decode(
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
) -> torch.Tensor:
    """Canonical [gate;up] weights, FP16/BF16 rounding, no atomics or sorting.

    This changes GEMM accumulation order and is not a bitwise-equivalent path.
    The only scratch is the call-private [M, topk, I] activation. Inplace output
    is safe because kernel 2 consumes only that activation, not x.
    """
    if not direct_decode_supported(x, w1, w2, ids, weights, b1, b2):
        raise ValueError("Unsupported direct MoE decode metadata")
    m, h = x.shape
    e, _, i = w2.shape
    topk = ids.shape[1]
    act = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    out = x if inplace else torch.empty_like(x)
    with torch.cuda.device(x.device):
        _gate_up[(m * topk, triton.cdiv(i, 4))](
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
            BN=4,
            BK=triton.next_power_of_2(h),
            num_warps=4,
            enable_fp_fusion=False,
        )
        _down_sum[(m, triton.cdiv(h, 4))](
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
            BT=triton.next_power_of_2(topk),
            BN=4,
            BK=triton.next_power_of_2(i),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
