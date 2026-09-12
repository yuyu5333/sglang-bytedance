"""Experimental unsorted expert-local tensor-core MoE, not runner-selected."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.direct_decode import direct_decode_supported


@triton.jit
def _expert_rows(
    IDS,
    M: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BT: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(IDS + route).to(tl.int64)
    rows = tl.arange(0, BM)
    slots = tl.arange(0, BT)
    routes = rows[:, None] * TOPK + slots[None, :]
    ids = tl.load(
        IDS + routes, mask=(rows[:, None] < M) & (slots[None, :] < TOPK), other=-1
    )
    matches = (
        (expert >= 0)
        & (expert < E)
        & (ids == expert)
        & (rows[:, None] < M)
        & (slots[None, :] < TOPK)
    )
    first = tl.min(tl.where(matches, slots[None, :], TOPK), 1)
    leader = tl.min(tl.where(first < TOPK, rows * TOPK + first, M * TOPK), 0)
    return expert, route == leader, rows, first


@triton.jit
def _grouped_gate_up(
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
    BM: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    expert, leader, rows, first = _expert_rows(IDS, M, E, TOPK, BM, BT)
    if leader:
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        pair = tl.arange(0, 2 * BN)
        feature = tl.program_id(1) * BN + pair // 2
        weight_row = feature + (pair % 2) * I
        k = tl.arange(0, BK)
        active = (rows < M) & (first < TOPK)
        gate_up = tl.full((BM, 2 * BN), 0, tl.float32)
        for start in range(tl.cdiv(H, BK)):
            kk = start * BK + k
            x = tl.load(
                X + rows[:, None] * H + kk[None, :],
                mask=active[:, None] & (kk[None, :] < H),
                other=0,
            )
            offset = expert * (2 * I * H) + weight_row[None, :] * H + kk[:, None]
            mask = (feature[None, :] < I) & (kk[:, None] < H)
            w = tl.load(W + offset, mask=mask, other=0)
            gate_up = tl.dot(x, w, gate_up)
        if HAS_BIAS:
            gate_up += tl.load(
                B + expert * (2 * I) + weight_row, mask=feature < I, other=0
            )[None, :]
        gate_up = gate_up.to(ACT.dtype.element_ty).to(tl.float32)
        gate, up = tl.split(tl.reshape(gate_up, (BM, BN, 2)))
        value = gate / (1.0 + tl.exp(-gate)) * up
        # Duplicate routes share the first slot's activation, but not its weight.
        tl.store(
            ACT + (rows * TOPK + first)[:, None] * I + n[None, :],
            value,
            mask=active[:, None] & (n[None, :] < I),
        )


@triton.jit
def _grouped_down(
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
    BM: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    expert, leader, rows, first = _expert_rows(IDS, M, E, TOPK, BM, BT)
    if leader:
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        active = (rows < M) & (first < TOPK)
        down = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(I, BK)):
            kk = start * BK + k
            a = tl.load(
                ACT + (rows * TOPK + first)[:, None] * I + kk[None, :],
                mask=active[:, None] & (kk[None, :] < I),
                other=0,
            )
            w = tl.load(
                W + expert * (H * I) + n[None, :] * I + kk[:, None],
                mask=(n[None, :] < H) & (kk[:, None] < I),
                other=0,
            )
            down = tl.dot(a, w, down)
        if HAS_BIAS:
            down += tl.load(B + expert * H + n, mask=n < H, other=0)[None, :]
        for slot in tl.static_range(TOPK):
            routes = rows * TOPK + slot
            ids = tl.load(IDS + routes, mask=rows < M, other=-1)
            selected = active & (ids == expert)
            weight = tl.load(WEIGHTS + routes, mask=selected, other=0)
            tl.store(
                ROUTES + routes[:, None] * H + n[None, :],
                down * weight[:, None],
                mask=selected[:, None] & (n[None, :] < H),
            )


@triton.jit
def _grouped_sum(
    ROUTES,
    IDS,
    OUT,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    slots = tl.arange(0, BT)
    routes = token * TOPK + slots
    ids = tl.load(IDS + routes, mask=slots < TOPK, other=-1)
    valid = (slots < TOPK) & (ids >= 0) & (ids < E)
    # Invalid routes are never written or read; an all-masked token sums to zero.
    values = tl.load(
        ROUTES + routes[:, None] * H + n[None, :],
        mask=valid[:, None] & (n[None, :] < H),
        other=0,
    ).to(tl.float32)
    tl.store(OUT + token * H + n, tl.sum(values, 0) * SCALE, mask=n < H)


def grouped_decode(
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
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Three launches with GPU-only expert leaders and call-private scratch.

    A leader processes all tokens selecting its expert. The activation is stored
    only in each token's first matching slot; down emits every matching route.
    The final reduction masks invalid routes before reading uninitialized slots.
    FP16/BF16 rounding boundaries match direct_decode, not accumulation order.
    """
    if not direct_decode_supported(x, w1, w2, ids, weights, b1, b2):
        raise ValueError("Unsupported grouped MoE decode metadata")
    if block_n not in (16, 32, 64) or block_k not in (32, 64, 128, 256):
        raise ValueError("Unsupported grouped MoE tile")
    m, h = x.shape
    e, _, i = w2.shape
    topk = ids.shape[1]
    act = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    routes = torch.empty((m * topk, h), device=x.device, dtype=x.dtype)
    out = x if inplace else torch.empty_like(x)
    launch = dict(
        BM=16,
        BT=triton.next_power_of_2(topk),
        BN=block_n,
        BK=block_k,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    with torch.cuda.device(x.device):
        _grouped_gate_up[(m * topk, triton.cdiv(i, block_n))](
            x, w1, b1, ids, act, m, h, i, e, topk, b1 is not None, **launch
        )
        _grouped_down[(m * topk, triton.cdiv(h, block_n))](
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
            **launch,
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
