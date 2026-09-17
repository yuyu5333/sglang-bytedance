"""Gather packed Main KV with the same FP8 rounding as the V4 cache writer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import _e2m1_code_to_fp32
from sglang.srt.mem_cache.dsv41_main_kv_layout import PackedMainKVView
from sglang.srt.mem_cache.dsv41_staging_workspace import (
    MainKVStagingWorkspace,
)


@triton.jit
def _load_legacy_nope(src, loc, valid, tile, PAGE_SLOTS: tl.constexpr):
    """One 64-channel group, including the intermediate BF16 store."""
    page = loc // PAGE_SLOTS
    slot = loc % PAGE_SLOTS
    base = page * (384 * PAGE_SLOTS)
    ch = tile * 64 + tl.arange(0, 64)
    packed = tl.load(src + base + slot * 224 + ch // 2, valid, other=0)
    code = (packed >> ((ch & 1) * 4)) & 15
    scale = tl.load(
        (src + base + PAGE_SLOTS * 224 + slot * 32 + ch // 16).to(
            tl.pointer_type(tl.float8e4nv)
        ),
        valid,
        other=0.0,
    ).to(tl.float32)
    value = (_e2m1_code_to_fp32(code) * scale).to(tl.bfloat16).to(tl.float32)
    # store.cuh uses max(amax, 1e-4) / 448, NOT max(amax / 448, 1e-4).
    raw_scale = tl.div_rn(tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4), 448.0)
    bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    fp8 = tl.clamp(value * inverse, -448.0, 448.0).to(tl.float8e4nv)
    return fp8, exponent


@triton.jit
def _stage_kernel(
    src,
    indices,
    lengths,
    dst,
    remap,
    INDEX_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    RESERVED_WIDTH: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    PAGE_SLOTS: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    DST_PAGE_BYTES: tl.constexpr,
):
    q = tl.program_id(0)
    j = tl.program_id(1)
    loc = tl.load(indices + q * INDEX_STRIDE + j).to(tl.int64)
    valid = (loc >= 0) & (loc < NUM_SLOTS)
    if HAS_LENGTHS:
        valid = valid & (j < tl.load(lengths + q * LENGTH_STRIDE))
    out_slot = q.to(tl.int64) * RESERVED_WIDTH + j
    tl.store(remap + out_slot, tl.where(valid, out_slot, -1).to(tl.int32))
    dst_page = out_slot // 256 * DST_PAGE_BYTES
    dst_slot = out_slot % 256
    for tile in tl.static_range(7):
        value, exponent = _load_legacy_nope(src, loc, valid, tile, PAGE_SLOTS)
        tl.store(
            (dst + dst_page + dst_slot * 576 + tile * 64 + tl.arange(0, 64)).to(
                tl.pointer_type(tl.float8e4nv)
            ),
            value,
        )
        tl.store(dst + dst_page + 256 * 576 + dst_slot * 8 + tile, exponent)
    tl.store(dst + dst_page + 256 * 576 + dst_slot * 8 + 7, 0)
    rope = tl.load(
        (src + loc // PAGE_SLOTS * (384 * PAGE_SLOTS) + 256 * PAGE_SLOTS).to(
            tl.pointer_type(tl.bfloat16)
        )
        + loc % PAGE_SLOTS * 64
        + tl.arange(0, 64),
        valid,
        other=0,
    )
    tl.store(
        (dst + dst_page + dst_slot * 576 + 448).to(tl.pointer_type(tl.bfloat16))
        + tl.arange(0, 64),
        rope,
    )


@triton.jit
def _gather_kernel(
    src,
    ids,
    out,
    OUT_STRIDE: tl.constexpr,
    ID_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    PAGE_SLOTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    loc = tl.load(ids + row * ID_STRIDE).to(tl.int64)
    valid = (loc >= 0) & (loc < NUM_SLOTS)
    for tile in tl.static_range(7):
        value, exponent = _load_legacy_nope(src, loc, valid, tile, PAGE_SLOTS)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        # FP8 output matches the existing Q8KV8 gather; BF16 output matches V4 dequant.
        dequantized = value.to(tl.float32) * scale
        tl.store(out + row * OUT_STRIDE + tile * 64 + tl.arange(0, 64), dequantized)
    rope = tl.load(
        (src + loc // PAGE_SLOTS * (384 * PAGE_SLOTS) + 256 * PAGE_SLOTS).to(
            tl.pointer_type(tl.bfloat16)
        )
        + loc % PAGE_SLOTS * 64
        + tl.arange(0, 64),
        valid,
        other=0,
    )
    tl.store(out + row * OUT_STRIDE + 448 + tl.arange(0, 64), rope)


def _validate_indices(view: PackedMainKVView, indices: torch.Tensor) -> None:
    if not view.storage.is_cuda or torch.version.cuda is None:
        raise ValueError("packed Main KV staging requires NVIDIA CUDA")
    if indices.device != view.storage.device or indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("indices must be int32/int64 on the cache device")


def stage_packed_main_kv(
    view: PackedMainKVView,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    workspace: MainKVStagingWorkspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage [query_rows, selected_width] without modifying source metadata."""
    _validate_indices(view, indices)
    if indices.ndim != 2 or indices.stride(-1) != 1:
        raise ValueError("staging indices must be 2D with contiguous selected rows")
    rows, width = indices.shape
    workspace.check_shape(rows, width)
    if workspace.pages.device != view.storage.device:
        raise ValueError("workspace and cache must be on the same device")
    if lengths is not None and (
        lengths.shape != (rows,)
        or lengths.dtype != torch.int32
        or lengths.device != indices.device
    ):
        raise ValueError("lengths must be int32 [query_rows] on the cache device")
    if rows and width:
        _stage_kernel[(rows, width)](
            view.storage,
            indices,
            lengths,
            workspace.pages,
            workspace.indices,
            indices.stride(0),
            width,
            workspace.width,
            view.storage.shape[0] * view.spec.page_slots,
            view.spec.page_slots,
            lengths is not None,
            lengths.stride(0) if lengths is not None else 1,
            workspace.pages.stride(0),
            num_warps=4,
        )
    return workspace.cache, workspace.indices[:rows, :width]


def gather_packed_main_kv(
    view: PackedMainKVView, token_ids: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Write legacy-compatible BF16/FP8 values into preallocated prefill scratch."""
    _validate_indices(view, token_ids)
    if token_ids.ndim != 1:
        raise ValueError("positional gather expects one-dimensional token IDs")
    if (
        out.shape != (token_ids.numel(), 1, 512)
        or out.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or out.device != view.storage.device
        or out.stride(-1) != 1
    ):
        raise ValueError("gather output must be BF16/FP8 [num_tokens, 1, 512]")
    if token_ids.numel():
        _gather_kernel[(token_ids.numel(),)](
            view.storage,
            token_ids,
            out,
            out.stride(0),
            token_ids.stride(0),
            view.storage.shape[0] * view.spec.page_slots,
            view.spec.page_slots,
            num_warps=4,
        )
    return out
