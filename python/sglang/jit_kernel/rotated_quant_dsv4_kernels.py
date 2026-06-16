"""DSv4 rotated-quant store/load helpers (Python orchestration over Triton).

This module is **M3.c.1**: wall-storage replacement for DSv4 nope.

Two host-side entry points used by ``RotatedQuantDeepSeekV4TokenToKVPool`` in
``mode='wall'``:

* :func:`rotated_store_to_packed` — given a BF16 input row ``[N, 512]``
  ``cat(nope, rope)`` plus per-layer ``(R, bits, scale, zero)``, produce the
  per-token packed bytes (nope-INT2/3/4 + rope-BF16) and scatter them into
  the paged ``cache`` buffer at the locations given by ``indices``.

* :func:`rotated_load_to_fp8_layout` — given a packed paged ``cache`` plus
  ``indices``, gather the requested rows, run inverse quant + inverse
  rotation on the GPU, then call :func:`rotated_dequant_to_fp8_layout` to
  emit the standard DSv4 ``[N, 576]`` slot bytes (and per-token ``[N, 8]``
  UE8M0 scale bytes) that FlashMLA expects. **M3.c.2** wires this into the
  attention prologue.

Layout of the packed paged cache (M3.c.1 v1):

    bytes_per_token = nope_packed_row_bytes + rope_bytes  # rope = 128 B
    bytes_per_page  = bytes_per_token * page_size

The page is **dense byte-packed** (no UE8M0 scale tail and no FP8 nope tail).
This is intentionally simpler than the upstream ``584 = 576 + 8`` layout
because the scale + R + zero come from the per-layer calib (host->device
constants) instead of being stored per-token.

Bit-pack/unpack reuse the M0 PyTorch reference implementation in
``rotated_kv_quant.bitpack_rowwise`` / ``bitunpack_rowwise``. The hot path
(Triton bit-pack/unpack) is M3.c.3.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.rotated_kv_quant import (
    RotatedQuantizerConfig,
    bitpack_rowwise,
    bitunpack_rowwise,
)

# Layout constants are duplicated here (instead of imported from
# triton_rotated_quant_dsv4) so CPU-only environments without Triton can use
# rotated_store_to_packed / rotated_load_to_fp8_layout_cpu_ref. The Triton
# kernel module is imported lazily inside rotated_load_to_fp8_layout.
_MLA_NOPE_DIM = 448
_MLA_TILE_SIZE = 64
_MLA_SLOT_BYTES = 576  # nope FP8 (448) + rope BF16 (128)
_MLA_SCALES_PER_TOKEN = 8

# Rope is always BF16 (2 bytes per element) regardless of the nope bit budget.
_ROPE_BYTES = _MLA_TILE_SIZE * 2  # 64 elements * 2 = 128 B


def packed_bytes_per_token(row_bytes_nope: int) -> int:
    """Total per-token byte count in the packed paged layout."""
    return int(row_bytes_nope) + _ROPE_BYTES


def _build_pack_meta_from_bits(
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build (dim_of_bit, bitpos_in_dim, row_bits, row_bytes) for Triton pack.

    Returns CPU tensors. Caller moves to device as needed.
    """
    bits_cpu = bits.cpu().to(torch.int32)
    D = int(bits_cpu.shape[0])
    bits_list = bits_cpu.tolist()
    row_bits = int(sum(bits_list))
    row_bytes = (row_bits + 7) // 8

    dim_of_bit = torch.zeros(row_bits, dtype=torch.int32)
    bitpos_in_dim = torch.zeros(row_bits, dtype=torch.int32)
    cursor = 0
    for d in range(D):
        b = bits_list[d]
        if b <= 0:
            continue
        dim_of_bit[cursor:cursor + b] = d
        bitpos_in_dim[cursor:cursor + b] = torch.arange(b, dtype=torch.int32)
        cursor += b
    return dim_of_bit, bitpos_in_dim, row_bits, row_bytes


# Per-config pack-meta cache keyed by device (small: ~448 int32 per dim).
_PACK_META_CACHE: dict[int, dict[str, object]] = {}


def _get_cached_pack_meta(
    cfg: RotatedQuantizerConfig, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (dim_of_bit_gpu, bitpos_in_dim_gpu) for this cfg+device."""
    key = id(cfg)
    entry = _PACK_META_CACHE.get(key)
    dev_key = str(device)
    if entry is not None and entry.get("device") == dev_key:
        return entry["dim_of_bit"], entry["bitpos_in_dim"]
    dim_of_bit, bitpos_in_dim, _rb, _rbytes = _build_pack_meta_from_bits(cfg.bits)
    if device and device.type == "cuda":
        dim_of_bit = dim_of_bit.to(device)
        bitpos_in_dim = bitpos_in_dim.to(device)
    _PACK_META_CACHE[key] = {
        "device": dev_key,
        "dim_of_bit": dim_of_bit,
        "bitpos_in_dim": bitpos_in_dim,
    }
    return dim_of_bit, bitpos_in_dim


# ----------------------------------------------------------------------
# Store path (write)
# ----------------------------------------------------------------------
def rotated_store_to_packed(
    input_bf16: torch.Tensor,  # [N, 512] BF16  (cat(nope, rope)
    cache: torch.Tensor,       # [num_pages, bytes_per_page] uint8
    indices: torch.Tensor,   # [N] int32, flat token-loc (page * page_size + slot
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> None:
    """Store rotated INT2/3/4 packed nope + raw BF16 rope into paged cache.

    Per-token bytes ``= row_bytes_nope + 128`` (rope BF16). GPU-only:
    ``cache[page, slot*Bpt : slot*Bpt + row_bytes_nope]`` is the packed
    nope, followed by ``128 B`` of raw BF16 rope.

    ``T3 path (Triton)：整个路径完全在 GPU 上： rotate + affine + quantise + clamp +
    bitpack，全部不走 CPU 不做任何 H2D/D2H 往返。
    """
    if input_bf16.dtype != torch.bfloat16:
        raise ValueError(f"input must be bf16, got {input_bf16.dtype}")
    if input_bf16.shape[-1] != _MLA_NOPE_DIM + _MLA_TILE_SIZE:
        raise ValueError(
            f"input last dim {input_bf16.shape[-1]} != {_MLA_NOPE_DIM + _MLA_TILE_SIZE}"
        )
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must be uint8, got {cache.dtype}")
    if cache.dim() != 2:
        raise ValueError(f"cache must be 2-D, got shape {tuple(cache.shape)}")

    N = input_bf16.shape[0]
    if N == 0:
        return
    row_bytes_nope = cfg.row_bytes
    bpt = packed_bytes_per_token(row_bytes_nope)
    bytes_per_page = cache.shape[1]
    if bytes_per_page != bpt * page_size:
        raise ValueError(
            f"cache bytes_per_page {bytes_per_page} != bpt({bpt}) * page_size({page_size})"
        )
    if indices.shape != (N,):
        raise ValueError(f"indices shape {tuple(indices.shape)} != ({N},)")
    indices_i64 = indices.to(torch.int64)

    nope = input_bf16[:, :_MLA_NOPE_DIM].contiguous()
    rope = input_bf16[:, _MLA_NOPE_DIM:].contiguous()

    # --- T3: 整个 store 全 GPU 路径
    #   1) rotate + affine quant + round + clamp 全走 GPU torch
    #   2) bitpack 走 Triton kernel (向量化位合并)
    R = cfg.R.to(device=input_bf16.device, dtype=torch.float32)
    scale = cfg.scale.to(device=input_bf16.device, dtype=torch.float32).clamp_min(1e-12)
    zero = cfg.zero.to(device=input_bf16.device, dtype=torch.float32)
    bits_gpu = cfg.bits.to(device=input_bf16.device, dtype=torch.int32)
    # levels = 2^bits - 1，用于 quantise 后的饱和值；留在 GPU
    levels_gpu = (
        torch.ones_like(bits_gpu, dtype=torch.int64) << bits_gpu.to(torch.int64)
    ) - 1

    K_rot = nope.to(torch.float32) @ R  # [N, 448]
    codes_f = ((K_rot - zero) / scale).round()
    codes_f = torch.clamp(
        codes_f, min=torch.zeros_like(codes_f), max=levels_gpu.to(codes_f.dtype)
    )
    codes_i32 = codes_f.to(torch.int32)  # 留在 GPU

    # --- Triton bitpack（T3）。
    dim_of_bit, bitpos_in_dim = _get_cached_pack_meta(cfg, input_bf16.device)
    from sglang.jit_kernel.triton_rotated_quant_dsv4 import triton_bitpack_rowwise
    packed = triton_bitpack_rowwise(
        codes_i32, dim_of_bit, bitpos_in_dim, cfg.row_bytes,
    )

    # Compose the [N, bpt] row on GPU (纯 view/cat，无 H2D)
    rope_bytes = rope.contiguous().view(torch.uint8).reshape(N, _ROPE_BYTES)
    full_row = torch.cat([packed, rope_bytes], dim=1)  # [N, bpt]

    # Scatter into paged cache:
    # 必须过滤 -1 sentinel（translate_loc_from_full_to_swa 对未映射槽返回 -1）
    cache_flat = cache.view(-1, bpt)
    idx_gpu = indices_i64.to(cache.device)
    if idx_gpu.numel() > 0:
        valid_mask = idx_gpu >= 0
        max_allowed = cache_flat.shape[0] - 1
        if not valid_mask.all():
            # 仅保留合法行再 scatter，避免 index_copy_ 触发设备端断言
            valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            idx_valid = idx_gpu[valid_idx]
            row_valid = full_row[valid_idx]
            # 附加安全钳位：如果 idx 中有超范围值（防御性编程）
            idx_clamped = idx_valid.clamp(min=0, max=max_allowed)
            cache_flat.index_copy_(0, idx_clamped, row_valid)
        else:
            idx_clamped = idx_gpu.clamp(min=0, max=max_allowed)
            cache_flat.index_copy_(0, idx_clamped, full_row)


# ----------------------------------------------------------------------
# Load path (read, used by attention prologue in M3.c.2)
# ----------------------------------------------------------------------
def rotated_load_to_fp8_layout(
    cache: torch.Tensor,       # [num_pages, bytes_per_page] uint8
    indices: torch.Tensor,     # [M] int32 flat token-locs to gather
    out_slot: torch.Tensor,    # [M, 576] uint8  (output, FlashMLA-compatible slot bytes)
    out_scale: torch.Tensor,   # [M, 8]   uint8  (output, UE8M0 per nope tile + 1 pad)
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> None:
    """Gather + dequant + inverse-rotate + emit standard FP8 slot bytes.

    Used by the M3.c.2 attention prologue: M attention-relevant token locs
    get expanded into FlashMLA's expected ``[M, 576]`` slot byte buffer plus
    ``[M, 8]`` UE8M0 scale bytes.
    """
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must be uint8, got {cache.dtype}")
    if out_slot.dtype != torch.uint8:
        raise ValueError(f"out_slot must be uint8, got {out_slot.dtype}")
    if out_scale.dtype != torch.uint8:
        raise ValueError(f"out_scale must be uint8, got {out_scale.dtype}")

    M = indices.shape[0]
    if M == 0:
        return
    if out_slot.shape != (M, _MLA_SLOT_BYTES):
        raise ValueError(
            f"out_slot shape {tuple(out_slot.shape)} != ({M},{_MLA_SLOT_BYTES})"
        )
    if out_scale.shape != (M, _MLA_SCALES_PER_TOKEN):
        raise ValueError(
            f"out_scale shape {tuple(out_scale.shape)} != ({M},{_MLA_SCALES_PER_TOKEN})"
        )

    row_bytes_nope = cfg.row_bytes
    bpt = packed_bytes_per_token(row_bytes_nope)
    if cache.shape[1] != bpt * page_size:
        raise ValueError(
            f"cache bytes_per_page {cache.shape[1]} != bpt({bpt}) * page_size({page_size})"
        )

    indices_i64 = indices.to(device=cache.device, dtype=torch.int64)
    cache_flat = cache.view(-1, bpt)  # [num_pages * page_size, bpt]
    rows = cache_flat.index_select(0, indices_i64)  # [M, bpt]
    packed_nope = rows[:, :row_bytes_nope].contiguous()  # uint8 [M, row_bytes_nope]
    rope_bytes = rows[:, row_bytes_nope:].contiguous()   # uint8 [M, 128]

    # --- T3: store 用 Triton bitpack（瓶颈），load 用 CPU bitunpack 缓解 GPU OOM
    # packed_nope 很小（uint8 [M, ~48]），CPU bitunpack 压力在 CPU RAM，
    # 刚好避免 GPU 上额外分配 200+ MiB 的 codes/codes_i32 中间张量。
    bits_cpu = cfg.bits.to(device="cpu", dtype=torch.int32)
    codes = bitunpack_rowwise(packed_nope.cpu(), bits_cpu, dim=_MLA_NOPE_DIM)
    codes = codes.to(device=cache.device, dtype=torch.float32)

    scale = cfg.scale.to(device=cache.device, dtype=torch.float32)
    zero = cfg.zero.to(device=cache.device, dtype=torch.float32)
    K_rot_hat = codes * scale + zero  # [M, 448]
    R = cfg.R.to(device=cache.device, dtype=torch.float32)
    nope_bf16 = (K_rot_hat @ R.t()).to(torch.bfloat16).contiguous()
    rope_bf16 = rope_bytes.view(torch.bfloat16).reshape(M, _MLA_TILE_SIZE).contiguous()

    # FP8 layout Triton kernel（GPU 端）
    from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
        rotated_dequant_to_fp8_layout,
    )

    rotated_dequant_to_fp8_layout(nope_bf16, rope_bf16, out_slot, out_scale)


# ----------------------------------------------------------------------
# CPU reference for unit tests (no GPU, no Triton)
# ----------------------------------------------------------------------
def rotated_load_to_fp8_layout_cpu_ref(
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-CPU reference: returns ``(nope_bf16_recon[M,448],
    rope_bf16[M,64], packed_bytes[M,row_bytes])``.

    Does **not** run the FP8 layout Triton kernel (which requires Triton +
    a CUDA device). Used by canary unit tests on CPU-only environments.
    """
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must be uint8, got {cache.dtype}")
    M = indices.shape[0]
    row_bytes_nope = cfg.row_bytes
    bpt = packed_bytes_per_token(row_bytes_nope)
    if cache.shape[1] != bpt * page_size:
        raise ValueError(
            f"cache bytes_per_page {cache.shape[1]} != bpt({bpt}) * page_size({page_size})"
        )

    indices_i64 = indices.to(torch.int64)
    cache_flat = cache.view(-1, bpt)
    rows = cache_flat.index_select(0, indices_i64)
    packed_nope = rows[:, :row_bytes_nope].contiguous()
    rope_bytes = rows[:, row_bytes_nope:].contiguous()

    bits_cpu = cfg.bits.to(torch.int32)
    codes = bitunpack_rowwise(packed_nope, bits_cpu, dim=_MLA_NOPE_DIM)
    K_rot_hat = codes.to(torch.float32) * cfg.scale + cfg.zero
    nope_bf16 = (K_rot_hat @ cfg.R.t().to(torch.float32)).to(torch.bfloat16).contiguous()
    rope_bf16 = rope_bytes.view(torch.bfloat16).reshape(M, _MLA_TILE_SIZE).contiguous()
    return nope_bf16, rope_bf16, packed_nope


__all__ = [
    "rotated_store_to_packed",
    "rotated_load_to_fp8_layout",
    "rotated_load_to_fp8_layout_cpu_ref",
    "quant_fp8_layout_cpu_ref",
    "packed_bytes_per_token",
]


# ----------------------------------------------------------------------
# CPU reference for the dequant->FP8 layout step (used by canary tests
# and by the pool's wall-mode prologue when no Triton runtime is present).
# ----------------------------------------------------------------------
def quant_fp8_layout_cpu_ref(
    nope_bf16: torch.Tensor,  # [M, 448] bf16
    rope_bf16: torch.Tensor,  # [M, 64]  bf16
):
    """Pure-PyTorch reference matching ``rotated_dequant_to_fp8_layout``.

    Layout (DSv4 FlashMLA-compatible):

      out_slot[m, 0:448]   = FP8 (E4M3) bytes of nope, per-tile UE8M0 scaling
      out_slot[m, 448:576] = raw BF16 bytes of rope (128 B)
      out_scale[m, 0:7]    = UE8M0 exponent bytes (one per 64-element tile)
      out_scale[m, 7]      = pad (0)

    UE8M0 scale = clip(ceil(log2(max(|x|) / FP8_E4M3_MAX)) + 127, 0, 255).
    Quantised value = round(x / 2^(scale-127)).
    """
    if nope_bf16.dim() != 2 or nope_bf16.shape[-1] != _MLA_NOPE_DIM:
        raise ValueError(
            f"nope_bf16 shape {tuple(nope_bf16.shape)} != (M, {_MLA_NOPE_DIM})"
        )
    if rope_bf16.dim() != 2 or rope_bf16.shape[-1] != _MLA_TILE_SIZE:
        raise ValueError(
            f"rope_bf16 shape {tuple(rope_bf16.shape)} != (M, {_MLA_TILE_SIZE})"
        )
    if nope_bf16.shape[0] != rope_bf16.shape[0]:
        raise ValueError("nope/rope batch mismatch")

    M = nope_bf16.shape[0]
    device = nope_bf16.device
    out_slot = torch.zeros((M, _MLA_SLOT_BYTES), dtype=torch.uint8, device=device)
    out_scale = torch.zeros(
        (M, _MLA_SCALES_PER_TOKEN), dtype=torch.uint8, device=device
    )

    nope_f = nope_bf16.to(torch.float32)
    num_tiles = _MLA_NOPE_DIM // _MLA_TILE_SIZE  # 7
    tiles = nope_f.reshape(M, num_tiles, _MLA_TILE_SIZE)
    abs_max = tiles.abs().amax(dim=-1).clamp_min(1e-4)
    fp8_max = 448.0
    scale_raw = abs_max / fp8_max
    log2_scale = torch.ceil(torch.log2(scale_raw))
    ue8m0 = (log2_scale + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    inv_scale = torch.pow(2.0, -(ue8m0.to(torch.float32) - 127.0))
    quantised = (tiles * inv_scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    fp8 = quantised.to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8).reshape(M, _MLA_NOPE_DIM)
    out_slot[:, :_MLA_NOPE_DIM] = fp8_bytes

    rope_bytes = rope_bf16.contiguous().view(torch.uint8).reshape(M, _ROPE_BYTES)
    out_slot[:, _MLA_NOPE_DIM:_MLA_SLOT_BYTES] = rope_bytes

    out_scale[:, :num_tiles] = ue8m0
    return out_slot, out_scale
