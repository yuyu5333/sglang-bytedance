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


# ----------------------------------------------------------------------
# Store path (write)
# ----------------------------------------------------------------------
def rotated_store_to_packed(
    input_bf16: torch.Tensor,  # [N, 512] BF16  (cat(nope, rope))
    cache: torch.Tensor,       # [num_pages, bytes_per_page] uint8
    indices: torch.Tensor,     # [N] int32, flat token-loc (page * page_size + slot)
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> None:
    """Store rotated INT2/3/4 packed nope + raw BF16 rope into paged cache.

    Per-token bytes ``= row_bytes_nope + 128`` (rope BF16). The whole row is
    written contiguously: ``cache[page, slot*Bpt : slot*Bpt + row_bytes_nope]``
    is the packed nope, followed by ``128 B`` of raw BF16 rope.
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

    # Rotate + affine quantise + clamp on the same device as `input_bf16`.
    R = cfg.R.to(device=input_bf16.device, dtype=torch.float32)
    scale = cfg.scale.to(device=input_bf16.device, dtype=torch.float32).clamp_min(1e-12)
    zero = cfg.zero.to(device=input_bf16.device, dtype=torch.float32)
    bits = cfg.bits.to(device="cpu", dtype=torch.int32)  # bitpack runs on CPU
    levels = (1 << bits.to(torch.int64)) - 1  # [D] cpu

    K_rot = nope.to(torch.float32) @ R  # [N, 448]
    codes_f = ((K_rot - zero) / scale).round()
    codes_f = torch.clamp(
        codes_f, min=torch.zeros_like(codes_f), max=levels.to(K_rot.device).to(K_rot.dtype)
    )
    codes_i64 = codes_f.to(torch.int64).cpu()  # [N, 448] cpu

    packed = bitpack_rowwise(codes_i64, bits)  # [N, row_bytes_nope] uint8 cpu
    packed = packed.to(device=input_bf16.device)

    # Compose the [N, bpt] row by concatenating packed nope + raw rope bytes.
    rope_bytes = rope.contiguous().view(torch.uint8).reshape(N, _ROPE_BYTES)
    full_row = torch.cat([packed, rope_bytes], dim=1)  # [N, bpt]

    # Scatter into paged cache: cache.view(-1, bpt)[loc] = full_row.
    # cache has shape [num_pages, bytes_per_page = bpt * page_size]; the flat
    # token-level row index is just `indices` (page * page_size + slot).
    cache_flat = cache.view(-1, bpt)  # [num_pages * page_size, bpt]
    cache_flat.index_copy_(0, indices_i64.to(cache.device), full_row)


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

    # Bit-unpack + inverse affine on CPU (M3.c.1 reference; M3.c.3 -> Triton).
    bits_cpu = cfg.bits.to(device="cpu", dtype=torch.int32)
    codes = bitunpack_rowwise(packed_nope.cpu(), bits_cpu, dim=_MLA_NOPE_DIM)
    codes = codes.to(device=cache.device, dtype=torch.float32)

    scale = cfg.scale.to(device=cache.device, dtype=torch.float32)
    zero = cfg.zero.to(device=cache.device, dtype=torch.float32)
    K_rot_hat = codes * scale + zero  # [M, 448]
    R = cfg.R.to(device=cache.device, dtype=torch.float32)
    nope_bf16 = (K_rot_hat @ R.t()).to(torch.bfloat16).contiguous()
    rope_bf16 = rope_bytes.view(torch.bfloat16).reshape(M, _MLA_TILE_SIZE).contiguous()

    # Triton import is lazy: CPU-only environments still get the writer +
    # cpu_ref reader without paying the import cost.
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
