"""DSv4 rotated-quant <-> standard FP8 layout shim (Triton).

This module is **M3.c.1**: it provides fused Triton kernels that take a
batch of BF16 nope vectors (already inverse-rotated back to the original
domain on the host via ``K_rot_hat @ R^T``) plus their BF16 rope tail, and
write them into a contiguous ``[N, 576]`` byte buffer in **exactly the
DSv4 standard layout** that FlashMLA expects:

    per-token slot (576 B) = nope FP8 (448 B, 7 UE8M0 tiles of 64 e4m3fn)
                           + rope BF16 (128 B, 64 elements)

Per-page UE8M0 scale bytes (8 B per token, in a separate page region) are
written into a parallel ``[N, 8]`` byte buffer. The caller is responsible
for tiling these two buffers into the page format used by FlashMLA when the
M3.c.2 attention shim is wired in. For M3.c.1 we expose a flat ``[N, 576]``
+ ``[N, 8]`` interface so unit tests can validate numerics without any
paging.

Why a dedicated kernel: the round-trip
``packed -> bitunpack -> affine -> @R^T -> ue8m0_fp8 -> 576B`` must be
bit-identical with the ground-truth ``triton_fused_store_flashmla`` write
path (the same UE8M0 cast / inv-scale + clamp). We therefore mirror that
kernel's tile structure (7 nope tiles of 64 e4m3 + 1 rope tile) and reuse
its scale-emission semantics.

Two-kernel design: nope tile kernel writes into ``out_slot.view(_FP8_DTYPE)``
and the per-tile UE8M0 scale byte; rope kernel writes BF16 into
``out_slot.view(torch.bfloat16)``. Splitting keeps Triton's pointer-type
contract clean (one element_ty per kernel).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)

# Mirrors triton_store_cache.py for binary compatibility with FlashMLA.
_MLA_NOPE_DIM = 448
_MLA_HEAD_DIM = 512  # nope (448) + rope (64)
_MLA_TILE_SIZE = 64
_MLA_NUM_NOPE_TILES = 7
_MLA_SLOT_BYTES = 576  # nope_fp8 (448) + rope_bf16 (128)
_MLA_SCALES_PER_TOKEN = 8  # 7 nope tiles + 1 padding (must remain 8 for layout)
_UE8M0_EXPONENT_BIAS = 127


@triton.jit
def _rotated_dequant_nope_kernel(
    nope_bf16_ptr,    # [N, 448] BF16
    out_slot_fp8_ptr, # [N, 576] viewed as e4m3fn (1B per element)
    out_scale_ptr,    # [N, 8]   uint8
    N,
    TILE_SIZE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    SCALES_PER_TOKEN: tl.constexpr,
    UE8M0_BIAS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    if token_id >= N:
        return

    lane = tl.arange(0, TILE_SIZE)
    x_bf16 = tl.load(
        nope_bf16_ptr + token_id * NOPE_DIM + tile_id * TILE_SIZE + lane
    )
    x_fp32 = x_bf16.to(tl.float32)

    abs_max = tl.max(tl.abs(x_fp32))
    scale = tl.maximum(abs_max, EPS) / FP8_MAX

    log2_scale = tl.log2(scale)
    ceil_log2 = tl.math.ceil(log2_scale)
    inv_scale = tl.exp2(-ceil_log2)

    x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
        out_slot_fp8_ptr.dtype.element_ty
    )

    nope_offset = token_id * SLOT_BYTES + tile_id * TILE_SIZE + lane
    tl.store(out_slot_fp8_ptr + nope_offset, x_fp8)

    ue8m0 = (ceil_log2.to(tl.int32) + UE8M0_BIAS).to(tl.uint8)
    scale_offset = token_id * SCALES_PER_TOKEN + tile_id
    tl.store(out_scale_ptr + scale_offset, ue8m0)


@triton.jit
def _rotated_dequant_rope_kernel(
    rope_bf16_ptr,     # [N, 64]  BF16
    out_slot_bf16_ptr, # [N, 576] viewed as bf16 (288 elements per row)
    N,
    TILE_SIZE: tl.constexpr,
    SLOT_BF16_ELEMS: tl.constexpr,  # 576 // 2 = 288
    ROPE_BF16_OFFSET: tl.constexpr,  # 448 // 2 = 224
):
    token_id = tl.program_id(0)
    if token_id >= N:
        return

    lane = tl.arange(0, TILE_SIZE)
    rope_vals = tl.load(rope_bf16_ptr + token_id * TILE_SIZE + lane)
    rope_offset = token_id * SLOT_BF16_ELEMS + ROPE_BF16_OFFSET + lane
    tl.store(out_slot_bf16_ptr + rope_offset, rope_vals)


def rotated_dequant_to_fp8_layout(
    nope_bf16: torch.Tensor,  # [N, 448] BF16, already inverse-rotated
    rope_bf16: torch.Tensor,  # [N, 64]  BF16
    out_slot: torch.Tensor,   # [N, 576] uint8 (output, slot bytes)
    out_scale: torch.Tensor,  # [N, 8]   uint8 (output, UE8M0 per nope tile + 1 pad)
) -> None:
    """Re-quantise rotated-back BF16 nope+rope into DSv4 FP8 slot layout.

    The output ``(out_slot, out_scale)`` matches FlashMLA's expected per-token
    layout:

    * ``out_slot[i, :448]`` = e4m3fn nope bytes (7 tiles of 64),
    * ``out_slot[i, 448:576]`` = rope BF16 bytes (64 elements),
    * ``out_scale[i, :7]``    = UE8M0 byte per nope tile,
    * ``out_scale[i, 7]``     = padding (untouched).

    The caller is responsible for scattering these into the paged FlashMLA
    cache. M3.c.1 unit tests use the flat ``[N, 576]`` form directly.
    """
    if nope_bf16.dtype != torch.bfloat16:
        raise ValueError(f"nope_bf16 must be bf16, got {nope_bf16.dtype}")
    if rope_bf16.dtype != torch.bfloat16:
        raise ValueError(f"rope_bf16 must be bf16, got {rope_bf16.dtype}")
    if out_slot.dtype != torch.uint8:
        raise ValueError(f"out_slot must be uint8, got {out_slot.dtype}")
    if out_scale.dtype != torch.uint8:
        raise ValueError(f"out_scale must be uint8, got {out_scale.dtype}")
    if nope_bf16.shape[-1] != _MLA_NOPE_DIM:
        raise ValueError(
            f"nope last dim {nope_bf16.shape[-1]} != {_MLA_NOPE_DIM}"
        )
    if rope_bf16.shape[-1] != _MLA_TILE_SIZE:
        raise ValueError(
            f"rope last dim {rope_bf16.shape[-1]} != {_MLA_TILE_SIZE}"
        )
    if out_slot.shape[-1] != _MLA_SLOT_BYTES:
        raise ValueError(
            f"out_slot last dim {out_slot.shape[-1]} != {_MLA_SLOT_BYTES}"
        )
    if out_scale.shape[-1] != _MLA_SCALES_PER_TOKEN:
        raise ValueError(
            f"out_scale last dim {out_scale.shape[-1]} != {_MLA_SCALES_PER_TOKEN}"
        )

    N = nope_bf16.shape[0]
    if N == 0:
        return
    if rope_bf16.shape[0] != N or out_slot.shape[0] != N or out_scale.shape[0] != N:
        raise ValueError("N mismatch among nope/rope/out_slot/out_scale")
    if not (
        nope_bf16.is_contiguous()
        and rope_bf16.is_contiguous()
        and out_slot.is_contiguous()
        and out_scale.is_contiguous()
    ):
        raise ValueError("all tensors must be contiguous")

    out_slot_fp8 = out_slot.view(_FP8_DTYPE)
    out_slot_bf16 = out_slot.view(torch.bfloat16)

    _rotated_dequant_nope_kernel[(N, _MLA_NUM_NOPE_TILES)](
        nope_bf16,
        out_slot_fp8,
        out_scale,
        N,
        TILE_SIZE=_MLA_TILE_SIZE,
        NOPE_DIM=_MLA_NOPE_DIM,
        SLOT_BYTES=_MLA_SLOT_BYTES,
        SCALES_PER_TOKEN=_MLA_SCALES_PER_TOKEN,
        UE8M0_BIAS=_UE8M0_EXPONENT_BIAS,
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )

    _rotated_dequant_rope_kernel[(N,)](
        rope_bf16,
        out_slot_bf16,
        N,
        TILE_SIZE=_MLA_TILE_SIZE,
        SLOT_BF16_ELEMS=_MLA_SLOT_BYTES // 2,
        ROPE_BF16_OFFSET=_MLA_NOPE_DIM // 2,
    )


# ---------------------------------------------------------------------------
# T3: Triton bitpack_rowwise — GPU-only, eliminates H2D/D2H round trip.
# ---------------------------------------------------------------------------
#
# Kernel design:
#   * Grid: (N, row_bytes). One program per (row, output_byte).
#   * Each program gathers 8 bit-contributions from its row, via
#     precomputed (dim_of_bit[], bitpos_in_dim[]) metadata.
#   * bits per dim vary (2/3/4) but are identical across rows, so
#     metadata is computed once per layer.
#   * Output byte = sum( (codes[row, dim] >> bitpos_in_dim) & 1 << (bit % 8) ).
# ---------------------------------------------------------------------------
@triton.jit
def _triton_bitpack_kernel(
    codes_ptr,         # [N, D] int32
    out_ptr,           # [N, row_bytes] uint8
    dim_of_bit_ptr,    # [row_bits] int32
    bitpos_in_dim_ptr, # [row_bits] int32
    N, D, row_bits, row_bytes,
):
    row = tl.program_id(0)
    byte = tl.program_id(1)
    if row >= N or byte >= row_bytes:
        return

    offs = byte * 8 + tl.arange(0, 8)
    mask = offs < row_bits

    dims = tl.load(dim_of_bit_ptr + offs, mask=mask, other=0)
    bpos = tl.load(bitpos_in_dim_ptr + offs, mask=mask, other=0)

    # Gather codes: one int32 per contributing dim.
    codes = tl.load(codes_ptr + row * D + dims, mask=mask, other=0)
    # Extract the right bit (LSB-first within each dim's value).
    bits = (codes >> bpos) & 1
    # Shift each bit to its position within this byte, then OR-reduce.
    bit_in_byte = tl.arange(0, 8)
    out = tl.sum(bits.to(tl.int32) << bit_in_byte, axis=0)
    tl.store(out_ptr + row * row_bytes + byte, out.to(tl.uint8))


def triton_bitpack_rowwise(
    codes: torch.Tensor,       # [N, D] int32 (cuda)
    dim_of_bit: torch.Tensor,  # [row_bits] int32 (cuda)
    bitpos_in_dim: torch.Tensor,  # [row_bits] int32 (cuda)
    row_bytes: int,
) -> torch.Tensor:
    """GPU-only bitpack. Returns [N, row_bytes] uint8 on same device."""
    if not codes.is_cuda:
        raise RuntimeError("triton_bitpack_rowwise requires CUDA tensors")
    if codes.dtype != torch.int32:
        raise ValueError(f"codes dtype must be int32, got {codes.dtype}")
    if codes.dim() != 2:
        raise ValueError(f"codes must be 2D, got shape {tuple(codes.shape)}")
    N, D = codes.shape
    row_bits = int(dim_of_bit.shape[0])
    if int(bitpos_in_dim.shape[0]) != row_bits:
        raise ValueError(
            f"dim_of_bit/bitpos_in_dim size mismatch: {row_bits} vs "
            f"{bitpos_in_dim.shape[0]}"
        )
    if not codes.is_contiguous():
        codes = codes.contiguous()

    out = torch.empty((N, row_bytes), dtype=torch.uint8, device=codes.device)
    if N == 0 or row_bytes == 0:
        return out

    grid = (N, row_bytes)
    _triton_bitpack_kernel[grid](
        codes, out, dim_of_bit, bitpos_in_dim,
        N, D, row_bits, row_bytes,
    )
    return out


# ---------------------------------------------------------------------------
# T3: Triton bitunpack_rowwise — GPU-only，消除 D2H/H2D 往返
# ---------------------------------------------------------------------------
#
# pack 语义（与 bitpack_rowwise 一致，LSB-first）：
#   对每个 dim d，其 bits[d] 个位按低位到高位顺序放入 packed 比特流
#   bit_start[d] = prefix_sum(bits[:d])
#   bit_of_dim = bit_start[d] + i，其中 i = 0..bits[d]-1
#   code[row, d] = Σ (packed_bit(row, bit_of_dim) << i)
#
# Kernel 设计：
#   * Grid: (N, D)，每个 program 负责一个 (row, dim)，通过前缀和查 bit_start
#   * 利用 Triton 向量化 load（一次 32b）+ shift/mask，避免 CPU 位运算循环
# ---------------------------------------------------------------------------
@triton.jit
def _triton_bitunpack_kernel(
    packed_ptr,        # [N, row_bytes] uint8
    codes_ptr,         # [N, D] int32 (output)
    bits_ptr,          # [D] int32
    prefix_sum_ptr,    # [D] int32，prefix_sum[d] = Σ bits[:d]
    N, D, row_bytes,
):
    row = tl.program_id(0)
    dim = tl.program_id(1)
    if row >= N or dim >= D:
        return

    bits_d = tl.load(bits_ptr + dim).to(tl.int32)
    bit_start = tl.load(prefix_sum_ptr + dim).to(tl.int32)

    if bits_d <= 0:
        tl.store(codes_ptr + row * D + dim, 0)
        return

    # 该 dim 需要读 bits_d 个位。位位置范围 [bit_start, bit_start + bits_d - 1]
    # 字节序: packed 是按 LSB-first 打包，每个字节对应 8 个连续 bit
    result: tl.int32 = 0
    # loop unroll by 8-bit byte reads; bits_d <= D <= 448 (实际很小: 2/3/4)
    # 由于 bits_d 是运行时值，用 while 循环（Triton 可展开小范围）
    bit_idx = 0
    while bit_idx < bits_d:
        global_bit = bit_start + bit_idx
        byte_off = global_bit // 8
        bit_in_byte = global_bit % 8
        byte_val = tl.load(packed_ptr + row * row_bytes + byte_off).to(tl.int32)
        bit_val = (byte_val >> bit_in_byte) & 1
        result = result | (bit_val << bit_idx)
        bit_idx += 1

    tl.store(codes_ptr + row * D + dim, result)


def triton_bitunpack_rowwise(
    packed: torch.Tensor,    # [N, row_bytes] uint8 (cuda)
    bits: torch.Tensor,      # [D] int32 (cuda or cpu)
) -> torch.Tensor:
    """GPU-only bitunpack. Returns [N, D] int32 on same device as ``packed``.

    ``bits`` 可以在 CPU 或 GPU；内部会把它 + 构造的前缀和移到 packed.device。
    """
    if not packed.is_cuda:
        raise RuntimeError("triton_bitunpack_rowwise requires CUDA tensors")
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed dtype must be uint8, got {packed.dtype}")
    if packed.dim() != 2:
        raise ValueError(f"packed must be 2D, got shape {tuple(packed.shape)}")
    if bits.dtype != torch.int32:
        bits = bits.to(torch.int32)
    if bits.dim() != 1:
        raise ValueError(f"bits must be 1D, got shape {tuple(bits.shape)}")

    N, row_bytes = packed.shape
    D = int(bits.shape[0])
    if D == 0 or N == 0:
        return torch.zeros((N, D), dtype=torch.int32, device=packed.device)

    # bits 可能在 CPU；构造前缀和并移动到 GPU
    bits_gpu = bits.to(packed.device)
    # prefix_sum: prefix_sum[d] = Σ bits[:d]
    prefix_sum = torch.zeros((D + 1,), dtype=torch.int64, device=bits_gpu.device)
    prefix_sum[1:] = bits_gpu.to(torch.int64).cumsum(0)
    prefix_sum = prefix_sum[:D].to(torch.int32).contiguous()
    bits_gpu = bits_gpu.contiguous()
    if not packed.is_contiguous():
        packed = packed.contiguous()

    codes = torch.empty((N, D), dtype=torch.int32, device=packed.device)

    grid = (N, D)
    _triton_bitunpack_kernel[grid](
        packed, codes, bits_gpu, prefix_sum,
        N, D, row_bytes,
    )
    return codes


__all__ = [
    "rotated_dequant_to_fp8_layout",
    "triton_bitpack_rowwise",
    "triton_bitunpack_rowwise",
    "_MLA_NOPE_DIM",
    "_MLA_HEAD_DIM",
    "_MLA_TILE_SIZE",
    "_MLA_SLOT_BYTES",
    "_MLA_SCALES_PER_TOKEN",
]
