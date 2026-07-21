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


# ---------------------------------------------------------------------------
# T_cgraph_safe: capture-stable per-token shadow scatter.
# ---------------------------------------------------------------------------
#
# 替代原来 PyTorch 的 gather + where + scatter_ 实现。原实现在 cudagraph
# capture/replay 下出现 partial corruption，根因：invalid token (loc=-1)
# 经 ``loc.clamp(min=0)`` 全部映射到 (page=0, slot=0)，与 valid token 在
# ``flat.scatter_(0, ...)`` 的重复 byte index 上 race；torch.where 只是
# 把 invalid 的 *值* 替换为 old，但 scatter_ 的重复 index winner 由
# 调度顺序决定，cudagraph 重放时调度差异导致 valid token 字节被 invalid
# 路径的 old byte 覆盖 → gsm8k 0.860 partial corruption。
#
# 修复策略：每 token 一个 program block，invalid 直接 return（无 read 无
# write，零 race surface）。valid token 之间天然无 byte 级别重叠
# （caller 保证每个 valid token 的 slot 唯一）。
# ---------------------------------------------------------------------------
@triton.jit
def _scatter_tokens_to_shadow_kernel(
    out_slot_ptr,    # [N, SLOT_BYTES] uint8
    out_scale_ptr,   # [N, SCALES] uint8
    loc_ptr,         # [N] integer (int32 or int64) (token slot ids, -1 for invalid)
    shadow_ptr,      # [num_pages * bytes_per_page] uint8 flat
    N,
    page_size,
    bytes_per_page,
    SLOT_BYTES: tl.constexpr,
    SCALES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,    # >= SLOT_BYTES, power-of-2
    BLOCK_SCALE: tl.constexpr,  # >= SCALES, power-of-2
):
    tid = tl.program_id(0)
    if tid >= N:
        return
    loc = tl.load(loc_ptr + tid).to(tl.int64)
    if loc < 0:
        return  # capture-safe: no read, no write for invalid token

    page = loc // page_size
    slot = loc % page_size

    # Slot bytes (576 not power-of-2 → BLOCK_VAL=1024 + mask).
    slot_lane = tl.arange(0, BLOCK_VAL)
    slot_mask = slot_lane < SLOT_BYTES
    val = tl.load(out_slot_ptr + tid * SLOT_BYTES + slot_lane, mask=slot_mask, other=0)
    slot_base = page * bytes_per_page + slot * SLOT_BYTES
    tl.store(shadow_ptr + slot_base + slot_lane, val, mask=slot_mask)

    # Scale bytes: scale region starts at page_size * SLOT_BYTES within page.
    scale_lane = tl.arange(0, BLOCK_SCALE)
    scale_mask = scale_lane < SCALES
    sval = tl.load(out_scale_ptr + tid * SCALES + scale_lane, mask=scale_mask, other=0)
    scale_base = page * bytes_per_page + page_size * SLOT_BYTES + slot * SCALES
    tl.store(shadow_ptr + scale_base + scale_lane, sval, mask=scale_mask)


def triton_scatter_tokens_to_shadow(
    out_slot: torch.Tensor,    # [N, 576] uint8
    out_scale: torch.Tensor,   # [N, 8]   uint8
    loc: torch.Tensor,         # [N]      int64 (with -1 sentinels)
    shadow: torch.Tensor,      # [num_pages, bytes_per_page] uint8
    page_size: int,
) -> None:
    """Capture-safe per-token shadow scatter.

    - 每 token 一个 program block；
    - ``loc < 0`` 的 token 整个 program block exit，零 read / 零 write，
      消除原 ``gather + where + scatter_`` 在 (page=0, slot=0) 上的重复
      index race；
    - valid token 之间 byte 不重叠（slot 唯一性由 caller 保证）。
    """
    if not (out_slot.is_cuda and out_scale.is_cuda and loc.is_cuda and shadow.is_cuda):
        raise RuntimeError("triton_scatter_tokens_to_shadow requires CUDA tensors")
    if out_slot.dtype != torch.uint8 or out_scale.dtype != torch.uint8 or shadow.dtype != torch.uint8:
        raise ValueError("out_slot/out_scale/shadow must be uint8")
    # Capture-stability: do NOT do ``loc.to(int64)`` here — it would alloc
    # a temporary tensor whose address is unstable across cudagraph
    # capture/replay. The kernel handles int32/int64 internally via
    # ``tl.load(...).to(tl.int64)``.
    if loc.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"loc must be int32 or int64 for cudagraph stability, got {loc.dtype}"
        )
    if not out_slot.is_contiguous():
        out_slot = out_slot.contiguous()
    if not out_scale.is_contiguous():
        out_scale = out_scale.contiguous()
    if not loc.is_contiguous():
        loc = loc.contiguous()

    N = out_slot.shape[0]
    if N == 0:
        return
    if out_slot.shape[-1] != _MLA_SLOT_BYTES:
        raise ValueError(
            f"out_slot last dim {out_slot.shape[-1]} != {_MLA_SLOT_BYTES}"
        )
    if out_scale.shape[-1] != _MLA_SCALES_PER_TOKEN:
        raise ValueError(
            f"out_scale last dim {out_scale.shape[-1]} != {_MLA_SCALES_PER_TOKEN}"
        )
    if shadow.dim() != 2:
        raise ValueError(f"shadow must be 2D [num_pages, bytes_per_page], got {tuple(shadow.shape)}")
    bytes_per_page = shadow.shape[1]
    shadow_flat = shadow.view(-1)

    _scatter_tokens_to_shadow_kernel[(N,)](
        out_slot,
        out_scale,
        loc,
        shadow_flat,
        N,
        int(page_size),
        int(bytes_per_page),
        SLOT_BYTES=_MLA_SLOT_BYTES,
        SCALES=_MLA_SCALES_PER_TOKEN,
        BLOCK_VAL=1024,   # next pow2 >= 576
        BLOCK_SCALE=8,    # already pow2
    )


# ---------------------------------------------------------------------------
# T5: Fused RMSNorm + RoPE (Triton).
# Replaces the Python fallback in set_swa_key_buffer_radix_fused_norm_rope
# (N*512 pow → N rsqrt → N*512 mul → nope/rope split → complex
# reshape → complex mul → cat → bf16, which allocates ~5x N*512
# intermediate tensors).  Fused kernel: one program per row, single kernel
# launch for the whole 512-dim vector, reading the row once in fp32,
# accumulating sq sum → rsqrt → fused weight-mul → nope passthrough +
# rope complex rotation → bf16 store.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_norm_rope_kernel(
    kv_ptr,        # [N, 512] BF16 (row stride = kv_row_stride, may be > ROW_DIM)
    weight_ptr,    # [512] BF16
    freqs_cis_ptr, # [max_pos, 32] complex64 -> fp32 interleaved [max_pos, 64]
    pos_ptr,       # [N] i64
    out_ptr,        # [N, 512] BF16 (contiguous, row stride = ROW_DIM)
    N,
    eps,
    kv_row_stride,  # runtime: input row stride in elements (qkv_a slice => 1536)
    ROW_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROPE_HALF: tl.constexpr,  # ROPE_DIM // 2
):
    row = tl.program_id(0)
    if row >= N:
        return

    in_base = row * kv_row_stride
    out_base = row * ROW_DIM

    # Step 1: accumulate mean of squares
    sum_sq = 0.0
    for b_start in range(0, ROW_DIM, BLOCK_SIZE):
        offs = b_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < ROW_DIM
        kv = tl.load(kv_ptr + in_base + offs, mask=mask, other=0.0)
        kv_f = kv.to(tl.float32)
        sum_sq += tl.sum(kv_f * kv_f, axis=0)
    norm = sum_sq / ROW_DIM
    rsqrt_val = tl.math.rsqrt(norm + eps)

    # Step 2: NOPE portion: columns [0, NOPE_DIM)
    for b_start in range(0, NOPE_DIM, BLOCK_SIZE):
        offs = b_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < NOPE_DIM
        kv = tl.load(kv_ptr + in_base + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = kv.to(tl.float32) * rsqrt_val * w.to(tl.float32)
        tl.store(out_ptr + out_base + offs, out.to(tl.bfloat16), mask=mask)

    # Step 3: RoPE portion: columns [NOPE_DIM, ROW_DIM).
    # kv rope layout: [re_0, im_0, re_1, im_1, ...] (interleaved).
    # freqs_cis layout: [re_0, im_0, re_1, im_1, ...] (interleaved).
    pos = tl.load(pos_ptr + row).to(tl.int64)
    i = tl.arange(0, ROPE_HALF)
    mask = i < ROPE_HALF
    re_offs = NOPE_DIM + 2 * i
    im_offs = re_offs + 1
    # Normalize + weight for each complex pair.
    re_kv = tl.load(kv_ptr + in_base + re_offs, mask=mask, other=0.0).to(tl.float32)
    im_kv = tl.load(kv_ptr + in_base + im_offs, mask=mask, other=0.0).to(tl.float32)
    re_w = tl.load(weight_ptr + re_offs, mask=mask, other=0.0).to(tl.float32)
    im_w = tl.load(weight_ptr + im_offs, mask=mask, other=0.0).to(tl.float32)
    re_kv = re_kv * rsqrt_val * re_w
    im_kv = im_kv * rsqrt_val * im_w
    # Apply RoPE rotation.
    fc_re = tl.load(freqs_cis_ptr + pos * ROPE_DIM + 2 * i, mask=mask, other=0.0)
    fc_im = tl.load(freqs_cis_ptr + pos * ROPE_DIM + 2 * i + 1, mask=mask, other=0.0)
    out_re = re_kv * fc_re - im_kv * fc_im
    out_im = re_kv * fc_im + im_kv * fc_re
    tl.store(out_ptr + out_base + re_offs, out_re.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + im_offs, out_im.to(tl.bfloat16), mask=mask)


def triton_fused_norm_rope(
    kv: torch.Tensor,       # [N, 512] BF16
    kv_weight: torch.Tensor,  # [512] BF16
    eps: float,
    freqs_cis: torch.Tensor, # [max_pos, 32] complex64 (or interleaved fp32 [max_pos, 64])
    positions: torch.Tensor, # [N] int64
) -> torch.Tensor:
    """Fused RMSNorm + RoPE, Triton GPU-only.

    Returns [N, 512] BF16 tensor, allocated on the same device as ``kv``.

    Accepts ``freqs_cis`` in either ``complex64`` layout (PyTorch default
    returned by ``precompute_freqs_cis``) or a raw ``fp32 [max_pos, 64]``
    interleaved (real, imag) tensor produced by explicit storage. The
    kernel treats freqs_cis memory as a flat ``fp32`` [max_pos, 64]
    sequence, which matches what ``complex64.to(torch.float32)`` produces
    when viewed as the underlying storage.
    """
    assert kv.is_cuda, "triton_fused_norm_rope requires CUDA"
    N = int(kv.shape[0])
    assert kv.shape == (N, _MLA_HEAD_DIM), (
        f"kv must be (N, {_MLA_HEAD_DIM}), got {tuple(kv.shape)}"
    )
    assert kv_weight.shape == (_MLA_HEAD_DIM,), (
        f"kv_weight must be ({_MLA_HEAD_DIM},), got {tuple(kv_weight.shape)}"
    )
    assert positions.shape == (N,), (
        f"positions must be (N,), got {tuple(positions.shape)}"
    )
    if N == 0:
        return torch.empty_like(kv)

    if freqs_cis.dtype in (torch.complex64, torch.complex32):
        # complex64: each element = 2 fp32 values. Convert to a plain
        # fp32 [max_pos, 64] view without copy where possible.
        freqs_fp32 = freqs_cis.view(torch.float32).reshape(freqs_cis.shape[0], -1)
    elif freqs_cis.dtype == torch.float32 and freqs_cis.shape[-1] == _MLA_TILE_SIZE:
        # Already [max_pos, 64] interleaved fp32.
        freqs_fp32 = freqs_cis
    else:
        freqs_fp32 = freqs_cis.to(torch.complex64).view(torch.float32).reshape(
            freqs_cis.shape[0], -1
        )
    assert freqs_fp32.shape[-1] == _MLA_TILE_SIZE, (
        f"freqs_cis last dim must be {_MLA_TILE_SIZE} (as interleaved fp32), "
        f"got {freqs_fp32.shape}"
    )

    # Output is always contiguous [N, 512]; the kernel writes row-major.
    # NOTE: torch.empty_like(kv) would inherit a strided layout when ``kv``
    # is a non-contiguous slice (e.g. qkv_a[..., q_lora_rank:] with
    # stride[0]=1536), so allocate a fresh contiguous buffer explicitly.
    out = torch.empty((N, _MLA_HEAD_DIM), dtype=kv.dtype, device=kv.device)
    # The kernel honours the input row stride so a non-contiguous ``kv``
    # slice (the common DSv4 case) is read correctly without a copy. The
    # innermost dim is assumed contiguous (stride 1), which holds for the
    # qkv_a[..., q_lora_rank:] slice.
    kv_row_stride = int(kv.stride(0))
    # Kernel launch: one program per row; each program processes the full
    # row in 128-column blocks for the sum-of-squares reduction and the
    # nope portion, and a single 64-wide load for rope.
    _fused_norm_rope_kernel[(N,)](
        kv, kv_weight, freqs_fp32, positions.to(torch.int64), out, N, eps,
        kv_row_stride,
        ROW_DIM=_MLA_HEAD_DIM,
        NOPE_DIM=_MLA_NOPE_DIM,
        ROPE_DIM=_MLA_TILE_SIZE,
        BLOCK_SIZE=128,
        ROPE_HALF=_MLA_TILE_SIZE // 2,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# T6: Triton fused packed→dequant→UE8M0→scatter-to-shadow (single kernel
# per token).  Replaces the Python loop in _refresh_shadow_pages which
# materializes a large [M, 576] out_slot + [M, 8] out_scale on GPU and
# then copies pages one-at-a-time via tensor copy_().
#
# Semantics (equivalent to Python fallback for correctness):
#   for each page p in page_indices:
#     for each slot s in 0..page_size-1:
#       (nope_bf16, rope_bf16) = dequant(packed[p, s*BPT : (s+1)*BPT])
#       out_slot[nope_bytes] = ue8m0_scale + e4m3 quant of nope_bf16
#       out_slot[rope_bytes] = rope_bf16 raw bytes
#       out_scale[7 tiles] = per-tile scale exponent byte + 1 pad byte
#       shadow[p, s*576:(s+1)*576] = out_slot
#       shadow[p, page_size*576 + s*8 : ...+8] = out_scale
#
# Design: one program per (page, slot) == one program per token-refresh.
# Each program:  (1) loads packed row for (page, slot);
#                (2) triton-bitunpack → int32 codes[448];
#                (3) affine: y = codes * scale + zero (per-column);
#                (4) matmul by R^T: y @ R.T → nope_bf16[448];
#                (5) per-tile UE8M0 scaling + e4m3fn store into out_slot;
#                (6) rope: load packed[..., nope_bytes : ...+64*2] bytes
#                    as bf16[64], store raw bytes into shadow rope region.
# Steps (2)-(4) are handled by triton_bitunpack_rowwise + torch ops in
# the current implementation; this kernel is a "fused scatter" which
# accepts the already-computed out_slot / out_scale and writes them into
# shadow pages with the correct per-slot offsets — this mirrors
# triton_scatter_tokens_to_shadow, but with page-aligned flat locs.
#
# For simplicity and correctness, this **T6_lite** kernel takes
# pre-computed out_slot / out_scale (from the existing Triton dequant
# pipeline) and scatters into shadow in a single pass. This replaces the
# Python ``for i, page in enumerate(flat_pages_cpu)`` loop which was the
# main latency driver of _refresh_shadow_pages for large batches.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_refresh_shadow_scatter_kernel(
    out_slot_ptr,   # [M_total, 576] uint8, packed contiguously as (num_pages, page_size, 576)
    out_scale_ptr,  # [M_total, 8] uint8
    page_indices_ptr,  # [P] int64 (unique pages to refresh)
    shadow_ptr,     # [num_pages_all, shadow_bytes_per_page] uint8
    page_size,
    shadow_bytes_per_page,
    P,              # num unique pages == page_indices_ptr.shape[0]
    SLOT_BYTES: tl.constexpr,  # 576
    SCALES_PER_TOKEN: tl.constexpr,  # 8
    BLOCK_VAL: tl.constexpr,  # 128 (bytes per lane write)
):
    pid = tl.program_id(0)
    if pid >= P:
        return
    page = tl.load(page_indices_ptr + pid).to(tl.int64)
    # shadow bytes layout:
    #   per-page: [page_size * SLOT_BYTES] of slot bytes, then
    #              [page_size * SCALES_PER_TOKEN] of scale bytes, then pad
    slot_bytes_off = page * shadow_bytes_per_page
    scale_bytes_off = slot_bytes_off + page_size * SLOT_BYTES
    # out_slot layout for this page: out_slot[pid * page_size + s, :]
    base_out_slot = pid * page_size * SLOT_BYTES
    base_out_scale = pid * page_size * SCALES_PER_TOKEN

    lane = tl.arange(0, BLOCK_VAL)
    # SLOT_BYTES region: loop over each slot; each slot is BLOCK_VAL bytes
    # per iteration to maximize memory coalescing.
    for s in range(page_size):
        slot_byte_off = s * SLOT_BYTES
        for b in range(0, SLOT_BYTES, BLOCK_VAL):
            mask = b + lane < SLOT_BYTES
            v = tl.load(out_slot_ptr + base_out_slot + slot_byte_off + b + lane,
                        mask=mask, other=0)
            tl.store(shadow_ptr + slot_bytes_off + slot_byte_off + b + lane, v, mask=mask)

        # SCALES_PER_TOKEN region: 8 bytes per slot.
        for b in range(0, SCALES_PER_TOKEN, BLOCK_VAL):
            mask = b + lane < SCALES_PER_TOKEN
            v = tl.load(out_scale_ptr + base_out_scale + s * SCALES_PER_TOKEN + b + lane,
                        mask=mask, other=0)
            tl.store(shadow_ptr + scale_bytes_off + s * SCALES_PER_TOKEN + b + lane,
                     v, mask=mask)


def triton_fused_refresh_shadow_scatter(
    out_slot: torch.Tensor,    # [P, page_size, 576] uint8 (contiguous)
    out_scale: torch.Tensor,   # [P, page_size, 8] uint8 (contiguous)
    page_indices: torch.Tensor, # [P] int64, unique page ids
    shadow: torch.Tensor,      # [num_pages, shadow_bytes_per_page] uint8
    page_size: int,
    shadow_bytes_per_page: int,
) -> None:
    """T6-lite: fused GPU scatter of pre-computed out_slot/out_scale into
    shadow pages. Replaces the CPU-side Python loop over unique pages.
    """
    assert out_slot.is_cuda and out_scale.is_cuda and page_indices.is_cuda
    assert out_slot.dtype == torch.uint8 and out_scale.dtype == torch.uint8
    P = int(page_indices.shape[0])
    if P == 0:
        return
    assert out_slot.shape == (P, page_size, _MLA_SLOT_BYTES), (
        f"out_slot shape {tuple(out_slot.shape)} != ({P},{page_size},{_MLA_SLOT_BYTES})"
    )
    assert out_scale.shape == (P, page_size, _MLA_SCALES_PER_TOKEN), (
        f"out_scale shape {tuple(out_scale.shape)} != ({P},{page_size},{_MLA_SCALES_PER_TOKEN})"
    )
    _fused_refresh_shadow_scatter_kernel[(P,)](
        out_slot.reshape(P * page_size, _MLA_SLOT_BYTES),
        out_scale.reshape(P * page_size, _MLA_SCALES_PER_TOKEN),
        page_indices.to(torch.int64),
        shadow,
        page_size, shadow_bytes_per_page, P,
        SLOT_BYTES=_MLA_SLOT_BYTES,
        SCALES_PER_TOKEN=_MLA_SCALES_PER_TOKEN,
        BLOCK_VAL=128,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# T_store_fused (bu4): single-kernel packed store for the uniform 4-bit path.
#
# Replaces the ~14 eager op tail of ``rotated_store_to_packed`` (reshape →
# amin → amax → range → step → round → clamp → cast → triton_bitpack →
# header fp16 stack → cat → index_select → where → index_copy_) with ONE
# Triton launch. Only the ``nope @ R`` matmul stays outside (a single
# efficient cublas gemm).
#
# Per-token program (grid = (N,)):
#   1. sentinel gate: loc = indices[t]; if loc < 0 → return (capture-safe,
#      no read / no write — same pattern proven correct by
#      _scatter_tokens_to_shadow_kernel).
#   2. for each of 7 groups of 64 dims:
#        - min/max over the 64 lanes → per-token×group affine
#        - codes = clamp(round((x - min)/step), 0, 15)   (step = range/15)
#        - nibble-pack 2 codes/byte → 32 bytes → cache[loc*bpt + g*32 + ..]
#        - fp16(min), fp16(range) → 4 header bytes at cache[loc*bpt+224+g*4]
#   3. copy 128 raw rope bytes → cache[loc*bpt + 252 + ..]
#
# Byte layout (bu4): [224 nope codes][28 fp16 header][128 rope] = 380 bpt,
# bit-identical to the legacy cat() path (LSB-first nibble pack, header =
# stack(fp16 min, fp16 range) per group).
# ---------------------------------------------------------------------------
@triton.jit
def _fused_store_bu4_kernel(
    k_rot_ptr,      # [N, 448] fp32 (already nope @ R)
    input_bf16_ptr, # [N, 512] bf16 (raw rope starts at element 448)
    cache_ptr,      # [num_pages * bpt] uint8 (flat)
    indices_ptr,    # [N] integer (int32/int64), -1 sentinel for invalid
    N,
    BPT,            # bytes per token (constexpr-like runtime int)
    NOPE_CODE_BYTES: tl.constexpr,   # 224
    HEADER_OFF: tl.constexpr,        # 224
    ROPE_OFF: tl.constexpr,          # 252
    GROUPS: tl.constexpr,            # 7
    GROUP_DIM: tl.constexpr,         # 64
    GROUP_BYTES: tl.constexpr,       # 32 (64 dims * 4 bit / 8)
    ROPE_BYTES: tl.constexpr,        # 128
    REUSE_GROUP_LOAD: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= N:
        return
    loc = tl.load(indices_ptr + t).to(tl.int64)
    if loc < 0:
        return  # capture-safe: invalid token does nothing

    row_base = loc * BPT
    L_f = 15.0

    for g in range(GROUPS):
        lane = tl.arange(0, GROUP_DIM)
        in_off = t * (GROUPS * GROUP_DIM) + g * GROUP_DIM + lane
        x = tl.load(k_rot_ptr + in_off)  # [64] fp32
        mn = tl.min(x, axis=0)
        mx = tl.max(x, axis=0)
        rng = tl.maximum(mx - mn, 1e-8)
        step = rng / L_f

        # Reuse the 64 values loaded for min/max when requested. The
        # reduction form avoids Triton tensor column indexing, which is not
        # supported by the container's frontend.
        blane = tl.arange(0, GROUP_BYTES)  # [32]
        reuse_load = tl.full((), 0, tl.int1)
        if REUSE_GROUP_LOAD:
            x_pairs = tl.reshape(x, (GROUP_BYTES, 2))
            pair_lane = tl.arange(0, 2)
            xlo = tl.sum(tl.where(pair_lane == 0, x_pairs, 0.0), axis=1)
            xhi = tl.sum(tl.where(pair_lane == 1, x_pairs, 0.0), axis=1)
        else:
            lo_off = t * (GROUPS * GROUP_DIM) + g * GROUP_DIM + 2 * blane
            hi_off = lo_off + 1
            xlo = tl.load(k_rot_ptr + lo_off)
            xhi = tl.load(k_rot_ptr + hi_off)
        clo = tl.floor((xlo - mn) / step + 0.5)
        chi = tl.floor((xhi - mn) / step + 0.5)
        clo = tl.minimum(tl.maximum(clo, 0.0), L_f).to(tl.int32)
        chi = tl.minimum(tl.maximum(chi, 0.0), L_f).to(tl.int32)
        byte_val = (clo | (chi << 4)).to(tl.uint8)
        tl.store(cache_ptr + row_base + g * GROUP_BYTES + blane, byte_val)

        # fp16 header: min then range, 2 bytes each (LSB-first)
        mn_h = mn.to(tl.float16).to(tl.uint16, bitcast=True)
        rng_h = rng.to(tl.float16).to(tl.uint16, bitcast=True)
        hb = row_base + HEADER_OFF + g * 4
        tl.store(cache_ptr + hb + 0, (mn_h & 0xFF).to(tl.uint8))
        tl.store(cache_ptr + hb + 1, ((mn_h >> 8) & 0xFF).to(tl.uint8))
        tl.store(cache_ptr + hb + 2, (rng_h & 0xFF).to(tl.uint8))
        tl.store(cache_ptr + hb + 3, ((rng_h >> 8) & 0xFF).to(tl.uint8))

    # Rope: read 64 BF16 values directly from the original contiguous input
    # and write their two raw bytes. This avoids materializing a contiguous
    # uint8 copy of the non-contiguous [N, 64] rope slice on every store.
    rlane = tl.arange(0, ROPE_BYTES // 2)
    rv = tl.load(input_bf16_ptr + t * (GROUPS * GROUP_DIM + ROPE_BYTES // 2) + 448 + rlane)
    rv_u16 = rv.to(tl.uint16, bitcast=True)
    tl.store(cache_ptr + row_base + ROPE_OFF + 2 * rlane, (rv_u16 & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + row_base + ROPE_OFF + 2 * rlane + 1, (rv_u16 >> 8).to(tl.uint8))


def triton_fused_store_bu4(
    k_rot: torch.Tensor,     # [N, 448] fp32 (nope @ R)
    input_bf16: torch.Tensor, # [N, 512] bf16, raw rope source
    cache: torch.Tensor,     # [num_pages, bpt] uint8
    indices: torch.Tensor,   # [N] int32/int64 (-1 sentinel)
    *,
    bpt: int,
) -> None:
    """Fused bu4 packed store: one Triton launch for the whole store tail.

    ``cache`` is written in place at ``cache.view(-1)[loc*bpt : loc*bpt+bpt]``
    for each valid ``loc``; ``loc < 0`` tokens are skipped (capture-safe).
    """
    assert k_rot.is_cuda and input_bf16.is_cuda and cache.is_cuda and indices.is_cuda
    assert k_rot.dtype == torch.float32, f"k_rot must be fp32, got {k_rot.dtype}"
    assert input_bf16.dtype == torch.bfloat16 and cache.dtype == torch.uint8
    assert indices.dtype in (torch.int32, torch.int64)
    N = int(k_rot.shape[0])
    if N == 0:
        return
    assert k_rot.shape[1] == _MLA_NOPE_DIM, f"k_rot dim {k_rot.shape[1]} != 448"
    assert input_bf16.shape == (N, 512), f"input_bf16 {tuple(input_bf16.shape)} != ({N},512)"
    if not k_rot.is_contiguous():
        k_rot = k_rot.contiguous()
    assert input_bf16.is_contiguous(), "input_bf16 must be contiguous"
    if not indices.is_contiguous():
        indices = indices.contiguous()
    cache_flat = cache.view(-1)
    import os
    num_warps = int(os.environ.get("SGLANG_RQ_FUSED_STORE_WARPS", "4"))
    if num_warps not in (1, 2, 4, 8):
        raise ValueError(
            f"SGLANG_RQ_FUSED_STORE_WARPS must be one of 1,2,4,8, got {num_warps}"
        )
    num_stages = int(os.environ.get("SGLANG_RQ_FUSED_STORE_STAGES", "2"))
    if num_stages not in (1, 2, 3, 4):
        raise ValueError(
            "SGLANG_RQ_FUSED_STORE_STAGES must be one of 1,2,3,4, "
            f"got {num_stages}"
        )
    _fused_store_bu4_kernel[(N,)](
        k_rot, input_bf16, cache_flat, indices,
        N, int(bpt),
        NOPE_CODE_BYTES=224,
        HEADER_OFF=224,
        ROPE_OFF=252,
        GROUPS=7,
        GROUP_DIM=64,
        GROUP_BYTES=32,
        ROPE_BYTES=128,
        REUSE_GROUP_LOAD=bool(
            os.environ.get("SGLANG_RQ_FUSED_STORE_REUSE_LOAD", "0") == "1"
        ),
        num_warps=num_warps,
        num_stages=num_stages,
    )


__all__ = [
    "rotated_dequant_to_fp8_layout",
    "triton_bitpack_rowwise",
    "triton_bitunpack_rowwise",
    "triton_scatter_tokens_to_shadow",
    "triton_fused_norm_rope",
    "triton_fused_refresh_shadow_scatter",
    "triton_fused_store_bu4",
    "_MLA_NOPE_DIM",
    "_MLA_HEAD_DIM",
    "_MLA_TILE_SIZE",
    "_MLA_SLOT_BYTES",
    "_MLA_SCALES_PER_TOKEN",
]
