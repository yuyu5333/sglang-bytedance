"""
Packing / quantization utilities for MXFP4A8 (weight E2M1 + block=32 E8M0 scale,
activation FP8 e4m3) on the CUTLASS w4a8 backend.

These helpers are the MXFP4 counterparts of ``int4fp8_utils.py``. They deliberately
reuse the SAME 4-bit nibble packing layout (``order_map = [0, 2, 4, 6, 1, 3, 5, 7]``)
as the int4a8 path, because the kernel-side DirectConvert prmt-LUT for E2M1 mirrors
the int4 one bit-for-bit. The only differences from int4a8 are:

  1. the 4-bit code is an E2M1 code (sign + 3-bit magnitude index), not a
     two's-complement int4 value, and
  2. the K-wise group size is 32 (E8M0 block) instead of 128, and the E8M0
     power-of-2 scale is pre-expanded to bf16 on the host so the kernel's
     post-MMA bf16 group-scale path is reused unchanged.

The int4a8 path (``int4fp8_utils.py``) is left untouched; this is a parallel module.
"""

import logging
from typing import Optional, Tuple

import torch
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

logger = logging.getLogger(__name__)

# E2M1 magnitudes indexed by the low-3-bit code. This ordering matches the
# kernel converter's POS_E4M3s LUT (mxfp4_numeric_conversion.hpp).
#   code: 0     1    2    3    4    5    6    7
#   mag : 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_E2M1_MAX = 6.0
MXFP4_BLOCK_SIZE = 32


def _round_to_e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """Map real values (already divided by their block scale) to the nearest
    E2M1 code (sign in bit 3, 3-bit magnitude index in bits 0..2). Returns an
    int8 tensor of nibble codes with the same shape as ``x``."""
    sign = (x < 0).to(torch.int8) << 3
    mag = x.abs().clamp(max=_E2M1_MAX)
    # nearest magnitude index
    grid = _E2M1_MAGNITUDES.to(x.device)  # [8]
    # |mag - grid| over last dim
    idx = (mag.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.int8)
    return (sign | idx).to(torch.int8)


def quantize_mxfp4_blockwise(
    w: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to MXFP4 (E2M1) with a per-block E8M0 scale.

    Args:
        w: weight tensor ``[..., K]``; K must be divisible by ``block_size``.
        block_size: K-wise block size (E8M0 block), default 32.

    Returns:
        codes: int8 tensor ``[..., K]`` of E2M1 nibble codes (sign|mag_idx).
        scale_e8m0: uint8 tensor ``[..., K // block_size]`` of E8M0 exponents
            (biased by 127, the standard E8M0 encoding).
    """
    orig_shape = w.shape
    k = orig_shape[-1]
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    wf = w.reshape(-1, k // block_size, block_size).float()

    # E8M0 scale: pick the power-of-2 that maps block absmax to <= E2M1_MAX.
    absmax = wf.abs().amax(dim=-1)  # [rows, nblocks]
    absmax = torch.clamp(absmax, min=1e-30)
    # exponent e such that absmax / 2^e <= E2M1_MAX  ->  e = ceil(log2(absmax / MAX))
    exp = torch.ceil(torch.log2(absmax / _E2M1_MAX))
    scale = torch.pow(2.0, exp)  # [rows, nblocks]

    scaled = wf / scale.unsqueeze(-1)
    codes = _round_to_e2m1_code(scaled).reshape(orig_shape)

    # E8M0 encoding: biased exponent (bias = 127), clamp to representable range.
    e8m0 = torch.clamp(exp + 127.0, 0.0, 255.0).to(torch.uint8)
    scale_e8m0 = e8m0.reshape(*orig_shape[:-1], k // block_size)
    return codes, scale_e8m0


def e8m0_to_bf16(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Expand an E8M0 (biased-exponent) power-of-2 scale to an exact bf16 value.

    Because every E8M0 value is a pure power of two, the expansion is lossless in
    bf16 and lets the kernel reuse the int4a8 post-MMA bf16 group-scale path
    unchanged.
    """
    exp = scale_e8m0.to(torch.float32) - 127.0
    return torch.pow(2.0, exp).to(torch.bfloat16)


_FP8_E4M3_MAX = 448.0


def quantize_activation_mxfp8_blockwise(
    x: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation tensor to mxfp8: FP8 (e4m3) data + a per-K-block
    scale, i.e. per-token (row) AND per-block (block_size along K).

    This is the activation-side counterpart of ``quantize_mxfp4_blockwise`` and
    is what makes the CUTLASS mxfp4a8 GEMM a true "per-token + per-block" (mxfp8)
    activation path on SM90, matching the SM100/SM120 native mxfp8 activation.

    Unlike the E8M0 weight scale, the activation block scale is a *general*
    fp32/bf16 amax-derived value (NOT restricted to a power of two): FP8 e4m3
    already carries a mantissa, so a real-valued per-block scale minimises the
    quantization error (this is how flashinfer / trtllm mxfp8 activation works in
    practice). The scale is emitted as bf16 so the kernel post-MMA path consumes
    the same bf16 element type as the (pre-expanded) weight block scale.

    Args:
        x: activation ``[M, K]`` (bf16/fp16/fp32); K must be divisible by
            ``block_size``.
        block_size: K-wise block size, default 32 (matches the E8M0 weight block).

    Returns:
        x_fp8: ``[M, K]`` float8_e4m3fn, the block-scaled activation.
        scale: ``[M, K // block_size]`` bf16, ``block_amax / 448`` per block.
            Dequant is ``x_fp8[m, k] * scale[m, k // block_size]``.
    """
    assert x.dim() == 2, "activation must be 2D [M, K]"
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    nblk = k // block_size

    xf = x.reshape(m, nblk, block_size).float()
    amax = xf.abs().amax(dim=-1)  # [M, nblk]
    amax = torch.clamp(amax, min=1e-12)
    scale = amax / _FP8_E4M3_MAX  # [M, nblk], real-valued
    xq = (xf / scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    x_fp8 = xq.reshape(m, k).to(torch.float8_e4m3fn)
    return x_fp8, scale.to(torch.bfloat16)


def interleave_act_scale_mxfp8(
    scale: torch.Tensor, alignment: int = 4
) -> torch.Tensor:
    """Interleave a per-token+per-block activation scale ``[M, K//block]`` into
    the kernel's physical activation-scale layout ``[K//(block*4), M*4]``.

    This mirrors the weight-scale 4-wide interleave (``interleave_scales`` in the
    int4a8 path) but tiled over tokens (M) instead of weight channels (N). The
    kernel's activation-scale TMA expects **token unit-stride** with the scale-K
    stride equal to M (tokens per expert), i.e. ``as_strides = M``. This exact
    layout was verified bit-exact (rel_mean = 0.0000) by the single-GEMM test
    ``tests/test_cutlass_mxfp4a8_moe_mm.py::run_case_mxfp8_act``.

    ``K//block`` must be a multiple of ``alignment`` (=4); for K a multiple of 128
    and block=32 this holds (K/32 multiple of 4).
    """
    m, nblk = scale.shape
    assert nblk % alignment == 0, f"K//block={nblk} not divisible by {alignment}"
    si = scale.reshape(m, nblk // alignment, alignment)  # [M, nblk/4, 4]
    si = si.permute(1, 0, 2)  # [nblk/4, M, 4]
    si = si.reshape(nblk // alignment, m * alignment)  # [nblk/4, M*4]
    return si.contiguous()


def quantize_activation_mxfp8_blockwise_grouped(
    x: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE, pad_even: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 block-quant helper: quantize an activation tensor to FP8
    (e4m3) data + a per-token+per-block bf16 scale, padding the token dimension
    to an even count so the kernel's TMA layout constraints are satisfied.

    This is the grouped counterpart of ``quantize_activation_mxfp8_blockwise``:
    it applies the same per-token + per-block (block_size along K) real-valued
    ``amax / 448`` scale, but additionally pads M up to the next even value
    (matching the kernel's even-padding requirement) with zeros.

    Args:
        x: activation ``[M, K]`` (bf16/fp16/fp32); K must be divisible by
            ``block_size``.
        block_size: K-wise block size, default 32 (matches the E8M0 weight block).
        pad_even: if True (default), pad M up to the next even value with zeros.

    Returns:
        x_fp8: ``[M_padded, K]`` float8_e4m3fn, the block-scaled activation.
        scale: ``[M_padded, K // block_size]`` bf16, ``block_amax / 448`` per block.
    """
    assert x.dim() == 2, "activation must be 2D [M, K]"
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"

    if pad_even and (m % 2 != 0):
        pad = torch.zeros(1, k, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=0)
        m = x.shape[0]

    x_fp8, scale = quantize_activation_mxfp8_blockwise(x, block_size=block_size)
    return x_fp8, scale


def _build_grouped_act_block_scale_compact(
    scale: torch.Tensor,
    expert_offsets_host,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Compact builder used by the eager path.

    This keeps the buffer tightly packed by concatenating the per-expert blocks
    with an expert-local even padding ``M_pad``. It is numerically exact and
    memory-efficient, but it uses host-side expert offsets and dynamic-shape
    tensor construction, so it must stay out of CUDA graph capture.
    """
    device = scale.device
    num_experts = len(expert_offsets_host) - 1
    nblk = scale.shape[1]
    assert nblk % 4 == 0, f"K//block={nblk} must be a multiple of 4"

    blocks = []
    m_pads = []
    for e in range(num_experts):
        s0 = int(expert_offsets_host[e])
        s1 = int(expert_offsets_host[e + 1])
        se = scale[s0:s1]  # [M_e, nblk]
        m_e = se.shape[0]
        m_pad = (m_e + 1) & ~1  # round up to even
        if m_pad != m_e:
            pad = torch.zeros(m_pad - m_e, nblk, dtype=se.dtype, device=device)
            se = torch.cat([se, pad], dim=0)
        # 4-wide interleave over K-blocks: [M_pad, nblk] -> [nblk/4, M_pad*4]
        si = se.reshape(m_pad, nblk // 4, 4).permute(1, 0, 2).reshape(nblk // 4, m_pad * 4)
        blocks.append(si.reshape(-1).contiguous())
        m_pads.append(m_pad)

    as_packed = (
        torch.cat(blocks).contiguous()
        if blocks
        else torch.zeros(0, dtype=scale.dtype, device=device)
    )
    as_strides = torch.tensor(
        [[m_pads[e]] * 2 for e in range(num_experts)],
        dtype=torch.int64,
        device=device,
    )
    return as_packed, as_strides


def _build_grouped_act_block_scale_capture_safe(
    scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Graph-safe builder for decode/spec CUDA graph capture.

    Unlike the eager compact builder, this path never calls ``.cpu()`` /
    ``.tolist()`` and never creates dynamic-shape outputs from per-expert token
    counts. Instead, every expert uses the SAME fixed even token stride
    ``M_stride = round_up_even(total_m)``, and the per-expert scale block is
    stored in a dense buffer of shape ``[E, K//(block*4), M_stride*4]``.

    This wastes some memory versus the compact eager layout, but for decode graph
    capture ``total_m`` is the captured batch size times ``topk`` (small and
    static), while the fixed even stride removes the original TMA alignment
    issue for odd per-expert token counts and makes the whole builder
    CUDA-graph-recordable.
    """
    assert torch.is_tensor(expert_offsets), "capture-safe path requires tensor expert_offsets"
    device = scale.device
    nblk = scale.shape[1]
    assert nblk % 4 == 0, f"K//block={nblk} must be a multiple of 4"

    num_experts = expert_offsets.numel() - 1
    total_m = scale.shape[0]
    m_stride = max(2, (total_m + 1) & ~1)

    grouped = torch.zeros(
        (num_experts, m_stride, nblk),
        dtype=scale.dtype,
        device=device,
    )

    if total_m > 0:
        row_ids = torch.arange(total_m, device=device, dtype=expert_offsets.dtype)
        expert_ids = torch.searchsorted(expert_offsets[1:], row_ids, right=True)
        row_offsets = expert_offsets.index_select(0, expert_ids)
        local_rows = (row_ids - row_offsets).to(torch.long)
        grouped[expert_ids.to(torch.long), local_rows, :] = scale

    packed = (
        grouped.reshape(num_experts, m_stride, nblk // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(num_experts, nblk // 4, m_stride * 4)
        .contiguous()
    )
    as_packed = packed.reshape(-1).contiguous()
    as_strides = torch.full(
        (num_experts, 2),
        m_stride,
        dtype=torch.int64,
        device=device,
    )
    return as_packed, as_strides


def build_grouped_act_block_scale(
    scale: torch.Tensor,
    expert_offsets,
    block_size: int = MXFP4_BLOCK_SIZE,
    capture_safe: Optional[bool] = None,
):
    """Build the activation block-scale buffer + strides consumed by CUTLASS.

    There are two layouts:

    1. eager compact layout: per-expert concatenation with expert-local even
       padding ``M_pad`` (minimal memory footprint).
    2. capture-safe layout: fixed even ``M_stride`` for every expert, fully
       device-side and static-shape so it is CUDA-graph-recordable.

    The capture-safe layout is selected automatically inside CUDA graph capture,
    or can be forced explicitly by passing ``capture_safe=True`` for testing.
    """
    if capture_safe is None:
        capture_safe = get_is_capture_mode()

    if capture_safe:
        return _build_grouped_act_block_scale_capture_safe(
            scale, expert_offsets, block_size=block_size
        )

    if torch.is_tensor(expert_offsets):
        expert_offsets_host = expert_offsets.detach().to("cpu").tolist()
    else:
        expert_offsets_host = expert_offsets
    return _build_grouped_act_block_scale_compact(
        scale, expert_offsets_host, block_size=block_size
    )


# E2M1 nibble interleave order used by the int4fp8 ``pack_*_to_int32`` helper.
# NOTE: the CUTLASS mxfp4a8 kernel does NOT expect this reorder for its packed
# int8 weight operand (see ``repack_hf_mxfp4_to_kernel`` below).
_ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]


def repack_hf_mxfp4_to_kernel(w_uint8: torch.Tensor) -> torch.Tensor:
    """Convert HF-packed MXFP4 (E2M1) bytes into the kernel's int8 weight layout.

    HF stores two E2M1 nibbles per byte in *natural* order (nibble ``2j`` in the
    low half of byte ``j``, nibble ``2j+1`` in the high half). This is EXACTLY
    the byte layout the CUTLASS mxfp4a8 grouped-GEMM expects for its packed int8
    weight operand ``b = [E, N, K//2]``.

    This was verified empirically with a single-GEMM bit-exact comparison
    (``tests/test_cutlass_mxfp4a8_moe_mm.py``): the natural nibble packing
    reproduces the bf16-dequant golden to rel_mean = 0, while applying the
    ``order_map = [0, 2, 4, 6, 1, 3, 5, 7]`` reorder produces garbage
    (rel_mean ~= 1.2). The ``order_map`` reorder only applies to the int4a8
    ``pack_int4_to_int32`` path (which packs into int32 words with a different
    prmt-LUT layout), NOT to this int8 grouped-GEMM path.

    Therefore this routine is a bit-preserving identity: the HF bytes are passed
    through unchanged (reinterpreted as int8 so two's-complement high bytes keep
    their bit pattern for the kernel's per-nibble decode).

    Args:
        w_uint8: HF-packed weights ``[..., cols]`` (uint8/int8), ``cols`` bytes
            per row, i.e. ``2*cols`` E2M1 codes along the last logical dim.

    Returns:
        int8 tensor of the same shape, bit-identical to the input bytes.
    """
    assert w_uint8.dtype in (torch.uint8, torch.int8)
    if w_uint8.dtype == torch.int8:
        return w_uint8.contiguous()
    # uint8 -> int8 is a pure bit reinterpretation (both 1 byte), which is what
    # the kernel's per-nibble decode operates on.
    return w_uint8.view(torch.int8).contiguous()


def pack_mxfp4_to_int32(to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
    """Pack E2M1 nibble codes into int32 words using the SAME interleave layout
    as ``pack_int4_to_int32``. ``to_pack`` holds 4-bit codes (0..15) as int8.
    """
    if to_pack.ndim > 2:
        raise ValueError("Pack: Only supports tensors with ndim <= 2.")

    order_map = [0, 2, 4, 6, 1, 3, 5, 7] if reorder else [0, 1, 2, 3, 4, 5, 6, 7]
    pack_num = 8
    if to_pack.ndim == 2:
        new_c = to_pack.shape[1] // pack_num
        packed = torch.zeros(
            to_pack.shape[0], new_c, dtype=torch.int32, device=to_pack.device
        )
        for c in range(new_c):
            for i in range(pack_num):
                col = (to_pack[:, c * pack_num + order_map[i]].to(torch.int32)) & 0x0F
                packed[:, c] = torch.bitwise_or(
                    packed[:, c], torch.bitwise_left_shift(col, i * 4)
                )
    elif to_pack.ndim == 0:
        packed = to_pack.to(torch.int32)
    else:
        new_c = to_pack.shape[0] // pack_num
        packed = torch.zeros(new_c, dtype=torch.int32, device=to_pack.device)
        for c in range(new_c):
            for i in range(pack_num):
                col = (to_pack[c * pack_num + order_map[i]].to(torch.int32)) & 0x0F
                packed[c] = torch.bitwise_or(
                    packed[c], torch.bitwise_left_shift(col, i * 4)
                )

    return packed.view(torch.uint32)
