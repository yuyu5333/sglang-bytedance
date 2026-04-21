from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
)
from sglang.srt.layers.quantization.w4afp8_kernel import (
    is_w4a8_fp8_linear_supported,
    w4a8_fp8_scaled_mm,
)

logger = logging.getLogger(__name__)

__all__ = [
    "cutlass_w4a8_fp8_linear",
    "quantize_input_to_fp8",
    "quantize_input_to_fp8_per_token",
]


def _unpack_int4_from_int8_packed(weight_packed_int8: torch.Tensor) -> torch.Tensor:
    """Expand int8-packed signed int4 values into one int8 per element."""
    packed = weight_packed_int8.to(torch.int16)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    unpacked = torch.stack((low, high), dim=-1)
    return unpacked.reshape(*weight_packed_int8.shape[:-1], -1).to(torch.int8)


def _dequantize_w4_groupwise(
    weight_packed_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference-only path used until the dense W4A8-FP8 kernel lands."""
    weight_int4 = _unpack_int4_from_int8_packed(weight_packed_int8).to(torch.float32)
    num_rows, input_size = weight_int4.shape
    num_groups = input_size // group_size

    assert (
        weight_scale.shape[0] == num_rows
    ), "weight_scale rows must match the weight rows"
    assert (
        weight_scale.shape[1] == num_groups
    ), "weight_scale columns must match the number of groups"

    dequant = weight_int4.reshape(num_rows, num_groups, group_size)
    dequant = dequant * weight_scale.to(torch.float32).unsqueeze(-1)
    return dequant.reshape(num_rows, input_size).to(output_dtype)


def quantize_input_to_fp8_per_token(
    input_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D activation tensor with dynamic per-token FP8 scales."""
    if input_2d.ndim != 2:
        raise ValueError(f"`input_2d` must be 2D, got shape={tuple(input_2d.shape)}")

    input_2d = input_2d.contiguous()
    return sglang_per_token_quant_fp8(input_2d)


def quantize_input_to_fp8(
    input_2d: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified activation quantization entry for dense W4A8-FP8 linear."""
    if input_2d.ndim != 2:
        raise ValueError(f"`input_2d` must be 2D, got shape={tuple(input_2d.shape)}")

    input_2d = input_2d.contiguous()
    if input_scale is None:
        return quantize_input_to_fp8_per_token(input_2d)

    if input_scale.numel() != 1:
        raise ValueError(
            "Static `input_scale` is currently expected to be scalar. "
            f"Got shape={tuple(input_scale.shape)}"
        )

    q_input, x_scale = scaled_fp8_quant(input_2d, input_scale)
    if x_scale.ndim == 1:
        x_scale = x_scale.view(1, 1).expand(input_2d.shape[0], 1)
    return q_input.contiguous(), x_scale.contiguous().to(torch.float32)


def cutlass_w4a8_fp8_linear(
    input: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dense W4AFP8 linear wrapper.

    The preferred path is: input -> FP8 quantization -> dense W4A8-FP8 kernel.
    A reference fallback is kept temporarily for unsupported runtimes.
    """
    assert (
        weight_packed.dtype == torch.int8
    ), "cutlass_w4a8_fp8_linear expects runtime int8-packed int4 weights"

    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    output_shape = [*input.shape[:-1], weight_packed.shape[0]]
    output_dtype = output_dtype or input.dtype

    if is_w4a8_fp8_linear_supported():
        q_input, x_scale = quantize_input_to_fp8(
            input_2d=input_2d,
            input_scale=input_scale,
        )
        output = w4a8_fp8_scaled_mm(
            q_input=q_input,
            weight_packed=weight_packed.contiguous(),
            x_scale=x_scale,
            weight_scale=weight_scale.contiguous(),
            group_size=group_size,
            out_dtype=output_dtype,
            bias=bias.contiguous() if bias is not None else None,
        )
        return output.reshape(*output_shape)

    logger.warning_once(
        "cutlass_w4a8_fp8_linear is falling back to the temporary reference path "
        "(dequantize + matmul) because the dense W4A8-FP8 kernel is unavailable."
    )

    weight = _dequantize_w4_groupwise(
        weight_packed_int8=weight_packed,
        weight_scale=weight_scale,
        group_size=group_size,
        output_dtype=output_dtype,
    )

    output = input_2d.to(output_dtype) @ weight.t()
    if bias is not None:
        output = output + bias
    return output.reshape(*output_shape)
