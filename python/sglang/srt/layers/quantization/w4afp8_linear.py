from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["cutlass_w4a8_fp8_linear"]


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

    The public call chain matches the future dense kernel entry point. Until the
    custom op is added, this function uses a reference path internally.
    """
    del input_scale

    assert (
        weight_packed.dtype == torch.int8
    ), "cutlass_w4a8_fp8_linear expects runtime int8-packed int4 weights"

    input_2d = input.reshape(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight_packed.shape[0]]
    output_dtype = output_dtype or input.dtype

    logger.warning_once(
        "cutlass_w4a8_fp8_linear is using the temporary reference path "
        "(dequantize + matmul)."
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
