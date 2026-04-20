from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch
from compressed_tensors import CompressionFormat
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.w4afp8_linear import cutlass_w4a8_fp8_linear

if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs

    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

__all__ = ["CompressedTensorsW4AFP8"]


def _unpack_repack_int32_to_cutlass_int8(
    weight_packed: torch.Tensor, num_bits: int
) -> torch.Tensor:
    """Convert pack_to_int32 layout into int8-packed signed int4 layout."""
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    offset = 1 << (num_bits - 1)
    pair_factor = pack_factor // 2

    out = torch.empty(
        (*weight_packed.shape[:-1], weight_packed.shape[-1], pair_factor),
        dtype=torch.int8,
        device=weight_packed.device,
    )
    for pair_idx in range(pair_factor):
        low_shift = num_bits * (2 * pair_idx)
        high_shift = low_shift + num_bits

        low_nibbles = ((weight_packed >> low_shift) & mask) - offset
        high_nibbles = ((weight_packed >> high_shift) & mask) - offset
        out[..., pair_idx] = ((high_nibbles << 4) | (low_nibbles & 0x0F)).to(
            torch.int8
        )

    return out.flatten(-2).contiguous()

class CompressedTensorsW4AFP8(CompressedTensorsLinearScheme):
    """Dense W4AFP8 linear scheme for compressed-tensors checkpoints.

    This implementation follows the compressed-tensors parameter registration
    contract so it can participate in the existing weight-loading flow.
    `apply_weights()` dispatches through `w4afp8_linear.py`, so the future dense
    kernel can replace the temporary implementation without changing the scheme.
    """

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",
        weight_quant: "QuantizationArgs",
        input_quant: "QuantizationArgs",
    ) -> None:
        self.quant_config = quant_config
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // self.num_bits
        self.group_size = weight_quant.group_size

        assert self.num_bits == 4, "CompressedTensorsW4AFP8 requires INT4 weights"
        assert self.group_size is not None, "W4AFP8 requires a group_size"
        assert weight_quant.symmetric, "Only symmetric INT4 weights are supported"
        assert not weight_quant.dynamic, "Dynamic weight quantization is unsupported"
        assert (
            input_quant is not None and input_quant.dynamic
        ), "W4AFP8 requires dynamic activation quantization"
        assert (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
        ), f"W4AFP8 requires pack-quantized format, got {self.quant_config.quant_format}"

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def _validate_shapes(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
    ) -> None:
        assert output_size_per_partition > 0, "Output size per partition must be > 0"
        assert (
            input_size_per_partition % self.packed_factor == 0
        ), "Input size per partition must align with int32 packing"
        assert (
            input_size_per_partition % self.group_size == 0
        ), "Input size per partition must align with group_size"

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        del input_size, output_size, kwargs

        output_size_per_partition = sum(output_partition_sizes)
        self._validate_shapes(input_size_per_partition, output_size_per_partition)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.group_size = self.group_size

        weight_packed = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.packed_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            packed_factor=self.packed_factor,
            packed_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight_packed)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # Compressed-tensors checkpoints commonly carry the original unpacked
        # shape alongside packed weights. Registering it avoids "not found in
        # params_dict" warnings during load even though inference does not need it.
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_w4afp8_converted", False):
            return

        weight_packed = layer.weight_packed.data
        if weight_packed.dtype == torch.int32:
            weight_packed = _unpack_repack_int32_to_cutlass_int8(
                weight_packed, self.num_bits
            )

        layer.weight_packed = Parameter(weight_packed.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data.contiguous(), requires_grad=False
        )
        layer.is_w4afp8_converted = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_packed = layer.weight_packed
        if weight_packed.dtype == torch.int32:
            weight_packed = _unpack_repack_int32_to_cutlass_int8(
                weight_packed, self.num_bits
            )

        return cutlass_w4a8_fp8_linear(
            input=x,
            weight_packed=weight_packed,
            weight_scale=layer.weight_scale,
            group_size=self.group_size,
            bias=bias,
            output_dtype=x.dtype,
        )
