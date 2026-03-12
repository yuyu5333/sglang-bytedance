from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs


class W4A8MoEFp8Config(QuantizationConfig):
    def __init__(
        self,
        moe_activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_moe_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4A8MoEFp8Config":
        moe_activation_scheme = "static"
        group_size = int(cls.get_from_keys_or(config, ["group_size"], default=128))
        return cls(moe_activation_scheme=moe_activation_scheme, group_size=group_size)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return UnquantizedLinearMethod()
        if isinstance(layer, FusedMoE):
            return W4A8MoEFp8Method(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W4A8MoEFp8Method(FusedMoEMethodBase):
    def __init__(self, quant_config: W4A8MoEFp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert "weight_loader" in extra_weight_attrs
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        tgt_group = int(self.quant_config.group_size)
        if hidden_size % 8 != 0 or intermediate_size_per_partition % 8 != 0:
            raise ValueError("hidden_size and intermediate_size must be divisible by 8.")
        if hidden_size % tgt_group != 0 or intermediate_size_per_partition % tgt_group != 0:
            raise ValueError("hidden_size/intermediate must be divisible by group_size.")

        kg13 = hidden_size // tgt_group
        kg2 = intermediate_size_per_partition // tgt_group
        align13 = 4 if kg13 % 4 == 0 else 1
        align2 = 4 if kg2 % 4 == 0 else 1
        self._w13_scale_alignment = align13
        self._w2_scale_alignment = align2

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        scale_weight_attrs = dict(extra_weight_attrs)
        scale_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )

        w13_weight_scale_inv = torch.nn.Parameter(
            torch.empty(
                num_experts,
                kg13 // align13,
                (2 * intermediate_size_per_partition) * align13,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale_inv)
        set_weight_attrs(w13_weight_scale_inv, scale_weight_attrs)

        w2_weight_scale_inv = torch.nn.Parameter(
            torch.empty(
                num_experts,
                kg2 // align2,
                hidden_size * align2,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale_inv)
        set_weight_attrs(w2_weight_scale_inv, scale_weight_attrs)

        w13_input_scale = torch.nn.Parameter(
            torch.ones((1,), dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones((1,), dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        device = layer.w13_weight.device
        self.a_strides1 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (num_experts, 3),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

        self.expert_offsets = torch.empty(
            (num_experts + 1), dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        self.moe_runner_config = moe_runner_config

    def apply(self, layer: Module, dispatch_output) -> Any:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        output = cutlass_w4a8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
        )
        return StandardCombineInput(hidden_states=output)
