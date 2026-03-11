from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.quantization.w4afp8 import interleave_scales
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

FP8_ABS_MAX = 448.0


def _unpack_int4_from_int32(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32:
        raise ValueError(f"Expected int32 packed tensor, got {packed.dtype}")

    x = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int64)
    nibbles = (x.unsqueeze(-1) >> shifts) & 0xF
    nibbles = nibbles.to(torch.int16)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).to(torch.int8)
    out = signed.reshape(*packed.shape[:-1], packed.shape[-1] * 8)
    return out


def _pack_int4_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.dtype != torch.int8:
        int4_values_interleaved = int4_values_interleaved.to(torch.int8)
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError("Last dim must be even for int4 packing.")

    low = int4_values_interleaved[..., 0::2]
    high = int4_values_interleaved[..., 1::2]
    packed = (high << 4) | (low & 0x0F)
    return packed.to(torch.int8)


def _requant_groupwise_int4(
    weight_fp: torch.Tensor, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight_fp.dim() != 3:
        raise ValueError(f"Expected [E, O, I] weight, got {tuple(weight_fp.shape)}")
    e, o, i = weight_fp.shape
    if i % group_size != 0:
        raise ValueError(f"I={i} must be divisible by group_size={group_size}")
    g = i // group_size
    w = weight_fp.to(torch.float32).reshape(e, o, g, group_size)
    max_abs = w.abs().amax(dim=-1).clamp(min=1e-12)
    scale = max_abs / 7.0
    q = torch.round(w / scale.unsqueeze(-1)).clamp(min=-8, max=7).to(torch.int8)
    return q.reshape(e, o, i), scale.to(torch.float32)


def _dequant_groupwise_int4_from_int32(
    packed_int32: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    q_int4 = _unpack_int4_from_int32(packed_int32)
    if q_int4.dim() != 3:
        raise ValueError(f"Expected [E,O,I] unpacked, got {tuple(q_int4.shape)}")
    e, o, i = q_int4.shape
    if scale.shape[:2] != (e, o):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} incompatible with (E,O)=({e},{o})"
        )
    if scale.dim() != 3:
        raise ValueError(f"Expected scale [E,O,G], got {tuple(scale.shape)}")
    g = scale.shape[2]
    if i != g * group_size:
        raise ValueError(
            f"Unpacked I={i} must equal scale_G={g} * group_size={group_size}"
        )
    w = q_int4.to(torch.float32).reshape(e, o, g, group_size) * scale.to(
        torch.float32
    ).unsqueeze(-1)
    return w.reshape(e, o, i).to(torch.bfloat16)


def _interleave_scales_2d(scales: torch.Tensor, alignment: int = 4) -> torch.Tensor:
    if scales.dim() != 2:
        raise ValueError(f"Expected 2D scales [O, Kg], got {tuple(scales.shape)}")
    o, kg = scales.shape
    if kg % alignment != 0:
        raise ValueError(f"Kg={kg} must be divisible by alignment={alignment}")
    t = scales.reshape(o, kg // alignment, alignment).permute(1, 0, 2).reshape(
        kg // alignment, o * alignment
    )
    return t


def _requant_and_pack_one_expert_from_group32_to_group128(
    packed_int32: torch.Tensor,
    scale32: torch.Tensor,
    src_group_size: int,
    dst_group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if src_group_size != 32 or dst_group_size != 128:
        raise ValueError("Only (src_group_size=32 -> dst_group_size=128) is supported.")
    if packed_int32.dtype != torch.int32:
        raise ValueError(f"Expected int32 packed, got {packed_int32.dtype}")
    if scale32.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unexpected scale dtype: {scale32.dtype}")

    q = _unpack_int4_from_int32(packed_int32).to(torch.int16)
    o, i = q.shape
    if i % 128 != 0:
        raise ValueError(f"I={i} must be divisible by 128")

    kg32 = i // 32
    kg128 = i // 128
    if scale32.shape != (o, kg32):
        raise ValueError(f"scale32 shape {tuple(scale32.shape)} expected {(o, kg32)}")

    q = q.reshape(o, kg32, 32)
    s32 = scale32.to(torch.float32).reshape(o, kg32, 1)

    prod = (q.abs().to(torch.float32) * s32).reshape(o, kg128, 4, 32)
    absmax128 = prod.amax(dim=-1).amax(dim=-1).clamp(min=1e-12)
    s128 = absmax128 / 7.0

    q32_f = q.to(torch.float32)
    q128 = torch.empty((o, kg32, 32), dtype=torch.int16)
    for sub in range(4):
        g32_slice = slice(sub, kg32, 4)
        g128_slice = slice(0, kg128)
        num = q32_f[:, g32_slice, :] * s32[:, g32_slice, :]
        den = s128[:, g128_slice].unsqueeze(-1)
        q_sub = torch.round(num / den).clamp(min=-8, max=7).to(torch.int16)
        q128[:, g32_slice, :] = q_sub

    q128 = q128.reshape(o, i).to(torch.int8)
    packed_int8 = _pack_int4_to_int8(q128).contiguous()
    scales128 = s128.to(torch.float32).contiguous()
    return packed_int8, scales128


class W4A8MoEFp8OnlineConfig(QuantizationConfig):
    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        source_group_size: int = 32,
        target_group_size: int = 128,
        calibration_steps: int = 1,
        calibration_max_tokens: int = 2048,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.source_group_size = source_group_size
        self.target_group_size = target_group_size
        self.calibration_steps = calibration_steps
        self.calibration_max_tokens = calibration_max_tokens

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_moe_fp8_online"

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
    def from_config(cls, config: Dict[str, Any]) -> "W4A8MoEFp8OnlineConfig":
        source_group_size = cls.get_from_keys_or(config, ["source_group_size"], default=None)
        if source_group_size is None:
            try:
                config_groups = config.get("config_groups", {})
                if isinstance(config_groups, dict):
                    for group_cfg in config_groups.values():
                        if not isinstance(group_cfg, dict):
                            continue
                        weights_cfg = group_cfg.get("weights", {})
                        if not isinstance(weights_cfg, dict):
                            continue
                        if "group_size" in weights_cfg:
                            source_group_size = weights_cfg["group_size"]
                            break
            except Exception:
                source_group_size = None
        source_group_size = int(source_group_size if source_group_size is not None else 32)
        target_group_size = int(
            cls.get_from_keys_or(config, ["group_size", "target_group_size"], default=128)
        )
        calibration_steps = int(
            cls.get_from_keys_or(config, ["calibration_steps"], default=1)
        )
        calibration_max_tokens = int(
            cls.get_from_keys_or(config, ["calibration_max_tokens"], default=2048)
        )
        return cls(
            source_group_size=source_group_size,
            target_group_size=target_group_size,
            calibration_steps=calibration_steps,
            calibration_max_tokens=calibration_max_tokens,
        )

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
            return W4A8MoEFp8OnlineMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W4A8MoEFp8OnlineMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: W4A8MoEFp8OnlineConfig):
        self.quant_config = quant_config
        self.process_weights_on_cpu = True

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

        src_group = int(self.quant_config.source_group_size)
        tgt_group = int(self.quant_config.target_group_size)
        if hidden_size % 8 != 0 or intermediate_size_per_partition % 8 != 0:
            raise ValueError("hidden_size and intermediate_size must be divisible by 8.")
        if hidden_size % tgt_group != 0 or intermediate_size_per_partition % tgt_group != 0:
            raise ValueError("hidden_size/intermediate must be divisible by target_group_size.")
        if hidden_size % src_group != 0 or intermediate_size_per_partition % src_group != 0:
            raise ValueError("hidden_size/intermediate must be divisible by source_group_size.")

        w13_packed_cols = hidden_size // 8
        w2_packed_cols = intermediate_size_per_partition // 8

        w13_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                w13_packed_cols,
                dtype=torch.int32,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight_packed)
        set_weight_attrs(w13_weight_packed, extra_weight_attrs)

        w2_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_packed_cols,
                dtype=torch.int32,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight_packed)
        set_weight_attrs(w2_weight_packed, extra_weight_attrs)

        scale_weight_attrs = dict(extra_weight_attrs)
        scale_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // src_group,
                dtype=torch.bfloat16,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // src_group,
                dtype=torch.bfloat16,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_weight_attrs)

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

        w13_weight_scale_inv = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // tgt_group,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale_inv)
        set_weight_attrs(w13_weight_scale_inv, scale_weight_attrs)

        w2_weight_scale_inv = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // tgt_group,
                dtype=torch.float32,
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

        layer._w4a8_calibration_steps_left = int(self.quant_config.calibration_steps)
        layer._w4a8_calibration_max_tokens = int(self.quant_config.calibration_max_tokens)
        layer._w4a8_a1_max_abs = 0.0
        layer._w4a8_a2_max_abs = 0.0

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

        self.expert_offsets = torch.empty((num_experts + 1), dtype=torch.int32, device=device)
        self.problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
        self.problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)

    def process_weights_after_loading(self, layer: Module) -> None:
        src_group = int(self.quant_config.source_group_size)
        tgt_group = int(self.quant_config.target_group_size)
        if src_group != 32 or tgt_group != 128:
            raise ValueError("w4a8_moe_fp8_online currently supports group32 -> group128 only.")

        device = layer.w13_weight.device
        num_experts = layer.w13_weight.shape[0]
        for e in range(num_experts):
            w13_packed_int8, w13_scale128 = _requant_and_pack_one_expert_from_group32_to_group128(
                layer.w13_weight_packed[e],
                layer.w13_weight_scale[e],
                src_group_size=src_group,
                dst_group_size=tgt_group,
            )
            w2_packed_int8, w2_scale128 = _requant_and_pack_one_expert_from_group32_to_group128(
                layer.w2_weight_packed[e],
                layer.w2_weight_scale[e],
                src_group_size=src_group,
                dst_group_size=tgt_group,
            )

            layer.w13_weight.data[e].copy_(w13_packed_int8.to(device))
            layer.w2_weight.data[e].copy_(w2_packed_int8.to(device))

            w13_scale_packed = _interleave_scales_2d(w13_scale128).to(torch.bfloat16)
            w2_scale_packed = _interleave_scales_2d(w2_scale128).to(torch.bfloat16)

            layer.w13_weight_scale_inv.data[e].copy_(w13_scale_packed.to(device))
            layer.w2_weight_scale_inv.data[e].copy_(w2_scale_packed.to(device))

        for name in ["w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"]:
            if hasattr(layer, name):
                try:
                    delattr(layer, name)
                except Exception:
                    pass
                if name in getattr(layer, "_parameters", {}):
                    layer._parameters.pop(name, None)

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        self.moe_runner_config = moe_runner_config

    def apply(self, layer: Module, dispatch_output) -> Any:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import (
            cutlass_w4a8_moe,
            cutlass_w4a8_moe_calibrate,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        if hasattr(layer, "_w4a8_calibration_steps_left") and layer._w4a8_calibration_steps_left > 0:
            max_tokens = int(getattr(layer, "_w4a8_calibration_max_tokens", x.shape[0]))
            x_calib = x[:max_tokens]
            topk_weights_calib = topk_weights[:max_tokens]
            topk_ids_calib = topk_ids[:max_tokens]

            output, a1_max, a2_max = cutlass_w4a8_moe_calibrate(
                x_calib,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk_weights_calib,
                topk_ids_calib,
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
                routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
            )

            layer._w4a8_a1_max_abs = max(float(layer._w4a8_a1_max_abs), float(a1_max))
            layer._w4a8_a2_max_abs = max(float(layer._w4a8_a2_max_abs), float(a2_max))
            layer._w4a8_calibration_steps_left -= 1
            if layer._w4a8_calibration_steps_left <= 0:
                a1_scale = max(layer._w4a8_a1_max_abs / FP8_ABS_MAX, 1e-12)
                a2_scale = max(layer._w4a8_a2_max_abs / FP8_ABS_MAX, 1e-12)
                layer.w13_input_scale = Parameter(
                    torch.tensor([a1_scale], dtype=torch.float32, device=x.device),
                    requires_grad=False,
                )
                layer.w2_input_scale = Parameter(
                    torch.tensor([a2_scale], dtype=torch.float32, device=x.device),
                    requires_grad=False,
                )

            if x_calib.shape[0] != x.shape[0]:
                output_full = cutlass_w4a8_moe(
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
                return StandardCombineInput(hidden_states=output_full)
            return StandardCombineInput(hidden_states=output)

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
