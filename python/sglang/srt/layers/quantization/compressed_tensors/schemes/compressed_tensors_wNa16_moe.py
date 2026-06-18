from __future__ import annotations

import enum
import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat

from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
    gptq_marlin_moe_repack,
)
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4DynamicMoEMethod,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_moe_permute_scales,
)
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )


__all__ = [
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
    "NPUCompressedTensorsW4A16Int4DynamicMoE",
]

_is_hip = is_hip()
_is_cuda = is_cuda()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    pass


logger = logging.getLogger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class CompressedTensorsWNA16MoE(CompressedTensorsMoEScheme):

    def __init__(self, quant_config: CompressedTensorsConfig, num_gpu_experts=-1):
        self.quant_config = quant_config
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, "Only symmetric quantization is supported for MoE"

        if not (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
            and self.num_bits in WNA16_SUPPORTED_BITS
        ):
            raise ValueError(
                "For Fused MoE layers, only ",
                f"{CompressionFormat.pack_quantized.value} ",
                "is supported for the following bits: ",
                f"{WNA16_SUPPORTED_BITS}",
            )
        self.num_gpu_experts = num_gpu_experts

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1

        if load_full_w2:
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size
        else:
            w2_scales_size = intermediate_size_per_partition

        self.is_k_full = (not self.actorder) or layer.moe_tp_size == 1

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        # Force record: these are the target GPTQ shapes for rollback.
        layer._original_shapes["w13_weight_packed"] = tuple(w13_weight.shape)
        layer._original_shapes["w2_weight_packed"] = tuple(w2_weight.shape)

        # Also record the shapes of the scales.
        layer._original_shapes["w2_weight_scale"] = tuple(w2_scale.shape)
        layer._original_shapes["w13_weight_scale"] = tuple(w13_scale.shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # Skip if the layer is already converted to Marlin format to prevent double-packing.
        if getattr(layer, "is_marlin_converted", False):
            return

        # [DEBUG W4A16] dump weight statistics BEFORE marlin repack so we can
        # see whether the checkpoint actually wrote into the params or whether
        # they are still at their `torch.empty` init values.
        # Per-expert breakdown: separate routed experts (0..127) from the
        # fused shared expert at index 128 to confirm shared-expert weights
        # were actually loaded.
        try:
            wp = layer.w13_weight_packed.data
            ws = layer.w13_weight_scale.data
            w2p = layer.w2_weight_packed.data
            w2s = layer.w2_weight_scale.data
            num_experts = wp.shape[0]
            for tag, sl in (
                ("routed[0:1]", slice(0, 1)),
                ("routed[mid]", slice(num_experts // 2, num_experts // 2 + 1)),
                ("last_idx", slice(num_experts - 1, num_experts)),
            ):
                wps = wp[sl]
                wss = ws[sl]
                w2ps = w2p[sl]
                w2ss = w2s[sl]
                logger.warning(
                    "[DEBUG W4A16] per-expert stats tag=%s "
                    "w13_packed nz=%.4f scale_min=%s scale_max=%s scale_mean=%s "
                    "w2_packed nz=%.4f scale_min=%s scale_max=%s scale_mean=%s",
                    tag,
                    (wps != 0).float().mean().item(),
                    wss.min().item(),
                    wss.max().item(),
                    wss.mean().item(),
                    (w2ps != 0).float().mean().item(),
                    w2ss.min().item(),
                    w2ss.max().item(),
                    w2ss.mean().item(),
                )
            logger.warning(
                "[DEBUG W4A16] pre-marlin stats "
                "w13_weight_packed shape=%s dtype=%s nonzero_frac=%.4f "
                "w13_weight_scale shape=%s dtype=%s min=%s max=%s mean=%s",
                tuple(wp.shape),
                wp.dtype,
                (wp != 0).float().mean().item() if wp.numel() else 0.0,
                tuple(ws.shape),
                ws.dtype,
                ws.min().item() if ws.numel() else "n/a",
                ws.max().item() if ws.numel() else "n/a",
                ws.mean().item() if ws.numel() else "n/a",
            )
        except Exception as e:
            logger.warning("[DEBUG W4A16] pre-marlin stats failed: %s", e)

        # [DEBUG W4A16] Save original (pack-quantized) tensors for layer 3
        # expert 0 so we can dequantize them with both interpretations
        # (compressed-tensors pack-quantized layout vs gptq layout) AFTER the
        # marlin repack and detect a packing-order mismatch.
        try:
            if getattr(layer, "layer_id", -1) == 3:
                logger.warning(
                    "[DEBUG W4A16] preserving layer3 expert0 pre-marlin tensors for dequant cross-check"
                )
                layer._dbg_pre_w13_packed = layer.w13_weight_packed[0].detach().clone()
                layer._dbg_pre_w13_scale = layer.w13_weight_scale[0].detach().clone()
                layer._dbg_pre_w2_packed = layer.w2_weight_packed[0].detach().clone()
                layer._dbg_pre_w2_scale = layer.w2_weight_scale[0].detach().clone()
                layer._dbg_pack_factor = self.packed_factor
                layer._dbg_group_size = self.group_size
                layer._dbg_num_bits = self.num_bits
        except Exception as e:
            logger.warning("[DEBUG W4A16] saving pre-marlin tensors failed: %s", e)

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        def replace_tensor(name, new_t):
            target_attr = getattr(layer, name)

            # Only save if the key doesn't exist to prevent overwriting with Marlin shapes.
            if name not in layer._original_shapes:
                # This is a safety check; `create_weights` usually handles this already.
                layer._original_shapes[name] = tuple(target_attr.shape)

            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            target_attr.resize_(new_t.shape)
            target_attr.copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            layer.w13_weight_packed.shape[2],
            layer.w13_weight_scale.shape[2],
            self.group_size,
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            layer.w2_weight_scale.shape[2],
            self.group_size,
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)

        layer.workspace = marlin_make_workspace(layer.w13_weight_packed.device, 4)
        layer.is_marlin_converted = True

        # [DEBUG W4A16] Dequant cross-check for layer 3, expert 0. We
        # interpret the *original* `weight_packed` int32 tensor with two
        # candidate layouts:
        #   A) compressed-tensors "pack-quantized": rows are output channels
        #      of size N, cols are packed input channels (K/pack_factor) where
        #      each int32 stores 8 contiguous 4-bit signed values along K
        #      (least-significant nibble = lowest input channel).
        #   B) GPTQ-style packing: rows are packed input channels
        #      (K/pack_factor), cols are output channels N. Each int32 stores
        #      8 contiguous 4-bit signed values along K (LSB = lowest k).
        # If the ckpt is layout (A) but `gptq_marlin_moe_repack` assumes
        # layout (B), the dequantized values come out shuffled and the GEMM
        # output is noise. We dump the dequantized first row of w13.expert0
        # under both interpretations so we can compare against the pristine
        # source ckpt offline.
        try:
            if hasattr(layer, "_dbg_pre_w13_packed"):
                pack_factor = layer._dbg_pack_factor
                group_size = layer._dbg_group_size
                num_bits = layer._dbg_num_bits
                pre_q = layer._dbg_pre_w13_packed
                pre_s = layer._dbg_pre_w13_scale
                logger.warning(
                    "[DEBUG W4A16] layer3.expert0 w13 pre-marlin shapes "
                    "packed=%s dtype=%s scale=%s scale_dtype=%s pack_factor=%d group_size=%d num_bits=%d",
                    tuple(pre_q.shape), pre_q.dtype,
                    tuple(pre_s.shape), pre_s.dtype,
                    pack_factor, group_size, num_bits,
                )

                def _unpack_int4_lsb_first(t: torch.Tensor) -> torch.Tensor:
                    # Unpack int32 -> 8 signed 4-bit values along last dim,
                    # LSB nibble first. Result dtype int8 in [-8, 7].
                    t32 = t.to(torch.int32)
                    shifts = torch.arange(0, 32, 4, device=t32.device, dtype=torch.int32)
                    expanded = t32.unsqueeze(-1) >> shifts
                    nibbles = expanded & 0xF
                    nibbles = nibbles.where(nibbles < 8, nibbles - 16)
                    return nibbles.to(torch.int8).reshape(*t32.shape[:-1], t32.shape[-1] * 8)

                # Layout A: pack-quantized -> packed.shape == (N_outer, K_packed)
                # so unpack along last dim to get (N_outer, K)
                unpackedA = _unpack_int4_lsb_first(pre_q)
                # Layout B: gptq -> packed.shape == (K_packed, N) so unpack
                # along *first* dim. We emulate by transposing first.
                unpackedB = _unpack_int4_lsb_first(pre_q.t().contiguous())

                logger.warning(
                    "[DEBUG W4A16] layer3.expert0 w13 unpacked layoutA shape=%s "
                    "first8 row0=%s last8 row0=%s minmax=(%d,%d) hist=%s",
                    tuple(unpackedA.shape),
                    unpackedA[0, :8].tolist(),
                    unpackedA[0, -8:].tolist(),
                    int(unpackedA.min().item()), int(unpackedA.max().item()),
                    torch.bincount(
                        (unpackedA[0].to(torch.int32) + 8).clamp(0, 15)
                    ).tolist(),
                )
                logger.warning(
                    "[DEBUG W4A16] layer3.expert0 w13 unpacked layoutB shape=%s "
                    "first8 col0=%s last8 col0=%s minmax=(%d,%d) hist=%s",
                    tuple(unpackedB.shape),
                    unpackedB[0, :8].tolist(),
                    unpackedB[0, -8:].tolist(),
                    int(unpackedB.min().item()), int(unpackedB.max().item()),
                    torch.bincount(
                        (unpackedB[0].to(torch.int32) + 8).clamp(0, 15)
                    ).tolist(),
                )

                # Multiply by scale[group_0, :] and inspect dequantized values.
                # In compressed-tensors pack-quantized:
                #   weight.shape == (N, K), scale.shape == (N, K // group_size)
                # So pre_s.shape should be (N, num_groups). Verify and dequant
                # column 0 of group 0 across the first 4 output channels.
                if pre_s.dim() == 2:
                    s0 = pre_s[:4, 0].to(torch.float32)  # 4 output rows
                    # layoutA: (N, K), pick row 0..3, k=0
                    if unpackedA.shape[0] >= 4 and unpackedA.shape[1] >= 1:
                        wA = unpackedA[:4, 0].to(torch.float32) * s0
                        logger.warning(
                            "[DEBUG W4A16] layer3.expert0 w13 layoutA dequant n=0..3,k=0: %s "
                            "(scales=%s)", wA.tolist(), s0.tolist(),
                        )
                    if unpackedB.shape[0] >= 4 and unpackedB.shape[1] >= 1:
                        wB = unpackedB[:4, 0].to(torch.float32) * s0
                        logger.warning(
                            "[DEBUG W4A16] layer3.expert0 w13 layoutB dequant n=0..3,k=0: %s",
                            wB.tolist(),
                        )

                # Free debug copies to release memory before forward.
                del layer._dbg_pre_w13_packed
                del layer._dbg_pre_w13_scale
                del layer._dbg_pre_w2_packed
                del layer._dbg_pre_w2_scale
        except Exception as e:
            logger.warning("[DEBUG W4A16] dequant cross-check failed: %s", e)

    def restore_weights_before_loading(self, layer: torch.nn.Module):
        """Forcibly resize parameters back to their original shapes (e.g., GPTQ format) before loading weights."""

        if not hasattr(layer, "_original_shapes"):
            return

        for name, orig_shape in layer._original_shapes.items():
            param = getattr(layer, name, None)

            if param is not None and param.shape != orig_shape:
                param.resize_(orig_shape)

        layer.is_marlin_converted = False

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def get_marlin_quant_info(self, layer):
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

        return MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight_packed,
            w2_qweight=layer.w2_weight_packed,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=getattr(layer, "w13_g_idx_sort_indices", None),
            w2_g_idx_sort_indices=getattr(layer, "w2_g_idx_sort_indices", None),
            weight_bits=self.num_bits,
            w13_g_idx=getattr(layer, "w13_weight_g_idx", None),
            w2_g_idx=getattr(layer, "w2_weight_g_idx", None),
            is_k_full=self.is_k_full,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, router_logits = topk_output

        # Get expert_map for EP support
        expert_map = None
        global_num_experts = -1
        if hasattr(layer, "dispatcher") and hasattr(
            layer.dispatcher, "local_expert_mapping"
        ):
            expert_map = layer.dispatcher.local_expert_mapping
            if expert_map is not None:
                global_num_experts = self.moe_runner_config.num_experts

        output = fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
            workspace=layer.workspace,
        )
        return StandardCombineInput(hidden_states=output)


class CompressedTensorsWNA16TritonMoE(CompressedTensorsWNA16MoE):
    """ROCm/HIP-compatible W4A16 MoE method using Triton kernels instead of Marlin.

    Inherits weight creation from CompressedTensorsWNA16MoE but converts
    weights to the uint8-packed format expected by the Triton fused MoE kernel
    instead of the Marlin-specific format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_triton_converted", False):
            return

        num_experts = layer.w13_weight_packed.shape[0]

        # Convert w13 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8
        w13 = layer.w13_weight_packed.data
        w13 = w13.transpose(1, 2).contiguous().view(torch.uint8)
        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)

        # Convert w2 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8
        w2 = layer.w2_weight_packed.data
        w2 = w2.transpose(1, 2).contiguous().view(torch.uint8)
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)

        # Convert w13 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w13_scale = layer.w13_weight_scale.data
        w13_scale = w13_scale.transpose(1, 2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)

        # Convert w2 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w2_scale = layer.w2_weight_scale.data
        w2_scale = w2_scale.transpose(1, 2).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        layer.is_triton_converted = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight_packed,
            w2_weight=layer.w2_weight_packed,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=[0, self.group_size],
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        quant_info = self.get_triton_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)


class NPUCompressedTensorsW4A16Int4DynamicMoE(CompressedTensorsMoEScheme):

    def __init__(self, quantization_config) -> None:
        self.pack_factor = 8  # weight dtype is int4,  but use int32 to create
        target = (
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"
        )
        if target in quantization_config.target_scheme_map:
            self.group_size = quantization_config.target_scheme_map[target][
                "weights"
            ].group_size
        else:
            self.group_size = 128

        self.kernel = NPUW4A16Int4DynamicMoEMethod()

    # TODO: See if we can merge this method's logic
    # with CompressedTensorsWNA16MoE. Need more models and tests.
    # @OrangeRedeng @TamirBaydasov
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        if (
            extra_weight_attrs.get(
                "moe_intermediate_size", intermediate_size_per_partition
            )
            // intermediate_size_per_partition
            > 1
        ):
            quant_method = FusedMoeWeightScaleSupported.GROUP.value
        else:
            quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        extra_weight_attrs.update({"quant_method": quant_method})
        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        weight_scale_dtype = torch.bfloat16
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.kernel.apply(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return self.kernel.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
