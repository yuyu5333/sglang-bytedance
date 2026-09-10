"""Budgeted adapters for standard local MoE execution.

Unsupported modes retain their original path with an explicit warning. No
collective, pre-permute allocation or device routing read is performed here.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe.workspace_policy import (
    MarlinWorkspaceEstimate,
    TritonWorkspaceEstimate,
    plan_workspace,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _warn_fallback(backend: str, reason: str) -> None:
    logger.warning(
        "MoE workspace budget is not enforced for %s: %s; using the original path",
        backend,
        reason,
    )


def _common_unsupported_reason(runner, dispatch, lora_info, custom_core):
    if torch.compiler.is_compiling():
        return "torch.compile"
    if custom_core:
        return "custom runner core"
    if runner.lora_enabled or lora_info is not None:
        return "LoRA/hooks"
    if dispatch.format.value != "standard":
        return "non-standard dispatch"
    if (
        runner.down_gemm_overlap_args is not None
        or runner.meta_overlap_args is not None
    ):
        return "communication/computation overlap"
    config = runner.config
    if config.no_combine or config.use_tp_all_gather_activation:
        return "no_combine or activation all-gather"
    if config.apply_router_weight_on_input:
        return "router weighting on input"
    if not config.is_gated or any(
        value is not None
        for value in (
            config.gemm1_alpha,
            config.gemm1_beta,
            config.gemm1_clamp_limit,
            config.swiglu_limit,
        )
    ):
        return "non-gated or modified activation"
    if (
        dispatch.hidden_states_scale is not None
        or dispatch.hidden_states_pre_quant is not None
    ):
        return "pre-quantized dispatch"
    x = dispatch.hidden_states
    if x.device.type != "cuda" or torch.version.hip is not None:
        return "only CUDA adapters are implemented"
    if (
        x.ndim != 2
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
    ):
        return "activation shape/dtype/strides"
    topk = dispatch.topk_output
    if not hasattr(topk, "topk_weights"):
        return "packed or ragged top-k"
    if (
        topk.topk_ids.ndim != 2
        or topk.topk_ids.shape != topk.topk_weights.shape
        or topk.topk_ids.shape[0] != x.shape[0]
        or topk.topk_ids.shape[1] == 0
        or topk.topk_weights.dtype != torch.float32
        or topk.topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk.topk_ids.is_contiguous()
        or not topk.topk_weights.is_contiguous()
        or topk.topk_ids.device != x.device
        or topk.topk_weights.device != x.device
    ):
        return "top-k shape/dtype/strides/device"
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec

    if not get_moe_a2a_backend().is_none():
        return "all-to-all backend"
    if get_exec().moe.enable_fused_moe_sum_all_reduce or is_allocation_symmetric():
        return "fused collective or symmetric-memory allocation"
    if is_batch_invariant_mode_enabled():
        return "batch-invariant execution"
    if envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get():
        return "experimental LoRA routing"
    return None


class MarlinWorkspaceAdapter:
    @staticmethod
    def run(dispatch, quant, config, budget):
        from sglang.srt.layers.moe.moe_runner.marlin import (
            MarlinMoeQuantInfo,
            fused_experts_none_to_marlin,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        if not isinstance(quant, MarlinMoeQuantInfo):
            return None, "non-Marlin quantization metadata"
        if config.activation != "silu":
            return None, "Marlin adapter requires gated SiLU"
        if any(
            value is not None and value.numel()
            for value in (
                quant.w13_g_idx,
                quant.w2_g_idx,
                quant.w13_g_idx_sort_indices,
                quant.w2_g_idx_sort_indices,
            )
        ):
            return None, "activation-order permutation"
        if quant.expert_map is not None:
            return None, "expert-parallel mapping"
        x = dispatch.hidden_states
        mxfp4 = (
            quant.weight_bits == 4
            and quant.w13_qzeros is None
            and quant.w2_qzeros is None
            and quant.w13_scales.dtype == torch.float8_e8m0fnu
            and quant.w2_scales.dtype == torch.float8_e8m0fnu
        )
        if mxfp4 and x.dtype != torch.bfloat16:
            return None, "MXFP4 activation conversion"
        if quant.weight_bits not in (4, 8):
            return None, "unsupported Marlin weight bits"
        if quant.w13_qweight.ndim != 3 or quant.w2_qweight.ndim != 3:
            return None, "invalid Marlin weight geometry"
        experts = quant.w13_qweight.shape[0]
        hidden, intermediate = x.shape[1], quant.w2_qweight.shape[1] * 16
        if min(experts, hidden, intermediate) <= 0:
            return None, "empty Marlin weight geometry"
        if quant.global_num_experts not in (-1, experts):
            return None, "global/local expert count mismatch"
        destination = zero_copy_context.get_moe_output(x)
        if (
            destination is not None
            and destination.untyped_storage().data_ptr()
            == x.untyped_storage().data_ptr()
            and destination.data_ptr() != x.data_ptr()
        ):
            return None, "offset input/output alias"
        if x.shape[0] == 0:
            # The regular Marlin runner allocates locks even for an empty batch.
            return StandardCombineInput(
                destination if destination is not None else torch.empty_like(x)
            ), None
        props = torch.cuda.get_device_properties(x.device)
        atomic = (x.dtype == torch.float16 or props.major >= 9) and not mxfp4
        estimate = MarlinWorkspaceEstimate(
            hidden,
            intermediate,
            dispatch.topk_output.topk_ids.shape[1],
            experts,
            props.multi_processor_count,
            not atomic,
        )
        plan = plan_workspace(x.shape[0], budget, estimate)
        # An explicit cap bypasses the old token-only experiment.
        output = fused_experts_none_to_marlin(
            dispatch, quant, config, chunk_size=plan.chunk_tokens
        )
        return output, None


class TritonWorkspaceAdapter:
    @staticmethod
    def run(dispatch, quant, config, budget):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
            _resolve_fused_moe_config,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        if not isinstance(quant, TritonMoeQuantInfo):
            return None, "non-Triton quantization metadata"
        if any(
            (
                quant.use_mxfp8,
                quant.use_fp8_w8a8,
                quant.use_int8_w8a8,
                quant.use_int8_w8a16,
                quant.use_int4_w4a16,
                quant.fuse_swiglu_interleaved,
            )
        ) or any(
            value is not None
            for value in (
                quant.w13_scale,
                quant.w2_scale,
                quant.w13_zp,
                quant.w2_zp,
                quant.a13_scale,
                quant.a2_scale,
                quant.block_shape,
            )
        ):
            return None, "quantized or interleaved Triton layout"
        if config.activation not in ("silu", "gelu"):
            return None, "Triton adapter requires gated SiLU/GELU"
        x, topk = dispatch.hidden_states, dispatch.topk_output
        w1, w2 = quant.w13_weight, quant.w2_weight
        if (
            w1.ndim != 3
            or w2.ndim != 3
            or w1.shape[0] != w2.shape[0]
            or w1.shape[1] != 2 * w2.shape[2]
            or w1.shape[2] != x.shape[1]
            or w2.shape[1] != x.shape[1]
            or w1.dtype != x.dtype
            or w2.dtype != x.dtype
            or w1.device != x.device
            or w2.device != x.device
            or not w1.is_contiguous()
            or not w2.is_contiguous()
            or min(w1.shape) <= 0
        ):
            return None, "Triton weight geometry/dtype/strides"
        if x.shape[0] == 0:
            return StandardCombineInput(
                x if config.inplace else torch.empty_like(x)
            ), None
        launch, down, down_tma, up_tma = _resolve_fused_moe_config(
            x,
            w1,
            w2,
            topk.topk_ids,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        if down_tma or up_tma:
            return None, "TMA layout"
        estimate = TritonWorkspaceEstimate(
            x.shape[1],
            w2.shape[2],
            topk.topk_ids.shape[1],
            w1.shape[0],
            launch["BLOCK_SIZE_M"],
        )
        plan = plan_workspace(x.shape[0], budget, estimate)
        cap, routes = plan.chunk_tokens, plan.chunk_tokens * estimate.topk
        scratch = (
            x.new_empty((routes, 2 * estimate.intermediate)),
            x.new_empty((routes, estimate.intermediate)),
            x.new_empty((cap, estimate.topk, estimate.hidden)),
        )
        output = x if config.inplace else torch.empty_like(x)
        for start in range(0, x.shape[0], cap):
            end = min(start + cap, x.shape[0])
            ids = topk.topk_ids[start:end]
            sorted_ids, expert_ids, num_padded = moe_align_block_size(
                ids, launch["BLOCK_SIZE_M"], estimate.experts
            )
            _fused_moe_kernel_sequence(
                x[start:end],
                w1,
                w2,
                topk.topk_weights[start:end],
                ids,
                sorted_ids,
                expert_ids,
                num_padded,
                launch,
                down,
                False,
                False,
                b1=quant.b13,
                b2=quant.b2,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                w1_scale=None,
                w2_scale=None,
                w1_zp=None,
                w2_zp=None,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation=config.activation,
                is_gated=True,
                no_combine=False,
                inplace=config.inplace,
                apply_router_weight_on_input=False,
                routed_scaling_factor=config.routed_scaling_factor,
                gemm1_alpha=None,
                gemm1_limit=None,
                filter_expert=(
                    config.num_experts is None
                    or config.num_experts != config.num_local_experts
                ),
                output_buffer=output[start:end],
                scratch=scratch,
            )
        return StandardCombineInput(output), None


def run_with_workspace_budget(
    runner, dispatch, quant, lora_info, budget: int, *, custom_core: bool = False
):
    """Return a combine input, or None for an explicitly unsupported path."""
    if envs.SGLANG_MARLIN_MOE_CHUNK_SIZE.get():
        raise ValueError(
            "Set only the shared MoE workspace budget or SGLANG_MARLIN_MOE_CHUNK_SIZE"
        )
    backend = runner.runner_backend.value
    adapters = {"marlin": MarlinWorkspaceAdapter, "triton": TritonWorkspaceAdapter}
    if backend not in adapters:
        _warn_fallback(backend, "no adapter")
        return None
    reason = _common_unsupported_reason(runner, dispatch, lora_info, custom_core)
    if reason is not None:
        _warn_fallback(backend, reason)
        return None
    output, reason = adapters[backend].run(dispatch, quant, runner.config, budget)
    if reason is not None:
        _warn_fallback(backend, reason)
    return output
