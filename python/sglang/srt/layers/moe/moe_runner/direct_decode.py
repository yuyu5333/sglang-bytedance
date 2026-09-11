"""Opt-in single-token dispatcher for the unsorted MoE experiment."""

from sglang.srt.layers.moe.moe_runner.workspace import _common_unsupported_reason


def try_direct_decode(runner, dispatch, quant, lora_info, custom_core):
    # Reuse the existing local-inference boundary: no hooks, overlap, collectives,
    # pre-quantized dispatch, compile or batch-invariant execution.
    if _common_unsupported_reason(runner, dispatch, lora_info, custom_core):
        return None
    if runner.runner_backend.value != "triton":
        return None
    from sglang.kernels.ops.moe.direct_decode import (
        direct_decode,
        direct_decode_supported,
    )
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.runtime_context import get_exec

    config = runner.config
    x = dispatch.hidden_states
    if (
        x.shape[0] != 1
        or x.shape[1] > 4096
        or config.activation != "silu"
        or get_tp_group().world_size != 1
        or get_exec().deterministic.enable_deterministic_inference
        or not isinstance(quant, TritonMoeQuantInfo)
    ):
        return None
    if any(
        (
            quant.use_mxfp8,
            quant.use_fp8_w8a8,
            quant.use_int8_w8a8,
            quant.use_int8_w8a16,
            quant.use_int4_w4a16,
            quant.per_channel_quant,
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
        return None
    topk = dispatch.topk_output
    args = (
        x,
        quant.w13_weight,
        quant.w2_weight,
        topk.topk_ids,
        topk.topk_weights,
        quant.b13,
        quant.b2,
    )
    if not direct_decode_supported(*args) or quant.w2_weight.shape[2] > 1024:
        return None
    return StandardCombineInput(
        direct_decode(
            *args[:5],
            b1=quant.b13,
            b2=quant.b2,
            scale=1.0
            if config.routed_scaling_factor is None
            else config.routed_scaling_factor,
            inplace=config.inplace,
        )
    )
