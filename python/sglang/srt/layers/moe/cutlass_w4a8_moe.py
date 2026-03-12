# SPDX-License-Identifier: Apache-2.0
"""Cutlass W4A8 MoE kernel."""

from typing import Optional

import os
from functools import lru_cache

import torch

from sglang.srt.utils import is_cuda_alike

_is_cuda_alike = is_cuda_alike()

if _is_cuda_alike:
    from sgl_kernel import (
        cutlass_w4a8_moe_mm,
        get_cutlass_w4a8_moe_mm_data,
    )

from sgl_kernel import silu_and_mul

from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8
from sglang.srt.distributed import get_moe_expert_parallel_world_size
from sglang.srt.layers.moe.ep_moe.kernels import (
    cutlass_w4_run_moe_ep_preproess,
    deepep_ll_get_cutlass_w4a8_moe_mm_data,
    deepep_permute_triton_kernel,
    deepep_post_reorder_triton_kernel,
    deepep_run_moe_deep_preprocess,
    post_reorder_for_cutlass_moe,
    pre_reorder_for_cutlass_moe,
    silu_and_mul_masked_post_per_tensor_quant_fwd,
    silu_mul_static_tensorwise_quant_for_cutlass_moe,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


@lru_cache(maxsize=1)
def _debug_cutlass_w4a8_moe() -> bool:
    return os.getenv("SGLANG_DEBUG_CUTLASS_W4A8_MOE", "0").lower() in ("1", "true")


_debug_cutlass_w4a8_moe_counter = 0


def _normalize_per_tensor_scale(
    scale: Optional[torch.Tensor], device: torch.device
) -> Optional[torch.Tensor]:
    if scale is None:
        return None
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device)
    if scale.numel() == 1 and scale.dim() == 0:
        scale = scale.view(1)
    return scale


def cutlass_w4a8_moe(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The ids of each token->expert mapping.
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    """
    global _debug_cutlass_w4a8_moe_counter
    if _debug_cutlass_w4a8_moe():
        _debug_cutlass_w4a8_moe_counter += 1
        if _debug_cutlass_w4a8_moe_counter <= 20 or _debug_cutlass_w4a8_moe_counter % 1000 == 0:
            try:
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_available() and torch.distributed.is_initialized()
                    else -1
                )
            except Exception:
                rank = -1
            try:
                topk_min = int(topk_ids.min().item()) if topk_ids.numel() else None
                topk_max = int(topk_ids.max().item()) if topk_ids.numel() else None
                topk_neg1 = int((topk_ids == -1).sum().item()) if topk_ids.numel() else 0
            except Exception:
                topk_min = topk_max = None
                topk_neg1 = -1
            print(
                "[SGLANG_DEBUG_CUTLASS_W4A8_MOE] "
                f"rank={rank} call={_debug_cutlass_w4a8_moe_counter} "
                f"a={tuple(a.shape)} {a.dtype} {a.device} "
                f"w1_q={tuple(w1_q.shape)} {w1_q.dtype} "
                f"w2_q={tuple(w2_q.shape)} {w2_q.dtype} "
                f"w1_scale={tuple(w1_scale.shape)} {w1_scale.dtype} "
                f"w2_scale={tuple(w2_scale.shape)} {w2_scale.dtype} "
                f"topk_ids={tuple(topk_ids.shape)} min={topk_min} max={topk_max} neg1={topk_neg1} "
                f"topk_weights={tuple(topk_weights.shape)} {topk_weights.dtype} "
                f"a1_scale={None if a1_scale is None else (tuple(a1_scale.shape), a1_scale.dtype)} "
                f"a2_scale={None if a2_scale is None else (tuple(a2_scale.shape), a2_scale.dtype)} "
                f"expert_offsets={tuple(expert_offsets.shape)} {expert_offsets.dtype} "
                f"problem_sizes1={tuple(problem_sizes1.shape)} {problem_sizes1.dtype} "
                f"problem_sizes2={tuple(problem_sizes2.shape)} {problem_sizes2.dtype}"
            )
            if a_strides1.numel() >= 3 and b_strides1.numel() >= 3 and c_strides1.numel() >= 3:
                print(
                    "[SGLANG_DEBUG_CUTLASS_W4A8_MOE] "
                    f"a_strides1[0]={a_strides1[0].tolist()} "
                    f"b_strides1[0]={b_strides1[0].tolist()} "
                    f"c_strides1[0]={c_strides1[0].tolist()} "
                    f"s_strides13[0]={s_strides13[0].tolist() if s_strides13.numel() >= 3 else None}"
                )
            if a_strides2.numel() >= 3 and b_strides2.numel() >= 3 and c_strides2.numel() >= 3:
                print(
                    "[SGLANG_DEBUG_CUTLASS_W4A8_MOE] "
                    f"a_strides2[0]={a_strides2[0].tolist()} "
                    f"b_strides2[0]={b_strides2[0].tolist()} "
                    f"c_strides2[0]={c_strides2[0].tolist()} "
                    f"s_strides2[0]={s_strides2[0].tolist() if s_strides2.numel() >= 3 else None}"
                )

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_local_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids.size(1)

    if apply_router_weight_on_input:
        assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"

    device = a.device
    if get_moe_expert_parallel_world_size() > 1:
        topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

    src2dst = cutlass_w4_run_moe_ep_preproess(
        topk_ids,
    )

    gateup_input = torch.empty(
        (m * topk, k),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    pre_reorder_for_cutlass_moe(
        a,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scale,
        num_local_experts,
        topk,
        m,
        k,
    )

    # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
    # they are kept to allow for a quick switch of the permutation logic
    # from the current triton kernel implementation to the cutlass-based one if needed.
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        n,
        k,
    )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        128,
        topk,
    )

    intermediate_q = torch.empty(
        (m * topk, n), dtype=torch.float8_e4m3fn, device=device
    )
    silu_mul_static_tensorwise_quant_for_cutlass_moe(
        c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
    )

    cutlass_w4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        128,
        topk,
    )

    output = torch.empty_like(a)

    post_reorder_for_cutlass_moe(
        c2,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        num_local_experts,
        topk,
        m,
        k,
        routed_scaling_factor,
    )
    return output


def cutlass_w4a8_moe_calibrate(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"

    num_local_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2
    n = w2_q.size(2) * 2
    topk = topk_ids.size(1)

    device = a.device
    if get_moe_expert_parallel_world_size() > 1:
        topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

    src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)

    a1_max_abs = a.abs().amax().to(torch.float32)
    a1_scale = (a1_max_abs / FP8_MAX).clamp(min=1e-12)
    a1_scale = _normalize_per_tensor_scale(a1_scale, device)
    assert a1_scale is not None and a1_scale.dim() == 1 and a1_scale.numel() == 1

    gateup_input = torch.empty((m * topk, k), device=device, dtype=torch.float8_e4m3fn)
    pre_reorder_for_cutlass_moe(
        a,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scale,
        num_local_experts,
        topk,
        m,
        k,
    )

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        n,
        k,
    )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        128,
        topk,
    )

    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.bfloat16)
    silu_and_mul(c1, intermediate)
    a2_max_abs = intermediate.abs().amax().to(torch.float32)
    a2_scale = (a2_max_abs / FP8_MAX).clamp(min=1e-12)
    a2_scale = _normalize_per_tensor_scale(a2_scale, device)
    assert a2_scale is not None and a2_scale.dim() == 1 and a2_scale.numel() == 1

    intermediate_q = torch.empty((m * topk, n), dtype=torch.float8_e4m3fn, device=device)
    per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)

    cutlass_w4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        128,
        topk,
    )

    output = torch.empty_like(a)
    post_reorder_for_cutlass_moe(
        c2,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        num_local_experts,
        topk,
        m,
        k,
        routed_scaling_factor,
    )
    return output, a1_max_abs, a2_max_abs


def cutlass_w4a8_moe_deepep_normal(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    """
    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2
    n = w2_q.size(2) * 2
    topk = topk_ids_.size(1)
    device = a.device

    reorder_topk_ids, src2dst, _ = deepep_run_moe_deep_preprocess(
        topk_ids_, num_experts
    )
    num_total_tokens = reorder_topk_ids.numel()
    gateup_input_pre_reorder = torch.empty(
        (int(num_total_tokens), a.shape[1]),
        device=device,
        dtype=a.dtype,
    )
    deepep_permute_triton_kernel[(a.shape[0],)](
        a,
        gateup_input_pre_reorder,
        src2dst,
        topk_ids_.to(torch.int64),
        None,
        topk,
        a.shape[1],
        BLOCK_SIZE=512,
    )
    gateup_input = torch.empty(
        gateup_input_pre_reorder.shape, dtype=torch.float8_e4m3fn, device=device
    )
    per_tensor_quant_fp8(gateup_input_pre_reorder, gateup_input, a1_scale.float(), True)
    del gateup_input_pre_reorder
    local_topk_ids = topk_ids_
    local_topk_ids = (
        torch.where(local_topk_ids == -1, num_experts, topk_ids_).to(torch.int32)
    ).contiguous()

    a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        local_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )
    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.zeros((m * topk, k), device=device, dtype=torch.bfloat16)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        128,
        topk,
    )
    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.bfloat16)
    silu_and_mul(c1, intermediate)

    intermediate_q = torch.empty(
        intermediate.shape, dtype=torch.float8_e4m3fn, device=device
    )
    per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)

    cutlass_w4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        128,
        topk,
    )
    num_tokens = src2dst.shape[0] // topk
    output = torch.empty(
        (num_tokens, c2.shape[1]),
        device=c2.device,
        dtype=torch.bfloat16,
    )
    deepep_post_reorder_triton_kernel[(num_tokens,)](
        c2,
        output,
        src2dst,
        topk_ids_,
        topk_weights,
        topk,
        c2.shape[1],
        BLOCK_SIZE=512,
    )

    return output


def cutlass_w4a8_moe_deepep_ll(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids_: torch.Tensor,
    masked_m: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    """
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[2] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_experts = w1_q.size(0)
    m = a.size(1)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)

    device = a.device

    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
        masked_m,
        problem_sizes1,
        problem_sizes2,
        num_experts,
        n,
        k,
    )

    gateup_input = torch.empty(a.shape, dtype=torch.float8_e4m3fn, device=device)
    per_tensor_quant_fp8(a, gateup_input, a1_scale.float(), True)
    c1 = torch.empty((num_experts, m, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((num_experts, m, k), device=device, dtype=torch.bfloat16)

    cutlass_w4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        128,
        topk,
    )

    intermediate_q = torch.empty(
        (num_experts, m, n), device=a.device, dtype=torch.float8_e4m3fn
    )
    silu_and_mul_masked_post_per_tensor_quant_fwd(
        c1, intermediate_q, masked_m, a2_scale
    )
    cutlass_w4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        128,
        topk,
    )

    return c2
