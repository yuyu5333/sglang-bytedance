# SPDX-License-Identifier: Apache-2.0
"""Cutlass W4A8 MoE kernel."""

from typing import Optional

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
    cutlass_w4_run_moe_ep_preproess_fast,
    deepep_ll_get_cutlass_w4a8_moe_mm_data,
    deepep_permute_triton_kernel,
    deepep_post_reorder_triton_kernel,
    deepep_run_moe_deep_preprocess,
    get_cutlass_w4a8_moe_mm_data_triton_kernel,
    post_reorder_for_cutlass_moe,
    pre_reorder_for_cutlass_moe,
    silu_and_mul_masked_post_per_tensor_quant_fwd,
    silu_mul_static_tensorwise_quant_for_cutlass_moe,
)
from sglang.srt.utils import get_bool_env_var
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8


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

    # 预处理：可选 fast 路径（无排序），避免 torch.sort 带来的 O(n log n) 开销
    use_fast_preproc = get_bool_env_var("SGLANG_CUTLASS_MOE_USE_FAST_PREPROC")
    if use_fast_preproc:
        src2dst, _seg_indptr = cutlass_w4_run_moe_ep_preproess_fast(
            topk_ids, num_local_experts
        )
    else:
        src2dst = cutlass_w4_run_moe_ep_preproess(
            topk_ids,
        )

    gateup_input = torch.empty(
        (m * topk, k),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    # 量化与重排：可选“预量化 + 重排按字节拷贝”路径，避免 Python 侧 a1_scale 计算与内核中逐元素缩放
    use_prequant = get_bool_env_var("SGLANG_CUTLASS_MOE_PREQUANT")
    if use_prequant:
        # 先在 GPU 上动态求 scale 并量化到临时缓冲，再重排时跳过缩放（a1_scales=None）
        a1_scale_tensor = torch.empty(1, device=device, dtype=torch.float32)
        gateup_input_pre_reorder = torch.empty_like(a, dtype=torch.float8_e4m3fn)
        # is_static=False: 同时计算 absmax 和量化输出；输出 scale 存入 a1_scale_tensor
        per_tensor_quant_fp8(a, gateup_input_pre_reorder, a1_scale_tensor, is_static=False)
        # 后续 GEMM 需要该 scale，用于从 FP8 反量化
        a1_scale = a1_scale_tensor

        pre_reorder_for_cutlass_moe(
            gateup_input_pre_reorder,  # 已是 fp8
            gateup_input,
            src2dst,
            topk_ids,
            None,  # a1_scales -> None，内核中跳过缩放，直接拷贝
            num_local_experts,
            topk,
            m,
            k,
        )
    else:
        # 兼容原路径：计算 a1_scale（GPU 上的归约），重排时乘缩放后写入 fp8
        if a1_scale is None:
            a1_scale = (
                torch.amax(a.abs())
                .to(torch.float32)
                .div_(torch.finfo(torch.float8_e4m3fn).max)
                .view(1)
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
    if not get_bool_env_var("SGLANG_USE_TRITON_PREP_NORMAL"):
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
    else:
        # use triton kernel to get problem sizes and expert offsets
        problem_sizes1, problem_sizes2, expert_offsets = (
            get_cutlass_w4a8_moe_mm_data_triton_kernel(
                topk_ids,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                num_local_experts,
                n,
                k,
            )
        )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)
    expected_m_per_group = int(m / num_local_experts * topk)
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
        expected_m_per_group,
    )

    intermediate_q = torch.empty(
        (m * topk, n), dtype=torch.float8_e4m3fn, device=device
    )

    # A2 量化路径：
    # - 默认（静态 scale）：沿用当前 fused kernel，a2_scale=amax(c1)/fp8_max
    # - 可选（动态融合）：先用 silu_and_mul 得到 bf16 中间值，再用 per_tensor_quant_fp8 同时产出 fp8 和 scale
    use_a2_prequant = get_bool_env_var("SGLANG_CUTLASS_MOE_PREQUANT_A2")
    if use_a2_prequant:
        # CUDA Graph 友好：不读取 host 标量，完全 device-side
        # 1) 计算全量 SiLU*mul 到中间缓冲
        intermediate_bf16 = torch.empty(
            (m * topk, n), dtype=torch.bfloat16, device=device
        )
        silu_and_mul(c1, intermediate_bf16)
        # 2) 构造设备侧有效行掩码 rows < expert_offsets[-1]
        vt = expert_offsets[-1:].to(torch.int32)  # shape [1], device
        rows = torch.arange(m * topk, device=device, dtype=torch.int32)
        mask_valid = rows < vt  # shape [m*topk]
        # 将无效行清零，避免参与 absmax 污染 scale
        intermediate_bf16.masked_fill_(~mask_valid.view(-1, 1), 0)
        # 3) 动态 per-tensor 量化（全量，但无效行为 0 不影响 absmax）
        a2_scale_tensor = torch.empty(1, device=device, dtype=torch.float32)
        per_tensor_quant_fp8(
            intermediate_bf16, intermediate_q, a2_scale_tensor, is_static=False
        )
        # 基本数值健壮性检查（全量检查；无效行已清零）
        assert torch.isfinite(a2_scale_tensor).all(), "a2_scale has inf/nan in A2 prequant path"
        assert torch.isfinite(intermediate_bf16).all(), "intermediate_bf16 has inf/nan in A2 prequant path"
        assert torch.isfinite(intermediate_q).all(), "intermediate_q has inf/nan in A2 prequant path"
        a2_scale = a2_scale_tensor
    else:
        if a2_scale is None:
            a2_scale = (
                torch.amax(c1.abs())
                .to(torch.float32)
                .div_(torch.finfo(torch.float8_e4m3fn).max)
                .view(1)
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
        expected_m_per_group,
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
    expected_m_per_group = int(m / num_experts)

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
        expected_m_per_group,
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
        expected_m_per_group,
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
    expected_m_per_group = int(m / num_experts)

    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
        masked_m,
        problem_sizes1,
        problem_sizes2,
        num_experts,
        n,
        k,
    )

    gateup_input = torch.empty(a.shape, dtype=torch.float8_e4m3fn, device=device)
    if get_bool_env_var("SGLANG_DEEPEP_BF16_DISPATCH"):
        per_tensor_quant_fp8(a, gateup_input, a1_scale.float(), True)
    else:
        gateup_input = a
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
        expected_m_per_group,
    )

    intermediate_q = torch.empty(
        (num_experts, m, n), device=a.device, dtype=torch.float8_e4m3fn
    )
    silu_and_mul_masked_post_per_tensor_quant_fwd(
        c1, intermediate_q, masked_m, a2_scale
    )
    del c1, gateup_input
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
        expected_m_per_group,
    )

    return c2
