# SPDX-License-Identifier: Apache-2.0
"""Cutlass W4A8 MoE kernel."""

import os
from typing import Optional

import torch

from sglang.srt.utils import is_cuda, is_cuda_alike

# 进程内只在第一次进入 cutlass_w4a8_moe 时落盘一次，避免每 step 都写盘。
_CUTLASS_W4A8_INNER_DUMPED = False

_is_cuda = is_cuda()
_is_cuda_alike = is_cuda_alike()

if _is_cuda_alike:
    from sgl_kernel import (
        cutlass_w4a8_moe_mm,
        get_cutlass_w4a8_moe_mm_data,
    )

if _is_cuda:
    from sglang.jit_kernel.activation import silu_and_mul
else:
    from sgl_kernel import silu_and_mul

from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8, per_tensor_absmax_fp8

from sglang.srt.distributed import get_moe_expert_parallel_world_size
from sglang.srt.layers.moe.ep_moe.kernels import (
    cutlass_w4_run_moe_ep_preproess,
    deepep_ll_get_cutlass_w4a8_moe_mm_data,
    deepep_permute_triton_kernel,
    deepep_post_reorder_triton_kernel,
    deepep_run_moe_deep_preprocess,
    fp8_per_token_to_per_tensor_quant_triton,
    post_reorder_for_cutlass_moe,
    pre_reorder_for_cutlass_moe,
    silu_and_mul_masked_post_per_tensor_quant_fwd,
    silu_mul_dynamic_tensorwise_quant_for_cutlass_moe,
    silu_mul_static_tensorwise_quant_for_cutlass_moe,
)


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

    # ------------------------------------------------------------------
    # DEBUG DUMP: 每个算子前后的输入/输出快照。
    # 通过 SGLANG_W4A8_INNER_DUMP_DIR 启用；只在进程内 dump 一次。
    # 命名规则: cutlass_w4a8_inner_rank{R}.pt，结构：
    #   {
    #     "step_00_inputs": {... 整个 cutlass_w4a8_moe 的入参 ...},
    #     "step_01_ep_preprocess":      {"in": {...},  "out": {...}},
    #     "step_02_a1_absmax":          {"in": {...},  "out": {...}},
    #     "step_03_pre_reorder":        {"in": {...},  "out": {...}},
    #     "step_04_get_mm_data":        {"in": {...},  "out": {...}},
    #     "step_05_gemm1":              {"in": {...},  "out": {...}},
    #     "step_06_silu_mul_quant":     {"in": {...},  "out": {...}},
    #     "step_07_gemm2":              {"in": {...},  "out": {...}},
    #     "step_08_post_reorder":       {"in": {...},  "out": {...}},
    #   }
    # ------------------------------------------------------------------
    global _CUTLASS_W4A8_INNER_DUMPED
    _inner_dump_dir = os.environ.get("SGLANG_W4A8_INNER_DUMP_DIR")
    _do_inner_dump = bool(_inner_dump_dir) and not _CUTLASS_W4A8_INNER_DUMPED
    _inner_dump: dict = {}

    def _snap(x):
        """Detach a tensor (or pass through scalars/None) onto CPU for safe storage."""
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").clone()
        return x

    def _record(label: str, payload: dict):
        if _do_inner_dump:
            _inner_dump[label] = {k: _snap(v) for k, v in payload.items()}

    if _do_inner_dump:
        _record(
            "step_00_inputs",
            {
                "a": a,
                "w1_q": w1_q,
                "w2_q": w2_q,
                "w1_scale": w1_scale,
                "w2_scale": w2_scale,
                "topk_weights": topk_weights,
                "topk_ids": topk_ids,
                "a_strides1": a_strides1,
                "b_strides1": b_strides1,
                "c_strides1": c_strides1,
                "a_strides2": a_strides2,
                "b_strides2": b_strides2,
                "c_strides2": c_strides2,
                "s_strides13": s_strides13,
                "s_strides2": s_strides2,
                "expert_offsets": expert_offsets,
                "problem_sizes1": problem_sizes1,
                "problem_sizes2": problem_sizes2,
                "a1_scale": a1_scale,
                "a2_scale": a2_scale,
                "apply_router_weight_on_input": apply_router_weight_on_input,
                "routed_scaling_factor": routed_scaling_factor,
                "num_local_experts": num_local_experts,
                "m": m,
                "k": k,
                "n": n,
                "topk": topk,
            },
        )

    # -------- step 1: cutlass_w4_run_moe_ep_preproess --------
    _step1_in = {"topk_ids": topk_ids}
    src2dst = cutlass_w4_run_moe_ep_preproess(
        topk_ids,
    )
    _record("step_01_ep_preprocess", {"in_topk_ids": _step1_in["topk_ids"], "out_src2dst": src2dst})

    gateup_input = torch.empty(
        (m * topk, k),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    # -------- step 2: per_tensor_absmax_fp8 (a1_scale fallback) --------
    # TODO: fuse per_tensor_absmax_fp8 and pre_reorder_for_cutlass_moe
    if a1_scale is None:
        _step2_in_a = a
        a1_scale = torch.zeros(1, dtype=torch.float32, device=device)
        per_tensor_absmax_fp8(a, a1_scale)
        _record(
            "step_02_a1_absmax",
            {"in_a": _step2_in_a, "out_a1_scale": a1_scale},
        )

    # -------- step 3: pre_reorder_for_cutlass_moe --------
    _step3_in = {
        "a": a,
        "src2dst": src2dst,
        "topk_ids": topk_ids,
        "a1_scale": a1_scale,
        "num_local_experts": num_local_experts,
        "topk": topk,
        "m": m,
        "k": k,
    }
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
    _record(
        "step_03_pre_reorder",
        {**{f"in_{k_}": v for k_, v in _step3_in.items()}, "out_gateup_input": gateup_input},
    )

    # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
    # they are kept to allow for a quick switch of the permutation logic
    # from the current triton kernel implementation to the cutlass-based one if needed.
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    # -------- step 4: get_cutlass_w4a8_moe_mm_data --------
    _step4_in = {
        "topk_ids": topk_ids,
        "num_local_experts": num_local_experts,
        "n": n,
        "k": k,
    }
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
    _record(
        "step_04_get_mm_data",
        {
            **{f"in_{k_}": v for k_, v in _step4_in.items()},
            "out_expert_offsets": expert_offsets,
            "out_problem_sizes1": problem_sizes1,
            "out_problem_sizes2": problem_sizes2,
            "out_a_map": a_map,
            "out_c_map": c_map,
        },
    )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)

    # -------- step 5: cutlass_w4a8_moe_mm (gemm 1) --------
    _step5_in = {
        "gateup_input": gateup_input,
        "w1_q": w1_q,
        "a1_scale_float": a1_scale.float(),
        "w1_scale": w1_scale,
        "expert_offsets_prefix": expert_offsets[:-1],
        "problem_sizes1": problem_sizes1,
        "a_strides1": a_strides1,
        "b_strides1": b_strides1,
        "c_strides1": c_strides1,
        "s_strides13": s_strides13,
        "topk": topk,
    }
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
    _record(
        "step_05_gemm1",
        {**{f"in_{k_}": v for k_, v in _step5_in.items()}, "out_c1": c1},
    )

    intermediate_q = torch.empty(
        (m * topk, n), dtype=torch.float8_e4m3fn, device=device
    )

    # -------- step 6: silu_mul + per-tensor quant (dynamic 或 static) --------
    if a2_scale is None:
        a2_scale = torch.zeros(1, dtype=torch.float32, device=device)
        _step6_in = {
            "c1": c1,
            "expert_offsets_last": expert_offsets[-1:],
            "m_topk": m * topk,
            "n": n,
            "branch": "dynamic",
        }
        silu_mul_dynamic_tensorwise_quant_for_cutlass_moe(
            c1, intermediate_q, a2_scale, expert_offsets[-1:], m * topk, n
        )
    else:
        _step6_in = {
            "c1": c1,
            "a2_scale_float": a2_scale.float(),
            "expert_offsets_last": expert_offsets[-1:],
            "m_topk": m * topk,
            "n": n,
            "branch": "static",
        }
        silu_mul_static_tensorwise_quant_for_cutlass_moe(
            c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
        )
    _record(
        "step_06_silu_mul_quant",
        {
            **{f"in_{k_}": v for k_, v in _step6_in.items()},
            "out_intermediate_q": intermediate_q,
            "out_a2_scale": a2_scale,
        },
    )

    # -------- step 7: cutlass_w4a8_moe_mm (gemm 2) --------
    _step7_in = {
        "intermediate_q": intermediate_q,
        "w2_q": w2_q,
        "a2_scale_float": a2_scale.float(),
        "w2_scale": w2_scale,
        "expert_offsets_prefix": expert_offsets[:-1],
        "problem_sizes2": problem_sizes2,
        "a_strides2": a_strides2,
        "b_strides2": b_strides2,
        "c_strides2": c_strides2,
        "s_strides2": s_strides2,
        "topk": topk,
    }
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
    _record(
        "step_07_gemm2",
        {**{f"in_{k_}": v for k_, v in _step7_in.items()}, "out_c2": c2},
    )

    output = torch.empty_like(a)

    # -------- step 8: post_reorder_for_cutlass_moe --------
    _step8_in = {
        "c2": c2,
        "src2dst": src2dst,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "num_local_experts": num_local_experts,
        "topk": topk,
        "m": m,
        "k": k,
        "routed_scaling_factor": routed_scaling_factor,
    }
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
    _record(
        "step_08_post_reorder",
        {**{f"in_{k_}": v for k_, v in _step8_in.items()}, "out_output": output},
    )

    if _do_inner_dump:
        try:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
        except Exception:
            rank = 0
        os.makedirs(_inner_dump_dir, exist_ok=True)
        _inner_dump_path = os.path.join(
            _inner_dump_dir, f"cutlass_w4a8_inner_rank{rank}.pt"
        )
        torch.save(_inner_dump, _inner_dump_path)
        _CUTLASS_W4A8_INNER_DUMPED = True

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
    a_states: torch.Tensor,
    a_scales: torch.Tensor,
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
    assert a_states.shape[2] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_experts = w1_q.size(0)
    m = a_states.size(1)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)

    device = a_states.device

    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
        masked_m,
        problem_sizes1,
        problem_sizes2,
        num_experts,
        n,
        k,
    )

    gateup_input = torch.empty(a_states.shape, dtype=torch.float8_e4m3fn, device=device)
    fp8_per_token_to_per_tensor_quant_triton(
        x=a_states,
        x_scale=a_scales,
        masked_m=masked_m,
        output_scale=a1_scale,
        output=gateup_input,
    )
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
        (num_experts, m, n), device=a_states.device, dtype=torch.float8_e4m3fn
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
