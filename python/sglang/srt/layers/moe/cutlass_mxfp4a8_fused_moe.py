# SPDX-License-Identifier: Apache-2.0
"""CUTLASS MXFP4A8 fused MoE runner.

This runner keeps the existing SM90 MXFP4A8 grouped GEMM kernel unchanged, but
moves the routing metadata preparation and input/output permutation onto the AOT
CUDA helper path used by the CUTLASS FP8 MoE runner:

  * ``prepare_moe_input`` builds expert offsets, GEMM problem sizes, and both
    permutations in one CUDA path.
  * ``shuffle_rows`` performs the gate/up input reorder.
  * ``apply_shuffle_mul_sum`` performs the final top-k reorder, router-weight
    multiply, and reduction.

The activation block-scale layout remains the graph-safe fixed-stride layout.
Do not switch this file to the compact layout unless the multi-expert numerical
issue is fixed and re-verified.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda_alike

_is_cuda_alike = is_cuda_alike()

if _is_cuda_alike:
    from sgl_kernel import (
        cutlass_mxfp4a8_moe_mm,
        get_cutlass_w4a8_moe_mm_data,
        prepare_moe_input,
    )

    try:
        from sgl_kernel import compact_cutlass_w4a8_moe_mm_data
    except ImportError:
        compact_cutlass_w4a8_moe_mm_data = None

    try:
        from sgl_kernel import get_cutlass_w4a8_moe_mm_data_with_permutation
    except ImportError:
        get_cutlass_w4a8_moe_mm_data_with_permutation = None

    from sglang.kernels.ops.quantization.fp8_kernel import (
        _run_per_token_group_quant_8bit_kernel,
        fp8_max,
        fp8_min,
    )


MXFP4_CHUNK_SIZE = 32
MXFP4_PACKED_SCALES = 4
_FP8_QUANT_EPS = 1e-10


@triton.jit
def _scatter_fixed_act_block_scale_kernel(
    scale_ptr,
    packed_ptr,
    expert_offsets_ptr,
    total_m,
    nblk,
    m_stride,
    num_experts: tl.constexpr,
    A: tl.constexpr,
    BLKN: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1) * BLKN + tl.arange(0, BLKN)
    mask = blk < nblk

    eid = tl.full((), 0, tl.int64)
    for e in tl.range(0, num_experts):
        next_off = tl.load(expert_offsets_ptr + e + 1).to(tl.int64)
        eid = tl.where(row >= next_off, e + 1, eid)
    local_row = row - tl.load(expert_offsets_ptr + eid).to(tl.int64)

    val = tl.load(scale_ptr + row * nblk + blk, mask=mask, other=0.0)
    expert_base = eid * m_stride * nblk
    dst = expert_base + (blk // A) * (m_stride * A) + local_row * A + (blk % A)
    tl.store(packed_ptr + dst, val.to(packed_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _apply_shuffle_mul_sum_fp32_factors_kernel(
    input_ptr,
    output_ptr,
    perm_ptr,
    factors_ptr,
    m,
    topk: tl.constexpr,
    row_stride: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = (token < m) & (offs < row_stride)

    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.range(0, topk):
        token_major_idx = token * topk + j
        src_row = tl.load(perm_ptr + token_major_idx).to(tl.int64)
        vals = tl.load(
            input_ptr + src_row * row_stride + offs, mask=mask, other=0.0
        ).to(tl.float32)
        factor = tl.load(factors_ptr + token_major_idx).to(tl.float32)
        acc += vals * factor * routed_scaling_factor

    tl.store(output_ptr + token * row_stride + offs, acc, mask=mask)


class CutlassMxfp4A8FusedMoeRunner:
    """Stateful MXFP4A8 MoE runner with reusable per-layer workspaces."""

    def __init__(self):
        self._workspace: Dict[Tuple[str, Tuple[int, ...], torch.dtype, int], torch.Tensor] = {}

    def _empty(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = ("tensor", name, dtype, device.index or 0)
        numel = 1
        for dim in shape:
            numel *= dim
        tensor = self._workspace.get(key)
        if tensor is None or tensor.device != device or tensor.numel() < numel:
            tensor = torch.empty((numel,), dtype=dtype, device=device)
            self._workspace[key] = tensor
        return tensor[:numel].view(shape)

    def _fixed_act_scale_strides(
        self,
        num_experts: int,
        m_stride: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = ("as_strides", (num_experts, m_stride), torch.int64, device.index or 0)
        tensor = self._workspace.get(key)
        if tensor is None or tensor.device != device:
            expert_prefix = torch.arange(
                num_experts, dtype=torch.int64, device=device
            ) * m_stride
            tensor = torch.stack(
                [
                    torch.full(
                        (num_experts,), m_stride, dtype=torch.int64, device=device
                    ),
                    expert_prefix,
                ],
                dim=1,
            ).contiguous()
            self._workspace[key] = tensor
        return tensor

    def _compact_moe_metadata(
        self,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compact_expert_offsets = self._empty(
            "compact_expert_offsets", (num_experts,), torch.int32, expert_offsets.device
        )
        compact_problem_sizes1 = self._empty(
            "compact_problem_sizes1", tuple(problem_sizes1.shape), torch.int32, expert_offsets.device
        )
        compact_problem_sizes2 = self._empty(
            "compact_problem_sizes2", tuple(problem_sizes2.shape), torch.int32, expert_offsets.device
        )
        compact_expert_ids = self._empty(
            "compact_expert_ids", (num_experts,), torch.int32, expert_offsets.device
        )
        compact_cutlass_w4a8_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            compact_expert_offsets,
            compact_problem_sizes1,
            compact_problem_sizes2,
            compact_expert_ids,
            num_experts,
        )
        return (
            compact_expert_offsets,
            compact_problem_sizes1,
            compact_problem_sizes2,
            compact_expert_ids,
        )

    def _build_fixed_act_block_scale(
        self,
        scale: torch.Tensor,
        expert_offsets: torch.Tensor,
        num_experts: int,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_m, nblk = scale.shape
        assert nblk % MXFP4_PACKED_SCALES == 0
        m_stride = max(2, (total_m + 1) & ~1)
        packed = self._empty(
            name,
            (num_experts * m_stride * nblk,),
            torch.bfloat16,
            scale.device,
        )
        strides = self._fixed_act_scale_strides(num_experts, m_stride, scale.device)
        if total_m > 0:
            _scatter_fixed_act_block_scale_kernel[
                (total_m, triton.cdiv(nblk, triton.next_power_of_2(nblk)))
            ](
                scale,
                packed,
                expert_offsets,
                total_m,
                nblk,
                m_stride,
                num_experts,
                A=MXFP4_PACKED_SCALES,
                BLKN=triton.next_power_of_2(nblk),
            )
        return packed, strides

    def _quantize_mxfp8_into(
        self,
        x: torch.Tensor,
        out_q: torch.Tensor,
        out_s: torch.Tensor,
        *,
        fuse_silu_and_mul: bool = False,
    ) -> None:
        if x.numel() == 0:
            return
        _run_per_token_group_quant_8bit_kernel(
            x,
            out_q,
            out_s,
            MXFP4_CHUNK_SIZE,
            _FP8_QUANT_EPS,
            fp8_min,
            fp8_max,
            scale_ue8m0=False,
            fuse_silu_and_mul=fuse_silu_and_mul,
            masked_m=None,
        )

    def _apply_shuffle_mul_sum_fp32_factors(
        self,
        c2: torch.Tensor,
        output: torch.Tensor,
        c_map: torch.Tensor,
        factors: torch.Tensor,
        routed_scaling_factor: float,
        topk: int,
    ) -> None:
        if output.numel() == 0:
            return
        m, row_stride = output.shape
        block = 256
        _apply_shuffle_mul_sum_fp32_factors_kernel[
            (m, triton.cdiv(row_stride, block))
        ](
            c2,
            output,
            c_map,
            factors,
            m,
            topk,
            row_stride,
            routed_scaling_factor,
            BLOCK=block,
        )

    def _silu_mul_quant_into(
        self,
        c1: torch.Tensor,
        out_q: torch.Tensor,
        out_s: torch.Tensor,
        n: int,
        swiglu_limit: Optional[float],
    ) -> None:
        if swiglu_limit is not None:
            lim = float(swiglu_limit)
            c1[:, :n].clamp_(max=lim)
            c1[:, n:].clamp_(min=-lim, max=lim)
        self._quantize_mxfp8_into(c1, out_q, out_s, fuse_silu_and_mul=True)

    def __call__(
        self,
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
        swiglu_limit: Optional[float] = None,
    ) -> torch.Tensor:
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert w1_q.dtype == torch.int8
        assert w2_q.dtype == torch.int8
        assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
        assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
        assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
        assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

        num_local_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(2) * 2
        n = w2_q.size(2) * 2
        topk = topk_ids.size(1)
        device = a.device

        if apply_router_weight_on_input:
            assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"

        # The AOT prepare/apply path does not materialize valid c_map entries for
        # the EP sentinel (-1 -> num_local_experts). Keep the legacy Triton path
        # for EP until the sentinel handling is added to prepare_moe_input.
        if get_parallel().moe_ep_size > 1:
            from sglang.srt.layers.moe.cutlass_mxfp4a8_moe import cutlass_mxfp4a8_moe

            return cutlass_mxfp4a8_moe(
                a,
                w1_q,
                w2_q,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                a_strides1,
                b_strides1,
                c_strides1,
                a_strides2,
                b_strides2,
                c_strides2,
                s_strides13,
                s_strides2,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                a1_scale,
                a2_scale,
                apply_router_weight_on_input,
                routed_scaling_factor,
                swiglu_limit,
            )

        topk_ids_i32 = topk_ids.contiguous()
        if topk_ids_i32.dtype != torch.int32:
            topk_ids_i32 = topk_ids_i32.to(torch.int32)

        a_map = self._empty("a_map", (topk_ids.numel(),), torch.int32, device)
        c_map = self._empty("c_map", (topk_ids.numel(),), torch.int32, device)
        if get_cutlass_w4a8_moe_mm_data_with_permutation is None:
            prepare_moe_input(
                topk_ids_i32,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                a_map,
                c_map,
                num_local_experts,
                n,
                k,
            )
            get_cutlass_w4a8_moe_mm_data(
                topk_ids_i32,
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
            get_cutlass_w4a8_moe_mm_data_with_permutation(
                topk_ids_i32,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                a_map,
                c_map,
                num_local_experts,
                n,
                k,
            )

        gateup_input = self._empty(
            "gateup_input_fp8", (m * topk, k), torch.float8_e4m3fn, device
        )
        a1_blk_scale = self._empty(
            "a1_blk_scale", (m * topk, k // MXFP4_CHUNK_SIZE), torch.float32, device
        )
        gateup_input_bf16 = self._empty(
            "gateup_input_bf16", (m * topk, k), a.dtype, device
        )
        torch.ops.sgl_kernel.shuffle_rows.default(a, a_map, gateup_input_bf16)
        self._quantize_mxfp8_into(gateup_input_bf16, gateup_input, a1_blk_scale)
        a1_as_packed, a1_as_strides = self._build_fixed_act_block_scale(
            a1_blk_scale, expert_offsets, num_local_experts, "a1_as_packed"
        )

        use_compact_groups = compact_cutlass_w4a8_moe_mm_data is not None and m <= 256
        if use_compact_groups:
            (
                expert_offsets_gemm,
                problem_sizes1_gemm,
                problem_sizes2_gemm,
                active_expert_ids,
            ) = self._compact_moe_metadata(
                expert_offsets, problem_sizes1, problem_sizes2, num_local_experts
            )
        else:
            expert_offsets_gemm = expert_offsets[:-1]
            problem_sizes1_gemm = problem_sizes1
            problem_sizes2_gemm = problem_sizes2
            active_expert_ids = None

        a_strides1_gemm = a_strides1
        b_strides1_gemm = b_strides1
        c_strides1_gemm = c_strides1
        s_strides13_gemm = s_strides13
        a_strides2_gemm = a_strides2
        b_strides2_gemm = b_strides2
        c_strides2_gemm = c_strides2
        s_strides2_gemm = s_strides2
        a1_as_strides_gemm = a1_as_strides

        c1 = self._empty("c1", (m * topk, n * 2), torch.bfloat16, device)
        c2 = self._empty("c2", (m * topk, k), torch.bfloat16, device)
        ones_scale = self._empty("ones_scale", (1,), torch.float32, device)
        ones_scale.fill_(1.0)

        cutlass_mxfp4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            ones_scale,
            w1_scale,
            expert_offsets_gemm,
            problem_sizes1_gemm,
            a_strides1_gemm,
            b_strides1_gemm,
            c_strides1_gemm,
            s_strides13_gemm,
            MXFP4_CHUNK_SIZE,
            topk,
            a1_as_packed,
            a1_as_strides_gemm,
            MXFP4_CHUNK_SIZE,
            active_expert_ids,
        )

        intermediate_q = self._empty(
            "intermediate_q", (m * topk, n), torch.float8_e4m3fn, device
        )
        a2_blk_scale = self._empty(
            "a2_blk_scale", (m * topk, n // MXFP4_CHUNK_SIZE), torch.float32, device
        )
        self._silu_mul_quant_into(c1, intermediate_q, a2_blk_scale, n, swiglu_limit)
        a2_as_packed, a2_as_strides = self._build_fixed_act_block_scale(
            a2_blk_scale, expert_offsets, num_local_experts, "a2_as_packed"
        )
        a2_as_strides_gemm = a2_as_strides

        cutlass_mxfp4a8_moe_mm(
            c2,
            intermediate_q,
            w2_q,
            ones_scale,
            w2_scale,
            expert_offsets_gemm,
            problem_sizes2_gemm,
            a_strides2_gemm,
            b_strides2_gemm,
            c_strides2_gemm,
            s_strides2_gemm,
            MXFP4_CHUNK_SIZE,
            topk,
            a2_as_packed,
            a2_as_strides_gemm,
            MXFP4_CHUNK_SIZE,
            active_expert_ids,
        )

        output = self._empty("output", tuple(a.shape), a.dtype, device)
        factors = topk_weights.reshape(-1).contiguous()
        self._apply_shuffle_mul_sum_fp32_factors(
            c2, output, c_map, factors, float(routed_scaling_factor), topk
        )
        return output


_DEFAULT_RUNNER = CutlassMxfp4A8FusedMoeRunner()


def cutlass_mxfp4a8_fused_moe(*args, **kwargs) -> torch.Tensor:
    return _DEFAULT_RUNNER(*args, **kwargs)
