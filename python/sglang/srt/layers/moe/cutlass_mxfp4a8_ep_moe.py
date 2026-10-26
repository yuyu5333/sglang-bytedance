# SPDX-License-Identifier: Apache-2.0
"""DeepEP FP8 group128 dispatch to fused-layout CUTLASS MXFP4 experts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_kernel import (
    cutlass_mxfp4a8_gemm1_blockscale,
    get_cutlass_w4a8_moe_mm_data,
)
from sglang.srt.layers.moe.cutlass_mxfp4a8_fused_moe import (
    CutlassMxfp4A8FusedMoeRunner,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)


@triton.jit(do_not_specialize=["scale_row_stride", "scale_group_stride"])
def _pack_normal(
    x,
    scales,
    order,
    sorted_ids,
    offsets,
    residual,
    packed_x,
    packed_s,
    factors,
    inverse,
    scale_row_stride,
    scale_group_stride,
    H: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    GP_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    route = tl.load(order + row)
    if tile == 0:
        tl.store(inverse + route, row)
    if row < tl.load(offsets + E):
        token = route // TOPK
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + token * H + cols, mask=cols < H, other=0.0)
        tl.store(packed_x + row * H + cols, values, mask=cols < H)
        if tile == 0:
            groups = tl.arange(0, GP_BLOCK)
            values_s = tl.load(
                scales + token * scale_row_stride + groups * scale_group_stride,
                mask=groups < G,
                other=0,
            )
            tl.store(packed_s + row * GP + groups, values_s, mask=groups < GP)
            expert = tl.load(sorted_ids + row)
            tl.store(factors + row, tl.load(residual + expert))


_PACK_NORMAL_PRELOADED = set()


def preload_cutlass_mxfp4a8_normal_pack(
    *, device: torch.device, hidden_size: int, num_experts: int, topk: int
) -> None:
    """Load the normal pack cubin before KV pools consume device memory."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (device, hidden_size, num_experts, topk)
    if key in _PACK_NORMAL_PRELOADED:
        return

    groups = hidden_size // 128
    padded_groups = triton.cdiv(groups, 4) * 4
    with torch.cuda.device(device):
        x = torch.empty((1, hidden_size), dtype=torch.float8_e4m3fn, device=device)
        scales = torch.empty((1, groups), dtype=torch.float32, device=device)
        packed_s = torch.empty((1, padded_groups), dtype=torch.float32, device=device)
        order = torch.zeros(1, dtype=torch.int64, device=device)
        sorted_ids = torch.zeros(1, dtype=torch.int32, device=device)
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        residual = torch.empty(1, dtype=torch.float32, device=device)
        factors = torch.empty(1, dtype=torch.float32, device=device)
        inverse = torch.empty(1, dtype=torch.int32, device=device)
        # Zero valid routes skip all payload reads. Launch (rather than compile
        # only) to initialize CUDA module handles using the serving signature.
        # Batch size belongs only in the launch grid, never in the cache key.
        _pack_normal[(1, triton.cdiv(hidden_size, 512))](
            x,
            scales,
            order,
            sorted_ids,
            offsets,
            residual,
            x,
            packed_s,
            factors,
            inverse,
            *scales.stride(),
            hidden_size,
            groups,
            padded_groups,
            triton.next_power_of_2(padded_groups),
            topk,
            num_experts,
            512,
        )
        torch.cuda.synchronize(device)
    _PACK_NORMAL_PRELOADED.add(key)


@triton.jit
def _ll_metadata(
    counts,
    offsets,
    p1,
    p2,
    E: tl.constexpr,
    CAP: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(counts + expert, mask=expert < E, other=0)
    tl.store(offsets + expert, expert * CAP, mask=expert <= E)
    tl.store(p1 + expert * 3, 2 * INTERMEDIATE, mask=expert < E)
    tl.store(p1 + expert * 3 + 1, count, mask=expert < E)
    tl.store(p1 + expert * 3 + 2, H, mask=expert < E)
    tl.store(p2 + expert * 3, H, mask=expert < E)
    tl.store(p2 + expert * 3 + 1, count, mask=expert < E)
    tl.store(p2 + expert * 3 + 2, INTERMEDIATE, mask=expert < E)


@triton.jit
def _pack_ll(
    x,
    scales,
    packed_x,
    packed_s,
    factors,
    residual,
    counts,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    GP_BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    WORKERS: tl.constexpr,
    IS_FP8: tl.constexpr,
    COPY_X: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    worker = tl.program_id(1)
    groups = tl.arange(0, GP_BLOCK)
    factor = tl.load(residual + expert)
    for row in tl.range(worker, tl.load(counts + expert), WORKERS):
        dst = expert.to(tl.int64) * CAP + row
        if IS_FP8:
            s = tl.load(
                scales + expert * S0 + row * S1 + groups * S2, mask=groups < G, other=0
            )
            tl.store(packed_s + dst * GP + groups, s, mask=groups < GP)
            if COPY_X:
                for base in tl.range(0, H, BLOCK):
                    cols = base + tl.arange(0, BLOCK)
                    v = tl.load(x + dst * H + cols, mask=cols < H, other=0.0)
                    tl.store(packed_x + dst * H + cols, v, mask=cols < H)
        else:
            # An explicit BF16 dispatch override is quantized only on valid rows.
            for group in tl.range(0, G):
                cols = group * 128 + tl.arange(0, 128)
                v = tl.load(x + dst * H + cols).to(tl.float32)
                s = tl.maximum(tl.max(tl.abs(v)), 1e-10) / 448.0
                tl.store(packed_x + dst * H + cols, tl.clamp(v / s, -448, 448))
                tl.store(packed_s + dst * GP + group, s)
            tl.store(
                packed_s + dst * GP + groups, 0, mask=(groups >= G) & (groups < GP)
            )
        tl.store(factors + dst, factor)


@triton.jit
def _swiglu_quant(
    c1,
    q,
    scales,
    residual,
    counts,
    offsets,
    INTERMEDIATE: tl.constexpr,
    CAP: tl.constexpr,
    E: tl.constexpr,
    IS_LL: tl.constexpr,
    WORKERS: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    worker = tl.program_id(1)
    start = expert * CAP if IS_LL else tl.load(offsets + expert)
    count = tl.load(counts + expert) if IS_LL else tl.load(offsets + expert + 1) - start
    cols = tl.arange(0, BLOCK)
    factor = tl.load(residual + expert)
    for local_row in tl.range(worker, count, WORKERS):
        row = (start + local_row).to(tl.int64)
        gate = tl.load(
            c1 + row * (2 * INTERMEDIATE) + cols,
            mask=cols < INTERMEDIATE,
            other=0,
        ).to(tl.float32)
        up = tl.load(
            c1 + row * (2 * INTERMEDIATE) + INTERMEDIATE + cols,
            mask=cols < INTERMEDIATE,
            other=0,
        ).to(tl.float32)
        if HAS_LIMIT:
            gate = tl.minimum(gate, LIMIT).to(tl.bfloat16).to(tl.float32)
            up = tl.clamp(up, -LIMIT, LIMIT).to(tl.bfloat16).to(tl.float32)
        value = gate * tl.sigmoid(gate) * up
        # Match EP1's BF16 activation before its per-row FP8 quantization.
        value = value.to(tl.bfloat16).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(value)), 1e-10) / 448.0
        tl.store(
            q + row * INTERMEDIATE + cols,
            tl.clamp(value / s, -448, 448),
            mask=cols < INTERMEDIATE,
        )
        tl.store(scales + row, s * factor)


@triton.jit
def _normal_reduce(
    c2,
    inverse,
    ids,
    weights,
    output,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.range(0, TOPK):
        route = token * TOPK + j
        expert = tl.load(ids + route)
        valid = (expert >= 0) & (expert < E)
        row = tl.load(inverse + route).to(tl.int64)
        value = tl.load(c2 + row * H + cols, mask=valid & (cols < H), other=0).to(
            tl.float32
        )
        weight = tl.load(weights + route)
        # Invalid DeepEP weights can be sentinels; do not multiply 0 by NaN.
        acc += tl.where(valid, value * weight, 0.0)
    tl.store(output + token * H + cols, acc, mask=cols < H)


class CutlassMxfp4A8EPMoeRunner(CutlassMxfp4A8FusedMoeRunner):
    """Workspaces are separate from EP1 and stable across graph replays."""

    def __call__(self, layer, dispatch_output, method):
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputFormat

        is_ll = dispatch_output.format == DispatchOutputFormat.DEEPEP_LL
        x = dispatch_output.hidden_states
        s = dispatch_output.hidden_states_scale
        e, h = layer.w13_weight_fused.shape[0], x.shape[-1]
        intermediate = layer.w2_weight_fused.shape[-1] * 2
        groups = h // 128
        padded_groups = triton.cdiv(groups, 4) * 4
        device = x.device
        topk = dispatch_output.topk_ids.shape[-1]
        if is_ll:
            assert x.ndim == 3 and x.shape[0] == e and x.is_contiguous()
            capacity = x.shape[1]
            rows = e * capacity
            counts = dispatch_output.masked_m
            workers = max(1, min(32, capacity))
            _ll_metadata[(triton.cdiv(e + 1, 128),)](
                counts,
                method.expert_offsets,
                method.problem_sizes1,
                method.problem_sizes2,
                e,
                capacity,
                h,
                intermediate,
                128,
            )
            if x.dtype == torch.float8_e4m3fn:
                assert s is not None and s.dtype == torch.float32
                assert tuple(s.shape) == (e, capacity, groups)
                packed_x = x.view(rows, h)
            else:
                assert x.dtype == torch.bfloat16
                packed_x = self._empty(
                    "packed_x", (rows, h), torch.float8_e4m3fn, device
                )
            packed_s = self._empty(
                "packed_s", (rows, padded_groups), torch.float32, device
            )
            factors = self._empty("factors1", (rows,), torch.float32, device)
            _pack_ll[(e, workers)](
                x,
                s if s is not None else x,
                packed_x,
                packed_s,
                factors,
                layer.w13_weight_residual_fused,
                counts,
                *(s.stride() if s is not None else (0, 0, 0)),
                h,
                groups,
                padded_groups,
                triton.next_power_of_2(padded_groups),
                capacity,
                workers,
                x.dtype == torch.float8_e4m3fn,
                False,
                512,
            )
        else:
            assert x.ndim == 2 and x.is_contiguous()
            if x.dtype != torch.float8_e4m3fn:
                from sglang.kernels.ops.quantization.fp8_kernel import (
                    sglang_per_token_group_quant_fp8,
                )

                x, s = sglang_per_token_group_quant_fp8(
                    x,
                    128,
                    column_major_scales=False,
                    scale_ue8m0=False,
                )
            assert s is not None and s.dtype == torch.float32
            assert tuple(s.shape) == (x.shape[0], groups)
            rows = dispatch_output.topk_ids.numel()
            capacity = 0
            counts = method.expert_offsets  # unused in the normal specialization
            workers = 32
            ids = dispatch_output.topk_ids.to(torch.int32).contiguous()
            ids = torch.where((ids >= 0) & (ids < e), ids, e)
            # The existing metadata kernel safely ignores the E sentinel.
            get_cutlass_w4a8_moe_mm_data(
                ids,
                method.expert_offsets,
                method.problem_sizes1,
                method.problem_sizes2,
                method.expert_offsets,
                method.expert_offsets,
                e,
                intermediate,
                h,
            )
            sorted_ids, order = torch.sort(ids.reshape(-1), stable=True)
            inverse = self._empty("inverse", (rows,), torch.int32, device)
            packed_x = self._empty("packed_x", (rows, h), torch.float8_e4m3fn, device)
            packed_s = self._empty(
                "packed_s", (rows, padded_groups), torch.float32, device
            )
            factors = self._empty("factors1", (rows,), torch.float32, device)
            if rows:
                _pack_normal[(rows, triton.cdiv(h, 512))](
                    x,
                    s,
                    order,
                    sorted_ids,
                    method.expert_offsets,
                    layer.w13_weight_residual_fused,
                    packed_x,
                    packed_s,
                    factors,
                    inverse,
                    *s.stride(),
                    h,
                    groups,
                    padded_groups,
                    triton.next_power_of_2(padded_groups),
                    topk,
                    e,
                    512,
                )

        c1 = self._empty("ep_c1", (rows, 2 * intermediate), torch.bfloat16, device)
        c2 = self._empty("ep_c2", (rows, h), torch.bfloat16, device)
        intermediate_q = self._empty(
            "ep_intermediate_q", (rows, intermediate), torch.float8_e4m3fn, device
        )
        factors2 = self._empty("factors2", (rows,), torch.float32, device)
        if rows:
            cutlass_mxfp4a8_gemm1_blockscale(
                c1,
                packed_x,
                layer.w13_weight_fused,
                factors,
                layer.w13_weight_scale_fused,
                method.expert_offsets[:-1],
                method.problem_sizes1,
                method.a_strides1,
                method.b_strides1,
                method.c_strides1,
                method.s_strides13,
                packed_s,
                config=100 if rows <= 64 else 101,
            )
            limit = method.moe_runner_config.swiglu_limit
            _swiglu_quant[(e, workers)](
                c1,
                intermediate_q,
                factors2,
                layer.w2_weight_residual_fused,
                counts,
                method.expert_offsets,
                intermediate,
                capacity,
                e,
                is_ll,
                workers,
                limit is not None,
                float(limit or 0),
                triton.next_power_of_2(intermediate),
                num_warps=4,
            )
            # K128 handles real DSv4 I2304 and independently padded TP shards.
            torch.ops.sgl_kernel.cutlass_mxfp4a8_fused_moe_mm.default(
                c2,
                intermediate_q,
                layer.w2_weight_fused,
                factors2,
                layer.w2_weight_scale_fused,
                method.expert_offsets[:-1],
                method.problem_sizes2,
                method.a_strides2,
                method.b_strides2,
                method.c_strides2,
                method.s_strides2,
                topk,
                100 if rows <= 64 else 101,
                None,
            )
        if is_ll:
            return DeepEPLLCombineInput(
                c2.view(e, capacity, h),
                dispatch_output.topk_ids,
                dispatch_output.topk_weights,
            )
        output = self._empty("ep_output", (x.shape[0], h), torch.bfloat16, device)
        if x.shape[0]:
            _normal_reduce[(x.shape[0], triton.cdiv(h, 512))](
                c2,
                inverse,
                ids,
                dispatch_output.topk_weights.contiguous(),
                output,
                h,
                topk,
                e,
                512,
            )
        return DeepEPNormalCombineInput(
            output,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )


_DEFAULT_RUNNER = CutlassMxfp4A8EPMoeRunner()


def cutlass_mxfp4a8_ep_moe(layer, dispatch_output, method):
    return _DEFAULT_RUNNER(layer, dispatch_output, method)
