from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CUDA_FILE = "gemm/w4a8_fp8_scaled_mm.cuh"


def _fake_w4a8_fp8_scaled_mm(
    q_input: torch.Tensor,
    weight_packed: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del x_scale, weight_scale, group_size, bias
    return q_input.new_empty((q_input.shape[0], weight_packed.shape[0]), dtype=out_dtype)


def has_w4a8_fp8_scaled_mm_jit_kernel() -> bool:
    return (Path(KERNEL_PATH) / "csrc" / _CUDA_FILE).exists()


@cache_once
def _jit_w4a8_fp8_scaled_mm_module(out_dtype: torch.dtype) -> Module:
    args = make_cpp_args(out_dtype)
    return load_jit(
        "w4a8_fp8_scaled_mm",
        *args,
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[("w4a8_fp8_scaled_mm", f"w4a8_fp8_scaled_mm<{args}>")],
    )


@register_custom_op(
    op_name="w4a8_fp8_scaled_mm",
    mutates_args=[],
    fake_impl=_fake_w4a8_fp8_scaled_mm,
)
def w4a8_fp8_scaled_mm(
    q_input: torch.Tensor,
    weight_packed: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """JIT entry for dense W4A8-FP8 matmul.

    Expected kernel contract:
    - q_input: [M, K], fp8
    - weight_packed: [N, K // 2], int8-packed int4
    - x_scale: [M, 1], fp32
    - weight_scale: [N, K // group_size], fp32/bf16
    - output: [M, N], out_dtype
    """
    if not has_w4a8_fp8_scaled_mm_jit_kernel():
        raise NotImplementedError(
            "JIT kernel source for W4A8-FP8 is not present. "
            f"Expected file: {Path(KERNEL_PATH) / 'csrc' / _CUDA_FILE}"
        )

    output = q_input.new_empty((q_input.shape[0], weight_packed.shape[0]), dtype=out_dtype)
    module = _jit_w4a8_fp8_scaled_mm_module(out_dtype)
    module.w4a8_fp8_scaled_mm(
        q_input,
        weight_packed,
        x_scale,
        weight_scale,
        output,
        group_size,
        bias,
    )
    return output
