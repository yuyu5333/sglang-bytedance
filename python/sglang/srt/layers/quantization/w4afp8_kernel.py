from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.utils import get_device_capability, is_cuda

logger = logging.getLogger(__name__)

__all__ = ["is_w4a8_fp8_linear_supported", "w4a8_fp8_scaled_mm"]

_is_cuda = is_cuda()
_w4a8_fp8_scaled_mm_op = None

if _is_cuda:
    try:
        from sgl_kernel import w4a8_fp8_scaled_mm as _w4a8_fp8_scaled_mm_op

        @torch.library.register_fake("sgl_kernel::w4a8_fp8_scaled_mm")
        def _w4a8_fp8_scaled_mm_abstract(
            q_input,
            weight_packed,
            x_scale,
            weight_scale,
            group_size,
            out_dtype,
            bias=None,
        ):
            del x_scale, weight_scale, group_size, bias
            m = q_input.shape[0]
            n = weight_packed.shape[0]
            return q_input.new_empty((m, n), dtype=out_dtype)

    except ImportError:
        _w4a8_fp8_scaled_mm_op = None


@lru_cache(maxsize=1)
def is_w4a8_fp8_linear_supported() -> bool:
    """Return whether the current runtime can execute the dense W4A8-FP8 op."""
    if not _is_cuda:
        return False

    major, _minor = get_device_capability()
    if major < 9:
        return False

    return _w4a8_fp8_scaled_mm_op is not None


def _validate_w4a8_fp8_inputs(
    q_input: torch.Tensor,
    weight_packed: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> None:
    if q_input.ndim != 2:
        raise ValueError(f"`q_input` must be 2D, got shape={tuple(q_input.shape)}")
    if weight_packed.ndim != 2:
        raise ValueError(
            f"`weight_packed` must be 2D, got shape={tuple(weight_packed.shape)}"
        )
    if x_scale.ndim not in (1, 2):
        raise ValueError(f"`x_scale` must be 1D or 2D, got shape={tuple(x_scale.shape)}")
    if weight_scale.ndim != 2:
        raise ValueError(
            f"`weight_scale` must be 2D, got shape={tuple(weight_scale.shape)}"
        )
    if group_size <= 0:
        raise ValueError(f"`group_size` must be positive, got {group_size}")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported `out_dtype`: {out_dtype}")

    m, k = q_input.shape
    n, packed_k = weight_packed.shape
    del n

    if packed_k * 2 != k:
        raise ValueError(
            "Packed weight shape is incompatible with q_input: "
            f"q_input.shape={tuple(q_input.shape)}, "
            f"weight_packed.shape={tuple(weight_packed.shape)}"
        )

    if k % group_size != 0:
        raise ValueError(
            f"`q_input.shape[1]` must be divisible by group_size, got K={k}, group_size={group_size}"
        )

    expected_x_scale_shape = (m, 1)
    if x_scale.numel() != m:
        raise ValueError(
            "Per-token x_scale must contain one scale per row: "
            f"x_scale.shape={tuple(x_scale.shape)}, expected numel={m}"
        )
    if x_scale.ndim == 2 and tuple(x_scale.shape) != expected_x_scale_shape:
        raise ValueError(
            "2D x_scale must have shape [M, 1]: "
            f"x_scale.shape={tuple(x_scale.shape)}, expected={expected_x_scale_shape}"
        )

    expected_weight_scale_shape = (weight_packed.shape[0], k // group_size)
    if tuple(weight_scale.shape) != expected_weight_scale_shape:
        raise ValueError(
            "weight_scale shape mismatch: "
            f"weight_scale.shape={tuple(weight_scale.shape)}, "
            f"expected={expected_weight_scale_shape}"
        )

    if q_input.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise ValueError(f"`q_input` must be FP8, got dtype={q_input.dtype}")
    if weight_packed.dtype != torch.int8:
        raise ValueError(
            f"`weight_packed` must be int8-packed int4, got dtype={weight_packed.dtype}"
        )
    if not x_scale.dtype.is_floating_point:
        raise ValueError(f"`x_scale` must be floating point, got dtype={x_scale.dtype}")
    if not weight_scale.dtype.is_floating_point:
        raise ValueError(
            f"`weight_scale` must be floating point, got dtype={weight_scale.dtype}"
        )

    if bias is not None:
        if bias.ndim != 1:
            raise ValueError(f"`bias` must be 1D, got shape={tuple(bias.shape)}")
        if bias.shape[0] != weight_packed.shape[0]:
            raise ValueError(
                "bias size mismatch: "
                f"bias.shape={tuple(bias.shape)}, expected=({weight_packed.shape[0]},)"
            )

    if not q_input.is_cuda:
        raise ValueError("`q_input` must be on CUDA")
    if not weight_packed.is_cuda or not x_scale.is_cuda or not weight_scale.is_cuda:
        raise ValueError("All W4A8-FP8 kernel inputs must be on CUDA")
    if bias is not None and not bias.is_cuda:
        raise ValueError("`bias` must be on CUDA when provided")


def w4a8_fp8_scaled_mm(
    q_input: torch.Tensor,
    weight_packed: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Thin Python wrapper around the dense W4A8-FP8 custom op.

    Args:
        q_input: [M, K], FP8 activation tensor.
        weight_packed: [N, K // 2], int8-packed signed int4 weights.
        x_scale: [M, 1], FP32 per-token activation scales.
        weight_scale: [N, K // group_size], FP32/BF16 group scales.
        group_size: quantization group size. MVP expects 128.
        out_dtype: output dtype, typically bf16.
        bias: optional [N] bias.

    Returns:
        [M, N] dense output tensor.
    """
    _validate_w4a8_fp8_inputs(
        q_input=q_input,
        weight_packed=weight_packed,
        x_scale=x_scale,
        weight_scale=weight_scale,
        group_size=group_size,
        out_dtype=out_dtype,
        bias=bias,
    )

    if not is_w4a8_fp8_linear_supported():
        raise NotImplementedError(
            "W4A8-FP8 dense kernel is not available in the current runtime. "
            "Expected CUDA SM90+ with `sgl_kernel.w4a8_fp8_scaled_mm` built and importable."
        )

    assert _w4a8_fp8_scaled_mm_op is not None
    return _w4a8_fp8_scaled_mm_op(
        q_input,
        weight_packed,
        x_scale,
        weight_scale,
        group_size,
        out_dtype,
        bias,
    )
