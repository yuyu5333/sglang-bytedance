"""Fixed-BU4 DSV4 decode wrapper isolated from the upstream FlashMLA ABI."""

from typing import Optional, Tuple

import torch
from sgl_kernel.flash_mla import FlashMLASchedMeta

try:
    from sgl_kernel import kvbit_flashmla_ops  # noqa: F401
except Exception as _e:
    _kvbit_flashmla_import_error = _e
else:
    _kvbit_flashmla_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.kvbit_flashmla_ops extension. "
    "Ensure CUDA Driver >= 12.4."
)


def kvbit_flash_mla_with_kvcache(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    sched_meta: FlashMLASchedMeta,
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    packed_kcache: torch.Tensor,
    scale_kcache: torch.Tensor,
    R_matrix: torch.Tensor,
    zero_point: torch.Tensor,
    dim_of_bit: torch.Tensor,
    bitpos_in_dim: torch.Tensor,
    bit_uniform: int,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    q_nope_is_folded: bool = False,
    identity_tail_bypass: bool = True,
    debug_u32_packed_load: bool = False,
    extra_packed_kcache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _kvbit_flashmla_import_error is not None:
        raise _IMPORT_ERROR from _kvbit_flashmla_import_error
    if indices is None:
        raise ValueError("KVBit FlashMLA requires sparse decode indices")
    if bit_uniform != 4:
        raise ValueError(f"KVBit FlashMLA supports only fixed BU4, got {bit_uniform}")

    out, lse, new_metadata, new_num_splits = (
        torch.ops.sgl_kernel.kvbit_sparse_decode_fwd.default(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            head_dim_v,
            softmax_scale,
            packed_kcache,
            scale_kcache,
            R_matrix,
            zero_point,
            dim_of_bit,
            bitpos_in_dim,
            bit_uniform,
            q_nope_is_folded,
            identity_tail_bypass,
            debug_u32_packed_load,
            extra_packed_kcache,
        )
    )
    sched_meta.tile_scheduler_metadata = new_metadata
    sched_meta.num_splits = new_num_splits
    return out, lse


def kvbit_mxint4_flash_mla_with_kvcache(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    sched_meta: FlashMLASchedMeta,
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    packed_kcache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    extra_packed_kcache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed 360-byte MXINT4+Hadamard MODEL1/H64 specialization."""
    if _kvbit_flashmla_import_error is not None:
        raise _IMPORT_ERROR from _kvbit_flashmla_import_error
    if indices is None:
        raise ValueError("KVBit MXINT4 FlashMLA requires sparse decode indices")

    out, lse, new_metadata, new_num_splits = (
        torch.ops.sgl_kernel.kvbit_mxint4_sparse_decode_fwd.default(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            head_dim_v,
            softmax_scale,
            packed_kcache,
            extra_packed_kcache,
        )
    )
    sched_meta.tile_scheduler_metadata = new_metadata
    sched_meta.num_splits = new_num_splits
    return out, lse
