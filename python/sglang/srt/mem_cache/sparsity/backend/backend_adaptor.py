import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
import triton
from sglang.srt.utils.common import is_cuda, is_hip
from sglang.srt.mem_cache.sparsity.kernel.flashattn_metadata_kernels import (
    update_page_table_triton,
    compute_sparse_seqlens_triton,
)

from sgl_kernel import quest_diff_and_update_sparse_metadata, quest_update_sparse_metadata


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

    def __init__(self, device: torch.device):
        self.device = device
        self._original_metadata = None

    def save_original_metadata(self, metadata: Any) -> None:
        """Save original metadata in the beginning of the forward pass."""
        pass

    @abstractmethod
    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        """
        Adapt attention metadata for sparse KVCache access.

        Transforms sparse retrieval results (logical indices of important KV pages/tokens)
        into backend-specific attention metadata format.

        Returns:
            Modified attention metadata compatible with the backend
        """
        pass


class NSABackendAdaptor(BackendAdaptor):
    """Adaptor for NSA (Native Sparse Attention) backend."""

    def __init__(
        self,
        device: torch.device,
        req_to_token_pool,
    ):
        super().__init__(device)
        self.req_to_token_pool = req_to_token_pool

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Transform logical page indices to physical device indices for NSA backend.
        """
        # TODO: Implement NSA backend adaptor logic
        pass


class FlashAttentionAdaptor(BackendAdaptor):
    """Adaptor for FlashAttention backend."""

    def __init__(
        self, device: torch.device, req_to_token_pool, sparse_kv_cache_manager
    ):
        super().__init__(device)
        self.req_to_token_pool = req_to_token_pool
        self.sparse_kv_cache_manager = sparse_kv_cache_manager

    def save_original_metadata(self, metadata: Any) -> None:
        self._original_metadata = {
            "page_table": metadata.page_table.clone(),
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "cu_seqlens_k": metadata.cu_seqlens_k.clone(),
            "max_seq_len_k": metadata.max_seq_len_k,
        }

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        """
        Adapt FlashAttention metadata for sparse KVCache access.

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.

        # TODO: Optimize performance
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        req_states = self.sparse_kv_cache_manager.req_states
        batch_size = sparse_mask.shape[0]

        topk_tokens_cnt = req_states.topk_tokens_cnt
        topk_pages = topk_tokens_cnt // page_size
        physical_pages = req_states.curr_device_indices[:batch_size, :topk_pages]
        topk_page_indices = selected_indices[:, :topk_pages].to(torch.int32).contiguous()

        if True:
            print(f"page_table: Python type={type(current_metadata.page_table)}, tensor dtype={current_metadata.page_table.dtype}, shape={current_metadata.page_table.shape}, continue?={current_metadata.page_table.is_contiguous()}")
            print(f"last_top_k_result: Python type={type(req_states.last_top_k_result)}, tensor dtype={req_states.last_top_k_result.dtype}, shape={req_states.last_top_k_result.shape}, continue?={req_states.last_top_k_result.is_contiguous()}")
            print(f"last_device_indices: Python type={type(req_states.last_device_indices)}, tensor dtype={req_states.last_device_indices.dtype}, shape={req_states.last_device_indices.shape}, continue?={req_states.last_device_indices.is_contiguous()}")
            print(f"topk_page_indices: Python type={type(topk_page_indices)}, tensor dtype={topk_page_page_indices.dtype}, shape={topk_page_page_indices.shape}, continue?={topk_page_indices.is_contiguous()}")
            print(f"req_pool_indices: Python type={type(forward_batch.req_pool_indices)}, tensor dtype={forward_batch.req_pool_indices.dtype}, shape={forward_batch.req_pool_indices.shape}, continue?={forward_batch.req_pool_indices.is_contiguous()}")
            print(f"seq_lens: Python type={type(forward_batch.seq_lens)}, tensor dtype={forward_batch.seq_lens.dtype}, shape={forward_batch.seq_lens.shape}, continue?={forward_batch.seq_lens.is_contiguous()}")
            print(f"valid_lengths: Python type={type(valid_lengths)}, tensor dtype={valid_lengths.dtype}, shape={valid_lengths.shape}, continue?={valid_lengths.is_contiguous()}")
            print(f"sparse_mask: Python type={type(sparse_mask)}, tensor dtype={sparse_mask.dtype}, shape={sparse_mask.shape}, continue?={sparse_mask.is_contiguous()}")
            print(f"req_to_tokens_host: Python type={type(req_states.req_to_tokens_host)}, tensor dtype={req_states.req_to_tokens_host.dtype}, shape={req_states.req_to_tokens_host.shape}, continue?={req_states.req_to_tokens_host.is_contiguous()}")
            print(f"physical_pages: Python type={type(physical_pages)}, tensor dtype={physical_pages.dtype}, shape={physical_pages.shape}, continue?={physical_pages.is_contiguous()}")
            print(f"should_load_device_indices: Python type={type(req_states.should_load_device_indices)}, tensor dtype={req_states.should_load_device_indices.dtype}, shape={req_states.should_load_device_indices.shape}, continue?={req_states.should_load_device_indices.is_contiguous()}")
            print(f"should_load_host_indices: Python type={type(req_states.should_load_host_indices)}, tensor dtype={req_states.should_load_host_indices.dtype}, shape={req_states.should_load_host_indices.shape}, continue?={req_states.should_load_host_indices.is_contiguous()}")
            print(f"cache_seqlens_int32 (current): Python type={type(current_metadata.cache_seqlens_int32)}, tensor dtype={current_metadata.cache_seqlens_int32.dtype}, shape={current_metadata.cache_seqlens_int32.shape}, continue?={current_metadata.cache_seqlens_int32.is_contiguous()}")
            print(f"cache_seqlens_int32 (original): Python type={type(self._original_metadata['cache_seqlens_int32'])}, tensor dtype={self._original_metadata['cache_seqlens_int32'].dtype}, shape={self._original_metadata['cache_seqlens_int32'].shape}, continue?={self._original_metadata['cache_seqlens_int32'].is_contiguous()}")
            print(f"\nlayer_id: Python type={type(layer_id)}, value={layer_id}")
            print(f"page_size: Python type={type(page_size)}, value={page_size}")

        quest_diff_and_update_sparse_metadata(
            current_metadata.page_table,
            req_states.last_top_k_result,
            req_states.last_device_indices,
            topk_page_indices,
            forward_batch.req_pool_indices.to(torch.int32).contiguous(),
            forward_batch.seq_lens.to(torch.int32).contiguous(),
            valid_lengths.to(torch.int32).contiguous(),
            sparse_mask.to(torch.int32).contiguous(),
            req_states.req_to_tokens_host,
            physical_pages,
            req_states.should_load_device_indices,
            req_states.should_load_host_indices,
            current_metadata.cache_seqlens_int32,
            self._original_metadata["cache_seqlens_int32"],
            layer_id,
            page_size
        )

        # Data Loading
        swap_target_device_slots = req_states.should_load_device_indices[:batch_size, :topk_tokens_cnt]
        swap_source_host_slots = req_states.should_load_host_indices[:batch_size, :topk_tokens_cnt]
        
        target_valid = swap_target_device_slots[swap_target_device_slots != -1]
        source_valid = swap_source_host_slots[swap_source_host_slots != -1]
        
        if target_valid.numel() > 0:
             self.sparse_kv_cache_manager.mem_pool_host.load_to_device_per_layer(
                self.sparse_kv_cache_manager.mem_pool_device,
                source_valid.flatten(),
                target_valid.flatten(),
                layer_id,
                "kernel"
             )

        if True:
            print(f"page_table: Python type={type(current_metadata.page_table)}, tensor dtype={current_metadata.page_table.dtype}, shape={current_metadata.page_table.shape}, continue?={current_metadata.page_table.is_contiguous()}")
            print(f"physical_pages: Python type={type(physical_pages)}, tensor dtype={physical_pages.dtype}, shape={physical_pages.shape}, continue?={physical_pages.is_contiguous()}")
            print(f"valid_lengths: Python type={type(valid_lengths)}, tensor dtype={valid_lengths.dtype}, shape={valid_lengths.shape}, continue?={valid_lengths.is_contiguous()}")
            print(f"seq_lens: Python type={type(forward_batch.seq_lens)}, tensor dtype={forward_batch.seq_lens.dtype}, shape={forward_batch.seq_lens.shape}, continue?={forward_batch.seq_lens.is_contiguous()}")
            print(f"cache_seqlens_int32 (current): Python type={type(current_metadata.cache_seqlens_int32)}, tensor dtype={current_metadata.cache_seqlens_int32.dtype}, shape={current_metadata.cache_seqlens_int32.shape}, continue?={current_metadata.cache_seqlens_int32.is_contiguous()}")
            print(f"sparse_mask: Python type={type(sparse_mask)}, tensor dtype={sparse_mask.dtype}, shape={sparse_mask.shape}, continue?={sparse_mask.is_contiguous()}")
            print(f"cache_seqlens_int32 (original): Python type={type(self._original_metadata['cache_seqlens_int32'])}, tensor dtype={self._original_metadata['cache_seqlens_int32'].dtype}, shape={self._original_metadata['cache_seqlens_int32'].shape}, continue?={self._original_metadata['cache_seqlens_int32'].is_contiguous()}")
            print(f"page_size: Python type={type(page_size)}, value={page_size}")

        quest_update_sparse_metadata(
            current_metadata.page_table,
            physical_pages,
            valid_lengths.to(torch.int32).contiguous(),
            sparse_mask.to(torch.int32).contiguous(),
            current_metadata.cache_seqlens_int32,
            forward_batch.seq_lens.to(torch.int32).contiguous(),
            self._original_metadata["cache_seqlens_int32"],
            page_size
        )

        current_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        current_metadata.max_seq_len_k = int(current_metadata.cache_seqlens_int32.max())
        return current_metadata


    def adapt_for_attn_metadata_python(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        """
        Adapt FlashAttention metadata for sparse KVCache access.

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.

        # TODO: Optimize performance
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        page_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]
        physical_pages = self.sparse_kv_cache_manager.swap_in_selected_pages(
            req_pool_indices=forward_batch.req_pool_indices,
            top_k_result=selected_indices,
            seq_lens=forward_batch.seq_lens,
            sparse_mask=sparse_mask,
            page_table=page_table,
            layer_id=layer_id,
            page_size=page_size,
            out_cache_loc=forward_batch.out_cache_loc,
        )
        max_selected = physical_pages.shape[1]
        valid_mask = torch.arange(max_selected, device=physical_pages.device).unsqueeze(
            0
        ) < valid_lengths.unsqueeze(1)
        update_mask = sparse_mask.unsqueeze(1) & valid_mask

        current_metadata.page_table[:, :max_selected] = torch.where(
            update_mask, physical_pages, current_metadata.page_table[:, :max_selected]
        )

        seq_lens = forward_batch.seq_lens
        positions_in_page = (seq_lens - 1) % page_size
        diff = page_size - positions_in_page - 1
        sparse_seq_lens = (valid_lengths * page_size - diff).to(torch.int32)

        current_metadata.cache_seqlens_int32 = torch.where(
            sparse_mask, sparse_seq_lens, self._original_metadata["cache_seqlens_int32"]
        )

        current_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        current_metadata.max_seq_len_k = int(current_metadata.cache_seqlens_int32.max())
        return current_metadata
