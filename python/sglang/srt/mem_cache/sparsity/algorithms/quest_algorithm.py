"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
"""

import logging

import torch
import triton

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_kernels import (
    quest_page_rep_kernel,
    quest_retrieval_score_kernel,
    quest_combine_indices_kernel,
)

logger = logging.getLogger(__name__)


class QuestAlgorithm(BaseSparseAlgorithmImpl):
    """Quest page-wise sparse attention using bounding-box criticality."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        for layer_id in range(start_layer, end_layer):
            self.page_k_min[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_k_max[layer_id] = torch.zeros_like(self.page_k_min[layer_id])
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        logger.info(
            "Initialized Quest page reps: %d pages, %d layers, head_num=%d, head_dim=%d",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
        )

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        req_to_token = self.req_to_token_pool.req_to_token
        head_num = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]

        # Determine BLOCK_DIM
        BLOCK_DIM = triton.next_power_of_2(head_dim)

        # Output tensors
        page_k_min = self.page_k_min[layer_id]
        page_k_max = self.page_k_max[layer_id]
        page_valid = self.page_valid[layer_id]

        grid = (n, max_pages, head_num)

        quest_page_rep_kernel[grid](
            page_k_min,
            page_k_max,
            page_valid,
            reqs,
            seq_lens,
            start_page,
            end_page,
            req_to_token,
            k_buffer,
            # Strides
            req_to_token.stride(0),
            req_to_token.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            page_k_min.stride(0),
            page_k_min.stride(1),
            page_k_min.stride(2),
            # Shapes
            req_to_token.shape[1],
            k_buffer.shape[0],
            # Constants
            PAGE_SIZE=self.page_size,
            HEAD_NUM=head_num,
            HEAD_DIM=head_dim,
            BLOCK_DIM=BLOCK_DIM,
        )

        if layer_id == 0:
            logger.info(
                f"Computed page representations for layer {layer_id}, start_page={start_page}, end_page={end_page}"
            )

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp pages to valid storage range
        phys_pages_clamped = phys_pages.clamp(0, self.page_k_min[layer_id].shape[0] - 1)

        k_min = self.page_k_min[layer_id][phys_pages_clamped]
        k_max = self.page_k_max[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]
        # Align query shape to KV heads.
        head_dim = k_min.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.view(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")

        kv_heads = k_min.shape[-2]
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            # Average grouped query heads to align with KV heads (approximation for MQA/GQA).
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(k_min.dtype).unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

        criticality = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=(2, 3))
        criticality = torch.where(
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))
        )

        return criticality

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        num_pages = seq_lens // self.page_size
        prompt_lens = self.states.prompt_lens[req_pool_indices]
        valid_mask = (
            ~self.states.repr_constructed[req_pool_indices]
            & (prompt_lens >= self.states.device_buffer_cnt)
            & (num_pages > 0)
        )

        if not valid_mask.any():
            return
        print(f"[DEBUG] [construct_representations] run _compute_page_representations")
        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            0,
            num_pages[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.repr_constructed[success_indices] = True
            self.states.last_constructed_page[success_indices] = num_pages[valid_mask]


    def update_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_decode_or_idle():
            return

        start_page = self.states.last_constructed_page[req_pool_indices]
        end_page = seq_lens // self.page_size
        valid_mask = self.states.repr_constructed[req_pool_indices] & (
            start_page < end_page
        )

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            end_page[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.last_constructed_page[success_indices] = end_page[valid_mask]

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        bs, device = queries.shape[0], queries.device
        
        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError("forward_batch with seq_lens is required for TopK retrieval")
        seq_lens = seq_lens_source.seq_lens.to(device)
        
        # Calculate dimensions
        num_pages = (seq_lens + self.page_size - 1) // self.page_size
        max_pages = int(num_pages.max().item())
        
        if max_pages == 0:
            return (
                torch.full((bs, 0), -1, dtype=torch.int32, device=device),
                torch.zeros(bs, dtype=torch.int32, device=device),
            )

        # Prepare kernel arguments
        req_to_token = self.req_to_token_pool.req_to_token
        page_k_min = self.page_k_min[layer_id]
        page_k_max = self.page_k_max[layer_id]
        
        head_num = page_k_min.shape[1]
        head_dim = page_k_min.shape[2]
        BLOCK_DIM = triton.next_power_of_2(head_dim)
        
        scores = torch.empty((bs, max_pages), dtype=torch.float32, device=device)
        
        grid = (bs, max_pages)

        # Handle 2D queries [bs, hidden_dim]
        if queries.dim() == 2:
            bs_q, hidden = queries.shape
            if hidden % head_dim != 0:
                 raise ValueError(f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}")
            q_heads = hidden // head_dim
            q = queries.view(bs_q, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")
        
        q_heads = q.shape[1]
        kv_heads = head_num
        
        GROUP_SIZE = 1
        if q_heads != kv_heads:
             if q_heads % kv_heads != 0:
                 raise ValueError(f"Query heads {q_heads} not divisible by KV heads {kv_heads}")
             GROUP_SIZE = q_heads // kv_heads
        
        # Ensure q is contiguous for Triton
        q = q.contiguous()
        
        quest_retrieval_score_kernel[grid](
            scores,
            req_pool_indices,
            seq_lens,
            req_to_token,
            page_k_min,
            page_k_max,
            q,
            # Strides
            scores.stride(0),
            scores.stride(1),
            req_to_token.stride(0),
            req_to_token.stride(1),
            page_k_min.stride(0),
            page_k_min.stride(1),
            page_k_min.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            # Shapes
            req_to_token.shape[1],
            page_k_min.shape[0],
            self.num_recent_pages,
            # Constants
            PAGE_SIZE=self.page_size,
            HEAD_NUM=head_num,
            HEAD_DIM=head_dim,
            BLOCK_DIM=BLOCK_DIM,
            GROUP_SIZE=GROUP_SIZE,
        )
        
        # Determine K per request
        recent_start = (num_pages - self.num_recent_pages).clamp(min=0)
        history_pages = recent_start.clamp(min=1)
        
        if self.fixed_topk_page_cnt is not None:
             k_target = max(self.fixed_topk_page_cnt - self.num_recent_pages, 1)
             k_per_req = torch.full((bs,), k_target, device=device)
        else:
             k_per_req = (history_pages * self.sparsity_ratio).long().clamp(min=1)
             
        k_per_req = torch.min(k_per_req, history_pages.long())
        
        # Apply sparse mask (if not sparse, we select 0 pages? or all? BaseAlgorithm logic says empty)
        # But we need to handle sparse_mask in combine kernel or here.
        # If sparse_mask is false, k=0?
        # In original code: if not sparse_mask[i]: continue (so empty list).
        k_per_req = k_per_req * sparse_mask.long()
        
        max_k = int(k_per_req.max().item())
        
        if max_k > 0:
            # Perform TopK with max_k
            # scores already has -inf for recent/invalid pages from kernel
            topk_vals, topk_indices = torch.topk(scores, k=min(max_k, max_pages), dim=1, sorted=False)
        else:
            topk_indices = torch.empty((bs, 0), dtype=torch.int64, device=device)
        
        # Combine and Sort using Kernel
        
        max_out = max_k + self.num_recent_pages
        # Initialize with INT_MAX so sort pushes padding to end
        out_indices = torch.full((bs, max_out), 2147483647, dtype=torch.int32, device=device)
        out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)
        
        combine_grid = (bs,)
        combine_block_size = 128
        
        quest_combine_indices_kernel[combine_grid](
            topk_indices,
            out_indices,
            out_lengths,
            seq_lens,
            k_per_req,
            # Strides
            topk_indices.stride(0) if topk_indices.numel() > 0 else 0,
            out_indices.stride(0),
            # Constants
            self.num_recent_pages,
            self.page_size,
            topk_indices.shape[1] if topk_indices.numel() > 0 else 0,
            BLOCK_SIZE=combine_block_size
        )
        
        # Sort indices (PyTorch sort is very fast)
        # sort(dim=1)
        out_indices, _ = out_indices.sort(dim=1)
        
        # Replace INT_MAX with -1 for padding
        out_indices.masked_fill_(out_indices == 2147483647, -1)
        
        return out_indices, out_lengths