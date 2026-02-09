from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.sparse import (
    load_cache_to_device_buffer,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

device_module = get_device_module()


class SparseKVCacheManager:
    """
    Manages KV cache offloading between device (GPU) and host (CPU) memory
    for hierarchical sparse attention.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        server_args: ServerArgs,
    ) -> None:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import (
            MHATokenToKVPoolHost,
            MLATokenToKVPoolHost,
        )

        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.next_offload_id = 0

        # Initialize host memory pool based on KV cache type
        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()
        self.is_mla_pool = False
        if isinstance(self.mem_pool_device, MHATokenToKVPool):
            self.mem_pool_host = MHATokenToKVPoolHost(
                self.mem_pool_device,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
        elif isinstance(self.mem_pool_device, MLATokenToKVPool):
            self.mem_pool_host = MLATokenToKVPoolHost(
                self.mem_pool_device,
                server_args.hicache_ratio,
                server_args.hicache_size,
                1,
                server_args.hicache_mem_layout,
            )
            self.is_mla_pool = True
        else:
            raise ValueError("Unsupported KV cache type for sparse attention offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # Track pending offload operations
        self.pending_sparse_decode_offloads = {}
        self.pending_sparse_prompt_offloads = {}

        # Separate ack queues for prompt and decode offloads
        self.ack_sparse_prompt_write_queue = []
        self.ack_sparse_decode_write_queue = []

        self.write_queue = []
        self.write_stream = device_module.Stream()
        self.device = self.mem_pool_device.device
        self.io_backend = "kernel"

        # Initialize bitmap for tracking KV cache locations
        max_pool_size = self.req_to_token_pool.req_to_token.shape[0]
        self.bitmap = torch.full(
            (max_pool_size, server_args.model_config.context_len),
            -1,
            dtype=torch.int16,
            device=server_args.device,
        )
        self.req_states = None

    def swap_in_selected_pages(
        self,
        req_pool_indices,
        top_k_result,
        seq_lens,
        sparse_mask,
        page_table,
        layer_id,
        page_size,
        out_cache_loc,
    ):
        """
        Swap in selected top-k pages/tokens from host to device memory.
        First step: Using diff kernel to identify the top-k pages/tokens that need to be swapped in.
        Second step: Using the io kernel to load the pages/tokens from host to device.

        Returns:
            Device indices of the selected pages/tokens
        """
        import os

        if os.environ.get("SGLANG_DISABLE_SPARSE_KV_SWAPIN", "0") != "0":
            page_table_req = page_table[req_pool_indices]
            indices = top_k_result.to(dtype=torch.int64)
            if indices.numel() > 0:
                indices = indices.clamp_(0, page_table_req.shape[1] - 1)
            return torch.gather(page_table_req, dim=1, index=indices)

        _debug = os.environ.get("SGLANG_DEBUG_SPARSE_SWAPIN", "0") != "0"
        _debug_sync = os.environ.get("SGLANG_DEBUG_SPARSE_SWAPIN_SYNC", "0") != "0"

        bs = sparse_mask.shape[0]
        block_size = 512 if top_k_result.size(1) == 2048 else 32
        if _debug:
            try:
                topk_min = int(top_k_result.min().item()) if top_k_result.numel() > 0 else None
                topk_max = int(top_k_result.max().item()) if top_k_result.numel() > 0 else None
            except Exception:
                topk_min, topk_max = None, None
            try:
                mask_any = bool(sparse_mask.any().item())
            except Exception:
                mask_any = None
            print(
                "[DEBUG][SparseKVCacheManager.swap_in_selected_pages][0] begin "
                f"pid={os.getpid()} layer_id={layer_id} bs={bs} page_size={page_size} "
                f"sparse_mask_any={mask_any} "
                f"topk_shape={tuple(top_k_result.shape)} topk_min={topk_min} topk_max={topk_max} "
                f"seq_lens_max={int(seq_lens.max().item()) if seq_lens.numel() > 0 else None} "
                f"page_table_shape={tuple(page_table.shape)}",
                flush=True,
            )
        if self.is_mla_pool:
            load_cache_to_device_buffer_mla(
                top_k_tokens=top_k_result,
                device_buffer_tokens=self.req_states.last_top_k_result,
                host_cache_locs=self.req_states.req_to_tokens_host,
                device_buffer_locs=self.req_states.last_device_indices,
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=self.mem_pool_device.kv_buffer[layer_id],
                top_k_device_locs=self.req_states.curr_device_indices,
                page_table=page_table,
                diff_map=self.bitmap,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                seq_lens=seq_lens,
                lru_slots=self.req_states.lru_slots,
                transfer_tasks_src=self.req_states.transfer_tasks_src,
                transfer_tasks_dst=self.req_states.transfer_tasks_dst,
                page_size=page_size,
                layer_id=layer_id,
                item_size_bytes=self.mem_pool_host.token_stride_size,
                block_size=block_size,
            )
        else:
            load_cache_to_device_buffer(
                top_k_tokens=top_k_result,
                device_buffer_tokens=self.req_states.last_top_k_result,
                host_cache_locs=self.req_states.req_to_tokens_host,
                device_buffer_locs=self.req_states.last_device_indices,
                host_cache_k=self.mem_pool_host.k_buffer[layer_id],
                host_cache_v=self.mem_pool_host.v_buffer[layer_id],
                device_buffer_k=self.mem_pool_device.k_buffer[layer_id],
                device_buffer_v=self.mem_pool_device.v_buffer[layer_id],
                top_k_device_locs=self.req_states.curr_device_indices,
                page_table=page_table,
                diff_map=self.bitmap,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                seq_lens=seq_lens,
                lru_slots=self.req_states.lru_slots,
                transfer_tasks_src=self.req_states.transfer_tasks_src,
                transfer_tasks_dst=self.req_states.transfer_tasks_dst,
                page_size=page_size,
                layer_id=layer_id,
                item_size_bytes=self.mem_pool_host.token_stride_size,
                block_size=block_size,
            )

        if _debug and _debug_sync and torch.cuda.is_available():
            print(
                "[DEBUG][SparseKVCacheManager.swap_in_selected_pages][1] synchronize begin "
                f"pid={os.getpid()} layer_id={layer_id}",
                flush=True,
            )
            torch.cuda.current_stream().synchronize()
            print(
                "[DEBUG][SparseKVCacheManager.swap_in_selected_pages][2] synchronize end "
                f"pid={os.getpid()} layer_id={layer_id}",
                flush=True,
            )

        result = self.req_states.curr_device_indices[
            :bs, : self.req_states.topk_tokens_cnt // page_size
        ]
        if _debug:
            print(
                "[DEBUG][SparseKVCacheManager.swap_in_selected_pages][3] return "
                f"pid={os.getpid()} layer_id={layer_id} result_shape={tuple(result.shape)}",
                flush=True,
            )
        return result

    def offload_decode_token_kvcache(
        self, req_pool_indices, device_cache_locs, seq_lens
    ):
        """
        Offload newly generated decode token KV cache from device to host.

        Returns:
            Offload operation ID for tracking completion
        """
        self.next_offload_id += 1
        offload_id = self.next_offload_id
        host_indices = self._write_to_host(
            device_indices=device_cache_locs.long(),
            node_id=offload_id,
            sparse_ack_type="decode_offload",
        )
        assert host_indices is not None, "Host out of memory"
        self.pending_sparse_decode_offloads[offload_id] = (
            host_indices,
            req_pool_indices,
            seq_lens,
        )
        return offload_id

    def poll_decode_offload_completion(self):
        """
        Poll and finalize completed decode token KV cache offload operations.

        Checks if pending decode offload operations have completed and updates
        the host indices mapping.
        """
        if len(self.pending_sparse_decode_offloads) == 0:
            return

        queue_sizes = torch.tensor(
            [len(self.ack_sparse_decode_write_queue)],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_sizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        completed_count = queue_sizes.tolist()[0]

        # Process all completed offload operations
        while completed_count > 0:
            _, finish_event, offload_ids = self.ack_sparse_decode_write_queue.pop(0)
            finish_event.synchronize()

            # Update host indices mapping
            host_indices, req_pool_indices, seq_lens = (
                self.pending_sparse_decode_offloads.pop(offload_ids[0])
            )
            self.req_states.req_to_tokens_host[req_pool_indices, seq_lens] = (
                host_indices.to(self.req_states.device)
            )
            completed_count -= 1

    def offload_prompt_kvcache(self, req):
        """
        Offload full prompt KV cache from device to host after prefill.

        Returns:
            Offload operation ID for tracking completion
        """
        prompt_len = len(req.origin_input_ids)
        token_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prompt_len
        ].long()

        self.next_offload_id += 1
        offload_id = self.next_offload_id
        host_indices = self._write_to_host(
            device_indices=token_indices,
            node_id=offload_id,
            sparse_ack_type="prompt_offload",
        )
        assert host_indices is not None, "Host out of memory"
        self.pending_sparse_prompt_offloads[offload_id] = (host_indices, req)
        return offload_id

    def poll_prompt_offload_completion(self):
        """
        Poll and finalize completed prompt KV cache offload operations.

        Checks if pending prompt offload operations have completed and updates
        the host indices mapping for each completed request.

        Returns:
            List of requests whose prompt KV cache offload has completed
        """
        completed_reqs = []
        if len(self.pending_sparse_prompt_offloads) == 0:
            return completed_reqs

        completed_count = 0
        for _, finish_event, _ in self.ack_sparse_prompt_write_queue:
            if not finish_event.query():
                break
            completed_count += 1

        # Sync completion count across TP ranks
        if self.tp_world_size > 1:
            queue_size = torch.tensor(completed_count, dtype=torch.int, device="cpu")
            torch.distributed.all_reduce(
                queue_size, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
            completed_count = int(queue_size.item())

        # Process all completed offload operations
        while completed_count > 0:
            _, finish_event, offload_ids = self.ack_sparse_prompt_write_queue.pop(0)
            finish_event.synchronize()

            host_indices, req = self.pending_sparse_prompt_offloads.pop(offload_ids[0])
            self.req_states.req_to_tokens_host[req.req_pool_idx][
                : len(host_indices)
            ] = host_indices.to(self.req_states.device)
            completed_reqs.append(req)
            completed_count -= 1

        return completed_reqs

    def block_poll_prompt_offload_completion(self):
        completed_reqs = []
        if len(self.pending_sparse_prompt_offloads) == 0:
            return completed_reqs

        while True:
            qsizes = torch.tensor(
                [
                    len(self.ack_sparse_prompt_write_queue),
                ],
                dtype=torch.int,
            )
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(
                    qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
            completed_count = qsizes.tolist()[0]
            if completed_count > 0:
                break

        # Process all completed offload operations
        while completed_count > 0:
            _, finish_event, offload_ids = self.ack_sparse_prompt_write_queue.pop(0)
            finish_event.synchronize()

            host_indices, req = self.pending_sparse_prompt_offloads.pop(offload_ids[0])
            self.req_states.req_to_tokens_host[req.req_pool_idx][
                : len(host_indices)
            ] = host_indices.to(self.req_states.device)
            completed_reqs.append(req)
            completed_count -= 1

        return completed_reqs

    def _write_to_host(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        sparse_ack_type="prompt_offload",
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        from sglang.srt.managers.cache_controller import CacheOperation

        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self._start_writing(sparse_ack_type=sparse_ack_type)
        return host_indices

    def _start_writing(self, sparse_ack_type: str) -> None:
        from sglang.srt.managers.cache_controller import CacheOperation, HiCacheAck

        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        # Route ack to appropriate queue
        ack = HiCacheAck(start_event, finish_event, op.node_ids)

        if sparse_ack_type == "prompt_offload":
            self.ack_sparse_prompt_write_queue.append(ack)
        elif sparse_ack_type == "decode_offload":
            self.ack_sparse_decode_write_queue.append(ack)

    def move_indices(self, op):
        """Move indices to device if needed."""
        host_indices = op.host_indices.to(self.device, non_blocking=True)
        device_indices = op.device_indices.to(self.device, non_blocking=True)
        return host_indices, device_indices
