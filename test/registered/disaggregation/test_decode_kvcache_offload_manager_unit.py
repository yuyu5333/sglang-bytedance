import unittest
from unittest.mock import MagicMock, patch
import torch
import time
from collections import defaultdict

from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.managers.schedule_batch import Req

class TestDecodeKVCacheOffloadManager(unittest.TestCase):
    def setUp(self):
        self.req_to_token_pool = MagicMock()
        self.token_to_kv_pool_allocator = MagicMock()
        
        # Mock KV cache type
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        mock_kv_cache = MagicMock(spec=MHATokenToKVPool)
        self.token_to_kv_pool_allocator.get_kvcache.return_value = mock_kv_cache

        self.tp_group = MagicMock()
        self.tree_cache = MagicMock()
        self.server_args = MagicMock()
        self.server_args.page_size = 16
        self.server_args.hicache_ratio = 2.0
        self.server_args.hicache_size = 1024
        self.server_args.hicache_mem_layout = "page_first"
        self.server_args.hicache_io_backend = "direct"
        self.server_args.hicache_storage_backend = "file"
        self.server_args.served_model_name = "test_model"
        self.server_args.hicache_storage_backend_extra_config = "{}"

        # Mock torch.distributed.get_world_size
        with patch("torch.distributed.get_world_size", return_value=1):
            # Mock memory pools
            with patch("sglang.srt.disaggregation.decode_kvcache_offload_manager.MHATokenToKVPoolHost") as MockPoolHost:
                with patch("sglang.srt.disaggregation.decode_kvcache_offload_manager.HiCacheController") as MockController:
                    # Fix isinstance check for mocks
                    with patch("sglang.srt.disaggregation.decode_kvcache_offload_manager.isinstance", side_effect=lambda obj, cls: True if (obj == mock_kv_cache and cls == MHATokenToKVPool) else isinstance(obj, cls)):
                        self.manager = DecodeKVCacheOffloadManager(
                            self.req_to_token_pool,
                            self.token_to_kv_pool_allocator,
                            self.tp_group,
                            self.tree_cache,
                            self.server_args
                        )
                        self.mock_controller = self.manager.cache_controller
                        self.mock_host_pool = self.manager.decode_host_mem_pool

    def test_abort_request_cleanup(self):
        """测试请求中断后的资源清理逻辑"""
        req = MagicMock(spec=Req)
        req.rid = "req_1"
        req.req_pool_idx = 0
        req.output_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # > 16 (page_size)
        req.origin_input_ids = [101, 102]
        req.finished.return_value = False
        req.pop_committed_kv_cache.return_value = 20
        req.pop_overallocated_kv_cache.return_value = (20, 20)
        req.prefix_indices = []
        
        # Mock req_to_token_pool with __getitem__ support for slice indexing
        tokens = torch.arange(100)
        def mock_getitem(index):
            if isinstance(index, tuple):
                idx, slc = index
                return tokens[slc]
            return tokens[index]
        self.req_to_token_pool.req_to_token.__getitem__.side_effect = mock_getitem
        
        # Mock cache_controller.write to return some host_indices
        host_indices = torch.tensor([10, 11])
        self.mock_controller.write.return_value = host_indices
        
        # 1. 触发一次卸载
        self.manager.offload_kv_cache(req)
        self.assertEqual(len(self.manager.ongoing_offload), 1)
        ack_id = self.manager.request_counter
        self.assertIn(ack_id, self.manager.req_to_ongoing_tasks[req.rid])

        # 2. 中断请求
        self.manager.abort_request(req)
        self.assertIn(req.rid, self.manager.aborted_requests)

        # 3. 模拟卸载完成
        # 构造 ack_write_queue 的元素: (start_event, finish_event, ack_list)
        mock_finish_event = MagicMock()
        self.mock_controller.ack_write_queue = [ (MagicMock(), mock_finish_event, [ack_id]) ]
        
        # 模拟分布式同步
        with patch("torch.distributed.all_reduce"):
            self.manager.check_offload_progress()

        # 4. 验证资源回收
        # 验证 _release_finished_req 被调用 (通过 free 调用验证)
        self.req_to_token_pool.free.assert_called_with(req)
        
        # 验证 host 内存被释放
        self.mock_host_pool.free.assert_called_with(host_indices)
        
        # 验证备份没有被触发
        self.mock_controller.write_storage.assert_not_called()
        
        # 验证追踪状态被清理
        self.assertEqual(len(self.manager.ongoing_offload), 0)
        self.assertEqual(len(self.manager.aborted_requests), 0)
        self.assertNotIn(req.rid, self.manager.req_to_ongoing_tasks)

    def test_release_finished_req_with_none_idx(self):
        """测试 req_pool_idx 为 None 时的安全性"""
        req = MagicMock(spec=Req)
        req.rid = "req_2"
        req.req_pool_idx = None
        
        # 不应抛出异常，且不应调用 free
        self.manager._release_finished_req(req, 0)
        self.req_to_token_pool.free.assert_not_called()

if __name__ == "__main__":
    unittest.main()
