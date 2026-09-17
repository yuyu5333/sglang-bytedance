"""FULL-page HiCache copies use the same buffers exported to PD."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.dsv41_main_kv_layout import make_dsv41_packed_main_kv_spec
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "layout,io",
    [
        ("layer_first", "kernel"),
        ("page_first", "kernel"),
        ("layer_first", "direct"),
        ("page_first_direct", "direct"),
    ],
)
def test_packed_full_page_roundtrip(layout, io):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=256))
    pool = DeepSeekV4TokenToKVPool(
        max_num_reqs=2,
        swa_size=512,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=5,
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=[0, 2, 2, 1, 1],
        kv_source_layers=[1, 3],
        full_size=512,
        main_kv_layout_specs={
            1: make_dsv41_packed_main_kv_spec(256),
            2: make_dsv41_packed_main_kv_spec(128),
        },
    )
    with patch.object(
        assembler, "get_memory", return_value=SimpleNamespace(hicache_mem_layout=layout)
    ):
        entries = assembler._dsv4_low_ratio_entries(pool, 256, 3, 5)
    assert len(entries) == 4  # CUDA C1/C2 Main and fused FP4 indexer pages
    assert {
        r.layout.layout_id
        for r in pool.get_kv_transfer_regions()
        if r.layout.kind == "indexer"
    } == {"indexer_fp4_fused_block32_v1"}
    region_ptrs = {r.buffer.data_ptr() for r in pool.get_kv_transfer_regions()}
    host_ids = torch.arange(256, device="cuda", dtype=torch.int64)
    source_ids = torch.arange(256, 512, device="cuda", dtype=torch.int64)
    target_ids = torch.arange(512, 768, device="cuda", dtype=torch.int64)
    for entry in entries:
        host = entry.host_pool
        assert host.page_aligned_only
        assert host.item_bytes == host.device_buffers[0].shape[1]
        expected = []
        for i, buffer in enumerate(host.device_buffers):
            assert buffer.data_ptr() in region_ptrs
            page = (
                (torch.arange(host.item_bytes, device="cuda") + 17 + i) % 256
            ).byte()
            buffer[1].copy_(page)
            buffer[2].fill_(211)
            expected.append(page.cpu())
        host.backup_from_device_all_layer(entry.device_pool, host_ids, source_ids, io)
        torch.cuda.synchronize()
        for i, buffer in enumerate(host.device_buffers):
            buffer[1].zero_()  # reused source page cannot supply the restored bytes
            host.load_to_device_per_layer(
                entry.device_pool, host_ids, target_ids, i, io
            )
        torch.cuda.synchronize()
        for i, buffer in enumerate(host.device_buffers):
            assert torch.equal(buffer[2].cpu(), expected[i])
            assert buffer[1].count_nonzero().item() == 0
        with pytest.raises(ValueError, match="page-aligned"):
            host.backup_from_device_all_layer(
                entry.device_pool, host_ids[:1], source_ids[:1], io
            )
        with pytest.raises(ValueError, match="page-aligned"):
            host.load_to_device_per_layer(
                entry.device_pool, host_ids[:1], target_ids[:1], 0, io
            )
