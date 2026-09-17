"""Exercise the runtime dispatcher with real H20 FlashMLA kernels."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
    gather_dequant_requant_fp8_paged,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4AttnMetadata,
    DeepseekV4AttnBackend,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import SparsePrefillWorkspace
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.dsv41_staging_workspace import MainKVStagingWorkspace
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

from test_packed_main_kv_staging import make_caches

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 FlashMLA required",
)


def make_case(rows, heads, page_slots):
    from sgl_kernel.flash_mla import get_mla_metadata

    view, _, legacy = make_caches(page_slots)
    _, _, swa = make_caches(128)
    # Avoid the deliberately enormous scale-boundary row in the staging fixture.
    ids = torch.randint(8, 2 * page_slots, (rows, 64), device="cuda", dtype=torch.int32)
    ids[:, 1] = -1
    ids[:, 2] = ids[:, 0]
    ids[:, 3] = 2 * page_slots
    lengths = torch.full((rows,), 64, device="cuda", dtype=torch.int32)
    lengths[0] = 0
    swa_ids = torch.arange(8, 136, device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
    core = DSV4AttnMetadata.__new__(DSV4AttnMetadata)
    core.swa_page_indices = swa_ids
    core.swa_topk_lengths = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
    ratio = 256 // page_slots
    setattr(core, f"c{ratio}_sparse_page_indices", ids)
    setattr(core, f"c{ratio}_sparse_topk_lengths", lengths)
    setattr(core, f"c{ratio}_flashmla_metadata", get_mla_metadata()[0])
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool.swa_page_size = 128
    pool.request_window = None
    pool.get_swa_key_buffer_radix = lambda _: swa
    pool.get_swa_key_layout = lambda: KVLayout.V4
    pool.get_swa_key_bytes_per_token = lambda: 584
    pool.get_extra_key_layout = lambda _: view.spec.layout_id
    pool.get_extra_key_view = lambda _: view
    pool.get_extra_key_page_size = lambda _: page_slots
    pool.get_extra_key_buffer = lambda _: legacy
    pool.get_extra_key_bytes_per_token = lambda _: 584
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.mtp_enabled = False
    backend.trtllm_attn = False
    backend.dsv41_main_kv_consumer = "staged"
    backend.main_kv_staging_workspace = MainKVStagingWorkspace(torch.device("cuda"), 64)
    backend.sparse_prefill_workspace = SparsePrefillWorkspace(torch.device("cuda"))
    backend.forward_metadata = SimpleNamespace(
        core_attn_metadata=core, sparse_prefill_cache=None, late_layer_tail=None
    )
    backend.token_to_kv_pool = pool
    backend.softmax_scale = 512**-0.5
    backend.head_dim_v = 512
    backend.dsv4_prefill_backend = "auto"
    q = torch.randn((rows, heads, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    sink = torch.linspace(-1, 1, heads, device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(layer_id=0, v_head_dim=512, tp_q_head_num=heads)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    kv = q[:, 0]

    def run():
        return backend._forward_attention(
            q, kv, kv, layer, batch, ratio, save_kv_cache=False, attn_sink=sink
        )

    def reference():
        from sgl_kernel.flash_mla import flash_mla_with_kvcache

        # The packed adapter promises bounds masking. The legacy SM90 reader
        # requires -1 for OOB slots, so normalize its reference input explicitly.
        reference_ids = torch.where(
            (ids >= 0) & (ids < 2 * page_slots), ids, -1
        )
        return flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=swa.as_strided((2, 128, 1, 584), (swa.stride(0), 584, 584, 1)),
            block_table=None, cache_seqlens=None, head_dim_v=512,
            tile_scheduler_metadata=get_mla_metadata()[0],
            softmax_scale=backend.softmax_scale, is_fp8_kvcache=True,
            indices=swa_ids.unsqueeze(1), topk_length=core.swa_topk_lengths,
            attn_sink=sink,
            extra_k_cache=legacy.as_strided(
                (2, page_slots, 1, 584), (legacy.stride(0), 584, 584, 1)
            ),
            extra_indices_in_kvcache=reference_ids.unsqueeze(1), extra_topk_length=lengths,
        )[0].squeeze(1)

    return backend, view, legacy, ids, lengths, run, reference


@pytest.mark.parametrize("rows,heads,page_slots", [(2, 64, 128), (65, 128, 256)])
def test_runtime_staged_matches_legacy(rows, heads, page_slots):
    backend, view, legacy, ids, lengths, run, reference = make_case(rows, heads, page_slots)
    original = ids.clone()
    torch.testing.assert_close(run(), reference(), atol=0.002, rtol=0.02)
    assert torch.equal(ids, original)
    # Gather goes through the backend's typed format adapter for both dtypes.
    selected = torch.tensor([8, page_slots - 1, page_slots, 8], device="cuda")
    for dtype, gather in (
        (torch.bfloat16, dequantize_k_cache_paged),
        (torch.float8_e4m3fn, gather_dequant_requant_fp8_paged),
    ):
        out = torch.empty((4, 1, 512), dtype=dtype, device="cuda")
        backend._gather_prefill_main(
            backend.token_to_kv_pool, 0, selected, out, q8=dtype != torch.bfloat16
        )
        expected = gather(legacy, selected, page_slots)
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def test_runtime_graph_replay_with_mutated_main():
    backend, view, legacy, ids, lengths, run, reference = make_case(65, 64, 128)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    ptr = backend.main_kv_staging_workspace.pages.data_ptr()
    for length in (3, 0, 64):
        ids.copy_(ids.flip(1))
        lengths.fill_(length)
        # Reuse the same physical pages with different data on both sides.
        view.storage.copy_(view.storage.flip(0))
        legacy.copy_(legacy.flip(0))
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=0.002, rtol=0.02)
        assert ptr == backend.main_kv_staging_workspace.pages.data_ptr()


@pytest.mark.parametrize("page_slots", [128, 256])
def test_prefill_attention_packed_matches_legacy(page_slots):
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        SparsePrefillChunkCache,
    )

    backend, view, legacy, _, _, _, _ = make_case(2, 64, page_slots)
    pool = backend.token_to_kv_pool
    ratio = 256 // page_slots
    ints = dict(device="cuda", dtype=torch.int32)
    lens = torch.tensor([192, 224], **ints)
    extend = torch.tensor([2, 2], **ints)
    positions = torch.tensor([190, 191, 222, 223], **ints)
    mapping = torch.arange(256, **ints)
    cache = SparsePrefillChunkCache.build(
        seq_lens=lens, extend_seq_lens=extend, query_lens=extend,
        query_pos=positions, req_pool_indices=torch.tensor([0, 1], **ints),
        req_to_token=mapping[None].repeat(2, 1), full_to_swa=mapping.long(),
        swa_window_size=128, swa_page_size=128, num_qo_tokens=4,
        max_seq_len=224, total_swa=258,
    )
    core = backend.forward_metadata.core_attn_metadata
    core.page_table = torch.zeros((4, 1), **ints)
    raw = torch.arange(8, 72, **ints)[None].repeat(4, 1)
    raw[:, 3] = -1
    setattr(core, f"c{ratio}_sparse_raw_indices", raw)
    backend.forward_metadata.sparse_prefill_cache = cache
    q = torch.randn((4, 1, 64, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    sink = torch.zeros(64, device="cuda", dtype=torch.float32)
    for method in (backend._forward_prefill_sparse, backend._forward_prefill_sparse_q8kv8):
        kwargs = dict(
            q=q, layer_id=0, compress_ratio=ratio, forward_batch=None,
            token_to_kv_pool=pool, core_attn_metadata=core, attn_sink=sink,
        )
        pool.get_extra_key_layout = lambda _: view.spec.layout_id
        actual = method(**kwargs)
        pool.get_extra_key_layout = lambda _: KVLayout.V4
        expected = method(**kwargs)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
