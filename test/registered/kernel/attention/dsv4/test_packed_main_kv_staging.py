import unittest

import torch

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
    gather_dequant_requant_fp8_paged,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.packed_main_kv_staging import (
    gather_packed_main_kv,
    stage_packed_main_kv,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_dsv41_packed_main_kv,
    quantize_dsv41_packed_main_kv,
)
from sglang.srt.mem_cache.dsv41_main_kv_layout import (
    PackedMainKVView,
    make_dsv41_packed_main_kv_spec,
)
from sglang.srt.mem_cache.dsv41_staging_workspace import MainKVStagingWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_caches(page_slots):
    torch.manual_seed(31)
    values = torch.randn((2, page_slots, 512), dtype=torch.bfloat16, device="cuda")
    values[0, :4] = 0
    values[0, 4] *= 0.001
    values[0, 5] *= 100
    spec = make_dsv41_packed_main_kv_spec(page_slots)
    view = PackedMainKVView(quantize_dsv41_packed_main_kv(values), spec)
    decoded = dequantize_dsv41_packed_main_kv(view.storage, page_slots).reshape(-1, 512)
    legacy = torch.zeros(
        (2, KVLayout.V4.page_bytes(page_slots)), dtype=torch.uint8, device="cuda"
    )
    fused_store_cache(
        input=decoded,
        cache=legacy,
        indices=torch.arange(2 * page_slots, dtype=torch.int32, device="cuda"),
        page_size=page_slots,
        type="flashmla",
        layout=KVLayout.V4,
    )
    return view, decoded, legacy


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.cuda is not None,
    "requires NVIDIA CUDA",
)
class TestPackedMainKVStaging(CustomTestCase):
    def test_stage_matches_actual_v4_writer(self):
        for p in (128, 256):
            with self.subTest(page_slots=p):
                view, decoded, legacy = make_caches(p)
                ws = MainKVStagingWorkspace(torch.device("cuda"), 64, query_tile=2)
                ids = (
                    torch.tensor(
                        [0, 1, 4, 5, p - 1, p, 2 * p - 1, -1, 2 * p, p],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    .repeat(13)[:128]
                    .view(2, 64)
                )
                original = ids.clone()
                lengths = torch.tensor([64, 17], dtype=torch.int32, device="cuda")
                _, remap = stage_packed_main_kv(view, ids, lengths, ws)
                expected_remap = torch.arange(128, device="cuda").view(2, 64)
                valid = (
                    (ids >= 0)
                    & (ids < 2 * p)
                    & (torch.arange(64, device="cuda") < lengths[:, None])
                )
                torch.testing.assert_close(
                    remap, torch.where(valid, expected_remap, -1).int(), atol=0, rtol=0
                )
                self.assertTrue(torch.equal(original, ids))
                selected = decoded[ids.clamp(0, 2 * p - 1).long()].reshape(-1, 512)
                selected[~valid.flatten()] = 0
                expected = torch.zeros_like(ws.pages)
                fused_store_cache(
                    input=selected,
                    cache=expected,
                    indices=torch.arange(128, dtype=torch.int32, device="cuda"),
                    page_size=256,
                    type="flashmla",
                    layout=KVLayout.V4,
                )
                # 比较完整物理页，包含清零的第八 scale 字节与 padding。
                self.assertTrue(torch.equal(ws.pages, expected))
                self.assertEqual(ws.cache.stride(0), KVLayout.V4.page_bytes(256))
                self.assertEqual(
                    ws.cache.untyped_storage().data_ptr(), ws.pages.data_ptr()
                )

    def test_positional_gather_bf16_and_fp8(self):
        for p in (128, 256):
            view, _, legacy = make_caches(p)
            ids = torch.tensor(
                [0, 4, 5, p - 1, p, 2 * p - 1, p], dtype=torch.int64, device="cuda"
            )
            for dtype, reference in (
                (torch.bfloat16, dequantize_k_cache_paged),
                (torch.float8_e4m3fn, gather_dequant_requant_fp8_paged),
            ):
                with self.subTest(page_slots=p, dtype=dtype):
                    out = torch.empty((ids.numel(), 1, 512), dtype=dtype, device="cuda")
                    gather_packed_main_kv(view, ids, out)
                    expected = reference(legacy, ids, p)
                    self.assertTrue(
                        torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
                    )
                    ids.fill_(-1)
                    gather_packed_main_kv(view, ids, out)
                    self.assertEqual(out.float().count_nonzero().item(), 0)
                    ids.copy_(
                        torch.tensor([0, 4, 5, p - 1, p, 2 * p - 1, p], device="cuda")
                    )

    def test_graph_replay_refreshes_data_indices_and_lengths(self):
        view, _, _ = make_caches(128)
        other, _, _ = make_caches(128)
        other.storage.copy_(view.storage.flip(0))
        ws = MainKVStagingWorkspace(torch.device("cuda"), 64, query_tile=2)
        eager_ws = MainKVStagingWorkspace(torch.device("cuda"), 64, query_tile=2)
        ids = torch.arange(128, dtype=torch.int32, device="cuda").view(2, 64)
        lengths = torch.tensor([64, 64], dtype=torch.int32, device="cuda")
        stage_packed_main_kv(view, ids, lengths, ws)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage_packed_main_kv(view, ids, lengths, ws)
        ptrs = (ws.pages.data_ptr(), ws.indices.data_ptr())
        for length in (1, 0, 33):
            view.storage.copy_(other.storage)
            ids.copy_(ids.flip(1))
            lengths.fill_(length)
            graph.replay()
            stage_packed_main_kv(view, ids, lengths, eager_ws)
            self.assertTrue(torch.equal(ws.pages, eager_ws.pages))
            self.assertTrue(torch.equal(ws.indices, eager_ws.indices))
            self.assertEqual(ptrs, (ws.pages.data_ptr(), ws.indices.data_ptr()))

    def test_no_lengths_empty_and_capacity(self):
        view, _, _ = make_caches(128)
        ws = MainKVStagingWorkspace(torch.device("cuda"), 64, query_tile=1)
        ids = torch.zeros((1, 64), dtype=torch.int32, device="cuda")
        _, remap = stage_packed_main_kv(view, ids, None, ws)
        torch.testing.assert_close(
            remap,
            torch.arange(64, dtype=torch.int32, device="cuda")[None, :],
            atol=0,
            rtol=0,
        )
        _, empty = stage_packed_main_kv(view, ids[:0], None, ws)
        self.assertEqual(empty.shape, (0, 64))
        with self.assertRaisesRegex(ValueError, "exceeds reserved"):
            stage_packed_main_kv(view, ids.expand(2, -1), None, ws)
        with self.assertRaisesRegex(ValueError, "lengths must"):
            stage_packed_main_kv(view, ids, torch.ones(2, device="cuda"), ws)


if __name__ == "__main__":
    unittest.main()
