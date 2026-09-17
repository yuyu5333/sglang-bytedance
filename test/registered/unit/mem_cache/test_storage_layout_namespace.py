import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.kv_region_layout import KVRegionLayout, kv_region_namespace
from sglang.srt.mem_cache.storage_layout_namespace import LayoutNamespacedStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLayoutNamespacedStorage(CustomTestCase):
    def test_file_pages_do_not_cross_layouts_or_sources(self):
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="same-model",
        )
        region = KVRegionLayout(
            1,
            "kv",
            "dsv41_main_kv_e2m1_block16_rope_bf16_v1",
            1,
            256,
            256,
            98304,
            3,
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = HiCacheFile(config, directory)
            packed = LayoutNamespacedStorage(backend, kv_region_namespace([region]))
            legacy = LayoutNamespacedStorage(
                backend,
                kv_region_namespace(
                    [replace(region, layout_id="v4", page_bytes=149760)]
                ),
            )
            other_source = LayoutNamespacedStorage(
                backend, kv_region_namespace([replace(region, source_layer_id=7)])
            )
            page = torch.arange(98304, dtype=torch.int64).to(torch.uint8)
            key = "identical-token-prefix"
            self.assertTrue(packed.set(key, page))
            self.assertTrue(packed.exists(key))
            self.assertFalse(legacy.exists(key))
            self.assertFalse(other_source.exists(key))
            self.assertFalse(backend.exists(key))
            out = torch.empty_like(page)
            restored = packed.get(key, out)
            self.assertTrue(torch.equal(restored, page))
            self.assertTrue(packed.batch_set([key], [page]))
            self.assertEqual(packed.batch_exists([key]), 1)
            self.assertTrue(torch.equal(packed.batch_get([key], [out])[0], page))

    def test_v1_v2_keys_and_prefixes_are_transformed_without_mutation(self):
        backend = Mock(spec=HiCacheFile)
        storage = LayoutNamespacedStorage(backend, "layout")
        keys = ["a", "b"]
        extra = HiCacheStorageExtraInfo(prefix_keys=["p"])
        transfer = PoolTransfer(PoolName.DEEPSEEK_V4_C1, keys=["a"])
        indices = torch.arange(2)
        for method in ("batch_get_v1", "batch_set_v1"):
            with self.subTest(method=method):
                getattr(storage, method)(keys, indices, extra)
                args = getattr(backend, method).call_args.args
                self.assertNotEqual(args[0], keys)
                self.assertNotEqual(args[2].prefix_keys, extra.prefix_keys)
                self.assertIs(args[1], indices)
        for method in ("batch_get_v2", "batch_set_v2"):
            with self.subTest(method=method):
                getattr(storage, method)([transfer], extra)
                args = getattr(backend, method).call_args.args
                self.assertNotEqual(args[0][0].keys, transfer.keys)
                self.assertNotEqual(args[1].prefix_keys, extra.prefix_keys)
        storage.batch_exists_v2(keys, [transfer], extra)
        exists_args = backend.batch_exists_v2.call_args.args
        self.assertEqual(exists_args[0][0], exists_args[1][0].keys[0])
        self.assertEqual(keys, ["a", "b"])
        self.assertEqual(extra.prefix_keys, ["p"])
        self.assertEqual(transfer.keys, ["a"])


if __name__ == "__main__":
    unittest.main()
