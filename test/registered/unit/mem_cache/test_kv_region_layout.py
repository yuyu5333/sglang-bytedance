import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.kv_region_layout import (
    KVRegionLayout,
    decode_kv_region_registration,
    encode_kv_region_registration,
    kv_region_namespace,
    match_kv_region_layouts,
    parse_kv_region_layouts,
    serialize_kv_region_layouts,
    validate_kv_region_registration,
    validate_kv_region_topology,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def main_region(source=3, ratio=1):
    return KVRegionLayout(
        1,
        "kv",
        KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1.value,
        ratio,
        256,
        256 // ratio,
        384 * (256 // ratio),
        source,
    )


class TestKVRegionLayout(CustomTestCase):
    def test_versioned_geometry_and_registration(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                region = main_region(ratio=ratio)
                args = SimpleNamespace(
                    kv_region_layouts=[asdict(region)], kv_item_lens=[region.page_bytes]
                )
                decoded, sizes = decode_kv_region_registration(
                    encode_kv_region_registration(args), 1
                )
                self.assertEqual(parse_kv_region_layouts(decoded), (region,))
                self.assertEqual(sizes, [98304 // ratio])
                self.assertEqual(serialize_kv_region_layouts([region]), decoded)
        for changes in (
            dict(schema_version=2),
            dict(page_bytes=98303),
            dict(page_slots=128),
            dict(source_layer_id=-1),
            dict(compression_ratio=True),
            dict(kind="indexer"),
            dict(global_page_size=128),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(main_region(), **changes)
        for layouts, sizes, count in (
            ([main_region()], [1], 1),
            ([main_region()], [98304], 2),
            ([main_region(), main_region()], [98304, 98304], 2),
        ):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_kv_region_registration(layouts, sizes, count)

    def test_pp_mapping_and_layout_rejection(self):
        first, second = main_region(3), main_region(9)
        self.assertEqual(match_kv_region_layouts([second], [first, second]), [(0, 1)])
        self.assertIsNone(match_kv_region_layouts(None, None))
        legacy = replace(first, layout_id="v4", page_bytes=149760)
        for source, destination in (
            ([first], None),
            (None, [first]),
            ([first], [legacy]),
            ([first], [second]),
            ([first, second], [second, first]),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                match_kv_region_layouts(source, destination)
        validate_kv_region_topology([second], {"0": [first], "1": [second]})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_kv_region_topology([first], {"0": [legacy]})
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_kv_region_topology([first], None)

    def test_namespace_tracks_format_and_source_only(self):
        region = main_region()
        baseline = kv_region_namespace([region])
        self.assertEqual(baseline, kv_region_namespace([asdict(region)]))
        for changed in (
            replace(region, source_layer_id=8),
            replace(region, layout_id="v4", page_bytes=149760),
            main_region(ratio=2),
        ):
            self.assertNotEqual(baseline, kv_region_namespace([changed]))


if __name__ == "__main__":
    unittest.main()
