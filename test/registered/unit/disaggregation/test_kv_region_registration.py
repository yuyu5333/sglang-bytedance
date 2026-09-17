import asyncio
import json
import struct
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.mem_cache.kv_region_layout import (
    KVRegionLayout,
    encode_kv_region_registration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def region(source=3):
    return KVRegionLayout(
        1,
        "kv",
        "dsv41_main_kv_e2m1_block16_rope_bf16_v1",
        1,
        256,
        256,
        98304,
        source,
    )


def manager(layouts):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(kv_region_layouts=[asdict(r) for r in layouts])
    return mgr


def peer(layouts):
    return SimpleNamespace(
        dst_kv_region_layouts=[asdict(r) for r in layouts],
        dst_kv_item_lens=[r.page_bytes for r in layouts],
        dst_kv_ptrs=[0x1000 + i * 0x100000 for i in range(len(layouts))],
        dst_kv_layer_ids=[r.source_layer_id for r in layouts],
        dst_dcp_size=1,
    )


class TestCommonKVRegionRegistration(CustomTestCase):
    def test_peer_geometry_mapping_and_rejection_before_registration(self):
        mgr = manager([region(9)])
        target = peer([region(3), region(9)])
        self.assertEqual(mgr.validate_peer_kv_regions(target), [(0, 1)])
        for field, value in (
            ("dst_kv_region_layouts", None),
            ("dst_kv_item_lens", [1, 98304]),
            ("dst_kv_layer_ids", [3, 8]),
            ("dst_dcp_size", 2),
        ):
            invalid = peer([region(3), region(9)])
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                mgr.validate_peer_kv_regions(invalid)

        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        nixl = NixlKVManager.__new__(NixlKVManager)
        nixl.kv_args = mgr.kv_args
        nixl.agent = Mock(spec=["add_remote_agent"])
        wrong = peer([replace(region(9), layout_id="v4", page_bytes=149760)])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            nixl._add_remote_peer(wrong)
        nixl.agent.add_remote_agent.assert_not_called()

    def test_bootstrap_preserves_pp_local_layouts_and_rejects_mixed_rank(self):
        server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
        for key in (
            "attn_tp_size",
            "attn_cp_size",
            "dp_size",
            "pp_size",
            "page_size",
            "kv_cache_dtype",
            "dsv41_spec_layout",
            "prefill_http_port",
            "follow_bootstrap_room",
            "enable_dsa_cache_layer_split",
        ):
            setattr(server, key, None)
        server.kv_region_layouts_by_pp = {}
        server.prefill_port_table = {}
        server._registered_count = 0
        server.lock = asyncio.Lock()

        class Request:
            def __init__(self, data):
                self.data = data
                self.query = {
                    key: "-1"
                    for key in (
                        "prefill_dp_rank",
                        "prefill_cp_rank",
                        "target_tp_rank",
                        "target_pp_rank",
                    )
                }

            async def json(self):
                return self.data

        async def run():
            payload = dict(
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                pp_size=2,
                pp_rank=0,
                system_dp_size=1,
                system_dp_rank=0,
                rank_ip="127.0.0.1",
                rank_port=12000,
                page_size=256,
                kv_cache_dtype="fp8_e4m3",
                kv_region_layouts=[asdict(region(3))],
            )
            response = await server._handle_route_put(Request(payload))
            self.assertEqual(response.status, 200)
            bad = dict(payload, kv_region_layouts=None)
            self.assertEqual((await server._handle_route_put(Request(bad))).status, 400)
            payload.update(pp_rank=1, kv_region_layouts=[asdict(region(9))])
            self.assertEqual(
                (await server._handle_route_put(Request(payload))).status, 200
            )
            result = await server._handle_route_get(Request({}))
            self.assertEqual(result.status, 200)
            layouts = json.loads(result.text)["kv_region_layouts_by_pp"]
            self.assertEqual(layouts["0"][0]["source_layer_id"], 3)
            self.assertEqual(layouts["1"][0]["source_layer_id"], 9)

        asyncio.run(run())


class TestKVRegionWire(CustomTestCase):
    def test_mooncake_wire_retains_staging_and_layout_contract(self):
        from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo

        contract = encode_kv_region_registration(
            SimpleNamespace(kv_region_layouts=[asdict(region())], kv_item_lens=[98304])
        )
        frames = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"session",
            struct.pack("Q", 0x1000),
            b"",
            b"",
            b"0",
            b"1",
            b"98304",
            b"",
            b"",
            struct.pack("I", 3),
            b"",
            b"",
            b"",
            b"1",
            b"0",
            b"",
        ]
        info = KVArgsRegisterInfo.from_zmq(frames + [contract])
        self.assertEqual(info.dst_kv_item_lens, [98304])
        self.assertEqual(info.dst_kv_region_layouts, [asdict(region())])
        self.assertEqual(manager([region()]).validate_peer_kv_regions(info), [(0, 0)])
        old = KVArgsRegisterInfo.from_zmq(frames)
        with self.assertRaisesRegex(ValueError, "missing"):
            manager([region()]).validate_peer_kv_regions(old)

    def test_nixl_wire_checks_descriptor_strides(self):
        from sglang.srt.disaggregation.nixl.conn import KVArgsRegisterInfo

        contract = encode_kv_region_registration(
            SimpleNamespace(kv_region_layouts=[asdict(region())], kv_item_lens=[98304])
        )
        frames = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            b"",
            b"",
            b"0",
            b"1",
            b"0",
            b"98304",
            b"",
            b"",
            b"",
            b"",
            b"3",
            b"VRAM",
            struct.pack("Q", 98304),
            b"",
            struct.pack("I", 3),
            b"1",
            b"0",
        ]
        info = KVArgsRegisterInfo.from_zmq(frames + [contract])
        self.assertEqual(manager([region()]).validate_peer_kv_regions(info), [(0, 0)])
        frames[18] = struct.pack("Q", 149760)
        with self.assertRaisesRegex(ValueError, "strides"):
            KVArgsRegisterInfo.from_zmq(frames + [contract])


if __name__ == "__main__":
    unittest.main()
