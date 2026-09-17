"""Versioned page geometry shared by HiCache and PD.

Addresses, allocation sizes and attention consumer choices are deliberately
absent: a page has the same wire format in differently sized pools.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import NamedTuple

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


@dataclass(frozen=True)
class KVRegionLayout:
    schema_version: int
    kind: str
    layout_id: str
    compression_ratio: int
    global_page_size: int
    page_slots: int
    page_bytes: int
    source_layer_id: int

    def __post_init__(self):
        for name in (
            "schema_version",
            "compression_ratio",
            "global_page_size",
            "page_slots",
            "page_bytes",
            "source_layer_id",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (
                0 if name == "source_layer_id" else 1
            ):
                raise ValueError(f"KV region layout: invalid {name}={value!r}")
        if self.schema_version != 1:
            raise ValueError("KV region layout: unsupported schema version")
        if self.kind not in ("kv", "indexer", "indexer_payload", "indexer_scale"):
            raise ValueError(f"KV region layout: unknown kind {self.kind!r}")
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("KV region layout: missing layout_id")
        if self.global_page_size != self.compression_ratio * self.page_slots:
            raise ValueError("KV region layout: slots do not tile a FULL page")
        if self.layout_id == KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1.value:
            if (
                self.kind != "kv"
                or self.compression_ratio not in (1, 2)
                or self.page_slots not in (128, 256)
                or self.page_bytes != self.page_slots * 384
            ):
                raise ValueError("KV region layout: invalid packed Main KV geometry")

    @property
    def identity(self):
        return self.kind, self.compression_ratio, self.source_layer_id


class KVTransferRegion(NamedTuple):
    buffer: object
    layout: KVRegionLayout


def parse_kv_region_layouts(value) -> tuple[KVRegionLayout, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("KV region layouts must be a list")
    try:
        regions = tuple(
            item if isinstance(item, KVRegionLayout) else KVRegionLayout(**item)
            for item in value
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("KV region layout: malformed descriptor") from exc
    if len({r.identity for r in regions}) != len(regions):
        raise ValueError("KV region layout: duplicate source region")
    return regions


def serialize_kv_region_layouts(regions):
    regions = parse_kv_region_layouts(regions)
    return None if regions is None else [asdict(region) for region in regions]


def validate_kv_region_registration(regions, item_lens, num_ptrs):
    regions = parse_kv_region_layouts(regions)
    if regions is None:
        return None
    if len(regions) != num_ptrs or len(item_lens) != num_ptrs:
        raise ValueError("KV region layout: descriptor/pointer/item_len count mismatch")
    if any(r.page_bytes != size for r, size in zip(regions, item_lens)):
        raise ValueError("KV region layout: page_bytes/item_len mismatch")
    return regions


def match_kv_region_layouts(source, destination):
    """Map a PP-local source list into a destination stage/full-model list.

    Unknown metadata is allowed only when both peers use the old protocol.
    Filtering the destination by source identity must preserve wire order.
    """
    source = parse_kv_region_layouts(source)
    destination = parse_kv_region_layouts(destination)
    if source is None and destination is None:
        return None
    if source is None or destination is None:
        raise ValueError("KV region layout metadata missing; upgrade both PD peers")
    dst = {r.identity: (i, r) for i, r in enumerate(destination)}
    pairs = []
    for i, region in enumerate(source):
        match = dst.get(region.identity)
        if match is None or match[1] != region:
            raise ValueError(f"KV region layout mismatch for {region.identity}")
        pairs.append((i, match[0]))
    if [j for _, j in pairs] != sorted(j for _, j in pairs):
        raise ValueError("KV region layout: source region order mismatch")
    return pairs


def kv_region_namespace(regions) -> str:
    encoded = json.dumps(
        serialize_kv_region_layouts(regions), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_kv_region_topology(local, remote_stages):
    """Bootstrap covers all P stages; only this D stage's sources must match."""
    local = parse_kv_region_layouts(local)
    if local is None and not remote_stages:
        return
    if local is None or not remote_stages:
        raise ValueError("KV region layout metadata missing; upgrade both PD peers")
    remote = {}
    for stage in remote_stages.values():
        for region in parse_kv_region_layouts(stage) or ():
            if region.identity in remote and remote[region.identity] != region:
                raise ValueError("KV region layout differs across prefill stages")
            remote[region.identity] = region
    for region in local:
        if remote.get(region.identity) != region:
            raise ValueError(f"KV region layout mismatch for {region.identity}")


def get_pool_transfer_info(pool):
    """Build pointers and their descriptors in one traversal."""
    if not getattr(pool, "has_kv_region_layouts", False):
        return (*pool.get_contiguous_buf_infos(), None)
    regions = pool.get_kv_transfer_regions()
    return (
        [r.buffer.data_ptr() for r in regions],
        [r.buffer.nbytes for r in regions],
        [r.layout.page_bytes for r in regions],
        serialize_kv_region_layouts([r.layout for r in regions]),
    )


def merge_draft_region_layouts(target, draft, num_draft_entries):
    # DeepSeek-V4.1 drafts have no Main KV entries. Keep unsupported mixed
    # representations explicit instead of publishing a partial pointer contract.
    if target is not None and num_draft_entries:
        raise ValueError("KV region layouts do not support draft Main KV entries")
    return target


def encode_kv_region_registration(args) -> bytes:
    return json.dumps(
        {
            "layouts": getattr(args, "kv_region_layouts", None),
            "item_lens": args.kv_item_lens,
        },
        separators=(",", ":"),
    ).encode()


def decode_kv_region_registration(data, num_ptrs):
    if not data:
        return None, None
    try:
        value = json.loads(data)
        layouts, item_lens = value["layouts"], value["item_lens"]
        validate_kv_region_registration(layouts, item_lens, num_ptrs)
    except (TypeError, KeyError) as exc:
        raise ValueError("KV region layout: malformed registration") from exc
    return layouts, item_lens
