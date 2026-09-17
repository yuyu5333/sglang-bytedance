"""Fixed-size scratch for the opt-in DSV4.1 staged Main-KV consumer."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

STAGING_QUERY_TILE = 64
STAGING_PAGE_SLOTS = 256


def staging_geometry(
    topk: int, query_tile: int = STAGING_QUERY_TILE
) -> tuple[int, int]:
    if topk <= 0 or query_tile <= 0:
        raise ValueError("staging topk and query tile must be positive")
    width = (topk + 63) // 64 * 64
    pages = (query_tile * width + STAGING_PAGE_SLOTS - 1) // STAGING_PAGE_SLOTS
    return width, pages


def staging_workspace_bytes(topk: int, query_tile: int = STAGING_QUERY_TILE) -> int:
    width, pages = staging_geometry(topk, query_tile)
    return pages * KVLayout.V4.page_bytes(STAGING_PAGE_SLOTS) + query_tile * width * 4


class MainKVStagingWorkspace:
    """One allocation per backend, shared by serial layers and query tiles.

    The owner must serialize calls that use this workspace, including graph
    replays. It never grows or memoizes converted data. A separate backend
    instance must be used for concurrently executing attention streams.
    """

    def __init__(
        self, device: torch.device, topk: int, query_tile: int = STAGING_QUERY_TILE
    ):
        self.query_tile = query_tile
        self.width, pages = staging_geometry(topk, query_tile)
        self.pages = torch.zeros(
            (pages, KVLayout.V4.page_bytes(STAGING_PAGE_SLOTS)),
            dtype=torch.uint8,
            device=device,
        )
        # This is an API view, not an AoS interpretation of the physical cache.
        # Explicit strides also preserve the padded page stride for a single page.
        self.cache = self.pages.as_strided(
            (pages, STAGING_PAGE_SLOTS, 1, 584),
            (self.pages.stride(0), 584, 584, 1),
        )
        self.indices = torch.empty(
            (query_tile, self.width), dtype=torch.int32, device=device
        )

    @property
    def nbytes(self) -> int:
        return self.pages.nbytes + self.indices.nbytes

    def check_shape(self, rows: int, width: int) -> None:
        if rows > self.query_tile or width > self.width:
            raise ValueError(
                f"staging shape {(rows, width)} exceeds reserved "
                f"{(self.query_tile, self.width)}; split queries before staging"
            )
