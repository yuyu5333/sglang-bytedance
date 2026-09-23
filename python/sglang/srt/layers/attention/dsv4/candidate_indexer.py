from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, NamedTuple, Optional, Union

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )


class CandidateMetadata:
    """Base of an implementation's published state on
    ``DSV4Metadata.candidate_metadata``."""


class _CandidateTopK(NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor
    fused_publication: bool


def _get_deepselect_topk(
    scores: torch.Tensor,
) -> Optional[Callable[..., tuple[Optional[torch.Tensor], torch.Tensor]]]:
    if not scores.is_cuda:
        return None
    try:
        from sglang.kernels.ops.deep_select import (
            is_deepselect_supported,
            topk,
        )
    except (ImportError, OSError):
        return None
    return topk if is_deepselect_supported(scores.device) else None


def _can_fuse_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    block_size: int,
) -> bool:
    return (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.stride(1) == 1
        and torch.is_tensor(compress_lens)
        and compress_lens.device == logits.device
        and compress_lens.dtype in (torch.int32, torch.int64)
        and compress_lens.is_contiguous()
        and compress_lens.numel() == logits.shape[0]
        and 0 < block_size <= 1024
    )


@dataclass(frozen=True)
class IndexerInputs:
    """One index-source layer's operands on the paged fp4 decode path (one query
    row per request, or per draft token under verify)."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    metadata: PagedIndexerMetadata  # this ratio's lengths, page table and plans
    # [rows] int, one request id per query row, the rows of one request
    # consecutive (verify: its draft tokens); None = every row its own request
    request_ids: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


def make_candidate_indexer(
    topk_blocks: int, block_size: int
) -> Optional[DeepGemmCandidateIndexer]:
    """The paged fp4 decode path's two-level indexer; None on Hopper, whose decode
    indexer selects through masks inline."""
    if topk_blocks <= 0 or get_platform().device_sm < 100:
        return None
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )

    return DeepGemmCandidateIndexer(topk_blocks, block_size)


# TODO(candidate): Hopper decode and prefill still select through these masks
# inline in the backend; move them behind the protocol as publish/select_prefill.
@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each


class PrefillCandidateBlocks(CandidateMetadata, msgspec.Struct):
    request_blocks: List[torch.Tensor]

    def tail(self, lengths: List[int]) -> PrefillCandidateBlocks:
        return PrefillCandidateBlocks(
            request_blocks=[
                blocks[blocks.shape[0] - length :]
                for blocks, length in zip(self.request_blocks, lengths)
            ]
        )


def published_masks(candidate) -> CandidateMasks:
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def _candidate_block_topk(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> _CandidateTopK:
    width = logits.size(-1)
    num_blocks = (width + block_size - 1) // block_size
    selected = min(topk_blocks, num_blocks)
    deepselect_topk = _get_deepselect_topk(logits) if selected > 0 else None
    if deepselect_topk is not None and _can_fuse_candidate_blocks(
        logits, compress_lens, block_size
    ):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_scores,
        )

        scores = candidate_block_scores(
            logits, compress_lens.reshape(-1), block_size=block_size
        )
        values, indices = deepselect_topk(
            scores, selected, indices_type=torch.int32
        )
        assert values is not None
        return _CandidateTopK(
            values=values, indices=indices, fused_publication=True
        )

    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    if deepselect_topk is not None:
        values, indices = deepselect_topk(
            scores, selected, indices_type=torch.int32
        )
        assert values is not None
        return _CandidateTopK(
            values=values, indices=indices, fused_publication=False
        )

    top = scores.topk(selected, dim=-1)
    return _CandidateTopK(
        values=top.values, indices=top.indices, fused_publication=False
    )


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    top = _candidate_block_topk(
        logits=logits,
        compress_lens=compress_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


def candidate_block_mask(
    blocks: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*blocks.shape[:-1], num_blocks + 1), dtype=torch.bool, device=blocks.device
    )
    keep.scatter_(-1, blocks.to(torch.int64).masked_fill(blocks < 0, num_blocks), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    top = _candidate_block_topk(
        logits=logits,
        compress_lens=compress_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    if top.fused_publication:
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            publish_candidate_block_mask,
        )

        return publish_candidate_block_mask(
            indices=top.indices,
            values=top.values,
            width=logits.shape[-1],
            block_size=block_size,
        )
    blocks = top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)
    return candidate_block_mask(
        blocks=blocks, width=logits.shape[-1], block_size=block_size
    )
