"""Candidate block reduction, publication, and sparse-row metadata."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

_DEEPSELECT_INPUT_ALIGNMENT_BYTES = 1024


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_block_scores_kernel(
    LOGITS,
    LENS,
    SCORES,
    WIDTH: tl.constexpr,
    LOGIT_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, BLOCK_PAD)
    cols = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    length = tl.load(LENS + row)
    in_bounds = (cols < WIDTH) & (offsets[None, :] < BLOCK_SIZE)
    values = tl.load(
        LOGITS + row * LOGIT_STRIDE + cols,
        in_bounds & (cols < length),
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)
    scores = tl.where(
        (length > 0) & (blocks == (length - 1) // BLOCK_SIZE),
        float("inf"),
        scores,
    )
    tl.store(SCORES + row * SCORE_STRIDE + blocks, scores, blocks < SCORE_STRIDE)


@triton.jit
def _publish_candidate_block_mask_kernel(
    INDICES,
    VALUES,
    KEEP,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    in_topk = i < TOPK * BLOCK_SIZE
    selected = tl.load(INDICES + row * INDEX_STRIDE + i // BLOCK_SIZE, in_topk, -1)
    score = tl.load(
        VALUES + row * VALUE_STRIDE + i // BLOCK_SIZE,
        in_topk,
        -float("inf"),
    )
    cols = selected * BLOCK_SIZE + i % BLOCK_SIZE
    valid = (
        in_topk
        & (selected >= 0)
        & (cols >= 0)
        & (cols < WIDTH)
        & (score > -float("inf"))
    )
    # Top-K block indices are unique, so each valid token has one writer.
    tl.store(KEEP + row * WIDTH + cols, True, valid)


def candidate_block_scores(
    logits: torch.Tensor, seq_lens: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Reduce visible token logits into 1024-byte aligned FP32 block-score rows."""
    assert logits.dim() == 2 and logits.is_cuda and logits.stride(1) == 1
    assert logits.dtype == torch.float32
    assert seq_lens.dim() == 1 and seq_lens.is_contiguous()
    assert seq_lens.device == logits.device
    assert 0 < block_size <= 1024
    rows, width = logits.shape
    blocks = triton.cdiv(width, block_size)
    alignment = _DEEPSELECT_INPUT_ALIGNMENT_BYTES // torch.float32.itemsize
    score_stride = triton.cdiv(blocks, alignment) * alignment
    scores = torch.empty(
        (rows, score_stride), dtype=torch.float32, device=logits.device
    )
    block_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // block_pad)
    _candidate_block_scores_kernel[(rows, triton.cdiv(score_stride, tile))](
        logits,
        seq_lens,
        scores,
        width,
        logits.stride(0),
        scores.stride(0),
        block_size,
        block_pad,
        tile,
    )
    return scores


def publish_candidate_block_mask(
    indices: torch.Tensor,
    values: torch.Tensor,
    width: int,
    block_size: int,
) -> torch.Tensor:
    """Publish selected block IDs directly as a token-level visibility mask."""
    assert indices.dim() == 2 and values.shape == indices.shape
    assert indices.is_cuda and values.device == indices.device
    rows, topk = indices.shape
    keep = torch.zeros((rows, width), dtype=torch.bool, device=indices.device)
    if rows == 0 or width == 0 or topk == 0:
        return keep
    _publish_candidate_block_mask_kernel[(rows, triton.cdiv(topk * block_size, 256))](
        indices,
        values,
        keep,
        width,
        block_size,
        topk,
        indices.stride(0),
        values.stride(0),
        256,
        num_warps=4,
    )
    return keep


@triton.jit
def _candidate_row_lens_kernel(
    LENS,
    NBLOCKS,
    VALID,
    ROWS,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = rows < ROWS
    if USE_PDL:
        tl.extra.cuda.gdc_wait()  # LENS is the previous kernel's output
    length = tl.load(LENS + rows, mask, 0).to(tl.int32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    nblocks = (length + (BLOCK - 1)) // BLOCK
    kept = tl.minimum(nblocks, TOPK)
    # the kept blocks laid out back to back, the newest one possibly partial
    valid = BLOCK * (kept - 1) + (length - 1) % BLOCK + 1
    valid = tl.where(length > 0, valid, 0)
    tl.store(NBLOCKS + rows, nblocks, mask)
    tl.store(VALID + rows, valid, mask)


def candidate_row_lens(
    seq_lens: torch.Tensor, topk_blocks: int, block_size: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: its number of blocks ``ceil(seq_len / block_size)`` and the
    length of its sparse logits row once the ``min(topk_blocks, blocks)`` kept
    blocks are laid out back to back (the newest block possibly partial):
    ``block_size * (kept - 1) + (seq_len - 1) % block_size + 1``. Both int32
    ``[rows]``; a zero-length row gets 0 for both."""
    assert seq_lens.dim() == 1 and seq_lens.is_contiguous()
    rows = seq_lens.numel()
    nblocks = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    valid = torch.empty_like(nblocks)
    tile = 256
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _candidate_row_lens_kernel[(triton.cdiv(rows, tile),)](
        seq_lens,
        nblocks,
        valid,
        rows,
        topk_blocks,
        block_size,
        tile,
        use_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    return nblocks, valid
