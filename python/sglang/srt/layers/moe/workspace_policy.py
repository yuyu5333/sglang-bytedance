"""CPU-only planning for call-private MoE scratch, not total GPU memory."""

from dataclasses import dataclass
from typing import Protocol


class WorkspaceEstimate(Protocol):
    def peak_bytes(self, tokens: int) -> int: ...


@dataclass(frozen=True)
class WorkspacePlan:
    tokens: int
    chunk_tokens: int
    budget_bytes: int
    estimated_peak_bytes: int


def plan_workspace(
    tokens: int, budget_bytes: int, estimate: WorkspaceEstimate
) -> WorkspacePlan:
    """Pick the largest fitting bucket without assuming monotone backend costs.

    Try the complete batch first, then descending powers of two. Fixed buckets
    keep capture shapes independent of GPU routing values or free-memory queries.
    The budget excludes weights, caller-owned tensors and the full output.
    """
    if tokens < 0 or budget_bytes <= 0:
        raise ValueError("tokens must be nonnegative and budget_bytes positive")
    if tokens == 0:
        return WorkspacePlan(0, 0, budget_bytes, 0)
    candidates = [tokens]
    bucket = 1 << (tokens.bit_length() - 1)
    while bucket:
        if bucket != tokens:
            candidates.append(bucket)
        bucket //= 2
    for chunk in candidates:
        peak = estimate.peak_bytes(chunk)
        if peak < 0:
            raise ValueError("backend scratch estimate must be nonnegative")
        if peak <= budget_bytes:
            return WorkspacePlan(tokens, chunk, budget_bytes, peak)
    raise ValueError(
        f"MoE workspace budget {budget_bytes} bytes cannot fit any supported "
        f"chunk bucket (one-token estimate: {estimate.peak_bytes(1)} bytes)"
    )


def _routing_bytes(tokens: int, topk: int, experts: int, block: int) -> int:
    routes = tokens * topk
    capacity = (
        routes * block if routes < experts + 1 else routes + (experts + 1) * (block - 1)
    )
    return 4 * (capacity + (capacity + block - 1) // block + experts + 3)


@dataclass(frozen=True)
class MarlinWorkspaceEstimate:
    hidden: int
    intermediate: int
    topk: int
    experts: int
    num_sms: int
    fp32_reduce: bool

    def block_m(self, tokens: int) -> int:
        for block in (8, 16, 32, 48, 64):
            if tokens * self.topk / self.experts / block < 0.9:
                break
        return block

    def peak_bytes(self, tokens: int) -> int:
        if tokens == 0:
            return 0
        block = self.block_m(tokens)
        routes = tokens * self.topk
        capacity = (
            routes * block
            if routes < self.experts + 1
            else routes + (self.experts + 1) * (block - 1)
        )
        width = max(2 * self.intermediate, self.hidden)
        activations = 2 * routes * (width + self.intermediate)
        locks = self.num_sms * 4 * 4
        reduction = 0
        if self.fp32_reduce:
            reduction = 4 * min(width * capacity, self.num_sms * 4 * block * 256)
            if block == 8:
                reduction *= 2
        # Old/new alignment tensors can coexist while Python assigns the tuple.
        return (
            activations
            + locks
            + reduction
            + 2 * _routing_bytes(tokens, self.topk, self.experts, block)
        )


@dataclass(frozen=True)
class TritonWorkspaceEstimate:
    hidden: int
    intermediate: int
    topk: int
    experts: int
    block_m: int

    def peak_bytes(self, tokens: int) -> int:
        if tokens == 0:
            return 0
        activations = 2 * tokens * self.topk * (3 * self.intermediate + self.hidden)
        return activations + 2 * _routing_bytes(
            tokens, self.topk, self.experts, self.block_m
        )
