"""Offline shared MoE budget planning and storage accounting, never GPU timing."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from analyze_marlin_chunking import routes, routing_cost

from sglang.srt.layers.moe.workspace_policy import (
    MarlinWorkspaceEstimate,
    TritonWorkspaceEstimate,
    plan_workspace,
)


def storage_check(estimate, tokens, cpu_limit):
    topk, experts = estimate.topk, estimate.experts
    block = (
        estimate.block_m(tokens)
        if isinstance(estimate, MarlinWorkspaceEstimate)
        else estimate.block_m
    )
    count = tokens * topk
    capacity = (
        count * block if count < experts + 1 else count + (experts + 1) * (block - 1)
    )
    h, i = estimate.hidden, estimate.intermediate
    parts = []
    if isinstance(estimate, MarlinWorkspaceEstimate):
        width = max(2 * i, h)
        parts.extend(
            [
                ("gate_down", (count * width,), torch.bfloat16),
                ("activation", (count, i), torch.bfloat16),
                ("locks", (estimate.num_sms * 4,), torch.int32),
            ]
        )
        if estimate.fp32_reduce:
            reduction = min(width * capacity, estimate.num_sms * 4 * block * 256)
            if block == 8:
                reduction *= 2
            parts.append(("reduction_bound", (reduction,), torch.float32))
    else:
        parts.extend(
            [
                ("gate_up", (count, 2 * i), torch.bfloat16),
                ("activation", (count, i), torch.bfloat16),
                ("down", (tokens, topk, h), torch.bfloat16),
            ]
        )
    for generation in range(2):
        parts.extend(
            [
                (f"sorted_{generation}", (capacity,), torch.int32),
                (f"experts_{generation}", (math.ceil(capacity / block),), torch.int32),
                (f"padded_{generation}", (1,), torch.int32),
                (f"cumsum_{generation}", (experts + 2,), torch.int32),
            ]
        )
    device = "cpu" if estimate.peak_bytes(tokens) <= cpu_limit else "meta"
    tensors = [
        (name, torch.empty(shape, dtype=dtype, device=device))
        for name, shape, dtype in parts
    ]
    sizes = {name: tensor.untyped_storage().nbytes() for name, tensor in tensors}
    assert sum(sizes.values()) == estimate.peak_bytes(tokens)
    return {
        "device": device,
        "storage_nbytes": sum(sizes.values()),
        "components": sizes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--num-sms", type=int, default=78)
    parser.add_argument("--triton-block-m", type=int, default=64)
    parser.add_argument("--cpu-limit-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.hidden,
            args.intermediate,
            args.tokens,
            args.num_sms,
            args.triton_block_m,
            args.cpu_limit_mib,
            *args.budgets_mib,
        )
        <= 0
        or not 0 < args.topk <= args.experts
    ):
        parser.error("positive dimensions/budgets and 0 < topk <= experts required")
    dimensions = (args.hidden, args.intermediate, args.topk, args.experts)
    estimates = {
        "marlin_fp32_reduce": MarlinWorkspaceEstimate(*dimensions, args.num_sms, True),
        "marlin_atomic": MarlinWorkspaceEstimate(*dimensions, args.num_sms, False),
        "triton_unquantized": TritonWorkspaceEstimate(*dimensions, args.triton_block_m),
    }
    rows = []
    ids_by_distribution = {
        name: routes(args.tokens, args.topk, args.experts, name, args.seed)
        for name in ("uniform", "hot")
    }
    for backend, estimate in estimates.items():
        before = storage_check(estimate, args.tokens, args.cpu_limit_mib * 2**20)
        for budget in args.budgets_mib:
            try:
                plan = plan_workspace(args.tokens, budget * 2**20, estimate)
            except ValueError as error:
                rows.append(
                    {"backend": backend, "budget_mib": budget, "error": str(error)}
                )
                continue
            after = storage_check(
                estimate, plan.chunk_tokens, args.cpu_limit_mib * 2**20
            )
            old_block = (
                estimate.block_m(args.tokens)
                if isinstance(estimate, MarlinWorkspaceEstimate)
                else estimate.block_m
            )
            new_block = (
                estimate.block_m(plan.chunk_tokens)
                if isinstance(estimate, MarlinWorkspaceEstimate)
                else estimate.block_m
            )
            costs = {}
            for name, ids in ids_by_distribution.items():
                old_pad, old_visits = routing_cost(ids, args.tokens, old_block)
                new_pad, new_visits = routing_cost(ids, plan.chunk_tokens, new_block)
                costs[name] = {
                    "full_padded_routes": old_pad,
                    "chunk_padded_routes": new_pad,
                    "full_expert_visits": old_visits,
                    "chunk_expert_visits": new_visits,
                }
            rows.append(
                {
                    "backend": backend,
                    "plan": asdict(plan),
                    "estimate_inputs": asdict(estimate),
                    "chunks": math.ceil(args.tokens / plan.chunk_tokens),
                    "full_block_m": old_block,
                    "chunk_block_m": new_block,
                    "full_storage_model": before,
                    "chunk_storage_model": after,
                    "routing_simulation": costs,
                }
            )
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        "python/sglang/srt/environ.py",
        "python/sglang/srt/layers/moe/workspace_policy.py",
        "python/sglang/srt/layers/moe/moe_runner/workspace.py",
        "python/sglang/srt/layers/moe/moe_runner/base.py",
        "python/sglang/srt/layers/moe/moe_runner/runner.py",
        "python/sglang/srt/layers/moe/moe_runner/marlin.py",
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py",
        "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py",
        "benchmark/kernels/fused_moe_triton/analyze_moe_workspace.py",
        "benchmark/kernels/fused_moe_triton/analyze_marlin_chunking.py",
    ]
    result = {
        "scope": "Synthetic local MoE scratch estimates and CPU/meta storage only",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "-C", str(root), "status", "--short"], text=True
        ).splitlines(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_paths
        },
        "torch": torch.__version__,
        "platform": platform.platform(),
        "assumptions": [
            "No checkpoint, request concurrency, KV dtype, TP/PP/DP or speculative decoding",
            "Triton block-M and Marlin SM count are supplied assumptions, not probed GPU settings",
            "Do not compare different backend/weight formats as a speed or accuracy A/B",
            "Storage checks materialize the conservative formula, not the runtime allocation trace",
            "CPU empty tensors do not touch every page; meta tensors allocate no data",
            "Budget excludes weights, caller tensors, full output, dispatch, allocator reserved memory, JIT and graph pools",
            "Two routing generations conservatively include tuple replacement and int64 activation-id normalization",
            "Sequential execution and call-private scratch are not GPU graph or concurrency validation",
            "Padding and expert visits are modeled work, not GPU latency or HBM counters",
        ],
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Wrote {len(rows)} budget plans to {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
