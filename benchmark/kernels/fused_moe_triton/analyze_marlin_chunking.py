"""Offline Marlin scratch/padding model, not a GPU latency benchmark."""

import argparse
import ast
import collections
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import torch


def block_size(tokens, topk, experts):
    for block in (8, 16, 32, 48, 64):
        if tokens * topk / experts / block < 0.9:
            break
    return block


def routes(tokens, topk, experts, distribution, seed):
    rng = random.Random(seed)
    population = range(experts)
    if distribution == "hot":
        population = range(max(topk, experts // 16))
    return [rng.sample(population, topk) for _ in range(tokens)]


def routing_cost(ids, cap, block):
    padded = 0
    expert_visits = 0
    for start in range(0, len(ids), cap):
        counts = collections.Counter(
            expert for row in ids[start : start + cap] for expert in row
        )
        padded += sum(math.ceil(count / block) * block for count in counts.values())
        expert_visits += len(counts)
    return padded, expert_visits


def sizes(tokens, topk, hidden, intermediate):
    # Gated FP16/BF16 route-major caches in fused_marlin_moe.
    cache13 = tokens * topk * max(2 * intermediate, hidden) * 2
    cache2 = tokens * topk * intermediate * 2
    return cache13, cache2


def allocation_check(tokens, topk, hidden, intermediate, max_bytes):
    shape13 = (tokens * topk * max(2 * intermediate, hidden),)
    shape2 = (tokens * topk, intermediate)
    expected = sum(sizes(tokens, topk, hidden, intermediate))
    # Real CPU tensor storage accounting, without touching every page. This
    # measures requested tensor storage, not RSS, GPU allocator reservations
    # or a CUDA Graph pool. Larger shapes use metadata-only tensors.
    device = "cpu" if expected <= max_bytes else "meta"
    one = torch.empty(shape13, dtype=torch.bfloat16, device=device)
    two = torch.empty(shape2, dtype=torch.bfloat16, device=device)
    measured = one.untyped_storage().nbytes() + two.untyped_storage().nbytes()
    assert measured == expected
    return {"device": device, "storage_nbytes": measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--batches", type=int, nargs="+", default=[128, 1024, 8192])
    parser.add_argument("--caps", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bandwidth-tbps", type=float, default=3.0)
    parser.add_argument("--launch-us", type=float, default=3.0)
    parser.add_argument("--cpu-allocation-limit-mib", type=int, default=512)
    args = parser.parse_args()
    assert args.hidden > 0 and args.intermediate > 0
    assert 0 < args.topk <= args.experts and min(args.batches + args.caps) > 0
    assert args.bandwidth_tbps > 0 and args.launch_us >= 0
    root = Path(__file__).resolve().parents[3]
    source = root / "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if getattr(node, "name", "") == "fused_marlin_moe"
    )
    # Confirm this checkout contains the prototype; storage shapes below are
    # an explicit model and must be reviewed if the implementation changes.
    assert any(
        isinstance(node, ast.For) and ast.unparse(node.target) == "start"
        for node in ast.walk(function)
    )
    begin = time.perf_counter()
    rows = []
    # MXFP4 G32 payload + one-byte E8M0 scales for gated W13 and W2.
    expert_weight_bytes = 3 * args.hidden * args.intermediate * 17 // 32
    for m in args.batches:
        baseline_block = block_size(m, args.topk, args.experts)
        baseline_bytes = sum(sizes(m, args.topk, args.hidden, args.intermediate))
        baseline_allocation = allocation_check(
            m,
            args.topk,
            args.hidden,
            args.intermediate,
            args.cpu_allocation_limit_mib * 2**20,
        )
        for distribution in ("uniform", "hot"):
            ids = routes(m, args.topk, args.experts, distribution, args.seed)
            old_padded, old_visits = routing_cost(ids, m, baseline_block)
            for limit in args.caps:
                cap = min(m, limit)
                block = block_size(cap, args.topk, args.experts)
                padded, visits = routing_cost(ids, cap, block)
                chunks = math.ceil(m / cap)
                pair_bytes = sum(sizes(cap, args.topk, args.hidden, args.intermediate))
                # At least align + two GEMMs + activation + top-k sum per chunk.
                extra_core_launches = 5 * (chunks - 1)
                # Conservative chunk mode also clears every GEMM2 destination.
                extra_down_zero_bytes = (
                    m * args.topk * args.hidden * 2 if chunks > 1 else 0
                )
                rows.append(
                    {
                        "tokens": m,
                        "distribution": distribution,
                        "chunk_tokens": cap,
                        "chunks": chunks,
                        "baseline_block_m": baseline_block,
                        "candidate_block_m": block,
                        "baseline_pair_bytes": baseline_bytes,
                        "candidate_pair_bytes": pair_bytes,
                        "saved_pair_bytes": baseline_bytes - pair_bytes,
                        "baseline_padded_routes": old_padded,
                        "candidate_padded_routes": padded,
                        "baseline_expert_chunk_visits": old_visits,
                        "candidate_expert_chunk_visits": visits,
                        "expert_visit_ratio": visits / old_visits,
                        "extra_core_launches_model": extra_core_launches,
                        "extra_down_zero_bytes": extra_down_zero_bytes,
                        "launch_sensitivity_us": extra_core_launches * args.launch_us,
                        "zero_bandwidth_floor_us": extra_down_zero_bytes
                        / (args.bandwidth_tbps * 1e6),
                        "cold_weight_visit_bytes_extra": (visits - old_visits)
                        * expert_weight_bytes,
                        "baseline_allocation": baseline_allocation,
                        "candidate_allocation": allocation_check(
                            cap,
                            args.topk,
                            args.hidden,
                            args.intermediate,
                            args.cpu_allocation_limit_mib * 2**20,
                        ),
                    }
                )
    print(
        json.dumps(
            {
                "scope": "synthetic gated MoE; CPU routing simulation and storage accounting only",
                "args": vars(args),
                "source_commit": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "worktree_status": subprocess.check_output(
                    ["git", "-C", str(root), "status", "--short"], text=True
                ).splitlines(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cpu_analysis_seconds": time.perf_counter() - begin,
                "assumptions": [
                    "No model checkpoint is loaded; dimensions are illustrative, not a model benchmark",
                    "Pair bytes exclude weights, input/output, routing metadata, act-order a_tmp and FP32 c_tmp",
                    "Positive cap bounds the route-expanded activation pair, not total process memory",
                    "Weight visits assume one logical full expert weight read per visited chunk, not measured HBM traffic",
                    "Launch model is 5 core launches per chunk, excluding zeroing and routing fast-path transitions",
                    "Changing Marlin block_m and split scheduling may change GPU rounding and runtime",
                    "Bandwidth/launch parameters are sensitivity assumptions, not measured latency or speedup",
                ],
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
