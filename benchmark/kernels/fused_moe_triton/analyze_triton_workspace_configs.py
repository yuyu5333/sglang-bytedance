"""Compare fixed/candidate config policies with explicit offline config snapshots."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from analyze_marlin_chunking import routes, routing_cost
from analyze_moe_workspace import storage_check

from sglang.srt.layers.moe.moe_runner import triton_utils
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe,
    fused_moe_triton_config,
)
from sglang.srt.layers.moe.moe_runner.workspace import (
    _TritonCandidateEstimate,
    _UnsupportedTritonWorkspace,
)
from sglang.srt.layers.moe.workspace_policy import (
    TritonWorkspaceEstimate,
    plan_workspace,
)
from sglang.srt.runtime_context import get_context


def read_config(path):
    if path is None:
        return None
    values = json.loads(path.read_text())
    return {int(key): value for key, value in values.items()}


def evaluate_policy(args, budget, candidate, x, w1, w2, ids, routing):
    resolved = []

    def resolve(*inputs, **kwargs):
        result = fused_moe._resolve_fused_moe_config(*inputs, **kwargs)
        launch, down, down_tma, up_tma = result
        resolved.append(
            {
                "tokens": kwargs.get("num_tokens", args.tokens),
                "up": launch,
                "down": down,
                "up_tma": up_tma,
                "down_tma": down_tma,
            }
        )
        return result

    try:
        if candidate:
            estimates = _TritonCandidateEstimate(x, w1, w2, ids, resolve)
            plan = plan_workspace(args.tokens, budget, estimates)
            launch, down, estimate = estimates.candidates[plan.chunk_tokens]
        else:
            launch, down, down_tma, up_tma = resolve(
                x,
                w1,
                w2,
                ids,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )
            if down_tma or up_tma:
                raise _UnsupportedTritonWorkspace("TMA layout for full batch")
            estimate = TritonWorkspaceEstimate(
                args.hidden,
                args.intermediate,
                args.topk,
                args.experts,
                launch["BLOCK_SIZE_M"],
            )
            plan = plan_workspace(args.tokens, budget, estimate)
    except _UnsupportedTritonWorkspace as error:
        return {"status": "unsupported", "reason": str(error), "resolved": resolved}
    except ValueError as error:
        if not str(error).startswith("MoE workspace budget"):
            raise
        return {"status": "budget_error", "reason": str(error), "resolved": resolved}

    costs = {}
    for distribution, token_routes in routing.items():
        padded, visits = routing_cost(token_routes, plan.chunk_tokens, estimate.block_m)
        costs[distribution] = {
            "padded_routes": padded,
            "expert_chunk_visits": visits,
        }
    return {
        "status": "planned",
        "plan": asdict(plan),
        "chunks": math.ceil(args.tokens / plan.chunk_tokens),
        "block_m": estimate.block_m,
        "up_config": launch,
        "down_config": down,
        "resolved": resolved,
        "storage_check": storage_check(
            estimate, plan.chunk_tokens, args.cpu_limit_mib * 2**20
        ),
        "routing_simulation": costs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--up-config", type=Path)
    parser.add_argument("--down-config", type=Path)
    parser.add_argument(
        "--tma-supported", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cpu-limit-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.tokens,
            args.hidden,
            args.intermediate,
            args.experts,
            args.topk,
            args.cpu_limit_mib,
            *args.budgets_mib,
        )
        <= 0
        or args.topk > args.experts
        or (args.down_config is not None and args.up_config is None)
    ):
        parser.error(
            "positive dimensions/budgets, topk <= experts and up before down required"
        )
    up = read_config(args.up_config)
    down = read_config(args.down_config) if args.down_config else up
    routing = {
        name: routes(args.tokens, args.topk, args.experts, name, args.seed)
        for name in ("uniform", "hot")
    }
    x = torch.empty(args.tokens, args.hidden, dtype=torch.bfloat16, device="meta")
    w1 = torch.empty(
        args.experts, 2 * args.intermediate, args.hidden, dtype=x.dtype, device="meta"
    )
    w2 = torch.empty(
        args.experts, args.hidden, args.intermediate, dtype=x.dtype, device="meta"
    )
    ids = torch.empty(args.tokens, args.topk, dtype=torch.int32, device="meta")

    # Replace only environment discovery: keep real config selection, planning,
    # TMA rejection and storage estimates. No CUDA operation is executed.
    with (
        get_context().override_server_args(enable_deterministic_inference=False),
        patch.object(triton_utils, "get_config", return_value=None),
        patch.object(
            fused_moe_triton_config,
            "get_moe_configs",
            side_effect=lambda *a, down_moe=False, **kw: down if down_moe else up,
        ),
        patch.object(fused_moe, "_moe_support_tma", return_value=args.tma_supported),
    ):
        rows = [
            {
                "budget_bytes": mib * 2**20,
                "fixed_full_config": evaluate_policy(
                    args, mib * 2**20, False, x, w1, w2, ids, routing
                ),
                "candidate_config": evaluate_policy(
                    args, mib * 2**20, True, x, w1, w2, ids, routing
                ),
            }
            for mib in args.budgets_mib
        ]
    root = Path(__file__).resolve().parents[3]
    sources = [
        "python/sglang/srt/layers/moe/workspace_policy.py",
        "python/sglang/srt/layers/moe/moe_runner/workspace.py",
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py",
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py",
        "benchmark/kernels/fused_moe_triton/analyze_marlin_chunking.py",
        "benchmark/kernels/fused_moe_triton/analyze_moe_workspace.py",
        "benchmark/kernels/fused_moe_triton/analyze_triton_workspace_configs.py",
    ]
    result = {
        "scope": "Offline configuration policy and CPU routing/storage simulation",
        "baseline_policy": "9af19d5b5d full-batch configuration held for all chunks",
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
            for name in sources
        },
        "config_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.up_config, args.down_config)
            if path is not None
        },
        "torch": torch.__version__,
        "platform": platform.platform(),
        "assumptions": [
            "No checkpoint, GPU execution, KV dtype, TP/PP/DP, request workload or model accuracy",
            "Explicit config snapshots bypass device/version discovery, not a live machine configuration",
            "Without config files the real default heuristic is used",
            "Up-only snapshots reuse that map for down, with all flags preserved",
            "TMA capability is an explicit assumption; TMA configs are never stripped to obtain a result",
            "Inputs/weights are metadata-only; storage checks materialize the formula on CPU/meta",
            "Kernel tiling, reduction order, JIT compile and latency are not measured",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(rows)} policy comparisons to {args.output}")


if __name__ == "__main__":
    main()
