"""Fresh single-GPU correctness and eager/graph A/B for direct MoE decode."""

import argparse
import contextlib
import hashlib
import json
import os
import random
import statistics
import subprocess
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from validate_workspace_cuda import (
    error_metrics,
    latency,
    make_weights,
    memory_probe,
    reference,
)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return graph, output


def validate(args, report):
    from sglang.kernels.ops.moe.direct_decode import direct_decode
    from sglang.kernels.ops.moe.grouped_decode import grouped_decode
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    dtype = getattr(torch, args.dtype)
    quant, ref_weights = make_weights(args, dtype)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=dtype) / 2
    original = x.clone()
    logits = torch.randn(args.tokens, args.experts, device="cuda")
    if args.routing == "hot":
        logits[:, args.topk :] = -100
    values, ids = logits.topk(args.topk, -1)
    ids = ids.to(getattr(torch, args.ids_dtype))
    weights = values.softmax(-1)
    if args.duplicate_experts and args.topk > 1:
        ids[:, -1] = ids[:, 0]
    if args.masked:
        ids[-1, -1] = -1
    if args.masked_token:
        ids[-1] = -1
    if args.all_masked:
        ids.fill_(-1)
    report["initial_ids"] = ids.cpu().tolist()
    cfg = MoeRunnerConfig(
        num_experts=args.experts
        * (2 if args.masked or args.masked_token or args.all_masked else 1),
        num_local_experts=args.experts,
        hidden_size=args.hidden,
        intermediate_size_per_partition=args.intermediate,
        top_k=args.topk,
        activation="silu",
        is_gated=True,
        inplace=args.inplace,
        routed_scaling_factor=args.scale,
        workspace_budget_bytes=0,
    )
    dispatch = StandardDispatchOutput(x, None, StandardTopKOutput(weights, ids, logits))
    runner = MoeRunner(MoeRunnerBackend.TRITON, cfg)

    def baseline():
        if args.inplace:
            x.copy_(original)
        with envs.SGLANG_MOE_DIRECT_DECODE.override(False):
            return runner.run(dispatch, quant).hidden_states

    def direct():
        if args.inplace:
            x.copy_(original)
        if args.runner_direct:
            with envs.SGLANG_MOE_DIRECT_DECODE.override(True):
                return runner.run(dispatch, quant).hidden_states
        return direct_decode(
            x,
            quant.w13_weight,
            quant.w2_weight,
            ids,
            weights,
            b1=quant.b13,
            b2=quant.b2,
            scale=args.scale,
            inplace=args.inplace,
        )

    def grouped():
        if args.inplace:
            x.copy_(original)
        return grouped_decode(
            x,
            quant.w13_weight,
            quant.w2_weight,
            ids,
            weights,
            b1=quant.b13,
            b2=quant.b2,
            scale=args.scale,
            inplace=args.inplace,
            block_n=args.block_n,
            block_k=args.block_k,
        )

    functions = {} if args.grouped_only else {"direct": direct}
    if args.grouped or args.grouped_only:
        functions["grouped"] = grouped
    if not args.direct_only and not args.grouped_only:
        functions["baseline"] = baseline
    context = (
        override_config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }
        )
        if args.config == "fixed"
        else contextlib.nullcontext()
    )
    report["cases"] = cases = {}
    expected = reference(original, ids, weights, ref_weights, "silu", args.scale)
    with context, torch.inference_mode():
        outputs = {}
        for name, fn in functions.items():
            if name == "direct" and args.runner_direct:
                with patch(
                    "sglang.kernels.ops.moe.direct_decode.direct_decode",
                    wraps=direct_decode,
                ) as observed:
                    output = fn()
                report["runner_direct_calls"] = observed.call_count
                assert observed.call_count == (0 if args.expect_fallback else 1)
            else:
                output = fn()
            torch.cuda.synchronize()
            outputs[name] = output.clone()
            cases[name] = {
                "reference": error_metrics(output, expected, args.atol, args.rtol),
            }
            print(name, cases[name]["reference"], flush=True)
            for _ in range(args.warmup):
                fn()
            cases[name]["memory"] = memory_probe(fn)
        if "baseline" in outputs:
            report["direct_vs_baseline"] = error_metrics(
                outputs["direct"], outputs["baseline"], args.atol, args.rtol
            )
            if "grouped" in outputs:
                report["grouped_vs_baseline"] = error_metrics(
                    outputs["grouped"], outputs["baseline"], args.atol, args.rtol
                )

        graphs = {name: capture(fn) for name, fn in functions.items()}
        saved_ids, saved_weights = ids.clone(), weights.clone()
        if args.grouped or args.grouped_only:
            changed_ids = (
                torch.arange(args.tokens, device="cuda")[:, None] * (args.topk + 1)
                + torch.arange(args.topk, device="cuda")[None, :]
                + 1
            ) % args.experts
            if args.routing == "uniform":
                changed_ids[:] = changed_ids[0].clone()
            if args.tokens > 1 and (
                args.masked or args.masked_token or args.all_masked
            ):
                changed_ids[0] = -1
            ids.copy_(changed_ids)
        else:
            ids.copy_(torch.where(ids < 0, ids, (ids + 1) % args.experts))
        weights.mul_(0.7)
        report["changed_ids"] = ids.cpu().tolist()
        changed = reference(original, ids, weights, ref_weights, "silu", args.scale)
        for name, (graph, output) in graphs.items():
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            cases[name]["changed_routing_graph"] = error_metrics(
                output, changed, args.atol, args.rtol
            )
        ids.copy_(saved_ids)
        weights.copy_(saved_weights)
        if not args.inplace:
            for name, fn in functions.items():
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                outputs = []
                for stream in streams:
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        outputs.append(fn())
                torch.cuda.synchronize()
                cases[name]["two_streams"] = [
                    error_metrics(out, expected, args.atol, args.rtol)
                    for out in outputs
                ]

        if not args.skip_timing:
            rng = random.Random(args.seed)
            for mode in ("eager", "graph"):
                for _ in range(args.repeats):
                    order = list(functions)
                    rng.shuffle(order)
                    for name in order:
                        fn = (
                            functions[name]
                            if mode == "eager"
                            else graphs[name][0].replay
                        )
                        timing = latency(fn, args.iterations)
                        cases[name].setdefault(mode, []).append(timing)
                for name in functions:
                    cases[name][mode + "_median_ms"] = statistics.median(
                        t["cuda_ms"] for t in cases[name][mode]
                    )
                    print(name, mode, cases[name][mode + "_median_ms"], flush=True)
        if args.profile:
            for name, fn in functions.items():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as prof:
                    fn()
                    torch.cuda.synchronize()
                target = args.output.with_name(args.output.stem + f"-{name}-trace.json")
                prof.export_chrome_trace(str(target))
                report.setdefault("profiles", {})[name] = str(target)
        report["pass"] = all(
            report.get(key, {"pass": True})["pass"]
            for key in ("direct_vs_baseline", "grouped_vs_baseline")
        ) and all(
            case["reference"]["pass"]
            and case["changed_routing_graph"]["pass"]
            and all(item["pass"] for item in case.get("two_streams", []))
            for case in cases.values()
        )
    destroy_model_parallel()
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--ids-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--routing", choices=["uniform", "hot"], default="uniform")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--masked-token", action="store_true")
    parser.add_argument("--all-masked", action="store_true")
    parser.add_argument("--duplicate-experts", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--config", choices=["runtime", "fixed"], default="runtime")
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--grouped-only", action="store_true")
    parser.add_argument("--block-n", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--block-k", type=int, choices=[32, 64, 128, 256], default=64)
    parser.add_argument("--runner-direct", action="store_true")
    parser.add_argument("--expect-fallback", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=29861)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.backend = "triton"
    if (
        min(
            args.tokens,
            args.hidden,
            args.intermediate,
            args.experts,
            args.topk,
            args.iterations,
            args.repeats,
        )
        <= 0
    ):
        parser.error("dimensions and timing counts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    import triton

    import sglang
    from sglang.kernels.ops.moe import direct_decode, grouped_decode

    report = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "-C", str(root), "status", "--short"], text=True
        ).splitlines(),
        "sglang": sglang.__file__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "source_sha256": hashlib.sha256(
            Path(direct_decode.__file__).read_bytes()
        ).hexdigest(),
        "grouped_source_sha256": hashlib.sha256(
            Path(grouped_decode.__file__).read_bytes()
        ).hexdigest(),
        "scope": "Synthetic BF16/FP16 MoE; TP/EP/PP/DP=1; no checkpoint/KV/HTTP/speculative/mem-fraction workload",
        "pass": False,
    }
    try:
        validate(args, report)
    except Exception:
        report["exception"] = traceback.format_exc()
        print(report["exception"], flush=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(args.output, "pass=", report["pass"], flush=True)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
