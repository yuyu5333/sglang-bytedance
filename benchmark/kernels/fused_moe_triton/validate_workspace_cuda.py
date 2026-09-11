"""Single-GPU MoE workspace correctness, memory and latency validation."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F


def make_weights(args, dtype):
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    e, h, i = args.experts, args.hidden, args.intermediate
    b1 = torch.randn(e, 2 * i, device="cuda", dtype=dtype) / 100 if args.bias else None
    b2 = torch.randn(e, h, device="cuda", dtype=dtype) / 100 if args.bias else None
    if args.backend == "triton":
        w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=dtype) / math.sqrt(h)
        w2 = torch.randn(e, h, i, device="cuda", dtype=dtype) / math.sqrt(i)
        return TritonMoeQuantInfo(w1, w2, b13=b1, b2=b2), (w1, w2, b1, b2)
    if args.backend == "marlin-mxfp4":
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def packed(n, k):
            value = torch.randint(256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
            scale = 2.0 ** math.floor(math.log2(1 / math.sqrt(k) / 3))
            ref = torch.stack(
                (table[(value & 15).long()], table[(value >> 4).long()]), -1
            )
            ref = (ref.reshape(e, n, k) * scale).to(dtype)
            scales = torch.full((e, n, k // 32), scale, dtype=dtype, device="cuda")
            return value, scales, ref

        p1, s1, w1 = packed(2 * i, h)
        p2, s2, w2 = packed(h, i)
        layer = torch.nn.Module()
        layer.orig_dtype = dtype
        for name, value in (
            ("w13_weight", p1),
            ("w2_weight", p2),
            ("w13_weight_scale", s1),
            ("w2_weight_scale", s2),
            ("w13_weight_bias", b1),
            ("w2_weight_bias", b2),
        ):
            if value is not None:
                setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
        prepare_moe_mxfp4_layer_for_marlin(layer)
        quant = MarlinMoeQuantInfo(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            None,
            None,
            4,
            w13_bias=getattr(layer, "w13_weight_bias", None),
            w2_bias=getattr(layer, "w2_weight_bias", None),
        )
        return quant, (w1, w2, b1, b2)

    from sgl_kernel.scalar_type import scalar_types

    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_bias
    from sglang.test.test_marlin_utils import marlin_quantize

    bits = 4 if args.backend == "marlin-int4" else 8
    scalar = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128

    def quantized(n, k):
        refs, weights, scales = [], [], []
        for _ in range(e):
            w = torch.randn(k, n, device="cuda", dtype=dtype) / math.sqrt(k)
            ref, packed, scale, _, _, _ = marlin_quantize(
                w, scalar, args.group_size, False
            )
            refs.append(ref.T.contiguous())
            weights.append(packed)
            scales.append(scale)
        return torch.stack(refs), torch.stack(weights), torch.stack(scales)

    w1, p1, s1 = quantized(2 * i, h)
    w2, p2, s2 = quantized(h, i)
    quant = MarlinMoeQuantInfo(
        p1,
        p2,
        s1,
        s2,
        None,
        None,
        bits,
        w13_bias=torch.stack([marlin_permute_bias(b) for b in b1])
        if b1 is not None
        else None,
        w2_bias=torch.stack([marlin_permute_bias(b) for b in b2])
        if b2 is not None
        else None,
    )
    return quant, (w1, w2, b1, b2)


def reference(x, ids, weights, tensors, activation, scale):
    w1, w2, b1, b2 = tensors
    routes = torch.zeros((*ids.shape, x.shape[1]), device=x.device, dtype=x.dtype)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        gate_up = x[token].float() @ w1[expert].float().T
        if b1 is not None:
            gate_up += b1[expert].float()
        gate, up = gate_up.to(x.dtype).float().chunk(2, -1)
        act = F.silu(gate) if activation == "silu" else F.gelu(gate)
        down = (act * up).to(x.dtype).float() @ w2[expert].float().T
        if b2 is not None:
            down += b2[expert].float()
        routes[token, slot] = (down * weights[token, slot, None]).to(x.dtype)
    return (routes.float().sum(1) * scale).to(x.dtype)


def error_metrics(actual, expected, atol, rtol):
    diff = (actual.float() - expected.float()).abs()
    finite = bool(torch.isfinite(actual).all())
    return {
        "finite": finite,
        "max_abs": float(diff.max()) if diff.numel() else 0,
        "rms": float(diff.square().mean().sqrt()) if diff.numel() else 0,
        "relative_l2": float(diff.norm() / expected.float().norm().clamp_min(1e-12)),
        "pass": finite
        and bool(torch.all(diff <= atol + rtol * expected.float().abs())),
        "atol": atol,
        "rtol": rtol,
    }


def memory_probe(fn):
    gc.collect()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    result = {
        "initial_allocated": initial,
        "peak_call_delta": torch.cuda.max_memory_allocated() - initial,
        "live_call_delta": torch.cuda.memory_allocated() - initial,
        "initial_reserved": reserved,
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }
    del output
    return result


def latency(fn, iterations):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return {
        "cuda_ms": start.elapsed_time(end) / iterations,
        "wall_ms": (time.perf_counter() - wall) * 1000 / iterations,
    }


def validate(args, report):
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _resolve_fused_moe_config,
    )
    from sglang.srt.layers.moe.moe_runner.workspace import (
        _common_unsupported_reason,
        _TritonCandidateEstimate,
        _UnsupportedTritonWorkspace,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.moe.workspace_policy import (
        MarlinWorkspaceEstimate,
        plan_workspace,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    dtype = getattr(torch, args.dtype)
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="workspace-kernel-validation")
    )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    quant, ref_weights = make_weights(args, dtype)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=dtype) / 2
    logits = torch.randn(args.tokens, args.experts, device="cuda", dtype=torch.float32)
    if args.routing == "hot":
        logits[:, max(args.topk, args.experts // 16) :] = -100
    values, ids = logits.topk(args.topk, dim=-1)
    ids = ids.to(getattr(torch, args.ids_dtype))
    weights = values.softmax(-1).contiguous()
    if args.masked:
        ids[-1] = -1
    if args.backend == "marlin-mxfp4":
        weights *= args.scale
    topk = StandardTopKOutput(weights, ids, logits)
    dispatch = StandardDispatchOutput(x, None, topk)
    cfg = MoeRunnerConfig(
        num_experts=args.experts * (2 if args.masked else 1),
        num_local_experts=args.experts,
        hidden_size=args.hidden,
        intermediate_size_per_partition=args.intermediate,
        top_k=args.topk,
        activation=args.activation,
        is_gated=True,
        inplace=args.inplace,
        routed_scaling_factor=args.scale,
        workspace_budget_bytes=0,
    )
    backend = (
        MoeRunnerBackend.TRITON if args.backend == "triton" else MoeRunnerBackend.MARLIN
    )
    runner = MoeRunner(backend, cfg)
    unsupported = _common_unsupported_reason(runner, dispatch, None, False)
    if unsupported:
        raise RuntimeError(f"Unexpected unsupported common path: {unsupported}")
    context = (
        override_config(
            {
                "BLOCK_SIZE_M": args.block_m,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }
        )
        if args.config == "fixed"
        else contextlib.nullcontext()
    )
    original_x = x.clone()
    effective_scale = 1.0 if args.backend == "marlin-mxfp4" else args.scale
    with context, torch.inference_mode():
        if args.backend == "triton":
            estimate = _TritonCandidateEstimate(
                x, quant.w13_weight, quant.w2_weight, ids, _resolve_fused_moe_config
            )
        else:
            props = torch.cuda.get_device_properties(0)
            estimate = MarlinWorkspaceEstimate(
                args.hidden,
                args.intermediate,
                args.topk,
                args.experts,
                props.multi_processor_count,
                args.backend == "marlin-mxfp4",
            )
        budgets = [int(value * 2**20) for value in args.budgets_mib]
        try:
            budgets += [estimate.peak_bytes(min(args.tokens, cap)) for cap in args.caps]
            plans = {
                budget: plan_workspace(args.tokens, budget, estimate)
                for budget in budgets
            }
        except _UnsupportedTritonWorkspace as error:
            report["unsupported"] = str(error)
            report["pass"] = True
            return
        report["plans"] = {str(key): asdict(value) for key, value in plans.items()}
        if args.backend == "triton":
            report["configs"] = {
                str(tokens): {"up": value[0], "down": value[1]}
                for tokens, value in estimate.candidates.items()
            }

        def run(budget):
            cfg.workspace_budget_bytes = budget
            if args.inplace:
                x.copy_(original_x)
            return runner.run(dispatch, quant).hidden_states

        expected = reference(
            original_x, ids, weights, ref_weights, args.activation, effective_scale
        )
        outputs = {}
        report["cases"] = cases = {}
        for budget in [0, *sorted(set(budgets), reverse=True)]:
            out = run(budget)
            torch.cuda.synchronize()
            outputs[budget] = out.clone()
            errors = error_metrics(out, expected, args.atol, args.rtol)
            cases[str(budget)] = {"reference": errors}
            print(f"budget={budget} reference={errors}", flush=True)
            for _ in range(args.warmup):
                run(budget)
            memory = memory_probe(lambda: run(budget))
            cases[str(budget)]["memory"] = memory
            if budget:
                cases[str(budget)]["vs_full"] = error_metrics(
                    out, outputs[0], args.atol, args.rtol
                )
                output_bytes = (
                    0
                    if args.inplace and args.backend == "triton"
                    else x.numel() * x.element_size()
                )
                memory["budget_plus_output"] = budget + output_bytes
                memory["within_budget_plus_output_1mib_allocator_allowance"] = (
                    memory["peak_call_delta"] <= budget + output_bytes + 2**20
                )

        if not args.skip_graphs:
            for budget in budgets:
                cfg.workspace_budget_bytes = budget
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run(budget)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run(budget)
                saved_ids, saved_weights = ids.clone(), weights.clone()
                ids.copy_(torch.where(ids < 0, ids, (ids + 1) % args.experts))
                weights.mul_(0.7)
                changed = reference(
                    original_x,
                    ids,
                    weights,
                    ref_weights,
                    args.activation,
                    effective_scale,
                )
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
                cases[str(budget)]["graph_changed_routing"] = error_metrics(
                    captured, changed, args.atol, args.rtol
                )
                ids.copy_(saved_ids)
                weights.copy_(saved_weights)
                del graph, captured, changed

        if not args.inplace:
            for budget in budgets:
                cfg.workspace_budget_bytes = budget
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                concurrent = []
                for stream in streams:
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        concurrent.append(run(budget))
                torch.cuda.synchronize()
                cases[str(budget)]["concurrent_streams"] = [
                    error_metrics(output, expected, args.atol, args.rtol)
                    for output in concurrent
                ]
                del concurrent

        order = [0, *sorted(set(budgets), reverse=True)]
        rng = random.Random(args.seed)
        for repeat in range(args.repeats):
            rng.shuffle(order)
            for budget in order:
                value = latency(lambda: run(budget), args.iterations)
                cases[str(budget)].setdefault("timings", []).append(value)
                print(f"repeat={repeat} budget={budget} timing={value}", flush=True)
        for value in cases.values():
            value["median_cuda_ms"] = statistics.median(
                t["cuda_ms"] for t in value["timings"]
            )
            value["median_wall_ms"] = statistics.median(
                t["wall_ms"] for t in value["timings"]
            )
        report["pass"] = all(
            value["reference"]["pass"]
            and value.get("vs_full", {"pass": True})["pass"]
            and value.get("graph_changed_routing", {"pass": True})["pass"]
            and all(item["pass"] for item in value.get("concurrent_streams", []))
            and value["memory"].get(
                "within_budget_plus_output_1mib_allocator_allowance", True
            )
            for value in cases.values()
        )
    destroy_model_parallel()
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["triton", "marlin-int4", "marlin-int8", "marlin-mxfp4"],
        required=True,
    )
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--ids-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--routing", choices=["uniform", "hot"], default="uniform")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--config", choices=["runtime", "fixed"], default="runtime")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--caps", type=int, nargs="*", default=[4])
    parser.add_argument("--budgets-mib", type=float, nargs="*", default=[])
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.masked and args.backend != "triton":
        parser.error("Marlin EP mapping is outside the shared adapter")
    if args.backend != "triton" and args.activation != "silu":
        parser.error("Marlin adapter requires SiLU")
    if args.backend == "marlin-mxfp4" and args.dtype != "bfloat16":
        parser.error("MXFP4 requires BF16 activation")
    for name in (
        "tokens",
        "hidden",
        "intermediate",
        "experts",
        "topk",
        "iterations",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    root = Path(__file__).resolve().parents[3]
    import sgl_kernel
    import triton

    import sglang

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
        "torch": torch.__version__,
        "triton": triton.__version__,
        "sglang": sglang.__file__,
        "sgl_kernel": sgl_kernel.__file__,
        "device": str(torch.cuda.get_device_properties(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Synthetic single-rank kernel workload; TP/PP/DP=1/1/1; no checkpoint/KV/speculative/HTTP workload",
        "pass": False,
    }
    try:
        validate(args, report)
    except Exception:
        report["exception"] = traceback.format_exc()
        print(report["exception"], flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Result {args.output}: pass={report['pass']}", flush=True)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
