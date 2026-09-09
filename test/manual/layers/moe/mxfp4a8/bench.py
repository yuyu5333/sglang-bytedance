"""Paired full-MoE benchmark for isolated, source-built SM90 libraries.

Weight preprocessing is outside timing. Routing, FP8 quantization, both GEMMs,
SwiGLU and ordered top-k combine are inside timing. No installed sgl_kernel
binary is loaded or replaced. See README.md for build and run commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def combine(
    input_ptr,
    output_ptr,
    perm_ptr,
    factors_ptr,
    m,
    topk: tl.constexpr,
    row_stride: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = (token < m) & (offs < row_stride)
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.range(0, topk):
        token_major_idx = token * topk + j
        src_row = tl.load(perm_ptr + token_major_idx).to(tl.int64)
        vals = tl.load(
            input_ptr + src_row * row_stride + offs, mask=mask, other=0.0
        ).to(tl.float32)
        factor = tl.load(factors_ptr + token_major_idx).to(tl.float32)
        acc += vals * factor * routed_scaling_factor
    tl.store(output_ptr + token * row_stride + offs, acc, mask=mask)


def default_configs(m):
    if m <= 64:
        return 100, 100
    if m == 2048:
        return 313, 313
    if m == 4096:
        return 320, 334
    if m >= 8192:
        return 322, 334
    return 101, 101


def make_weights(experts, hidden, inter, seed):
    from flashinfer.fused_moe import (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    weights = []
    for n, k in ((2 * inter, hidden), (hidden, inter)):
        raw = torch.randint(
            0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
        )
        scales = torch.randint(
            125, 130, (experts, n, k // 32),
            dtype=torch.uint8, device="cuda", generator=gen,
        )
        q, offsets, residual = preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            raw, scales
        )
        weights.extend(
            (q.view(torch.int8).contiguous(), offsets.view(torch.uint8).contiguous(),
             (residual.float() * 64.0).contiguous())
        )
    return weights


class Runner:
    def __init__(self, ops, x, ids, factors, weights, args, configs):
        self.ops, self.x, self.ids, self.factors = ops, x, ids, factors
        self.weights, self.configs, self.args = weights, configs, args
        self.m, self.hidden = x.shape
        self.topk, self.inter, self.experts = ids.shape[1], args.inter, args.experts
        rows = ids.numel()

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=x.device)

        self.a_map = empty((rows,), torch.int32)
        self.c_map = empty((rows,), torch.int32)
        self.offsets = empty((self.experts + 1,), torch.int32)
        self.gemm_offsets = self.offsets[:-1]
        self.problems1 = empty((self.experts, 3), torch.int32)
        self.problems2 = empty((self.experts, 3), torch.int32)
        self.a1 = empty((rows, self.hidden), torch.float8_e4m3fn)
        self.s1 = empty((rows,), torch.float32)
        self.c1 = empty((rows, 2 * self.inter), torch.bfloat16)
        self.a2 = empty((rows, self.inter), torch.float8_e4m3fn)
        self.s2 = empty((rows,), torch.float32)
        self.c2 = empty((rows, self.hidden), torch.bfloat16)
        self.output = empty(x.shape, x.dtype)
        self.strides = [
            torch.full((self.experts,), v, dtype=torch.int64, device=x.device)
            for v in (self.hidden, self.hidden, 2 * self.inter, 2 * self.inter,
                      self.inter, self.inter, self.hidden, self.hidden)
        ]
        self.graph = None

    def run(self):
        prepare = self.m <= 64
        if not prepare:
            self.ops.metadata(
                self.ids, self.offsets, self.problems1, self.problems2,
                self.a_map, self.c_map, self.experts, self.inter, self.hidden,
            )
        self.ops.core(
            self.c1, self.c2, self.x, self.ids, self.a_map, self.c_map, self.x,
            self.a1, self.s1, self.a2, self.s2, *self.weights,
            self.offsets, self.gemm_offsets, self.problems1, self.problems2,
            *self.strides, self.topk, *self.configs, self.experts, self.inter,
            self.hidden, self.args.clamp or 0.0, self.args.clamp is not None,
            prepare, None,
        )
        combine[(self.m, triton.cdiv(self.hidden, 256))](
            self.c2, self.output, self.c_map, self.factors, self.m, self.topk,
            self.hidden, self.args.routed_scale, BLOCK=256,
        )
        return self.output

    def __call__(self):
        if self.graph is None:
            return self.run()
        self.graph.replay()
        return self.output

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.run()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run()


def assert_equal(left, right):
    left()
    right()
    torch.cuda.synchronize()
    result = {}
    for name in ("a_map", "c_map", "offsets", "a1", "s1", "c1", "a2", "s2",
                 "c2", "output"):
        a, b = getattr(left, name), getattr(right, name)
        equal = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        if a.dtype in (torch.float32, torch.bfloat16):
            if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                raise AssertionError(f"{name}: nonfinite values")
        result[name] = equal
        if not equal:
            diff = (a.float() - b.float()).abs()
            raise AssertionError(
                f"{name}: max_abs={diff.max().item()} mean_abs={diff.mean().item()}"
            )
    return result


def paired_times(runners, warmup, iters):
    for _ in range(warmup):
        for runner in runners:
            runner()
    torch.cuda.synchronize()
    times = [[] for _ in runners]
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in runners]
    for step in range(iters):
        order = range(len(runners)) if step % 2 == 0 else reversed(range(len(runners)))
        for i in order:
            start, end = events[i]
            start.record()
            runners[i]()
            end.record()
            end.synchronize()
            times[i].append(start.elapsed_time(end))
    return [
        dict(median_ms=statistics.median(t), min_ms=min(t), samples_ms=t)
        for t in times
    ]


def library_info(path, namespace):
    path = Path(path).resolve()
    torch.ops.load_library(str(path))
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return getattr(torch.ops, namespace), dict(path=str(path), sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--baseline-namespace", default="mxfp4a8_baseline")
    parser.add_argument("--candidate-namespace", default="mxfp4a8_candidate")
    parser.add_argument("--configs", type=int, nargs=2)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[4, 16, 64, 256, 1024, 2048, 4096, 8192])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--inter", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--clamp", type=float)
    parser.add_argument("--routed-scale", type=float, default=1.0)
    parser.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.hidden % 128 or args.inter % 128 or not 0 < args.topk <= args.experts:
        parser.error("hidden/inter must be multiples of 128; 0 < topk <= experts")
    if min(args.tokens) < 1 or args.iters < 1 or args.warmup < 0:
        parser.error("tokens/iters must be positive; warmup must be nonnegative")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This benchmark requires SM90")

    baseline, baseline_info = library_info(args.baseline, args.baseline_namespace)
    candidate, candidate_info = (
        library_info(args.candidate, args.candidate_namespace) if args.candidate
        else (baseline, baseline_info)
    )
    weights = make_weights(args.experts, args.hidden, args.inter, args.seed)
    report = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, baseline=baseline_info, candidate=candidate_info,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        results=[],
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    for m in args.tokens:
        gen = torch.Generator(device="cuda").manual_seed(args.seed + m)
        x = torch.randn((m, args.hidden), device="cuda", dtype=torch.bfloat16,
                        generator=gen) * 0.1
        logits = torch.randn((m, args.experts), device="cuda", generator=gen)
        if args.routing == "skewed":
            logits[:, :args.topk] += 8
        factors, ids = logits.softmax(-1).topk(args.topk, dim=-1)
        factors = (factors / factors.sum(-1, keepdim=True)).contiguous()
        ids = ids.to(torch.int32).contiguous()
        left = Runner(baseline, x, ids, factors, weights, args, default_configs(m))
        right = Runner(candidate, x, ids, factors, weights, args,
                       args.configs or default_configs(m))
        equality = assert_equal(left, right)
        if args.graph:
            left.capture()
            right.capture()
            equality = assert_equal(left, right)
            # Replay with changed values at fixed addresses, not cached outputs.
            x.mul_(0.75)
            ids.copy_(ids.roll(1, dims=0))
            equality = assert_equal(left, right)
        results = paired_times((left, right), args.warmup, args.iters)
        row = dict(tokens=m, equal=equality, baseline=results[0],
                   candidate=results[1], configs=[left.configs, right.configs],
                   reduction_pct=100 * (1 - results[1]["median_ms"] /
                                        results[0]["median_ms"]))
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        del left, right


if __name__ == "__main__":
    main()
