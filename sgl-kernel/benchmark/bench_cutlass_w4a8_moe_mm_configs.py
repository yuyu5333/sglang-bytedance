#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for CUTLASS W4A8 grouped GEMM config selection.

This benchmark bypasses the full MoE path and directly times `cutlass_w4a8_moe_mm`
for the concrete TP4 / TP8 shapes observed in production:

- TP4 PD gemm1: n=1024, k=4096
- TP4 PD gemm2: n=4096, k=512
- TP8 colocated gemm1: n=512, k=4096
- TP8 colocated gemm2: n=4096, k=256

For each shape and each requested m value, it force-selects one kernel config via
environment variables, runs the kernel repeatedly, and emits both a wide raw table
and a per-m best-config summary in Markdown.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, List, Sequence

import torch

from sgl_kernel import cutlass_w4a8_moe_mm
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8


CO_TOKENS = ["16_c111", "16_c211", "32_c111", "32_c211", "64_c111", "64_c211"]
PP_TOKENS = ["64x16_c111", "128x32_c111", "128x32_c211", "128x64_c111"]
DEFAULT_M_VALUES = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192]

SHAPES: Dict[str, Dict[str, object]] = {
    "tp4_gemm1": {
        "label": "TP4 PD gemm1",
        "n": 1024,
        "k": 4096,
        "kind": "co",
        "env_for_m": lambda m: (
            "SGLANG_W4A8_FORCE_N1024_K4096_LE32"
            if m <= 32
            else "SGLANG_W4A8_FORCE_N1024_K4096_LE1024"
            if m <= 1024
            else "SGLANG_W4A8_FORCE_N1024_K4096_GT1024"
        ),
    },
    "tp4_gemm2": {
        "label": "TP4 PD gemm2",
        "n": 4096,
        "k": 512,
        "kind": "co",
        "env_for_m": lambda m: (
            "SGLANG_W4A8_FORCE_N4096_K512_LE32"
            if m <= 32
            else "SGLANG_W4A8_FORCE_N4096_K512_LE1024"
            if m <= 1024
            else "SGLANG_W4A8_FORCE_N4096_K512_GT1024"
        ),
    },
    "tp8_gemm1": {
        "label": "TP8 colocated gemm1",
        "n": 512,
        "k": 4096,
        "kind": "co",
        "env_for_m": lambda m: (
            "SGLANG_W4A8_FORCE_N512_K4096_LE32"
            if m <= 32
            else "SGLANG_W4A8_FORCE_N512_K4096_LE1024"
            if m <= 1024
            else "SGLANG_W4A8_FORCE_N512_K4096_GT1024"
        ),
    },
    "tp8_gemm2": {
        "label": "TP8 colocated gemm2",
        "n": 4096,
        "k": 256,
        "kind": "pp",
        "env_for_m": lambda m: (
            "SGLANG_W4A8_FORCE_N4096_K256_LE8"
            if m <= 8
            else "SGLANG_W4A8_FORCE_N4096_K256_LE32"
            if m <= 32
            else "SGLANG_W4A8_FORCE_N4096_K256_GT32"
        ),
    },
}

ALL_FORCE_ENVS = sorted(
    {
        shape["env_for_m"](m)  # type: ignore[index]
        for shape in SHAPES.values()
        for m in (4, 16, 64, 2048)
    }
)


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    input_tensor_int8 = int4_values_interleaved.to(torch.int8)
    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]
    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts: int, ref_weight: torch.Tensor, ref_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8).contiguous()

    alignment = 4 if k % 512 == 0 else 1
    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0],
        ref_scale.shape[1],
        ref_scale.shape[2] // alignment,
        alignment,
    )
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    )
    return w_q, scale_interleaved.contiguous()


def per_tensor_quant_fp8_wrapper(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(1, device=x.device, dtype=torch.float32)
    per_tensor_quant_fp8(x, x_q, x_s, is_static=False)
    return x_q, x_s


def parse_m_values(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def clear_force_envs() -> None:
    for env_name in ALL_FORCE_ENVS:
        os.environ.pop(env_name, None)


@contextmanager
def forced_shape_config(env_name: str, token: str):
    clear_force_envs()
    os.environ[env_name] = token
    try:
        yield
    finally:
        os.environ.pop(env_name, None)


def benchmark_cuda_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    return sum(times_ms) / len(times_ms)


def build_case_tensors(n: int, k: int, m: int, dtype: torch.dtype, seed: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = "cuda"
    num_experts = 1

    a = torch.randn(m, k, dtype=dtype, device=device)
    ref_w = torch.randint(-8, 8, (num_experts, n, k), dtype=torch.int8, device=device)
    ref_w_scale = torch.randn(num_experts, n, k // 128, dtype=dtype, device=device) * 0.005

    a_q, a_scale = per_tensor_quant_fp8_wrapper(a)
    w_q, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)

    expert_offsets = torch.tensor([0], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    b_strides = a_strides
    d_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    s_strides = d_strides
    d = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    return {
        "d": d,
        "a_q": a_q.contiguous(),
        "w_q": w_q,
        "a_scale": a_scale,
        "w_scale": w_scale,
        "expert_offsets": expert_offsets,
        "problem_sizes": problem_sizes,
        "a_strides": a_strides,
        "b_strides": b_strides,
        "d_strides": d_strides,
        "s_strides": s_strides,
    }


def run_one_case(shape_name: str, m: int, token: str, warmup: int, iters: int, seed: int) -> float:
    shape = SHAPES[shape_name]
    n = int(shape["n"])
    k = int(shape["k"])
    env_name = shape["env_for_m"](m)  # type: ignore[index]
    tensors = build_case_tensors(n=n, k=k, m=m, dtype=torch.bfloat16, seed=seed)

    def runner() -> None:
        cutlass_w4a8_moe_mm(
            tensors["d"],
            tensors["a_q"],
            tensors["w_q"],
            tensors["a_scale"],
            tensors["w_scale"],
            tensors["expert_offsets"],
            tensors["problem_sizes"],
            tensors["a_strides"],
            tensors["b_strides"],
            tensors["d_strides"],
            tensors["s_strides"],
            128,
            1,
        )

    with forced_shape_config(env_name=env_name, token=token):
        return benchmark_cuda_fn(runner, warmup=warmup, iters=iters)


def build_markdown(results: Dict[str, Dict[int, Dict[str, float]]]) -> str:
    lines: List[str] = []
    for shape_name, shape_results in results.items():
        shape = SHAPES[shape_name]
        kind = str(shape["kind"])
        tokens = CO_TOKENS if kind == "co" else PP_TOKENS
        lines.append(f"## {shape['label']} (`n={shape['n']}, k={shape['k']}`)")
        lines.append("")
        lines.append("| m | best_config | best_ms | second_best | delta_ms |")
        lines.append("| ---: | --- | ---: | --- | ---: |")
        for m in sorted(shape_results):
            ordered = sorted(shape_results[m].items(), key=lambda item: item[1])
            best_token, best_ms = ordered[0]
            second_token, second_ms = ordered[1] if len(ordered) > 1 else ("-", best_ms)
            lines.append(
                f"| {m} | `{best_token}` | {format_ms(best_ms)} | `{second_token}` | {format_ms(second_ms - best_ms)} |"
            )
        lines.append("")
        lines.append("| m | " + " | ".join(f"`{token}`" for token in tokens) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in tokens) + " |")
        for m in sorted(shape_results):
            row = [format_ms(shape_results[m][token]) for token in tokens]
            lines.append(f"| {m} | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUTLASS W4A8 grouped GEMM configs.")
    parser.add_argument(
        "--shapes",
        default="tp4_gemm1,tp4_gemm2,tp8_gemm1,tp8_gemm2",
        help="Comma-separated shape keys.",
    )
    parser.add_argument(
        "--m-values",
        default=",".join(str(v) for v in DEFAULT_M_VALUES),
        help="Comma-separated m values to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--markdown-out", default="", help="Optional markdown output path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    requested_shapes = [item.strip() for item in args.shapes.split(",") if item.strip()]
    for shape_name in requested_shapes:
        if shape_name not in SHAPES:
            raise ValueError(f"Unknown shape key: {shape_name}")
    m_values = parse_m_values(args.m_values)

    results: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    bench_start = time.time()
    for shape_name in requested_shapes:
        kind = str(SHAPES[shape_name]["kind"])
        tokens = CO_TOKENS if kind == "co" else PP_TOKENS
        for m in m_values:
            results[shape_name][m] = {}
            for idx, token in enumerate(tokens):
                seed = args.seed + requested_shapes.index(shape_name) * 100000 + m * 10 + idx
                ms = run_one_case(shape_name, m, token, warmup=args.warmup, iters=args.iters, seed=seed)
                results[shape_name][m][token] = ms
                print(
                    f"[bench] shape={shape_name} m={m} token={token} mean_ms={ms:.3f}",
                    flush=True,
                )
            torch.cuda.empty_cache()

    markdown = build_markdown(results)
    total_s = time.time() - bench_start
    footer = f"\nBenchmark finished in {total_s:.1f}s\n"
    print(markdown)
    print(footer, flush=True)

    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as fout:
            fout.write(markdown)


if __name__ == "__main__":
    main()
