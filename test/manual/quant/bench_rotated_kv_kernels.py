"""算子层 microbench：对比 Triton 实现 vs CPU 参考实现。

目的：定位 T3 端到端 hang 的真实瓶颈是不是算子本身。

跑法（容器内）：
    cd /workspace/sglang-bytedance/python && \
    python3 ../test/manual/quant/bench_rotated_kv_kernels.py
"""
from __future__ import annotations

import time

import torch

from sglang.srt.layers.quantization.rotated_kv_quant import (
    bitpack_rowwise as cpu_bitpack,
    bitunpack_rowwise as cpu_bitunpack,
)
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    _build_pack_meta_from_bits,
)
from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
    triton_bitpack_rowwise,
    triton_bitunpack_rowwise,
)


D = 448  # MLA nope dim


def _make_bits_dsv4() -> torch.Tensor:
    """模拟 DSv4 的 bits 分布：前若干 dim 4-bit，其余 2-bit，sum ≈ 1024。"""
    bits = torch.full((D,), 2, dtype=torch.int32)
    # 让前 64 维 4 bits，其余 2 bits → row_bits = 64*4 + 384*2 = 1024
    bits[:64] = 4
    return bits


def _gpu_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_pack(N: int, bits: torch.Tensor, device: torch.device) -> dict:
    """对比 CPU bitpack 和 Triton bitpack。"""
    print(f"\n=== bitpack  N={N:>8d} ===")
    bits_cpu = bits.to("cpu")
    codes_cpu = torch.randint(0, 4, (N, D), dtype=torch.int32)

    # --- CPU 参考 ---
    t0 = time.perf_counter()
    out_cpu = cpu_bitpack(codes_cpu, bits_cpu)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU  bitpack: {t_cpu*1000:>10.2f} ms")

    out_triton_bytes = None
    t_triton = None
    if device.type == "cuda":
        codes_gpu = codes_cpu.to(device)
        dim_of_bit, bitpos_in_dim, row_bits, row_bytes = (
            _build_pack_meta_from_bits(bits_cpu)
        )
        dim_of_bit_gpu = dim_of_bit.to(device)
        bitpos_in_dim_gpu = bitpos_in_dim.to(device)

        # warmup
        for _ in range(2):
            triton_bitpack_rowwise(
                codes_gpu, dim_of_bit_gpu, bitpos_in_dim_gpu, row_bytes
            )
        _gpu_sync()
        t0 = time.perf_counter()
        for _ in range(5):
            out_triton = triton_bitpack_rowwise(
                codes_gpu, dim_of_bit_gpu, bitpos_in_dim_gpu, row_bytes
            )
        _gpu_sync()
        t_triton = (time.perf_counter() - t0) / 5
        print(f"  GPU  bitpack: {t_triton*1000:>10.2f} ms (avg of 5)")
        out_triton_bytes = out_triton.cpu()

        # 正确性
        eq = torch.equal(out_cpu, out_triton_bytes)
        print(f"  match: {eq}")

    return {
        "N": N,
        "cpu_ms": t_cpu * 1000,
        "gpu_ms": t_triton * 1000 if t_triton else None,
    }


def bench_unpack(N: int, bits: torch.Tensor, device: torch.device) -> dict:
    print(f"\n=== bitunpack N={N:>8d} ===")
    bits_cpu = bits.to("cpu")
    codes_cpu = torch.randint(0, 4, (N, D), dtype=torch.int32)
    packed_cpu = cpu_bitpack(codes_cpu, bits_cpu)

    # --- CPU 参考 ---
    t0 = time.perf_counter()
    out_cpu = cpu_bitunpack(packed_cpu, bits_cpu, dim=D)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU  bitunpack: {t_cpu*1000:>10.2f} ms")

    t_triton = None
    if device.type == "cuda":
        packed_gpu = packed_cpu.to(device)
        bits_gpu = bits_cpu.to(device)

        # warmup
        for _ in range(2):
            triton_bitunpack_rowwise(packed_gpu, bits_gpu)
        _gpu_sync()
        t0 = time.perf_counter()
        for _ in range(5):
            out_triton = triton_bitunpack_rowwise(packed_gpu, bits_gpu)
        _gpu_sync()
        t_triton = (time.perf_counter() - t0) / 5
        print(f"  GPU  bitunpack: {t_triton*1000:>10.2f} ms (avg of 5)")

        out_triton_cpu = out_triton.cpu().to(torch.int64)
        eq = torch.equal(out_cpu, out_triton_cpu)
        print(f"  match: {eq}")

    return {
        "N": N,
        "cpu_ms": t_cpu * 1000,
        "gpu_ms": t_triton * 1000 if t_triton else None,
    }


def main() -> None:
    bits = _make_bits_dsv4()
    print(f"bits dsv4-like: sum={int(bits.sum())} (row_bits)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    sizes = [10_000, 100_000, 1_000_000]
    pack_results = []
    unpack_results = []
    for N in sizes:
        pack_results.append(bench_pack(N, bits, device))
        unpack_results.append(bench_unpack(N, bits, device))

    print("\n=== summary (ms) ===")
    print(f"{'op':<10}{'N':>10}{'CPU ms':>14}{'GPU ms':>14}{'speedup':>10}")
    for r in pack_results:
        sp = r["cpu_ms"] / r["gpu_ms"] if r["gpu_ms"] else float("nan")
        print(
            f"{'pack':<10}{r['N']:>10}{r['cpu_ms']:>14.2f}"
            f"{(r['gpu_ms'] or 0):>14.2f}{sp:>10.1f}x"
        )
    for r in unpack_results:
        sp = r["cpu_ms"] / r["gpu_ms"] if r["gpu_ms"] else float("nan")
        print(
            f"{'unpack':<10}{r['N']:>10}{r['cpu_ms']:>14.2f}"
            f"{(r['gpu_ms'] or 0):>14.2f}{sp:>10.1f}x"
        )


if __name__ == "__main__":
    main()
