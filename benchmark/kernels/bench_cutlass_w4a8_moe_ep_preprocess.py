import argparse
from dataclasses import dataclass

import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    cutlass_w4_run_moe_ep_preproess_torch,
    cutlass_w4_run_moe_ep_preproess_triton,
)


@dataclass
class BenchResult:
    name: str
    mean_us: float
    min_us: float
    max_us: float


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark MoE EP preprocess (src2dst) implementations."
    )
    parser.add_argument("--m-list", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-sentinel",
        action="store_true",
        help="Allow generating the EP sentinel id (= num_experts) in topk_ids.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run correctness checks before benchmarking.",
    )
    return parser.parse_args()


def _make_topk_ids(
    m: int, topk: int, num_experts: int, seed: int, include_sentinel: bool
) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    upper = num_experts + 1 if include_sentinel else num_experts
    return torch.randint(
        0,
        upper,
        (m, topk),
        device="cuda",
        dtype=torch.int32,
        generator=gen,
    )


def _dst2src(src2dst: torch.Tensor) -> torch.Tensor:
    return torch.argsort(src2dst.to(torch.int64))


def _validate_grouping(topk_ids: torch.Tensor, src2dst: torch.Tensor):
    flat_ids = topk_ids.reshape(-1)
    numel = flat_ids.numel()
    sorted_src2dst = torch.sort(src2dst.to(torch.int64)).values
    expected = torch.arange(numel, device=topk_ids.device, dtype=torch.int64)
    assert torch.equal(sorted_src2dst, expected), "src2dst must be a permutation"

    grouped_ids = flat_ids[_dst2src(src2dst)]
    expected_grouped_ids = torch.sort(flat_ids, stable=True).values
    assert torch.equal(
        grouped_ids, expected_grouped_ids
    ), "grouped expert ids do not match sorted ids"


def _time_cuda_fn(fn, warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    elapsed_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    elapsed_us = [v * 1000.0 for v in elapsed_ms]
    return BenchResult(
        name="",
        mean_us=sum(elapsed_us) / len(elapsed_us),
        min_us=min(elapsed_us),
        max_us=max(elapsed_us),
    )


def _run_one_case(args, m: int):
    topk_ids = _make_topk_ids(
        m,
        args.topk,
        args.num_experts,
        seed=args.seed + m,
        include_sentinel=args.include_sentinel,
    )

    triton_src2dst = cutlass_w4_run_moe_ep_preproess_triton(topk_ids)
    torch_stable_src2dst = cutlass_w4_run_moe_ep_preproess_torch(topk_ids, stable=True)
    torch_unstable_src2dst = cutlass_w4_run_moe_ep_preproess_torch(topk_ids, stable=False)

    if args.check:
        assert torch.equal(
            triton_src2dst, torch_stable_src2dst
        ), "torch stable path must exactly match triton path"
        _validate_grouping(topk_ids, triton_src2dst)
        _validate_grouping(topk_ids, torch_unstable_src2dst)

    providers = {
        "triton_stable": lambda: cutlass_w4_run_moe_ep_preproess_triton(topk_ids),
        "torch_stable": lambda: cutlass_w4_run_moe_ep_preproess_torch(topk_ids, stable=True),
        "torch_unstable": lambda: cutlass_w4_run_moe_ep_preproess_torch(
            topk_ids, stable=False
        ),
    }

    results = {}
    for name, fn in providers.items():
        result = _time_cuda_fn(fn, warmup=args.warmup, iters=args.iters)
        result.name = name
        results[name] = result

    return results


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    print(
        "m topk numel triton_us torch_stable_us torch_unstable_us "
        "stable_speedup unstable_speedup"
    )
    for m in args.m_list:
        results = _run_one_case(args, m)
        triton_mean = results["triton_stable"].mean_us
        torch_stable_mean = results["torch_stable"].mean_us
        torch_unstable_mean = results["torch_unstable"].mean_us
        print(
            f"{m} {args.topk} {m * args.topk} "
            f"{triton_mean:.2f} {torch_stable_mean:.2f} {torch_unstable_mean:.2f} "
            f"{triton_mean / torch_stable_mean:.3f}x {triton_mean / torch_unstable_mean:.3f}x"
        )


if __name__ == "__main__":
    main()
