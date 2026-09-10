"""Standalone same-format FlashInfer timing; one shape/cache per process."""

import argparse
import json
from pathlib import Path

import torch
from bench import make_weights, paired_times
from flashinfer import __version__ as flashinfer_version
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import cutlass_fused_moe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    m, hidden, inter, experts, topk = args.tokens, 4096, 2048, 256, 6
    w1, s1, r1, w2, s2, r2 = make_weights(experts, hidden, inter, args.seed)
    w1, w2 = w1.view(torch.uint8), w2.view(torch.uint8)
    gen = torch.Generator(device="cuda").manual_seed(args.seed + m)
    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16,
                    generator=gen) * 0.1
    logits = torch.randn((m, experts), device="cuda", generator=gen)
    factors, ids = logits.softmax(-1).topk(topk, dim=-1)
    factors = (factors / factors.sum(-1, keepdim=True)).contiguous()
    ids = ids.to(torch.int32).contiguous()
    output = torch.empty_like(x)
    scales = [s1.view(torch.int32), r1, torch.ones((), device="cuda"),
              s2.view(torch.int32), r2]

    def run():
        cutlass_fused_moe(
            input=x,
            token_selected_experts=ids,
            token_final_scales=factors,
            fc1_expert_weights=w1,
            fc2_expert_weights=w2,
            output_dtype=torch.bfloat16,
            quant_scales=scales,
            use_w4_group_scaling=True,
            use_wfp4afp8_humming=True,
            output=output,
        )
        return output

    with autotune(True, cache=args.cache):
        run()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise AssertionError("Nonfinite FlashInfer output")
    result = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        flashinfer=flashinfer_version, tokens=m, hidden=hidden, inter=inter,
        experts=experts, topk=topk, seed=args.seed, cache=args.cache,
        warmup=args.warmup, iters=args.iters,
        scope="Standalone process; diagnostic unless container isolation is verified",
        **paired_times((run,), args.warmup, args.iters)[0],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
