"""Standalone full-MoE timing after a bitwise upstream correctness check."""

import argparse
import json
from pathlib import Path

import torch
from bench import (
    Runner,
    assert_equal,
    default_configs,
    library_info,
    make_weights,
    paired_times,
    production_config_selector,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    if args.tokens < 1 or args.warmup < 0 or args.iters < 1:
        parser.error("tokens/iters must be positive; warmup must be nonnegative")
    args.hidden, args.inter, args.experts, args.topk = 4096, 2048, 256, 6
    args.clamp, args.routed_scale = None, 1.0
    baseline, baseline_info = library_info(args.baseline, "mxfp4a8_baseline")
    candidate, candidate_info = library_info(args.candidate, "mxfp4a8_candidate")
    weights = make_weights(args.experts, args.hidden, args.inter, args.seed)
    gen = torch.Generator(device="cuda").manual_seed(args.seed + args.tokens)
    x = torch.randn((args.tokens, args.hidden), device="cuda",
                    dtype=torch.bfloat16, generator=gen) * 0.1
    logits = torch.randn((args.tokens, args.experts), device="cuda", generator=gen)
    factors, ids = logits.softmax(-1).topk(args.topk, dim=-1)
    factors = (factors / factors.sum(-1, keepdim=True)).contiguous()
    ids = ids.to(torch.int32).contiguous()
    configs = production_config_selector()(
        args.tokens, args.hidden, args.inter, args.experts, args.topk
    )
    reference = Runner(baseline, x, ids, factors, weights, args,
                       default_configs(args.tokens))
    runner = Runner(candidate, x, ids, factors, weights, args, configs)
    equal = assert_equal(reference, runner)
    del reference
    result = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, baseline=baseline_info, candidate=candidate_info,
        tokens=args.tokens, hidden=args.hidden, inter=args.inter,
        experts=args.experts, topk=args.topk, seed=args.seed, configs=configs,
        warmup=args.warmup, iters=args.iters, equal=equal,
        scope="Standalone full MoE; acceptance requires verified container isolation",
        **paired_times((runner,), args.warmup, args.iters)[0],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
