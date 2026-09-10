"""Check the real Python runner against the isolated upstream CUDA baseline."""

import argparse
from types import SimpleNamespace

import torch
from bench import Runner, default_configs, library_info, make_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[4, 16, 64, 256, 640, 1024, 1536, 2048, 2049, 4096, 8192])
    args = parser.parse_args()
    baseline, _ = library_info(args.baseline, "mxfp4a8_baseline")
    candidate, _ = library_info(args.candidate, "mxfp4a8_candidate")

    from sglang.srt.layers.moe import cutlass_mxfp4a8_fused_moe as module
    from sglang.srt.runtime_context import get_parallel

    # Redirect only this process. Do not replace the installed sgl_kernel library.
    module.get_cutlass_w4a8_moe_mm_data_with_permutation = candidate.metadata
    torch.ops.sgl_kernel.cutlass_mxfp4a8_fused_moe_core = candidate.core
    options = SimpleNamespace(
        hidden=4096, inter=2048, experts=256, topk=6, clamp=None, routed_scale=1.0
    )
    weights = make_weights(options.experts, options.hidden, options.inter, 20260910)
    w1, s1, r1, w2, s2, r2 = weights
    py_runner = module.CutlassMxfp4A8FusedMoeRunner()
    with get_parallel().override(moe_ep_size=1):
        for m in args.tokens:
            gen = torch.Generator(device="cuda").manual_seed(20260910 + m)
            x = torch.randn((m, options.hidden), dtype=torch.bfloat16,
                            device="cuda", generator=gen) * 0.1
            logits = torch.randn((m, options.experts), device="cuda", generator=gen)
            factors, ids = logits.softmax(-1).topk(options.topk, dim=-1)
            factors = (factors / factors.sum(-1, keepdim=True)).contiguous()
            ids = ids.to(torch.int32).contiguous()
            reference = Runner(baseline, x, ids, factors, weights, options, default_configs(m))

            def run():
                a1, b1, c1, scale1, a2, b2, c2, scale2 = reference.strides
                return py_runner(
                    x, w1, w2, s1, s2, w1, w2, s1, s2, r1, r2, factors, ids,
                    a1, b1, c1, a2, b2, c2, scale1, scale2, reference.offsets,
                    reference.problems1, reference.problems2,
                )

            expected = reference().clone()
            actual = run()
            torch.cuda.synchronize()
            if not torch.equal(expected, actual):
                raise AssertionError(f"eager Python runner mismatch at M={m}")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = run()
            x.mul_(0.75)
            ids.copy_(ids.roll(1, dims=0))
            graph.replay()
            expected = reference().clone()
            torch.cuda.synchronize()
            if not torch.equal(expected, graph_output):
                raise AssertionError(f"graph Python runner mismatch at M={m}")
            print(f"M={m}: real Python runner eager/changed-input graph bit-exact", flush=True)
            del graph, reference


if __name__ == "__main__":
    main()
