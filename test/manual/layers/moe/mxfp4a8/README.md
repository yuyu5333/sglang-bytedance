# SM90 MXFP4A8 Isolated Benchmark

This target compiles five production CUDA translation units and one registration
file. It does not build all of SGLang or overwrite an installed library. Use
matching, existing CUTLASS and FlashInfer dependency checkouts from the AOT build.
No dependencies are downloaded by this target.

Build the unchanged baseline before editing the kernel:

```bash
cmake -S test/manual/layers/moe/mxfp4a8 -B /tmp/mxfp4a8-baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXPERIMENT_NAMESPACE=mxfp4a8_baseline \
  -DCUTLASS_ROOT=/path/to/repo-cutlass-src \
  -DFLASHINFER_ROOT=/path/to/repo-flashinfer-src
ninja -C /tmp/mxfp4a8-baseline -n
ninja -C /tmp/mxfp4a8-baseline -j 6
```

After updating the source, configure a separate `/tmp/mxfp4a8-candidate` build
with `-DEXPERIMENT_NAMESPACE=mxfp4a8_candidate`. Do not rebuild the baseline
directory from modified source. Keep its commit, compiler flags and library
SHA256 with the results.

```bash
python test/manual/layers/moe/mxfp4a8/bench.py \
  --baseline /tmp/mxfp4a8-baseline/libmxfp4a8_baseline.so \
  --candidate /tmp/mxfp4a8-candidate/libmxfp4a8_candidate.so \
  --warmup 10 --iters 100 --output /tmp/mxfp4a8-paired.json
```

Run again with `--graph` for graph capture/replay, `--routing skewed` for empty
experts and routing imbalance, and `--clamp 7 --routed-scale 0.5` for optional
activation/finalization semantics. `--configs GEMM1 GEMM2` overrides only the
candidate tactics. Default tactics reproduce the upstream production runner.

The benchmark compares intermediate data and final output bitwise before timing
and rejects nonfinite BF16/FP32 outputs. Each variant owns independent workspaces.
Timing alternates baseline/candidate order and includes metadata, input
quantization, GEMM1, SwiGLU/quantization, GEMM2 and the unchanged ordered combine.
Weight preprocessing and workspace allocation are outside timing. Graph
validation changes inputs and routing at fixed addresses before replay.

These are isolated A/B diagnostic results. They are not a substitute for the
production AOT build, INT4 regression suite, model accuracy evaluation, or the
separate-container FlashInfer W4A8 acceptance benchmark.
