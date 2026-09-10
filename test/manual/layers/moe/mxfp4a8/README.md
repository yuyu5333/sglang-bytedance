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

For a distinct namespace without recompiling CUDA, pass
`-DREUSE_CUDA_OBJECTS="/absolute/first.cu.o;...;/absolute/fifth.cu.o"` to a
separate build directory. Supply all five verified CUDA objects from matching
toolchain builds; only `bindings.cpp` uses `EXPERIMENT_NAMESPACE`.
Record the object paths and hashes with the resulting library. This supports
an incremental stable-patch baseline, for example unchanged GEMM objects plus
the verified SwiGLU object. Do not rebuild source object directories while
linking, or mistake the current source HEAD for the reused objects' provenance.

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
`--baseline-configs GEMM1 GEMM2` supports direct tactic comparisons.
`--production-configs` loads the candidate's actual pure dispatch method via AST,
without importing the installed `sgl_kernel` package.

The benchmark compares intermediate data and final output bitwise before timing
and rejects nonfinite BF16/FP32 outputs. Each variant owns independent workspaces.
Atomic routing can change expert-local row order. The checker verifies both
permutations and compares intermediates in original token-route order.
Timing alternates baseline/candidate order and includes metadata, input
quantization, GEMM1, SwiGLU/quantization, GEMM2 and the unchanged ordered combine.
Weight preprocessing and workspace allocation are outside timing. Graph
validation changes inputs and routing at fixed addresses before replay.

Validate the real Python wrapper, with process-local operator redirection:

```bash
PYTHONPATH=python python test/manual/layers/moe/mxfp4a8/check_runner.py \
  --baseline /tmp/mxfp4a8-baseline/libmxfp4a8_baseline.so \
  --candidate /tmp/mxfp4a8-candidate/libmxfp4a8_candidate.so
```

`flashinfer_bench.py` runs a standalone, same-format W4A8 comparison using the
same random inputs. Run each `--tokens M` in a new process with its own `--cache`
and `--output` paths; it never loads the CUTLASS experiment libraries.

These are isolated A/B diagnostic results. They are not a substitute for the
production AOT build, INT4 regression suite, model accuracy evaluation, or the
separate-container FlashInfer W4A8 acceptance benchmark.
