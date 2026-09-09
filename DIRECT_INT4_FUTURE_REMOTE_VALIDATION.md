# Direct INT4 Future: Remote Validation

## Status

Kernel validation and a fresh model A/B are complete on H20. With the common
configuration below, default Direct INT4 increases the token-pool capacity
29.27%, but reduces output throughput 16.72% versus native FP8 KV. GSM8K differs
by one correct answer in this single paired evaluation, which does not
establish accuracy equivalence. This document supersedes the local-only
verification status in `DIRECT_INT4_FUTURE.md`.

The subsequent producer optimization and fresh before/after INT4 comparison
are recorded in [Round 2](DIRECT_INT4_FUTURE_ROUND2.md). The measurements below
remain the original round-1 results.

- Host: `115.190.141.215`, hostname `iv-yehwog4ni84c5qw9eqe0`.
- Hardware: eight NVIDIA H20 GPUs, 97,871 MiB reported memory per GPU.
- Kernel-validation container: `kvbit-ds`.
- Fresh A/B container: `kvbit-future-ab-20260908`.
- Tested source commit: `276e514d01`.
- Branch: `feat/dsv4-direct-int4-g32-e4m3-future`.
- Source: `/sgl-workspace/sglang-bytedance`; clean working tree.
- Explicit `PYTHONPATH=/sgl-workspace/sglang-bytedance/python`.
- Host model: `/mnt/nvme2/DeepSeek-V4-Flash`.
- The existing container maps that model to
  `/data00/nvme2/DeepSeek-V4-Flash`.
- A/B source: `/sgl-workspace/dsv4-future-ab`; model:
  `/models/DeepSeek-V4-Flash`, read-only mount. Explicit A/B
  `PYTHONPATH=/sgl-workspace/dsv4-future-ab/python`.
- A/B endpoint: `127.0.0.1:31080`; both services have been stopped after testing.

No SGLang or KVBit source was edited in a remote container. Source changes
were committed and pushed locally, then pulled with `--ff-only`.

## Validation Matrix

| Check | Scalar | BF16x2 |
|---|---|---|
| Focused CMake build, CUDA 13.0.88 | Pass | Pass |
| GPU numerical, writer-byte, and graph conformance | 4 tests and 30 subtests pass | 4 tests and 30 subtests pass |
| Final memcheck, CUDA API reporting disabled | 0 errors | 0 errors |
| Final racecheck | 0 errors, 0 warnings | 0 errors, 0 warnings |
| Final synccheck | 0 errors | 0 errors |
| 100 repeated high-logit split-K comparisons | Pass, maximum absolute output error 0.0009765625 | Not repeated |
| Model smoke / CUDA Graph startup | Pass, including target verify and both draft graphs | Not scheduled |
| Same-configuration performance / GSM8K | Complete; results below | Not scheduled |

The final combined CPU regression reports **77 passed, 58 subtests passed**.
The host C++ math test covers 120 sink/LSE combinations. Pre-commit checks
pass. No coverage threshold was requested or configured for this task; no
coverage percentage is claimed.

The newly added TP-padding regression is one test method with five supported
geometry subcases; three of them failed before the startup-validator fix.
Existing CPU and GPU tests were retained. Final all-pass verification did not
relax any numerical assertion. Test preparation and report flush completed.

GPU cases cover both 368/384-byte layouts, SWA plus C4/C128 sources, invalid
indices, noncontiguous source rows and location vectors, multiple query
positions, forced split/no-split scheduling, sink-only rows, and graph replay
after query mutation. The strided norm/RoPE case compares strided and contiguous
inputs to the same Triton implementation. It does not establish bitwise
equivalence with the previous JIT norm/RoPE implementation.

## Failures and Changes

### Tensor Parallel Head Padding

The checkpoint has 64 total query heads. At TP8, each rank has eight real
heads, but `DeepseekV4Attention.forward` and `_local_attn_sink` pad the kernel
input to 64. The initial future-branch validator incorrectly rejected this
supported configuration.

Regression cases first reproduced the rejection. Commit `858a766231` changes
the check to follow the model's padding contract. It retains rejection of
invalid divisibility, zero heads, unsupported unpadded TP1 geometry, and more
than 64 local heads.

### Focused Build

The normal top-level AOT configuration downloads unrelated dependencies before
configuring FlashMLA. Commit `b5b1b8cde6` adds the default-off
`SGL_KERNEL_DSV4_INT4_ONLY` option. It uses the normal pinned FlashMLA and
CUTLASS FetchContent declarations, Torch ABI, and private installation
component. It does not change the default full build.

Pinned dependencies:

| Dependency | Commit | Archive SHA256 |
|---|---|---|
| FlashMLA | `c1dee569a494b184811a08171a690ece21420262` | `77d3f1714b5903dc8f7a99fbc3a5d9a2b886e449f994b6c4b1d2feeacb467b1e` |
| CUTLASS | `147f5673d0c1c3dcf66f78d677fd647e4a020219` | `9f6c53320a85b4a570975e557918cde65168cd311f081920446c238437347dc6` |

The first focused build reused cache entries after checking these exact pins
and hashes. Old installed wrapper/extension files in `kvbit-ds` were not
overwritten; tests loaded isolated CMake installations and checked ABI 2.

Equivalent final scalar build, from the AOT source directory in `kvbit-ds`:

```bash
cmake -S . -B /tmp/kvbit-future-20260908-lineinfo-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake \
  -DSKBUILD_SABI_VERSION=3.9 \
  -DSKBUILD_SABI_COMPONENT=Development.SABIModule \
  -DSGL_KERNEL_DSV4_INT4_ONLY=ON \
  -DSGL_KERNEL_ENABLE_DSV4_INT4=ON \
  -DSGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT=OFF \
  -DFETCHCONTENT_BASE_DIR=/sgl-workspace/sglang-bytedance/python/sglang/kernels/aot/build/_deps \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas=-v,-warn-spills"
cmake --build /tmp/kvbit-future-20260908-lineinfo-build \
  --target kvbit_flashmla_ops -j 4
cmake --install /tmp/kvbit-future-20260908-lineinfo-build \
  --component kvbit_flashmla --prefix /tmp/kvbit-future-20260908-final-scalar
```

BF16x2 uses its own build/install directories and switches
`SGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT=ON`; it was built without `-lineinfo`.

### Cross-Warpgroup Synchronization

Numerical tests alone passed before the synchronization changes. Sanitizers
found additional evidence:

| Revision | synccheck | racecheck |
|---|---|---|
| `b5b1b8cde6`, original consumer barriers | Divergent-thread barrier errors | 12 reported access-pair groups involving `sScale` |
| `817cf5e4d5`, unaligned barrier operations | 0 errors | 12 reported groups remain |
| `2bdea2307b`, two-way free rendezvous | 0 errors | One reported group, 64 byte hazards |
| `276e514d01`, two-way ready and free rendezvous | 0 errors | 0 hazards, scalar and BF16x2 |

Line-info builds located the read in the second consumer warpgroup and the
write in `scale_softmax`. The original SASS locations were `LDS.64` at
`0xae50` and `STS.64` at `0x100b0`.

The consumers execute different control-flow paths. CUTLASS's ordinary named
barrier methods use aligned PTX instructions; the private kernel now uses its
unaligned variants, including the cross-warpgroup batch rendezvous. The
ready/free exchange for the shared softmax buffer now waits on both sides.
Thread counts, the two K buffers, and QK/PV WGMMA geometry are unchanged.
Native FlashMLA is not modified.

This establishes a sanitizer-clean implementation for the exercised cases.
No pre-fix numerical failure or model-quality impact was demonstrated, and
the report does not infer one from sanitizer output alone.

### CUDA API Probe Noise

Unfiltered memcheck exits 99 and reports 34
`CUDA_ERROR_INVALID_VALUE` calls to `cuGetProcAddress_v2`. A process executing
only `import sgl_kernel`, without loading this INT4 extension or running its
kernel, reproduces the same 34 errors in dependency initialization.

Final memcheck uses `--report-api-errors no`; device-memory checking, including
the default Hopper bulk-copy and WGMMA checks, remains enabled. Both the raw
unfiltered logs and the import-only control are retained. This is **not** a
claim that unfiltered memcheck passed.

## Build Resources

Both final variants target SM90a with CUDA 13.0.88 and the same pinned
dependencies. Scalar includes `-lineinfo` for diagnostic attribution.

| Main Decode Kernel | Scalar | BF16x2 |
|---|---:|---:|
| Registers reported by ptxas | 168 | 168 |
| Stack bytes | 32 | 32 |
| Spill-store bytes | 4 | 8 |
| Spill-load bytes | 4 | 16 |
| Barrier resources reported by ptxas | 13 | 13 |

These compilation statistics are not measurements of end-to-end speed or
occupancy. BF16x2 remains opt-in.

| Installed Artifact | SHA256 |
|---|---|
| Scalar | `d76709f0a0cb0efd27c488c4e62c7b3e9d56ea29d41ffeb51cdab27629525335` |
| BF16x2 | `2c4becf102ed4d2d7c2d4b1ef59b22ebd83b23a4de4c039394f367648f3a7455` |

Final CMake installations are also available on the host under
`/mnt/nvme2/kvbit-future-20260908/aot-scalar` and `aot-vector`.

## A/B Protocol

The original base image was checked in a newly created probe container. Its
packages differ from the already validated `kvbit-ds` environment:

| Package | Original Base Image | Validated Environment |
|---|---|---|
| Python | 3.12.3 | 3.12.3 |
| Torch | 2.11.0+cu130 | 2.13.0+cu130 |
| Triton | 3.6.0 | 3.7.1 |
| sglang-kernel | 0.4.5 | 0.4.6.post1 |
| Transformers | 5.12.1 | 5.12.1 |
| FlashInfer | 0.6.15.post1 | 0.6.15.post1 |

The base-image probe is stopped and supplies neither A/B result. A local
snapshot of the validated environment was created; no packages were upgraded
or downgraded. The model mount is excluded from the image. The snapshot ID is
`sha256:2a607a6e39a81912432553239d3e96b5a1399f7d8cd06fbeb061707eb01a185e`.

The fresh container has private PID/IPC namespaces, host networking, 100 GiB
shared memory, and a read-only model mount. An independent Git-protocol clone
from the snapshot's clean repository was created with `--no-local`, then its
origin was set to GitHub. Both phases ran that new checkout at `276e514d01`.
The scalar CMake component was installed only in the new container. Its
wrapper, ABI 2, extension hash, actual `sglang` import path, and GPU conformance
were checked again.

Common measured settings:

| Setting | Value |
|---|---|
| Model | Same DeepSeek-V4-Flash checkpoint |
| TP / PP / DP | 8 / 1 / 1 |
| Memory fraction | 0.8 |
| Speculative | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| Draft KV | Explicit FP8 E4M3 in both phases |
| CUDA Graph | Target verify, draft decode and draft extend enabled, maximum batch 32; ordinary prefill graph disabled in both |
| Requests / maximum concurrency | 128 / 32 |
| Performance input / output | 4096 / 256 tokens, `random-ids`, `--tokenize-prompt`, `--random-range-ratio 1` |
| Request rate / temperature | Infinite / 0 |
| Server / benchmark seed | 42 / 42 |
| Repetitions | Three per KV format; 32 warmup requests, then flush prefix cache before every measurement |
| A | Native FP8 E4M3 target KV |
| B | Direct INT4 G32 E4M3 target KV, scalar, `aos_368`, strided norm off |
| GSM8K | Repository legacy 8-shot evaluator, 1319 questions, max 1024 output tokens, parallel 32, temperature 0 |

Both phases set `SGLANG_DEFAULT_THINKING=1`,
`SGLANG_DSV4_FP4_EXPERTS=1`, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`,
`SGLANG_DSV4_INT4_LAYOUT=aos_368`,
`SGLANG_DSV4_INT4_STRIDED_NORM_ROPE=0`, `GLOO_SOCKET_IFNAME=eth0`,
`NCCL_MIN_NCHANNELS=24`, and `NCCL_IB_QPS_PER_CONNECTION=8`.
Both use the Marlin MoE backend and an 8192-token prefill chunk.
No weights were converted for either phase.

Manifests match in source commit, model-config hash, package versions,
environment, extension ABI and hash. Service commands differ only in target
KV dtype. A recursive comparison of resolved server configuration found only
the expected KV dtype and measured startup/memory/capacity differences.
Every measured run completed 128 requests, exactly 524,288 input tokens and
32,768 output tokens, with no request errors.

An initial runner incorrectly used `--random-range-ratio 0`, which this
benchmark implements as variable length, and decoded random IDs back to text.
Those two completed runs are retained only in `/artifacts/fp8`; the third was
stopped. None are used here. The corrected runner uses ratio 1, passes integer
IDs directly, and asserts every request's input/output length. All results
below are fresh runs after that correction.

The legacy evaluator takes its eight exemplars from the beginning of the test
file. Report both its full 1319-question score and the 1311-question score
excluding those exemplars. Per-question outputs are retained for paired
comparison. This protocol is not presented as an independent standard
train-exemplar GSM8K evaluation.

Only same-configuration fresh results can support the A/B conclusion. Historical
results from other commits are not used as the baseline. Report total output
throughput and its division by eight GPUs separately; neither establishes a
globally optimal TP configuration.

## Measured Results

These are three-run medians unless stated otherwise. Latency rows take the
median of the three per-run means. Configuration is the fixed-workload,
fixed-memory-fraction protocol above, not an equal-token-capacity experiment.

| Metric | Native FP8 KV | Direct INT4 KV | INT4 Change |
|---|---:|---:|---:|
| Total output throughput, tokens/s | 588.22 | 489.87 | -16.72% |
| Output throughput / 8 GPUs, tokens/s/GPU | 73.53 | 61.23 | -16.72% |
| Mean TPOT, ms | 45.64 | 54.71 | +19.89% |
| Mean TTFT, ms | 2066.02 | 2507.44 | +21.37% |
| Mean end-to-end latency, ms | 13694.10 | 16461.87 | +20.21% |
| Speculative accept length | 2.6621 | 2.7273 | Diagnostic, not an isolated cause |
| Full token-pool capacity, startup log | 6,157,312 | 7,959,296 | +29.27% |
| SWA token-pool capacity | 615,680 | 795,904 | +29.27% |
| Target-verify graph memory, GiB, TP0 | 0.719 | 0.764 | +0.045 GiB |
| Target packed row bytes | 584 | 368 | -36.99% |

| Repeat | FP8 Output Tokens/s | INT4 Output Tokens/s | FP8 TPOT ms | INT4 TPOT ms |
|---|---:|---:|---:|---:|
| 1 | 585.27 | 487.46 | 45.66 | 54.90 |
| 2 | 592.53 | 490.11 | 45.28 | 54.61 |
| 3 | 588.22 | 489.87 | 45.64 | 54.71 |

The smaller packed row does not translate into the same percentage capacity
gain because indexer, compression state, draft pools and other storage remain.
INT4 logs confirm `packed_row=368`, all 43 target layers, 21 C4 layers, 20 C128
layers, and `scratch=disabled`. All eight scheduler processes map the rebuilt
`kvbit_flashmla_ops.abi3.so`. The service parent has the explicit A/B
`PYTHONPATH` and working directory. Scheduler process-title rewriting leaves
their `/proc` environment snapshots empty, so those snapshots are not used as
import-path evidence.

### Accuracy

Input prompts are 1261-1424 tokens, identical for corresponding questions.
The FP8 phase generated 125,569 output tokens; INT4 generated 126,946.
Output lengths are variable, bounded at 1024. Both phases use 8-shot prompts,
temperature 0, parallelism 32, TP8, EAGLE and the graph settings above.

| Metric | Native FP8 KV | Direct INT4 KV |
|---|---:|---:|
| Full legacy GSM8K | 1260/1319, 95.5269% | 1259/1319, 95.4511% |
| Excluding eight exemplars | 1252/1311, 95.4996% | 1251/1311, 95.4233% |
| Invalid parsed answers | 0 | 1 |
| Responses reaching 1024-token limit | 0 | 1 |
| Request retractions | 0 | 0 |

Paired correct/incorrect outcomes: FP8 alone is correct on 12 questions; INT4
alone on 11. The full-score delta is -0.0758 percentage points. Exact two-sided
McNemar p=1.0 does **not** establish equivalence. There was one accuracy run per
format and no repeated FP8 control for batching-related generation variation.
Predicted numeric answers match on 1288/1319 questions; complete text matches
on 449/1319.

The single invalid INT4 output is zero-based question 1176. It continues a
geometric chalk-percentage calculation until the 1024-token limit. Its final
partial decimal ends in `3.023`; the legacy integer-regex parser extracts
`023`, which `ast.literal_eval` rejects. The response itself is a successful
generation with `finish_reason.type=length`, not an HTTP/CUDA failure. FP8
generates a parseable correct answer in 160 tokens. The raw result and the
unchanged evaluator score are retained.

### Interpretation and Limits

- Default Direct INT4 is functional in this model/configuration, but is a
  capacity-for-speed tradeoff here, not a throughput improvement over native
  FP8 KV.
- The decode producer reads fewer persistent bytes but expands selected rows
  into BF16 shared-memory tiles and performs explicit dequantization.
  Compilation still reports register spills and a large shared-memory plan.
  These are plausible cost sources, not a measured bottleneck attribution.
- Speculative accept-length ranges overlap; a simple claim that lower
  acceptance caused the slowdown is not supported by these results.
- No NCU/NSight/Perfetto profile, isolated producer timing, memory-bandwidth
  attribution, or equal-capacity run was collected. The costs of unpacking,
  synchronization, writer work and BF16 attention are not separated.
- The A/B order was FP8 then INT4, three runs each, not randomized/interleaved.
  Both phases ran without other observed GPU workloads.
- BF16x2, 384-byte stride and strided norm/RoPE have kernel-level tests, but no
  end-to-end performance/accuracy comparison here. Their defaults remain off.
- No old INT4 commit was rerun; this experiment cannot quantify a speedup
  over the earlier implementation.

## Evidence Locations

- Container build/test/sanitizer logs:
  `kvbit-ds:/tmp/kvbit-future-20260908-logs`.
- Local log archive:
  `../validation/dsv4-future-20260908/build-test-sanitizer-logs.tar.gz`.
- Local phase runner:
  `../validation/dsv4-future-20260908/run_phase.py`.
- Local manifest-checking comparator:
  `../validation/dsv4-future-20260908/compare_phases.py`.
- Host A/B artifacts: `/mnt/nvme2/kvbit-future-20260908`.
- Fresh raw phase directories: `fp8-fixed` and `int4-fixed`, under both the
  host artifact directory and local `../validation/dsv4-future-20260908`.
- Local validated summary: `../validation/dsv4-future-20260908/comparison.json`.
- Each phase includes the manifest, resolved server configuration, smoke
  response, launch/benchmark commands, complete server log, three detailed
  performance JSONL files, GSM8K metrics and 1319 answer records.

The A/B container remains available but idle. Its server process groups have
exited, and `nvidia-smi` reports no compute processes. The stopped base-image
and entrypoint probe containers are retained, not deleted. Other users'
containers and the original `kvbit-ds` installation were not stopped or
modified.
