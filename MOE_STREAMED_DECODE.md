# Output-Owned Streamed SIMT MoE Decode

Date: 2026-09-14. Branch: `feat/moe-direct-decode-streamed`.
Base: `366a6d331c8565d8a8b39915e617545ea06a931c`.
Kernel implementation: `447c642d87a4d088369fbc88ea7da37d1f6957cc`.
Final tested harness: `109a00cca58a0154ae9fd4cf70343a399bd32f7d`.

## Decision

**Keep this as a low-level experiment; do not change the runner selector.**
Explicitly selected launch parameters reduce BF16 M1 graph latency at
H4096/I512/E256/top-k6 by **18.40%-19.04%** versus original Triton across three
GPU1 seeds. The result does not generalize to all M1 shapes or to FP16.
M4/M8 uniform routing improves, but hot routing is **45.62%/113.79% slower**
than original Triton. These are synthetic operator results, not model results.

A same-process, same-tile ablation on idle GPU2 isolates an additional
**7.06% at M1 and 7.98% at M4** reduction from replacing the old vectorized
down implementation with the streamed version. Tile changes also contribute;
the full gain relative to the old default direct path is not attributed to
route streaming alone.

The original direct kernel, grouped kernel, runner, selector and environment
switches are unchanged from the base. `SGLANG_MOE_DIRECT_DECODE` still defaults
off and, when eligible, calls the old M1 direct implementation, not this one.
There is no runtime routing readback, online tuner, new automatic selector,
checkpoint accuracy claim, or serving-throughput claim.

CPU tests, numerical checks, changed-route graphs, two-stream checks and eight
filtered device sanitizer runs pass. Unfiltered memcheck exits 86 with 34
import-time CUDA API errors. This is not an unqualified sanitizer all-pass.

## Hypothesis And Implementation

The [previous SIMT direct kernel](MOE_DIRECT_DECODE.md) combines route,
output-feature and reduction dimensions in its down kernel. This experiment
changes only the down reduction algorithm and exposes launch parameters for
offline testing:

1. Reuse `_gate_up`: each route reads its expert weights directly, computes
   gate/up in FP32, applies the dtype roundtrip and SiLU, and writes activation.
2. `_streamed_down_sum` assigns one CTA to one token/output-feature tile.
   It processes routes in a loop, loads a two-dimensional `[BN, BK]` weight
   tile per route, computes down, adds bias, applies router weight, rounds
   each route to output dtype, and accumulates into an FP32 `[BN]` result.
3. Apply the final scale and store output once.

This retains **two kernel launches**. It uses SIMT reductions, not tensor-core
matrix multiplication. There is no sorting, route-output tensor, atomic
addition, global lock, weight repacking or persistent shared scratch.
Loop unrolling can keep more than one route's compiler state live; source
tensor shape alone does not establish physical register allocation.

Call-private allocations, for FP16/BF16:

```text
activation scratch: 2 * M * top-k * I bytes
separate output when not inplace: 2 * M * H bytes
```

The down kernel never reads `x`, so inplace writes occur after the first
kernel has consumed the input in the same stream. Invalid expert IDs mask
weight, activation, bias and router-weight loads. The shared metadata gate
allows contiguous same-device CUDA FP16/BF16 tensors, M1..8, H128..8192,
I128..4096, top-k1..8, INT32/INT64 IDs, FP32 router weights, and optional
matching biases. It does not read GPU routing values on the host.

Gate/up is rounded before SiLU, activation is materialized in input dtype,
and each weighted down route is rounded before FP32 top-k summation.
`enable_fp_fusion=False` remains set. Reduction order can differ; bitwise
equivalence is not promised. This work does not add quantized or distributed
execution, special SwiGLU variants, or production dispatch integration.

### Selected Parameters Are Not Defaults

| Configuration | Up N | Down N | Up/Down Warps | Route Unroll |
|---|---:|---:|---|---:|
| Wrapper and CLI defaults | 4 | 16 | 4/4 | 1 |
| Selected streamed experiment | 16 | 16 | 4/4 | 2 |
| Same-tile vectorized control | 16 | 16 | 4/4 | Not used |
| Old direct implementation | 4 | 4 | 4/4 | Not used |

All selected-result tables below require explicit `up_n=16, down_n=16,
up_warps=4, down_warps=4, unroll=2`. They are not default-call speedup claims.
`vectorized_down=True` reuses the old `_down_sum` at the requested tile;
`--ablate-vector` runs that control alongside streamed in the same process.

## Environment And Measurement

| Item | Actual Value |
|---|---|
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| Dedicated container | `kvbit-future-ab-20260908` |
| Repository | `/sgl-workspace/sglang-bytedance` |
| Actual package import | `/sgl-workspace/sglang-bytedance/python/sglang/__init__.py` |
| Actual candidate import | `<repo>/python/sglang/kernels/ops/moe/streamed_decode.py` |
| Devices | H20, SM90, 78 SMs, 60 MiB L2 |
| Primary matrix / replication | Physical GPU1, `GPU-a6568ac4-4e7d-6432-49af-b4fdeb771eaa` |
| Paired ablation / unfiltered check | Physical GPU2, `GPU-159d2b76-8430-73da-3de3-3c781ddeaf28` |
| Earlier pilot / search / matrix | Physical GPU0, `GPU-1c2c22aa-44de-d80b-fff2-03e1842c1b32` |
| Python / Torch / Triton | 3.12.3 / 2.13.0+cu130 / 3.7.1 |
| CUDA compiler / sanitizer | 13.0.88 / 2025.3.1.0 |
| Host driver / loaded CUDA | 535.161.08 / compat `libcuda.so.580.82.07` |
| Loaded AOT binary | `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm90/common_ops.abi3.so` |
| Model | Synthetic unquantized MoE; no checkpoint loaded |
| Main workload | H4096/I512/E256/top-k6, M listed per row |
| Main math | BF16 weights/activations, plain SiLU, scale1, no bias/mask, outplace |
| Parallelism | TP=EP=PP=DP=1; one measured operator call at a time |
| Routing | Uniform distinct top-k, or hot: all tokens choose experts 0..5 |
| Runtime | Explicit `PYTHONPATH=<repo>/python`, selected `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`, `HF_HUB_OFFLINE=1` |
| Main timing | 20 warmups, 7 rounds x 200 calls, randomized implementation order |
| Statistic | Median of per-round CUDA-event mean latency |
| CUDA Graph | Enabled in primary tables; eager results also retained |
| Not applicable | Sequence input/output lengths, request concurrency/count, mem-fraction, KV dtype, speculative decoding, HTTP, TTFT/TPOT |

Within every comparison process, implementations share input, weights, routes,
dtype, device, scale and timing protocol. Initial routing is restored before
timing. Graphs reuse addresses and weights; this is not a cold-cache multilayer
serving workload. Eager timings include Python and launch gaps.

The selected GPU was checked at 0 MiB before each launch. All 50 recorded
experiment processes are non-overlapping. GPU0 was later occupied by an external
VLLM service, so the complete six-case matrix was rerun on GPU1. GPU1 was later
also occupied; paired ablations used GPU2. Preflight refusals launched no
benchmark. Neither external service was stopped or changed. The host was not
wholly exclusive, and measurements from different GPUs are not pooled.
No packages, installed binaries, clocks or GPU reset state were changed.

Original Triton uses runtime config resolution, including the existing 3.5.1
H20 fallback when 3.7.1 tuning files are absent. Down TMA remains enabled.
Neither baseline settings nor tuning JSON files were changed:

| M | Up M/N/K | Up Warps/Stages | Down M/N/K | Down Warps/Stages |
|---:|---|---|---|---|
| 1 | 16/64/128 | 4/4 | 16/32/256 | 4/2, TMA |
| 4 | 16/64/64 | 4/4 | 16/32/256 | 4/2, TMA |
| 8 | 16/64/64 | 4/3 | 16/32/256 | 4/2, TMA |

## Fresh GPU1 Results

Latency is microseconds. Positive change means slower than original Triton.
Grouped uses N32/K256 from the [grouped experiment](MOE_GROUPED_DECODE.md);
streamed uses the selected parameters above. Seed42:

| M | Routing | Original Triton | Old Direct | Grouped | Streamed | Streamed Change |
|---:|---|---:|---:|---:|---:|---:|
| 1 | uniform | 39.1555 | 37.1566 | 42.9411 | 31.9512 | -18.40% |
| 1 | hot | 39.2123 | 37.1947 | 43.0578 | 31.9560 | -18.51% |
| 4 | uniform | 97.7042 | 108.3682 | 128.1688 | 93.9024 | -3.89% |
| 4 | hot | 42.9869 | 81.4941 | 45.0784 | 62.5960 | +45.62% |
| 8 | uniform | 195.9251 | 207.4293 | 246.5501 | 176.2362 | -10.05% |
| 8 | hot | 53.9226 | 151.3973 | 50.4861 | 115.2803 | +113.79% |

M8 uniform has wider round variation: original 194.6694..211.4928 us,
streamed 171.9589..183.3795 us. Only M1/M4 uniform received additional
seed replication; M8 is not a three-seed result.

### Uniform-Routing Replication

Each seed changes all random tensors. Every row is its own same-input A/B:

| M | Seed | Original us | Old Direct us | Streamed us | Reduction vs Original | Streamed Round Range us |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 42 | 39.1555 | 37.1566 | 31.9512 | 18.40% | 31.8936..31.9824 |
| 1 | 7 | 39.7629 | 37.5736 | 32.2403 | 18.92% | 32.1240..32.2520 |
| 1 | 123 | 39.8427 | 37.6158 | 32.2550 | 19.04% | 32.2126..32.2800 |
| 4 | 42 | 97.7042 | 108.3682 | 93.9024 | 3.89% | 93.8685..94.3896 |
| 4 | 7 | 100.1978 | 109.3730 | 94.8597 | 5.33% | 94.5214..95.4698 |
| 4 | 123 | 98.5757 | 109.6342 | 95.1173 | 3.51% | 94.5490..95.7442 |

### Negative Shape And Dtype Checks

These use selected streamed parameters, seed42 and the same timing protocol.
The BF16 narrower-shape comparison is independent of the main-shape result:

| BF16 Shape | Original us | Old Direct us | Streamed us |
|---|---:|---:|---:|
| M1/H2048/I1024/E64/top-k4 | 28.7552 | 20.3038 | 31.6187 |

The FP16 comparison is independent of all BF16 results:

| FP16 Shape | Original us | Old Direct us | Streamed us |
|---|---:|---:|---:|
| M1/H4096/I512/E256/top-k6 | 39.9355 | 37.8710 | 40.0427 |

Both pass numerical checks, but neither supports adopting this launch
configuration. There is no claim that M1 alone is a sufficient selector.

### Allocation And Eager Behavior

Eager call peak includes output and explicit scratch, excluding already resident
weights and caller input. It is not process memory or KV capacity:

| M | Original KiB | Old Direct KiB | Grouped KiB | Streamed KiB |
|---:|---:|---:|---:|---:|
| 1 | 345.5 | 14 | 62 | 14 |
| 4 | 1378.5 | 56 | 248 | 56 |
| 8 | 2756 | 112 | 496 | 112 |

Streamed does not reduce allocation relative to old direct. For M1 uniform,
eager CUDA-event latency was 297.0602 us original, 50.7939 us old direct and
51.4936 us streamed. Thus the M1 graph gain does not imply a measured eager
gain over old direct; host launch overhead can dominate this short operator.

## Same-Process GPU2 Ablation

Both candidates share Up N16, Down N16 and 4/4 warps. The vectorized control
uses old `_down_sum`; streamed uses `_streamed_down_sum`, unroll2. Main BF16
shape, uniform routes, seed42, seven rounds x 200 calls:

| M | Original us | Old Default Direct us | Same-Tile Vectorized us | Streamed us | Reduction vs Vectorized |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.4464 | 37.1490 | 34.1904 | 31.7765 | 7.06% |
| 4 | 98.4472 | 108.9923 | 102.8760 | 94.6622 | 7.98% |

Vectorized/streamed round ranges are 34.1589..34.2018 / 31.7024..31.8202 us
at M1, and 102.3328..103.5736 / 93.9790..96.2054 us at M4.
Numerical, changed-route graph and two-stream checks pass for all four paths.
These absolute times are not compared against GPU1 measurements.

Tile changes alone reduce old direct latency by 7.96% at M1 and 5.61% at M4
in these processes. Replacing down at the same tile adds the reductions above.
This isolates an implementation/configuration effect, not a single hardware
counter or a dtype-independent algorithmic guarantee.

### Trace Evidence And Limits

Single-call GPU2 M4 trace diagnostics, not timing-round medians:

| Path | Up us | Down us | Up Registers/Thread | Down Registers/Thread | Up Grid | Down Grid |
|---|---:|---:|---:|---:|---|---|
| Old default direct | 65.089 | 45.216 | 62 | 62 | 24 x 128 | 4 x 1024 |
| Same-tile vectorized | 59.264 | 43.777 | 115 | 136 | 24 x 32 | 4 x 256 |
| Streamed | 63.457 | 34.080 | 115 | 72 | 24 x 32 | 4 x 256 |

The same-tile down comparison supports reduced compiled register use,
136 to 72, and shared memory falls from 2048 to 1024 bytes. However, selected
streamed uses more registers per thread than old default down (72 vs 62);
Up N16 also increases registers (115 vs 62). The original hypothesis cannot
be reported as a general register-pressure reduction versus the old default.
The two up measurements use the same implementation/configuration and their
single-call variation must not be attributed to the down algorithm.

Larger tiles reduce CTA count fourfold. Streaming removes the padded route
tensor dimension in the source, but it does not recover cross-token expert
weight reuse. Original Triton groups tokens, whereas direct/streamed reread
weights for each route. For main BF16 shape, one expert's gate/up/down weights
total 12 MiB. M4 uniform has 24 unique experts; hot routing has only six.
These logical working sets help explain why route distribution matters,
but are not measured DRAM traffic. The precise cause of the FP16 and narrower
shape regressions has not been isolated.

No new NCU measurement or measured occupancy/traffic claim is made.
`trace_processor` was not on PATH; Chrome trace JSON was parsed directly.
Profiler occupancy estimates are not treated as hardware-counter evidence.

### Earlier Offline Diagnostics

The twelve GPU0 M4-uniform candidate-only runs used three rounds x 100 calls.
They are retained as search diagnostics, not final paired A/B:

| Up N | Down N | Up/Down Warps | Unroll | Down Algorithm | Graph us |
|---:|---:|---|---:|---|---:|
| 4 | 4 | 4/4 | 1 | streamed | 102.2899 |
| 4 | 8 | 4/4 | 1 | streamed | 100.0093 |
| 4 | 32 | 4/4 | 1 | streamed | 104.4762 |
| 4 | 64 | 4/4 | 1 | streamed | 102.0445 |
| 4 | 16 | 4/4 | 2 | streamed | 98.8067 |
| 4 | 16 | 4/4 | 8 | streamed | 100.4595 |
| 4 | 8 | 4/4 | - | vectorized | 103.1795 |
| 4 | 16 | 4/4 | - | vectorized | 106.8499 |
| 8 | 16 | 4/4 | 1 | streamed | 115.9299 |
| 16 | 16 | 4/4 | 1 | streamed | 96.2448 |
| 8 | 16 | 8/4 | 1 | streamed | 101.5142 |
| 4 | 16 | 4/8 | 1 | streamed | 104.9472 |

Earlier GPU1 separate-process controls are also retained: M1 vectorized
Up N16 measured 34.7048 us; M4 vectorized Up N16 measured 103.0826 us;
M1 streamed Up N4/unroll2 measured 37.6790 us. The GPU2 paired table, not
cross-process subtraction of these values, is the attribution evidence.
The earlier six GPU0 A/B runs remain in `ab-*`; primary results use `gpu1-ab-*`.

## Validation And Failures

### CPU

New `test_streamed_decode.py`: **35 passed on the first run**. It loads the
actual wrapper and metadata gate and records allocation/device/launch
dependencies; it does not emulate GPU arithmetic. Cases cover metadata and
tile rejection before allocation, launch order/grid/kwargs, exclusive `BT`
versus `UNROLL`, bias/scale forwarding, inplace output, call-private scratch,
both down variants and boundary shapes.

Final combined rerun at `109a00cca5`: **470 passed**, two expected macOS
platform warnings, 24.80 s. This includes the previous 435 cases. The
seven-step unit-test workflow and `utree flush` completed; coverage statistics
were skipped under the no-gate/non-flux rule. No assertions or tolerances
were weakened.

### GPU Numerical Matrix

All use unchanged `atol=0.003, rtol=0.03`, selected streamed parameters and
an independent reference. Each checks five changed-routing graph replays;
non-inplace cases also check two independent calls on separate CUDA streams.

| Case | Coverage |
|---|---|
| `ragged` | M3/H288/I160/E64/T3, FP16, bias, scale1.7, INT64, duplicate expert, masked route |
| `topk1-inplace` | M1/H1024/I512/E64/T1, FP16, bias, scale0.5, inplace |
| `all-masked` | M4/H256/I128/E64/T3, BF16, bias, initially all IDs -1, later valid and masked tokens |
| `zero-scale` | M8/H1024/I512/E64/T8, BF16, scale0, masked token |
| `max-geometry` | M8/H8192/I4096/E8/T8, BF16, hot and duplicate routes |
| Timed runs | Main uniform/hot matrix, seed replication, tile controls, narrower shape and FP16 |

Maximum-geometry reference max-absolute error is 0.0009765625. Graph checks
change routing membership, reuse distribution and router weights while keeping
addresses fixed. They do not establish accuracy for a real checkpoint.
GPU sanitizer coverage is for the selected launch configuration, not every
legal combination of all exposed tile/warp/unroll options.

### Sanitizers

| Scope | Tools | Process Result |
|---|---|---|
| GPU1 FP16 M3/H288/I160/E64/T3, INT64, duplicate, masked token, bias | memcheck / racecheck / synccheck / initcheck | Four exit0; zero errors; racecheck zero warnings |
| GPU1 BF16 M8/H4096/I512/E256/T6, hot, duplicate, masked route, bias | memcheck / racecheck / synccheck / initcheck | Four exit0; zero errors; racecheck zero warnings |
| GPU2 FP16 ragged, same selected parameters, API reporting unfiltered | memcheck | Exit86; 34 CUDA API errors |

The eight zero-error runs explicitly use `--report-api-errors no`. They
support scoped device access/race/synchronization/initialization checks,
not a clean CUDA API environment. In the unfiltered run, all 34 reported
errors are `CUDA_ERROR_INVALID_VALUE` from `cuGetProcAddress_v2` in import
stacks. Numerical/graph/two-stream checks still return true, but aggregate
acceptance is false because the process exits 86. The failure is retained;
no package change was attempted to suppress it.

Ledger: **50 GPU experiment processes, 49 successful and one retained
unfiltered API failure**. This is process accounting, not a statistical
accuracy rate. Preflight SSH/GPU-busy refusals are separately recorded.
The old AOT alignment barrier and Marlin shared-memory race described in
[workspace validation](MOE_WORKSPACE_CUDA_VALIDATION.md) are not fixed or
revalidated by the streamed-only sanitizer cases.

## Reproduction And Artifacts

| Commit | Purpose |
|---|---|
| `447c642d87` | Streamed kernel, configurable launches and initial validation entry |
| `109a00cca5` | 35 CPU cases and same-process vectorized ablation |

Kernel source SHA256, unchanged throughout all 50 runs:

```text
3d1c1dc6e5c23f20165153971cd21a09fca2045c246c3350236f227dfef276c2
```

Choose an idle GPU and a new output name before reproducing. Example from
the container, with physical GPU2 selected explicitly:

```bash
env PYTHONPATH=/sgl-workspace/sglang-bytedance/python \
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  python3 /sgl-workspace/sglang-bytedance/benchmark/kernels/fused_moe_triton/validate_direct_decode.py \
  --streamed --ablate-vector --tokens 1 \
  --up-n 16 --down-n 16 --up-warps 4 --down-warps 4 --unroll 2 \
  --warmup 20 --iterations 200 --repeats 7 --profile \
  --output /artifacts/moe-streamed-decode-20260914/new-paired-m1.json
```

Change only `--tokens` and output name for M4 paired replication. Add
`--grouped --block-k 256` and omit `--ablate-vector` for the four-way matrix.
The repository-external `run_remote.py` helper records the exact command,
source commit, GPU status/UUID, timestamps, output and process return code;
it rejects dirty/mismatched checkouts, busy GPUs and reused local output names.

Evidence locations:

```text
Local:
/Users/bytedance/Desktop/WYZ/TREA_auto/20260724-kvbit-dev-plugin/validation/moe-streamed-decode-20260914/

Container raw JSON/traces:
/artifacts/moe-streamed-decode-20260914/

Host backing directory:
/mnt/nvme2/kvbit-future-20260908/moe-streamed-decode-20260914/
```

The local directory contains `*.json`, `*.run.json`, raw logs, Chrome traces,
`summary.json`, `tables.md`, `environment.json`, `preflight-events.md`, the
execution/aggregation helpers and CPU JUnit XML. Source and H20 config hashes,
actual imports, loaded CUDA/AOT paths and process non-overlap are recorded.
Local and remote source hashes match; both tested checkouts are clean at
`109a00cca5`. The final documentation commit does not alter runtime code.

After GPU validation, the experiment container contained only init and sleep,
and GPU2 was back to 0 MiB. External VLLM PIDs 2140219 on GPU0 and 2426345 on
GPU1 remained untouched. This is not a claim that all eight GPUs were free.

## Remaining Boundary

No production selector change or formal PR was created. Before promoting a
narrow M1 configuration, it still needs representative model routing and
weight distributions, real model quality checks, multilayer cache behavior
and serving measurements. The dtype/shape regressions need direct compiler
or hardware evidence before a broader selector can be justified. The present
result is a verified low-level candidate with explicit applicability limits.
