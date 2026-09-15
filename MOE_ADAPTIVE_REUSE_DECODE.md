# GPU-Only Expert-Reuse Adaptive Decode

Date: 2026-09-15. Branch: `feat/moe-adaptive-reuse-decode`.
Base: `d91ead9db624c3d07975227f50d9600636b5c38a`.
Implementation: `7e885c84465f211bffec0e8ef7645dbee3cb5815`.
Tests: `27c1794a562a96f5b96fafa79f5208b74acd7083`.

## Decision

**Do not integrate this candidate into the runner.** GPU-only per-expert
selection works numerically in the tested cases, but combining SIMT and
tensor-core branches in the same compiled kernels does not recover the best
performance of each separate implementation.

Default adaptive M4/M8 uniform is 122.90%/117.88% slower than original Triton.
M4/M8 hot is 12.72%/14.65% slower. M8 mixed, where both branches really execute,
is 113.70% slower. Smaller tiles mitigate the uniform regression but do not
provide a common winning configuration. M1 is faster than original Triton,
but slower than the existing streamed candidate; its grouped branch is
eliminated at compile time, so it is not a dynamic-selection benefit.

Keep the code as an explicitly invoked experiment and retain the negative
results. Old direct, grouped, streamed, runner, selector and environment
switches are unchanged. No formal PR, checkpoint test, serving test or
production performance claim is part of this round.

## Hypothesis And Implementation

The [streamed experiment](MOE_STREAMED_DECODE.md) improved low-reuse cases but
lost badly on hot routes. The [grouped experiment](MOE_GROUPED_DECODE.md)
recovered cross-token reuse, but lost on uniform routing. The new hypothesis
was to select per expert inside the GPU rather than read routing values back
to the host or choose one implementation for the whole batch.

There are three launches:

1. `_adaptive_gate_up`: scan the at-most 8 x 8 routing IDs and count distinct
   tokens selecting the route's expert. At or above `min_reuse`, call the
   existing leader-based grouped tensor-core gate/up; otherwise call the
   existing route-local SIMT gate/up.
2. `_adaptive_down`: recompute the same decision from unchanged IDs. Grouped
   experts consume the first matching activation slot; SIMT experts consume
   their own route slot. Both write each weighted route in output dtype.
3. Reuse `_grouped_sum`: mask invalid expert IDs, reduce routes in FP32, apply
   final scale and write output.

All routes selecting one expert make the same decision. Duplicate slots
inside one token do not inflate the reuse count. Grouped leaders write all
matching weighted output slots, including duplicates; SIMT routes own their
individual output slots. There are no output atomics, global locks, sorting,
host routing decisions or shared mutable scratch.

Per-call allocations for FP16/BF16:

```text
activation: 2 * M * top-k * I bytes
route output: 2 * M * top-k * H bytes
separate output when not inplace: 2 * M * H bytes
```

The first launch consumes `x`; down and sum never read it, so inplace writes
remain ordered after input consumption. Invalid routes may leave scratch
uninitialized but the final sum masks them before loading.

The existing metadata gate allows contiguous same-device CUDA FP16/BF16
tensors, M1..8, H128..8192, I128..4096, top-k1..8, INT32/INT64 IDs, FP32 router
weights and optional matching biases. No quantized or distributed execution,
interleaved weights, capped activation or production dispatch is added.
The original dtype materialization boundaries are retained:
gate/up before SiLU, activation before down, and weighted routes before sum.
`enable_fp_fusion=False`; different accumulation orders are not bitwise
equivalent by contract.

Wrapper defaults are `min_reuse=4`, `simt_n=16`, `group_n=32`, `group_k=256`,
four warps and three stages. Threshold9 forces SIMT by compile-time M bounds.
Threshold1 routes every valid expert to grouped, but does not remove the
dynamic branch and extra grid work from the source implementation.
The harness retains the old shared `--block-k` default64, so reproductions
must explicitly pass `--block-k 256` for the default-wrapper matrix.

## Environment And Protocol

| Item | Actual Configuration |
|---|---|
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| Dedicated container | `kvbit-future-ab-20260908` |
| Repository | `/sgl-workspace/sglang-bytedance` |
| Primary device | GPU6 H20, SM90, 78 SMs, 60 MiB L2 |
| GPU6 UUID | `GPU-adbf4722-ba6c-4109-111a-4bfeb0f3a365` |
| Earlier pilot/threshold controls | GPU2, `GPU-159d2b76-8430-73da-3de3-3c781ddeaf28` |
| Python / Torch / Triton | 3.12.3 / 2.13.0+cu130 / 3.7.1 |
| CUDA compiler / sanitizer | 13.0.88 / 2025.3.1.0 |
| Host driver / loaded CUDA | 535.161.08 / compat `libcuda.so.580.82.07` |
| Actual candidate import | `<repo>/python/sglang/kernels/ops/moe/adaptive_decode.py` |
| Loaded AOT binary | `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm90/common_ops.abi3.so` |
| Model | Synthetic unquantized MoE; no checkpoint |
| Shape | H4096/I512/E256/top-k6; M specified per row |
| Math | BF16, plain SiLU, scale1, no bias/mask, outplace unless explicitly overridden |
| Parallelism | TP=EP=PP=DP=1, single measured operator at a time |
| Runtime | Explicit `PYTHONPATH=<repo>/python`, selected `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`, `HF_HUB_OFFLINE=1` |
| Primary timing | 20 warmups, seven rounds x 200 calls, randomized implementation order, seed42 |
| Reported statistic | Median of per-round CUDA-event mean latency |
| CUDA Graph | Enabled in latency tables; eager also retained |
| Not applicable | Sequence lengths, requests/concurrency, mem-fraction, KV dtype, speculative decoding, HTTP, TTFT/TPOT |

Every comparison process uses the same input, weights, routing, dtype, device
and timing protocol across implementations. Routing is restored before timing.
Repeated graphs reuse addresses and weights; they are not cold-cache multilayer
serving measurements. Eager includes Python and launch gaps.

The runtime baseline uses existing H20 config resolution with the 3.5.1
fallback when 3.7.1 JSON is absent; down `USE_TMA` stays enabled. Tuning files
and baseline settings were not changed. Main-shape settings:

| M | Up M/N/K | Warps/Stages | Down M/N/K | Warps/Stages |
|---:|---|---|---|---|
| 1 | 16/64/128 | 4/4 | 16/32/256 | 4/2, TMA |
| 4 | 16/64/64 | 4/4 | 16/32/256 | 4/2, TMA |
| 8 | 16/64/64 | 4/3 | 16/32/256 | 4/2, TMA |

The first three pilots and nine threshold controls used idle GPU2. An external
task later occupied GPU2..5; the next preflight refused to launch. The dedicated
container contained only init/sleep. Remaining diagnostics and the complete
fresh matrix used idle GPU6. No external process was stopped or modified.
GPU0/1 also retained external VLLM services. The host was not exclusive;
absolute times from GPU2 and GPU6 are not pooled.

## Fresh GPU6 Matrix

Microseconds; positive change means slower than original Triton. Grouped
uses N32/K256. Streamed uses Up N16/Down N16, four warps, unroll2. Adaptive
uses the wrapper-default parameters stated above.

| M | Routing | Original | Old Direct | Grouped | Streamed | Adaptive | Adaptive Change |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | uniform | 39.4088 | 37.1517 | 43.1664 | 31.6926 | 32.7690 | -16.85% |
| 4 | uniform | 97.1566 | 108.2637 | 125.7776 | 93.8125 | 216.5658 | +122.90% |
| 4 | hot | 42.9534 | 81.5232 | 44.9965 | 62.6643 | 48.4179 | +12.72% |
| 4 | mixed | 77.0242 | 95.1726 | 93.6402 | 81.3491 | 209.8354 | +172.43% |
| 8 | uniform | 194.3034 | 205.7162 | 229.6534 | 173.1782 | 423.3429 | +117.88% |
| 8 | hot | 53.7429 | 151.8051 | 50.2186 | 115.5910 | 61.6157 | +14.65% |
| 8 | mixed | 118.8134 | 173.9986 | 147.6176 | 134.5674 | 253.9061 | +113.70% |

Uniform uses per-token distinct top-k from random logits. Hot restricts all
tokens to experts0..5. Mixed restricts the first half of the tokens to these
six experts and leaves the other half uniform. M4 mixed selects no expert
with four distinct tokens, so all 24 routes take SIMT. M8 mixed selects
25 grouped and 23 SIMT routes, with 29 unique experts. Thus its negative
result really covers simultaneous branch use, not only an artificial label.

M1 has no dynamic grouped branch at threshold4. It also uses an extra route
tensor and launch relative to streamed. A lower M1 time than original Triton
does not justify adopting this candidate over the existing narrow experiment.

## Diagnostic Ablations

### Threshold Controls On GPU2

Each cell below is adaptive/original latency from its own same-input process.
Ten warmups, three rounds x 100 calls; N32/K256, SIMT N16. These are diagnostic
configurations, not a same-process isolation of individual hardware effects.

| Case | Threshold1 us | Threshold2 us | Threshold9 us |
|---|---|---|---|
| M4 uniform | 130.1062 / 98.2112 | 216.1843 / 98.2714 | 98.4010 / 98.3475 |
| M8 hot | 62.8832 / 55.0998 | 62.9667 / 55.1158 | 121.5731 / 55.1034 |
| M8 mixed | 156.2966 / 119.6317 | 251.2394 / 119.7542 | 139.3283 / 122.3974 |

No tested threshold solves the mixed-distribution problem. Threshold9 removes
the grouped branch at compilation, but gives up hot-route tensor-core reuse.

### Smaller Tiles On GPU6

Ten warmups, three rounds x 100 calls; threshold4, SIMT N16. Candidate/original
latency pairs from independent processes:

| Group N/K | M4 Uniform us | M8 Hot us |
|---|---|---|
| 16/32 | 102.1814 / 98.6973 | 101.0707 / 55.1366 |
| 16/64 | 101.9856 / 98.6957 | 74.0560 / 55.1437 |
| 32/64 | 102.1162 / 98.6189 | 64.6016 / 55.2605 |

These tiles reduce the uniform penalty, but remain slower than baseline and
lose hot-route performance. No online tuning or automatic tile selection is
introduced.

Fresh N16/K32 confirmation uses the primary 20-warmup, seven-round x 200-call
protocol, on GPU6 with the same BF16 shape and seed:

| Case | Original us | Adaptive us |
|---|---:|---:|
| M4 uniform | 98.9024 | 103.4104 |
| M8 hot | 55.1283 | 100.9338 |
| M8 mixed | 120.1976 | 177.9790 |

The negative result survives the longer measurement. It is not a production
fallback configuration.

### Evidence For The Resource Tradeoff

Single-call traces, not timing-round medians:

| Case / Device | Gate-Up us | Gate-Up Registers/Thread | Gate-Up Shared Bytes | Down Shared Bytes |
|---|---:|---:|---:|---:|
| Default M4 uniform / GPU6 | 163.332 | 69 | 81920 | 49152 |
| N16/K32 M4 uniform / GPU6 | 65.825 | 110 | 6144 | 4096 |
| Threshold9 M4 uniform / GPU2 | 62.177 | 113 | 256 | 1024 |

The default kernel retains 80 KiB gate/up shared memory even when every
runtime route chooses SIMT. Smaller grouped tiles change that compiled
footprint; threshold9 eliminates the branch entirely. Fewer registers in the
default mixed kernel are not evidence of a faster kernel or higher occupancy.
The measurements support a resource/layout tradeoff; they do not isolate
shared memory from compiler layouts, route scans, pipeline changes or other
effects. No NCU hardware-counter attribution is claimed.

Hot routes pay additional scans and excess CTAs. With SIMT N16 and grouped
N32, both mixed grids are twice as wide as the standalone grouped grid.
High-reuse CTAs outside the grouped range scan routing before exiting;
surviving grouped work also performs leader discovery. Both branches share
one compiled resource envelope. Merely placing fast standalone algorithms
behind a dynamic branch does not preserve their standalone cost.

`trace_processor` was unavailable on PATH; Chrome JSON was parsed directly.
Profiler occupancy estimates are not treated as measured hardware occupancy.
The precise contribution of each source of overhead remains unisolated.

## CPU Validation

Scope: `adaptive_decode` wrapper in the new module. The test uses the actual
wrapper and shared metadata gate, with controlled allocation/device/launch
dependencies; it does not emulate CUDA arithmetic.

Defect analysis found no confirmed wrapper defect. Allocation order, branch
activation ownership and inplace lifetime concerns were checked against
the actual producer/consumer code and filtered with reasons recorded.

Generated cases: 12 launch-contract combinations plus 12 rejection cases.
They cover inplace/outplace, bias/no bias, boundary/ragged shapes, thresholds
1/4/9, launch grids/parameters/order, output aliasing, call-private allocation,
metadata rejection and invalid tile/threshold rejection before allocation.

Verification: **24/24 passed on the first run** (16.85 s); no test fix or
assertion weakening. Full related suite: **494 passed**, two expected macOS
platform warnings, 27.71 s. The seven-step workflow and `utree flush` completed.
Coverage statistics were skipped under the no-gate/non-flux rule.

## GPU Numerical Validation

All listed cases pass independent-reference checks at unchanged
`atol=0.003, rtol=0.03`. Each includes five changed-routing graph replays;
non-inplace cases also check two calls on independent CUDA streams.

| Case | Configuration / Coverage |
|---|---|
| `ragged` | M5/H288/I160/E64/T3, FP16, hot, duplicate expert, masked route, INT64, bias, scale1.7 |
| `topk1-inplace` | M4/H1024/I512/E64/T1, FP16, hot, bias, scale0.5, inplace |
| `all-masked` | M8/H256/I128/E64/T3, BF16, bias, initially all IDs -1 |
| `zero-scale` | M8/H1024/I512/E64/T8, BF16, mixed, masked token, scale0 |
| `max-geometry` | M8/H8192/I4096/E8/T8, BF16, hot and duplicate routes |
| Timed processes | Main uniform/hot/mixed routing, forced-threshold controls and tile diagnostics |

The maximum-geometry max-absolute error is 0.001953125. In main M4 uniform,
graph replay changes all 24 routes from SIMT to grouped; main M4 hot changes
all 24 in the other direction. M8 mixed changes from 25 grouped/23 SIMT routes
to 48 SIMT routes. Tensor addresses stay fixed. This checks data-dependent
branch changes within captured graphs, not only a static hot-route graph.

Eager call peak, including output but excluding already resident input and
weights, is 62/248/496 KiB for adaptive M1/M4/M8, versus 14/56/112 KiB for
old direct or streamed and 345.5/1378.5/2756 KiB for original Triton.
Adaptive matches grouped's allocation, not streamed's smaller allocation.
No KV capacity inference follows from this single-call metric.

## Sanitizers And Limits

All device checks below run adaptive-only on GPU6, with threshold4, SIMT N16
and grouped N32/K256. They exercise both dynamic branches through changed
routing, not all possible legal launch configurations.

| Scope | Tool | Result |
|---|---|---|
| FP16 M5/H288/I160/E64/T3, hot, duplicate, masked route, INT64, bias | memcheck / racecheck / synccheck / initcheck | Four exit0; zero errors; racecheck zero warnings |
| BF16 M8/H4096/I512/E256/T6, mixed, duplicate, masked route, bias | memcheck / racecheck / synccheck / initcheck | Four exit0; zero errors; racecheck zero warnings |
| BF16 M8/H256/I128/E64/T3, bias, initially all masked, then grouped valid routes | initcheck | Exit0, zero errors |
| FP16 ragged configuration, API reporting unfiltered | memcheck | Exit86, 34 import-time CUDA API errors |

The nine zero-error runs explicitly use `--report-api-errors no`. They are
filtered device checks, not an unqualified sanitizer all-pass. All 34
unfiltered reports are `CUDA_ERROR_INVALID_VALUE` on `cuGetProcAddress_v2`
from import stacks. Numerical/graph/two-stream checks remain true, but the
process failure is retained. No package change was made to suppress it.
The older AOT alignment barrier and Marlin shared-memory race from
[workspace validation](MOE_WORKSPACE_CUDA_VALIDATION.md) are neither fixed
nor revalidated by these adaptive-only checks.

Ledger: **43 experiment processes, 42 successful, one retained unfiltered
API failure**. All 43 numerical reports pass and all use the same adaptive
source SHA256. Process intervals do not overlap. The busy-GPU preflight
refusal is recorded separately because no benchmark was launched.
This is experiment accounting, not a statistical model-accuracy rate.

No model, quantized weights, distributed communication, quality benchmark
or serving behavior was tested. No multi-seed claim is made for this negative
round. The baseline remains the actual runtime-selected configuration, not
an exhaustively tuned upper bound. External jobs ran on other GPUs; paired
same-process tests control inputs and the selected GPU, not whole-host power
or host scheduling.

## Reproduction And Provenance

Commits `7e885c8446` and `27c1794a56` contain the candidate/harness and CPU
tests, respectively. Runtime code was not changed after the first GPU pilot.
The final report commit is documentation-only.

Adaptive source SHA256:

```text
9e5e0893a045f60c6da6880647781cde2205f478816be2e252abe5ae1931d28d
```

Choose an idle physical GPU and unused output name. Example inside the
dedicated container:

```bash
env PYTHONPATH=/sgl-workspace/sglang-bytedance/python \
  CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  python3 /sgl-workspace/sglang-bytedance/benchmark/kernels/fused_moe_triton/validate_direct_decode.py \
  --adaptive --streamed --grouped --tokens 8 --routing mixed \
  --min-reuse 4 --simt-n 16 --block-n 32 --block-k 256 \
  --up-n 16 --down-n 16 --unroll 2 \
  --warmup 20 --iterations 200 --repeats 7 --profile \
  --output /artifacts/moe-adaptive-decode-20260915/new-m8-mixed.json
```

For sanitizer reproduction, use `--adaptive-only --skip-timing --warmup 1`
with the documented shape/mask flags, and prefix Python with
`compute-sanitizer --tool <tool> --error-exitcode 86 --target-processes all`.
Adding `--report-api-errors no` changes the scope to filtered device checks
and must be recorded as such.

Evidence locations:

```text
Local:
/Users/bytedance/Desktop/WYZ/TREA_auto/20260724-kvbit-dev-plugin/validation/moe-adaptive-decode-20260915/

Container JSON/traces:
/artifacts/moe-adaptive-decode-20260915/

Host backing directory:
/mnt/nvme2/kvbit-future-20260908/moe-adaptive-decode-20260915/
```

Local evidence includes raw JSON, command/exit/timestamp sidecars, raw logs,
Chrome traces, CPU JUnit XML, execution helpers, `summary.json`, `tables.md`,
`environment.json` and `preflight-events.md`. Source/config hashes, actual
imports and loaded CUDA/AOT paths match the intended checkout. Source was
modified only locally, committed/pushed, then synchronized through Git.
No package upgrade, binary replacement, GPU reset or clock change was made.

Post-validation GPU6 memory was 0 MiB; the dedicated container contained only
init/sleep. External VLLM on GPU0/1 and Python jobs on GPU2..5 were untouched.
The original workspace's unrelated harness formatting change was preserved.

## Next Direction

The evidence argues against putting these two resource-heavy algorithms
behind one CTA-level branch. A future experiment could first classify routing
once into compact GPU metadata, then use separate resource-specialized kernels,
with explicit accounting for the extra launches and scratch. That is a new
hypothesis, not an implementation or speedup delivered here. It should only
be promoted after fresh paired measurements show that classification and launch
overhead do not erase the benefit. This round stops at a validated negative
result with production behavior unchanged.
