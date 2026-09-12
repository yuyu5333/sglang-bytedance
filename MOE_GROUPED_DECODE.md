# Unsorted Expert-Local Tensor-Core Decode

Date: 2026-09-12. Branch: `feat/moe-direct-decode-grouped`.
Base: `07c9e4fa16` from the [single-token direct experiment](MOE_DIRECT_DECODE.md).
Final tested implementation: `ba2e4f7dd3659cfec248e02a1d3a73cdb6673020`.

## Decision

**Do not extend the runner selector to M4/M8.** The broad performance hypothesis
was not supported. Keep this low-level candidate as an explicitly invoked
experiment, with the original Triton and SIMT direct implementations unchanged.

One narrow result survives fresh replication: BF16 M8 with all tokens selecting
the same six experts, using N32/K256, reduces graph latency by **6.23%-6.60%**
over the original Triton path across three seeds. The same tile on M8 uniform
routing is **25.40% slower**. Shape-only selection cannot distinguish these cases.
There is no routing-to-host decision, new environment switch, runtime tuner,
checkpoint accuracy claim, or serving-throughput claim.

The final kernel passes the numerical matrix, changed-routing CUDA Graph replay,
two-stream checks, and the scoped device sanitizer checks below. Unfiltered
memcheck still exits 86 with 34 import-time CUDA API errors; this is not an
unqualified sanitizer all-pass or production acceptance.

## Hypothesis And Implementation

The previous SIMT direct kernel rereads expert weights for each selected token.
The candidate attempts to preserve its unsorted dispatch while recovering
expert-local token reuse and tensor-core computation:

1. Each route CTA scans at most 8 tokens x 8 slots. Only the first route
   selecting an expert is its leader. Other CTAs exit without matrix work.
2. `_grouped_gate_up` batches that expert's selected tokens in a 16-row tile.
   Canonical gate/up weights are gathered into paired columns of one `tl.dot`.
   The FP32 accumulator is rounded to FP16/BF16 before SiLU, and the result is
   stored only at each token's first matching route slot.
3. `_grouped_down` reads that first slot and performs a batched down projection.
   Every matching route gets its own weighted, dtype-rounded result, including
   duplicate expert slots with distinct router weights.
4. `_grouped_sum` masks invalid expert IDs before reading route results,
   sums in FP32, applies the final scale, and writes output.

There are **three kernels**, not the previous two. No sort, global lock, atomic
output addition, communication, weight repacking, or persistent shared scratch
is introduced. Scratch is call-private:

```text
activation: 2 * M * top-k * I bytes
route output: 2 * M * top-k * H bytes
separate output when not inplace: 2 * M * H bytes
```

Unused activation and invalid route slots are intentionally not initialized.
All consumers mask them before reading. The initcheck case covers initially
all-masked routes, transition to valid routes with a masked token, and graph
replay. Inplace output is written only after input consumption is complete.

The shared metadata check allows CUDA FP16/BF16, M1..8, H128..8192, I128..4096,
top-k1..8, contiguous same-device weights, INT32/INT64 IDs, FP32 router weights,
and optional matching biases. FP8/INT4, capped/interleaved SwiGLU, distributed
execution and production dispatch integration are not part of this experiment.
Accumulation order differs; bitwise equivalence is not promised.

Cached PTX was read back, not inferred from `tl.dot` syntax alone. Gate/up and
down variants contain `mma.sync.aligned.m16n8k16` FP16/BF16 instructions.
This is not a WGMMA or TMA implementation.

## Environment And Measurement

| Item | Actual Value |
|---|---|
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| Dedicated container | `kvbit-future-ab-20260908` |
| Repository / import | `/sgl-workspace/sglang-bytedance/python/sglang/__init__.py` |
| Device | GPU0 H20, SM90, 78 SMs, 60 MiB L2 |
| GPU UUID | `GPU-1c2c22aa-44de-d80b-fff2-03e1842c1b32` |
| Python / Torch / Triton | 3.12.3 / 2.13.0+cu130 / 3.7.1 |
| CUDA compiler / sanitizer | 13.0.88 / 2025.3.1.0 |
| Host driver / loaded CUDA | 535.161.08 / compat `libcuda.so.580.82.07` |
| Parallelism | TP=EP=PP=DP=1, single operator call at a time |
| Model | Synthetic unquantized MoE; no checkpoint loaded |
| Workload | H4096/I512/E256/top-k6, M listed per row |
| Math | BF16 weights/activations, plain SiLU, scale1, no bias/mask, outplace |
| Routing | Uniform distinct top-k, or hot: all tokens select experts 0..5 |
| Runtime | Explicit `PYTHONPATH=<repo>/python`, `CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=1`, `HF_HUB_OFFLINE=1` |
| Timing | 20 warmups, 7 rounds x 200 calls, randomized implementation order |
| Reported statistic | Median of per-round CUDA-event mean latency |
| CUDA Graph | Enabled for primary latency tables; eager also recorded |
| Not applicable | Sequence input/output lengths, request count/concurrency, mem-fraction, KV dtype, speculative decoding, HTTP, TTFT/TPOT |

All implementations in an A/B process share weights, inputs, routes, dtype,
scaling, device and timing protocol. Graph routing is restored before timing.
All 27 timed processes are non-overlapping; each selected GPU was checked idle
before launch. No package changes, GPU resets or clock locking were performed.
These are repeated single-layer measurements, not a cold-cache multilayer
serving trace.

Baseline retains runtime config resolution, including the existing 3.5.1 H20
fallback when 3.7.1 tuning JSON is absent. The down projection keeps `USE_TMA`.
No baseline TMA option was disabled to improve the candidate's relative result.
Baseline tile settings:

| M | Up M/N/K | Up Warps/Stages | Down M/N/K | Down Warps/Stages |
|---:|---|---|---|---|
| 1 | 16/64/128 | 4/4 | 16/32/256 | 4/2, TMA |
| 4 | 16/64/64 | 4/4 | 16/32/256 | 4/2, TMA |
| 8 | 16/64/64 | 4/3 | 16/32/256 | 4/2, TMA |

## Fresh A/B Results

Final implementation, grouped default M16/N32/K64, 4 warps, 3 stages.
Latency is microseconds; positive change means slower than baseline.

| M | Routing | Unique Experts / Routes | Original Triton | SIMT Direct | Grouped | Grouped Change |
|---:|---|---:|---:|---:|---:|---:|
| 1 | uniform | 6/6 | 40.0467 | 37.6603 | 49.3997 | +23.36% |
| 1 | hot | 6/6 | 39.9461 | 37.6962 | 49.5445 | +24.03% |
| 4 | uniform | 24/24 | 98.3594 | 108.9366 | 109.9398 | +11.77% |
| 4 | hot | 6/24 | 43.9280 | 81.9075 | 55.0328 | +25.28% |
| 8 | uniform | 47/48 | 199.9646 | 212.8646 | 227.1101 | +13.58% |
| 8 | hot | 6/48 | 55.1762 | 151.8606 | 57.6749 | +4.53% |

The broad default is a negative result, including M1. It does not replace the
previous M1 SIMT path.

### Offline Tile Search And Confirmation

Twelve candidate-only diagnostic processes tested N16/N32/N64 crossed with
K128/K256 for M4 uniform and hot routing. They used the initial two-dot
gate/up revision `03c750ec83`, 3 rounds x 100 calls. They are not final A/B
results. All passed numerical checks, but no single tile won both distributions.
For example, N32/K256 measured 122.4310 us uniform and 44.3290 us hot.
Full negative search results are retained in the raw evidence.

The single-dot revision was then checked with fresh original/SIMT/grouped A/B:

| M | Routing | Grouped N/K | Original Triton us | Grouped us | Change |
|---:|---|---|---:|---:|---:|
| 4 | uniform | 64/128 | 98.2813 | 104.5506 | +6.38% |
| 4 | hot | 32/256 | 43.8523 | 46.3842 | +5.77% |
| 8 | hot | 32/256 | 55.0274 | 51.3858 | -6.62% |
| 8 | uniform | 32/256 | 197.2824 | 247.3861 | +25.40% |

Additional M8 hot N32/K256 replication changes all random tensors with the seed:

| Seed | Original us | Grouped us | Reduction | Original Round Range us | Grouped Round Range us |
|---:|---:|---:|---:|---|---|
| 42 | 55.2544 | 51.6062 | 6.60% | 55.1562..55.3026 | 51.3443..51.7512 |
| 7 | 55.1360 | 51.7005 | 6.23% | 55.0258..55.1685 | 51.3536..51.7701 |
| 123 | 55.1832 | 51.7298 | 6.26% | 55.0534..55.1968 | 51.3597..51.7499 |

These are independent within-seed A/B results, not a cross-seed accuracy ranking.
The route distribution remains deliberately concentrated, not representative
evidence of routing concentration in a real model.

### Allocation

Measured eager call peak includes the output and excludes already resident
weights and caller inputs. It is not process memory or a KV capacity measurement.

| M | Original KiB | SIMT Direct KiB | Grouped KiB |
|---:|---:|---:|---:|
| 1 | 345.5 | 14 | 62 |
| 4 | 1378.5 | 56 | 248 |
| 8 | 2756 | 112 | 496 |

Grouped needs more scratch than SIMT direct because it materializes down routes.
Reducing this small allocation relative to baseline is not a reason to accept
the uniform-routing latency regression.

## Why The Broad Hypothesis Failed

The route counts explain the available reuse without assuming a hardware
bottleneck. For M4 uniform, all 24 routes select different experts. Grouping
cannot reduce their weight working set. M8 uniform has only one repeated expert
among 48 routes. A 16-row tensor-core tile then does mostly masked-row work.
M8 hot has eight useful rows per expert and only six unique experts.

Each expert's BF16 gate/up/down weights total 12 MiB at this shape. Unique
selected weights are 288 MiB at M4 uniform, 564 MiB at M8 uniform, and 72 MiB
at hot routing. These are logical working sets, not measured DRAM traffic.
The original sorted Triton path already groups tokens by expert, so restoring
reuse relative to SIMT is not automatically an advantage over the proper baseline.

Single-call traces provide a narrower observation:

| Case / Implementation | Alignment us | Gate/Up us | Activation us | Down us | Sum us |
|---|---:|---:|---:|---:|---:|
| Initial M4 uniform / original | 1.664 | 61.248 | 1.856 | 31.809 | 1.824 |
| Initial M4 uniform / grouped | - | 60.513 | fused | 45.600 | 2.752 |
| Final M8 hot N32/K256 / original | 4.193 | 32.320 | 1.984 | 13.089 | 1.568 |
| Final M8 hot N32/K256 / grouped | - | 23.872 | fused | 23.072 | 3.584 |

Removing two launches saves overhead, but grouped down and sum remain more
expensive. Larger K tiles trade loop overhead against shared-memory footprint
and occupancy; leader scans and inactive CTAs also remain. These are plausible
mechanisms, not separately measured causal contributions.

Combining gate/up into one dot did not materially improve M4 default latency:
108.3958 us before versus 108.4470 us after in separate pilot A/B runs.
It is retained as the tested implementation, not claimed as a speedup.

Trace tables are diagnostics, not medians. `trace_processor` was unavailable
on PATH; Chrome JSON was parsed directly. No Perfetto SQL or NCU result is
claimed. A future optimization needs measured down-kernel memory/occupancy
evidence, not an assertion that tensor cores or fewer launches must win.

## Validation

### CPU Scope And Results

The new `test_grouped_decode.py` covers the actual Python wrapper's rejection
before allocation, tile validation, launch geometry/order, bias and scale
forwarding, inplace output, call-private scratch and boundary shapes.
CUDA operations are recorded dependencies, not an emulation of the arithmetic.

New cases: **20/20 passed on the first run**. Existing cases: **415 passed**.
Final combined rerun: **435 passed**, 2 expected platform warnings, 20.24 s.
The seven-step unit-test workflow and `utree flush` completed. Static analysis
confirmed no new wrapper defect; no assertions were weakened. Coverage
statistics were skipped under the workflow's no-gate/non-flux rule.

### GPU Numerical Matrix

All final-revision cases below pass at unchanged `atol=0.003, rtol=0.03`.
Each includes five changed-routing graph replays and, except inplace, two
independent output calls on separate CUDA streams.

| Case | Main Coverage |
|---|---|
| `final-ragged-fp16` | M3/H288/I160/E64/T3, FP16, bias, scale1.7, INT64, duplicate expert, masked route |
| `final-hot-m8-masked-token` | M8/H1024/I512/E64/T8, BF16, hot routes, entire token masked |
| `final-topk1-inplace` | M1/H1024/I512/E64/T1, FP16, bias, scale0.5, inplace |
| `final-zero-scale` | M4/H1024/I512/E64/T4, BF16, bias, scale0 |
| `final-all-masked` | M4/H256/I128/E64/T3, BF16, bias, all routes invalid before graph transition |
| `final-max-geometry` | M8/H8192/I4096/E8/T8, BF16, hot/duplicate routes; grouped versus reference only |
| Final A/B processes | M1/M4/M8 main shape, both route distributions, default and selected tiles |

Graph checks change expert membership, leader positions, weights and reuse
distribution without changing tensor addresses. For masked cases the first
token becomes invalid, moving leaders to later tokens. The timings use the
restored initial IDs, not these diagnostic IDs.

### Device Sanitizers And Failures

| Scope | Tool | Result |
|---|---|---|
| FP16 ragged, duplicate, bias, INT64, masked token | memcheck / racecheck / synccheck | All 0 errors; racecheck 0 warnings |
| BF16 M8/H4096/I512/E256/T6 hot, duplicate, bias, masked route | memcheck / racecheck / synccheck | All 0 errors; racecheck 0 warnings |
| Same main geometry, N32/K256, duplicate, bias, masked token | memcheck / racecheck / synccheck | All 0 errors; racecheck 0 warnings |
| Main N32/K256, initially all masked, changed-route replay | initcheck | 0 errors |
| FP16 ragged without API filtering | memcheck | 34 `cuGetProcAddress_v2` invalid-value API errors; process exit 86 |

The ten zero-error runs explicitly use `--report-api-errors no`; they validate
device accesses/synchronization for those cases, not a clean CUDA API environment.
The unfiltered harness numerical result is true, but the aggregate process
result is false because its exit code is 86. All 34 raw error entries are
import-time API calls, not grouped-kernel device access reports.

One earlier pilot also failed. The new harness injected `-1` during graph replay
while baseline `num_experts == num_local_experts` disabled expert filtering.
The original Triton graph then performed an illegal access. Both direct
graphs had completed correctly. Commit `03c750ec83` restricts new masked IDs
to explicit masked test configurations; the same M4 pilot subsequently passed.
This was a harness contract violation, not a production-kernel fix.

Raw accounting: **52 processes, 50 successful, 2 retained failures** (the pilot
contract error and unfiltered API errors). This is an experiment ledger, not
a statistical pass rate or 52 independent model accuracy evaluations.

The older AOT alignment barrier and Marlin shared-memory race from
[workspace validation](MOE_WORKSPACE_CUDA_VALIDATION.md) are not fixed or
revalidated by this candidate. No installed binary was rebuilt or replaced.

## Reproduction And Artifacts

Source commits:

| Commit | Purpose |
|---|---|
| `250211dfc4` | Expert leaders, tensor-core candidate, harness, 20 CPU cases |
| `03c750ec83` | Correct graph-replay expert-filter contract |
| `ba2e4f7dd3` | Single-dot gate/up implementation; final GPU-tested source |

Example narrow positive case, from the repository:

```bash
PYTHONPATH="$PWD/python" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 python3 benchmark/kernels/fused_moe_triton/validate_direct_decode.py \
  --grouped --tokens 8 --hidden 4096 --intermediate 512 --experts 256 --topk 6 \
  --dtype bfloat16 --routing hot --block-n 32 --block-k 256 \
  --warmup 20 --iterations 200 --repeats 7 --seed 42 \
  --output /artifacts/grouped-reproduction.json
```

Repeat with `--routing uniform` to check the corresponding regression.
The grouped path is invoked explicitly by the harness, never by the runner.

Raw evidence in the shared workspace, outside this Git repository:

```text
../validation/moe-grouped-decode-20260912/
  <run>.json, <run>.log, <run>.run.json
  *-trace.json
  summary.json, tables.md, environment.json, codegen.json
  cpu-initial.xml, cpu-grouped.xml, cpu-final.xml, cpu-final-paired.xml
  run_remote.py, run_matrix.py, run_tiles.py, run_confirm.py, run_final.py
  summarize.py, record_environment.py, inspect_codegen.py
```

Container artifacts: `/artifacts/moe-grouped-decode-20260912/`.
The ledger records commands, host/container, source HEAD, process return codes,
start/end times, GPU state, import path, source hashes and numerical metrics.
Environment collection matched all ten local/remote source/config hashes.

No real checkpoint, model accuracy, TTFT/TPOT, KV capacity or serving throughput
was measured. The mounted DeepSeek-V4-Flash checkpoint is native FP8 with capped
SwiGLU and is outside this unquantized plain-SiLU path.

## Next Decision

Retain the experiment and evidence, reject broad runner integration for now.
A useful next candidate must reduce down-projection and dispatch overhead while
remaining competitive with the tuned original path on uniform routing.
Alternatively, a device-side policy would need its own cost measurement and
real routing-distribution evidence before exploiting the narrow hot case.
Neither is assumed implemented or validated here.
