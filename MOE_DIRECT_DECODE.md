# Two-Launch Direct MoE Decode

Date: 2026-09-12. Branch: `feat/moe-direct-decode`.
Base: `527aafd7ed` on `feat/marlin-moe-bounded-workspace`.
Kernel prototype: `d7e48efde4`; tested runner integration: `f43ae5bcce`.

## Decision

Retain an **opt-in single-token experiment**, not a prefill optimization
or a new default. This direction removes work instead of splitting it into
more launches: unsorted GEMV with fused activation and fused output reduction
replaces the ordinary sorted/padded GEMM pipeline.

Two fresh H20 single-token workloads reduce CUDA Graph latency by **6.52%**
and **30.31%** against their own original Triton runner. Call peak allocation
also falls. Eager improvements are larger, but include reduced Python and
launch overhead; they must not be reported as kernel speedups.

The same low-level algorithm at M4 is **11.24% slower under CUDA Graph**.
The runner therefore selects it only for M1; M4 fallback is verified.
Different shapes/backends are independent workloads, not a cross-format
performance or accuracy ranking.

## Direction Selection

| Candidate | Source Observation | Decision |
|---|---|---|
| Continue bounded prefill scratch | Previous H20 tests reduced memory but increased latency at every measured chunked setting | Keep that experiment separate |
| Add an interleaved SwiGLU epilogue | `fused_moe_kernel` already implements it, with corresponding tests | Do not duplicate |
| Rewrite Marlin synchronization/epilogue | Prior AOT barrier and Marlin race reports are unresolved; altering them would mix a correctness repair with a new performance claim | Leave the evidence intact; separate follow-up |
| Direct token/top-k decode | M1 has little cross-token expert reuse, yet pays alignment, padded GEMM rows, an activation launch and output reduction | Implement and measure |

The branch is intentionally stacked on the previous tested head, in a new
worktree `../sglang-moe-direct-decode`. The original worktree and its existing
`validate_workspace_cuda.py` formatting change were not modified.

## Algorithm

Canonical expert weights remain `[E, 2I, H]` with `[gate; up]` halves and
`[E, H, I]`. No load-time repacking or quantization changes are introduced.

1. `_gate_up`: each CTA owns a routed token and an intermediate-feature tile.
   It reads the expert ID directly, computes gate/up GEMVs with FP32 reduction,
   applies optional bias, rounds to the activation dtype as the original
   GEMM1 would, then computes SiLU times up and stores `[M, top-k, I]`.
2. `_down_sum`: each CTA owns a token and an output-feature tile. It computes
   all selected experts' down projections, adds bias and router weights,
   rounds each weighted route to the output dtype as GEMM2 would, then sums
   routes in FP32 and applies the final routed scaling factor.

There is no route sort, expert block padding, atomic output accumulation,
global lock, full gate/up tensor, full route-expanded down tensor, or separate
activation/reduction launch. `torch.empty` allocations are private to the
call. The only scratch storage is:

```text
2 * M * top-k * I bytes
```

The output is separate unless the existing inplace option is selected.
Inplace is safe for this dataflow because the second kernel reads the saved
activation, not the original input. Same-stream kernel ordering supplies
the dependency between stages.

This is **not bitwise equivalent** to tensor-core GEMM: FP32 reduction order
changes. Explicit dtype round trips preserve existing materialization
boundaries, but cannot make two different reductions bitwise identical.
Deterministic and batch-invariant modes are excluded by the runner selector.

## Integration Boundary

Set before warmup/capture:

```bash
export SGLANG_MOE_DIRECT_DECODE=1
export SGLANG_MOE_WORKSPACE_BUDGET_BYTES=0
```

The default is `SGLANG_MOE_DIRECT_DECODE=0`. `MoeRunner.run` checks this path
only when the effective workspace budget is zero. A positive budget retains
the previous policy and never silently substitutes the direct algorithm.

`try_direct_decode` reuses the existing local-inference capability check,
then checks the unquantized metadata and the measured single-token boundary.
An unsupported case returns to the original path. No collective is added.

| Dimension | Runner Selection |
|---|---|
| Backend and device | Built-in Triton, CUDA, TP world size 1 |
| Tokens | Exactly 1 |
| Hidden / intermediate | 128..4096 / 128..1024 |
| Top-k | 1..8 |
| Activations and weights | Contiguous FP16/BF16; same dtype/device |
| Router weights / IDs | Contiguous FP32 / INT32 or INT64 |
| Expert layout | Canonical split gate/up; no interleaved fusion metadata |
| Activation | Gated plain SiLU; no alpha/beta/clamp modifiers |
| Bias, output, route scale | Supported bias, inplace/out-of-place, final scale |
| Quantization flags/scales/zero points | Rejected |
| LoRA, custom core, packed routes, pre-quantized dispatch | Rejected |
| A2A, overlap, activation all-gather, fused collective, symmetric allocation | Rejected |
| torch.compile, deterministic, batch-invariant | Rejected |

The low-level experimental operator accepts M1..8, H up to 8192 and I up to
4096 for research. Those larger limits are not automatic runner selection
or a performance recommendation. IDs must follow the existing valid-local-ID
or `-1` filtered-route contract.

## Environment and Method

| Item | Observed Value |
|---|---|
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| Container | `kvbit-future-ab-20260908` |
| Remote checkout | `/sgl-workspace/sglang-bytedance`, clean |
| Performance GPU | H20, SM90, 78 SMs, GPU 0 UUID `1c2c22aa-44de-d80b-fff2-03e1842c1b32` |
| Driver / compiler | `535.161.08` / CUDA `13.0.88` |
| Python / Torch / Triton | `3.12.3` / `2.13.0+cu130` / `3.7.1` |
| Compute Sanitizer | `2025.3.1.0` |
| Source import | `/sgl-workspace/sglang-bytedance/python/sglang/__init__.py` |
| Runtime setup | Offline `ServerArgs(model_path="dummy")`; Gloo single-rank groups |
| Topology / concurrency | TP/EP/PP/DP=1; sequential calls for latency, two streams checked separately |
| Model, sequence lengths, requests, KV dtype, mem-fraction | N/A: synthetic operator workload |
| Speculative decoding / HTTP | None / not tested |

Every run explicitly sets `PYTHONPATH=<checkout>/python`,
`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`, and `HF_HUB_OFFLINE=1`.
Source changes were local commits/pushes followed by container Git
fetch/checkout or fast-forward pull. No remote source patch, package change,
model modification, or replacement of installed AOT binaries was performed.

For each A/B process, the same normalized random input, BF16 weights, top-k
routes, router weights and seed 42 are reused for both implementations.
FP32 reference matmul disables TF32. Performance runs are sequential on GPU 0,
without a concurrent compilation or sanitizer workload. No clock/power
settings were changed.

The main repeated measurements warm up 10 calls and use 7 randomized-order
rounds of 200 calls per implementation and mode. Each table value is the
median of the round means. Earlier confirmation runs used 5 x 100.
CUDA-event eager measurements include host dispatch gaps. Graph measurements
replay captured calls with fixed addresses. Both modes reuse hot weights
and routes; this is not a multi-layer serving-cache simulation. The harness
toggles the feature flag only between controlled benchmark calls, not in a
live server.

## Fresh Runner A/B

Both rows use M1, BF16 activation/weights, gated SiLU, scale 1, no bias,
uniform distinct top-k, INT32 IDs, and out-of-place output. Each compares
its own original runtime-selected Triton implementation against direct
decode with identical inputs. `runner_direct_calls=1` verifies selection.

| H / I / E / Top-k | Original Runtime Config | Graph Baseline us | Graph Direct us | Reduction | Eager Baseline / Direct us | Call Peak Baseline / Direct KiB |
|---|---|---:|---:|---:|---:|---:|
| 4096 / 512 / 256 / 6 | H20 paired Triton-3.5.1 tables, TMA down path | 40.270 | 37.646 | 6.52% | 296.703 / 93.955 | 345.5 / 14.0 |
| 2048 / 1024 / 64 / 4 | Existing heuristic, non-TMA | 29.093 | 20.276 | 30.31% | 241.953 / 93.741 | 45.5 / 12.0 |

Raw files: `runner-m1-repeat.json`, `runner-narrow-m1-repeat.json`.
Graph round ranges are respectively:

| Workload | Baseline Range us | Direct Range us |
|---|---|---|
| H4096/I512/E256/T6 | 40.187..40.360 | 37.602..37.690 |
| H2048/I1024/E64/T4 | 29.054..29.131 | 20.119..20.313 |

The 5 x 100 confirmation runs measured reductions of 6.43% and 30.11%.
These are repeated operator measurements, not confidence intervals or
model-level speedup estimates.

Call peak is `max_memory_allocated - allocated_before_call`, not process
VRAM. It includes an 8 KiB / 4 KiB output in the two rows; direct scratch
is 6 KiB / 8 KiB respectively. Baseline TMA has padded intermediate storage.
Allocator reserved peaks do not shrink in these runs. Nothing here changes
KV pool sizing or proves an OOM avoidance guarantee.

### Negative Result and Dispatch Choice

With M4/H4096/I512/E256/T6, the raw operator measured graph latency
`98.329 -> 109.385 us`, **11.24% slower**, despite eager latency
`271.621 -> 110.997 us`. Source: `initial-m4.json` at `d7e48efde4`.
The direct approach rereads weights for every route and cannot exploit
cross-token GEMM reuse. That is a plausible mechanism, not a measured HBM
counter attribution.

`runner-m4-fallback.json` at `f43ae5bcce` confirms zero calls to the direct
kernel, identical baseline output, graph replay and two-stream correctness.
This is why the selector is M1-only.

### Trace Evidence

Separate post-timing Torch profiler captures confirm **5 kernels -> 2**:

```text
Original: align -> GEMM1 -> SiLU/multiply -> GEMM2 -> top-k reduce
Direct:   gate/up/SiLU -> down/router-weight/top-k sum
```

At H2048/I1024, one diagnostic trace records direct kernels at 12.800 and
7.233 us. The original trace records 1.280, 15.392, 1.184, 8.864 and 1.249 us.
At H4096/I512, direct kernels are 23.584 and 13.440 us; its original GEMMs
are 22.753 and 12.000 us, plus alignment/activation/reduction.
The larger shape therefore gets much less net GPU benefit from the fusion.

These one-call traces establish launch structure and provide diagnostic
durations, not the timing-table medians or bandwidth/cache counters.
Chrome trace JSON is retained for Perfetto. `trace_processor` was not on
PATH; launch counts/durations were read from the trace event JSON.

## Validation

### Numerical and Runtime Matrix

Fixed tolerance throughout: `atol=0.003, rtol=0.03`. The reference evaluates
each expert with FP32 matmul, explicit GEMM1 rounding, SiLU, GEMM2 weighting
and rounding, then FP32 top-k reduction. Passing synthetic tests does not
establish model accuracy or bitwise identity.

| Artifact | Coverage | Result |
|---|---|---|
| `runner-m1`, `runner-narrow-m1`, their `-repeat` variants | Actual M1 selector, independent reference and baseline | Pass |
| `runner-m4-fallback` | M4 is not selected | Pass, direct call count 0 |
| `runner-fp16-bias` | FP16, H512/I256/E8/T3, bias, scale 1.7, masked route, INT64 | Pass |
| `runner-topk1-inplace` | FP16, top-k 1, scale 0.5, bias, inplace | Pass |
| `runner-zero-scale` | BF16, top-k 1, zero scale | Pass |
| `raw-ragged` | M3/H288/I160/E4/T3, bias, mask, INT64, scale 0.5 | Pass |
| `raw-m8-topk8` | M8/H512/I256/E8/T8, low-level research path | Pass |
| `raw-max-geometry` | M1/H8192/I4096/E8/T8, direct/reference only | Pass |

Every row also captures and replays after changing valid expert IDs and
multiplying router weights by 0.7. It replays five times and compares to
a recomputed reference. Two independent streams are checked except for
the inplace case. These are bounded checks, not long-running stress tests.

### Sanitizer

All runs use `--error-exitcode 86 --target-processes all`, direct-only
execution and the same numerical/graph/two-stream validation.

| Workload | memcheck | racecheck | synccheck |
|---|---|---|---|
| M3/H256/I128/E4/T3, bias/mask/INT64/scale1.7 | 0 device errors with API reporting disabled | 0 hazards/errors/warnings | 0 errors |
| Actual M1 runner, H4096/I512/E256/T6 | 0 device errors with API reporting disabled | 0 hazards/errors/warnings, bias/mask/INT64 | 0 errors, bias/mask/INT64 |

Unfiltered `direct-memcheck.log` reports 34 `cuGetProcAddress_v2` API
errors during import and exits 86. This matches the prior import-only
control in `../validation/moe-workspace-cuda-20260911/`.
Filtered checks explicitly use `--report-api-errors no`.
They are **not unfiltered memcheck passes**. Race/synchronization errors
were never suppressed.

The initial racecheck and synccheck attempts failed SSH authentication
before remote execution. The read-only hostname probe and `klist -s`
succeeded; no tickets or packages were changed. A dedicated SSH control
connection was established and retry artifacts retain successful results.

This path avoids the old small-expert AOT alignment and Marlin kernels;
it does not repair their previously reported barrier/race failures.
Unsupported fallback still has those existing risks.

### CPU Tests

Scope: metadata gate, actual selector control flow, parameter forwarding,
and `MoeRunner.run` switch/budget precedence. Kernel arithmetic is not
substituted for GPU proof.

Defect analysis: no new confirmed functional defect in this scope. M4
performance regression is recorded and excluded from runner selection;
the existing CUDA failures remain separately documented.

Generated cases: **89**, comprising 45 metadata/early-error cases,
32 selector cases and 12 runner switch/budget combinations. Invalid dtype,
shape, device, strides, quantization and unsupported modes are tested.

Verification: each module's focused run passed on its first run, then
**415 passed** including 326 existing cases. The final post-format rerun
also passed 415 in 20.04 s, with two existing macOS AWQ/GGUF warnings.
The test-generation workflow and `utree flush` completed. Coverage was not
requested or measured. Pre-commit passed after its formatting-only changes.

## Reproduction and Evidence

Tracked harness: `benchmark/kernels/fused_moe_triton/validate_direct_decode.py`.
Local artifacts: `../validation/moe-direct-decode-20260912/`.
Remote artifacts: `/artifacts/moe-direct-decode-20260912/`.
Each run retains raw JSON, complete log and a `.run.json` command/commit/
GPU-occupancy/timestamp/exit-code sidecar. `summary.json` and `tables.md`
are derived from those files. `environment.json` includes actual imports,
versions, loaded CUDA library paths and source/config hashes.

Inside the verified container:

```bash
export PYTHONPATH=/sgl-workspace/sglang-bytedance/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
cd /sgl-workspace/sglang-bytedance

python3 benchmark/kernels/fused_moe_triton/validate_direct_decode.py \
  --runner-direct --iterations 200 --repeats 7 \
  --output /artifacts/moe-direct-decode-20260912/runner-m1-repeat.json
```

For the second row add `--hidden 2048 --intermediate 1024 --experts 64
--topk 4` and a distinct output name. For fallback testing use
`--runner-direct --tokens 4 --expect-fallback --skip-timing`.
Use new filenames when repeating to preserve original evidence.

## Limits and Next Decision

No real checkpoint or serving endpoint was loaded. DeepSeek-V4's native FP8
and modified activation are not supported by this experiment. No claim is
made about TTFT/TPOT, requests per second, full-model accuracy, speculative
decoding, KV capacity, multi-GPU communication, FP8 or INT4 performance.

The useful next step is to validate the M1 path on a supported real
unquantized checkpoint before widening the selector. Extending weight-only
quantization inside the direct loads is a separate hypothesis that needs
its own rounding, bandwidth and accuracy tests; it is not implemented here.

The feature remains default-off. The test processes and JIT workers have
exited, all GPUs returned to zero allocated process memory, and the isolated
container is retained with its artifacts. No formal PR was published.
