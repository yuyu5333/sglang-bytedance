# MoE Workspace Budget: H20 Validation

Date: 2026-09-11. Branch: `feat/marlin-moe-bounded-workspace`.
Tested implementation: `1e1848c1eb4f4c5e395daea41e2e5c9f404c5abe`.

## Decision

Keep the feature **default-off and experimental**. Single-rank CUDA tests
confirm that chunking can substantially reduce the call's allocated peak,
but every measured chunked prefill configuration is slower than its own
unchunked baseline. This is a memory/latency tradeoff, not a speedup.

Two Python defects found during testing were fixed and retested. Device
validation is **not all-pass**: the installed small-expert AOT alignment
kernel fails synccheck, and existing Marlin GEMMs report shared-memory
races, including with the workspace budget disabled. Those reports remain
unresolved; numerical agreement does not establish synchronization safety.

This report supersedes the "not yet tested on GPU" status of the earlier
[shared-budget](MOE_WORKSPACE_BUDGET.md) and
[candidate-config](MOE_WORKSPACE_CANDIDATE_CONFIG.md) reports. Their CPU
storage/routing simulations remain historical evidence, not GPU results.

## Environment and Scope

| Item | Observed Value |
|---|---|
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| Container | `kvbit-future-ab-20260908`, ID prefix `af0763b0e263` |
| Repository | `/sgl-workspace/sglang-bytedance` |
| GPUs | 8 H20, SM90, 78 SMs, 60 MiB L2; each test uses one GPU |
| Performance GPU | GPU 0, UUID `1c2c22aa-44de-d80b-fff2-03e1842c1b32` |
| Host driver / CUDA compiler | `535.161.08` / `13.0.88` |
| Loaded CUDA compatibility library | `/usr/local/cuda-13.0/compat/libcuda.so.580.82.07` |
| Python / Torch / Triton | `3.12.3` / `2.13.0+cu130` / `3.7.1` |
| Compute Sanitizer | `2025.3.1.0` |
| `sglang` import | `/sgl-workspace/sglang-bytedance/python/sglang/__init__.py` |
| `sgl_kernel` import | `/usr/local/lib/python3.12/dist-packages/sgl_kernel/__init__.py` |
| Runtime initialization | Offline `ServerArgs(model_path="dummy")`, single-rank Gloo groups |
| Topology | TP/PP/DP/EP = 1/1/1/1; no A2A or fused collective |
| Model, input/output sequence length, requests, mem-fraction, KV dtype | N/A: synthetic MoE operators, not a serving workload |
| Speculative decoding / HTTP | None / not tested |
| Package or installed AOT changes | None |

Every execution sets `PYTHONPATH=/sgl-workspace/sglang-bytedance/python`,
`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`, and `HF_HUB_OFFLINE=1`.
The remote checkout was clean before validation. Source changes were made
locally, committed, pushed, then synchronized with Git fast-forward only.
Marlin JIT compiled against the target checkout. No source was copied or
patched inside the container.

The mounted checkpoint `/models/DeepSeek-V4-Flash` was inspected but not
loaded for computation. Its config has FP8 weights and `swiglu_limit=10.0`,
outside these adapters' plain-activation/nonquantized-Triton support.
The following tests do not validate DeepSeek-V4 model accuracy, token-pool
capacity, TTFT, TPOT, or throughput.

## Fixes Found on GPU

| Issue | Evidence and Cause | Fix and Verification |
|---|---|---|
| Shared policy incorrectly rejects ordinary single-rank calls | At `81210a5aef`, both smoke tests stop before kernels. `is_allocation_symmetric()` is an allocation-policy preference and is true without DP; it does not establish that symmetric allocation is enabled. | `b75c56f07e`: require symmetric memory enabled, symmetric allocation policy, and TP world size greater than one. Eight CPU flag/world-size combinations pass; subsequent real CUDA calls enter the adapters. |
| Triton top-k 1 with non-unit scaling reads unwritten down scratch | `triton-fp16-topk1.json` at `b75c56f07e`: budget 0 produces nonfinite values; positive budgets are finite. GEMM2 writes directly to output, but scaled reduction reads `intermediate_cache3`. | `1e1848c1eb`: non-unit scaling forces GEMM2 to write the intermediate. Same FP16/GELU/top-k 1/scale 0.5/inplace/bias GPU case passes; max absolute error `1.52587890625e-5`. |

The second issue is in the default Triton dataflow, not the budgeted
scratch path. Its failed baseline is retained and excluded from valid A/B
results. The fix changes the destination selected before reduction; it
does not alter the CUDA reduction kernel or relax tolerances.

The initial validation harness used a nonexistent model name, triggering
Hugging Face lookup. Those two processes were terminated before GPU work.
`81210a5aef` changed the harness to the existing offline `"dummy"` convention.
That setup failure is not counted as a kernel test.

### CPU Regression

Scope: planner, adapters, Triton/Marlin Python dataflow, runner extensions,
and config selection. CUDA operations are substituted in CPU dataflow tests;
they do not certify CUDA kernels.

Defect analysis: the two fixes above precede regression generation. The
new cases assert correct behavior, not the previously observed failures.

Generated cases: 8 symmetric-allocation combinations and 3 top-k 1 scaling
cases (`0`, `0.5`, `1.7`), in addition to the previous 315 cases. The scaling
test checks finite output, exact scaling of the unscaled top-k 1 result,
and default/scratch equality.

Validation: **326 passed, 2 existing platform warnings** in 11.23 s.
The focused scaling run passed 6 cases. Pre-commit and `git diff --check`
passed. Test-generation steps and `utree flush` completed. Coverage was
not requested or measured. The final rerun also passed 326 cases in 12.10 s.
Evidence: `cpu-gate-fix.xml`, `cpu-topk-fix.xml`, `cpu-final.xml`.

## CUDA Numerical Matrix

Default small geometry is `M17/H256/I128/E4`, out-of-place, uniform routing,
INT32 IDs, scale 1, plain SiLU, no bias. All rows compare real `MoeRunner`
execution with an independent per-expert FP32 matmul reference, rounded at
activation/output boundaries. `atol=0.03, rtol=0.03` was fixed throughout.
This is an operator tolerance check, not model accuracy or bitwise
equivalence across Marlin tilings.

| Artifact Stem | Backend / Activation Dtype | Changes from Defaults | Positive Caps | Numerical / Changed-Route Graph / Two Streams |
|---|---|---|---|---|
| `triton-smoke-fixed` | Triton / BF16 | top-k 2, bias, masked routes, INT64 IDs | 4 | Pass / Pass / Pass |
| `triton-fp16-topk1-fixed` | Triton / FP16 | top-k 1, GELU, scale 0.5, bias, inplace | 4, 1 | Pass / Pass / N/A, inplace |
| `triton-bf16-zero` | Triton / BF16 | top-k 1, scale 0 | 4, 1 | Pass / Pass / Pass |
| `triton-bf16-tail` | Triton / BF16 | M257/H512/I256/E16, top-k 3, GELU, scale 1.7, hot routes, masks, bias, INT64 | 64, 16 | Pass / Pass / Pass |
| `int4-fp16-smoke` | Marlin INT4 / FP16 | top-k 3, scale 1.7, bias | 4, 1 | Pass / Pass / Pass |
| `int4-bf16-smoke` | Marlin INT4 / BF16 | top-k 2, scale 1.7, bias | 8, 1 | Pass / Pass / Pass |
| `int8-bf16-smoke` | Marlin INT8 / BF16 | top-k 3, scale 0.5, bias, hot routes | 4, 1 | Pass / Pass / Pass |
| `mxfp4-smoke-fixed` | Marlin MXFP4 / BF16 | top-k 2, bias | 4 | Pass / Pass / Pass |
| `mxfp4-no-bias` | Marlin MXFP4 / BF16 | top-k 2, scale 0.5, INT64 IDs | 8, 1 | Pass / Pass / Pass |

The first Triton and MXFP4 smoke reports use `b75c56f07e`; all remaining
rows use `1e1848c1eb`. Marlin INT4/INT8 weights use the existing quantization
helper and group 128; MXFP4 uses E2M1 packed values and group-32 E8M0 scales
through the actual Marlin repacking path.

Graphs are captured for positive budgets, then replayed five times after
valid expert IDs are shifted modulo E and router weights multiplied by
0.7. Tails reuse the selected cap's config. Out-of-place calls are also
run on two independent CUDA streams. This is a bounded test, not a
long-duration stress test or graph-pool memory measurement.

`triton-h20-tma.json` separately confirms that runtime selection for
`M8192/H4096/I512/E256/top-k6` finds the paired Triton-3.5.1 H20 tables and
requests TMA. The planner reports `TMA layout for 8192-token candidate`.
The harness stops before MoE execution in this case. Its `pass=true`
means expected unsupported detection, **not** budgeted execution or
verification of the fallback CUDA path.

## Prefill Memory and Latency

These are four independent, fresh A/B workloads on GPU 0. Only comparisons
within a backend/format/workload are meaningful. The same generated
weights, activations, IDs, seed 42, route weights, activation and topology
are retained while changing the byte budget. No old INT4-KV measurements
are used as baselines.

Method: no other test or JIT compilation was running when each performance
process started; all eight GPUs were idle before the series. Warmup is
10 calls per budget, then 5 randomized-order rounds of 20 calls per budget.
CUDA events enclose eager dispatch, including host launch gaps. Wall-clock
values are also retained. No clock/power settings were changed, and
Triton's round-to-round variation is reported below.

CUDA Graph timing is disabled here. Concurrent-stream correctness runs
finish before timing. These numbers are not kernel-only latency or
request throughput. All four reports pass numerical and two-stream checks.

### Runtime E64 Workloads

Common configuration: `M8192/H2048/I1024/E64/top-k4`, BF16 activations,
SiLU, scale 1, uniform distinct top-k, no bias, no masks, out-of-place.
Triton uses the existing nonquantized heuristic, block-M/N/K=64/64/32,
group-M=8, no TMA. Marlin uses its existing block-M/split-K heuristics.
INT4 uses group 128; MXFP4 uses group 32.

Call peak is `max_memory_allocated - allocated_before_call`, including
the new **32 MiB full output**, excluding already-live inputs/weights.

| Backend / Weight Format | Budget MiB | Cap / Chunks | Call Peak MiB | Peak Reduction | CUDA Median ms [Round Min, Max] | Latency Change |
|---|---:|---|---:|---:|---|---:|
| Triton BF16 | 0 | 8192 / 1 | 352.1436 | - | 3.3639 [3.3194, 3.9571] | - |
| Triton BF16 | 64 | 1024 / 8 | 72.0649 | 79.54% | 5.0520 [4.7392, 5.7500] | +50.18% |
| Triton BF16 | 16 | 256 / 32 | 42.0415 | 88.06% | 13.2476 [13.0102, 14.9935] | +293.81% |
| Marlin INT4 | 0 | 8192 / 1 | 224.1450 | - | 5.8962 [5.8958, 5.9019] | - |
| Marlin INT4 | 64 | 2048 / 4 | 80.0986 | 64.26% | 7.1200 [7.1164, 7.1340] | +20.76% |
| Marlin INT4 | 16 | 512 / 16 | 44.0430 | 80.35% | 8.8359 [8.8340, 8.8415] | +49.86% |
| Marlin MXFP4 | 0 | 8192 / 1 | 224.1450 | - | 5.8785 [5.8760, 5.8816] | - |
| Marlin MXFP4 | 64 | 1024 / 8 | 75.5337 | 66.30% | 8.3302 [8.3256, 8.3323] | +41.71% |
| Marlin MXFP4 | 16 | 256 / 32 | 47.7642 | 78.69% | 12.7668 [12.7638, 12.7705] | +117.18% |

Sources: `prefill-triton-runtime.json`, `prefill-int4-runtime.json`,
`prefill-mxfp4-runtime.json`. Marlin timing is diagnostic while the
kernel-level race reports below remain unresolved.

### Explicit Non-TMA E256 Diagnostic

Configuration: Triton BF16, `M8192/H4096/I512/E256/top-k6`, otherwise the
same method. Both baseline and budgeted execution explicitly override
block-M/N/K=16/64/32, group-M=8, no TMA. This is not the runtime TMA path
and not a benchmark of the earlier up-only tuning-table snapshot.

The new full output is **64 MiB**. Source:
`prefill-triton-fixed-e256.json`.

| Budget MiB | Cap / Chunks | Call Peak MiB | Peak Reduction | CUDA Median ms [Round Min, Max] | Latency Change |
|---:|---|---:|---:|---|---:|
| 0 | 8192 / 1 | 592.2158 | - | 11.0529 [10.8722, 11.2142] | - |
| 1024 | 8192 / 1 | 592.2173 | Approximately zero | 11.1765 [10.8506, 11.6558] | +1.12% |
| 64 | 512 / 16 | 97.8179 | 83.48% | 21.9024 [21.7567, 22.3500] | +98.16% |
| 16 | 128 / 64 | 72.2915 | 87.79% | 79.3334 [78.8237, 79.6834] | +617.76% |

The full-fitting-budget timing overlaps the baseline range; the observed
1.12% difference is not a statistically established overhead estimate.
The chunked paths are substantially slower in these measurements.

### Budget Accounting

The policy covers modeled explicit call-private scratch, not output,
weights, caller inputs, reserved allocator memory, graph pools or the
whole process. The validation check is
`call_peak <= budget + full_output_bytes + 1 MiB`; the extra allowance
is a test allowance for allocator/lifetime overhead, not a strict-byte
guarantee. All positive-budget measurements satisfy it.

| Workload | Budget MiB | Planned Scratch MiB | Measured Call Peak Minus Full Output MiB |
|---|---:|---:|---:|
| Triton E64 | 64 / 16 | 40.0640 / 10.0402 | 40.0649 / 10.0415 |
| Marlin INT4 E64 | 64 / 16 | 48.0969 / 12.0415 | 48.0986 / 12.0430 |
| Marlin MXFP4 E64 | 64 / 16 | 43.5652 / 15.7756 | 43.5337 / 15.7642 |
| Triton fixed E256 | 1024 / 64 / 16 | 528.4317 / 33.0581 / 8.2895 | 528.2173 / 33.8179 / 8.2915 |

The E256/64 MiB case exceeds its tensor-byte estimate by about 0.760 MiB,
though it remains well below the configured budget after subtracting
output. An allocation-by-allocation trace was not taken, so this gap is
not assigned to a specific allocator action. Peak-minus-output is also
not a full tensor-lifetime trace, particularly for Marlin's delayed output.
Reserved peaks remain high after full-batch warmup: this experiment does
not demonstrate returning memory to the driver or increasing the KV pool.

More chunks repeat alignment, two GEMMs, activation and reduction, and
can change expert padding/tiling and weight reuse. MXFP4 additionally
budgets non-atomic FP32 reduction scratch, selecting a smaller cap than
INT4 at the same budget. These code-level costs explain why lower live
storage need not improve speed. Their individual HBM/launch contributions
were not profiled, and no causal percentage is assigned to them.

## Sanitizer Results and Remaining Defects

Sanitizer timings are excluded from the performance tables. The process
exit code is authoritative even when numerical checks write `pass=true`.
All checks use `--error-exitcode 86 --target-processes all`.

| Artifact Stem | Check | Result |
|---|---|---|
| `triton-memcheck`, `mxfp4-memcheck` | Unfiltered memcheck, M9/E4/T2, cap4, bias | Exit 86, each 34 CUDA API errors during import |
| `import-only-memcheck` | Only import `sgl_kernel`, no MoE | Exit 86, same 34 `cuGetProcAddress_v2` errors |
| `triton-memcheck-device`, `mxfp4-memcheck-device` | Same workload, `--report-api-errors no` | Exit 0, 0 device errors; graph and two-stream checks pass |
| `triton-racecheck` | M9/E4/T2, bias/masks/INT64, cap4 | Exit 0, 0 hazards/errors/warnings |
| `triton-synccheck`, `mxfp4-synccheck` | M9/E4/T2, cap4 | Exit 86, 32 barrier errors in AOT alignment on first budget-0 call |
| `triton-synccheck-e64`, `mxfp4-synccheck-e64` | M128/E64/T2, cap64, bias | Exit 0, 0 errors; graph and two-stream checks pass |
| `mxfp4-racecheck` | M9/E4/T2, cap4, bias, graph/two streams | Exit 86, 90 displayed race reports; numerical checks pass |
| `mxfp4-racecheck-baseline` | Budget 0 only, no graph, no warmup, bias | Exit 86, 4 displayed race reports |
| `mxfp4-racecheck-no-bias` | Same baseline-only test, no bias | Exit 86, 4 displayed race reports |
| `int4-racecheck` | M9/E4/T2, budget0/cap4, bias, no graph | Exit 86, 17 displayed race reports |

Report counts depend on the number of launches; 90 versus 4 is not a
like-for-like measure of race severity. No race or barrier errors were
suppressed. The API-only filter was used only after the import-only
control reproduced the API errors; this is not an unfiltered memcheck pass.

### Small-Expert Alignment Barrier

Both backends fail before their first GEMM, in the installed
`common_ops.abi3.so` symbol
`moe_align_block_size_small_batch_expert_kernel<...,256>` at offset `0x910`.
This occurs with INT32 and INT64 IDs while the budget is disabled.

The corresponding checkout source is
`python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu:240-320`.
Its fill threads execute three barriers in one branch and return; the
histogram threads execute three other barriers after that branch.
Synccheck reports divergent block synchronization. The dispatcher selects
this kernel for fewer than 1024 routes and at most 64 internal buckets.
The E64 control uses 65 buckets after the invalid-expert bucket is added
and passes through the other alignment path.

This source pattern is consistent with the runtime report, but the
installed AOT binary was not rebuilt or certified against this source.
Its loaded path and SHA256 are recorded separately. Resolving it requires
a matched-source AOT build and an isolated small-expert reproduction;
changing the installed package was outside this test's authorized
environment changes.

### Marlin Shared-Memory Race

The report is within the existing JIT `sglang::device::marlin_moe::Marlin`
kernel, not between independent calls. MXFP4/BF16 with bias reports read
offset `0x4860` and write offset `0x7fc0`; without bias the offsets are
`0x45c0` and `0x7b60`. INT4/BF16 reports the same class of read/write hazard.
Baseline-only execution reproduces it without chunking, graph replay,
concurrency or bias. These controls rule out those features as necessary
conditions, but do not prove absence of additional chunk-specific hazards.

`python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h` and the
GEMM wrapper are unchanged relative to branch base `fb45f5ffbc`.
The kernel aliases weight-pipeline and reduction shared storage and has
multiple reuse/synchronization boundaries. The precise offending source
instructions have not been mapped from SASS; a missing barrier is a
candidate explanation, not an established root cause. A line-info build,
instruction mapping and targeted synchronization experiment are needed
before declaring a fix or a tool false positive.

Do not enable this feature by default or label Marlin synchronization
validated until these reports are resolved. This test intentionally does
not replace another user's packages, stop other containers, or broaden
the change into an unverified CUDA kernel rewrite.

## Reproduction and Artifacts

Tracked harness:
`benchmark/kernels/fused_moe_triton/validate_workspace_cuda.py`.
SHA256: `08bb0723dabf9c1c7919f70fb3406f2288000443a88bd37a94fb6dfc002cf2df`.

Local raw evidence:
`../validation/moe-workspace-cuda-20260911/`.
Remote JSON:
`/artifacts/moe-workspace-cuda-20260911/`, backed by host
`/mnt/nvme2/kvbit-future-20260908/moe-workspace-cuda-20260911/`.

Each named report is a `.json`; runner executions also retain local
`.log` and `.run.json` with command, commit, GPU occupancy before execution,
timestamps and exit code. The first smoke and gate-failure reports were
retrieved from direct SSH runs and do not have `.run.json` sidecars.
`summary.json` distinguishes failure from unsupported, records raw hashes,
and retains the failed runs. `environment.json` records source/binary hashes,
actual loaded library paths, versions, container identity and model config.
`prefill-tables.md` is generated from raw JSON, not from remembered values.

Run inside the verified container with explicit environment:

```bash
export PYTHONPATH=/sgl-workspace/sglang-bytedance/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
cd /sgl-workspace/sglang-bytedance

python3 benchmark/kernels/fused_moe_triton/validate_workspace_cuda.py \
  --backend triton --tokens 8192 --hidden 2048 --intermediate 1024 \
  --experts 64 --topk 4 --caps --budgets-mib 64 16 --skip-graphs \
  --warmup 10 --iterations 20 --repeats 5 --port 29763 \
  --output /artifacts/moe-workspace-cuda-20260911/prefill-triton-runtime.json
```

The INT4/MXFP4 E64 runs use the same arguments except for `--backend
marlin-int4` or `marlin-mxfp4` and output name. The E256 diagnostic uses
`--hidden 4096 --intermediate 512 --experts 256 --topk 6 --config fixed
--block-m 16 --budgets-mib 1024 64 16`.

Example unfiltered sanitizer reproducer:

```bash
compute-sanitizer --tool synccheck --error-exitcode 86 \
  --target-processes all python3 \
  benchmark/kernels/fused_moe_triton/validate_workspace_cuda.py \
  --backend triton --tokens 9 --bias --masked --ids-dtype int64 \
  --caps 4 --warmup 1 --iterations 1 --repeats 1 --port 29763 \
  --output /artifacts/moe-workspace-cuda-20260911/triton-synccheck.json
```

Use distinct output names to preserve original evidence. Full commands for
all runner-managed cases are in their `.run.json` sidecars.

## Remaining Acceptance

1. Resolve the AOT alignment barrier and Marlin race reports, then repeat
   the same failing checks without filtering synchronization errors.
2. Add a tensor-allocation trace and separate CUDA Graph pool accounting
   before making a strict scratch-limit claim.
3. Validate supported real checkpoints and serving workloads, including
   accuracy and separately controlled prefill/decode tests. None was run.
4. TP/EP communication, LoRA, modified activations, quantized Triton, TMA,
   torch.compile, graph recapture and long-running concurrency remain
   unsupported or outside this validation.

All test/JIT processes have exited. The isolated container is retained
with its original init/sleep process and artifacts; other containers,
services, package versions and model files were not changed. No formal
PR was published.
