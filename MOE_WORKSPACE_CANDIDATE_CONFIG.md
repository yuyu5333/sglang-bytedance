# Triton Workspace: Candidate-Specific Configurations

This is the historical local-only iteration at `70cf5aa406`. Subsequent
device tests and fixes are recorded in
[MOE_WORKSPACE_CUDA_VALIDATION.md](MOE_WORKSPACE_CUDA_VALIDATION.md).
They show a memory/latency tradeoff and unresolved sanitizer findings,
not a GPU speedup or production acceptance.

## Decision

Continue the default-off budget prototype with **per-candidate existing
configuration selection**. The initial implementation selected a Triton
config for the full batch and held it for every chunk. A much smaller
chunk could therefore retain a large block-M and process excess padding.

| Item | Value |
|---|---|
| Baseline implementation | `9af19d5b5d` |
| Branch | `feat/marlin-moe-bounded-workspace` |
| Production changes | Two Python files; Triton config resolution and budget adapter |
| Marlin / CUDA kernels / communication | Unchanged |
| Shared budget | Still default-off; same byte-budget contract and exclusions |
| Remote / GPU / package changes in this iteration | None |

This change reduces **modeled padded work in some configurations**, not
measured latency. Default heuristic results at 64 MiB remain unchanged.
The repository's explicit H20 paired TMA configuration still falls back
and receives no budget guarantee. It is not evidence of H20 acceleration.

## Implementation

`_resolve_fused_moe_config` accepts an optional candidate token count.
Omitting it preserves full-batch selection. A supplied count must be
positive and no larger than the input batch. All existing tuning-table,
override and default-heuristic selection logic is reused.

The Triton adapter uses a call-private `_TritonCandidateEstimate`:

1. The shared planner visits the full batch, then descending power-of-two
   buckets, exactly as before.
2. For each visited candidate, resolve the existing up/down configs and
   compute scratch using that candidate's block-M.
3. Retain the resolved configs and estimate in one call-private cache.
4. Execute the selected cap with the exact retained config for all chunks
   and the tail. Do not reselect a tail config.

The shared planner still makes no monotonicity assumption. Configs and
estimates remain independent of GPU routing values and available memory.
The new count argument avoids creating input views during planning.
No activation, routing or output data storage is allocated until selection
finishes. Config lookup may allocate ordinary host dictionaries.

If any visited candidate requests active TMA, return the explicit
unsupported warning and original path. The adapter does not silently
disable TMA, change its layout or skip to a smaller candidate. Ordinary
budget failure and configuration errors still propagate as errors.
This is conservative: a full batch with non-TMA config may now fall back
if a later candidate chooses TMA.

The selected up and down block-M remain equal under the existing resolver
contract. Cached source dictionaries are copied before removing TMA flags.
Candidate caches do not survive the invocation. Calls with sufficient budget
for the full batch resolve just one config.

## Offline Method

`analyze_triton_workspace_configs.py` executes the production resolver,
candidate estimator and shared planner. It substitutes only device/version
configuration discovery and TMA capability:

- Real input/weight values are not loaded; tensors are metadata-only.
- Config files are parsed into explicitly chosen tuning snapshots.
- The default case uses the production default heuristic with no tables.
- An up-only snapshot reuses the same map for down, preserving all flags.
- The paired case loads both up/down maps, preserving TMA flags.
- CPU routing uses seed 42 and distinct top-k experts, uniform or hot.
- Storage checks materialize the same conservative tensor-byte formula
  as the shared-budget report, on CPU or meta according to size.

The baseline side reproduces the `9af19d5b5d` fixed-config policy using the
same estimator. No GPU kernel, allocator trace, timing, checkpoint, KV
dtype, TP/PP/DP runtime, serving memory fraction, concurrency, speculative
decoder or model accuracy is involved.

Synthetic dimensions: BF16 activation storage, `M=8192, H=4096, I=512,
E=256, top-k=6`. TMA capability is explicitly assumed available. A named
H20 config file is **not** a live device/runtime selection: cross-version
fallback on a real server may find a different down table.

## Results

### Default Heuristic

The default nonquantized heuristic changes block-M from 64 to 16 only
when candidate tokens are no larger than the expert count.

| Budget MiB | Cap, Unchanged | Before / After Block-M | Before / After Uniform Padded Routes |
|---|---:|---|---|
| 16 | 128 | 64 / 16 | 996,928 / 249,232 |
| 64 | 512 | 64 / 64 | 262,144 / 262,144 |
| 256 | 2048 | 64 / 64 | 66,176 / 66,176 |

The 16 MiB case reduces modeled padded rows by 75%. The other two cases
do not reduce padded work. This is not a claim that the existing default
heuristic is optimal for chunked execution.

### Explicit Non-TMA Snapshot

Source: `triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`, deliberately
supplied as an up-only map and reused for down. Its 512-token entry selects
block-M 16; its large-batch entry selects 64.

| Budget MiB | Cap | Before / After Block-M | Before / After Uniform Padded Routes | Candidate Estimate MiB |
|---|---:|---|---|---:|
| 16 | 128 | 64 / 16 | 996,928 / 249,232 | 8.289 |
| 64 | 512 | 64 / 16 | 262,144 / 71,824 | 33.058 |
| 256 | 2048 | 64 / 64 | 66,176 / 66,176 | 132.223 |

At 64 MiB, padded rows decrease **72.6013%** relative to fixed-config
chunking. Expert-chunk visits remain **4,096** and chunk count remains
**16**. The unchunked model has 57,024 padded rows and 256 expert visits,
so the remaining chunking overhead is substantial.

The scratch estimate only changes from 34,761,592 to 34,663,960 bytes in
that example: activation scratch is unchanged, routing capacity decreases.
The improvement is in modeled padded computation, not a major additional
memory saving.

### Explicit Paired TMA Snapshot

Source: the `triton_3_5_1` H20 up and `_down` files with the same E/N.
The down map requests TMA. Both policies report **unsupported** for all
three budgets. No flags were stripped and no bounded-memory or padded-work
result is claimed for this case.

### Costs and Limits

- Candidate config queries increase from one to **7 / 5 / 3** for the
  16 / 64 / 256 MiB examples. This is CPU planning overhead, not measured
  inference overhead.
- Top-k route count, repeated expert visits and chunk launch count remain
  unchanged in these examples.
- Smaller block-M may reduce tensor-core efficiency or change weight reuse.
- Config entries tuned for stand-alone batches need not be optimal for a
  sequence of chunks. No new entries are generated here.
- Changes to kernel tiling may change numerical rounding. CPU exact
  agreement does not establish CUDA bitwise equality.
- The budget still excludes full output, weights, caller tensors, dispatch,
  allocator reservations and graph pools.

## Validation

### Scope

Tests extend `test_workspace_adapters.py` and retain its CPU substitutes
for CUDA operations. The real Python candidate resolver and executor are
used. Existing planner, Marlin, runner-extension and config tests remain
in the regression set.

### Defect Analysis

No new confirmed functional defect was found in this scope. Tests verify
that configuration errors are not hidden as budget fallback, TMA rejection
occurs before scratch allocation, and estimates match the execution config.
Device behavior remains unverified.

### Generated Cases

**18 new cases**: optional/valid/invalid token counts, varying configs,
inplace/outplace, scaled top-k, tail config stability, TMA on full/intermediate
candidates and either GEMM, budget/config errors, repeated calls, and
nonmonotone candidate costs.

The first adapter run passed **128/128**, including the previous 110 cases.
No failing assertions were weakened or production logic changed during the
test loop. Coverage was not requested or measured. `utree flush` completed.

### Final Results

| Check | Result |
|---|---|
| Combined CPU regression | 315 passed: 18 new and 297 previous cases |
| Adapter / planner / Marlin / runner-config | 128 / 26 / 150 / 11 passed |
| Repository pre-commit and diff checks | Passed |
| Real production imports in offline script | Passed with checkout PYTHONPATH |
| Local environment | macOS arm64, Python 3.13.14, Torch 2.14.0, no CUDA |
| GPU numerical/allocator/graph/sanitizer/performance validation | Not run |

Pre-commit initially reformatted the test and analysis script; re-running
passed. No hook was skipped. The two warnings in the final CPU run are the
existing AWQ/GGUF unsupported-platform warnings, not test failures.

## Reproduction

Run with the local compatible Python environment and explicit checkout:

```bash
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  test/registered/unit/layers/moe/test_workspace_adapters.py \
  test/registered/unit/layers/moe/test_workspace_policy.py \
  test/registered/unit/layers/moe/test_marlin_chunked_workspace.py \
  test/registered/unit/layers/moe/test_moe_runner_extensions.py \
  test/registered/unit/layers/moe/test_fused_moe_triton_config.py

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_triton_workspace_configs.py \
  --output ../validation/moe-workspace-candidate-config-20260911/heuristic.json

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_triton_workspace_configs.py \
  --up-config 'python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json' \
  --output ../validation/moe-workspace-candidate-config-20260911/h20-up-only-snapshot.json
```

For the paired TMA case, pass the `triton_3_5_1` up file and matching
`_down.json` via `--down-config`. The JSON records explicit inputs, selected
configs, candidate-query sequence, code/config hashes and Git state.
An additional `--tokens 8193 --budgets-mib 16 64 600` run covers a tail
and a full-batch fitting budget with the non-TMA snapshot.

No remote validation, push or formal PR publication was part of this local
iteration. The subsequent H20 validation was pushed and synchronized; no
formal PR has been published.
