# Budgeted MoE Workspace: Marlin and Triton

## Status

Implemented a default-off, shared **call-private scratch budget policy** with
Marlin and Triton adapters. This is a local memory-management prototype.
No GPU memory, speed, accuracy, CUDA Graph or concurrency claim is made.

The initial implementation is `9af19d5b5d`. The subsequent
[candidate-configuration iteration](MOE_WORKSPACE_CANDIDATE_CONFIG.md)
selects Triton configs per candidate chunk rather than from the full batch.
The initial fixed-config measurement tables below are retained as history.

| Item | Value |
|---|---|
| Branch | `feat/marlin-moe-bounded-workspace` |
| Original branch base | `fb45f5ffbc`, `feat/dsv4-direct-int4-g32-e4m3` |
| Previous Marlin token-cap prototype | `0ca13a9cdf` |
| Shared control | `SGLANG_MOE_WORKSPACE_BUDGET_BYTES=0` |
| Runner override | `MoeRunnerConfig.workspace_budget_bytes` |
| Execution scope | Standard local CUDA MoE, after dispatch and before pre-permute |
| Local validation | macOS arm64, Python 3.13.14, Torch 2.14.0, CUDA unavailable |
| Kernel changes | None |
| Remote access / deployment / PR publication | Not performed |

The earlier [Marlin token-cap report](MARLIN_MOE_BOUNDED_WORKSPACE.md) is
historical evidence for the activation-pair prototype. Its storage figures
exclude routing/locks/reduction and must not be presented as this policy's
complete scratch estimates.

## Configuration and Contract

Example environment setting, before runner warmup:

```bash
export SGLANG_MOE_WORKSPACE_BUDGET_BYTES=67108864
export SGLANG_MARLIN_MOE_CHUNK_SIZE=0
```

Runner configuration takes precedence:

| Value | Behavior |
|---|---|
| `None` | Inherit environment budget |
| `0` | Explicitly disable shared policy |
| Positive integer | Budget in bytes |
| Negative value | `ValueError` before execution |

The old `SGLANG_MARLIN_MOE_CHUNK_SIZE` token-cap experiment remains available
when shared budgeting is disabled. Enabling both controls raises an error.
Direct calls that bypass `MoeRunner.run` do not inherit the shared policy.
The low-level Marlin function now also accepts an optional `chunk_size`;
omitting it preserves the old token-cap behavior.

The budget covers the modeled explicit scratch tensor storage in one
supported invocation. It includes activation scratch, aligned routing,
Marlin locks and its non-atomic FP32 reduction temporary. It excludes:

- Expert weights/scales/biases and caller-owned activations/top-k/router logits.
- Full output, dispatch/all-gather/all-to-all buffers and caller intermediates.
- Allocator reservations/rounding/fragmentation, JIT and CUDA Graph pools.
- Other invocations, streams, layers or model subsystems.

This is not a process VRAM limit, an allocator-enforced quota or a promise to
avoid OOM. It does not resize the KV token pool.

Unsupported modes emit an explicit, rate-limited warning and use the original
path: `MoE workspace budget is not enforced for ...; using the original path`.
Their scratch is **not bounded by this policy**. When a supported nonempty
input has no fitting candidate, planning raises `ValueError` before
alignment or scratch allocation. Empty supported inputs return empty outputs
without locks or route scratch.

## Execution

`MoeRunner.run` checks the budget before fused-function dispatch and before
pre-permute. `workspace_policy.py` is CPU-importable and independent of
runner/kernel imports. `moe_runner/workspace.py` owns capability checks and
the two backend adapters.

Planning tries the full batch first, followed by descending powers of two
not larger than the batch. It chooses the largest fitting **candidate**, not
necessarily the largest fitting integer. It does not assume monotone costs:
Marlin block-M changes and FP32 reduction sizing can break that assumption.

Inputs are tensor metadata, backend configuration and a fixed byte budget.
No routing-value D2H read, free-memory query or new collective is added.
Do not mutate these controls on a live server after warmup/capture.

### Marlin Adapter

The adapter derives the existing Marlin block-M heuristic and reduction
mode, plans a cap, and forwards it to `fused_experts_none_to_marlin`. The
existing token-chunk executor reuses one gate/down storage, one gated
activation storage and per-call locks. It keeps the chosen block-M for the
tail. One-token alignment uses the existing fast path when eligible.

Marlin retains its output ownership: matching zero-copy destination or a
separate output. The runner wrapper does not adopt Triton's inplace setting.
An offset alias into the input returns to the original path with a warning;
exact aliases retain existing semantics.

### Triton Adapter

Configuration selection was separated from routing alignment. The adapter
now resolves an existing configuration for each candidate token count,
rejects TMA, and plans before allocating route scratch. The selected
candidate's configuration determines its estimate and is reused for every
chunk, including the tail. Routing IDs are independently aligned for each
chunk. No GPU tuning is performed by this selection.

Three call-private buffers are reused: gate/up `[C*T, 2I]`, activation
`[C*T, I]` and down `[C, T, H]`. The full output follows the original Triton
`inplace` setting. Each chunk writes only its output slice. The outer
dispatcher combine is not split or repeated.

`_fused_moe_kernel_sequence` accepts optional scratch and output buffers,
validates geometry/dtype/device and uses smaller views for tails. Without
these arguments it keeps its original allocation path. Scratch execution
uses an initialized down buffer even for top-k 1; it uses the existing
reduction op rather than the small-batch torch.compile reduction path.
Both cached up and down config dictionaries are copied before removing
TMA flags, so a rejected adapter probe cannot alter fallback configuration.

## Capability Matrix

| Mode | Marlin | Triton |
|---|---|---|
| Contiguous 2-D CUDA FP16/BF16 activation | Yes | Yes |
| Standard unpacked top-k, FP32 weights, INT32/INT64 IDs | Yes | Yes |
| Gated plain SiLU | Yes | Yes |
| Gated plain GELU | No | Yes |
| Existing 4/8-bit Marlin metadata | Yes, subject to existing kernel contracts | N/A |
| MXFP4 with E8M0 scales | BF16 activation only | No |
| Unquantized FP16/BF16 expert weights | N/A | Yes |
| FP8/INT8/INT4/MXFP8 Triton flags or quantization scales | N/A | No |
| Act-order permutation / explicit expert map | No | N/A |
| TMA / interleaved fused SwiGLU | N/A | No |
| Standard local masked expert IDs | No explicit EP map accepted | Existing filter behavior |
| Packed/ragged top-k or pre-quantized dispatch | No | No |
| LoRA/hooks / modified or nongated activation | No | No |
| Router weighting on input / no-combine | No | No |
| A2A / activation all-gather / communication overlap | No | No |
| Fused sum-all-reduce / symmetric-memory allocation | No | No |
| torch.compile / batch-invariant mode / custom runner core | No | No |
| CUDA Graph capture/replay | Unverified | Unverified |

CUDA only is intentional for this version. HIP and CPU fall back. The
Triton weight geometry is `[E, 2I, H]` and `[E, H, I]`. A backend not in this
table has no adapter. DeepGEMM's existing masked/compact policy is unchanged.
Allowlisted metadata is not certification of every quantized model.

## Storage Model

For a cap `C`, top-k `T`, local expert count `E`, hidden `H`,
intermediate `I`, block-M `B`, and assumed/probed SM count `S`, define:

```text
R = C * T
P = R * B                        if R < E + 1
    R + (E + 1) * (B - 1)       otherwise
Q = 4 * (P + ceil(P / B) + E + 3)

Marlin activations = 2 * R * (max(2I, H) + I)
Marlin locks       = 16 * S
Marlin reduction   = 4 * min(max(2I, H) * P, S * 4 * B * 256)
                     doubled for B=8, zero for atomic mode
Marlin estimate    = activations + locks + reduction + 2Q

Triton activations = 2 * R * (3I + H)
Triton estimate    = activations + 2Q
```

`Q` includes sorted IDs, expert IDs, padded-count scalar and cumsum.
The second routing allowance conservatively covers old/new tuple
replacement. It also covers filtered activation's possible INT64-to-INT32
ID conversion (`4R <= Q`); those lifetimes do not overlap. Full-cap routing
capacity bounds tail allocations with the same B. Empty-batch estimates
are zero.

The implementation uses exact tensor-byte arithmetic, not allocator block
sizes. The Marlin reduction term is a maximum across the serial GEMMs,
not their sum. Act-order and eager modified activations are excluded by
capability checks because their temporaries are not modeled here.

## Initial Local Measurements

This section records `9af19d5b5d`, which held the full-batch Triton config.
Current candidate-config results are in
[MOE_WORKSPACE_CANDIDATE_CONFIG.md](MOE_WORKSPACE_CANDIDATE_CONFIG.md).

These are **synthetic storage checks and routing simulations**, not model
benchmarks. Inputs: `M=8192, T=6, H=4096, I=512, E=256`, BF16 tensor
storage, assumed `S=78`, supplied Triton `B=64`, seed 42. No checkpoint,
KV dtype, TP/PP/DP runtime, mem-fraction, speculative decoding, request
concurrency or CUDA Graph is involved.

Each row is an independent backend/mode storage model. Do not interpret
different weight formats or backends as an accuracy/throughput A/B.

| Backend Model | Budget MiB | Cap | Chunks | Full-Batch Model MiB | Planned Model MiB |
|---|---:|---:|---:|---:|---:|
| Marlin non-atomic FP32 reduction | 16 | 128 | 64 | 452.009 | 11.650 |
| Marlin non-atomic FP32 reduction | 64 | 1024 | 8 | 452.009 | 63.864 |
| Marlin non-atomic FP32 reduction | 256 | 4096 | 2 | 452.009 | 235.819 |
| Marlin atomic | 16 | 256 | 32 | 432.509 | 13.532 |
| Marlin atomic | 64 | 1024 | 8 | 432.509 | 54.114 |
| Marlin atomic | 256 | 4096 | 2 | 432.509 | 216.319 |
| Triton unquantized, B=64 | 16 | 128 | 64 | 528.508 | 8.383 |
| Triton unquantized, B=64 | 64 | 512 | 16 | 528.508 | 33.151 |
| Triton unquantized, B=64 | 256 | 2048 | 4 | 528.508 | 132.223 |

`storage_check` materializes all formula components as CPU tensors up to
128 MiB and meta tensors above that, and compares storage `nbytes` exactly.
This checks formula consistency, not the runtime allocator trace. Empty
CPU pages need not be resident. Meta tensors allocate no data.

For uniform distinct top-k routing and a 64 MiB budget:

| Backend Model | Full / Chunk Block-M | Full / Chunk Padded Routes | Full / Chunk Expert Visits |
|---|---|---|---|
| Marlin, both reduction modes | 64 / 32 | 57,024 / 68,448 | 256 / 2,048 |
| Triton unquantized | 64 / 64 | 57,024 / 262,144 | 256 / 4,096 |

Triton's fixed configuration gives a particularly large padding increase
at this cap. Repeated alignment, GEMM launches and expert weight reads can
dominate any locality benefit. Bucket selection can leave substantial
unused budget. Neither these counts nor reduced storage predict throughput.
Per-candidate config selection is now implemented; device testing and
per-chunk GPU tuning remain outstanding.

## Tests and Reproduction

### Scope

- `test_workspace_policy.py`: direct imports of the pure planner/estimators.
- `test_workspace_adapters.py`: real Python adapter, runner and executor
  control flow, CPU operator substitutes and FakeTensor capability metadata.
- `test_marlin_chunked_workspace.py`: previous 150-case CPU regression.

AST extraction avoids CUDA execution/registration but retains the tested
function bodies. Real package imports were also run separately with
`PYTHONPATH=<this checkout>/python`; `sglang.__file__` resolved to this
checkout. Torch reported CUDA unavailable.

### Defect Analysis

No unresolved functional defect was established in this local scope.
Implementation review corrected the top-k 1 scratch destination, config
dictionary mutation on fallback, and empty Marlin lock allocation. This
does not establish absence of device-level defects.

### Generated Cases and Results

The counts in this subsection are the initial implementation's validation.
The candidate-config report lists its additional cases and final regression.

Planner: **26 passed**. Adapters: **110 passed**. Previous Marlin:
**150 passed**. Existing runner-extension/config tests: **11 passed**.
Total: **297 passed**, including 136 new cases and 161 previous cases.
New cases cover deterministic/nonmonotone buckets, budget
failure, precedence, disabled/fallback paths, shapes and storage reuse,
FP16/BF16, SiLU/GELU, tails, masks, bias, zero/scaled routing, top-k 1/2/3,
inplace/outplace, isolated calls, invalid scratch/output and empty batches.

The adapter test extension initially had three fixture errors on two runs:
this macOS Torch build attempted CUDA access for FakeTensor view operations.
Using directly constructed fake shape/stride metadata fixed the fixture
without changing production code or weakening assertions. All adapter
tests were rerun afterward. Four empty-Marlin cases passed in a separate
post-review cycle. Test workflow `utree flush` was executed for both cycles.
Coverage was not requested or measured.

Scope-specific pre-commit and `git diff --check` passed. The first hook run
formatted four files and exposed system Python 3.9 parsing errors in two
repository checkers. Re-running with the existing Python 3.13 environment
on PATH passed, without changing packages or skipping hooks. The registered
test check used the actual branch base
`GITHUB_BASE_REF=feat/dsv4-direct-int4-g32-e4m3`.

Run from the repository root with a compatible Python environment:

```bash
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  test/registered/unit/layers/moe/test_workspace_policy.py \
  test/registered/unit/layers/moe/test_workspace_adapters.py \
  test/registered/unit/layers/moe/test_marlin_chunked_workspace.py \
  test/registered/unit/layers/moe/test_moe_runner_extensions.py \
  test/registered/unit/layers/moe/test_fused_moe_triton_config.py

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_moe_workspace.py \
  --output ../validation/moe-workspace-budget-20260911/budget-plans.json

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_moe_workspace.py \
  --tokens 8193 --intermediate 2048 --budgets-mib 1 64 256 \
  --output ../validation/moe-workspace-budget-20260911/budget-wide-tail.json
```

The JSON records configuration, source hashes, Git state, storage
components and routing counts. JUnit results live beside it. The second
command includes non-power-of-two tail planning and a 1 MiB budget failure
for non-atomic Marlin: its one-token estimate is 1,650,344 bytes.

## Remaining Validation

Before deployment or enabling by default:

1. Run fresh per-backend GPU output comparisons at identical configuration,
   covering routing changes, tails, bias, scaling and supported quantization.
2. Check allocated/reserved peak memory against the modeled components.
   Separate graph-pool retention and full output from call-private scratch.
3. Validate Marlin locks and both backends with sanitizer, repeated graph
   capture/replay and concurrent independent calls.
4. Measure prefill latency and decode regression separately at multiple
   budgets. Profile routing, padding, HBM traffic and launch overhead.
5. Run end-to-end model accuracy and identical-configuration fresh A/B.

No remote or GPU validation was performed for this change.
