# Marlin MoE: Bounded Route-Expanded Scratch

## Decision

Retain an **opt-in local prototype** for token-chunked Marlin MoE. The
demonstrated benefit is smaller route-expanded activation storage, not
improved GPU throughput. Local simulation also finds substantial costs:
additional expert visits, padded rows, zeroing and launches. It is a
memory-pressure option for large prefill batches, not a default decode
optimization or a deployment-ready performance claim.

| Item | Value |
|---|---|
| Base branch | `feat/dsv4-direct-int4-g32-e4m3` |
| Base commit | `fb45f5ffbc` |
| Prototype branch | `feat/marlin-moe-bounded-workspace` |
| New control | `SGLANG_MARLIN_MOE_CHUNK_SIZE`, default `0` |
| Production scope | Marlin MoE Python orchestration and one environment field |
| Local environment | macOS arm64, Python 3.13.14, Torch 2.14.0 |
| Remote activity | No SSH, server/container access, package changes or GPU tests |
| Kernel changes | None; existing alignment, activation, GEMM and reduction kernels |

The user's upstream already includes the third INT4 iteration. This branch
starts there and does not rewrite the original future branches. Prior H20
throughput/accuracy measurements are not evidence for this prototype.

## Candidate Comparison

The selection is based on this checkout's code, not a full-model profile.
None of these rows claims to rank measured wall-time bottlenecks.

| Direction | Concrete Source Observation | Potential Benefit | Cost or Missing Evidence | Decision |
|---|---|---|---|---|
| INT4 KV norm/RoPE/pack fusion | `deepseek_v4_memory_pool.py:set_swa_key_buffer_radix_fused_norm_rope` runs norm/RoPE then the packed writer; default may first materialize a contiguous KV view | Remove a materialization and launch; roughly 2048 temporary bytes/token for one BF16-512 write/read boundary | Must preserve staged BF16 rounding, scale rounding and invalid-location behavior; value is small per decode token and needs GPU timing | Defer; ties the new PR to the previous INT4 work |
| DP communication/computation overlap | DSV4 already has gather/reduce-scatter and two-batch overlap paths; `single_batch_overlap.py` exposes down-GEMM/combine overlap for selected backends | Approach `max(compute, communication)` instead of their serial sum in favorable cases | SM/HBM contention, stream ordering, graph lifecycle and topology cannot be established from CPU timing | Defer; not a reliable local-only first implementation |
| Marlin activation fusion | Clamped SwiGLU and GPT-OSS activation in `fused_marlin_moe.py` still contain eager elementwise sequences | Fewer launches and intermediate writes | Related fused activation kernels already exist elsewhere; preserving every rounding boundary and extending Marlin epilogues needs device checks | Candidate for a separate measured PR |
| Marlin GEMM2/top-k reduction fusion | GEMM2 writes `[M * topk, H]`, then sums routes | Remove a route-major output write/read pair | Tokens span experts/CTAs; atomic or staged reduction changes determinism, synchronization and split-K handling | Defer; larger mathematical and concurrency change |
| Marlin bounded route scratch | `fused_marlin_moe.py` allocates an activation pair proportional to the full routed batch before both GEMMs | Bound the largest route-expanded tensor pair by chunk tokens, independently of full prefill length | Repeated routing, weights, padding and GEMMs can hurt latency | Implement as a default-off memory prototype |

For the communication row, even the ideal overlap bound is not a speedup
prediction: communication and compute can compete for the same resources.
For the KV and reduction rows, logical bytes removed do not automatically
equal HBM bytes saved.

## Implementation

For each token chunk, run the existing sequence:

```text
local top-k slice -> expert alignment -> gate/up GEMM
-> activation -> down GEMM -> top-k sum into output[start:end]
```

One call-private `intermediate_cache13` aliases gate/up and down storage,
as before. One call-private `intermediate_cache2` holds gated activations.
Both are reused across chunks. The existing per-call Marlin lock workspace
is reused serially; it is never cached across separate calls or streams.
The runtime still accepts a caller-supplied workspace as before.

Important boundaries:

- `0` retains one full-batch iteration. A cap at least as large as the
  current batch also uses one iteration. Negative caps raise `ValueError`.
- Chunking is based on the local tensor shape, not a device-to-host routing
  count. No new host synchronization or collective is introduced.
- Routing IDs are sliced with activations and are aligned independently;
  token IDs inside each chunk are local, matching GEMM input/output views.
- The full chunk determines the existing block-M heuristic. The tail uses
  that same block size and smaller views; a one-token tail retains the
  existing single-token alignment fast path when eligible.
- The shared gate/down allocation is cleared between chunks. Chunked
  execution also clears each down destination conservatively, preserving
  neutral values for skipped rows. Existing EP clearing remains.
- Exact input/output aliasing is permitted after both GEMMs for that
  chunk finish. A destination sharing input storage at another offset
  disables chunking, since it could overwrite future input tokens.
- New output allocation is delayed until the first reduction, avoiding an
  earlier full output allocation on the default path.
- Non-gated activations do not retain an unused gated-activation buffer.
- Weight format, expert maps, bias, activation formulas, top-k weighting,
  routed scaling, quantized GEMM arguments and communication APIs are unchanged.

Changing batch partitioning changes Marlin launch geometry and potentially
split-K accumulation. Unchanged formulas therefore do **not** imply
bitwise-equal CUDA results. The public custom-op signature is unchanged,
but graph capture/replay and torch.compile execution remain unverified.
Set the control before warmup/capture and do not change it on a live server.

## Storage Model

For gated FP16/BF16 MoE, let:

- `M`: input tokens on this rank.
- `T`: routes per token.
- `H`: local GEMM input/output hidden dimension.
- `I`: expert intermediate dimension on this rank.
- `C`: `min(M, chunk_limit)`, or `M` when disabled.

The two explicit route-expanded activation buffers occupy:

```text
S(M) = 2 * M * T * (max(2 * I, H) + I) bytes
S(C) = 2 * C * T * (max(2 * I, H) + I) bytes
```

This is a live tensor-pair calculation, not total process memory or measured
CUDA allocator usage. It excludes:

- Quantized expert weights/scales, router logits and full top-k metadata.
- Full input/output tensors and communication buffers.
- Alignment metadata and activation-order permutation buffers.
- FP32 split-K temporary storage allocated by the JIT GEMM wrapper.
- Extra eager activation intermediates, allocator fragmentation, graph pools
  and concurrent calls.

For example, the JIT wrapper's FP32 temporary is bounded separately by
`min(size_n * sorted_ids_capacity, SMs * 4 * block_m * 256) * 4` bytes,
with another factor of two for block-M 8. At an assumed 78 SMs, the second
term gives 19.5 MiB for block-M 64 and 9.75 MiB for block-M 32.
Those are allocation bounds, not observed device memory. This prototype
does not share that temporary across the two GEMMs.

### Local Storage Checks

All configurations below are synthetic gated MoE, FP16/BF16 activations,
256 experts and MXFP4-G32/E8M0 weights for the logical weight-byte model.
There is no checkpoint, KV format, speculative decoder, TP/PP/DP runtime,
serving memory fraction or request concurrency in these local calculations.
`I` is already the per-rank dimension; do not divide it by a guessed TP.

| M / T / H / I | Cap | Baseline Pair MiB | Chunk Pair MiB | Reduction | Local Storage Check |
|---|---:|---:|---:|---:|---|
| 128 / 6 / 4096 / 512 | 1024 | 6.75 | 6.75 | 0% | CPU / CPU |
| 1024 / 6 / 4096 / 512 | 512 | 54 | 27 | 50% | CPU / CPU |
| 8192 / 6 / 4096 / 512 | 1024 | 432 | 54 | 87.5% | CPU / CPU |
| 8192 / 6 / 4096 / 2048 | 1024 | 576 | 72 | 87.5% | Meta / CPU |
| 32768 / 8 / 7168 / 2048 | 2048 | 4608 | 288 | 93.75% | Meta / CPU |

CPU checks allocate `torch.empty` tensors and compare their storage `nbytes`
to the formula. Pages are not all touched, so this is not RSS measurement.
Meta checks allocate metadata only. Neither measures GPU allocation,
CUDA Graph pool retention, OOM avoidance or performance.

## Costs and Negative Signals

The script also generates distinct top-k routes using seed 42. Uniform
routing selects from all 256 experts; the hot case selects from 16.
Block-M follows the actual heuristic for the full batch and each cap.

| M / T / Cap | Distribution | Baseline Block-M | Chunk Block-M | Baseline Padded Routes | Chunk Padded Routes | Expert-Chunk Visit Ratio |
|---|---|---:|---:|---:|---:|---:|
| 8192 / 6 / 512 | Uniform | 64 | 16 | 57,024 | 71,824 | 16x |
| 8192 / 6 / 1024 | Uniform | 64 | 32 | 57,024 | 68,448 | 8x |
| 8192 / 6 / 2048 | Uniform | 64 | 64 | 57,024 | 66,176 | 4x |
| 8192 / 6 / 1024 | Hot | 64 | 32 | 49,664 | 51,136 | 8x |
| 32768 / 8 / 2048 | Uniform | 64 | 64 | 270,080 | 383,808 | 16x |

At `M=8192, T=6, H=4096, I=512, cap=1024`, uniform padded work increases
20.03%. Eight chunks replace one. A five-core-launch model (align, two GEMMs,
activation, reduction) adds 35 launches, excluding zeroing and alignment
fast-path changes. An assumed 3 us/launch gives 105 us of sensitivity, not
measured CUDA launch latency.

Conservative down-buffer clearing adds 384 MiB of logical writes in that
non-EP example. At an assumed effective bandwidth of 3 TB/s, its
bandwidth-only floor is 134.2 us. Do not add these optimistic models to
infer an end-to-end speedup or slowdown.

Each expert contains `3 * H * I * 17/32` logical bytes under the MXFP4
weight/scales model. If each chunk visit rereads its full expert weights,
the uniform example adds 5.578 GiB of logical weight visits, versus
0.349 GiB for hot routing. These are not HBM counters: cache retention,
tile scheduling and repeated weight reads within a chunk are unmodeled.
The smaller activation pair alone does not prove it fits in L2 with all
other working data.

These negative signals are why the prototype stays off by default. A
larger cap or no chunking is preferable when memory is sufficient. No
claim is made that freeing scratch automatically increases token-pool
capacity; cache budgeting is a separate mechanism.

## Local Validation

### Scope

`test/registered/unit/layers/moe/test_marlin_chunked_workspace.py` executes
the production Python function bodies extracted with AST, avoiding only
CUDA/Triton imports and custom-op registration. CUDA GEMM, routing,
activation and reduction dependencies are replaced by deterministic CPU
implementations. The independent reference evaluates each routed expert
directly.

This checks Python control/data flow, shapes, argument forwarding, allocation
identity and numerical assembly under the CPU surrogate. It does not test
Marlin quantized arithmetic, GPU reduction order, kernel validation, PDL,
stream concurrency or inter-block lock reset.

### Defect Analysis

No baseline functional defect is claimed. Large scratch allocation is an
optimization opportunity, not proof of an OOM. Existing EP output clearing
already addresses skipped experts. Offset output aliases are a new chunking
risk, handled by the fallback instead of changing the zero-copy API.

### Generated Cases

150 parameterized CPU cases cover FP16/BF16, scalar-type selection, seven
activation configurations, biases, route weighting, zero routed scaling,
disabled/capped execution, empty/single-token batches, one/two routes,
tail chunks, masked experts, zero-copy/inplace/offset-alias outputs,
workspace and activation-order argument forwarding, repeated calls and
invalid configuration.

The gated path asserts a single scratch storage and activation storage
across chunks. Tests assert zero-tolerance equality to the CPU reference;
that is not a zero-tolerance CUDA guarantee.

### Results

| Check | Result |
|---|---|
| Initial CPU generation/verification loop | 137 passed |
| Added boundary cases, second loop | 150 passed |
| Final implementation regression | 150 passed |
| Scope-specific pre-commit | Pass |
| `git diff --check` | Pass |
| Test workflow `utree flush` | Completed |
| Coverage | Not requested or measured |
| CUDA build, numerical tests, sanitizer, graph replay | Not run |
| GPU memory, throughput, TTFT/TPOT, model accuracy | Not run |

The registered-test hook initially compared against `origin/main`, pulling
in unrelated historical taxonomy violations. Running it with this PR's
actual base, `GITHUB_BASE_REF=feat/dsv4-direct-int4-g32-e4m3`, passes.
No hook was skipped and no historical test was moved or weakened.

## Reproduction

From the repository root, using a compatible local Python environment:

```bash
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  test/registered/unit/layers/moe/test_marlin_chunked_workspace.py

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_marlin_chunking.py

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_marlin_chunking.py \
  --intermediate 2048 --batches 8192 --caps 1024 \
  --cpu-allocation-limit-mib 128

PYTHONPATH="$PWD/python" python \
  benchmark/kernels/fused_moe_triton/analyze_marlin_chunking.py \
  --hidden 7168 --intermediate 2048 --topk 8 --experts 256 \
  --batches 32768 --caps 2048 --cpu-allocation-limit-mib 384
```

Local artifacts are under
`../validation/marlin-moe-bounded-workspace-20260910/`: `cpu.xml`,
`offline-analysis.json`, `offline-wide.json`, and `offline-large.json`.
Analysis JSON records dimensions, assumptions, source hash, Git state,
Torch/platform and local analysis duration. That duration measures the
analysis script, not inference.

## Future Acceptance

Before promoting this beyond a draft PR:

1. Verify quantized outputs for MXFP4, GPTQ/AWQ and NVFP4 as applicable,
   both activation dtypes, EP masks, tails and activation-order inputs.
2. Run memcheck/racecheck/synccheck and repeated graph replay with changing
   routing. Validate lock reset when reusing the workspace across chunks.
3. Measure peak allocated/reserved memory, graph pools, HBM traffic and
   launch counts at fixed full batch and several caps.
4. Measure prefill latency and decode regression separately, with fresh
   identical model, dtype, topology and scheduler settings.
5. Retain the feature only where memory improvement justifies the latency
   cost. Do not enable by default on the basis of the local storage model.

No remote validation is performed as part of this request.
