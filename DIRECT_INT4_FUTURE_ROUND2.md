# Direct INT4 Future: Token-Interleaved Producer

## Status

The producer change has passed numerical and sanitizer validation. In the
fresh, same-configuration model comparison, output throughput improves 6.35%
and mean TPOT decreases 6.06%, with unchanged token-pool capacity. Twelve
synthetic attention cases are bitwise identical to the previous implementation
and take 8.6%-17.3% less time. Legacy GSM8K scores are 1258/1319 before and
1260/1319 after, which does not establish an accuracy improvement or
equivalence. The candidate is retained.

| Item | Value |
|---|---|
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| GPU | H20 SM90, 78 SMs per GPU; eight GPUs available |
| Container | `kvbit-future-ab-20260908` |
| Repository | `/sgl-workspace/dsv4-future-ab` |
| Branch | `feat/dsv4-direct-int4-g32-e4m3-future` |
| Baseline source | `e08050d32d`, same implementation as `276e514d01` |
| Candidate source | `4ed0960eb2` |
| Model | Host `/mnt/nvme2/DeepSeek-V4-Flash`, container `/models/DeepSeek-V4-Flash` |
| Runtime | Python 3.12.3, Torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0.88 |

The previous native-FP8 comparison is recorded in
[Round 1](DIRECT_INT4_FUTURE_REMOTE_VALIDATION.md). This round compares the
previous and candidate **INT4 implementations**, not INT4 against a fresh FP8
baseline.

## Diagnosis

Nsight Compute 2025.3.1 collected the main sparse attention kernel with
`--set full`, one selected NVTX-range launch, 39 replay passes, and
`--clock-control none`. Both profiles use the same GPU and synthetic input:

- Batch 128, one query position, 64 physical heads, QK/V dimension 512.
- SWA top-k 128, C4 top-k 512, FP32 sink logits, BF16 query/output.
- Direct INT4 G32 E4M3, scalar dequantization, 368-byte rows.
- Two pools of 262,144 rows, seed 42, uniformly random sparse indices.
- Attention-only, no model weights, TP, speculative decoding or serving
  memory fraction. The profile is an eager launch; the timing benchmark below
  uses CUDA Graph replay.

| Main-Kernel Counter | Before | After |
|---|---:|---:|
| Profiled duration, us | 327.616 | 282.560 |
| Shared-store bank conflicts | 4,605,782 | 13,819 |
| Shared-store wavefronts | 5,865,466 | 871,366 |
| Excessive shared wavefronts, source counters | 4,592,640 | 5,120 |
| Executed warp instructions | 16,870,516 | 13,528,784 |
| Global-load requests | 742,992 | 384,592 |
| Global-load sectors | 2,382,406 | 3,949,820 |
| L1/TEX hit rate | 69.29% | 78.60% |
| DRAM throughput, percent of peak | 3.07% | 3.54% |
| Compute throughput, percent of peak | 22.20% | 26.26% |
| Registers per thread | 168 | 168 |
| Local spilling requests | 1,448 | 1,448 |

The original layout sends a warp's feature fragments to the same shared-memory
banks. Splitting stores into two lane phases does not remove the underlying
bank conflict. The new mapping nearly removes that overhead, while reducing
the executed instruction count.

There is a tradeoff: sparse global loads visit more sectors. Most of the
additional traffic is served by L1; measured DRAM read traffic is nearly
unchanged. This is not an HBM-bandwidth optimization. The unchanged 208,896-byte
shared-memory plan still permits only one CTA per SM.

NCU durations include profiling/replay effects and are not the benchmark
durations. Frequencies were not locked, and six CTC transfer metrics were
unavailable. Initial counter access failed with `ERR_NVGPUCTRPERM`; only the
profiling exec was retried with Docker's `--privileged` flag. No global driver,
clock, or counter-permission setting was changed.

## Implementation

Only `sm90/decode/sparse_fp8/packed_int4.cuh` changes:

```text
token_in_warp = lane & 7
word_in_group = lane >> 3
token = warp * 16 + round * 8 + token_in_warp
feature = group * 32 + word_in_group * 8
```

Each warp processes eight tokens at once in two rounds. The lowest eight
lanes read token indices and group scales; shuffles share them with the
other feature lanes for the same token. Every lane writes a 16-byte fragment.
Eight adjacent lanes cover every shared-memory bank, producing the minimum
four wavefronts for a 512-byte full-warp store.

An index enumeration checks that all 64 x 512 elements are written exactly
once. Negative indices, out-of-range indices and short top-k lengths still
produce zero payload with invalid masks. RoPE remains copied as BF16.

The following do not change:

- Packed format, codec, scale rounding, layout offsets or pool capacity.
- Default scalar dequantization and final BF16 rounding.
- Query/K/V geometry, QK/PV accumulation order and split scheduling.
- Shared-memory layout, two K buffers and register budgets.
- Producer publication, ready/free barriers and consumer synchronization.
- Writer, native FlashMLA, Python serving API and extension ABI 2.

## Kernel Measurements

The microbenchmark excludes pool construction and packing. It initializes
scheduler metadata before timing, captures five attention calls per graph,
then measures seven samples with approximately 100 ms per sample. Values
are sample medians. All cases use the two 262,144-row pools, seed 42, one query
position, 64 physical heads, SWA top-k 128 and 368-byte INT4 rows.

`extra=0` means SWA only; `extra=32` uses a C128-shaped page size of two;
`extra=512` uses C4 page size 64. Inputs are synthetic, not captured model KV.

| Batch | Extra Top-k | Before us | After us | Latency Change |
|---|---:|---:|---:|---:|
| 1 | 0 | 19.760 | 18.052 | -8.64% |
| 1 | 32 | 20.061 | 18.314 | -8.71% |
| 1 | 512 | 21.996 | 20.089 | -8.67% |
| 32 | 0 | 26.536 | 22.735 | -14.33% |
| 32 | 32 | 35.858 | 29.715 | -17.13% |
| 32 | 512 | 79.619 | 65.854 | -17.29% |
| 128 | 0 | 54.333 | 47.239 | -13.06% |
| 128 | 32 | 77.403 | 65.842 | -14.94% |
| 128 | 512 | 306.740 | 261.036 | -14.90% |
| 1024 | 0 | 553.010 | 490.626 | -11.28% |
| 1024 | 32 | 776.511 | 675.238 | -13.04% |
| 1024 | 512 | 2159.852 | 1866.690 | -13.57% |

All twelve output tensors and all twelve LSE tensors match the baseline
bitwise. Repeating the old binary after the candidate run changes its
medians by -0.284% to +0.041%, smaller than the candidate improvement.

The previous BF16x2 binary was also measured on the same twelve cases. Its
outputs/LSE are bitwise identical, but latency is higher than the old scalar
binary in every case. This is a binary-level diagnostic: the old scalar
includes `-lineinfo` and the old vector build does not, so it does not isolate
the dequantization switch alone. BF16x2 remains opt-in. The new mapping is
tested with both dequantization implementations, but only scalar is selected
for model A/B.

## Validation

| Check | Scalar | BF16x2 |
|---|---|---|
| Focused SM90a CMake build | Pass | Pass |
| GPU numerical/writer/graph suite | 4 tests, 30 subtests pass | 4 tests, 30 subtests pass under each sanitizer |
| memcheck, API reports disabled | 0 errors | 0 errors |
| racecheck | 0 hazards, 0 warnings | 0 hazards, 0 warnings |
| synccheck | 0 errors | 0 errors |
| Exact old/new comparison, 12 synthetic shapes | Pass | Not measured for the new mapping |

The existing regression suite includes 368/384-byte rows, C4/C128 page
geometry, invalid indices, empty attention, multiple query positions,
split/no-split paths and CUDA Graph replay after mutation. Assertions are
unchanged.

CPU regression: 77 tests and 58 subtests pass. Pre-commit passes. No test
coverage percentage is claimed.

Both new binaries use `-lineinfo` and report 168 registers, 13 barriers,
32 stack bytes, four spill-store bytes and four spill-load bytes. The scalar
baseline has the same settings and resource counts.

| Artifact | SHA256 |
|---|---|
| Baseline scalar | `d76709f0a0cb0efd27c488c4e62c7b3e9d56ea29d41ffeb51cdab27629525335` |
| Candidate scalar | `188a6e1128e787b009571a27d18b086e92b170c50cefeabcdecd6d63e4d4b0dd` |
| Candidate BF16x2 | `38b60dad7ff540722ebd59efa6f26c45e59b940afb64e66618a4a29244a35a0e` |

As in round 1, memcheck disables CUDA API error reports because an isolated
`import sgl_kernel` reproduces dependency-initialization API errors without
executing this extension. Device-memory and Hopper checks remain enabled.
This does not claim an unfiltered memcheck pass.

The original `kvbit-ds` package installation is untouched. Candidate
extensions are built from the Git-synchronized source into independent CMake
prefixes. Only the scalar component in the isolated A/B container's
`site-packages` is updated for serving; no dependency packages change.

## Model A/B

Both phases use the same read-only checkpoint and service command:

| Setting | Value |
|---|---|
| Model | DeepSeek-V4-Flash |
| TP / PP / DP | 8 / 1 / 1 |
| Memory fraction | 0.8 |
| Target KV | Direct INT4 G32 E4M3, `aos_368`, scalar |
| Draft KV | FP8 E4M3 |
| Speculative | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| CUDA Graph | Target verify, draft decode and draft extend on; maximum batch 32; ordinary prefill graph off |
| Performance input / output | Fixed 4096 / 256 tokens |
| Requests / concurrency / repetitions | 128 / 32 / 3 |
| Warmup | 32 requests, then flush prefix cache before measurement |
| Random input protocol | `random-ids`, `--tokenize-prompt`, `--random-range-ratio 1` |
| Seed / temperature / request rate | 42 / 0 / infinite |
| MoE / prefill chunk | Marlin / 8192 tokens |
| GSM8K | Legacy 8-shot, 1319 questions, parallel 32, temperature 0, output limit 1024 |

Environment is identical: explicit
`PYTHONPATH=/sgl-workspace/dsv4-future-ab/python`,
`SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV4_FP4_EXPERTS=1`,
`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`,
`SGLANG_DSV4_INT4_LAYOUT=aos_368`,
`SGLANG_DSV4_INT4_STRIDED_NORM_ROPE=0`,
`GLOO_SOCKET_IFNAME=eth0`, `NCCL_MIN_NCHANNELS=24`,
`NCCL_IB_QPS_PER_CONNECTION=8`.

The comparator checks manifests, launch commands, resolved server settings,
benchmark commands, exact per-request lengths, all request errors, and
per-question prompt-token counts. The intentionally different fields are
source commit and compiled extension hash. Measurements are serial; no other
GPU workload runs during either performance phase.

### Performance

Three-run medians are shown below. Latency statistics are medians of the
three per-run means. All six runs completed without request errors and each
contains exactly 524,288 input and 32,768 output tokens.

| Metric | Before INT4 | After INT4 | Change |
|---|---:|---:|---:|
| Total output throughput, tokens/s | 489.677 | 520.758 | +6.35% |
| Output throughput / 8 GPUs, tokens/s/GPU | 61.210 | 65.095 | +6.35% |
| Mean TPOT, ms | 54.666 | 51.356 | -6.06% |
| Mean TTFT, ms | 2502.728 | 2353.428 | -5.97% |
| Mean end-to-end latency, ms | 16452.068 | 15457.500 | -6.05% |
| Speculative accept length | 2.6955 | 2.6840 | Diagnostic |
| Full token-pool capacity | 7,959,296 | 7,959,296 | Unchanged |

| Repeat | Before Output Tokens/s | After Output Tokens/s |
|---|---:|---:|
| 1 | 486.358 | 520.608 |
| 2 | 489.677 | 520.967 |
| 3 | 490.247 | 520.758 |

This is one TP8 configuration, not a search for optimal total or per-GPU
throughput. The service logs confirm target-verify, draft-decode and
draft-extend CUDA Graph capture in both phases. All eight candidate scheduler
processes map the rebuilt extension and run from the intended checkout.

### Accuracy

Both phases use the same 8-shot prompts, temperature zero, concurrency 32 and
serving configuration above. Input lengths are 1261-1424 tokens, identical
for corresponding questions, and output is limited to 1024 tokens.

| Metric | Before INT4 | After INT4 |
|---|---:|---:|
| Full legacy GSM8K | 1258/1319, 95.3753% | 1260/1319, 95.5269% |
| Excluding eight test-file exemplars | 1250/1311, 95.3471% | 1252/1311, 95.4996% |
| Generated tokens | 126,042 | 125,044 |
| Invalid parsed answers | 0 | 0 |
| Responses reaching the output limit | 0 | 0 |

The before version alone answers 10 questions correctly; the after version
alone answers 12. Numeric predictions match on 1292/1319 questions and full
text matches on 426/1319. The full-score difference is +0.1516 percentage
points, but this is one evaluation per revision with batch-dependent
generation. It is not evidence of improved precision or statistical
equivalence.

The legacy evaluator uses the first eight test-file rows as its exemplars.
The full score is not an independent train-exemplar GSM8K result; the
exemplar-excluded score is reported separately. The dataset SHA256 remains
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

## Decision and Limits

- Retain `4ed0960eb2`: the unchanged-format producer improves the fresh
  fixed-workload model measurement without reducing capacity, and all
  exercised numerical and sanitizer cases pass.
- NCU identifies a measured source of overhead: feature-major producer
  stores cause shared bank conflicts. The change reduces those conflicts
  99.70% and also changes instruction count and global access patterns.
  The end-to-end gain cannot be attributed to bank conflicts alone.
- Both model phases use INT4. Native FP8 was not rerun in this round; no
  new FP8-relative speedup is claimed.
- The model order is before then after, three performance repetitions per
  phase, not randomized or interleaved. A separate old-binary microbenchmark
  repeat checks kernel timing drift but is not an interleaved model control.
- No full model-stage trace or individual producer/consumer cycle breakdown
  was collected. Shared-memory footprint, global-load latency and remaining
  register spills are not eliminated.
- Synthetic bitwise equality covers the listed cases, not all possible
  model inputs. Long-context, other concurrency/TP settings, 384-byte model
  serving and end-to-end BF16x2 remain outside this round.
- No new dependencies, model conversion, native scratch cache, relaxed
  accuracy tolerances or weaker synchronization were introduced.

## Evidence

- Host: `/mnt/nvme2/kvbit-future-20260908/round2`.
- Container: `/artifacts/round2`.
- Local scripts: `../validation/dsv4-future-round2/microbench.py` and
  `compare_round.py`.
- Unchanged phase runner: `../validation/dsv4-future-20260908/run_phase.py`.
- Local raw archive: `../validation/dsv4-future-round2/round2-evidence.tar.gz`;
  it contains manifests, benchmark/evaluation outputs, NCU reports and
  build/sanitizer logs. Large synthetic `.pt` outputs and installed binaries
  remain in the host artifact directory rather than in this local archive.
- Microbench directories: `baseline-micro`, `candidate-micro`,
  `baseline-repeat-micro`, `vector-micro`.
- NCU reports: `baseline-c4-ncu-admin.ncu-rep`, `candidate-c4-ncu.ncu-rep`.
- A/B directories: `int4-before`, `int4-after`.
- Builds: `/tmp/kvbit-future-round2-scalar-build`,
  `/tmp/kvbit-future-round2-vector-build`.
- Candidate installation prefixes: `/artifacts/round2/aot-scalar`,
  `/artifacts/round2/aot-vector`.
- Old installations are retained under `/artifacts/aot-scalar` and
  `/artifacts/aot-vector`.

Both model process groups have exited after phase completion. The isolated
container remains idle; all eight GPUs report zero allocated memory and no
compute processes. Other users' services and the original `kvbit-ds` Python
package installation were not modified. Both repositories are synchronized
through Git; reports do not replace or overwrite historical raw results.
