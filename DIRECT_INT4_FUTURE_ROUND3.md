# Direct INT4 Round 3: Whole-Group Loads

## Status

The round is complete; the scalar candidate is retained on the experiment
branch. Fresh model output throughput improves 9.11% and mean TPOT decreases
8.60%, with unchanged token-pool capacity. All exercised kernel numerical
and sanitizer checks pass. Legacy GSM8K scores are 1255/1319 before and
1253/1319 after. This single evaluation does not establish accuracy
equivalence or identify the cause of the two-question difference.

| Item | Value |
|---|---|
| Upstream baseline | `8435df17606ba8c0bcea28e1930e0c2526b0f67c` |
| Upstream branch | `feat/dsv4-direct-int4-g32-e4m3` |
| Experiment branch | `feat/dsv4-direct-int4-g32-e4m3-future-r3` |
| Candidate | `5cb3842499817358d32424b354e4b5a5179da9d4` |
| Host | `115.190.141.215`, `iv-yehwog4ni84c5qw9eqe0` |
| GPU | Eight H20 SM90 GPUs, 78 SMs each |
| A/B container | `kvbit-future-ab-20260908` |
| A/B checkout | `/sgl-workspace/dsv4-future-ab` |
| Model | `/models/DeepSeek-V4-Flash`, read-only bind mount from `/mnt/nvme2/DeepSeek-V4-Flash` |

The user cherry-picked rounds 1 and 2 onto upstream. Both remote repositories
were already on that updated branch when this round started. This experiment
therefore starts directly from `8435df1760`, not from the older `future`
branch. The upstream CP interface updates are retained. Neither upstream
history nor the original future branch is rewritten.

The baseline kernel source matches the round-2 implementation and loads the
same scalar binary SHA256:
`188a6e1128e787b009571a27d18b086e92b170c50cefeabcdecd6d63e4d4b0dd`.
All model comparisons in this round use fresh runs on the updated upstream
framework. Historical model results are not used as the baseline.

## Hypothesis

Round 2 reduced producer shared-memory bank conflicts, but each feature lane
still performs one 32-bit code load and a scale shuffle for each of 14 G32
groups. Fresh NCU measurement at batch 128, SWA top-k 128 plus C4 top-k 512
reports about 49.5% of average warp cycles between issued instructions as
L1TEX long-scoreboard dependencies. This includes global/local memory
dependencies; it is not an isolated producer timing.

The candidate changes only
`python/sglang/kernels/aot/csrc/kvbit/flashmla/sm90/decode/sparse_fp8/packed_int4.cuh`.
Each feature lane owns an entire G32 group:

```text
token = warp * 16 + round * 8 + (lane & 7)
group = group_base + (lane >> 3)
codes = aligned uint4 load at row + group * 16
scale = stored E4M3 scale[group]
```

The four 32-bit words in the loaded group reuse the same scale in registers.
Four groups are processed per iteration, with the last two unused groups
masked. This preserves the eight-token lane interleave for shared stores
while reducing repeated loads and removing group-scale shuffles.

Safety and numerical invariants:

- Every 368/384-byte row starts at a 16-byte-aligned address. Each 16-byte
  group load is aligned and stays inside code bytes `[0, 224)`.
- Group 13 ends at byte 223. Groups 14/15 do not load or write.
- Negative, out-of-pool and top-k-masked tokens perform no source load and
  write zero payload with an invalid mask.
- An index enumeration covers every element of the 64 x 512 tile exactly
  once, including the unchanged BF16 RoPE path.
- Scalar and optional BF16x2 dequantization formulas are unchanged.
- Quantization, layouts, pool budgeting, QK/PV accumulation, scheduler,
  sink/LSE handling, barriers and two K buffers are unchanged.

## Protocol

### Kernel Measurements

The unchanged round-2 `microbench.py` is used with a new output directory.
It records actual source/import paths, extension hashes and numerical
outputs. Setup, packing and initial metadata generation are excluded.

| Setting | Value |
|---|---|
| Data | Synthetic BF16 Q and stored INT4 KV, seed 42 |
| Batch sizes | 1, 32, 128, 1024 |
| Query positions / physical heads / head dimension | 1 / 64 / 512 |
| SWA top-k / extra top-k | 128 / 0, 32, 512 |
| Main / extra pool size | 262,144 rows each |
| Row stride | 368 bytes |
| Timing | CUDA Graph, five calls per graph, seven samples per shape |
| Reference | Baseline output and LSE saved and compared with zero tolerance |
| Domain | Single-GPU attention only; no model, TP or speculative decoding |

NCU uses the batch-128/C4-512 case, an eager NVTX-selected launch, `--set full`,
39 replay passes and `--clock-control none`. Only profiler processes use
privileged Docker exec for counter access. No driver permissions or clock
settings are changed. NCU timing is diagnostic, not CUDA Graph benchmark
timing.

### Model Measurements

Both phases use the same checkpoint, runner, model configuration hash, Python
packages, serving environment and command. Only source commit and extension
binary are intended to differ.

| Setting | Value |
|---|---|
| Model | DeepSeek-V4-Flash |
| TP / PP / DP | 8 / 1 / 1 |
| Memory fraction | 0.8 |
| Target KV | Direct INT4 G32 E4M3, `aos_368`, scalar |
| Draft KV | FP8 E4M3 |
| Speculative | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| CUDA Graph | Target verify, draft decode and draft extend on; maximum batch 32; ordinary prefill graph off |
| Input / output | Fixed 4096 / 256 tokens |
| Requests / concurrency / repetitions | 128 / 32 / 3 |
| Warmup | 32 requests, then prefix-cache flush |
| Dataset CLI | `random-ids`, `--random-range-ratio 1`, `--tokenize-prompt` |
| Server and benchmark seeds | 42 |
| Temperature / request rate | 0 / infinite |
| MoE backend / prefill chunk | Marlin / 8192 tokens |
| GSM8K | Legacy 8-shot, 1319 questions, parallel 32, temperature 0, maximum output 1024 |

Environment:
`PYTHONPATH=/sgl-workspace/dsv4-future-ab/python`,
`SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV4_FP4_EXPERTS=1`,
`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`, `SGLANG_DSV4_INT4_LAYOUT=aos_368`,
`SGLANG_DSV4_INT4_STRIDED_NORM_ROPE=0`, `GLOO_SOCKET_IFNAME=eth0`,
`NCCL_MIN_NCHANNELS=24`, `NCCL_IB_QPS_PER_CONNECTION=8`.

The legacy GSM8K evaluator uses its first eight test-file questions as
exemplars. Both the full score and exemplar-excluded score must be reported;
neither is presented as a standard independent train-exemplar evaluation.

## Kernel Results

All twelve scalar and BF16x2 output/LSE pairs match the saved baseline with
`atol=0, rtol=0`. Values below are medians of seven CUDA Graph timing samples
under the kernel protocol above, in microseconds.

| Batch | Extra Top-k | Before Scalar | After Scalar | Scalar Change | After BF16x2 |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 18.012 | 14.852 | -17.55% | 14.629 |
| 1 | 32 | 18.254 | 15.146 | -17.03% | 14.864 |
| 1 | 512 | 20.075 | 17.039 | -15.12% | 16.695 |
| 32 | 0 | 22.649 | 17.621 | -22.20% | 17.242 |
| 32 | 32 | 29.666 | 22.716 | -23.43% | 22.281 |
| 32 | 512 | 65.738 | 49.033 | -25.41% | 48.257 |
| 128 | 0 | 47.324 | 34.227 | -27.67% | 33.728 |
| 128 | 32 | 65.780 | 44.550 | -32.27% | 43.954 |
| 128 | 512 | 261.132 | 118.406 | -54.66% | 117.994 |
| 1024 | 0 | 490.721 | 256.301 | -47.77% | 250.765 |
| 1024 | 32 | 675.190 | 329.461 | -51.20% | 321.320 |
| 1024 | 512 | 1865.110 | 797.817 | -57.22% | 784.270 |

The old binary was measured again after both candidates. Its median drift
is -0.140% to +0.401%, smaller than the measured scalar improvement.
The repeat also matches the saved output/LSE with zero tolerance.

Both new variants use the same compiler configuration, including `-lineinfo`.
BF16x2 is 0.35%-2.47% faster than the new scalar in these measurements.
That small additional difference has no repeated or interleaved variant
control and is not a model-level result. BF16x2 remains opt-in; only scalar
is used for the model comparison.

### Profiled Mechanism

Fresh before/after NCU results for the same batch-128, SWA-128 plus C4-512
eager launch:

| Main-Kernel Metric | Before | After |
|---|---:|---:|
| Duration, us | 282.592 | 120.608 |
| Measured SM frequency, GHz | 1.817313 | 1.816546 |
| Global-load requests | 384,592 | 179,792 |
| Global-load sectors | 3,949,820 | 2,720,541 |
| Executed warp instructions | 13,528,460 | 13,161,670 |
| Long-scoreboard cycles per issued instruction | 14.0513 | 2.9720 |
| Barrier-wait cycles per issued instruction | 11.4687 | 6.3948 |
| Total warp cycles per issued instruction | 28.4074 | 12.4597 |
| Shared-store wavefronts | 870,904 | 985,063 |
| Shared-store bank conflicts | 13,945 | 28,082 |
| Excessive shared wavefronts, source counter | 5,120 | 5,120 |
| Local spilling requests | 1,448 | 1,448 |
| Registers per thread | 168 | 168 |
| Dynamic shared memory, bytes | 208,896 | 208,896 |
| Theoretical / achieved occupancy | 18.75% / 18.75% | 18.75% / 18.75% |
| DRAM throughput, percent of peak | 3.54% | 8.30% |
| Compute throughput, percent of peak | 26.18% | 61.36% |

Main-kernel SASS confirms the intended aligned load. The following are
static opcode counts, not executed load counts:

| Load Opcode | Before | After Scalar |
|---|---:|---:|
| `LDG.E.CONSTANT` | 41 | 13 |
| `LDG.E.128.CONSTANT` | 0 | 8 |
| `LDG.E.U8` | 28 | 8 |
| `LDG.E.128` | 2 | 2 |
| `LDG.E` | 16 | 16 |

The measured mechanism is consistent with fewer fine-grained load requests
and shorter memory-dependency waits: requests fall 53.25%, sectors fall
31.12%, and long-scoreboard cycles per issued instruction fall 78.85%.
Total executed instructions fall only 2.71%. This is not evidence that
instruction count alone accounts for the 57.32% profiled duration reduction,
nor that HBM bandwidth was saturated.

There is a cost: shared-store wavefronts increase 13.11%, and measured bank
conflicts increase despite preserving the eight-token interleave. The
source-level excessive-wavefront counter remains 5,120. These counters
describe different scopes and should not be treated as interchangeable.
No claim is made that all memory metrics improve.

Barrier waits now account for about 51.3% of average warp issue latency,
versus about 40.4% before, although absolute barrier-wait cycles decrease.
This ratio is not a wall-time fraction and does not justify removing
correctness barriers. Occupancy and the shared-memory footprint are unchanged.
No isolated producer/consumer cycle breakdown was collected.

Both NCU captures completed 39 passes. Six CTC transfer metrics were
unavailable. GPU clocks were not locked; their measured frequencies are
shown above. Profiler speedup suggestions are not used as measured results.

## Validation

CPU regression on the updated upstream framework: 79 tests and 63 subtests
pass. The candidate commit passes pre-commit checks. No test assertions or
tolerances were relaxed, and no test coverage percentage is claimed.

| Check | Scalar | BF16x2 |
|---|---|---|
| Focused SM90a CMake build | Pass | Pass |
| GPU numerical/writer/graph regression | 4 tests, 30 subtests pass | 4 tests, 30 subtests pass |
| memcheck, API reports disabled | 0 errors | 0 errors |
| racecheck | 0 hazards, 0 warnings | 0 hazards, 0 warnings |
| synccheck | 0 errors | 0 errors |
| Output and LSE against old binary, 12 cases | Zero-tolerance match | Zero-tolerance match |

Each sanitizer executes the same complete four-test suite. It includes
368/384-byte rows, page sizes 2/64/256, invalid indices and empty attention,
multiple query positions, split/no-split paths, writer byte conformance
and CUDA Graph replay after changing the query.

As established by the import-only control in the previous rounds,
`import sgl_kernel` can produce dependency-initialization CUDA API errors
without running this extension. The memcheck invocation uses
`--report-api-errors no --error-exitcode 99`; device-memory and Hopper checks
remain enabled. This is not an unfiltered memcheck pass.

Baseline and candidate main kernels report 168 registers, 13 barriers,
32 stack bytes, four spill-store bytes and four spill-load bytes.

| Installed Binary | SHA256 |
|---|---|
| Baseline scalar | `188a6e1128e787b009571a27d18b086e92b170c50cefeabcdecd6d63e4d4b0dd` |
| Candidate scalar | `fcf6e36f1a65106ac0011f2a0fbd2095b022c8117fcaf08ee71bae474f46d18c` |
| Candidate BF16x2 | `e812bc665117fc3d0150804fffb7c4a8727a22e5d1290c21b604ddbd96fe3ed4` |

Only the candidate scalar CMake component is installed into the isolated
A/B container's serving package directory. No dependency versions change.
Python 3.12.3, Torch 2.13.0+cu130, Triton 3.7.1, Transformers 5.12.1,
sglang-kernel 0.4.6.post1 and FlashInfer 0.6.15.post1 are retained. Package
metadata reports Torch as 2.13.0; `torch.__version__` includes `+cu130`.
The CUDA compiler is 13.0.88; NCU and compute-sanitizer are 2025.3.1.

## Model Results

### Performance

All six runs completed 128 requests without request errors, with exactly
524,288 input and 32,768 output tokens per run. Each individual request
has 4096 input and 256 output tokens. Results below use the model protocol
above and three-run medians; latency values are medians of per-run means.

| Metric | Before INT4 | After INT4 | Change |
|---|---:|---:|---:|
| Total output throughput, tokens/s | 527.516 | 575.555 | +9.11% |
| Output throughput / 8 GPUs, tokens/s/GPU | 65.939 | 71.944 | +9.11% |
| Mean TPOT, ms | 50.921 | 46.543 | -8.60% |
| Mean TTFT, ms | 2295.194 | 2042.681 | -11.00% |
| Mean end-to-end latency, ms | 15281.388 | 13938.489 | -8.79% |
| Speculative accept length | 2.6906 | 2.6972 | Diagnostic |
| Full token-pool capacity | 7,959,296 | 7,959,296 | Unchanged |

| Repeat | Before Output Tokens/s | After Output Tokens/s |
|---|---:|---:|
| 1 | 527.516 | 571.602 |
| 2 | 527.988 | 575.555 |
| 3 | 525.697 | 580.282 |

Manifests agree on the service command, model configuration SHA256, package
versions, environment, extension ABI and import locations. Resolved server
settings also match after excluding startup-time and memory-observation
fields. Capacity is checked separately rather than assumed equal.
The model configuration SHA256 is
`52b5a1aa87606cb5be4f3158d706594edb1c4ce97ce6b1cd6079f15df075d7f5`.

Both phases complete HTTP health and generation probes. Logs confirm target
verify, draft decode and draft extend CUDA Graph capture. All eight candidate
scheduler processes map the installed scalar extension, use the intended
checkout and expose its explicit `PYTHONPATH`.

This is one fixed TP8 configuration, not a search for the best total
throughput or the best per-GPU throughput. Model measurements run before
then after, not in randomized/interleaved order. The candidate's three
throughput values trend upward, and speculative acceptance varies between
runs. The old-binary kernel repeat bounds observed microbenchmark drift;
it is not an interleaved model control. No full-model stage trace was
collected to attribute the end-to-end gain entirely to one kernel or stall.

### Accuracy

Both runs use the same 1319 questions, 8-shot prompts, temperature zero,
parallelism 32 and the serving configuration above. Corresponding questions
have identical input-token counts, ranging from 1261 to 1424 tokens.
Maximum output length is 1024 tokens; the server seed is 42.

| Metric | Before INT4 | After INT4 |
|---|---:|---:|
| Full legacy GSM8K | 1255/1319, 95.1478% | 1253/1319, 94.9962% |
| Excluding eight test-file exemplars | 1247/1311, 95.1182% | 1245/1311, 94.9657% |
| Generated tokens | 125,405 | 125,771 |
| Invalid parsed answers | 0 | 0 |
| Responses reaching the output limit | 1 | 0 |

The before version alone answers 14 questions correctly; the after version
alone answers 12. Numeric predictions match on 1288/1319 questions, and
full text matches on 393/1319. The full-score change is -0.1516 percentage
points. The single length-capped baseline answer, zero-based index 255,
is retained and scored as incorrect; no capped answer is removed or rerun.

Batch-dependent generation can vary even at temperature zero. This one
paired evaluation does not establish statistical equivalence, nor does it
isolate the cause of the score difference. Synthetic zero-tolerance kernel
results cover the tested inputs, not every possible model execution.
Independent repeated accuracy evaluation remains a release-validation gap.

The evaluator uses the first eight test-file rows as exemplars, so this is
not an independent train-exemplar GSM8K protocol. The separate 1311-question
score makes that limitation visible but does not remove all evaluation
limitations. The dataset SHA256 is
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

## Decision and Limits

- Retain implementation commit `5cb3842499`: one CUDA header changes, with
  unchanged format, dequantization formulas, cache capacity and synchronization.
  The measured fixed-workload improvement is supported by numerical checks,
  sanitizer results and a fresh model comparison.
- The new aligned 128-bit group loads and shared scale reuse reduce
  fine-grained load requests. NCU supports that mechanism, but does not
  attribute the whole end-to-end gain to one instruction class.
- Keep scalar and `aos_368` as defaults. BF16x2 remains optional, and no
  end-to-end BF16x2 or 384-byte-layout result is claimed.
- Both model phases use INT4. Native FP8 was not rerun; this round does not
  update the earlier FP8-relative performance or accuracy conclusions.
- Other models, long-context workloads, other TP/concurrency settings and
  randomized model A/B are outside this round. This is not a search for
  either the best total throughput or the best single-GPU efficiency.
- No dependencies, weights, native attention code, accuracy tolerances,
  scratch-cache policy or correctness barriers were changed.

## Evidence

- Remote host: `/mnt/nvme2/kvbit-future-20260908/round3`.
- Container: `/artifacts/round3`.
- Local archive: `../validation/dsv4-future-round3/round3-evidence.tar.gz`,
  with extracted files under `raw/`. It contains build/test/sanitizer logs,
  SASS, NCU reports, manifests, complete benchmark outputs, per-question
  answers and the paired comparison. Large synthetic `.pt` tensors and
  installed `.so` binaries remain in the remote artifact directory.
- Local analysis: `kernel-results.json`, `kernel-summary.json`,
  `summarize_kernel.py`, `performance-summary.json`, `comparison.json`,
  `after-worker-maps.json` and `cleanup-check.json` in that directory.
- NCU exports: `baseline-c4-raw.csv`, `candidate-c4-raw.csv` and the matching
  `*-details.txt` files in the local analysis directory.
- Build and sanitizer commands: `*-command.json` in the raw archive.
- Validation driver: `/artifacts/round3/validate_candidate.py`.
- Unchanged microbenchmark: `/artifacts/round2/microbench.py`.
- Unchanged model phase runner: `/artifacts/run_phase.py`.
- Unchanged paired comparator: `/artifacts/round2/compare_round.py`.
- CMake build directories: `/tmp/kvbit-future-round3-scalar-build` and
  `/tmp/kvbit-future-round3-vector-build`.
- Candidate prefixes: `/artifacts/round3/aot-scalar` and `aot-vector`.
- Baseline scalar prefix and build remain at `/artifacts/round2/aot-scalar`
  and `/tmp/kvbit-future-round2-scalar-build` for controlled rollback.
- Baseline source and package paths are recorded in each phase manifest.

The comparator was rerun locally from the recovered per-run/per-question
files. Its output SHA256 matches the remote result:
`fd8df25385e773ab9f63788860da74a06040fa69f91daeae1211315c61c618f6`.
The unchanged phase runner, microbenchmark and comparator also have matching
local/remote hashes. The raw archive SHA256 is
`24598e46340f55a55d53d577960390a3c698f833aca6dd43a0b4a2dff3b57645`.

All production-source changes reached the A/B checkout through Git.
The implementation is a single commit on top of the user's cherry-picked
upstream baseline; this report is separate from that implementation commit.
The original `kvbit-ds:/sgl-workspace/sglang-bytedance` stays on upstream
`8435df1760`, and its installed package is untouched by this round.

Both model phases completed and their process groups exited. The candidate
runner records `phase_complete` before `server_stopped`, with return code
`-9` during cleanup, not during measurement. The A/B container has only its
init/sleep processes after cleanup.

At the recorded 22:07 +08:00 check on 2026-09-09, GPUs 0-3 are occupied by a
different service in `dji-qwen`. Its processes started at 21:22 +08:00,
after this round ended at 16:23 +08:00. Those processes were not stopped or
modified. The host is therefore not reported as entirely idle.
