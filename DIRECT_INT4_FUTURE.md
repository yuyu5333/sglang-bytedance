# Direct INT4 Future Branch

## Status

Branch: `feat/dsv4-direct-int4-g32-e4m3-future`.
Starting commit: `13119ebba6`.

This is a local research iteration, not a deployment-ready revision. No SSH,
remote build, GPU execution, model evaluation, service launch, or performance
benchmark was performed. Historical H20 results do not validate this revision.
Defaults retain the compact format and scalar dequantization, but the producer
and combine implementation have changed even with all experimental switches off.

## Implemented

| Area | Change | Evidence Available Here |
|---|---|---|
| Format definition | One JSON codec specification generates Python and C++ constants and a schema hash | Generator drift check; CPU codec tests; host C++ compilation |
| Physical layout | `aos_368` and `aos_384` use identical payload offsets; aligned rows add trailing padding | Byte equality and lossless repacking on CPU |
| Capacity | SWA, C4, C128 charge the selected physical row size | Pool factory and complete pool-configurator test file |
| Startup | Probe the AOT extension, ABI version 2, registered op, and CUDA dispatch implementation | Dependency and dispatch failure tests with native dependencies mocked |
| Geometry | Reject unsupported local heads, MQA/value dimensions, NoPE/RoPE dimensions, and sparse width before packed pool allocation | CPU contract tests |
| Writer | Support row and location strides, 64-bit address arithmetic, empty writes, and both row sizes; clamp before integer conversion | Host validation; GPU byte-conformance cases written but not executed |
| Rounding | Explicit `tl.div_rn` and `cvt.rni.s32.f32` for the intended RNE contract | CPU boundary fixtures; GPU fixtures pending execution |
| Producer | Flat row addressing, warp index broadcast, bounds masks; remove legacy rotated/affine branches and destructive probes | Static inspection only |
| Dequantization | Optional BF16x2 mantissa insertion and scale broadcast within four-lane groups | CPU algebra checks over every nibble and finite nonnegative E4M3 scale |
| SWA norm/RoPE | Optional reuse of the existing strided Triton implementation, removing the tail-view copy | Routing implemented; GPU equivalence case not executed |
| LSE | Include sink in both no-split and private split-K combine, using stable log-sum-exp | 120 host C++ math cases; full GPU output/LSE comparison not executed |

The new code does not add a native shadow cache, a packed-to-native decode
scratch buffer, rotation, an external KVBit dependency, or mixed-format pages.
The existing attention split-accumulation workspace still exists.

## Format and Controls

The source specification is
`python/sglang/srt/mem_cache/kvbit_dsv4_format.json`.
This is a deliberately restricted generator for one codec, not a general
format compiler. Unsupported codec/schema changes are rejected.

```bash
python3 scripts/kvbit/generate_dsv4_layout.py
python3 scripts/kvbit/generate_dsv4_layout.py --check
```

| Control | Default | Alternative |
|---|---|---|
| `--kv-cache-dtype` | Existing server default | `int4` enables the target-worker path |
| `SGLANG_DSV4_INT4_LAYOUT` | `aos_368` | `aos_384` |
| `SGLANG_DSV4_INT4_STRIDED_NORM_ROPE` | `0` | `1` uses the strided Triton norm/RoPE kernel |
| CMake `SGL_KERNEL_ENABLE_DSV4_INT4` | `ON` | `OFF` omits the private extension |
| CMake `SGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT` | `OFF` | `ON` uses BF16x2 arithmetic and cooperative scale loads |

Set environment controls before startup, consistently on all target ranks.
Do not change layout after capacity planning, pool allocation, or graph capture.
Both sources in one attention call must use the same physical layout.
The extension build gate currently requires CUDA 12.8 or newer; a minimum-version
CUDA build matrix was not run.

| Layout | Codes | Scales and Header Padding | BF16 RoPE | Trailing Padding | Bytes/Row |
|---|---|---|---|---|---:|
| `aos_368` | `[0,224)` | `[224,240)` | `[240,368)` | None | 368 |
| `aos_384` | `[0,224)` | `[224,240)` | `[240,368)` | `[368,384)` | 384 |

Scales occupy `[224,238)`; `[238,240)` is zero padding. The codec remains
signed INT4, G32, E4M3 nearest-even scales, and unrotated NoPE. The writer uses
the stored scale to quantize codes in `[-7,7]`. RoPE stays BF16.

Relative to a 584-byte native FP8 row, the storage reductions are 36.99% and
34.25%, respectively. These are byte-accounting results, not token-capacity or
throughput measurements. State, indexer, and draft pools are unchanged.

## Why These Experiments

### Stride Alignment

For a 128-byte-aligned base and eight successive rows, address-only enumeration
of the 368-byte payload gives:

| Layout | 32-Byte Sectors Touched Per Isolated Row | Average 128-Byte Spans |
|---|---:|---:|
| `aos_368` | 12 | 3.75 |
| `aos_384` | 12 | 3.00 |

Thus alignment does **not** prove fewer sector transactions or lower HBM traffic.
Cache reuse, issued load widths, sparse index distribution, and execution order
are missing from this calculation. The 384-byte format costs 4.35% more storage
than 368 bytes and should remain opt-in until a controlled GPU comparison exists.
RoPE remains at offset 240, so this is a stride experiment, not a separate
RoPE-alignment experiment.

### BF16x2 Dequantization

For a nibble `n`, `n ^ 8` is the biased signed code. Inserting that value in the
BF16 mantissa at exponent 128 gives `128 + (n ^ 8)` exactly. Subtracting 136
recovers `(n ^ 8) - 8`. Both the integer code and E4M3 scale are exactly
representable in BF16.

CPU tests compare the proposed BF16 multiplication with the scalar FP32 product
followed by BF16 conversion for all 16 nibbles and 127 finite nonnegative E4M3
encodings: 2,032 combinations, including zero and the reserved `-8` reader code.
This checks the algebra, not CUDA instruction lowering, register usage, or speed.

### Producer Simplification

The API requires contiguous packed rows, so
`page * page_size * stride + slot * stride` reduces to `token_index * stride`.
One lane loads an index and broadcasts it within the warp. Negative and
out-of-pool indices are masked before dereferencing. The same warp stages
NoPE and RoPE, eliminating the shared pointer table and producer-only barriers.
The two K buffers, QK/PV WGMMA shapes, consumer register budgets, and split
scheduler policy remain unchanged.

No occupancy improvement is claimed. Even reducing K buffers from two to one
would not shrink their shared-memory union below the existing FP32 output
accumulator: `64 * 520 * 4 = 133,120` bytes. This exceeds two BF16 K tiles'
`2 * 64 * 512 * 2 = 131,072` bytes. Merely tuning the buffer count is therefore
not sufficient to unlock a second resident CTA.

### LSE Contract

The CMake-pinned upstream combine source writes LSE before applying the sink,
just as the old no-split epilogue omits it. The private INT4 combine now uses
the same sink-inclusive denominator for output and LSE. Native FlashMLA's
combine and public ABI are untouched.

Final LSE is natural-log; intermediate split LSE is base-2 and excludes sink.
The sink is added once after merging splits. A sink-only row returns zero
output and the sink's logit as LSE. The upstream `+inf` empty-state sentinel
is retained for a sink-free, all-masked row.

## Verification

Local environment: macOS arm64, Python 3.13.14, PyTorch 2.14.0,
Transformers 5.12.1, in a dedicated local environment. This is not the
server's pinned CUDA environment. `PYTHONPATH` pointed explicitly to this
checkout's `python` directory; the imported `sglang.__file__` was verified.
The final environment is under
`~/.cache/kvbit-future-validation-20260908/venv`, after package files disappeared
from the earlier system-temporary environment.

| Check | Result |
|---|---|
| Combined focused CPU regression | 76 passed; 50 subtests passed |
| GPU conformance collection | 4 skipped because SM90 is unavailable |
| Host C++ ABI/math executable | Compiled and passed 120 sink/LSE cases |
| Generated ABI consistency | Passed |
| Python AST, import lint, formatting, spelling | Passed in local checks |
| Full staged pre-commit | Passed with Python 3.13 and staged change scope |
| Coverage threshold | Not requested; no coverage percentage claimed |
| CUDA compilation, sanitizers, graph execution | Not performed |
| Task quality, service smoke, throughput/latency | Not performed |

Added CPU coverage includes 18 new test methods: codec/contract (8), startup
probe (5), schema/host ABI (3), aligned pool allocation (1), and aligned budget
(1). Existing route tests now exercise both layouts.

The four GPU test methods are in
`test/registered/kernel/mem_cache/test_kvbit_dsv4_cuda.py`. They cover writer
bytes and strides; forced no-split and split-K attention, including multiple
query positions and SWA+C4/C128; graph replay with an updated query; and strided
norm/RoPE versus the contiguous Triton implementation. Their tolerances and
fixtures have not been calibrated against an actual GPU run.

With a Python 3.10+ environment and the test dependencies installed:

```bash
PYTHONPATH="$PWD/python" python3 -m pytest \
  test/registered/unit/mem_cache/test_kvbit_dsv4.py \
  test/registered/unit/mem_cache/test_kvbit_dsv4_codec.py \
  test/registered/unit/mem_cache/test_kvbit_dsv4_runtime.py \
  test/registered/unit/mem_cache/test_kvbit_dsv4_abi.py \
  test/registered/unit/model_executor/test_pool_configurator.py -q
```

The test-generation preparation, scope selection, defect analysis, generation
and verification loop were completed. Early collection failures were missing
local dependencies, not changed assertions. Existing tests were retained.

## Boundaries and Next Experiments

- Rebuild the private AOT extension before any GPU use; stale ABI versions fail
  at startup. CPU tests cannot establish CUDA compilation or synchronization.
- Defaults and experimental variants all need SM90 numerical validation,
  compute-sanitizer, and graph capture/replay before serving traffic.
- Validate compact/scalar first, then change only stride, then only vector
  dequantization. Combine switches only after each isolated comparison passes.
- The strided norm switch changes the norm/RoPE implementation as well as
  removing the copy. It is not a pure copy-only timing comparison.
- Record register counts, spills, shared memory, barrier stalls, L2 sectors,
  writer time, decode time, and combine time. Do not infer end-to-end gains from
  fewer source instructions or smaller persistent storage.
- Re-run fixed-workload and fixed-memory-budget comparisons separately, with
  model, lengths, requests, concurrency, TP/PP/DP, speculative settings, graph
  mode, KV format, memory fraction, seed, and build commit recorded.
- Multi-format page tables, hot/cold migration, additional models, and SM100
  backends remain design directions, not implemented capabilities.
