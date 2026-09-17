# DSV4.1 packed Main KV: staged consumer validation

| H20, Main K=1024, SWA K=128 | Legacy V4 | Packed staged | Packed direct |
|---|---:|---:|---:|
| B2, h64/h128, P128/P256 | 17.38–20.41 us | 42.61–47.33 us | 17.94–20.42 us |
| B32, h64/h128, P128/P256 | 61.43–96.55 us | 241.05–268.62 us | 61.99–96.18 us |
| Persistent Main storage / 65,536 slots / source | 36.5625 MiB | 24 MiB | 24 MiB |
| Fixed staging + remap, K1024 | 0 | 36.8125 MiB | 0 |
| Net saving with one source, excluding common allocations | 0 | -24.25 MiB | 12.5625 MiB |
| Single-call graph private pool, across these shapes | 12–40 MiB | 12–40 MiB | 12–40 MiB |

The staged path is an opt-in compatibility consumer. It converts selected packed
FP4 Main slots to legacy V4 FP8 pages and calls the existing FlashMLA attention.
It is slower than direct consumption in this measurement. `auto` is unchanged.
Enable with `--dsv41-main-kv-layout packed_fp4 --dsv41-main-kv-consumer staged`.
The supported configuration is SM90 with the FlashMLA backend.

The conversion preserves the legacy writer's rounding after decoding packed
values to BF16. It does not recover values lost by the original FP4 quantization.
Large sparse prefill uses positional BF16/FP8 gather. Decode, verify and small
prefill use a fixed 64-query workspace, serially reused across tiles and layers.
Each backend instance owns its workspace and is charged by the pool configurator.

Net memory savings depend on distinct KV sources, compression ratios, capacity,
top-k and backend ownership. Do not multiply savings by reader layer count.
For N sources each holding 65,536 slots and one K1024 workspace, the saving is
`N * 12.5625 - 36.8125 MiB`, before other model allocations. The checkpoint's
actual top-k may be 512, which needs an 18.40625 MiB workspace.

## Reproduce the kernel comparison

Build/install the repository's pinned SM90 FlashMLA extension and matching
`sgl_kernel` Python wrapper. Confirm the actual interpreter and extension paths
before measuring. The direct API requires the PR3-capable extension.

```bash
PYTHONPATH=python python benchmark/kernels/attention/bench_dsv41_staged_main.py \
  --output /tmp/staged-ab.json
PYTHONPATH=python python benchmark/kernels/attention/bench_dsv41_staged_main.py \
  --memory-only --output /tmp/staged-memory.json
```

The default comparison uses 65,536 slots, K1024, B2/B32, h64/h128 and P128/P256.
Each shape has one sample of 20 graph invocations, with bounded warmup; shapes
run serially. All paths use the same input and indices. Legacy is built from
decoded packed values through the actual V4 writer. Staged and legacy outputs
matched exactly in all eight measured cases. Direct has a different intermediate
quantization path, so its difference from legacy is reported separately.

Staged total calls the real `_forward_staged_main` method and includes scheduler
work and output copy. The separate staged-attention measurement also includes
these operations. These are kernel measurements, not serving QPS or P95.

The memory-only run captures one invocation and records both live allocations
and private-pool reservations. The latter accounts for kernel scratch retained
by a graph even after its Python tensor is gone. It does not measure a complete
model's multi-layer, multi-bucket graph residency.

## Reproduce FULL-page transport

Use an isolated container with an idle GPU, RDMA device access, host networking
and unlimited memlock. The script uses GPU0 as seen inside the container.

```bash
PYTHONPATH=python python benchmark/kernels/attention/smoke_dsv41_page_transfer.py \
  --transport mooncake
PYTHONPATH=python UCX_LOG_LEVEL=info \
  python benchmark/kernels/attention/smoke_dsv41_page_transfer.py --transport nixl
```

Two processes create real C1/C2 pools, validate matching region descriptors, and
transfer source page 1 to destination page 2. Each backend transferred 173,568 B
across four Main/Indexer regions with exact byte equality and intact neighboring
pages. Mooncake uses RDMA explicitly. NIXL restricts UCX to `rc,cuda_copy`; logs
confirmed `rc_mlx5`, excluding TCP/shared-memory/CUDA-IPC data paths.

HiCache reuses FULL-page geometry and namespaces persistent storage keys by
layout. PD registration carries versioned region descriptors; both Python and
Rust bootstrap preserve PP-local source lists, including empty SWA-only stages.
DSpark state-layout restrictions remain enforced.

Validation includes real V4-writer comparisons, mutated-data graph replay,
cross-tile queries, BF16/FP8 prefill, HiCache page round trips, CUDA memcheck,
Python wire/PP regressions and Rust bootstrap regressions. The RDMA smoke covers
two processes on one host/GPU. Cross-host model PD, request ACK/retraction
lifecycle, model accuracy and serving QPS/P95 have not been qualified.

Measurements used H20, Python 3.12, Torch 2.13.0+cu130, Triton 3.7.1, FlashMLA
`4960de014f73eb4475b5302b7c36ec2083a01d4a`. Timings were recorded at SGLang
`88dfca9bf5`; graph reservation and transport runs used `9c5a75358e`.
