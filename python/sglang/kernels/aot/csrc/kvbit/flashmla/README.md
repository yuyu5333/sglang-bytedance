# KVBit FlashMLA specialization

This directory contains the SM90 sparse-decode specialization used by the
DeepSeek V4 INT4 KV cache. It is derived from FlashMLA commit
`98751d47134c8f2f1a4df5b07875144c3d8075d1` and is distributed under the
MIT license in `LICENSE`.

The code is intentionally compiled into `kvbit_flashmla_ops`, separately from
the regular `flashmla_ops` extension. The regular extension is built from the
official `sgl-project/FlashMLA` dependency and retains its public ABI.

The `kvbit_int4_sparse_decode_fwd` specialization consumes a 368-byte payload:

- 224 bytes: 448 signed two's-complement int4 values, even dimension in the
  low nibble
- 14 bytes: fourteen E4M3 `absmax / 7` steps, one per 32 NoPE dimensions
- 2 bytes: zero padding
- 128 bytes: 64 BF16 RoPE values

The specialization uses MODEL1/H64 and consumes NoPE directly without a
Hadamard transform. SWA and optional extra KV use the same row format and scale
contract. The official FlashMLA ABI remains unchanged.

The future branch also supports a 384-byte stride with 16 trailing zero bytes.
Payload offsets do not move. `packed_layout.h` is generated from
`python/sglang/srt/mem_cache/kvbit_dsv4_format.json`; run
`python3 scripts/kvbit/generate_dsv4_layout.py --check` from the repository root
to detect drift. The private extension ABI is version 2.

The producer now addresses flat contiguous rows, masks out-of-range token
indices, and no longer includes the historical rotated/affine reconstruction
or destructive performance probes. An optional BF16x2 implementation is selected
at build time with `SGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT=ON`. It is off by default.
Both no-split and split-K LSE include the attention sink. The private combine
kernel is derived from the CMake-pinned FlashMLA
`c1dee569a494b184811a08171a690ece21420262`; native FlashMLA is not modified.

These changes require a fresh AOT build and GPU conformance before deployment.
The future branch has no GPU performance or task-quality validation.
