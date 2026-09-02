# KVBit FlashMLA specialization

This directory contains the SM90 sparse-decode specializations used by
DeepSeek V4 fixed-BU4 and MXINT4 KV caches. It is derived from FlashMLA commit
`98751d47134c8f2f1a4df5b07875144c3d8075d1` and is distributed under the
MIT license in `LICENSE`.

The code is intentionally compiled into `kvbit_flashmla_ops`, separately from
the regular `flashmla_ops` extension. The regular extension is built from the
official `sgl-project/FlashMLA` dependency and retains its public ABI.

The specialization consumes the SGLang-owned 380-byte packed row:

- 224 bytes: 448 fixed 4-bit NoPE values
- 28 bytes: seven FP16 `(minimum, range)` pairs
- 128 bytes: 64 BF16 RoPE values

The independent `kvbit_mxint4_sparse_decode_fwd` specialization consumes a
360-byte row:

- 224 bytes: 448 signed two's-complement int4 values, even dimension in the
  low nibble
- 7 bytes: one UE8M0 scale per group of 64 NoPE dimensions
- 1 byte: zero padding
- 128 bytes: 64 BF16 RoPE values

Both specializations use MODEL1/H64 and restore the NoPE key with normalized
H256 on dimensions `[0, 256)` plus identity on `[256, 448)`. SWA and optional
extra KV use the same row format and scale contract. MXINT4 has its own Torch
op and Python wrapper, `kvbit_mxint4_flash_mla_with_kvcache`; the official
FlashMLA ABI and the existing fixed-BU4 ABI remain unchanged.
