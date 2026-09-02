# KVBit FlashMLA specialization

This directory contains the SM90 sparse-decode specialization used by
DeepSeek V4 fixed-BU4 KV caches. It is derived from FlashMLA commit
`98751d47134c8f2f1a4df5b07875144c3d8075d1` and is distributed under the
MIT license in `LICENSE`.

The code is intentionally compiled into `kvbit_flashmla_ops`, separately from
the regular `flashmla_ops` extension. The regular extension is built from the
official `sgl-project/FlashMLA` dependency and retains its public ABI.

The specialization consumes the SGLang-owned 380-byte packed row:

- 224 bytes: 448 fixed 4-bit NoPE values
- 28 bytes: seven FP16 `(minimum, range)` pairs
- 128 bytes: 64 BF16 RoPE values

Only the fixed-BU4 production path is called by SGLang. The mirrored
FlashMLA structure is retained in this first migration to preserve the
producer/consumer barriers, shared-memory layout, and WGMMA pipeline exactly.
New features should not be added to this copy unless they are required by the
fixed-BU4 ABI.
