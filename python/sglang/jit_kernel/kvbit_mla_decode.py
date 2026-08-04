"""JIT-compiled CUDA stage-1 kernel for kvbit no_alloc MLA decode attention.

Replaces the Triton ``_mla_decode_kernel`` (kvbit.triton_kernels) with a
hand-written CUDA kernel that fuses 4bit dequant + qk + online softmax + V
accumulation with vectorized loads and split-K parallelism. The Triton kernel
was 5.6x-29x slower than fa3 at bs=1 (13.7ms vs 2.8ms for 78 layers) because
bs=1 decode is a GEMV (M=1) where tensor cores don't help — fa3's speed is
memory-path efficiency, which this kernel mirrors for the 4bit KV (4x less
HBM than bf16).

This module exports only the stage-1 per-split kernel. The host-side
log-sum-exp merge across splits (``_reduce_splits``) and the V-path
inverse-rotate (``out @ R``) stay in ``kvbit.triton_kernels.mla_decode_fwd`` /
``dsa_backend._forward_kvbit`` — this kernel is a drop-in replacement for the
Triton stage-1 kernel only, producing the same (att_out, att_lse) tensors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_kvbit_mla_decode_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "kvbit_mla_decode_stage1",
        *args,
        cuda_files=["kvbit/mla_decode_stage1.cuh"],
        cuda_wrappers=[
            (
                "kvbit_mla_decode_stage1",
                f"KvbitMlaDecodeStage1Kernel<{args}>::run",
            )
        ],
    )


def kvbit_mla_decode_stage1(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kvbit_packed: torch.Tensor,
    rope_buf: torch.Tensor,
    sm_scale: float,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    num_kv_splits: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    max_splits: int,
) -> None:
    """Run the CUDA stage-1 kernel in-place into att_out / att_lse.

    Inputs (must be CUDA, contiguous in the leading dims):
      q_nope        (B, H, 512) bf16 — Q-FHT folded.
      q_pe          (B, H, 64)  bf16.
      kvbit_packed  (num_slots, 288) uint8 — 4bit nope packed (rope_dim=0).
      rope_buf      (num_slots, 1, 64) bf16.
      page_table    (B, max_len) int32.
      cache_seqlens (B,) int32.
      num_kv_splits (B,) int32.
    Outputs (pre-allocated, written in-place):
      att_out       (B, H, max_splits, 512) fp32 — normalized per-split output.
      att_lse       (B, H, max_splits) fp32 — logsum-exp per split.
    """
    if q_nope.device.type != "cuda":
        raise RuntimeError("kvbit_mla_decode_stage1 requires CUDA tensors")
    module = _jit_kvbit_mla_decode_module(is_arch_support_pdl())
    module.kvbit_mla_decode_stage1(
        q_nope, q_pe, kvbit_packed, rope_buf, sm_scale,
        page_table, cache_seqlens, num_kv_splits,
        att_out, att_lse, max_splits,
    )
