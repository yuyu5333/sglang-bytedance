"""Standalone single-GPU microbench that drives the packed FP8 sparse-decode
FlashMLA kernel (use_packed=true branch) so ncu / nsys can attach and report
the definitive stall-reason / memory-throughput / occupancy breakdown.

Why: clock64 segment profiling localized WHERE the time goes (producer
nope_rebuild ~477K cyc/block, fill_sX 231K = per-token scattered global read),
but never answered WHY it stalls (memory-latency-bound vs occupancy-bound vs
barrier-bound). Those imply opposite levers (byte-reduction/vectorize vs raise
occupancy). This target lets ncu settle it with WarpState + Memory Workload.

Approach: reuse FlashMLA's own validated testcase generator to build a legal
MODEL1_FP8Sparse decode workload (q / k_cache / indices / sched_meta). Then
build a byte-correct packed swa cache (rotate+affine+bitpack) whose row layout
matches indices_in_kvcache exactly, and inject the 6 packed kwargs so the
kernel runs the use_packed inner loop (no IMA).

Usage (inside fp8-dsv4, cwd /workspace/FlashMLA):
    CUDA_VISIBLE_DEVICES=7 python3 tests/ncu_packed_probe.py         # sanity
    CUDA_VISIBLE_DEVICES=7 ncu --set full -k regex:splitkv.*sparse -c 3 \
        python3 tests/ncu_packed_probe.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ for lib,quant
sys.path.insert(0, "/workspace/sglang-bytedance/python")        # sglang fork

import flash_mla
import lib
from lib import RawTestParamForDecode as RawTestParam

from sglang.srt.mem_cache.rotated_quant_dsv4_memory_pool import (
    build_synthetic_dsv4_calibration,
)
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    packed_bytes_per_token,
    rotated_store_to_packed,
    _get_cached_cfg_gpu,
)
from sglang.jit_kernel.hadamard import _jit_hadamard_module, hadamard_transform

dev = torch.device("cuda:0")
# The FlashMLA testcase generator (lib.generate_testcase_for_decode ->
# _randperm_batch) allocates helper tensors on the *default* device. Match the
# validated tests (test_flash_mla_sparse_decoding.py:240) so cuda tensors and
# cpu helpers don't collide.
torch.set_default_device(dev)
torch.cuda.set_device(dev)
# MODEL1_FP8Sparse layout hardcodes 2-byte (bf16) rope; the generator uses the
# default dtype, so it must be bf16 (matching the validated test main()).
torch.set_default_dtype(torch.bfloat16)

# ---- production-like MODEL1 swa decode workload ----
B = int(os.environ.get("PROBE_B", "32"))
# Production DSV4 uses num_attention_heads=64 (config.json), so h_q=64 ->
# CLUSTER_SIZE=1 -> wgmma_uniform_supported path reads R_bf16 (correct). H_Q=128
# would give CLUSTER_SIZE=2 -> legacy branch that reads the null fp32 R_matrix_ptr
# (IMA at splitkv_mla.cuh:1488). Match production exactly.
H_Q = int(os.environ.get("PROBE_HQ", "64"))
S_KV = int(os.environ.get("PROBE_SKV", "4096"))
TOPK = int(os.environ.get("PROBE_TOPK", "512"))
BLOCK_SIZE = int(os.environ.get("PROBE_BLK", "64"))
ITERS = int(os.environ.get("PROBE_ITERS", "50"))
QK_NOPE = 448
D_QK = 512
BIT_UNIFORM = int(os.environ.get("PROBE_BIT_UNIFORM", "3"))
Q_FOLD = int(os.environ.get("PROBE_Q_FOLD", "0"))
COMPARE = int(os.environ.get("PROBE_COMPARE", "0"))
TIMING = int(os.environ.get("PROBE_TIMING", "0"))
R_IDENTITY = int(os.environ.get("PROBE_R_IDENTITY", "0"))
IDENTITY_TAIL_BYPASS = int(os.environ.get("PROBE_IDENTITY_TAIL_BYPASS", "0"))
DEBUG_U32_LOAD = int(os.environ.get("PROBE_DEBUG_U32_LOAD", "0"))
COMPARE_U32_LOAD = int(os.environ.get("PROBE_COMPARE_U32_LOAD", "0"))

p = RawTestParam(
    b=B, h_q=H_Q, s_q=1, h_kv=1, s_kv=S_KV, is_varlen=False, topk=TOPK,
    have_topk_length=False, enable_attn_sink=True,
    block_size=BLOCK_SIZE, d_qk=D_QK, d_v=512,
    check_correctness=False, num_runs=0, seed=0,
).to_test_param()

t = lib.generate_testcase_for_decode(p)
sched_meta = flash_mla.get_mla_metadata()[0]

kv_scope = t.kv_scope
k_cache = kv_scope.get_kvcache_for_flash_mla()   # MODEL1_FP8Sparse quantized
indices = kv_scope.indices_in_kvcache            # [b, s_q, topk] int32
# dequantized bf16 kv (rotate source): kv_scope.blocked_k after quant_and_dequant_
# is the dequantized bf16 [num_blocks, block_size, h_kv, d_qk].
kv_bf16 = kv_scope.blocked_k.view(-1, D_QK).contiguous()   # [num_rows, 512]
num_rows = kv_bf16.shape[0]
print(f"[probe] b={B} topk={TOPK} block={BLOCK_SIZE} num_rows={num_rows} "
      f"k_cache.shape={tuple(k_cache.shape)}")

# ---- build byte-correct packed cache, row layout == indices_in_kvcache ----
cfg = build_synthetic_dsv4_calibration(1, QK_NOPE)[0]
if R_IDENTITY:
    cfg.R = torch.eye(QK_NOPE, dtype=torch.float32)
    print("[probe] PROBE_R_IDENTITY=1 (store/load R is identity)")
row_bytes_nope = cfg.row_bytes
bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
print(f"[probe] bit_uniform={cfg.bit_uniform} row_bytes_nope={row_bytes_nope} "
      f"packed_bpt={bpt}  (native FP8 = 584)")

# indices_in_kvcache = block_index * block_size + offset. That equals the flat
# row into k_cache.view(-1, d_qk) i.e. our num_rows. Use page_size=block_size so
# the packed store's page*page_size+slot maps identically.
PAGE_SIZE = BLOCK_SIZE
num_pages = (num_rows + PAGE_SIZE - 1) // PAGE_SIZE
packed_bytes_per_page = bpt * PAGE_SIZE
packed_cache = torch.zeros(num_pages, packed_bytes_per_page,
                           dtype=torch.uint8, device=dev)

loc = torch.arange(num_rows, dtype=torch.int32, device=dev)
CHUNK = 16384
# replace NaN (unused slots) with 0 so pack does not propagate NaN
kv_src = torch.nan_to_num(kv_bf16.to(torch.bfloat16), nan=0.0)
for s in range(0, num_rows, CHUNK):
    e = min(s + CHUNK, num_rows)
    rotated_store_to_packed(
        kv_src[s:e], packed_cache, loc[s:e], page_size=PAGE_SIZE, cfg=cfg,
    )
torch.cuda.synchronize()
packed_rows = packed_cache.view(num_pages * PAGE_SIZE, bpt)

cfg_gpu = _get_cached_cfg_gpu(cfg, dev)
_bu = int(cfg.bit_uniform)
packed_kwargs = {
    "packed_kcache": packed_rows,
    "scale_kcache": cfg_gpu["scale"],
    "R_matrix": cfg_gpu["R_bf16"] if _bu > 0 else cfg_gpu["R"],
    "zero_point": cfg_gpu["zero"],
    "dim_of_bit": cfg_gpu["dim_of_bit"],
    "bitpos_in_dim": cfg_gpu["bitpos_in_dim"],
    "bit_uniform": _bu,
}

# [Route H step6] PROBE_NATIVE=1 drops ALL packed kwargs so the SAME kernel
# takes the native FP8 producer branch (params.packed_kcache_ptr == nullptr)
# on the IDENTICAL k_cache / indices / q workload. This is the apples-to-apples
# native-vs-packed comparison NCU needs to attribute the 2.18x decode gap to
# either the producer skeleton (Stall Barrier), memory latency, or occupancy.
PROBE_NATIVE = int(os.environ.get("PROBE_NATIVE", "0"))
if PROBE_NATIVE:
    packed_kwargs = {}
    print("[probe] PROBE_NATIVE=1 -> native FP8 producer path (no packed kwargs)")

q_call = t.q
q_nope_is_folded = False
q_folded = None


def fold_q_exact(q):
    if _bu != 4 or not IDENTITY_TAIL_BYPASS:
        raise RuntimeError(
            "exact folded-Q requires PROBE_BIT_UNIFORM=4 and "
            "PROBE_IDENTITY_TAIL_BYPASS=1"
        )
    folded = q.clone()
    folded[..., :256] = hadamard_transform(q[..., :256], scale=0.0625)
    return folded


def hadamard256_inplace(x):
    rows = x.reshape(-1, x.shape[-1])
    prefix = rows.as_strided(
        (rows.shape[0], 256),
        (rows.stride(0), rows.stride(1)),
    )
    _jit_hadamard_module(x.dtype).hadamard_transform(prefix, prefix, 0.0625)
    return x


if Q_FOLD:
    q_folded = fold_q_exact(t.q)
    q_call = q_folded
    q_nope_is_folded = True
    print(
        "[probe] q_nope_is_folded=True "
        "(exact FP32 H256 butterfly, producer writes rotated K directly)"
    )


def one_call_full(
    q_arg=q_call,
    folded=q_nope_is_folded,
    debug_u32_load=DEBUG_U32_LOAD,
):
    return flash_mla.flash_mla_with_kvcache(
        q=q_arg,
        k_cache=k_cache,
        head_dim_v=p.d_v,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=sched_meta,
        softmax_scale=t.sm_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=kv_scope.topk_length,
        attn_sink=t.attn_sink,
        q_nope_is_folded=folded,
        identity_tail_bypass=bool(IDENTITY_TAIL_BYPASS),
        debug_u32_packed_load=bool(debug_u32_load),
        **packed_kwargs,
    )


def one_call(q_arg=q_call, folded=q_nope_is_folded, debug_u32_load=DEBUG_U32_LOAD):
    return one_call_full(q_arg, folded, debug_u32_load)[0]


def print_diff(name, ref, cur):
    ref_f = ref.float().flatten()
    cur_f = cur.float().flatten()
    diff = (ref_f - cur_f).abs()
    denom = ref_f.abs().mean().clamp_min(1e-8)
    cos = torch.nn.functional.cosine_similarity(ref_f, cur_f, dim=0)
    print(
        f"[probe][compare] {name}: "
        f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g} "
        f"rel_mean={float(diff.mean() / denom):.6g} cos={cos.item():.9f}"
    )


out = one_call()
torch.cuda.synchronize()
print(f"[probe] out.shape={tuple(out.shape)} "
      f"finite={torch.isfinite(out).all().item()}")

if COMPARE_U32_LOAD and _bu > 0:
    print(
        "[probe] PROBE_COMPARE_U32_LOAD=1; this is meaningful only when "
        "FlashMLA was built with FMLA_ENABLE_U32_LOAD_ORACLE."
    )
    out_byte = one_call(t.q, False, debug_u32_load=0)
    torch.cuda.synchronize()
    out_u32 = one_call(t.q, False, debug_u32_load=1)
    torch.cuda.synchronize()
    print_diff("byte_load_vs_u32_load", out_byte, out_u32)

if COMPARE:
    out_base, lse_base = one_call_full(t.q, False)
    torch.cuda.synchronize()
    if q_folded is None:
        q_folded = fold_q_exact(t.q)
    out_fold, lse_fold = one_call_full(q_folded, True)
    out_fold_restored = out_fold.clone()
    hadamard256_inplace(out_fold_restored)
    torch.cuda.synchronize()
    print_diff("inverse_k_vs_folded_q_out_rotated", out_base, out_fold)
    print_diff("inverse_k_vs_folded_q_out_restored", out_base, out_fold_restored)
    print_diff("inverse_k_vs_folded_q_lse", lse_base, lse_fold)


if TIMING:
    timing_iters = int(os.environ.get("PROBE_TIMING_ITERS", "100"))
    q_timing = fold_q_exact(t.q)
    out_timing, _ = one_call_full(q_timing, True)
    q_inplace_timing = t.q.clone()
    out_inplace_timing = out_timing.clone()

    def folded_pipeline():
        q_timed = fold_q_exact(t.q)
        out_timed, _ = one_call_full(q_timed, True)
        return torch.cat(
            (
                hadamard_transform(out_timed[..., :256], scale=0.0625),
                out_timed[..., 256:],
            ),
            dim=-1,
        )

    for _ in range(10):
        one_call(t.q, False)
        folded_pipeline()
    torch.cuda.synchronize()
    for name, fn in (
        ("inverse_k", lambda: one_call(t.q, False)),
        ("q_fht_cat", lambda: fold_q_exact(t.q)),
        ("folded_k_kernel", lambda: one_call(q_timing, True)),
        (
            "out_fht_cat",
            lambda: torch.cat(
                (
                    hadamard_transform(out_timing[..., :256], scale=0.0625),
                    out_timing[..., 256:],
                ),
                dim=-1,
            ),
        ),
        ("q_fht_inplace", lambda: hadamard256_inplace(q_inplace_timing)),
        ("out_fht_inplace", lambda: hadamard256_inplace(out_inplace_timing)),
        ("folded_q_full_pipeline", folded_pipeline),
    ):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timing_iters):
            fn()
        end.record()
        end.synchronize()
        print(
            f"[probe][timing] {name}: "
            f"{start.elapsed_time(end) * 1000.0 / timing_iters:.3f} us"
        )


for _ in range(ITERS):
    one_call()
torch.cuda.synchronize()
print("[probe] done")
