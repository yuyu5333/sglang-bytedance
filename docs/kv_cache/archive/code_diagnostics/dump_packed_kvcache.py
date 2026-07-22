"""Dump 一份真实的 3-bit packed KV cache（wall packed 布局）并解读其字节结构。

复用 ncu_packed_probe.py 的 workload 构造：用 FlashMLA 自带 testcase 生成器造出
MODEL1_FP8Sparse decode 的 bf16 KV，然后走 rotated_store_to_packed 打包成
uniform-3bit packed 布局，最后：

  1. 打印 packed cache 的形状 / 每 token 字节结构分解
  2. dump 前若干 token 的原始 packed bytes（hex）
  3. 用 CPU reference 解回 bf16，和打包前的原始 bf16 对照（cos / max_abs）
  4. 把 packed cache + 元信息存成 .pt 供离线分析

Usage (inside fp8-dsv4, cwd /workspace/FlashMLA):
    CUDA_VISIBLE_DEVICES=7 SGLANG_RQ_BIT_UNIFORM=3 \
        python3 tests/dump_packed_kvcache.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/sglang-bytedance/python")

import lib
from lib import RawTestParamForDecode as RawTestParam

from sglang.srt.mem_cache.rotated_quant_dsv4_memory_pool import (
    build_synthetic_dsv4_calibration,
)
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    packed_bytes_per_token,
    rotated_store_to_packed,
    rotated_load_to_fp8_layout_cpu_ref,
    _UNIFORM_HEADER_BYTES,
    _UNIFORM_GROUPS,
    _MLA_NOPE_DIM,
    _MLA_TILE_SIZE,
    _ROPE_BYTES,
)

dev = torch.device("cuda:0")
torch.set_default_device(dev)
torch.cuda.set_device(dev)
torch.set_default_dtype(torch.bfloat16)

B = int(os.environ.get("PROBE_B", "4"))
H_Q = int(os.environ.get("PROBE_HQ", "64"))
S_KV = int(os.environ.get("PROBE_SKV", "4096"))
TOPK = int(os.environ.get("PROBE_TOPK", "512"))
BLOCK_SIZE = int(os.environ.get("PROBE_BLK", "64"))
QK_NOPE = 448
D_QK = 512
N_DUMP = int(os.environ.get("DUMP_N_TOKENS", "4"))
OUT_PT = os.environ.get("DUMP_OUT", "/tmp/packed_kvcache_3bit.pt")

p = RawTestParam(
    b=B, h_q=H_Q, s_q=1, h_kv=1, s_kv=S_KV, is_varlen=False, topk=TOPK,
    have_topk_length=False, enable_attn_sink=True,
    block_size=BLOCK_SIZE, d_qk=D_QK, d_v=512,
    check_correctness=False, num_runs=0, seed=0,
).to_test_param()

t = lib.generate_testcase_for_decode(p)
kv_scope = t.kv_scope
kv_bf16 = kv_scope.blocked_k.view(-1, D_QK).contiguous()   # [num_rows, 512]
num_rows = kv_bf16.shape[0]

cfg = build_synthetic_dsv4_calibration(1, QK_NOPE)[0]
row_bytes_nope = cfg.row_bytes
bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)

print("=" * 72)
print("Packed 3-bit KV cache 布局解读")
print("=" * 72)
print(f"bit_uniform          = {cfg.bit_uniform}  (每个 nope 维度用几 bit)")
print(f"nope 维度            = {_MLA_NOPE_DIM}")
print(f"rope 维度            = {_MLA_TILE_SIZE}  (BF16, 固定不量化)")
print(f"uniform groups       = {_UNIFORM_GROUPS}  (每 {_MLA_TILE_SIZE} 维一组)")
print("-" * 72)
print("每 token 字节结构 (packed_bytes_per_token):")
print(f"  [0 : {row_bytes_nope}]           nope codes  = {_MLA_NOPE_DIM}dim x {cfg.bit_uniform}bit / 8 = {row_bytes_nope} B")
print(f"  [{row_bytes_nope} : {row_bytes_nope + _UNIFORM_HEADER_BYTES}]       affine header = {_UNIFORM_GROUPS}组 x (fp16 min + fp16 range) = {_UNIFORM_HEADER_BYTES} B")
print(f"  [{row_bytes_nope + _UNIFORM_HEADER_BYTES} : {bpt}]       rope BF16   = {_MLA_TILE_SIZE}dim x 2B = {_ROPE_BYTES} B")
print(f"  合计 bytes_per_token = {bpt} B   (对比 native FP8 = 584 B, 省 {584 - bpt} B = {(584 - bpt) / 584 * 100:.1f}%)")
print("=" * 72)

# ---- 打包 ----
PAGE_SIZE = BLOCK_SIZE
num_pages = (num_rows + PAGE_SIZE - 1) // PAGE_SIZE
packed_bytes_per_page = bpt * PAGE_SIZE
packed_cache = torch.zeros(num_pages, packed_bytes_per_page,
                           dtype=torch.uint8, device=dev)
loc = torch.arange(num_rows, dtype=torch.int32, device=dev)
kv_src = torch.nan_to_num(kv_bf16.to(torch.bfloat16), nan=0.0)
CHUNK = 16384
for s in range(0, num_rows, CHUNK):
    e = min(s + CHUNK, num_rows)
    rotated_store_to_packed(kv_src[s:e], packed_cache, loc[s:e],
                            page_size=PAGE_SIZE, cfg=cfg)
torch.cuda.synchronize()
packed_rows = packed_cache.view(num_pages * PAGE_SIZE, bpt)

print(f"\npacked_cache.shape = {tuple(packed_cache.shape)}  "
      f"(num_pages={num_pages}, bytes_per_page={packed_bytes_per_page})")
print(f"packed_rows.shape  = {tuple(packed_rows.shape)}  dtype={packed_rows.dtype}")
print(f"总字节数           = {packed_cache.numel()} B "
      f"({packed_cache.numel() / 1024 / 1024:.2f} MiB)")

# ---- 找有真实数据的 token（testcase 前面的行常是未用 slot=0）----
row_absmax = kv_src[:, :QK_NOPE].abs().amax(dim=1)
nonzero_rows = torch.nonzero(row_absmax > 1e-6, as_tuple=False).flatten()
if nonzero_rows.numel() > 0:
    dump_ids = nonzero_rows[:N_DUMP].cpu().tolist()
else:
    dump_ids = list(range(min(N_DUMP, num_rows)))
print(f"\n[info] 非零(有真实KV)的 token 行数 = {nonzero_rows.numel()} / {num_rows}")
print(f"[info] 将 dump 这些行: {dump_ids}")

# ---- dump N 个有数据 token 的原始字节 ----
print("\n" + "=" * 72)
print(f"选取的 {len(dump_ids)} 个有真实数据 token 的原始 packed bytes (hex)")
print("=" * 72)
for i in dump_ids:
    row = packed_rows[i].cpu()
    nope_codes = row[:row_bytes_nope]
    header = row[row_bytes_nope:row_bytes_nope + _UNIFORM_HEADER_BYTES]
    rope = row[row_bytes_nope + _UNIFORM_HEADER_BYTES:]
    # 解读 header 的 fp16 min/range
    hdr_fp16 = header.view(torch.float16).reshape(_UNIFORM_GROUPS, 2)
    print(f"\n--- token[{i}] ---")
    print(f"  nope codes (前 32B): {nope_codes[:32].numpy().tolist()}")
    print(f"  affine header ({_UNIFORM_HEADER_BYTES}B) 解读为 {_UNIFORM_GROUPS} 组 (min, range):")
    for g in range(_UNIFORM_GROUPS):
        print(f"    group[{g}]: min={hdr_fp16[g, 0].item():+.4f}  range={hdr_fp16[g, 1].item():.4f}")
    print(f"  rope (前 16B hex): {rope[:16].numpy().tobytes().hex()}")

# ---- 解回 bf16, 和原始对照 ----
print("\n" + "=" * 72)
print("解包正确性验证 (CPU reference dequant vs 原始 bf16)")
print("=" * 72)
M_check = min(256, num_rows)
# CPU reference: cfg.R 在 CPU，且 bitunpack 内部会用默认 device 建 arange，
# 因此临时把默认 device 切回 CPU，全程 CPU 张量，避免 device 混用。
torch.set_default_device("cpu")
idx = torch.arange(M_check, dtype=torch.int64, device="cpu")
nope_recon, rope_recon, _ = rotated_load_to_fp8_layout_cpu_ref(
    packed_cache.cpu(), idx, page_size=PAGE_SIZE, cfg=cfg,
)
torch.set_default_device(dev)
orig_nope = kv_src[:M_check, :QK_NOPE].float().cpu()
recon_nope = nope_recon.float().cpu()
diff = (orig_nope - recon_nope).abs()
cos = torch.nn.functional.cosine_similarity(
    orig_nope.flatten(), recon_nope.flatten(), dim=0
)
print(f"nope  max_abs_err = {diff.max().item():.6g}")
print(f"nope  mean_abs_err= {diff.mean().item():.6g}")
print(f"nope  cos_sim     = {cos.item():.9f}   (量化误差, 3bit 预期 cos>0.99)")

orig_rope = kv_src[:M_check, QK_NOPE:].float().cpu()
recon_rope = rope_recon.float().cpu()
rope_diff = (orig_rope - recon_rope).abs()
print(f"rope  max_abs_err = {rope_diff.max().item():.6g}   (rope BF16 无损, 应≈0)")

# ---- 存盘 ----
torch.save({
    "packed_cache": packed_cache.cpu(),
    "packed_rows": packed_rows[:1024].cpu(),
    "bit_uniform": cfg.bit_uniform,
    "row_bytes_nope": row_bytes_nope,
    "bytes_per_token": bpt,
    "num_pages": num_pages,
    "page_size": PAGE_SIZE,
    "uniform_header_bytes": _UNIFORM_HEADER_BYTES,
    "uniform_groups": _UNIFORM_GROUPS,
    "nope_dim": _MLA_NOPE_DIM,
    "rope_dim": _MLA_TILE_SIZE,
    "R_hadamard": cfg.R.cpu(),
    "orig_kv_bf16_sample": kv_src[:1024].cpu(),
}, OUT_PT)
print(f"\n已保存到 {OUT_PT}")
print("  keys: packed_cache, packed_rows[:1024], bit_uniform, row_bytes_nope,")
print("        bytes_per_token, R_hadamard, orig_kv_bf16_sample[:1024] ...")
