"""Minimal hang diagnosis."""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from sgl_kernel import flashmla_ops
from sgl_kernel.flash_mla import get_mla_metadata

print("CUDA_LAUNCH_BLOCKING=", os.environ.get("CUDA_LAUNCH_BLOCKING"))
print("banner =", torch.ops.sgl_kernel.flashmla_fork_probe())

batch_size = 1
seq_len_q = 1
seq_len_k = 64
num_heads_q = 128
num_heads_k = 1
head_dim_k = 576
head_dim_v = 512
page_size = 64
qk_nope = 512
num_pages = 1

torch.manual_seed(42)
q_fp8 = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim_k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
descale_q = torch.ones(1, device="cuda", dtype=torch.float32)
descale_k = torch.ones(1, device="cuda", dtype=torch.float32)
k_cache = torch.zeros(num_pages, page_size, num_heads_k, head_dim_k, device="cuda", dtype=torch.float8_e4m3fn)
block_table = torch.zeros(batch_size, 1, device="cuda", dtype=torch.int32)
cache_seqlens = torch.tensor([seq_len_k], device="cuda", dtype=torch.int32)

tile_md, num_splits = get_mla_metadata(
    cache_seqlens, seq_len_q * num_heads_q // num_heads_k,
    num_heads_k, is_fp8_kvcache=True,
)

# minimal calib
num_rows = num_pages * page_size
scale_kcache = torch.zeros(num_rows, qk_nope, device="cuda", dtype=torch.float32)
zero_point = torch.zeros(qk_nope, device="cuda", dtype=torch.float32)
R_matrix = torch.eye(qk_nope, device="cuda", dtype=torch.float32)
nope_bytes = qk_nope // 8
rope_bytes = 64 * 2
packed_row_bytes = nope_bytes + rope_bytes
packed_kcache = torch.zeros(num_rows, packed_row_bytes, device="cuda", dtype=torch.uint8)
dim_of_bit = torch.arange(qk_nope, device="cuda", dtype=torch.int32)
bitpos_in_dim = torch.zeros(qk_nope, device="cuda", dtype=torch.int32)

print("about to call packed kernel ...", flush=True)
import time
t0 = time.time()
out_packed, lse_packed = torch.ops.sgl_kernel.fwd_kvcache_mla_packed_fp8(
    q_fp8, k_cache, head_dim_v, cache_seqlens, block_table,
    head_dim_k ** (-0.5), True,
    tile_md, num_splits,
    descale_q, descale_k,
    packed_kcache, scale_kcache, R_matrix, zero_point,
    dim_of_bit, bitpos_in_dim,
)
torch.cuda.synchronize()
t1 = time.time()
print(f"packed kernel returned in {t1-t0:.2f}s")
print("out shape:", out_packed.shape)
print("OK")
