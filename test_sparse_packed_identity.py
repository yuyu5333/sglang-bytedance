import sys
sys.path.insert(0, "/workspace/FlashMLA/tests")
import torch
import lib
from lib import RawTestParamForDecode as RawTestParam
import flash_mla.cuda as fmc

tp = RawTestParam(b=1, h_q=64, s_q=1, h_kv=1, s_kv=128, is_varlen=True, topk=64,
    have_topk_length=False, enable_attn_sink=True,
    extra_s_k=None, extra_topk=None, block_size=64, extra_block_size=None,
    have_extra_topk_length=False, d_qk=512, check_correctness=True, num_runs=0)

tp = tp.to_test_param()
scope = tp.kv_scope0
q = tp.q
kv = scope.get_kvcache_for_flash_mla()
indices = scope.indices_in_kvcache
topk_length = scope.topk_length
attn_sink = tp.attn_sink

print(f"q shape: {q.shape}, dtype: {q.dtype}")
print(f"kv shape: {kv.shape}, dtype: {kv.dtype}")
print(f"indices shape: {indices.shape}, dtype: {indices.dtype}")

out_dense, lse_dense, _, _ = fmc.sparse_decode_fwd(
    q, kv, indices, topk_length, attn_sink,
    None, None, None, None, None,
    tp.d_v, tp.sm_scale)
print(f"out shape: {out_dense.shape}")
print(f"lse shape: {lse_dense.shape}")
print("Dense path (all-None) PASSED")

# Now test with packed identity calib
import numpy as np
device = "cuda"
num_blocks = kv.shape[0]
page_block_size = kv.shape[1]
h_kv = kv.shape[2]
bytes_per_token = kv.shape[3]
num_rows = num_blocks * page_block_size
d_qk = tp.d_qk
qk_nope = d_qk - 64  # rope=64
rope_dim = 64

print(f"\nnum_blocks={num_blocks}, page_block_size={page_block_size}, h_kv={h_kv}, bytes_per_token={bytes_per_token}")
print(f"d_qk={d_qk}, qk_nope={qk_nope}, rope_dim={rope_dim}")

# Get dequantized KV in bf16 to build packed data
# blocked_k is in bf16, shape [num_blocks, page_block_size, h_kv, d_qk]
kv_bf16 = scope.blocked_k  # already dequantized
print(f"kv_bf16 shape: {kv_bf16.shape}, dtype: {kv_bf16.dtype}")

# Build packed data (1 bit per dim identity calib)
nope_bytes = qk_nope // 8
packed_row_bytes = nope_bytes + 128  # rope 64 bf16 = 128 bytes

kv_flat = kv_bf16.reshape(num_rows, h_kv, d_qk)[:, 0, :]  # [num_rows, d_qk]

packed_kcache = torch.zeros(num_rows, packed_row_bytes, dtype=torch.uint8, device=device)

# Encode nope: 1 bit per dim
for d in range(qk_nope):
    byte_idx = d // 8
    bit_idx = d % 8
    bit_vals = (kv_flat[:, d] > 0).to(torch.uint8)
    packed_kcache[:, byte_idx] |= bit_vals << bit_idx

# Encode rope: 64 bf16 at end
rope_bf16 = kv_flat[:, qk_nope:qk_nope+rope_dim]
rope_bytes = rope_bf16.view(torch.uint8).reshape(num_rows, -1)
packed_kcache[:, nope_bytes:nope_bytes + rope_dim * 2] = rope_bytes

scale_kcache = torch.ones(num_rows, qk_nope, dtype=torch.float32, device=device)
R_matrix = torch.eye(qk_nope, dtype=torch.float32, device=device)
zero_point = torch.zeros(qk_nope, dtype=torch.float32, device=device)
row_bits = qk_nope
dim_of_bit = torch.arange(qk_nope, dtype=torch.int32, device=device)
bitpos_in_dim = torch.zeros(row_bits, dtype=torch.int32, device=device)

print(f"\npacked_kcache shape: {packed_kcache.shape}")
print(f"scale_kcache shape: {scale_kcache.shape}")
print(f"R_matrix shape: {R_matrix.shape}")

out_packed, lse_packed, _, _ = fmc.sparse_decode_fwd(
    q, kv, indices, topk_length, attn_sink,
    None, None, None, None, None,
    tp.d_v, tp.sm_scale,
    packed_kcache=packed_kcache,
    scale_kcache=scale_kcache,
    R_matrix=R_matrix,
    zero_point=zero_point,
    dim_of_bit=dim_of_bit,
    bitpos_in_dim=bitpos_in_dim,
)

out_dense_f = out_dense.to(torch.float32)
out_packed_f = out_packed.to(torch.float32)
lse_dense_f = lse_dense.to(torch.float32)
lse_packed_f = lse_packed.to(torch.float32)

out_max_diff = (out_dense_f - out_packed_f).abs().max().item()
lse_max_diff = (lse_dense_f - lse_packed_f).abs().max().item()

print(f"\nout_max_diff = {out_max_diff:.6e}")
print(f"lse_max_diff = {lse_max_diff:.6e}")
print(f"out_allclose (atol=1e-2) = {torch.allclose(out_dense_f, out_packed_f, atol=1e-2, rtol=1e-2)}")
print(f"lse_allclose (atol=1e-2) = {torch.allclose(lse_dense_f, lse_packed_f, atol=1e-2, rtol=1e-2)}")
