"""Packed-FP8 path numerical verification with synthetic identity calib.

Construction (so packed path output should bit-exact match dense path):
- R_matrix = identity (qk_nope x qk_nope)
- zero_point = 0
- bits[d] = 1 for all d (1-bit per channel)
- packed_kcache = all 0xFF (every bit = 1, so codes[d] = 1 after bit-unpack)
- scale_kcache[token, d] = fp32(kcache[token, d])  (so dequant -> fp32(kcache))
- rope half = bf16(fp32(kcache_rope))  (round-trip-exact for fp8 e4m3)
"""
import torch
from sgl_kernel import flashmla_ops
from sgl_kernel.flash_mla import get_mla_metadata, flash_mla_with_kvcache

batch_size = 1
seq_len_q = 1
seq_len_k = 64
num_heads_q = 128
num_heads_k = 1
head_dim_k = 576
head_dim_v = 512
page_size = 64
qk_nope = 512

num_pages_per_seq = (seq_len_k + page_size - 1) // page_size
num_pages = num_pages_per_seq * batch_size

torch.manual_seed(42)

q = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim_k, device="cuda", dtype=torch.bfloat16)
k_full = torch.randn(batch_size, seq_len_k, num_heads_k, head_dim_k, device="cuda", dtype=torch.bfloat16)

q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k_full.to(torch.float8_e4m3fn)
descale_q = torch.ones(1, device="cuda", dtype=torch.float32)
descale_k = torch.ones(1, device="cuda", dtype=torch.float32)

k_cache = torch.zeros(num_pages, page_size, num_heads_k, head_dim_k,
                      device="cuda", dtype=torch.float8_e4m3fn)
block_table = torch.zeros(batch_size, num_pages_per_seq, device="cuda", dtype=torch.int32)
cache_seqlens = torch.tensor([seq_len_k] * batch_size, device="cuda", dtype=torch.int32)
for b in range(batch_size):
    for i in range(num_pages_per_seq):
        block_table[b, i] = b * num_pages_per_seq + i
        s = i * page_size
        e = min(s + page_size, seq_len_k)
        k_cache[block_table[b, i], :e - s] = k_fp8[b, s:e]

tile_md, num_splits = get_mla_metadata(
    cache_seqlens, seq_len_q * num_heads_q // num_heads_k,
    num_heads_k, is_fp8_kvcache=True,
)
out_dense, lse_dense = flash_mla_with_kvcache(
    q_fp8, k_cache, block_table, cache_seqlens, head_dim_v,
    tile_md, num_splits,
    softmax_scale=head_dim_k ** (-0.5), causal=True,
    descale_q=descale_q, descale_k=descale_k, is_fp8_kvcache=True,
)

num_rows = num_pages * page_size
kcache_rows = k_cache.view(num_rows, num_heads_k, head_dim_k).select(1, 0)
kcache_nope_fp8 = kcache_rows[:, :qk_nope]
kcache_rope_fp8 = kcache_rows[:, qk_nope:]

scale_kcache = kcache_nope_fp8.to(torch.float32).contiguous()
zero_point = torch.zeros(qk_nope, device="cuda", dtype=torch.float32)
R_matrix = torch.eye(qk_nope, device="cuda", dtype=torch.float32)

nope_bytes = qk_nope // 8
rope_bytes = 64 * 2
packed_row_bytes = nope_bytes + rope_bytes

packed_kcache = torch.zeros(num_rows, packed_row_bytes, device="cuda", dtype=torch.uint8)
packed_kcache[:, :nope_bytes] = 0xFF
kcache_rope_bf16 = kcache_rope_fp8.to(torch.float32).to(torch.bfloat16).contiguous()
packed_kcache[:, nope_bytes:] = kcache_rope_bf16.view(torch.uint8).view(num_rows, rope_bytes)

dim_of_bit = torch.arange(qk_nope, device="cuda", dtype=torch.int32)
bitpos_in_dim = torch.zeros(qk_nope, device="cuda", dtype=torch.int32)

out_packed, lse_packed = torch.ops.sgl_kernel.fwd_kvcache_mla_packed_fp8(
    q_fp8, k_cache, head_dim_v, cache_seqlens, block_table,
    head_dim_k ** (-0.5), True,
    tile_md, num_splits,
    descale_q, descale_k,
    packed_kcache, scale_kcache, R_matrix, zero_point,
    dim_of_bit, bitpos_in_dim,
)

print("out_dense  shape:", out_dense.shape, "dtype:", out_dense.dtype)
print("out_packed shape:", out_packed.shape, "dtype:", out_packed.dtype)

out_equal = torch.equal(out_dense, out_packed)
lse_equal = torch.equal(lse_dense, lse_packed)
print("out bit-exact:", out_equal)
print("lse bit-exact:", lse_equal)

if not out_equal:
    diff = (out_dense.float() - out_packed.float()).abs()
    print("out max diff:", diff.max().item())
    print("out mean diff:", diff.mean().item())
    print("out frac bit-exact:", (out_dense == out_packed).float().mean().item())
if not lse_equal:
    diff = (lse_dense - lse_packed).abs()
    print("lse max diff:", diff.max().item())
    print("lse mean diff:", diff.mean().item())

out_close = torch.allclose(out_dense.float(), out_packed.float(), atol=0.05, rtol=0.05)
lse_close = torch.allclose(lse_dense, lse_packed, atol=0.05, rtol=0.05)
print("out close (atol=rtol=0.05):", out_close)
print("lse close (atol=rtol=0.05):", lse_close)

print("Test PASSED!" if (out_close and lse_close) else "Test FAILED (loose)!")
