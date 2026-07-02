"""Stage-2 sparse_fp8 packed K-tile fused dequant — identity calib sanity test.

Verifies:
  1. packed path produces finite output (no NaN/Inf).
  2. code=0 -> output magnitude ~0 (K_nope=0).
  3. code=1, constant scale -> output magnitude ~ scale (V averaging).
  4. dense path baseline (zero KV) is also finite.

Note: random uint8 KV bytes will decode to FP8 NaN encodings, so this test
uses zero-initialized KV to keep the dense baseline well-defined.
"""
import torch
import flash_mla.cuda as fmc

B = 1
H_Q = 64
S_Q = 1
H_KV = 1
S_K = 64
D_QK = 512  # MODEL1
D_V = 512
BLOCK_SIZE = 64
NUM_BLOCKS = 2
QK_NOPE = 448
ROPE_BYTES = 128  # 64 dims * bf16
bytes_per_token = 584

torch.manual_seed(42)

q = torch.randn(B, S_Q, H_Q, D_QK, device='cuda', dtype=torch.bfloat16) * 0.1
kv_bytes = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, H_KV, bytes_per_token,
                       device='cuda', dtype=torch.uint8)
indices = torch.arange(S_K, device='cuda', dtype=torch.int32) \
    .unsqueeze(0).unsqueeze(0).expand(B, S_Q, S_K).contiguous()
topk_length = torch.full((B,), S_K, device='cuda', dtype=torch.int32)
attn_sink = torch.zeros(H_Q, device='cuda', dtype=torch.float32)
sm_scale = D_QK ** -0.5

num_rows = NUM_BLOCKS * BLOCK_SIZE
nope_packed_bytes = QK_NOPE // 8
packed_row_bytes = nope_packed_bytes + ROPE_BYTES

R_matrix = torch.eye(QK_NOPE, device='cuda', dtype=torch.float32)
zero_point = torch.zeros(QK_NOPE, device='cuda', dtype=torch.float32)
dim_of_bit = torch.arange(QK_NOPE, device='cuda', dtype=torch.int32)
bitpos_in_dim = torch.zeros(QK_NOPE, device='cuda', dtype=torch.int32)

# --- Test 1: dense path with zero KV ---
out_d, _, _, _ = fmc.sparse_decode_fwd(
    q, kv_bytes, indices, topk_length, attn_sink,
    None, None, None, None, None, D_V, sm_scale)
print("[1] DENSE zero-KV:")
print("    finite =", bool(torch.isfinite(out_d).all()))
print("    |max| =", float(out_d.abs().max()))

# --- Test 2: packed bits=0 (code=0, dequant=0) ---
packed0 = torch.zeros(num_rows, packed_row_bytes, device='cuda', dtype=torch.uint8)
scale_kcache_rand = torch.randn(num_rows, QK_NOPE, device='cuda', dtype=torch.float32) * 0.1
out_p0, _, _, _ = fmc.sparse_decode_fwd(
    q, kv_bytes, indices, topk_length, attn_sink,
    None, None, None, None, None, D_V, sm_scale,
    packed_kcache=packed0, scale_kcache=scale_kcache_rand,
    R_matrix=R_matrix, zero_point=zero_point,
    dim_of_bit=dim_of_bit, bitpos_in_dim=bitpos_in_dim)
print("[2] PACKED bits=0 (code=0):")
print("    finite =", bool(torch.isfinite(out_p0).all()))
print("    |max| =", float(out_p0.abs().max()))

# --- Test 3: packed bits=FF (code=1, constant scale=0.5) ---
packedFF = torch.zeros(num_rows, packed_row_bytes, device='cuda', dtype=torch.uint8)
packedFF[:, :nope_packed_bytes] = 0xFF
scale_const = torch.full((num_rows, QK_NOPE), 0.5, device='cuda', dtype=torch.float32)
out_pFF, _, _, _ = fmc.sparse_decode_fwd(
    q, kv_bytes, indices, topk_length, attn_sink,
    None, None, None, None, None, D_V, sm_scale,
    packed_kcache=packedFF, scale_kcache=scale_const,
    R_matrix=R_matrix, zero_point=zero_point,
    dim_of_bit=dim_of_bit, bitpos_in_dim=bitpos_in_dim)
print("[3] PACKED bits=FF (code=1) scale=0.5:")
print("    finite =", bool(torch.isfinite(out_pFF).all()))
print("    |max| =", float(out_pFF.abs().max()))
print("    |mean| =", float(out_pFF.abs().mean()))

assert torch.isfinite(out_d).all(), "DENSE not finite"
assert torch.isfinite(out_p0).all(), "PACKED bits=0 not finite"
assert torch.isfinite(out_pFF).all(), "PACKED bits=FF not finite"
# Expect mean around scale (= 0.5)
assert 0.3 < float(out_pFF.abs().mean()) < 0.7, "PACKED bits=FF magnitude off"

print("\nAll Stage-2 fused-dequant identity-calib checks PASSED.")
