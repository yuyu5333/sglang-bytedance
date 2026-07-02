import torch
import flash_mla.cuda as fmc

B = 1
H_Q = 64
S_Q = 1
H_KV = 1
S_K = 64
D_QK = 512
D_V = 512
BLOCK_SIZE = 64
NUM_BLOCKS = 4

q = torch.randn(B, S_Q, H_Q, D_QK, device='cuda', dtype=torch.bfloat16)

kv_bytes = torch.randint(0, 256, (NUM_BLOCKS, BLOCK_SIZE, H_KV, 584),
    device='cuda', dtype=torch.uint8).contiguous()

S_K = 64  # must be multiple of 64 for FlashMLA
indices_data = torch.arange(S_K, device='cuda', dtype=torch.int32)
# indices format: block_idx * page_block_size + rel_idx_in_block
indices = indices_data.unsqueeze(0).unsqueeze(0).expand(B, S_Q, S_K).contiguous()

topk_length = torch.full((B,), S_K, device='cuda', dtype=torch.int32)

attn_sink = torch.full((H_Q,), 50.0, device='cuda', dtype=torch.float32)

sm_scale = D_QK ** -0.5

out_dense, lse_dense, _, _ = fmc.sparse_decode_fwd(
    q, kv_bytes, indices, topk_length, attn_sink,
    None, None, None, None, None,
    D_V, sm_scale)

print("dense path OK")
print("  out shape:", out_dense.shape)
print("  lse shape:", lse_dense.shape)
print("  out max:", float(out_dense.abs().max().item()))
