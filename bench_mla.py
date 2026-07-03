import torch
import time
import sys
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

b = 32
s_q = 1
h_q = 128
h_kv = 1
d = 576
dv = 512
block_size = 64
mean_sk = 4096
causal = False
varlen = False
torch_dtype = torch.float8_e4m3fn

device = torch.device("cuda:0")
torch.set_default_device(device)
torch.manual_seed(0)

use_fp8 = True
cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
total_seqlens = cache_seqlens.sum().item()
max_seqlen = cache_seqlens.max().item()
max_seqlen_pad = (max_seqlen + 255) // 256 * 256

q = torch.randn(b, s_q, h_q, d)
block_table = torch.arange(
    b * max_seqlen_pad // block_size, dtype=torch.int32
).view(b, max_seqlen_pad // block_size)
blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
blocked_v = blocked_k[..., :dv]

tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens, s_q * h_q // h_kv, h_kv, is_fp8_kvcache=use_fp8
)

fp8_dtype = torch.float8_e4m3fn
descale_q = torch.ones((1), dtype=torch.float32)
descale_k = torch.ones((1), dtype=torch.float32)

q = q.to(fp8_dtype)
blocked_k = blocked_k.to(fp8_dtype)
blocked_v = blocked_v.to(fp8_dtype)

def run_kernel():
    return flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        dv,
        tile_scheduler_metadata,
        num_splits,
        causal=causal,
        descale_q=descale_q,
        descale_k=descale_k,
    )

for _ in range(5):
    run_kernel()
torch.cuda.synchronize()

n_iter = 100
t0 = time.time()
for _ in range(n_iter):
    run_kernel()
torch.cuda.synchronize()
t1 = time.time()

elapsed = t1 - t0
ms_per_iter = (elapsed / n_iter) * 1000
print(f"avg: {ms_per_iter:.2f} ms/iter, {n_iter/elapsed:.1f} iter/s")
print(f"total tokens processed: {b * s_q * n_iter}")
print("Done.")
