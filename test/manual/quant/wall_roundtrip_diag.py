"""Wall-mode end-to-end round-trip diagnosis (T3 score=0 root cause).

Runs in the container's docker env (CUDA + Triton). Loads /tmp/calib_dsv4.pt
and exercises the EXACT wall path used by the server:

    BF16 input  --rotated_store_to_packed-->  packed bytes
                                              (Triton bitpack on GPU)

    packed  --rotated_load_to_fp8_layout-->  FP8 slot bytes + UE8M0 scale
            (Triton bitunpack + dequant + matmul + quant_to_fp8)

    FP8 slot  --decode-->  reconstructed bf16

Then compares reconstructed against original to bound the per-step quant
error. If recon is wildly off, the pipeline is broken (pipeline bug, not
calib quality).

Usage (in container):
    cd /workspace/sglang-bytedance/python && \
      python3 ../test/manual/quant/wall_roundtrip_diag.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

from sglang.srt.layers.quantization.rotated_kv_quant import RotatedQuantizerConfig
from sglang.srt.mem_cache.rotated_quant_dsv4_memory_pool import (
    load_rotated_quant_dsv4_calibration,
)
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    rotated_store_to_packed,
    rotated_load_to_fp8_layout,
    packed_bytes_per_token,
    _MLA_NOPE_DIM,
    _MLA_TILE_SIZE,
    _MLA_SLOT_BYTES,
    _MLA_SCALES_PER_TOKEN,
)


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda:0")

    print("Loading /tmp/calib_dsv4.pt ...")
    cfgs = load_rotated_quant_dsv4_calibration(
        path="/tmp/calib_dsv4.pt",
        layer_num=43,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        compression_ratios=[0, 4, 128, 4],
    )
    cfg = cfgs[0]  # use layer 0
    print(f"layer 0: row_bits={cfg.row_bits} row_bytes={cfg.row_bytes}")

    # Make a paged cache: page_size=64, 4 pages = 256 slots
    page_size = 64
    num_pages = 4
    bpt = packed_bytes_per_token(cfg.row_bytes)
    cache = torch.zeros(
        (num_pages, bpt * page_size), dtype=torch.uint8, device=device
    )

    # Synthesize input: N=128 tokens of [N, 512] BF16 cat(nope, rope)
    torch.manual_seed(0)
    N = 128
    input_bf16 = torch.randn(
        N, _MLA_NOPE_DIM + _MLA_TILE_SIZE, dtype=torch.bfloat16, device=device
    )
    indices = torch.arange(N, dtype=torch.int32, device=device)

    # Store
    rotated_store_to_packed(
        input_bf16, cache, indices, page_size=page_size, cfg=cfg,
    )
    print("[ok] store done")

    # Load: gather the same tokens back through wall layout
    out_slot = torch.empty(
        (N, _MLA_SLOT_BYTES), dtype=torch.uint8, device=device
    )
    out_scale = torch.empty(
        (N, _MLA_SCALES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    rotated_load_to_fp8_layout(
        cache, indices, out_slot, out_scale,
        page_size=page_size, cfg=cfg,
    )
    print("[ok] load done")

    # Decode FP8 slot back to bf16: nope_fp8(448) + rope_bf16(128)
    fp8_dtype = torch.float8_e4m3fn
    nope_fp8 = out_slot[:, :_MLA_NOPE_DIM].view(fp8_dtype)
    nope_recon_per_tile = nope_fp8.to(torch.float32).reshape(N, 7, 64)
    # ue8m0 -> 2^(ue8m0 - 127)
    ue8m0 = out_scale[:, :7].to(torch.float32)
    tile_scale = torch.pow(2.0, ue8m0 - 127.0).unsqueeze(-1)
    nope_recon = (nope_recon_per_tile * tile_scale).reshape(N, _MLA_NOPE_DIM)
    rope_recon = (
        out_slot[:, _MLA_NOPE_DIM:].contiguous()
          .view(torch.bfloat16).reshape(N, _MLA_TILE_SIZE).to(torch.float32)
    )
    full_recon = torch.cat([nope_recon, rope_recon], dim=-1)

    src = input_bf16.to(torch.float32)

    # Per-segment cosine sim & max abs diff
    nope_src = src[:, :_MLA_NOPE_DIM]
    rope_src = src[:, _MLA_NOPE_DIM:]

    def _cos(a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

    print(f"nope cos_sim = {_cos(nope_src, nope_recon):.4f}")
    print(f"rope cos_sim = {_cos(rope_src, rope_recon):.4f}")
    print(f"nope max|err| = {(nope_src - nope_recon).abs().max().item():.4f}")
    print(f"rope max|err| = {(rope_src - rope_recon).abs().max().item():.4f}")
    print(f"rope mean|err| = {(rope_src - rope_recon).abs().mean().item():.4f}")
    print(f"nope mean|err| = {(nope_src - nope_recon).abs().mean().item():.4f}")
    print(f"src     [:8]: {src[0, :8].tolist()}")
    print(f"recon   [:8]: {full_recon[0, :8].tolist()}")

    # Sanity: rope must round-trip exactly (bytes preserved)
    rope_diff = (rope_src - rope_recon).abs().max().item()
    print(
        "[rope-byte-roundtrip]",
        "PASS" if rope_diff < 1e-2 else f"FAIL diff={rope_diff:.6f}",
    )


if __name__ == "__main__":
    main()
