"""Wall-mode round-trip diag with REAL calib + REAL KV dump samples.

Goal: rule out calib mismatch as the root cause of T3 wall token salad.
This script uses /data00/calib_dsv4_real.pt (calib derived from real
forward-pass KV dump) and feeds REAL post-norm-pre-rope nope samples
from the same dump into the wall store/load pipeline. If cos_sim is
high here, then the calib is fine and the bug must be elsewhere
(prologue layout, FP8 cast, or interaction with FlashMLA at runtime).

Usage (in container):
    cd /workspace/sglang-bytedance/python && \
      python3 ../test/manual/quant/wall_real_calib_diag.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

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


def _cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda:0")

    calib_path = os.environ.get("CALIB_PATH", "/data00/calib_dsv4_real.pt")
    dump_path = os.environ.get(
        "KV_DUMP_PATH", "/data00/kv_dump_dsv4_rank0.pt"
    )
    print(f"calib  : {calib_path}")
    print(f"kv dump: {dump_path}")

    cfgs = load_rotated_quant_dsv4_calibration(
        path=calib_path,
        layer_num=43,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        compression_ratios=[0, 4, 128, 4],
    )
    print(f"loaded {len(cfgs)} layer configs")

    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    sample_layers = [0, 5, 20, 42]
    page_size = 64

    for lid in sample_layers:
        cfg = cfgs[lid]
        print(f"\n========== layer {lid} ==========")
        print(f"  row_bits={int(cfg.bits.sum())} row_bytes={cfg.row_bytes} "
              f"b_mean={float(cfg.bits.float().mean()):.3f}")

        # Real KV samples: dump[lid]["nope"] is [N, 1, 448] post-norm bf16/f32.
        nope_real = dump[lid]["nope"].squeeze(1).to(torch.bfloat16)
        rope_real = dump[lid]["rope"].squeeze(1).to(torch.bfloat16)
        N = min(256, nope_real.shape[0])
        nope_in = nope_real[:N].contiguous().to(device)
        rope_in = rope_real[:N].contiguous().to(device)
        # Build wall packer input: [N, 512] cat(nope, rope)
        wall_in = torch.cat([nope_in, rope_in], dim=-1).contiguous()

        # Build packed paged cache for this batch
        num_pages = (N + page_size - 1) // page_size
        bpt = packed_bytes_per_token(cfg.row_bytes)
        cache = torch.zeros(
            (num_pages, bpt * page_size), dtype=torch.uint8, device=device
        )
        indices = torch.arange(N, dtype=torch.int32, device=device)

        rotated_store_to_packed(
            wall_in, cache, indices, page_size=page_size, cfg=cfg,
        )

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

        # Decode FP8 slot back to bf16 for comparison
        fp8 = torch.float8_e4m3fn
        nope_fp8 = out_slot[:, :_MLA_NOPE_DIM].view(fp8)
        ue8m0 = out_scale[:, :7].to(torch.float32)
        tile_scale = torch.pow(2.0, ue8m0 - 127.0).unsqueeze(-1)
        nope_recon = (
            nope_fp8.to(torch.float32).reshape(N, 7, 64) * tile_scale
        ).reshape(N, _MLA_NOPE_DIM)
        rope_recon = (
            out_slot[:, _MLA_NOPE_DIM:].contiguous()
              .view(torch.bfloat16).reshape(N, _MLA_TILE_SIZE)
              .to(torch.float32)
        )

        nope_src = nope_in.to(torch.float32)
        rope_src = rope_in.to(torch.float32)

        cs_n = _cos(nope_src, nope_recon)
        cs_r = _cos(rope_src, rope_recon)
        max_n = (nope_src - nope_recon).abs().max().item()
        max_r = (rope_src - rope_recon).abs().max().item()
        mean_n = (nope_src - nope_recon).abs().mean().item()
        mean_r = (rope_src - rope_recon).abs().mean().item()
        # source magnitudes
        nope_std = nope_src.std().item()
        nope_max_abs = nope_src.abs().max().item()

        print(f"  nope: cos={cs_n:.4f} max|err|={max_n:.4f} "
              f"mean|err|={mean_n:.4f} src_std={nope_std:.4f} "
              f"src_maxabs={nope_max_abs:.4f}")
        print(f"  rope: cos={cs_r:.4f} max|err|={max_r:.4f} "
              f"mean|err|={mean_r:.4f}")

        # 取 codes 看分位是否饱和
        from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
            triton_bitunpack_rowwise,
        )
        cache_flat = cache.view(-1, bpt)[:N]
        packed_nope = cache_flat[:, :cfg.row_bytes].contiguous()
        codes = triton_bitunpack_rowwise(packed_nope, cfg.bits.to(device))
        levels = ((1 << cfg.bits.to(device).to(torch.int64)) - 1)
        # 饱和率：codes==0 or codes==levels
        sat_lo = (codes == 0).float().mean().item()
        sat_hi = (codes == levels.unsqueeze(0)).float().mean().item()
        print(f"  codes: sat_lo={sat_lo:.3f} sat_hi={sat_hi:.3f} "
              f"min={codes.min().item()} max={codes.max().item()}")


if __name__ == "__main__":
    main()
