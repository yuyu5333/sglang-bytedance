"""[M3.c.4 Stage-5] Shadow-writer bisect: Triton chain vs native CUDA.

Decisive GPU experiment to locate the Stage-5 token-salad root cause that
was isolated to the Triton "write shadow" chain (drop_packed=0) vs the
native CUDA writer (drop_packed=1).

Two independent writers should produce **byte-identical** FP8+UE8M0 shadow
pages for the same BF16 ``kv`` input:

  * Writer N (native, coherent):
        fused_k_norm_rope_flashmla(kv, kv_weight, eps, freqs_cis,
                                   positions, out_loc=loc, kvcache=shadowN)
  * Writer T (triton, salad):
        cat = triton_fused_norm_rope(kv, kv_weight, eps, freqs_cis, positions)
        rotated_dequant_to_fp8_layout(cat[:,:448], cat[:,448:],
                                      out_slot, out_scale)
        triton_scatter_tokens_to_shadow(out_slot, out_scale, loc, shadowT)

The test bisects the chain:

  1. ``kv`` contiguous vs strided slice (qkv_a[..., q_lora_rank:] layout) →
     does triton_fused_norm_rope honour the row stride?
  2. cat (triton) vs cat_ref (pytorch fp32 reference) → norm+rope numeric.
  3. shadow bytes N vs T → end-to-end byte diff.

Run inside the container:
    cd /workspace/sglang-bytedance/python && \
    PYTHONPATH=/workspace/sglang-bytedance/python:$PYTHONPATH \
    python ../test/manual/quant/test_shadow_writer_bisect.py
"""

from __future__ import annotations

import torch


def _ref_cat(kv_bf16, kv_weight, eps, freqs_cis, positions, nope_dim=448):
    """PyTorch fp32 reference for fused norm+rope (matches CPU fallback)."""
    kv_f = kv_bf16.to(torch.float32)
    w = kv_weight.to(torch.float32)
    var = kv_f.pow(2).mean(dim=-1, keepdim=True)
    kv_norm = kv_f * torch.rsqrt(var + eps) * w
    nope = kv_norm[..., :nope_dim]
    rope = kv_norm[..., nope_dim:]
    rope_dim = rope.shape[-1]
    rope_c = torch.view_as_complex(
        rope.reshape(*rope.shape[:-1], rope_dim // 2, 2).contiguous()
    )
    fr = freqs_cis.index_select(0, positions.to(torch.long))
    rope_rot = torch.view_as_real(rope_c * fr).reshape(*rope.shape[:-1], rope_dim)
    return torch.cat([nope, rope_rot], dim=-1).to(torch.bfloat16)


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    dev = "cuda"
    torch.manual_seed(0)

    N = 17
    HEAD = 512
    NOPE = 448
    ROPE = 64
    q_lora_rank = 1024
    eps = 1e-6
    page_size = 64
    max_pos = 4096

    # --- inputs ---
    kv_weight = torch.randn(HEAD, device=dev, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (N,), device=dev, dtype=torch.int32)

    # freqs_cis complex64 [max_pos, 32]
    ang = torch.randn(max_pos, ROPE // 2, device=dev)
    freqs_cis = torch.polar(torch.ones_like(ang), ang).to(torch.complex64)

    # contiguous kv [N, 512]
    kv_contig = torch.randn(N, HEAD, device=dev, dtype=torch.bfloat16)

    # strided kv: emulate qkv_a[..., q_lora_rank:] (row stride = 1024+512)
    qkv_a = torch.randn(N, q_lora_rank + HEAD, device=dev, dtype=torch.bfloat16)
    qkv_a[:, q_lora_rank:] = kv_contig  # same values, different layout
    kv_strided = qkv_a[:, q_lora_rank:]
    assert not kv_strided.is_contiguous(), "expected strided slice"
    assert kv_strided.stride(0) == q_lora_rank + HEAD

    from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
        triton_fused_norm_rope,
        rotated_dequant_to_fp8_layout,
        triton_scatter_tokens_to_shadow,
    )

    # ---- step 1+2: cat numeric (contig vs strided vs ref) ----
    cat_ref = _ref_cat(kv_contig, kv_weight, eps, freqs_cis, positions)
    cat_contig = triton_fused_norm_rope(kv_contig, kv_weight, eps, freqs_cis, positions)
    cat_strided = triton_fused_norm_rope(kv_strided, kv_weight, eps, freqs_cis, positions)

    def cos(a, b):
        a = a.to(torch.float32).reshape(-1)
        b = b.to(torch.float32).reshape(-1)
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    print("=== STEP 1+2: cat (norm+rope) ===")
    print(f"cos(cat_contig, cat_ref)   = {cos(cat_contig, cat_ref):.6f}")
    print(f"cos(cat_strided, cat_ref)  = {cos(cat_strided, cat_ref):.6f}")
    print(f"cos(cat_strided, cat_contig)= {cos(cat_strided, cat_contig):.6f}")
    print(f"max|cat_strided - cat_contig| = "
          f"{(cat_strided.float()-cat_contig.float()).abs().max().item():.4f}")

    # ---- step 3: shadow byte diff (use contiguous kv so cat is correct) ----
    from sglang.jit_kernel.deepseek_v4 import fused_k_norm_rope_flashmla

    bytes_per_page = ((584 * page_size + 575) // 576) * 576
    num_pages = 8
    loc = torch.arange(N, device=dev, dtype=torch.int32)  # slots 0..N-1 in page 0

    shadowN = torch.zeros(num_pages, bytes_per_page, device=dev, dtype=torch.uint8)
    fused_k_norm_rope_flashmla(
        kv=kv_contig, kv_weight=kv_weight, eps=eps, freqs_cis=freqs_cis,
        positions=positions, out_loc=loc, kvcache=shadowN, page_size=page_size,
    )

    shadowT = torch.zeros(num_pages, bytes_per_page, device=dev, dtype=torch.uint8)
    cat = cat_contig  # already triton norm+rope on contiguous kv
    nope = cat[:, :NOPE].contiguous()
    rope = cat[:, NOPE:].contiguous()
    out_slot = torch.zeros(N, 576, device=dev, dtype=torch.uint8)
    out_scale = torch.zeros(N, 8, device=dev, dtype=torch.uint8)
    rotated_dequant_to_fp8_layout(nope, rope, out_slot, out_scale)
    triton_scatter_tokens_to_shadow(out_slot, out_scale, loc, shadowT, page_size)

    print("\n=== STEP 3: shadow bytes (native N vs triton T) ===")
    diff = (shadowN.int() - shadowT.int()).abs()
    nz = (diff != 0)
    print(f"total differing bytes: {int(nz.sum().item())} / {diff.numel()}")
    # break down: value region [0, page_size*576) vs scale region
    val_end = page_size * 576
    page0N = shadowN[0]
    page0T = shadowT[0]
    val_diff = (page0N[:val_end].int() - page0T[:val_end].int()).abs()
    scale_region = page0N[val_end:val_end + page_size * 8]
    scl_diff = (scale_region.int()
                - page0T[val_end:val_end + page_size * 8].int()).abs()
    print(f"page0 value-region differing bytes: {int((val_diff!=0).sum())}")
    print(f"page0 scale-region differing bytes: {int((scl_diff!=0).sum())}")

    # per-token slot 0 detail
    for t in [0, 1, 2]:
        sN = page0N[t * 576:(t + 1) * 576]
        sT = page0T[t * 576:(t + 1) * 576]
        d = (sN.int() - sT.int()).abs()
        nope_d = int((d[:448] != 0).sum())
        rope_d = int((d[448:576] != 0).sum())
        print(f"  token {t}: nope-byte diff={nope_d}/448  rope-byte diff={rope_d}/128")
        if nope_d:
            idx = torch.nonzero(d[:448] != 0).reshape(-1)[:8].tolist()
            print(f"    nope first-diff idx={idx}")
            print(f"    N[{idx}]={sN[:448][idx].tolist()}")
            print(f"    T[{idx}]={sT[:448][idx].tolist()}")
        # scale bytes for this token
        scN = page0N[val_end + t * 8:val_end + t * 8 + 8]
        scT = page0T[val_end + t * 8:val_end + t * 8 + 8]
        print(f"    scaleN={scN.tolist()}  scaleT={scT.tolist()}")


if __name__ == "__main__":
    main()
