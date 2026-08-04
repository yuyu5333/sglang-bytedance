"""Correctness test for the CUDA kvbit_mla_decode_stage1 kernel vs the Triton kernel.

Both kernels must produce the same (att_out, att_lse) for the same inputs (up
to fp noise from op-order differences), since the host-side _reduce_splits +
out@R are identical downstream. This test builds a kvbit 4bit store, runs both
stage-1 kernels, and compares the final merged output.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")

CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
@pytest.mark.parametrize("B,H,S,max_splits", [(1, 1, 64, 1), (1, 1, 256, 4), (2, 2, 128, 2), (1, 1, 3000, 47)])
def test_cuda_stage1_matches_triton(B, H, S, max_splits):
    import sys
    sys.path.insert(0, "/sgl-workspace")
    from kvbit.triton_kernels import mla_decode_fwd as triton_mla_decode_fwd
    from kvbit.store import encode_kv_rows
    from kvbit.rotation import build_hadamard
    from kvbit.layout import packed_row_bytes
    from sglang.jit_kernel.kvbit_mla_decode import kvbit_mla_decode_stage1

    dev = "cuda"
    Dn, Dr, gs, bits = 512, 64, 64, 4
    sm_scale = 1.0 / (Dn + Dr) ** 0.5
    torch.manual_seed(42)
    R = build_hadamard(Dn, dtype=torch.bfloat16, device=dev)
    Rf = R.float()

    kn = (torch.randn(1, S, Dn) * 3.0).to(torch.bfloat16).cuda()
    kr = (torch.randn(1, S, Dr) * 3.0).to(torch.bfloat16).cuda()
    qn = (torch.randn(1, H, Dn) * 3.0).to(torch.bfloat16).cuda()
    qp = (torch.randn(1, H, Dr) * 3.0).to(torch.bfloat16).cuda()
    # broadcast KV to B sequences (same KV for all batches in this test)
    kn = kn.expand(B, S, Dn).contiguous()
    kr = kr.expand(B, S, Dr).contiguous()
    qnf = (qn.float() @ Rf).to(torch.bfloat16)

    rb = packed_row_bytes(dim=Dn, bits=bits, rope_dim=0, group_size=gs)
    packed = torch.zeros(S, rb, dtype=torch.uint8, device=dev)
    rope_buf = torch.zeros(S, 1, Dr, dtype=torch.bfloat16, device=dev)
    for s in range(S):
        e = encode_kv_rows(kn[0, s].unsqueeze(0), bits=bits, nope_dim=Dn, rope_dim=0,
                           group_size=gs, rotation=None, bf16_rotate=True, rotate=True)
        packed[s] = e.data[0]
        rope_buf[s, 0] = kr[0, s]
    # page_table: each batch maps to the same S slots
    pt = torch.arange(S, dtype=torch.int32, device=dev).unsqueeze(0).repeat(B, 1)
    csl = torch.full((B,), S, dtype=torch.int32, device=dev)
    nks = torch.full((B,), max_splits, dtype=torch.int32, device=dev)

    # --- Triton path (full mla_decode_fwd includes reduce + returns bf16) ---
    out_triton = triton_mla_decode_fwd(
        qnf, qp, packed, rope_buf, pt, csl,
        num_kv_splits=nks, max_kv_splits=max_splits, sm_scale=sm_scale,
        kv_lora_rank=Dn, qk_rope_head_dim=Dr, bits=bits, group_size=gs,
    )

    # --- CUDA stage-1 + host reduce ---
    att_out = torch.zeros(B, H, max_splits, Dn, dtype=torch.float32, device=dev)
    att_lse = torch.full((B, H, max_splits), float("-inf"), dtype=torch.float32, device=dev)
    kvbit_mla_decode_stage1(
        qnf, qp, packed, rope_buf, sm_scale, pt, csl, nks,
        att_out, att_lse, max_splits,
    )
    # replicate _reduce_splits
    max_val = att_lse.max(dim=-1).values
    is_valid = att_lse > -float("inf")
    weights = torch.exp(att_lse - max_val.unsqueeze(-1))
    weights = torch.where(is_valid, weights, torch.zeros_like(weights))
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    out_cuda = (att_out * weights.unsqueeze(-1)).sum(dim=-2).to(q_nope_dtype := qnf.dtype)

    assert out_cuda.shape == out_triton.shape, (out_cuda.shape, out_triton.shape)
    assert torch.isfinite(out_cuda).all(), "CUDA output non-finite"
    cos = torch.nn.functional.cosine_similarity(
        out_cuda.float().flatten(), out_triton.float().flatten(), dim=0
    ).item()
    maxdiff = (out_cuda.float() - out_triton.float()).abs().max().item()
    # both kernels compute the same math; allow fp noise from op order.
    assert cos > 0.97, f"B{B}H{H}S{S}s{max_splits}: cos {cos} < 0.97, maxdiff {maxdiff}"
    assert maxdiff < 3.0, f"B{B}H{H}S{S}s{max_splits}: maxdiff {maxdiff} >= 3.0"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
