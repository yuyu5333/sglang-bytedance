"""Single-GEMM numerical comparison for cutlass_mxfp4a8_moe_mm.

Goal: locate the correctness bug in the MXFP4A8 grouped GEMM by comparing the
kernel output against a bf16-dequant golden reference, and empirically decide
which nibble packing order the kernel expects (NATURAL vs order_map
[0,2,4,6,1,3,5,7]).

The mxfp4a8 kernel reuses the int4a8 mainloop DirectConvert path, so the byte
packing convention SHOULD be identical to the int4a8 test
(``pack_int4_values_to_int8`` = natural order).  We verify that directly here.

Run on remote (SM90 required):
    cd /sgl-workspace/sglang-bytedance/python/sglang/kernels/aot
    PYTHONPATH=... python tests/test_cutlass_mxfp4a8_moe_mm.py
"""

import sys

import torch
from sgl_kernel import cutlass_mxfp4a8_moe_mm

from sglang.kernels.ops.quantization.per_tensor_quant_fp8 import per_tensor_quant_fp8

# E2M1 magnitude table indexed by the 3-bit magnitude index (exp<<1 | mant).
# Matches mxfp4_numeric_conversion.hpp: idx 0..7 -> {0,.5,1,1.5,2,3,4,6}.
E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

CHUNK = 32  # MXFP4 block size in K
ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]


def _per_tensor_quant_fp8(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    assert x.is_contiguous(), "`x` is not contiguous"
    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(1, device=x.device, dtype=torch.float32)
    per_tensor_quant_fp8(x, x_q, x_s, is_static=False)
    return x_q, x_s


def make_e2m1_weights(num_experts, n, k, device):
    """Return (nibble_codes[int8], values[float32]) both shape [E,N,K].

    nibble_codes in [0,16): bit3=sign, bits2-0=magnitude index.
    values = decoded signed float on the E2M1 grid.
    """
    idx = torch.randint(0, 8, (num_experts, n, k), device=device)  # magnitude idx
    sign = torch.randint(0, 2, (num_experts, n, k), device=device)  # 0/1
    mag = E2M1_MAG.to(device)[idx]
    values = torch.where(sign.bool(), -mag, mag).to(torch.float32)
    codes = (sign << 3) | idx  # int8 nibble code
    return codes.to(torch.int8), values


def pack_nibbles_natural(codes: torch.Tensor) -> torch.Tensor:
    """codes [E,N,K] int8 -> packed [E,N,K//2] int8.

    Natural order (same as int4a8 test pack_int4_values_to_int8):
    low nibble = even K index (2i), high nibble = odd K index (2i+1).
    """
    codes = codes.to(torch.int8)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return ((high << 4) | (low & 0x0F)).to(torch.int8)


def pack_nibbles_ordermap(codes: torch.Tensor) -> torch.Tensor:
    """Apply order_map [0,2,4,6,1,3,5,7] within each group of 8 along K,
    then pack natural. Mirrors the production int4fp8 reorder packing."""
    e, n, k = codes.shape
    assert k % 8 == 0
    reordered = codes.reshape(e, n, k // 8, 8)[..., ORDER_MAP].reshape(e, n, k)
    return pack_nibbles_natural(reordered)


def interleave_scales(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, K//CHUNK] -> [E, K//(CHUNK*4), N*4] (4-wide interleave)."""
    s0, s1, s2 = scales.shape
    alignment = 4 if s2 % 4 == 0 else 1
    si = scales.reshape(s0, s1, s2 // alignment, alignment)
    si = si.permute(0, 2, 1, 3)
    si = si.reshape(s0, s2 // alignment, s1 * alignment)
    return si.contiguous()


def ref_grouped_gemm(a_q, a_scale, w_values, w_scale, num_experts, sel):
    """Golden: dequant w (block=32) then matmul in fp32.

    w_values [E,N,K] fp32 (E2M1 grid), w_scale [E,N,K//CHUNK] fp32.
    """
    dtype = torch.bfloat16
    m = a_q.shape[0]
    n = w_values.shape[1]
    c_ref = torch.zeros((m, n), dtype=dtype, device=a_q.device)
    for i in range(num_experts):
        tok = torch.where(sel == i)[0]
        if len(tok) == 0:
            continue
        a = a_q[tok].to(torch.float32)
        scale_rep = w_scale[i].repeat_interleave(CHUNK, dim=1).to(torch.float32)
        w = w_values[i] * scale_rep  # [N,K]
        c = torch.matmul(a, w.t()) * a_scale
        c_ref[tok] = c.to(dtype)
    return c_ref


def run_case(pack_fn, label, num_experts, m, k, n, device, seed=0):
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    a = torch.randn(m, k, dtype=dtype, device=device)
    codes, w_values = make_e2m1_weights(num_experts, n, k, device)

    # E8M0 power-of-2 scale, block=32 along K -> [E,N,K//CHUNK]
    exps = torch.randint(-4, 3, (num_experts, n, k // CHUNK), device=device)
    w_scale = (2.0 ** exps.to(torch.float32))  # exact powers of two

    b_packed = pack_fn(codes).view(num_experts, n, k // 2).contiguous()
    b_scale = interleave_scales(w_scale.to(torch.bfloat16)).contiguous()

    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides

    a_q, a_scale = _per_tensor_quant_fp8(a)

    c = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    cutlass_mxfp4a8_moe_mm(
        c, a_q, b_packed, a_scale, b_scale,
        expert_offsets[:-1], problem_sizes,
        a_strides, b_strides, c_strides, s_strides,
        CHUNK, 8,
    )
    c = c.to(dtype)

    sel = torch.zeros((m,), dtype=torch.long, device=device)
    c_ref = ref_grouped_gemm(a_q, a_scale.item(), w_values, w_scale, num_experts, sel)

    max_abs = torch.max(torch.abs(c.float() - c_ref.float())).item()
    mean_abs = torch.mean(torch.abs(c.float() - c_ref.float())).item()
    ref_scale = c_ref.float().abs().mean().item() + 1e-9
    print(
        f"[{label:9s}] m={m} k={k} n={n}  max_abs={max_abs:.4f} "
        f"mean_abs={mean_abs:.4f} rel_mean={mean_abs/ref_scale:.4f}"
    )
    return mean_abs / ref_scale


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    device = "cuda"
    # small shapes; k must be multiple of 128 (chunk*4) for 4-wide interleave
    for (m, k, n) in [(4, 256, 512), (8, 512, 1024), (16, 1024, 2048)]:
        rel_nat = run_case(pack_nibbles_natural, "natural", 1, m, k, n, device)
        rel_ord = run_case(pack_nibbles_ordermap, "order_map", 1, m, k, n, device)
        winner = "natural" if rel_nat < rel_ord else "order_map"
        print(f"  -> winner: {winner}  (natural={rel_nat:.4f}, order_map={rel_ord:.4f})\n")


if __name__ == "__main__":
    main()
