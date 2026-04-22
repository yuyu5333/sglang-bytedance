import pytest
import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    cutlass_w4_run_moe_ep_preproess_torch,
    cutlass_w4_run_moe_ep_preproess_triton,
)


def _make_topk_ids(
    m: int, topk: int, num_experts: int, include_sentinel: bool
) -> torch.Tensor:
    upper = num_experts + 1 if include_sentinel else num_experts
    return torch.randint(0, upper, (m, topk), device="cuda", dtype=torch.int32)


def _validate_grouping(topk_ids: torch.Tensor, src2dst: torch.Tensor):
    flat_ids = topk_ids.reshape(-1)
    numel = flat_ids.numel()
    sorted_src2dst = torch.sort(src2dst.to(torch.int64)).values
    expected = torch.arange(numel, device=topk_ids.device, dtype=torch.int64)
    assert torch.equal(sorted_src2dst, expected)

    dst2src = torch.argsort(src2dst.to(torch.int64))
    grouped_ids = flat_ids[dst2src]
    expected_grouped_ids = torch.sort(flat_ids, stable=True).values
    assert torch.equal(grouped_ids, expected_grouped_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("m", "topk", "num_experts", "include_sentinel"),
    [
        (32, 2, 8, False),
        (64, 4, 16, False),
        (128, 8, 64, True),
        (257, 8, 256, True),
    ],
)
def test_cutlass_w4a8_moe_ep_preprocess_variants(m, topk, num_experts, include_sentinel):
    torch.manual_seed(0)
    topk_ids = _make_topk_ids(m, topk, num_experts, include_sentinel)

    src2dst_triton = cutlass_w4_run_moe_ep_preproess_triton(topk_ids)
    src2dst_torch_stable = cutlass_w4_run_moe_ep_preproess_torch(topk_ids, stable=True)
    src2dst_torch_unstable = cutlass_w4_run_moe_ep_preproess_torch(
        topk_ids, stable=False
    )

    torch.testing.assert_close(src2dst_torch_stable, src2dst_triton)
    _validate_grouping(topk_ids, src2dst_triton)
    _validate_grouping(topk_ids, src2dst_torch_unstable)
