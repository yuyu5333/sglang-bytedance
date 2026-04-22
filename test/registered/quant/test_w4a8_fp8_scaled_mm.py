import unittest
from typing import Optional

import torch

from sglang.srt.layers.quantization.w4afp8_kernel import (
    is_w4a8_fp8_linear_supported,
    w4a8_fp8_scaled_mm,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-small-1-gpu")


def _fp8_dtype() -> torch.dtype:
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise unittest.SkipTest("PyTorch FP8 dtype is not available in this runtime.")


def _pack_int4_to_int8(weight_int4: torch.Tensor) -> torch.Tensor:
    assert weight_int4.ndim == 2
    assert weight_int4.dtype == torch.int8
    assert weight_int4.shape[1] % 2 == 0
    assert torch.all(weight_int4 >= -8) and torch.all(weight_int4 <= 7)

    low = (weight_int4[:, 0::2].to(torch.int16) & 0x0F).contiguous()
    high = (weight_int4[:, 1::2].to(torch.int16) & 0x0F).contiguous()
    packed = low | (high << 4)
    return packed.to(torch.int8).contiguous()


def _unpack_int4_from_int8(weight_packed: torch.Tensor) -> torch.Tensor:
    packed = weight_packed.to(torch.int16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).reshape(weight_packed.shape[0], -1).to(
        torch.float32
    )


def _reference_w4a8_fp8_scaled_mm(
    q_input: torch.Tensor,
    weight_packed: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m, k = q_input.shape
    n = weight_packed.shape[0]
    num_groups = k // group_size

    x = q_input.to(torch.float32) * x_scale.reshape(m, 1).to(torch.float32)

    weight_int4 = _unpack_int4_from_int8(weight_packed)
    weight = weight_int4.reshape(n, num_groups, group_size)
    weight = weight * weight_scale.to(torch.float32).unsqueeze(-1)
    weight = weight.reshape(n, k)

    out = x @ weight.t()
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)


class TestW4A8FP8ScaledMM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("This test requires a CUDA device.")
        if not is_w4a8_fp8_linear_supported():
            raise unittest.SkipTest(
                "W4A8-FP8 dense kernel is not supported on this runtime."
            )
        torch.set_default_device("cuda")
        cls.fp8_dtype = _fp8_dtype()

    def _make_inputs(
        self,
        m: int,
        k: int,
        n: int,
        *,
        with_bias: bool,
        x_scale_rank: int,
        out_dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        assert k % 128 == 0
        assert x_scale_rank in (1, 2)

        torch.manual_seed(0)

        q_input = torch.clamp(
            0.25 * torch.randn((m, k), dtype=torch.float16, device="cuda"),
            min=-0.8,
            max=0.8,
        ).to(self.fp8_dtype)

        weight_int4 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
        weight_packed = _pack_int4_to_int8(weight_int4)

        base_x_scale = 0.05 + 0.1 * torch.rand((m,), dtype=torch.float32, device="cuda")
        x_scale = base_x_scale if x_scale_rank == 1 else base_x_scale.view(m, 1)

        weight_scale = 0.05 + 0.1 * torch.rand(
            (n, k // 128), dtype=torch.float32, device="cuda"
        )

        bias = None
        if with_bias:
            bias = 0.01 * torch.randn((n,), dtype=out_dtype, device="cuda")

        return q_input, weight_packed, x_scale, weight_scale, bias

    def test_smoke_compile_and_cache(self):
        q_input, weight_packed, x_scale, weight_scale, bias = self._make_inputs(
            8,
            128,
            16,
            with_bias=True,
            x_scale_rank=2,
            out_dtype=torch.bfloat16,
        )

        out1 = w4a8_fp8_scaled_mm(
            q_input=q_input,
            weight_packed=weight_packed,
            x_scale=x_scale,
            weight_scale=weight_scale,
            group_size=128,
            out_dtype=torch.bfloat16,
            bias=bias,
        )
        out2 = w4a8_fp8_scaled_mm(
            q_input=q_input,
            weight_packed=weight_packed,
            x_scale=x_scale,
            weight_scale=weight_scale,
            group_size=128,
            out_dtype=torch.bfloat16,
            bias=bias,
        )

        self.assertEqual(out1.shape, (8, 16))
        self.assertEqual(out1.dtype, torch.bfloat16)
        torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)

    def test_numerical_parity_without_bias(self):
        test_configs = [
            (1, 128, 3, 1, torch.float32),
            (16, 128, 32, 2, torch.bfloat16),
            (32, 256, 64, 1, torch.float16),
        ]

        for m, k, n, x_scale_rank, out_dtype in test_configs:
            with self.subTest(
                m=m, k=k, n=n, x_scale_rank=x_scale_rank, out_dtype=out_dtype
            ):
                q_input, weight_packed, x_scale, weight_scale, _bias = self._make_inputs(
                    m,
                    k,
                    n,
                    with_bias=False,
                    x_scale_rank=x_scale_rank,
                    out_dtype=out_dtype,
                )

                out = w4a8_fp8_scaled_mm(
                    q_input=q_input,
                    weight_packed=weight_packed,
                    x_scale=x_scale,
                    weight_scale=weight_scale,
                    group_size=128,
                    out_dtype=out_dtype,
                )
                ref = _reference_w4a8_fp8_scaled_mm(
                    q_input=q_input,
                    weight_packed=weight_packed,
                    x_scale=x_scale,
                    weight_scale=weight_scale,
                    group_size=128,
                    out_dtype=out_dtype,
                )

                atol = 1e-4 if out_dtype == torch.float32 else 1e-2
                rtol = 1e-4 if out_dtype == torch.float32 else 1e-2
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_numerical_parity_with_bias_and_signed_int4_edges(self):
        m, k, n = 2, 128, 4
        q_input = torch.tensor(
            [
                [-0.75, -0.5, -0.25, 0.0] * 32,
                [0.75, 0.5, 0.25, -0.125] * 32,
            ],
            dtype=torch.float16,
            device="cuda",
        ).to(self.fp8_dtype)

        weight_int4 = torch.tensor(
            [
                [-8, -1, 0, 7] * 32,
                [7, 0, -1, -8] * 32,
                [-8, 7, -8, 7] * 32,
                [0, 1, 2, 3] * 32,
            ],
            dtype=torch.int8,
            device="cuda",
        )
        weight_packed = _pack_int4_to_int8(weight_int4)
        x_scale = torch.tensor([0.125, 0.25], dtype=torch.float32, device="cuda")
        weight_scale = torch.tensor(
            [[0.5], [0.25], [0.125], [0.75]], dtype=torch.float32, device="cuda"
        )
        bias = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.bfloat16, device="cuda")

        out = w4a8_fp8_scaled_mm(
            q_input=q_input,
            weight_packed=weight_packed,
            x_scale=x_scale,
            weight_scale=weight_scale,
            group_size=128,
            out_dtype=torch.bfloat16,
            bias=bias,
        )
        ref = _reference_w4a8_fp8_scaled_mm(
            q_input=q_input,
            weight_packed=weight_packed,
            x_scale=x_scale,
            weight_scale=weight_scale,
            group_size=128,
            out_dtype=torch.bfloat16,
            bias=bias,
        )

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
