import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.w4afp8_linear import (
    cutlass_w4a8_fp8_linear,
    quantize_input_to_fp8,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="default")


class TestW4AFP8LinearWrapper(unittest.TestCase):
    def test_dispatch_path_uses_quant_and_kernel(self):
        input_tensor = torch.randn(2, 3, 4, dtype=torch.float32)
        weight_packed = torch.randint(-128, 127, (5, 2), dtype=torch.int8)
        weight_scale = torch.randn(5, 2, dtype=torch.float32)
        bias = torch.randn(5, dtype=torch.float32)

        q_input = torch.randn(6, 4, dtype=torch.float32)
        x_scale = torch.randn(6, 1, dtype=torch.float32).abs()
        kernel_output = torch.randn(6, 5, dtype=torch.float32)

        with patch(
            "sglang.srt.layers.quantization.w4afp8_linear.is_w4a8_fp8_linear_supported",
            return_value=True,
        ), patch(
            "sglang.srt.layers.quantization.w4afp8_linear.quantize_input_to_fp8",
            return_value=(q_input, x_scale),
        ) as mock_quant, patch(
            "sglang.srt.layers.quantization.w4afp8_linear.w4a8_fp8_scaled_mm",
            return_value=kernel_output,
        ) as mock_kernel:
            output = cutlass_w4a8_fp8_linear(
                input=input_tensor,
                weight_packed=weight_packed,
                weight_scale=weight_scale,
                group_size=2,
                bias=bias,
                output_dtype=torch.float32,
            )

        self.assertEqual(output.shape, (2, 3, 5))
        torch.testing.assert_close(output, kernel_output.reshape(2, 3, 5))

        mock_quant.assert_called_once()
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertIsNone(quant_kwargs["input_scale"])
        self.assertEqual(quant_kwargs["input_2d"].shape, (6, 4))
        self.assertTrue(quant_kwargs["input_2d"].is_contiguous())
        torch.testing.assert_close(quant_kwargs["input_2d"], input_tensor.reshape(6, 4))

        mock_kernel.assert_called_once()
        kernel_kwargs = mock_kernel.call_args.kwargs
        self.assertIs(kernel_kwargs["q_input"], q_input)
        self.assertIs(kernel_kwargs["x_scale"], x_scale)
        self.assertEqual(kernel_kwargs["group_size"], 2)
        self.assertEqual(kernel_kwargs["out_dtype"], torch.float32)
        torch.testing.assert_close(kernel_kwargs["weight_packed"], weight_packed)
        torch.testing.assert_close(kernel_kwargs["weight_scale"], weight_scale)
        torch.testing.assert_close(kernel_kwargs["bias"], bias)

    def test_fallback_path_uses_reference_matmul(self):
        input_tensor = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, 2.0]]], dtype=torch.float32
        )
        weight_packed = torch.randint(-128, 127, (3, 2), dtype=torch.int8)
        weight_scale = torch.randn(3, 2, dtype=torch.float32)
        dequant_weight = torch.tensor(
            [
                [1.0, 0.0, 0.5, -1.0],
                [0.0, 1.0, -0.5, 2.0],
                [1.5, -1.0, 0.0, 0.25],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

        with patch(
            "sglang.srt.layers.quantization.w4afp8_linear.is_w4a8_fp8_linear_supported",
            return_value=False,
        ), patch(
            "sglang.srt.layers.quantization.w4afp8_linear._dequantize_w4_groupwise",
            return_value=dequant_weight,
        ) as mock_dequant, patch(
            "sglang.srt.layers.quantization.w4afp8_linear.quantize_input_to_fp8"
        ) as mock_quant, patch(
            "sglang.srt.layers.quantization.w4afp8_linear.w4a8_fp8_scaled_mm"
        ) as mock_kernel:
            output = cutlass_w4a8_fp8_linear(
                input=input_tensor,
                weight_packed=weight_packed,
                weight_scale=weight_scale,
                group_size=2,
                bias=bias,
                output_dtype=torch.float32,
            )

        expected = input_tensor.reshape(-1, 4) @ dequant_weight.t()
        expected = expected + bias
        expected = expected.reshape(1, 2, 3)

        mock_dequant.assert_called_once()
        mock_quant.assert_not_called()
        mock_kernel.assert_not_called()
        torch.testing.assert_close(output, expected)

    def test_quantize_input_to_fp8_uses_per_token_when_dynamic(self):
        input_2d = torch.randn(4, 8, dtype=torch.float32)
        q_input = torch.randn(4, 8, dtype=torch.float32)
        x_scale = torch.randn(4, 1, dtype=torch.float32).abs()

        with patch(
            "sglang.srt.layers.quantization.w4afp8_linear.sglang_per_token_quant_fp8",
            return_value=(q_input, x_scale),
        ) as mock_per_token, patch(
            "sglang.srt.layers.quantization.w4afp8_linear.scaled_fp8_quant"
        ) as mock_scaled:
            got_q_input, got_x_scale = quantize_input_to_fp8(input_2d)

        mock_per_token.assert_called_once()
        mock_scaled.assert_not_called()
        self.assertIs(got_q_input, q_input)
        self.assertIs(got_x_scale, x_scale)

    def test_quantize_input_to_fp8_with_scalar_scale_expands_to_per_token_shape(self):
        input_2d = torch.randn(3, 4, dtype=torch.float32)
        input_scale = torch.tensor([0.25], dtype=torch.float32)
        q_input = torch.randn(3, 4, dtype=torch.float32)
        scalar_scale = torch.tensor([0.25], dtype=torch.float32)

        with patch(
            "sglang.srt.layers.quantization.w4afp8_linear.scaled_fp8_quant",
            return_value=(q_input, scalar_scale),
        ) as mock_scaled:
            got_q_input, got_x_scale = quantize_input_to_fp8(input_2d, input_scale)

        mock_scaled.assert_called_once()
        self.assertIs(got_q_input, q_input)
        self.assertEqual(got_x_scale.shape, (3, 1))
        self.assertTrue(torch.allclose(got_x_scale, torch.full((3, 1), 0.25)))

    def test_quantize_input_to_fp8_rejects_non_scalar_static_scale(self):
        input_2d = torch.randn(2, 4, dtype=torch.float32)
        input_scale = torch.ones(2, 1, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "expected to be scalar"):
            quantize_input_to_fp8(input_2d, input_scale)


if __name__ == "__main__":
    unittest.main(verbosity=2)
