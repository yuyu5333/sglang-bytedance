import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.nn import Parameter

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8 import (
    CompressedTensorsW4AFP8,
    _unpack_repack_int32_to_cutlass_int8,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="default")


def _make_quant_args(
    num_bits: int = 4,
    type_name: str = "int",
    symmetric: bool = True,
    dynamic: bool = False,
    strategy: str = "group",
    group_size: int = 128,
):
    return SimpleNamespace(
        num_bits=num_bits,
        type=type_name,
        symmetric=symmetric,
        dynamic=dynamic,
        strategy=strategy,
        group_size=group_size,
    )


def _make_w4afp8_scheme(group_size: int = 128):
    weight_quant = _make_quant_args(
        num_bits=4,
        type_name="int",
        symmetric=True,
        dynamic=False,
        strategy="group",
        group_size=group_size,
    )
    input_quant = _make_quant_args(
        num_bits=8,
        type_name="float",
        symmetric=True,
        dynamic=True,
        strategy="token",
        group_size=None,
    )
    quant_config = SimpleNamespace(
        quant_format="pack-quantized",
    )
    return CompressedTensorsW4AFP8(
        quant_config=quant_config,
        weight_quant=weight_quant,
        input_quant=input_quant,
    )


class TestCreateWeightsRegistersExpectedParams(unittest.TestCase):
    def test_registers_weight_packed_weight_scale_weight_shape(self):
        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        k = 512
        n_partition = 256
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n_partition],
            input_size=k,
            output_size=n_partition,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        self.assertIn("weight_packed", dict(layer.named_parameters()))
        self.assertIn("weight_scale", dict(layer.named_parameters()))
        self.assertIn("weight_shape", dict(layer.named_parameters()))

        wp = layer.weight_packed
        ws = layer.weight_scale
        self.assertEqual(wp.shape, (n_partition, k // 8))
        self.assertEqual(wp.data.dtype, torch.int32)
        self.assertEqual(ws.shape, (n_partition, k // 128))
        self.assertEqual(ws.data.dtype, torch.float32)

    def test_registers_attributes_on_layer(self):
        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        scheme.create_weights(
            layer=layer,
            input_size_per_partition=256,
            output_partition_sizes=[128],
            input_size=256,
            output_size=128,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        self.assertEqual(layer.logical_widths, [128])
        self.assertEqual(layer.input_size_per_partition, 256)
        self.assertEqual(layer.output_size_per_partition, 128)
        self.assertEqual(layer.group_size, 128)
        self.assertEqual(layer.orig_dtype, torch.bfloat16)

    def test_fused_output_partition_sizes(self):
        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        scheme.create_weights(
            layer=layer,
            input_size_per_partition=512,
            output_partition_sizes=[256, 256],
            input_size=512,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        self.assertEqual(layer.logical_widths, [256, 256])
        self.assertEqual(layer.output_size_per_partition, 512)
        self.assertEqual(layer.weight_packed.shape[0], 512)


class TestProcessWeightsAfterLoadingRepacksWeightLayout(unittest.TestCase):
    def test_converts_int32_to_int8_packed(self):
        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        k = 256
        n = 64
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        layer.weight_packed.data = torch.randint(
            -100, 100, (n, k // 8), dtype=torch.int32
        )
        layer.weight_scale.data = torch.randn(n, k // 128, dtype=torch.float32)

        scheme.process_weights_after_loading(layer)

        wp = layer.weight_packed
        self.assertIsInstance(wp, Parameter)
        self.assertEqual(wp.dtype, torch.int8)
        self.assertEqual(wp.shape, (n, k // 2))
        self.assertTrue(wp.is_contiguous())
        self.assertTrue(getattr(layer, "is_w4afp8_converted", False))

    def test_idempotent_when_already_converted(self):
        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        k = 256
        n = 64
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        layer.weight_packed.data = torch.randint(
            -100, 100, (n, k // 8), dtype=torch.int32
        )
        layer.weight_scale.data = torch.randn(n, k // 128, dtype=torch.float32)

        scheme.process_weights_after_loading(layer)
        shape_after_first = layer.weight_packed.shape

        scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight_packed.shape, shape_after_first)


class TestApplyWeightsCallsCutlassW4A8FP8Linear(unittest.TestCase):
    @patch(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8.cutlass_w4a8_fp8_linear"
    )
    def test_passes_correct_arguments(self, mock_cutlass):
        mock_cutlass.return_value = torch.randn(2, 64, dtype=torch.bfloat16)

        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        k = 256
        n = 64
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        layer.weight_packed = Parameter(
            torch.randint(-128, 127, (n, k // 2), dtype=torch.int8),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.randn(n, k // 128, dtype=torch.float32),
            requires_grad=False,
        )
        layer.is_w4afp8_converted = True

        x = torch.randn(2, k, dtype=torch.bfloat16)
        bias = torch.randn(n, dtype=torch.bfloat16)

        output = scheme.apply_weights(layer, x, bias=bias)

        mock_cutlass.assert_called_once()
        call_kwargs = mock_cutlass.call_args.kwargs
        self.assertIs(call_kwargs["input"], x)
        torch.testing.assert_close(
            call_kwargs["weight_packed"], layer.weight_packed.data
        )
        torch.testing.assert_close(
            call_kwargs["weight_scale"], layer.weight_scale.data
        )
        self.assertEqual(call_kwargs["group_size"], 128)
        torch.testing.assert_close(call_kwargs["bias"], bias)
        self.assertEqual(call_kwargs["output_dtype"], torch.bfloat16)

    @patch(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8.cutlass_w4a8_fp8_linear"
    )
    def test_repacks_int32_on_the_fly_if_not_converted(self, mock_cutlass):
        mock_cutlass.return_value = torch.randn(1, 32, dtype=torch.bfloat16)

        scheme = _make_w4afp8_scheme(group_size=128)
        layer = torch.nn.Module()
        weight_loader = MagicMock()

        k = 256
        n = 32
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )

        layer.weight_packed = Parameter(
            torch.zeros(n, k // 8, dtype=torch.int32), requires_grad=False
        )
        layer.weight_scale = Parameter(
            torch.randn(n, k // 128, dtype=torch.float32), requires_grad=False
        )

        x = torch.randn(1, k, dtype=torch.bfloat16)
        scheme.apply_weights(layer, x)

        call_kwargs = mock_cutlass.call_args.kwargs
        self.assertEqual(call_kwargs["weight_packed"].dtype, torch.int8)
        self.assertEqual(call_kwargs["weight_packed"].shape[0], n)
        self.assertEqual(call_kwargs["weight_packed"].shape[1], k // 2)


class TestUnpackRepackInt32ToCutlassInt8(unittest.TestCase):
    def test_roundtrip_with_known_values(self):
        n = 4
        k = 128
        group_size = 128

        # 包含全部 signed int4 边界值 [-8, 7]
        weight_int4 = torch.randint(-8, 8, (n, k), dtype=torch.int8)

        # compressed-tensors `pack_to_int32` 采用 unsigned-offset 编码：
        # 存储前将每个 signed int4 值加上 zero-point=8，得到 [0, 15] 的 nibble。
        # 这与 `_unpack_repack_int32_to_cutlass_int8` 中 `- offset` 的解码互逆。
        pack_factor = 8
        offset = 1 << (4 - 1)  # = 8
        mask = (1 << 4) - 1  # = 0x0F
        packed_int32 = torch.zeros(n, k // pack_factor, dtype=torch.int32)
        for col in range(k):
            group = col // pack_factor
            shift = (col % pack_factor) * 4
            val = (weight_int4[:, col].to(torch.int32) + offset) & mask
            packed_int32[:, group] |= val << shift

        result = _unpack_repack_int32_to_cutlass_int8(packed_int32, num_bits=4)

        self.assertEqual(result.shape, (n, k // 2))
        self.assertEqual(result.dtype, torch.int8)

        for row in range(n):
            for col in range(k):
                packed_col = col // 2
                byte = result[row, packed_col].item() & 0xFF
                if col % 2 == 0:
                    nibble = byte & 0x0F
                else:
                    nibble = (byte >> 4) & 0x0F
                if nibble >= 8:
                    nibble -= 16
                self.assertEqual(
                    nibble,
                    weight_int4[row, col].item(),
                    f"mismatch at [{row}, {col}]: expected {weight_int4[row, col].item()}, got {nibble}",
                )

    def test_output_shape_and_dtype(self):
        packed = torch.randint(0, 1000, (8, 16), dtype=torch.int32)
        result = _unpack_repack_int32_to_cutlass_int8(packed, num_bits=4)
        self.assertEqual(result.shape, (8, 64))
        self.assertEqual(result.dtype, torch.int8)


class TestIsW4AFP8Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from compressed_tensors.quantization import QuantizationType

            cls.QType = QuantizationType
        except ImportError:
            raise unittest.SkipTest(
                "compressed_tensors package is not available."
            )

        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        cls.ConfigCls = CompressedTensorsConfig

    def _make_quant_args(
        self,
        num_bits: int = 4,
        qtype=None,
        symmetric: bool = True,
        dynamic: bool = False,
        strategy: str = "group",
        group_size: int = 128,
    ):
        if qtype is None:
            qtype = self.QType.INT if num_bits == 4 else self.QType.FLOAT
        return SimpleNamespace(
            num_bits=num_bits,
            type=qtype,
            symmetric=symmetric,
            dynamic=dynamic,
            strategy=strategy,
            group_size=group_size,
        )

    def test_detects_valid_w4afp8(self):
        # `_is_w4afp8` 是实例方法，签名为 (self, weight_quant, input_quant)。
        # 当前实现不依赖 self 的任何属性，但仍需用实例调用以避免 self 被错位成
        # 第一个业务参数。这里用 SimpleNamespace 作为"类实例"的轻量替身：
        # 通过 `__get__` 绑定把函数转成 bound method。
        fake_self = SimpleNamespace()
        is_w4afp8 = self.ConfigCls._is_w4afp8.__get__(fake_self)

        weight_quant = self._make_quant_args(
            num_bits=4, qtype=self.QType.INT, symmetric=True, dynamic=False
        )
        input_quant = self._make_quant_args(
            num_bits=8, qtype=self.QType.FLOAT, symmetric=True, dynamic=True
        )

        self.assertTrue(is_w4afp8(weight_quant, input_quant))

    def test_rejects_non_int_weight(self):
        fake_self = SimpleNamespace()
        is_w4afp8 = self.ConfigCls._is_w4afp8.__get__(fake_self)

        weight_quant = self._make_quant_args(
            num_bits=4, qtype=self.QType.FLOAT, symmetric=True, dynamic=False
        )
        input_quant = self._make_quant_args(
            num_bits=8, qtype=self.QType.FLOAT, symmetric=True, dynamic=True
        )

        self.assertFalse(is_w4afp8(weight_quant, input_quant))

    def test_rejects_static_activation(self):
        fake_self = SimpleNamespace()
        is_w4afp8 = self.ConfigCls._is_w4afp8.__get__(fake_self)

        weight_quant = self._make_quant_args(
            num_bits=4, qtype=self.QType.INT, symmetric=True, dynamic=False
        )
        input_quant = self._make_quant_args(
            num_bits=8, qtype=self.QType.FLOAT, symmetric=True, dynamic=False
        )

        self.assertFalse(is_w4afp8(weight_quant, input_quant))

    def test_rejects_asymmetric_weight(self):
        fake_self = SimpleNamespace()
        is_w4afp8 = self.ConfigCls._is_w4afp8.__get__(fake_self)

        weight_quant = self._make_quant_args(
            num_bits=4, qtype=self.QType.INT, symmetric=False, dynamic=False
        )
        input_quant = self._make_quant_args(
            num_bits=8, qtype=self.QType.FLOAT, symmetric=True, dynamic=True
        )

        self.assertFalse(is_w4afp8(weight_quant, input_quant))

    def test_rejects_wrong_num_bits(self):
        fake_self = SimpleNamespace()
        is_w4afp8 = self.ConfigCls._is_w4afp8.__get__(fake_self)

        weight_quant = self._make_quant_args(
            num_bits=8, qtype=self.QType.INT, symmetric=True, dynamic=False
        )
        input_quant = self._make_quant_args(
            num_bits=8, qtype=self.QType.FLOAT, symmetric=True, dynamic=True
        )

        self.assertFalse(is_w4afp8(weight_quant, input_quant))

    def test_rejects_none_quant_args(self):
        fake_self = SimpleNamespace()
        is_w4afp8 = self.ConfigCls._is_w4afp8.__get__(fake_self)

        self.assertFalse(is_w4afp8(None, None))
        self.assertFalse(
            is_w4afp8(
                self._make_quant_args(num_bits=4, qtype=self.QType.INT),
                None,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
