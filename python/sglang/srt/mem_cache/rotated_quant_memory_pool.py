"""
Rotated + non-uniform bit-allocation KV cache pool (M1).

This is the SGLang runtime adapter for the M0 quantizer in
``layers/quantization/rotated_kv_quant.py``. It subclasses
:class:`MHATokenToKVPool` and replaces the dense BF16/FP8 buffer with a
``uint8`` packed buffer of shape ``[size+page_size, head_num, row_bytes]``,
where ``row_bytes`` is determined by the per-layer bit table loaded from
calibration.

Calibration file schema (a single ``.pt`` file)::

    {
        layer_id (int): {
            "k": {"R": Tensor[D, D] fp32,
                   "bits": Tensor[D] int32,
                   "scale": Tensor[D] fp32,
                   "zero":  Tensor[D] fp32},
            "v": { ... same ... },
        },
        ...
    }

For now the rotation/bits/scale/zero are *layer-shared across heads* (the
same per-coordinate table is broadcast over ``head_num``). Per-head
calibration is left to M2.

Known M1 limitations (tracked in KVRoadMap.md, "启发三 / 风险与回退"):
  * Hadamard rotation is applied to the entire ``head_dim`` -- if the
    model uses RoPE, RoPE must be disabled for the rotated channels or
    moved before the quantizer. The model-side wiring is not yet done in
    this file; this pool only handles the storage layer.
  * GQA / MLA layouts are not supported; this pool targets vanilla MHA.
  * ``get_key_buffer`` / ``get_value_buffer`` perform a full-pool dequant
    on every call, which is correct but slow. Use
    :py:meth:`get_dequant_workspace` from attention backends.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.layers.quantization.rotated_kv_quant import (
    RotatedQuantizer,
    RotatedQuantizerConfig,
    bitpack_rowwise,
    bitunpack_rowwise,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Calibration loading
# ----------------------------------------------------------------------
_REQUIRED_KEYS = ("R", "bits", "scale", "zero")


def _validate_side(side: Dict[str, torch.Tensor], dim: int, tag: str) -> None:
    for k in _REQUIRED_KEYS:
        if k not in side:
            raise ValueError(f"calib[{tag}] missing key '{k}'")
    R = side["R"]
    bits = side["bits"]
    scale = side["scale"]
    zero = side["zero"]
    if R.shape != (dim, dim):
        raise ValueError(f"calib[{tag}].R shape {tuple(R.shape)} != ({dim},{dim})")
    if bits.shape != (dim,) or scale.shape != (dim,) or zero.shape != (dim,):
        raise ValueError(f"calib[{tag}] bits/scale/zero must all be shape ({dim},)")
    if int(bits.min().item()) < 1 or int(bits.max().item()) > 8:
        raise ValueError(
            f"calib[{tag}].bits must be in [1, 8], got [{int(bits.min())}, "
            f"{int(bits.max())}]"
        )


def load_rotated_quant_calibration(
    path: str, layer_num: int, head_dim: int, v_head_dim: Optional[int] = None
) -> Dict[int, Dict[str, RotatedQuantizerConfig]]:
    """Read a ``.pt`` calibration file and build per-layer quantizer configs.

    Returns ``{layer_id: {"k": cfg_k, "v": cfg_v}}``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"rotated-kv-quant-config not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"calibration file {path} must be a dict, got {type(raw)}")

    v_dim = v_head_dim if v_head_dim is not None else head_dim
    out: Dict[int, Dict[str, RotatedQuantizerConfig]] = {}
    for layer_id in range(layer_num):
        if layer_id not in raw:
            raise ValueError(f"calibration missing layer_id={layer_id}")
        entry = raw[layer_id]
        if "k" not in entry or "v" not in entry:
            raise ValueError(f"calibration[{layer_id}] missing 'k' or 'v'")
        _validate_side(entry["k"], head_dim, f"layer {layer_id} k")
        _validate_side(entry["v"], v_dim, f"layer {layer_id} v")
        out[layer_id] = {
            "k": RotatedQuantizerConfig(
                R=entry["k"]["R"].to(torch.float32),
                bits=entry["k"]["bits"].to(torch.int32),
                scale=entry["k"]["scale"].to(torch.float32),
                zero=entry["k"]["zero"].to(torch.float32),
            ),
            "v": RotatedQuantizerConfig(
                R=entry["v"]["R"].to(torch.float32),
                bits=entry["v"]["bits"].to(torch.int32),
                scale=entry["v"]["scale"].to(torch.float32),
                zero=entry["v"]["zero"].to(torch.float32),
            ),
        }
    return out


# ----------------------------------------------------------------------
# Per-layer quantizer holder bound to a device
# ----------------------------------------------------------------------
class _DeviceBoundQuantizer:
    """Wraps a :class:`RotatedQuantizer` config moved to ``device``.

    Adds a vectorised ``encode`` / ``decode`` operating on
    ``[..., D]`` tensors. The implementation reuses the M0 reference
    bit-pack / bit-unpack routines (pure PyTorch). M2 will swap them for
    Triton kernels.
    """

    def __init__(self, cfg: RotatedQuantizerConfig, device: torch.device):
        self.device = device
        self.R = cfg.R.to(device=device, dtype=torch.float32)
        self.bits_cpu = cfg.bits.to(torch.int32).cpu()  # bit-pack runs on cpu (M1)
        self.bits_dev = cfg.bits.to(device=device, dtype=torch.int32)
        self.scale = cfg.scale.to(device=device, dtype=torch.float32)
        self.zero = cfg.zero.to(device=device, dtype=torch.float32)
        self.row_bits = int(self.bits_cpu.sum().item())
        self.row_bytes = (self.row_bits + 7) // 8
        levels = (1 << self.bits_dev.to(torch.int64)) - 1  # [D]
        self.levels = levels.to(torch.float32)
        self.D = int(self.R.shape[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x: [..., D]`` real -> ``uint8 [..., row_bytes]`` packed."""
        if x.shape[-1] != self.D:
            raise ValueError(
                f"encode expects last dim {self.D}, got {x.shape[-1]}"
            )
        x_rot = x.to(torch.float32) @ self.R  # [..., D]
        codes = ((x_rot - self.zero) / self.scale.clamp_min(1e-12)).round()
        codes = codes.clamp_(min=0.0).minimum(self.levels)
        codes_int = codes.to(torch.int64).cpu()  # bitpack on cpu in M1
        packed_cpu = bitpack_rowwise(codes_int, self.bits_cpu)
        return packed_cpu.to(self.device)

    def decode(
        self, packed: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """``uint8 [..., row_bytes]`` -> dtype ``[..., D]``."""
        if packed.shape[-1] != self.row_bytes:
            raise ValueError(
                f"decode expects last dim {self.row_bytes} bytes, "
                f"got {packed.shape[-1]}"
            )
        codes = bitunpack_rowwise(packed.cpu(), self.bits_cpu, dim=self.D)
        codes_dev = codes.to(self.device).to(torch.float32)
        x_rot_hat = codes_dev * self.scale + self.zero
        x_hat = x_rot_hat @ self.R.t()
        return x_hat.to(dtype)


# ----------------------------------------------------------------------
# Pool
# ----------------------------------------------------------------------
class RotatedQuantMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool whose tensor storage is rotated + bit-packed uint8.

    Calibration is loaded eagerly from ``calib_path`` at construction time;
    every layer must have a matching entry. The buffer shape per layer is
    ``[size+page_size, head_num, row_bytes_{k,v}]`` (uint8).

    Notes
    -----
    * ``store_dtype`` is overridden to ``torch.uint8``; ``self.dtype``
      remains the *logical* dtype (e.g. bf16) so attention backends still
      see bf16 after dequant.
    * ``set_kv_buffer`` is rewritten end-to-end (no JIT store_cache); we
      do a vanilla ``index_put`` after quantization.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        calib_path: str,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        # Load calibration first; we need row_bytes to size buffers.
        v_dim = v_head_dim if v_head_dim is not None else head_dim
        self._calib_cfgs = load_rotated_quant_calibration(
            calib_path, layer_num=layer_num, head_dim=head_dim, v_head_dim=v_dim
        )
        self._calib_path = calib_path

        # Quantizer bookkeeping; will be created in _create_buffers once
        # device is known.
        self._quant_k: Dict[int, _DeviceBoundQuantizer] = {}
        self._quant_v: Dict[int, _DeviceBoundQuantizer] = {}
        # Stash row_bytes; same across heads in M1.
        sample = next(iter(self._calib_cfgs.values()))
        self._row_bytes_k = sample["k"].row_bytes
        self._row_bytes_v = sample["v"].row_bytes
        # Sanity: every layer must agree on row_bytes (M1 assumption).
        for lid, cfg in self._calib_cfgs.items():
            if cfg["k"].row_bytes != self._row_bytes_k:
                raise ValueError(
                    f"layer {lid} k row_bytes {cfg['k'].row_bytes} != "
                    f"layer 0 row_bytes {self._row_bytes_k}"
                )
            if cfg["v"].row_bytes != self._row_bytes_v:
                raise ValueError(
                    f"layer {lid} v row_bytes {cfg['v'].row_bytes} != "
                    f"layer 0 row_bytes {self._row_bytes_v}"
                )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,  # JIT store_cache assumes dense -> off
        )
        logger.info(
            "RotatedQuantMHATokenToKVPool ready: dtype=%s, store_dtype=uint8, "
            "row_bytes_k=%d, row_bytes_v=%d, calib=%s",
            self.dtype,
            self._row_bytes_k,
            self._row_bytes_v,
            self._calib_path,
        )

    # ------------------------------------------------------------------
    # Buffer allocation: override store_dtype to uint8 + custom shape.
    # ------------------------------------------------------------------
    def _create_buffers(self):
        # Force uint8 packed storage regardless of self.dtype.
        self.store_dtype = torch.uint8

        from contextlib import nullcontext

        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self._row_bytes_k),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self._row_bytes_v),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        # Bind quantizer configs to device now that device is known.
        device = torch.device(self.device)
        for lid, cfg in self._calib_cfgs.items():
            self._quant_k[lid] = _DeviceBoundQuantizer(cfg["k"], device=device)
            self._quant_v[lid] = _DeviceBoundQuantizer(cfg["v"], device=device)

        # Skip building data_ptrs / data_strides used by JIT store_cache;
        # we override set_kv_buffer to bypass that path entirely.
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        import numpy as np

        self.data_strides = torch.tensor(
            [
                int(np.prod(x.shape[1:]) * x.dtype.itemsize)
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Write path: quantize -> uint8 -> index_put
    # ------------------------------------------------------------------
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        rel_id = layer_id - self.start_layer

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k.div(k_scale)
            if v_scale is not None:
                cache_v = cache_v.div(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        # Expect cache_k / cache_v shape [N, head_num, head_dim].
        qk = self._quant_k[layer_id]
        qv = self._quant_v[layer_id]
        packed_k = qk.encode(cache_k)  # [N, H, row_bytes_k]
        packed_v = qv.encode(cache_v)

        self.k_buffer[rel_id][loc] = packed_k
        self.v_buffer[rel_id][loc] = packed_v

    # ------------------------------------------------------------------
    # Read path: dequant on demand.
    # ------------------------------------------------------------------
    def get_dequant_workspace(
        self,
        layer_id: int,
        loc: torch.Tensor,
        side: str = "k",
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Decode the rows at ``loc`` of layer ``layer_id`` to dense ``dtype``.

        Returns shape ``[len(loc), head_num, head_dim_or_v_head_dim]``. This
        is the recommended fast path for attention backends -- only the
        currently-used tokens get dequantized.
        """
        rel_id = layer_id - self.start_layer
        target_dtype = dtype if dtype is not None else self.dtype
        if side == "k":
            packed = self.k_buffer[rel_id][loc]
            q = self._quant_k[layer_id]
        elif side == "v":
            packed = self.v_buffer[rel_id][loc]
            q = self._quant_v[layer_id]
        else:
            raise ValueError(f"side must be 'k' or 'v', got {side}")
        return q.decode(packed, dtype=target_dtype)

    def _full_dequant(self, layer_id: int, side: str) -> torch.Tensor:
        rel_id = layer_id - self.start_layer
        if side == "k":
            packed = self.k_buffer[rel_id]
            q = self._quant_k[layer_id]
        else:
            packed = self.v_buffer[rel_id]
            q = self._quant_v[layer_id]
        return q.decode(packed, dtype=self.dtype)

    def _get_key_buffer(self, layer_id: int):
        # Slow fallback: fully dequant the layer.
        return self._full_dequant(layer_id, "k")

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        return self._full_dequant(layer_id, "v")

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)


__all__ = [
    "RotatedQuantMHATokenToKVPool",
    "load_rotated_quant_calibration",
]
