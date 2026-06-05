"""
旋转 + 非均匀 bit 分配 MLA KV cache pool (M3.a)。

适配 MLA layout（DeepSeek-V2 / V3 / V4-MLA 等）：每 token 的 KV 在
``MLATokenToKVPool`` 中是一段长度为 ``kv_lora_rank + qk_rope_head_dim``
的 latent 向量（buffer shape ``[N, 1, kv_lora_rank + qk_rope_head_dim]``）。

由于 RoPE 与 Hadamard 不可交换（``H · RoPE(x) ≠ RoPE(H · x)``），本实现：

* 仅对前 ``kv_lora_rank`` 维（latent / nope 段）做旋转 + INT2/3/4 量化；
* ``qk_rope_head_dim`` 维（rope 段）保留原 dtype（bf16/fp16）原样存储。

物理 buffer 布局（按 token 顺序，每 row）::

    [ packed_latent (uint8, latent_row_bytes) | rope_raw (uint8 view of bf16, rope_bytes) ]

整张 buffer 是 ``[size + page_size, 1, latent_row_bytes + rope_bytes]`` 的 uint8。

Calibration schema (.pt)::

    {
        "_meta": {
            "mode": "mla",
            "kv_lora_rank": int,
            "qk_rope_head_dim": int,
            "layer_num": int,
        },
        layer_id (int): {
            "latent": {"R": Tensor[L, L] fp32,
                        "bits": Tensor[L] int32,
                        "scale": Tensor[L] fp32,
                        "zero":  Tensor[L] fp32},
        },
        ...
    }

其中 ``L = kv_lora_rank``。

已知限制（追踪在 KVRoadMap.md "M3.a"）：

* DSA / DeepSeek-V4 多比例分级压缩路径不支持（断言已拒绝）。
* attention backend 默认通过 ``get_key_buffer`` / ``get_value_buffer`` 读取，
  会触发整层 dequant；建议在性能路径切换到 ``get_dequant_workspace``。
* bitpack/bitunpack 走 CPU 往返（与 M1 MHA 版一致），M2 用 Triton 优化。
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.layers.quantization.rotated_kv_quant import (
    RotatedQuantizerConfig,
    bitpack_rowwise,
    bitunpack_rowwise,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 校准加载
# ----------------------------------------------------------------------
_REQUIRED_KEYS = ("R", "bits", "scale", "zero")


def _validate_latent(side: Dict[str, torch.Tensor], dim: int, tag: str) -> None:
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
        raise ValueError(
            f"calib[{tag}] bits/scale/zero must all be shape ({dim},)"
        )
    if int(bits.min().item()) < 1 or int(bits.max().item()) > 8:
        raise ValueError(
            f"calib[{tag}].bits must be in [1, 8], got "
            f"[{int(bits.min())}, {int(bits.max())}]"
        )


def load_rotated_quant_mla_calibration(
    path: str,
    layer_num: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> Dict[int, RotatedQuantizerConfig]:
    """读取 MLA 模式 .pt 校准文件。

    返回 ``{layer_id: cfg_latent}``。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"rotated-kv-quant-config not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(
            f"calibration file {path} must be a dict, got {type(raw)}"
        )

    meta = raw.get("_meta")
    if meta is None or meta.get("mode") != "mla":
        raise ValueError(
            "calibration file is not in MLA mode; expected "
            "_meta={'mode': 'mla', 'kv_lora_rank': L, 'qk_rope_head_dim': R}. "
            "Use --mla-mode when running build_rotated_kv_calib.py."
        )
    if int(meta.get("kv_lora_rank", -1)) != kv_lora_rank:
        raise ValueError(
            f"calibration kv_lora_rank={meta.get('kv_lora_rank')} "
            f"!= model kv_lora_rank={kv_lora_rank}"
        )
    if int(meta.get("qk_rope_head_dim", -1)) != qk_rope_head_dim:
        raise ValueError(
            f"calibration qk_rope_head_dim={meta.get('qk_rope_head_dim')} "
            f"!= model qk_rope_head_dim={qk_rope_head_dim}"
        )

    out: Dict[int, RotatedQuantizerConfig] = {}
    for layer_id in range(layer_num):
        if layer_id not in raw:
            raise ValueError(f"calibration missing layer_id={layer_id}")
        entry = raw[layer_id]
        if "latent" not in entry:
            raise ValueError(f"calibration[{layer_id}] missing 'latent'")
        _validate_latent(
            entry["latent"], kv_lora_rank, f"layer {layer_id} latent"
        )
        out[layer_id] = RotatedQuantizerConfig(
            R=entry["latent"]["R"].to(torch.float32),
            bits=entry["latent"]["bits"].to(torch.int32),
            scale=entry["latent"]["scale"].to(torch.float32),
            zero=entry["latent"]["zero"].to(torch.float32),
        )
    return out


# ----------------------------------------------------------------------
# 设备绑定的 latent 量化器
# ----------------------------------------------------------------------
class _DeviceBoundLatentQuantizer:
    """latent 段（kv_lora_rank 维）的量化器，绑定到指定 device。

    输入 / 输出形状均为 ``[..., kv_lora_rank]``，packed 形状最后一维为
    ``row_bytes``。
    """

    def __init__(self, cfg: RotatedQuantizerConfig, device: torch.device):
        self.device = device
        self.R = cfg.R.to(device=device, dtype=torch.float32)
        self.bits_cpu = cfg.bits.to(torch.int32).cpu()
        self.bits_dev = cfg.bits.to(device=device, dtype=torch.int32)
        self.scale = cfg.scale.to(device=device, dtype=torch.float32)
        self.zero = cfg.zero.to(device=device, dtype=torch.float32)
        self.row_bits = int(self.bits_cpu.sum().item())
        self.row_bytes = (self.row_bits + 7) // 8
        levels = (1 << self.bits_dev.to(torch.int64)) - 1
        self.levels = levels.to(torch.float32)
        self.D = int(self.R.shape[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x: [..., D]`` -> ``uint8 [..., row_bytes]``."""
        if x.shape[-1] != self.D:
            raise ValueError(
                f"latent encode expects last dim {self.D}, got {x.shape[-1]}"
            )
        x_rot = x.to(torch.float32) @ self.R
        codes = ((x_rot - self.zero) / self.scale.clamp_min(1e-12)).round()
        codes = codes.clamp_(min=0.0).minimum(self.levels)
        codes_int = codes.to(torch.int64).cpu()
        packed_cpu = bitpack_rowwise(codes_int, self.bits_cpu)
        return packed_cpu.to(self.device)

    def decode(
        self, packed: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """``uint8 [..., row_bytes]`` -> ``dtype [..., D]``."""
        if packed.shape[-1] != self.row_bytes:
            raise ValueError(
                f"latent decode expects last dim {self.row_bytes} bytes, "
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
class RotatedQuantMLATokenToKVPool(MLATokenToKVPool):
    """MLA KV pool：latent 段做旋转 + 非均匀 bit 量化，rope 段保留原 dtype。

    Buffer 布局（每 token-head row）::

        [ packed_latent (uint8 × latent_row_bytes) | rope_raw (uint8 view of bf16) ]

    其中 latent_row_bytes 来自校准 ``Σ b[d] / 8``，rope_bytes =
    ``qk_rope_head_dim * itemsize(self.dtype)``。

    实现要点：
    * 重写 ``_create_buffers`` —— 强制 ``store_dtype = uint8``，buffer 总宽度
      = ``latent_row_bytes + rope_bytes``。
    * 重写 ``set_mla_kv_buffer(layer, loc, k_nope, k_rope)`` —— 这是 MLA
      模型实际写入 API；分别量化 latent 段、view rope 段为 uint8。
    * 重写 ``set_kv_buffer(layer, loc, cache_k, cache_v=None)`` —— 兼容
      cache_k 拼接形式（``[..., L+R]``），切片后委托给 set_mla_kv_buffer。
    * 重写 ``get_key_buffer`` —— 整层 dequant，返回 ``[N, 1, L+R]`` 拼接形式。
    * 重写 ``get_value_buffer`` —— 仅 latent 段 dequant，返回 ``[N, 1, L]``。
    * 重写 ``get_mla_kv_buffer`` —— 仅对 ``loc`` 处 dequant，返回
      ``(k_nope, k_rope)`` 两个 tensor，是 attention 的快路径。
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        calib_path: str,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        # 先加载校准（MLATokenToKVPool 的 _create_buffers 会用到 row_bytes）
        self._calib_cfgs = load_rotated_quant_mla_calibration(
            calib_path,
            layer_num=layer_num,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self._calib_path = calib_path

        # row_bytes 一致性校验（M3.a 假设所有层一致）
        sample = next(iter(self._calib_cfgs.values()))
        self._latent_row_bytes = (int(sample.bits.sum().item()) + 7) // 8
        for lid, cfg in self._calib_cfgs.items():
            row_bytes = (int(cfg.bits.sum().item()) + 7) // 8
            if row_bytes != self._latent_row_bytes:
                raise ValueError(
                    f"layer {lid} latent row_bytes {row_bytes} != "
                    f"layer 0 row_bytes {self._latent_row_bytes}"
                )

        # rope 段以 uint8 视图存放 raw dtype 字节
        self._rope_bytes = qk_rope_head_dim * torch.empty([], dtype=dtype).element_size()
        self._row_total_bytes = self._latent_row_bytes + self._rope_bytes
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim

        # quantizer holders；在 _create_buffers 完成后绑定 device
        self._quant_latent: Dict[int, _DeviceBoundLatentQuantizer] = {}

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            use_dsa=False,
            override_kv_cache_dim=None,
        )
        logger.info(
            "RotatedQuantMLATokenToKVPool ready: dtype=%s store_dtype=uint8 "
            "kv_lora_rank=%d qk_rope_head_dim=%d "
            "latent_row_bytes=%d rope_bytes=%d total_row=%d calib=%s",
            self.dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            self._latent_row_bytes,
            self._rope_bytes,
            self._row_total_bytes,
            self._calib_path,
        )

    # ------------------------------------------------------------------
    # Buffer 分配：覆写 store_dtype 为 uint8，shape 为 [N, 1, total_bytes]
    # ------------------------------------------------------------------
    def _create_buffers(self):
        # 强制 uint8 packed 存储
        self.store_dtype = torch.uint8
        # MLATokenToKVPool 用 self.kv_cache_dim 决定 buffer 尺寸；这里
        # 我们直接用自定义 _row_total_bytes，避免与基类 kv_cache_dim 冲突。

        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # padded slot 0 用于 padded token 的 dummy 写入
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self._row_total_bytes),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        # 设备已知，绑定 quantizer
        device = torch.device(self.device)
        for lid, cfg in self._calib_cfgs.items():
            self._quant_latent[lid] = _DeviceBoundLatentQuantizer(
                cfg, device=device
            )

    # ------------------------------------------------------------------
    # 内部辅助：拿到某层 buffer 的 latent / rope 切片视图
    # ------------------------------------------------------------------
    def _latent_slice(self, layer_buf: torch.Tensor) -> torch.Tensor:
        """``[N, 1, total]`` -> ``[N, 1, latent_row_bytes]`` (uint8 view)."""
        return layer_buf[..., : self._latent_row_bytes]

    def _rope_slice(self, layer_buf: torch.Tensor) -> torch.Tensor:
        """``[N, 1, total]`` -> ``[N, 1, rope_bytes]`` (uint8 view)."""
        return layer_buf[..., self._latent_row_bytes :]

    def _rope_as_dtype(self, rope_uint8: torch.Tensor) -> torch.Tensor:
        """``uint8 [..., rope_bytes]`` -> ``self.dtype [..., qk_rope_head_dim]``。

        要求底层是 contiguous 才能 view。这里通过 reshape + view 实现。
        """
        # 必须 contiguous 才能合法 view 到不同 dtype
        rope_contig = rope_uint8.contiguous()
        # 把最后一维（rope_bytes）变成 qk_rope_head_dim 个 self.dtype 元素
        return rope_contig.view(self.dtype).reshape(
            *rope_uint8.shape[:-1], self._qk_rope_head_dim
        )

    def _dtype_as_rope_uint8(self, rope_dtype_t: torch.Tensor) -> torch.Tensor:
        """``self.dtype [..., qk_rope_head_dim]`` -> ``uint8 [..., rope_bytes]``."""
        # 转到 self.dtype 再 view 成 uint8
        if rope_dtype_t.dtype != self.dtype:
            rope_dtype_t = rope_dtype_t.to(self.dtype)
        rope_contig = rope_dtype_t.contiguous()
        return rope_contig.view(torch.uint8).reshape(
            *rope_dtype_t.shape[:-1], self._rope_bytes
        )

    # ------------------------------------------------------------------
    # 写路径：set_mla_kv_buffer (主路径) + set_kv_buffer (兼容)
    # ------------------------------------------------------------------
    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """MLA 模型的标准写入 API。

        参数：
            cache_k_nope: ``[N, 1, kv_lora_rank]``，将被旋转 + 量化。
            cache_k_rope: ``[N, 1, qk_rope_head_dim]``，按原 dtype 写入。
        """
        layer_id = layer.layer_id
        rel_id = layer_id - self.start_layer

        # latent 量化
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
        if cache_k_rope.dtype != self.dtype:
            cache_k_rope = cache_k_rope.to(self.dtype)

        q = self._quant_latent[layer_id]
        packed_latent = q.encode(cache_k_nope)  # [N, 1, latent_row_bytes]
        rope_u8 = self._dtype_as_rope_uint8(cache_k_rope)  # [N, 1, rope_bytes]

        layer_buf = self.kv_buffer[rel_id]
        # 注意：必须先写 rope 再写 latent 还是反之都可以；二者写在不同的最后维切片
        # 但是 advanced indexing on the same tensor 时要避免重叠 alias。
        # 我们用 index_put 到完整 row 的拼接结果，最稳。
        full_row = torch.cat([packed_latent, rope_u8], dim=-1)
        layer_buf[loc] = full_row

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: Optional[torch.Tensor] = None,
    ):
        """兼容路径：cache_k 是 ``[N, 1, kv_lora_rank + qk_rope_head_dim]``。

        基类 ``MLATokenToKVPool.set_kv_buffer`` 把整段当作单 tensor 写入，
        我们这里切片成 (nope, rope) 后委托给 ``set_mla_kv_buffer``。
        ``cache_v`` 在 MLA 中由 latent 推导，因此被忽略。
        """
        del cache_v  # MLA: V 由 latent 推导
        if cache_k.shape[-1] != self._kv_lora_rank + self._qk_rope_head_dim:
            raise ValueError(
                f"set_kv_buffer expects cache_k last dim "
                f"{self._kv_lora_rank + self._qk_rope_head_dim}, "
                f"got {cache_k.shape[-1]}"
            )
        cache_k_nope = cache_k[..., : self._kv_lora_rank]
        cache_k_rope = cache_k[..., self._kv_lora_rank :]
        self.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    # ------------------------------------------------------------------
    # 读路径
    # ------------------------------------------------------------------
    def _full_dequant_concat(self, layer_id: int) -> torch.Tensor:
        """整层 dequant，返回 ``[N, 1, kv_lora_rank + qk_rope_head_dim]``。"""
        rel_id = layer_id - self.start_layer
        layer_buf = self.kv_buffer[rel_id]
        latent_packed = self._latent_slice(layer_buf)
        rope_u8 = self._rope_slice(layer_buf)

        q = self._quant_latent[layer_id]
        latent_dq = q.decode(latent_packed, dtype=self.dtype)  # [N, 1, L]
        rope_dq = self._rope_as_dtype(rope_u8)  # [N, 1, R]
        return torch.cat([latent_dq, rope_dq], dim=-1)

    def get_key_buffer(self, layer_id: int):
        """返回 ``[N, 1, kv_lora_rank + qk_rope_head_dim]``，logical dtype。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._full_dequant_concat(layer_id)

    def get_value_buffer(self, layer_id: int):
        """返回 ``[N, 1, kv_lora_rank]``（仅 latent 段）。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        rel_id = layer_id - self.start_layer
        layer_buf = self.kv_buffer[rel_id]
        latent_packed = self._latent_slice(layer_buf)
        q = self._quant_latent[layer_id]
        return q.decode(latent_packed, dtype=self.dtype)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """attention 的快路径：仅对 ``loc`` 行做 dequant。

        返回 ``(k_nope[N,1,L], k_rope[N,1,R])``。
        """
        layer_id = layer.layer_id
        rel_id = layer_id - self.start_layer
        target_dtype = dst_dtype if dst_dtype is not None else self.dtype

        layer_buf = self.kv_buffer[rel_id]
        rows = layer_buf[loc]  # [len(loc), 1, total]
        latent_packed = rows[..., : self._latent_row_bytes]
        rope_u8 = rows[..., self._latent_row_bytes :]

        q = self._quant_latent[layer_id]
        k_nope = q.decode(latent_packed, dtype=target_dtype)
        k_rope = self._rope_as_dtype(rope_u8).to(target_dtype)
        return k_nope, k_rope

    def get_dequant_workspace(
        self,
        layer_id: int,
        loc: torch.Tensor,
        side: str = "latent",
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """通用快路径，对照 MHA 版接口名称。

        side="latent" -> [N, 1, kv_lora_rank]
        side="rope"   -> [N, 1, qk_rope_head_dim]
        side="full"   -> [N, 1, kv_lora_rank + qk_rope_head_dim]
        """
        rel_id = layer_id - self.start_layer
        target_dtype = dtype if dtype is not None else self.dtype
        layer_buf = self.kv_buffer[rel_id]
        rows = layer_buf[loc]

        if side == "latent":
            latent_packed = rows[..., : self._latent_row_bytes]
            return self._quant_latent[layer_id].decode(
                latent_packed, dtype=target_dtype
            )
        elif side == "rope":
            rope_u8 = rows[..., self._latent_row_bytes :]
            return self._rope_as_dtype(rope_u8).to(target_dtype)
        elif side == "full":
            latent_packed = rows[..., : self._latent_row_bytes]
            rope_u8 = rows[..., self._latent_row_bytes :]
            latent_dq = self._quant_latent[layer_id].decode(
                latent_packed, dtype=target_dtype
            )
            rope_dq = self._rope_as_dtype(rope_u8).to(target_dtype)
            return torch.cat([latent_dq, rope_dq], dim=-1)
        else:
            raise ValueError(
                f"side must be 'latent' | 'rope' | 'full', got {side}"
            )

    # ------------------------------------------------------------------
    # CPU offloading 兼容（基类用 self.kv_buffer 做 indexing，OK）
    # 但 disagg 信息要返回 packed 字节
    # ------------------------------------------------------------------
    # MLATokenToKVPool.get_contiguous_buf_infos 已经直接读取 self.kv_buffer
    # 的 data_ptr / nbytes，对 uint8 packed buffer 同样有效，无需 override。

    def get_kv_size_bytes(self):
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += kv_cache.nbytes
        return kv_size_bytes


__all__ = [
    "RotatedQuantMLATokenToKVPool",
    "load_rotated_quant_mla_calibration",
]
