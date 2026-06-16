"""
旋转 + 非均匀 bit 分配 DeepSeek-V4 KV cache pool.

两个工作模式:

* ``mode='eval'`` (M3.b)
    主存储不动 (DSv4 原生 FP8 layout, FlashMLA / fused_store_cache 兼容).
    包装类持有 per-layer ``RotatedQuantizer``, 提供 ``simulate_quantize_nope``
    离线评估接口. 用于在不动 kernel 的前提下度量我们的 INT2/3/4 方案在 DSv4
    nope 段上的精度.

* ``mode='wall'`` (M3.c.2, **真存储替换 + 三池同步 + attention shim**)
    ``swa_kv_pool`` / ``c4_kv_pool`` / ``c128_kv_pool`` 三个子池的
    ``kv_buffer`` 被替换成 packed layout::

        bytes_per_token = row_bytes_nope + 128  # rope = BF16 64 elems
        bytes_per_page  = bytes_per_token * page_size_of_pool

    写入路径走 :func:`rotated_store_to_packed`: BF16 ``[N, 512]`` (cat(nope,
    rope)) -> 旋转 + INT2/3/4 affine 量化 + bit-pack -> packed slot 写入 paged
    buffer; rope 段保留原 BF16 字节直接拷贝. 读取路径在 attention prologue
    一次性 dequant 当前 batch 涉及到的 page 范围里所有 token, 把字节填回
    一份 **shadow FP8 buffer**, layout 与原生 DSv4 ``[num_pages,
    bytes_per_page_padded(584/576)]`` 一致, FlashMLA 直接吃 shadow, 不需要
    任何修改.

校准 schema (与 M3.b 一致): ``_meta.mode='dsv4'``, 每层 ``nope`` 段
``{R, bits, scale, zero}``. 详见 :func:`load_rotated_quant_dsv4_calibration`.

Indexer / compress_state 在 wall 模式下保持原 FP8 不动 (它们是元数据,
不是逐 token 主存储).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional, Tuple

import torch

from sglang.srt.layers.quantization.rotated_kv_quant import (
    RotatedQuantizer,
    RotatedQuantizerConfig,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Calibration loader (DSv4 mode)
# ----------------------------------------------------------------------
_REQUIRED_KEYS = ("R", "bits", "scale", "zero")


def _validate_side(side: Dict[str, torch.Tensor], dim: int, tag: str) -> None:
    for k in _REQUIRED_KEYS:
        if k not in side:
            raise ValueError(f"calib[{tag}] missing key '{k}'")
    if side["R"].shape != (dim, dim):
        raise ValueError(
            f"calib[{tag}].R shape {tuple(side['R'].shape)} != ({dim},{dim})"
        )
    for k in ("bits", "scale", "zero"):
        if side[k].shape != (dim,):
            raise ValueError(
                f"calib[{tag}].{k} shape {tuple(side[k].shape)} != ({dim},)"
            )
    if int(side["bits"].min().item()) < 1 or int(side["bits"].max().item()) > 8:
        raise ValueError(
            f"calib[{tag}].bits out of range "
            f"[{int(side['bits'].min())}, {int(side['bits'].max())}]"
        )


def load_rotated_quant_dsv4_calibration(
    path: str,
    layer_num: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    compression_ratios: List[int],
) -> Dict[int, RotatedQuantizerConfig]:
    """加载 DSv4 模式 calib.pt.

    Schema::

        {
            "_meta": {
                "mode": "dsv4",
                "qk_nope_head_dim": int,
                "qk_rope_head_dim": int,
                "compression_ratios": [int, ...],
                "layer_num": int,
            },
            layer_id (int): {"nope": {R, bits, scale, zero}},
            ...
        }

    返回 ``{layer_id: cfg_nope}``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"rotated-kv-quant-config not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"calibration file must be dict, got {type(raw)}")

    meta = raw.get("_meta")
    if meta is None or meta.get("mode") != "dsv4":
        raise ValueError(
            "calibration file is not in DSv4 mode; expected _meta.mode='dsv4'. "
            "Use --dsv4-mode when running build_rotated_kv_calib.py."
        )
    if int(meta.get("qk_nope_head_dim", -1)) != qk_nope_head_dim:
        raise ValueError(
            f"calib qk_nope_head_dim={meta.get('qk_nope_head_dim')} != "
            f"model {qk_nope_head_dim}"
        )
    if int(meta.get("qk_rope_head_dim", -1)) != qk_rope_head_dim:
        raise ValueError(
            f"calib qk_rope_head_dim={meta.get('qk_rope_head_dim')} != "
            f"model {qk_rope_head_dim}"
        )
    calib_ratios = list(meta.get("compression_ratios", []))
    if calib_ratios and calib_ratios != list(compression_ratios):
        logger.warning(
            "DSv4 calib compression_ratios %s != model %s; "
            "calibration will be applied per-layer regardless.",
            calib_ratios,
            list(compression_ratios),
        )

    out: Dict[int, RotatedQuantizerConfig] = {}
    for lid in range(layer_num):
        if lid not in raw:
            raise ValueError(f"calibration missing layer_id={lid}")
        entry = raw[lid]
        if "nope" not in entry:
            raise ValueError(f"calib[{lid}] missing 'nope'")
        _validate_side(entry["nope"], qk_nope_head_dim, f"layer {lid} nope")
        out[lid] = RotatedQuantizerConfig(
            R=entry["nope"]["R"].to(torch.float32),
            bits=entry["nope"]["bits"].to(torch.int32),
            scale=entry["nope"]["scale"].to(torch.float32),
            zero=entry["nope"]["zero"].to(torch.float32),
        )
    return out


# ----------------------------------------------------------------------
# Pool 包装类
# ----------------------------------------------------------------------
Mode = Literal["eval", "wall"]


# DSv4 native FP8 layout: 584 bytes/token, page padded to multiple of 576.
_DSV4_NATIVE_BPT = 584
_DSV4_SLOT_BYTES = 576
_DSV4_NOPE_FP8_BYTES = 448
_DSV4_ROPE_BF16_BYTES = 128
_DSV4_SCALES_PER_TOKEN = 8


def _native_bytes_per_page(page_size: int) -> int:
    """DSv4 native paged FP8 layout: ``ceil(584 * P / 576) * 576``."""
    return ((_DSV4_NATIVE_BPT * page_size + _DSV4_SLOT_BYTES - 1) //
            _DSV4_SLOT_BYTES) * _DSV4_SLOT_BYTES


class _WallPoolEntry:
    """Per-pool wall-storage state.

    Holds the packed paged buffers (real KV storage) plus a shadow
    paged buffer in DSv4-native FP8 layout, ready for FlashMLA. The
    attention prologue dequants packed -> writes shadow per-batch.
    """

    __slots__ = (
        "kind",
        "pool",
        "packed_buffers",
        "shadow_buffers",
        "packed_bpt",
        "packed_bytes_per_page",
        "shadow_bytes_per_page",
        "page_size",
        "num_pages",
    )

    def __init__(
        self,
        kind: str,
        pool: DeepSeekV4SingleKVPool,
        packed_buffers: List[torch.Tensor],
        shadow_buffers: List[torch.Tensor],
        packed_bpt: int,
        packed_bytes_per_page: int,
        shadow_bytes_per_page: int,
        page_size: int,
        num_pages: int,
    ):
        self.kind = kind
        self.pool = pool
        self.packed_buffers = packed_buffers
        self.shadow_buffers = shadow_buffers
        self.packed_bpt = packed_bpt
        self.packed_bytes_per_page = packed_bytes_per_page
        self.shadow_bytes_per_page = shadow_bytes_per_page
        self.page_size = page_size
        self.num_pages = num_pages


class RotatedQuantDeepSeekV4TokenToKVPool(DeepSeekV4TokenToKVPool):
    """DSv4 + 旋转量化 KV pool.

    Args:
        calib_path: DSv4-mode calibration .pt 路径.
        mode: ``'eval'`` (M3.b) 或 ``'wall'`` (M3.c.2).
        其余参数透传给 ``DeepSeekV4TokenToKVPool``.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        calib_path: str,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        mode: Mode = "eval",
    ):
        super().__init__(
            max_num_reqs=max_num_reqs,
            swa_size=swa_size,
            c4_size=c4_size,
            c128_size=c128_size,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            page_size=page_size,
            swa_page_size=swa_page_size,
            dtype=dtype,
            state_dtype=state_dtype,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            indexer_head_dim=indexer_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            compression_ratios=compression_ratios,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_hisparse=enable_hisparse,
        )

        if mode not in ("eval", "wall"):
            raise ValueError(f"unknown mode {mode!r}; expected 'eval'|'wall'")
        self._mode: Mode = mode
        self._calib_path = calib_path
        cfgs = load_rotated_quant_dsv4_calibration(
            calib_path,
            layer_num=layer_num,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            compression_ratios=list(compression_ratios),
        )
        self._nope_cfgs: Dict[int, RotatedQuantizerConfig] = cfgs
        self._nope_quantizers: Dict[int, RotatedQuantizer] = {
            lid: RotatedQuantizer(c) for lid, c in cfgs.items()
        }
        sample = next(iter(cfgs.values()))
        self._sim_row_bytes = (int(sample.bits.sum().item()) + 7) // 8

        # Wall-storage state.
        self._wall_pools: Dict[str, _WallPoolEntry] = {}

        if self._mode == "wall":
            self._install_wall_storage()
            mode_msg = (
                "WALL-STORAGE MODE: swa/c4/c128 main buffers replaced with "
                "INT2/3/4 packed nope + raw BF16 rope; attention prologue "
                "dequants + writes shadow FP8 buffers in DSv4-native layout "
                "for FlashMLA. Indexer / compress_state remain native FP8."
            )
        else:
            mode_msg = (
                "EVALUATION MODE: underlying DSv4 KV storage unchanged (FP8). "
                "Use simulate_quantize_nope() for offline accuracy evaluation."
            )

        logger.warning(
            "RotatedQuantDeepSeekV4TokenToKVPool active. %s "
            "calib=%s qk_nope_head_dim=%d packed_row_bytes=%d b_mean=%.2f "
            "wall_pools=%s",
            mode_msg,
            calib_path,
            qk_nope_head_dim,
            self._sim_row_bytes,
            float(sample.bits.float().mean()),
            list(self._wall_pools.keys()),
        )

    # ------------------------------------------------------------------
    # Wall-storage installation (M3.c.2: swa + c4 + c128)
    # ------------------------------------------------------------------
    def _install_wall_storage(self) -> None:
        """Replace ``swa/c4/c128 kv_buffer`` with packed bytes_per_page,
        and allocate matching shadow buffers in DSv4-native FP8 layout.

        Lifetime model: packed buffers are the **canonical** KV storage.
        Shadow buffers are touched on demand by the attention prologue
        with the M tokens needed by the current batch. No-op if a pool
        has 0 layers (e.g. PP shard without c128 layers).
        """
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            packed_bytes_per_token,
        )

        bpt_packed = packed_bytes_per_token(self._sim_row_bytes)
        self._wall_bpt = bpt_packed

        for kind, pool in (
            ("swa", self.swa_kv_pool),
            ("c4", self.c4_kv_pool),
            ("c128", self.c128_kv_pool),
        ):
            if pool is None or pool.layer_num <= 0:
                continue
            old_buffers = pool.kv_buffer
            if not old_buffers:
                continue
            num_pages = old_buffers[0].shape[0]
            device = old_buffers[0].device
            page_size = pool.page_size
            packed_bytes_per_page = bpt_packed * page_size
            shadow_bytes_per_page = _native_bytes_per_page(page_size)

            # Drop the existing FP8 buffers; reallocate as packed + shadow.
            pool.kv_buffer = []  # type: ignore[assignment]
            del old_buffers
            if device.type == "cuda":
                torch.cuda.empty_cache()

            packed_buffers = [
                torch.zeros(
                    num_pages,
                    packed_bytes_per_page,
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(pool.layer_num)
            ]
            # Shadow stays zero-initialised; prologue refills before each
            # FlashMLA call using the indices it will read.
            shadow_buffers = [
                torch.zeros(
                    num_pages,
                    shadow_bytes_per_page,
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(pool.layer_num)
            ]

            # The pool's main kv_buffer is now the PACKED storage. Reading
            # via super().get_swa_key_buffer_radix would return packed bytes
            # to FlashMLA -- our overrides redirect to shadow_buffers.
            pool.kv_buffer = packed_buffers  # type: ignore[assignment]

            # IMPORTANT: keep ``pool.bytes_per_page_padded`` at the FP8 size
            # because downstream code (e.g. backend.forward reshapes the
            # buffer to ``[num_pages, P, 1, 584]``) reads this attribute via
            # ``kv_cache_total_dim`` and assumes the native layout. The
            # SHADOW buffer matches that shape exactly.
            entry = _WallPoolEntry(
                kind=kind,
                pool=pool,
                packed_buffers=packed_buffers,
                shadow_buffers=shadow_buffers,
                packed_bpt=bpt_packed,
                packed_bytes_per_page=packed_bytes_per_page,
                shadow_bytes_per_page=shadow_bytes_per_page,
                page_size=page_size,
                num_pages=num_pages,
            )
            self._wall_pools[kind] = entry

        if "swa" in self._wall_pools:
            swa_entry = self._wall_pools["swa"]
            self._wall_bytes_per_page = swa_entry.packed_bytes_per_page
        else:
            self._wall_bytes_per_page = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Map absolute layer_id to local index of swa_kv_pool.kv_buffer.

        Parent class (DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused
        and ...norm_rope) uses ``self.swa_kv_pool.kv_buffer[layer_id]``
        directly. ``swa_kv_pool`` is a DeepSeekV4SingleKVPool sized by
        ``layer_num`` (full layer count), so the absolute layer_id is the
        correct index. Mirror parent's behavior verbatim.
        """
        return layer_id

    def _layer_id_for_extra(self, layer_id: int) -> Tuple[str, int]:
        """Map absolute layer_id to (pool_kind in {'c4','c128'}, local_layer)."""
        compress_ratio, compress_layer_id, compress_kv_pool = self.layer_mapping[
            layer_id
        ]
        assert compress_kv_pool is not None, (
            f"layer {layer_id} ratio={compress_ratio} has no compress_kv_pool"
        )
        if compress_ratio == 4:
            return "c4", compress_layer_id
        if compress_ratio == 128:
            return "c128", compress_layer_id
        raise ValueError(
            f"unsupported compress_ratio {compress_ratio} on layer {layer_id}"
        )

    def _wall_kv_input(self, kv: torch.Tensor) -> torch.Tensor:
        """Ensure shape ``[N, 512]`` BF16 (cat(nope, rope)) for the packer."""
        if kv.dtype != torch.bfloat16:
            kv = kv.to(torch.bfloat16)
        if kv.dim() != 2 or kv.shape[-1] != 512:
            raise ValueError(
                f"wall packer expects [N, 512] bf16, got {tuple(kv.shape)} "
                f"{kv.dtype}"
            )
        return kv.contiguous()

    # ------------------------------------------------------------------
    # Wall-mode write overrides
    # ------------------------------------------------------------------
    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self._mode != "wall":
            return super().set_swa_key_buffer_radix_fused(layer_id, raw_loc, cache_k)
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            rotated_store_to_packed,
        )

        # Mirror the parent's translation+cache logic
        # (DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused).
        if self._should_cache_swa:
            if layer_id == 0:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        local_layer_id = self._swa_local_layer_id(layer_id)
        cfg = self._nope_cfgs[layer_id]
        rotated_store_to_packed(
            self._wall_kv_input(cache_k),
            self.swa_kv_pool.kv_buffer[local_layer_id],
            swa_loc,
            page_size=self.swa_kv_pool.page_size,
            cfg=cfg,
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if self._mode != "wall":
            return super().set_swa_key_buffer_radix_fused_norm_rope(
                layer_id, raw_loc, kv, kv_weight, eps, freqs_cis, positions
            )
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            rotated_store_to_packed,
        )

        # Mirror the parent's translation+cache logic
        # (DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope).
        if self._should_cache_swa:
            if layer_id == self.start_layer or self.cached_loc is None:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        local_layer_id = self._swa_local_layer_id(layer_id)
        cfg = self._nope_cfgs[layer_id]

        # Reproduce the fused norm+rope computation in PyTorch; M3.c.3
        # will replace this with a Triton kernel that also writes packed.
        # NOTE: must mirror jit_kernel/csrc/.../main_norm_rope.cuh exactly:
        # RMSNorm + kv_weight act on the FULL kHeadDim=512 vector; only the
        # last kRopeDim=64 lanes get rotated by RoPE afterwards.
        kv_f = kv.to(torch.float32)
        var = kv_f.pow(2).mean(dim=-1, keepdim=True)
        kv_norm = (kv_f * torch.rsqrt(var + eps)).to(kv.dtype) * kv_weight
        nope_norm = kv_norm[..., : self.qk_nope_head_dim]
        rope_norm = kv_norm[..., self.qk_nope_head_dim :]
        # Apply RoPE on rope half. freqs_cis is complex64 [max_pos, rope_dim/2];
        # gather per position then compute rotated rope = view_as_real(rope_complex * freqs).
        rope_dim = rope_norm.shape[-1]
        rope_complex = rope_norm.float().reshape(
            *rope_norm.shape[:-1], rope_dim // 2, 2
        )
        rope_complex = torch.view_as_complex(rope_complex.contiguous())
        freqs = freqs_cis.index_select(0, positions.to(torch.long))
        rope_rotated = (
            torch.view_as_real(rope_complex * freqs)
            .reshape(*rope_norm.shape[:-1], rope_dim)
            .to(kv.dtype)
        )
        cat = torch.cat([nope_norm, rope_rotated], dim=-1)
        rotated_store_to_packed(
            self._wall_kv_input(cat),
            self.swa_kv_pool.kv_buffer[local_layer_id],
            swa_loc,
            page_size=self.swa_kv_pool.page_size,
            cfg=cfg,
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self._mode != "wall":
            return super().set_extra_key_buffer_fused(layer_id, loc, cache_k)
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            rotated_store_to_packed,
        )

        kind, local_layer_id = self._layer_id_for_extra(layer_id)
        entry = self._wall_pools[kind]
        cfg = self._nope_cfgs[layer_id]
        rotated_store_to_packed(
            self._wall_kv_input(cache_k),
            entry.packed_buffers[local_layer_id],
            loc,
            page_size=entry.page_size,
            cfg=cfg,
        )

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ) -> None:
        if self._mode != "wall":
            return super().set_extra_key_buffer(
                layer_id, loc, cache_nope_fp8_rope_bf16_pack
            )
        # The non-fused path takes a NopeFp8RopeBf16Pack; in wall mode we
        # never use it (the model is configured with FUSED_STORE_CACHE=True).
        # Keep an explicit error so silent corruption is impossible.
        raise NotImplementedError(
            "wall mode requires SGLANG_OPT_USE_FUSED_STORE_CACHE=true; "
            "non-fused set_extra_key_buffer is not supported."
        )

    # ------------------------------------------------------------------
    # Wall-mode read overrides + attention prologue
    # ------------------------------------------------------------------
    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        if self._mode != "wall":
            return super().get_swa_key_buffer_radix(layer_id)
        self.wait_layer_transfer(layer_id)
        local_layer_id = self._swa_local_layer_id(layer_id)
        return self._wall_pools["swa"].shadow_buffers[local_layer_id]

    def get_extra_key_buffer(self, layer_id: int):
        if self._mode != "wall":
            return super().get_extra_key_buffer(layer_id)
        self.wait_layer_transfer(layer_id)
        kind, local_layer_id = self._layer_id_for_extra(layer_id)
        return self._wall_pools[kind].shadow_buffers[local_layer_id]

    def _refresh_shadow_pages(
        self,
        entry: _WallPoolEntry,
        local_layer_id: int,
        cfg: RotatedQuantizerConfig,
        page_indices: torch.Tensor,
    ) -> None:
        """Dequant all valid (page_idx >= 0) tokens in ``page_indices`` from
        the packed buffer of ``entry`` into its shadow buffer. Writes happen
        per-token directly into the shadow's [page * shadow_bytes_per_page +
        slot * 576 :] regions plus the per-tile UE8M0 scale tail.

        Per-batch refresh (no caching): correctness > perf for M3.c.2.
        """
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            packed_bytes_per_token,
            rotated_load_to_fp8_layout,
            rotated_load_to_fp8_layout_cpu_ref,
            _MLA_NOPE_DIM,
            _MLA_TILE_SIZE,
            _MLA_SLOT_BYTES,
            _MLA_SCALES_PER_TOKEN,
        )

        if page_indices.numel() == 0:
            return

        device = entry.shadow_buffers[local_layer_id].device
        page_size = entry.page_size

        # Flatten page_indices, drop sentinels (-1).
        flat_pages = page_indices.reshape(-1).to(torch.int64)
        flat_pages = flat_pages[flat_pages >= 0]
        if flat_pages.numel() == 0:
            return
        # Deduplicate to avoid redundant work when the same page is hit
        # from multiple queries.
        flat_pages = torch.unique(flat_pages)

        # Build the flat token-loc list: for each unique page, all P slots.
        # loc = page * page_size + slot
        slot_range = torch.arange(page_size, device=device, dtype=torch.int64)
        loc = (flat_pages.to(device).unsqueeze(1) * page_size +
               slot_range.unsqueeze(0)).reshape(-1).to(torch.int32)

        cache = entry.packed_buffers[local_layer_id]
        M = loc.numel()

        # On GPU we use the Triton dequant; on CPU (test path) use the
        # pure-PyTorch reference + manual FP8 quant of the BF16 nope.
        if cache.device.type == "cuda":
            try:
                import triton  # noqa: F401
                use_triton = True
            except ImportError:
                use_triton = False
        else:
            use_triton = False

        if use_triton:
            out_slot = torch.empty(
                (M, _MLA_SLOT_BYTES), dtype=torch.uint8, device=device
            )
            out_scale = torch.empty(
                (M, _MLA_SCALES_PER_TOKEN), dtype=torch.uint8, device=device
            )
            rotated_load_to_fp8_layout(
                cache, loc, out_slot, out_scale,
                page_size=page_size, cfg=cfg,
            )
        else:
            from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
                quant_fp8_layout_cpu_ref,
            )
            nope_bf16, rope_bf16, _ = rotated_load_to_fp8_layout_cpu_ref(
                cache, loc, page_size=page_size, cfg=cfg,
            )
            out_slot, out_scale = quant_fp8_layout_cpu_ref(nope_bf16, rope_bf16)

        # Scatter into shadow: for each (page, slot), write 576 value bytes
        # at offset slot * 576 plus 8 scale bytes at (page_size * 576) +
        # slot * 8 within the page.
        shadow = entry.shadow_buffers[local_layer_id]
        # Reshape shadow page region into (num_pages, page_size, 576) for
        # the value half and (num_pages, page_size, 8) for the scale half.
        bytes_per_page = entry.shadow_bytes_per_page
        # slot value region: page bytes [0, page_size * 576)
        # scale region:      page bytes [page_size * 576, page_size * 584)
        # The remaining bytes (if bytes_per_page > page_size * 584) are pad.
        num_pages = entry.num_pages
        # We use an explicit loop over unique pages to keep the index
        # arithmetic simple and CUDA-graph-safe (no dynamic shapes).
        # Per-page block size is constant (page_size * 576 + page_size * 8).
        flat_pages_cpu = flat_pages.to("cpu").tolist()
        out_slot_view = out_slot.reshape(len(flat_pages_cpu), page_size, _MLA_SLOT_BYTES)
        out_scale_view = out_scale.reshape(len(flat_pages_cpu), page_size, _MLA_SCALES_PER_TOKEN)
        for i, page in enumerate(flat_pages_cpu):
            page_buf = shadow[page]
            value_region = page_buf[:page_size * _MLA_SLOT_BYTES].view(
                page_size, _MLA_SLOT_BYTES
            )
            scale_region = page_buf[
                page_size * _MLA_SLOT_BYTES :
                page_size * _MLA_SLOT_BYTES + page_size * _MLA_SCALES_PER_TOKEN
            ].view(page_size, _MLA_SCALES_PER_TOKEN)
            value_region.copy_(out_slot_view[i])
            scale_region.copy_(out_scale_view[i])

    def _rotated_quant_attention_prologue(
        self,
        layer_id: int,
        core_attn_metadata,
        compress_ratio: int,
    ) -> None:
        """Refill shadow FP8 buffers for the pages this batch will read.

        Called by ``DeepseekV4AttnBackend.forward`` (and similar paths)
        right after ``store_cache`` and before
        ``get_swa_key_buffer_radix`` / ``get_extra_key_buffer``.

        Args:
            layer_id: absolute layer id.
            core_attn_metadata: ``DSV4AttnMetadata``-like object exposing
                ``swa_page_indices`` plus c4/c128 sparse page indices.
            compress_ratio: ``0|4|128`` -- selects which extra pool to also
                refresh (0 means swa-only).
        """
        if self._mode != "wall":
            return
        cfg = self._nope_cfgs[layer_id]

        swa_pages = getattr(core_attn_metadata, "swa_page_indices", None)
        if swa_pages is not None and "swa" in self._wall_pools:
            entry = self._wall_pools["swa"]
            local_layer_id = self._swa_local_layer_id(layer_id)
            self._refresh_shadow_pages(entry, local_layer_id, cfg, swa_pages)

        if compress_ratio == 4:
            extra_pages = getattr(
                core_attn_metadata, "c4_sparse_page_indices", None
            )
        elif compress_ratio == 128:
            extra_pages = getattr(core_attn_metadata, "c128_page_indices", None)
        else:
            extra_pages = None

        if extra_pages is not None and compress_ratio in (4, 128):
            kind, local_layer_id = self._layer_id_for_extra(layer_id)
            entry = self._wall_pools[kind]
            self._refresh_shadow_pages(entry, local_layer_id, cfg, extra_pages)

    # ------------------------------------------------------------------
    # Wall-mode read API (M3.c.1 carry-over for unit tests / debugging)
    # ------------------------------------------------------------------
    def dequant_swa_to_fp8_layout(
        self,
        layer_id: int,
        loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather + dequant + inverse-rotate -> standard DSv4 FP8 slot bytes.

        Returns ``(out_slot[M, 576] uint8, out_scale[M, 8] uint8)``. This
        is the per-token form used by canary tests; the production path
        is :meth:`_rotated_quant_attention_prologue` which fills shadow
        pages in bulk.
        """
        if self._mode != "wall":
            raise RuntimeError(
                "dequant_swa_to_fp8_layout only valid in wall mode"
            )
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            rotated_load_to_fp8_layout,
            rotated_load_to_fp8_layout_cpu_ref,
            _MLA_SCALES_PER_TOKEN,
            _MLA_SLOT_BYTES,
        )

        cfg = self._nope_cfgs[layer_id]
        local_layer_id = self._swa_local_layer_id(layer_id)
        entry = self._wall_pools["swa"]
        cache = entry.packed_buffers[local_layer_id]
        M = loc.shape[0]
        if cache.device.type == "cuda":
            try:
                import triton  # noqa: F401
                use_triton = True
            except ImportError:
                use_triton = False
        else:
            use_triton = False

        if use_triton:
            out_slot = torch.empty(
                (M, _MLA_SLOT_BYTES), dtype=torch.uint8, device=cache.device
            )
            out_scale = torch.empty(
                (M, _MLA_SCALES_PER_TOKEN), dtype=torch.uint8, device=cache.device
            )
            rotated_load_to_fp8_layout(
                cache, loc, out_slot, out_scale,
                page_size=entry.page_size, cfg=cfg,
            )
        else:
            from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
                quant_fp8_layout_cpu_ref,
            )
            nope_bf16, rope_bf16, _ = rotated_load_to_fp8_layout_cpu_ref(
                cache, loc, page_size=entry.page_size, cfg=cfg,
            )
            out_slot, out_scale = quant_fp8_layout_cpu_ref(nope_bf16, rope_bf16)
        return out_slot, out_scale

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def wall_bytes_per_page(self) -> Optional[int]:
        return getattr(self, "_wall_bytes_per_page", None)

    # ------------------------------------------------------------------
    # Eval-mode interface (M3.b)
    # ------------------------------------------------------------------
    def simulate_quantize_nope(
        self,
        layer_id: int,
        nope: torch.Tensor,
        return_packed: bool = False,
    ):
        if layer_id not in self._nope_quantizers:
            raise KeyError(
                f"no quantizer for layer_id={layer_id}; "
                f"available={sorted(self._nope_quantizers)}"
            )
        q = self._nope_quantizers[layer_id]
        nope_f32 = nope.detach().to(torch.float32).cpu()
        packed = q.quantize(nope_f32)
        nope_dq = q.dequantize(packed, dtype=torch.float32)
        nope_dq = nope_dq.to(device=nope.device, dtype=nope.dtype)
        if return_packed:
            return nope_dq, packed
        return nope_dq

    def simulated_compression_ratio(self) -> float:
        bf16_bytes = self.qk_nope_head_dim * 2
        return float(bf16_bytes) / float(self._sim_row_bytes)

    def wall_compression_ratio(self) -> Optional[float]:
        """Wall-mode compression ratio = native bytes / packed bpt.

        Reflects the swa/c4/c128 main-storage savings (M3.c.2 covers all
        three). Indexer / compress_state are unchanged.
        """
        if self._mode != "wall":
            return None
        return float(_DSV4_NATIVE_BPT) / float(self._wall_bpt)


# ----------------------------------------------------------------------
# CPU reference: BF16 nope/rope -> DSv4 FP8 slot bytes is now provided by
# ``sglang.jit_kernel.rotated_quant_dsv4_kernels.quant_fp8_layout_cpu_ref``.
# Re-export here for back-compat with anything that imported the private
# helper from this module.
# ----------------------------------------------------------------------
from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
    quant_fp8_layout_cpu_ref as _quant_fp8_layout_cpu_ref,  # noqa: F401
)


__all__ = [
    "RotatedQuantDeepSeekV4TokenToKVPool",
    "load_rotated_quant_dsv4_calibration",
]
