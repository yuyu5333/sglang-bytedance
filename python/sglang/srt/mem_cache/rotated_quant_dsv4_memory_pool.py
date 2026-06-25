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


def _wall_token_shadow_enabled() -> bool:
    """Token-level shadow mirror mode (fix for token salad).

    When ``SGLANG_RQ_WALL_TOKEN_SHADOW=1``, the store path additionally
    quantizes the BF16 input to FP8+UE8M0 layout and scatters those bytes
    *per token* into the shadow buffer. The attention prologue then skips
    the page-level refresh (which used to fill garbage from
    ``dequant(packed=0) = 0*scale + zero @ R.t() != 0`` into never-written
    slots, corrupting FlashMLA reads).

    The packed buffer is still maintained (for offline parity / future
    eviction) but its content is not read back at attention time when this
    flag is on. BYPASS_QUANT remains an orthogonal diagnostic switch.
    """
    return os.environ.get("SGLANG_RQ_WALL_TOKEN_SHADOW", "0") == "1"


def _wall_drop_packed_enabled() -> bool:
    """Drop the packed buffer entirely (memory milestone path).

    When ``SGLANG_RQ_WALL_DROP_PACKED=1`` (only valid when
    ``SGLANG_RQ_WALL_TOKEN_SHADOW=1``):

    * ``packed_buffers`` is **not allocated** (saves ~38% bytes/token vs
      native FP8 in the previous wall config; **eliminates** the +49%
      reverse-overhead the dual-buffer architecture used to pay).
    * Store paths skip the ``rotated_store_to_packed`` call entirely.
    * The ``set_swa_key_buffer_radix_fused_norm_rope`` PyTorch fp32
      fused-norm-rope fallback is replaced with the native CUDA
      ``fused_k_norm_rope_flashmla`` kernel writing directly to the
      shadow buffer — capture-safe and identical to baseline FP8 path.
    * ``pool.kv_buffer`` is aliased to ``shadow_buffers`` so any parent
      access pattern sees a sane FP8-layout buffer.

    This is the production milestone configuration: equivalent baseline
    accuracy, full cudagraph perf, zero memory overhead vs FP8 baseline
    (modulo a single calib.pt of fp32 weights that's already cheap).
    """
    return (
        os.environ.get("SGLANG_RQ_WALL_DROP_PACKED", "0") == "1"
        and _wall_token_shadow_enabled()
    )


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
        # T3 优化: 每个 local-layer 一个 [num_pages] bool tensor，
        # True 表示该页 packed 已被写过但 shadow 还没刷新；
        # _refresh_shadow_pages 只对 dirty=True 的页做 bitunpack，
        # 刷完清掉 dirty。store 路径将写入的 page mark dirty=True。
        "dirty_pages",
        # T_cgraph_safe: 跨 forward 复用的 staging buffer，避免在
        # cudagraph capture 期间反复 ``torch.empty`` 临时 tensor 导致
        # replay 时指针不稳定。形状 ``[max_tokens_per_forward, ...]``。
        "staging_slot",
        "staging_scale",
        # T_packed_only: per-(page, slot) 是否被 store 写过的 mask。
        # _refresh_shadow_pages dequant 之后，invalid slot (False) 的
        # out_slot / out_scale 字节强置 0，避免 dequant(packed=0) =
        # zero @ R.t() ≠ 0 的 garbage 写入 shadow 污染 FlashMLA。
        # 形状: List[Tensor[num_pages, page_size] bool] × layer_num。
        # 内存开销: num_pages × page_size × 1B × layer_num；DSv4
        # SWA 池 ≈ 152 MB total（可忽略 vs shadow ≈ 23 GB）。
        "valid_slots",
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
        dirty_pages: List[torch.Tensor],
        staging_slot: torch.Tensor,
        staging_scale: torch.Tensor,
        valid_slots: List[torch.Tensor],
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
        self.dirty_pages = dirty_pages
        self.staging_slot = staging_slot
        self.staging_scale = staging_scale
        self.valid_slots = valid_slots


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
            "wall_pools=%s token_shadow=%s drop_packed=%s",
            mode_msg,
            calib_path,
            qk_nope_head_dim,
            self._sim_row_bytes,
            float(sample.bits.float().mean()),
            list(self._wall_pools.keys()),
            _wall_token_shadow_enabled(),
            _wall_drop_packed_enabled(),
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

        Env knob ``SGLANG_RQ_WALL_KINDS`` (comma-separated subset of
        ``swa,c4,c128``; default ``swa,c4,c128``) controls which sub-pools
        are placed under wall storage. **Diagnostic use**: c4/c128 store
        compressor outputs whose distribution differs from the raw
        post-RMSNorm KV used to build calibration; when calib was built
        from a SWA-only kv-dump, applying the same per-layer (R,bits,
        scale,zero) to c4/c128 is statistically wrong. Setting this to
        ``swa`` keeps SWA on the packed/shadow path while c4/c128 stay on
        native FP8, isolating SWA pipeline correctness from calib
        mismatch on the compressor side.
        """
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            packed_bytes_per_token,
        )

        bpt_packed = packed_bytes_per_token(self._sim_row_bytes)
        self._wall_bpt = bpt_packed

        env_kinds = os.environ.get("SGLANG_RQ_WALL_KINDS", "swa,c4,c128")
        # Special token "none" / empty string => keep ALL pools on native FP8;
        # used as a clean baseline to verify the wrapper itself doesn't
        # corrupt parent FP8 storage.
        if env_kinds.strip().lower() in ("", "none"):
            wall_kinds: set[str] = set()
        else:
            wall_kinds = {
                k.strip() for k in env_kinds.split(",") if k.strip()
            }
            for k in wall_kinds:
                if k not in ("swa", "c4", "c128"):
                    raise ValueError(
                        f"SGLANG_RQ_WALL_KINDS contains unknown kind {k!r}; "
                        f"allowed: swa,c4,c128 (or 'none' for native FP8)"
                    )
        logger.warning(
            "wall-storage install scope: SGLANG_RQ_WALL_KINDS=%s",
            sorted(wall_kinds),
        )

        for kind, pool in (
            ("swa", self.swa_kv_pool),
            ("c4", self.c4_kv_pool),
            ("c128", self.c128_kv_pool),
        ):
            if kind not in wall_kinds:
                continue
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
            ] if not _wall_drop_packed_enabled() else [
                # Drop-packed mode: no allocation. Use a 1-byte sentinel
                # so any code path that still indexes packed_buffers[i]
                # gets a clear shape error instead of silent bad reads.
                torch.zeros(1, dtype=torch.uint8, device=device)
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

            # T3 优化: dirty_pages mask, 初始全 True 表示首次都要冷启动刷新。
            # store 路径将写过的 page idx 标 True; refresh 后清零。
            dirty_pages = [
                torch.ones(num_pages, dtype=torch.bool, device=device)
                for _ in range(pool.layer_num)
            ]

            # T_packed_only: per-(page, slot) valid mask。初始全 False
            # （所有 slot 未写过）。store 路径调 _mark_slots_valid_from_loc
            # 标 True；prologue dequant 之后用此 mask 把 invalid slot
            # 的字节强置 0，与 baseline 未写 FP8 buffer 等价。
            valid_slots = [
                torch.zeros(
                    num_pages, page_size,
                    dtype=torch.bool, device=device,
                )
                for _ in range(pool.layer_num)
            ]

            # T_cgraph_safe: 跨 forward 复用 staging buffer，避免
            # _write_tokens_to_shadow 在 cudagraph capture/replay 期间
            # 反复 ``torch.empty`` 临时 tensor 导致指针不稳定。
            # 容量上限 = num_pages * page_size = 池中所有 slot 数（一次
            # forward 写入 token 数不可能超过这个上限）。
            staging_capacity = num_pages * page_size
            staging_slot = torch.zeros(
                staging_capacity, _DSV4_SLOT_BYTES,
                dtype=torch.uint8, device=device,
            )
            staging_scale = torch.zeros(
                staging_capacity, _DSV4_SCALES_PER_TOKEN,
                dtype=torch.uint8, device=device,
            )

            # The pool's main kv_buffer is now the PACKED storage. Reading
            # via super().get_swa_key_buffer_radix would return packed bytes
            # to FlashMLA -- our overrides redirect to shadow_buffers.
            # drop_packed 模式下 packed_buffers 是 1B 占位，必须把
            # pool.kv_buffer 别名到 shadow_buffers，否则任何走 super()
            # 路径或 attention backend 直接 reshape kv_buffer 的代码会
            # 拿到错的 shape，触发 OOB 读写。
            if _wall_drop_packed_enabled():
                pool.kv_buffer = shadow_buffers  # type: ignore[assignment]
            else:
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
                dirty_pages=dirty_pages,
                staging_slot=staging_slot,
                staging_scale=staging_scale,
                valid_slots=valid_slots,
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

    def wait_layer_transfer(self, layer_id: int) -> None:
        """No-op layer-transfer barrier for wall mode.

        Wall packed_buffers are written by our own store overrides on the
        same CUDA stream as attention; shadow_buffers are refreshed on
        demand by the prologue. Neither participates in the parent's
        layer_transfer_counter pipeline (hicache / async H2D), so there
        is nothing to wait on. If hicache support is added later this
        method can be promoted to wait on swa_kv_pool's counter.
        """
        return

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
    # T3 dirty-page tracking
    # ------------------------------------------------------------------
    def _mark_pages_dirty_from_loc(
        self,
        entry: _WallPoolEntry,
        local_layer_id: int,
        loc: torch.Tensor,
    ) -> None:
        """根据 store 路径写入的 token-loc，把对应的 page mark dirty=True。

        ``loc`` 是 ``[N]`` int32 token slot 索引，含 -1 sentinel 表示未映射；
        page_idx = loc // page_size。**完全 GPU 化、无同步**。

        注意: 不预先 mask -1。loc=-1 → page_idx=-1 → 取 abs / clamp_min(0) 落
        到 page 0（page 0 反正也被冷启动 dirty=True，多刷无损），换取无
        ``.all().item()`` 同步。
        """
        # T_cgraph_safe: token_shadow 模式下 prologue 整体跳过 page-level
        # refresh，dirty bit 是 dead code。提前 short-circuit 既避免了
        # 不必要的 ``loc.to(int64)`` 临时 tensor（cudagraph capture/replay
        # 不稳定），又略提速。
        if _wall_token_shadow_enabled():
            return
        if loc.numel() == 0:
            return
        dirty = entry.dirty_pages[local_layer_id]
        loc_i64 = loc.to(dtype=torch.int64, device=dirty.device).reshape(-1)
        page_idx = (loc_i64 // entry.page_size).clamp_(0, entry.num_pages - 1)
        dirty.index_fill_(0, page_idx, True)

    def _mark_slots_valid_from_loc(
        self,
        entry: _WallPoolEntry,
        local_layer_id: int,
        loc: torch.Tensor,
    ) -> None:
        """T_packed_only: 标记 store 真正写入过的 (page, slot) 为 valid=True。

        prologue dequant 后用 valid_slots mask 把 invalid slot 的字节
        强置 0，避免 ``dequant(packed=0) = zero @ R.t() ≠ 0`` 的 garbage
        污染 shadow / FlashMLA 读路径。

        ``loc`` 是 store 的 [N] int32 token slot 索引，含 -1 sentinel；
        slot 落在 page=loc//P, off=loc%P。-1 sentinel 用 clamp_min(0) 落
        到 (page=0, slot=0)，被 token_shadow 之外的代码忽略不影响正确性
        （valid_slots 覆盖范围内的 slot 一定来自有效 loc）。

        TODO(cgraph): 当前用 ``valid_flat[idx] = True`` advanced indexing，
        在 cuda graph capture 下需要稳定的 idx 指针——Step2 改 Triton
        kernel 时一并迁移到 capture-safe API。Step1 主要验证量化数学
        在 ``--disable-cuda-graph`` 下可达 gsm8k ≥ 0.94。
        """
        if loc.numel() == 0:
            return
        valid = entry.valid_slots[local_layer_id]  # [num_pages, page_size] bool
        loc_i64 = loc.to(dtype=torch.int64, device=valid.device).reshape(-1)
        # 把 -1 sentinel 落到 (0, 0)：scatter True 到 page 0 slot 0；
        # 这个 slot 一旦被真实 token 写过就也是 True（无副作用）。
        loc_i64 = loc_i64.clamp_min_(0)
        page_idx = loc_i64 // entry.page_size
        slot_idx = loc_i64 % entry.page_size
        flat = valid.reshape(-1)
        flat_idx = page_idx * entry.page_size + slot_idx
        flat.index_fill_(0, flat_idx, True)

    # ------------------------------------------------------------------
    # Token-level shadow write: avoid garbage in unwritten slots.
    # ------------------------------------------------------------------
    def _write_tokens_to_shadow(
        self,
        entry: _WallPoolEntry,
        local_layer_id: int,
        kv_bf16_512: torch.Tensor,  # [N, 512] BF16 (cat(nope, rope))
        loc: torch.Tensor,           # [N] int32 token slot ids (with -1 sentinels)
    ) -> None:
        """Quantize [N, 512] BF16 → DSv4 native FP8 slot bytes, scatter into
        ``shadow_buffers[local_layer_id]``.

        Why this exists: dequant(packed=0) ≠ 0 because of ``codes*scale +
        zero`` with non-zero ``zero``. If the prologue refreshes a whole
        page, slots that were never store()'d get filled with garbage and
        FlashMLA reads them.  The fix is to keep shadow as the *authoritative
        per-token mirror*: store path writes a token → shadow gets that
        token's bytes; unwritten slots stay zero (== FP8 numerical 0,
        masked out by attention anyway).
        """
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            _MLA_NOPE_DIM, _MLA_SLOT_BYTES, _MLA_SCALES_PER_TOKEN,
        )
        from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
            rotated_dequant_to_fp8_layout,
            triton_scatter_tokens_to_shadow,
        )
        N = kv_bf16_512.shape[0]
        if N == 0:
            return
        device = kv_bf16_512.device
        nope_bf16 = kv_bf16_512[..., :_MLA_NOPE_DIM].contiguous()
        rope_bf16 = kv_bf16_512[..., _MLA_NOPE_DIM:].contiguous()
        # T_cgraph_safe: 复用 entry 级 staging buffer 而不是 ``torch.empty``，
        # 避免 cudagraph capture 时分配的临时 tensor 在 replay 阶段被
        # caching allocator 复用但 graph kernel binding 仍旧指向 capture
        # 时的地址，导致 replay 时 kernel 写入 stale slot。staging 容量 =
        # num_pages * page_size，前 N 行对应当前 forward。
        out_slot = entry.staging_slot[:N]
        out_scale = entry.staging_scale[:N]
        rotated_dequant_to_fp8_layout(nope_bf16, rope_bf16, out_slot, out_scale)

        shadow = entry.shadow_buffers[local_layer_id]
        page_size = entry.page_size

        # Capture-safe per-token scatter: 每 token 一个 program block，
        # invalid (loc<0) 直接 return 零 read 零 write，彻底消除原
        # PyTorch ``gather + where + scatter_`` 在 cudagraph capture/replay
        # 下的 byte-level race（详见 triton_scatter_tokens_to_shadow 注释）。
        # 关键：直接传原始 loc（int32），避免 ``loc.to(int64)`` 在 cudagraph
        # capture 期间 alloc 临时 int64 tensor —— 这会让 kernel 在 replay
        # 时拿到不稳定指针。Triton kernel 内部会 cast 到 int64。
        loc_flat = loc.reshape(-1)
        if not loc_flat.is_contiguous():
            loc_flat = loc_flat.contiguous()
        triton_scatter_tokens_to_shadow(
            out_slot,
            out_scale,
            loc_flat,
            shadow,
            page_size,
        )


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
        if "swa" not in self._wall_pools:
            return super().set_swa_key_buffer_radix_fused(layer_id, raw_loc, cache_k)
        # BYPASS: NSA-CP path. cache_k is already normed+rope-applied BF16
        # [N, 512]. Quant to FP8+UE8M0 layout via the same path used for
        # the rotated dequant output, then scatter into shadow_buffer.
        if os.environ.get("SGLANG_RQ_WALL_BYPASS_QUANT", "0") == "1":
            from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
                _MLA_NOPE_DIM, _MLA_TILE_SIZE,
                _MLA_SLOT_BYTES, _MLA_SCALES_PER_TOKEN,
            )
            if self._should_cache_swa:
                if layer_id == 0:
                    self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
                swa_loc = self.cached_loc
            else:
                swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
            local_layer_id = self._swa_local_layer_id(layer_id)
            shadow = self._wall_pools["swa"].shadow_buffers[local_layer_id]
            page_size = self.swa_kv_pool.page_size
            ck = self._wall_kv_input(cache_k)
            nope_bf16 = ck[..., :_MLA_NOPE_DIM].contiguous()
            rope_bf16 = ck[..., _MLA_NOPE_DIM:].contiguous()
            M = nope_bf16.shape[0]
            out_slot = torch.empty(
                (M, _MLA_SLOT_BYTES), dtype=torch.uint8, device=ck.device
            )
            out_scale = torch.empty(
                (M, _MLA_SCALES_PER_TOKEN), dtype=torch.uint8, device=ck.device
            )
            from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
                rotated_dequant_to_fp8_layout,
            )
            rotated_dequant_to_fp8_layout(nope_bf16, rope_bf16, out_slot, out_scale)
            # scatter into shadow according to swa_loc
            page_idx = (swa_loc // page_size).to(torch.long)
            slot_idx = (swa_loc % page_size).to(torch.long)
            for i in range(M):
                pi = int(page_idx[i].item())
                si = int(slot_idx[i].item())
                page_buf = shadow[pi]
                page_buf[si * _MLA_SLOT_BYTES:(si + 1) * _MLA_SLOT_BYTES].copy_(
                    out_slot[i]
                )
                scale_off = page_size * _MLA_SLOT_BYTES + si * _MLA_SCALES_PER_TOKEN
                page_buf[scale_off:scale_off + _MLA_SCALES_PER_TOKEN].copy_(
                    out_scale[i]
                )
            return
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
        # drop_packed 模式 packed_buffers 是 1B 占位，跳过 rotated_store_to_packed
        # （写入路径是 dead code：prologue 整体 short-circuit，packed 不会被读）。
        if not _wall_drop_packed_enabled():
            rotated_store_to_packed(
                self._wall_kv_input(cache_k),
                self._wall_pools["swa"].packed_buffers[local_layer_id],
                swa_loc,
                page_size=self.swa_kv_pool.page_size,
                cfg=cfg,
            )
        # T3 token-shadow 模式: 同步写一份 token 级 shadow，避免
        # prologue 整 page refresh 把未写 slot 填成 dequant(0) 的 garbage。
        if _wall_token_shadow_enabled():
            self._write_tokens_to_shadow(
                self._wall_pools["swa"], local_layer_id,
                self._wall_kv_input(cache_k), swa_loc,
            )
        # T3: mark dirty pages so prologue 只刷新被本次写过的页
        self._mark_pages_dirty_from_loc(
            self._wall_pools["swa"], local_layer_id, swa_loc,
        )
        # T_packed_only (β): mark per-(page, slot) valid so prologue
        # dequant 后能把 invalid slot 的 garbage 字节清零。
        self._mark_slots_valid_from_loc(
            self._wall_pools["swa"], local_layer_id, swa_loc,
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
        if "swa" not in self._wall_pools:
            return super().set_swa_key_buffer_radix_fused_norm_rope(
                layer_id, raw_loc, kv, kv_weight, eps, freqs_cis, positions
            )
        # ------------------------------------------------------------------
        # 诊断开关 SGLANG_RQ_WALL_BYPASS_QUANT=1: 完全绕开 packed/shadow
        # 的量化数学链路, 直接调 native fused_k_norm_rope_flashmla 把
        # FP8+UE8M0 写入 shadow_buffer (layout 与原生 FP8 buffer 一致).
        # 如果此模式下输出通顺 -> 证明 shadow buffer 的字节布局/get 路径/
        # FlashMLA 消费均正确, 问题一定出在 packed→shadow 的量化数学;
        # 如果仍 salad -> 问题出在 shadow buffer 的 layout/dtype 或
        # get_swa_key_buffer_radix 替换破坏了 backend 期待的语义.
        #
        # T_milestone (2026-06-17): drop_packed 模式同样直走这条 CUDA
        # kernel 路径（控制实验 #2 已实测 gsm8k 0.955 / tps 1538 在
        # cudagraph 下达成 baseline 等价），消除 PyTorch fp32 fallback 在
        # capture/replay 时产生的临时 tensor 别名问题。
        # ------------------------------------------------------------------
        if (
            os.environ.get("SGLANG_RQ_WALL_BYPASS_QUANT", "0") == "1"
            or _wall_drop_packed_enabled()
        ):
            from sglang.jit_kernel.deepseek_v4 import fused_k_norm_rope_flashmla

            if self._should_cache_swa:
                if layer_id == self.start_layer or self.cached_loc is None:
                    self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
                swa_loc = self.cached_loc
            else:
                swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
            local_layer_id = self._swa_local_layer_id(layer_id)
            shadow = self._wall_pools["swa"].shadow_buffers[local_layer_id]
            fused_k_norm_rope_flashmla(
                kv=kv,
                kv_weight=kv_weight,
                eps=eps,
                freqs_cis=freqs_cis,
                positions=positions,
                out_loc=swa_loc,
                kvcache=shadow,
                page_size=self.swa_kv_pool.page_size,
            )
            return
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

        # T5: Triton fused RMSNorm + RoPE.
        # Produces a single BF16 [N, 512] output with zero Python-side
        # fp32 intermediates (eliminates the prior N*512 pow / rsqrt /
        # mul / nope-split / complex-mul / cast chain, which created
        # ~5x N*512 intermediate tensors and multiple kernel launches).
        # Fallback to the PyTorch chain on CPU or if the Triton kernel
        # is unavailable for any reason.
        if kv.is_cuda:
            from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
                triton_fused_norm_rope,
            )
            cat = triton_fused_norm_rope(
                kv=kv, kv_weight=kv_weight, eps=eps,
                freqs_cis=freqs_cis, positions=positions,
            )
        else:
            # CPU fallback — only used in offline canary tests.
            kv_weight_f = kv_weight.to(torch.float32)
            kv_f = kv.to(torch.float32)
            var = kv_f.pow(2).mean(dim=-1, keepdim=True)
            kv_norm_f = kv_f * torch.rsqrt(var + eps) * kv_weight_f
            nope_norm_f = kv_norm_f[..., : self.qk_nope_head_dim]
            rope_norm_f = kv_norm_f[..., self.qk_nope_head_dim :]
            rope_dim = rope_norm_f.shape[-1]
            rope_complex = rope_norm_f.reshape(
                *rope_norm_f.shape[:-1], rope_dim // 2, 2
            ).contiguous()
            rope_complex = torch.view_as_complex(rope_complex)
            freqs = freqs_cis.index_select(0, positions.to(torch.long))
            rope_rotated_f = torch.view_as_real(
                rope_complex * freqs
            ).reshape(*rope_norm_f.shape[:-1], rope_dim)
            cat = torch.cat([nope_norm_f, rope_rotated_f], dim=-1).to(kv.dtype)
        rotated_store_to_packed(
            self._wall_kv_input(cat),
            self._wall_pools["swa"].packed_buffers[local_layer_id],
            swa_loc,
            page_size=self.swa_kv_pool.page_size,
            cfg=cfg,
        )
        # T3 token-shadow 模式: cat 已经是 norm+rope 后的 BF16 [N, 512]，
        # 直接量化写 shadow，prologue 跳过 page refresh。
        if _wall_token_shadow_enabled():
            self._write_tokens_to_shadow(
                self._wall_pools["swa"], local_layer_id,
                self._wall_kv_input(cat), swa_loc,
            )
        # T3: mark dirty pages so prologue 只刷新被本次写过的页
        self._mark_pages_dirty_from_loc(
            self._wall_pools["swa"], local_layer_id, swa_loc,
        )
        # T_packed_only (β)
        self._mark_slots_valid_from_loc(
            self._wall_pools["swa"], local_layer_id, swa_loc,
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self._mode != "wall":
            return super().set_extra_key_buffer_fused(layer_id, loc, cache_k)
        kind, _ = self._layer_id_for_extra(layer_id)
        if kind not in self._wall_pools:
            # SGLANG_RQ_WALL_KINDS excluded this kind; keep native FP8.
            return super().set_extra_key_buffer_fused(layer_id, loc, cache_k)
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            rotated_store_to_packed,
        )

        kind, local_layer_id = self._layer_id_for_extra(layer_id)
        entry = self._wall_pools[kind]
        cfg = self._nope_cfgs[layer_id]
        # drop_packed 模式跳过 rotated_store_to_packed（packed_buffers 是
        # 1B 占位，prologue 整体 short-circuit，无人读这块字节）。
        if not _wall_drop_packed_enabled():
            rotated_store_to_packed(
                self._wall_kv_input(cache_k),
                entry.packed_buffers[local_layer_id],
                loc,
                page_size=entry.page_size,
                cfg=cfg,
            )
        # T3 token-shadow 模式: 同步写 shadow，绕开 page-level refresh
        # 的 dequant(0) garbage 污染。
        if _wall_token_shadow_enabled():
            self._write_tokens_to_shadow(
                entry, local_layer_id,
                self._wall_kv_input(cache_k), loc,
            )
        # T3: mark dirty pages
        self._mark_pages_dirty_from_loc(entry, local_layer_id, loc)
        # T_packed_only (β)
        self._mark_slots_valid_from_loc(entry, local_layer_id, loc)

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
        if "swa" not in self._wall_pools:
            return super().get_swa_key_buffer_radix(layer_id)
        self.wait_layer_transfer(layer_id)
        local_layer_id = self._swa_local_layer_id(layer_id)
        return self._wall_pools["swa"].shadow_buffers[local_layer_id]

    def get_extra_key_buffer(self, layer_id: int):
        if self._mode != "wall":
            return super().get_extra_key_buffer(layer_id)
        self.wait_layer_transfer(layer_id)
        kind, local_layer_id = self._layer_id_for_extra(layer_id)
        if kind not in self._wall_pools:
            # SGLANG_RQ_WALL_KINDS excluded this kind; native FP8 buffer
            # is intact, fall back to parent.
            return super().get_extra_key_buffer(layer_id)
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

        # Flatten page_indices, drop sentinels (-1) AND out-of-range pages
        # (TP may pass page indices that are valid for the main KV pool
        # but >= entry.num_pages for the packed sub-buffer).
        flat_pages = page_indices.reshape(-1).to(torch.int64)
        max_page = entry.num_pages
        flat_pages = flat_pages[(flat_pages >= 0) & (flat_pages < max_page)]
        if flat_pages.numel() == 0:
            return
        # Deduplicate to avoid redundant work when the same page is hit
        # from multiple queries.
        flat_pages = torch.unique(flat_pages)

        # T3 优化: 只刷新 dirty=True 的页。每个 decode step 真正改写
        # packed buffer 的页只有 1-2 个（新 token 落入的那 1 页），但
        # page_indices 一般覆盖整个 SWA 区域 (10w+ token / 256 pgsz ≈
        # 数百页) 。dirty filter 把 refresh 工作量从 O(pages_in_view) 砍
        # 到 O(pages_written_since_last_refresh)。
        #
        # 启用门槛：env var SGLANG_RQ_DIRTY_FILTER=1 时启用过滤；默认
        # 关闭以排查 dirty-mask 漏刷导致 KV cache 损坏的根因。mark / clear
        # 仍始终执行（无副作用），便于打开开关后无需重启。
        dirty = entry.dirty_pages[local_layer_id]
        if os.environ.get("SGLANG_RQ_DIRTY_FILTER", "0") == "1":
            flat_pages_dev = flat_pages.to(dirty.device)
            page_is_dirty = dirty.index_select(0, flat_pages_dev)
            flat_pages_dev = flat_pages_dev[page_is_dirty]
            if flat_pages_dev.numel() == 0:
                # 全部命中 cache，无需刷新
                return
            flat_pages = flat_pages_dev

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
            # T_packed_only (β) byte-layout fix: must be torch.zeros, not
            # torch.empty. The Triton dequant kernel only writes
            # out_scale[:, :7] (7 nope tiles) and leaves out_scale[:, 7]
            # uninitialized. The native CUDA fused_k_norm_rope_flashmla
            # path leaves shadow's 8th scale byte at its initial value 0
            # (shadow = torch.zeros at install time). When we scatter
            # out_scale into shadow we overwrite all 8 bytes, so the
            # uninitialized 8th byte becomes random garbage in shadow.
            # FlashMLA may load scales as a vec64 and use any non-zero
            # 8th byte as an exponent, yielding 2^(garbage-127) ~ 2^128
            # ⇒ catastrophic numerics ⇒ token salad.
            # Same logic for out_slot: nope kernel writes [:, :448] and
            # rope kernel writes [:, 448:576], so all 576 bytes are set.
            # Keep out_slot = empty for perf; force out_scale = zeros so
            # the 8th byte stays 0 (parity with native FP8 path).
            out_slot = torch.empty(
                (M, _MLA_SLOT_BYTES), dtype=torch.uint8, device=device
            )
            out_scale = torch.zeros(
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

        # T_packed_only (β): apply valid_slots mask. dequant(packed=0) =
        # zero @ R.t() ≠ 0；未被 store 写过的 slot 在 packed_buffer 仍是 0
        # 但 dequant 出来不是 0，会污染 shadow 让 FlashMLA 读到 garbage
        # （token salad 根因）。这里 gather 当前 refresh 的 (page, slot)
        # mask，invalid 行字节清零，等价于 baseline native FP8 buffer 中
        # 未写 slot 的 0 字节，FlashMLA 读到也只是 numerical 0（被
        # attention mask 屏蔽），不会污染输出。
        valid = entry.valid_slots[local_layer_id]  # [num_pages, page_size] bool
        # gather 同一 flat_pages 顺序的 mask: [P_unique, page_size]
        valid_rows = valid.index_select(0, flat_pages.to(valid.device))
        valid_flat = valid_rows.reshape(-1)  # [M]
        invalid_flat = ~valid_flat
        if invalid_flat.any():
            out_slot[invalid_flat] = 0
            out_scale[invalid_flat] = 0

        # T6: Triton fused scatter into shadow — replaces the Python
        # for (page in flat_pages) loop which synchronized to CPU and
        # executed one copy-kernel launch per page (bad for large
        # batches with many pages). A single kernel launch does the
        # full scatter.
        shadow = entry.shadow_buffers[local_layer_id]
        bytes_per_page = entry.shadow_bytes_per_page
        P = int(flat_pages.numel())
        if use_triton and shadow.is_cuda:
            from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
                triton_fused_refresh_shadow_scatter,
            )
            triton_fused_refresh_shadow_scatter(
                out_slot.reshape(P, page_size, _MLA_SLOT_BYTES).contiguous(),
                out_scale.reshape(P, page_size, _MLA_SCALES_PER_TOKEN).contiguous(),
                flat_pages.to(torch.int64).to(shadow.device),
                shadow,
                page_size,
                bytes_per_page,
            )
        else:
            # Python fallback (CPU or tests). Kept byte-equivalent.
            flat_pages_cpu = flat_pages.to("cpu").tolist()
            out_slot_view = out_slot.reshape(P, page_size, _MLA_SLOT_BYTES)
            out_scale_view = out_scale.reshape(P, page_size, _MLA_SCALES_PER_TOKEN)
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

        # T3: clear dirty bits for refreshed pages
        dirty.index_fill_(0, flat_pages.to(dirty.device), False)

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
        # [M3.c.4 Stage-4] cudagraph-safe short-circuit.
        #
        # 当 sparse-path 把 packed_kwargs (packed_kcache + scale + R + zero +
        # dim_of_bit + bitpos_in_dim) 全部传给 FlashMLA 时，kernel 内部走
        # fused bit-unpack + per-dim dequant from packed_kcache (commit
        # d21761c sm90/decode/sparse_fp8/splitkv_mla.cuh use_packed=true
        # 分支)，**完全不读** swa_k_cache / shadow_buffer。此时 prologue
        # 的 page-level shadow refresh 是 dead work：
        #   * _refresh_shadow_pages 含 boolean indexing
        #     (`flat_pages[(flat_pages >= 0) & (flat_pages < max_page)]`),
        #     torch.unique(flat_pages), invalid_flat.any() 三处 capture-
        #     fatal op，是当前必须 `--disable-cuda-graph` 的唯一原因；
        #   * shadow 即使被刷新也无人消费 (sparse_decode.h validator 见 6
        #     个 packed kwargs 全非 None 即 use_packed=true，绕过 shadow)。
        # → 默认整体跳过 prologue (env=1)，捎带把整条 capture-unsafe 链
        #   一并扫掉，让 server 可以去掉 `--disable-cuda-graph` 跑 graph。
        #
        # 退路：SGLANG_RQ_SKIP_SHADOW_REFRESH=0 时退回原 shadow refresh
        # 行为，用于 dense-path 调试 / 回归对照。
        if os.environ.get("SGLANG_RQ_SKIP_SHADOW_REFRESH", "1") == "1":
            return
        # 诊断模式 SGLANG_RQ_WALL_BYPASS_QUANT=1: shadow_buffer 已被 store
        # 路径用 native FP8 kernel 直接写入真值, refresh 会用 packed (全 0)
        # 覆盖它 -> 跳过 SWA refresh; c4/c128 在 bypass 模式下仍走 packed,
        # 它们的 shadow 仍由 packed -> dequant 重建.
        bypass_quant = os.environ.get("SGLANG_RQ_WALL_BYPASS_QUANT", "0") == "1"
        # token-shadow 模式: store 路径已经把每个 token 的 FP8+UE8M0 字节
        # 写进 shadow，此时 page-level refresh 反而会把未写 slot 用
        # dequant(packed=0)=zero@R.t() 的 garbage 覆盖，必须整体跳过
        # （SWA + c4/c128 全部跳）。
        token_shadow = _wall_token_shadow_enabled()
        if token_shadow:
            return
        cfg = self._nope_cfgs[layer_id]

        swa_pages = getattr(core_attn_metadata, "swa_page_indices", None)
        if (
            swa_pages is not None
            and "swa" in self._wall_pools
            and not bypass_quant
        ):
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
            if kind in self._wall_pools:
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
            # See _refresh_shadow_pages for why this must be torch.zeros.
            out_scale = torch.zeros(
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
    # [M3.c.4 Stage-3] FlashMLA sparse-path packed-FP8 kwargs.
    # ------------------------------------------------------------------
    def get_rotated_packed_kwargs(
        self, layer_id: int, kind: str = "swa"
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return a dict suitable for ``flash_mla.flash_mla_with_kvcache``'s
        six packed kwargs, or ``None`` if this pool is not in wall mode /
        ``kind`` is not under wall storage / drop_packed is active (no
        packed buffer to expose).

        Returned dict keys (match the kernel kwargs verbatim)::

            {
                "packed_kcache": uint8[num_rows, packed_row_bytes],
                "scale_kcache":  float32[qk_nope],
                "R_matrix":      float32[qk_nope, qk_nope],
                "zero_point":    float32[qk_nope],
                "dim_of_bit":    int32[row_bits],
                "bitpos_in_dim": int32[row_bits],
            }

        ``packed_kcache`` is a *view* over the layer's packed buffer
        reshaped from ``[num_pages, packed_bytes_per_page]`` to
        ``[num_pages * page_size, packed_row_bytes]``. The kernel reads
        token ``t`` at row ``page_index(t) * page_size + slot(t)`` which
        matches our store layout exactly.
        """
        if self._mode != "wall":
            return None
        if kind not in self._wall_pools:
            return None
        if _wall_drop_packed_enabled():
            # No packed buffer to expose; sparse-path falls back to the
            # shadow FP8 path (kernel runs the all-None branch).
            return None
        entry = self._wall_pools[kind]
        if kind == "swa":
            local_layer_id = self._swa_local_layer_id(layer_id)
        else:
            _kind, local_layer_id = self._layer_id_for_extra(layer_id)
            assert _kind == kind, (
                f"layer_id={layer_id} maps to kind {_kind!r}, "
                f"requested {kind!r}"
            )
        if layer_id not in self._nope_cfgs:
            return None
        cfg = self._nope_cfgs[layer_id]
        packed_page_buf = entry.packed_buffers[local_layer_id]
        device = packed_page_buf.device
        page_size = entry.page_size
        num_pages = entry.num_pages
        num_rows = num_pages * page_size
        # ``packed_row_bytes`` includes the rope BF16 tail; kernel walks
        # the same stride for both nope/rope reads (rope read uses
        # ``pk_row + nope_bytes``).
        packed_row_bytes = entry.packed_bpt
        # Sanity (cheap, no copy).
        assert packed_page_buf.numel() == num_rows * packed_row_bytes, (
            f"packed buffer numel {packed_page_buf.numel()} != "
            f"num_rows({num_rows}) * packed_row_bytes({packed_row_bytes})"
        )
        packed_rows = packed_page_buf.view(num_rows, packed_row_bytes)

        # Pull cached GPU-resident calib tensors. _get_cached_cfg_gpu
        # populates these on first call (capture-safe).
        from sglang.jit_kernel.rotated_quant_dsv4_kernels import (
            _get_cached_cfg_gpu,
        )

        cfg_gpu = _get_cached_cfg_gpu(cfg, device)
        return {
            "packed_kcache": packed_rows,
            "scale_kcache": cfg_gpu["scale"],
            "R_matrix": cfg_gpu["R"],
            "zero_point": cfg_gpu["zero"],
            "dim_of_bit": cfg_gpu["dim_of_bit"],
            "bitpos_in_dim": cfg_gpu["bitpos_in_dim"],
        }

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
