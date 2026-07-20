"""DSv4 rotated-quant store/load helpers (Python orchestration over Triton).

This module is **M3.c.1**: wall-storage replacement for DSv4 nope.

Two host-side entry points used by ``RotatedQuantDeepSeekV4TokenToKVPool`` in
``mode='wall'``:

* :func:`rotated_store_to_packed` — given a BF16 input row ``[N, 512]``
  ``cat(nope, rope)`` plus per-layer ``(R, bits, scale, zero)``, produce the
  per-token packed bytes (nope-INT2/3/4 + rope-BF16) and scatter them into
  the paged ``cache`` buffer at the locations given by ``indices``.

* :func:`rotated_load_to_fp8_layout` — given a packed paged ``cache`` plus
  ``indices``, gather the requested rows, run inverse quant + inverse
  rotation on the GPU, then call :func:`rotated_dequant_to_fp8_layout` to
  emit the standard DSv4 ``[N, 576]`` slot bytes (and per-token ``[N, 8]``
  UE8M0 scale bytes) that FlashMLA expects. **M3.c.2** wires this into the
  attention prologue.

Layout of the packed paged cache (M3.c.1 v1):

    bytes_per_token = nope_packed_row_bytes + rope_bytes  # rope = 128 B
    bytes_per_page  = bytes_per_token * page_size

The page is **dense byte-packed** (no UE8M0 scale tail and no FP8 nope tail).
This is intentionally simpler than the upstream ``584 = 576 + 8`` layout
because the scale + R + zero come from the per-layer calib (host->device
constants) instead of being stored per-token.

Bit-pack/unpack reuse the M0 PyTorch reference implementation in
``rotated_kv_quant.bitpack_rowwise`` / ``bitunpack_rowwise``. The hot path
(Triton bit-pack/unpack) is M3.c.3.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.rotated_kv_quant import (
    RotatedQuantizerConfig,
    bitpack_rowwise,
    bitunpack_rowwise,
)

# Layout constants are duplicated here (instead of imported from
# triton_rotated_quant_dsv4) so CPU-only environments without Triton can use
# rotated_store_to_packed / rotated_load_to_fp8_layout_cpu_ref. The Triton
# kernel module is imported lazily inside rotated_load_to_fp8_layout.
_MLA_NOPE_DIM = 448
_MLA_TILE_SIZE = 64
_MLA_SLOT_BYTES = 576  # nope FP8 (448) + rope BF16 (128)
_MLA_SCALES_PER_TOKEN = 8

# Rope is always BF16 (2 bytes per element) regardless of the nope bit budget.
_ROPE_BYTES = _MLA_TILE_SIZE * 2  # 64 elements * 2 = 128 B

# Route G uniform-bit layout: per-token × per-group affine header.
# 7 groups of 64 dims each (HEAD_DIM_NOPE / GROUP_DIM = 448 / 64 = 7).
# Each group stores (fp16 min, fp16 range) = 4 bytes.
_UNIFORM_GROUPS = _MLA_NOPE_DIM // _MLA_TILE_SIZE  # 7
_UNIFORM_HEADER_BYTES_PER_GROUP = 4  # fp16 min + fp16 range
_UNIFORM_HEADER_BYTES = _UNIFORM_GROUPS * _UNIFORM_HEADER_BYTES_PER_GROUP  # 28 B


def uniform_row_bytes_nope(bit_uniform: int) -> int:
    """Packed nope byte count for a uniform-N-bit row (N in [1..8])."""
    if bit_uniform <= 0:
        raise ValueError("uniform_row_bytes_nope requires bit_uniform > 0")
    total_bits = _MLA_NOPE_DIM * int(bit_uniform)
    if total_bits % 8 != 0:
        raise ValueError(
            f"uniform bit_uniform={bit_uniform} * dim={_MLA_NOPE_DIM} "
            f"is not byte-aligned"
        )
    return total_bits // 8


def packed_bytes_per_token(row_bytes_nope: int, bit_uniform: int = 0) -> int:
    """Total per-token byte count in the packed paged layout.

    When ``bit_uniform > 0``, an extra ``_UNIFORM_HEADER_BYTES`` (28 B)
    is reserved between the nope codes and the rope bytes to hold the
    per-token × per-64-dim-group (fp16 min, fp16 range) header. That
    header is what makes the wall path dynamic per-token affine instead
    of the legacy static per-dim ``(scale, zero)`` from calib.
    """
    extra = _UNIFORM_HEADER_BYTES if int(bit_uniform) > 0 else 0
    return int(row_bytes_nope) + extra + _ROPE_BYTES


def _build_pack_meta_from_bits(
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build (dim_of_bit, bitpos_in_dim, row_bits, row_bytes) for Triton pack.

    Returns CPU tensors. Caller moves to device as needed.
    """
    bits_cpu = bits.cpu().to(torch.int32)
    D = int(bits_cpu.shape[0])
    bits_list = bits_cpu.tolist()
    row_bits = int(sum(bits_list))
    row_bytes = (row_bits + 7) // 8

    dim_of_bit = torch.zeros(row_bits, dtype=torch.int32)
    bitpos_in_dim = torch.zeros(row_bits, dtype=torch.int32)
    cursor = 0
    for d in range(D):
        b = bits_list[d]
        if b <= 0:
            continue
        dim_of_bit[cursor:cursor + b] = d
        bitpos_in_dim[cursor:cursor + b] = torch.arange(b, dtype=torch.int32)
        cursor += b
    return dim_of_bit, bitpos_in_dim, row_bits, row_bytes


# Per-config pack-meta cache keyed by device (small: ~448 int32 per dim).
_PACK_META_CACHE: dict[int, dict[str, object]] = {}

# KDUMP5: per-process one-shot guard for env-gated writer/reader dumps.
_PYDUMP_SEEN_CFGS: set = set()


def _get_cached_pack_meta(
    cfg: RotatedQuantizerConfig, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (dim_of_bit_gpu, bitpos_in_dim_gpu) for this cfg+device."""
    entry = _get_cached_cfg_gpu(cfg, device)
    return entry["dim_of_bit"], entry["bitpos_in_dim"]


def _get_cached_cfg_gpu(
    cfg: RotatedQuantizerConfig, device,
) -> dict:
    """Return cached GPU copies of {R, scale, zero, bits, levels, dim_of_bit,
    bitpos_in_dim} for this cfg+device.

    All tensors are copied once at first call and reused for every
    subsequent ``rotated_store_to_packed``. This is critical for CUDA graph
    capture: any per-call ``.to(device)`` H2D copy would invalidate the
    stream capture (cudaErrorStreamCaptureInvalidated).
    """
    key = id(cfg)
    dev_key = str(device)
    entry = _PACK_META_CACHE.get(key)
    if entry is not None and entry.get("device") == dev_key:
        return entry
    dim_of_bit, bitpos_in_dim, _rb, _rbytes = _build_pack_meta_from_bits(cfg.bits)
    if device is not None and getattr(device, "type", None) == "cuda":
        dim_of_bit = dim_of_bit.to(device)
        bitpos_in_dim = bitpos_in_dim.to(device)
        R_gpu = cfg.R.to(device=device, dtype=torch.float32)
        scale_gpu = cfg.scale.to(device=device, dtype=torch.float32).clamp_min(1e-12)
        zero_gpu = cfg.zero.to(device=device, dtype=torch.float32)
        bits_gpu = cfg.bits.to(device=device, dtype=torch.int32)
        levels_gpu = (
            (torch.ones_like(bits_gpu, dtype=torch.int64) << bits_gpu.to(torch.int64))
            - 1
        ).to(torch.float32)
    else:
        R_gpu = cfg.R.to(torch.float32)
        scale_gpu = cfg.scale.to(torch.float32).clamp_min(1e-12)
        zero_gpu = cfg.zero.to(torch.float32)
        bits_gpu = cfg.bits.to(torch.int32)
        levels_gpu = (
            (torch.ones_like(bits_gpu, dtype=torch.int64) << bits_gpu.to(torch.int64))
            - 1
        ).to(torch.float32)
    entry = {
        "device": dev_key,
        "dim_of_bit": dim_of_bit,
        "bitpos_in_dim": bitpos_in_dim,
        "R": R_gpu,
        # [step3r] BF16 prestore of R for the uniform-bit FlashMLA fill_sR
        #   path. The kernel already truncates R to bf16 before the gemm, so
        #   this is value-identical (RNE) while halving the R L2 load width
        #   and dropping the per-element fp32->bf16 convert. The fp32 R_gpu
        #   above is still used by the Python store-path rotation.
        "R_bf16": R_gpu.to(torch.bfloat16),
        "scale": scale_gpu,
        "zero": zero_gpu,
        "bits": bits_gpu,
        "levels": levels_gpu,
    }
    _PACK_META_CACHE[key] = entry
    return entry


# ----------------------------------------------------------------------
# Store path (write)
# ----------------------------------------------------------------------
def rotated_store_to_packed(
    input_bf16: torch.Tensor,  # [N, 512] BF16  (cat(nope, rope)
    cache: torch.Tensor,       # [num_pages, bytes_per_page] uint8
    indices: torch.Tensor,   # [N] int32, flat token-loc (page * page_size + slot
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> None:
    """Store rotated INT2/3/4 packed nope + raw BF16 rope into paged cache.

    Per-token bytes ``= row_bytes_nope + 128`` (rope BF16). GPU-only:
    ``cache[page, slot*Bpt : slot*Bpt + row_bytes_nope]`` is the packed
    nope, followed by ``128 B`` of raw BF16 rope.

    ``T3 path (Triton)：整个路径完全在 GPU 上： rotate + affine + quantise + clamp +
    bitpack，全部不走 CPU 不做任何 H2D/D2H 往返。
    """
    if input_bf16.dtype != torch.bfloat16:
        raise ValueError(f"input must be bf16, got {input_bf16.dtype}")
    if input_bf16.shape[-1] != _MLA_NOPE_DIM + _MLA_TILE_SIZE:
        raise ValueError(
            f"input last dim {input_bf16.shape[-1]} != {_MLA_NOPE_DIM + _MLA_TILE_SIZE}"
        )
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must be uint8, got {cache.dtype}")
    if cache.dim() != 2:
        raise ValueError(f"cache must be 2-D, got shape {tuple(cache.shape)}")

    N = input_bf16.shape[0]
    if N == 0:
        return

    # [Route H step3g] STORE-NULL PERF PROBE toggle.
    #   When SGLANG_RQ_STORE_NULL_PROBE=1, skip the ENTIRE per-token store
    #   (nope @ R rotation + per-group affine min/max/round/clamp + Triton
    #   bitpack + scatter into the packed paged cache). This is the ONLY
    #   heavy per-decode-step, per-layer GPU op that the packed-wall path
    #   runs but the native FP8 baseline does NOT (native has no store-side
    #   rotation/quant — it writes FP8 bytes directly via set_kv_buffer).
    #
    #   Rationale (step3f correction): reading splitkv_mla.cuh proved the
    #   producer->consumer handshake (bar_k_local_ready.arrive at L1468) is
    #   SHARED verbatim by both the packed and dense K-load branches, and
    #   the dense/native K path is ALSO manual load+dequant+smem-store (not
    #   TMA async) — so step3f's "packed replaced TMA async with named
    #   barrier" mechanism is factually wrong. step3b already zeroed the
    #   entire producer nope reconstruction inside the kernel with zero tps
    #   gain, and the consumer WG is shared with native. The last untested
    #   packed-specific per-step op is this store rotation/quant, which
    #   fires ~13 eager kernel launches/layer x 60 layers/step. This probe
    #   zeroes it in one cut:
    #     tps jumps toward native cgoff (~398) -> the store path IS the
    #       21.5x bottleneck (per-step host-launch-bound eager store);
    #     tps stays ~19.5 -> store is exonerated too, pivot elsewhere.
    #   Output is intentionally salad (packed cache stays stale); this only
    #   measures the full-load 32-req decode tps ceiling.
    import os as _os
    if _os.environ.get("SGLANG_RQ_STORE_NULL_PROBE", "0") == "1":
        return

    row_bytes_nope = cfg.row_bytes
    bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
    bytes_per_page = cache.shape[1]
    if bytes_per_page != bpt * page_size:
        raise ValueError(
            f"cache bytes_per_page {bytes_per_page} != bpt({bpt}) * page_size({page_size})"
        )
    if indices.shape != (N,):
        raise ValueError(f"indices shape {tuple(indices.shape)} != ({N},)")
    nope = input_bf16[:, :_MLA_NOPE_DIM].contiguous()

    # --- T3: 整个 store 全 GPU 路径
    #   1) rotate + affine quant + round + clamp 全走 GPU torch
    #   2) bitpack 走 Triton kernel (向量化位合并)
    #
    # 关键约束：CUDA graph capture 期间任何 H2D copy 都会触发
    # cudaErrorStreamCaptureInvalidated，所以 R/scale/zero/levels 必须
    # 在第一次（capture 前）就缓存到 GPU。_get_cached_cfg_gpu 一次性建好
    # 所有 GPU constants，后续每次 store 直接复用。
    cfg_gpu = _get_cached_cfg_gpu(cfg, input_bf16.device)
    R = cfg_gpu["R"]
    R_bf16 = cfg_gpu["R_bf16"]
    scale = cfg_gpu["scale"]
    zero = cfg_gpu["zero"]
    levels_f = cfg_gpu["levels"]

    # [P3 experiment] Use BF16 Tensor Core GEMM for the rotation and promote
    # only its result to FP32 for the existing affine/packing path. The
    # production default remains the FP32 reference; enable this isolated
    # experiment with SGLANG_RQ_BF16_ROTATE=1.
    if _os.environ.get("SGLANG_RQ_BF16_ROTATE", "0") == "1":
        rotate_impl = _os.environ.get("SGLANG_RQ_BF16_ROTATE_IMPL", "matmul")
        if rotate_impl == "linear":
            K_rot = torch._C._nn.linear(nope, R_bf16.t()).to(torch.float32)
        elif rotate_impl == "mm":
            K_rot = torch.mm(nope, R_bf16).to(torch.float32)
        elif rotate_impl == "matmul":
            K_rot = (nope @ R_bf16).to(torch.float32)
        else:
            raise ValueError(
                "SGLANG_RQ_BF16_ROTATE_IMPL must be matmul,mm,linear, "
                f"got {rotate_impl}"
            )
    else:
        K_rot = nope.to(torch.float32) @ R  # [N, 448]

    # [T_store_fused] bu4 single-kernel store fast path.
    #   When SGLANG_RQ_FUSED_STORE=1 AND the layout is the uniform 4-bit
    #   packed row (bit_uniform==4, bpt==380), fuse the entire store tail
    #   (per-group affine min/max/round/clamp + nibble pack + fp16 header +
    #   rope byte copy + sentinel-safe scatter) into ONE Triton launch,
    #   replacing the ~14 eager ops below. The nope@R matmul above is the
    #   only op left outside. Byte layout is bit-identical to the legacy
    #   cat() path (verified by the shared 224/28/128 offsets).
    if (
        _os.environ.get("SGLANG_RQ_FUSED_STORE", "0") == "1"
        and int(cfg.bit_uniform) == 4
        and bpt == 380
        and row_bytes_nope == 224
    ):
        from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
            triton_fused_store_bu4,
        )
        triton_fused_store_bu4(
            K_rot, input_bf16, cache, indices, bpt=bpt,
        )
        return

    # The fused bu4 path accepts the original int32/int64 indices directly.
    # Keep this conversion in the legacy path only; otherwise every fused
    # store paid for an unused int64 device copy before the early return.
    indices_i64 = indices.to(torch.int64)
    rope = input_bf16[:, _MLA_NOPE_DIM:].contiguous()

    # Route G uniform path: per-token × per-group(64) dynamic affine.
    # The per-dim calib (scale,zero,levels) is bypassed in favor of a
    # per-token header (fp16 min + fp16 range, 7 groups × 4 B = 28 B)
    # which is bitwise-stable across cudagraph replay (header bytes
    # captured into the same packed cache slot as the bit codes).
    if cfg.bit_uniform > 0:
        bu = int(cfg.bit_uniform)
        L_int = (1 << bu) - 1
        L_f = float(L_int)
        # [N, 7, 64]
        K_rot_g = K_rot.reshape(N, _UNIFORM_GROUPS, _MLA_TILE_SIZE)
        # Dynamic per-token × per-group min / max -> affine.
        kmin = K_rot_g.amin(dim=2, keepdim=True)         # [N, 7, 1] fp32
        kmax = K_rot_g.amax(dim=2, keepdim=True)         # [N, 7, 1] fp32
        krange = (kmax - kmin).clamp_min(1e-8)           # [N, 7, 1]
        step = krange / L_f                              # [N, 7, 1]
        codes_g = ((K_rot_g - kmin) / step).round()
        codes_g = torch.clamp(codes_g, min=0.0, max=L_f)
        codes_i32 = codes_g.reshape(N, _MLA_NOPE_DIM).to(torch.int32)
    else:
        codes_f = ((K_rot - zero) / scale).round()
        codes_f = torch.clamp(
            codes_f, min=torch.zeros_like(codes_f), max=levels_f
        )
        codes_i32 = codes_f.to(torch.int32)  # 留在 GPU

    # KDUMP5: writer-side one-shot dump (env-gated). Cross-checks the
    # codes/sk/zp/R the writer hands to bitpack against the kernel KDUMP4
    # printout. One line per unique cfg object to keep stdout sane.
    import os as _os
    if _os.environ.get("SGLANG_RQ_PYDUMP", "0") == "1":
        _cfg_key = id(cfg)
        if _cfg_key not in _PYDUMP_SEEN_CFGS:
            _PYDUMP_SEEN_CFGS.add(_cfg_key)
            try:
                _r0 = R[0, :4].detach().to("cpu").tolist()
                _sk0 = scale[:4].detach().to("cpu").tolist()
                _zp0 = zero[:4].detach().to("cpu").tolist()
                _nope0 = nope[0, :4].detach().to(torch.float32).cpu().tolist()
                _krot0 = K_rot[0, :4].detach().cpu().tolist()
                _codes0 = codes_i32[0, :4].detach().cpu().tolist()
                _idx0 = int(indices_i64[0].detach().cpu().item()) if N > 0 else -1
                print(
                    f"[KDUMP5-store] cfg_id%1000={_cfg_key % 1000} "
                    f"N={N} idx[0]={_idx0} "
                    f"R[0,0:4]={_r0} sk[0:4]={_sk0} zp[0:4]={_zp0} "
                    f"nope[0,0:4]={_nope0} K_rot[0,0:4]={_krot0} "
                    f"codes[0,0:4]={_codes0}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[KDUMP5-store] dump failed: {_e}", flush=True)

        _dim_key = ("dims16_432", id(cfg))
        if _dim_key not in _PYDUMP_SEEN_CFGS:
            _PYDUMP_SEEN_CFGS.add(_dim_key)
            try:
                _x_hat = codes_i32.to(torch.float32) * scale + zero
                _recon = (_x_hat @ R.t()).to(torch.float32)
                _idx0 = int(indices_i64[0].detach().cpu().item()) if N > 0 else -1
                print(
                    f"[KDUMP7-store-dims] cfg_id%1000={id(cfg) % 1000} "
                    f"N={N} idx[0]={_idx0} "
                    f"nope[16:20]={nope[0, 16:20].detach().to(torch.float32).cpu().tolist()} "
                    f"codes[16:20]={codes_i32[0, 16:20].detach().cpu().tolist()} "
                    f"s_x[16:20]={_x_hat[0, 16:20].detach().cpu().tolist()} "
                    f"sk[16:20]={scale[16:20].detach().cpu().tolist()} "
                    f"zp[16:20]={zero[16:20].detach().cpu().tolist()} "
                    f"recon[16:20]={_recon[0, 16:20].detach().cpu().tolist()}",
                    flush=True,
                )
                print(
                    f"[KDUMP7-store-dims] cfg_id%1000={id(cfg) % 1000} "
                    f"nope[432:436]={nope[0, 432:436].detach().to(torch.float32).cpu().tolist()} "
                    f"codes[432:436]={codes_i32[0, 432:436].detach().cpu().tolist()} "
                    f"s_x[432:436]={_x_hat[0, 432:436].detach().cpu().tolist()} "
                    f"sk[432:436]={scale[432:436].detach().cpu().tolist()} "
                    f"zp[432:436]={zero[432:436].detach().cpu().tolist()} "
                    f"recon[432:436]={_recon[0, 432:436].detach().cpu().tolist()}",
                    flush=True,
                )
                _nope0 = nope[0].detach().to(torch.float32)
                _recon0 = _recon[0].detach().to(torch.float32)
                _cos = torch.nn.functional.cosine_similarity(
                    _nope0.unsqueeze(0), _recon0.unsqueeze(0), dim=-1
                ).item()
                _max_abs = (_nope0 - _recon0).abs().max().item()
                _rmse = torch.sqrt(torch.mean((_nope0 - _recon0) ** 2)).item()
                print(
                    f"[KDUMP8-store-error] cfg_id%1000={id(cfg) % 1000} "
                    f"idx[0]={_idx0} nope_vs_recon_cos={_cos:.8f} "
                    f"rmse={_rmse:.8f} max_abs={_max_abs:.8f}",
                    flush=True,
                )
                # KDUMP9: separate "calib scale/zero suboptimal" from
                # "bit-budget SNR ceiling". Recompute IDEAL per-dim
                # min-max scale/zero on THIS batch's real rotated nope
                # (K_rot), using the SAME per-dim bit levels as calib,
                # then sweep fixed 2/3/4/6/8-bit. R is orthogonal so
                # cosine in K_rot space == cosine in nope space.
                try:
                    _Kr = K_rot.detach().to(torch.float32)      # [N, D]
                    _nope_f = nope.detach().to(torch.float32)   # [N, D]
                    _lv = levels_f.detach().to(torch.float32)   # [D] = 2^bits-1
                    if _lv.dim() == 2:
                        _lv = _lv[0]
                    _kmin = _Kr.min(dim=0, keepdim=True).values  # [1, D]
                    _kmax = _Kr.max(dim=0, keepdim=True).values
                    _is = ((_kmax - _kmin) /
                           _lv.clamp(min=1).unsqueeze(0)).clamp(min=1e-8)
                    _ic = ((_Kr - _kmin) / _is).round().clamp(min=0)
                    _ic = torch.minimum(_ic, _lv.unsqueeze(0))
                    _ixh = _ic * _is + _kmin
                    _irec = _ixh @ R.t()
                    _ideal_cos = torch.nn.functional.cosine_similarity(
                        _nope_f, _irec, dim=-1).mean().item()
                    _sw = {}
                    for _bb in (2, 3, 4, 6, 8):
                        _L = float((1 << _bb) - 1)
                        _s = ((_kmax - _kmin) / _L).clamp(min=1e-8)
                        _c = ((_Kr - _kmin) / _s).round().clamp(min=0)
                        _c = _c.clamp(max=_L)
                        _xh = _c * _s + _kmin
                        _rc = _xh @ R.t()
                        _sw[_bb] = torch.nn.functional.cosine_similarity(
                            _nope_f, _rc, dim=-1).mean().item()
                    print(
                        f"[KDUMP9-sweep] cfg_id%1000={id(cfg) % 1000} N={N} "
                        f"ideal_perdim_cos={_ideal_cos:.6f} "
                        f"b2={_sw[2]:.6f} b3={_sw[3]:.6f} b4={_sw[4]:.6f} "
                        f"b6={_sw[6]:.6f} b8={_sw[8]:.6f}",
                        flush=True,
                    )
                    # KDUMP10: PER-TOKEN dynamic scale/zero (route A).
                    # Each token gets its own min/max -> scale/zero, then
                    # fixed b-bit RTN. Two granularities:
                    #  (pt)  per-token single scalar over all 448 dims
                    #  (pg)  per-token per-group=64 (7 groups) scale/zero
                    # Reported in K_rot space (R orthogonal => cos preserved).
                    def _pt_cos(_bb, _grp):
                        _L = float((1 << _bb) - 1)
                        if _grp == 0:
                            _mn = _Kr.min(dim=1, keepdim=True).values  # [N,1]
                            _mx = _Kr.max(dim=1, keepdim=True).values
                            _s = ((_mx - _mn) / _L).clamp(min=1e-8)
                            _c = ((_Kr - _mn) / _s).round().clamp(min=0)
                            _c = _c.clamp(max=_L)
                            _xh = _c * _s + _mn
                        else:
                            _D = _Kr.shape[1]
                            _ng = _D // _grp
                            _kr = _Kr.reshape(N, _ng, _grp)
                            _mn = _kr.min(dim=2, keepdim=True).values  # [N,ng,1]
                            _mx = _kr.max(dim=2, keepdim=True).values
                            _s = ((_mx - _mn) / _L).clamp(min=1e-8)
                            _c = ((_kr - _mn) / _s).round().clamp(min=0)
                            _c = _c.clamp(max=_L)
                            _xh = (_c * _s + _mn).reshape(N, _D)
                        _rc = _xh @ R.t()
                        return torch.nn.functional.cosine_similarity(
                            _nope_f, _rc, dim=-1).mean().item()
                    print(
                        f"[KDUMP10-pertoken] cfg_id%1000={id(cfg) % 1000} N={N} "
                        f"pt_b2={_pt_cos(2,0):.6f} pt_b3={_pt_cos(3,0):.6f} "
                        f"pt_b4={_pt_cos(4,0):.6f} "
                        f"pg64_b2={_pt_cos(2,64):.6f} pg64_b3={_pt_cos(3,64):.6f} "
                        f"pg64_b4={_pt_cos(4,64):.6f}",
                        flush=True,
                    )
                except Exception as _e9:
                    print(f"[KDUMP9-sweep] failed: {_e9}", flush=True)
            except Exception as _e:
                print(f"[KDUMP7-store-dims] dump failed: {_e}", flush=True)

    # --- Triton bitpack（T3）。
    dim_of_bit = cfg_gpu["dim_of_bit"]
    bitpos_in_dim = cfg_gpu["bitpos_in_dim"]
    from sglang.jit_kernel.triton_rotated_quant_dsv4 import triton_bitpack_rowwise
    packed = triton_bitpack_rowwise(
        codes_i32, dim_of_bit, bitpos_in_dim, cfg.row_bytes,
    )

    # Compose the [N, bpt] row on GPU (纯 view/cat，无 H2D).
    # Uniform path: insert 28-byte per-token×7-group header between
    # packed nope codes and rope bytes. Header layout (per group):
    #   bytes [0..2)  = fp16(min)
    #   bytes [2..4)  = fp16(range = max - min)
    rope_bytes = rope.contiguous().view(torch.uint8).reshape(N, _ROPE_BYTES)
    if cfg.bit_uniform > 0:
        # kmin / krange are [N, 7, 1] fp32 from above. Pack to fp16 and
        # reinterpret as uint8 to splice into the byte row.
        header_pairs = torch.stack(
            (kmin.squeeze(-1).to(torch.float16),
             krange.squeeze(-1).to(torch.float16)),
            dim=-1,
        )  # [N, 7, 2] fp16
        header_bytes = (
            header_pairs.contiguous()
            .view(torch.uint8)
            .reshape(N, _UNIFORM_HEADER_BYTES)
        )
        full_row = torch.cat([packed, header_bytes, rope_bytes], dim=1)
    else:
        full_row = torch.cat([packed, rope_bytes], dim=1)  # [N, bpt]

    # KDUMP6-store-rope: dump first token's rope half (BF16 values + raw
    # uint8 bytes) so we can cross-check that the kernel's
    # `rope_bf16 = (bf16*)(pk_row + nope_bytes)` indexing reads the exact
    # same 8 BF16 elements (= 16 raw bytes) that we wrote here. Gated by
    # the same env + once-per-cfg key as KDUMP5-store to avoid log spam.
    if _os.environ.get("SGLANG_RQ_PYDUMP", "0") == "1":
        _rope_key = ("rope", id(cfg))
        if _rope_key not in _PYDUMP_SEEN_CFGS:
            _PYDUMP_SEEN_CFGS.add(_rope_key)
            try:
                _rope0 = rope[0, :8].detach().to(torch.float32).cpu().tolist()
                _ropeb0 = rope_bytes[0, :16].detach().cpu().tolist()
                _idx0 = int(indices_i64[0].detach().cpu().item()) if N > 0 else -1
                _bpt = full_row.shape[1]
                _row_bytes_nope = packed.shape[1]
                print(
                    f"[KDUMP6-store-rope] cfg_id%1000={id(cfg) % 1000} "
                    f"N={N} idx[0]={_idx0} bpt={_bpt} row_bytes_nope={_row_bytes_nope} "
                    f"rope[0,0:8]={_rope0} rope_bytes[0,0:16]={_ropeb0}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[KDUMP6-store-rope] dump failed: {_e}", flush=True)

    # Scatter into paged cache:
    # 必须过滤 -1 sentinel（translate_loc_from_full_to_swa 对未映射槽返回 -1）。
    # 关键：CUDA graph capture 期间任何 .all()/.nonzero()/Python branch
    # on a GPU bool 都会触发隐式 D2H sync，导致
    # cudaErrorStreamCaptureInvalidated。所以这里走 unconditional GPU
    # 路径：把 invalid row 用 "读旧值再写回" 替换成 no-op，整体只有 GPU op。
    cache_flat = cache.view(-1, bpt)
    idx_gpu = indices_i64.to(cache.device)
    if idx_gpu.numel() > 0:
        max_allowed = cache_flat.shape[0] - 1
        valid = (idx_gpu >= 0).unsqueeze(-1)  # [N, 1]
        idx_safe = idx_gpu.clamp(min=0, max=max_allowed)
        old_rows = cache_flat.index_select(0, idx_safe)
        new_rows = torch.where(valid, full_row, old_rows)
        cache_flat.index_copy_(0, idx_safe, new_rows)


# ----------------------------------------------------------------------
# Load path (read, used by attention prologue in M3.c.2)
# ----------------------------------------------------------------------
def rotated_load_to_fp8_layout(
    cache: torch.Tensor,       # [num_pages, bytes_per_page] uint8
    indices: torch.Tensor,     # [M] int32 flat token-locs to gather
    out_slot: torch.Tensor,    # [M, 576] uint8  (output, FlashMLA-compatible slot bytes)
    out_scale: torch.Tensor,   # [M, 8]   uint8  (output, UE8M0 per nope tile + 1 pad)
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> None:
    """Gather + dequant + inverse-rotate + emit standard FP8 slot bytes.

    Used by the M3.c.2 attention prologue: M attention-relevant token locs
    get expanded into FlashMLA's expected ``[M, 576]`` slot byte buffer plus
    ``[M, 8]`` UE8M0 scale bytes.
    """
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must be uint8, got {cache.dtype}")
    if out_slot.dtype != torch.uint8:
        raise ValueError(f"out_slot must be uint8, got {out_slot.dtype}")
    if out_scale.dtype != torch.uint8:
        raise ValueError(f"out_scale must be uint8, got {out_scale.dtype}")

    M = indices.shape[0]
    if M == 0:
        return
    if out_slot.shape != (M, _MLA_SLOT_BYTES):
        raise ValueError(
            f"out_slot shape {tuple(out_slot.shape)} != ({M},{_MLA_SLOT_BYTES})"
        )
    if out_scale.shape != (M, _MLA_SCALES_PER_TOKEN):
        raise ValueError(
            f"out_scale shape {tuple(out_scale.shape)} != ({M},{_MLA_SCALES_PER_TOKEN})"
        )

    row_bytes_nope = cfg.row_bytes
    bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
    if cache.shape[1] != bpt * page_size:
        raise ValueError(
            f"cache bytes_per_page {cache.shape[1]} != bpt({bpt}) * page_size({page_size})"
        )

    # --- T3: chunked GPU load path.
    #   * 用 Triton GPU bitunpack + chunk=8192，兼顾 GPU 内存和速度。
    #   * 之前 CPU bitunpack 太慢（448 次 Python 循环，M=10^6 → 300s+）。
    #   * CUDA OOM 用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments 解决。
    #
    # 同 store path：cfg constants 必须缓存到 GPU，避免 capture-期 H2D
    # 触发 cudaErrorStreamCaptureInvalidated。
    cfg_gpu = _get_cached_cfg_gpu(cfg, cache.device)
    scale = cfg_gpu["scale"]
    zero = cfg_gpu["zero"]
    R = cfg_gpu["R"]
    bits_gpu = cfg_gpu["bits"]

    indices_i64 = indices.to(device=cache.device, dtype=torch.int64)
    cache_flat = cache.view(-1, bpt)  # [num_pages * page_size, bpt] (view, no alloc)
    max_allowed = cache_flat.shape[0]

    # 防御性钳位：负索引 / 越界索引钳到合法范围。CUDA graph capture
    # 不允许任何 Python branch on GPU tensor（.all() 会做隐式 D2H sync），
    # 所以这里走 unconditional GPU clamp 而不是 boolean indexing 后改 M。
    # 攻击者：若 indices 里真有 -1，会读到 row 0 的内容，被写到 out_slot 的
    # 对应槽位 — 后续 attention 会用 attention mask 把那些位置忽略。
    indices_i64 = indices_i64.clamp(min=0, max=max_allowed - 1)

    # FP8 layout + GPU Triton kernels
    from sglang.jit_kernel.triton_rotated_quant_dsv4 import (
        rotated_dequant_to_fp8_layout,
        triton_bitunpack_rowwise,
    )

    is_uniform = cfg.bit_uniform > 0
    if is_uniform:
        bu = int(cfg.bit_uniform)
        L_f = float((1 << bu) - 1)
        header_off = row_bytes_nope
        rope_off = header_off + _UNIFORM_HEADER_BYTES
    else:
        rope_off = row_bytes_nope

    CHUNK = 8192  # 每块 ~ 8K tokens → 峰值 ~30 MB → 不会 OOM
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        chunk_idx = indices_i64[start:end]
        chunk_rows = cache_flat.index_select(0, chunk_idx)  # [chunk, bpt]
        chunk_packed = chunk_rows[:, :row_bytes_nope].contiguous()
        chunk_rope = chunk_rows[:, rope_off:].contiguous()
        # GPU bitunpack → codes_i32 (ch, 448) int32 (bits is GPU-cached)
        codes_chunk = triton_bitunpack_rowwise(chunk_packed, bits_gpu)
        ch = end - start
        if is_uniform:
            # Read [ch, 28] header bytes -> [ch, 7, 2] fp16 -> kmin / krange.
            hdr_bytes = chunk_rows[:, header_off:header_off + _UNIFORM_HEADER_BYTES].contiguous()
            hdr_fp16 = hdr_bytes.view(torch.float16).reshape(ch, _UNIFORM_GROUPS, 2)
            kmin_g = hdr_fp16[..., 0].to(torch.float32).unsqueeze(-1)   # [ch, 7, 1]
            krange_g = hdr_fp16[..., 1].to(torch.float32).unsqueeze(-1)
            step_g = krange_g / L_f                                     # [ch, 7, 1]
            codes_g = codes_chunk.reshape(ch, _UNIFORM_GROUPS, _MLA_TILE_SIZE).to(torch.float32)
            K_rot_hat = (codes_g * step_g + kmin_g).reshape(ch, _MLA_NOPE_DIM)
            nope_bf16 = (K_rot_hat @ R.t()).to(torch.bfloat16).contiguous()
        else:
            # Legacy variable-bit path: per-dim static (scale,zero).
            nope_bf16 = ((codes_chunk.to(torch.float32) * scale + zero) @ R.t()).to(
                torch.bfloat16
            ).contiguous()
        rope_bf16 = chunk_rope.view(torch.bfloat16).reshape(
            ch, _MLA_TILE_SIZE,
        ).contiguous()
        rotated_dequant_to_fp8_layout(
            nope_bf16, rope_bf16,
            out_slot[start:end], out_scale[start:end],
        )
        del chunk_rows, chunk_packed, chunk_rope, codes_chunk, nope_bf16, rope_bf16


# ----------------------------------------------------------------------
# CPU reference for unit tests (no GPU, no Triton)
# ----------------------------------------------------------------------
def rotated_load_to_fp8_layout_cpu_ref(
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    cfg: RotatedQuantizerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-CPU reference: returns ``(nope_bf16_recon[M,448],
    rope_bf16[M,64], packed_bytes[M,row_bytes])``.

    Does **not** run the FP8 layout Triton kernel (which requires Triton +
    a CUDA device). Used by canary unit tests on CPU-only environments.
    """
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must be uint8, got {cache.dtype}")
    M = indices.shape[0]
    row_bytes_nope = cfg.row_bytes
    bpt = packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
    if cache.shape[1] != bpt * page_size:
        raise ValueError(
            f"cache bytes_per_page {cache.shape[1]} != bpt({bpt}) * page_size({page_size})"
        )

    indices_i64 = indices.to(torch.int64)
    cache_flat = cache.view(-1, bpt)
    rows = cache_flat.index_select(0, indices_i64)
    packed_nope = rows[:, :row_bytes_nope].contiguous()
    if cfg.bit_uniform > 0:
        bu = int(cfg.bit_uniform)
        L_f = float((1 << bu) - 1)
        header_off = row_bytes_nope
        rope_off = header_off + _UNIFORM_HEADER_BYTES
        hdr_bytes = rows[:, header_off:rope_off].contiguous()
        hdr_fp16 = hdr_bytes.view(torch.float16).reshape(M, _UNIFORM_GROUPS, 2)
        kmin_g = hdr_fp16[..., 0].to(torch.float32).unsqueeze(-1)   # [M, 7, 1]
        krange_g = hdr_fp16[..., 1].to(torch.float32).unsqueeze(-1)
        step_g = krange_g / L_f
        rope_bytes = rows[:, rope_off:].contiguous()
        bits_cpu = cfg.bits.to(torch.int32)
        codes = bitunpack_rowwise(packed_nope, bits_cpu, dim=_MLA_NOPE_DIM)
        codes_g = codes.reshape(M, _UNIFORM_GROUPS, _MLA_TILE_SIZE).to(torch.float32)
        K_rot_hat = (codes_g * step_g + kmin_g).reshape(M, _MLA_NOPE_DIM)
    else:
        rope_bytes = rows[:, row_bytes_nope:].contiguous()
        bits_cpu = cfg.bits.to(torch.int32)
        codes = bitunpack_rowwise(packed_nope, bits_cpu, dim=_MLA_NOPE_DIM)
        K_rot_hat = codes.to(torch.float32) * cfg.scale + cfg.zero
    nope_bf16 = (K_rot_hat @ cfg.R.t().to(torch.float32)).to(torch.bfloat16).contiguous()
    rope_bf16 = rope_bytes.view(torch.bfloat16).reshape(M, _MLA_TILE_SIZE).contiguous()
    return nope_bf16, rope_bf16, packed_nope


__all__ = [
    "rotated_store_to_packed",
    "rotated_load_to_fp8_layout",
    "rotated_load_to_fp8_layout_cpu_ref",
    "quant_fp8_layout_cpu_ref",
    "packed_bytes_per_token",
    "uniform_row_bytes_nope",
]


# ----------------------------------------------------------------------
# CPU reference for the dequant->FP8 layout step (used by canary tests
# and by the pool's wall-mode prologue when no Triton runtime is present).
# ----------------------------------------------------------------------
def quant_fp8_layout_cpu_ref(
    nope_bf16: torch.Tensor,  # [M, 448] bf16
    rope_bf16: torch.Tensor,  # [M, 64]  bf16
):
    """Pure-PyTorch reference matching ``rotated_dequant_to_fp8_layout``.

    Layout (DSv4 FlashMLA-compatible):

      out_slot[m, 0:448]   = FP8 (E4M3) bytes of nope, per-tile UE8M0 scaling
      out_slot[m, 448:576] = raw BF16 bytes of rope (128 B)
      out_scale[m, 0:7]    = UE8M0 exponent bytes (one per 64-element tile)
      out_scale[m, 7]      = pad (0)

    UE8M0 scale = clip(ceil(log2(max(|x|) / FP8_E4M3_MAX)) + 127, 0, 255).
    Quantised value = round(x / 2^(scale-127)).
    """
    if nope_bf16.dim() != 2 or nope_bf16.shape[-1] != _MLA_NOPE_DIM:
        raise ValueError(
            f"nope_bf16 shape {tuple(nope_bf16.shape)} != (M, {_MLA_NOPE_DIM})"
        )
    if rope_bf16.dim() != 2 or rope_bf16.shape[-1] != _MLA_TILE_SIZE:
        raise ValueError(
            f"rope_bf16 shape {tuple(rope_bf16.shape)} != (M, {_MLA_TILE_SIZE})"
        )
    if nope_bf16.shape[0] != rope_bf16.shape[0]:
        raise ValueError("nope/rope batch mismatch")

    M = nope_bf16.shape[0]
    device = nope_bf16.device
    out_slot = torch.zeros((M, _MLA_SLOT_BYTES), dtype=torch.uint8, device=device)
    out_scale = torch.zeros(
        (M, _MLA_SCALES_PER_TOKEN), dtype=torch.uint8, device=device
    )

    nope_f = nope_bf16.to(torch.float32)
    num_tiles = _MLA_NOPE_DIM // _MLA_TILE_SIZE  # 7
    tiles = nope_f.reshape(M, num_tiles, _MLA_TILE_SIZE)
    abs_max = tiles.abs().amax(dim=-1).clamp_min(1e-4)
    fp8_max = 448.0
    scale_raw = abs_max / fp8_max
    log2_scale = torch.ceil(torch.log2(scale_raw))
    ue8m0 = (log2_scale + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    inv_scale = torch.pow(2.0, -(ue8m0.to(torch.float32) - 127.0))
    quantised = (tiles * inv_scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    fp8 = quantised.to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8).reshape(M, _MLA_NOPE_DIM)
    out_slot[:, :_MLA_NOPE_DIM] = fp8_bytes

    rope_bytes = rope_bf16.contiguous().view(torch.uint8).reshape(M, _ROPE_BYTES)
    out_slot[:, _MLA_NOPE_DIM:_MLA_SLOT_BYTES] = rope_bytes

    out_scale[:, :num_tiles] = ue8m0
    return out_slot, out_scale
