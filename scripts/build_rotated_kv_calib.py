"""Build a calibration ``.pt`` file for the rotated + non-uniform-bit KV cache pool.

This is the offline companion of
``python/sglang/srt/mem_cache/rotated_quant_memory_pool.py``. It produces a
file consumable by ``--rotated-kv-quant-config``.

Output schema (matches ``load_rotated_quant_calibration``)::

    {
        layer_id (int): {
            "k": {"R": fp32 [D, D], "bits": int32 [D],
                   "scale": fp32 [D], "zero": fp32 [D]},
            "v": { ... same with v_head_dim ... },
        },
        ...
    }

Two modes are supported:

1. ``--from-kv-dump <file>`` — load a real KV dump produced by some
   external instrumented forward pass. Expected schema::

       {
           layer_id (int): {"k": fp [N, H, D], "v": fp [N, H, VD]}
       }

   where ``N`` is the number of tokens collected (across the whole
   calibration corpus), ``H`` is ``num_kv_heads`` and ``D`` /
   ``VD`` are the head dimensions. Layers may differ in ``N``.

2. ``--synthetic`` — generate a fake heteroscedastic Gaussian dump in
   memory with per-coordinate ``sigma`` shaped to model an outlier
   distribution. Useful for smoke testing the whole pipeline without a
   real model. ``--num-layers``, ``--head-num``, ``--head-dim``,
   ``--v-head-dim``, ``--num-tokens`` and ``--seed`` control its size.

Common flags:

    --b-mean      per-coordinate average bit count (e.g. 2.5)
    --b-min/max   bit-allocation clamp (defaults 1, 4)
    --num-bins    histogram bins for quantile estimation (default 2048)
    --q-lo/--q-hi outer quantiles used as the affine [zero, zero+scale*max]
                  range (defaults 1e-3 and 1 - 1e-3)
    -o/--output   destination ``.pt`` path

Run ``python scripts/build_rotated_kv_calib.py --help`` for the full list.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from typing import Dict, Tuple

import torch

# Load the M0 module directly, bypassing ``sglang.__init__`` (which pulls in
# CUDA-only deps like triton). This keeps the calibration script usable on
# CPU-only machines.
_M0_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "python",
        "sglang",
        "srt",
        "layers",
        "quantization",
        "rotated_kv_quant.py",
    )
)
_spec = importlib.util.spec_from_file_location("_rkq_m0", _M0_PATH)
_rkq = importlib.util.module_from_spec(_spec)
sys.modules["_rkq_m0"] = _rkq  # required for dataclass repr / pickling
_spec.loader.exec_module(_rkq)
KVCalibrator = _rkq.KVCalibrator
RotatedQuantizer = _rkq.RotatedQuantizer
build_hadamard = _rkq.build_hadamard

logger = logging.getLogger("build_rotated_kv_calib")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _synthetic_kv(
    num_layers: int,
    num_tokens: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    seed: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Heteroscedastic Gaussian K/V with per-coord sigma in [0.1, 3.0].

    Produces a tensor shape ``[num_tokens, head_num, head_dim]`` per layer per
    side. Pure CPU, fp32, used only for smoke tests.
    """
    g = torch.Generator().manual_seed(seed)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for lid in range(num_layers):
        # sigma profile: linear ramp + a few outlier coordinates.
        sigma_k = torch.linspace(0.1, 3.0, head_dim)
        outlier_idx = torch.randperm(head_dim, generator=g)[: max(1, head_dim // 16)]
        sigma_k[outlier_idx] *= 4.0
        sigma_v = torch.linspace(0.1, 3.0, v_head_dim)
        outlier_idx_v = torch.randperm(v_head_dim, generator=g)[
            : max(1, v_head_dim // 16)
        ]
        sigma_v[outlier_idx_v] *= 4.0

        k = torch.randn(num_tokens, head_num, head_dim, generator=g) * sigma_k
        v = torch.randn(num_tokens, head_num, v_head_dim, generator=g) * sigma_v
        out[lid] = {"k": k, "v": v}
    return out


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------
def _calibrate_side(
    samples: torch.Tensor,
    R: torch.Tensor,
    *,
    b_mean: float,
    b_min: int,
    b_max: int,
    num_bins: int,
    q_lo: float,
    q_hi: float,
    chunk_tokens: int,
) -> Dict[str, torch.Tensor]:
    """Run the calibrator on the *rotated* samples and build the per-coord table.

    ``samples`` is ``[N, H, D]``. We flatten head + token, rotate, feed to
    ``KVCalibrator`` in chunks, then derive ``RotatedQuantizer`` parameters
    via ``RotatedQuantizer.from_calibration``.
    """
    if samples.ndim != 3:
        raise ValueError(f"expected [N, H, D], got {tuple(samples.shape)}")
    n, h, d = samples.shape
    flat = samples.reshape(n * h, d).to(torch.float32)
    R_f32 = R.to(torch.float32)

    cal = KVCalibrator(dim=d, num_bins=num_bins, q_lo=q_lo, q_hi=q_hi)
    # Rotate then observe in chunks (memory-friendly).
    chunk = max(1, chunk_tokens)
    for start in range(0, flat.shape[0], chunk):
        cal.observe(flat[start : start + chunk] @ R_f32)
    stats = cal.finalize()

    quantizer = RotatedQuantizer.from_calibration(
        stats, R=R_f32, b_mean=b_mean, b_min=b_min, b_max=b_max
    )
    cfg = quantizer.config
    return {
        "R": cfg.R.to(torch.float32),
        "bits": cfg.bits.to(torch.int32),
        "scale": cfg.scale.to(torch.float32),
        "zero": cfg.zero.to(torch.float32),
    }


def _synthetic_mla_kv(
    num_layers: int,
    num_tokens: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    seed: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """合成 MLA KV dump：每层 latent ``[N, 1, kv_lora_rank]`` +
    rope ``[N, 1, qk_rope_head_dim]``（rope 段实际不被校准，这里只为 schema）。
    """
    g = torch.Generator().manual_seed(seed)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for lid in range(num_layers):
        sigma = torch.linspace(0.1, 3.0, kv_lora_rank)
        outlier_idx = torch.randperm(kv_lora_rank, generator=g)[
            : max(1, kv_lora_rank // 16)
        ]
        sigma[outlier_idx] *= 4.0
        latent = (
            torch.randn(num_tokens, 1, kv_lora_rank, generator=g) * sigma
        )
        # rope 段，仅占位（当前算法不校准）
        rope = torch.randn(num_tokens, 1, qk_rope_head_dim, generator=g)
        out[lid] = {"latent": latent, "rope": rope}
    return out


def build_mla_calibration(
    kv_dump: Dict[int, Dict[str, torch.Tensor]],
    *,
    qk_rope_head_dim: int,
    b_mean: float,
    b_min: int,
    b_max: int,
    num_bins: int,
    q_lo: float,
    q_hi: float,
    chunk_tokens: int,
) -> Dict:
    """MLA 模式校准：仅对 latent 段做旋转 + bit 分配；rope 段不校准。

    输入 schema::

        {layer_id: {"latent": fp [N, 1 or H, kv_lora_rank],
                     "rope":   fp [N, 1 or H, qk_rope_head_dim] (optional)}}

    输出 schema 见 ``rotated_quant_mla_memory_pool.py``。
    """
    if not kv_dump:
        raise ValueError("kv_dump is empty")

    sample = next(iter(kv_dump.values()))
    if "latent" not in sample:
        raise ValueError(
            "MLA calibration requires 'latent' key per layer; got "
            f"{list(sample.keys())}"
        )
    kv_lora_rank = sample["latent"].shape[-1]
    R_latent = build_hadamard(kv_lora_rank)

    out: Dict = {
        "_meta": {
            "mode": "mla",
            "kv_lora_rank": int(kv_lora_rank),
            "qk_rope_head_dim": int(qk_rope_head_dim),
            "layer_num": len(kv_dump),
        }
    }
    for lid in sorted(kv_dump.keys()):
        entry = kv_dump[lid]
        latent_samples = entry["latent"]
        # 接受 [N, H, L] 或 [N, 1, L] 或 [N, L]
        if latent_samples.ndim == 2:
            latent_samples = latent_samples.unsqueeze(1)
        if latent_samples.ndim != 3:
            raise ValueError(
                f"layer {lid} latent shape {tuple(latent_samples.shape)} "
                "must be [N, H, L] or [N, L]"
            )
        if latent_samples.shape[-1] != kv_lora_rank:
            raise ValueError(
                f"layer {lid} latent kv_lora_rank "
                f"{latent_samples.shape[-1]} != layer 0 {kv_lora_rank}"
            )
        t0 = time.time()
        side = _calibrate_side(
            latent_samples,
            R_latent,
            b_mean=b_mean,
            b_min=b_min,
            b_max=b_max,
            num_bins=num_bins,
            q_lo=q_lo,
            q_hi=q_hi,
            chunk_tokens=chunk_tokens,
        )
        logger.info(
            "[mla] layer %d done in %.1fs: bits[latent] mean=%.2f",
            lid,
            time.time() - t0,
            float(side["bits"].float().mean()),
        )
        out[lid] = {"latent": side}
    return out


# ----------------------------------------------------------------------
# DSv4 mode (M3.b)
# ----------------------------------------------------------------------
def _synthetic_dsv4_kv(
    num_layers: int,
    num_tokens: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    seed: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """合成 DSv4 KV dump：每层 nope ``[N, 1, qk_nope_head_dim]`` +
    rope ``[N, 1, qk_rope_head_dim]``（rope 不校准，仅占位）。
    """
    g = torch.Generator().manual_seed(seed)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for lid in range(num_layers):
        sigma = torch.linspace(0.1, 3.0, qk_nope_head_dim)
        outlier_idx = torch.randperm(qk_nope_head_dim, generator=g)[
            : max(1, qk_nope_head_dim // 16)
        ]
        sigma[outlier_idx] *= 4.0
        nope = torch.randn(num_tokens, 1, qk_nope_head_dim, generator=g) * sigma
        rope = torch.randn(num_tokens, 1, qk_rope_head_dim, generator=g)
        out[lid] = {"nope": nope, "rope": rope}
    return out


def build_dsv4_calibration(
    kv_dump: Dict[int, Dict[str, torch.Tensor]],
    *,
    qk_rope_head_dim: int,
    compression_ratios: list,
    b_mean: float,
    b_min: int,
    b_max: int,
    num_bins: int,
    q_lo: float,
    q_hi: float,
    chunk_tokens: int,
) -> Dict:
    """DSv4 模式校准：仅对 nope 段做旋转 + bit 分配；rope / indexer 段不校准。

    输入 schema::

        {layer_id: {"nope": fp [N, 1 or H, qk_nope_head_dim],
                     "rope": fp [N, 1 or H, qk_rope_head_dim] (optional)}}

    输出 schema 见 ``rotated_quant_dsv4_memory_pool.py``。
    """
    if not kv_dump:
        raise ValueError("kv_dump is empty")
    sample = next(iter(kv_dump.values()))
    if "nope" not in sample:
        raise ValueError(
            "DSv4 calibration requires 'nope' key per layer; got "
            f"{list(sample.keys())}"
        )
    qk_nope_head_dim = sample["nope"].shape[-1]
    R_nope = build_hadamard(qk_nope_head_dim)

    out: Dict = {
        "_meta": {
            "mode": "dsv4",
            "qk_nope_head_dim": int(qk_nope_head_dim),
            "qk_rope_head_dim": int(qk_rope_head_dim),
            "compression_ratios": list(compression_ratios) if compression_ratios else [],
            "layer_num": len(kv_dump),
        }
    }
    for lid in sorted(kv_dump.keys()):
        entry = kv_dump[lid]
        samples = entry["nope"]
        if samples.ndim == 2:
            samples = samples.unsqueeze(1)
        if samples.ndim != 3:
            raise ValueError(
                f"layer {lid} nope shape {tuple(samples.shape)} "
                "must be [N, H, D] or [N, D]"
            )
        if samples.shape[-1] != qk_nope_head_dim:
            raise ValueError(
                f"layer {lid} nope dim {samples.shape[-1]} != "
                f"layer 0 {qk_nope_head_dim}"
            )
        t0 = time.time()
        side = _calibrate_side(
            samples,
            R_nope,
            b_mean=b_mean,
            b_min=b_min,
            b_max=b_max,
            num_bins=num_bins,
            q_lo=q_lo,
            q_hi=q_hi,
            chunk_tokens=chunk_tokens,
        )
        logger.info(
            "[dsv4] layer %d done in %.1fs: bits[nope] mean=%.2f",
            lid,
            time.time() - t0,
            float(side["bits"].float().mean()),
        )
        out[lid] = {"nope": side}
    return out


def build_calibration(
    kv_dump: Dict[int, Dict[str, torch.Tensor]],
    *,
    b_mean: float,
    b_min: int,
    b_max: int,
    num_bins: int,
    q_lo: float,
    q_hi: float,
    chunk_tokens: int,
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    """Fold the dump into the on-disk calibration schema."""
    if not kv_dump:
        raise ValueError("kv_dump is empty")

    # Infer head_dim / v_head_dim from layer 0 and reuse R across layers
    # whose dim agrees (saves time + memory; correct because R only needs
    # to be orthonormal -- it does not depend on data).
    sample = next(iter(kv_dump.values()))
    head_dim = sample["k"].shape[-1]
    v_head_dim = sample["v"].shape[-1]
    R_k = build_hadamard(head_dim)
    R_v = build_hadamard(v_head_dim) if v_head_dim != head_dim else R_k

    out: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for lid in sorted(kv_dump.keys()):
        entry = kv_dump[lid]
        if "k" not in entry or "v" not in entry:
            raise ValueError(f"kv_dump[{lid}] missing 'k' or 'v'")
        k_samples = entry["k"]
        v_samples = entry["v"]
        if k_samples.shape[-1] != head_dim:
            raise ValueError(
                f"layer {lid} k head_dim {k_samples.shape[-1]} != layer 0 "
                f"{head_dim}"
            )
        if v_samples.shape[-1] != v_head_dim:
            raise ValueError(
                f"layer {lid} v v_head_dim {v_samples.shape[-1]} != layer 0 "
                f"{v_head_dim}"
            )
        t0 = time.time()
        side_k = _calibrate_side(
            k_samples, R_k, b_mean=b_mean, b_min=b_min, b_max=b_max,
            num_bins=num_bins, q_lo=q_lo, q_hi=q_hi, chunk_tokens=chunk_tokens,
        )
        side_v = _calibrate_side(
            v_samples, R_v, b_mean=b_mean, b_min=b_min, b_max=b_max,
            num_bins=num_bins, q_lo=q_lo, q_hi=q_hi, chunk_tokens=chunk_tokens,
        )
        logger.info(
            "layer %d done in %.1fs: bits[k] mean=%.2f, bits[v] mean=%.2f",
            lid,
            time.time() - t0,
            float(side_k["bits"].float().mean()),
            float(side_v["bits"].float().mean()),
        )
        out[lid] = {"k": side_k, "v": side_v}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Offline calibration for the rotated + non-uniform-bit KV cache. "
            "Produces a .pt consumable by --rotated-kv-quant-config."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-kv-dump",
        type=str,
        help="path to a .pt with schema {layer_id: {'k': [N,H,D], 'v': [N,H,VD]}}",
    )
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="generate synthetic heteroscedastic Gaussian samples (smoke test)",
    )

    # Synthetic-only knobs
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--head-num", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--v-head-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)

    # MLA mode (M3.a)
    p.add_argument(
        "--mla-mode",
        action="store_true",
        help=(
            "Build a MLA-mode calibration (latent only). Output schema is "
            "{_meta:{mode:'mla',...}, layer_id:{'latent':{...}}}."
        ),
    )
    p.add_argument(
        "--kv-lora-rank",
        type=int,
        default=512,
        help="MLA kv_lora_rank (latent dim). Default matches DeepSeek-V2/V3.",
    )
    p.add_argument(
        "--qk-rope-head-dim",
        type=int,
        default=64,
        help="MLA qk_rope_head_dim (rope dim, kept BF16). Default 64.",
    )

    # DSv4 mode (M3.b)
    p.add_argument(
        "--dsv4-mode",
        action="store_true",
        help=(
            "Build a DSv4-mode calibration (nope only). Output schema is "
            "{_meta:{mode:'dsv4',...}, layer_id:{'nope':{...}}}. "
            "DSv4 main KV storage is unchanged at runtime; the calib drives "
            "RotatedQuantDeepSeekV4TokenToKVPool.simulate_quantize_nope() "
            "for offline accuracy evaluation."
        ),
    )
    p.add_argument(
        "--qk-nope-head-dim",
        type=int,
        default=448,
        help=(
            "DSv4 qk_nope_head_dim (the dim that gets rotated + INT2/3/4 "
            "quantised). DeepSeek-V4 standard = 448."
        ),
    )
    p.add_argument(
        "--compression-ratios",
        type=str,
        default="",
        help=(
            "DSv4 compression_ratios as comma-separated ints, e.g. "
            "'0,4,128,4'. Used only as metadata in the calib file; the "
            "actual ratios used at runtime come from the model config."
        ),
    )

    # Calibration knobs
    p.add_argument("--b-mean", type=float, default=2.5)
    p.add_argument("--b-min", type=int, default=1)
    p.add_argument("--b-max", type=int, default=4)
    p.add_argument("--num-bins", type=int, default=2048)
    p.add_argument("--q-lo", type=float, default=1e-3)
    p.add_argument("--q-hi", type=float, default=1.0 - 1e-3)
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=4096,
        help="chunk size when streaming samples into KVCalibrator.observe",
    )

    p.add_argument("-o", "--output", type=str, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.mla_mode and args.dsv4_mode:
        raise ValueError("--mla-mode and --dsv4-mode are mutually exclusive")

    if args.synthetic:
        if args.mla_mode:
            logger.info(
                "Synthesising MLA KV dump: layers=%d, tokens=%d, "
                "kv_lora_rank=%d, qk_rope_head_dim=%d",
                args.num_layers,
                args.num_tokens,
                args.kv_lora_rank,
                args.qk_rope_head_dim,
            )
            dump = _synthetic_mla_kv(
                num_layers=args.num_layers,
                num_tokens=args.num_tokens,
                kv_lora_rank=args.kv_lora_rank,
                qk_rope_head_dim=args.qk_rope_head_dim,
                seed=args.seed,
            )
        elif args.dsv4_mode:
            logger.info(
                "Synthesising DSv4 KV dump: layers=%d, tokens=%d, "
                "qk_nope_head_dim=%d, qk_rope_head_dim=%d",
                args.num_layers,
                args.num_tokens,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
            )
            dump = _synthetic_dsv4_kv(
                num_layers=args.num_layers,
                num_tokens=args.num_tokens,
                qk_nope_head_dim=args.qk_nope_head_dim,
                qk_rope_head_dim=args.qk_rope_head_dim,
                seed=args.seed,
            )
        else:
            logger.info(
                "Synthesising KV dump: layers=%d, tokens=%d, H=%d, D=%d, VD=%d",
                args.num_layers,
                args.num_tokens,
                args.head_num,
                args.head_dim,
                args.v_head_dim,
            )
            dump = _synthetic_kv(
                num_layers=args.num_layers,
                num_tokens=args.num_tokens,
                head_num=args.head_num,
                head_dim=args.head_dim,
                v_head_dim=args.v_head_dim,
                seed=args.seed,
            )
    else:
        logger.info("Loading KV dump from %s", args.from_kv_dump)
        dump = torch.load(
            args.from_kv_dump, map_location="cpu", weights_only=False
        )
        if not isinstance(dump, dict):
            raise ValueError(f"--from-kv-dump must be a dict, got {type(dump)}")

    if args.mla_mode:
        calib = build_mla_calibration(
            dump,
            qk_rope_head_dim=args.qk_rope_head_dim,
            b_mean=args.b_mean,
            b_min=args.b_min,
            b_max=args.b_max,
            num_bins=args.num_bins,
            q_lo=args.q_lo,
            q_hi=args.q_hi,
            chunk_tokens=args.chunk_tokens,
        )
    elif args.dsv4_mode:
        compression_ratios = (
            [int(x) for x in args.compression_ratios.split(",") if x.strip()]
            if args.compression_ratios
            else []
        )
        calib = build_dsv4_calibration(
            dump,
            qk_rope_head_dim=args.qk_rope_head_dim,
            compression_ratios=compression_ratios,
            b_mean=args.b_mean,
            b_min=args.b_min,
            b_max=args.b_max,
            num_bins=args.num_bins,
            q_lo=args.q_lo,
            q_hi=args.q_hi,
            chunk_tokens=args.chunk_tokens,
        )
    else:
        calib = build_calibration(
            dump,
            b_mean=args.b_mean,
            b_min=args.b_min,
            b_max=args.b_max,
            num_bins=args.num_bins,
            q_lo=args.q_lo,
            q_hi=args.q_hi,
            chunk_tokens=args.chunk_tokens,
        )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(calib, out_path)

    if args.mla_mode:
        layer_ids = sorted([k for k in calib.keys() if isinstance(k, int)])
        sample = calib[layer_ids[0]]
        logger.info(
            "Wrote MLA calibration: %s (layers=%d, kv_lora_rank=%d, "
            "qk_rope_head_dim=%d, latent_row_bytes=%d)",
            out_path,
            len(layer_ids),
            sample["latent"]["R"].shape[0],
            args.qk_rope_head_dim,
            (int(sample["latent"]["bits"].sum()) + 7) // 8,
        )
    elif args.dsv4_mode:
        layer_ids = sorted([k for k in calib.keys() if isinstance(k, int)])
        sample = calib[layer_ids[0]]
        logger.info(
            "Wrote DSv4 calibration: %s (layers=%d, qk_nope_head_dim=%d, "
            "qk_rope_head_dim=%d, nope_row_bytes=%d)",
            out_path,
            len(layer_ids),
            sample["nope"]["R"].shape[0],
            args.qk_rope_head_dim,
            (int(sample["nope"]["bits"].sum()) + 7) // 8,
        )
    else:
        layer_ids = sorted(calib.keys())
        sample = calib[layer_ids[0]]
        logger.info(
            "Wrote calibration: %s (layers=%d, head_dim=%d, v_head_dim=%d, "
            "row_bytes_k=%d, row_bytes_v=%d)",
            out_path,
            len(layer_ids),
            sample["k"]["R"].shape[0],
            sample["v"]["R"].shape[0],
            (int(sample["k"]["bits"].sum()) + 7) // 8,
            (int(sample["v"]["bits"].sum()) + 7) // 8,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
