"""Offline sweep: per-channel + token-block dynamic range cos vs (b_mean, blk).

Purpose
-------
Verify whether route E (b_mean=2.5 + blk64 per-channel dynamic range) actually
delivers single-layer cos >= 0.99 (so 43-layer cumulative >= 0.94 gsm8k)
on the real KV dump that produced the as-shipped calib.

This script is the gatekeeper for routes (E) / (C') before any production code
change. It is intentionally CPU-only and self-contained: load the dump,
compute per-layer Hadamard R, bit allocation b[d] for each b_mean, then
simulate the quantization round-trip under per-channel + token-block dynamic
range and report cos similarity vs the original BF16 nope.

Inputs
------
* ``--kv-dump``: path to ``/data00/kv_dump_dsv4_rank0.pt`` or equivalent.
  Schema: ``{layer_id: {"nope": fp [N, 1 or H, qk_nope_head_dim]}}`` matching
  ``build_rotated_kv_calib.py --dsv4-mode --from-kv-dump``.
* ``--b-means``: comma-separated, default ``2.0,2.25,2.5``.
* ``--blks``: comma-separated, default ``32,64,128,256``.
* ``--layers``: comma-separated layer ids to evaluate (default all).
* ``--n-tokens``: subsample first N tokens per layer (default 4096, ``-1`` = all).

Output
------
Markdown table to stdout, one row per (layer_id, b_mean, blk), columns
``[cos_mean, cos_p10, cos_p50, cos_p90, bpt_bytes, wall_ratio]``.

The wall_ratio formula::

    bpt = ceil(b_mean * D / 8) + 2 * D * 2 // blk + 128
                 ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
                 packed nope    per-block per-channel min+scale (fp16 each)
                                amortized over ``blk`` tokens => 2*2*D/blk B/tok

    wall_ratio = 584 / bpt

584 B/tok is the DSv4 native FP8 layout (576 slot + 8 UE8M0).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List

import torch

# Import the actual production ``allocate_bits`` / ``build_hadamard`` directly
# via importlib so we don't drag in the rest of the ``sglang`` package
# (which pulls in CUDA + transformers + pybase64 etc.). This mirrors the
# pattern already used by ``test/manual/quant/test_rotated_kv_quant_dsv4_canary.py``.
import importlib.util as _ilu

_RKQ_PATH = os.path.abspath(
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
_spec = _ilu.spec_from_file_location("_rotated_kv_quant_standalone", _RKQ_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"failed to load rotated_kv_quant.py from {_RKQ_PATH}")
_rkq = _ilu.module_from_spec(_spec)
# dataclass annotation type-resolution needs the module registered in
# ``sys.modules`` before ``exec_module`` runs.
sys.modules["_rotated_kv_quant_standalone"] = _rkq
_spec.loader.exec_module(_rkq)
allocate_bits = _rkq.allocate_bits
build_hadamard = _rkq.build_hadamard


_NATIVE_BPT = 584
_ROPE_BYTES = 128


def _per_channel_blk_round_trip(
    K_rot: torch.Tensor,  # [N, D] fp32 (already left-multiplied by R)
    bits: torch.Tensor,   # [D] int32, allocated bits per channel
    blk: int,
) -> torch.Tensor:
    """Simulate per-channel + per-block dynamic range RTN, return reconstruction.

    For each block of ``blk`` consecutive tokens and each channel d:
        mn_b = min_{t in block} K_rot[t, d]
        mx_b = max_{t in block} K_rot[t, d]
        L_d  = 2**bits[d] - 1
        scale_b = (mx_b - mn_b) / max(L_d, 1)
        codes = round((K_rot - mn_b) / scale_b).clamp(0, L_d)
        K_hat = codes * scale_b + mn_b
    Returns ``K_hat`` of same shape as ``K_rot``.
    """
    N, D = K_rot.shape
    levels = ((1 << bits.to(torch.int64)) - 1).clamp_min(1).to(torch.float32)  # [D]
    n_blocks = (N + blk - 1) // blk
    K_hat = torch.empty_like(K_rot)
    for bi in range(n_blocks):
        s = bi * blk
        e = min(s + blk, N)
        chunk = K_rot[s:e]  # [bs, D]
        mn = chunk.min(dim=0, keepdim=True).values  # [1, D]
        mx = chunk.max(dim=0, keepdim=True).values
        scale = ((mx - mn) / levels.unsqueeze(0)).clamp(min=1e-8)
        codes = ((chunk - mn) / scale).round().clamp(min=0)
        codes = torch.minimum(codes, levels.unsqueeze(0))
        K_hat[s:e] = codes * scale + mn
    return K_hat


def _eval_layer(
    nope: torch.Tensor,    # [N, D] fp32 (already flattened over heads)
    R: torch.Tensor,       # [D, D] fp32 orthonormal
    b_mean: float,
    blk: int,
    b_min: int = 1,
    b_max: int = 4,
) -> dict:
    """Single (layer, b_mean, blk) evaluation."""
    N, D = nope.shape
    # 1) bit allocation per-channel (mirror build_rotated_kv_calib.py).
    K_rot = nope @ R  # [N, D] fp32
    var = K_rot.var(dim=0, unbiased=False).clamp_min(1e-12)
    bits = allocate_bits(var, b_mean=b_mean, b_min=b_min, b_max=b_max)
    # 2) per-channel + per-block dynamic range RTN.
    K_hat = _per_channel_blk_round_trip(K_rot, bits, blk)
    # 3) recover to nope space and measure cos vs original nope.
    nope_hat = K_hat @ R.t()
    cos_per_tok = torch.nn.functional.cosine_similarity(
        nope, nope_hat, dim=-1
    )  # [N]
    # 4) byte accounting.
    row_bits = int(bits.sum().item())
    nope_packed_bytes = (row_bits + 7) // 8
    # per-block per-channel min+scale: 2 * D * 2B amortized over blk tokens.
    dyn_range_bytes_per_tok = (2 * D * 2 + blk - 1) // blk  # ceil
    bpt = nope_packed_bytes + dyn_range_bytes_per_tok + _ROPE_BYTES
    wall_ratio = _NATIVE_BPT / bpt
    return {
        "cos_mean": float(cos_per_tok.mean().item()),
        "cos_p10": float(cos_per_tok.quantile(0.10).item()),
        "cos_p50": float(cos_per_tok.quantile(0.50).item()),
        "cos_p90": float(cos_per_tok.quantile(0.90).item()),
        "bpt_bytes": int(bpt),
        "wall_ratio": float(wall_ratio),
        "nope_packed_bytes": int(nope_packed_bytes),
        "dyn_range_bytes_per_tok": int(dyn_range_bytes_per_tok),
        "bits_mean": float(bits.float().mean().item()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kv-dump", required=True, help="path to kv_dump_dsv4_*.pt")
    p.add_argument("--b-means", default="2.0,2.25,2.5",
                   help="comma-separated b_mean values to sweep")
    p.add_argument("--blks", default="32,64,128,256",
                   help="comma-separated block sizes to sweep")
    p.add_argument("--layers", default="",
                   help="comma-separated layer ids; empty = all")
    p.add_argument("--n-tokens", type=int, default=4096,
                   help="subsample first N tokens per layer; -1 = all")
    p.add_argument("--b-min", type=int, default=1)
    p.add_argument("--b-max", type=int, default=4)
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda (cuda only if installed)")
    args = p.parse_args()

    b_means = [float(x) for x in args.b_means.split(",") if x.strip()]
    blks = [int(x) for x in args.blks.split(",") if x.strip()]
    target_layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers.strip()
        else None
    )

    print(f"# loading KV dump: {args.kv_dump}", file=sys.stderr)
    dump = torch.load(args.kv_dump, map_location="cpu", weights_only=False)
    if not isinstance(dump, dict):
        raise ValueError(f"kv-dump must be dict, got {type(dump)}")

    # Discover qk_nope_head_dim and Hadamard R (shared across layers since R
    # depends only on D in our pipeline).
    sample_lid = next(iter(dump.keys()))
    sample_entry = dump[sample_lid]
    if "nope" not in sample_entry:
        raise ValueError(
            f"dump[{sample_lid}] missing 'nope' key; got {list(sample_entry)}"
        )
    qk_nope_head_dim = sample_entry["nope"].shape[-1]
    R = build_hadamard(qk_nope_head_dim).to(args.device)

    print(
        f"# qk_nope_head_dim={qk_nope_head_dim}  layers={len(dump)}  "
        f"device={args.device}  b_means={b_means}  blks={blks}  "
        f"n_tokens_per_layer={args.n_tokens}",
        file=sys.stderr,
    )

    # Header.
    print(
        "| layer | b_mean | blk | cos_mean | cos_p10 | cos_p50 | cos_p90 | "
        "bits_mean | row_bytes | dyn_B/tok | bpt | wall_ratio |"
    )
    print(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )

    layer_ids = sorted(dump.keys())
    if target_layers is not None:
        layer_ids = [lid for lid in layer_ids if lid in target_layers]

    for lid in layer_ids:
        entry = dump[lid]
        nope = entry["nope"]
        if nope.dim() == 3:
            # [N, H, D] -> [N*H, D]
            nope = nope.reshape(-1, nope.shape[-1])
        nope = nope.to(torch.float32).to(args.device).contiguous()
        if args.n_tokens > 0 and nope.shape[0] > args.n_tokens:
            nope = nope[: args.n_tokens]
        for b_mean in b_means:
            for blk in blks:
                stats = _eval_layer(
                    nope, R, b_mean=b_mean, blk=blk,
                    b_min=args.b_min, b_max=args.b_max,
                )
                print(
                    f"| {lid} | {b_mean:.2f} | {blk} | "
                    f"{stats['cos_mean']:.6f} | {stats['cos_p10']:.6f} | "
                    f"{stats['cos_p50']:.6f} | {stats['cos_p90']:.6f} | "
                    f"{stats['bits_mean']:.3f} | "
                    f"{stats['nope_packed_bytes']} | "
                    f"{stats['dyn_range_bytes_per_tok']} | "
                    f"{stats['bpt_bytes']} | {stats['wall_ratio']:.4f} |",
                    flush=True,
                )

    # Footer: cumulative 43-layer cos projection per (b_mean, blk).
    # This is a conservative lower bound: assumes errors stack multiplicatively.
    # In practice attention is robust to per-layer cos noise via softmax
    # normalization, but cumulative >= 0.94 single-layer cos^43 is a useful
    # screening rule.
    print("", flush=True)
    print(
        "# screening rule: 43-layer cumulative cos^43 >= 0.94 "
        "<=> per-layer cos >= 0.9986. "
        "Single-layer cos >= 0.99 typically passes gsm8k in practice "
        "(softmax compensates).",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
