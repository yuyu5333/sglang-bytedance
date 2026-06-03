"""
Rotated + non-uniform bit-allocation KV cache quantizer (QuaRot/SpinQuant +
Lloyd reverse water-filling).

This module is M0 of the "rotation + non-uniform bit allocation" roadmap (see
KVRoadMap.md, "启发三"): it is a stand-alone numerical implementation that can
be exercised on any tensor without touching the SGLang runtime. M1 will plug
the resulting :class:`RotatedQuantizer` into a packed ``KVCache`` subclass.

Pipeline:

    write:  K_raw  --(left-multiply Hadamard R)-->  K_rot
                   --(per-coordinate affine, b[d] bits)-->  codes
                   --(rowwise bit-pack)-->  uint8 packed buffer

    read:   uint8  --(rowwise bit-unpack)-->  codes
                   --(inverse affine)-->  K_rot_hat
                   --(right-multiply R^T)-->  K_hat   (≈ K_raw)

The forward Hadamard ``R`` is orthonormal so for any query ``q`` the inner
product ``<q, K> = <q R, K R>`` -- attention is mathematically unchanged when
both Q and K are rotated by the same R, which is what makes "compress storage
in the rotated domain" lossless in expectation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


# ----------------------------------------------------------------------
# 1. Orthonormal rotation: Walsh-Hadamard via Sylvester construction
# ----------------------------------------------------------------------
def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def build_hadamard(
    dim: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build a ``dim x dim`` orthonormal rotation.

    For powers of two we return the standard normalised Walsh-Hadamard matrix
    (entries ±1/√dim). For other dims we fall back to a block-diagonal matrix
    composed of the largest power-of-two Hadamard block plus an identity tail
    -- still orthogonal, still cheap, just less aggressive on the residual
    coordinates. (M2 will replace this with a fast WHT kernel.)
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    device = device or torch.device("cpu")

    if _is_power_of_two(dim):
        H = torch.tensor([[1.0]], device=device, dtype=dtype)
        while H.shape[0] < dim:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
            )
        return H / math.sqrt(dim)

    # Largest power-of-two prefix; identity tail on the remainder.
    block = 1 << (dim.bit_length() - 1)
    out = torch.eye(dim, device=device, dtype=dtype)
    out[:block, :block] = build_hadamard(block, device=device, dtype=dtype)
    return out


# ----------------------------------------------------------------------
# 2. Streaming calibrator: per-coordinate variance + robust quantile range
# ----------------------------------------------------------------------
class KVCalibrator:
    """Online statistics collector over the rotated coordinate axis.

    Accumulates Welford-style mean/variance and a coarse histogram so that we
    can pull robust quantiles (default 0.001 / 0.999) without storing the
    full sample. Designed to be fed in mini-batches:

        cal = KVCalibrator(dim=D, num_bins=2048, q_lo=1e-3, q_hi=1-1e-3)
        for K_batch in stream:                   # K_batch shape [..., D]
            cal.observe(K_batch @ R)             # rotate first
        stats = cal.finalize()                   # dict of per-d stats
    """

    def __init__(
        self,
        dim: int,
        num_bins: int = 2048,
        q_lo: float = 1e-3,
        q_hi: float = 1.0 - 1e-3,
        clip_init: float = 8.0,
    ):
        if not (0.0 < q_lo < q_hi < 1.0):
            raise ValueError("require 0 < q_lo < q_hi < 1")
        self.dim = dim
        self.num_bins = num_bins
        self.q_lo = q_lo
        self.q_hi = q_hi

        # Welford accumulators (per coordinate, fp64 for stability).
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)

        # Histogram bookkeeping. We bracket each coordinate with a symmetric
        # range [-clip_init * sigma, +clip_init * sigma] computed lazily after
        # the first batch; the first batch sets the histogram extents.
        self.hist_lo: Optional[torch.Tensor] = None  # [D]
        self.hist_hi: Optional[torch.Tensor] = None  # [D]
        self.hist: Optional[torch.Tensor] = None  # [D, num_bins] int64
        self.clip_init = clip_init

    def _maybe_init_histogram(self, x: torch.Tensor) -> None:
        if self.hist is not None:
            return
        flat = x.reshape(-1, self.dim).to(torch.float64)
        std = flat.std(dim=0).clamp_min(1e-6)
        mean = flat.mean(dim=0)
        self.hist_lo = (mean - self.clip_init * std).cpu()
        self.hist_hi = (mean + self.clip_init * std).cpu()
        self.hist = torch.zeros(self.dim, self.num_bins, dtype=torch.int64)

    def observe(self, x: torch.Tensor) -> None:
        """Feed a tensor whose last dimension is the rotated coordinate."""
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"observe expects last dim {self.dim}, got {x.shape[-1]}"
            )
        flat = x.reshape(-1, self.dim).to(torch.float64).cpu()
        n = flat.shape[0]
        if n == 0:
            return

        # Welford merge.
        batch_mean = flat.mean(dim=0)
        batch_m2 = ((flat - batch_mean) ** 2).sum(dim=0)
        delta = batch_mean - self.mean
        new_count = self.count + n
        self.mean = self.mean + delta * (n / new_count)
        self.m2 = self.m2 + batch_m2 + delta ** 2 * (self.count * n / new_count)
        self.count = new_count

        # Histogram (rebin into pre-allocated [-clip*sigma, +clip*sigma]).
        self._maybe_init_histogram(x)
        lo = self.hist_lo  # [D]
        hi = self.hist_hi  # [D]
        width = (hi - lo).clamp_min(1e-12)
        # Map every sample to a bin index per coordinate; clamp into range.
        idx = ((flat - lo) / width * self.num_bins).floor().to(torch.int64)
        idx.clamp_(0, self.num_bins - 1)
        # Scatter-add into per-coord histograms.
        for d in range(self.dim):
            self.hist[d].scatter_add_(
                0, idx[:, d], torch.ones(n, dtype=torch.int64)
            )

    def finalize(self) -> Dict[str, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError("KVCalibrator.finalize called with <2 samples")
        var = (self.m2 / max(self.count - 1, 1)).clamp_min(1e-12)

        # Robust quantiles from the histogram.
        if self.hist is None:
            raise RuntimeError("histogram never initialised; call observe first")
        cdf = self.hist.cumsum(dim=1).to(torch.float64) / float(self.count)
        bin_centers = (
            self.hist_lo.unsqueeze(1)
            + (self.hist_hi - self.hist_lo).unsqueeze(1)
            * (torch.arange(self.num_bins).to(torch.float64) + 0.5)
            / self.num_bins
        )

        def _quantile(target: float) -> torch.Tensor:
            # First bin where cdf >= target, per coordinate.
            mask = cdf >= target
            idx = mask.float().argmax(dim=1)  # [D]
            # If no bin satisfies (target=1 exactly), fall back to last bin.
            no_hit = ~mask.any(dim=1)
            idx[no_hit] = self.num_bins - 1
            return bin_centers.gather(1, idx.unsqueeze(1)).squeeze(1)

        q_lo = _quantile(self.q_lo)
        q_hi = _quantile(self.q_hi)

        return {
            "mean": self.mean.to(torch.float32),
            "var": var.to(torch.float32),
            "q_lo": q_lo.to(torch.float32),
            "q_hi": q_hi.to(torch.float32),
            "count": torch.tensor(self.count, dtype=torch.int64),
        }


# ----------------------------------------------------------------------
# 3. Reverse water-filling integer bit allocation
# ----------------------------------------------------------------------
def allocate_bits(
    var: torch.Tensor, b_mean: float, b_min: int = 1, b_max: int = 4
) -> torch.Tensor:
    """Lloyd reverse water-filling -> integer bit table b[d] in [b_min, b_max].

    The continuous optimum minimising sum_d sigma_d^2 * 2^{-2 b[d]} subject
    to mean(b[d]) = b_mean is ``b_d = b_mean + 0.5 * log2(sigma_d^2 / G)``
    with G the geometric mean of the variances. We round it to integers in
    ``[b_min, b_max]``, then iteratively shuffle ±1 between the most under-
    and over-resourced coordinates until ``sum(b[d]) == round(b_mean * D)``.
    """
    if var.dim() != 1:
        raise ValueError("var must be 1-D, shape [D]")
    if not (b_min <= b_mean <= b_max):
        raise ValueError(f"b_mean {b_mean} outside [b_min, b_max]")

    D = var.numel()
    var = var.to(torch.float64).clamp_min(1e-12)
    log_var = var.log2()
    G = log_var.mean()  # log2 of geometric mean
    cont = b_mean + 0.5 * (log_var - G)
    bits = cont.round().clamp(b_min, b_max).to(torch.int64)

    target = int(round(b_mean * D))
    target = max(min(target, b_max * D), b_min * D)

    # Use the *unclamped* continuous score to break ties correctly.
    score = cont - bits.to(torch.float64)

    # Greedy fix-up: each iteration moves one bit toward the target.
    while True:
        diff = int(bits.sum().item()) - target
        if diff == 0:
            break
        if diff > 0:
            # Need to remove bits: pick the d with largest *negative* slack
            # (i.e. most over-allocated relative to its continuous optimum)
            # that can still afford to lose one bit.
            cand = (bits > b_min).nonzero(as_tuple=False).squeeze(-1)
            if cand.numel() == 0:
                break
            d = cand[score[cand].argmin()]
            bits[d] -= 1
            score[d] += 1.0
        else:
            cand = (bits < b_max).nonzero(as_tuple=False).squeeze(-1)
            if cand.numel() == 0:
                break
            d = cand[score[cand].argmax()]
            bits[d] += 1
            score[d] -= 1.0

    return bits.to(torch.int32)


# ----------------------------------------------------------------------
# 4. Rowwise bit-packing helpers (M0: pure PyTorch reference impl)
# ----------------------------------------------------------------------
def bitpack_rowwise(codes: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Pack ``codes`` (..., D) of integer codes into a uint8 tensor.

    Each code uses ``bits[d]`` bits; coordinates are concatenated in order
    of d and then padded with zero bits to the next byte boundary. Returns
    shape ``(..., row_bytes)`` where ``row_bytes = ceil(sum(bits) / 8)``.

    This is a reference implementation used for correctness testing in M0;
    M2 will replace it with a Triton kernel.
    """
    if codes.shape[-1] != bits.numel():
        raise ValueError("last dim of codes must match len(bits)")
    if codes.dtype not in (torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"codes dtype must be integer, got {codes.dtype}")

    leading = codes.shape[:-1]
    flat = codes.reshape(-1, codes.shape[-1]).to(torch.int64)
    n_rows, D = flat.shape
    row_bits = int(bits.sum().item())
    row_bytes = (row_bits + 7) // 8

    # Build a [n_rows, row_bits] bit matrix, then reshape to bytes.
    # bit_matrix[i, p] is the p-th bit of the row-i payload (LSB-first within
    # each coordinate, coordinates concatenated in d order).
    out_bits = torch.zeros(n_rows, row_bytes * 8, dtype=torch.int64)
    cursor = 0
    bits_list = bits.tolist()
    for d, b in enumerate(bits_list):
        if b == 0:
            continue
        # Take the b LSBs of flat[:, d].
        col = flat[:, d] & ((1 << b) - 1)
        for k in range(b):
            out_bits[:, cursor + k] = (col >> k) & 1
        cursor += b
    # Fold 8 bits at a time into a uint8.
    weights = (1 << torch.arange(8, dtype=torch.int64)).view(1, 1, 8)  # LSB-first
    grouped = out_bits.view(n_rows, row_bytes, 8)
    packed = (grouped * weights).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*leading, row_bytes)


def bitunpack_rowwise(
    packed: torch.Tensor, bits: torch.Tensor, dim: int
) -> torch.Tensor:
    """Inverse of :func:`bitpack_rowwise`. Returns int64 codes ``(..., D)``."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}")
    leading = packed.shape[:-1]
    row_bytes = packed.shape[-1]
    flat = packed.reshape(-1, row_bytes).to(torch.int64)
    n_rows = flat.shape[0]
    row_bits = row_bytes * 8

    # Expand each byte to 8 bits, LSB-first.
    bit_matrix = (
        (flat.unsqueeze(-1) >> torch.arange(8, dtype=torch.int64)) & 1
    ).reshape(n_rows, row_bits)

    out = torch.zeros(n_rows, dim, dtype=torch.int64)
    cursor = 0
    bits_list = bits.tolist()
    for d, b in enumerate(bits_list):
        if b == 0:
            continue
        weights = 1 << torch.arange(b, dtype=torch.int64)
        out[:, d] = (bit_matrix[:, cursor : cursor + b] * weights).sum(dim=-1)
        cursor += b
    return out.reshape(*leading, dim)


# ----------------------------------------------------------------------
# 5. End-to-end quantizer
# ----------------------------------------------------------------------
@dataclass
class RotatedQuantizerConfig:
    """Per-(layer, kv-side) quantizer configuration."""

    R: torch.Tensor        # [D, D] orthonormal rotation
    bits: torch.Tensor     # [D] int32 in [1, 4]
    scale: torch.Tensor    # [D] fp32 affine scale
    zero: torch.Tensor     # [D] fp32 affine zero (q_lo)
    row_bits: int = field(init=False)
    row_bytes: int = field(init=False)

    def __post_init__(self):
        self.row_bits = int(self.bits.sum().item())
        self.row_bytes = (self.row_bits + 7) // 8


class RotatedQuantizer:
    """Stateless quantize/dequantize given a :class:`RotatedQuantizerConfig`."""

    def __init__(self, config: RotatedQuantizerConfig):
        self.config = config

    @staticmethod
    def from_calibration(
        calib_stats: Dict[str, torch.Tensor],
        R: torch.Tensor,
        b_mean: float,
        b_min: int = 1,
        b_max: int = 4,
    ) -> "RotatedQuantizer":
        """Build a quantizer from variance + (q_lo, q_hi) calibration."""
        var = calib_stats["var"]
        q_lo = calib_stats["q_lo"]
        q_hi = calib_stats["q_hi"]
        bits = allocate_bits(var, b_mean=b_mean, b_min=b_min, b_max=b_max)
        # Affine scale per coordinate: cover [q_lo, q_hi] with 2^b - 1 levels.
        levels = (1 << bits.to(torch.int64)) - 1  # [D]
        scale = (q_hi - q_lo) / levels.to(torch.float32).clamp_min(1.0)
        return RotatedQuantizer(
            RotatedQuantizerConfig(R=R, bits=bits, scale=scale, zero=q_lo)
        )

    # --- core API -------------------------------------------------------
    def quantize(self, K: torch.Tensor) -> torch.Tensor:
        """K: ``(..., D)`` real-valued  ->  uint8 packed ``(..., row_bytes)``."""
        cfg = self.config
        if K.shape[-1] != cfg.R.shape[0]:
            raise ValueError(
                f"K last dim {K.shape[-1]} != rotation dim {cfg.R.shape[0]}"
            )
        K_rot = K.to(torch.float32) @ cfg.R.to(K.dtype if K.is_floating_point() else torch.float32)
        # Affine encode -> integer codes.
        codes = ((K_rot - cfg.zero) / cfg.scale.clamp_min(1e-12)).round()
        # Saturate to per-coordinate range.
        levels = (1 << cfg.bits.to(torch.int64)) - 1
        codes = codes.clamp(min=torch.zeros_like(levels).to(codes.dtype), max=levels.to(codes.dtype))
        codes_int = codes.to(torch.int64)
        return bitpack_rowwise(codes_int, cfg.bits)

    def dequantize(
        self, packed: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """uint8 packed ``(..., row_bytes)``  ->  fp ``(..., D)``."""
        cfg = self.config
        D = cfg.R.shape[0]
        codes = bitunpack_rowwise(packed, cfg.bits, dim=D).to(torch.float32)
        K_rot_hat = codes * cfg.scale + cfg.zero
        # Inverse rotation: orthonormal R => R^T is the inverse.
        K_hat = K_rot_hat @ cfg.R.t()
        return K_hat.to(dtype)

    # --- compression metric --------------------------------------------
    def storage_ratio_vs(self, ref_dtype: torch.dtype) -> float:
        """Return packed_bytes / ref_dtype_bytes per token row."""
        D = self.config.R.shape[0]
        ref_bytes = D * torch.tensor([], dtype=ref_dtype).element_size()
        return self.config.row_bytes / ref_bytes


__all__ = [
    "build_hadamard",
    "KVCalibrator",
    "allocate_bits",
    "bitpack_rowwise",
    "bitunpack_rowwise",
    "RotatedQuantizerConfig",
    "RotatedQuantizer",
]
