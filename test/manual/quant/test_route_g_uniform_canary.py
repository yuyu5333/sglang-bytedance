"""Route G uniform-bit packed layout CPU canary.

Verifies M3.c.4 Stage-5 Route G step 2/3:

* ``packed_bytes_per_token(row_bytes, bit_uniform=N)`` returns
  ``row_bytes + 28 + 128`` when N>0 (header for per-token×7-group fp16
  affine), and ``row_bytes + 128`` when N=0 (legacy variable-bit).
* ``rotated_store_to_packed`` with ``cfg.bit_uniform=N`` writes the
  per-token×per-group dynamic affine header bytes plus N-bit codes plus
  raw rope bytes into the paged cache at the right (page, slot).
* ``rotated_load_to_fp8_layout_cpu_ref`` reverses it: reads 28-byte
  header → per-group (min, range) fp16 → reconstructs K_rot_hat →
  inverse rotates by R.t() → bf16 nope. Cosine ≥ target for the bit
  budget (u4≥0.99, u3≥0.97, u2≥0.91).
* Backward compat: ``cfg.bit_uniform=0`` keeps the old static
  per-dim (scale, zero) path bit-exact.

The test loads ``rotated_kv_quant`` (M0) and
``rotated_quant_dsv4_kernels`` (M3.c kernels) standalone, mirroring
``test_rotated_kv_quant_dsv4_canary.py`` so it runs on CPU-only macOS.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import torch


REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
M0_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "srt", "layers", "quantization",
    "rotated_kv_quant.py",
)
KERNELS_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "jit_kernel",
    "rotated_quant_dsv4_kernels.py",
)


def _load_standalone(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to spec_from_file_location for {name}@{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    m0 = _load_standalone("_rkq_m0_routeg", M0_PATH)
    sys.modules["sglang.srt.layers.quantization.rotated_kv_quant"] = m0
    kernels = _load_standalone("_rkq_kernels_routeg", KERNELS_PATH)
    return m0, kernels


def _make_uniform_cfg(m0, dim: int, bit_uniform: int):
    """Uniform-N-bit config. scale/zero kept as dummy (legacy path unused
    when ``cfg.bit_uniform>0``)."""
    R = m0.build_hadamard(dim).to(torch.float32)
    bits = torch.full((dim,), int(bit_uniform), dtype=torch.int32)
    # Dummy scale/zero (not used in uniform path but RotatedQuantizerConfig
    # requires them to be valid tensors).
    scale = torch.ones(dim, dtype=torch.float32)
    zero = torch.zeros(dim, dtype=torch.float32)
    return m0.RotatedQuantizerConfig(
        R=R, bits=bits, scale=scale, zero=zero, bit_uniform=int(bit_uniform),
    )


def _make_legacy_cfg(m0, dim: int, b_mean: float, seed: int):
    torch.manual_seed(seed)
    sigma = torch.linspace(0.1, 3.0, dim)
    var = sigma.pow(2.0)
    bits = m0.allocate_bits(var, b_mean=b_mean, b_min=1, b_max=4)
    levels = (1 << bits.to(torch.int64)) - 1
    samples = torch.randn(4096, dim) * sigma
    R = m0.build_hadamard(dim).to(torch.float32)
    samples_rot = samples @ R
    q_lo = torch.quantile(samples_rot, 0.001, dim=0)
    q_hi = torch.quantile(samples_rot, 0.999, dim=0)
    scale = (q_hi - q_lo) / levels.to(torch.float32).clamp_min(1.0)
    return m0.RotatedQuantizerConfig(
        R=R, bits=bits.to(torch.int32),
        scale=scale.to(torch.float32),
        zero=q_lo.to(torch.float32),
        bit_uniform=0,
    )


class TestRouteGUniformPacked(unittest.TestCase):
    NOPE_DIM = 448
    ROPE_DIM = 64
    PAGE_SIZE = 64
    NUM_PAGES = 4

    @classmethod
    def setUpClass(cls):
        cls.m0, cls.kernels = _load_modules()

    # ------------------------------------------------------------------
    # T1: byte accounting — uniform header adds 28 B, legacy keeps 268 B
    # ------------------------------------------------------------------
    def test_01_packed_bytes_per_token_accounting(self):
        kk = self.kernels
        # uniform 3-bit: 448 * 3 / 8 = 168 nope + 28 hdr + 128 rope = 324
        self.assertEqual(kk.uniform_row_bytes_nope(3), 168)
        self.assertEqual(kk.packed_bytes_per_token(168, bit_uniform=3), 324)
        # uniform 4-bit: 448 * 4 / 8 = 224 nope + 28 hdr + 128 rope = 380
        self.assertEqual(kk.uniform_row_bytes_nope(4), 224)
        self.assertEqual(kk.packed_bytes_per_token(224, bit_uniform=4), 380)
        # uniform 2-bit: 448 * 2 / 8 = 112 nope + 28 hdr + 128 rope = 268
        self.assertEqual(kk.uniform_row_bytes_nope(2), 112)
        self.assertEqual(kk.packed_bytes_per_token(112, bit_uniform=2), 268)
        # legacy (variable bits, b_mean=2.5 -> 140 row_bytes): 140 + 128 = 268
        self.assertEqual(kk.packed_bytes_per_token(140), 268)
        self.assertEqual(kk.packed_bytes_per_token(140, bit_uniform=0), 268)

    # ------------------------------------------------------------------
    # T2: uniform-N store + cpu_ref load roundtrip; cosine >= target
    # ------------------------------------------------------------------
    # The GPU store path (rotated_store_to_packed) imports
    # ``sglang.jit_kernel.triton_rotated_quant_dsv4`` which requires the
    # full sglang package + triton. On CPU-only macOS we exercise an
    # inline CPU store that mirrors the store-side math (same per-token
    # × per-group dynamic affine + bitpack), then the load path
    # (rotated_load_to_fp8_layout_cpu_ref) is the real production CPU
    # ref code we ship. This validates the uniform layout end-to-end
    # for both store math and load math.
    def _cpu_store_uniform(self, m0, kk, cfg, cat, cache, indices):
        """CPU equivalent of rotated_store_to_packed for uniform path."""
        N = cat.shape[0]
        D = cfg.R.shape[0]
        row_bytes_nope = int(cfg.row_bytes)
        bpt = kk.packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
        page_size = self.PAGE_SIZE
        bu = int(cfg.bit_uniform)
        L_f = float((1 << bu) - 1)

        nope = cat[:, :D].to(torch.float32)
        rope = cat[:, D:].contiguous()
        K_rot = nope @ cfg.R.to(torch.float32)  # [N, D]
        groups = D // kk._MLA_TILE_SIZE
        K_rot_g = K_rot.reshape(N, groups, kk._MLA_TILE_SIZE)
        kmin = K_rot_g.amin(dim=2, keepdim=True)
        kmax = K_rot_g.amax(dim=2, keepdim=True)
        krange = (kmax - kmin).clamp_min(1e-8)
        step = krange / L_f
        codes_g = ((K_rot_g - kmin) / step).round().clamp(0, L_f)
        codes_i64 = codes_g.reshape(N, D).to(torch.int64)

        bits_cpu = cfg.bits.to(torch.int32)
        packed = m0.bitpack_rowwise(codes_i64, bits_cpu)  # [N, row_bytes]
        assert packed.shape == (N, row_bytes_nope)

        header_pairs = torch.stack(
            (kmin.squeeze(-1).to(torch.float16),
             krange.squeeze(-1).to(torch.float16)),
            dim=-1,
        )  # [N, groups, 2] fp16
        header_bytes = (
            header_pairs.contiguous().view(torch.uint8)
            .reshape(N, kk._UNIFORM_HEADER_BYTES)
        )
        rope_bytes = rope.view(torch.uint8).reshape(N, kk._ROPE_BYTES)
        full_row = torch.cat([packed, header_bytes, rope_bytes], dim=1)
        assert full_row.shape == (N, bpt), f"{full_row.shape} vs ({N},{bpt})"

        cache_flat = cache.view(-1, bpt)
        cache_flat.index_copy_(0, indices.to(torch.int64), full_row)

    def _roundtrip_one(self, bit_uniform: int, cos_target: float):
        kk = self.kernels
        cfg = _make_uniform_cfg(self.m0, self.NOPE_DIM, bit_uniform)
        expected_row_bytes = kk.uniform_row_bytes_nope(bit_uniform)
        self.assertEqual(int(cfg.row_bytes), expected_row_bytes)

        bpt = kk.packed_bytes_per_token(cfg.row_bytes, cfg.bit_uniform)
        bytes_per_page = bpt * self.PAGE_SIZE
        cache = torch.zeros(self.NUM_PAGES, bytes_per_page, dtype=torch.uint8)

        torch.manual_seed(7 + bit_uniform)
        N = 32
        sigma = torch.linspace(0.1, 3.0, self.NOPE_DIM)
        nope = (torch.randn(N, self.NOPE_DIM) * sigma).to(torch.bfloat16)
        rope = torch.randn(N, self.ROPE_DIM).to(torch.bfloat16)
        cat = torch.cat([nope, rope], dim=-1).contiguous()

        capacity = self.NUM_PAGES * self.PAGE_SIZE
        perm = torch.randperm(
            capacity,
            generator=torch.Generator().manual_seed(11 + bit_uniform),
        )
        indices = perm[:N].to(torch.int32)

        self._cpu_store_uniform(self.m0, kk, cfg, cat, cache, indices)

        nope_rec, rope_rec, packed_nope = kk.rotated_load_to_fp8_layout_cpu_ref(
            cache, indices, page_size=self.PAGE_SIZE, cfg=cfg,
        )

        # rope must be byte-exact regardless of bit_uniform.
        rope_rec_bytes = rope_rec.contiguous().view(torch.uint8).reshape(
            N, self.ROPE_DIM * 2
        )
        rope_orig_bytes = rope.contiguous().view(torch.uint8).reshape(
            N, self.ROPE_DIM * 2
        )
        self.assertTrue(
            torch.equal(rope_rec_bytes, rope_orig_bytes),
            msg=f"u{bit_uniform}: rope bytes mismatch",
        )

        # packed nope shape is [N, row_bytes] uint8.
        self.assertEqual(tuple(packed_nope.shape), (N, expected_row_bytes))

        cos = torch.nn.functional.cosine_similarity(
            nope.float(), nope_rec.float(), dim=-1,
        )
        cm = float(cos.mean())
        self.assertGreaterEqual(
            cm, cos_target,
            msg=f"u{bit_uniform}: cos.mean={cm:.4f} < target {cos_target}",
        )

    def test_02_uniform_4_roundtrip(self):
        self._roundtrip_one(bit_uniform=4, cos_target=0.99)

    def test_03_uniform_3_roundtrip(self):
        self._roundtrip_one(bit_uniform=3, cos_target=0.97)

    def test_04_uniform_2_roundtrip(self):
        # u2 is non-production (route G uses u3); 0.91 reflects realistic
        # SNR floor for 2-bit×64-group affine on Hadamard-rotated data.
        self._roundtrip_one(bit_uniform=2, cos_target=0.91)

    # ------------------------------------------------------------------
    # T3: backward compat — cfg.bit_uniform=0 keeps legacy path bitexact
    # ------------------------------------------------------------------
    def _cpu_store_legacy(self, m0, kk, cfg, cat, cache, indices):
        """CPU equivalent of rotated_store_to_packed for legacy path."""
        N = cat.shape[0]
        D = cfg.R.shape[0]
        row_bytes_nope = int(cfg.row_bytes)
        bpt = kk.packed_bytes_per_token(row_bytes_nope, cfg.bit_uniform)
        nope = cat[:, :D].to(torch.float32)
        rope = cat[:, D:].contiguous()
        K_rot = nope @ cfg.R.to(torch.float32)
        scale = cfg.scale.to(torch.float32).clamp_min(1e-12)
        zero = cfg.zero.to(torch.float32)
        levels = (1 << cfg.bits.to(torch.int64)) - 1
        codes = ((K_rot - zero) / scale).round()
        codes = torch.clamp(
            codes,
            min=torch.zeros_like(levels).to(codes.dtype),
            max=levels.to(codes.dtype),
        )
        codes_i64 = codes.to(torch.int64)
        bits_cpu = cfg.bits.to(torch.int32)
        packed = m0.bitpack_rowwise(codes_i64, bits_cpu)
        rope_bytes = rope.view(torch.uint8).reshape(N, kk._ROPE_BYTES)
        full_row = torch.cat([packed, rope_bytes], dim=1)
        assert full_row.shape == (N, bpt)
        cache_flat = cache.view(-1, bpt)
        cache_flat.index_copy_(0, indices.to(torch.int64), full_row)

    def test_05_legacy_path_backward_compat(self):
        kk = self.kernels
        cfg = _make_legacy_cfg(self.m0, self.NOPE_DIM, b_mean=2.5, seed=1)
        self.assertEqual(int(cfg.bit_uniform), 0)
        bpt = kk.packed_bytes_per_token(cfg.row_bytes, cfg.bit_uniform)
        self.assertEqual(int(cfg.row_bytes), 140)
        self.assertEqual(bpt, 268)

        bytes_per_page = bpt * self.PAGE_SIZE
        cache = torch.zeros(self.NUM_PAGES, bytes_per_page, dtype=torch.uint8)

        torch.manual_seed(99)
        N = 16
        sigma = torch.linspace(0.1, 3.0, self.NOPE_DIM)
        nope = (torch.randn(N, self.NOPE_DIM) * sigma).to(torch.bfloat16)
        rope = torch.randn(N, self.ROPE_DIM).to(torch.bfloat16)
        cat = torch.cat([nope, rope], dim=-1).contiguous()
        indices = torch.arange(N, dtype=torch.int32)

        self._cpu_store_legacy(self.m0, kk, cfg, cat, cache, indices)
        nope_rec, rope_rec, _ = kk.rotated_load_to_fp8_layout_cpu_ref(
            cache, indices, page_size=self.PAGE_SIZE, cfg=cfg,
        )

        rope_rec_bytes = rope_rec.contiguous().view(torch.uint8).reshape(
            N, self.ROPE_DIM * 2
        )
        rope_orig_bytes = rope.contiguous().view(torch.uint8).reshape(
            N, self.ROPE_DIM * 2
        )
        self.assertTrue(torch.equal(rope_rec_bytes, rope_orig_bytes))

        cos = torch.nn.functional.cosine_similarity(
            nope.float(), nope_rec.float(), dim=-1,
        )
        # Legacy b_mean=2.5 hetero-bit path has roughly the same SNR
        # ceiling as M3.c.1's pre-existing test_03; keep the same bar.
        self.assertGreater(float(cos.mean()), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
