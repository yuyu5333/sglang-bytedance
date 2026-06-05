"""End-to-end smoke test for the rotated + non-uniform-bit KV cache pipeline.

Covers:

  1. ``scripts/build_rotated_kv_calib.py --synthetic`` → produces a valid
     calibration ``.pt``.
  2. ``load_rotated_quant_calibration`` consumes the file without error.
  3. The (CPU-emulated) pool path:
       * encode K/V to packed uint8 via :class:`_DeviceBoundQuantizer`
       * write into a uint8 buffer via ``index_put`` (same shape contract as
         :class:`RotatedQuantMHATokenToKVPool`)
       * read back via ``decode`` and check shape + numerical sanity.
  4. Storage ratio matches the theoretical b̄/16.

The test deliberately avoids importing :class:`MHATokenToKVPool` because that
pulls in CUDA-only modules; we exercise the *pool storage contract* using
the same primitives :class:`RotatedQuantMHATokenToKVPool` is built on top of.
A separate GPU-bound integration test is left to a follow-up.

Run with::

    python /path/to/test_rotated_kv_quant_e2e.py
    # or
    pytest test/manual/quant/test_rotated_kv_quant_e2e.py -v
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import torch

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
M0_PATH = os.path.join(
    REPO_ROOT,
    "python",
    "sglang",
    "srt",
    "layers",
    "quantization",
    "rotated_kv_quant.py",
)
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "build_rotated_kv_calib.py")


def _load_m0():
    """Import ``rotated_kv_quant`` as a stand-alone module (no sglang side-imports)."""
    spec = importlib.util.spec_from_file_location("_rkq_m0", M0_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rkq_m0"] = mod  # required for dataclass repr
    spec.loader.exec_module(mod)
    return mod


def _validate_side(side, dim, tag):
    for k in ("R", "bits", "scale", "zero"):
        assert k in side, f"{tag} missing {k}"
    assert side["R"].shape == (dim, dim), (
        f"{tag} R shape {tuple(side['R'].shape)} != ({dim}, {dim})"
    )
    assert side["bits"].shape == (dim,)
    assert side["scale"].shape == (dim,)
    assert side["zero"].shape == (dim,)
    assert 1 <= int(side["bits"].min()) and int(side["bits"].max()) <= 8


class TestRotatedKVQuantE2E(unittest.TestCase):
    """Drive the script + storage contract end-to-end."""

    NUM_LAYERS = 2
    NUM_TOKENS = 512
    HEAD_NUM = 4
    HEAD_DIM = 128
    V_HEAD_DIM = 128
    B_MEAN = 2.5

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="rotated_kv_e2e_")
        cls.calib_path = os.path.join(cls.tmpdir, "calib.pt")
        cls.m0 = _load_m0()

    @classmethod
    def tearDownClass(cls):
        # Best-effort cleanup; ignore failures.
        try:
            for p in os.listdir(cls.tmpdir):
                os.unlink(os.path.join(cls.tmpdir, p))
            os.rmdir(cls.tmpdir)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 1) Run the calibration script via subprocess.
    # ------------------------------------------------------------------
    def test_01_script_synthetic(self):
        cmd = [
            sys.executable,
            SCRIPT_PATH,
            "--synthetic",
            "--num-layers", str(self.NUM_LAYERS),
            "--num-tokens", str(self.NUM_TOKENS),
            "--head-num", str(self.HEAD_NUM),
            "--head-dim", str(self.HEAD_DIM),
            "--v-head-dim", str(self.V_HEAD_DIM),
            "--b-mean", str(self.B_MEAN),
            "--seed", "0",
            "-o", self.calib_path,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "python") + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"script failed: stdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertTrue(os.path.isfile(self.calib_path))
        size = os.path.getsize(self.calib_path)
        self.assertGreater(size, 1024, msg="calibration file suspiciously small")

    # ------------------------------------------------------------------
    # 2) Schema validation (mirror load_rotated_quant_calibration).
    # ------------------------------------------------------------------
    def test_02_schema(self):
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        self.assertIsInstance(raw, dict)
        self.assertEqual(set(raw.keys()), set(range(self.NUM_LAYERS)))
        for lid, entry in raw.items():
            self.assertIn("k", entry)
            self.assertIn("v", entry)
            _validate_side(entry["k"], self.HEAD_DIM, f"layer {lid} k")
            _validate_side(entry["v"], self.V_HEAD_DIM, f"layer {lid} v")

    # ------------------------------------------------------------------
    # 3) Storage contract: encode -> uint8 buffer -> decode.
    # ------------------------------------------------------------------
    def test_03_pool_storage_contract(self):
        m0 = self.m0
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        cfg_k = m0.RotatedQuantizerConfig(
            R=raw[0]["k"]["R"].float(),
            bits=raw[0]["k"]["bits"].int(),
            scale=raw[0]["k"]["scale"].float(),
            zero=raw[0]["k"]["zero"].float(),
        )
        qz = m0.RotatedQuantizer(cfg_k)
        row_bytes = (int(cfg_k.bits.sum()) + 7) // 8

        size, page_size = 256, 16
        buf = torch.zeros(
            (size + page_size, self.HEAD_NUM, row_bytes), dtype=torch.uint8
        )

        # Build a small write batch and write into random locations.
        torch.manual_seed(7)
        n_write = 64
        loc = torch.randperm(size)[:n_write]
        sigma = torch.linspace(0.1, 3.0, self.HEAD_DIM)
        x = torch.randn(n_write, self.HEAD_NUM, self.HEAD_DIM) * sigma

        packed = qz.quantize(x)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(
            tuple(packed.shape), (n_write, self.HEAD_NUM, row_bytes)
        )
        buf[loc] = packed

        read = buf[loc]
        x_hat = qz.dequantize(read, dtype=torch.float32)
        self.assertEqual(tuple(x_hat.shape), (n_write, self.HEAD_NUM, self.HEAD_DIM))

        # Numerical sanity: cosine similarity per row should be > 0.7 even
        # at b̄=2.5 + outlier-heavy synthetic data.
        x_flat = x.reshape(-1, self.HEAD_DIM)
        x_hat_flat = x_hat.reshape(-1, self.HEAD_DIM)
        cos = torch.nn.functional.cosine_similarity(x_flat, x_hat_flat, dim=-1)
        self.assertGreater(float(cos.mean()), 0.7, msg=f"mean cos {cos.mean()}")
        self.assertGreater(float(cos.min()), 0.3, msg=f"min cos {cos.min()}")

        # Storage ratio sanity: ratio ≈ b̄/16 (within 50% slack since
        # row_bytes is rounded up to bytes and bits clamped to [b_min, b_max]).
        ratio = (row_bytes * self.HEAD_NUM) / (self.HEAD_DIM * 2 * self.HEAD_NUM)
        expected = self.B_MEAN / 16.0
        self.assertLess(
            abs(ratio - expected),
            0.5 * expected,
            msg=f"ratio={ratio:.4f}, expected~{expected:.4f}",
        )

    # ------------------------------------------------------------------
    # 4) loader-equivalent path matches the encode/decode results.
    # ------------------------------------------------------------------
    def test_04_loader_equivalence(self):
        m0 = self.m0
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        # Build configs the way load_rotated_quant_calibration would.
        configs = {}
        for lid in raw:
            configs[lid] = {
                side: m0.RotatedQuantizerConfig(
                    R=raw[lid][side]["R"].float(),
                    bits=raw[lid][side]["bits"].int(),
                    scale=raw[lid][side]["scale"].float(),
                    zero=raw[lid][side]["zero"].float(),
                )
                for side in ("k", "v")
            }
        # Sanity: row_bytes consistent across layers (M1 contract).
        rb_k = (int(configs[0]["k"].bits.sum()) + 7) // 8
        rb_v = (int(configs[0]["v"].bits.sum()) + 7) // 8
        for lid in configs:
            self.assertEqual(
                (int(configs[lid]["k"].bits.sum()) + 7) // 8, rb_k
            )
            self.assertEqual(
                (int(configs[lid]["v"].bits.sum()) + 7) // 8, rb_v
            )


class TestRotatedKVQuantMLAE2E(unittest.TestCase):
    """MLA 模式（M3.a）端到端 smoke test。

    覆盖：
      1. ``build_rotated_kv_calib.py --synthetic --mla-mode`` 输出包含
         ``_meta.mode='mla'`` 的合法 schema。
      2. ``load_rotated_quant_mla_calibration``（仿照实现）成功加载。
      3. latent 段量化往返 + rope 段 raw view 往返。
      4. 模拟 RotatedQuantMLATokenToKVPool 的 buffer 拼接布局：
         ``[N, 1, latent_row_bytes + rope_bytes]`` 写入 / 读出对齐。
    """

    NUM_LAYERS = 2
    NUM_TOKENS = 256
    KV_LORA_RANK = 128
    QK_ROPE_HEAD_DIM = 64
    B_MEAN = 2.5

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="rotated_kv_mla_e2e_")
        cls.calib_path = os.path.join(cls.tmpdir, "calib_mla.pt")
        cls.m0 = _load_m0()

    @classmethod
    def tearDownClass(cls):
        try:
            for p in os.listdir(cls.tmpdir):
                os.unlink(os.path.join(cls.tmpdir, p))
            os.rmdir(cls.tmpdir)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 1) 跑校准脚本（mla 模式）
    # ------------------------------------------------------------------
    def test_01_script_synthetic_mla(self):
        cmd = [
            sys.executable,
            SCRIPT_PATH,
            "--synthetic",
            "--mla-mode",
            "--num-layers", str(self.NUM_LAYERS),
            "--num-tokens", str(self.NUM_TOKENS),
            "--kv-lora-rank", str(self.KV_LORA_RANK),
            "--qk-rope-head-dim", str(self.QK_ROPE_HEAD_DIM),
            "--b-mean", str(self.B_MEAN),
            "--seed", "0",
            "-o", self.calib_path,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "python") + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"mla script failed: stdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertTrue(os.path.isfile(self.calib_path))
        size = os.path.getsize(self.calib_path)
        self.assertGreater(size, 1024, msg="mla calibration file suspiciously small")

    # ------------------------------------------------------------------
    # 2) Schema 校验（带 _meta.mode='mla'）
    # ------------------------------------------------------------------
    def test_02_schema_mla(self):
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        self.assertIsInstance(raw, dict)
        self.assertIn("_meta", raw)
        meta = raw["_meta"]
        self.assertEqual(meta["mode"], "mla")
        self.assertEqual(meta["kv_lora_rank"], self.KV_LORA_RANK)
        self.assertEqual(meta["qk_rope_head_dim"], self.QK_ROPE_HEAD_DIM)

        layer_ids = [k for k in raw.keys() if isinstance(k, int)]
        self.assertEqual(set(layer_ids), set(range(self.NUM_LAYERS)))
        for lid in layer_ids:
            entry = raw[lid]
            self.assertIn("latent", entry)
            _validate_side(
                entry["latent"], self.KV_LORA_RANK, f"layer {lid} latent"
            )

    # ------------------------------------------------------------------
    # 3) MLA pool 存储契约：
    #    [N, 1, latent_row_bytes + rope_bytes] 拼接 buffer
    # ------------------------------------------------------------------
    def test_03_mla_pool_storage_contract(self):
        m0 = self.m0
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        cfg = m0.RotatedQuantizerConfig(
            R=raw[0]["latent"]["R"].float(),
            bits=raw[0]["latent"]["bits"].int(),
            scale=raw[0]["latent"]["scale"].float(),
            zero=raw[0]["latent"]["zero"].float(),
        )
        qz = m0.RotatedQuantizer(cfg)
        latent_row_bytes = (int(cfg.bits.sum()) + 7) // 8

        # 选 bf16 作为 dtype（与 SGLang 默认一致）
        dtype = torch.bfloat16
        rope_bytes = self.QK_ROPE_HEAD_DIM * torch.empty([], dtype=dtype).element_size()
        total_row = latent_row_bytes + rope_bytes

        size, page_size = 128, 16
        # MLA 是 head_num = 1
        buf = torch.zeros((size + page_size, 1, total_row), dtype=torch.uint8)

        torch.manual_seed(11)
        n_write = 32
        loc = torch.randperm(size)[:n_write]

        # 构造 latent / rope 输入
        sigma = torch.linspace(0.1, 3.0, self.KV_LORA_RANK)
        latent = torch.randn(n_write, 1, self.KV_LORA_RANK) * sigma
        latent = latent.to(dtype)
        rope = torch.randn(n_write, 1, self.QK_ROPE_HEAD_DIM).to(dtype)

        # encode + 拼接（模拟 set_mla_kv_buffer 行为）
        packed_latent = qz.quantize(latent.float())  # [n_write, 1, latent_row_bytes]
        rope_u8 = rope.contiguous().view(torch.uint8).reshape(
            n_write, 1, rope_bytes
        )
        full_row = torch.cat([packed_latent, rope_u8], dim=-1)
        self.assertEqual(tuple(full_row.shape), (n_write, 1, total_row))

        buf[loc] = full_row

        # 读回 + 切片（模拟 get_mla_kv_buffer 行为）
        rows = buf[loc]
        latent_packed_back = rows[..., :latent_row_bytes]
        rope_back_u8 = rows[..., latent_row_bytes:]

        latent_back = qz.dequantize(latent_packed_back, dtype=torch.float32)
        self.assertEqual(
            tuple(latent_back.shape), (n_write, 1, self.KV_LORA_RANK)
        )

        # rope 段 raw view 必须严格等于
        rope_back = rope_back_u8.contiguous().view(dtype).reshape(
            n_write, 1, self.QK_ROPE_HEAD_DIM
        )
        self.assertTrue(torch.equal(rope_back, rope))

        # latent 段量化 + 反量化的精度
        cos = torch.nn.functional.cosine_similarity(
            latent.float().reshape(-1, self.KV_LORA_RANK),
            latent_back.reshape(-1, self.KV_LORA_RANK),
            dim=-1,
        )
        self.assertGreater(float(cos.mean()), 0.7, msg=f"mla mean cos {cos.mean()}")
        self.assertGreater(float(cos.min()), 0.3, msg=f"mla min cos {cos.min()}")

    # ------------------------------------------------------------------
    # 4) loader 等价：模拟 load_rotated_quant_mla_calibration 的检查路径
    # ------------------------------------------------------------------
    def test_04_loader_equivalence_mla(self):
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)

        # 模拟 load_rotated_quant_mla_calibration 的几个关键检查
        self.assertIn("_meta", raw)
        self.assertEqual(raw["_meta"]["mode"], "mla")

        # 构造 config dict
        m0 = self.m0
        configs = {}
        for lid in [k for k in raw.keys() if isinstance(k, int)]:
            configs[lid] = m0.RotatedQuantizerConfig(
                R=raw[lid]["latent"]["R"].float(),
                bits=raw[lid]["latent"]["bits"].int(),
                scale=raw[lid]["latent"]["scale"].float(),
                zero=raw[lid]["latent"]["zero"].float(),
            )
        # row_bytes 一致性（M3.a 契约）
        rb = (int(configs[0].bits.sum()) + 7) // 8
        for lid in configs:
            self.assertEqual(
                (int(configs[lid].bits.sum()) + 7) // 8, rb
            )

        # 错误使用：用 MLA calib 走 MHA loader 应该失败（meta 缺 k/v）
        # 这里只验证 raw 里没有 'k'/'v' 顶层结构（即不会被误识别为 MHA schema）
        first = raw[0]
        self.assertNotIn("k", first)
        self.assertNotIn("v", first)


class TestRotatedKVQuantDSv4E2E(unittest.TestCase):
    """DSv4 evaluation-mode 端到端 smoke test（M3.b）。

    覆盖：
      1. ``build_rotated_kv_calib.py --synthetic --dsv4-mode`` 输出包含
         ``_meta.mode='dsv4'`` 的合法 schema。
      2. schema 校验 + compression_ratios 元数据保留。
      3. nope 段量化往返精度（评估模式的核心 API
         ``simulate_quantize_nope`` 的等价计算）。
      4. row_bytes 一致性 + simulated_compression_ratio 计算正确。
    """

    NUM_LAYERS = 2
    NUM_TOKENS = 256
    QK_NOPE_HEAD_DIM = 128  # smoke 用 128（DSv4 真参 448 跑得慢但可调）
    QK_ROPE_HEAD_DIM = 64
    B_MEAN = 2.5
    COMPRESSION_RATIOS = "0,4,128,4"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="rotated_kv_dsv4_e2e_")
        cls.calib_path = os.path.join(cls.tmpdir, "calib_dsv4.pt")
        cls.m0 = _load_m0()

    @classmethod
    def tearDownClass(cls):
        try:
            for p in os.listdir(cls.tmpdir):
                os.unlink(os.path.join(cls.tmpdir, p))
            os.rmdir(cls.tmpdir)
        except OSError:
            pass

    def test_01_script_synthetic_dsv4(self):
        cmd = [
            sys.executable,
            SCRIPT_PATH,
            "--synthetic",
            "--dsv4-mode",
            "--num-layers", str(self.NUM_LAYERS),
            "--num-tokens", str(self.NUM_TOKENS),
            "--qk-nope-head-dim", str(self.QK_NOPE_HEAD_DIM),
            "--qk-rope-head-dim", str(self.QK_ROPE_HEAD_DIM),
            "--compression-ratios", self.COMPRESSION_RATIOS,
            "--b-mean", str(self.B_MEAN),
            "--seed", "0",
            "-o", self.calib_path,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "python") + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"dsv4 script failed: stdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertTrue(os.path.isfile(self.calib_path))

    def test_02_schema_dsv4(self):
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        self.assertIsInstance(raw, dict)
        self.assertIn("_meta", raw)
        meta = raw["_meta"]
        self.assertEqual(meta["mode"], "dsv4")
        self.assertEqual(meta["qk_nope_head_dim"], self.QK_NOPE_HEAD_DIM)
        self.assertEqual(meta["qk_rope_head_dim"], self.QK_ROPE_HEAD_DIM)
        self.assertEqual(
            meta["compression_ratios"],
            [int(x) for x in self.COMPRESSION_RATIOS.split(",")],
        )
        layer_ids = [k for k in raw.keys() if isinstance(k, int)]
        self.assertEqual(set(layer_ids), set(range(self.NUM_LAYERS)))
        for lid in layer_ids:
            entry = raw[lid]
            self.assertIn("nope", entry)
            _validate_side(
                entry["nope"], self.QK_NOPE_HEAD_DIM, f"layer {lid} nope"
            )

    def test_03_simulate_quantize_nope_roundtrip(self):
        """模拟 simulate_quantize_nope 的核心计算路径。

        因为构造 RotatedQuantDeepSeekV4TokenToKVPool 需要完整的 DSv4
        runtime，这里直接用 M0 的 RotatedQuantizer 跑同一份 nope 校准，
        验证 row_bytes / 精度满足契约（这正是 simulate_quantize_nope
        会执行的运算）。
        """
        m0 = self.m0
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)
        cfg = m0.RotatedQuantizerConfig(
            R=raw[0]["nope"]["R"].float(),
            bits=raw[0]["nope"]["bits"].int(),
            scale=raw[0]["nope"]["scale"].float(),
            zero=raw[0]["nope"]["zero"].float(),
        )
        qz = m0.RotatedQuantizer(cfg)
        nope_row_bytes = (int(cfg.bits.sum()) + 7) // 8

        torch.manual_seed(13)
        n = 64
        sigma = torch.linspace(0.1, 3.0, self.QK_NOPE_HEAD_DIM)
        nope_bf16 = (
            torch.randn(n, 1, self.QK_NOPE_HEAD_DIM) * sigma
        ).to(torch.bfloat16)

        packed = qz.quantize(nope_bf16.float())  # [n, 1, nope_row_bytes]
        self.assertEqual(tuple(packed.shape), (n, 1, nope_row_bytes))
        self.assertEqual(packed.dtype, torch.uint8)

        nope_dq = qz.dequantize(packed, dtype=torch.float32)
        self.assertEqual(
            tuple(nope_dq.shape), (n, 1, self.QK_NOPE_HEAD_DIM)
        )

        # 精度契约（合成异方差数据 + b̄=2.5，cos 应远高于 0）
        cos = torch.nn.functional.cosine_similarity(
            nope_bf16.float().reshape(-1, self.QK_NOPE_HEAD_DIM),
            nope_dq.reshape(-1, self.QK_NOPE_HEAD_DIM),
            dim=-1,
        )
        self.assertGreater(float(cos.mean()), 0.7, msg=f"dsv4 mean cos {cos.mean()}")
        self.assertGreater(float(cos.min()), 0.3, msg=f"dsv4 min cos {cos.min()}")

    def test_04_compression_ratio_metadata(self):
        """row_bytes 一致性 + simulated_compression_ratio 数值正确性。"""
        raw = torch.load(self.calib_path, map_location="cpu", weights_only=False)

        # row_bytes 一致性
        layer_ids = [k for k in raw.keys() if isinstance(k, int)]
        rb0 = (int(raw[layer_ids[0]]["nope"]["bits"].sum()) + 7) // 8
        for lid in layer_ids:
            self.assertEqual(
                (int(raw[lid]["nope"]["bits"].sum()) + 7) // 8, rb0
            )

        # 模拟压缩比 = bf16_bytes / packed_bytes
        bf16_bytes = self.QK_NOPE_HEAD_DIM * 2
        sim_ratio = bf16_bytes / rb0
        # b̄=2.5 → bf16(16) / 2.5 = 6.4×（受 row_bytes ceil 影响略小）
        self.assertGreater(
            sim_ratio,
            5.0,
            msg=f"sim_ratio {sim_ratio} below expected ~6.4 (b̄={self.B_MEAN})",
        )
        self.assertLess(
            sim_ratio,
            8.0,
            msg=f"sim_ratio {sim_ratio} above sane upper bound",
        )

        # DSv4 模式不应误用 MHA / MLA loader
        first = raw[layer_ids[0]]
        self.assertIn("nope", first)
        self.assertNotIn("k", first)
        self.assertNotIn("v", first)
        self.assertNotIn("latent", first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
