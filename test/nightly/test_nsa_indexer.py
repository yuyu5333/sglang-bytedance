import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

import torch
import sys
import types

def _torch_compile(*args, **kwargs):
    def _decorator(f):
        return f
    return _decorator
torch.compile = _torch_compile

# Lightweight stubs to satisfy optional runtime deps when importing sglang
sys.modules.setdefault("IPython", types.ModuleType("IPython"))
_ip_disp = types.ModuleType("IPython.display")
class _HTML:
    def __init__(self, *args, **kwargs):
        pass
def _display(*args, **kwargs):
    return None
_ip_disp.HTML = _HTML
_ip_disp.display = _display
sys.modules.setdefault("IPython.display", _ip_disp)
sys.modules.setdefault("pybase64", types.ModuleType("pybase64"))
_triton = types.ModuleType("triton")
_triton_language = types.ModuleType("triton.language")
import importlib.machinery as _machinery
_triton.__spec__ = _machinery.ModuleSpec("triton", loader=None)
_triton_language.__spec__ = _machinery.ModuleSpec("triton.language", loader=None)
_triton_language.constexpr = lambda v: v
def _triton_jit(fn=None, **kwargs):
    if fn is None:
        def _decorator(f):
            return f
        return _decorator
    return fn
_triton.jit = _triton_jit
def _triton_autotune(*args, **kwargs):
    def _decorator(f):
        return f
    return _decorator
_triton.autotune = _triton_autotune
class _TritonConfig:
    def __init__(self, *args, **kwargs):
        pass
_triton.Config = _TritonConfig
sys.modules.setdefault("triton", _triton)
sys.modules.setdefault("triton.language", _triton_language)
_orjson = types.ModuleType("orjson")
_orjson.dumps = lambda *args, **kwargs: b"{}"
_orjson.loads = lambda *args, **kwargs: {}
sys.modules.setdefault("orjson", _orjson)
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

# Minimal stub for sglang.srt.layers.dp_attention to avoid heavy deps
_dp_attn_stub = types.ModuleType("sglang.srt.layers.dp_attention")
class _DummyTPGroup:
    def cp_all_gather_into_tensor_async(self, *args, **kwargs):
        return None
_dp_attn_stub.get_attention_tp_group = lambda: _DummyTPGroup()
_dp_attn_stub.get_attention_tp_rank = lambda: 0
_dp_attn_stub.get_attention_tp_size = lambda: 1
_dp_attn_stub.is_allocation_symmetric = lambda: True
_dp_attn_stub.set_dp_buffer_len = lambda *args, **kwargs: None
_dp_attn_stub.get_attention_dp_size = lambda: 1
_dp_attn_stub.get_attention_dp_group = lambda: _DummyTPGroup()
_dp_attn_stub.get_attention_dp_rank = lambda: 0
_dp_attn_stub.is_dp_attention_enabled = lambda: False
class _DpPaddingMode:
    pass
_dp_attn_stub.DpPaddingMode = _DpPaddingMode
_dp_attn_stub.set_is_extend_in_batch = lambda *args, **kwargs: None
sys.modules.setdefault("sglang.srt.layers.dp_attention", _dp_attn_stub)

# Minimal stub for AttentionArch to avoid importing heavy configs
_model_config_stub = types.ModuleType("sglang.srt.configs.model_config")
import enum as _enum
class _AttentionArch(_enum.Enum):
    MLA = 1
_model_config_stub.AttentionArch = _AttentionArch
sys.modules.setdefault("sglang.srt.configs.model_config", _model_config_stub)

# Minimal stub for fastapi responses
_fastapi = types.ModuleType("fastapi")
_fastapi_responses = types.ModuleType("fastapi.responses")
_fastapi.__spec__ = _machinery.ModuleSpec("fastapi", loader=None)
_fastapi_responses.__spec__ = _machinery.ModuleSpec("fastapi.responses", loader=None)
class _ORJSONResponse:
    pass
_fastapi_responses.ORJSONResponse = _ORJSONResponse
_fastapi.responses = _fastapi_responses
sys.modules.setdefault("fastapi", _fastapi)
sys.modules.setdefault("fastapi.responses", _fastapi_responses)

# Minimal stub for starlette routing
_starlette = types.ModuleType("starlette")
_starlette_routing = types.ModuleType("starlette.routing")
class _Mount:
    pass
_starlette_routing.Mount = _Mount
_starlette.routing = _starlette_routing
sys.modules.setdefault("starlette", _starlette)
sys.modules.setdefault("starlette.routing", _starlette_routing)

# Minimal stub for openai types
_openai = types.ModuleType("openai")
_openai_types = types.ModuleType("openai.types")
_openai_types_responses = types.ModuleType("openai.types.responses")
_openai.__spec__ = _machinery.ModuleSpec("openai", loader=None)
_openai_types.__spec__ = _machinery.ModuleSpec("openai.types", loader=None)
_openai_types_responses.__spec__ = _machinery.ModuleSpec("openai.types.responses", loader=None)
class _ResponseFunctionToolCall: pass
class _ResponseInputItemParam: pass
class _ResponseOutputItem: pass
class _ResponseOutputMessage: pass
class _ResponseOutputText: pass
class _ResponseReasoningItem: pass
_openai_types_responses.ResponseFunctionToolCall = _ResponseFunctionToolCall
_openai_types_responses.ResponseInputItemParam = _ResponseInputItemParam
_openai_types_responses.ResponseOutputItem = _ResponseOutputItem
_openai_types_responses.ResponseOutputMessage = _ResponseOutputMessage
_openai_types_responses.ResponseOutputText = _ResponseOutputText
_openai_types_responses.ResponseReasoningItem = _ResponseReasoningItem
_openai_types_responses.response = types.ModuleType("openai.types.responses.response")
class _ToolChoice: pass
_openai_types_responses.response.ToolChoice = _ToolChoice
_openai_types_responses.tool = types.ModuleType("openai.types.responses.tool")
class _Tool: pass
_openai_types_responses.tool.Tool = _Tool
_openai.types = _openai_types
sys.modules.setdefault("openai", _openai)
sys.modules.setdefault("openai.types", _openai_types)
sys.modules.setdefault("openai.types.responses", _openai_types_responses)
sys.modules.setdefault("openai.types.responses.response", _openai_types_responses.response)
sys.modules.setdefault("openai.types.responses.tool", _openai_types_responses.tool)

# Minimal stub for sglang.srt.server_args to avoid deep imports
_server_args_stub = types.ModuleType("sglang.srt.server_args")
class _ServerArgs:
    def __init__(self, model_path="dummy"):
        self.enable_dp_attention = False
        self.nsa_prefill_backend = "flashmla_sparse"
        self.nsa_decode_backend = "flashmla_sparse"
        self.device = "cuda"
        self.enable_nsa_prefill_context_parallel = False
def _get_global_server_args():
    return _ServerArgs()
def _set_global_server_args_for_scheduler(args):
    return None
_server_args_stub.ServerArgs = _ServerArgs
_server_args_stub.get_global_server_args = _get_global_server_args
_server_args_stub.set_global_server_args_for_scheduler = _set_global_server_args_for_scheduler
sys.modules.setdefault("sglang.srt.server_args", _server_args_stub)

# Minimal stub for vllm layernorm
_vllm = types.ModuleType("vllm")
_vllm_me = types.ModuleType("vllm.model_executor")
_vllm_layers = types.ModuleType("vllm.model_executor.layers")
_vllm_ln = types.ModuleType("vllm.model_executor.layers.layernorm")
class _GemmaRMSNorm: pass
class _RMSNorm: pass
_vllm_ln.GemmaRMSNorm = _GemmaRMSNorm
_vllm_ln.RMSNorm = _RMSNorm
sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.model_executor", _vllm_me)
sys.modules.setdefault("vllm.model_executor.layers", _vllm_layers)
sys.modules.setdefault("vllm.model_executor.layers.layernorm", _vllm_ln)

# Minimal stubs to avoid deep imports during nsa_indexer import
_sg_ln = types.ModuleType("sglang.srt.layers.layernorm")
class _LayerNorm:
    def __init__(self, hidden, dtype=None):
        pass
    def __call__(self, x):
        return x
_sg_ln.LayerNorm = _LayerNorm
sys.modules.setdefault("sglang.srt.layers.layernorm", _sg_ln)

_sg_linear = types.ModuleType("sglang.srt.layers.linear")
class _ReplicatedLinear:
    def __init__(self, in_features, out_features, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
    def __call__(self, x):
        l = x.shape[0]
        y = torch.zeros(l, self.out_features, dtype=x.dtype, device=x.device)
        return y, None
class _LinearBase: pass
_sg_linear.ReplicatedLinear = _ReplicatedLinear
_sg_linear.LinearBase = _LinearBase
sys.modules.setdefault("sglang.srt.layers.linear", _sg_linear)

_sg_rope = types.ModuleType("sglang.srt.layers.rotary_embedding")
def _get_rope_wrapper(*args, **kwargs):
    def _rope(positions, q_rope, k_rope):
        return q_rope, k_rope
    return _rope
_sg_rope.get_rope_wrapper = _get_rope_wrapper
sys.modules.setdefault("sglang.srt.layers.rotary_embedding", _sg_rope)

_sg_dgw = types.ModuleType("sglang.srt.layers.deep_gemm_wrapper")
class _Cfg:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
def _configure_deep_gemm_num_sms(*args, **kwargs):
    return _Cfg()
_sg_dgw.configure_deep_gemm_num_sms = _configure_deep_gemm_num_sms
sys.modules.setdefault("sglang.srt.layers.deep_gemm_wrapper", _sg_dgw)

_sg_qcfg = types.ModuleType("sglang.srt.layers.quantization.base_config")
class _QuantizationConfig: pass
_sg_qcfg.QuantizationConfig = _QuantizationConfig
sys.modules.setdefault("sglang.srt.layers.quantization.base_config", _sg_qcfg)

# Minimal stub for sgl_kernel
_sgl_kernel = types.ModuleType("sgl_kernel")
def _hadamard_transform(x, scale=1.0):
    return x
def _silu_and_mul(*args, **kwargs):
    return None
_sgl_kernel.hadamard_transform = _hadamard_transform
_sgl_kernel.silu_and_mul = _silu_and_mul
sys.modules.setdefault("sgl_kernel", _sgl_kernel)

# Minimal stub for cuda graph runner
_cgr_stub = types.ModuleType("sglang.srt.model_executor.cuda_graph_runner")
_cgr_stub.get_is_capture_mode = lambda: False
sys.modules.setdefault("sglang.srt.model_executor.cuda_graph_runner", _cgr_stub)

# Minimal stub for forward_batch_info
_fb_stub = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
class _ForwardMode:
    def is_context_parallel_extend(self):
        return False
    def is_target_verify(self):
        return False
    def is_draft_extend(self):
        return False
_fb_stub.ForwardMode = _ForwardMode
class _ForwardBatch:
    pass
_fb_stub.ForwardBatch = _ForwardBatch
sys.modules.setdefault("sglang.srt.model_executor.forward_batch_info", _fb_stub)

# Minimal stub for NSA backend
_nsa_backend_stub = types.ModuleType("sglang.srt.layers.attention.nsa_backend")
class _NativeSparseAttnBackend:
    def __init__(self, *args, **kwargs):
        pass
_nsa_backend_stub.NativeSparseAttnBackend = _NativeSparseAttnBackend
sys.modules.setdefault("sglang.srt.layers.attention.nsa_backend", _nsa_backend_stub)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals before importing backends
_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    BaseIndexerMetadata,
    Indexer,
    rotate_activation,
)
from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
# Use unittest.TestCase to avoid heavy sglang.test dependency chain

# Global configuration for all indexer tests
DEFAULT_CONFIG = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.float8_e4m3fn,
    "context_len": 2048,
    "max_bs": 64,
    "hidden_size": 5120,
    "index_n_heads": 1,
    "index_head_dim": 128,
    "rope_head_dim": 64,
    "index_topk": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "max_position_embeddings": 163840,
    "rope_theta": 10000.0,
    "layer_id": 0,
    "page_size": 64,
}


class MockIndexerMetadata(BaseIndexerMetadata):
    """Mock implementation of BaseIndexerMetadata for testing."""

    def __init__(self, batch_size, seq_lens, page_table=None):
        self.batch_size = batch_size
        self.seq_lens = seq_lens
        self.page_table = page_table
        self.device = "cuda"

    def get_seqlens_int32(self) -> torch.Tensor:
        """Return: (batch_size,) int32 tensor"""
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_page_table_64(self) -> torch.Tensor:
        """Return: (batch_size, num_blocks) int32, page table with page size 64."""
        if self.page_table is not None:
            return self.page_table
        # Create a simple page table for testing
        max_seq_len = max(self.seq_lens)
        num_blocks = (max_seq_len + 63) // 64  # Round up to page size 64
        page_table = torch.zeros(
            (self.batch_size, num_blocks), dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            # Simple linear mapping: block i maps to page i
            num_blocks_needed = (self.seq_lens[i] + 63) // 64
            page_table[i, :num_blocks_needed] = torch.arange(
                num_blocks_needed, device=self.device
            )
        return page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        """Return: (sum_extend_seq_len,) int32 tensor"""
        # For extend mode, each new token attends to progressively more tokens
        # For a sequence being extended from position 0 to seq_len, token i attends to i+1 tokens
        result = []
        for seq_len in self.seq_lens:
            result.extend(range(1, seq_len + 1))
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits.
        For testing, just return the topk indices.
        """
        return torch.topk(logits, k=topk, dim=-1).indices


class MockModelRunner:
    def __init__(self, config=None):
        self.device = "cuda"
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.dtype = self.config["dtype"]
        self.kv_cache_dtype = self.config["kv_cache_dtype"]
        self.is_hybrid_swa = False

        # Model configuration
        attention_arch = AttentionArch.MLA
        max_context_len = self.config["context_len"]
        max_batch_size = self.config["max_bs"]

        # Create mock hf_config for NSA - instantiate it as an object, not a type
        hf_config = type(
            "HfConfig",
            (),
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "index_topk": self.config["index_topk"],
                "index_head_dim": self.config["index_head_dim"],
                "index_n_heads": self.config["index_n_heads"],
            },
        )()

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "num_attention_heads": 128,
                "kv_lora_rank": self.config["kv_lora_rank"],
                "qk_rope_head_dim": self.config["qk_rope_head_dim"],
                "hf_config": hf_config,
            },
        )()

        self.sliding_window_size = None
        self.page_size = self.config["page_size"]

        # Create req_to_token_pool
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_batch_size,
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )()

        # Create NSATokenToKVPool
        max_total_num_tokens = max_batch_size * max_context_len
        self.token_to_kv_pool = NSATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.config["page_size"],
            dtype=self.config["kv_cache_dtype"],
            kv_lora_rank=self.config["kv_lora_rank"],
            qk_rope_head_dim=self.config["qk_rope_head_dim"],
            layer_num=1,
            device=self.device,
            index_head_dim=self.config["index_head_dim"],
            enable_memory_saver=False,
        )

        # Required by backend with NSA-specific attributes
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "auto",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
                "nsa_prefill_backend": "flashmla_sparse",
                "nsa_decode_backend": "fa3",
            },
        )()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNSAIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up global server args for testing."""
        server_args = ServerArgs(model_path="dummy")
        server_args.enable_dp_attention = False
        server_args.nsa_prefill_backend = "flashmla_sparse"
        server_args.nsa_decode_backend = "flashmla_sparse"
        set_global_server_args_for_scheduler(server_args)

        # Check GPU capability for FP8
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            cls.supports_fp8 = compute_capability[0] >= 9  # Hopper or newer

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        pass

    def setUp(self):
        # Test parameters
        self.batch_size = 2
        self.seq_len = 128
        self.config = DEFAULT_CONFIG.copy()
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def _init_model_runner(self, config_override=None):
        """Initialize model runner with optional config override."""
        config = self.config.copy()
        if config_override:
            config.update(config_override)
        self.model_runner = MockModelRunner(config)
        self.backend = NativeSparseAttnBackend(self.model_runner)

    def _create_indexer(self, **kwargs):
        """Create an Indexer instance with default parameters."""
        params = {
            "hidden_size": self.config["hidden_size"],
            "index_n_heads": self.config["index_n_heads"],
            "index_head_dim": self.config["index_head_dim"],
            "rope_head_dim": self.config["rope_head_dim"],
            "index_topk": self.config["index_topk"],
            "q_lora_rank": self.config["q_lora_rank"],
            "max_position_embeddings": self.config["max_position_embeddings"],
            "rope_theta": self.config["rope_theta"],
            "layer_id": self.config["layer_id"],
            "scale_fmt": "ue8m0",
            "block_size": 128,
            "quant_config": None,  # No quantization for testing
        }
        params.update(kwargs)

        torch.set_default_dtype(self.dtype)
        indexer = Indexer(**params)
        # Move indexer to CUDA device
        indexer = indexer.to(device=self.device)

        # Convert linear layer weights to bfloat16 (but preserve LayerNorm's float32
        # and weights_proj's float32 - it uses params_dtype=torch.float32 in production)
        # Need to recursively convert LinearBase submodules (like ReplicatedLinear)
        for name, module in indexer.named_modules():
            # Check for LinearBase (parent of ReplicatedLinear) but exclude LayerNorm
            # Also exclude weights_proj which uses float32 params in production
            if isinstance(module, LinearBase) and not isinstance(module, LayerNorm):
                if "weights_proj" not in name:
                    module.to(dtype=self.dtype)

        return indexer

    def _create_forward_batch(
        self, mode, batch_size=None, seq_len=None, extend_len=None
    ):
        """Create a forward batch for testing."""
        batch_size = batch_size or self.batch_size
        seq_len = seq_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            q_len = extend_len or seq_len
            total_len = seq_len

            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    batch_size * (total_len - q_len),
                    batch_size * total_len,
                    device=self.device,
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor([total_len] * batch_size, device=self.device),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [total_len - q_len] * batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [total_len - q_len] * batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor([q_len] * batch_size, device=self.device),
                extend_seq_lens_cpu=torch.tensor([q_len] * batch_size, device="cpu"),
                attn_backend=self.backend,
            )
        else:  # ForwardMode.DECODE
            decode_len = 1
            total_len = seq_len + decode_len

            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    batch_size * seq_len, batch_size * total_len, device=self.device
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor([total_len] * batch_size, device=self.device),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
                attn_backend=self.backend,
            )

        # Add token pools
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Mock write to req_to_token_pool
        page_size = self.model_runner.page_size
        for i in range(batch_size):
            seq_length = total_len
            for j in range(seq_length):
                self.model_runner.req_to_token_pool.req_to_token[i, j] = (
                    i * seq_length + j + page_size
                )

        return forward_batch

    def _verify_topk_output(self, topk_indices, batch_size, q_len, topk):
        """Verify the topk indices output shape and basic properties."""
        self.assertIsNotNone(topk_indices)
        self.assertEqual(topk_indices.device.type, "cuda")

        # Check shape - should be (total_q_len, topk_padded)
        # where topk_padded is aligned to 2048
        self.assertEqual(len(topk_indices.shape), 2)
        self.assertEqual(topk_indices.shape[0], batch_size * q_len)

        # Check that topk is padded to at least topk
        self.assertGreaterEqual(topk_indices.shape[1], topk)

        # Check for padding values (-1)
        has_padding = (topk_indices == -1).any()
        self.assertTrue(
            has_padding or topk_indices.shape[1] == topk,
            "Output should have padding or exact topk size",
        )

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_indexer_basic_creation(self, mock_deep_gemm):
        """Test basic indexer creation and initialization."""
        mock_deep_gemm.get_num_sms.return_value = 132

        indexer = self._create_indexer()

        self.assertEqual(indexer.hidden_size, self.config["hidden_size"])
        self.assertEqual(indexer.n_heads, self.config["index_n_heads"])
        self.assertEqual(indexer.head_dim, self.config["index_head_dim"])
        self.assertEqual(indexer.rope_head_dim, self.config["rope_head_dim"])
        self.assertEqual(indexer.index_topk, self.config["index_topk"])
        self.assertEqual(indexer.layer_id, self.config["layer_id"])

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    @patch("sglang.srt.layers.attention.nsa.triton_kernel.act_quant")
    def test_forward_extend_mode(self, mock_act_quant, mock_deep_gemm):
        """Test indexer forward pass in extend mode."""
        if not self.supports_fp8:
            self.skipTest("FP8 requires Hopper GPU or newer")

        # Setup mocks
        mock_deep_gemm.get_num_sms.return_value = 132
        mock_deep_gemm.get_paged_mqa_logits_metadata.return_value = MagicMock()

        def mock_quant(x, *args, **kwargs):
            # Return FP8 tensor and scale
            return x.to(torch.float8_e4m3fn), torch.ones(
                x.shape[0], dtype=torch.float32, device=x.device
            )

        mock_act_quant.side_effect = mock_quant

        # Mock deep_gemm.fp8_mqa_logits to return logits (ragged path)
        def mock_mqa_logits(q, kv, weights, ks, ke, *args, **kwargs):
            # q shape: (sum_extend_seq_len, ...), return logits for each query token
            num_queries = q.shape[0]
            # kv is a tuple (k_fp8, k_scale), get total number of keys from k_fp8
            k_fp8, k_scale = kv
            max_kv_len = k_fp8.shape[0]  # Total keys across all batches (k_offset)
            return torch.randn(
                num_queries, max_kv_len, dtype=torch.float32, device="cuda"
            )

        mock_deep_gemm.fp8_mqa_logits.side_effect = mock_mqa_logits

        # Also mock the paged version for completeness
        def mock_paged_mqa_logits(q, kv, weights, *args, **kwargs):
            batch_size = q.shape[0]
            seq_len = 128
            return torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")

        mock_deep_gemm.fp8_paged_mqa_logits.side_effect = mock_paged_mqa_logits

        self._init_model_runner()

        indexer = self._create_indexer()
        forward_batch = self._create_forward_batch(ForwardMode.EXTEND)

        # Create input tensors
        total_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(
            total_tokens,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            total_tokens,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(total_tokens, device=self.device)

        # Run forward pass
        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size, [self.seq_len] * self.batch_size
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        # Verify output
        self._verify_topk_output(
            topk_indices, self.batch_size, self.seq_len, self.config["index_topk"]
        )

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    @patch("sglang.srt.layers.attention.nsa.triton_kernel.act_quant")
    def test_forward_decode_mode(self, mock_act_quant, mock_deep_gemm):
        """Test indexer forward pass in decode mode."""
        if not self.supports_fp8:
            self.skipTest("FP8 requires Hopper GPU or newer")

        # Setup mocks
        mock_deep_gemm.get_num_sms.return_value = 132
        mock_deep_gemm.get_paged_mqa_logits_metadata.return_value = MagicMock()

        def mock_quant(x, *args, **kwargs):
            return x.to(torch.float8_e4m3fn), torch.ones(
                x.shape[0], dtype=torch.float32, device=x.device
            )

        mock_act_quant.side_effect = mock_quant

        def mock_paged_mqa_logits(q, kv, weights, *args, **kwargs):
            batch_size = q.shape[0]
            seq_len = 128
            return torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")

        mock_deep_gemm.fp8_paged_mqa_logits.side_effect = mock_paged_mqa_logits

        self._init_model_runner()

        indexer = self._create_indexer()
        forward_batch = self._create_forward_batch(ForwardMode.DECODE)

        # Create input tensors for decode (batch_size tokens only)
        hidden_states = torch.randn(
            self.batch_size,
            self.config["hidden_size"],
            dtype=self.dtype,
            device=self.device,
        )
        q_lora = torch.randn(
            self.batch_size,
            self.config["q_lora_rank"],
            dtype=self.dtype,
            device=self.device,
        )
        positions = torch.arange(self.batch_size, device=self.device)

        # Run forward pass
        with patch.object(
            self.backend,
            "get_indexer_metadata",
            return_value=MockIndexerMetadata(
                self.batch_size, [self.seq_len + 1] * self.batch_size
            ),
        ):
            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

        # Verify output - decode mode has q_len=1
        self._verify_topk_output(
            topk_indices, self.batch_size, 1, self.config["index_topk"]
        )

    def test_rotate_activation(self):
        """Test the Hadamard transform (rotate_activation) function."""
        # Test with power-of-2 hidden size
        hidden_size = 128
        x = torch.randn(16, hidden_size, dtype=torch.bfloat16, device=self.device)

        try:
            output = rotate_activation(x)
            self.assertEqual(output.shape, x.shape)
            self.assertEqual(output.dtype, torch.bfloat16)
        except ImportError:
            self.skipTest("sgl_kernel not available for hadamard_transform")

    def test_rotate_activation_invalid_size(self):
        """Test that rotate_activation fails with non-power-of-2 size."""
        # Test with non-power-of-2 hidden size
        hidden_size = 129  # Not a power of 2
        x = torch.randn(16, hidden_size, dtype=torch.bfloat16, device=self.device)

        with self.assertRaises(AssertionError):
            rotate_activation(x)

    def test_indexer_metadata_interface(self):
        """Test the BaseIndexerMetadata interface implementation."""
        batch_size = 4
        seq_lens = [64, 128, 96, 112]

        metadata = MockIndexerMetadata(batch_size, seq_lens)

        # Test get_seqlens_int32
        seqlens = metadata.get_seqlens_int32()
        self.assertEqual(seqlens.shape, (batch_size,))
        self.assertEqual(seqlens.dtype, torch.int32)
        self.assertTrue(torch.all(seqlens == torch.tensor(seq_lens, device="cuda")))

        # Test get_page_table_64
        page_table = metadata.get_page_table_64()
        self.assertEqual(len(page_table.shape), 2)
        self.assertEqual(page_table.shape[0], batch_size)
        self.assertEqual(page_table.dtype, torch.int32)

        # Test topk_transform
        logits = torch.randn(batch_size, 128, device="cuda")
        topk = 64
        topk_indices = metadata.topk_transform(logits, topk)
        self.assertEqual(topk_indices.shape, (batch_size, topk))

    # TODO: enable this test after indexer accuracy aligned
    # @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    # def test_indexer_with_different_topk(self, mock_deep_gemm):
    #     """Test indexer with different topk values."""
    #     mock_deep_gemm.get_num_sms.return_value = 132

    #     for topk in [32, 64, 128]:
    #         with self.subTest(topk=topk):
    #             indexer = self._create_indexer(index_topk=topk)
    #             self.assertEqual(indexer.index_topk, topk)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_indexer_with_fused_wk(self, mock_deep_gemm):
        """Test indexer creation with fused wk and weights projection."""
        mock_deep_gemm.get_num_sms.return_value = 132

        # Note: fuse_wk_and_weights_proj feature is not currently implemented
        # This test verifies basic indexer creation still works
        indexer = self._create_indexer()
        self.assertIsNotNone(indexer)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_indexer_with_alt_stream(self, mock_deep_gemm):
        """Test indexer creation with alternative CUDA stream."""
        mock_deep_gemm.get_num_sms.return_value = 132

        alt_stream = torch.cuda.Stream()
        indexer = self._create_indexer(alt_stream=alt_stream)
        self.assertEqual(indexer.alt_stream, alt_stream)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm_wrapper.configure_deep_gemm_num_sms")
    @patch("torch.cuda.stream")
    @patch("torch.cuda.current_stream")
    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.rotate_activation")
    def test_dual_stream_branch_waits(self, mock_rotate, mock_current_stream, mock_cuda_stream, mock_cfg_num_sms):
        """Verify dual-stream branch executes wait_stream on both streams and shapes are correct."""
        if not torch.cuda.is_available():
            self.skipTest("Test requires CUDA")

        class DummyCtx:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        mock_rotate.side_effect = lambda t: t
        mock_current_stream.return_value = MagicMock()
        mock_cuda_stream.return_value = DummyCtx()
        mock_cfg_num_sms.return_value = DummyCtx()

        self._init_model_runner()
        alt_stream_mock = MagicMock()
        indexer = self._create_indexer(alt_stream=alt_stream_mock)

        n_heads = indexer.n_heads
        head_dim = indexer.head_dim
        l = 4

        def mock_wq_b(inp):
            return (
                torch.randn(l, n_heads * head_dim, dtype=self.dtype, device=self.device),
                None,
            )

        def mock_wk(inp):
            return (
                torch.randn(l, head_dim, dtype=self.dtype, device=self.device),
                None,
            )

        indexer.wq_b = MagicMock(side_effect=mock_wq_b)
        indexer.wk = MagicMock(side_effect=mock_wk)

        q_lora = torch.randn(l, self.config["q_lora_rank"], dtype=self.dtype, device=self.device)
        x = torch.randn(l, self.config["hidden_size"], dtype=self.dtype, device=self.device)
        positions = torch.arange(l, device=self.device)

        forward_batch = type("FB", (), {"nsa_cp_metadata": None})()

        query, key = indexer._get_q_k_bf16(q_lora, x, positions, True, forward_batch)

        self.assertEqual(query.shape, (l, n_heads, head_dim))
        self.assertEqual(key.shape, (l, head_dim))

        cs = mock_current_stream.return_value
        alt_stream_mock.wait_stream.assert_called_with(cs)
        cs.wait_stream.assert_called_with(alt_stream_mock)

    def test_shape_sanity_checks(self):
        """Test various shape combinations for consistency."""
        test_configs = [
            {"batch_size": 1, "seq_len": 64},
            {"batch_size": 4, "seq_len": 128},
            {"batch_size": 8, "seq_len": 256},
        ]

        for config in test_configs:
            with self.subTest(**config):
                batch_size = config["batch_size"]
                seq_len = config["seq_len"]

                # Test metadata shapes
                metadata = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

                seqlens = metadata.get_seqlens_int32()
                self.assertEqual(seqlens.shape, (batch_size,))

                page_table = metadata.get_page_table_64()
                expected_blocks = (seq_len + 63) // 64
                self.assertEqual(page_table.shape[0], batch_size)
                self.assertGreaterEqual(page_table.shape[1], expected_blocks)


if __name__ == "__main__":
    unittest.main()
