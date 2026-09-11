"""Exercise production Python adapters with CPU operators, never CUDA kernels."""

import ast
import contextlib
import dataclasses
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional

import pytest
import test_marlin_chunked_workspace as marlin_cpu
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe.workspace_policy import (
    MarlinWorkspaceEstimate,
    TritonWorkspaceEstimate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]
RUNNER = ROOT / "python/sglang/srt/layers/moe/moe_runner"


def load_nodes(path, names, namespace=None):
    tree = ast.parse(path.read_text(), str(path))
    body = [node for node in tree.body if getattr(node, "name", "") in names]
    assert {node.name for node in body} == set(names)
    for node in body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    ns = dict(
        __name__=__name__,
        torch=torch,
        Optional=Optional,
        Any=Any,
        Dict=Dict,
        List=List,
        dataclass=dataclasses.dataclass,
        MoeQuantInfo=object,
    )
    ns.update(namespace or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def install_module(monkeypatch, name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class CombineInput(NamedTuple):
    hidden_states: torch.Tensor


def config(**overrides):
    values = dict(
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        no_combine=False,
        inplace=False,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        use_tp_all_gather_activation=False,
        num_experts=4,
        num_local_experts=4,
        routed_scaling_factor=None,
        workspace_budget_bytes=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def dispatch(x, ids, weights):
    return SimpleNamespace(
        hidden_states=x,
        hidden_states_scale=None,
        hidden_states_pre_quant=None,
        format=SimpleNamespace(value="standard"),
        topk_output=SimpleNamespace(
            topk_ids=ids, topk_weights=weights, router_logits=torch.zeros(x.shape[0], 4)
        ),
    )


@pytest.fixture
def adapters(monkeypatch):
    name = "_test_workspace_adapters_impl"
    spec = importlib.util.spec_from_file_location(name, RUNNER / "workspace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SGLANG_MARLIN_MOE_CHUNK_SIZE", "0")
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_LORA_OPTI", "0")
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.token_dispatcher.standard",
        StandardCombineInput=CombineInput,
    )
    return module


@pytest.fixture
def triton_runtime(monkeypatch):
    generator = torch.Generator().manual_seed(42)
    calls, aligns = [], []
    launch = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}
    down = dict(launch)

    def align(ids, block, experts):
        helper = marlin_cpu.CpuMarlin(torch.bfloat16)
        aligns.append((ids.clone(), block))
        return helper.align(ids, block, experts)

    def invoke(
        a,
        b,
        bias,
        out,
        a_scale,
        b_scale,
        zp,
        weights,
        ids,
        sorted_ids,
        experts,
        num_padded,
        mul_weight,
        top_k,
        launch_config,
        **kw,
    ):
        assert a_scale is None and b_scale is None and zp is None
        assert not kw.get("a_use_tma") and not kw.get("b_use_tma")
        assert not kw.get("fuse_sum_all_reduce")
        flat = out.view(-1, out.shape[-1])
        calls.append((mul_weight, a.shape[0], flat.untyped_storage().data_ptr()))
        for index in range(int(num_padded[0])):
            route = int(sorted_ids[index])
            if route >= flat.shape[0]:
                continue
            expert = int(experts[index // launch_config["BLOCK_SIZE_M"]])
            if expert < 0:
                flat[route].zero_()
                continue
            value = a[route // top_k].float() @ b[expert].float().T
            if bias is not None:
                value += bias[expert].float()
            if mul_weight:
                value *= weights.reshape(-1)[route]
            flat[route].copy_(value)

    def activate(x, out, activation="silu", **kwargs):
        gate, up = x.float().chunk(2, dim=-1)
        value = F.silu(gate) if activation == "silu" else F.gelu(gate)
        out.copy_(value * up)

    ns = load_nodes(
        RUNNER / "triton_utils/fused_moe.py",
        [
            "_resolve_fused_moe_config",
            "_prepare_fused_moe_run",
            "_fused_moe_kernel_sequence",
        ],
        dict(
            padding_size=0,
            _use_aiter=False,
            _is_cuda=True,
            _is_hip=False,
            _is_xpu=False,
            _is_musa=False,
            _has_vllm_ops=False,
            tl=SimpleNamespace(bfloat16=torch.bfloat16, float16=torch.float16),
            get_exec=lambda: SimpleNamespace(
                moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
            ),
            get_config_dtype_str=lambda **kwargs: "bf16",
            try_get_optimal_moe_config=lambda *args, **kwargs: (launch, (down, None)),
            _moe_support_tma=lambda: True,
            logger=SimpleNamespace(warning_once=lambda *args: None),
            moe_align_block_size=align,
            invoke_fused_moe_kernel=invoke,
            silu_and_mul=activate,
            gelu_and_mul=lambda x, out, **kwargs: activate(x, out, "gelu", **kwargs),
            moe_sum_reduce=lambda x, out, scale: out.copy_(x.float().sum(1) * scale),
            _use_moe_sum_reduce_torch_compile=lambda m: False,
            get_tp_group=lambda: None,
            is_allocation_symmetric=lambda: False,
            use_symmetric_memory=lambda *args, **kwargs: contextlib.nullcontext(),
        ),
    )
    quant_cls = load_nodes(RUNNER / "triton.py", ["TritonMoeQuantInfo"])[
        "TritonMoeQuantInfo"
    ]
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.triton",
        TritonMoeQuantInfo=quant_cls,
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
        **{
            key: ns[key]
            for key in ("_resolve_fused_moe_config", "_fused_moe_kernel_sequence")
        },
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size",
        moe_align_block_size=align,
    )
    return SimpleNamespace(
        ns=ns,
        quant_cls=quant_cls,
        calls=calls,
        aligns=aligns,
        launch=launch,
        down=down,
        generator=generator,
    )


def sequence_inputs(rt, topk=2, scale=1.0, tokens=3):
    x = torch.randn(tokens, 64, generator=rt.generator).to(torch.bfloat16)
    w1 = torch.randn(4, 64, 64, generator=rt.generator).to(x.dtype) * 0.1
    w2 = torch.randn(4, 64, 32, generator=rt.generator).to(x.dtype) * 0.1
    ids = (torch.arange(tokens * topk).view(tokens, topk) % 4).to(torch.int32)
    weights = torch.rand(tokens, topk, generator=rt.generator)
    sorted_ids, expert_ids, padded = rt.ns["moe_align_block_size"](ids, 8, 4)
    return dict(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=padded,
        config=rt.launch,
        down_config=rt.down,
        down_moe_use_tma=False,
        up_moe_use_tma=False,
        b1=None,
        b2=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        activation="silu",
        is_gated=True,
        no_combine=False,
        inplace=False,
        apply_router_weight_on_input=False,
        routed_scaling_factor=scale,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=False,
    )


@pytest.mark.parametrize(
    "topk,scale", [(1, 0.0), (1, 0.5), (1, 1.0), (1, 1.7), (2, 1.0), (3, 0.5)]
)
def test_triton_default_and_scratch_paths_agree(triton_runtime, topk, scale):
    args = sequence_inputs(triton_runtime, topk, scale)
    run = triton_runtime.ns["_fused_moe_kernel_sequence"]
    expected = run(**args)
    assert torch.isfinite(expected).all()
    if topk == 1:
        unscaled = run(**dict(args, routed_scaling_factor=1.0))
        torch.testing.assert_close(
            expected, (unscaled.float() * scale).to(expected.dtype), atol=0, rtol=0
        )
    x = args["hidden_states"]
    scratch = (
        x.new_empty((5 * topk, 64)),
        x.new_empty((5 * topk, 32)),
        x.new_empty((5, topk, 64)),
    )
    output = torch.empty_like(x)
    actual = run(**args, scratch=scratch, output_buffer=output)
    assert actual is output
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "case",
    [
        "short",
        "shape",
        "dtype",
        "stride",
        "device",
        "tuple",
        "output_shape",
        "output_dtype",
        "output_stride",
        "output_device",
        "no_combine",
        "hooks",
        "up_tma",
        "down_tma",
        "quantized",
        "nongated",
        "activation",
        "interleaved",
        "modifier",
        "input_weight",
    ],
)
def test_triton_scratch_validation_precedes_gemm(triton_runtime, case):
    args = sequence_inputs(triton_runtime)
    x = args["hidden_states"]
    scratch = [x.new_empty((6, 64)), x.new_empty((6, 32)), x.new_empty((3, 2, 64))]
    output = torch.empty_like(x)
    if case == "short":
        scratch[0] = x.new_empty((5, 64))
    elif case == "shape":
        scratch[1] = x.new_empty((6, 33))
    elif case == "dtype":
        scratch[2] = scratch[2].float()
    elif case == "stride":
        scratch[0] = x.new_empty((64, 6)).T
    elif case == "device":
        scratch[0] = torch.empty(6, 64, device="meta", dtype=x.dtype)
    elif case == "tuple":
        scratch.pop()
    elif case == "output_shape":
        output = x.new_empty((4, 64))
    elif case == "output_dtype":
        output = x.float()
    elif case == "output_stride":
        output = x.new_empty((64, 3)).T
    elif case == "output_device":
        output = torch.empty_like(x, device="meta")
    elif case == "hooks":
        args["hooks"] = SimpleNamespace(after_gate_up=None, after_down=None)
    else:
        name, value = {
            "no_combine": ("no_combine", True),
            "up_tma": ("up_moe_use_tma", True),
            "down_tma": ("down_moe_use_tma", True),
            "quantized": ("use_int8_w8a8", True),
            "nongated": ("is_gated", False),
            "activation": ("activation", "situ"),
            "interleaved": ("fuse_swiglu_interleaved", True),
            "modifier": ("gemm1_alpha", 1.0),
            "input_weight": ("apply_router_weight_on_input", True),
        }[case]
        args[name] = value
    with pytest.raises(ValueError, match="Triton"):
        triton_runtime.ns["_fused_moe_kernel_sequence"](
            **args, scratch=tuple(scratch), output_buffer=output
        )
    assert not triton_runtime.calls


@pytest.mark.parametrize("tma", [False, True])
def test_config_resolution_preserves_prepare_and_cached_configs(triton_runtime, tma):
    rt = triton_runtime
    args = sequence_inputs(rt)
    rt.aligns.clear()
    rt.launch["USE_TMA"] = tma
    rt.down["USE_TMA"] = tma
    keys = (
        "hidden_states",
        "w1",
        "w2",
        "topk_ids",
        "use_fp8_w8a8",
        "use_int8_w8a8",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "per_channel_quant",
        "block_shape",
    )
    selected = {key: args[key] for key in keys}
    resolved = rt.ns["_resolve_fused_moe_config"](**selected)
    assert not rt.aligns
    prepared = rt.ns["_prepare_fused_moe_run"](**selected)
    assert len(rt.aligns) == 1
    assert prepared[:4] == resolved
    assert resolved[2:] == (tma, tma)
    assert "USE_TMA" not in resolved[0] and "USE_TMA" not in resolved[1]
    assert rt.launch["USE_TMA"] is tma and rt.down["USE_TMA"] is tma
    assert resolved[0] is not rt.launch and resolved[1] is not rt.down


@pytest.mark.parametrize("tokens", [None, 1, 2, 3, 0, -1, 4])
def test_config_candidate_token_override(triton_runtime, monkeypatch, tokens):
    rt = triton_runtime
    args = sequence_inputs(rt)
    rt.aligns.clear()
    seen = []

    def select(w1, w2, topk, dtype, num_tokens, **kwargs):
        seen.append(num_tokens)
        return rt.launch, (rt.down, None)

    monkeypatch.setitem(rt.ns, "try_get_optimal_moe_config", select)
    keys = (
        "hidden_states",
        "w1",
        "w2",
        "topk_ids",
        "use_fp8_w8a8",
        "use_int8_w8a8",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "per_channel_quant",
        "block_shape",
    )
    selected = {key: args[key] for key in keys}
    if tokens is not None and not 0 < tokens <= 3:
        with pytest.raises(ValueError, match="Candidate token count"):
            rt.ns["_resolve_fused_moe_config"](**selected, num_tokens=tokens)
        assert seen == []
    else:
        result = rt.ns["_resolve_fused_moe_config"](**selected, num_tokens=tokens)
        assert seen == [3 if tokens is None else tokens]
        assert result[:2] == (rt.launch, rt.down)
        assert result[2:] == (False, False)
    assert not rt.aligns and not rt.calls
    assert args["hidden_states"].shape == (3, 64)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("topk,scale", [(2, 1.0), (3, 0.5)])
def test_triton_uses_candidate_config_for_every_chunk(
    adapters, triton_runtime, monkeypatch, inplace, topk, scale
):
    rt = triton_runtime
    args = sequence_inputs(rt, topk, scale, tokens=7)
    expected = rt.ns["_fused_moe_kernel_sequence"](**args)
    rt.calls.clear()
    rt.aligns.clear()
    selected, launched = [], []
    up_configs = {
        tokens: dict(rt.launch, BLOCK_SIZE_M=block)
        for tokens, block in ((7, 64), (4, 32), (2, 8), (1, 128))
    }
    down_configs = {
        tokens: dict(launch, BLOCK_SIZE_N=64) for tokens, launch in up_configs.items()
    }

    def select(w1, w2, topk, dtype, num_tokens, **kwargs):
        assert not rt.aligns and not rt.calls
        selected.append(num_tokens)
        return up_configs[num_tokens], (down_configs[num_tokens], None)

    invoke = rt.ns["invoke_fused_moe_kernel"]

    def record(*args, **kwargs):
        launched.append(dict(args[14]))
        return invoke(*args, **kwargs)

    monkeypatch.setitem(rt.ns, "try_get_optimal_moe_config", select)
    monkeypatch.setitem(rt.ns, "invoke_fused_moe_kernel", record)
    quant = rt.quant_cls(args["w1"], args["w2"])
    inp = dispatch(args["hidden_states"], args["topk_ids"], args["topk_weights"])
    budget = TritonWorkspaceEstimate(64, 32, topk, 4, 8).peak_bytes(2)
    result, why = adapters.TritonWorkspaceAdapter.run(
        inp, quant, config(inplace=inplace, routed_scaling_factor=scale), budget
    )
    assert why is None
    assert selected == [7, 4, 2]
    assert [(ids.shape[0], block) for ids, block in rt.aligns] == [
        (2, 8),
        (2, 8),
        (2, 8),
        (1, 8),
    ]
    assert launched == [up_configs[2], down_configs[2]] * 4
    assert len({call[2] for call in rt.calls[::2]}) == 1
    assert len({call[2] for call in rt.calls[1::2]}) == 1
    assert (result.hidden_states is inp.hidden_states) is inplace
    torch.testing.assert_close(result.hidden_states, expected, atol=0, rtol=0)


@pytest.mark.parametrize("tma_tokens", [7, 4])
@pytest.mark.parametrize("side", ["up", "down"])
def test_triton_candidate_tma_falls_back_before_allocation(
    adapters, triton_runtime, monkeypatch, tma_tokens, side
):
    rt = triton_runtime
    args = sequence_inputs(rt, tokens=7)
    inp = dispatch(args["hidden_states"], args["topk_ids"], args["topk_weights"])
    quant = rt.quant_cls(args["w1"], args["w2"])
    rt.aligns.clear()
    seen = []
    configs = {}

    def select(w1, w2, topk, dtype, num_tokens, **kwargs):
        seen.append(num_tokens)
        up, down = dict(rt.launch), dict(rt.down)
        if num_tokens == tma_tokens:
            (up if side == "up" else down)["USE_TMA"] = True
        configs[num_tokens] = up, down
        return up, (down, None)

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported candidate must fall back before scratch allocation")

    monkeypatch.setitem(rt.ns, "try_get_optimal_moe_config", select)
    monkeypatch.setattr(torch.Tensor, "new_empty", unexpected)
    monkeypatch.setattr(torch, "empty_like", unexpected)
    result, why = adapters.TritonWorkspaceAdapter.run(inp, quant, config(), 1)
    assert result is None and why == f"TMA layout for {tma_tokens}-token candidate"
    assert seen == ([7] if tma_tokens == 7 else [7, 4])
    assert configs[tma_tokens][0 if side == "up" else 1]["USE_TMA"] is True
    assert not rt.aligns and not rt.calls


@pytest.mark.parametrize("failure", ["budget", "config"])
def test_triton_candidate_errors_propagate(
    adapters, triton_runtime, monkeypatch, failure
):
    rt = triton_runtime
    args = sequence_inputs(rt, tokens=7)
    inp = dispatch(args["hidden_states"], args["topk_ids"], args["topk_weights"])
    quant = rt.quant_cls(args["w1"], args["w2"])
    rt.aligns.clear()
    seen = []

    def select(w1, w2, topk, dtype, num_tokens, **kwargs):
        seen.append(num_tokens)
        if failure == "config" and num_tokens == 4:
            raise ValueError("invalid launch configuration")
        return rt.launch, (rt.down, None)

    monkeypatch.setitem(rt.ns, "try_get_optimal_moe_config", select)
    for _ in range(2):
        seen.clear()
        with pytest.raises(
            ValueError, match="cannot fit" if failure == "budget" else "invalid launch"
        ):
            adapters.TritonWorkspaceAdapter.run(inp, quant, config(), 1)
        # The one-token estimate appears twice in the planner but resolves once.
        assert seen == ([7, 4, 2, 1] if failure == "budget" else [7, 4])
    assert not rt.aligns and not rt.calls


def test_triton_candidate_costs_can_be_nonmonotone(
    adapters, triton_runtime, monkeypatch
):
    rt = triton_runtime
    args = sequence_inputs(rt, tokens=13)
    expected = rt.ns["_fused_moe_kernel_sequence"](**args)
    rt.calls.clear()
    rt.aligns.clear()
    selected = []

    def select(w1, w2, topk, dtype, num_tokens, **kwargs):
        selected.append(num_tokens)
        launch = dict(rt.launch, BLOCK_SIZE_M=8 if num_tokens == 4 else 1024)
        return launch, (dict(launch), None)

    monkeypatch.setitem(rt.ns, "try_get_optimal_moe_config", select)
    budget = TritonWorkspaceEstimate(64, 32, 2, 4, 8).peak_bytes(4)
    assert TritonWorkspaceEstimate(64, 32, 2, 4, 1024).peak_bytes(1) > budget
    result, why = adapters.TritonWorkspaceAdapter.run(
        dispatch(args["hidden_states"], args["topk_ids"], args["topk_weights"]),
        rt.quant_cls(args["w1"], args["w2"]),
        config(),
        budget,
    )
    assert why is None and selected == [13, 8, 4]
    assert [(ids.shape[0], block) for ids, block in rt.aligns] == [
        (4, 8),
        (4, 8),
        (4, 8),
        (1, 8),
    ]
    torch.testing.assert_close(result.hidden_states, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize(
    "topk,scale", [(1, None), (1, 0.5), (2, 1.0), (3, 0.0), (3, 1.7)]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_triton_execution(
    adapters, triton_runtime, dtype, activation, topk, scale, inplace
):
    rt = triton_runtime
    m, h, i, e = 7, 64, 32, 4
    x = torch.randn(m, h, generator=rt.generator).to(dtype)
    original = x.clone()
    w1 = torch.randn(e, 2 * i, h, generator=rt.generator).to(dtype) * 0.1
    w2 = torch.randn(e, h, i, generator=rt.generator).to(dtype) * 0.1
    ids = (torch.arange(m * topk).view(m, topk) % e).to(torch.int32)
    ids[-1] = -1
    weights = torch.rand(m, topk, generator=rt.generator)
    quant = rt.quant_cls(w1, w2, b13=torch.ones(e, 2 * i, dtype=dtype) * 0.1)
    cfg = config(
        activation=activation,
        inplace=inplace,
        routed_scaling_factor=scale,
        num_experts=8,
    )
    inp = dispatch(x, ids, weights)
    expected = []
    for token in range(m):
        values = []
        for slot, expert in enumerate(ids[token].tolist()):
            if expert < 0:
                values.append(torch.zeros(h))
                continue
            gu = (
                original[token].float() @ w1[expert].float().T
                + quant.b13[expert].float()
            ).to(dtype)
            gate, up = gu.float().chunk(2)
            a = ((F.silu(gate) if activation == "silu" else F.gelu(gate)) * up).to(
                dtype
            )
            values.append(
                ((a.float() @ w2[expert].float().T) * weights[token, slot])
                .to(dtype)
                .float()
            )
        expected.append(torch.stack(values).sum(0) * (1.0 if scale is None else scale))
    estimate = TritonWorkspaceEstimate(h, i, topk, e, 8)
    result, reason = adapters.TritonWorkspaceAdapter.run(
        inp, quant, cfg, estimate.peak_bytes(2)
    )
    assert reason is None
    torch.testing.assert_close(
        result.hidden_states, torch.stack(expected).to(dtype), atol=0, rtol=0
    )
    assert [x[0].shape[0] for x in rt.aligns] == [2, 2, 2, 1]
    assert len(rt.calls) == 8
    assert len({call[2] for call in rt.calls[::2]}) == 1
    assert len({call[2] for call in rt.calls[1::2]}) == 1
    assert (result.hidden_states.data_ptr() == x.data_ptr()) is inplace
    if not inplace:
        torch.testing.assert_close(x, original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("quant", "quantized"),
        ("tma", "TMA"),
        ("act", "SiLU/GELU"),
        ("shape", "geometry"),
    ],
)
def test_triton_rejects_before_alignment(adapters, triton_runtime, case, reason):
    rt = triton_runtime
    x = torch.zeros(7, 64, dtype=torch.bfloat16)
    quant = rt.quant_cls(
        torch.zeros(4, 64, 64, dtype=x.dtype), torch.zeros(4, 64, 32, dtype=x.dtype)
    )
    cfg = config()
    if case == "quant":
        quant.use_fp8_w8a8 = True
    elif case == "tma":
        rt.down["USE_TMA"] = True
    elif case == "act":
        cfg.activation = "situ"
    else:
        quant.w2_weight = torch.empty(4, 65, 32, dtype=x.dtype)
    inp = dispatch(x, torch.zeros(7, 2, dtype=torch.int32), torch.ones(7, 2))
    result, why = adapters.TritonWorkspaceAdapter.run(inp, quant, cfg, 10**9)
    assert result is None and reason in why
    assert not rt.aligns and not rt.calls
    if case == "tma":
        assert rt.down["USE_TMA"] is True


def test_triton_empty_and_budget_failure(adapters, triton_runtime):
    rt = triton_runtime
    x = torch.zeros(0, 64, dtype=torch.bfloat16)
    quant = rt.quant_cls(
        torch.zeros(4, 64, 64, dtype=x.dtype), torch.zeros(4, 64, 32, dtype=x.dtype)
    )
    inp = dispatch(x, torch.zeros(0, 2, dtype=torch.int32), torch.ones(0, 2))
    result, why = adapters.TritonWorkspaceAdapter.run(inp, quant, config(), 1)
    assert why is None and result.hidden_states.shape == x.shape
    quant.use_fp8_w8a8 = True
    result, why = adapters.TritonWorkspaceAdapter.run(inp, quant, config(), 1)
    assert result is None and "quantized" in why
    quant.use_fp8_w8a8 = False
    inp = dispatch(
        torch.zeros(7, 64, dtype=x.dtype),
        torch.zeros(7, 2, dtype=torch.int32),
        torch.ones(7, 2),
    )
    with pytest.raises(ValueError, match="cannot fit"):
        adapters.TritonWorkspaceAdapter.run(inp, quant, config(), 1)
    assert not rt.aligns and not rt.calls


def test_triton_calls_own_separate_scratch(adapters, triton_runtime, monkeypatch):
    rt = triton_runtime
    args = sequence_inputs(rt)
    quant = rt.quant_cls(args["w1"], args["w2"])
    inp = dispatch(args["hidden_states"], args["topk_ids"], args["topk_weights"])
    retained = []
    original = rt.ns["_fused_moe_kernel_sequence"]

    def run(*args, **kwargs):
        retained.append(kwargs["scratch"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"],
        "_fused_moe_kernel_sequence",
        run,
    )
    budget = TritonWorkspaceEstimate(64, 32, 2, 4, 8).peak_bytes(2)
    first, why = adapters.TritonWorkspaceAdapter.run(inp, quant, config(), budget)
    assert why is None
    second, why = adapters.TritonWorkspaceAdapter.run(inp, quant, config(), budget)
    assert why is None and len(retained) == 4
    assert retained[0] is retained[1] and retained[2] is retained[3]
    for left, right in zip(retained[0], retained[2]):
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
    torch.testing.assert_close(
        first.hidden_states, second.hidden_states, atol=0, rtol=0
    )


@pytest.mark.parametrize("mxfp4", [False, True])
def test_marlin_adapter_executes_real_orchestration(adapters, monkeypatch, mxfp4):
    engine = marlin_cpu.CpuMarlin(torch.bfloat16, mxfp4=mxfp4)
    args = engine.inputs(7)
    expected = engine.reference(args)
    low = marlin_cpu.runtime.__wrapped__()
    ns = load_nodes(
        RUNNER / "marlin.py",
        ["MarlinMoeQuantInfo", "fused_experts_none_to_marlin"],
        dict(StandardDispatchOutput=Any, MoeRunnerConfig=Any, StandardCombineInput=Any),
    )
    quant = ns["MarlinMoeQuantInfo"](
        args["w1"],
        args["w2"],
        args["w1_scale"],
        args["w2_scale"],
        None,
        None,
        args["num_bits"],
    )
    inp = dispatch(args["hidden_states"], args["topk_ids"], args["topk_weights"])
    estimate = MarlinWorkspaceEstimate(64, 32, 2, 4, 78, mxfp4)
    with marlin_cpu._bind(monkeypatch, low, engine, 0):
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device: SimpleNamespace(major=9, multi_processor_count=78),
        )
        install_module(
            monkeypatch,
            "sglang.srt.layers.moe.moe_runner.marlin",
            MarlinMoeQuantInfo=ns["MarlinMoeQuantInfo"],
            fused_experts_none_to_marlin=ns["fused_experts_none_to_marlin"],
        )
        install_module(
            monkeypatch,
            "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe",
            fused_marlin_moe=low["fused_marlin_moe"],
        )
        install_module(
            monkeypatch,
            "sglang.srt.layers.quantization.marlin_utils",
            marlin_make_workspace=lambda device, max_blocks_per_sm: torch.zeros(
                312, dtype=torch.int32
            ),
        )
        result, reason = adapters.MarlinWorkspaceAdapter.run(
            inp, quant, config(), estimate.peak_bytes(2)
        )
        assert reason is None
        assert len(engine.calls) == 8
        torch.testing.assert_close(result.hidden_states, expected, atol=0, rtol=0)
        quant.w13_g_idx = torch.arange(64)
        result, reason = adapters.MarlinWorkspaceAdapter.run(
            inp, quant, config(), 10**9
        )
        assert result is None and "activation-order" in reason
        quant.w13_g_idx = None
        quant.expert_map = torch.arange(4)
        result, reason = adapters.MarlinWorkspaceAdapter.run(
            inp, quant, config(), 10**9
        )
        assert result is None and "expert-parallel" in reason


@pytest.mark.parametrize("mxfp4", [False, True])
@pytest.mark.parametrize("destination", [False, True])
def test_marlin_empty_batch_does_not_allocate_locks(
    adapters, monkeypatch, mxfp4, destination
):
    args = marlin_cpu.CpuMarlin(torch.bfloat16, mxfp4=mxfp4).inputs(0)
    cls = load_nodes(RUNNER / "marlin.py", ["MarlinMoeQuantInfo"])["MarlinMoeQuantInfo"]
    quant = cls(
        args["w1"],
        args["w2"],
        args["w1_scale"],
        args["w2_scale"],
        None,
        None,
        args["num_bits"],
    )

    def unexpected(*args, **kwargs):
        pytest.fail("empty batch must not query CUDA or enter the allocating runner")

    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.marlin",
        MarlinMoeQuantInfo=cls,
        fused_experts_none_to_marlin=unexpected,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    x = args["hidden_states"]
    output = torch.empty_like(x)
    inp = dispatch(x, args["topk_ids"], args["topk_weights"])
    context = (
        zero_copy_context.set_moe_output(output)
        if destination
        else contextlib.nullcontext()
    )
    with context:
        result, why = adapters.MarlinWorkspaceAdapter.run(
            inp, quant, config(inplace=True), 1
        )
        assert why is None and result.hidden_states.shape == x.shape
        assert result.hidden_states is not x
        assert (result.hidden_states is output) is destination
        result, why = adapters.MarlinWorkspaceAdapter.run(
            inp, quant, config(activation="situ"), 1
        )
        assert result is None and "SiLU" in why
        quant.expert_map = torch.arange(4)
        result, why = adapters.MarlinWorkspaceAdapter.run(inp, quant, config(), 1)
        assert result is None and "expert-parallel" in why


@pytest.fixture
def guarded_dispatch(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(7, 64, device="cuda", dtype=torch.bfloat16)
        ids = torch.empty(7, 2, device="cuda", dtype=torch.int32)
        weights = torch.empty(7, 2, device="cuda")
    inp = dispatch(x, ids, weights)
    runner = SimpleNamespace(
        config=config(),
        runner_backend=SimpleNamespace(value="triton"),
        lora_enabled=False,
        down_gemm_overlap_args=None,
        meta_overlap_args=None,
    )
    state = SimpleNamespace(
        a2a=False,
        fused=False,
        symmetric=False,
        invariant=False,
        symmetric_enabled=True,
        world_size=2,
    )
    install_module(
        monkeypatch,
        "sglang.srt.distributed.device_communicators.pynccl_allocator",
        is_symmetric_memory_enabled=lambda: state.symmetric_enabled,
    )
    install_module(
        monkeypatch,
        "sglang.srt.distributed.parallel_state",
        get_tp_group=lambda: SimpleNamespace(world_size=state.world_size),
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.dp_attention",
        is_allocation_symmetric=lambda: state.symmetric,
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.utils",
        get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: not state.a2a),
    )
    install_module(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_exec=lambda: SimpleNamespace(
            moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=state.fused)
        ),
    )
    install_module(
        monkeypatch,
        "sglang.srt.batch_invariant_ops",
        is_batch_invariant_mode_enabled=lambda: state.invariant,
    )
    return runner, inp, state


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("world_size", [1, 2])
def test_symmetric_policy_requires_active_allocator(
    adapters, guarded_dispatch, enabled, symmetric, world_size
):
    runner, inp, state = guarded_dispatch
    state.symmetric_enabled = enabled
    state.symmetric = symmetric
    state.world_size = world_size
    reason = adapters._common_unsupported_reason(runner, inp, None, False)
    if enabled and symmetric and world_size > 1:
        assert reason == "fused collective or symmetric-memory allocation"
    else:
        assert reason is None


@pytest.mark.parametrize(
    "case,reason",
    [
        ("lora", "LoRA"),
        ("format", "dispatch"),
        ("overlap", "overlap"),
        ("no_combine", "no_combine"),
        ("input_weight", "input"),
        ("activation", "modified"),
        ("prequant", "pre-quantized"),
        ("packed", "packed"),
        ("a2a", "all-to-all"),
        ("fused", "collective"),
        ("symmetric", "symmetric"),
        ("invariant", "invariant"),
        ("custom", "custom"),
        ("cpu", "CUDA"),
        ("compile", "compile"),
    ],
)
def test_unsupported_modes_are_explicit(
    adapters, guarded_dispatch, monkeypatch, caplog, case, reason
):
    runner, inp, state = guarded_dispatch
    custom = False
    if case == "lora":
        runner.lora_enabled = True
    elif case == "format":
        inp.format.value = "deepep_ll"
    elif case == "overlap":
        runner.meta_overlap_args = {}
    elif case == "no_combine":
        runner.config.no_combine = True
    elif case == "input_weight":
        runner.config.apply_router_weight_on_input = True
    elif case == "activation":
        runner.config.swiglu_limit = 10
    elif case == "prequant":
        inp.hidden_states_pre_quant = (inp.hidden_states, None)
    elif case == "packed":
        inp.topk_output = SimpleNamespace(packed_topk_ids=inp.topk_output.topk_ids)
    elif case in ("a2a", "fused", "symmetric", "invariant"):
        setattr(state, case, True)
    elif case == "custom":
        custom = True
    elif case == "cpu":
        inp.hidden_states = torch.empty(7, 64, dtype=torch.bfloat16)
    else:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with caplog.at_level(logging.WARNING):
        output = adapters.run_with_workspace_budget(
            runner, inp, object(), None, 1024, custom_core=custom
        )
    assert output is None
    assert "not enforced" in caplog.text and reason in caplog.text


def test_supported_guard_and_conflicting_controls(
    adapters, guarded_dispatch, monkeypatch
):
    runner, inp, _ = guarded_dispatch
    assert adapters._common_unsupported_reason(runner, inp, None, False) is None
    monkeypatch.setenv("SGLANG_MARLIN_MOE_CHUNK_SIZE", "128")
    with pytest.raises(ValueError, match="Set only"):
        adapters.run_with_workspace_budget(runner, inp, object(), None, 1024)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("hidden_dtype", "activation"),
        ("hidden_stride", "activation"),
        ("ids_dtype", "top-k"),
        ("weights_dtype", "top-k"),
        ("ids_stride", "top-k"),
        ("shape", "top-k"),
        ("zero_topk", "top-k"),
        ("lora_info", "LoRA"),
        ("lora_routing", "experimental"),
        ("unknown_backend", "no adapter"),
        ("metadata", "metadata"),
    ],
)
def test_additional_capability_boundaries(
    adapters, guarded_dispatch, triton_runtime, monkeypatch, caplog, case, reason
):
    runner, inp, _ = guarded_dispatch
    topk = inp.topk_output
    lora = None
    with inp.hidden_states.fake_mode:
        if case == "hidden_dtype":
            inp.hidden_states = torch.empty(7, 64, device="cuda")
        elif case == "hidden_stride":
            inp.hidden_states = torch.empty_strided(
                (7, 64), (1, 7), device="cuda", dtype=torch.bfloat16
            )
        elif case == "ids_dtype":
            topk.topk_ids = torch.empty(7, 2, device="cuda")
        elif case == "weights_dtype":
            topk.topk_weights = torch.empty(7, 2, device="cuda", dtype=torch.float16)
        elif case == "ids_stride":
            topk.topk_ids = torch.empty_strided(
                (7, 2), (1, 7), device="cuda", dtype=torch.int32
            )
        elif case == "shape":
            topk.topk_weights = torch.empty(3, 2, device="cuda")
        elif case == "zero_topk":
            topk.topk_ids = torch.empty(7, 0, device="cuda", dtype=torch.int32)
            topk.topk_weights = torch.empty(7, 0, device="cuda")
    if case == "lora_info":
        lora = {}
    elif case == "lora_routing":
        monkeypatch.setenv("SGLANG_EXPERIMENTAL_LORA_OPTI", "1")
    elif case == "unknown_backend":
        runner.runner_backend.value = "deep_gemm"
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            result = adapters.run_with_workspace_budget(
                runner, inp, object(), lora, 1024
            )
            assert result is None
    assert reason in caplog.text
    assert len(caplog.records) == 1
    assert not triton_runtime.aligns and not triton_runtime.calls


@pytest.mark.parametrize(
    "field,environment,expected",
    [(None, "1234", 1234), (0, "1234", 0), (4321, "1234", 4321)],
)
@pytest.mark.parametrize("accepted", [False, True])
def test_runner_budget_precedence(monkeypatch, field, environment, expected, accepted):
    tree = ast.parse((RUNNER / "runner.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MoeRunner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    # Compile annotations as strings, just as this production module does.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = dict(envs=envs, _CUSTOM_RUNNER_CORE_FACTORIES={})
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(RUNNER / "runner.py"),
            "exec",
        ),
        ns,
    )
    seen = []
    policy_result, fallback = object(), object()
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.workspace",
        run_with_workspace_budget=lambda runner, inp, quant, lora, budget, **kwargs: (
            seen.append(budget) or (policy_result if accepted else None)
        ),
    )
    monkeypatch.setenv("SGLANG_MOE_WORKSPACE_BUDGET_BYTES", environment)
    runner = SimpleNamespace(
        config=config(workspace_budget_bytes=field),
        runner_backend=SimpleNamespace(value="triton"),
        lora_enabled=False,
        fused_func=lambda *args: fallback,
    )
    result = ns["run"](runner, object(), object())
    assert result is (policy_result if expected and accepted else fallback)
    assert seen == ([expected] if expected else [])
    runner.config.workspace_budget_bytes = -1
    with pytest.raises(ValueError, match="nonnegative"):
        ns["run"](runner, object(), object())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
