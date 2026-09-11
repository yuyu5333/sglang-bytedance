"""CPU metadata and dispatch checks; CUDA arithmetic is tested separately."""

import ast
from types import SimpleNamespace

import pytest
import torch
from test_workspace_adapters import (
    ROOT,
    RUNNER,
    config,
    dispatch,
    install_module,
    load_nodes,
)

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
KERNEL = ROOT / "python/sglang/kernels/ops/moe/direct_decode.py"


def metadata(shape, dtype=torch.bfloat16, device="cuda", contiguous=True):
    return SimpleNamespace(
        shape=torch.Size(shape),
        ndim=len(shape),
        dtype=dtype,
        device=torch.device(device),
        is_contiguous=lambda: contiguous,
    )


def inputs(m=1, h=256, i=128, e=4, topk=2, dtype=torch.bfloat16):
    return [
        metadata((m, h), dtype),
        metadata((e, 2 * i, h), dtype),
        metadata((e, h, i), dtype),
        metadata((m, topk), torch.int32),
        metadata((m, topk), torch.float32),
        metadata((e, 2 * i), dtype),
        metadata((e, h), dtype),
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m,h,i,e,topk", [(1, 128, 128, 1, 1), (8, 8192, 4096, 8, 8), (3, 288, 160, 4, 3)]
)
def test_supported_metadata_boundaries(dtype, m, h, i, e, topk):
    fn = load_nodes(KERNEL, ["direct_decode_supported"])["direct_decode_supported"]
    args = inputs(m, h, i, e, topk, dtype)
    assert fn(*args)
    args[3].dtype = torch.int64
    assert fn(*args)
    assert fn(*args[:5])


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"device": torch.device("cpu")}),
        (0, {"ndim": 3}),
        (0, {"dtype": torch.float32}),
        (0, {"shape": torch.Size((0, 256))}),
        (0, {"shape": torch.Size((9, 256))}),
        (0, {"shape": torch.Size((1, 127))}),
        (0, {"shape": torch.Size((1, 8193))}),
        (1, {"ndim": 2}),
        (2, {"ndim": 2}),
        (3, {"ndim": 1}),
        (3, {"shape": torch.Size((1, 0))}),
        (3, {"dtype": torch.float32}),
        (4, {"dtype": torch.bfloat16}),
        (4, {"shape": torch.Size((2, 2))}),
        (1, {"shape": torch.Size((4, 128, 256))}),
        (2, {"shape": torch.Size((1, 256, 128))}),
        (2, {"shape": torch.Size((4, 512, 128))}),
        (2, {"shape": torch.Size((4, 256, 127))}),
        (2, {"shape": torch.Size((4, 256, 4097))}),
        (1, {"dtype": torch.float16}),
        (2, {"dtype": torch.float16}),
        (5, {"shape": torch.Size((4, 128))}),
        (6, {"shape": torch.Size((4, 128))}),
        (5, {"dtype": torch.float32}),
        (6, {"dtype": torch.float32}),
    ]
    + [(i, {"device": torch.device("cuda:1")}) for i in range(1, 7)]
    + [(i, {"is_contiguous": lambda: False}) for i in range(7)],
)
def test_unsupported_metadata_rejected_before_launch(index, change):
    ns = load_nodes(KERNEL, ["direct_decode_supported", "direct_decode"])
    args = inputs()
    vars(args[index]).update(change)
    assert not ns["direct_decode_supported"](*args)
    with pytest.raises(ValueError, match="Unsupported direct MoE"):
        ns["direct_decode"](*args[:5], b1=args[5], b2=args[6])


def test_hip_is_not_selected(monkeypatch):
    fn = load_nodes(KERNEL, ["direct_decode_supported"])["direct_decode_supported"]
    monkeypatch.setattr(torch.version, "hip", "test")
    assert not fn(*inputs())


@pytest.fixture
def selector(monkeypatch):
    gate = load_nodes(KERNEL, ["direct_decode_supported"])["direct_decode_supported"]
    quant_cls = load_nodes(RUNNER / "triton.py", ["TritonMoeQuantInfo"])[
        "TritonMoeQuantInfo"
    ]
    state = SimpleNamespace(common=None, world_size=1, deterministic=False)
    calls = []
    sentinel = object()
    install_module(
        monkeypatch,
        "sglang.kernels.ops.moe.direct_decode",
        direct_decode_supported=gate,
        direct_decode=lambda *args, **kwargs: calls.append((args, kwargs)) or sentinel,
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.triton",
        TritonMoeQuantInfo=quant_cls,
    )
    install_module(
        monkeypatch,
        "sglang.srt.distributed.parallel_state",
        get_tp_group=lambda: SimpleNamespace(world_size=state.world_size),
    )
    install_module(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_exec=lambda: SimpleNamespace(
            deterministic=SimpleNamespace(
                enable_deterministic_inference=state.deterministic
            )
        ),
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.token_dispatcher.standard",
        StandardCombineInput=lambda value: SimpleNamespace(hidden_states=value),
    )
    ns = load_nodes(
        RUNNER / "direct_decode.py",
        ["try_direct_decode"],
        {
            "_common_unsupported_reason": lambda *args: state.common,
        },
    )
    args = inputs()
    inp = dispatch(args[0], args[3], args[4])
    quant = quant_cls(args[1], args[2], b13=args[5], b2=args[6])
    runner = SimpleNamespace(
        config=config(), runner_backend=SimpleNamespace(value="triton")
    )
    return SimpleNamespace(
        run=ns["try_direct_decode"],
        runner=runner,
        inp=inp,
        quant=quant,
        state=state,
        calls=calls,
        sentinel=sentinel,
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("scale", [None, 0.0, 0.5, 1.7])
def test_selector_forwards_supported_call(selector, inplace, scale):
    rt = selector
    rt.runner.config.inplace = inplace
    rt.runner.config.routed_scaling_factor = scale
    result = rt.run(rt.runner, rt.inp, rt.quant, None, False)
    assert result.hidden_states is rt.sentinel
    assert len(rt.calls) == 1
    args, kwargs = rt.calls[0]
    assert args[0] is rt.inp.hidden_states
    assert args[1] is rt.quant.w13_weight and args[2] is rt.quant.w2_weight
    assert args[3] is rt.inp.topk_output.topk_ids
    assert kwargs == dict(
        b1=rt.quant.b13,
        b2=rt.quant.b2,
        scale=1.0 if scale is None else scale,
        inplace=inplace,
    )


@pytest.mark.parametrize(
    "case",
    [
        "common",
        "backend",
        "tokens",
        "hidden",
        "intermediate",
        "activation",
        "tp",
        "deterministic",
        "metadata_class",
        "metadata_dtype",
        "use_mxfp8",
        "use_fp8_w8a8",
        "use_int8_w8a8",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "per_channel_quant",
        "fuse_swiglu_interleaved",
        "w13_scale",
        "w2_scale",
        "w13_zp",
        "w2_zp",
        "a13_scale",
        "a2_scale",
        "block_shape",
    ],
)
def test_selector_unsupported_preserves_original_path(selector, case):
    rt = selector
    if case == "common":
        rt.state.common = "LoRA or communication"
    elif case == "backend":
        rt.runner.runner_backend.value = "marlin"
    elif case == "tokens":
        rt.inp.hidden_states.shape = torch.Size((4, 256))
    elif case == "hidden":
        rt.inp.hidden_states.shape = torch.Size((1, 8192))
    elif case == "intermediate":
        rt.quant.w13_weight.shape = torch.Size((4, 4096, 256))
        rt.quant.w2_weight.shape = torch.Size((4, 256, 2048))
        rt.quant.b13 = None
    elif case == "activation":
        rt.runner.config.activation = "gelu"
    elif case == "tp":
        rt.state.world_size = 2
    elif case == "deterministic":
        rt.state.deterministic = True
    elif case == "metadata_class":
        rt.quant = object()
    elif case == "metadata_dtype":
        rt.quant.w13_weight.dtype = torch.float32
    else:
        setattr(rt.quant, case, True)
    assert rt.run(rt.runner, rt.inp, rt.quant, None, False) is None
    assert not rt.calls


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize(
    "field,environment,budget_active",
    [
        (0, "4096", False),
        (None, "4096", True),
        (1024, "0", True),
    ],
)
def test_runner_direct_switch_and_budget_priority(
    monkeypatch, enabled, accepted, field, environment, budget_active
):
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
    calls = []
    fast, fallback = object(), object()
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.direct_decode",
        try_direct_decode=lambda *args, **kwargs: (
            calls.append("direct") or (fast if accepted else None)
        ),
    )
    install_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.workspace",
        run_with_workspace_budget=lambda *args, **kwargs: (
            calls.append("budget") or None
        ),
    )
    monkeypatch.setenv("SGLANG_MOE_DIRECT_DECODE", "1" if enabled else "0")
    monkeypatch.setenv("SGLANG_MOE_WORKSPACE_BUDGET_BYTES", environment)
    runner = SimpleNamespace(
        config=config(workspace_budget_bytes=field),
        runner_backend=SimpleNamespace(value="triton"),
        lora_enabled=False,
        fused_func=lambda *args: fallback,
    )
    result = ns["run"](runner, object(), object())
    selected = enabled and not budget_active
    assert result is (fast if selected and accepted else fallback)
    assert calls == (["budget"] if budget_active else ["direct"] if enabled else [])
