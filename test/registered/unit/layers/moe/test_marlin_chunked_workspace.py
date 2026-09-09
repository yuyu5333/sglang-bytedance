"""CPU dataflow checks; these do not validate Marlin CUDA arithmetic or locks."""

import ast
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers import zero_copy_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
)


def _activate(x, activation, gated, limit, alpha):
    if not gated:
        return F.silu(x) if activation == "silu" else torch.relu(x).square()
    gate, up = x.chunk(2, dim=-1)
    if activation == "situ":
        beta = 4.0 if alpha is None else alpha
        gate = beta * torch.tanh(gate.float() / beta) * torch.sigmoid(gate.float())
        up = up.float()
        if limit is not None:
            up = limit * torch.tanh(up / limit)
        return (gate * up).to(x.dtype)
    if alpha is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * torch.sigmoid(gate * alpha) * (up + 1)
    if limit is not None:
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return F.silu(gate) * up
    return (F.silu(gate.float()) * up.float()).to(x.dtype)


class CpuMarlin:
    def __init__(self, dtype, gated=True, mxfp4=False, n=32, bias=False):
        self.dtype = dtype
        self.n = n
        self.k = 64
        self.experts = 4
        self.gated = gated
        self.calls = []
        self.align_calls = []
        generator = torch.Generator().manual_seed(42)
        self.w13 = torch.randn(
            self.experts, n * (2 if gated else 1), self.k, generator=generator
        ) / math.sqrt(self.k)
        self.w2 = torch.randn(self.experts, self.k, n, generator=generator) / math.sqrt(
            n
        )
        self.b13 = (
            torch.randn(self.experts, self.w13.shape[1], generator=generator).to(dtype)
            if bias
            else None
        )
        self.b2 = (
            torch.randn(self.experts, self.k, generator=generator).to(dtype)
            if bias
            else None
        )
        self.mxfp4 = mxfp4

    def inputs(self, m, ids_dtype=torch.int64, ep=False):
        generator = torch.Generator().manual_seed(123)
        hidden = torch.randn(m, self.k, generator=generator).to(self.dtype)
        ids = torch.stack(
            (torch.arange(m) % self.experts, (torch.arange(m) + 1) % self.experts),
            dim=1,
        ).to(ids_dtype)
        if ep and m:
            ids[1::2, 0] = -1
            ids[-1] = -1
        weights = torch.rand(m, 2, generator=generator)
        scales_dtype = torch.float8_e8m0fnu if self.mxfp4 else self.dtype
        bits = 4 if self.mxfp4 else 8
        return dict(
            hidden_states=hidden,
            w1=torch.empty(self.experts, self.k // 16, 16, dtype=torch.int32),
            w2=torch.empty(
                self.experts, self.n // 16, self.k * (bits // 2), dtype=torch.int32
            ),
            w1_scale=torch.empty(
                self.experts, 1, self.w13.shape[1], dtype=scales_dtype
            ),
            w2_scale=torch.empty(self.experts, 1, self.k, dtype=scales_dtype),
            gating_output=torch.zeros(m, self.experts),
            topk_ids=ids,
            topk_weights=weights,
            expert_map=torch.arange(6) if ep else None,
            global_num_experts=6 if ep else -1,
            num_bits=bits,
            is_gated=self.gated,
            w1_bias=self.b13,
            w2_bias=self.b2,
        )

    def align(self, ids, block, experts=None):
        self.align_calls.append((ids.shape[0], block, experts))
        flat = ids.reshape(-1)
        sorted_ids, expert_ids = [], []
        for expert in sorted(flat.unique().tolist()):
            rows = (flat == expert).nonzero().flatten().tolist()
            padding = (-len(rows)) % block
            sorted_ids.extend(rows + [flat.numel()] * padding)
            expert_ids.extend([expert] * ((len(rows) + padding) // block))
        return (
            torch.tensor(sorted_ids, dtype=torch.int32),
            torch.tensor(expert_ids, dtype=torch.int32),
            torch.tensor([len(sorted_ids)], dtype=torch.int32),
        )

    def gemm(
        self,
        a,
        c,
        weight,
        bias,
        scale,
        global_scale,
        zeros,
        g_idx,
        perm,
        workspace,
        sorted_ids,
        experts,
        num_padded,
        topk_weights,
        **kwargs,
    ):
        down = kwargs["mul_topk_weights"]
        topk = kwargs["top_k"]
        assert a.shape == (kwargs["size_m"], kwargs["size_k"])
        assert c.shape == (kwargs["size_m"] * topk, kwargs["size_n"])
        assert c.is_contiguous()
        assert (
            a.shape[0] == topk_weights.numel()
            if down
            else a.shape[0] == topk_weights.shape[0]
        )
        assert kwargs["use_fp32_reduce"] is True
        assert kwargs["use_atomic_add"] is (not self.mxfp4)
        assert kwargs["b_q_type"] == ("fp4" if self.mxfp4 else "int8")
        assert workspace.dtype == torch.int32 and not workspace.count_nonzero()
        if not down:
            assert not c.count_nonzero(), "stale data entering gate-up scratch"
        if kwargs["is_ep"] and down:
            assert not c.count_nonzero(), "masked routes must have neutral down output"
        self.calls.append(
            dict(
                down=down,
                rows=c.shape[0],
                cache_ptr=c.untyped_storage().data_ptr(),
                cache_bytes=c.untyped_storage().nbytes(),
                activation_ptr=a.untyped_storage().data_ptr(),
                workspace_ptr=workspace.data_ptr(),
                g_idx=g_idx,
                perm=perm,
            )
        )
        matrix = self.w2 if down else self.w13
        block = kwargs["moe_block_size"]
        for i in range(int(num_padded[0])):
            row = int(sorted_ids[i])
            expert = int(experts[i // block])
            if row >= c.shape[0] or expert < 0:
                continue
            value = a[row // topk].float() @ matrix[expert].T
            if bias is not None:
                value = value + bias[expert].float()
            value = value.to(a.dtype)
            if down:
                value = (value.float() * topk_weights.reshape(-1)[row]).to(a.dtype)
            c[row].copy_(value)
        return c

    def reference(self, args):
        hidden = args["hidden_states"]
        output = torch.empty_like(hidden)
        for row in range(hidden.shape[0]):
            routed = []
            for slot, expert in enumerate(args["topk_ids"][row].tolist()):
                if expert < 0:
                    routed.append(torch.zeros(self.k, dtype=self.dtype))
                    continue
                gate_up = hidden[row].float() @ self.w13[expert].T
                if self.b13 is not None:
                    gate_up += self.b13[expert].float()
                activated = _activate(
                    gate_up.to(self.dtype),
                    args.get("activation", "silu"),
                    self.gated,
                    args.get("clamp_limit"),
                    args.get("gemm1_alpha"),
                )
                down = activated.float() @ self.w2[expert].T
                if self.b2 is not None:
                    down += self.b2[expert].float()
                weighted = down.to(self.dtype).float() * args["topk_weights"][row, slot]
                routed.append(weighted.to(self.dtype))
            factor = args.get("routed_scaling_factor")
            scaling = 1.0 if self.mxfp4 or factor is None else factor
            output[row] = torch.stack(routed).float().sum(dim=0) * scaling
        return output


@pytest.fixture
def runtime():
    # Compile the real function bodies, avoiding only CUDA/Triton import and
    # custom-op registration. No production branch or allocation is replaced.
    names = {
        "fused_marlin_moe",
        "get_scalar_type",
        "swiglu_limit_func",
        "swiglu_gpt_oss_sigmoid_alpha_contiguous",
    }
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in body} == names
    for node in body:
        node.decorator_list = []
    namespace = dict(
        torch=torch,
        F=F,
        Optional=Optional,
        envs=envs,
        zero_copy_context=zero_copy_context,
    )
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


def _bind(monkeypatch, runtime, engine, limit):
    monkeypatch.setenv("SGLANG_MARLIN_MOE_CHUNK_SIZE", str(limit))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=78),
    )
    modules = {}
    for name, function_name, function in (
        (
            "sglang.srt.layers.moe.fused_moe_triton",
            "moe_align_block_size",
            engine.align,
        ),
        (
            "sglang.kernels.ops.moe.moe_align_single_token",
            "moe_align_single_token",
            engine.align,
        ),
        (
            "sglang.kernels.ops.moe.moe_topk_sum",
            "moe_topk_sum",
            lambda x, out: out.copy_(x.float().sum(dim=1)),
        ),
    ):
        module = ModuleType(name)
        setattr(module, function_name, function)
        modules[name] = module
    scalar = ModuleType("sgl_kernel.scalar_type")
    scalar.scalar_types = SimpleNamespace(
        float4_e2m1f="fp4", uint4="zp4", uint4b8="int4", uint8b128="int8"
    )
    modules[scalar.__name__] = scalar
    runtime["moe_wna16_marlin_gemm"] = engine.gemm
    runtime["silu_and_mul"] = lambda x, out: out.copy_(
        _activate(x, "silu", True, None, None)
    )
    runtime["situ_and_mul"] = lambda out, x, situ_beta, linear_beta: out.copy_(
        _activate(x, "situ", True, linear_beta, situ_beta)
    )
    runtime["moe_sum_reduce"] = lambda x, out, scale: out.copy_(
        x.float().sum(dim=1) * scale
    )
    return patch.dict(sys.modules, modules)


@pytest.mark.parametrize("limit", [0, 1, 3, 7, 99])
@pytest.mark.parametrize(
    "dtype,mxfp4",
    [(torch.float16, False), (torch.bfloat16, False), (torch.bfloat16, True)],
)
@pytest.mark.parametrize(
    "activation,gated,clamp,alpha",
    [
        ("silu", True, None, None),
        ("silu", True, 1.5, None),
        ("silu", True, 0.0, None),
        ("silu", True, 1.5, 1.702),
        ("situ", True, 2.0, 4.0),
        ("silu", False, None, None),
        ("relu2", False, None, None),
    ],
)
def test_chunked_dataflow(
    runtime, monkeypatch, limit, dtype, mxfp4, activation, gated, clamp, alpha
):
    engine = CpuMarlin(dtype, gated=gated, mxfp4=mxfp4, bias=True)
    args = engine.inputs(7)
    args.update(
        activation=activation,
        clamp_limit=clamp,
        gemm1_alpha=alpha,
        routed_scaling_factor=1.75,
    )
    expected = engine.reference(args)
    original = args["hidden_states"].clone()
    with _bind(monkeypatch, runtime, engine, limit):
        output = runtime["fused_marlin_moe"](**args)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(args["hidden_states"], original, atol=0, rtol=0)
    cap = min(limit, 7) if limit else 7
    assert len(engine.calls) == 2 * math.ceil(7 / cap)
    assert len({call["cache_ptr"] for call in engine.calls}) == 1
    assert len({call["workspace_ptr"] for call in engine.calls}) == 1
    assert {call["cache_bytes"] for call in engine.calls} == {
        cap * 2 * max(engine.w13.shape[1], engine.k) * 2
    }
    if gated:
        assert (
            len({call["activation_ptr"] for call in engine.calls if call["down"]}) == 1
        )


@pytest.mark.parametrize("n", [32, 128])
@pytest.mark.parametrize("limit", [0, 1, 3])
@pytest.mark.parametrize("m", [0, 1, 7, 8])
def test_masked_rows_and_tail(runtime, monkeypatch, n, limit, m):
    engine = CpuMarlin(torch.bfloat16, mxfp4=True, n=n, bias=True)
    args = engine.inputs(m, ep=True)
    expected = engine.reference(args)
    with _bind(monkeypatch, runtime, engine, limit):
        output = runtime["fused_marlin_moe"](**args)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert all(row[2] == 6 for row in engine.align_calls)
    if m == 0:
        assert not engine.calls and output.shape == (0, engine.k)


@pytest.mark.parametrize(
    "mode", ["inplace", "destination", "exact_alias", "offset_alias"]
)
def test_output_ownership(runtime, monkeypatch, mode):
    engine = CpuMarlin(torch.bfloat16, mxfp4=True)
    args = engine.inputs(7, ids_dtype=torch.int32)
    expected = engine.reference(args)
    if mode == "inplace":
        args["inplace"] = True
        destination = args["hidden_states"]
    elif mode == "offset_alias":
        backing = torch.empty(8, engine.k, dtype=engine.dtype)
        backing[:7].copy_(args["hidden_states"])
        args["hidden_states"] = backing[:7]
        destination = backing[1:]
    elif mode == "exact_alias":
        destination = args["hidden_states"]
    else:
        destination = torch.empty_like(args["hidden_states"])
    with (
        _bind(monkeypatch, runtime, engine, 3),
        zero_copy_context.set_moe_output(destination),
    ):
        result = runtime["fused_marlin_moe"](**args)
    assert result.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert len(engine.calls) == (2 if mode == "offset_alias" else 6)
    if mode != "offset_alias":
        assert engine.align_calls[-1][0] == 1
        assert engine.align_calls[-1][2] is None


def test_forwarded_workspace_and_ordering(runtime, monkeypatch):
    engine = CpuMarlin(torch.bfloat16)
    args = engine.inputs(7)
    workspace = torch.zeros(312, dtype=torch.int32)
    order1, order2 = torch.arange(engine.k), torch.arange(engine.n)
    args.update(
        workspace=workspace,
        g_idx1=order1,
        sort_indices1=order1,
        g_idx2=order2,
        sort_indices2=order2,
    )
    with _bind(monkeypatch, runtime, engine, 3):
        runtime["fused_marlin_moe"](**args)
    for call in engine.calls:
        order = order2 if call["down"] else order1
        assert call["workspace_ptr"] == workspace.data_ptr()
        assert call["g_idx"] is order and call["perm"] is order


@pytest.mark.parametrize("m", [0, 1, 6, 7])
@pytest.mark.parametrize("factor", [None, 0.0, 1.5])
def test_single_route_and_scaling(runtime, monkeypatch, m, factor):
    engine = CpuMarlin(torch.bfloat16)
    args = engine.inputs(m, ids_dtype=torch.int32)
    args["topk_ids"] = args["topk_ids"][:, :1].contiguous()
    args["topk_weights"] = args["topk_weights"][:, :1].contiguous()
    args["routed_scaling_factor"] = factor
    expected = engine.reference(args)
    with _bind(monkeypatch, runtime, engine, 3):
        output = runtime["fused_marlin_moe"](**args)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert len(engine.calls) == 2 * math.ceil(m / 3)


def test_default_and_repeated_calls(runtime, monkeypatch):
    engine = CpuMarlin(torch.bfloat16, mxfp4=True)
    with _bind(monkeypatch, runtime, engine, 0):
        monkeypatch.delenv("SGLANG_MARLIN_MOE_CHUNK_SIZE")
        assert envs.SGLANG_MARLIN_MOE_CHUNK_SIZE.get() == 0
        args = engine.inputs(7)
        first = runtime["fused_marlin_moe"](**args)
        first_saved = first.clone()
        assert len(engine.calls) == 2
        monkeypatch.setenv("SGLANG_MARLIN_MOE_CHUNK_SIZE", "3")
        args["topk_ids"] = args["topk_ids"].flip(0).contiguous()
        second = runtime["fused_marlin_moe"](**args)
    torch.testing.assert_close(second, engine.reference(args), atol=0, rtol=0)
    torch.testing.assert_close(first, first_saved, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()
    assert len(engine.calls) == 8


@pytest.mark.parametrize(
    "invalid,match",
    [
        ({"activation": "invalid"}, "Unsupported activation"),
        ({"gemm1_alpha": 1.702}, "requires clamp_limit"),
        ({"limit": -1}, "must be nonnegative"),
    ],
)
def test_configuration_errors(runtime, monkeypatch, invalid, match):
    engine = CpuMarlin(torch.bfloat16)
    args = engine.inputs(7)
    settings = dict(invalid)
    limit = settings.pop("limit", 3)
    args.update(settings)
    with _bind(monkeypatch, runtime, engine, limit):
        with pytest.raises(ValueError, match=match):
            runtime["fused_marlin_moe"](**args)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
