"""CPU wrapper contracts; real CUDA validates the adaptive arithmetic."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_direct_decode import KERNEL, inputs, metadata
from test_workspace_adapters import ROOT, load_nodes

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
SOURCE = ROOT / "python/sglang/kernels/ops/moe/adaptive_decode.py"


@pytest.fixture
def runtime():
    allocations, calls, devices = [], [], []

    def empty(shape, *, device, dtype):
        value = metadata(shape, dtype, device)
        allocations.append(value)
        return value

    def scope(device):
        devices.append(device)
        return nullcontext()

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((self.name, grid, args, kwargs))

    gate = load_nodes(KERNEL, ["direct_decode_supported"])["direct_decode_supported"]
    ns = load_nodes(
        SOURCE,
        ["adaptive_decode"],
        {
            "torch": SimpleNamespace(
                Tensor=torch.Tensor,
                empty=empty,
                empty_like=lambda x: empty(x.shape, device=x.device, dtype=x.dtype),
                cuda=SimpleNamespace(device=scope),
            ),
            "triton": SimpleNamespace(
                cdiv=lambda a, b: (a + b - 1) // b,
                next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            ),
            "direct_decode_supported": gate,
            "_adaptive_gate_up": Launch("up"),
            "_adaptive_down": Launch("down"),
            "_grouped_sum": Launch("sum"),
        },
    )
    return SimpleNamespace(
        run=ns["adaptive_decode"], allocations=allocations, calls=calls, devices=devices
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "m,h,i,topk,sn,gn,gk,reuse",
    [
        (1, 128, 128, 1, 4, 16, 32, 1),
        (5, 288, 160, 3, 8, 32, 64, 4),
        (8, 8192, 4096, 8, 16, 64, 256, 9),
    ],
)
def test_adaptive_decode_launch_contract(
    runtime, inplace, bias, m, h, i, topk, sn, gn, gk, reuse
):
    rt = runtime
    args = inputs(m=m, h=h, i=i, e=8, topk=topk)
    b1, b2 = args[5:] if bias else (None, None)
    out = rt.run(
        *args[:5],
        b1=b1,
        b2=b2,
        scale=1.7,
        inplace=inplace,
        min_reuse=reuse,
        simt_n=sn,
        group_n=gn,
        group_k=gk,
    )
    assert len(rt.allocations) == (2 if inplace else 3)
    assert [tuple(x.shape) for x in rt.allocations[:2]] == [
        (m * topk, i),
        (m * topk, h),
    ]
    assert out is (args[0] if inplace else rt.allocations[2])
    assert all(
        t.dtype == args[0].dtype and t.device == args[0].device for t in rt.allocations
    )
    assert rt.devices == [args[0].device]
    assert len(rt.calls) == 3
    up, down, combine = rt.calls
    assert [(c[0], c[1]) for c in rt.calls] == [
        ("up", (m * topk, (i + min(sn, gn) - 1) // min(sn, gn))),
        ("down", (m * topk, (h + min(sn, gn) - 1) // min(sn, gn))),
        ("sum", (m, (h + 255) // 256)),
    ]
    assert up[2] == (
        args[0],
        args[1],
        b1,
        args[3],
        rt.allocations[0],
        m,
        h,
        i,
        8,
        topk,
        bias,
    )
    assert down[2] == (
        rt.allocations[0],
        args[2],
        b2,
        args[3],
        args[4],
        rt.allocations[1],
        m,
        h,
        i,
        8,
        topk,
        bias,
    )
    assert combine[2] == (rt.allocations[1], args[3], out, h, 8, topk, 1.7)
    options = dict(
        BT=1 << (topk - 1).bit_length(),
        SN=sn,
        GN=gn,
        GK=gk,
        MIN_REUSE=reuse,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    assert up[3] == dict(SK=1 << (h - 1).bit_length(), **options)
    assert down[3] == dict(SK=1 << (i - 1).bit_length(), **options)
    assert combine[3] == dict(
        BT=1 << (topk - 1).bit_length(),
        BN=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    previous = list(rt.allocations)
    rt.run(*args[:5])
    assert all(a is not b for a in previous for b in rt.allocations[len(previous) :])


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "empty",
        "fp32",
        "weight",
        "bias",
        "strided",
        "reuse_zero",
        "reuse_negative",
        "reuse_large",
        "simt_n",
        "group_n",
        "group_k",
    ],
)
def test_adaptive_decode_rejects_before_allocation(runtime, case):
    args = inputs()
    kwargs = dict(b1=args[5], b2=args[6])
    message = "Unsupported adaptive MoE decode metadata"
    if case == "cpu":
        args[0].device = torch.device("cpu")
    elif case == "empty":
        args[0].shape = torch.Size((0, 256))
    elif case == "fp32":
        args[0].dtype = torch.float32
    elif case == "weight":
        args[2].shape = torch.Size((4, 256, 127))
    elif case == "bias":
        args[5].shape = torch.Size((4, 128))
    elif case == "strided":
        args[1].is_contiguous = lambda: False
    else:
        if case.startswith("reuse"):
            kwargs["min_reuse"] = {
                "reuse_zero": 0,
                "reuse_negative": -1,
                "reuse_large": 10,
            }[case]
        else:
            kwargs[case] = 3
        message = "Unsupported adaptive MoE launch configuration"
    with pytest.raises(ValueError, match=message):
        runtime.run(*args[:5], **kwargs)
    assert not runtime.allocations
    assert not runtime.calls
    assert not runtime.devices
