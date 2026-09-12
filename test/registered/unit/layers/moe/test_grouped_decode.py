"""CPU launch-contract tests; tensor-core arithmetic needs the GPU harness."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_direct_decode import KERNEL, inputs, metadata
from test_workspace_adapters import ROOT, load_nodes

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
GROUPED = ROOT / "python/sglang/kernels/ops/moe/grouped_decode.py"


@pytest.fixture
def runtime():
    allocations, calls, devices = [], [], []

    def empty(shape, *, device, dtype):
        value = metadata(shape, dtype, device)
        allocations.append(value)
        return value

    def device_scope(device):
        devices.append(device)
        return nullcontext()

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((self.name, grid, args, kwargs))

    gate = load_nodes(KERNEL, ["direct_decode_supported"])["direct_decode_supported"]
    ns = load_nodes(
        GROUPED,
        ["grouped_decode"],
        {
            "torch": SimpleNamespace(
                Tensor=torch.Tensor,
                empty=empty,
                empty_like=lambda x: empty(x.shape, device=x.device, dtype=x.dtype),
                cuda=SimpleNamespace(device=device_scope),
            ),
            "triton": SimpleNamespace(
                cdiv=lambda a, b: (a + b - 1) // b,
                next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            ),
            "direct_decode_supported": gate,
            "_grouped_gate_up": Launch("up"),
            "_grouped_down": Launch("down"),
            "_grouped_sum": Launch("sum"),
        },
    )
    return SimpleNamespace(
        run=ns["grouped_decode"],
        allocations=allocations,
        calls=calls,
        devices=devices,
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "m,h,i,e,topk,bn,bk",
    [
        (1, 128, 128, 1, 1, 16, 32),
        (3, 288, 160, 4, 3, 32, 64),
        (8, 8192, 4096, 8, 8, 64, 256),
    ],
)
def test_grouped_decode_launch_contract(
    runtime, inplace, bias, m, h, i, e, topk, bn, bk
):
    rt = runtime
    args = inputs(m, h, i, e, topk)
    b1, b2 = args[5:] if bias else (None, None)
    result = rt.run(
        *args[:5], b1=b1, b2=b2, scale=1.7, inplace=inplace, block_n=bn, block_k=bk
    )
    assert [tuple(t.shape) for t in rt.allocations[:2]] == [
        (m * topk, i),
        (m * topk, h),
    ]
    assert len(rt.allocations) == (2 if inplace else 3)
    assert all(
        t.dtype == args[0].dtype and t.device == args[0].device for t in rt.allocations
    )
    assert result is (args[0] if inplace else rt.allocations[2])
    assert rt.devices == [args[0].device]
    up, down, combine = rt.calls
    assert [(c[0], c[1]) for c in rt.calls] == [
        ("up", (m * topk, (i + bn - 1) // bn)),
        ("down", (m * topk, (h + bn - 1) // bn)),
        ("sum", (m, (h + 255) // 256)),
    ]
    assert up[2][:4] == (args[0], args[1], b1, args[3])
    assert up[2][4] is rt.allocations[0]
    assert up[2][5:] == (m, h, i, e, topk, bias)
    assert down[2][:6] == (
        rt.allocations[0],
        args[2],
        b2,
        args[3],
        args[4],
        rt.allocations[1],
    )
    assert down[2][6:] == (m, h, i, e, topk, bias)
    assert combine[2][:3] == (rt.allocations[1], args[3], result)
    assert combine[2][3:] == (h, e, topk, 1.7)
    assert (
        up[3]
        == down[3]
        == dict(
            BM=16,
            BT=1 << (topk - 1).bit_length(),
            BN=bn,
            BK=bk,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
    )
    previous = list(rt.allocations)
    rt.run(*args[:5])
    assert all(a is not b for a in previous for b in rt.allocations[len(previous) :])


@pytest.mark.parametrize(
    "case",
    ["empty", "cpu", "fp32", "strided", "bad_weight", "bad_bias", "tile_n", "tile_k"],
)
def test_grouped_decode_rejects_before_allocation(runtime, case):
    args = inputs()
    kwargs = dict(b1=args[5], b2=args[6])
    message = "Unsupported grouped MoE metadata"
    if case == "empty":
        args[0].shape = torch.Size((0, 256))
    elif case == "cpu":
        args[0].device = torch.device("cpu")
    elif case == "fp32":
        args[0].dtype = torch.float32
    elif case == "strided":
        args[1].is_contiguous = lambda: False
    elif case == "bad_weight":
        args[2].shape = torch.Size((4, 256, 127))
    elif case == "bad_bias":
        args[5].shape = torch.Size((4, 128))
    else:
        kwargs["block_n" if case == "tile_n" else "block_k"] = 0
        message = "Unsupported grouped MoE tile"
    if "tile" not in case:
        message = "Unsupported grouped MoE decode metadata"
    with pytest.raises(ValueError, match=message):
        runtime.run(*args[:5], **kwargs)
    assert not runtime.allocations
    assert not runtime.calls
    assert not runtime.devices
