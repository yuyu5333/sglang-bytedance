"""CPU launch-contract tests, separate from CUDA numerical validation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_direct_decode import KERNEL, inputs, metadata
from test_workspace_adapters import ROOT, load_nodes

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
SOURCE = ROOT / "python/sglang/kernels/ops/moe/streamed_decode.py"


@pytest.fixture
def runtime():
    allocations, calls = [], []

    def empty(shape, *, device, dtype):
        value = metadata(shape, dtype, device)
        allocations.append(value)
        return value

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((self.name, grid, args, kwargs))

    gate = load_nodes(KERNEL, ["direct_decode_supported"])["direct_decode_supported"]
    ns = load_nodes(
        SOURCE,
        ["streamed_decode"],
        {
            "torch": SimpleNamespace(
                Tensor=torch.Tensor,
                empty=empty,
                empty_like=lambda x: empty(x.shape, device=x.device, dtype=x.dtype),
                cuda=SimpleNamespace(device=lambda _: nullcontext()),
            ),
            "triton": SimpleNamespace(
                cdiv=lambda a, b: (a + b - 1) // b,
                next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            ),
            "direct_decode_supported": gate,
            "_gate_up": Launch("up"),
            "_down_sum": Launch("vectorized"),
            "_streamed_down_sum": Launch("streamed"),
        },
    )
    return SimpleNamespace(
        run=ns["streamed_decode"], allocations=allocations, calls=calls
    )


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "m,h,i,topk,un,dn,uw,dw,ul",
    [
        (1, 128, 128, 1, 2, 2, 4, 4, 1),
        (3, 288, 160, 3, 4, 16, 4, 8, 2),
        (8, 8192, 4096, 8, 16, 64, 8, 8, 8),
    ],
)
def test_streamed_decode_launch_contract(
    runtime, vectorized, inplace, bias, m, h, i, topk, un, dn, uw, dw, ul
):
    rt = runtime
    args = inputs(m=m, h=h, i=i, e=8, topk=topk)
    b1, b2 = args[5:] if bias else (None, None)
    out = rt.run(
        *args[:5],
        b1=b1,
        b2=b2,
        scale=0.5,
        inplace=inplace,
        up_n=un,
        down_n=dn,
        up_warps=uw,
        down_warps=dw,
        unroll=ul,
        vectorized_down=vectorized,
    )
    assert len(rt.allocations) == (1 if inplace else 2)
    assert tuple(rt.allocations[0].shape) == (m * topk, i)
    assert out is (args[0] if inplace else rt.allocations[1])
    assert all(
        t.dtype == args[0].dtype and t.device == args[0].device for t in rt.allocations
    )
    assert len(rt.calls) == 2
    up, down = rt.calls
    assert up[:2] == ("up", (m * topk, (i + un - 1) // un))
    assert down[:2] == (
        "vectorized" if vectorized else "streamed",
        (m, (h + dn - 1) // dn),
    )
    assert up[2][:5] == (args[0], args[1], b1, args[3], rt.allocations[0])
    assert up[2][5:] == (h, i, 8, topk, bias)
    assert down[2][:6] == (rt.allocations[0], args[2], b2, args[3], args[4], out)
    assert down[2][6:] == (h, i, 8, topk, 0.5, bias)
    assert up[3] == dict(
        BN=un, BK=1 << (h - 1).bit_length(), num_warps=uw, enable_fp_fusion=False
    )
    options = dict(BT=1 << (topk - 1).bit_length()) if vectorized else dict(UNROLL=ul)
    assert down[3] == dict(
        BN=dn,
        BK=1 << (i - 1).bit_length(),
        num_warps=dw,
        enable_fp_fusion=False,
        **options,
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
        "up_n",
        "down_n",
        "up_warps",
        "down_warps",
        "unroll",
    ],
)
def test_streamed_decode_rejects_before_allocation(runtime, case):
    args = inputs()
    kwargs = dict(b1=args[5], b2=args[6])
    message = "Unsupported streamed MoE decode metadata"
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
        kwargs[case] = 3
        message = "Unsupported streamed MoE launch configuration"
    with pytest.raises(ValueError, match=message):
        runtime.run(*args[:5], **kwargs)
    assert not runtime.allocations
    assert not runtime.calls
