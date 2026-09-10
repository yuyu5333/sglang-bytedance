import sys
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.workspace_policy import (
    MarlinWorkspaceEstimate,
    TritonWorkspaceEstimate,
    plan_workspace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "tokens,budget,expected",
    [
        (0, 1, 0),
        (1, 10, 1),
        (13, 130, 13),
        (13, 129, 8),
        (13, 40, 4),
        (13, 10, 1),
        (1024, 10239, 512),
    ],
)
def test_largest_fitting_bucket(tokens, budget, expected):
    estimate = SimpleNamespace(peak_bytes=lambda size: size * 10)
    plan = plan_workspace(tokens, budget, estimate)
    assert plan.tokens == tokens
    assert plan.chunk_tokens == expected
    assert plan.budget_bytes == budget
    assert plan.estimated_peak_bytes == expected * 10


@pytest.mark.parametrize("tokens,budget", [(-1, 100), (2, 0), (2, -1)])
def test_invalid_budget(tokens, budget):
    with pytest.raises(ValueError, match="nonnegative"):
        plan_workspace(tokens, budget, SimpleNamespace(peak_bytes=lambda size: size))


def test_nonmonotone_and_impossible_estimates():
    # Smaller block-M may choose a larger FP32-reduction scratch allocation.
    estimate = SimpleNamespace(
        peak_bytes=lambda size: {13: 500, 8: 70, 4: 200, 2: 30, 1: 50}[size]
    )
    assert plan_workspace(13, 70, estimate).chunk_tokens == 8
    assert plan_workspace(13, 30, estimate).chunk_tokens == 2
    with pytest.raises(ValueError, match="cannot fit"):
        plan_workspace(13, 29, estimate)
    with pytest.raises(ValueError, match="nonnegative"):
        plan_workspace(1, 100, SimpleNamespace(peak_bytes=lambda size: -1))


@pytest.mark.parametrize("backend", ["marlin", "triton"])
@pytest.mark.parametrize("tokens", [1, 7, 42, 43, 256, 1024, 8192])
def test_backend_estimates_cover_activation_storage(backend, tokens):
    if backend == "marlin":
        estimate = MarlinWorkspaceEstimate(4096, 512, 6, 256, 78, True)
        activation = 2 * tokens * 6 * (4096 + 512)
    else:
        estimate = TritonWorkspaceEstimate(4096, 512, 6, 256, 64)
        activation = 2 * tokens * 6 * (4096 + 3 * 512)
    assert estimate.peak_bytes(0) == 0
    assert estimate.peak_bytes(tokens) > activation
    plan = plan_workspace(tokens, estimate.peak_bytes(tokens), estimate)
    assert plan.chunk_tokens == tokens


def test_reference_budget_and_fp32_reduction():
    marlin = MarlinWorkspaceEstimate(4096, 512, 6, 256, 78, True)
    atomic = MarlinWorkspaceEstimate(4096, 512, 6, 256, 78, False)
    triton = TritonWorkspaceEstimate(4096, 512, 6, 256, 64)
    assert marlin.peak_bytes(1024) == 66_966_456
    assert marlin.peak_bytes(1024) - atomic.peak_bytes(1024) == 78 * 4 * 32 * 256 * 4
    assert plan_workspace(8192, 64 * 2**20, marlin).chunk_tokens == 1024
    assert plan_workspace(8192, 64 * 2**20, triton).chunk_tokens == 512


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
