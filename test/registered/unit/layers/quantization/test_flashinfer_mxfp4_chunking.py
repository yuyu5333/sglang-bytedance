import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    _slice_standard_dispatch_output,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput


def test_slice_standard_dispatch_output_slices_token_major_fields():
    hidden_states = torch.arange(20).reshape(5, 4)
    hidden_states_scale = torch.arange(5)
    topk = StandardTopKOutput(
        topk_weights=torch.arange(10).reshape(5, 2),
        topk_ids=torch.arange(10).reshape(5, 2),
        router_logits=torch.arange(15).reshape(5, 3),
    )
    dispatch = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        topk_output=topk,
        hidden_states_pre_quant=(object(),),
    )

    chunk = _slice_standard_dispatch_output(dispatch, topk, 1, 4)

    assert torch.equal(chunk.hidden_states, hidden_states[1:4])
    assert torch.equal(chunk.hidden_states_scale, hidden_states_scale[1:4])
    assert torch.equal(chunk.topk_output.topk_ids, topk.topk_ids[1:4])
    assert torch.equal(chunk.topk_output.topk_weights, topk.topk_weights[1:4])
    assert chunk.hidden_states_pre_quant is None


def test_slice_standard_dispatch_output_preserves_scalar_scale():
    topk = StandardTopKOutput(
        topk_weights=torch.ones(4, 2),
        topk_ids=torch.zeros(4, 2, dtype=torch.int32),
        router_logits=torch.ones(4, 3),
    )
    scale = torch.ones(())
    dispatch = StandardDispatchOutput(torch.ones(4, 8), scale, topk)

    chunk = _slice_standard_dispatch_output(dispatch, topk, 0, 2)

    assert chunk.hidden_states_scale is scale
