from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.dsv41_request_window import RequestWindow


class _FakePool:
    def __init__(self, size, layers):
        self.size = size
        self.kv_buffer = [torch.zeros(1) for _ in range(layers)]


def test_request_window_keeps_context_per_stream():
    window = RequestWindow(
        _FakePool,
        num_slots=2,
        layers=1,
        page_size=4,
        capacity=4,
        workspace_rows=8,
    )
    stream = [11]
    window._stream_key = lambda: stream[0]
    prefill_layout = SimpleNamespace(size=8)
    decode_layout = SimpleNamespace(size=8)

    window.activate(prefill_layout)
    prefill_context = window._contexts[11]
    prefill_context.prepared = (0, False)

    stream[0] = 22
    window.activate(decode_layout)
    decode_context = window._contexts[22]

    assert decode_context is not prefill_context
    assert decode_context.workspace is not prefill_context.workspace
    assert decode_context.layout is decode_layout
    assert prefill_context.layout is prefill_layout
    assert prefill_context.prepared == (0, False)


def test_request_window_reset_invalidates_every_stream():
    window = RequestWindow(
        _FakePool,
        num_slots=2,
        layers=1,
        page_size=4,
        capacity=4,
        workspace_rows=8,
    )
    stream = [11]
    window._stream_key = lambda: stream[0]
    layout = SimpleNamespace(size=8)

    window.activate(layout)
    window._contexts[11].prepared = (0, False)
    stream[0] = 22
    window.activate(layout)
    window._contexts[22].prepared = (0, False)

    window.reset(torch.tensor([1]))

    assert window._contexts[11].prepared is None
    assert window._contexts[22].prepared is None
