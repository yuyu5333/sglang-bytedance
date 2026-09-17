from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import SparsePrefillWorkspace
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_workspace_dtype_capacity_and_capture():
    ws = SparsePrefillWorkspace(torch.device("cuda"))
    bf16 = ws.get(256)
    fp8 = ws.get(64, torch.float8_e4m3fn)
    assert bf16.data_ptr() == ws.get(8).data_ptr()
    assert fp8.data_ptr() == ws.get(16, torch.float8_e4m3fn).data_ptr()
    assert bf16.data_ptr() != fp8.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ws.get(64).fill_(3)
    graph.replay()
    torch.testing.assert_close(bf16[:64], torch.full_like(bf16[:64], 3))
    # Isolate the capture guard without intentionally invalidating a live graph.
    with patch("torch.cuda.is_current_stream_capturing", return_value=True) as capturing:
        for rows, dtype in ((257, torch.bfloat16), (65, torch.float8_e4m3fn), (8, torch.float16)):
            with pytest.raises(RuntimeError, match="reserve during graph warmup"):
                ws.get(rows, dtype)
        assert capturing.call_count == 3
    grown = ws.get(512)
    assert grown.shape == (512, 1, 512)
    assert grown.data_ptr() != bf16.data_ptr()
    graph.replay()
    # An older graph keeps its allocation alive after later eager growth.
    torch.testing.assert_close(bf16[:64], torch.full_like(bf16[:64], 3))
    assert fp8.data_ptr() == ws.get(64, torch.float8_e4m3fn).data_ptr()
