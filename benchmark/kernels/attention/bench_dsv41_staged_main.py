"""Bounded SM90 legacy/staged/direct A/B; no model-serving metrics.

PYTHONPATH=python python benchmark/kernels/attention/bench_dsv41_staged_main.py \
    --output /workspace/results/staged-ab.json
"""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from sgl_kernel.flash_mla import (
    flash_mla_with_kvcache,
    flash_mla_with_mixed_kvcache,
    get_mla_metadata,
)

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.packed_main_kv_staging import (
    stage_packed_main_kv,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    dequantize_dsv41_packed_main_kv,
    quantize_dsv41_packed_main_kv,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.mem_cache.dsv41_main_kv_layout import (
    PackedMainKVView,
    make_dsv41_packed_main_kv_spec,
)
from sglang.srt.mem_cache.dsv41_staging_workspace import MainKVStagingWorkspace


def v4_cache(values, page_slots):
    pages = torch.zeros(
        (values.shape[0] // page_slots, KVLayout.V4.page_bytes(page_slots)),
        dtype=torch.uint8,
        device="cuda",
    )
    fused_store_cache(
        input=values,
        cache=pages,
        indices=torch.arange(values.shape[0], device="cuda", dtype=torch.int32),
        page_size=page_slots,
        type="flashmla",
        layout=KVLayout.V4,
    )
    return pages.as_strided(
        (pages.shape[0], page_slots, 1, 584), (pages.stride(0), 584, 584, 1)
    )


def measure(fn, iterations, memory_only):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    # Include private-pool reservations: allocated bytes alone miss scratch
    # whose Python Tensor has died but whose address is still held by the graph.
    before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    resident = torch.cuda.memory_allocated() - before
    reserved = torch.cuda.memory_reserved() - reserved_before
    result = {
        "single_graph_resident_bytes": resident,
        "single_graph_reserved_bytes": reserved,
    }
    del output, graph
    gc.collect()
    if memory_only:
        return result
    # One sample, bounded work. Capture amortizes Python launch overhead.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return result | {
        "us": start.elapsed_time(end) * 1000 / iterations,
        "samples": 1,
    }


def run_case(rows, heads, page_slots, slots, topk, iterations, memory_only):
    torch.manual_seed(20260917 + rows + heads + page_slots)
    values = (
        torch.randn(
            (slots // page_slots, page_slots, 512), device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    )
    view = PackedMainKVView(
        quantize_dsv41_packed_main_kv(values),
        make_dsv41_packed_main_kv_spec(page_slots),
    )
    # Legacy sees the same packed values, round-tripped through the real writer.
    decoded = dequantize_dsv41_packed_main_kv(view.storage, page_slots)
    legacy = v4_cache(decoded.reshape(-1, 512), page_slots)
    swa = v4_cache(values.reshape(-1, 512)[:4096].contiguous(), 128)
    del values, decoded
    q = torch.randn((rows, 1, heads, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.randint(slots, (rows, topk), device="cuda", dtype=torch.int32)
    swa_ids = torch.randint(4096, (rows, 1, 128), device="cuda", dtype=torch.int32)
    lens = torch.full((rows,), topk, device="cuda", dtype=torch.int32)
    swa_lens = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
    sink = torch.linspace(-1, 1, heads, device="cuda", dtype=torch.float32)
    ws = MainKVStagingWorkspace(torch.device("cuda"), topk)
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.main_kv_staging_workspace = ws
    backend.head_dim_v = 512
    backend.softmax_scale = 512**-0.5
    legacy_meta = get_mla_metadata()[0]
    direct_meta = get_mla_metadata()[0]
    common = {
        "q": q,
        "head_dim_v": 512,
        "softmax_scale": 512**-0.5,
        "attn_sink": sink,
    }

    def attention(cache, indices, metadata):
        return flash_mla_with_kvcache(
            **common,
            k_cache=swa,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=metadata,
            is_fp8_kvcache=True,
            indices=swa_ids,
            topk_length=swa_lens,
            extra_k_cache=cache,
            extra_indices_in_kvcache=indices,
            extra_topk_length=lens,
        )[0]

    def legacy_fn():
        return attention(legacy, ids.unsqueeze(1), legacy_meta)

    def conversion():
        return stage_packed_main_kv(view, ids, lens, ws)

    def staged_attention():
        output = torch.empty_like(q)
        output.copy_(
            attention(ws.cache, ws.indices[:rows].unsqueeze(1), get_mla_metadata()[0])
        )
        return output

    def staged():
        return backend._forward_staged_main(
            q, swa, swa_ids, swa_lens, view, ids.unsqueeze(1), lens, sink
        )

    def direct():
        return flash_mla_with_mixed_kvcache(
            **common,
            swa_cache=swa,
            swa_indices=swa_ids,
            swa_topk_length=swa_lens,
            main_cache_bytes=view.storage,
            main_indices=ids.unsqueeze(1),
            main_topk_length=lens,
            swa_layout=KVLayout.V4.value,
            main_layout=view.spec.layout_id.value,
            main_page_slots=page_slots,
            main_page_bytes=view.spec.page_bytes,
            tile_scheduler_metadata=direct_meta,
        )[0]

    expected = legacy_fn()
    actual = staged()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
    direct_out = direct()
    assert torch.isfinite(direct_out).all()
    result = {
        "batch": rows,
        "heads": heads,
        "page_slots": page_slots,
        "slots": slots,
        "topk": topk,
        "legacy_bytes": legacy.untyped_storage().nbytes(),
        "packed_bytes": view.storage.untyped_storage().nbytes(),
        "workspace_bytes": ws.nbytes,
        "staged_legacy_max_abs": (actual.float() - expected.float()).abs().max().item(),
        "direct_legacy_max_abs": (direct_out.float() - expected.float())
        .abs()
        .max()
        .item(),
    }
    del actual, expected, direct_out
    fns = [
        ("legacy", legacy_fn),
        ("conversion", conversion),
        ("staged_attention", staged_attention),
        ("staged_total", staged),
        ("direct", direct),
    ]
    if (heads == 128) ^ (page_slots == 256):
        fns.reverse()
    for name, fn in fns:
        result[name] = measure(fn, iterations, memory_only)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--memory-only", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability()[0] == 9
    assert args.slots % 256 == 0 and args.topk % 64 == 0
    result = {
        "python": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "iterations": args.iterations,
        "repeat": 1,
        "memory_only": args.memory_only,
        "cases": [],
    }
    for rows in (2, 32):
        for heads in (64, 128):
            for page_slots in (128, 256):
                case = run_case(
                    rows,
                    heads,
                    page_slots,
                    args.slots,
                    args.topk,
                    args.iterations,
                    args.memory_only,
                )
                result["cases"].append(case)
                print(json.dumps(case), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
