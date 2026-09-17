"""Two-process, one-shot GPU page transfer through Mooncake RDMA or NIXL UCX.

This checks real FULL-page bytes, not the model-level PD lifecycle. NIXL runs
with UCX_TLS=rc,cuda_copy to exclude TCP, shared-memory and CUDA IPC data paths.
Run in a dedicated container exposing GPU0 and /dev/infiniband.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.dsv41_main_kv_layout import make_dsv41_packed_main_kv_spec
from sglang.srt.mem_cache.kv_region_layout import (
    get_pool_transfer_info,
    match_kv_region_layouts,
    validate_kv_region_registration,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def make_pool():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=256))
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=2,
        swa_size=512,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=5,
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=[0, 2, 2, 1, 1],
        kv_source_layers=[1, 3],
        full_size=512,
        main_kv_layout_specs={
            1: make_dsv41_packed_main_kv_spec(256),
            2: make_dsv41_packed_main_kv_spec(128),
        },
    )


def pattern(size, i):
    return ((torch.arange(size, device="cuda") + 17 + i) % 256).byte()


def receive(pipe):
    if not pipe.poll(45):
        raise TimeoutError("peer did not reply within 45s")
    value = pipe.recv()
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def worker(pipe, transport, sender):
    try:
        torch.cuda.set_device(0)
        pool = make_pool()
        regions = pool.get_kv_transfer_regions()
        ptrs, sizes, items, layouts = get_pool_transfer_info(pool)
        validate_kv_region_registration(layouts, items, len(ptrs))
        for i, region in enumerate(regions):
            region.buffer.fill_(211)
            if sender:
                region.buffer[1].copy_(pattern(items[i], i))
        torch.cuda.synchronize()
        name = f"pr2-{os.getpid()}"
        if transport == "mooncake":
            from mooncake.engine import TransferEngine

            engine = TransferEngine()
            assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "mlx5_0") == 0
            assert engine.batch_register_memory(ptrs, sizes) == 0
            metadata = f"127.0.0.1:{engine.get_rpc_port()}"
        else:
            from nixl._api import nixl_agent

            engine = nixl_agent(name)
            registration = engine.register_memory(
                [(ptr, size, 0, "") for ptr, size in zip(ptrs, sizes)], "VRAM"
            )
            metadata = engine.get_agent_metadata()
        pipe.send(
            {
                "ptrs": ptrs,
                "items": items,
                "layouts": layouts,
                "metadata": metadata,
                "name": name,
            }
        )
        peer = receive(pipe)
        match_kv_region_layouts(layouts, peer["layouts"])
        validate_kv_region_registration(
            peer["layouts"], peer["items"], len(peer["ptrs"])
        )
        if sender:
            source = [ptr + size for ptr, size in zip(ptrs, items)]
            target = [ptr + 2 * size for ptr, size in zip(peer["ptrs"], items)]
            if transport == "mooncake":
                assert (
                    engine.batch_transfer_sync_write(
                        peer["metadata"], source, target, items
                    )
                    == 0
                )
            else:
                engine.add_remote_agent(peer["metadata"])
                local = engine.get_xfer_descs(
                    [(p, n, 0) for p, n in zip(source, items)], "VRAM"
                )
                remote = engine.get_xfer_descs(
                    [(p, n, 0) for p, n in zip(target, items)], "VRAM"
                )
                handle = engine.initialize_xfer("WRITE", local, remote, peer["name"])
                state = engine.transfer(handle)
                deadline = time.monotonic() + 30
                while state == "PROC" and time.monotonic() < deadline:
                    state = engine.check_xfer_state(handle)
                assert state == "DONE", state
                engine.release_xfer_handle(handle)
            pipe.send({"done": True})
            assert receive(pipe)["verified"]
        else:
            assert receive(pipe)["done"]
            torch.cuda.synchronize()
            for i, region in enumerate(regions):
                assert torch.equal(region.buffer[2], pattern(items[i], i))
                assert torch.all(region.buffer[0] == 211)
                assert torch.all(region.buffer[1] == 211)
            pipe.send({"verified": True})
            print(
                json.dumps(
                    {
                        "transport": transport,
                        "bytes": sum(items),
                        "regions": layouts,
                        "verified": True,
                        "topology": "two processes, same host/GPU, source page 1 -> destination page 2",
                    }
                ),
                flush=True,
            )
        if transport == "mooncake":
            assert engine.batch_unregister_memory(ptrs) == 0
        else:
            engine.deregister_memory(registration)
    except BaseException:
        error = traceback.format_exc()
        pipe.send({"error": error})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["mooncake", "nixl"], required=True)
    args = parser.parse_args()
    if args.transport == "nixl":
        os.environ["UCX_TLS"] = "rc,cuda_copy"
        os.environ["UCX_NET_DEVICES"] = "mlx5_0:1"
    ctx = mp.get_context("spawn")
    left, right = ctx.Pipe()
    workers = [
        ctx.Process(target=worker, args=(left, args.transport, True)),
        ctx.Process(target=worker, args=(right, args.transport, False)),
    ]
    for process in workers:
        process.start()
    deadline = time.monotonic() + 100
    try:
        for process in workers:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()
