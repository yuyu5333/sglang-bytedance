from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.encoder_swa_replay import run_encoder_swa_replay
from sglang.srt.multiplex.multiplexing_mixin import (
    merge_completed_split_prefill,
    stash_pdmux_chunked_request,
)


def test_middle_chunk_stays_out_of_running_batch():
    chunked_req = object()
    running_batch = Mock()
    running_batch.is_empty.return_value = False
    prefill_batch = Mock()
    prefill_batch.is_empty.return_value = True

    result = merge_completed_split_prefill(
        running_batch, prefill_batch, chunked_req
    )

    prefill_batch.filter_batch.assert_called_once_with(
        chunked_req_to_exclude=chunked_req
    )
    running_batch.merge_batch.assert_not_called()
    assert result is running_batch


def test_completed_prefill_merges_into_running_batch():
    running_batch = Mock()
    running_batch.is_empty.return_value = False
    prefill_batch = Mock()
    prefill_batch.is_empty.return_value = False

    result = merge_completed_split_prefill(running_batch, prefill_batch, None)

    prefill_batch.filter_batch.assert_called_once_with(
        chunked_req_to_exclude=None
    )
    running_batch.merge_batch.assert_called_once_with(prefill_batch)
    assert result is running_batch


def test_completed_prefill_becomes_running_batch_when_decode_is_empty():
    running_batch = Mock()
    running_batch.is_empty.return_value = True
    prefill_batch = Mock()
    prefill_batch.is_empty.return_value = False

    result = merge_completed_split_prefill(running_batch, prefill_batch, None)

    assert result is prefill_batch


def test_middle_chunk_is_stashed_before_its_continuation():
    scheduler = Mock()
    scheduler.chunked_req.prefix_indices = [1, 2]
    scheduler.chunked_req.extend_range.end = 5
    scheduler.tp_worker.model_runner.token_to_kv_pool.request_window = None

    stash_pdmux_chunked_request(scheduler)

    scheduler.stash_chunked_request.assert_called_once_with(scheduler.chunked_req)


def test_request_window_middle_chunk_keeps_request_owned_kv():
    scheduler = Mock()
    scheduler.chunked_req.prefix_indices = torch.tensor([1, 2])
    scheduler.chunked_req.extend_range.end = 5
    scheduler.chunked_req.kv.req_pool_idx = 1
    scheduler.tp_worker.model_runner.token_to_kv_pool.request_window = object()
    scheduler.req_to_token_pool.req_to_token = torch.tensor(
        [[0, 0, 0, 0, 0], [10, 11, 12, 13, 14]]
    )

    stash_pdmux_chunked_request(scheduler)

    scheduler.stash_chunked_request.assert_not_called()
    assert torch.equal(
        scheduler.chunked_req.prefix_indices, torch.tensor([10, 11, 12, 13, 14])
    )


def test_encoder_swa_replay_resets_radix_hit_request():
    worker = Mock()
    window = worker.model_runner.token_to_kv_pool.request_window
    batch = Mock()
    batch.forward_mode.is_extend_without_speculative.return_value = True
    batch.reqs = [Mock()]
    batch.encoder_swa_reset = [False]
    batch.req_pool_indices = torch.tensor([3])
    batch.prefix_lens = [0]

    run_encoder_swa_replay(worker, batch)

    window.reset.assert_called_once()
    assert torch.equal(window.reset.call_args.args[0], torch.tensor([3]))


@patch("sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new")
def test_encoder_swa_replay_uses_full_extend_forward(init_forward_batch):
    worker = Mock()
    runner = worker.model_runner
    runner.device = "cpu"
    runner.req_to_token_pool.req_to_token = torch.tensor(
        [[0, 0], [0, 0], [0, 0], [10, 11]]
    )
    runner.model.model.engram_hasher = None
    req = SimpleNamespace(full_untruncated_fill_ids=[1, 2])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.SPLIT_PREFILL,
        reqs=[req],
        encoder_swa_reset=[False],
        req_pool_indices=torch.tensor([3]),
        req_pool_indices_cpu=torch.tensor([3]),
        prefix_lens=[2],
    )
    forward_batch = Mock()
    init_forward_batch.return_value = forward_batch

    run_encoder_swa_replay(worker, batch)

    replay_batch = init_forward_batch.call_args.args[0]
    assert replay_batch.forward_mode == ForwardMode.EXTEND
    runner.forward.assert_called_once_with(forward_batch)


def test_parked_chunk_without_new_kv_is_not_stashed():
    scheduler = Mock()
    scheduler.chunked_req.prefix_indices = [1, 2]
    scheduler.chunked_req.extend_range.end = 2

    stash_pdmux_chunked_request(scheduler)

    scheduler.stash_chunked_request.assert_not_called()


@patch("sglang.srt.managers.tp_worker.ForwardBatch.init_new")
@patch("sglang.srt.managers.tp_worker.get_exec")
@patch("sglang.srt.model_executor.encoder_swa_replay.run_encoder_swa_replay")
def test_split_prefill_initializes_encoder_swa_replay(
    replay, get_exec, init_forward_batch
):
    get_exec.return_value.features.enable_encoder_swa_bounded_replay = True
    forward_batch = Mock()
    init_forward_batch.return_value = forward_batch
    worker = Mock()
    worker.model_runner.forward.return_value = SimpleNamespace(
        logits_output=None,
        can_run_graph=False,
        expert_distribution_metrics=None,
    )
    batch = Mock(split_index=0, split_forward_count=1, split_prefill_finished=False)

    TpModelWorker.forward_batch_split_prefill(worker, batch)

    replay.assert_called_once_with(worker, batch)
    init_forward_batch.assert_called_once()
    worker.model_runner.forward.assert_called_once_with(
        forward_batch, split_forward_count=1
    )
