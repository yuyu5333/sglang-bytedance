from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.tp_worker import TpModelWorker
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

    stash_pdmux_chunked_request(scheduler)

    scheduler.stash_chunked_request.assert_called_once_with(scheduler.chunked_req)


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
