from unittest.mock import Mock

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
