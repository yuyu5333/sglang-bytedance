import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillCandidateBlocks,
    candidate_block_mask,
    select_candidate_block_ids,
    select_candidate_blocks,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrefillCandidateBlocks(CustomTestCase):
    def test_causal_partial_blocks_and_forced_newest_block(self):
        scores = torch.tensor([[100.0] * 8 + [50.0] * 8 + [-10.0] * 3] * 4)
        lengths = torch.tensor([[0], [1], [9], [19]])
        scores.masked_fill_(torch.arange(19)[None, :] >= lengths, -torch.inf)
        original_scores = scores.clone()
        expected = torch.tensor(
            [
                [False] * 19,
                [True] * 8 + [False] * 11,
                [True] * 16 + [False] * 3,
                [True] * 8 + [False] * 8 + [True] * 3,
            ]
        )
        blocks = select_candidate_block_ids(
            logits=scores, compress_lens=lengths, topk_blocks=2, block_size=8
        )
        self.assertEqual(blocks.dtype, torch.int32)
        self.assertEqual(tuple(blocks.shape), (4, 2))
        torch.testing.assert_close(blocks[0], torch.tensor([-1, -1], dtype=torch.int32))
        torch.testing.assert_close(
            candidate_block_mask(blocks=blocks, width=19, block_size=8), expected
        )
        torch.testing.assert_close(
            select_candidate_blocks(
                logits=scores, compress_lens=lengths, topk_blocks=2, block_size=8
            ),
            expected,
        )
        torch.testing.assert_close(scores, original_scores)

    def test_underfilled_and_empty_candidates(self):
        for width in (0, 1, 7, 8, 9):
            with self.subTest(width=width):
                scores = torch.zeros((2, width))
                blocks = select_candidate_block_ids(
                    logits=scores, compress_lens=width, topk_blocks=2048, block_size=8
                )
                torch.testing.assert_close(
                    candidate_block_mask(blocks=blocks, width=width, block_size=8),
                    torch.ones_like(scores, dtype=torch.bool),
                )
        blocks = torch.full((3, 2), -1, dtype=torch.int32)
        self.assertFalse(
            candidate_block_mask(blocks=blocks, width=19, block_size=8).any()
        )

    def test_replay_tail_keeps_request_boundaries_and_empty_tails(self):
        requests = [torch.arange(n * 2).reshape(n, 2) for n in (5, 0, 3)]
        candidates = PrefillCandidateBlocks(request_blocks=requests)
        tail = candidates.tail([2, 0, 0])
        self.assertEqual(
            [tuple(b.shape) for b in tail.request_blocks], [(2, 2), (0, 2), (0, 2)]
        )
        torch.testing.assert_close(tail.request_blocks[0], requests[0][3:])
        self.assertEqual(tail.request_blocks[0].data_ptr(), requests[0][3:].data_ptr())
        self.assertEqual([b.shape[0] for b in candidates.request_blocks], [5, 0, 3])

    def test_nonfinite_blocks_match_mask_selection(self):
        logits = torch.tensor([[float("nan")] * 8 + [1.0] * 8 + [-torch.inf] * 8])
        kwargs = dict(logits=logits, compress_lens=24, topk_blocks=3, block_size=8)
        blocks = select_candidate_block_ids(**kwargs)
        torch.testing.assert_close(
            candidate_block_mask(blocks=blocks, width=24, block_size=8),
            select_candidate_blocks(**kwargs),
        )

    def test_deepselect_receives_reduced_block_scores(self):
        calls = []

        def fake_deepselect(scores, topk, *, indices_type):
            calls.append((scores.clone(), topk, indices_type))
            top = torch.topk(scores, topk, dim=-1, sorted=False)
            return top.values, top.indices.to(indices_type)

        logits = torch.arange(48, dtype=torch.float32).reshape(2, 24)
        lengths = torch.tensor([[17], [24]], dtype=torch.int32)
        with (
            patch(
                "sglang.srt.layers.attention.dsv4.candidate_indexer."
                "_get_deepselect_topk",
                return_value=fake_deepselect,
            ),
            patch(
                "sglang.srt.layers.attention.dsv4.candidate_indexer."
                "_can_fuse_candidate_blocks",
                return_value=False,
            ),
        ):
            blocks = select_candidate_block_ids(logits, lengths, 2, 8)

        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][0].shape), (2, 3))
        self.assertEqual(calls[0][1:], (2, torch.int32))
        self.assertEqual(blocks.dtype, torch.int32)

    def test_deepselect_routes_through_fused_reduction_and_publication(self):
        calls = []
        logits = torch.arange(48, dtype=torch.float32).reshape(2, 24)
        lengths = torch.tensor([[17], [24]], dtype=torch.int32)
        expected = select_candidate_blocks(logits, lengths, 2, 8)

        def fake_scores(input_logits, seq_lens, block_size):
            calls.append(("scores", input_logits.shape, seq_lens.shape, block_size))
            scores = input_logits.unflatten(-1, (-1, block_size)).amax(dim=-1)
            last = (seq_lens[:, None] - 1) // block_size
            return scores.masked_fill(
                torch.arange(scores.shape[1]) == last, torch.inf
            )

        def fake_deepselect(scores, topk, *, indices_type):
            calls.append(("topk", scores.shape, topk, indices_type))
            top = scores.topk(topk, dim=-1, sorted=False)
            return top.values, top.indices.to(indices_type)

        def fake_publish(indices, values, width, block_size):
            calls.append(("publish", indices.shape, values.shape, width, block_size))
            blocks = indices.masked_fill(~(values > -torch.inf), -1)
            return candidate_block_mask(blocks, width, block_size)

        with (
            patch(
                "sglang.srt.layers.attention.dsv4.candidate_indexer."
                "_get_deepselect_topk",
                return_value=fake_deepselect,
            ),
            patch(
                "sglang.srt.layers.attention.dsv4.candidate_indexer."
                "_can_fuse_candidate_blocks",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.candidate_blocks."
                "candidate_block_scores",
                side_effect=fake_scores,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.candidate_blocks."
                "publish_candidate_block_mask",
                side_effect=fake_publish,
            ),
        ):
            actual = select_candidate_blocks(logits, lengths, 2, 8)

        torch.testing.assert_close(actual, expected)
        self.assertEqual([call[0] for call in calls], ["scores", "topk", "publish"])

    def test_deepselect_falls_back_when_unavailable(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(1, 24)
        kwargs = dict(logits=logits, compress_lens=24, topk_blocks=2, block_size=8)
        expected = select_candidate_block_ids(**kwargs)
        with patch(
            "sglang.srt.layers.attention.dsv4.candidate_indexer."
            "_get_deepselect_topk",
            return_value=None,
        ):
            actual = select_candidate_block_ids(**kwargs)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
