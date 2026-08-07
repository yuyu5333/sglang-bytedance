import unittest
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_config import (
    normalize_dspark_draft_hf_config,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    _draft_seq_lens_host_bound,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DSparkPPVerifyInputRaw,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    _resolve_dspark_draft_parallel_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )

    def test_pipeline_parallel_preserves_cuda_graph(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.pp_size = 4
        server_args.disable_cuda_graph = False
        _handle_dspark(server_args)
        self.assertFalse(server_args.disable_cuda_graph)


class TestLegacySpeculatorsDsparkConfig(CustomTestCase):
    def test_legacy_anchor_export_is_normalized_to_sglang_gamma(self):
        config = SimpleNamespace(
            architectures=["DSparkDraftModel"],
            aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
            block_size=8,
            draft_vocab_size=154880,
            dspark_bonus_anchor=False,
            speculators_config={
                "proposal_methods": [{"speculative_tokens": 7}]
            },
            transformer_layer_config={
                "model_type": "qwen3",
                "hidden_size": 6144,
                "num_hidden_layers": 5,
            },
        )

        normalize_dspark_draft_hf_config(config)
        parsed = parse_dspark_draft_config(draft_hf_config=config)

        self.assertEqual(config.hidden_size, 6144)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(config.num_target_layers, 71)
        self.assertEqual(config.dspark_block_size, 7)
        self.assertEqual(parsed.gamma, 7)


class TestDsparkPPTensorRelay(CustomTestCase):
    def test_draft_is_replicated_across_target_attention_cp(self):
        target_ps = ParallelState.trivial(
            tp_rank=1,
            tp_size=2,
            pp_rank=3,
            pp_size=4,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_cp_rank=1,
            attn_cp_size=2,
        )

        draft_ps = _resolve_dspark_draft_parallel_state(target_ps)

        self.assertEqual(draft_ps.pp_rank, 0)
        self.assertEqual(draft_ps.tp_rank, 0)
        self.assertEqual(draft_ps.tp_size, 1)
        self.assertEqual(draft_ps.attn_tp_rank, 0)
        self.assertEqual(draft_ps.attn_tp_size, 1)
        self.assertEqual(draft_ps.attn_cp_rank, 0)
        self.assertEqual(draft_ps.attn_cp_size, 1)

    def test_draft_host_bound_does_not_read_device_prefix_lens(self):
        committed = torch.tensor([100, 200], dtype=torch.int64)
        host_bound, host_bound_sum = _draft_seq_lens_host_bound(committed, gamma=4)

        self.assertEqual(host_bound.tolist(), [110, 210])
        self.assertEqual(host_bound_sum, 320)

    def test_tensor_fields_are_flattened_for_device_transport(self):
        raw = DSparkPPVerifyInputRaw(
            bonus_tokens=torch.tensor([10, 20]),
            draft_tokens=torch.tensor([[11, 12], [21, 22]]),
            new_seq_lens=torch.tensor([7, 9]),
            confidence=torch.tensor([0.5, 0.75]),
            accept_lens=torch.tensor([2, 3], dtype=torch.int32),
            cap_trim_lens=torch.tensor([0, 1], dtype=torch.int32),
            verify_lens=torch.tensor([3, 3], dtype=torch.int32),
        )

        payload = raw.to_tensor_dict()
        self.assertNotIn("pp_spec_output", payload)
        self.assertTrue(DSparkPPVerifyInputRaw.has_pp_outputs(payload))
        rebuilt = DSparkPPVerifyInputRaw.from_pp_outputs(payload)

        self.assertIs(rebuilt.bonus_tokens, raw.bonus_tokens)
        self.assertIs(rebuilt.draft_tokens, raw.draft_tokens)
        self.assertIs(rebuilt.accept_lens, raw.accept_lens)

    def test_dummy_and_batch_ops_stay_in_tensor_form(self):
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(), SimpleNamespace()],
            input_ids=torch.tensor([101, 202]),
            seq_lens=torch.tensor([8, 13]),
        )
        raw = DSparkPPVerifyInputRaw.build_dummy_for_decode(batch, num_draft=5)

        self.assertEqual(raw.draft_tokens.shape, (2, 4))
        self.assertEqual(raw.draft_tokens.tolist(), [[101] * 4, [202] * 4])
        self.assertTrue(torch.is_tensor(raw.accept_lens))

        raw.filter_batch(torch.tensor([1]))
        self.assertEqual(raw.bonus_tokens.tolist(), [202])

        other = DSparkPPVerifyInputRaw.build_dummy_for_decode(
            SimpleNamespace(
                reqs=[SimpleNamespace()],
                input_ids=torch.tensor([303]),
                seq_lens=torch.tensor([21]),
            ),
            num_draft=5,
        )
        raw.merge_batch(other)
        self.assertEqual(raw.bonus_tokens.tolist(), [202, 303])
        self.assertEqual(raw.draft_tokens.shape, (2, 4))

    def test_merge_drops_partial_whole_batch_optional_fields(self):
        running = DSparkPPVerifyInputRaw(
            bonus_tokens=torch.tensor([10, 20]),
            draft_tokens=torch.tensor([[11, 12], [21, 22]]),
            new_seq_lens=torch.tensor([7, 9]),
            accept_lens=torch.tensor([2, 3], dtype=torch.int32),
            confidence=None,
            cap_trim_lens=torch.tensor([0, 1], dtype=torch.int32),
            verify_lens=None,
        )
        newly_prefilled = DSparkPPVerifyInputRaw.build_dummy_for_decode(
            SimpleNamespace(
                reqs=[SimpleNamespace()],
                input_ids=torch.tensor([303]),
                seq_lens=torch.tensor([21]),
            ),
            num_draft=3,
        )

        running.merge_batch(newly_prefilled)

        self.assertEqual(running.bonus_tokens.tolist(), [10, 20, 303])
        self.assertEqual(running.draft_tokens.shape, (3, 2))
        self.assertIsNone(running.confidence)
        self.assertIsNone(running.verify_lens)
        running.filter_batch(
            torch.tensor([0, 2]),
            has_been_filtered=False,
            new_indices_cpu=[0, 2],
        )
        self.assertEqual(running.bonus_tokens.tolist(), [10, 303])


if __name__ == "__main__":
    unittest.main()
