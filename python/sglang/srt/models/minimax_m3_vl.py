# SPDX-License-Identifier: Apache-2.0
# MiniMax M3 VL — vision tower + M3 (mixed sparse/dense MoE) text backbone.

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.minimax_m3 import (
    MiniMaxM3Model,
    MiniMaxM3SparseForCausalLM,
    build_minimax_fused_qkv_index,
    get_spec_layer_idx_from_weight_name,
)
from sglang.srt.models.minimax_vl_common import (
    CLIPVisionConfig,
    MiniMaxVLVisionModel,
    get_image_feature,
    get_video_feature,
    load_vision_weight,
    merge_vit_qkv_weights,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_device_sm, is_cuda, log_info_on_rank0
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_device_sm = get_device_sm()


class MiniMaxM3SparseForConditionalGeneration(nn.Module):
    """MiniMax M3 VL: shared vision tower + M3 LLM with mixed sparse/dense attention.

    Always loaded as the mixed sparse/dense backbone: which layers are sparse
    vs dense is decided by ``config.text_config.sparse_attention_config``. A
    checkpoint that omits ``sparse_attention_config`` will produce a pure-dense
    model.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        self.num_fused_shared_experts = 0
        self._determine_num_fused_shared_experts()

        vision_config_raw = config.vision_config
        assert vision_config_raw is not None, "vision_config is required"
        if hasattr(vision_config_raw, "to_dict"):
            vision_config_dict = vision_config_raw.to_dict()
        else:
            vision_config_dict = vision_config_raw
        vision_config = CLIPVisionConfig.from_dict(vision_config_dict)
        self.vision_config = vision_config

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        # Vision model skips quantization: CLIP dimensions (head_dim=80) are not
        # compatible with MXFP8 kernel alignment requirements (128).
        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=None,
            prefix=add_prefix("vision_tower", prefix),
        )

        # Language model: M3 (with optional sparse attention).
        # The unified MiniMaxM3Model reads ``text_config.sparse_attention_config``
        # to decide per-layer whether to construct dense or sparse attention,
        # so no branching is needed here.
        text_config = config.text_config
        self.model = MiniMaxM3Model(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model.model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("language_model.lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()

        _, text_rope_scaling = get_rope_config(text_config)
        self.is_mrope_enabled = (
            text_rope_scaling is not None and "mrope_section" in text_rope_scaling
        )

        self.logits_processor = LogitsProcessor(text_config)

    def _determine_num_fused_shared_experts(self) -> None:
        text_config = self.config.text_config
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if not getattr(text_config, "n_shared_experts", None):
            disable_reason = "No shared experts are defined in the config."
        elif not _is_cuda:
            disable_reason = "Shared experts fusion currently requires CUDA devices."
        elif _is_cuda and (_device_sm is not None) and (_device_sm < 80):
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = (
                "Shared experts fusion is not supported together with expert "
                "parallelism yet."
            )
        elif get_moe_a2a_backend().is_deepep():
            disable_reason = (
                "Shared experts fusion is not supported when Deepep MoE backend "
                "is enabled."
            )
        elif self.quant_config is not None and any(
            "shared_experts" in pat
            for pat in (getattr(self.quant_config, "ignore", []) or [])
        ):
            # Mixed-precision ckpt (e.g. W4A16 with bf16 shared_experts):
            # routed experts are quantized into the fused-MoE packed grid,
            # but shared_experts are stored as unquantized bf16 raw weight.
            # Fusing them into the quantized expert-128 slot would silently
            # drop the bf16 weight (no matching `experts.w*_weight` param
            # exists; only `_weight_packed/_scale/_shape` do), leaving
            # expert 128 at its torch.ones() init and corrupting every
            # token. Force-disable fusion so shared_experts goes through the
            # independent MiniMaxM3MLP path (UnquantizedLinearMethod + bf16).
            disable_reason = (
                "Quant config ignores shared_experts (ckpt stores them as "
                "unquantized bf16), which cannot be fused into the quantized "
                "expert grid."
            )

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = text_config.n_shared_experts
        assert (
            self.num_fused_shared_experts == 1
        ), "Only 1 fused shared expert is supported"
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        # EP looks up this hook on the top-level arch class to build expert-location
        # metadata (else ExpertLocationDispatchInfo.init_new asserts). The VL config
        # nests the LM config under text_config, so delegate there; fall back to
        # config itself when text_config is absent (LM config passed directly).
        text_config = getattr(config, "text_config", None) or config
        return MiniMaxM3SparseForCausalLM.get_model_config_for_expert_location(
            text_config
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return get_image_feature(self.vision_tower, items, self.use_data_parallel)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return get_video_feature(self.vision_tower, items, self.use_data_parallel)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank and not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load checkpoint weights for the vision tower and the M3 LLM.

        M3 LLM differs from M2 in:
        - MoE path is ``mlp.experts.*`` (not ``block_sparse_moe.experts.*``);
          checkpoints saved with the M2 naming are remapped on the fly.
        - Optional shared experts fusion: ``mlp.shared_experts`` is mapped onto
          a synthetic ``mlp.experts.{num_local_experts}`` slot.
        - PP layer skipping via ``get_layer_id``.
        - MTP / spec-decode layers are skipped.
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        # [DEBUG W4A16] Tracking counters/sets for end-of-load summary. These
        # are passed by reference (closure) into _load_llm_weight via attrs.
        self._dbg_expert_loaded = 0
        self._dbg_expert_not_in_params = 0
        self._dbg_expert_fell_through = 0
        self._dbg_stacked_loaded = 0
        self._dbg_stacked_not_in_params = 0
        self._dbg_default_loaded = 0
        self._dbg_default_missing = 0
        self._dbg_seen_expert_suffixes: set = set()
        self._dbg_shared_ckpt_names: set = set()
        self._dbg_shared_loaded_targets: set = set()
        self._dbg_shared_missed: list = []
        self._dbg_dense_ckpt_names: set = set()
        self._dbg_dense_loaded_targets: set = set()
        self._dbg_dense_missed: list = []
        # [DEBUG W4A16-VL2] Capture every ckpt key that mentions "shared" OR
        # is a top-level "experts.<id>." pattern with id >= num_local_experts
        # (i.e. >= 128). Also collect a "template" of every ckpt key (with
        # layer ids and expert ids replaced by "*") to find any unfamiliar
        # naming hiding the shared-expert weights.
        self._dbg_total_ckpt_keys = 0
        self._dbg_any_shared_keys: list = []
        self._dbg_high_expert_id_keys: list = []
        self._dbg_key_templates: set = set()

        # ``.qkv_proj`` (with the leading dot) prevents matching e.g.
        # ``index_q_proj`` in the sparse-attention branch.
        llm_stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Mirror the LLM's fused index projection (see MiniMaxM3.load_weights):
        # restack the separate index_q/k/v projections into one index_qkv_proj.
        # The leading "." makes these match only the index_*_proj weights.
        if (
            getattr(self.config.text_config, "sparse_attention_config", None)
            is not None
        ):
            llm_stacked_params_mapping += [
                (".index_qkv_proj", ".index_q_proj", "q"),
                (".index_qkv_proj", ".index_k_proj", "k"),
                (".index_qkv_proj", ".index_v_proj", "v"),
            ]

        num_experts = getattr(self.config.text_config, "num_local_experts", 0)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=num_experts + self.num_fused_shared_experts,
            )
            if num_experts > 0
            else []
        )

        params_dict = dict(self.named_parameters())
        vit_qkv_weights: dict = {}
        vit_qkv_biases: dict = {}

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # [DEBUG W4A16-VL2] capture EVERY raw ckpt key (before any rename)
            # so we can see how shared experts are named in the W4A16 ckpt.
            self._dbg_total_ckpt_keys += 1
            if "shared" in name:
                if len(self._dbg_any_shared_keys) < 64:
                    self._dbg_any_shared_keys.append(name)
            # Detect any "experts.<id>." with id >= num_experts (would be a
            # baked-in shared expert slot in the ckpt itself).
            import re as _re

            m = _re.search(r"experts\.(\d+)\.", name)
            if m and num_experts > 0 and int(m.group(1)) >= num_experts:
                if len(self._dbg_high_expert_id_keys) < 64:
                    self._dbg_high_expert_id_keys.append(name)
            # Build a "template" key: replace `.layers.N.` with `.layers.*.`
            # and `experts.N.` with `experts.*.` so we get a small set.
            tmpl = _re.sub(r"\.layers\.\d+\.", ".layers.*.", name)
            tmpl = _re.sub(r"experts\.\d+\.", "experts.*.", tmpl)
            self._dbg_key_templates.add(tmpl)

            if name.startswith("language_model."):
                self._load_llm_weight(
                    name[len("language_model.") :],
                    loaded_weight,
                    params_dict,
                    llm_stacked_params_mapping,
                    expert_params_mapping,
                )
                continue

            load_vision_weight(
                name, loaded_weight, params_dict, vit_qkv_weights, vit_qkv_biases
            )

        merge_vit_qkv_weights(vit_qkv_weights, vit_qkv_biases, params_dict)

        # [DEBUG W4A16] one-shot summary so we can see, in the *real* loader
        # path used by MiniMaxM3SparseForConditionalGeneration, where every
        # checkpoint key was dispatched.
        logger.warning(
            "[DEBUG W4A16-VL] load_weights summary: "
            "num_fused_shared_experts=%d "
            "expert_loaded=%d expert_fell_through=%d expert_not_in_params=%d "
            "stacked_loaded=%d stacked_not_in_params=%d "
            "default_loaded=%d default_missing=%d "
            "unique_expert_ckpt_suffixes=%d "
            "shared_ckpt_seen=%d shared_loaded_targets=%d shared_missed=%d "
            "dense_ckpt_seen=%d dense_loaded_targets=%d dense_missed=%d",
            self.num_fused_shared_experts,
            self._dbg_expert_loaded,
            self._dbg_expert_fell_through,
            self._dbg_expert_not_in_params,
            self._dbg_stacked_loaded,
            self._dbg_stacked_not_in_params,
            self._dbg_default_loaded,
            self._dbg_default_missing,
            len(self._dbg_seen_expert_suffixes),
            len(self._dbg_shared_ckpt_names),
            len(self._dbg_shared_loaded_targets),
            len(self._dbg_shared_missed),
            len(self._dbg_dense_ckpt_names),
            len(self._dbg_dense_loaded_targets),
            len(self._dbg_dense_missed),
        )
        for s in sorted(self._dbg_seen_expert_suffixes)[:32]:
            logger.warning("[DEBUG W4A16-VL] expert ckpt suffix sample: %s", s)
        for s in sorted(self._dbg_shared_ckpt_names):
            logger.warning("[DEBUG W4A16-VL] shared_experts ckpt name: %s", s)
        for s in sorted(self._dbg_shared_loaded_targets):
            logger.warning("[DEBUG W4A16-VL] shared_experts loaded target: %s", s)
        for s in self._dbg_shared_missed[:32]:
            logger.warning("[DEBUG W4A16-VL] shared_experts MISSED: %s", s)
        for s in sorted(self._dbg_dense_ckpt_names):
            logger.warning("[DEBUG W4A16-VL] dense ckpt name: %s", s)
        for s in sorted(self._dbg_dense_loaded_targets):
            logger.warning("[DEBUG W4A16-VL] dense loaded target: %s", s)
        for s in self._dbg_dense_missed[:32]:
            logger.warning("[DEBUG W4A16-VL] dense MISSED: %s", s)

        # [DEBUG W4A16-VL2] dump every "shared" key, every "experts.>=128" key,
        # total ckpt key count, and a sorted list of all key templates so we
        # can pinpoint how (or whether) the W4A16 ckpt actually stores the
        # shared-expert weights.
        logger.warning(
            "[DEBUG W4A16-VL2] total_ckpt_keys=%d num_shared_keys=%d "
            "num_high_expert_id_keys=%d num_key_templates=%d",
            self._dbg_total_ckpt_keys,
            len(self._dbg_any_shared_keys),
            len(self._dbg_high_expert_id_keys),
            len(self._dbg_key_templates),
        )
        for s in self._dbg_any_shared_keys:
            logger.warning("[DEBUG W4A16-VL2] ckpt key with 'shared': %s", s)
        for s in self._dbg_high_expert_id_keys:
            logger.warning(
                "[DEBUG W4A16-VL2] ckpt key with high expert id (>=num_local_experts): %s",
                s,
            )
        for s in sorted(self._dbg_key_templates):
            logger.warning("[DEBUG W4A16-VL2] ckpt key template: %s", s)

        # Fuse main qkv_proj + sparse index_qkv_proj into one GEMM per sparse
        # attention layer (see MiniMaxM3.load_weights for the rationale).
        build_minimax_fused_qkv_index(self)

    def _load_llm_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        llm_stacked_params_mapping: list,
        expert_params_mapping: list,
    ) -> None:
        # [DEBUG W4A16-VL] capture original ckpt key for shared_experts and
        # dense layers 0/1/2 BEFORE any rename so we can see exactly what the
        # checkpoint stores (.weight vs .weight_packed/.weight_scale).
        if "mlp.shared_experts" in name:
            self._dbg_shared_ckpt_names.add(name)
        if any(f".layers.{i}." in name for i in (0, 1, 2)) and (
            ".mlp." in name and "experts." not in name
        ):
            self._dbg_dense_ckpt_names.add(name)

        # Older checkpoints used the M2-style ``block_sparse_moe`` naming.
        if "block_sparse_moe" in name:
            name = name.replace("block_sparse_moe", "mlp")

        layer_id = get_layer_id(name)
        if layer_id is not None and (
            layer_id < self.model.start_layer or layer_id >= self.model.end_layer
        ):
            return

        # [DEBUG W4A16-VL] track that the ckpt key reached dispatch (i.e. not
        # filtered out as out-of-range PP layer).
        is_shared_orig = "mlp.shared_experts" in name
        is_dense_orig = any(
            f".layers.{i}." in name for i in (0, 1, 2)
        ) and (".mlp." in name and "experts." not in name)

        if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
            name = name.replace(
                "mlp.shared_experts",
                f"mlp.experts.{self.config.text_config.num_local_experts}",
            )
            name = name.replace("gate_proj", "w1")
            name = name.replace("down_proj", "w2")
            name = name.replace("up_proj", "w3")

        if (
            get_spec_layer_idx_from_weight_name(self.config.text_config, name)
            is not None
        ):
            return

        # [DEBUG W4A16-VL] record any "experts." key (post-rename) so we know
        # what suffixes the ckpt actually delivers (e.g. ".weight_packed",
        # ".weight_scale", ".weight").
        if "experts." in name:
            tail = name[name.rfind("experts.") :]
            self._dbg_seen_expert_suffixes.add(tail)

        for param_name, weight_name, shard_id in llm_stacked_params_mapping:
            if weight_name not in name:
                continue
            if "mlp.experts." in name:
                # Experts are handled by expert_params_mapping below.
                continue
            new_name = name.replace(weight_name, param_name)
            if new_name.endswith(".bias") and new_name not in params_dict:
                continue
            if new_name not in params_dict:
                self._dbg_stacked_not_in_params += 1
                if is_shared_orig:
                    self._dbg_shared_missed.append(
                        f"stacked-not-in-params orig={name} target={new_name}"
                    )
                if is_dense_orig:
                    self._dbg_dense_missed.append(
                        f"stacked-not-in-params orig={name} target={new_name}"
                    )
                continue
            param = params_dict[new_name]
            param.weight_loader(param, loaded_weight, shard_id)
            self._dbg_stacked_loaded += 1
            if is_shared_orig:
                self._dbg_shared_loaded_targets.add(new_name)
            if is_dense_orig:
                self._dbg_dense_loaded_targets.add(new_name)
            return

        is_expert_weight = False
        for mapping in expert_params_mapping:
            param_name, weight_name, expert_id, shard_id = mapping
            if weight_name not in name:
                continue
            is_expert_weight = True
            new_name = name.replace(weight_name, param_name)
            if new_name not in params_dict:
                self._dbg_expert_not_in_params += 1
                continue
            param = params_dict[new_name]
            param.weight_loader(
                param,
                loaded_weight,
                new_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            self._dbg_expert_loaded += 1
            return
        if is_expert_weight:
            self._dbg_expert_fell_through += 1
            return

        if name.endswith(".bias") and name not in params_dict:
            return
        remapped = maybe_remap_kv_scale_name(name, params_dict)
        if remapped is None:
            return
        if remapped not in params_dict:
            self._dbg_default_missing += 1
            if is_shared_orig:
                self._dbg_shared_missed.append(
                    f"default-not-in-params orig={name} target={remapped}"
                )
            if is_dense_orig:
                self._dbg_dense_missed.append(
                    f"default-not-in-params orig={name} target={remapped}"
                )
            logger.warning(f"Parameter {remapped} not found in params_dict")
            return
        param = params_dict[remapped]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            weight_loader(param, loaded_weight)
            self._dbg_default_loaded += 1
            if is_shared_orig:
                self._dbg_shared_loaded_targets.add(remapped)
            if is_dense_orig:
                self._dbg_dense_loaded_targets.add(remapped)
        except Exception as e:
            logger.warning(f"Error loading weight {remapped}: {e}")


EntryClass = [MiniMaxM3SparseForConditionalGeneration]
