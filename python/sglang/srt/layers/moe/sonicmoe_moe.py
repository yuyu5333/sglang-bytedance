import logging
from typing import Optional

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput


logger = logging.getLogger(__name__)


class SonicMoEFusedMoE(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            from sonicmoe import MoE, KernelBackendMoE
        except Exception as e:
            raise ImportError(
                "SonicMoE backend selected but 'sonicmoe' package is not available"
            ) from e

        num_experts = self.num_experts
        top_k = self.top_k if self.top_k is not None else 1
        hidden_size = self.hidden_size
        intermediate_size = (
            self.intermediate_size_per_partition * get_moe_tensor_parallel_world_size()
        )

        self._sonicmoe_backend = KernelBackendMoE.sonicmoe
        self._sonicmoe_layer = MoE(
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            is_glu=True,
            add_bias=False,
            std=0.02,
        ).to(
            device=(
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        output, _ = self._sonicmoe_layer(
            hidden_states, kernel_backend_moe=self._sonicmoe_backend
        )

        if self.reduce_results and (
            get_moe_tensor_parallel_world_size() > 1
            or get_moe_expert_parallel_world_size() > 1
        ):
            output = tensor_model_parallel_all_reduce(output)
        return output
