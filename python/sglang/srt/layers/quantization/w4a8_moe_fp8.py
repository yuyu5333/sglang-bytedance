from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod


class W4A8MoEFp8Config(QuantizationConfig):
    def __init__(
        self,
        moe_activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_moe_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4A8MoEFp8Config":
        moe_activation_scheme = "static"
        group_size = int(cls.get_from_keys(config, ["group_size"], default=128))
        return cls(moe_activation_scheme=moe_activation_scheme, group_size=group_size)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return UnquantizedLinearMethod()
        if isinstance(layer, FusedMoE):
            return W4AFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

