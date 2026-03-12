from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import safetensors
import safetensors.torch
import torch


@dataclass(frozen=True)
class ExpertWeightKeys:
    base: str
    packed_key: str
    scale_key: str


def _list_safetensors_files(model_dir: str) -> List[str]:
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise ValueError(f"model_dir must be a directory, got: {model_dir}")
    return sorted(str(p) for p in model_path.glob("*.safetensors"))


def _load_index(model_dir: str) -> Tuple[Dict[str, str] | None, str | None]:
    index_name = "model.safetensors.index.json"
    index_path = Path(model_dir) / index_name
    if not index_path.is_file():
        return None, None
    with index_path.open("r") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        return None, None
    return weight_map, index_name


def _build_index(model_dir: str, safetensors_files: List[str]) -> Dict[str, str]:
    weight_map: Dict[str, str] = {}
    for st in safetensors_files:
        filename = os.path.basename(st)
        with safetensors.safe_open(st, framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = filename
    return weight_map


def _unpack_int4_from_int32(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32:
        raise ValueError(f"Expected int32 packed tensor, got {packed.dtype}")

    x = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int64)
    nibbles = (x.unsqueeze(-1) >> shifts) & 0xF
    nibbles = nibbles.to(torch.int16)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).to(torch.int8)
    out = signed.reshape(*packed.shape[:-1], packed.shape[-1] * 8)
    return out


def _pack_int4_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.dtype != torch.int8:
        int4_values_interleaved = int4_values_interleaved.to(torch.int8)
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError("Last dim must be even for int4 packing.")

    low = int4_values_interleaved[..., 0::2]
    high = int4_values_interleaved[..., 1::2]
    packed = (high << 4) | (low & 0x0F)
    return packed.to(torch.int8)


def _groupwise_int4_quantize(
    weight: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    out, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} must be divisible by group_size {group_size}"
        )
    num_groups = in_features // group_size

    w = weight.to(torch.float32).reshape(out, num_groups, group_size)
    max_abs = w.abs().amax(dim=-1).clamp(min=1e-12)
    scale = max_abs / 7.0

    q = torch.round(w / scale.unsqueeze(-1)).clamp(min=-8, max=7).to(torch.int8)
    q = q.reshape(out, in_features)
    return q, scale.to(torch.float32)


def _dequant_from_packed_and_scale(
    packed_int32: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    q_int4 = _unpack_int4_from_int32(packed_int32)
    out, in_features = q_int4.shape
    if scale.dim() != 2 or scale.shape[0] != out:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} incompatible with out={out}"
        )
    group_size = in_features // scale.shape[1]
    if in_features % scale.shape[1] != 0:
        raise ValueError(
            f"Cannot infer group_size from in_features={in_features} and scale_last={scale.shape[1]}"
        )
    scale_expanded = scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    w = q_int4.to(torch.float32) * scale_expanded
    return w.to(torch.bfloat16)


def _iter_expert_weight_keys(keys: Iterable[str]) -> Iterator[ExpertWeightKeys]:
    # Kimi format: w13_weight, w2_weight, w13_input_scale, w2_weight_scale_inv
    # Standard format: gate_proj.weight_packed, down_proj.weight_packed
    kimi_pat = re.compile(
        r"^(?P<base>.+\.mlp\.experts\.\d+\.)(w13_weight|w2_weight)$"
    )
    standard_pat = re.compile(
        r"^(?P<base>.+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj))\.weight_packed$"
    )
    for k in keys:
        m_kimi = kimi_pat.match(k)
        m_std = standard_pat.match(k)
        if m_kimi:
            base = m_kimi.group("base")
            weight_suffix = m_kimi.group(2)
            if weight_suffix == "w13":
                yield ExpertWeightKeys(
                    base=base.rstrip("."),
                    packed_key=f"{base}w13_weight",
                    scale_key=f"{base}w13_input_scale",
                )
            elif weight_suffix == "w2":
                yield ExpertWeightKeys(
                    base=base.rstrip("."),
                    packed_key=f"{base}w2_weight",
                    scale_key=f"{base}w2_weight_scale_inv",
                )
        elif m_std:
            base = m_std.group("base")
            yield ExpertWeightKeys(
                base=base, packed_key=f"{base}.weight_packed", scale_key=f"{base}.weight_scale"
            )


def _copy_non_safetensors_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            continue
        if name.endswith(".safetensors"):
            continue
        shutil.copyfile(src, dst)


def _copy_safetensors_files(src_dir: str, dst_dir: str) -> List[str]:
    src_files = _list_safetensors_files(src_dir)
    out_files: List[str] = []
    for f in src_files:
        dst = os.path.join(dst_dir, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copyfile(f, dst)
        out_files.append(dst)
    return out_files


def convert(
    model_dir: str,
    save_dir: str,
    group_size: int = 128,
    out_safetensors_name: str = "moe-w4a8-moe-fp8.safetensors",
    quant_method_name: str = "w4a8_moe_fp8",
) -> None:
    model_dir = os.path.abspath(model_dir)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    _copy_non_safetensors_files(model_dir, save_dir)
    out_safetensors_files = _copy_safetensors_files(model_dir, save_dir)

    weight_map, index_name = _load_index(model_dir)
    
    # Validate weight_map and rebuild if needed
    if weight_map is not None:
        sample_file = next(iter(weight_map.values()), None)
        if sample_file and not os.path.exists(os.path.join(model_dir, sample_file)):
            print(f"Warning: Files in weight_map not found in model_dir, rebuilding index...")
            weight_map = None
    
    if weight_map is None:
        weight_map = _build_index(model_dir, _list_safetensors_files(model_dir))
        index_name = "model.safetensors.index.json"

    converted: Dict[str, torch.Tensor] = {}

    keys_to_process = [
        k for k in weight_map.keys() 
        if k.endswith(".weight_packed") or k.endswith(".w13_weight") or k.endswith(".w2_weight")
    ]
    for w in _iter_expert_weight_keys(keys_to_process):
        packed_file = weight_map.get(w.packed_key)
        scale_file = weight_map.get(w.scale_key)
        if packed_file is None or scale_file is None or packed_file != scale_file:
            raise RuntimeError(
                f"Expected {w.packed_key} and {w.scale_key} in same shard, got {packed_file} and {scale_file}"
            )
        shard_path = os.path.join(model_dir, packed_file)
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
            packed = f.get_tensor(w.packed_key)
            scale = f.get_tensor(w.scale_key)

        dense = _dequant_from_packed_and_scale(packed, scale)
        q_int4, new_scale = _groupwise_int4_quantize(dense, group_size=group_size)
        packed_int8 = _pack_int4_to_int8(q_int4)

        converted[f"{w.base}.weight"] = packed_int8.contiguous()
        converted[f"{w.base}.weight_scale_inv"] = new_scale.contiguous()

    out_path = os.path.join(save_dir, out_safetensors_name)
    safetensors.torch.save_file(converted, out_path, metadata={"format": "pt"})

    out_filename = os.path.basename(out_path)
    for k in converted.keys():
        weight_map[k] = out_filename

    out_index_path = os.path.join(save_dir, index_name)
    index_data = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(out_index_path, "w") as f:
        json.dump(index_data, f, indent=2, sort_keys=True)

    out_quant_cfg_path = os.path.join(save_dir, "hf_quant_config.json")
    with open(out_quant_cfg_path, "w") as f:
        json.dump(
            {
                "quant_method": quant_method_name,
                "group_size": group_size,
                "format": "w4a8-moe-fp8",
            },
            f,
            indent=2,
            sort_keys=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--out-safetensors-name", type=str, default="moe-w4a8-moe-fp8.safetensors"
    )
    parser.add_argument("--quant-method-name", type=str, default="w4a8_moe_fp8")
    args = parser.parse_args()

    convert(
        model_dir=args.model_dir,
        save_dir=args.save_dir,
        group_size=args.group_size,
        out_safetensors_name=args.out_safetensors_name,
        quant_method_name=args.quant_method_name,
    )


if __name__ == "__main__":
    main()

