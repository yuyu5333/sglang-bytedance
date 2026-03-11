from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import safetensors
import safetensors.torch
import torch


FP8_ABS_MAX = 448.0


@dataclass(frozen=True)
class LayerSpec:
    layer_prefix: str
    num_experts: int


def _load_text_config(model_dir: str) -> dict:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with config_path.open("r") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config") or cfg
    if not isinstance(text_cfg, dict):
        raise ValueError("Invalid config.json: text_config is not a dict")
    return text_cfg


def _discover_layer_prefixes(weight_map: Dict[str, str]) -> List[str]:
    prefixes: set[str] = set()
    pat = re.compile(r"^(?P<prefix>.+\.model\.layers\.\d+)\.")
    for k in weight_map.keys():
        m = pat.match(k)
        if m:
            prefixes.add(m.group("prefix"))
    if prefixes:
        return sorted(prefixes)
    pat2 = re.compile(r"^(?P<prefix>model\.layers\.\d+)\.")
    for k in weight_map.keys():
        m = pat2.match(k)
        if m:
            prefixes.add(m.group("prefix"))
    return sorted(prefixes)


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


def _build_index(model_dir: str) -> Dict[str, str]:
    weight_map: Dict[str, str] = {}
    for st in sorted(Path(model_dir).glob("*.safetensors")):
        filename = st.name
        with safetensors.safe_open(str(st), framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = filename
    return weight_map


def _write_input_scales(
    model_dir: str,
    save_dir: str,
    layer_specs: List[LayerSpec],
    a1_scales: Dict[str, float],
    a2_scales: Dict[str, float],
    out_safetensors_name: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    weight_map, index_name = _load_index(save_dir)
    if weight_map is None:
        weight_map, _ = _load_index(model_dir)
    if weight_map is None:
        weight_map = _build_index(save_dir)
        index_name = "model.safetensors.index.json"
    if index_name is None:
        index_name = "model.safetensors.index.json"

    out: Dict[str, torch.Tensor] = {}
    for spec in layer_specs:
        a1 = float(a1_scales[spec.layer_prefix])
        a2 = float(a2_scales[spec.layer_prefix])
        for e in range(spec.num_experts):
            base = f"{spec.layer_prefix}.mlp.experts.{e}"
            out[f"{base}.w1.input_scale"] = torch.tensor(a1, dtype=torch.float32)
            out[f"{base}.w3.input_scale"] = torch.tensor(a1, dtype=torch.float32)
            out[f"{base}.w2.input_scale"] = torch.tensor(a2, dtype=torch.float32)

    out_path = os.path.join(save_dir, out_safetensors_name)
    safetensors.torch.save_file(out, out_path, metadata={"format": "pt"})

    out_filename = os.path.basename(out_path)
    for k in out.keys():
        weight_map[k] = out_filename

    out_index_path = os.path.join(save_dir, index_name)
    with open(out_index_path, "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)


def _heuristic_scales(layer_specs: List[LayerSpec]) -> Tuple[Dict[str, float], Dict[str, float]]:
    a1: Dict[str, float] = {}
    a2: Dict[str, float] = {}
    for spec in layer_specs:
        a1[spec.layer_prefix] = 1.0
        a2[spec.layer_prefix] = 1.0
    return a1, a2


def _activation_dump_scales(
    activation_json: str, layer_specs: List[LayerSpec]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(activation_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("activation_json must be a dict")

    a1: Dict[str, float] = {}
    a2: Dict[str, float] = {}
    for spec in layer_specs:
        layer_data = data.get(spec.layer_prefix)
        if not isinstance(layer_data, dict):
            raise ValueError(f"Missing layer entry for {spec.layer_prefix}")
        x_max = float(layer_data.get("a1_max_abs"))
        inter_max = float(layer_data.get("a2_max_abs"))
        a1[spec.layer_prefix] = x_max / FP8_ABS_MAX
        a2[spec.layer_prefix] = inter_max / FP8_ABS_MAX
    return a1, a2


def calibrate(
    model_dir: str,
    save_dir: str,
    mode: str,
    activation_json: str | None,
    out_safetensors_name: str,
) -> None:
    text_cfg = _load_text_config(model_dir)
    num_layers = int(text_cfg["num_hidden_layers"])
    num_experts = int(text_cfg.get("n_routed_experts", text_cfg.get("num_experts", 0)))
    if num_experts <= 0:
        raise ValueError("Cannot infer n_routed_experts from config")

    weight_map, _ = _load_index(model_dir)
    if weight_map is None:
        weight_map = _build_index(model_dir)
    prefixes = _discover_layer_prefixes(weight_map)
    if prefixes:
        layer_prefixes = prefixes
    else:
        layer_prefixes = [f"model.layers.{i}" for i in range(num_layers)]

    layer_specs = [LayerSpec(layer_prefix=p, num_experts=num_experts) for p in layer_prefixes]

    if mode == "activation_json":
        if activation_json is None:
            raise ValueError("--activation-json is required for activation_json mode")
        a1_scales, a2_scales = _activation_dump_scales(activation_json, layer_specs)
    elif mode == "heuristic":
        a1_scales, a2_scales = _heuristic_scales(layer_specs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _write_input_scales(
        model_dir=model_dir,
        save_dir=save_dir,
        layer_specs=layer_specs,
        a1_scales=a1_scales,
        a2_scales=a2_scales,
        out_safetensors_name=out_safetensors_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "activation_json"],
    )
    parser.add_argument("--activation-json", type=str, default=None)
    parser.add_argument(
        "--out-safetensors-name", type=str, default="moe-input-scales.safetensors"
    )
    args = parser.parse_args()
    calibrate(
        model_dir=args.model_dir,
        save_dir=args.save_dir,
        mode=args.mode,
        activation_json=args.activation_json,
        out_safetensors_name=args.out_safetensors_name,
    )


if __name__ == "__main__":
    main()

