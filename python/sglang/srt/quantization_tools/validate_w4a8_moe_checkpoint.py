from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def _load_index(model_dir: str) -> Dict[str, str]:
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors.index.json in {model_dir}")
    with index_path.open("r") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid index: weight_map is not a dict")
    return weight_map


def _detect_prefix(weight_map: Dict[str, str]) -> str:
    candidates = [
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    ]
    for c in candidates:
        if c in weight_map:
            return c.split(".mlp.experts.0.gate_proj.weight")[0]
    for k in weight_map.keys():
        if ".model.layers.0.mlp.experts.0.gate_proj.weight" in k:
            return k.split(".mlp.experts.0.gate_proj.weight")[0]
    raise ValueError("Cannot detect model prefix from index.")


def validate(model_dir: str) -> None:
    model_dir = os.path.abspath(model_dir)
    weight_map = _load_index(model_dir)
    prefix = _detect_prefix(weight_map)

    required: List[str] = []
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        required.append(f"{prefix}.mlp.experts.0.{proj}.weight")
        required.append(f"{prefix}.mlp.experts.0.{proj}.weight_scale_inv")
    for w in ["w1", "w2", "w3"]:
        required.append(f"{prefix}.mlp.experts.0.{w}.input_scale")

    missing = [k for k in required if k not in weight_map]
    if missing:
        raise SystemExit(
            "Missing required keys:\n" + "\n".join(missing[:50])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()
    validate(args.model_dir)


if __name__ == "__main__":
    main()

