from __future__ import annotations

import argparse
import os
from pathlib import Path


def validate(path: str) -> None:
    p = Path(path)
    if not p.is_dir():
        raise SystemExit(f"Not a directory: {path}")

    files = list(p.glob("*.safetensors"))
    if not files:
        raise SystemExit("No *.safetensors found in output dir.")

    rank_files = [f for f in files if "model-rank-" in f.name]
    if not rank_files:
        raise SystemExit("No model-rank-*-part-*.safetensors found (sharded_state missing).")

    config_json = p / "config.json"
    if not config_json.is_file():
        raise SystemExit("Missing config.json in output dir.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    validate(os.path.abspath(args.path))


if __name__ == "__main__":
    main()

