import dataclasses
import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from sglang import Engine, ServerArgs


def _copy_metadata_files(model_path: str, output: str) -> None:
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            src = os.path.join(model_path, file)
            dst = os.path.join(output, file)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    continue
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, output)


def main() -> None:
    parser = ArgumentParser()
    ServerArgs.add_cli_args(parser)

    parser.add_argument(
        "--output", "-o", required=True, type=str, help="path to output checkpoint"
    )
    parser.add_argument(
        "--file-pattern", type=str, help="string pattern of saved filenames"
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=5 * 1024**3,
        help="max size (in bytes) of each safetensors file",
    )
    args = parser.parse_args()

    engine_args = ServerArgs.from_cli_args(args)
    model_path = engine_args.model_path
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")

    engine_args.quantization = "w4a8_moe_fp8_online"

    llm = Engine(**dataclasses.asdict(engine_args))
    Path(args.output).mkdir(exist_ok=True, parents=True)
    llm.save_sharded_model(
        path=args.output, pattern=args.file_pattern, max_size=args.max_file_size
    )

    _copy_metadata_files(model_path, args.output)

    hf_quant_config_path = os.path.join(args.output, "hf_quant_config.json")
    with open(hf_quant_config_path, "w") as f:
        json.dump(
            {
                "quant_method": "w4a8_moe_fp8",
                "group_size": 128,
                "format": "w4a8-moe-fp8",
                "note": "Exported from w4a8_moe_fp8_online (CPU conversion) into sharded_state.",
            },
            f,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()

