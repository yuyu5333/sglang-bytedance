"""Repack verified installed FlashInfer distributions, excluding runtime state."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from wheel.wheelfile import WheelFile


def snapshot(name, output):
    dist = importlib.metadata.distribution(name)
    wheel_metadata = Parser().parsestr(dist.read_text("WHEEL"))
    tags = wheel_metadata.get_all("Tag")
    if not tags or len(tags) != 1:
        raise ValueError(f"{name}: expected one wheel compatibility tag")
    filename = f"{name.replace('-', '_')}-{dist.version}-{tags[0]}.whl"
    target = output / filename
    if target.exists():
        raise FileExistsError(target)
    installed = dist.files
    record = next(f for f in installed if str(f).endswith(".dist-info/RECORD"))
    record_hash = hashlib.sha256(dist.locate_file(record).read_bytes()).hexdigest()
    paths = []
    for entry in installed:
        path = PurePosixPath(str(entry))
        if path.is_absolute() or ".." in path.parts:
            if path == PurePosixPath("../../../bin/flashinfer"):
                continue  # Recreated by pip from the wheel entry_points metadata.
            raise ValueError(f"{name}: path outside site-packages: {path}")
        if entry.hash is None:
            continue
        if path.name in ("INSTALLER", "REQUESTED", "direct_url.json", "RECORD"):
            continue
        if "__pycache__" in path.parts:
            continue
        paths.append(entry)
    try:
        with WheelFile(target, "w", compression=zipfile.ZIP_STORED) as wheel:
            for entry in paths:
                source = dist.locate_file(entry)
                with source.open("rb") as stream:
                    digest = hashlib.file_digest(stream, entry.hash.mode)
                encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
                if encoded != entry.hash.value or source.stat().st_size != entry.size:
                    raise ValueError(f"{name}: installed file differs from RECORD: {entry}")
                wheel.write(source, str(entry))
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        distribution=name,
        version=dist.version,
        installed_record_sha256=record_hash,
        file_count=len(paths),
        wheel=str(target),
        bytes=target.stat().st_size,
        sha256=digest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    records = []
    for name in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
        records.append(snapshot(name, args.output))
        print(json.dumps(records[-1]), flush=True)
        manifest.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
