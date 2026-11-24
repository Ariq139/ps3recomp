#!/usr/bin/env python3
"""Recover exact embedded P5 SPU images for import into Ghidra."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json_atomic(path: Path, data: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_game = repo_root / "p5_npeb02436_gameonly"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=default_game / "src" / "ps3" / "spu",
                        help="SPU lift folders and spu_prescanned.json")
    parser.add_argument("--output-dir", type=Path,
                        default=default_game / "ghidra_inputs" / "spu_images",
                        help="Separate folder for raw images to import into Ghidra")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_dir.resolve()
    metadata_path = source_root / "spu_prescanned.json"
    if not metadata_path.is_file():
        parser.error(f"SPU image inventory is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = {entry.get("symbol_prefix"): entry for entry in metadata
               if entry.get("symbol_prefix")}
    lift_dirs = sorted(path for path in source_root.iterdir()
                       if path.is_dir() and (path / "spu_recomp.c").is_file())
    exported = []
    for folder in lift_dirs:
        include = folder / "image_bytes.inc"
        if not include.is_file():
            continue
        values = re.findall(r"0x([0-9a-fA-F]{2})(?![0-9a-fA-F])",
                            include.read_text(encoding="utf-8"))
        if not values:
            parser.error(f"no byte literals found in {include}")
        image = bytes(int(value, 16) for value in values)
        image_hash = sha256(image)
        record = records.get(folder.name)
        if record is None or str(record.get("region_sha256", "")).lower() != image_hash:
            parser.error(f"image bytes do not match inventory provenance: {folder.name}")

        output = output_root / (folder.name + ".bin")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(output, image)
        exported.append({
            "image": folder.name,
            "output": output.name,
            "size": len(image),
            "sha256": image_hash,
            "region_sha256": record["region_sha256"],
            "image_id": record.get("image_id"),
            "fingerprint": record.get("fingerprint"),
            "load_address": 0,
            "ghidra_language": "SPU processor, big-endian, 32-bit",
        })

    if not exported:
        parser.error(f"no verifiable image_bytes.inc files found under {source_root}")
    try:
        source_label = source_root.relative_to(repo_root).as_posix()
    except ValueError:
        source_label = str(source_root)
    manifest = {
        "schema_version": 1,
        "source_root": source_label,
        "inventory_sha256": sha256(metadata_path.read_bytes()),
        "images": exported,
        "not_exported": [folder.name for folder in lift_dirs
                         if not (folder / "image_bytes.inc").is_file()],
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    print(json.dumps({
        "exported_images": len(exported),
        "verified_region_hashes": len(exported),
        "without_embedded_image_bytes": len(manifest["not_exported"]),
        "output_dir": str(output_root),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
