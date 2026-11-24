#!/usr/bin/env python3
"""Generate verified readable companions for the P5 lifted SPU sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_bytes_from_include(path: Path) -> bytes:
    values = re.findall(r"0x([0-9a-fA-F]{2})(?![0-9a-fA-F])",
                        path.read_text(encoding="utf-8"))
    if not values:
        raise ValueError(f"no byte literals found in {path}")
    return bytes(int(value, 16) for value in values)


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True))
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = (repo_root / "p5_npeb02436_gameonly" / "src" / "ps3" / "spu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root,
                        help="SPU image folders containing spu_recomp.c")
    parser.add_argument("--names-dir", type=Path,
                        help="Optional per-image Ghidra maps: <dir>/<image>/names.json")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    tool = Path(__file__).with_name("spu_decompiled_view.py").resolve()
    if not tool.is_file():
        parser.error(f"readable-view tool is missing: {tool}")
    if not source_root.is_dir():
        parser.error(f"SPU source folder does not exist: {source_root}")
    names_root = args.names_dir.resolve() if args.names_dir else None
    if names_root is not None and not names_root.is_dir():
        parser.error(f"symbol map folder does not exist: {names_root}")

    metadata_path = source_root / "spu_prescanned.json"
    if names_root is not None and not metadata_path.is_file():
        parser.error("symbol maps require spu_prescanned.json to verify image provenance")
    image_metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_metadata = {entry.get("symbol_prefix"): entry for entry in metadata
                          if entry.get("symbol_prefix")}

    sources = sorted(source_root.glob("*/spu_recomp.c"))
    if not sources:
        parser.error(f"no image-level spu_recomp.c files found under {source_root}")

    tool_hash = sha256(tool.read_bytes())
    records: list[dict] = []
    for source in sources:
        output = source.with_name("spu_recomp.readable.c")
        report_path = source.with_name("spu_recomp.readable.json")
        temporary_output = output.with_name(output.name + ".tmp")
        temporary_report = report_path.with_name(report_path.name + ".tool.tmp")
        original = source.read_bytes()
        names_path = None
        names_sha256 = None
        try:
            command = [sys.executable, str(tool), str(source), "--output",
                       str(temporary_output), "--report", str(temporary_report)]
            if names_root is not None:
                names_path = names_root / source.parent.name / "names.json"
                if not names_path.is_file():
                    flat_map = names_root / (source.parent.name + ".json")
                    names_path = flat_map if flat_map.is_file() else None
                if names_path is not None:
                    image_include = source.parent / "image_bytes.inc"
                    image_record = image_metadata.get(source.parent.name)
                    if not image_include.is_file() or image_record is None:
                        raise ValueError(
                            "refusing symbols: this source has no embedded image bytes and "
                            "matching EBOOT provenance record")
                    image_hash = sha256(image_bytes_from_include(image_include))
                    if image_record.get("region_sha256", "").lower() != image_hash:
                        raise ValueError(
                            "embedded SPU image bytes do not match spu_prescanned.json")
                    name_document = json.loads(names_path.read_text(encoding="utf-8"))
                    name_meta = name_document.get("_meta", {})
                    if not isinstance(name_meta, dict) or str(
                            name_meta.get("image_sha256", "")).lower() != image_hash:
                        raise ValueError(
                            "Ghidra symbol map hash does not match this exact SPU image")
                    names_sha256 = sha256(names_path.read_bytes())
                    command.extend(["--names", str(names_path),
                                    "--image-sha256", image_hash])
            subprocess.run(command, check=True, capture_output=True, text=True)
            transform = json.loads(temporary_report.read_text(encoding="utf-8"))
            readable = temporary_output.read_bytes()
            normalized_input = source.read_text(encoding="utf-8").encode("utf-8")
            normalized_output = temporary_output.read_text(encoding="utf-8").encode("utf-8")
            if transform.get("input_sha256") != sha256(normalized_input):
                raise RuntimeError("transform report input hash does not match its source")
            if transform.get("output_sha256") != sha256(normalized_output):
                raise RuntimeError("transform report output hash does not match its output")
            if source.read_bytes() != original:
                raise RuntimeError("canonical lifted source changed during transformation")

            os.replace(temporary_output, output)
            record = {
                "image": source.parent.name,
                "source": source.name,
                "source_file_sha256": sha256(original),
                "readable": output.name,
                "readable_file_sha256": sha256(readable),
                "symbol_map": ({
                    "path": names_path.relative_to(names_root).as_posix()
                            if names_path.is_relative_to(names_root) else str(names_path),
                    "file_sha256": names_sha256,
                    "image_sha256": image_hash,
                } if names_path is not None else None),
                "transform": transform,
            }
            write_json_atomic(report_path, record)
            records.append(record)
            print(json.dumps({"image": record["image"], **transform}, sort_keys=True))
        except (OSError, UnicodeError, subprocess.CalledProcessError,
                json.JSONDecodeError, RuntimeError) as error:
            detail = ((error.stderr or "").strip()
                      if isinstance(error, subprocess.CalledProcessError) else str(error))
            if not detail:
                detail = str(error)
            print(f"{source}: {detail}", file=sys.stderr)
            temporary_output.unlink(missing_ok=True)
            temporary_report.unlink(missing_ok=True)
            return 1
        finally:
            temporary_report.unlink(missing_ok=True)

    try:
        source_root_label = source_root.relative_to(repo_root).as_posix()
    except ValueError:
        source_root_label = str(source_root)
    manifest = {
        "schema_version": 1,
        "tool": "tools/spu_decompiled_view.py",
        "tool_file_sha256": tool_hash,
        "driver": "tools/build_spu_readable_view.py",
        "driver_file_sha256": sha256(Path(__file__).read_bytes()),
        "symbol_map_root": str(names_root) if names_root is not None else None,
        "source_root": source_root_label,
        "image_count": len(records),
        "images": records,
    }
    write_json_atomic(source_root / "readable_view_manifest.json", manifest)
    totals = {
        key: sum(int(record["transform"].get(key, 0)) for record in records)
        for key in ("functions", "annotated_ls_accesses", "formatted_trampolines",
                    "split_dense_statements", "annotated_symbol_references")
    }
    print(json.dumps({"images": len(records), **totals}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
