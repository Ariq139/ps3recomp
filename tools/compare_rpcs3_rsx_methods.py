#!/usr/bin/env python3
"""Name-level comparison of RPCS3-bound RSX methods and local video references."""
from __future__ import annotations

import argparse
import json
import pathlib
import re


METHOD = re.compile(r"\bNV[0-9A-Fa-f]{4}_[A-Za-z0-9_]+\b")
BOUND_SINGLE = re.compile(r"\bbind\s*\(\s*(NV[0-9A-Fa-f]{4}_[A-Za-z0-9_]+)")
BOUND_RANGE = re.compile(r"\bbind_range\s*<\s*(NV[0-9A-Fa-f]{4}_[A-Za-z0-9_]+)")
COMMENT = re.compile(r"/\*.*?\*/|//[^\r\n]*", re.S)
STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def source_text(path: pathlib.Path) -> str:
    text = path.read_text(encoding="latin1")
    return STRING.sub(" ", COMMENT.sub(" ", text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=pathlib.Path, required=True,
                        help="ps3recomp checkout root")
    parser.add_argument("--rpcs3", type=pathlib.Path, required=True,
                        help="RPCS3 checkout root")
    parser.add_argument("--json-out", type=pathlib.Path,
                        help="optional JSON report path")
    args = parser.parse_args()

    oracle = args.rpcs3 / "rpcs3" / "Emu" / "RSX" / "rsx_methods.cpp"
    oracle_text = source_text(oracle)
    bound = set(BOUND_SINGLE.findall(oracle_text))
    bound.update(BOUND_RANGE.findall(oracle_text))

    local_files = sorted(
        path for path in (args.local / "libs" / "video").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".c", ".h", ".m", ".mm"}
        and "tests" not in path.parts
    )
    local_names: set[str] = set()
    for path in local_files:
        local_names.update(METHOD.findall(source_text(path)))

    missing = sorted(bound - local_names)
    print(f"RPCS3 RSX method source: {oracle}")
    print(f"RPCS3 explicitly bound method names: {len(bound)}")
    print(f"Local video source files scanned: {len(local_files)}")
    print(f"RPCS3 bound names absent from local video identifiers: {len(missing)}")
    for name in missing:
        print(f"  {name}")

    if args.json_out:
        report = {
            "rpcs3_source": str(oracle),
            "local_source_count": len(local_files),
            "rpcs3_bound_count": len(bound),
            "local_referenced_count": len(local_names),
            "missing_name_count": len(missing),
            "missing_names": missing,
            "note": "Name-level only. Ranges, generic method storage, and aliases need manual review.",
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"Complete inventory: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
