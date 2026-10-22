#!/usr/bin/env python3
"""Inventory RPCS3-registered Cell module exports absent from local libs/*.c.

This is a name-level gap finder, not proof that an API is behaviorally
implemented. It includes the explicit manifest for registered unprefixed
guest exports and intentionally does not copy or execute RPCS3 code.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


RPCS3_REG = re.compile(
    r"REG_FUNC\s*\(\s*([^,]+?)\s*,\s*([A-Za-z_]\w*)\s*\)")
LOCAL_DEF = re.compile(
    r"^(?!static)[A-Za-z_][\w \t\*]*?\b(_?(?:cell|sys|sce|_sys)\w+)\s*"
    r"\([^;]*?\)\s*\{", re.M)
GENERIC_DEF = re.compile(
    r"^(?!static)[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*"
    r"\([^;]*?\)\s*\{", re.M)
MODULE_DECL = re.compile(r"DECLARE\s*\(\s*ppu_module_manager::(\w+)\s*\)")
SPECIAL_EXPORTS_PATH = pathlib.Path(__file__).with_name("hle_special_exports.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=pathlib.Path, required=True,
                        help="ps3recomp checkout root")
    parser.add_argument("--rpcs3", type=pathlib.Path, required=True,
                        help="RPCS3 checkout root")
    parser.add_argument("--json-out", type=pathlib.Path,
                        help="optional path for the complete gap inventory")
    args = parser.parse_args()

    local_exports: set[str] = set()
    local_special: set[str] = set()
    special_manifest = json.loads(SPECIAL_EXPORTS_PATH.read_text(encoding="utf-8"))
    special_by_module = special_manifest["modules"]
    local_sources = sorted((args.local / "libs").glob("**/*.c"))
    for source in local_sources:
        text = source.read_text(encoding="latin1")
        local_exports.update(m.group(1) for m in LOCAL_DEF.finditer(text))
        special = set(special_by_module.get(source.stem, ()))
        definitions = {m.group(1) for m in GENERIC_DEF.finditer(text)}
        matches = special & definitions
        local_exports.update(matches)
        local_special.update(matches)

    rpcs3_modules: dict[str, set[str]] = collections.defaultdict(set)
    module_sources = sorted(
        (args.rpcs3 / "rpcs3" / "Emu" / "Cell" / "Modules").glob("**/*.cpp"))
    for source in module_sources:
        text = source.read_text(encoding="latin1")
        declaration = MODULE_DECL.search(text)
        fallback_module = source.stem
        for match in RPCS3_REG.finditer(text):
            owner = match.group(1).strip()
            module = declaration.group(1) if declaration else owner or fallback_module
            rpcs3_modules[module].add(match.group(2))

    all_rpcs3 = set().union(*rpcs3_modules.values()) if rpcs3_modules else set()
    missing_by_module = {
        module: sorted(names - local_exports)
        for module, names in sorted(rpcs3_modules.items())
        if names - local_exports
    }
    missing_count = len(all_rpcs3 - local_exports)
    print(f"RPCS3 Cell module files scanned: {len(module_sources)}")
    print(f"RPCS3 registered export names: {len(all_rpcs3)}")
    print(f"ps3recomp exported-looking libs/*.c names: {len(local_exports)}")
    print(f"  including verified unprefixed exports: {len(local_special)}")
    print(f"RPCS3 names absent from local libs: {missing_count}")
    for module, names in sorted(missing_by_module.items(),
                                key=lambda item: (-len(item[1]), item[0])):
        print(f"  {module}: {len(names)}")

    if args.json_out:
        report = {
            "local_root": str(args.local),
            "rpcs3_root": str(args.rpcs3),
            "rpcs3_module_file_count": len(module_sources),
            "rpcs3_export_name_count": len(all_rpcs3),
            "local_lib_export_name_count": len(local_exports),
            "local_special_export_names": sorted(local_special),
            "special_export_reference": special_manifest["reference"],
            "missing_name_count": missing_count,
            "missing_by_module": missing_by_module,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"Complete inventory: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
