#!/usr/bin/env python3
"""Find RPCS3 Cell HLE call targets with no same-named local implementation.

This supplements export and syscall-slot inventories with internal call
dependencies (for example ppu_execute<&sys_fs_open>). A missing name is only a
review candidate: ps3recomp may provide equivalent behavior under another
name or through a different runtime boundary. The tool does not copy or run
RPCS3 code.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


CALL = re.compile(
    r"\b(sys_[A-Za-z][A-Za-z0-9_]*)\s*(?=\()|(?<=&)"
    r"(sys_[A-Za-z][A-Za-z0-9_]*)\s*(?=>)"
)
DEFINITION = re.compile(
    r"^\s*(?:(?:static|inline|constexpr|extern)\s+)*"
    r"[\w:<>,*&\s]+?\b(sys_[A-Za-z][A-Za-z0-9_]*)\s*"
    r"\([^;{}]*\)\s*(?:const\s*)?\{",
    re.M,
)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp"}


def source_files(root: pathlib.Path, pattern: str):
    return (path for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=pathlib.Path, required=True,
                        help="ps3recomp checkout root")
    parser.add_argument("--rpcs3", type=pathlib.Path, required=True,
                        help="RPCS3 checkout root")
    parser.add_argument("--json-out", type=pathlib.Path,
                        help="optional path for the complete candidate inventory")
    parser.add_argument("--top", type=int, default=40,
                        help="show this many most frequently referenced candidates")
    args = parser.parse_args()

    local_defs: dict[str, list[str]] = collections.defaultdict(list)
    for path in source_files(args.local, "libs/**/*"):
        text = path.read_text(encoding="latin1", errors="replace")
        for match in DEFINITION.finditer(text):
            local_defs[match.group(1)].append(str(path.relative_to(args.local)))
    for path in source_files(args.local, "runtime/**/*"):
        text = path.read_text(encoding="latin1", errors="replace")
        for match in DEFINITION.finditer(text):
            local_defs[match.group(1)].append(str(path.relative_to(args.local)))

    callers: dict[str, set[str]] = collections.defaultdict(set)
    call_counts: collections.Counter[str] = collections.Counter()
    module_root = args.rpcs3 / "rpcs3" / "Emu" / "Cell" / "Modules"
    module_files = sorted(source_files(module_root, "**/*"))
    for path in module_files:
        text = path.read_text(encoding="latin1", errors="replace")
        for match in CALL.finditer(text):
            name = match.group(1) or match.group(2)
            if name.endswith("_t"):
                continue
            call_counts[name] += 1
            callers[name].add(path.name)

    missing = {
        name: {
            "occurrences": call_counts[name],
            "rpcS3_module_files": sorted(callers[name]),
        }
        for name in sorted(call_counts)
        if name not in local_defs
    }
    print(f"RPCS3 Cell HLE module files scanned: {len(module_files)}")
    print(f"Unique sys_* call targets found: {len(call_counts)}")
    print(f"Same-named local sys_* definitions: {len(local_defs)}")
    print(f"Call-target names absent locally: {len(missing)}")
    for name, info in sorted(missing.items(),
                              key=lambda item: (-item[1]["occurrences"], item[0]))[:max(0, args.top)]:
        print(f"  {name}: {info['occurrences']} calls in {len(info['rpcS3_module_files'])} module files")

    if args.json_out:
        report = {
            "local_root": str(args.local),
            "rpcs3_root": str(args.rpcs3),
            "rpcs3_module_file_count": len(module_files),
            "unique_sys_call_target_count": len(call_counts),
            "local_sys_definition_count": len(local_defs),
            "missing_call_target_count": len(missing),
            "missing_call_targets": missing,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"Complete inventory: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
