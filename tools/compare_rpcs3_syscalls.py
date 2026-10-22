#!/usr/bin/env python3
"""Compare RPCS3's named LV2 syscall slots with ps3recomp registrations.

This is an inventory, not an ABI or behavior-equivalence proof. Same-slot
handlers may still differ in name, arguments, outputs, or semantics.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re


RPCS3_ENTRY = re.compile(
    r"\b(BIND_SYSC|NULL_FUNC)\s*\(\s*([A-Za-z_]\w*)\s*\)"
    r"[^\r\n]*?//\s*(\d+)\s*\(0x[0-9A-Fa-f]+\)")
DEFINE = re.compile(r"^\s*#\s*define\s+(SYS_[A-Za-z0-9_]+)\s+(\d+)\b", re.M)
REGISTER = re.compile(
    r"lv2_syscall_register\s*\(\s*(?:tbl|&g_lv2_syscalls)\s*,\s*"
    r"(SYS_[A-Za-z0-9_]+|\d+)\s*,\s*([A-Za-z_]\w*)\s*\)", re.S)


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="latin1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=pathlib.Path, required=True)
    ap.add_argument("--rpcs3", type=pathlib.Path, required=True)
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args()

    oracle = args.rpcs3 / "rpcs3/Emu/Cell/lv2/lv2.cpp"
    rpcs3: dict[int, dict[str, str]] = {}
    for kind, name, number in RPCS3_ENTRY.findall(read(oracle)):
        rpcs3[int(number)] = {"kind": kind, "name": name}

    constants: dict[str, int] = {}
    for header in sorted((args.local / "runtime").glob("**/*.h")):
        constants.update({name: int(value) for name, value in DEFINE.findall(read(header))})
    local_by_num: dict[int, set[str]] = {}
    local_sources = sorted((args.local / "runtime").glob("**/*.c"))
    local_sources += sorted((args.local / "libs").glob("**/*.c"))
    for source in local_sources:
        for symbol, handler in REGISTER.findall(read(source)):
            number = int(symbol) if symbol.isdigit() else constants.get(symbol)
            if number is not None:
                local_by_num.setdefault(number, set()).add(handler)

    missing_slot = []
    local_aliases = []
    null_slots = []
    for number, entry in sorted(rpcs3.items()):
        local_handlers = sorted(local_by_num.get(number, set()))
        item = {"number": number, **entry, "local_handlers": local_handlers}
        if entry["kind"] == "NULL_FUNC":
            null_slots.append(item)
        elif not local_handlers:
            missing_slot.append(item)
        elif entry["name"] not in local_handlers:
            local_aliases.append(item)

    print(f"RPCS3 named LV2 entries with explicit slot numbers: {len(rpcs3)}")
    print(f"Local syscall numbers with a registered handler: {len(local_by_num)}")
    print(f"RPCS3 BIND_SYSC slots unregistered locally: {len(missing_slot)}")
    print(f"RPCS3 BIND_SYSC slots handled locally under another symbol: {len(local_aliases)}")
    print(f"RPCS3 explicit NULL_FUNC slots: {len(null_slots)}")
    for item in missing_slot:
        print(f"  missing slot {item['number']:>3}: {item['name']}")
    for item in local_aliases:
        print(f"  local alias {item['number']:>3}: {item['name']} -> {', '.join(item['local_handlers'])}")

    if args.json_out:
        report = {
            "rpcs3_source": str(oracle),
            "rpcs3_named_slot_count": len(rpcs3),
            "local_registered_slot_count": len(local_by_num),
            "unregistered_bind_slots": missing_slot,
            "bind_slots_handled_under_other_symbols": local_aliases,
            "rpcs3_null_func_slots": null_slots,
            "note": "Slot/name inventory only; it does not prove ABI or behavior equivalence.",
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Complete inventory: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
