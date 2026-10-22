#!/usr/bin/env python3
"""Turn decoded unsupported PPU statements in an existing lift into logged no-ops.

New lifts do this in ppu_lifter.py. This one-time migration brings already
generated game sources to the same behavior without regenerating the ELF.
Raw `.word` placeholders are left alone because they may be embedded data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from pathlib import Path


COMMENT_STMT = re.compile(rb"^(\s*)/\*\s*(.*?)\s*\*/;\s*$")
MARKED_STMT = re.compile(
    rb'^(\s*)ps3_ppu_unimplemented_instruction\(0u, __FILE__, __func__, __LINE__, (".*")\);\s*$'
)
MARKERS = (b"todo:", b"unhandled", b"unsupported", b"unmodeled", b"unmodelled", b"not implemented")


def migrate(path: Path) -> tuple[int, int]:
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    marked = 0
    restored_words = 0
    try:
        with os.fdopen(fd, "wb") as dst:
            with path.open("rb") as src:
                for line_number, line in enumerate(src, 1):
                    body = line.rstrip(b"\r\n")
                    ending = line[len(body):]
                    match = COMMENT_STMT.match(body)
                    if match and any(marker in match.group(2).lower() for marker in MARKERS):
                        operation = match.group(2).strip().decode("latin-1")
                        if operation.lower().startswith("todo:"):
                            operation = operation[5:].strip()
                        if operation.lower().startswith(".word"):
                            dst.write(line)
                            continue
                        replacement = (
                            match.group(1)
                            + b"ps3_ppu_unimplemented_instruction(0u, __FILE__, __func__, __LINE__, "
                            + json.dumps(operation).encode("ascii")
                            + b");"
                            + ending
                        )
                        dst.write(replacement)
                        marked += 1
                    elif marked := MARKED_STMT.match(body):
                        operation = json.loads(marked.group(2).decode("ascii"))
                        if operation.lower().startswith(".word"):
                            restored = (
                                marked.group(1) + b"/* TODO: "
                                + operation.encode("latin-1") + b" */;" + ending
                            )
                            dst.write(restored)
                            restored_words += 1
                        else:
                            dst.write(line)
                    else:
                        dst.write(line)

        if marked or restored_words:
            os.chmod(temp_name, stat.S_IMODE(path.stat().st_mode))
            os.replace(temp_name, path)
        else:
            os.unlink(temp_name)
        return marked, restored_words
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="existing lifter output directory")
    parser.add_argument("--apply", action="store_true", help="rewrite matching .cpp files")
    args = parser.parse_args()
    if not args.apply:
        parser.error("pass --apply to rewrite generated files")
    if not args.directory.is_dir() or not (args.directory / "ppu_recomp.h").is_file():
        parser.error(f"not a PPU lift directory: {args.directory}")

    total_marked = 0
    total_restored = 0
    for source in sorted(args.directory.glob("*.cpp")):
        marked, restored = migrate(source)
        if marked or restored:
            print(f"{source.name}: marked {marked}; restored raw-word placeholders {restored}")
            total_marked += marked
            total_restored += restored
    print(f"Total marked: {total_marked}; raw-word placeholders restored: {total_restored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
