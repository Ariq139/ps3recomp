#!/usr/bin/env python3
"""Build and run the SPU image fingerprint cache regression."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
if not cc:
    raise SystemExit("Set CC to an installed C compiler path")

with tempfile.TemporaryDirectory(prefix="spu-fingerprint-cache-") as tmp:
    binary = Path(tmp) / ("fingerprint-cache.exe" if os.name == "nt" else "fingerprint-cache")
    subprocess.run([
        cc, "-std=c17", "-O2", "-UNDEBUG",
        str(root / "runtime/spu/tests/test_image_fingerprint_cache.c"),
        "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
