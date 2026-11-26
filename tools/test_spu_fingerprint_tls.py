#!/usr/bin/env python3
"""Compile the real fingerprint-cache TLS owner and exercise native thread teardown."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="spu-fingerprint-tls-") as tmp:
    exe = Path(tmp) / ("fixture.exe" if os.name == "nt" else "fixture")
    cmd = [os.environ.get("CC", "clang" if os.name == "nt" else "cc"),
           "-std=c11", "-O2", "-UNDEBUG", "-Wall", "-Wextra", "-Werror"]
    if os.name != "nt":
        cmd.append("-pthread")
    cmd += [str(ROOT / "runtime/spu/tests/test_image_fingerprint_tls.c"), "-o", str(exe)]
    subprocess.run(cmd, check=True, timeout=60)
    subprocess.run([str(exe)], check=True, timeout=30)
