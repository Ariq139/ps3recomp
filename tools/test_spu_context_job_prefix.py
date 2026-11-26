#!/usr/bin/env python3
"""Compare the SPURS prefix initializer with complete context initialization."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
if not cc:
    raise SystemExit("Set CC to an installed C compiler path")

source = root / "runtime/spu/tests/test_spu_context_job_prefix.c"
with tempfile.TemporaryDirectory(prefix="spu-context-prefix-") as tmp:
    exe = Path(tmp) / ("context-prefix.exe" if os.name == "nt" else "context-prefix")
    subprocess.run([
        cc, "-std=gnu17", "-O2", "-UNDEBUG",
        "-I" + str(root / "include"), "-I" + str(root / "runtime/spu"),
        str(source), "-o", str(exe),
    ], check=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True,
                            text=True, timeout=30)
    print(result.stdout.strip())
