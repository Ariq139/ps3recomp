#!/usr/bin/env python3
"""Build and run the synthetic PRX export registry regression."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        if os.environ.get("CI"):
            raise SystemExit("FAIL: no C compiler found")
        print("SKIP: no C compiler")
        return
    with tempfile.TemporaryDirectory(prefix="prx-export-registry-") as tmp:
        binary = Path(tmp) / ("test.exe" if os.name == "nt" else "test")
        subprocess.run([cc, "-std=c17", "-O2", "-UNDEBUG",
                        str(ROOT / "runtime/prx/prx_loader.c"),
                        str(ROOT / "runtime/prx/tests/test_prx_export_registry.c"),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()
