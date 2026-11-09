#!/usr/bin/env python3
"""Check SHUFB selector semantics; optionally compare current/previous SIMD cost."""
import argparse
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--benchmark', action='store_true', help='run observational paired benchmark')
parser.add_argument('--avx2', action='store_true', help='use AVX2 flags (requires an AVX2 host)')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
cc = os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
if not cc:
    raise SystemExit('Set CC to an installed C compiler path')
x86 = platform.machine().lower() in ('amd64', 'x86_64', 'x64')
flags = ['-mavx2' if args.avx2 else '-msse4.1'] if x86 else []
with tempfile.TemporaryDirectory(prefix='spu-shufb-') as directory:
    tests = ['test_spu_shufb']
    if args.benchmark:
        if not x86:
            raise SystemExit('The frozen baseline benchmark requires x86 SSE4.1')
        tests.append('bench_spu_shufb')
    for name in tests:
        binary = Path(directory) / (name + ('.exe' if os.name == 'nt' else ''))
        subprocess.run([cc, '-std=gnu17', '-O2', '-Wno-deprecated-declarations',
                        *flags, '-I' + str(root / 'include'),
                        '-I' + str(root / 'runtime/platform'),
                        str(root / 'runtime/spu/tests' / (name + '.c')),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=120)
