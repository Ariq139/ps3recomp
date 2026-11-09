#!/usr/bin/env python3
"""Validate AVX2 SPU FMA rounding; optionally run a paired synthetic benchmark."""
import argparse
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--benchmark', action='store_true', help='compare frozen RN baseline vs current; observational only')
args = parser.parse_args()
if platform.machine().lower() not in ('amd64', 'x86_64', 'x64'):
    raise SystemExit('This AVX2-specific fixture requires an x86 host')
root = Path(__file__).resolve().parents[1]
cc = os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
if not cc:
    raise SystemExit('Set CC to an installed C compiler path')
with tempfile.TemporaryDirectory(prefix='spu-fma-boundary-') as tmp:
    cases = ['test_spu_fma_boundary']
    if args.benchmark:
        cases.append('bench_spu_fma_boundary')
    for name in cases:
        binary = Path(tmp) / (name + ('.exe' if os.name == 'nt' else ''))
        subprocess.run([cc, '-std=gnu17', '-O2', '-mavx2', '-ffp-contract=off',
                        '-Wno-deprecated-declarations', '-I' + str(root/'include'),
                        '-I' + str(root/'runtime/platform'),
                        str(root/'runtime/spu/tests'/(name+'.c')), '-o', str(binary),
                        *([] if os.name == 'nt' else ['-lm'])], check=True)
        subprocess.run([str(binary)], check=True, timeout=120)
