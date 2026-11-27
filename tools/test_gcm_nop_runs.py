#!/usr/bin/env python3
"""The FIFO walker passes over NOP runs at once, in order, stopping at put and
at an IO page end (libs/video/tests/test_gcm_nop_runs.c, which compiles
cellGcmSys.c in)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
compiler = os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
if not compiler:
    raise SystemExit('Set CC to an installed C compiler path')
with tempfile.TemporaryDirectory(prefix='gcm-nop-runs-') as tmp:
    binary = Path(tmp) / 'test'
    if os.name == 'nt':
        linker = ['-Wl,/OPT:REF']
    elif sys.platform == 'darwin':
        linker = ['-Wl,-dead_strip']
    else:
        linker = ['-Wl,--gc-sections', '-pthread', '-lm']
    subprocess.run([compiler, '-std=gnu17', '-O1', '-UNDEBUG',
                    '-ffunction-sections', '-fdata-sections',
                    '-I' + str(root / 'include'), '-I' + str(root / 'libs/video'),
                    str(root / 'libs/video/tests/test_gcm_nop_runs.c'), *linker,
                    '-o', str(binary)], check=True)
    base = {k: v for k, v in os.environ.items()
            if k not in ('GCM_FIFO_HISTORY', 'GCM_WORD_STATS', 'GCM_CIRCLE_DUMP')}
    for mode, extra in (
        ('default', {}),
        ('history-off', {'GCM_FIFO_HISTORY': '0'}),
        ('word-stats', {'GCM_FIFO_HISTORY': '0', 'GCM_WORD_STATS': '1'}),
        ('circle-dump', {'GCM_FIFO_HISTORY': '0', 'GCM_CIRCLE_DUMP': '1'}),
    ):
        result = subprocess.run([str(binary)], timeout=30, capture_output=True,
                                text=True, env={**base, **extra})
        print(f'{mode}: {result.stdout.strip()}')
        if result.returncode:
            print(result.stderr[-2000:])
            raise SystemExit(f'test_gcm_nop_runs failed: {mode}')
