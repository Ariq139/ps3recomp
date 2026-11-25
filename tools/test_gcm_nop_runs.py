#!/usr/bin/env python3
"""The FIFO walker passes over NOP runs at once, in order, stopping at put and
at an IO page end (libs/video/tests/test_gcm_nop_runs.c, which compiles
cellGcmSys.c in)."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
if os.name == 'nt':
    print('SKIP: the GCM NOP-run fixture uses POSIX linking')
    raise SystemExit(0)
with tempfile.TemporaryDirectory(prefix='gcm-nop-runs-') as tmp:
    binary = Path(tmp) / 'test'
    linker = ['-Wl,-dead_strip'] if sys.platform == 'darwin' else ['-Wl,--gc-sections', '-pthread', '-lm']
    subprocess.run([os.environ.get('CC', 'cc'), '-std=gnu17', '-O1', '-UNDEBUG',
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
