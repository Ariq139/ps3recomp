#!/usr/bin/env python3
"""The opt-in async SPU task thread pool (runtime/spu/spu_task_pool.h):
reuse, no task queued behind another, multi-submitter stress and idle
retirement. Synthetic: tests/test_spu_task_pool.c runs plain host tasks."""
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='ps3-spu-task-pool-') as tmp:
    windows = os.name == 'nt'
    binary = Path(tmp) / ('test.exe' if windows else 'test')
    source = 'test_spu_task_pool_win.c' if windows else 'test_spu_task_pool.c'
    subprocess.run([os.environ.get('CC', 'clang' if windows else 'cc'), '-std=gnu17', '-O1', '-Wall', '-Wextra', '-Werror',
                    *([] if windows else ['-pthread']), '-I' + str(root / 'runtime/spu'),
                    str(root / 'tests' / source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=120)
