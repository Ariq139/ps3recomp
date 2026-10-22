"""Compile the production registry in isolation and exercise cache invalidation.

The registry currently shares a translation unit with DMA and SPURS services.
Extract that unchanged region so this test needs neither guest memory services
nor linker-dependent elimination of unrelated runtime functions.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--compiler', default=os.environ.get('CC', 'clang-cl' if os.name == 'nt' else 'cc'))
args = parser.parse_args()
source = (root / 'runtime/spu/spu_channels.c').read_text(encoding='utf-8')
begin = source.index('typedef void (*spu_fn)(spu_context*);')
end = source.index('/* ---- Swappable code overlays', begin)
with tempfile.TemporaryDirectory(prefix='spu-registry-test-') as directory:
    temp = Path(directory)
    unit = temp / 'production_registry.c'
    unit.write_text('#include "spu_interp.h"\n#include <stdatomic.h>\n'
                    '#include <stdio.h>\n#include <string.h>\n' + source[begin:end], encoding='utf-8')
    fixture = root / 'runtime/spu/tests/test_spu_lookup_cache.c'
    includes = [root / 'include', root / 'runtime/spu']
    binary = temp / ('test.exe' if os.name == 'nt' else 'test')
    if os.name == 'nt':
        command = [args.compiler, '/nologo', '/O2', '/UNDEBUG', '/D_CRT_SECURE_NO_WARNINGS',
                   *['/I' + str(path) for path in includes], str(fixture), str(unit),
                   '/Fo:' + str(temp) + os.sep, '/Fe:' + str(binary)]
    else:
        command = [args.compiler, '-std=c17', '-O2', '-UNDEBUG',
                   *['-I' + str(path) for path in includes], str(fixture), str(unit), '-o', str(binary)]
    subprocess.run(command, check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
