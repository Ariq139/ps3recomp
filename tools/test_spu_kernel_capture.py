#!/usr/bin/env python3
"""Native capture/replay checks, including the real direct-trampoline drain."""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from spu_kernel_replay import extract, promote_registers

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler')
parser.add_argument('--cflag', action='append', default=[])
args = parser.parse_args()
cc = args.compiler or os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
if not cc:
    raise SystemExit('pass --compiler with the native C compiler')


def run(command, expected=0):
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    if p.returncode != expected:
        print(p.stdout, p.stderr)
        raise SystemExit('unexpected exit '+str(p.returncode)+': '+str(command))
    return p


with tempfile.TemporaryDirectory(prefix='spu-kernel-test-') as folder:
    folder = Path(folder)
    source = ROOT/'runtime/spu/tests/test_kernel_capture.c'
    original = source.read_bytes().replace(b'\r\n', b'\n')
    # Preserve public registration references and unrelated definitions, with
    # both generated-source newline conventions and non-ASCII comments.
    original += ('\n/* byte canary: π */\n'
        'void (*const synthetic_registered)(spu_context*) = synthetic_kernel;\n'
        'void unrelated_canary(spu_context* ctx){ctx->gpr[99]._u32[0]=123;}\n').encode()
    for style in ('lf', 'crlf'):
        raw = original if style == 'lf' else original.replace(b'\n', b'\r\n')
        full_input, full_output = folder/(style+'-input.c'), folder/(style+'-copy.c')
        full_input.write_bytes(raw)
        run([sys.executable, str(ROOT/'tools/promote_spu_kernel.py'), '--source', str(full_input),
             '--function', 'synthetic_kernel', '--output', str(full_output)])
        text = raw.decode()
        _, _, (begin,end) = extract(text, 'synthetic_kernel', with_span=True)
        emitted = full_output.read_bytes()
        assert emitted.startswith(text[:begin].encode())
        assert emitted.endswith(text[end:].encode())
        assert b'synthetic_registered)(spu_context*) = synthetic_kernel;' in emitted
        assert b'#define synthetic_kernel' not in emitted
        exe = folder/(style+'-full-copy'+('.exe' if os.name == 'nt' else ''))
        command = [cc, '-std=gnu17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
                   '-ffunction-sections', '-fdata-sections', '-Iinclude', '-Iruntime/spu',
                   '-I'+str(source.parent), *args.cflag, str(full_output), '-o', str(exe)]
        command += ['-fuse-ld=lld', '-Wl,/opt:ref'] if os.name == 'nt' else ['-lm']
        run(command)
        prefix = folder/(style+'-public-copy')
        run([str(exe), str(prefix)])
        command = [sys.executable, str(ROOT/'tools/spu_kernel_replay.py'), str(prefix.with_suffix('.bin')),
                   '--source', str(full_output), '--function', 'synthetic_kernel', '--compiler', cc,
                   '--transformed', *['--cflag='+flag for flag in args.cflag]]
        assert 'kernel replay: PASS' in run(command).stdout
    body, _ = extract(source.read_text(), 'synthetic_kernel')
    for invalid in (
        '(ctx->gpr[3]) = ctx->gpr[4];',
        '(ctx->gpr[3]._u32[0]) += 1;',
        '++(ctx->gpr[3]._u32[0]);',
        'ctx->ls = (uint8_t*)ctx->gpr;',
        'other->gpr[3] = ctx->gpr[4];',
        '(*fn)(ctx);',
        'g_spu_ls_rdiag = 1;',
        'u128* alias = &ctx->gpr[3];',
        'ctx->gpr[3] = ctx->gpr[ctx->gpr[2]._u32[0]];',
        'synthetic_kernel(ctx);',
        'ctx->pc++;',
        "ctx->gpr[4]._u32[0] = 'ctx->gpr[3]';",
        'u128 kr3 = ctx->gpr[4]; ctx->gpr[3] = kr3;',
    ):
        bad = body.replace('ctx->gpr[3]._u32[0] += 9;', invalid)
        try:
            pure, _ = extract(bad, 'synthetic_kernel')
            promote_registers(pure, 'synthetic_kernel')
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported kernel accepted: '+invalid)
    print('full-source public wrapper: LF/CRLF bytes preserved; unsupported alias/mutation forms rejected')
    for mode in ('wrapper', 'drain', 'macro'):
        exe = folder/(mode + ('.exe' if os.name == 'nt' else ''))
        command = [cc, '-std=gnu17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
                   '-ffunction-sections', '-fdata-sections', '-Iinclude', '-Iruntime/spu',
                   *args.cflag, str(source), '-o', str(exe)]
        if mode in ('drain', 'macro'):
            command += ['-DSPU_CAPTURE_USE_DRAIN']
        if mode == 'macro':
            command += ['-DSPU_CAPTURE_USE_MACRO']
        command += ['-fuse-ld=lld', '-Wl,/opt:ref'] if os.name == 'nt' else ['-lm']
        run(command)
        prefix = folder/(mode+'-capture')
        run([str(exe), str(prefix)])
        capture = prefix.with_suffix('.bin')
        data = capture.read_bytes()
        assert len(data) == 40+2*(13*4+128*16+256*1024)
        # Only the first invocation was saved. The continuation runs during
        # real guest execution, but its writes must not enter the after image.
        gpr_before = 40+13*4
        gpr_after = 40+(13*4+128*16+256*1024)+13*4
        assert struct.unpack_from('<I', data, gpr_before+3*16)[0] == 5
        assert struct.unpack_from('<I', data, gpr_after+3*16)[0] == 14
        assert struct.unpack_from('<I', data, gpr_after+4*16)[0] == 0
        replay = [sys.executable, str(ROOT/'tools/spu_kernel_replay.py'), str(capture),
                  '--source', str(source), '--function', 'synthetic_kernel', '--compiler', cc]
        replay += ['--cflag='+flag for flag in args.cflag]
        assert 'kernel replay: PASS' in run(replay).stdout
        assert 'kernel replay: PASS' in run(replay+['--promote-registers']).stdout
        if mode == 'wrapper':
            timed = run(replay+['--promote-registers', '--benchmark-iterations', '16'])
            assert timed.stdout.count('paired kernel bench run') == 6
            assert 'promotion guards:' in timed.stdout
        odd_prefix = folder/(mode+'-odd')
        run([str(exe), str(odd_prefix), '6'])
        odd_replay = replay.copy()
        odd_replay[2] = str(odd_prefix.with_suffix('.bin'))
        assert 'kernel replay: PASS' in run(odd_replay).stdout
        assert 'kernel replay: PASS' in run(odd_replay+['--promote-registers']).stdout
        mutated = folder/'mutated.c'
        mutated.write_text(source.read_text().replace('+= 9;', '+= 10;'))
        changed = replay.copy()
        changed[changed.index('--source')+1] = str(mutated)
        p = run(changed, expected=1)
        assert 'kernel replay: FAIL' in p.stdout and 'GPR mismatch' in p.stderr
        # Capture I/O failure cannot suppress or retry guest execution.
        p = run([str(exe), str(folder/'nonexistent-directory'/mode)])
        assert 'FAILED (guest execution continued)' in p.stderr
        print(mode+': capture, exact image/PC, one-shot, replay, changed-output rejection, I/O failure PASS')
    truncated = folder/'truncated.bin'
    truncated.write_bytes(data[:100])
    command = replay.copy()
    command[2] = str(truncated)
    assert 'incomplete' in run(command, expected=1).stderr
    leaf = ROOT/'runtime/spu/tests/test_kernel_capture_leaf.c'
    binary = folder/('leaf.exe' if os.name == 'nt' else 'leaf')
    command = [cc, '-std=gnu17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
               '-ffunction-sections', '-fdata-sections', '-Iinclude', '-Iruntime/spu',
               *args.cflag, str(leaf), '-o', str(binary)]
    command += ['-fuse-ld=lld', '-Wl,/opt:ref'] if os.name == 'nt' else ['-lm']
    run(command)
    prefix = folder/'leaf-capture'
    run([str(binary), str(prefix)])
    capture = prefix.with_suffix('.bin')
    contents = capture.read_bytes()
    assert struct.unpack_from('<I', contents, 20)[0] == 0x658
    assert struct.unpack_from('<I', contents, 40)[0] == 0x40
    assert struct.unpack_from('<I', contents, 40+13*4+128*16+256*1024)[0] == 0x40
    command = [sys.executable, str(ROOT/'tools/spu_kernel_replay.py'), str(capture),
               '--source', str(leaf), '--function', 'synthetic_leaf', '--compiler', cc,
               '--leaf-block', *['--cflag='+flag for flag in args.cflag]]
    assert 'kernel replay: PASS' in run(command).stdout
    without_leaf = command.copy()
    without_leaf.remove('--leaf-block')
    assert 'must expose a continuation' in run(without_leaf, expected=1).stderr
    assert 'kernel replay: PASS' in run(command+['--promote-registers', '--benchmark-iterations', '16']).stdout
    missing = folder/'nonexistent-leaf-directory'/'capture'
    assert 'FAILED (guest execution continued)' in run([str(binary), str(missing)]).stderr
    print('leaf API: exact logical target, architectural PC/TLS preservation, replay and promotion PASS')
    print('truncated capture rejected; all kernel capture checks passed')
