#!/usr/bin/env python3
"""Replay one captured pure LS/GPR function, without following its continuation.

Capture at boot with SPU_KERNEL_CAPTURE_PREFIX=<path-without-.bin>,
SPU_KERNEL_CAPTURE_IMAGE=<exact registered image id> and
SPU_KERNEL_CAPTURE_PC=0x16a0. No generated source is modified.

Example: python tools/spu_kernel_replay.py capture.bin --source path/spu_recomp.c
         --function p5_job_A96F7BFC_spu_func_000016A0 --cflag=-mavx2 --cflag=-mfma
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_BYTES = 13 * 4 + 128 * 16 + 256 * 1024


def extract(source, name, *, with_span=False, validate=True, allow_leaf=False):
    # Ignore comments and strings while balancing the selected function's braces.
    masked = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda m: ' ' * len(m[0]), source, flags=re.S)
    match = re.search(r'\bvoid\s+' + re.escape(name) +
                      r'\s*\(spu_context\s*\*\s*ctx\)\s*\{', masked)
    if not match:
        raise ValueError('selected function definition not found')
    depth, end = 1, match.end()
    while end < len(masked) and depth:
        depth += (masked[end] == '{') - (masked[end] == '}')
        end += 1
    if depth:
        raise ValueError('unbalanced generated function')
    body = source[match.start():end]
    plain = masked[match.start():end]
    if not validate:
        return (body, [], (match.start(), end)) if with_span else (body, [])
    fields = set(re.findall(r'ctx\s*->\s*(\w+)', plain))
    if fields - {'gpr', 'ls', 'pc', 'status', 'stop_code'}:
        raise ValueError('kernel accesses unsupported context fields: ' + str(sorted(fields)))
    # This is a pure-block tool. In particular, never replay channels, DMA,
    # interrupts, waits or nested guest calls using no-op runtime stubs.
    calls = set(re.findall(r'\b([A-Za-z_]\w*)\s*\(', plain))
    helper_text = (ROOT/'runtime/spu/spu_helpers.h').read_text()
    helpers = set(re.findall(r'\b(spu_\w+)\s*\(', helper_text))
    ls_helpers = {'spu_ls_read128', 'spu_ls_write128', 'spu_ls_read32',
                  'spu_ls_write32', 'spu_ls_read16', 'spu_ls_write16', 'spu_link'}
    forbidden = [c for c in calls if c.startswith(('spu_wrch', 'spu_rdch',
        'spu_rchcnt', 'spu_halt', 'spu_stop', 'spu_drain', 'spu_indirect'))]
    allowed_words = {'if', 'for', 'while', 'switch', 'sizeof', name}
    forbidden += [c for c in calls if c not in helpers | ls_helpers | allowed_words]
    if forbidden:
        raise ValueError('kernel contains unsupported calls: ' + ', '.join(sorted(set(forbidden))))
    # Context may be accessed by field or passed to a known LS accessor only.
    # Returning/storing it or passing it to an arbitrary helper is an escape.
    uses = plain[plain.index('{')+1:]
    uses = re.sub(r'\bspu_ls_(?:read|write)(?:128|32|16)\s*\(\s*ctx\b', '', uses)
    uses = re.sub(r'\bctx\s*->', '', uses)
    if re.search(r'\bctx\b', uses):
        raise ValueError('kernel context escapes the pure LS/GPR block')
    targets = sorted(set(re.findall(r'g_spu_trampoline_fn\s*=\s*(\w+)', plain)))
    if not targets and not allow_leaf:
        raise ValueError('kernel must expose a continuation; arbitrary complete jobs are unsupported')
    return (body, targets, (match.start(), end)) if with_span else (body, targets)


def promote_registers(body, name):
    """Temporary pure-leaf candidate; the production generated file is untouched.

    Only constant-index vector-register accesses are promoted. All referenced
    entry values are loaded conservatively; the native compiler removes loads
    dominated by assignments. Every modified register is committed on exit.
    """
    def mask_token(match):
        if match[0].startswith(('"', "'")):
            raise ValueError('promotion rejects string and character literals')
        return ' ' * len(match[0])
    plain = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                   mask_token, body, flags=re.S)
    code = plain[plain.index('{')+1:]
    if re.search(r'\b'+re.escape(name)+r'\s*\(', code):
        raise ValueError('promotion rejects recursive calls')
    # This pass supports the lifter's explicit GPR/LS-helper subset, not
    # arbitrary C alias analysis. Reject shapes whose writes could evade the
    # modified-register set or change the assumptions of the entry guard.
    if '*' in code or re.search(r'&\s*(?:\(\s*)*ctx\s*->', code):
        raise ValueError('promotion rejects pointer syntax or context-field addresses')
    literal = r'(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*'
    terminal_words = r'(?:(?:ctx\s*->\s*(?:pc|status|stop_code)\s*=\s*'+literal+r'|'+\
                     r'g_spu_trampoline_fn\s*=\s*[A-Za-z_]\w*);\s*)+'
    terminal_code = re.sub(terminal_words+r'(?:return\s*;|(?=\s*}\s*$))', '', code)
    if re.search(r'ctx\s*->\s*(?:pc|status|stop_code)\b|\bg_spu_trampoline_fn\b', terminal_code):
        raise ValueError('promotion requires terminal literal PC/status and continuation writes')
    lhs = re.compile(r'(?:ctx\s*->\s*gpr\s*\[\s*\d+\s*\]'
                     r'(?:\s*\.\s*_[usf]\d+\s*\[\s*\d+\s*\])?'
                     r'|ctx\s*->\s*(?:pc|status|stop_code)|g_spu_trampoline_fn)\s*$')
    for op in re.finditer(r'(?<![=!<>])(?:<<=|>>=|[+\-/%&^|]=|=(?!=)|\+\+|--)', code):
        if not lhs.search(code[:op.start()]):
            raise ValueError('promotion rejects noncanonical or parenthesized mutation')
    allowed = {'ctx', 'gpr', 'pc', 'status', 'stop_code', 'g_spu_trampoline_fn',
               'if', 'else', 'goto', 'return', 'while', 'for', 'do', 'break',
               'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int32_t', 'uint32_t',
               'int64_t', 'uint64_t', 'size_t'}
    helpers = set(re.findall(r'\b(spu_\w+)\s*\(',
                            (ROOT/'runtime/spu/spu_helpers.h').read_text()))
    allowed |= helpers | {'spu_ls_read128', 'spu_ls_write128', 'spu_ls_read32',
                          'spu_ls_write32', 'spu_ls_read16', 'spu_ls_write16', 'spu_link'}
    allowed |= set(re.findall(r'g_spu_trampoline_fn\s*=\s*([A-Za-z_]\w*)', code))
    for token in re.findall(r'\b[A-Za-z_]\w*\b', code):
        if token not in allowed and not re.fullmatch(r'loc_[0-9A-Fa-f]+|_[usf](?:8|16|32|64)', token):
            raise ValueError('promotion rejects unrecognized identifier/alias: '+token)
    if re.search(r'\bkr\d+\b', plain):
        raise ValueError('promotion register-local names collide with existing source')
    ref = re.compile(r'\bctx\s*->\s*gpr\s*\[\s*(\d+)\s*\]')
    refs = list(ref.finditer(plain))
    all_refs = list(re.finditer(r'\bctx\s*->\s*gpr\b', plain))
    if len(refs) != len(all_refs) or any(int(m[1]) >= 128 for m in refs):
        raise ValueError('promotion requires constant GPR indices in [0,127]')
    written = set()
    for m in refs:
        tail = plain[m.end():]
        field = re.match(r'\s*\.\s*(_[us](?:8|16|32|64)|_f(?:32|64))\s*\[\s*(\d+)\s*\]', tail)
        if re.match(r'\s*\.', tail) and not field:
            raise ValueError('promotion requires constant lane indices')
        if field:
            bits = int(re.search(r'\d+', field[1])[0])
            if int(field[2]) >= 128 // bits:
                raise ValueError('promotion rejects out-of-range register lanes')
            tail = tail[field.end():]
        if re.match(r'\s*(?:=(?!=)|[+\-*/^|&%]=|<<=|>>=|\+\+|--)', tail):
            written.add(int(m[1]))
        # Prefix mutation is not emitted by the lifter. Reject rather than
        # silently missing it in the modified-register set.
        if re.search(r'(?:\+\+|--)\s*$', plain[:m.start()]):
            raise ValueError('promotion rejects prefix register mutation')
    promoted = body
    for m in reversed(refs):
        promoted = promoted[:m.start()] + 'kr' + m[1] + promoted[m.end():]
    commits = ''.join('ctx->gpr[%d] = kr%d; ' % (r, r) for r in sorted(written))
    terminal = re.compile(r'((?:(?:ctx\s*->\s*(?:pc|status|stop_code)\s*=[^;]+|'
                          r'g_spu_trampoline_fn\s*=\s*\w+);\s*)+)return\s*;')
    promoted = terminal.sub(lambda m: commits + m[1] + 'return /* committed */;', promoted)
    promoted = re.sub(r'\breturn\s*;', lambda m: '{ '+commits+'return; }', promoted)
    promoted = promoted.replace('return /* committed */;', 'return;')
    fallthrough = re.compile(r'((?:(?:ctx\s*->\s*(?:pc|status|stop_code)\s*=[^;]+|'
                             r'g_spu_trampoline_fn\s*=\s*\w+);\s*)+)(}\s*)$')
    promoted = fallthrough.sub(lambda m: commits+m[1]+m[2], promoted)
    closing = promoted.rfind('}')
    promoted = promoted[:closing] + '\n    '+commits+'\n' + promoted[closing:]
    reference = re.sub(r'\bvoid\s+'+re.escape(name)+r'\b',
                       'void '+name+'_reference', body, count=1)
    predicate = '''
static int @SAFE@(const spu_context* ctx) {
    const uintptr_t lslo=(uintptr_t)ctx->ls, ctxlo=(uintptr_t)ctx;
    return !(g_spu_ls_rdiag || g_spu_ls_wdiag || g_spu_ls_watch_n ||
        g_spu_ls_probe || g_spu_smc_watch || g_spu_wws_probe ||
        lslo > UINTPTR_MAX-SPU_LS_SIZE ||
        (lslo != (uintptr_t)ctx->ls_store &&
         lslo < ctxlo+sizeof(*ctx) && lslo+SPU_LS_SIZE > ctxlo));
}
'''.replace('@SAFE@', name+'_registers_safe')
    guard = '''
    if (!@SAFE@(ctx)) {
        @REFERENCE@(ctx); return;
    }
'''.replace('@REFERENCE@', name+'_reference').replace('@SAFE@', name+'_registers_safe')
    loads = ''.join('    u128 kr%d = ctx->gpr[%d];\n' % (r, r)
                    for r in sorted({int(m[1]) for m in refs}))
    opening = promoted.index('{')+1
    promoted = promoted[:opening] + guard + loads + promoted[opening:]
    return reference+'\n'+predicate+'\n'+promoted


def extract_transformed(source, name, *, allow_leaf=False):
    """Accept only the exact guarded transform reproduced from its pure baseline."""
    baseline, targets, ref_span = extract(source, name+'_reference', with_span=True, allow_leaf=allow_leaf)
    original = re.sub(r'\bvoid\s+'+re.escape(name+'_reference')+r'\b',
                      'void '+name, baseline, count=1)
    expected = promote_registers(original.replace('\r\n', '\n'), name)
    _, _, public_span = extract(source, name, with_span=True, validate=False)
    actual = source[ref_span[0]:public_span[1]]
    if actual.replace('\r\n', '\n') != expected:
        raise ValueError('transformed public wrapper does not match the guarded transform')
    return actual, targets


HARNESS = r'''
#include "spu_helpers.h"
#include "tests/spu_ls_watch_fixture.h"
#include "spu_kernel_capture.h"
#include <assert.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#if SPU_KERNEL_CAPTURE_X86
#include <immintrin.h>
#endif
SPU_THREAD_LOCAL void (*g_spu_trampoline_fn)(spu_context*);
@CONTINUATIONS@
@FUNCTION@
static int fields_equal(const spu_context* a, const spu_context* b) {
    return a->pc==b->pc && a->status==b->status && a->stop_code==b->stop_code &&
      a->image_id==b->image_id && a->spu_id==b->spu_id && a->int_enable==b->int_enable &&
      a->event_mask==b->event_mask && spu_event_load(a)==spu_event_load(b) &&
      a->srr0==b->srr0 && a->host_depth==b->host_depth &&
      a->drain_ret_pc==b->drain_ret_pc && a->policy_mode==b->policy_mode;
}
static uint64_t kernel_now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;LARGE_INTEGER q;
    if(!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&q);
    return (uint64_t)(q.QuadPart/frequency.QuadPart)*1000000000ull+
           (uint64_t)(q.QuadPart%frequency.QuadPart)*1000000000ull/frequency.QuadPart;
#else
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
#endif
}
int main(int argc, char** argv) {
    if(argc!=2) return 2;
    FILE* f=fopen(argv[1],"rb"); if(!f) return 2;
    char magic[8]; uint32_t h[8];
    if(fread(magic,1,8,f)!=8 || memcmp(magic,"SPUKRN01",8)) return 2;
    for(unsigned i=0;i<8;i++) if(!spu_kernel_get32(f,&h[i])) return 2;
    if(h[0]!=1 || h[1]!=SPU_KERNEL_SNAPSHOT_BYTES) return 2;
    spu_context *actual=(spu_context*)calloc(1,sizeof(spu_context));
    spu_context *gold=(spu_context*)calloc(1,sizeof(spu_context));
    if(!actual||!gold) return 2;
    spu_context_init(actual,0);spu_context_init(gold,0);
    uint32_t before_tramp,after_tramp;
    if(!spu_kernel_snapshot_read(f,actual,&before_tramp) ||
       !spu_kernel_snapshot_read(f,gold,&after_tramp) || fgetc(f)!=EOF) return 2;
    fclose(f);
    /* h[3] identifies an outlined block; its enclosing architectural PC can
     * differ and is preserved in both snapshots. */
    if((uint32_t)actual->image_id!=h[2]) return 2;
    spu_context* seed=(spu_context*)malloc(sizeof(*seed));if(!seed)return 2;
    memcpy(seed,actual,sizeof(*seed));seed->ls=seed->ls_store;
    const int rounds[4]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
#if SPU_KERNEL_CAPTURE_X86
    const uint32_t prior_mxcsr=_mm_getcsr();
#endif
    fenv_t prior_env;
    if(h[6]>3 || feholdexcept(&prior_env)!=0 || fesetround(rounds[h[6]])!=0) return 2;
#if SPU_KERNEL_CAPTURE_X86
    if(h[4]==1) {
        _Alignas(16) unsigned char fx[512]; uint32_t supported;
        _fxsave(fx);memcpy(&supported,fx+28,4);
        if(!supported) supported=0xFFBFu; /* architectural default MXCSR_MASK */
        const uint32_t controls=h[5]&0xE040u; /* rounding, FTZ, DAZ only */
        if(controls&~supported) { fesetenv(&prior_env);_mm_setcsr(prior_mxcsr);return 2; }
        _mm_setcsr((controls|0x1F80u)&supported); /* all exceptions masked */
    }
#else
    if(h[4]==1 && (h[5]&0x8040u)) { fesetenv(&prior_env);return 2; }
#endif
    g_spu_ls_probe=g_spu_smc_watch=g_spu_wws_probe=0;
    g_spu_ls_rdiag=g_spu_ls_wdiag=0;
    g_spu_trampoline_fn=before_tramp ? @FIRST_CONTINUATION@ : NULL;
    @NAME@(actual); /* exactly one native function: never drain its continuation */
    int failed=!fields_equal(actual,gold) ||
        ((g_spu_trampoline_fn!=NULL)!=(after_tramp!=0));
    for(unsigned r=0;r<128;r++) for(unsigned lane=0;lane<4;lane++)
      if(actual->gpr[r]._u32[lane]!=gold->gpr[r]._u32[lane]) {
        fprintf(stderr,"GPR mismatch r%u lane%u: %08X expected %08X\n",r,lane,
          actual->gpr[r]._u32[lane],gold->gpr[r]._u32[lane]);failed=1;goto gpr_done;
      }
gpr_done:
    for(unsigned a=0;a<SPU_LS_SIZE;a++) if(actual->ls[a]!=gold->ls[a]) {
      fprintf(stderr,"LS mismatch at0x%05X: %02X expected %02X\n",a,
        actual->ls[a],gold->ls[a]);failed=1;break;
    }
    @GUARD_CHECKS@
    @BENCHMARK@
    fesetenv(&prior_env);
#if SPU_KERNEL_CAPTURE_X86
    _mm_setcsr(prior_mxcsr);
#endif
    printf("kernel replay: %s logical=0x%05X actual-entry=0x%05X exit=0x%05X expected=0x%05X\n",
      failed?"FAIL":"PASS",h[3],seed->pc,actual->pc,gold->pc);
    free(seed);free(actual);free(gold);return failed?1:0;
}
'''

GUARD_CHECKS = r'''
    g_spu_ls_rdiag=1;
    assert(!@NAME@_registers_safe(actual));g_spu_ls_rdiag=0;
    g_spu_ls_wdiag=1;
    assert(!@NAME@_registers_safe(actual));g_spu_ls_wdiag=0;
    uint8_t* ls_saved=actual->ls;actual->ls=(uint8_t*)actual->gpr;
    assert(!@NAME@_registers_safe(actual));actual->ls=ls_saved;
    actual->ls=actual->ls_store+16;
    assert(!@NAME@_registers_safe(actual));actual->ls=ls_saved;
    assert(@NAME@_registers_safe(actual));
    puts("promotion guards: read/write diagnostics and LS/context aliases rejected");
'''

BENCHMARK = r'''
    if(!failed) {
      spu_context* ref=(spu_context*)malloc(sizeof(*ref));
      if(!ref) failed=1;
      else {
        uint64_t timer_ns=0;
        kernel_now_ns();
        for(unsigned i=0;i<128;i++) {uint64_t t=kernel_now_ns();timer_ns+=kernel_now_ns()-t;}
        printf("timer empty interval=%llu ns (unsubtracted)\n",(unsigned long long)(timer_ns/128));
        for(unsigned pair=0;pair<@RUNS@;pair++) {
         uint64_t reference_ns=0,promoted_ns=0;
         for(unsigned i=0;i<@ITERATIONS@;i++) {
          memcpy(ref,seed,sizeof(*ref));ref->ls=ref->ls_store;
          memcpy(actual,seed,sizeof(*actual));actual->ls=actual->ls_store;
          /* Alternate order; cloning and all byte comparisons are outside
           * measured intervals. Both kernels see identical captured inputs. */
          void (*reference_tramp)(spu_context*)=NULL,(*promoted_tramp)(spu_context*)=NULL;
          for(unsigned order=0;order<2;order++) {
            int native=((pair+i+order)&1u)!=0;
            g_spu_trampoline_fn=before_tramp?@FIRST_CONTINUATION@:NULL;
            uint64_t begin=kernel_now_ns();
            if(native) @NAME@(actual);else @NAME@_reference(ref);
            uint64_t elapsed=kernel_now_ns()-begin;
            if(native) {promoted_ns+=elapsed;promoted_tramp=g_spu_trampoline_fn;}
            else {reference_ns+=elapsed;reference_tramp=g_spu_trampoline_fn;}
          }
          if(reference_tramp!=promoted_tramp||!fields_equal(actual,ref)||memcmp(actual->gpr,ref->gpr,sizeof(ref->gpr))||
             memcmp(actual->ls,ref->ls,SPU_LS_SIZE)) {
            fprintf(stderr,"paired kernel mismatch iteration%u\n",i);failed=1;break;
          }
         }
         printf("paired kernel bench run%u: %u inputs reference=%llu ns/call promoted=%llu ns/call\n",
          pair+1,@ITERATIONS@,(unsigned long long)(reference_ns/@ITERATIONS@),
          (unsigned long long)(promoted_ns/@ITERATIONS@));
         if(failed) break;
        }
        free(ref);
      }
    }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--function', required=True)
    parser.add_argument('--compiler')
    parser.add_argument('--cflag', action='append', default=[])
    parser.add_argument('--promote-registers', action='store_true',
                        help='test temporary guarded GPR-local candidate; no generated files changed')
    parser.add_argument('--transformed', action='store_true',
                        help='verify/replay a full source emitted by promote_spu_kernel.py')
    parser.add_argument('--leaf-block', action='store_true',
                        help='explicitly allow a pure outlined block with no trampoline write')
    parser.add_argument('--benchmark-iterations', type=int, default=0,
                        help='paired reference/promoted calls on cloned inputs; requires promotion')
    parser.add_argument('--benchmark-runs', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if args.transformed and args.promote_registers:
        parser.error('transformed source is already promoted')
    promoted_variant = args.promote_registers or args.transformed
    if args.benchmark_iterations < 0 or args.benchmark_iterations > 1000000:
        parser.error('benchmark iterations must be between 0 and 1000000')
    if args.benchmark_iterations and not promoted_variant:
        parser.error('benchmark requires --promote-registers')
    if not 1 <= args.benchmark_runs <= 100:
        parser.error('benchmark runs must be between 1 and 100')
    data = args.capture.read_bytes()
    if len(data) != 40 + 2 * SNAPSHOT_BYTES or data[:8] != b'SPUKRN01':
        raise SystemExit('incomplete or unsupported kernel capture')
    h = struct.unpack_from('<8I', data, 8)
    if h[0] != 1 or h[1] != SNAPSHOT_BYTES or h[6] > 3:
        raise SystemExit('unsupported capture version/layout')
    text = args.source.read_text()
    if args.transformed:
        body, targets = extract_transformed(text, args.function, allow_leaf=args.leaf_block)
    else:
        body, targets = extract(text, args.function, allow_leaf=args.leaf_block)
        if args.promote_registers:
            body = promote_registers(body, args.function)
    if not targets:
        # Only used to preserve a pre-existing TLS value in leaf replay. It is
        # never executed or written into the real capture.
        targets = ['spu_kernel_leaf_continuation']
    continuations = '\n'.join('void '+t+'(spu_context* ctx){(void)ctx;abort();}' for t in targets)
    harness = HARNESS.replace('@CONTINUATIONS@', continuations).replace('@FUNCTION@', body)
    harness = harness.replace('@GUARD_CHECKS@', GUARD_CHECKS if promoted_variant else '')
    harness = harness.replace('@BENCHMARK@', BENCHMARK.replace('@ITERATIONS@', str(args.benchmark_iterations))
                              .replace('@RUNS@', str(args.benchmark_runs))
                              if args.benchmark_iterations else '')
    harness = harness.replace('@NAME@', args.function).replace('@FIRST_CONTINUATION@', targets[0])
    cc = args.compiler or os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
    if not cc:
        raise SystemExit('pass --compiler with a native C compiler')
    with tempfile.TemporaryDirectory(prefix='spu-kernel-replay-') as folder:
        source, binary = Path(folder)/'replay.c', Path(folder)/('replay.exe' if os.name=='nt' else 'replay')
        source.write_text(harness)
        command = [cc, '-std=gnu17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
                   '-ffunction-sections', '-fdata-sections',
                   '-I'+str(ROOT/'include'), '-I'+str(ROOT), '-I'+str(ROOT/'runtime/spu'),
                   *args.cflag, str(source), '-o', str(binary)]
        command += ['-fuse-ld=lld', '-Wl,/opt:ref', '-Wl,/opt:noicf'] if os.name=='nt' else ['-lm']
        subprocess.run(command, check=True, timeout=90)
        print('capture sha256='+hashlib.sha256(data).hexdigest(), flush=True)
        before, after = data[40:40+SNAPSHOT_BYTES], data[40+SNAPSHOT_BYTES:]
        changed_ls = sum(a != b for a, b in zip(before[13*4+128*16:], after[13*4+128*16:]))
        a = struct.unpack_from('<512I', before, 13*4)
        b = struct.unpack_from('<512I', after, 13*4)
        actual_pc = struct.unpack_from('<I', before)[0]
        print(f'capture image={h[2]} logical=0x{h[3]:05X} architectural-entry=0x{actual_pc:05X} changedLS={changed_ls} bytes '
              f'changedGPR={sum(x != y for x, y in zip(a,b))} words', flush=True)
        print('variant='+('promoted-registers' if promoted_variant else 'lifted-reference'), flush=True)
        result = subprocess.run([str(binary), str(args.capture.resolve())], timeout=args.timeout)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
