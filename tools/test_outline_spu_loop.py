#!/usr/bin/env python3
"""Check pure-loop CFG validation, unchanged bytes and cache build chaining."""
import argparse
import hashlib
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from outline_spu_loop import outline
from promote_spu_kernel import transform

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--compiler')
args=parser.parse_args()
cc=args.compiler or os.environ.get('CC') or shutil.which('clang') or shutil.which('cc')
if not cc: raise SystemExit('pass --compiler')
name='p5_job_A96F7BFC_spu_func_00000040'
actual=ROOT/'p5_feb16/spu/spu_job_A96F7BFCEB376D57/spu_recomp.c'
if actual.exists():
    data=actual.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    assert outline(data,name,0x654,0x658,0x1324,'OFF')[0]==data
    for mode in ('CAPTURE','NATIVE'):
        out,info=outline(data,name,0x654,0x658,0x1324,mode)
        assert info['instructions']==820 and info['source_sha256']==digest
        assert actual.read_bytes()==data
        # Source prefix and suffix remain byte-identical apart from inserted
        # helpers/include; every neighboring function is preserved intact.
        text=data.decode(); begin=text.index('void '+name+'(')
        end=text.index('\nvoid ',begin+1)
        assert out.endswith(data[end:])
        # CAPTURE must retain every original loop byte, including labels and
        # instruction order; promotion is separately replay-verified.
        if mode=='CAPTURE':
            raw=text[text.index('loc_00000654:',begin):]
            close=re.search(r'[^\n]*if[^\n]*goto loc_00000654;[^\n]*\n',raw)
            assert close and raw[:close.end()].encode() in out
        prefix=data[:begin]
        if mode=='CAPTURE':
            nl=b'\r\n' if b'\r\n' in data else b'\n'
            assert out.startswith(b'#include "spu_kernel_capture_api.h"'+nl+prefix)
        else: assert out.startswith(prefix)
    print('actual A96 loop: 820 instructions; OFF identity; neighboring source unchanged PASS')

SOURCE='''#include "image_header.h"
void kernel(spu_context* ctx) {
    goto loc_00000104;
loc_00000100:
    ctx->gpr[3] = spu_ai(ctx->gpr[3], 1);
loc_00000104:
    ctx->gpr[4] = spu_ai(ctx->gpr[3], 2);
    spu_ls_write128(ctx, 0, ctx->gpr[4]);
    if (ctx->gpr[4]._u32[0] < 5) goto loc_00000100;
    ctx->pc = 0x110; g_spu_trampoline_fn = after; return;
}
'''

def emit(source=SOURCE,mode='NATIVE',exit_pc=0x110):
    return outline(source.encode(),'kernel',0x100,0x104,exit_pc,mode)[0]

negatives=[
 SOURCE.replace("    goto loc_00000104;", "    goto *target;\n    goto loc_00000104;"),
 SOURCE.replace("    goto loc_00000104;", "    target = &&loc_00000104;\n    goto loc_00000104;"),
 SOURCE.replace('    goto loc_00000104;', '    if (ctx->gpr[7]._u32[0]) goto loc_00000104;'),
 SOURCE.replace('    goto loc_00000104;', '    if (ctx->gpr[7]._u32[0])\n        goto loc_00000104;'),
 SOURCE.replace('    goto loc_00000104;', '    if (ctx->gpr[7]._u32[0]) {\n        goto loc_00000104;\n    }'),
 SOURCE.replace('    goto loc_00000104;','    if (ctx->gpr[1]._u32[0]) goto loc_00000100;\n    goto loc_00000104;'),
 SOURCE.replace('    goto loc_00000104;','    ctx->gpr[2] = spu_il(0);'),
 SOURCE.replace('    spu_ls_write128(ctx, 0, ctx->gpr[4]);','    goto loc_00000110;'),
 SOURCE.replace('    spu_ls_write128(ctx, 0, ctx->gpr[4]);','    spu_wrch(ctx, 0, ctx->gpr[4]);'),
 SOURCE.replace('    spu_ls_write128(ctx, 0, ctx->gpr[4]);','    ctx->ls = external_ls;'),
 SOURCE.replace('    spu_ls_write128(ctx, 0, ctx->gpr[4]);','    (ctx->gpr[4]) = spu_il(0);'),
 SOURCE.replace('if (ctx->gpr[4]._u32[0] < 5) goto loc_00000100;', 'goto loc_00000100;'),
 SOURCE.replace('    spu_ls_write128(ctx, 0, ctx->gpr[4]);','    if (ctx->gpr[0]._u32[0]) goto loc_00000100;'),
]
for bad in negatives:
    try: emit(bad)
    except ValueError: pass
    else: raise AssertionError('unsupported CFG/body accepted')
try: emit(exit_pc=0x114)
except ValueError: pass
else: raise AssertionError('wrong instruction-count boundary accepted')
print('fourteen unsupported CFG/alias/channel/boundary cases rejected PASS')

DRIVER=r'''
#include "spu_helpers.h"
#include "tests/spu_ls_watch_fixture.h"
#include <assert.h>
#include <string.h>
SPU_THREAD_LOCAL void (*g_spu_trampoline_fn)(spu_context*);
void after(spu_context* c){(void)c;}
static unsigned captures;
void spu_kernel_capture_invoke(spu_context* c,uint32_t id,void(*fn)(spu_context*)){
 assert(id==0x104 && c->pc==0x40);++captures;fn(c);assert(c->pc==0x40);
}
#define kernel baseline
#include "original.c"
#undef kernel
#include "outlined.c"
int main(void){
 static spu_context a,b;spu_context_init(&a,0);spu_context_init(&b,0);
 a.gpr[3]._u32[0]=b.gpr[3]._u32[0]=1;a.pc=b.pc=0x40;
 baseline(&a);kernel(&b);
 assert(a.gpr[3]._u32[0]==3 && a.gpr[4]._u32[0]==5);
 assert(!memcmp(a.gpr,b.gpr,sizeof(a.gpr)) && !memcmp(a.ls,b.ls,SPU_LS_SIZE));
 assert(a.pc==b.pc && a.status==b.status && a.stop_code==b.stop_code);
 assert(captures==EXPECTED_CAPTURE);
 puts("outlined loop real reference/native execution and metadata preservation PASS");return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='spu-loop-outline-') as folder:
    folder=Path(folder)
    (folder/'image_header.h').write_text('#pragma once\n#include "spu_helpers.h"\nvoid after(spu_context*);\n')
    (folder/'original.c').write_text(SOURCE)
    (folder/'driver.c').write_text(DRIVER)
    for newline in ('\n','\r\n'):
        for mode in ('OFF','CAPTURE','NATIVE'):
            source=SOURCE.replace('\n',newline)
            (folder/'outlined.c').write_bytes(emit(source,mode))
            exe=folder/('run.exe' if os.name=='nt' else 'run')
            command=[cc,'-std=gnu17','-O2','-UNDEBUG','-D_CRT_SECURE_NO_WARNINGS',
                     '-DEXPECTED_CAPTURE='+('1' if mode=='CAPTURE' else '0'),
                     '-I'+str(folder),'-I'+str(ROOT),'-I'+str(ROOT/'include'),
                     '-I'+str(ROOT/'runtime/spu'),'-I'+str(ROOT/'runtime/platform'),
                     str(folder/'driver.c'),'-o',str(exe)]
            if os.name!='nt': command += ['-lm']
            subprocess.run(command,check=True)
            subprocess.run([str(exe)],check=True,timeout=5)
    # The shared CMake helper must chain after another source transform and
    # preserve per-source flags, definitions and original header lookup.
    for mode in ('OFF','CAPTURE','NATIVE'):
        project=folder/('cmake-'+mode);project.mkdir()
        (project/'image_header.h').write_text((folder/'image_header.h').read_text())
        probe='#if !defined(OUTLINE_PROBE) || !defined(__AVX2__)\n#error lost source compile properties\n#endif\n'+SOURCE+'\nvoid cull(spu_context* ctx) { ctx->gpr[1] = spu_il(1); ctx->pc = 0x20; g_spu_trampoline_fn = after; return; }\n'
        (project/'original.c').write_text(probe)
        cmake='''cmake_minimum_required(VERSION 3.20)
project(OutlineProbe C)
set(PS3RECOMP_DIR "@ROOT@")
add_library(probe OBJECT original.c)
set_source_files_properties(original.c PROPERTIES COMPILE_DEFINITIONS OUTLINE_PROBE=1 COMPILE_OPTIONS "-mavx2" INCLUDE_DIRECTORIES "@ROOT@/include;@ROOT@/runtime/spu;@ROOT@/runtime/platform")
include("@ROOT@/cmake/PS3RecompSPUKernel.cmake")
ps3recomp_promote_spu_kernel(probe original.c cull first_copy)
ps3recomp_outline_spu_loop(probe "${first_copy}" kernel 0x100 0x104 0x110 @MODE@ final_copy)
'''.replace('@ROOT@',ROOT.as_posix()).replace('@MODE@',mode)
        (project/'CMakeLists.txt').write_text(cmake)
        build=project/'build'
        subprocess.run(['cmake','-S',str(project),'-B',str(build),'-G','Ninja',
                        '-DCMAKE_C_COMPILER='+cc],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['cmake','--build',str(build)],check=True,stdout=subprocess.DEVNULL)
    print('CMake OFF/CAPTURE/NATIVE cache chain, AVX2/source definitions/includes PASS')
