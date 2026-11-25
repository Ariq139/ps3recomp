#!/usr/bin/env python3
"""Source-extracted host dispatch state and rollback fixture, no game required."""
import os, subprocess, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'runtime/spu/spu_workload.c').read_text()
start=s.index('#define TS_TRACK 8')
end=s.index('/* CellSpursTaskset2 keeps',start)
harness=r'''
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdatomic.h>
typedef long LONG;
static LONG InterlockedExchange(volatile LONG* p,LONG v){return __atomic_exchange_n(p,v,__ATOMIC_SEQ_CST);}
static void Sleep(unsigned n){(void)n;}
static uint8_t mem[4096]; uint8_t* vm_base=mem;
static unsigned locked, notifications;
void spu_lockline_lock(void){assert(!locked);locked=1;}
void spu_lockline_unlock(void){assert(locked);locked=0;}
void spu_coh_notify_write(uint32_t ea){assert(locked&&ea==0x100);notifications++;}
'''
checks=r'''
int main(void) {
    const uint32_t ea=0x100,t=3;
    const uint8_t mask=0x10,other=0x20;
    spu_taskset_mark_running(ea,t);
    assert(ts_states(ea)[t]==TS_STARTING);
    spu_taskset_state(ea,t,TS_RUNNING);
    assert(ts_states(ea)[t]==TS_RUNNING);
    spu_taskset_state(ea,t,TS_UNAVAILABLE);
    assert(ts_states(ea)[t]==TS_UNAVAILABLE);
    spu_taskset_mark_running(ea,t);
    for(unsigned off=0;off<=0x50;off+=0x10)mem[ea+off]=mask|other;
    spu_taskset_abort_creation(ea,t);
    assert(ts_states(ea)[t]==TS_IDLE && notifications==1);
    for(unsigned off=0;off<=0x50;off+=0x10)assert(mem[ea+off]==other);
    puts("PASS SPU reservation transitions and atomic bitset rollback");
}
'''
with tempfile.TemporaryDirectory() as d:
    c=Path(d)/'test.c'; exe=Path(d)/('test.exe' if os.name=='nt' else 'test')
    c.write_text(harness+s[start:end]+checks)
    subprocess.run([os.environ.get('CC','cc'),'-std=gnu17','-UNDEBUG',str(c),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=10)
