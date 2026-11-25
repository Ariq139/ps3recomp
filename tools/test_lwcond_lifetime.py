#!/usr/bin/env python3
"""Test experimental HLE logical mutex lifetime independently of slot allocation.

Uses the real registry/create/destroy handlers with stub guest memory and slot
operations. It does not validate the full guest wait/signal or raw LV2 ABI.
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(src, name):
    m = re.search(r"(?m)^static void " + name + r"\(ppu_context\* ctx\)\s*\{", src)
    assert m, name
    i, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[m.start():i]


HARNESS = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <new>
#include "ps3emu/error_codes.h"
struct ppu_context { uint64_t gpr[32]{}; uint64_t thread_id=1; };
static uint8_t mem[0x10000];
static uint32_t vm_read32(uint32_t a) {
    return uint32_t(mem[a])<<24 | uint32_t(mem[a+1])<<16 | uint32_t(mem[a+2])<<8 | mem[a+3];
}
static void vm_write32(uint32_t a,uint32_t v) {
    for(unsigned i=0;i<4;++i) mem[a+i]=(uint8_t)(v>>(24-i*8));
}
#define LWM_SIZE 0x18u
struct LwmSlot { volatile long state=0; };
static LwmSlot slot;  // permanently allocated, even when logically destroyed
static const uint32_t MUTEX=0x1000, COND=0x2000;
static LwmSlot* lwm_find(uint32_t a) { return a==MUTEX ? &slot : nullptr; }
static LwmSlot* lwm_slot(uint32_t a) { return lwm_find(a); }
static uint32_t lwm_guest_address_error(uint32_t a,uint32_t n) {
    if (!a || uint64_t(a)+n>sizeof mem) return CELL_EFAULT;
    if (a&3) return CELL_EALIGN;
    return CELL_OK;
}
static long InterlockedCompareExchange(volatile long* p,long value,long expected) {
    long old=*p; if(old==expected)*p=value; return old;
}
static void sys_lwmutex_create(ppu_context* c) { slot.state=0; c->gpr[3]=CELL_OK; }
static void sys_lwmutex_destroy_counted(ppu_context* c) { c->gpr[3]=CELL_OK; }
REGISTRY
DESTROY
static uint32_t call(void (*fn)(ppu_context*),uint32_t a,uint32_t b=0) {
    ppu_context c{}; c.gpr[3]=a; c.gpr[4]=b; fn(&c); return (uint32_t)c.gpr[3];
}
int main() {
    assert(lwm_find(MUTEX)); // a slot alone must not prove a live mutex
    assert(call(lwc_create_event,COND,MUTEX)==CELL_ESRCH);
    assert(call(lwc_mutex_create_checked,MUTEX)==CELL_OK);
    uint64_t first=s_lwc_live_mutexes.at(MUTEX);
    assert(call(lwc_mutex_create_checked,MUTEX)==CELL_EBUSY);
    assert(call(lwc_create_event,COND,MUTEX)==CELL_OK);
    assert(lwc_find(COND) && lwc_find(COND)->mutex_generation==first);
    assert(call(lwc_mutex_destroy_checked,MUTEX)==CELL_EBUSY);
    LwcState stale=*lwc_find(COND);
    assert(call(lwc_destroy_event,COND)==CELL_OK);
    slot.state=1;
    assert(call(lwc_mutex_destroy_checked,MUTEX)==CELL_EBUSY);
    slot.state=0;
    assert(call(lwc_mutex_destroy_checked,MUTEX)==CELL_OK);
    assert(lwm_find(MUTEX) && !s_lwc_live_mutexes.count(MUTEX));
    assert(call(lwc_create_event,COND,MUTEX)==CELL_ESRCH);
    assert(call(lwc_mutex_create_checked,MUTEX)==CELL_OK);
    assert(s_lwc_live_mutexes.at(MUTEX)!=first);
    s_lwc.emplace(COND,stale); // stale generation with matching guest words
    vm_write32(COND,MUTEX); vm_write32(COND+4,stale.id);
    assert(!lwc_find(COND));
    s_lwc.erase(COND);
    assert(call(lwc_create_event,COND,MUTEX)==CELL_OK && lwc_find(COND));
    assert(call(lwc_destroy_event,COND)==CELL_OK);
    assert(call(lwc_mutex_destroy_checked,MUTEX)==CELL_OK);
    puts("PASS: logical mutex lifetime, generation and paired condition checks");
}
'''


def main():
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        if os.environ.get("CI"):
            raise SystemExit("FAIL: no C++ compiler")
        print("SKIP: no C++ compiler")
        return
    src = (ROOT / "runtime/ppu/ppu_sysprx.cpp").read_text()
    start = src.index("static int lwc_events_mode()")
    end = src.index("static void lwc_wait_event", start)
    code = HARNESS.replace("REGISTRY", src[start:end]).replace(
        "DESTROY", extract(src, "lwc_mutex_destroy_checked"))
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "lifetime.cpp"
        exe = Path(tmp) / ("lifetime.exe" if os.name == "nt" else "lifetime")
        c.write_text(code)
        flags = [] if os.name == "nt" else ["-pthread"]
        subprocess.run([cxx, "-std=c++17", "-O2", "-UNDEBUG", "-I" + str(ROOT / "include"),
                        *flags, str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == "__main__":
    main()
