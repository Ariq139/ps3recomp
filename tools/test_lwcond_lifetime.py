#!/usr/bin/env python3
"""Test experimental HLE logical mutex lifetime independently of slot allocation.

Uses real registry and HLE wait/signal handlers with stub guest memory and
mutex-slot operations. Does not validate raw LV2 or the actual host mutex queue.
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
#include <chrono>
#include <thread>
#include <atomic>
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
#define LWM_ATTR 8u
#define LWM_OWNER 0u
#define LWM_RECUR 12u
#define LWM_SELF(ctx) ((uint32_t)(ctx)->thread_id)
struct LwmSlot { volatile long state=0; std::mutex gate; std::condition_variable ready; };
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
    __atomic_compare_exchange_n(p,&expected,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);
    return expected;
}
static void ppu_thread_note_block(uint64_t) {}
extern "C" int32_t ppu_thread_priority_snapshot(uint64_t tid) { return tid == 3 ? 100 : 1000; }
static int lwm_acquire(LwmSlot* sl, uint64_t) {
    std::unique_lock<std::mutex> hold(sl->gate);
    sl->ready.wait(hold,[&]{return InterlockedCompareExchange(&sl->state,1,0)==0;});
    return 1;
}
static void lwm_release(LwmSlot* sl) {
    std::lock_guard<std::mutex> hold(sl->gate);
    __atomic_store_n(&sl->state,0,__ATOMIC_SEQ_CST); sl->ready.notify_one();
}
static void sys_lwmutex_create(ppu_context* c) { slot.state=0; c->gpr[3]=CELL_OK; }
static void sys_lwmutex_destroy_counted(ppu_context* c) { c->gpr[3]=CELL_OK; }
static void sys_lwmutex_lock(ppu_context* c) {
    lwm_acquire(&slot,0); vm_write32(MUTEX,LWM_SELF(c)); vm_write32(MUTEX+LWM_RECUR,1); c->gpr[3]=CELL_OK;
}
static void sys_lwmutex_trylock(ppu_context* c) {
    if (InterlockedCompareExchange(&slot.state,1,0)) {c->gpr[3]=CELL_EBUSY;return;}
    vm_write32(MUTEX,LWM_SELF(c)); vm_write32(MUTEX+LWM_RECUR,1); c->gpr[3]=CELL_OK;
}
static void sys_lwmutex_unlock(ppu_context* c) {
    vm_write32(MUTEX,0); vm_write32(MUTEX+LWM_RECUR,0); lwm_release(&slot); c->gpr[3]=CELL_OK;
}
REGISTRY
DESTROY
static uint32_t call(void (*fn)(ppu_context*),uint32_t a,uint64_t b=0,uint64_t tid=1) {
    ppu_context c{}; c.thread_id=tid; c.gpr[3]=a; c.gpr[4]=b; fn(&c); return (uint32_t)c.gpr[3];
}
static void wait_queued(unsigned count) {
    for (unsigned n=0;n<3000;++n) {
        {std::lock_guard<std::mutex> hold(s_lwc_mutex);
         auto* c=lwc_find(COND); unsigned ready=0;
         for(auto* w=c->head;w;w=w->next) if(!w->notified) ++ready;
         if(ready==count) return;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(!"waiter did not queue");
}
static void wait_contract() {
    using namespace std::chrono;
    assert(lwc_events_mode()==2);
    assert(call(lwc_mutex_create_checked,MUTEX)==CELL_OK);
    assert(call(lwc_create_event,COND,MUTEX)==CELL_OK);
    assert(call(lwc_mutex_lock_checked,MUTEX)==CELL_OK);
    const auto start=steady_clock::now();
    assert(call(lwc_wait_event,COND,30000)==CELL_ETIMEDOUT);
    assert(steady_clock::now()-start>=milliseconds(25));
    assert(vm_read32(MUTEX)==1 && vm_read32(MUTEX+LWM_RECUR)==1);
    assert(call(lwc_mutex_unlock_checked,MUTEX)==CELL_OK);
    assert(call(lwc_signal_event,COND)==CELL_OK); // no saved notification
    std::atomic<unsigned> done{0};
    auto waiter=[&](uint64_t tid){
        assert(call(lwc_mutex_lock_checked,MUTEX,0,tid)==CELL_OK);
        assert(call(lwc_wait_event,COND,0,tid)==CELL_OK);
        assert(vm_read32(MUTEX)==tid && vm_read32(MUTEX+LWM_RECUR)==1);
        done.fetch_add(1);
        assert(call(lwc_mutex_unlock_checked,MUTEX,0,tid)==CELL_OK);
    };
    std::thread a(waiter,2); wait_queued(1);
    std::thread b(waiter,3); wait_queued(2);
    std::this_thread::sleep_for(milliseconds(15));
    assert(done.load()==0);
    assert(call(lwc_destroy_event,COND)==CELL_EBUSY);
    assert(call(lwc_signal_to_event,COND,99)==CELL_ESRCH);
    assert(call(lwc_signal_to_event,COND,3)==CELL_OK);
    b.join(); assert(done.load()==1);
    assert(call(lwc_signal_all_event,COND)==CELL_OK);
    a.join(); assert(done.load()==2);
    // Verify single-signal selection with the queue held in a known order.
    for (uint32_t protocol : {1u, 2u}) {
        done.store(0);
        vm_write32(MUTEX+LWM_ATTR,protocol);
        std::thread first(waiter,2); wait_queued(1);
        std::thread second(waiter,3); wait_queued(2);
        assert(call(lwc_signal_event,COND)==CELL_OK);
        if (protocol==1) first.join(); // FIFO: oldest first
        else second.join();            // priority: tid 3 has lower priority number
        assert(done.load()==1);
        assert(call(lwc_signal_event,COND)==CELL_OK);
        if (first.joinable()) first.join();
        if (second.joinable()) second.join();
        assert(done.load()==2);
    }
    assert(call(lwc_destroy_event,COND)==CELL_OK);
    assert(call(lwc_mutex_destroy_checked,MUTEX)==CELL_OK);
    assert(call(lwc_mutex_lock_checked,MUTEX)==CELL_ESRCH);
}
int main() {
    assert(lwm_find(MUTEX)); // a slot alone must not prove a live mutex
    assert(call(lwc_create_event,COND,MUTEX)==CELL_ESRCH);
    assert(call(lwc_mutex_create_checked,MUTEX)==CELL_OK);
    uint64_t first=s_lwc_live_mutexes.at(MUTEX).generation;
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
    assert(s_lwc_live_mutexes.at(MUTEX).generation!=first);
    s_lwc.emplace(COND,stale); // stale generation with matching guest words
    vm_write32(COND,MUTEX); vm_write32(COND+4,stale.id);
    assert(!lwc_find(COND));
    s_lwc.erase(COND);
    assert(call(lwc_create_event,COND,MUTEX)==CELL_OK && lwc_find(COND));
    assert(call(lwc_destroy_event,COND)==CELL_OK);
    assert(call(lwc_mutex_destroy_checked,MUTEX)==CELL_OK);
    wait_contract();
    puts("PASS: logical mutex lifetime, exact HLE timeout, notification and reacquisition checks");
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
    end = src.index("/* sys_ppu_thread_get_id", start)
    code = HARNESS.replace("REGISTRY", src[start:end]).replace(
        "DESTROY", extract(src, "lwc_mutex_destroy_checked"))
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "lifetime.cpp"
        exe = Path(tmp) / ("lifetime.exe" if os.name == "nt" else "lifetime")
        c.write_text(code)
        flags = [] if os.name == "nt" else ["-pthread"]
        subprocess.run([cxx, "-std=c++17", "-O2", "-UNDEBUG", "-I" + str(ROOT / "include"),
                        *flags, str(c), "-o", str(exe)], check=True)
        env = dict(os.environ, PS3RECOMP_LWCOND_EVENTS="hle-strict")
        subprocess.run([str(exe)], check=True, timeout=30, env=env)


if __name__ == "__main__":
    main()
