#!/usr/bin/env python3
"""Exercise the actual lightweight-mutex slot lookup with a single-thread harness.

The existing Windows/compat synchronization implementation is not replaced.
This fixture extracts the table, hash and lookup functions, and substitutes
only the platform spinlock primitive for a single-threaded test.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def extract(src, name):
    m = re.search(r"(?m)^static (?:inline )?[^\n]*\b" + name + r"\([^;\n]*\)\s*\{", src)
    assert m, name
    i, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[m.start():i]


HARNESS = r'''
#include <cstdint>
#include <cstdio>
#include <cassert>
using HANDLE = void*;
static long _InterlockedExchange(volatile long* p, long n) {
    long old = *p; *p = n; return old;
}
static void YieldProcessor() {}
TABLE
static volatile long g_lwm_tab_lock = 0;
FIND
SLOT
int main() {
    assert(lwm_slot(0) == nullptr && lwm_find(0) == nullptr);
    /* Identical low 16 bits; existing and new slot paths must agree. */
    constexpr unsigned N = 1024;
    LwmSlot* slots[N];
    for (unsigned i = 0; i < N; ++i) {
        uint32_t a = 0x1000u + i * 0x10000u;
        assert(!lwm_find(a));
        slots[i] = lwm_slot(a);
        assert(slots[i] && uint32_t(slots[i]->addr) == a);
        assert(lwm_find(a) == slots[i]);
        slots[i]->state = long(i + 1);
    }
    for (unsigned pass = 0; pass < 4; ++pass) {
        for (unsigned i = 0; i < N; ++i) {
            unsigned j = (i * 73u) % N;
            uint32_t a = 0x1000u + j * 0x10000u;
            assert(lwm_slot(a) == slots[j]);
            assert(lwm_slot(a) == slots[j]);  // warm thread-local hit
            assert(lwm_find(a) == slots[j] && slots[j]->state == long(j + 1));
        }
    }
    // Fill every remaining slot. Failed lookups may not create entries
    // or invalidate a previously cached pointer when the table is full.
    for (unsigned i = 0; i < LWM_HASH - N; ++i) {
        uint32_t a = 0x80000000u + i * 4u;
        LwmSlot* sl = lwm_slot(a);
        assert(sl && uint32_t(sl->addr) == a);
    }
    assert(!lwm_slot(0xf0000000u) && !lwm_find(0xf0000000u));
    assert(lwm_slot(0x1000u) == slots[0] && slots[0]->state == 1);
    puts("PASS: lwmutex slot hash, cache, pointer identity and full table");
}
'''


def main():
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        if os.environ.get("CI"):
            raise SystemExit("FAIL: no C++ compiler")
        print("SKIP: no C++ compiler")
        return
    src = (ROOT / "runtime/ppu/ppu_sysprx.cpp").read_text()
    start = src.index("#define LWM_HASH_BITS")
    end = src.index("extern \"C\" { extern volatile long long g_gcm_ref_pub_count;", start)
    code = HARNESS.replace("TABLE", src[start:end]).replace("FIND", extract(src, "lwm_find")).replace("SLOT", extract(src, "lwm_slot"))
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "t.cpp"
        exe = Path(tmp) / ("t.exe" if os.name == "nt" else "t")
        c.write_text(code)
        subprocess.run([cxx, "-std=c++17", "-O2", "-UNDEBUG", str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == "__main__":
    main()
