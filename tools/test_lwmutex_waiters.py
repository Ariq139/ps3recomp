#!/usr/bin/env python3
"""Exercise the lwmutex FIFO handoff, timeout removal and failed-event fallback.

The fixture compiles the actual queue and lock helpers from ppu_sysprx.cpp.
One build uses real Windows events; the second forces event creation/waits to
fail and checks that the grant flag plus bounded backoff still make progress.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def extract(src, name):
    match = re.search(
        r"(?m)^static (?:inline )?[^\n]*\b" + name + r"\([^;\n]*\)\s*\{", src
    )
    assert match, name
    i, depth = match.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[match.start():i]


HARNESS = r'''
#include <windows.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#ifdef TEST_INVALID_EVENT
static HANDLE fixture_CreateEventA(LPSECURITY_ATTRIBUTES, BOOL, BOOL, LPCSTR) { return nullptr; }
static DWORD fixture_WaitForSingleObject(HANDLE, DWORD) { return WAIT_FAILED; }
static BOOL fixture_SetEvent(HANDLE) { return FALSE; }
#define CreateEventA fixture_CreateEventA
#define WaitForSingleObject fixture_WaitForSingleObject
#define SetEvent fixture_SetEvent
#endif

/* LWM_STRUCTS */
/* LWM_NOW */
/* LWM_TRACE */
/* LWM_TRACE_FOR */
/* LWM_QLOCK */
/* LWM_QUNLOCK */
/* LWM_EVENT */
/* LWM_TRY */
/* LWM_ACQUIRE */
/* LWM_RELEASE */

static size_t waiter_count(LwmSlot* slot)
{
    size_t count = 0;
    lwm_qlock(slot);
    for (LwmWaiter* waiter = slot->qhead; waiter; waiter = waiter->next) ++count;
    lwm_qunlock(slot);
    return count;
}

static void await_waiter_count(LwmSlot* slot, size_t expected)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (waiter_count(slot) == expected) return;
        Sleep(1);
    }
    std::fprintf(stderr, "timed out waiting for %zu queued waiter(s)\n", expected);
    std::abort();
}

int main()
{
    const uint32_t target = 0x015AA1C0u;
    const auto trace_expected = [](uint32_t address) {
        if (!getenv("PS3_LWMUTEX_TRACE")) return false;
        const char* setting = getenv("PS3_LWMUTEX_TRACE_ADDR");
        const uint32_t watched = setting ? (uint32_t)strtoul(setting, nullptr, 0) : 0u;
        return !watched || watched == address;
    };
    assert(lwm_trace_for(target) == trace_expected(target));
    assert(lwm_trace_for(target + 4) == trace_expected(target + 4));

    LwmSlot slot{};
    assert(lwm_try(&slot));

    std::mutex result_mutex;
    std::condition_variable result_changed;
    std::vector<int> acquisition_order;
    std::atomic<bool> release_first{false};

    std::thread first([&] {
        assert(lwm_acquire(&slot, 0));
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            acquisition_order.push_back(1);
        }
        result_changed.notify_all();
        while (!release_first.load(std::memory_order_acquire)) Sleep(1);
        lwm_release(&slot);
    });
    await_waiter_count(&slot, 1);

    std::thread second([&] {
        assert(lwm_acquire(&slot, 0));
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            acquisition_order.push_back(2);
        }
        result_changed.notify_all();
        lwm_release(&slot);
    });
    await_waiter_count(&slot, 2);

    lwm_release(&slot);  /* hand the held lock to the first queued waiter */
    {
        std::unique_lock<std::mutex> lock(result_mutex);
        assert(result_changed.wait_for(lock, std::chrono::seconds(2), [&] {
            return !acquisition_order.empty();
        }));
        assert(acquisition_order.front() == 1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::lock_guard<std::mutex> lock(result_mutex);
        assert(acquisition_order.size() == 1); /* second cannot steal the handoff */
    }
    release_first.store(true, std::memory_order_release);
    {
        std::unique_lock<std::mutex> lock(result_mutex);
        assert(result_changed.wait_for(lock, std::chrono::seconds(2), [&] {
            return acquisition_order.size() == 2;
        }));
    }
    first.join();
    second.join();
    assert(acquisition_order[0] == 1 && acquisition_order[1] == 2);
    assert(slot.state.load(std::memory_order_acquire) == 0);

    LwmSlot timed{};
    assert(lwm_try(&timed));
    std::atomic<bool> timed_out{false};
    std::thread timeout_waiter([&] {
        timed_out.store(!lwm_acquire(&timed, 30000), std::memory_order_release);
    });
    await_waiter_count(&timed, 1);
    timeout_waiter.join();
    assert(timed_out.load(std::memory_order_acquire));
    assert(waiter_count(&timed) == 0);
    assert(timed.state.load(std::memory_order_acquire) == 1);
    lwm_release(&timed);
    assert(timed.state.load(std::memory_order_acquire) == 0);

    std::puts("PASS: FIFO handoff, timeout unlink and event-failure progress");
    return 0;
}
'''


def main():
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise SystemExit("FAIL: no C++17 compiler")

    src = (ROOT / "runtime/ppu/ppu_sysprx.cpp").read_text()
    struct_start = src.index("struct LwmWaiter {")
    struct_end = src.index('\nextern "C" { long long gcm_ref_pub_count_snapshot(void);', struct_start)
    structs = (src[struct_start:struct_end]
               .replace("static struct LwmSlot {", "struct LwmSlot {")
               .replace("} g_lwm[LWM_HASH];", "};"))

    source = HARNESS
    replacements = {
        "/* LWM_STRUCTS */": structs,
        "/* LWM_NOW */": extract(src, "lwm_now_us"),
        "/* LWM_TRACE */": extract(src, "lwm_trace"),
        "/* LWM_TRACE_FOR */": extract(src, "lwm_trace_for"),
        "/* LWM_QLOCK */": extract(src, "lwm_qlock"),
        "/* LWM_QUNLOCK */": extract(src, "lwm_qunlock"),
        "/* LWM_EVENT */": extract(src, "lwm_thread_event"),
        "/* LWM_TRY */": extract(src, "lwm_try"),
        "/* LWM_ACQUIRE */": extract(src, "lwm_acquire"),
        "/* LWM_RELEASE */": extract(src, "lwm_release"),
    }
    for marker, code in replacements.items():
        source = source.replace(marker, code)

    with tempfile.TemporaryDirectory(prefix="lwmutex-waiters-") as tmp:
        cpp = Path(tmp) / "test.cpp"
        cpp.write_text(source)
        for invalid_event in (False, True):
            exe = Path(tmp) / ("failed-event.exe" if invalid_event else "real-event.exe")
            command = [compiler, "-std=c++17", "-O2", "-UNDEBUG", "-Wno-deprecated-declarations", str(cpp), "-o", str(exe)]
            if invalid_event:
                command.insert(1, "-DTEST_INVALID_EVENT")
            subprocess.run(command, check=True, timeout=60)
            for trace_mode in ("filtered", "all", "off"):
                env = os.environ.copy()
                env.pop("PS3_LWMUTEX_TRACE", None)
                env.pop("PS3_LWMUTEX_TRACE_ADDR", None)
                if trace_mode != "off":
                    env["PS3_LWMUTEX_TRACE"] = "1"
                if trace_mode == "filtered" or trace_mode == "off":
                    env["PS3_LWMUTEX_TRACE_ADDR"] = "0x015AA1C0"
                run = subprocess.run([str(exe)], check=True, timeout=10, capture_output=True, text=True, env=env)
                print(("invalid-event" if invalid_event else "real-event") + "/" + trace_mode + ": " + run.stdout.strip())


if __name__ == "__main__":
    main()
