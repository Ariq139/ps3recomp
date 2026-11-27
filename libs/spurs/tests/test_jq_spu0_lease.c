/* A synchronous CopyPush fallback must not overlap worker 0 on another queue. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include "../cellSpursJq.c"

static uint8_t guest[0x4000];
uint8_t* vm_base = guest;
int g_resv_store_active;
uint32_t g_ww_lo, g_ww_hi;
void ppu_resv_break_store(uint64_t ea) { (void)ea; }
void ps3_ww_report_inline(uint32_t a, uint64_t v, int w) { (void)a; (void)v; (void)w; }
int spu_coh_is_reserved(uint32_t a) { (void)a; return 0; }
void spu_coh_notify_write(uint32_t a) { (void)a; }
void spu_lockline_lock(void) {}
void spu_lockline_unlock(void) {}
s32 sys_ppu_thread_yield(void) { SwitchToThread(); return 0; }
int32_t ps3_guest_memory_alloc(uint32_t n, uint32_t a, uint32_t* out)
{ (void)n; (void)a; *out = 0; return -1; }
int32_t ps3_guest_memory_free(uint32_t ea) { (void)ea; return 0; }
void spurs_job_set_spu_num(uint32_t n) { assert(n == 0); }
void spu_workload_job_cache_thread_cleanup(void) {}

static HANDLE worker_entered, release_worker, fallback_entered;
int spu_workload_dispatch_job_snapshot(const uint8_t* image, uint32_t size,
                                        uint32_t ea, uint32_t ds, const uint8_t* desc)
{
    (void)image; (void)size; (void)ds; (void)desc;
    if (ea == 0x1400) {
        SetEvent(worker_entered);
        assert(WaitForSingleObject(release_worker, 5000) == WAIT_OBJECT_0);
    } else {
        assert(ea == 0x1500);
        SetEvent(fallback_entered);
    }
    return 1;
}
static DWORD WINAPI fallback_submit(void* unused)
{
    (void)unused;
    assert(jq_push_slot_mode(1, 0x1500, 128, 0, 0, 1) == CELL_OK);
    return 0;
}
int main(void)
{
    assert(_putenv_s("SPURS_JQ_WORKERS", "1") == 0);
    worker_entered = CreateEvent(NULL, TRUE, FALSE, NULL);
    release_worker = CreateEvent(NULL, TRUE, FALSE, NULL);
    fallback_entered = CreateEvent(NULL, TRUE, FALSE, NULL);
    assert(worker_entered && release_worker && fallback_entered);
    assert(jq_create_state(0x1000, 4) == 0);
    assert(jq_create_state(0x1100, 4) == 1);
    for (unsigned ea = 0x1400; ea <= 0x1500; ea += 0x100) {
        vm_write64(ea, 0x1800); vm_write16(ea + 8, 1);
    }
    assert(jq_push_slot_mode(0, 0x1400, 128, 0, 0, 0) == CELL_OK);
    assert(WaitForSingleObject(worker_entered, 1000) == WAIT_OBJECT_0);
    HANDLE producer = CreateThread(NULL, 0, fallback_submit, NULL, 0, NULL);
    assert(producer);
    int admitted = 0;
    for (unsigned i = 0; i < 2000 && !admitted; ++i) {
        jq_lock(); admitted = s_queues[1].jobCount == 1; jq_unlock();
        if (!admitted) Sleep(1);
    }
    assert(admitted);
    assert(WaitForSingleObject(fallback_entered, 100) == WAIT_TIMEOUT);
    SetEvent(release_worker);
    assert(WaitForSingleObject(fallback_entered, 2000) == WAIT_OBJECT_0);
    assert(WaitForSingleObject(producer, 2000) == WAIT_OBJECT_0);
    assert(cellSpursShutdownJobQueue() == CELL_OK);
    CloseHandle(producer); CloseHandle(worker_entered);
    CloseHandle(release_worker); CloseHandle(fallback_entered);
    puts("JQ logical SPU 0 is exclusive between async and fallback jobs");
    return 0;
}
