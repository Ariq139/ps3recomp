/* PushSync must accept a tag mask and order outstanding jobs, without
 * dispatching that mask as a descriptor or writing through spare registers. */
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

static HANDLE job_entered, release_job, sync_done;
static LONG dispatch_calls;
int spu_workload_dispatch_job(const uint8_t* image, uint32_t image_size,
                             uint32_t job_ea, uint32_t descriptor_size)
{
    assert(image && image_size == 16);
    assert(job_ea == 0x1400 && descriptor_size == 128);
    InterlockedIncrement(&dispatch_calls);
    SetEvent(job_entered);
    assert(WaitForSingleObject(release_job, 2000) == WAIT_OBJECT_0);
    return 1;
}
int32_t ps3_guest_memory_alloc(uint32_t size,uint32_t alignment,uint32_t* out) {(void)size;(void)alignment;*out=0;return -1;}
int32_t ps3_guest_memory_free(uint32_t ea) {(void)ea;return 0;}
void spurs_job_set_spu_num(uint32_t n) {(void)n;}
void spu_workload_job_cache_thread_cleanup(void) {}
int spu_workload_dispatch_job_snapshot(const uint8_t* image,uint32_t size,uint32_t ea,uint32_t ds,const uint8_t* descriptor) {
 (void)descriptor;return spu_workload_dispatch_job(image,size,ea,ds);
}
static DWORD WINAPI submit(void* unused)
{
    (void)unused;
    assert(cellSpursJobQueuePush((CellSpursJobQueue*)(uintptr_t)0x1000,
                               (void*)(uintptr_t)0x1400, 128, 7, NULL) == 0);
    return 0;
}
static DWORD WINAPI sync_port(void* unused)
{
    (void)unused;
    assert(cellSpursJobQueuePort2PushSync(0x1200, 1u << 7, 0,
                                        0, 0x1600, 0, 0, 0) == 0);
    SetEvent(sync_done);
    return 0;
}
int main(void)
{
    assert(cellSpursJobQueuePort2Create((CellSpursJobQueuePort*)(uintptr_t)0x1200,
                                      (CellSpursJobQueue*)(uintptr_t)0x1000) == 0);
    /* This exact three-register shape aborted P5's battle-loading worker. */
    assert(cellSpursJobQueuePort2PushSync(0x1200, 1u << 7, 0, 0, 0, 0, 0, 0) == 0);
    assert((s32)cellSpursJobQueuePort2PushSync(0, 1, 0, 0, 0, 0, 0, 0) ==
           (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT);
    assert((s32)cellSpursJobQueuePort2PushSync(0x1300, 1, 0, 0, 0, 0, 0, 0) ==
           (s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED);

    job_entered = CreateEvent(NULL, TRUE, FALSE, NULL);
    release_job = CreateEvent(NULL, TRUE, FALSE, NULL);
    sync_done = CreateEvent(NULL, TRUE, FALSE, NULL);
    vm_write64(0x1400, 0x1800);
    vm_write16(0x1408, 1);
    vm_write32(0x1600, 0xA5A5A5A5);
    HANDLE producer = CreateThread(NULL, 0, submit, NULL, 0, NULL);
    assert(producer && WaitForSingleObject(job_entered, 1000) == WAIT_OBJECT_0);
    HANDLE waiter = CreateThread(NULL, 0, sync_port, NULL, 0, NULL);
    assert(waiter && WaitForSingleObject(sync_done, 50) == WAIT_TIMEOUT);
    SetEvent(release_job);
    assert(WaitForSingleObject(producer, 1000) == WAIT_OBJECT_0);
    assert(WaitForSingleObject(waiter, 1000) == WAIT_OBJECT_0);
    assert(dispatch_calls == 1 && vm_read32(0x1600) == 0xA5A5A5A5);
    assert(s_queues[0].submitCount == 1 && s_queues[0].completeCount == 1);
    assert(cellSpursShutdownJobQueue()==0);
    CloseHandle(producer); CloseHandle(waiter);
    CloseHandle(job_entered); CloseHandle(release_job); CloseHandle(sync_done);
    puts("job queue sync: tag mask ABI, completion ordering, no extra job/writes PASS");
    return 0;
}
