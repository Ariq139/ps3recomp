/*
 * ps3recomp - cellSpursJq HLE implementation
 *
 * SPURS Job Queue — manages job submission and tracking.
 * Bounded asynchronous worker pool for lifted SPU job descriptors.
 * A single worker is the conservative default; concurrent lanes require
 * explicit opt-in until other SPU runtime global state is audited.
 */

#include "cellSpursJq.h"
#include "../../runtime/ppu/ppu_memory.h"   /* GUEST_PTR, vm_write*: guest EA -> host pointer */
#include "../../runtime/spu/spu_workload.h"
#include "../system/sysPrxForUser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>
#ifdef _WIN32
# include <windows.h>
#else
# include <pthread.h>
#endif

/* All queue, port, job-id, and pending-job state is protected by the pool
 * mutex, including in synchronous-compatibility mode. Never hold it while
 * calling lifted SPU code or guest memory callbacks. */
#define JQ_WORKER_MAX 6
#define CELL_SPURS_JQ_MAX_PORTS 64

typedef struct JqTask JqTask;
struct JqTask {
    JqTask* next;
    u32 queue_slot, tag, id, job_ea, descriptor_size, binary_size;
    uint64_t tag_serial;
    u32 owned_descriptor, flags;
    uint64_t sequence;
    uint8_t descriptor[256];
    uint8_t* binary;
};
typedef struct {
    int in_use, accepting, copy_sync_active;
    u32 handle_ea, maxJobs, jobCount, nextJobId, submitCount, completeCount;
    u32 completedIds[CELL_SPURS_JQ_MAX_JOBS];
    s32 completionCodes[CELL_SPURS_JQ_MAX_JOBS];
    u32 completedHead;
    JqTask* pending[CELL_SPURS_JQ_MAX_JOBS];
} JqState;
typedef struct { int in_use; u32 port_ea, queue_slot; } JqPortState;

static JqState s_queues[CELL_SPURS_JQ_MAX_QUEUES];
static JqPortState s_ports[CELL_SPURS_JQ_MAX_PORTS];
static atomic_uint s_active_job_dispatches;
static atomic_int s_queue_shutdown_requested;
static atomic_uint_fast64_t s_tag_submitted[CELL_SPURS_JQ_MAX_QUEUES][32];
static atomic_uint_fast64_t s_tag_completed[CELL_SPURS_JQ_MAX_QUEUES][32];
static uint64_t s_task_sequence;
static JqTask* s_task_head;
static JqTask* s_task_tail;
static int s_pool_workers, s_pool_stopping;
#ifdef _WIN32
static SRWLOCK s_jq_mutex = SRWLOCK_INIT;
static CONDITION_VARIABLE s_jq_cond = CONDITION_VARIABLE_INIT;
static HANDLE s_jq_threads[JQ_WORKER_MAX];
static void jq_lock(void) { AcquireSRWLockExclusive(&s_jq_mutex); }
static void jq_unlock(void) { ReleaseSRWLockExclusive(&s_jq_mutex); }
static void jq_notify(void) { WakeAllConditionVariable(&s_jq_cond); }
static void jq_await(void) { SleepConditionVariableSRW(&s_jq_cond, &s_jq_mutex, INFINITE, 0); }
#else
static pthread_mutex_t s_jq_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t s_jq_cond = PTHREAD_COND_INITIALIZER;
static pthread_t s_jq_threads[JQ_WORKER_MAX];
static void jq_lock(void) { pthread_mutex_lock(&s_jq_mutex); }
static void jq_unlock(void) { pthread_mutex_unlock(&s_jq_mutex); }
static void jq_notify(void) { pthread_cond_broadcast(&s_jq_cond); }
static void jq_await(void) { pthread_cond_wait(&s_jq_cond, &s_jq_mutex); }
#endif

static void jq_trace_port2(const char* op, u32 port_ea, u32 queue_ea,
                           u32 job_ea, u32 descriptor_size,
                           u32 descriptor_pool_size, u32 tag,
                           int slot, s32 rc)
{
    static atomic_uint emitted;
    if (!getenv("SPURS_JQ_TRACE")) return;
    if (atomic_fetch_add_explicit(&emitted, 1, memory_order_relaxed) >= 64) return;
    fprintf(stderr,
            "[spurs-jq] %s port=%08X align=%u queue=%08X job=%08X "
            "desc=%u pool=%u tag=%u slot=%d rc=%08X\n",
            op, port_ea, port_ea & (CELL_SPURS_JQ_PORT2_ALIGN - 1u),
            queue_ea, job_ea, descriptor_size, descriptor_pool_size, tag,
            slot, (u32)rc);
}

static int jq_requested_workers(void)
{
    /* Opt out with 0; 1 is an asynchronous PPU/offload lane. Multiple lanes
     * may expose other titles' non-thread-safe SPU globals: opt in deliberately. */
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("SPURS_JQ_WORKERS");
        int n = e ? atoi(e) : 1;
        if (n < 0) n = 0;
        if (n > JQ_WORKER_MAX) n = JQ_WORKER_MAX;
        cached = n;
    }
    return cached;
}
static s32 jq_status_for_id(const JqState* q, u32 id)
{
    for (u32 i = 0; i < CELL_SPURS_JQ_MAX_JOBS; ++i)
        if (q->completedIds[i] == id) return q->completionCodes[i];
    return (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND;
}
static int jq_pending_id(const JqState* q, u32 id)
{
    for (u32 i = 0; i < CELL_SPURS_JQ_MAX_JOBS; ++i)
        if (q->pending[i] && q->pending[i]->id == id) return 1;
    return 0;
}
static int jq_pending_tag_before(u32 slot, u32 mask, const uint64_t target[32])
{
    const JqState* q = &s_queues[slot];
    for (u32 i = 0; i < CELL_SPURS_JQ_MAX_JOBS; ++i) {
        const JqTask* t = q->pending[i];
        if (t && (mask & (1u << t->tag)) && t->tag_serial <= target[t->tag])
            return 1;
    }
    return 0;
}
/* Both sync and async completions use this path under the lock. */
static void jq_finish_locked(JqTask* t, s32 status)
{
    JqState* q = &s_queues[t->queue_slot];
    for (u32 i = 0; i < CELL_SPURS_JQ_MAX_JOBS; ++i)
        if (q->pending[i] == t) { q->pending[i] = NULL; break; }
    q->completedIds[q->completedHead] = t->id;
    q->completionCodes[q->completedHead] = status;
    q->completedHead = (q->completedHead + 1u) % CELL_SPURS_JQ_MAX_JOBS;
    q->completeCount++;
    if (q->jobCount) q->jobCount--;
    atomic_fetch_add_explicit(&s_tag_completed[t->queue_slot][t->tag], 1,
                              memory_order_release);
    atomic_fetch_sub_explicit(&s_active_job_dispatches, 1,
                              memory_order_acq_rel);
    jq_notify();
}
extern int32_t ps3_guest_memory_alloc(uint32_t, uint32_t, uint32_t*);
extern int32_t ps3_guest_memory_free(uint32_t);

/* Optional bounded allocation recycling. The allocation is private until
 * jq_free_task, after its pending entry has retired and guest-owned copy has
 * been freed. The existing queue mutex protects the pool as well; allocation
 * and disposal callers run outside that lock. */
#define JQ_ALLOC_CLASSES 11u /* 256 B through 256 KiB */
#define JQ_ALLOC_PER_CLASS 2u
static JqTask* s_alloc_free[JQ_ALLOC_CLASSES];
static unsigned s_alloc_count[JQ_ALLOC_CLASSES];

static int jq_recycle_enabled(void)
{
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (!value) {
        const char* e = getenv("SPURS_JQ_ALLOC_RECYCLE");
        value = e && !strcmp(e, "1") ? 2 : 1;
        atomic_store_explicit(&cached, value, memory_order_relaxed);
    }
    return value == 2;
}
static unsigned jq_alloc_class(u32 bytes)
{
    unsigned c = 0;
    while (bytes > (256u << c)) ++c; /* caller validated 1..0x40000 */
    return c;
}
static JqTask* jq_alloc_task(u32 bytes)
{
    if (!jq_recycle_enabled()) return (JqTask*)malloc(sizeof(JqTask) + bytes);
    const unsigned c = jq_alloc_class(bytes);
    jq_lock();
    JqTask* t = s_alloc_free[c];
    if (t) {
        s_alloc_free[c] = t->next;
        --s_alloc_count[c];
    }
    jq_unlock();
    return t ? t : (JqTask*)malloc(sizeof(JqTask) + (256u << c));
}
static void jq_free_task(JqTask* t)
{
    if (!t) return;
    if (t->owned_descriptor) ps3_guest_memory_free(t->owned_descriptor);
    if (jq_recycle_enabled()) {
        const unsigned c = jq_alloc_class(t->binary_size);
        jq_lock();
        if (!atomic_load_explicit(&s_queue_shutdown_requested, memory_order_acquire) &&
            s_alloc_count[c] < JQ_ALLOC_PER_CLASS) {
            t->next = s_alloc_free[c];
            s_alloc_free[c] = t;
            ++s_alloc_count[c];
            jq_unlock();
            return;
        }
        jq_unlock();
    }
    free(t);
}
static void jq_release_recycled_locked(void)
{
    for (unsigned c = 0; c < JQ_ALLOC_CLASSES; ++c) {
        while (s_alloc_free[c]) {
            JqTask* t = s_alloc_free[c];
            s_alloc_free[c] = t->next;
            free(t);
        }
        s_alloc_count[c] = 0;
    }
}
static s32 jq_execute(JqTask* t)
{
    /* The generated SPU code gets a fresh local store. Descriptor/binary reads
     * performed by the runner are from immutable copies; DMA payload EAs remain
     * guest-owned and must remain valid until completion. */
    return spu_workload_dispatch_job_snapshot(t->binary, t->binary_size,
                                               t->job_ea, t->descriptor_size,
                                               t->descriptor)
        ? CELL_OK : (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND;
}
enum { JQ_FLAG_SYNC_JOB=1u, JQ_FLAG_EXCLUSIVE_JOB=2u, JQ_FLAG_NON_BLOCKING=4u };
static int jq_task_runnable(const JqTask* task)
{
    const JqState* q=&s_queues[task->queue_slot];
    for(u32 i=0;i<CELL_SPURS_JQ_MAX_JOBS;++i) {
        const JqTask* earlier=q->pending[i];
        if(!earlier || earlier==task || earlier->sequence>=task->sequence) continue;
        if((task->flags | earlier->flags) & (JQ_FLAG_SYNC_JOB|JQ_FLAG_EXCLUSIVE_JOB)) return 0;
    }
    return 1;
}
static JqTask* jq_take_runnable(void)
{
    JqTask* previous=NULL;
    for(JqTask* t=s_task_head;t;previous=t,t=t->next) {
        if(!jq_task_runnable(t)) continue;
        if(previous) previous->next=t->next; else s_task_head=t->next;
        if(s_task_tail==t) s_task_tail=previous;
        t->next=NULL;return t;
    }
    return NULL;
}
#ifdef _WIN32
static DWORD WINAPI jq_worker_main(LPVOID arg)
#else
static void* jq_worker_main(void* arg)
#endif
{
    u32 worker_no = (u32)(uintptr_t)arg;
#ifdef _WIN32
    { ULONG guard = 256u * 1024u; SetThreadStackGuarantee(&guard); }
#endif
    spurs_job_set_spu_num(worker_no);
    for (;;) {
        jq_lock();
        JqTask* t=NULL;
        while (!(t=jq_take_runnable()) && !s_pool_stopping) jq_await();
        if (!t) { jq_unlock(); break; }
        jq_unlock();
        s32 result = jq_execute(t);
        jq_lock();
        jq_finish_locked(t, result);
        jq_unlock();
        jq_free_task(t);
    }
    spu_workload_job_cache_thread_cleanup();
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}
/* Called with the lock. On thread creation failure, run synchronously instead
 * of losing a queued job; partially started workers remain available. */
static int jq_pool_start_locked(void)
{
    if (s_pool_workers) return 1;
    int requested = jq_requested_workers();
    if (!requested) return 0;
    s_pool_stopping = 0;
    for (int i = 0; i < requested; ++i) {
#ifdef _WIN32
        HANDLE h = CreateThread(NULL, 1u << 20, jq_worker_main,
                                (void*)(uintptr_t)i, 0, NULL);
        if (!h) break;
        s_jq_threads[i] = h;
#else
        if (pthread_create(&s_jq_threads[i], NULL, jq_worker_main,
                           (void*)(uintptr_t)i) != 0) break;
#endif
        s_pool_workers++;
    }
    if (!s_pool_workers) return 0;
    fprintf(stderr, "[spurs-jq] async pool: %d worker(s) (requested %d)\n",
            s_pool_workers, requested);
    return 1;
}
static void jq_wait_for_dispatches(void)
{
    jq_lock();
    while (atomic_load_explicit(&s_active_job_dispatches, memory_order_acquire))
        jq_await();
    jq_unlock();
}
/* PushSync snapshots the selected tags' submissions, so later pushes cannot
 * keep it waiting; a completed COUNT alone is incorrect with out-of-order
 * workers, so inspect the pending serials instead. */
static void jq_wait_for_tag_mask(u32 slot, u32 mask)
{
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !mask) return;
    uint64_t target[32] = {0};
    jq_lock();
    for (u32 tag = 0; tag < 32; ++tag)
        if (mask & (1u << tag))
            target[tag] = atomic_load_explicit(&s_tag_submitted[slot][tag],
                                                 memory_order_acquire);
    while (jq_pending_tag_before(slot, mask, target)) jq_await();
    jq_unlock();
}
static void jq_wait_for_queue_dispatches(u32 slot)
{
    jq_wait_for_tag_mask(slot, UINT32_MAX);
}
static int jq_alloc(void)
{
    for (int i = 0; i < CELL_SPURS_JQ_MAX_QUEUES; i++) {
        if (!s_queues[i].in_use) return i;
    }
    return -1;
}

static int jq_find_handle(u32 handle_ea)
{
    for (int i = 0; i < CELL_SPURS_JQ_MAX_QUEUES; i++)
        if (s_queues[i].in_use && s_queues[i].handle_ea == handle_ea)
            return i;
    return -1;
}

static int jq_create_state(u32 handle_ea, u32 max_jobs)
{
    if (!vm_base || !handle_ea || handle_ea > UINT32_MAX - 4u)
        return -1;
    jq_lock();
    int slot = jq_find_handle(handle_ea);
    if (slot < 0) slot = jq_alloc();
    if (slot >= 0 && !s_queues[slot].in_use) {
        JqState* q = &s_queues[slot];
        memset(q, 0, sizeof(*q));
        q->in_use = q->accepting = 1;
        q->handle_ea = handle_ea;
        q->maxJobs = max_jobs ? max_jobs : CELL_SPURS_JQ_MAX_JOBS;
        q->nextJobId = 1;
        atomic_store_explicit(&s_queue_shutdown_requested, 0, memory_order_release);
        for (u32 tag = 0; tag < 32; ++tag) {
            atomic_store(&s_tag_submitted[slot][tag], 0);
            atomic_store(&s_tag_completed[slot][tag], 0);
        }
    }
    jq_unlock();
    if (slot >= 0) vm_write32(handle_ea, (u32)slot);
    return slot;
}

static int jq_port_slot(u32 port_ea)
{
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u) return -1;
    jq_lock();
    int result = -1;
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; ++i) {
        if (s_ports[i].in_use && s_ports[i].port_ea == port_ea) {
            u32 slot = s_ports[i].queue_slot;
            if (slot < CELL_SPURS_JQ_MAX_QUEUES && s_queues[slot].in_use)
                result = (int)slot;
            break;
        }
    }
    jq_unlock();
    return result;
}
static s32 jq_register_port(u32 port_ea, u32 queue_slot)
{
    jq_lock();
    if (queue_slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[queue_slot].in_use) {
        jq_unlock();
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    }
    const u32 queue_ea = s_queues[queue_slot].handle_ea;
    int free_slot = -1;
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; ++i) {
        if (s_ports[i].in_use && s_ports[i].port_ea == port_ea) {
            s_ports[i].queue_slot = queue_slot;
            free_slot = -2;
            break;
        }
        if (!s_ports[i].in_use && free_slot < 0) free_slot = i;
    }
    if (free_slot == -1) {
        jq_unlock();
        return (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;
    }
    if (free_slot >= 0) {
        s_ports[free_slot].in_use = 1;
        s_ports[free_slot].port_ea = port_ea;
        s_ports[free_slot].queue_slot = queue_slot;
    }
    jq_unlock();

    /* SDK Port2 layout: self EA at +0, atomic job counters at +0x10,
     * queue EA at +0x18, and the queue handle at +0x1C. */
    vm_write32(port_ea + 0x00, port_ea);
    vm_write32(port_ea + 0x10, 0);
    vm_write32(port_ea + 0x14, 0);
    vm_write32(port_ea + 0x18, queue_ea);
    vm_write32(port_ea + 0x1C, queue_slot);
    return CELL_OK;
}

/* API */

s32 cellSpursJobQueueAttributeInitialize(CellSpursJobQueueAttribute* attr)
{
    const u32 ea = (u32)(uintptr_t)attr;
    if (!vm_base || !ea || ea > UINT32_MAX - sizeof(*attr))
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    memset(vm_base + ea, 0, sizeof(*attr));
    vm_write32(ea + 0, 64);
    vm_write32(ea + 4, 1);
    /* Default priority: normal on all SPUs */
    for (int i = 0; i < 8; i++)
        vm_write8(ea + 8 + (u32)i, CELL_SPURS_JQ_PRIORITY_NORMAL);

    return CELL_OK;
}

s32 cellSpursJobQueueAttributeSetMaxGrab(CellSpursJobQueueAttribute* attr, u32 maxGrab)
{
    const u32 ea = (u32)(uintptr_t)attr;
    if (!vm_base || !ea || ea > UINT32_MAX - sizeof(*attr))
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    vm_write32(ea + 4, maxGrab);
    return CELL_OK;
}

/* This import's ABI is not documented in the local headers. Keep it as a
 * permissive success stub; no attribute fields are changed. */
u64 cellSpursJobQueueAttributeSetDoBusyWaiting(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)a4; (void)a5; (void)a6; (void)a7;
    return (u64)(s64)(s32)CELL_OK;
}

/* SDK ABI: revision and internal-version arguments precede CellSpurs and the
 * queue handle, placing those pointers in the third and fourth saved slots.
 * This inline HLE registers queue state without an asynchronous pool. */
u64 _cellSpursCreateJobQueueWithJobDescriptorPool(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a0; (void)a1; (void)a4; (void)a5; (void)a6; (void)a7;
    const u32 spurs_ea = (u32)a2;
    const u32 jq_ea = (u32)a3;
    if (!vm_base || !spurs_ea || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    int slot = jq_create_state(jq_ea, CELL_SPURS_JQ_MAX_JOBS);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;
    return CELL_OK;
}

s32 cellSpursCreateJobQueue(void* spurs, CellSpursJobQueue* jq,
                            const CellSpursJobQueueAttribute* attr,
                            void* buffer, u32 bufferSize)
{
    const u32 attr_ea = (u32)(uintptr_t)attr;
    const u32 jq_ea = (u32)(uintptr_t)jq;
    (void)spurs; (void)buffer; (void)bufferSize;
    printf("[cellSpursJq] CreateJobQueue()\n");

    if (!vm_base || !jq_ea || !attr_ea ||
        jq_ea > UINT32_MAX - 4u || attr_ea > UINT32_MAX - sizeof(*attr))
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    const u32 max_jobs = vm_read32(attr_ea + 0);
    if (!max_jobs || max_jobs > CELL_SPURS_JQ_MAX_JOBS)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    int slot = jq_create_state(jq_ea, max_jobs);
    if (slot < 0) return (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;
    return CELL_OK;
}

s32 cellSpursDestroyJobQueue(CellSpursJobQueue* jq)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    JqState* q = &s_queues[slot];
    if (!q->in_use || q->handle_ea != jq_ea) {
        jq_unlock(); return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    }
    q->accepting = 0;
    while (q->jobCount) jq_await();
    q->in_use = 0;
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; ++i)
        if (s_ports[i].in_use && s_ports[i].queue_slot == slot)
            s_ports[i].in_use = 0;
    jq_unlock();
    return CELL_OK;
}

s32 cellSpursShutdownJobQueue(void)
{
    atomic_store_explicit(&s_queue_shutdown_requested, 1, memory_order_release);
    jq_wait_for_dispatches();
    jq_lock();
    s_pool_stopping = 1;
    jq_notify();
    int n = s_pool_workers;
    jq_unlock();
    for (int i = 0; i < n; ++i) {
#ifdef _WIN32
        WaitForSingleObject(s_jq_threads[i], INFINITE);
        CloseHandle(s_jq_threads[i]);
        s_jq_threads[i] = NULL;
#else
        pthread_join(s_jq_threads[i], NULL);
#endif
    }
    jq_lock();
    s_pool_workers = 0;
    s_pool_stopping = 0;
    jq_release_recycled_locked();
    jq_unlock();
    return CELL_OK;
}

s32 cellSpursJoinJobQueue(void)
{
    jq_wait_for_dispatches();
    return CELL_OK;
}

/* Prepare immutable job header+image at submission time. DMA payload buffers
 * remain guest-owned and may be mutated by the title; hardware likewise
 * requires titles to synchronize those buffers with job execution. */
static JqTask* jq_prepare(u32 job_ea, u32 descriptor_size)
{
    if (!vm_base || !job_ea ||
        (descriptor_size != 128u && descriptor_size != 256u) ||
        job_ea > UINT32_MAX - descriptor_size) return NULL;
    uint8_t descriptor[256];
    /* Capture before decoding, exactly as before. Never defer guest reads
     * until the worker runs: the submitting thread may reuse the descriptor. */
    memcpy(descriptor, vm_base + job_ea, descriptor_size);
    const uint8_t* h = descriptor;
    u64 ea64 = 0;
    for (unsigned i = 0; i < 8; ++i) ea64 = (ea64 << 8) | h[i];
    ea64 &= ~1ull;
    const u32 binary_size = (((u32)h[8] << 8) | h[9]) << 4;
    if (ea64 > UINT32_MAX || ea64 == 0 || binary_size == 0 ||
        binary_size > 0x40000u || ea64 > UINT32_MAX - binary_size)
        return NULL;

    /* One allocation instead of a header allocation plus a binary allocation.
     * Keep a private byte-for-byte snapshot for every in-flight job. No
     * shared-image lifetime, guest-memory, or completion rules change. */
    JqTask* t = jq_alloc_task(binary_size);
    if (!t) return NULL;
    memset(t, 0, sizeof(*t));
    memcpy(t->descriptor, descriptor, descriptor_size);
    t->binary = (uint8_t*)(t + 1);
    t->binary_size = binary_size;
    t->job_ea = job_ea;
    t->descriptor_size = descriptor_size;
    memcpy(t->binary, vm_base + (u32)ea64, binary_size);
    return t;
}

static s32 jq_push_slot_mode_flags(u32 slot, u32 job_ea, u32 jobSize,
                             u32 tag, u32 id_ea, int copy_push, u32 flags)
{
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || tag >= 32u || (flags & ~7u))
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    static int order_trace=-1;
    if(order_trace<0)order_trace=getenv("SPURS_JQ_ORDER_TRACE")!=NULL;
    if(order_trace) {
        static atomic_uint records;
        if(atomic_fetch_add_explicit(&records,1,memory_order_relaxed)<32)
            fprintf(stderr,"[jq-order] queue=%u tag=%u flags=%u copy=%d job=%08X\n",slot,tag,flags,copy_push,job_ea);
    }
    JqTask* t = jq_prepare(job_ea, jobSize);
    if (!t) return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    static int async_copy_enabled = -1;
    if (async_copy_enabled < 0) {
        const char* setting=getenv("SPURS_JQ_ASYNC_COPY");
        async_copy_enabled=!setting || setting[0]!='0';
    }
    if (copy_push && async_copy_enabled) {
        u32 copied_ea = 0;
        if (ps3_guest_memory_alloc(jobSize, 128u, &copied_ea) == 0 && copied_ea) {
            t->owned_descriptor = copied_ea;
            memcpy(vm_base + copied_ea, t->descriptor, jobSize);
            t->job_ea = copied_ea;
        }
    }
    const int synchronous_copy = copy_push && !t->owned_descriptor;
    jq_lock();
    JqState* q = &s_queues[slot];
    /* A copied descriptor has an owned guest EA, so ordinary SPU DMA sees
     * the snapshot after the submitter reuses its input. Allocation failure
     * retains the synchronous lifetime guarantee. */
    while (q->in_use && q->accepting &&
           (q->copy_sync_active || (synchronous_copy && q->jobCount) ||
            q->jobCount>=q->maxJobs || q->jobCount>=CELL_SPURS_JQ_MAX_JOBS)) {
        if(flags & JQ_FLAG_NON_BLOCKING) {
            const s32 result = q->jobCount>=q->maxJobs || q->jobCount>=CELL_SPURS_JQ_MAX_JOBS
                ? (s32)CELL_SPURS_JQ_ERROR_QUEUE_FULL : (s32)CELL_SPURS_JQ_ERROR_BUSY;
            jq_unlock();jq_free_task(t);return result;
        }
        jq_await();
    }
    if (!q->in_use || !q->accepting ||
        atomic_load_explicit(&s_queue_shutdown_requested, memory_order_acquire)) {
        jq_unlock(); jq_free_task(t); return (s32)CELL_SPURS_JQ_ERROR_ABORT;
    }
    if (q->jobCount >= q->maxJobs || q->jobCount >= CELL_SPURS_JQ_MAX_JOBS) {
        jq_unlock(); jq_free_task(t); return (s32)CELL_SPURS_JQ_ERROR_QUEUE_FULL;
    }
    u32 free_slot;
    for (free_slot = 0; free_slot < CELL_SPURS_JQ_MAX_JOBS; ++free_slot)
        if (!q->pending[free_slot]) break;
    if (free_slot == CELL_SPURS_JQ_MAX_JOBS) {
        jq_unlock(); jq_free_task(t); return (s32)CELL_SPURS_JQ_ERROR_QUEUE_FULL;
    }
    if (synchronous_copy) q->copy_sync_active = 1;
    t->queue_slot = slot;
    t->flags=flags;
    t->sequence=++s_task_sequence;
    t->tag = tag;
    t->id = q->nextJobId++;
    if (q->nextJobId == 0) q->nextJobId = 1;
    t->tag_serial = atomic_fetch_add_explicit(&s_tag_submitted[slot][tag], 1,
                                               memory_order_acq_rel) + 1u;
    q->pending[free_slot] = t;
    q->jobCount++;
    q->submitCount++;
    atomic_fetch_add_explicit(&s_active_job_dispatches, 1, memory_order_acq_rel);
    if (id_ea) vm_write32(id_ea, t->id);
    int asynchronous = !synchronous_copy && jq_pool_start_locked();
    if (asynchronous) {
        if (s_task_tail) s_task_tail->next = t;
        else s_task_head = t;
        s_task_tail = t;
        jq_notify();
        jq_unlock();
        return CELL_OK;
    }
    if (!synchronous_copy) {
        q->copy_sync_active=1;
        while (!jq_task_runnable(t)) jq_await();
    }
    jq_unlock();
    /* SPURS_JQ_WORKERS=0 retains synchronous execution. */
    spurs_job_set_spu_num(0);
    s32 result = jq_execute(t);
    jq_lock();
    jq_finish_locked(t, result);
    q->copy_sync_active = 0; jq_notify();
    jq_unlock();
    jq_free_task(t);
    return result;
}

static s32 jq_push_slot_mode(u32 slot,u32 job_ea,u32 jobSize,u32 tag,u32 id_ea,int copy_push)
{
    return jq_push_slot_mode_flags(slot,job_ea,jobSize,tag,id_ea,copy_push,JQ_FLAG_NON_BLOCKING);
}

static s32 jq_push_slot(u32 slot, u32 job_ea, u32 jobSize,
                        u32 tag, u32 id_ea)
{
    return jq_push_slot_mode(slot, job_ea, jobSize, tag, id_ea, 0);
}

s32 cellSpursJobQueuePush(CellSpursJobQueue* jq, const void* job,
                          u32 jobSize, u32 tag, CellSpursJobId* id)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    const u32 id_ea = (u32)(uintptr_t)id;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    return jq_push_slot(vm_read32(jq_ea), (u32)(uintptr_t)job,
                        jobSize, tag, id_ea);
}

s32 cellSpursJobQueuePushJob(CellSpursJobQueue* jq, const CellSpursJob256* job,
                             u32 tag, CellSpursJobId* id)
{
    return cellSpursJobQueuePush(jq, job, 256, tag, id);
}

s32 cellSpursJobQueuePort2Create(CellSpursJobQueuePort* port, CellSpursJobQueue* jq)
{
    const u32 port_ea = (u32)(uintptr_t)port;
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || !port_ea || jq_ea > UINT32_MAX - 4u ||
        port_ea > UINT32_MAX - CELL_SPURS_JQ_PORT2_SIZE) {
        jq_trace_port2("create", port_ea, jq_ea, 0, 0, 0, 0, -1,
                       (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT);
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    }
    if (port_ea & (CELL_SPURS_JQ_PORT2_ALIGN - 1u)) {
        jq_trace_port2("create", port_ea, jq_ea, 0, 0, 0, 0, -1,
                       (s32)CELL_SPURS_JQ_ERROR_ALIGN);
        return (s32)CELL_SPURS_JQ_ERROR_ALIGN;
    }

    /* Port creation may be the first HLE entry reached for a guest queue. */
    int slot = jq_create_state(jq_ea, CELL_SPURS_JQ_MAX_JOBS);
    if (slot < 0) {
        jq_trace_port2("create", port_ea, jq_ea, 0, 0, 0, 0, slot,
                       (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY);
        return (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;
    }
    s32 rc = jq_register_port(port_ea, (u32)slot);
    jq_trace_port2("create", port_ea, jq_ea, 0, 0, 0, 0, slot, rc);
    return rc;
}

s32 cellSpursJobQueuePort2Destroy(CellSpursJobQueuePort* port)
{
    const u32 port_ea = (u32)(uintptr_t)port;
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - CELL_SPURS_JQ_PORT2_SIZE)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    if (port_ea & (CELL_SPURS_JQ_PORT2_ALIGN - 1u))
        return (s32)CELL_SPURS_JQ_ERROR_ALIGN;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; i++)
        if (s_ports[i].in_use && s_ports[i].port_ea == port_ea)
            memset(&s_ports[i], 0, sizeof(s_ports[i]));
    jq_unlock();
    vm_write32(port_ea + 0x00, 0);
    vm_write32(port_ea + 0x18, 0);
    vm_write32(port_ea + 0x1C, 0);
    return CELL_OK;
}

u64 cellSpursJobQueuePort2PushSync(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    /* SDK ABI: port, job-tag mask, DMA tag, flags. The tag mask is a value,
     * not a guest pointer; only matching jobs on this port's queue are joined. */
    (void)a2; (void)a3; (void)a4; (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    jq_wait_for_tag_mask((u32)slot, (u32)a1);
    return CELL_OK;
}

/* Flush joins all jobs submitted on this queue before the call. */
u64 cellSpursJobQueuePort2PushFlush(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a1; (void)a2; (void)a3; (void)a4;
    (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    jq_wait_for_queue_dispatches((u32)slot);
    return CELL_OK;
}

s32 cellSpursJobQueueSendSignal(CellSpursJobQueue* jq, CellSpursJobId id)
{
    const u32 ea = (u32)(uintptr_t)jq;
    if (!vm_base || !ea || ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const u32 slot = vm_read32(ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES) return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    s32 result = s_queues[slot].in_use ? jq_status_for_id(&s_queues[slot], id)
                                        : (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_unlock();
    return result;
}

s32 cellSpursJobQueueWait(CellSpursJobQueue* jq, CellSpursJobId id)
{
    const u32 ea = (u32)(uintptr_t)jq;
    if (!vm_base || !ea || ea > UINT32_MAX - 4u || id == 0)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const u32 slot = vm_read32(ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    JqState* q = &s_queues[slot];
    if (!q->in_use) { jq_unlock(); return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT; }
    while (jq_pending_id(q, id)) jq_await();
    s32 result = jq_status_for_id(q, id);
    jq_unlock();
    return result;
}

s32 cellSpursJobQueueTryWait(CellSpursJobQueue* jq, CellSpursJobId id)
{
    const u32 ea = (u32)(uintptr_t)jq;
    if (!vm_base || !ea || ea > UINT32_MAX - 4u || id == 0)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const u32 slot = vm_read32(ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    JqState* q = &s_queues[slot];
    s32 result = !q->in_use ? (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT :
                 jq_pending_id(q, id) ? (s32)CELL_SPURS_JQ_ERROR_BUSY :
                 jq_status_for_id(q, id);
    if (result == (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND && !q->submitCount)
        result = (s32)CELL_SPURS_JQ_ERROR_BUSY;
    jq_unlock();
    return result;
}

s32 cellSpursJobQueueGetCount(CellSpursJobQueue* jq, u32* count)
{
    const u32 ea = (u32)(uintptr_t)jq, ce = (u32)(uintptr_t)count;
    if (!vm_base || !ea || !ce || ea > UINT32_MAX - 4u || ce > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    u32 slot = vm_read32(ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    if (!s_queues[slot].in_use) { jq_unlock(); return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT; }
    u32 n = s_queues[slot].jobCount;
    jq_unlock();
    vm_write32(ce, n);
    return CELL_OK;
}

s32 cellSpursJobQueueSemaphoreTryAcquire(CellSpursJobQueue* jq)
{
    const u32 ea = (u32)(uintptr_t)jq;
    if (!vm_base || !ea || ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    u32 slot = vm_read32(ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    jq_lock();
    s32 result = !s_queues[slot].in_use ? (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT :
                 s_queues[slot].jobCount ? (s32)CELL_SPURS_JQ_ERROR_BUSY : CELL_OK;
    jq_unlock();
    return result;
}

u64 cellSpursJobQueuePort2Sync(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a1; (void)a2; (void)a3; (void)a4;
    (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    jq_wait_for_queue_dispatches((u32)slot);
    return CELL_OK;
}

u64 _cellSpursJobQueuePort2CopyPushJobBody(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    const u32 job_ea = (u32)a1;
    const u32 descriptor_size = (u32)a2;
    const u32 descriptor_pool_size = (u32)a3;
    const u32 tag = (u32)a4;
    const int slot = jq_port_slot(port_ea);
    if (getenv("GFD_QUEUE_TRACE") && slot != 1 && slot != 2) {
        static unsigned seen = 0;
        if (seen++ < 12)
            fprintf(stderr, "[gfd-copy-push] port=%08X job=%08X size=%u slot=%d\n",
                    port_ea, job_ea, descriptor_size, slot);
        fprintf(stderr, "[gfd-copy-header] binary=%016llX size16=%04X\n",
                (unsigned long long)vm_read64(job_ea), vm_read16(job_ea + 8));
    }
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - CELL_SPURS_JQ_PORT2_SIZE) {
        jq_trace_port2("copy-push", port_ea, 0, job_ea, descriptor_size,
                       descriptor_pool_size, tag, slot,
                       (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT);
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    }
    if (port_ea & (CELL_SPURS_JQ_PORT2_ALIGN - 1u)) {
        jq_trace_port2("copy-push", port_ea, 0, job_ea, descriptor_size,
                       descriptor_pool_size, tag, slot,
                       (s32)CELL_SPURS_JQ_ERROR_ALIGN);
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_ALIGN;
    }
    if (slot < 0) {
        jq_trace_port2("copy-push", port_ea, 0, job_ea, descriptor_size,
                       descriptor_pool_size, tag, slot,
                       (s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED);
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    }
    s32 rc = jq_push_slot_mode_flags((u32)slot, job_ea, descriptor_size,
                               tag, 0, 1, (u32)a5);
    jq_trace_port2("copy-push", port_ea, vm_read32(port_ea + 0x18), job_ea,
                   descriptor_size, descriptor_pool_size, tag, slot, rc);
    if (getenv("GFD_QUEUE_TRACE") && slot == 0)
        fprintf(stderr, "[gfd-copy-result] rc=%08X\n", (u32)rc);
    return (u64)(s64)rc;
}

u64 _cellSpursJobQueuePort2PushJobBody(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    const u32 job_ea = (u32)a1;
    const u32 descriptor_size = (u32)a2;
    const u32 tag = (u32)a3;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    return (u64)(s64)jq_push_slot_mode_flags((u32)slot, job_ea, descriptor_size, tag, 0, 0, (u32)a4);
}
