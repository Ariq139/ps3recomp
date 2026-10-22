/*
 * ps3recomp - cellSpursJq HLE implementation
 *
 * SPURS Job Queue — manages job submission and tracking.
 * Synchronous descriptors dispatch through the runtime's lifted SPU job
 * registry; jobs without a registered lifted binary fail explicitly.
 */

#include "cellSpursJq.h"
#include "../../runtime/ppu/ppu_memory.h"   /* GUEST_PTR, vm_write*: guest EA -> host pointer */
#include "../../runtime/spu/spu_workload.h"
#include "../system/sysPrxForUser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/* Internal state */

typedef struct {
    int in_use;
    u32 handle_ea;      /* guest CellSpursJobQueue object */
    u32 maxJobs;
    u32 jobCount;       /* number of pending (not-yet-completed) jobs */
    u32 nextJobId;
    u32 submitCount;    /* total jobs submitted over lifetime */
    u32 completeCount;  /* total jobs completed over lifetime */
    /*
     * Per-job completion ring: tracks the last CELL_SPURS_JQ_MAX_JOBS
     * job IDs that have completed.  Wait/TryWait check this.
     */
    u32 completedIds[CELL_SPURS_JQ_MAX_JOBS];
    u32 completedHead;  /* next write position in the ring */
} JqState;

typedef struct {
    int in_use;
    u32 port_ea;        /* guest CellSpursJobQueuePort output object */
    u32 queue_slot;
} JqPortState;

static JqState s_queues[CELL_SPURS_JQ_MAX_QUEUES];
#define CELL_SPURS_JQ_MAX_PORTS 64
static JqPortState s_ports[CELL_SPURS_JQ_MAX_PORTS];
static atomic_uint s_active_job_dispatches;
static atomic_int s_queue_shutdown_requested;
static void jq_wait_for_dispatches(void);

/* RPCS3_REFERENCE_REVIEWED: descriptor binary-pointer and size fields were
 * checked against RPCS3 headers. This local decoder copies no implementation.
 * A job descriptor is guest data: the first 8 bytes hold the binary EA and
 * byte 8 stores its size in 16-byte units. The runtime has a real SPU job
 * staging/dispatch path; use it instead of claiming every submitted job
 * completed without running. */
static s32 jq_dispatch_descriptor(u32 job_ea, u32 descriptor_size)
{
    if (atomic_load_explicit(&s_queue_shutdown_requested, memory_order_acquire))
        return (s32)CELL_SPURS_JQ_ERROR_ABORT;
    if (!vm_base || !job_ea ||
        (descriptor_size != 128 && descriptor_size != 256) ||
        job_ea > UINT32_MAX - descriptor_size)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    const u64 binary_ea64 = vm_read64(job_ea) & ~1ull;
    const u32 binary_ea = (u32)binary_ea64;
    const u32 binary_size = (u32)vm_read16(job_ea + 8) << 4;
    if (!binary_ea || !binary_size || binary_ea64 > UINT32_MAX ||
        binary_ea > UINT32_MAX - binary_size)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    atomic_fetch_add_explicit(&s_active_job_dispatches, 1, memory_order_acq_rel);
    if (atomic_load_explicit(&s_queue_shutdown_requested, memory_order_acquire)) {
        atomic_fetch_sub_explicit(&s_active_job_dispatches, 1, memory_order_acq_rel);
        return (s32)CELL_SPURS_JQ_ERROR_ABORT;
    }
    const int ran = spu_workload_dispatch_job(vm_base + binary_ea, binary_size,
                                               job_ea, descriptor_size);
    atomic_fetch_sub_explicit(&s_active_job_dispatches, 1, memory_order_acq_rel);
    return ran ? CELL_OK : (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND;
}

/* Push helpers execute jobs synchronously in this HLE. Sync/flush wait for
 * any worker that entered the dispatcher concurrently. */
static void jq_wait_for_dispatches(void)
{
    while (atomic_load_explicit(&s_active_job_dispatches, memory_order_acquire) != 0)
        sys_ppu_thread_yield();
}

/* Check whether a job ID has completed in the given queue */
static int jq_is_completed(const JqState* q, CellSpursJobId id)
{
    /* Job ID 0 is invalid / never assigned */
    if (id == 0) return 0;

    for (int i = 0; i < CELL_SPURS_JQ_MAX_JOBS; i++) {
        if (q->completedIds[i] == id)
            return 1;
    }
    return 0;
}

/* Record a job as completed */
static void jq_mark_completed(JqState* q, CellSpursJobId id)
{
    q->completedIds[q->completedHead] = id;
    q->completedHead = (q->completedHead + 1) % CELL_SPURS_JQ_MAX_JOBS;
    q->completeCount++;
    if (q->jobCount > 0)
        q->jobCount--;
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

    int existing = jq_find_handle(handle_ea);
    if (existing >= 0)
        return existing;

    int slot = jq_alloc();
    if (slot < 0)
        return -1;

    JqState* q = &s_queues[slot];
    memset(q, 0, sizeof(*q));
    q->in_use = 1;
    q->handle_ea = handle_ea;
    q->maxJobs = max_jobs ? max_jobs : CELL_SPURS_JQ_MAX_JOBS;
    q->nextJobId = 1;
    vm_write32(handle_ea, (u32)slot);
    return slot;
}

static int jq_port_slot(u32 port_ea)
{
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u)
        return -1;
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; i++) {
        if (s_ports[i].in_use && s_ports[i].port_ea == port_ea) {
            int slot = (int)s_ports[i].queue_slot;
            return slot >= 0 && slot < CELL_SPURS_JQ_MAX_QUEUES &&
                   s_queues[slot].in_use ? slot : -1;
        }
    }
    return -1;
}

static s32 jq_register_port(u32 port_ea, u32 queue_slot)
{
    int free_slot = -1;
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; i++) {
        if (s_ports[i].in_use && s_ports[i].port_ea == port_ea) {
            s_ports[i].queue_slot = queue_slot;
            vm_write32(port_ea, queue_slot);
            return CELL_OK;
        }
        if (!s_ports[i].in_use && free_slot < 0)
            free_slot = i;
    }
    if (free_slot < 0)
        return (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;

    s_ports[free_slot].in_use = 1;
    s_ports[free_slot].port_ea = port_ea;
    s_ports[free_slot].queue_slot = queue_slot;
    vm_write32(port_ea, queue_slot);
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

/* P5's call site passes CellSpurs in r5 and the CellSpursJobQueue object in
 * r6. The HLE runs jobs synchronously, so it registers the queue handle
 * without creating Sony's asynchronous descriptor pool. */
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

    int slot = jq_alloc();
    if (slot < 0) return (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;

    memset(&s_queues[slot], 0, sizeof(JqState));
    atomic_store_explicit(&s_queue_shutdown_requested, 0, memory_order_release);
    s_queues[slot].in_use = 1;
    s_queues[slot].handle_ea = jq_ea;
    s_queues[slot].maxJobs = max_jobs;
    s_queues[slot].jobCount = 0;
    s_queues[slot].nextJobId = 1;
    s_queues[slot].submitCount = 0;
    s_queues[slot].completeCount = 0;
    s_queues[slot].completedHead = 0;

    vm_write32(jq_ea, (u32)slot);
    return CELL_OK;
}

s32 cellSpursDestroyJobQueue(CellSpursJobQueue* jq)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    const u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    printf("[cellSpursJq] DestroyJobQueue(%u)\n", slot);
    s_queues[slot].in_use = 0;
    return CELL_OK;
}

s32 cellSpursShutdownJobQueue(void)
{
    atomic_store_explicit(&s_queue_shutdown_requested, 1, memory_order_release);
    jq_wait_for_dispatches();
    return CELL_OK;
}

s32 cellSpursJoinJobQueue(void)
{
    jq_wait_for_dispatches();
    return CELL_OK;
}

static s32 jq_push_slot(u32 slot, u32 job_ea, u32 jobSize,
                        u32 tag, u32 id_ea)
{
    (void)tag;
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    JqState* q = &s_queues[slot];

    if (q->jobCount >= q->maxJobs)
        return (s32)CELL_SPURS_JQ_ERROR_QUEUE_FULL;

    const s32 dispatch_rc = jq_dispatch_descriptor(job_ea, jobSize);
    if (dispatch_rc != CELL_OK) return dispatch_rc;

    u32 jobId = q->nextJobId++;
    q->jobCount++;
    q->submitCount++;

    if (id_ea) vm_write32(id_ea, jobId);

    /* Only a successfully executed job reaches completion bookkeeping. */
    jq_mark_completed(q, jobId);

    return CELL_OK;
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
        port_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    int slot = jq_find_handle(jq_ea);
    if (slot < 0) {
        const u32 raw = vm_read32(jq_ea);
        if (raw < CELL_SPURS_JQ_MAX_QUEUES && s_queues[raw].in_use &&
            s_queues[raw].handle_ea == jq_ea)
            slot = (int)raw;
    }
    /* RPCS3's current HLE also accepts this call as a success-only stub. P5
     * allocates this queue object at its call site, so create a synchronous
     * local queue on first use instead of aborting guest initialization. */
    if (slot < 0)
        slot = jq_create_state(jq_ea, CELL_SPURS_JQ_MAX_JOBS);
    if (slot < 0)
        return (s32)CELL_SPURS_JQ_ERROR_OUT_OF_MEMORY;
    return jq_register_port(port_ea, (u32)slot);
}

s32 cellSpursJobQueuePort2Destroy(CellSpursJobQueuePort* port)
{
    const u32 port_ea = (u32)(uintptr_t)port;
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < CELL_SPURS_JQ_MAX_PORTS; i++)
        if (s_ports[i].in_use && s_ports[i].port_ea == port_ea)
            memset(&s_ports[i], 0, sizeof(s_ports[i]));
    vm_write32(port_ea, 0);
    return CELL_OK;
}

u64 cellSpursJobQueuePort2PushSync(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    /* P5's A78F1C wrapper passes the port in r3, a tag bitmask in r4 and
     * zero in r5. This is a synchronization command, not a job descriptor.
     * Do not interpret the mask as a guest pointer or write a job ID through
     * an unrelated argument register. With inline jobs, waiting for every
     * outstanding dispatcher also satisfies a dependency on selected tags. */
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    if (!vm_base || !port_ea || port_ea > UINT32_MAX - 4u)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    jq_wait_for_dispatches();
    return CELL_OK;
}

/* Jobs are executed inline, so there is no deferred batch; flush waits for
 * concurrently submitted inline jobs to return. */
s32 cellSpursJobQueuePort2PushFlush(void)
{
    jq_wait_for_dispatches();
    return CELL_OK;
}

s32 cellSpursJobQueueSendSignal(CellSpursJobQueue* jq, CellSpursJobId id)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    if (!jq_is_completed(&s_queues[slot], id))
        return (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND;
    return CELL_OK;
}

s32 cellSpursJobQueueWait(CellSpursJobQueue* jq, CellSpursJobId id)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    const JqState* q = &s_queues[slot];

    /*
     * If no jobs have ever been submitted, treat as an error —
     * nothing can complete if nothing was pushed.
     */
    if (q->submitCount == 0)
        return (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND;

    if (!jq_is_completed(q, id))
        return (s32)CELL_SPURS_JQ_ERROR_JOB_NOT_FOUND;

    return CELL_OK;
}

s32 cellSpursJobQueueTryWait(CellSpursJobQueue* jq, CellSpursJobId id)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    const JqState* q = &s_queues[slot];

    /* No jobs submitted yet — return BUSY */
    if (q->submitCount == 0)
        return (s32)CELL_SPURS_JQ_ERROR_BUSY;

    if (!jq_is_completed(q, id))
        return (s32)CELL_SPURS_JQ_ERROR_BUSY;

    return CELL_OK;
}

s32 cellSpursJobQueueGetCount(CellSpursJobQueue* jq, u32* count)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    const u32 count_ea = (u32)(uintptr_t)count;
    if (!vm_base || !jq_ea || !count_ea ||
        jq_ea > UINT32_MAX - 4u || count_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;

    vm_write32(count_ea, s_queues[slot].jobCount);
    return CELL_OK;
}

s32 cellSpursJobQueueSemaphoreTryAcquire(CellSpursJobQueue* jq)
{
    const u32 jq_ea = (u32)(uintptr_t)jq;
    if (!vm_base || !jq_ea || jq_ea > UINT32_MAX - 4u)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    const u32 slot = vm_read32(jq_ea);
    if (slot >= CELL_SPURS_JQ_MAX_QUEUES || !s_queues[slot].in_use)
        return (s32)CELL_SPURS_JQ_ERROR_INVALID_ARGUMENT;
    return atomic_load_explicit(&s_active_job_dispatches, memory_order_acquire) == 0
        ? CELL_OK : (s32)CELL_SPURS_JQ_ERROR_BUSY;
}

u64 cellSpursJobQueuePort2Sync(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)a4; (void)a5; (void)a6; (void)a7;
    jq_wait_for_dispatches();
    return CELL_OK;
}

u64 _cellSpursJobQueuePort2CopyPushJobBody(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a3; (void)a4; (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    const u32 job_ea = (u32)a1;
    const u32 descriptor_size = (u32)a2;
    const int slot = jq_port_slot(port_ea);
    if (getenv("GFD_QUEUE_TRACE") && slot != 1 && slot != 2) {
        static unsigned seen = 0;
        if (seen++ < 12)
            fprintf(stderr, "[gfd-copy-push] port=%08X job=%08X size=%u slot=%d\n",
                    port_ea, job_ea, descriptor_size, slot);
        fprintf(stderr, "[gfd-copy-header] binary=%016llX size16=%04X\n",
                (unsigned long long)vm_read64(job_ea), vm_read16(job_ea + 8));
    }
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    s32 rc = jq_push_slot((u32)slot, job_ea, descriptor_size, 0, 0);
    if (getenv("GFD_QUEUE_TRACE") && slot == 0)
        fprintf(stderr, "[gfd-copy-result] rc=%08X\n", (u32)rc);
    return (u64)(s64)rc;
}

u64 _cellSpursJobQueuePort2PushJobBody(
    u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7)
{
    (void)a3; (void)a4; (void)a5; (void)a6; (void)a7;
    const u32 port_ea = (u32)a0;
    const u32 job_ea = (u32)a1;
    const u32 descriptor_size = (u32)a2;
    const int slot = jq_port_slot(port_ea);
    if (slot < 0)
        return (u64)(s64)(s32)CELL_SPURS_JQ_ERROR_NOT_INITIALIZED;
    return (u64)(s64)jq_push_slot((u32)slot, job_ea, descriptor_size, 0, 0);
}
