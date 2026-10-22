/*
 * ps3recomp - cellFiber HLE implementation
 *
 * The SDK's scheduler-based PPU fibers (see cellFiber.h). RunFibers runs on the calling PPU thread and switches to one
 * fiber at a time; a fiber runs the guest entry through the PPU thread trampoline on its own host stack with its own
 * ppu_context, whose r1 points into the guest stack the caller passed to CreateFiber. A fiber hands control back to
 * RunFibers when it blocks (WaitSignal, JoinFiber), yields or ends.
 *
 * Not modelled: the scheduler attribute and fiber attribute contents (both are zeroed on initialisation and ignored),
 * and the fiber context-switch trace/ufiber utilities of the SDK.
 */

/* Darwin gates the ucontext routines AND the shape of ucontext_t itself on _XOPEN_SOURCE, so this has to come before
 * the first system header any include below reaches -- see the static assertion further down for what defining it
 * late costs. Nothing above this line. */
#if defined(__APPLE__) && !defined(_XOPEN_SOURCE)
#  define _XOPEN_SOURCE 600
#  define _DARWIN_C_SOURCE 1   /* and back: _XOPEN_SOURCE alone hides MAP_ANONYMOUS, which vm.h uses */
#endif

#include "cellFiber.h"
#include "../../runtime/ppu/ppu_memory.h"   /* vm_base, vm_write32, vm_memset: guest EA -> host */
#include "../../runtime/syscalls/sys_ppu_thread.h"   /* ppu_context, g_ppu_thread_entry_trampoline */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <ucontext.h>

#if defined(__APPLE__)
/* getcontext() points uc_mcontext at ucontext_t's own __mcontext_data member and writes uc_mcsize (816) bytes there.
 * That member is declared only when _XOPEN_SOURCE was defined before <sys/_types/_ucontext.h> was first pulled in;
 * without it the struct is the 64-byte header alone and every getcontext, swapcontext and makecontext writes 752
 * bytes off the end of it. The define above is the fix; this is the tripwire, because moving an #include up here
 * would silently undo it. */
_Static_assert(sizeof(ucontext_t) > 64,
               "ucontext_t has no embedded machine state: _XOPEN_SOURCE was "
               "defined too late and getcontext will write past the struct");
#endif
#endif

extern PPU_THREAD_LOCAL ppu_context* g_active_ctx;

/* Host stack of each fiber. The guest stack is separate: it holds the guest's own frames, this one the host frames of
 * the lifted code and of the trampoline. */
#define FIBER_HOST_STACK   (4u * 1024u * 1024u)
/* Smallest guest stack CreateFiber accepts. */
#define FIBER_MIN_STACK    0x400u
/* Guest stack bytes kept free above the initial r1 (the ABI's back chain and register save area). */
#define FIBER_STACK_TOP_PAD 0x100u
/* r3 for the fiber's thread id when it is created outside a PPU thread. */
#define FIBER_FALLBACK_THREAD_ID 0x7FFF0000ull

/* The retail cellFiber PPU scheduler API is different from the small
 * CellFiber handle API below. A guest fiber is a PPU function running on a
 * host fiber: switching the host fiber preserves the recompiled C call stack
 * and its live ppu_context while cellFiberPpuYield switches back to the guest
 * scheduler. */
typedef struct PpuFiberSlot {
    int in_use;
    u32 scheduler_ea;
    u32 fiber_ea;
    u32 entry_opd;
    u64 arg;
    u32 priority;
    u32 guest_stack_ea;
    u32 guest_stack_size;
    u32 on_exit_opd;
    u64 on_exit_arg;
    s32 exit_status;
    int exit_status_set;
    int exit_callback_called;
    int state;                 /* 0 ready, 1 running, 2 waiting, 3 terminated */
    int wait_target;
    u32 run_epoch;
    int force_resume;
#ifdef _WIN32
    LPVOID native_fiber;
#else
    ucontext_t context;
    u8* host_stack;
    size_t host_stack_size;
#endif
} PpuFiberSlot;

typedef struct PpuSchedulerSlot {
    int in_use;
    u32 scheduler_ea;
    u32 epoch;
} PpuSchedulerSlot;

#define CELL_FIBER_MAX_FIBERS 256
#define CELL_FIBER_PPU_MAX_SCHEDULERS 8
#define CELL_FIBER_PPU_HOST_STACK_SIZE (1024u * 1024u)
static PpuFiberSlot s_ppu_fibers[CELL_FIBER_MAX_FIBERS];
static PpuSchedulerSlot s_ppu_schedulers[CELL_FIBER_PPU_MAX_SCHEDULERS];
#ifdef _WIN32
static __declspec(thread) LPVOID s_ppu_scheduler_host = NULL;
static __declspec(thread) int s_ppu_running_slot = -1;
static __declspec(thread) u32 s_ppu_running_scheduler = 0;
#else
static _Thread_local ucontext_t s_ppu_scheduler_host;
static _Thread_local int s_ppu_running_slot = -1;
static _Thread_local u32 s_ppu_running_scheduler = 0;
#endif

extern u64 ppu_guest_call_on_stack(u32 opd_addr, u32 guest_sp,
                                   u64 a0, u64 a1, u64 a2, u64 a3,
                                   u64 a4, u64 a5, u64 a6, u64 a7);
extern u64 ppu_guest_call(u32 opd_addr, u64 a0, u64 a1, u64 a2, u64 a3,
                          u64 a4, u64 a5, u64 a6, u64 a7);
extern void ps3_hle_register_ctx(u32 nid, const char* name,
                                void (*fn)(ppu_context*));
extern PPU_THREAD_LOCAL ppu_context* g_active_ctx;

static PpuSchedulerSlot* ppu_scheduler_find(u32 scheduler_ea)
{
    for (int i = 0; i < CELL_FIBER_PPU_MAX_SCHEDULERS; i++)
        if (s_ppu_schedulers[i].in_use &&
            s_ppu_schedulers[i].scheduler_ea == scheduler_ea)
            return &s_ppu_schedulers[i];
    return NULL;
}

static int ppu_fiber_find(u32 fiber_ea)
{
    for (int i = 0; i < CELL_FIBER_MAX_FIBERS; i++)
        if (s_ppu_fibers[i].in_use && s_ppu_fibers[i].fiber_ea == fiber_ea)
            return i;
    return -1;
}

static void ppu_fiber_finish(PpuFiberSlot* f)
{
    if (!f->exit_status_set) {
        f->exit_status = 0;
        f->exit_status_set = 1;
    }
    if (f->on_exit_opd && !f->exit_callback_called) {
        f->exit_callback_called = 1;
        (void)ppu_guest_call(f->on_exit_opd, f->on_exit_arg,
                             (u64)(s64)f->exit_status, 0, 0, 0, 0, 0, 0);
    }
    f->state = 3;
    s_ppu_running_slot = -1;
    s_ppu_running_scheduler = 0;
#ifdef _WIN32
    SwitchToFiber(s_ppu_scheduler_host);
#else
    setcontext(&s_ppu_scheduler_host);
#endif
}

#ifdef _WIN32
static VOID CALLBACK ppu_fiber_trampoline(LPVOID param)
{
    PpuFiberSlot* f = (PpuFiberSlot*)param;
    int slot = (int)(f - s_ppu_fibers);
    s_ppu_running_slot = slot;
    s_ppu_running_scheduler = f->scheduler_ea;
    /* Reserve the PPC64 caller frame below the end of the allocation.
     * A callee saves LR at entry SP + 0x10; using the allocation end as SP
     * would overwrite the next heap chunk's free-list links. */
    u32 guest_sp = ((f->guest_stack_ea + f->guest_stack_size) & ~15u) - 0x70u;
    vm_write64(guest_sp, 0); /* terminate the guest stack backchain */
    u64 result = ppu_guest_call_on_stack(
        f->entry_opd, guest_sp, f->arg, 0, 0, 0, 0, 0, 0, 0);
    f->exit_status = (s32)(u32)result;
    f->exit_status_set = 1;
    ppu_fiber_finish(f);
}
#else
static void ppu_fiber_trampoline(int slot)
{
    PpuFiberSlot* f = &s_ppu_fibers[(u32)slot];
    s_ppu_running_slot = slot;
    s_ppu_running_scheduler = f->scheduler_ea;
    u32 guest_sp = ((f->guest_stack_ea + f->guest_stack_size) & ~15u) - 0x70u;
    vm_write64(guest_sp, 0);
    u64 result = ppu_guest_call_on_stack(
        f->entry_opd, guest_sp, f->arg, 0, 0, 0, 0, 0, 0, 0);
    f->exit_status = (s32)(u32)result;
    f->exit_status_set = 1;
    ppu_fiber_finish(f);
}
#endif

static void ppu_fiber_switch_to_host(PpuFiberSlot* f)
{
    ppu_context* scheduler_ctx = g_active_ctx;
    int slot = (int)(f - s_ppu_fibers);
    f->state = 1;
    s_ppu_running_slot = slot;
    s_ppu_running_scheduler = f->scheduler_ea;
#ifdef _WIN32
    SwitchToFiber(f->native_fiber);
#else
    swapcontext(&s_ppu_scheduler_host, &f->context);
#endif
    g_active_ctx = scheduler_ctx;
    s_ppu_running_slot = -1;
    s_ppu_running_scheduler = 0;
}

static void ppu_fiber_suspend_to_host(PpuFiberSlot* f)
{
    /* Native fibers share thread-local storage. Restore this fiber's guest
     * context on resume so reservation checks use its own lwarx state. */
    ppu_context* fiber_ctx = g_active_ctx;
#ifdef _WIN32
    SwitchToFiber(s_ppu_scheduler_host);
#else
    swapcontext(&f->context, &s_ppu_scheduler_host);
#endif
    g_active_ctx = fiber_ctx;
}

static int ppu_scheduler_pick(PpuSchedulerSlot* scheduler, u32 epoch)
{
    int best = -1;
    for (int i = 0; i < CELL_FIBER_MAX_FIBERS; i++) {
        PpuFiberSlot* f = &s_ppu_fibers[i];
        if (!f->in_use || f->scheduler_ea != scheduler->scheduler_ea ||
            f->state != 0 || (f->run_epoch == epoch && !f->force_resume))
            continue;
        if (best < 0 || f->priority < s_ppu_fibers[best].priority)
            best = i;
    }
    return best;
}

static void ppu_scheduler_wake_joiners(u32 scheduler_ea)
{
    for (int i = 0; i < CELL_FIBER_MAX_FIBERS; i++) {
        PpuFiberSlot* f = &s_ppu_fibers[i];
        if (!f->in_use || f->scheduler_ea != scheduler_ea || f->state != 2)
            continue;
        if (f->wait_target >= 0 && f->wait_target < CELL_FIBER_MAX_FIBERS &&
            s_ppu_fibers[f->wait_target].in_use &&
            s_ppu_fibers[f->wait_target].state == 3) {
            f->state = 0;
            f->force_resume = 1;
        }
    }
}

/* Give each runnable fiber one turn per epoch. Yielded fibers run again after
 * the other ready fibers have had a turn; a JoinFiber waiter resumes after its
 * target completes. */
static int ppu_scheduler_run_pass(PpuSchedulerSlot* scheduler)
{
    if (!scheduler) return 0;
    int ran = 0;
    u32 epoch = ++scheduler->epoch;
    if (!epoch) epoch = ++scheduler->epoch;
#ifdef _WIN32
    if (!s_ppu_scheduler_host) {
        s_ppu_scheduler_host = ConvertThreadToFiber(NULL);
        if (!s_ppu_scheduler_host && GetLastError() == ERROR_ALREADY_FIBER)
            s_ppu_scheduler_host = GetCurrentFiber();
    }
    if (!s_ppu_scheduler_host) return 0;
#else
    getcontext(&s_ppu_scheduler_host);
#endif
    for (;;) {
        ppu_scheduler_wake_joiners(scheduler->scheduler_ea);
        int slot = ppu_scheduler_pick(scheduler, epoch);
        if (slot < 0) {
            int more_ready = 0;
            for (int i = 0; i < CELL_FIBER_MAX_FIBERS; i++) {
                PpuFiberSlot* f = &s_ppu_fibers[i];
                if (f->in_use && f->scheduler_ea == scheduler->scheduler_ea &&
                    f->state == 0) {
                    more_ready = 1;
                    break;
                }
            }
            if (!more_ready) break;
            epoch = ++scheduler->epoch;
            if (!epoch) epoch = ++scheduler->epoch;
            continue;
        }
        PpuFiberSlot* f = &s_ppu_fibers[slot];
        f->run_epoch = epoch;
        f->force_resume = 0;
        ppu_fiber_switch_to_host(f);
        ran++;
    }
    return ran;
}

/* ---------------------------------------------------------------------------
 * State
 *
 * One lock covers all of it, and one condition variable wakes everyone who waits on it: a RunFibers loop with nothing
 * to run (its fibers wait for a signal another thread sends), and JoinFiber callers outside any fiber. Guest
 * addresses name fibers and schedulers, never host pointers, so a waiter that wakes after FinalizeScheduler finds the
 * object gone instead of freed memory.
 * -----------------------------------------------------------------------*/

enum { F_READY, F_RUNNING, F_WAIT_SIGNAL, F_WAIT_JOIN, F_DONE };

typedef struct Sched Sched;
typedef struct Fiber Fiber;

struct Fiber {
    Fiber*  next;                /* every fiber of every scheduler */
    Sched*  sched;
    u32     ea;                  /* the guest CellFiberPpu */
    u32     opd;
    u32     priority;
    u32     stack_ea;
    u32     stack_size;
    u64     arg;
    u64     seq;                 /* position in the run queue: creation order, then the last yield */
    int     state;
    int     signal_pending;      /* a SendSignal arrived before the WaitSignal */
    u32     join_ea;             /* F_WAIT_JOIN: the fiber waited for */
    s32     status;
#ifdef _WIN32
    void*   host_fiber;
#else
    ucontext_t uc;
    void*   host_stack;
#endif
    ppu_context ctx;
};

struct Sched {
    Sched*  next;
    u32     ea;                  /* the guest CellFiberPpuScheduler */
    int     running;             /* a thread is inside RunFibers */
#ifdef _WIN32
    void*   back;                /* RunFibers' fiber while it runs one */
#else
    ucontext_t back;             /* RunFibers' context while it runs one */
#endif
};

static Sched* s_scheds;
static Fiber* s_fibers;
static u64    s_seq;

/* Lifecycle events only: no logging or environment lookup on fiber switches.
 * Call with s_lock held, including the one-time environment lookup. */
static int fiber_lifecycle_trace(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* value = getenv("PPU_FIBER_TRACE");
        enabled = value && *value && strcmp(value, "0") != 0;
    }
    return enabled;
}

/* The fiber running on this host thread, if any. */
static PPU_THREAD_LOCAL Fiber* t_cur;

#ifdef _WIN32
static SRWLOCK            s_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_cv   = CONDITION_VARIABLE_INIT;
#  define LOCK()    AcquireSRWLockExclusive(&s_lock)
#  define UNLOCK()  ReleaseSRWLockExclusive(&s_lock)
#  define WAIT()    SleepConditionVariableSRW(&s_cv, &s_lock, INFINITE, 0)
#  define WAKE()    WakeAllConditionVariable(&s_cv)
#else
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  s_cv   = PTHREAD_COND_INITIALIZER;
#  define LOCK()    pthread_mutex_lock(&s_lock)
#  define UNLOCK()  pthread_mutex_unlock(&s_lock)
#  define WAIT()    pthread_cond_wait(&s_cv, &s_lock)
#  define WAKE()    pthread_cond_broadcast(&s_cv)
#endif

/* Lookups; the lock is held. */
static Sched* find_sched(u32 ea)
{
    for (Sched* s = s_scheds; s; s = s->next)
        if (s->ea == ea) return s;
    return NULL;
}

static Fiber* find_fiber(u32 ea)
{
    for (Fiber* f = s_fibers; f; f = f->next)
        if (f->ea == ea) return f;
    return NULL;
}

/* The READY fiber of the scheduler with the lowest priority number; creation (or last yield) order among equals. */
static Fiber* pick_runnable(const Sched* s)
{
    Fiber* best = NULL;
    for (Fiber* f = s_fibers; f; f = f->next) {
        if (f->sched != s || f->state != F_READY) continue;
        if (!best || f->priority < best->priority ||
            (f->priority == best->priority && f->seq < best->seq))
            best = f;
    }
    return best;
}

/* Every fiber of the scheduler has ended (or it has none). */
static int all_done(const Sched* s)
{
    for (const Fiber* f = s_fibers; f; f = f->next)
        if (f->sched == s && f->state != F_DONE) return 0;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Host context switching: ucontext on POSIX, fibers on Windows. RunFibers' own context is the "back" one every fiber
 * returns to when it blocks, yields or ends.
 * -----------------------------------------------------------------------*/

static void fiber_body(Fiber* f);

#ifdef _WIN32
static VOID CALLBACK fiber_entry(void* p)
{
    fiber_body((Fiber*)p);
}

static int host_create(Fiber* f)
{
    f->host_fiber = CreateFiberEx(0, FIBER_HOST_STACK, 0, fiber_entry, f);
    return f->host_fiber != NULL;
}

/* Releases the host stack. The fiber must not be the one running, nor be resumed again. */
static void host_destroy(Fiber* f)
{
    if (f->host_fiber) DeleteFiber(f->host_fiber);
    f->host_fiber = NULL;
}

/* RunFibers -> fiber. The calling thread becomes a fiber itself on first use. */
static void host_resume(Sched* s, Fiber* f)
{
    if (!IsThreadAFiber()) ConvertThreadToFiber(NULL);
    s->back = GetCurrentFiber();
    SwitchToFiber(f->host_fiber);
}

static void host_yield(Fiber* f)
{
    SwitchToFiber(f->sched->back);
}
#else
static void fiber_entry(unsigned hi, unsigned lo)
{
    fiber_body((Fiber*)(((uintptr_t)hi << 32) | lo));
}

static int host_create(Fiber* f)
{
    f->host_stack = malloc(FIBER_HOST_STACK);
    if (!f->host_stack) return 0;
    getcontext(&f->uc);
    f->uc.uc_stack.ss_sp   = f->host_stack;
    f->uc.uc_stack.ss_size = FIBER_HOST_STACK;
    f->uc.uc_link          = NULL;
    uintptr_t p = (uintptr_t)f;
    makecontext(&f->uc, (void (*)(void))fiber_entry, 2, (unsigned)(p >> 32), (unsigned)(p & 0xFFFFFFFFu));
    return 1;
}

/* Releases the host stack. The fiber must not be the one running, nor be resumed again. */
static void host_destroy(Fiber* f)
{
    free(f->host_stack);
    f->host_stack = NULL;
}

static void host_resume(Sched* s, Fiber* f)
{
    swapcontext(&s->back, &f->uc);
}

static void host_yield(Fiber* f)
{
    swapcontext(&f->uc, &f->sched->back);
}
#endif

/* The fiber has ended with `status`: record it and make the fibers that joined it runnable. The lock is held. */
static void fiber_finish_locked(Fiber* f, s32 status)
{
    f->status = status;
    f->state  = F_DONE;
    if (fiber_lifecycle_trace())
        fprintf(stderr, "[fiber-life] finish ea=%08X entry=%08X status=%08X\n",
                f->ea, f->opd, (u32)status);
    for (Fiber* g = s_fibers; g; g = g->next)
        if (g->state == F_WAIT_JOIN && g->join_ea == f->ea) g->state = F_READY;
}

/* Leaves the fiber for good, back into RunFibers. */
static void fiber_leave(Fiber* f)
{
#ifdef _WIN32
    SwitchToFiber(f->sched->back);
#else
    setcontext(&f->sched->back);
#endif
    abort();                     /* an ended fiber is never resumed */
}

/* The fiber's first and last frame, on its own host stack. */
static void fiber_body(Fiber* f)
{
    f->ctx.gpr[1] = (f->stack_ea + f->stack_size - FIBER_STACK_TOP_PAD) & ~0xFull;
    f->ctx.gpr[3] = f->arg;
    f->ctx.cia    = f->opd;
    if (g_ppu_thread_entry_trampoline)
        g_ppu_thread_entry_trampoline(&f->ctx);   /* runs the entry to completion; its r3 is the status */

    LOCK();
    fiber_finish_locked(f, (s32)f->ctx.gpr[3]);
    UNLOCK();
    WAKE();
    fiber_leave(f);
}

/* ---------------------------------------------------------------------------
 * Initialisation
 * -----------------------------------------------------------------------*/

s32 _cellFiberPpuInitialize(void)
{
    return CELL_OK;
}

s32 _cellFiberPpuSchedulerAttributeInitialize(CellFiberPpuSchedulerAttribute* attr, u32 sdkVersion)
{
    (void)sdkVersion;
    if (!attr) return (s32)CELL_FIBER_ERROR_NULL_POINTER;
    vm_memset((u32)(uintptr_t)attr, 0, CELL_FIBER_PPU_SCHEDULER_ATTRIBUTE_SIZE);
    return CELL_OK;
}

s32 _cellFiberPpuAttributeInitialize(CellFiberPpuAttribute* attr, u32 sdkVersion)
{
    (void)sdkVersion;
    if (!attr) return (s32)CELL_FIBER_ERROR_NULL_POINTER;
    vm_memset((u32)(uintptr_t)attr, 0, CELL_FIBER_PPU_ATTRIBUTE_SIZE);
    return CELL_OK;
}

s32 cellFiberPpuInitializeScheduler(CellFiberPpuScheduler* scheduler, const CellFiberPpuSchedulerAttribute* attr)
{
    (void)attr;
    const u32 ea = (u32)(uintptr_t)scheduler;
    if (!ea) return (s32)CELL_FIBER_ERROR_NULL_POINTER;
    if (ea % CELL_FIBER_PPU_SCHEDULER_ALIGN) return (s32)CELL_FIBER_ERROR_ALIGN;

    LOCK();
    s32 rc = CELL_OK;
    if (find_sched(ea)) {
        rc = (s32)CELL_FIBER_ERROR_STAT;
    } else {
        Sched* s = (Sched*)calloc(1, sizeof *s);
        if (!s) {
            rc = (s32)CELL_FIBER_ERROR_NOMEM;
        } else {
            s->ea   = ea;
            s->next = s_scheds;
            s_scheds = s;
        }
    }
    UNLOCK();
    return rc;
}

s32 cellFiberPpuFinalizeScheduler(CellFiberPpuScheduler* scheduler)
{
    const u32 ea = (u32)(uintptr_t)scheduler;
    if (!ea) return (s32)CELL_FIBER_ERROR_NULL_POINTER;

    LOCK();
    Sched* s = find_sched(ea);
    s32 rc = CELL_OK;
    if (!s) {
        rc = (s32)CELL_FIBER_ERROR_INVAL;
    } else if (s->running) {
        rc = (s32)CELL_FIBER_ERROR_STAT;
    } else {
        /* Fibers that never ran to their end are dropped with their host stacks. Nothing resumes them: RunFibers is
         * not running, and a JoinFiber waiting for one finds it gone and fails with INVAL. */
        for (Fiber** pf = &s_fibers; *pf;) {
            Fiber* f = *pf;
            if (f->sched == s) {
                *pf = f->next;
                host_destroy(f);
                free(f);
            } else {
                pf = &f->next;
            }
        }
        for (Sched** ps = &s_scheds; *ps; ps = &(*ps)->next) {
            if (*ps == s) { *ps = s->next; break; }
        }
        free(s);
    }
    UNLOCK();
    WAKE();
    return rc;
}

s32 cellFiberPpuCreateFiber(CellFiberPpuScheduler* scheduler, CellFiberPpu* fiber, u32 entry, u64 arg,
                            u32 priority, void* eaStack, u32 sizeStack, const CellFiberPpuAttribute* attr)
{
    (void)attr;
    const u32 sched_ea = (u32)(uintptr_t)scheduler;
    const u32 fiber_ea = (u32)(uintptr_t)fiber;
    const u32 stack_ea = (u32)(uintptr_t)eaStack;
    if (!sched_ea || !fiber_ea || !entry || !stack_ea) return (s32)CELL_FIBER_ERROR_NULL_POINTER;
    if (fiber_ea % CELL_FIBER_PPU_ALIGN) return (s32)CELL_FIBER_ERROR_ALIGN;
    if (sizeStack < FIBER_MIN_STACK) return (s32)CELL_FIBER_ERROR_INVAL;

    Fiber* f = (Fiber*)calloc(1, sizeof *f);
    if (!f) return (s32)CELL_FIBER_ERROR_NOMEM;
    f->ea         = fiber_ea;
    f->opd        = entry;
    f->arg        = arg;
    f->priority   = priority;
    f->stack_ea   = stack_ea;
    f->stack_size = sizeStack;
    f->state      = F_READY;

    /* r13 stays the creator's, so the fiber's TLS is the creating thread's (0 lets the trampoline pick a block). */
    const ppu_context* parent = g_active_ctx;
    f->ctx.gpr[13]   = parent ? parent->gpr[13] : 0;
    f->ctx.thread_id = parent ? parent->thread_id : FIBER_FALLBACK_THREAD_ID;

    if (!host_create(f)) {
        free(f);
        return (s32)CELL_FIBER_ERROR_NOMEM;
    }

    LOCK();
    Sched* s = find_sched(sched_ea);
    s32 rc = CELL_OK;
    if (!s) {
        rc = (s32)CELL_FIBER_ERROR_INVAL;
    } else if (find_fiber(fiber_ea)) {
        Fiber* previous = find_fiber(fiber_ea);
        fprintf(stderr, "[fiber-life] create rejected ea=%08X entry=%08X "
                "previous_state=%d previous_entry=%08X previous_status=%08X\n",
                fiber_ea, entry, previous->state, previous->opd, (u32)previous->status);
        rc = (s32)CELL_FIBER_ERROR_STAT;
    } else {
        f->sched = s;
        f->seq   = s_seq++;
        f->next  = s_fibers;
        s_fibers = f;
        if (fiber_lifecycle_trace())
            fprintf(stderr, "[fiber-life] create ea=%08X entry=%08X arg=%llX "
                    "stack=%08X+%X scheduler=%08X\n", fiber_ea, entry,
                    (unsigned long long)arg, stack_ea, sizeStack, sched_ea);
    }
    UNLOCK();
    if (rc != CELL_OK) {
        host_destroy(f);
        free(f);
        return rc;
    }
    WAKE();
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Running
 * -----------------------------------------------------------------------*/

s32 cellFiberPpuRunFibers(CellFiberPpuScheduler* scheduler)
{
    const u32 ea = (u32)(uintptr_t)scheduler;
    if (!ea) return (s32)CELL_FIBER_ERROR_NULL_POINTER;

    LOCK();
    Sched* s = find_sched(ea);
    if (!s) { UNLOCK(); return (s32)CELL_FIBER_ERROR_INVAL; }
    if (s->running || t_cur) { UNLOCK(); return (s32)CELL_FIBER_ERROR_STAT; }
    s->running = 1;

    for (;;) {
        Fiber* f = pick_runnable(s);
        if (!f) {
            if (all_done(s)) break;
            WAIT();              /* its fibers wait for a signal, or a join, that another thread delivers */
            continue;
        }
        f->state = F_RUNNING;
        UNLOCK();

        ppu_context* saved_active = g_active_ctx;
        t_cur        = f;
        g_active_ctx = &f->ctx;
        host_resume(s, f);       /* back here when f blocks, yields or ends */
        g_active_ctx = saved_active;
        t_cur        = NULL;

        LOCK();
        if (f->state == F_DONE)
            host_destroy(f);     /* off its stack now; the Fiber stays for JoinFiber's status */
    }
    s->running = 0;
    UNLOCK();
    return CELL_OK;
}

s32 cellFiberPpuWaitSignal(void)
{
    Fiber* f = t_cur;
    if (!f) return (s32)CELL_FIBER_ERROR_PERM;

    LOCK();
    if (f->signal_pending) {
        f->signal_pending = 0;
        UNLOCK();
        return CELL_OK;
    }
    f->state = F_WAIT_SIGNAL;
    UNLOCK();
    host_yield(f);               /* RunFibers runs the others until SendSignal makes this one ready */
    return CELL_OK;
}

s32 cellFiberPpuSendSignal(CellFiberPpu* fiber, u32* numWorker)
{
    const u32 ea = (u32)(uintptr_t)fiber;
    if (!ea) return (s32)CELL_FIBER_ERROR_NULL_POINTER;

    LOCK();
    Fiber* f = find_fiber(ea);
    if (!f) { UNLOCK(); return (s32)CELL_FIBER_ERROR_INVAL; }
    if (f->state == F_WAIT_SIGNAL)  f->state = F_READY;
    else if (f->state != F_DONE)    f->signal_pending = 1;
    UNLOCK();
    WAKE();
    if (numWorker) vm_write32((u32)(uintptr_t)numWorker, 0);
    return CELL_OK;
}

s32 cellFiberPpuYield(void)
{
    Fiber* f = t_cur;
    if (!f) return (s32)CELL_FIBER_ERROR_PERM;

    LOCK();
    f->state = F_READY;
    f->seq   = s_seq++;          /* behind the other fibers of its priority */
    UNLOCK();
    host_yield(f);
    return CELL_OK;
}

s32 cellFiberPpuExit(s32 status)
{
    Fiber* f = t_cur;
    if (!f) return (s32)CELL_FIBER_ERROR_PERM;

    LOCK();
    fiber_finish_locked(f, status);
    UNLOCK();
    WAKE();
    fiber_leave(f);
    return CELL_OK;              /* not reached */
}

CellFiberPpu* cellFiberPpuSelf(void)
{
    const Fiber* f = t_cur;
    return (CellFiberPpu*)(uintptr_t)(f ? f->ea : 0);
}

/* Waits for the fiber to end and returns its exit status. Called from a fiber it hands control back to RunFibers
 * meanwhile (blocking the host thread would stop the scheduler that has to run the fiber waited for); from any other
 * thread it blocks. */
s32 cellFiberPpuJoinFiber(CellFiberPpu* fiber, s32* status)
{
    const u32 ea = (u32)(uintptr_t)fiber;
    if (!ea) return (s32)CELL_FIBER_ERROR_NULL_POINTER;

    Fiber* self = t_cur;
    LOCK();
    Fiber* f = find_fiber(ea);
    if (!f || f == self) { UNLOCK(); return (s32)CELL_FIBER_ERROR_INVAL; }
    if (self && f->state != F_DONE) {
        self->state   = F_WAIT_JOIN;
        self->join_ea = ea;
        UNLOCK();
        host_yield(self);        /* ready again once the fiber has ended */
        LOCK();
        self->join_ea = 0;
    }
    while ((f = find_fiber(ea)) != NULL && f->state != F_DONE)
        WAIT();
    s32 rc = CELL_OK;
    if (!f) rc = (s32)CELL_FIBER_ERROR_INVAL;      /* the scheduler was finalized under the join */
    else if (status) vm_write32((u32)(uintptr_t)status, (u32)f->status);
    if (fiber_lifecycle_trace())
        fprintf(stderr, "[fiber-life] join ea=%08X rc=%08X state=%d\n",
                ea, (u32)rc, f ? f->state : -1);
    UNLOCK();
    return rc;
}
