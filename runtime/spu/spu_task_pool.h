/*
 * ps3recomp - a host thread pool for async SPU tasks (Windows/POSIX)
 *
 * spu_workload_dispatch_async_owned runs each SPU task on its own detached host
 * thread, and a title can start several per frame. Each new thread pays the
 * creation, a cold stack and fresh thread-local state again. This pool runs
 * the tasks on reused threads without changing what a task can rely on:
 *
 *  - Every task still starts at once, on a thread of its own. A task is handed
 *    to the pool only when more pool threads are idle and waiting than tasks
 *    are already queued for them; otherwise a new pool thread is created for
 *    it, exactly as without the pool. So a task that waits for another task
 *    can never be queued behind it, and there are never fewer threads than
 *    running tasks.
 *  - A pool thread that has been idle for idle_ms without work exits.
 *
 * The pool owns no task memory. spu_task_pool_submit() takes an item the
 * caller embeds in its own task record; the item must stay valid until the
 * task starts running, and the run function may free it.
 *
 * Header-only so the synthetic test (tests/test_spu_task_pool.c) can drive it
 * directly. Windows uses SRW locks and CRT-aware native workers.
 */
#ifndef PS3RECOMP_SPU_TASK_POOL_H
#define PS3RECOMP_SPU_TASK_POOL_H

#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <limits.h>
#else
#include <errno.h>
#include <pthread.h>
#include <time.h>
#endif

struct spu_task_pool;

typedef struct spu_task_pool_item {
    void (*run)(void* arg);
    void* arg;
    struct spu_task_pool* pool;
    struct spu_task_pool_item* next;
} spu_task_pool_item;

typedef struct spu_task_pool {
#ifdef _WIN32
    SRWLOCK mu;
    CONDITION_VARIABLE cv;
#else
    pthread_mutex_t mu;
    pthread_cond_t  cv;
#endif
    spu_task_pool_item* head;
    spu_task_pool_item* tail;
    unsigned queued;               /* handed off, not yet picked up */
    unsigned idle;                 /* pool threads waiting for work */
    unsigned threads;              /* pool threads alive */
    unsigned long long reused;     /* tasks handed to an idle thread */
    unsigned long long created;    /* tasks that got a new thread */
    unsigned long long retired;    /* threads that exited idle */
    unsigned idle_ms;              /* idle time before a thread exits */
    size_t   stack_size;           /* per-thread stack reservation; 0 = default */
} spu_task_pool;

#ifdef _WIN32
#define SPU_TASK_POOL_INIT(idle_ms, stack_size) \
    { SRWLOCK_INIT, CONDITION_VARIABLE_INIT, NULL, NULL, 0, 0, 0, 0, 0, 0, (idle_ms), (stack_size) }
static inline void spu_task_pool_lock(spu_task_pool* p) { AcquireSRWLockExclusive(&p->mu); }
static inline void spu_task_pool_unlock(spu_task_pool* p) { ReleaseSRWLockExclusive(&p->mu); }
static inline void spu_task_pool_signal(spu_task_pool* p) { WakeConditionVariable(&p->cv); }
#else
#define SPU_TASK_POOL_INIT(idle_ms, stack_size) \
    { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, NULL, 0, 0, 0, 0, 0, 0, (idle_ms), (stack_size) }
static inline void spu_task_pool_lock(spu_task_pool* p) { pthread_mutex_lock(&p->mu); }
static inline void spu_task_pool_unlock(spu_task_pool* p) { pthread_mutex_unlock(&p->mu); }
static inline void spu_task_pool_signal(spu_task_pool* p) { pthread_cond_signal(&p->cv); }
#endif

#ifdef _WIN32
static unsigned __stdcall spu_task_pool_thread(void* p)
#else
static inline void* spu_task_pool_thread(void* p)
#endif
{
    spu_task_pool_item* it = (spu_task_pool_item*)p;
    spu_task_pool* pool = it->pool;
    for (;;) {
        /* Copy out first: the task may free the record the item lives in. */
        void (*run)(void*) = it->run;
        void* arg = it->arg;
        run(arg);

        spu_task_pool_lock(pool);
        pool->idle++;
#ifdef _WIN32
        const ULONGLONG deadline=GetTickCount64()+pool->idle_ms;
        while (!pool->head) {
            const ULONGLONG now=GetTickCount64();
            if (now>=deadline) break;
            if (!SleepConditionVariableSRW(&pool->cv,&pool->mu,(DWORD)(deadline-now),0)
                && GetLastError()!=ERROR_TIMEOUT) break;
        }
#else
        struct timespec until;
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_sec  += (time_t)(pool->idle_ms / 1000u);
        until.tv_nsec += (long)(pool->idle_ms % 1000u) * 1000000L;
        if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
        int timed_out = 0;
        while (!pool->head && !timed_out)
            timed_out = pthread_cond_timedwait(&pool->cv, &pool->mu, &until) != 0;
#endif
        pool->idle--;
        /* A queued task is taken even after a timeout: it was handed off
         * while this thread still counted as idle. */
        if (!pool->head) {
            pool->threads--;
            pool->retired++;
            spu_task_pool_unlock(pool);
            return 0;
        }
        it = pool->head;
        pool->head = it->next;
        if (!pool->head) pool->tail = NULL;
        pool->queued--;
        spu_task_pool_unlock(pool);
    }
}

/* Start run(arg) now: on an idle pool thread if one is free, else on a new
 * one. 1 when the task is running or about to, 0 when no thread could be
 * created (the item is then unused and the caller still owns its task). */
static inline int spu_task_pool_submit(spu_task_pool* pool, spu_task_pool_item* it,
                                void (*run)(void*), void* arg)
{
    it->run = run;
    it->arg = arg;
    it->pool = pool;
    it->next = NULL;

    spu_task_pool_lock(pool);
    if (pool->idle > pool->queued) {
        if (pool->tail) pool->tail->next = it; else pool->head = it;
        pool->tail = it;
        pool->queued++;
        pool->reused++;
        spu_task_pool_signal(pool);
        spu_task_pool_unlock(pool);
        return 1;
    }
    /* Counted before the thread runs, so a later hand-off sees it. */
    pool->threads++;
    pool->created++;
    spu_task_pool_unlock(pool);

    #ifdef _WIN32
    uintptr_t th = 0;
    if (pool->stack_size <= UINT_MAX)
        th = _beginthreadex(NULL, (unsigned)pool->stack_size, spu_task_pool_thread, it,
                           pool->stack_size ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0, NULL);
    if (!th) {
        spu_task_pool_lock(pool); pool->threads--; pool->created--; spu_task_pool_unlock(pool);
        return 0;
    }
    CloseHandle((HANDLE)th);
    #else
    pthread_t th;
    int rc = EINVAL;
    if (pool->stack_size) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        if (pthread_attr_setstacksize(&attr, pool->stack_size) == 0)
            rc = pthread_create(&th, &attr, spu_task_pool_thread, it);
        pthread_attr_destroy(&attr);
    }
    /* A host that will not reserve that much still gets the thread, on the
     * default stack, as the thread-per-task path does. */
    if (rc != 0)
        rc = pthread_create(&th, NULL, spu_task_pool_thread, it);
    if (rc != 0) {
        spu_task_pool_lock(pool);
        pool->threads--;
        pool->created--;
        spu_task_pool_unlock(pool);
        return 0;
    }
    pthread_detach(th);
    #endif
    return 1;
}

#endif /* PS3RECOMP_SPU_TASK_POOL_H */
