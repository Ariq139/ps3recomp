/* Actual Win32 pool: reuse, simultaneous mutually dependent tasks, retirement. */
#include "spu_task_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
static SRWLOCK gate_lock=SRWLOCK_INIT;
static CONDITION_VARIABLE gate_cv=CONDITION_VARIABLE_INIT;
static unsigned started,done,peak,running;
typedef struct { spu_task_pool_item item; unsigned gate; } task;
static void run_task(void* p)
{
    task* t=(task*)p;
    AcquireSRWLockExclusive(&gate_lock);
    started++; running++; if(running>peak)peak=running;
    WakeAllConditionVariable(&gate_cv);
    while(started<t->gate)
        if(!SleepConditionVariableSRW(&gate_cv,&gate_lock,10000,0))abort();
    running--;done++;WakeAllConditionVariable(&gate_cv);
    ReleaseSRWLockExclusive(&gate_lock);
    free(t);
}
static void submit(spu_task_pool* pool,unsigned gate)
{
    task* t=(task*)calloc(1,sizeof *t);assert(t);t->gate=gate;
    assert(spu_task_pool_submit(pool,&t->item,run_task,t)==1);
}
static void wait_done(unsigned n)
{
    AcquireSRWLockExclusive(&gate_lock);
    while(done<n)if(!SleepConditionVariableSRW(&gate_cv,&gate_lock,10000,0))abort();
    ReleaseSRWLockExclusive(&gate_lock);
}
static void wait_count(spu_task_pool* pool,unsigned count,int idle)
{
    for(unsigned i=0;i<10000;i++){
        spu_task_pool_lock(pool); unsigned n=idle?pool->idle:pool->threads;spu_task_pool_unlock(pool);
        if(n==count)return;Sleep(1);
    }
    abort();
}
int main(void)
{
    static spu_task_pool pool=SPU_TASK_POOL_INIT(2000u,1u<<20);
    for(unsigned i=0;i<50;i++){submit(&pool,0);wait_done(i+1);wait_count(&pool,1,1);}
    spu_task_pool_lock(&pool);assert(pool.created==1&&pool.reused==49);spu_task_pool_unlock(&pool);
    enum{K=8};
    for(unsigned i=0;i<K;i++)submit(&pool,50+K);
    wait_done(50+K);assert(peak==K);
    wait_count(&pool,K,1);
    wait_count(&pool,0,0);
    spu_task_pool_lock(&pool);assert(!pool.queued&&!pool.idle&&!pool.head&&pool.created==pool.retired);
    spu_task_pool_unlock(&pool);
    puts("PASS Windows SPU task pool reuse, concurrent dependencies, retirement");
}
