#include "runtime/platform/win32_compat.h"
#include "runtime/syscalls/sys_cond.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t* vm_base;
static uint32_t mid, cid;
static int failures;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); failures++; } } while (0)
static int32_t call(int64_t (*f)(ppu_context*), ppu_context* c, uint64_t a, uint64_t b, uint64_t d) {
    c->gpr[3]=a; c->gpr[4]=b; c->gpr[5]=d;
    return (int32_t)f(c);
}
static uint32_t be_read(unsigned off) {
    uint8_t* p=vm_base+off;
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3];
}
static DWORD WINAPI signaler(void* raw) {
    ppu_context c={0}; c.thread_id=2;
    Sleep(20);
    if (raw) {
        sys_cond_test_spurious(cid);
        /* Signal a non-waiting thread: must not wake another guest waiter. */
        CHECK(call(sys_cond_signal_to,&c,cid,999,0)==(int32_t)CELL_ESRCH);
        CHECK(call(sys_cond_signal_to,&c,cid,4,0)==(int32_t)CELL_EPERM);
    } else {
        int32_t rc=call(sys_mutex_lock,&c,mid,200000,0);
        if (rc==CELL_OK) {
            CHECK(call(sys_cond_signal_to,&c,cid,1,0)==CELL_OK);
            call(sys_mutex_unlock,&c,mid,0,0);
        }
    }
    return 0;
}
static volatile LONG woke[4], entered[4];
static DWORD WINAPI waiter_thread(void* raw) {
    unsigned tid=(unsigned)(uintptr_t)raw;
    ppu_context c={0}; c.thread_id=tid;
    if (call(sys_mutex_lock,&c,mid,0,0)!=CELL_OK) return 1;
    InterlockedExchange(&entered[tid],1);
    int32_t rc=call(sys_cond_wait,&c,cid,5000000,0);
    InterlockedExchange(&woke[tid],rc==CELL_OK?1:-1);
    call(sys_mutex_unlock,&c,mid,0,0);
    return 0;
}
static void wait_flag(volatile LONG* flag) {
    for (unsigned i=0;i<4000&&!InterlockedCompareExchange(flag,0,0);i++) Sleep(1);
    CHECK(InterlockedCompareExchange(flag,0,0)==1);
}
int main(int argc, char** argv) {
    vm_base=calloc(1,65536);
    ppu_context c={0}; c.thread_id=1;
    vm_base[0x103]=1; vm_base[0x107]=SYS_SYNC_RECURSIVE;
    CHECK(call(sys_mutex_create,&c,0x200,0x100,0)==CELL_OK);
    mid=be_read(0x200);
    CHECK(call(sys_cond_create,&c,0x204,mid,0)==CELL_OK);
    cid=be_read(0x204);
    CHECK(call(sys_mutex_destroy,&c,mid,0,0)==(int32_t)CELL_EBUSY);
    const char* mode=argc>1?argv[1]:"recursive";
    if (!strcmp(mode,"targeted") || !strcmp(mode,"priority")) {
        if (!strcmp(mode,"priority")) g_sys_mutexes[mid-1].protocol=SYS_SYNC_PRIORITY;
        HANDLE a=CreateThread(NULL,0,waiter_thread,(void*)(uintptr_t)2,0,NULL);
        wait_flag(&entered[2]);
        CHECK(call(sys_mutex_lock,&c,mid,0,0)==CELL_OK);
        CHECK(call(sys_mutex_unlock,&c,mid,0,0)==CELL_OK);
        HANDLE b=CreateThread(NULL,0,waiter_thread,(void*)(uintptr_t)3,0,NULL);
        wait_flag(&entered[3]);
        CHECK(call(sys_mutex_lock,&c,mid,0,0)==CELL_OK);
        CHECK((!strcmp(mode,"priority") ? call(sys_cond_signal,&c,cid,0,0) :
                                      call(sys_cond_signal_to,&c,cid,3,0))==CELL_OK);
        CHECK(call(sys_mutex_unlock,&c,mid,0,0)==CELL_OK);
        wait_flag(&woke[3]);
        CHECK(InterlockedCompareExchange(&woke[2],0,0)==0);
        CHECK(call(sys_cond_signal_to,&c,cid,2,0)==CELL_OK);
        CHECK(WaitForSingleObject(a,6000)==WAIT_OBJECT_0);
        CHECK(WaitForSingleObject(b,6000)==WAIT_OBJECT_0);
        CloseHandle(a); CloseHandle(b);
        CHECK(woke[2]==1 && woke[3]==1);
    } else if (!strcmp(mode,"owner")) {
        CHECK(call(sys_cond_wait,&c,cid,1000,0)==(int32_t)CELL_EPERM);
    } else {
        CHECK(call(sys_mutex_lock,&c,mid,0,0)==CELL_OK);
        int spurious=!strcmp(mode,"spurious");
        if (!spurious) CHECK(call(sys_mutex_lock,&c,mid,0,0)==CELL_OK);
        HANDLE h=CreateThread(NULL,0,signaler,(void*)(uintptr_t)spurious,0,NULL);
        CHECK(h!=NULL);
        int32_t rc=call(sys_cond_wait,&c,cid,100000,0);
        CHECK(rc==(spurious?(int32_t)CELL_ETIMEDOUT:CELL_OK));
        CHECK(g_sys_mutexes[mid-1].owner_tid==1);
        CHECK(g_sys_mutexes[mid-1].lock_count==(spurious?1:2));
        CHECK(call(sys_mutex_unlock,&c,mid,0,0)==CELL_OK);
        if (!spurious) CHECK(call(sys_mutex_unlock,&c,mid,0,0)==CELL_OK);
        CHECK(WaitForSingleObject(h,1000)==WAIT_OBJECT_0);
        CloseHandle(h);
    }
    CHECK(call(sys_cond_destroy,&c,cid,0,0)==CELL_OK);
    CHECK(call(sys_mutex_destroy,&c,mid,0,0)==CELL_OK);
    printf("condition %s: %s\n",mode,failures?"FAIL":"PASS");
    return failures?1:0;
}
