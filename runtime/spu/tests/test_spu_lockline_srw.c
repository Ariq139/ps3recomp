/* Exercise the actual optional Windows coherency lock, not a mock or POSIX lock. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <windows.h>
#ifndef PS3RECOMP_SPU_LOCKLINE_SRW
#error Windows SRW test must select the SRW implementation
#endif
#include "../spu_coherency.c"

void spu_ch_wake(spu_context* ctx) { (void)ctx; }

static HANDLE waiting, entered;
static uint64_t count, mirror=UINT64_MAX;
static DWORD WINAPI blocked_reader(void* unused)
{
    (void)unused;
    assert(SetEvent(waiting));
    spu_lockline_lock();
    assert(count==42 && mirror==~UINT64_C(42));
    assert(SetEvent(entered));
    spu_lockline_unlock();
    return 0;
}
static DWORD WINAPI writer(void* unused)
{
    (void)unused;
    for (unsigned i=0;i<50000;++i) {
        spu_lockline_lock();
        assert(mirror==~count);
        ++count;
        mirror=~count;
        spu_lockline_unlock();
    }
    return 0;
}
int main(void)
{
    waiting=CreateEventA(NULL,TRUE,FALSE,NULL);
    entered=CreateEventA(NULL,TRUE,FALSE,NULL);
    assert(waiting && entered);
    spu_lockline_lock();
    HANDLE reader=CreateThread(NULL,0,blocked_reader,NULL,0,NULL);
    assert(reader && WaitForSingleObject(waiting,5000)==WAIT_OBJECT_0);
    assert(WaitForSingleObject(entered,100)==WAIT_TIMEOUT);
    count=42; mirror=~count;
    spu_lockline_unlock();
    assert(WaitForSingleObject(reader,5000)==WAIT_OBJECT_0);
    assert(WaitForSingleObject(entered,0)==WAIT_OBJECT_0);
    CloseHandle(reader); CloseHandle(waiting); CloseHandle(entered);
    HANDLE threads[8];
    count=0; mirror=~count;
    for(unsigned i=0;i<8;++i) {
        threads[i]=CreateThread(NULL,0,writer,NULL,0,NULL);
        assert(threads[i]);
    }
    assert(WaitForMultipleObjects(8,threads,TRUE,20000)==WAIT_OBJECT_0);
    for(unsigned i=0;i<8;++i) CloseHandle(threads[i]);
    assert(count==400000 && mirror==~UINT64_C(400000));
    puts("PASS: Windows SRW coherency lock exclusion and publication");
    return 0;
}
