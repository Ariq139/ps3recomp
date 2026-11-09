#define main gcm_nop_fixture_main
#include "test_gcm_nop_runs.c"
#undef main
#include <assert.h>

static u32 emit(u32 io,u32 subch,u32 method,u32 value)
{
    gw32(IO_ADDR+io,(1u<<18)|(subch<<13)|method);
    gw32(IO_ADDR+io+4,value);
    return io+8;
}
static void drain_to(u32 end)
{
    vm_write32(GCM_CONTROL_GUEST_ADDR,end);
    cellGcm_rsx_process_fifo();
}
#ifdef _WIN32
static DWORD WINAPI producer(LPVOID unused)
{
    (void)unused;
    for(u32 i=1;i<=2000;++i)gcm_ref_push_at(i,0);
    return 0;
}
static DWORD WINAPI consumer(LPVOID unused)
{
    (void)unused;
    for(u32 expected=1;expected<=2000;) {
        AcquireSRWLockExclusive(&s_ref_pub_lock);
        const int published=gcm_ref_publish_one();
        if(published) {
            assert(vm_read32(GCM_CONTROL_GUEST_ADDR+8)==expected);
            ++expected;
        }
        ReleaseSRWLockExclusive(&s_ref_pub_lock);
        if(!published)SwitchToThread();
    }
    return 0;
}
#endif
int main(int argc,char** argv)
{
    const int bypass=argc==2 && !strcmp(argv[1],"--bypass");
#ifdef _WIN32
    _putenv_s("GCM_SEMA_ACQUIRE",bypass?"0":"");
    _putenv_s("PERF_GCM","1");
#else
    if(bypass)setenv("GCM_SEMA_ACQUIRE","0",1);else unsetenv("GCM_SEMA_ACQUIRE");
    setenv("PERF_GCM","1",1);
#endif
    setvbuf(stdout,NULL,_IONBF,0);
    ppu_vm_size=ppu_hle_inject_base+0x10000u;
    vm_base=(u8*)calloc(1,ppu_vm_size);assert(vm_base);
    u32 out=host_alloc(4,4);
    cellGcmSetupContext(out,CMD_SIZE,IO_SIZE,IO_ADDR,host_alloc,host_w32);
    s_fifo_getoff=0;
    vm_write32(GCM_CONTROL_GUEST_ADDR+4,0);
    u32 io=emit(0,1,0,RSX_OBJECT_NV0039);
    io=emit(io,1,NV0039_LINE_LENGTH_IN,42);
    io=emit(io,1,0,RSX_OBJECT_NV4097);
    io=emit(io,1,PROBE_METHOD,123);
    io=emit(io,3,0,RSX_OBJECT_NV4097);
    io=emit(io,3,PROBE_METHOD,456);
    drain_to(io);
    assert(s_gcm_bound_object[1]==RSX_OBJECT_NV4097);
    assert(s_gcm_bound_object[3]==RSX_OBJECT_NV4097);
    assert(s_nv0039.line_length==42);
    assert(seen_n==2 && seen_d[0]==123 && seen_d[1]==456);

    const u32 wait_at=io;
    io=emit(io,0,NV406E_SEMAPHORE_OFFSET,0x20);
    io=emit(io,0,NV406E_SEMAPHORE_ACQUIRE,0x1234);
    io=emit(io,0,PROBE_METHOD,789);
    vm_write32(GCM_LABEL_GUEST_BASE+0x20,0);
    drain_to(io);
    if(!bypass) {
        assert(s_fifo_getoff==wait_at+8 && seen_n==2);
        assert(s_gcm_perf.on && s_gcm_perf.wait_since);
        drain_to(io);
        assert(s_fifo_getoff==wait_at+8 && seen_n==2);
        vm_write32(GCM_LABEL_GUEST_BASE+0x20,0x1234);
        drain_to(io);
    }
    assert(s_fifo_getoff==io && seen_n==3 && seen_d[2]==789);
    assert(!s_gcm_perf.wait_since);
    if(!bypass)assert(s_gcm_perf.sema_waits==1);
    s_nv0039.input_dma=s_nv0039.output_dma=0xFEED0001u;
    s_nv0039.input_offset=0x2000;s_nv0039.output_offset=0x3000;
    s_nv0039.line_length=32;s_nv0039.line_count=1;s_nv0039.format=0x101;
    for(u32 i=0;i<32;++i)vm_base[IO_ADDR+0x2000+i]=(u8)(i*13u);
    nv0039_buffer_notify();
    assert(!memcmp(vm_base+IO_ADDR+0x2000,vm_base+IO_ADDR+0x3000,32));
    assert(s_gcm_perf.copies==1 && s_gcm_perf.copy_bytes==32);
    io=emit(io,0,0x1808,0);drain_to(io);
    assert(s_gcm_perf.draws==1);

    s_ref_qhead=s_ref_qtail=0;s_ref_last_pub=0;
    gcm_ref_push_at(10,0);gcm_ref_push_at(11,0);
    AcquireSRWLockExclusive(&s_ref_pub_lock);
    assert(gcm_ref_publish_one());
    LARGE_INTEGER now;QueryPerformanceCounter(&now);
    s_ref_last_pub=now.QuadPart+s_ref_frequency.QuadPart;
    assert(!gcm_ref_publish_one());
    ReleaseSRWLockExclusive(&s_ref_pub_lock);
    cellGcm_ref_on_poll();
    assert(vm_read32(GCM_CONTROL_GUEST_ADDR+8)==10 && s_ref_qtail-s_ref_qhead==1);
    s_ref_last_pub=0;
    AcquireSRWLockExclusive(&s_ref_pub_lock);
    assert(gcm_ref_publish_one());
    ReleaseSRWLockExclusive(&s_ref_pub_lock);
    assert(vm_read32(GCM_CONTROL_GUEST_ADDR+8)==11);

    s_ref_qhead=s_ref_qtail=0;s_ref_last_pub=0;
    for(u32 i=1;i<=GCM_REF_QLEN+1;++i)gcm_ref_push_at(i,0);
    assert(s_ref_qhead==1 && s_ref_qtail==GCM_REF_QLEN+1);
    assert(vm_read32(GCM_CONTROL_GUEST_ADDR+8)==1);
    for(u32 i=2;i<=GCM_REF_QLEN+1;++i) {
        AcquireSRWLockExclusive(&s_ref_pub_lock);
        s_ref_last_pub=0;
        assert(gcm_ref_publish_one());
        assert(vm_read32(GCM_CONTROL_GUEST_ADDR+8)==i);
        ReleaseSRWLockExclusive(&s_ref_pub_lock);
    }
    assert(s_ref_qhead==s_ref_qtail);
    s_ref_qhead=s_ref_qtail=0xFFFFFFFEu;
    for(u32 i=1;i<=3;++i)gcm_ref_push_at(i,0);
    for(u32 i=1;i<=3;++i) {
        AcquireSRWLockExclusive(&s_ref_pub_lock);s_ref_last_pub=0;
        assert(gcm_ref_publish_one());
        assert(vm_read32(GCM_CONTROL_GUEST_ADDR+8)==i);
        ReleaseSRWLockExclusive(&s_ref_pub_lock);
    }
#ifdef _WIN32
    s_ref_qhead=s_ref_qtail=0;s_ref_last_pub=0;
    const long long before=g_gcm_ref_pub_count;
    HANDLE threads[2]={CreateThread(NULL,0,producer,NULL,0,NULL),CreateThread(NULL,0,consumer,NULL,0,NULL)};
    assert(threads[0]&&threads[1]);
    assert(WaitForMultipleObjects(2,threads,TRUE,10000)==WAIT_OBJECT_0);
    CloseHandle(threads[0]);CloseHandle(threads[1]);
    assert(s_ref_qhead==s_ref_qtail && g_gcm_ref_pub_count-before==2000);
#endif
    free(vm_base);
    puts("GCM ordering: bindings, rebinds, acquire/resume, shared pacing, queue overflow, wrap and concurrent publication passed");
    return 0;
}
