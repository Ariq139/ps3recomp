/* Exercise the real staging/entry/cleanup path with a deterministic clock. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../spurs_job.c"
static uint8_t guest[65536];
uint8_t* vm_base = guest;
static unsigned clocks, entries, unregistered, mfc_released;
double spu_job_profile_now(void) { return ++clocks * 0.001; }
void spu_coh_unregister(spu_context* ctx) { (void)ctx; ++unregistered; }
void spu_mfc_release(spu_context* ctx) { (void)ctx; ++mfc_released; }
int spu_workload_get_job_abi(int image_id, spu_job_abi* abi)
{ (void)image_id; memset(abi,0,sizeof(*abi)); abi->context_tag=20; return 0; }
int spu_run_with_halt(void (*entry)(spu_context*), spu_context* ctx)
{ entry(ctx); return 0; }
static uint32_t local32(const uint8_t* p)
{ return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3]; }
static void entry(spu_context* ctx)
{
    ++entries;
    assert(ctx->ls[0] == 0x5A); /* unchanged binary staging */
    uint32_t io = local32(ctx->ls + ctx->gpr[3]._u32[0] + JC_IO_BUFFER);
    assert(memcmp(ctx->ls + io, guest + 0x3000, 16) == 0);
}
static void expect_time(double a,double b) { assert(fabs(a-b)<1e-12); }
int main(void)
{
    const uint32_t job=0x1000;
    memset(guest + 0x2000,0x5A,128); memset(guest+0x3000,0x42,16);
    ls32(guest,job+JH_EA_BINARY+4,0x2000);
    ls16(guest,job+JH_SIZE_BINARY,8);
    ls16(guest,job+JH_SIZE_DMA_LIST,8);
    ls32(guest,job+JH_SIZE_IN_OR_INOUT,16);
    ls32(guest,job+JH_SIZE,16); ls32(guest,job+JH_SIZE+4,0x3000);
    assert(spu_run_spurs_job_with_abi(entry,17,job,256,NULL)==0);
    assert(clocks==0 && entries==1 && unregistered==1 && mfc_released==1);
    spu_job_phase_times times={9,9,9,9,9};
    assert(spu_run_spurs_job_profiled(entry,17,job,256,NULL,NULL,NULL,&times)==0);
    assert(clocks==5 && entries==2 && unregistered==2 && mfc_released==2);
    expect_time(times.lookup,0); expect_time(times.reset,.001); expect_time(times.stage,.001);
    expect_time(times.execute,.001); expect_time(times.teardown,.001);
    ls16(guest,job+JH_SIZE_BINARY,0);
    times.lookup=times.reset=times.stage=times.execute=times.teardown=9;
    assert(spu_run_spurs_job_profiled(entry,17,job,256,NULL,NULL,NULL,&times)<0);
    assert(clocks==7 && entries==2 && unregistered==2 && mfc_released==2);
    expect_time(times.reset,.001); expect_time(times.lookup,0); expect_time(times.stage,0);
    expect_time(times.execute,0); expect_time(times.teardown,0);
    assert(spu_run_spurs_job_with_abi(entry,17,job,256,NULL)<0);
    assert(clocks==7);
    puts("job phases: staged payload preserved; clocks disabled by default; phases reset PASS");
    return 0;
}
