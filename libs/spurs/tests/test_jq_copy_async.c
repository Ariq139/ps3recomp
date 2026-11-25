/* PushSync must accept a tag mask and order outstanding jobs, without
 * dispatching that mask as a descriptor or writing through spare registers. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include "../cellSpursJq.c"

static uint8_t guest[0x10000];
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

static HANDLE entered, release_job;
static volatile LONG freed;
static int fail_alloc;
int32_t ps3_guest_memory_alloc(uint32_t size,uint32_t align,uint32_t* out) {assert(size==128 && align==128);if(fail_alloc)return -1;*out=0x3000;return 0;}
int32_t ps3_guest_memory_free(uint32_t ea) {assert(ea==0x3000);InterlockedIncrement(&freed);return 0;}
void spurs_job_set_spu_num(uint32_t n) {(void)n;}
void spu_workload_job_cache_thread_cleanup(void) {}
int spu_workload_dispatch_job_snapshot(const uint8_t* image,uint32_t size,uint32_t ea,uint32_t ds,const uint8_t* desc) {
 assert(size==16 && ds==128 && image[0]==0xCC);assert(ea==(fail_alloc?0x1400u:0x3000u));
 assert(vm_read32(ea+0x60)==0x12345678 && desc[0x60]==0x12);SetEvent(entered);
 assert(WaitForSingleObject(release_job,3000)==WAIT_OBJECT_0);assert(vm_read32(ea+0x60)==0x12345678);return 1;
}
static void test_shared_images(void) {
 assert(_putenv_s("SPURS_JQ_SHARE_IMAGES","1")==0);
 vm_write64(0x1400,0x1800);vm_write16(0x1408,1);guest[0x1800]=0x11;
 JqTask* a=jq_prepare(0x1400,128);
 JqTask* b=jq_prepare(0x1400,128);
 assert(a && b && a!=b && a->image && a->binary==b->binary);
 assert(a->image->refs==2 && a->binary[0]==0x11);
 b->descriptor[0x60]=0x77;assert(a->descriptor[0x60]!=0x77);
 guest[0x1800]=0x22;
 JqTask* c=jq_prepare(0x1400,128);
 assert(c && c->binary!=a->binary && c->binary[0]==0x22 && a->binary[0]==0x11);
 jq_free_task(a);assert(b->binary[0]==0x11);jq_free_task(b);
 JqTask* d=jq_prepare(0x1400,128);
 assert(d && d->binary==c->binary);
 jq_free_task(c);jq_free_task(d);
 for(unsigned i=0;i<JQ_IMAGE_SLOTS;++i)assert(!s_jq_images[i]);
}
int main(int argc,char** argv) {
 (void)argv;
 test_shared_images();
 assert(_putenv_s("SPURS_JQ_ASYNC_COPY",argc>1 ? "0" : "1")==0);
 entered=CreateEvent(NULL,TRUE,FALSE,NULL);release_job=CreateEvent(NULL,TRUE,FALSE,NULL);assert(jq_create_state(0x1000,4)==0);
 vm_write64(0x1400,0x1800);vm_write16(0x1408,1);vm_write32(0x1460,0x12345678);guest[0x1800]=0xCC;
 if(argc>1) {
  fail_alloc=1;SetEvent(release_job);
  assert(jq_push_slot_mode(0,0x1400,128,7,0,1)==0);
  assert(s_queues[0].jobCount==0 && freed==0);
  assert(cellSpursShutdownJobQueue()==0);
  CloseHandle(entered);CloseHandle(release_job);
  puts("CopyPush synchronous compatibility: no borrowed descriptor survives return passed");return 0;
 }
 assert(jq_register_port(0x1200,0)==0);
 assert((s32)_cellSpursJobQueuePort2CopyPushJobBody(0x1200,0x1400,128,128,7,1,0,0)==0);assert(WaitForSingleObject(entered,1000)==WAIT_OBJECT_0);
 memset(guest+0x1400,0xEE,128);guest[0x1800]=0xDD;assert(s_queues[0].jobCount==1 && freed==0);
 SetEvent(release_job);assert(cellSpursShutdownJobQueue()==0);assert(freed==1);
 fail_alloc=1;int slot=jq_create_state(0x2000,4);assert(slot>=0);
 vm_write64(0x1400,0x1800);vm_write16(0x1408,1);vm_write32(0x1460,0x12345678);guest[0x1800]=0xCC;
 assert(jq_push_slot_mode(slot,0x1400,128,7,0,1)==0);assert(s_queues[slot].jobCount==0 && freed==1);
 CloseHandle(entered);CloseHandle(release_job);puts("CopyPush: async return, descriptor lifetime, binary reuse, cleanup, allocation fallback passed");
}
