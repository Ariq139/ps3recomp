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

static HANDLE entered_a, release_a, entered_b, release_b, entered_c, entered_other, blocking_returned;
int32_t ps3_guest_memory_alloc(uint32_t n,uint32_t a,uint32_t* out) {(void)n;(void)a;*out=0;return -1;}
int32_t ps3_guest_memory_free(uint32_t ea) {(void)ea;return 0;}
void spurs_job_set_spu_num(uint32_t n) {(void)n;}
void spu_workload_job_cache_thread_cleanup(void) {}
int spu_workload_dispatch_job_snapshot(const uint8_t* image,uint32_t size,uint32_t ea,uint32_t ds,const uint8_t* desc) {
 (void)image;(void)size;(void)ds;(void)desc;
 if(ea==0x1400){SetEvent(entered_a);assert(WaitForSingleObject(release_a,3000)==WAIT_OBJECT_0);}
 else if(ea==0x1500){SetEvent(entered_b);assert(WaitForSingleObject(release_b,3000)==WAIT_OBJECT_0);}
 else if(ea==0x1600)SetEvent(entered_c);
 else {assert(ea==0x1700);SetEvent(entered_other);}
 return 1;
}
static void submit(unsigned port,unsigned job,unsigned flags) {
 assert((s32)_cellSpursJobQueuePort2PushJobBody(port,job,128,0,flags,0,0,0)==0);
}
static DWORD WINAPI blocking_submit(void* ignored) {
 (void)ignored;submit(0x1200,0x1700,0);SetEvent(blocking_returned);return 0;
}
/* Exercise out-of-order removal independently of worker scheduling. */
static void test_dense_pending(void) {
 JqState* q=&s_queues[0];
 JqTask a={0}, b={0}, c={0}, d={0};
 a.id=1; a.sequence=1; a.tag=7; a.tag_serial=1;
 b.id=2; b.sequence=2; b.flags=JQ_FLAG_SYNC_JOB;
 c.id=3; c.sequence=3; c.tag=7; c.tag_serial=2;
 d.id=4; d.sequence=4; d.tag=7; d.tag_serial=3;
 uint64_t target[32]={0}; target[7]=2;
 jq_lock();
 jq_pending_insert(q,&a); jq_pending_insert(q,&b); jq_pending_insert(q,&c);
 assert(q->jobCount==3 && !jq_task_runnable(&b) && !jq_task_runnable(&c));
 jq_pending_remove(q,&a);
 assert(q->jobCount==2 && q->pending[0]==&c && c.pending_index==0);
 assert(!jq_pending_id(q,1) && jq_pending_id(q,3));
 assert(jq_task_runnable(&b) && !jq_task_runnable(&c));
 assert(jq_pending_tag_before(0,1u<<7,target));
 jq_pending_insert(q,&d);
 jq_pending_remove(q,&b);
 assert(jq_task_runnable(&c) && jq_task_runnable(&d));
 assert(d.pending_index==1 && q->pending[1]==&d);
 jq_pending_remove(q,&c);
 assert(!jq_pending_tag_before(0,1u<<7,target));
 assert(jq_pending_id(q,4) && d.pending_index==0);
 jq_pending_remove(q,&d);
 assert(q->jobCount==0 && q->pending[0]==NULL);
 jq_unlock();
}
int main(int argc,char** argv) {
 (void)argv;
 test_dense_pending();

 assert(_putenv_s("SPURS_JQ_WORKERS","4")==0);
 blocking_returned=CreateEvent(NULL,TRUE,FALSE,NULL);
 entered_a=CreateEvent(NULL,TRUE,FALSE,NULL);release_a=CreateEvent(NULL,TRUE,FALSE,NULL);
 entered_b=CreateEvent(NULL,TRUE,FALSE,NULL);release_b=CreateEvent(NULL,TRUE,FALSE,NULL);
 entered_c=CreateEvent(NULL,TRUE,FALSE,NULL);entered_other=CreateEvent(NULL,TRUE,FALSE,NULL);
 assert(jq_create_state(0x1000,3)==0 && jq_create_state(0x2000,3)==1);
 assert(jq_register_port(0x1200,0)==0 && jq_register_port(0x2200,1)==0);
 for(unsigned ea=0x1400;ea<=0x1700;ea+=0x100){vm_write64(ea,0x1800);vm_write16(ea+8,1);}
 submit(0x1200,0x1400,0);assert(WaitForSingleObject(entered_a,1000)==WAIT_OBJECT_0);
 submit(0x1200,0x1500,argc>1 ? JQ_FLAG_EXCLUSIVE_JOB : JQ_FLAG_SYNC_JOB);submit(0x1200,0x1600,0);
 submit(0x2200,0x1700,0);assert(WaitForSingleObject(entered_other,1000)==WAIT_OBJECT_0);
 assert(WaitForSingleObject(entered_b,30)==WAIT_TIMEOUT && WaitForSingleObject(entered_c,30)==WAIT_TIMEOUT);
 assert((s32)_cellSpursJobQueuePort2PushJobBody(0x1200,0x1700,128,0,JQ_FLAG_NON_BLOCKING,0,0,0)==(s32)CELL_SPURS_JQ_ERROR_QUEUE_FULL);
 HANDLE blocked=CreateThread(NULL,0,blocking_submit,NULL,0,NULL);assert(blocked);
 assert(WaitForSingleObject(blocking_returned,30)==WAIT_TIMEOUT);
 SetEvent(release_a);assert(WaitForSingleObject(entered_b,1000)==WAIT_OBJECT_0);
 assert(WaitForSingleObject(blocking_returned,1000)==WAIT_OBJECT_0);
 assert(WaitForSingleObject(blocked,1000)==WAIT_OBJECT_0);CloseHandle(blocked);CloseHandle(blocking_returned);

 assert(WaitForSingleObject(entered_c,30)==WAIT_TIMEOUT);
 SetEvent(release_b);assert(WaitForSingleObject(entered_c,1000)==WAIT_OBJECT_0);
 assert(cellSpursShutdownJobQueue()==0);
 CloseHandle(entered_a);CloseHandle(release_a);CloseHandle(entered_b);CloseHandle(release_b);CloseHandle(entered_c);CloseHandle(entered_other);
 puts("JQ flags: sync barrier, younger-job exclusion, independent queue progress, blocking capacity and nonblocking capacity passed");
}
