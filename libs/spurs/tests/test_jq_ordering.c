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
/* Compare the maintained frontiers against the original scan after arbitrary
 * insertions and out-of-order removals, including multiple independent queues. */
static u32 frontier_rng = 0x8f33d7u;
static u32 frontier_next(void) {
 frontier_rng ^= frontier_rng << 13;
 frontier_rng ^= frontier_rng >> 17;
 frontier_rng ^= frontier_rng << 5;
 return frontier_rng;
}
static int scan_runnable(const JqTask* task) {
 const JqState* q = &s_queues[task->queue_slot];
 for (u32 i=0; i<q->jobCount; ++i) {
  const JqTask* e=q->pending[i];
  if (e != task && e->sequence < task->sequence &&
      ((e->flags | task->flags) & (JQ_FLAG_SYNC_JOB | JQ_FLAG_EXCLUSIVE_JOB))) return 0;
 }
 return 1;
}
static void check_frontiers(JqState* q) {
 u32 n=0, b=0, expected_b=0;
 JqTask* prev=NULL;
 for (JqTask* t=q->pending_head; t; t=t->pending_next) {
  assert(n++ < q->jobCount);
  assert(t->pending_prev==prev && (!prev || prev->sequence<t->sequence));
  assert(q->pending[t->pending_index]==t);
  assert(jq_task_runnable(t)==scan_runnable(t));
  expected_b += !!(t->flags & (JQ_FLAG_SYNC_JOB|JQ_FLAG_EXCLUSIVE_JOB));
  prev=t;
 }
 assert(n==q->jobCount && prev==q->pending_tail);
 prev=NULL;
 for (JqTask* t=q->barrier_head; t; t=t->barrier_next) {
  assert(b++ < q->jobCount);
  assert(t->barrier_prev==prev && (!prev || prev->sequence<t->sequence));
  assert(t->flags & (JQ_FLAG_SYNC_JOB|JQ_FLAG_EXCLUSIVE_JOB));
  prev=t;
 }
 assert(b==expected_b && prev==q->barrier_tail);
}
static void test_frontiers(void) {
 static JqTask tasks[2][CELL_SPURS_JQ_MAX_JOBS];
 unsigned char used[2][CELL_SPURS_JQ_MAX_JOBS]={0};
 uint64_t seq=0;
 jq_lock();
 for (u32 step=0; step<20000; ++step) {
  u32 slot=frontier_next()%2, i=frontier_next()%CELL_SPURS_JQ_MAX_JOBS;
  JqState* q=&s_queues[slot];
  JqTask* t=&tasks[slot][i];
  if (used[slot][i]) { jq_pending_remove(q,t); used[slot][i]=0; }
  else {
   memset(t,0,sizeof(*t)); t->queue_slot=slot; t->sequence=++seq;
   t->flags=frontier_next()&7u;
   jq_pending_insert(q,t); used[slot][i]=1;
  }
  check_frontiers(q);
 }
 for (u32 slot=0; slot<2; ++slot) {
  JqState* q=&s_queues[slot];
  while(q->jobCount) { jq_pending_remove(q,q->pending[q->jobCount-1]); check_frontiers(q); }
 }
 jq_unlock();
}

/* Direct comparison with the original outstanding-set scan. Include
 * later submissions past a retained target, zero masks and tag 31. */
static int scan_tag_before(const JqState* q, u32 mask, const uint64_t target[32])
{
 for (u32 i=0; i<q->jobCount; ++i) {
  const JqTask* t=q->pending[i];
  if ((mask & (1u<<t->tag)) && t->tag_serial<=target[t->tag]) return 1;
 }
 return 0;
}
static void check_tags(u32 slot, const uint64_t submitted[32], const uint64_t target[32])
{
 const JqState* q=&s_queues[slot];
 u32 mask=0, total=0;
 for(u32 tag=0;tag<32;++tag) {
  const JqTask* prev=NULL;
  for(const JqTask* t=q->tag_head[tag];t;t=t->tag_next) {
   assert(total++<q->jobCount && t->tag==tag);
   assert(t->tag_prev==prev && (!prev || prev->tag_serial<t->tag_serial));
   assert(t->tag_serial<=submitted[tag] && q->pending[t->pending_index]==t);
   prev=t;
  }
  assert(prev==q->tag_tail[tag]);
  if(prev) mask|=1u<<tag;
 }
 assert(total==q->jobCount && mask==q->pending_tag_mask);
 const u32 masks[]={0u, UINT32_MAX, 1u<<31, 1u<<(frontier_next()%32u), frontier_next()};
 for(u32 k=0;k<sizeof(masks)/sizeof(masks[0]);++k) {
  assert(jq_pending_tag_before(slot,masks[k],target)==scan_tag_before(q,masks[k],target));
  assert(jq_pending_tag_before(slot,masks[k],submitted)==scan_tag_before(q,masks[k],submitted));
 }
}
static void test_tag_frontiers(void)
{
 static JqTask tasks[2][CELL_SPURS_JQ_MAX_JOBS];
 unsigned char used[2][CELL_SPURS_JQ_MAX_JOBS]={0};
 uint64_t submitted[2][32]={{0}}, target[2][32]={{0}}, sequence=0;
 jq_lock();
 for(u32 step=0;step<20000;++step) {
  u32 slot=frontier_next()%2u, i=frontier_next()%CELL_SPURS_JQ_MAX_JOBS;
  JqState* q=&s_queues[slot]; JqTask* t=&tasks[slot][i];
  if(step%37u==0) memcpy(target[slot],submitted[slot],sizeof target[slot]);
  if(used[slot][i]) { jq_pending_remove(q,t); used[slot][i]=0; }
  else {
   memset(t,0,sizeof(*t)); t->queue_slot=slot; t->sequence=++sequence;
   t->tag=frontier_next()%32u; t->tag_serial=++submitted[slot][t->tag];
   t->flags=frontier_next()&7u; jq_pending_insert(q,t); used[slot][i]=1;
  }
  check_tags(slot,submitted[slot],target[slot]);
 }
 for(u32 slot=0;slot<2;++slot) {
  JqState* q=&s_queues[slot];
  while(q->jobCount) { jq_pending_remove(q,q->pending[q->jobCount-1]); check_tags(slot,submitted[slot],target[slot]); }
 }
 jq_unlock();
}

int main(int argc,char** argv) {
 (void)argv;
 test_dense_pending();
 test_frontiers();
 test_tag_frontiers();

 assert(_putenv_s("SPURS_JQ_WORKERS","4")==0);
 assert(_putenv_s("SPURS_JQ_REJECT_PROFILE","1")==0);
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
 jq_lock();
 assert(s_jq_reject_profile.full==1 && s_jq_reject_profile.busy==0 && s_jq_reject_profile.aborted==0);
 assert(s_jq_reject_profile.binary_bytes==16 && s_jq_reject_profile.descriptor_bytes==256);
 jq_unlock();
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
