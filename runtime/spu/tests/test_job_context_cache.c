#include <windows.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
static volatile LONG freed;
static void tracked_free(void* p) { if(p) InterlockedIncrement(&freed); free(p); }
#define free tracked_free
#include "../spu_job_context_cache.h"
#undef free
static DWORD WINAPI worker(void* arg) {
 (void)arg;
 void* a=job_context_take(0x40000),*b=job_context_take(0x40000);
 assert(a && b && a!=b);
 job_context_put(a);
 void* c=job_context_take(0x40000); assert(c==a);
 job_context_put(c); job_context_put(b);
 return 0;
}
int main(void) {
 HANDLE threads[4];
 for(unsigned i=0;i<4;++i) {threads[i]=CreateThread(NULL,0,worker,NULL,0,NULL);assert(threads[i]);}
 assert(WaitForMultipleObjects(4,threads,TRUE,10000)==WAIT_OBJECT_0);
 for(unsigned i=0;i<4;++i) CloseHandle(threads[i]);
 assert(freed==8);
 puts("job context cache: reuse, nested checkout, thread isolation and exit cleanup passed");
}
