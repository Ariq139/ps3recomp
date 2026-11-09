#ifndef PS3RECOMP_SPU_JOB_CONTEXT_CACHE_H
#define PS3RECOMP_SPU_JOB_CONTEXT_CACHE_H
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
static DWORD s_job_context_key = FLS_OUT_OF_INDEXES;
static INIT_ONCE s_job_context_once = INIT_ONCE_STATIC_INIT;
static VOID CALLBACK job_context_destroy(void* p) { free(p); }
static BOOL CALLBACK job_context_key_init(PINIT_ONCE once, PVOID arg, PVOID* context)
{
 (void)once; (void)arg; (void)context;
 s_job_context_key = FlsAlloc(job_context_destroy);
 return TRUE;
}
static int job_context_key_ready(void)
{
 InitOnceExecuteOnce(&s_job_context_once, job_context_key_init, NULL, NULL);
 return s_job_context_key != FLS_OUT_OF_INDEXES;
}
static void* job_context_take(size_t bytes)
{
 if (job_context_key_ready()) {
  void* p = FlsGetValue(s_job_context_key);
  if (p && FlsSetValue(s_job_context_key, NULL)) return p;
 }
 return malloc(bytes);
}
static void job_context_put(void* p)
{
 if (job_context_key_ready() && !FlsGetValue(s_job_context_key) && FlsSetValue(s_job_context_key,p)) return;
 free(p);
}
#else
#include <pthread.h>
static pthread_key_t s_job_context_key;
static pthread_once_t s_job_context_once = PTHREAD_ONCE_INIT;
static int s_job_context_key_valid;
static void job_context_key_init(void) { s_job_context_key_valid = pthread_key_create(&s_job_context_key, free) == 0; }
static int job_context_key_ready(void) { pthread_once(&s_job_context_once, job_context_key_init); return s_job_context_key_valid; }
static void* job_context_take(size_t bytes)
{
 if (job_context_key_ready()) {
  void* p = pthread_getspecific(s_job_context_key);
  if (p && !pthread_setspecific(s_job_context_key,NULL)) return p;
 }
 return malloc(bytes);
}
static void job_context_put(void* p)
{
 if (job_context_key_ready() && !pthread_getspecific(s_job_context_key) && !pthread_setspecific(s_job_context_key,p)) return;
 free(p);
}
#endif
#endif
