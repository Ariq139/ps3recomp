#ifndef PS3RECOMP_SPU_IMAGE_FINGERPRINT_TLS_H
#define PS3RECOMP_SPU_IMAGE_FINGERPRINT_TLS_H

#include "spu_image_fingerprint_cache.h"

/* The general dispatcher may be entered on any host thread, not just a JQ
 * worker with an explicit cleanup hook. Use a destructor-bearing native key
 * so all owned snapshots are released when that execution owner exits.
 * Keep the key for process lifetime, like the job-context cache. */
#ifdef _WIN32
#include <windows.h>
static DWORD s_fingerprint_key = FLS_OUT_OF_INDEXES;
static INIT_ONCE s_fingerprint_once = INIT_ONCE_STATIC_INIT;
static VOID CALLBACK fingerprint_tls_destroy(void* p)
{
    spu_image_fingerprint_cache_destroy((spu_image_fingerprint_cache*)p);
    free(p);
}
static BOOL CALLBACK fingerprint_tls_init(PINIT_ONCE once, PVOID arg, PVOID* context)
{
    (void)once; (void)arg; (void)context;
    s_fingerprint_key = FlsAlloc(fingerprint_tls_destroy);
    return TRUE;
}
static inline spu_image_fingerprint_cache* spu_image_fingerprint_thread_cache(void)
{
    if (!InitOnceExecuteOnce(&s_fingerprint_once, fingerprint_tls_init, NULL, NULL) ||
        s_fingerprint_key == FLS_OUT_OF_INDEXES) return NULL;
    spu_image_fingerprint_cache* cache =
        (spu_image_fingerprint_cache*)FlsGetValue(s_fingerprint_key);
    if (cache) return cache;
    cache = (spu_image_fingerprint_cache*)calloc(1, sizeof(*cache));
    if (cache && !FlsSetValue(s_fingerprint_key, cache)) {
        free(cache);
        return NULL;
    }
    return cache;
}
#else
#include <pthread.h>
static pthread_key_t s_fingerprint_key;
static pthread_once_t s_fingerprint_once = PTHREAD_ONCE_INIT;
static int s_fingerprint_key_valid;
/* The pthread key owns the allocation; the compiler TLS pointer only borrows
 * it. Hot lookups avoid pthread_once/pthread_getspecific. FLS remains the
 * lookup path on Windows because a host thread may switch fibers. */
#ifdef __cplusplus
static thread_local spu_image_fingerprint_cache* s_fingerprint_fast;
#else
static _Thread_local spu_image_fingerprint_cache* s_fingerprint_fast;
#endif
static void fingerprint_tls_destroy(void* p)
{
    s_fingerprint_fast = NULL;
    spu_image_fingerprint_cache_destroy((spu_image_fingerprint_cache*)p);
    free(p);
}
static void fingerprint_tls_init(void)
{
    s_fingerprint_key_valid = pthread_key_create(&s_fingerprint_key, fingerprint_tls_destroy) == 0;
}
static inline spu_image_fingerprint_cache* spu_image_fingerprint_thread_cache(void)
{
    if (s_fingerprint_fast) return s_fingerprint_fast;
    if (pthread_once(&s_fingerprint_once, fingerprint_tls_init) ||
        !s_fingerprint_key_valid) return NULL;
    spu_image_fingerprint_cache* cache =
        (spu_image_fingerprint_cache*)pthread_getspecific(s_fingerprint_key);
    if (cache) return s_fingerprint_fast = cache;
    cache = (spu_image_fingerprint_cache*)calloc(1, sizeof(*cache));
    if (cache && pthread_setspecific(s_fingerprint_key, cache)) {
        free(cache);
        return NULL;
    }
    return s_fingerprint_fast = cache;
}
#endif
#endif
