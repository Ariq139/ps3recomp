#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif
static atomic_uint frees;
static void counted_free(void* p)
{
    if (p) atomic_fetch_add(&frees, 1);
    free(p);
}
#define free counted_free
#include "../spu_image_fingerprint_tls.h"
#undef free

static uint64_t hash_image(const void* data, size_t n)
{
    const unsigned char* b = data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
static void exercise(void)
{
    unsigned char a[257] = {1}, b[257] = {2};
    spu_image_fingerprint_cache* c = spu_image_fingerprint_thread_cache();
    assert(c && c->clock == 0);
    assert(spu_image_fingerprint_cached(c, a, sizeof(a), hash_image) == hash_image(a, sizeof(a)));
    assert(spu_image_fingerprint_cached(c, b, sizeof(b), hash_image) == hash_image(b, sizeof(b)));
    assert(spu_image_fingerprint_thread_cache() == c);
    assert(c->clock == 2);
    assert(c->entries[0].snapshot && c->entries[1].snapshot);
    /* Simulate multiple tasks on one long-lived pool worker. A hit still
     * compares the entire image, including changes at an identical address. */
    for (unsigned i = 0; i < 64; ++i) {
        assert(spu_image_fingerprint_thread_cache() == c);
        assert(spu_image_fingerprint_cached(c, a, sizeof(a), hash_image) == hash_image(a, sizeof(a)));
        assert(spu_image_fingerprint_cached(c, b, sizeof(b), hash_image) == hash_image(b, sizeof(b)));
    }
    a[256] ^= 1;
    assert(spu_image_fingerprint_cached(c, a, sizeof(a), hash_image) == hash_image(a, sizeof(a)));
    assert(spu_image_fingerprint_thread_cache() == c);
}
#ifdef _WIN32
static unsigned __stdcall worker(void* p) { (void)p; exercise(); return 0; }
#else
static void* worker(void* p) { (void)p; exercise(); return NULL; }
#endif
int main(void)
{
    enum { N = 32, ROUNDS = 4 };
    for (unsigned round = 0; round < ROUNDS; ++round) {
#ifdef _WIN32
        HANDLE threads[N];
        for (unsigned i = 0; i < N; ++i) {
            threads[i] = (HANDLE)_beginthreadex(NULL, 0, worker, NULL, 0, NULL);
            assert(threads[i]);
        }
        assert(WaitForMultipleObjects(N, threads, TRUE, INFINITE) == WAIT_OBJECT_0);
        for (unsigned i = 0; i < N; ++i) CloseHandle(threads[i]);
#else
        pthread_t threads[N];
        for (unsigned i = 0; i < N; ++i) assert(!pthread_create(&threads[i], NULL, worker, NULL));
        for (unsigned i = 0; i < N; ++i) assert(!pthread_join(threads[i], NULL));
#endif
        /* Two snapshots and one cache allocation per finished worker. */
        assert(atomic_load(&frees) == 3u * N * (round + 1));
    }
    puts("PASS fingerprint snapshots released across 128 worker exits");
    return 0;
}
