#ifdef NDEBUG
#undef NDEBUG
#endif
#include <string.h>
#include <stddef.h>
static unsigned comparisons;
static int counted_memcmp(const void* a, const void* b, size_t n)
{
    ++comparisons;
    return memcmp(a, b, n);
}
#define memcmp counted_memcmp
#include "../spu_image_fingerprint_cache.h"
#undef memcmp
#include <assert.h>
#include <stdio.h>

static unsigned hashes;
static uint64_t fingerprint(const void* data, size_t size)
{
    const unsigned char* p = (const unsigned char*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
static uint64_t counted_hash(const void* data, size_t size)
{
    ++hashes;
    return fingerprint(data, size);
}

int main(void)
{
    spu_image_fingerprint_cache cache = {0}, other_thread = {0};
    unsigned char images[SPU_IMAGE_FINGERPRINT_SLOTS + 3][257] = {{0}};
    for (unsigned i = 0; i < sizeof images / sizeof images[0]; ++i)
        for (unsigned j = 0; j < sizeof images[0]; ++j)
            images[i][j] = (unsigned char)(i * 31 + j);
    const size_t size = sizeof images[0];
    uint64_t h = spu_image_fingerprint_cached(&cache, images[0], size, counted_hash);
    assert(h == fingerprint(images[0], size) && hashes == 1);
    for (unsigned i = 0; i < 1000; ++i)
        assert(spu_image_fingerprint_cached(&cache, images[0], size, counted_hash) == h);
    assert(hashes == 1);

    /* Fill the table, then repeatedly dispatch its newest image from a
     * different allocation. Exactly one complete comparison per hit,
     * with no recomputed hash. Changing a final byte must still miss. */
    spu_image_fingerprint_cache relocated = {0};
    unsigned char moved[257];
    for (unsigned i = 0; i < SPU_IMAGE_FINGERPRINT_SLOTS; ++i)
        assert(spu_image_fingerprint_cached(&relocated, images[i], size, fingerprint)
               == fingerprint(images[i], size));
    memcpy(moved, images[SPU_IMAGE_FINGERPRINT_SLOTS-1], size);
    unsigned cmp_before = comparisons;
    for (unsigned i = 0; i < 1000; ++i) {
        const void* p = i & 1 ? moved : images[SPU_IMAGE_FINGERPRINT_SLOTS-1];
        assert(spu_image_fingerprint_cached(&relocated, p, size, counted_hash)
               == fingerprint(p, size));
    }
    assert(comparisons - cmp_before == 1000 && hashes == 1);
    moved[size-1] ^= 1;
    assert(spu_image_fingerprint_cached(&relocated, moved, size, fingerprint)
           == fingerprint(moved, size));
    /* Changed address can match a non-recent cached image. */
    memcpy(moved, images[0], size);
    assert(spu_image_fingerprint_cached(&relocated, moved, size, fingerprint)
           == fingerprint(moved, size));
    for (unsigned i = 0; i < SPU_IMAGE_FINGERPRINT_SLOTS; ++i)
        free(relocated.entries[i].snapshot);

    /* Same address, modified final byte: sparse probes would miss this. */
    images[0][size - 1] ^= 1;
    assert(spu_image_fingerprint_cached(&cache, images[0], size, counted_hash)
           == fingerprint(images[0], size));
    assert(hashes == 2);
    assert(spu_image_fingerprint_cached(&cache, images[0], size - 1, counted_hash)
           == fingerprint(images[0], size - 1));
    assert(hashes == 3);
    for (unsigned i = 1; i < sizeof images / sizeof images[0]; ++i)
        assert(spu_image_fingerprint_cached(&cache, images[i], size, counted_hash)
               == fingerprint(images[i], size));
    assert(spu_image_fingerprint_cached(&cache, images[0], size, counted_hash)
           == fingerprint(images[0], size));
    const unsigned before = hashes;
    assert(spu_image_fingerprint_cached(&other_thread, images[0], size, counted_hash)
           == fingerprint(images[0], size));
    assert(hashes == before + 1);
    for (unsigned i = 0; i < SPU_IMAGE_FINGERPRINT_SLOTS; ++i) {
        free(cache.entries[i].snapshot);
        free(other_thread.entries[i].snapshot);
    }
    puts("image fingerprint cache: unchanged images, mutation, size, eviction, isolation PASS");
    return 0;
}
