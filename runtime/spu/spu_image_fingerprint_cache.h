#ifndef PS3RECOMP_SPU_IMAGE_FINGERPRINT_CACHE_H
#define PS3RECOMP_SPU_IMAGE_FINGERPRINT_CACHE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Caller-owned cache: keep one per dispatcher thread. Comparing a retained
 * copy preserves full-image validation while avoiding FNV's serial dependency
 * chain for unchanged job binaries. A reused guest address is never sufficient. */
#define SPU_IMAGE_FINGERPRINT_SLOTS 16
typedef struct {
    const void* source;
    size_t size;
    unsigned char* snapshot;
    uint64_t fingerprint, last_use;
} spu_image_fingerprint_entry;
typedef struct {
    spu_image_fingerprint_entry entries[SPU_IMAGE_FINGERPRINT_SLOTS];
    uint64_t clock;
} spu_image_fingerprint_cache;

static inline void spu_image_fingerprint_cache_destroy(spu_image_fingerprint_cache* cache)
{
    if (!cache) return;
    for (unsigned i = 0; i < SPU_IMAGE_FINGERPRINT_SLOTS; ++i)
        free(cache->entries[i].snapshot);
    memset(cache, 0, sizeof(*cache));
}

static inline uint64_t spu_image_fingerprint_cached(
    spu_image_fingerprint_cache* cache, const void* data, size_t size,
    uint64_t (*hash)(const void*, size_t))
{
    if (!data || !size || size > 0x40000u) return hash(data, size);
    unsigned slot = 0;
    unsigned changed = SPU_IMAGE_FINGERPRINT_SLOTS;
    /* Prefer the original address fast path, but a fresh asynchronous job
     * snapshot need not have the previous job's host address. Address is an
     * accelerator only: every hit still validates ALL bytes. */
    for (unsigned i = 0; i < SPU_IMAGE_FINGERPRINT_SLOTS; ++i) {
        spu_image_fingerprint_entry* e = &cache->entries[i];
        if (e->last_use < cache->entries[slot].last_use) slot = i;
        if (e->snapshot && e->source == data && e->size == size) {
            if (!memcmp(e->snapshot, data, size)) {
                e->last_use = ++cache->clock;
                return e->fingerprint;
            }
            changed = i;
        }
    }
    /* Immutable snapshots often move between allocations. Reuse a known
     * fingerprint by content, never by an address or a sparse checksum alone.
     * A changed same-address entry must not prevent finding an equal image
     * in another slot. */
    for (unsigned i = 0; i < SPU_IMAGE_FINGERPRINT_SLOTS; ++i) {
        spu_image_fingerprint_entry* e = &cache->entries[i];
        if (i != changed && e->snapshot && e->size == size &&
            !memcmp(e->snapshot, data, size)) {
            e->source = data;
            e->last_use = ++cache->clock;
            return e->fingerprint;
        }
    }
    if (changed != SPU_IMAGE_FINGERPRINT_SLOTS) slot = changed;
    const uint64_t fingerprint = hash(data, size);
    spu_image_fingerprint_entry* e = &cache->entries[slot];
    if (e->size != size || !e->snapshot) {
        unsigned char* snapshot = (unsigned char*)malloc(size);
        if (!snapshot) return fingerprint;
        free(e->snapshot);
        e->snapshot = snapshot;
    }
    memcpy(e->snapshot, data, size);
    e->source = data;
    e->size = size;
    e->fingerprint = fingerprint;
    e->last_use = ++cache->clock;
    return fingerprint;
}
#endif
