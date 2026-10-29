#ifndef PS3RECOMP_RSX_VERTEX_DECODE_CACHE_H
#define PS3RECOMP_RSX_VERTEX_DECODE_CACHE_H

#include <stdint.h>

#define RSX_VERTEX_DECODE_CACHE_SLOTS 256u

typedef struct {
    uint64_t epoch;
    uint32_t vertex;
    float values[64];
} rsx_vertex_decode_cache_entry;

typedef struct {
    rsx_vertex_decode_cache_entry entries[RSX_VERTEX_DECODE_CACHE_SLOTS];
} rsx_vertex_decode_cache;

static inline float* rsx_vertex_decode_cache_find(
    rsx_vertex_decode_cache* cache, uint64_t epoch, uint32_t vertex)
{
    rsx_vertex_decode_cache_entry* entry =
        &cache->entries[vertex & (RSX_VERTEX_DECODE_CACHE_SLOTS - 1u)];
    return epoch && entry->epoch == epoch && entry->vertex == vertex
        ? entry->values : 0;
}

static inline float* rsx_vertex_decode_cache_store(
    rsx_vertex_decode_cache* cache, uint64_t epoch, uint32_t vertex)
{
    rsx_vertex_decode_cache_entry* entry =
        &cache->entries[vertex & (RSX_VERTEX_DECODE_CACHE_SLOTS - 1u)];
    entry->epoch = epoch;
    entry->vertex = vertex;
    return entry->values;
}

#endif
