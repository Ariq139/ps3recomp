#ifndef PS3RECOMP_RSX_RAW_UPLOAD_CACHE_H
#define PS3RECOMP_RSX_RAW_UPLOAD_CACHE_H
#include "rsx_vertex_raw.h"
#include <stdlib.h>

#define RSX_RAW_UPLOAD_CACHE_SLOTS 16
#define RSX_RAW_UPLOAD_CACHE_MAX_BYTES (512u * 1024u)
typedef struct {
    rsx_raw_vertex_plan plan;
    u8* snapshot;
    u32 capacity, offset, index_count;
    int valid;
} rsx_raw_upload_entry;
typedef struct {
    rsx_raw_upload_entry entries[RSX_RAW_UPLOAD_CACHE_SLOTS];
} rsx_raw_upload_cache;

static void rsx_raw_upload_reset(rsx_raw_upload_cache* cache)
{
    for (u32 i = 0; i < RSX_RAW_UPLOAD_CACHE_SLOTS; ++i) cache->entries[i].valid = 0;
}
static void rsx_raw_upload_destroy(rsx_raw_upload_cache* cache)
{
    for (u32 i = 0; i < RSX_RAW_UPLOAD_CACHE_SLOTS; ++i) free(cache->entries[i].snapshot);
    memset(cache, 0, sizeof(*cache));
}
static int rsx_raw_upload_reuse(rsx_raw_upload_cache* cache,
    const rsx_raw_vertex_plan* plan, const u32* vertices, const u32* indices,
    u32 index_count, const u8* guest, u32 offset, void* destination, u32* reused_offset)
{
    const u32 bytes = plan->bytes + ((index_count * 4u + 15u) & ~15u);
    if (bytes > RSX_RAW_UPLOAD_CACHE_MAX_BYTES) {
        rsx_raw_vertex_write(plan, vertices, guest, destination);
        if (index_count) memcpy((u8*)destination + plan->bytes, indices, index_count * 4u);
        if (bytes > plan->bytes + index_count * 4u)
            memset((u8*)destination + plan->bytes + index_count * 4u, 0, bytes - plan->bytes - index_count * 4u);
        *reused_offset = offset;
        return 0;
    }
    u32 hash = plan->header[1] ^ index_count;
    for (u32 i = 0; i < plan->range_count; ++i)
        hash = (hash * 16777619u) ^ plan->ranges[i].source ^ plan->ranges[i].size;
    rsx_raw_upload_entry* entry = &cache->entries[hash % RSX_RAW_UPLOAD_CACHE_SLOTS];
    int match = entry->valid && entry->index_count == index_count &&
        !memcmp(&entry->plan, plan, sizeof(*plan)) &&
        !memcmp(entry->snapshot + RSX_RAW_VERTEX_HEADER_BYTES, vertices, plan->header[1] * 4u) &&
        (!index_count || !memcmp(entry->snapshot + plan->bytes, indices, index_count * 4u));
    for (u32 i = 0; match && i < plan->range_count; ++i) {
        const rsx_raw_vertex_range* r = &plan->ranges[i];
        match = !memcmp(entry->snapshot + r->destination, guest + r->source, r->size);
    }
    if (match) { *reused_offset = entry->offset; return 1; }
    entry->valid = 0;
    if (entry->capacity < bytes) {
        u8* snapshot = (u8*)realloc(entry->snapshot, bytes);
        if (!snapshot) {
            rsx_raw_vertex_write(plan, vertices, guest, destination);
            if (index_count) memcpy((u8*)destination + plan->bytes, indices, index_count * 4u);
            *reused_offset = offset;
            return 0;
        }
        entry->snapshot = snapshot; entry->capacity = bytes;
    }
    rsx_raw_vertex_write(plan, vertices, guest, entry->snapshot);
    if (index_count) memcpy(entry->snapshot + plan->bytes, indices, index_count * 4u);
    if (bytes > plan->bytes + index_count * 4u)
        memset(entry->snapshot + plan->bytes + index_count * 4u, 0, bytes - plan->bytes - index_count * 4u);
    memcpy(destination, entry->snapshot, bytes);
    entry->plan = *plan; entry->offset = offset;
    entry->index_count = index_count; entry->valid = 1;
    *reused_offset = offset;
    return 0;
}
#endif
