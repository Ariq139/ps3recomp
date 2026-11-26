#ifndef PS3RECOMP_RSX_D3D12_SRV_CACHE_H
#define PS3RECOMP_RSX_D3D12_SRV_CACHE_H

#include <stdint.h>
#include <string.h>
#include <d3d12.h>

#define D3D12_SRV_CACHE_SLOTS 2048u
#define D3D12_SRV_BATCH_MAX 20u
#define D3D12_SRV_HEAP_SLOTS (D3D12_SRV_CACHE_SLOTS + D3D12_SRV_BATCH_MAX)
typedef struct {
    ID3D12Resource* resource;
    D3D12_SHADER_RESOURCE_VIEW_DESC desc;
} rsx_d3d12_srv_request;
typedef struct {
    ID3D12Resource* resource;
    D3D12_SHADER_RESOURCE_VIEW_DESC desc;
    int valid;
} rsx_d3d12_cached_srv;

typedef struct {
    ID3D12DescriptorHeap* heap;
    rsx_d3d12_cached_srv entries[D3D12_SRV_CACHE_SLOTS];
    int attempted;
    uint64_t creates, copies, hits;
} rsx_d3d12_srv_cache;

static inline void rsx_d3d12_srv_cache_reset(rsx_d3d12_srv_cache* cache)
{
    for (unsigned i = 0; i < D3D12_SRV_CACHE_SLOTS; i++) {
        rsx_d3d12_cached_srv* entry = &cache->entries[i];
        if (entry->valid && entry->resource)
            entry->resource->lpVtbl->Release(entry->resource);
        entry->valid = 0;
        entry->resource = NULL;
    }
}

static inline void rsx_d3d12_srv_cache_shutdown(rsx_d3d12_srv_cache* cache)
{
    rsx_d3d12_srv_cache_reset(cache);
    if (cache->heap) cache->heap->lpVtbl->Release(cache->heap);
    memset(cache, 0, sizeof(*cache));
}

static inline int rsx_d3d12_srv_cache_prepare(rsx_d3d12_srv_cache* cache,
    ID3D12Device* device)
{
    if (!cache->attempted) {
        cache->attempted = 1;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {0};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = D3D12_SRV_HEAP_SLOTS;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        (void)device->lpVtbl->CreateDescriptorHeap(
            device, &hd, &IID_ID3D12DescriptorHeap, (void**)&cache->heap);
    }
    return cache->heap != NULL;
}

static inline unsigned rsx_d3d12_srv_index(ID3D12Resource* resource,
    const D3D12_SHADER_RESOURCE_VIEW_DESC* desc)
{
    const uint64_t pointer = (uint64_t)(uintptr_t)resource;
    return (unsigned)((pointer >> 4) ^ (pointer >> 32) ^
        ((uint64_t)desc->Format * 2654435761u) ^
        ((uint64_t)desc->Shader4ComponentMapping * 2246822519u) ^
        ((uint64_t)desc->ViewDimension * 3266489917u)) & (D3D12_SRV_CACHE_SLOTS - 1u);
}

static inline int rsx_d3d12_srv_match(const rsx_d3d12_cached_srv* entry,
    ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc)
{
    return entry->valid && entry->resource == resource &&
        memcmp(&entry->desc, desc, sizeof(*desc)) == 0;
}

static inline void rsx_d3d12_srv_cache_fill(rsx_d3d12_srv_cache* cache,
    ID3D12Device* device, unsigned index, D3D12_CPU_DESCRIPTOR_HANDLE source,
    ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc)
{
    rsx_d3d12_cached_srv* entry = &cache->entries[index];
    if (!rsx_d3d12_srv_match(entry, resource, desc)) {
        /* Retain the identity until reset so a released COM pointer cannot
         * be mistaken for a different resource allocated at the same address. */
        if (resource) resource->lpVtbl->AddRef(resource);
        if (entry->valid && entry->resource)
            entry->resource->lpVtbl->Release(entry->resource);
        device->lpVtbl->CreateShaderResourceView(device, resource, desc, source);
        entry->resource = resource;
        entry->desc = *desc;
        entry->valid = 1;
        cache->creates++;
    } else {
        cache->hits++;
    }
}
static inline void rsx_d3d12_srv_cache_write(rsx_d3d12_srv_cache* cache,
    ID3D12Device* device, UINT increment, D3D12_CPU_DESCRIPTOR_HANDLE destination,
    ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc)
{
    if (!rsx_d3d12_srv_cache_prepare(cache, device)) {
        device->lpVtbl->CreateShaderResourceView(device, resource, desc, destination);
        cache->creates++;
        return;
    }
    const unsigned index = rsx_d3d12_srv_index(resource, desc);
    D3D12_CPU_DESCRIPTOR_HANDLE source;
    cache->heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(cache->heap, &source);
    source.ptr += (uint64_t)index * increment;
    rsx_d3d12_srv_cache_fill(cache, device, index, source, resource, desc);
    /* D3D12 descriptor-copy sources must be non-shader-visible. */
    device->lpVtbl->CopyDescriptorsSimple(device, 1, destination, source,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    cache->copies++;
}

/* Resolve all sources before one CopyDescriptors call. A direct-mapped
 * collision must not overwrite a source already selected in this batch.
 * Spill to its dedicated CPU-only slot instead. The caller retains every
 * request's resource through the call; normal GPU retirement is unchanged. */
static inline void rsx_d3d12_srv_cache_batch(rsx_d3d12_srv_cache* cache,
    ID3D12Device* device, UINT increment, D3D12_CPU_DESCRIPTOR_HANDLE destination,
    const rsx_d3d12_srv_request* requests, UINT count)
{
    if (!count) return;
    if (count > D3D12_SRV_BATCH_MAX || !rsx_d3d12_srv_cache_prepare(cache, device)) {
        for (UINT i = 0; i < count; ++i) {
            rsx_d3d12_srv_cache_write(cache, device, increment, destination,
                requests[i].resource, &requests[i].desc);
            destination.ptr += increment;
        }
        return;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE base, sources[D3D12_SRV_BATCH_MAX];
    unsigned buckets[D3D12_SRV_BATCH_MAX];
    cache->heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(cache->heap, &base);
    for (UINT i = 0; i < count; ++i) {
        const rsx_d3d12_srv_request* req = &requests[i];
        unsigned index = rsx_d3d12_srv_index(req->resource, &req->desc);
        int pinned = 0;
        for (UINT j = 0; j < i; ++j) if (buckets[j] == index) { pinned = 1; break; }
        if (pinned && !rsx_d3d12_srv_match(&cache->entries[index], req->resource, &req->desc)) {
            index = D3D12_SRV_CACHE_SLOTS + i;
            sources[i].ptr = base.ptr + (uint64_t)index * increment;
            device->lpVtbl->CreateShaderResourceView(device, req->resource, &req->desc, sources[i]);
            cache->creates++;
        } else {
            sources[i].ptr = base.ptr + (uint64_t)index * increment;
            rsx_d3d12_srv_cache_fill(cache, device, index, sources[i], req->resource, &req->desc);
        }
        buckets[i] = index;
    }
    device->lpVtbl->CopyDescriptors(device, 1, &destination, &count,
        count, sources, NULL, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    cache->copies++; /* API calls, not descriptor count */
}
#endif
