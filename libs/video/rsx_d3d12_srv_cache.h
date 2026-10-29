#ifndef PS3RECOMP_RSX_D3D12_SRV_CACHE_H
#define PS3RECOMP_RSX_D3D12_SRV_CACHE_H

#include <stdint.h>
#include <string.h>
#include <d3d12.h>

#define D3D12_SRV_CACHE_SLOTS 2048u
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

static inline void rsx_d3d12_srv_cache_write(rsx_d3d12_srv_cache* cache,
    ID3D12Device* device, UINT increment, D3D12_CPU_DESCRIPTOR_HANDLE destination,
    ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc)
{
    if (!cache->attempted) {
        cache->attempted = 1;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {0};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = D3D12_SRV_CACHE_SLOTS;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        (void)device->lpVtbl->CreateDescriptorHeap(
            device, &hd, &IID_ID3D12DescriptorHeap, (void**)&cache->heap);
    }
    if (!cache->heap) {
        device->lpVtbl->CreateShaderResourceView(device, resource, desc, destination);
        cache->creates++;
        return;
    }
    const uint64_t pointer = (uint64_t)(uintptr_t)resource;
    const unsigned index = (unsigned)((pointer >> 4) ^ (pointer >> 32) ^
        ((uint64_t)desc->Format * 2654435761u) ^
        ((uint64_t)desc->Shader4ComponentMapping * 2246822519u) ^
        ((uint64_t)desc->ViewDimension * 3266489917u)) & (D3D12_SRV_CACHE_SLOTS - 1u);
    rsx_d3d12_cached_srv* entry = &cache->entries[index];
    D3D12_CPU_DESCRIPTOR_HANDLE source;
    cache->heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(cache->heap, &source);
    source.ptr += (uint64_t)index * increment;
    if (!entry->valid || entry->resource != resource ||
        memcmp(&entry->desc, desc, sizeof(*desc)) != 0) {
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
    /* D3D12 descriptor-copy sources must be non-shader-visible. */
    device->lpVtbl->CopyDescriptorsSimple(device, 1, destination, source,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    cache->copies++;
}

#endif
