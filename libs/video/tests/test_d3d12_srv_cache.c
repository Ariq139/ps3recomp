#include <windows.h>
#include <initguid.h>
#include "../rsx_d3d12_srv_cache.h"
#include <assert.h>
#include <stdio.h>

static rsx_d3d12_srv_cache test_cache;
static ID3D12Device test_device;

static ID3D12DescriptorHeap test_cpu_heap, test_gpu_heap;
static ID3D12Resource test_resource;
static ULONG test_refs;
static unsigned test_creates, test_copies;
static D3D12_SHADER_RESOURCE_VIEW_DESC test_cpu_views[D3D12_SRV_HEAP_SLOTS];
static D3D12_SHADER_RESOURCE_VIEW_DESC test_gpu_views[8];

static D3D12_CPU_DESCRIPTOR_HANDLE* STDMETHODCALLTYPE test_heap_start(
    ID3D12DescriptorHeap* heap, D3D12_CPU_DESCRIPTOR_HANDLE* handle)
{
    handle->ptr = heap == &test_cpu_heap ? 0x10000u : 0x100000u;
    return handle;
}

static HRESULT STDMETHODCALLTYPE test_create_heap(ID3D12Device* device,
    const D3D12_DESCRIPTOR_HEAP_DESC* desc, REFIID iid, void** heap)
{
    (void)device; (void)iid;
    assert(desc->Flags == D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    assert(desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    assert(desc->NumDescriptors == D3D12_SRV_HEAP_SLOTS);
    *heap = &test_cpu_heap;
    return S_OK;
}

static void STDMETHODCALLTYPE test_create_view(ID3D12Device* device, ID3D12Resource* resource,
    const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
    (void)device; (void)resource;
    test_creates++;
    if (destination.ptr < 0x100000u) {
        const SIZE_T index = (destination.ptr - 0x10000u) / 32u;
        assert(index < D3D12_SRV_HEAP_SLOTS);
        test_cpu_views[index] = *desc;
    } else {
        const SIZE_T index = (destination.ptr - 0x100000u) / 32u;
        assert(index < 8);
        test_gpu_views[index] = *desc;
    }
}

static void STDMETHODCALLTYPE test_copy_view(ID3D12Device* device, UINT count,
    D3D12_CPU_DESCRIPTOR_HANDLE destination, D3D12_CPU_DESCRIPTOR_HANDLE source,
    D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    (void)device;
    assert(count == 1 && type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    assert(source.ptr >= 0x10000u && source.ptr < 0x100000u);
    const SIZE_T src = (source.ptr - 0x10000u) / 32u;
    const SIZE_T dst = (destination.ptr - 0x100000u) / 32u;
    assert(src < D3D12_SRV_HEAP_SLOTS && dst < 8);
    test_gpu_views[dst] = test_cpu_views[src];
    test_copies++;
}

static void STDMETHODCALLTYPE test_copy_batch(ID3D12Device* device, UINT nd,
    const D3D12_CPU_DESCRIPTOR_HANDLE* dst, const UINT* dst_sizes, UINT ns,
    const D3D12_CPU_DESCRIPTOR_HANDLE* src, const UINT* src_sizes,
    D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    assert(nd == 1 && dst_sizes[0] == ns && !src_sizes);
    for (UINT i = 0; i < ns; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE out = {dst[0].ptr + i * 32u};
        test_copy_view(device, 1, out, src[i], type);
    }
}

static ULONG STDMETHODCALLTYPE test_addref(ID3D12Resource* resource)
{ assert(resource == &test_resource); return ++test_refs; }
static ULONG STDMETHODCALLTYPE test_release(ID3D12Resource* resource)
{ assert(resource == &test_resource && test_refs); return --test_refs; }

static void test_write(unsigned slot, ID3D12Resource* resource,
                       DXGI_FORMAT format, UINT mapping, int cube)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc = {0};
    desc.Format = format;
    desc.Shader4ComponentMapping = mapping;
    desc.ViewDimension = cube ? D3D12_SRV_DIMENSION_TEXTURECUBE : D3D12_SRV_DIMENSION_TEXTURE2D;
    if (cube) desc.TextureCube.MipLevels = 1;
    else desc.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE destination = {0x100000u + slot * 32u};
    rsx_d3d12_srv_cache_write(&test_cache, &test_device, 32, destination, resource, &desc);
}

int main(void)
{
    static ID3D12DescriptorHeapVtbl heap_vtable = {
        .GetCPUDescriptorHandleForHeapStart = test_heap_start
    };
    static ID3D12DeviceVtbl device_vtable = {
        .CreateDescriptorHeap = test_create_heap,
        .CreateShaderResourceView = test_create_view,
        .CopyDescriptorsSimple = test_copy_view,
        .CopyDescriptors = test_copy_batch
    };
    static ID3D12ResourceVtbl resource_vtable = {
        .AddRef = test_addref, .Release = test_release
    };
    test_device.lpVtbl = &device_vtable;
    test_cpu_heap.lpVtbl = test_gpu_heap.lpVtbl = &heap_vtable;
    test_resource.lpVtbl = &resource_vtable;
    const UINT mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    test_write(0, &test_resource, DXGI_FORMAT_R8_UNORM, mapping, 0);
    test_write(1, &test_resource, DXGI_FORMAT_R8_UNORM, mapping, 0);
    assert(test_creates == 1 && test_copies == 2 && test_refs == 1);
    assert(memcmp(&test_gpu_views[0], &test_gpu_views[1], sizeof(test_gpu_views[0])) == 0);
    test_write(2, &test_resource, DXGI_FORMAT_R8_UNORM, mapping, 1);
    assert(test_creates == 2 && test_gpu_views[2].ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE);
    assert(test_gpu_views[0].ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D);
    test_write(3, NULL, DXGI_FORMAT_R8_UNORM, mapping, 0);
    test_write(4, NULL, DXGI_FORMAT_R8_UNORM, mapping, 0);
    assert(test_creates == 3 && test_copies == 5);
    rsx_d3d12_srv_cache_reset(&test_cache);
    assert(test_refs == 0);
    test_write(5, &test_resource, DXGI_FORMAT_R8_UNORM, mapping, 0);
    assert(test_creates == 4 && test_refs == 1);
    rsx_d3d12_srv_cache_reset(&test_cache);
    assert(test_refs == 0);
    test_cache.heap = NULL;
    test_write(6, &test_resource, DXGI_FORMAT_R8_UNORM, mapping, 0);
    assert(test_creates == 5 && test_copies == 6 && test_refs == 0);
    test_cache.heap = &test_cpu_heap;
    test_write(0, &test_resource, DXGI_FORMAT_R8_UNORM, mapping, 0);
    D3D12_SHADER_RESOURCE_VIEW_DESC changed = test_gpu_views[0];
    changed.Texture2D.MipLevels = 2;
    D3D12_CPU_DESCRIPTOR_HANDLE destination = {0x100000u + 32u};
    rsx_d3d12_srv_cache_write(&test_cache, &test_device, 32, destination,
        &test_resource, &changed);
    assert(test_refs == 1 && test_creates == 7);
    assert(test_gpu_views[0].Texture2D.MipLevels == 1 &&
           test_gpu_views[1].Texture2D.MipLevels == 2);
    rsx_d3d12_srv_cache_reset(&test_cache);
    assert(test_refs == 0);
    /* Same bucket, different view (MipLevels is not part of the bucket).
     * The batch must spill instead of replacing the first source. */
    rsx_d3d12_srv_request requests[3] = {0};
    requests[0].resource = requests[1].resource = &test_resource;
    requests[0].desc = requests[1].desc = test_gpu_views[0];
    requests[1].desc.Texture2D.MipLevels = 2;
    requests[2] = requests[0];
    const uint64_t calls = test_cache.copies;
    D3D12_CPU_DESCRIPTOR_HANDLE batch_dst = {0x100000u};
    rsx_d3d12_srv_cache_batch(&test_cache, &test_device, 32, batch_dst, requests, 3);
    assert(test_cache.copies == calls + 1);
    assert(test_gpu_views[0].Texture2D.MipLevels == 1 &&
           test_gpu_views[1].Texture2D.MipLevels == 2 &&
           test_gpu_views[2].Texture2D.MipLevels == 1);
    rsx_d3d12_srv_cache_reset(&test_cache);
    assert(test_refs == 0);
    puts("D3D12 SRV cache: reuse, dimensions, nulls, reset, references, collisions and batching passed");
    return 0;
}
