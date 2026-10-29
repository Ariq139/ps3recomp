#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include "rsx_d3d12_srv_cache.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    ID3D12Debug* debug = NULL;
    const int debug_enabled = SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&debug));
    if (debug_enabled) {
        debug->lpVtbl->EnableDebugLayer(debug);
        debug->lpVtbl->Release(debug);
    }
    IDXGIFactory4* factory = NULL;
    IDXGIAdapter* adapter = NULL;
    ID3D12Device* device = NULL;
    assert(SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory)));
    assert(SUCCEEDED(factory->lpVtbl->EnumWarpAdapter(factory, &IID_IDXGIAdapter, (void**)&adapter)));
    assert(SUCCEEDED(D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_11_0,
        &IID_ID3D12Device, (void**)&device)));
    ID3D12InfoQueue* queue = NULL;
    (void)device->lpVtbl->QueryInterface(device, &IID_ID3D12InfoQueue, (void**)&queue);
    if (queue) queue->lpVtbl->ClearStoredMessages(queue);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {0};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 3;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ID3D12DescriptorHeap* heap = NULL;
    assert(SUCCEEDED(device->lpVtbl->CreateDescriptorHeap(
        device, &heap_desc, &IID_ID3D12DescriptorHeap, (void**)&heap)));
    D3D12_HEAP_PROPERTIES properties = {0};
    properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC resource_desc = {0};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resource_desc.Width = resource_desc.Height = 4;
    resource_desc.DepthOrArraySize = resource_desc.MipLevels = 1;
    resource_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    resource_desc.SampleDesc.Count = 1;
    ID3D12Resource* resource = NULL;
    assert(SUCCEEDED(device->lpVtbl->CreateCommittedResource(device, &properties,
        D3D12_HEAP_FLAG_NONE, &resource_desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        NULL, &IID_ID3D12Resource, (void**)&resource)));
    D3D12_SHADER_RESOURCE_VIEW_DESC view = {0};
    view.Format = resource_desc.Format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    const UINT increment = device->lpVtbl->GetDescriptorHandleIncrementSize(
        device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE destination;
    heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(heap, &destination);
    static rsx_d3d12_srv_cache cache;
    rsx_d3d12_srv_cache_write(&cache, device, increment, destination, resource, &view);
    destination.ptr += increment;
    rsx_d3d12_srv_cache_write(&cache, device, increment, destination, resource, &view);
    destination.ptr += increment;
    rsx_d3d12_srv_cache_write(&cache, device, increment, destination, NULL, &view);
    assert(cache.creates == 2 && cache.hits == 1 && cache.copies == 3);
    rsx_d3d12_srv_cache_shutdown(&cache);
    resource->lpVtbl->Release(resource);
    heap->lpVtbl->Release(heap);
    unsigned errors = 0;
    if (queue) {
        const UINT64 count = queue->lpVtbl->GetNumStoredMessagesAllowedByRetrievalFilter(queue);
        for (UINT64 i = 0; i < count; i++) {
            SIZE_T bytes = 0;
            queue->lpVtbl->GetMessage(queue, i, NULL, &bytes);
            D3D12_MESSAGE* message = (D3D12_MESSAGE*)malloc(bytes);
            assert(message && SUCCEEDED(queue->lpVtbl->GetMessage(queue, i, message, &bytes)));
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                fprintf(stderr, "%s\n", message->pDescription);
                errors++;
            }
            free(message);
        }
        queue->lpVtbl->Release(queue);
    }
    device->lpVtbl->Release(device);
    adapter->lpVtbl->Release(adapter);
    factory->lpVtbl->Release(factory);
    printf("WARP descriptor copies: %u errors, debug layer %s\n",
        errors, debug_enabled ? "enabled" : "unavailable");
    return errors ? 1 : 0;
}
