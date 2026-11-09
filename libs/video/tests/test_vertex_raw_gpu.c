#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "rsx_vertex_raw.h"
#include "rsx_vertex_fetch.h"
#include "rsx_vp_decompiler.h"

u8* vm_base;
u32 cellGcmResolveLocated(int local, u32 offset) { (void)local; return offset; }
u32 cellGcmResolveOffset(u32 offset) { return offset; }
static u32 fragmented_resolve(int local, u32 offset)
{
    (void)local;
    return offset >= 0x100000 ? offset + 0x100000 : offset;
}

static void be16(u8* p, u32 value) { p[0] = (u8)(value >> 8); p[1] = (u8)value; }
static void be32(u8* p, u32 value)
{
    p[0] = (u8)(value >> 24); p[1] = (u8)(value >> 16);
    p[2] = (u8)(value >> 8); p[3] = (u8)value;
}

static ID3DBlob* compile(const char* source, const char* target)
{
    ID3DBlob* code = NULL;
    ID3DBlob* errors = NULL;
    HRESULT hr = D3DCompile(source, strlen(source), "raw_vertex_test", NULL, NULL,
        "main", target, D3DCOMPILE_WARNINGS_ARE_ERRORS, 0, &code, &errors);
    if (FAILED(hr) && errors)
        fprintf(stderr, "%s\n", (char*)errors->lpVtbl->GetBufferPointer(errors));
    if (errors) errors->lpVtbl->Release(errors);
    assert(SUCCEEDED(hr) && code);
    return code;
}

static ID3D12Resource* buffer(ID3D12Device* device, u64 bytes,
    D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_HEAP_PROPERTIES properties = {0}; properties.Type = heap;
    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1;
    desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
    ID3D12Resource* result = NULL;
    assert(SUCCEEDED(device->lpVtbl->CreateCommittedResource(device, &properties,
        D3D12_HEAP_FLAG_NONE, &desc, state, NULL, &IID_ID3D12Resource, (void**)&result)));
    return result;
}

static int validate_corpus(const char* directory)
{
    char pattern[1024]; snprintf(pattern, sizeof(pattern), "%s/*.vp", directory);
    WIN32_FIND_DATAA data;
    HANDLE search = FindFirstFileA(pattern, &data);
    assert(search != INVALID_HANDLE_VALUE);
    static u8 ucode[65536];
    static char source[256 * 1024], wrapped[280 * 1024];
    u32 total = 0;
    do {
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", directory, data.cFileName);
        FILE* file = fopen(path, "rb"); assert(file);
        const u32 bytes = (u32)fread(ucode, 1, sizeof(ucode), file); fclose(file);
        rsx_vp_input_analysis analysis = {0xFFFF,0};
        assert(rsx_vp_analyze_inputs(ucode, bytes, &analysis) >= 0);
        assert(rsx_vp_decompile_compact_textures(ucode, bytes, 0, 0, 0,
            analysis.input_mask, source, sizeof(source)) > 0);
        assert(rsx_raw_vertex_wrap_hlsl(source, analysis.input_mask, wrapped, sizeof(wrapped)));
        ID3DBlob* blob = compile(wrapped, "vs_5_0");
        blob->lpVtbl->Release(blob); ++total;
    } while (FindNextFileA(search, &data));
    FindClose(search);
    printf("RPCS3 vertex corpus: %u raw-fetch shaders compiled\n", total);
    return total ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc == 3 && !strcmp(argv[1], "--corpus")) return validate_corpus(argv[2]);
    const u32 strip[] = {0,1,2,1,3,2,2,3,4};
    const u32 fan[] = {0,1,2,0,2,3,0,3,4};
    const u32 quad[] = {0,1,2,0,2,3,4,5,6,4,6,7};
    const u32 flat_strip[] = {2,0,1,2,1,3,4,2,3};
    assert(!rsx_raw_vertex_count(RSX_PRIMITIVE_TRIANGLES, 256));
    assert(rsx_raw_vertex_count(RSX_PRIMITIVE_QUADS, 8) == 12);
    assert(rsx_raw_vertex_count(RSX_PRIMITIVE_TRIANGLE_STRIP, 5) == 9);
    assert(rsx_raw_vertex_count(RSX_PRIMITIVE_TRIANGLE_FAN, 5) == 9);
    for (u32 i = 0; i < 9; ++i) {
        assert(rsx_raw_vertex_source(RSX_PRIMITIVE_TRIANGLE_STRIP, i, 0) == strip[i]);
        assert(rsx_raw_vertex_source(RSX_PRIMITIVE_TRIANGLE_FAN, i, 0) == fan[i]);
        assert(rsx_raw_vertex_source(RSX_PRIMITIVE_TRIANGLE_STRIP, i, 1) == flat_strip[i]);
    }
    for (u32 i = 0; i < 12; ++i)
        assert(rsx_raw_vertex_source(RSX_PRIMITIVE_QUADS, i, 0) == quad[i]);
    const int indexed_test = argc == 2 && (!strcmp(argv[1], "--indexed") ||
        !strcmp(argv[1], "--hardware-indexed"));
    vm_base = (u8*)calloc(1, 65536);
    assert(vm_base);
    rsx_state state = {0};
    const u32 types[] = {2,3,4,1,5,6,7,0,2,2,2,2,1,3,4,0};
    const u32 sizes[] = {3,2,4,3,4,1,3,0,1,1,4,1,1,4,1,0};
    state.vertex_data_base_offset = 64;
    for (u32 a = 0; a < 16; ++a) {
        state.vertex_data4f[a][0] = (float)a;
        state.vertex_data4f[a][3] = 1;
        if (!types[a]) continue;
        rsx_vertex_attrib* d = &state.vertex_attribs[a];
        d->type = types[a]; d->size = sizes[a]; d->enabled = 1;
        d->stride = 16; d->offset = 1025 + a * 512;
        d->frequency = a == 8 ? 2 : a == 9 ? 3 : 1;
    }
    state.frequency_divider_op = (1u << 9) | (1u << 14);
    state.vertex_attribs[0].offset |= 0x10000000u;
    state.vertex_attribs[1].offset |= 0x80000000u;
    state.vertex_attribs[10].stride = 0;
    state.vertex_attribs[11].offset = state.vertex_attribs[0].offset + 12;
    state.vertex_data4f[7][0] = -4.5f;
    state.vertex_data4f[7][1] = 12.0f;
    for (u32 a = 0; a < 16; ++a) {
        const rsx_vertex_attrib* d = &state.vertex_attribs[a];
        if (!d->enabled) continue;
        for (u32 i = 0; i < 8; ++i) {
            u8* p = vm_base + ((state.vertex_data_base_offset +
                (d->offset & 0x7FFFFFFFu)) & 0x0FFFFFFFu) + i * d->stride;
            for (u32 k = 0; k < d->size; ++k) {
                if (d->type == 2) {
                    float f = (float)((int)i - 3) * (k + 1) + 0.25f;
                    u32 bits; memcpy(&bits, &f, 4); be32(p + k * 4, bits);
                } else if (d->type == 3) {
                    const u32 halves[] = {0x0001,0x8000,0x3C00,0xBC00,0x7C00,0xFC00,0x7E01,0x7BFF};
                    be16(p + k * 2, halves[(i + k) & 7]);
                } else if (d->type == 1 || d->type == 5) {
                    const u32 words[] = {0x8000,0xFFFF,0,0x7FFF};
                    be16(p + k * 2, words[(i + k) & 3]);
                } else if (d->type == 6) {
                    be32(p, (i & 1) ? 0x80200400u : 0x7FDFFBFFu);
                } else p[k] = (u8)(i * 31 + k * 3);
            }
        }
    }
    const u32 vertices[] = {3,7,3,0,5,6,1,2};
    const u32 vertex_count = sizeof(vertices) / sizeof(vertices[0]);
    rsx_raw_vertex_plan plan;
    assert(rsx_raw_vertex_prepare(&state, 0xFFFF, vertices, vertex_count, 65536,
        cellGcmResolveLocated, &plan));
    assert(plan.range_count == 13); /* two interleaved attributes share a range */
    rsx_raw_vertex_plan too_small;
    assert(rsx_raw_vertex_prepare(&state, 0xFFFF, vertices, vertex_count, plan.bytes,
        cellGcmResolveLocated, &too_small));
    assert(!rsx_raw_vertex_prepare(&state, 0xFFFF, vertices, vertex_count, plan.bytes - 1,
        cellGcmResolveLocated, &too_small));
    assert(!rsx_raw_vertex_prepare(&state, 0xFFFF, vertices, vertex_count, 32,
        cellGcmResolveLocated, &too_small));
    rsx_state fragmented = {0};
    fragmented.vertex_attribs[0] = (rsx_vertex_attrib){0,0xFFFF0,16,4,2,1,1};
    const u32 crossing[] = {0,2};
    assert(!rsx_raw_vertex_prepare(&fragmented, 1, crossing, 2, 65536,
        fragmented_resolve, &too_small));
    const u32 sparse[] = {0,1000000};
    assert(!rsx_raw_vertex_prepare(&state, 1, sparse, 2, 65536,
        cellGcmResolveLocated, &too_small));
    const u32 max_output = vertex_count * 16 * 4 * sizeof(float);
    ID3D12Debug* debug = NULL;
    const int debug_enabled = SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&debug));
    if (debug_enabled) { debug->lpVtbl->EnableDebugLayer(debug); debug->lpVtbl->Release(debug); }
    IDXGIFactory4* factory = NULL; IDXGIAdapter* adapter = NULL; ID3D12Device* device = NULL;
    assert(SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory)));
    if (argc == 2 && (!strcmp(argv[1], "--hardware") || !strcmp(argv[1], "--hardware-indexed"))) {
        IDXGIFactory6* performance_factory = NULL;
        assert(SUCCEEDED(factory->lpVtbl->QueryInterface(factory, &IID_IDXGIFactory6,
            (void**)&performance_factory)));
        assert(SUCCEEDED(performance_factory->lpVtbl->EnumAdapterByGpuPreference(
            performance_factory, 0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            &IID_IDXGIAdapter, (void**)&adapter)));
        performance_factory->lpVtbl->Release(performance_factory);
    } else {
        assert(SUCCEEDED(factory->lpVtbl->EnumWarpAdapter(factory, &IID_IDXGIAdapter, (void**)&adapter)));
    }
    DXGI_ADAPTER_DESC adapter_info;
    assert(SUCCEEDED(adapter->lpVtbl->GetDesc(adapter, &adapter_info)));
    printf("Adapter: %ls\n", adapter_info.Description);
    assert(SUCCEEDED(D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_11_0,
        &IID_ID3D12Device, (void**)&device)));
    ID3D12InfoQueue* info_queue = NULL;
    device->lpVtbl->QueryInterface(device, &IID_ID3D12InfoQueue, (void**)&info_queue);
    if (info_queue) info_queue->lpVtbl->ClearStoredMessages(info_queue);
    ID3D12Resource* upload = buffer(device, plan.bytes + vertex_count * 4u, D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* output = buffer(device, max_output, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource* readback = buffer(device, max_output, D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    void* mapping = NULL; D3D12_RANGE empty = {0,0};
    assert(SUCCEEDED(upload->lpVtbl->Map(upload, 0, &empty, &mapping)));
    rsx_raw_vertex_write(&plan, vertices, vm_base, mapping);
    for (u32 i = 0; i < vertex_count; ++i)
        ((u32*)((u8*)mapping + plan.bytes))[i] = vertex_count - 1u - i;
    upload->lpVtbl->Unmap(upload, 0, NULL);
    char source[16384];
    snprintf(source, sizeof(source), "%s\nRWStructuredBuffer<float4> result:register(u0);"
        "[numthreads(16,1,1)]void main(uint3 id:SV_DispatchThreadID){"
        "uint vi=rsx_raw_vertices.Load(rsx_raw_vertices.Load(0u)+id.y*4u);"
        "result[id.y*16u+id.x]=rsx_raw_attribute(id.x,vi);}", rsx_raw_vertex_hlsl);
    if (indexed_test) snprintf(source, sizeof(source),
        "%s\nRWStructuredBuffer<float4> result:register(u0);"
        "float4 main(uint id:SV_VertexID):SV_Position{"
        "uint vi=rsx_raw_vertices.Load(rsx_raw_vertices.Load(0u)+id*4u);"
        "for(uint a=0;a<16;++a)result[id*16+a]=rsx_raw_attribute(a,vi);"
        "return float4(0,0,0,1);}", rsx_raw_vertex_hlsl);
    ID3DBlob* shader = compile(source, indexed_test ? "vs_5_0" : "cs_5_0");
    const char* vs_source = "struct VSInput{float4 a0:ATTR0;float4 a7:ATTR7;};"
        "struct VSOutput{float4 pos:SV_Position;};"
        "VSOutput main(VSInput input){VSOutput o;o.pos=input.a0+input.a7;return o;}";
    char wrapped[16384];
    assert(rsx_raw_vertex_wrap_hlsl(vs_source, 0x81, wrapped, sizeof(wrapped)));
    ID3DBlob* vs = compile(wrapped, "vs_5_0");
    vs->lpVtbl->Release(vs);
    D3D12_ROOT_PARAMETER parameters[2] = {0};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[0].Descriptor.ShaderRegister = 20;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC root_desc = {0};
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    root_desc.NumParameters = 2; root_desc.pParameters = parameters;
    ID3DBlob* serialized = NULL; ID3DBlob* errors = NULL;
    assert(SUCCEEDED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &errors)));
    ID3D12RootSignature* root = NULL;
    assert(SUCCEEDED(device->lpVtbl->CreateRootSignature(device, 0,
        serialized->lpVtbl->GetBufferPointer(serialized), serialized->lpVtbl->GetBufferSize(serialized),
        &IID_ID3D12RootSignature, (void**)&root)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {0}; pipeline_desc.pRootSignature = root;
    pipeline_desc.CS.pShaderBytecode = shader->lpVtbl->GetBufferPointer(shader);
    pipeline_desc.CS.BytecodeLength = shader->lpVtbl->GetBufferSize(shader);
    ID3D12PipelineState* pipeline = NULL;
    if (indexed_test) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics = {0};
        graphics.pRootSignature = root; graphics.VS = pipeline_desc.CS;
        graphics.SampleMask = UINT_MAX; graphics.SampleDesc.Count = 1;
        graphics.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        graphics.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        graphics.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        assert(SUCCEEDED(device->lpVtbl->CreateGraphicsPipelineState(device, &graphics,
            &IID_ID3D12PipelineState, (void**)&pipeline)));
    } else {
    assert(SUCCEEDED(device->lpVtbl->CreateComputePipelineState(device, &pipeline_desc,
        &IID_ID3D12PipelineState, (void**)&pipeline)));
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    ID3D12CommandQueue* queue = NULL; ID3D12CommandAllocator* allocator = NULL;
    ID3D12GraphicsCommandList* list = NULL;
    assert(SUCCEEDED(device->lpVtbl->CreateCommandQueue(device, &queue_desc,
        &IID_ID3D12CommandQueue, (void**)&queue)));
    assert(SUCCEEDED(device->lpVtbl->CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT,
        &IID_ID3D12CommandAllocator, (void**)&allocator)));
    assert(SUCCEEDED(device->lpVtbl->CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        allocator, pipeline, &IID_ID3D12GraphicsCommandList, (void**)&list)));
    if (indexed_test) {
        list->lpVtbl->SetGraphicsRootSignature(list, root);
        list->lpVtbl->SetGraphicsRootShaderResourceView(list, 0, upload->lpVtbl->GetGPUVirtualAddress(upload));
        list->lpVtbl->SetGraphicsRootUnorderedAccessView(list, 1, output->lpVtbl->GetGPUVirtualAddress(output));
        D3D12_INDEX_BUFFER_VIEW ibv = {0};
        ibv.BufferLocation = upload->lpVtbl->GetGPUVirtualAddress(upload) + plan.bytes;
        ibv.SizeInBytes = vertex_count * 4u; ibv.Format = DXGI_FORMAT_R32_UINT;
        list->lpVtbl->IASetIndexBuffer(list, &ibv);
        list->lpVtbl->IASetPrimitiveTopology(list, D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
        D3D12_VIEWPORT viewport = {0,0,1,1,0,1}; D3D12_RECT scissor = {0,0,1,1};
        list->lpVtbl->RSSetViewports(list, 1, &viewport);
        list->lpVtbl->RSSetScissorRects(list, 1, &scissor);
        list->lpVtbl->DrawIndexedInstanced(list, vertex_count, 1, 0, 0, 0);
    } else {
    list->lpVtbl->SetComputeRootSignature(list, root);
    list->lpVtbl->SetComputeRootShaderResourceView(list, 0, upload->lpVtbl->GetGPUVirtualAddress(upload));
    list->lpVtbl->SetComputeRootUnorderedAccessView(list, 1, output->lpVtbl->GetGPUVirtualAddress(output));
    list->lpVtbl->Dispatch(list, 1, vertex_count, 1);
    }
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = output;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->lpVtbl->ResourceBarrier(list, 1, &barrier);
    list->lpVtbl->CopyResource(list, readback, output);
    assert(SUCCEEDED(list->lpVtbl->Close(list)));
    ID3D12CommandList* commands[] = {(ID3D12CommandList*)list};
    queue->lpVtbl->ExecuteCommandLists(queue, 1, commands);
    ID3D12Fence* fence = NULL;
    assert(SUCCEEDED(device->lpVtbl->CreateFence(device, 0, D3D12_FENCE_FLAG_NONE,
        &IID_ID3D12Fence, (void**)&fence)));
    HANDLE event = CreateEventA(NULL, FALSE, FALSE, NULL);
    assert(event && SUCCEEDED(queue->lpVtbl->Signal(queue, fence, 1)));
    assert(SUCCEEDED(fence->lpVtbl->SetEventOnCompletion(fence, 1, event)));
    assert(WaitForSingleObject(event, 30000) == WAIT_OBJECT_0);
    D3D12_RANGE read_range = {0,max_output};
    assert(SUCCEEDED(readback->lpVtbl->Map(readback, 0, &read_range, &mapping)));
    const float* values = (const float*)mapping;
    u32 comparisons = 0;
    for (u32 v = 0; v < vertex_count; ++v) for (u32 a = 0; a < 16; ++a) {
        float expected[4]; rsx_fetch_attrib(&state, (int)a, vertices[v], expected);
        /* RPCS3 applies the signed-normalized bias before supplying absent W. */
        if (state.vertex_attribs[a].enabled && state.vertex_attribs[a].type == 1)
            for (u32 k = state.vertex_attribs[a].size; k < 3; ++k)
                expected[k] = 0.5f / 32767.5f;
        for (u32 k = 0; k < 4; ++k) {
            const float actual = values[(v * 16 + a) * 4 + k];
            if (isnan(expected[k])) assert(isnan(actual));
            else if (isinf(expected[k])) assert(actual == expected[k]);
            else if (fabsf(actual - expected[k]) > 0.000002f) {
                fprintf(stderr, "v%u a%u c%u: expected %.9g got %.9g\n", v,a,k,expected[k],actual);
                abort();
            }
            ++comparisons;
        }
    }
    readback->lpVtbl->Unmap(readback, 0, &empty);
    u32 validation_errors = 0;
    if (info_queue) {
        const UINT64 count = info_queue->lpVtbl->GetNumStoredMessagesAllowedByRetrievalFilter(info_queue);
        for (UINT64 i = 0; i < count; ++i) {
            SIZE_T bytes = 0; info_queue->lpVtbl->GetMessage(info_queue, i, NULL, &bytes);
            D3D12_MESSAGE* message = (D3D12_MESSAGE*)malloc(bytes);
            assert(message && SUCCEEDED(info_queue->lpVtbl->GetMessage(info_queue, i, message, &bytes)));
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                fprintf(stderr, "%s\n", message->pDescription); ++validation_errors;
            }
            free(message);
        }
        info_queue->lpVtbl->Release(info_queue);
    }
    CloseHandle(event); fence->lpVtbl->Release(fence);
    list->lpVtbl->Release(list); allocator->lpVtbl->Release(allocator); queue->lpVtbl->Release(queue);
    pipeline->lpVtbl->Release(pipeline); root->lpVtbl->Release(root);
    serialized->lpVtbl->Release(serialized); shader->lpVtbl->Release(shader);
    readback->lpVtbl->Release(readback); output->lpVtbl->Release(output); upload->lpVtbl->Release(upload);
    device->lpVtbl->Release(device); adapter->lpVtbl->Release(adapter); factory->lpVtbl->Release(factory);
    free(vm_base);
    printf("GPU raw vertex fetch: %u components matched, %u validation errors, debug layer %s\n",
        comparisons, validation_errors, debug_enabled ? "enabled" : "unavailable");
    return validation_errors ? 1 : 0;
}
