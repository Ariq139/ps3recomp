#!/usr/bin/env python3
"""Exercise the production disk-cache functions with the real Windows compiler.

No window or guest data is required. Pipeline persistence is also checked when
a D3D12 device is available. Extract the unchanged functions from the backend
so corruption and path tests cannot use a separate cache model.
Run from a Visual Studio developer shell with clang-cl available.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'win32', 'requires Windows D3DCompiler')
class DiskCache(unittest.TestCase):
    def test_roundtrip_corruption_and_relative_directories(self):
        compiler = shutil.which('clang-cl')
        if not compiler or not os.environ.get('INCLUDE'):
            self.skipTest('requires a Visual Studio developer shell')
        backend = (ROOT / 'libs/video/rsx_d3d12_backend.c').read_text(encoding='utf-8')
        start = backend.index('static u64 d3d12_pso_hash_bytes(')
        end = backend.index('static void d3d12_pipeline_cache_open(void)\n{', start)
        cache = backend[start:end]
        library = backend[end:backend.index('static void d3d12_pipeline_library_name(', end)]
        start = backend.index('static HRESULT d3d12_pipeline_library_get_or_create(')
        library += backend[start:backend.index('static void d3d12_vp_fp_hash_remove(', start)]
        start = backend.index('static const GUID s_iid_d3d12_device1')
        guids = backend[start:backend.index('/* Link libraries */', start)]
        workload = (ROOT / 'runtime/spu/spu_workload.c').read_text(encoding='utf-8')
        start = workload.index('uint64_t spu_workload_fingerprint(')
        fingerprint = workload[start:workload.index('/* ---- registry', start)]
        start = workload.index('typedef struct SpuDumpSeen')
        capture = fingerprint + workload[start:workload.index('#ifndef SPU_WORKLOAD_MAX', start)]
        preamble = r'''
#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
static struct {
    char shader_cache_dir[MAX_PATH];
    char pipeline_library_path[MAX_PATH];
    ID3D12Device* device;
    ID3D12Device1* device1;
    ID3D12PipelineLibrary* pipeline_library;
    void* pipeline_library_data;
    SIZE_T pipeline_library_data_size;
    CRITICAL_SECTION pipeline_library_lock;
    int pipeline_library_lock_ready;
    volatile LONG pipeline_library_dirty;
} s_d3d;
static int s_perf_disk_shader_hits, s_perf_disk_shader_misses;
static int s_perf_disk_pso_hits, s_perf_disk_pso_misses;
'''
        main = r'''
static const char shader[] =
    "float4 main() : SV_Target { return float4(1, 0, 0, 1); }";

static ID3DBlob* compile(void)
{
    ID3DBlob* code = NULL;
    ID3DBlob* errors = NULL;
    HRESULT hr = d3d12_compile_cached(shader, sizeof(shader) - 1, "cache-test",
        NULL, NULL, "main", "ps_5_0", 0, 0, &code, &errors);
    if (errors) {
        fprintf(stderr, "%s", (char*)errors->lpVtbl->GetBufferPointer(errors));
        errors->lpVtbl->Release(errors);
    }
    assert(SUCCEEDED(hr) && code);
    return code;
}

static DWORD WINAPI pipeline_worker(void* opaque)
{
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = d3d12_pipeline_library_get_or_create(L"parallel_test_v1",
        (const D3D12_GRAPHICS_PIPELINE_STATE_DESC*)opaque, &pso);
    if (pso) pso->lpVtbl->Release(pso);
    return (DWORD)hr;
}

static void check_pipeline_cache(int restored)
{
    HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                  &IID_ID3D12Device, (void**)&s_d3d.device);
    if (FAILED(hr)) { puts("D3D12 device unavailable; PSO check skipped"); return; }
    d3d12_pipeline_cache_open();
    if (!s_d3d.pipeline_library) {
        s_d3d.device->lpVtbl->Release(s_d3d.device);
        s_d3d.device = NULL;
        puts("pipeline library unavailable; PSO check skipped");
        return;
    }
    D3D12_ROOT_SIGNATURE_DESC root_desc = {0};
    ID3DBlob* signature = NULL;
    assert(SUCCEEDED(D3D12SerializeRootSignature(&root_desc,
        D3D_ROOT_SIGNATURE_VERSION_1, &signature, NULL)));
    ID3D12RootSignature* root = NULL;
    assert(SUCCEEDED(s_d3d.device->lpVtbl->CreateRootSignature(s_d3d.device, 0,
        signature->lpVtbl->GetBufferPointer(signature),
        signature->lpVtbl->GetBufferSize(signature),
        &IID_ID3D12RootSignature, (void**)&root)));
    signature->lpVtbl->Release(signature);
    const char vs[] = "float4 main(uint id:SV_VertexID):SV_Position{return float4(0,0,0,1);}";
    ID3DBlob* vertex = NULL;
    assert(SUCCEEDED(d3d12_compile_cached(vs, sizeof(vs)-1, "pipeline-test",
        NULL, NULL, "main", "vs_5_0", 0, 0, &vertex, NULL)));
    ID3DBlob* pixel = compile();
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {0};
    desc.pRootSignature = root;
    desc.VS.pShaderBytecode = vertex->lpVtbl->GetBufferPointer(vertex);
    desc.VS.BytecodeLength = vertex->lpVtbl->GetBufferSize(vertex);
    desc.PS.pShaderBytecode = pixel->lpVtbl->GetBufferPointer(pixel);
    desc.PS.BytecodeLength = pixel->lpVtbl->GetBufferSize(pixel);
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO;
    desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    desc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    desc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    ID3D12PipelineState* pso = NULL;
    assert(SUCCEEDED(d3d12_pipeline_library_get_or_create(L"cache_test_v1", &desc, &pso)));
    assert(pso);
    pso->lpVtbl->Release(pso);
    assert(s_perf_disk_pso_hits == restored);
    assert(s_perf_disk_pso_misses == !restored);
    assert(SUCCEEDED(d3d12_pipeline_library_get_or_create(L"cache_test_v1", &desc, &pso)));
    pso->lpVtbl->Release(pso);
    assert(s_perf_disk_pso_hits == restored + 1);
    HANDLE workers[8];
    for (unsigned i = 0; i < 8; i++) {
        workers[i] = CreateThread(NULL, 0, pipeline_worker, &desc, 0, NULL);
        assert(workers[i]);
    }
    assert(WaitForMultipleObjects(8, workers, TRUE, 30000) == WAIT_OBJECT_0);
    for (unsigned i = 0; i < 8; i++) {
        DWORD result;
        assert(GetExitCodeThread(workers[i], &result) && result == S_OK);
        CloseHandle(workers[i]);
    }
    d3d12_pipeline_cache_save();
    assert(GetFileAttributesA(s_d3d.pipeline_library_path) != INVALID_FILE_ATTRIBUTES);
    s_d3d.pipeline_library->lpVtbl->Release(s_d3d.pipeline_library);
    s_d3d.pipeline_library = NULL;
    free(s_d3d.pipeline_library_data);
    s_d3d.pipeline_library_data = NULL;
    DeleteCriticalSection(&s_d3d.pipeline_library_lock);
    s_d3d.pipeline_library_lock_ready = 0;
    s_d3d.device1->lpVtbl->Release(s_d3d.device1);
    s_d3d.device1 = NULL;
    root->lpVtbl->Release(root);
    vertex->lpVtbl->Release(vertex);
    pixel->lpVtbl->Release(pixel);
    s_d3d.device->lpVtbl->Release(s_d3d.device);
    s_d3d.device = NULL;
    puts(restored ? "pipeline cache reused across processes" : "pipeline cache saved");
}

int main(int argc, char** argv)
{
    if (argc > 1 && !strcmp(argv[1], "disabled")) {
        assert(_putenv_s("SPU_DUMP_JOB_IMAGES", "0") == 0);
        assert(spu_job_dump_directory() == NULL);
        puts("SPU capture disabled with zero");
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "default")) {
        char profile[MAX_PATH], expected[MAX_PATH];
        assert(GetCurrentDirectoryA(sizeof(profile), profile));
        strcat(profile, "\\profile");
        assert(_putenv_s("PS3RECOMP_CACHE_DIR", "") == 0);
        assert(_putenv_s("LOCALAPPDATA", profile) == 0);
        assert(_putenv_s("SPU_DUMP_JOB_IMAGES", "1") == 0);
        d3d12_cache_directory_init();
        snprintf(expected, sizeof(expected), "%s\\ps3recomp\\cache", profile);
        assert(!strcmp(s_d3d.shader_cache_dir, expected));
        strcat(expected, "\\spu_jobs");
        assert(!strcmp(spu_job_dump_directory(), expected));
        assert(GetFileAttributesA(expected) != INVALID_FILE_ATTRIBUTES);
        puts("default shader and SPU roots agree");
        return 0;
    }
    char short_dir[] = "x";
    d3d12_cache_mkdirs(short_dir);
    assert(GetFileAttributesA("x") != INVALID_FILE_ATTRIBUTES);
    assert(GetFileAttributesA("x") & FILE_ATTRIBUTE_DIRECTORY);
    char spu_dir[] = "p/q/cache";
    spu_job_dump_mkdirs(spu_dir);
    assert(GetFileAttributesA(spu_dir) != INVALID_FILE_ATTRIBUTES);
    assert(_putenv_s("PS3RECOMP_CACHE_DIR", "a/b/cache") == 0);

    if (argc > 1) {
        ID3DBlob* restored = compile();
        restored->lpVtbl->Release(restored);
        assert(s_perf_disk_shader_hits == 1 && s_perf_disk_shader_misses == 0);
        puts("cache reused across processes");
        check_pipeline_cache(1);
        return 0;
    }

    ID3DBlob* first = compile();
    ID3DBlob* second = compile();
    assert(s_perf_disk_shader_hits == 1 && s_perf_disk_shader_misses == 1);
    assert(first->lpVtbl->GetBufferSize(first) == second->lpVtbl->GetBufferSize(second));
    assert(!memcmp(first->lpVtbl->GetBufferPointer(first),
                   second->lpVtbl->GetBufferPointer(second),
                   first->lpVtbl->GetBufferSize(first)));
    first->lpVtbl->Release(first);
    second->lpVtbl->Release(second);

    u64 key_a, key_b;
    d3d12_shader_cache_key(shader, sizeof(shader) - 1, "cache-test", "main",
                           "ps_5_0", 0, 0, &key_a, &key_b);
    char path[MAX_PATH];
    snprintf(path, sizeof(path), "a/b/cache/%016llX_%016llX.cso",
             (unsigned long long)key_a, (unsigned long long)key_b);

    /* Same-length corruption must cause recompilation rather than a PSO fail. */
    FILE* file = fopen(path, "r+b");
    assert(file && fseek(file, -1, SEEK_END) == 0);
    int byte = fgetc(file);
    assert(byte != EOF && fseek(file, -1, SEEK_END) == 0);
    assert(fputc(byte ^ 0xff, file) != EOF && fclose(file) == 0);
    first = compile();
    first->lpVtbl->Release(first);
    assert(s_perf_disk_shader_hits == 1 && s_perf_disk_shader_misses == 2);

    /* An overflowing size in an otherwise valid header must be rejected. */
    D3D12ShaderCacheHeader header;
    file = fopen(path, "r+b");
    assert(file && fread(&header, sizeof(header), 1, file) == 1);
    header.bytecode_size = UINT64_MAX;
    assert(fseek(file, 0, SEEK_SET) == 0);
    assert(fwrite(&header, sizeof(header), 1, file) == 1 && fclose(file) == 0);
    first = compile();
    first->lpVtbl->Release(first);
    assert(s_perf_disk_shader_misses == 3);

    file = fopen(path, "wb");
    assert(file && fputc(0, file) == 0 && fclose(file) == 0);
    first = compile();
    first->lpVtbl->Release(first);
    second = compile();
    second->lpVtbl->Release(second);
    assert(s_perf_disk_shader_hits == 2 && s_perf_disk_shader_misses == 4);

    /* Exercise both the shared root and the uncapped, deduplicated SPU writer. */
    assert(_putenv_s("SPU_DUMP_JOB_IMAGES", "1") == 0);
    const char* dir = spu_job_dump_directory();
    assert(dir && !strcmp(dir, "a/b/cache\\spu_jobs"));
    for (unsigned i = 0; i < 70; i++) {
        uint8_t image[32];
        memset(image, (int)i, sizeof(image));
        uint64_t fp = spu_workload_fingerprint(image, sizeof(image));
        spu_dump_job_image(dir, fp, image, sizeof(image));
        spu_dump_job_image(dir, fp, image, sizeof(image));
        snprintf(path, sizeof(path), "%s/spujob_%016llX_32.bin", dir,
                 (unsigned long long)fp);
        assert(spu_dump_file_matches(path, image, sizeof(image)));
    }
    WIN32_FIND_DATAA found;
    snprintf(path, sizeof(path), "%s/*.bin", dir);
    HANDLE search = FindFirstFileA(path, &found);
    assert(search != INVALID_HANDLE_VALUE);
    unsigned count = 0;
    do { count++; } while (FindNextFileA(search, &found));
    FindClose(search);
    assert(count == 70);
    check_pipeline_cache(0);
    puts("cache roundtrip, corruption recovery, counters, and directories passed");
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='ps3recomp-cache-test-') as tmp:
            source = Path(tmp) / 'cache_test.c'
            source.write_text(preamble + guids + cache + library + capture + main, encoding='utf-8')
            exe = Path(tmp) / 'cache_test.exe'
            result = subprocess.run([compiler, '/nologo', '/TC', '/std:c11', '/W0',
                                     str(source), '/Fe:' + str(exe),
                                     '/link', 'd3dcompiler.lib', 'd3d12.lib'],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('corruption recovery', result.stdout)
            pso_supported = 'pipeline cache saved' in result.stdout
            result = subprocess.run([str(exe), 'reload'], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('reused across processes', result.stdout)
            if pso_supported:
                self.assertIn('pipeline cache reused across processes', result.stdout)
            else:
                print('D3D12 pipeline library unsupported; shader/SPU tests completed')
            result = subprocess.run([str(exe), 'disabled'], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe), 'default'], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
