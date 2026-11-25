/*
 * ps3recomp - D3D12 RSX Backend
 *
 * Translates RSX GPU state to D3D12 rendering commands.
 *
 * Phase 1 implementation:
 *   - Win32 window + D3D12 device + swap chain
 *   - Clear render target to RSX clear color
 *   - Present with vsync
 *   - Basic vertex-colored triangle rendering
 *
 * This file is C with COM calls (D3D12 is a COM API).
 * We use the C interface (__uuidof not available in C, so we
 * define GUIDs manually).
 */

#ifdef _WIN32

#include <time.h>
#include "ps3emu/log_first.h"
#include "rsx_d3d12_backend.h"
#include "cellGcmSys.h"
#include "rsx_primitives.h"
#include "rsx_vertex_fetch.h"
#include "rsx_vertex_raw.h"
#include "rsx_raw_upload_cache.h"
#include "rsx_fp_decompiler.h"
#include "rsx_vp_decompiler.h"
#include "rsx_texture_layout.h"
#include "../input/cellKb.h"
#include "../input/cellMouse.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

/* Recognized by hybrid-graphics drivers when exported by the game EXE.
 * DXGI adapter selection below remains authoritative for our D3D12 device. */
__declspec(dllexport) DWORD NvOptimusEnablement = 1;
__declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 1;

/* We need these GUIDs — define them here to avoid uuid.lib dependency.
 * initguid.h only turns DEFINE_GUID into a definition for headers included
 * after it, so it has to come first. */
#include <initguid.h>

/* D3D12 headers */
#include <d3d12.h>
#include <d3d12sdklayers.h>   /* ID3D12Debug / ID3D12InfoQueue */
#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
static const char* d3d12_setting_subvp_dbg(void);

/* D3D12 pipeline libraries are optional on older runtimes/drivers. Keep the
 * interface IDs local so this C translation unit does not need dxguid.lib. */
static const GUID s_iid_d3d12_device1 =
    {0x77acce80, 0x638e, 0x4e65, {0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e}};
static const GUID s_iid_d3d12_pipeline_library =
    {0xc64226a8, 0x9201, 0x46af, {0xb4, 0xcc, 0x53, 0xfb, 0x9f, 0xf7, 0x41, 0x4f}};

#include "rsx_d3d12_device.h"
#include "rsx_d3d12_shadow_border.h"

/* Link libraries */
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

static HRESULT d3d12_compile_cached(
    LPCVOID source, SIZE_T source_size, LPCSTR source_name,
    const D3D_SHADER_MACRO* defines, ID3DInclude* include,
    LPCSTR entry_point, LPCSTR target, UINT flags1, UINT flags2,
    ID3DBlob** code, ID3DBlob** errors);
static void d3d12_pipeline_cache_open(void);
static void d3d12_pipeline_cache_save(void);

/* ---------------------------------------------------------------------------
 * Constants
 * -----------------------------------------------------------------------*/

#define FRAME_COUNT         2   /* double buffering */
#define MAX_VERTICES    393216  /* per-frame vertex buffer (dbgfont submits
                                 * ~7.5k verts/frame; leave generous headroom) */
#define RSX_CLEAR_DEPTH_MASK   0x01u
#define RSX_CLEAR_STENCIL_MASK 0x02u
#define RSX_CLEAR_COLOR_MASK   0xF0u
#define RSX_TEXTURE_UNITS 16  /* Fragment texture registers t0-t15. */
#define RSX_PIXEL_NORMAL_SAMPLERS 2 /* legacy sampler slots used only on overflow. */
#define RSX_SHADOW_COMPARE_FIRST_REGISTER RSX_PIXEL_NORMAL_SAMPLERS
#define RSX_SHADOW_COMPARE_FUNCS 8 /* raw ZFUNC values 0..7 at s2-s9 */
#define RSX_SAMPLER_TABLE_SIZE 16
#define RSX_SAMPLER_TABLE_COUNT 128 /* 128 * 16 == the D3D12 visible sampler limit. */
#define MAX_DRAWS         8192  /* Host-side staging capacity. The previous
                                 * 2048 limit left little headroom over the
                                 * measured P5 peak (~1841). This affects host
                                 * allocations, not the guest RSX command ABI. */
/* Per-draw VP constant-buffer slot: vp_c[512] + posscale + posoffset
 * (514 vec4 = 8224 B) rounded up to D3D12's 256-byte CBV alignment.
 * Constants are snapshotted at RECORD time -- wave's passes each set their
 * own texScale/offset uniforms, so one per-frame snapshot ran every pass
 * with the LAST pass's constants. */
#define VP_CB_STRIDE      8448
/* A 4096-byte FP can contain at most 128 instruction/CONST pairs.
 * b1 matches PSConstants: fp_constants[max(count,1)], then fp_alpha. */
#define VP_FP_MAX_CONSTANTS 128u
#define VP_FPCB_STRIDE    2304
#define VERTEX_STRIDE       36  /* bytes per host vertex: pos3 + col4 + uv2 */
#define VP_VERT_STRIDE (16u * sizeof(float) * 4u) /* 16 float4 RSX attributes */

/* Guest stencil state snapshotted with each draw. D3D12 supports per-face
 * compare/ops but has one read mask, write mask, and reference for both faces;
 * the back-face values are retained so the PSO key and diagnostics remain
 * faithful even when the host API cannot express an asymmetric mask/ref. */
typedef struct {
    u32 enable, two_sided;
    u32 front_func, front_ref, front_read_mask, front_write_mask;
    u32 front_fail, front_zfail, front_zpass;
    u32 back_func, back_ref, back_read_mask, back_write_mask;
    u32 back_fail, back_zfail, back_zpass;
} RSXStencilState;

/* D3D12 permits only sixteen pixel-sampler registers. Guest programs can use
 * different RSX samplers per texture unit, so the HLSL remapper assigns the
 * active units compact registers and the draw binds the matching descriptors. */
typedef struct {
    u8 normal_slot[RSX_TEXTURE_UNITS];
    u8 shadow_slot[RSX_TEXTURE_UNITS];
    u16 normal_mask;
    u16 shadow_mask;
    u8 legacy_fallback;
} D3D12FPSamplerLayout;

typedef struct {
    u32 vb_byte_offset; /* offset into vb_mapped where this draw's verts live */
    u32 vertex_count;
    u32 topology;       /* rsx_topology (NOT a D3D value; see topo_to_d3d) */
    int textured;       /* 1 = sample the bound font/atlas texture (dbgfont) */
    int is_vp;          /* 1 = real VP path: vb_byte_offset indexes vp_vb (float4) */
    /* VP path per-draw shader/texture state, captured at draw_arrays time. */
    u32 fp_addr;        /* SET_SHADER_PROGRAM value (guest FP ucode location)   */
    int fp_exp32;       /* SET_SHADER_CONTROL 32-bit-exports bit at draw time   */
    float alpha_ref;    /* surface-format-decoded alpha-test reference */
    float blend_color[4]; /* dynamic blend factor in RGBA order */
    u8 gamma_masks[RSX_TEXTURE_UNITS]; /* sampled-channel sRGB decode mask */
    u32 color_mask_mrt, blend_enable_mrt;
    u32 shader_packer;
    int srgb_output;
    u32 alpha_ctl;      /* alpha test: enable<<16 | (func&0xFF)<<8 | ref */
    u32 begin_epoch;    /* SET_BEGIN_END generation, for batch concatenation */
    int flat_shading;
    int two_sided_lighting;
    u32 cull;           /* packed face culling: bit0 enable, bit1 cull FRONT
                         * (else BACK), bit2 front face is CCW. RSX culls back
                         * faces on most solid geometry; rendering everything
                         * double-sided lets a shell's interior faces show
                         * through and shade unlit (black). */
    int poly_fill;
    u32 poly_scale_bits, poly_bias_bits;
    u32 fog_mode, fog_param0_bits, fog_param1_bits;
    u32 shader_window;
    u32 texcoord_control_mask, anti_aliasing_control;
    u32 cmask;          /* D3D write mask from SET_COLOR_MASK at draw time
                         * (wave's sim passes write single lanes of the height
                         * maps; ignoring the mask stomped persistent state) */
    /* Per-unit textures (t0-t15): decompiled FPs sample all 16 RSX units
     * (demosaic's interpolation passes read 3). Captured at draw time. */
    struct {
        u32 off;        /* resolved vm offset (guest upload source), 0 = none */
        u32 raw;        /* raw RSX offset (offscreen-RT matching) */
        u32 w, h, fmt;  /* dims + RSX base format */
        u32 ctrl1;      /* NV4097 TEXTURE_CONTROL1: component remap crossbar */
        u32 filter, address, control0, pitch; /* sampler state + row pitch */
        u32 border_color; /* guest TEXTURE_BORDER_COLOR, snapshotted per draw */
        u32 mips;       /* SET_TEXTURE_FORMAT bits 16..31: mipmap level count.
                         * Cube faces sit one whole mip pyramid apart, so this
                         * is what sets the face stride. */
        int cube;       /* SET_TEXTURE_FORMAT bit 2: a cube texture. Sampled
                         * with a 3-component direction, not a 2D uv -- treating
                         * one as 2D is what makes an environment-mapped chrome
                         * surface come out with black patches and banding. */
        int set;
    } tex[RSX_TEXTURE_UNITS];
    u32 tex_shadow_mask;
    u64 tex_shadow_compare_key; /* one raw ZFUNC nibble per fragment unit */
    rsx_fp_shadow_desc tex_shadow[RSX_TEXTURE_UNITS];
    int tex_rt[RSX_TEXTURE_UNITS];      /* pre-pass: OffRT index sampled by unit, -1 = none */
    int tex_feedback[RSX_TEXTURE_UNITS]; /* snapshot the active color target before this draw */
    /* NV40 vertex TXL units use t16-t19 and are captured independently from
     * the fragment texture bank. 2D, 1D-as-one-row-2D, and cube are supported. */
    struct {
        u32 off, raw, w, h, fmt, ctrl1, pitch, mips;
        int cube, is_1d, set;
    } vtex[RSX_MAX_VERTEX_TEXTURES];
    int vtex_rt[RSX_MAX_VERTEX_TEXTURES];
    u32 vtex_mask, vtex_cube_mask, vtex_1d_mask;
    int tex_slot;       /* legacy single-slot path (atlas); -1 = none */
    int vs_idx;         /* VPVSEntry slot for this draw's vertex program (-1 = primary) */
    u32 vp_input_mask;  /* original RSX attributes present in the packed VP stream */
    u32 vp_stride;      /* packed float4 bytes per VP vertex; 256 for safe fallback */
    u32 vp_raw_bytes;
    u32 vp_raw_index_offset;
    int blend;          /* guest blend enable at draw time */
    u32 blend_key;      /* packed guest blend state (factors+equation), PSO key */
    int depth_test;     /* guest depth-test enable at draw time */
    int depth_write;    /* guest depth write mask at draw time */
    u32 depth_func;     /* NV4097/GL comparison enum at draw time */
    u32 zmin_max_control; /* depth clip/clamp state at draw time */
    RSXStencilState stencil;
    /* Render target key: registered display buffers use a reserved synthetic
     * key; all other surfaces use their resolved guest VM offset. Zero remains
     * the legacy direct-to-swapchain target. */
    u32 rt_off;
    int rt_none;            /* CELL_GCM_SURFACE_TARGET_NONE: depth-only pass */
    u32 zeta_off, zeta_ea, zeta_pitch, zeta_format;
                             /* guest depth surface identity at draw time */
    int tex_depth[RSX_TEXTURE_UNITS];
    int vtex_depth[RSX_MAX_VERTEX_TEXTURES];
    /* Which per-draw constant slot this record's constants were written to.
     * vp_record_cb writes them keyed by the RECORD index, so any pass that
     * reorders or compacts s_d3d.draws (DRAW_KEEP_TEX, DRAW_LAST_TEX,
     * DRAW_LIMIT) would otherwise make every surviving draw read some other
     * object's MVP -- silently transforming it somewhere else entirely. */
    u32 cb_slot;
    u32 vp_cb_slot;
    u32 rt_mrt[3];      /* colour targets B,C,D (MRT1/2/3), 0 = none. Deferred
                         * shading G-buffers write 3-4 targets in one pass. */
    u32 rt_w, rt_h;     /* surface clip dims at record time (offscreen RT size) */
    u32 rt_fmt;         /* RSX surface colour format (SET_SURFACE_FORMAT [4:0]) */
    /* Guest viewport rect at draw time (target pixels). Sub-viewport layouts
     * (wave's debug tiles) position quads with this, not with constants --
     * forcing full-target viewports drew every tile window-sized. */
    u32 vp_x, vp_y, vp_w, vp_h;
    u32 scissor_x, scissor_y, scissor_w, scissor_h;
    float vp_min_depth, vp_max_depth;
    /* Ordered clear op (offscreen surfaces only; display clears stay the
     * frame-start backbuffer clear). is_clear records also set is_vp so the
     * legacy replay pass skips them. */
    int   is_clear;
    u32   clear_flags;
    float clear_depth;
    u8    clear_stencil;
    float cc[4];
    u32   clear_mask;   /* CLEAR_SURFACE bits: 0x01 depth, 0x02 stencil, 0xF0 colour */
    float clear_z;
    u8    clear_s;
} D3D12DrawRecord;

/* Non-swapchain surface (guest display or render-to-texture). Persistent across
 * frames so a selected display buffer retains its contents until the next flip. */
typedef struct {
    ID3D12Resource*       res;
    ID3D12Resource*       up;          /* init-upload staging (kept alive) */
    ID3D12Resource*       feedback;    /* copy used for same-surface sampling */
    u32                   off, w, h;   /* raw RSX offset + dims */
    u32                   dxgi;        /* DXGI_FORMAT of the resource */
    D3D12_RESOURCE_STATES st;          /* tracked resource state */
    int                   used;        /* referenced this frame */
    int                   contents_valid; /* display buffer has a known image */
    u64                   color_serial; /* most recent guest color write */
} OffRT;

/* Persistent guest zeta surfaces. Reusing by resolved guest address preserves
 * depth contents across passes and frames, including depth-only targets. */
typedef struct {
    ID3D12Resource*       res;
    u32                   ea, w, h, pitch, zeta_format;
    DXGI_FORMAT           resource_format, dsv_format, srv_format;
    D3D12_RESOURCE_STATES st;
    int                   used;
    int                   initialized;
    u64                   depth_serial; /* most recent guest depth write */
} OffDepth;
#define MAX_OFF_RTS  64  /* Preserve multi-pass and alternating-frame surface chains. */
#define MAX_OFF_DEPTHS 32
#define DISPLAY_RT_KEY_BASE 0xFFFF0000u
/* Reserve disjoint ranges for cached guest textures, render targets, depth
 * surfaces, per-slot atlases, and per-draw windows. Writing
 * texture slot 1+i over RT/depth/draw descriptors corrupts already-built tables. */
#define VP_TEX_SLOTS 512
#define RT_SRV_BASE (1 + VP_TEX_SLOTS)
#define RT_DEPTH_SRV_BASE (RT_SRV_BASE + MAX_OFF_RTS)
#define ATLAS_SRV_BASE (RT_DEPTH_SRV_BASE + MAX_OFF_DEPTHS)
/* Per-draw SRV windows: each VP draw gets t0-t15 for fragment textures and
 * t16-t19 for vertex TXL textures. */
#define DRAW_SRV_BASE (ATLAS_SRV_BASE + FRAME_COUNT)
#define DRAW_SRV_STRIDE (RSX_TEXTURE_UNITS + RSX_MAX_VERTEX_TEXTURES)
#define DRAW_SRV_FRAME_STRIDE (MAX_DRAWS * DRAW_SRV_STRIDE)
#define DRAW_SRV_FRAME_BASE(frame) (DRAW_SRV_BASE + (u32)(frame) * DRAW_SRV_FRAME_STRIDE)
#define DRAW_SRV_SLOT(frame, draw, unit) \
    (DRAW_SRV_FRAME_BASE(frame) + (u32)(draw) * DRAW_SRV_STRIDE + (u32)(unit))
#define SRV_HEAP_SIZE (DRAW_SRV_BASE + FRAME_COUNT * DRAW_SRV_FRAME_STRIDE)

/* Keep host-side capacity choices within the guaranteed D3D12 limits for the
 * feature level requested below. These checks fail at compile time if a later
 * capacity increase would exceed the API contract. */
#if SRV_HEAP_SIZE > D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1
#error "D3D12 CBV/SRV/UAV heap exceeds the guaranteed shader-visible limit"
#endif
#if (RSX_SAMPLER_TABLE_COUNT * RSX_SAMPLER_TABLE_SIZE) > D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE
#error "D3D12 sampler heap exceeds the shader-visible sampler limit"
#endif
#if DRAW_SRV_STRIDE > D3D12_COMMONSHADER_INPUT_RESOURCE_REGISTER_COUNT
#error "D3D12 shader resource table exceeds the per-stage register limit"
#endif
#if RSX_SAMPLER_TABLE_SIZE > D3D12_COMMONSHADER_SAMPLER_REGISTER_COUNT
#error "D3D12 sampler table exceeds the per-stage sampler register limit"
#endif

/* Per-frame VP texture slot: a guest texture uploaded for this frame's VP
 * draws (re-uploaded every frame -- gcm/cube's plasma animates in guest
 * memory). SRV lives at heap index 1+slot. */
typedef struct {
    ID3D12Resource* res;
    ID3D12Resource* up[FRAME_COUNT]; /* upload buffers are submission-slot local */
    u32 off, w, h, fmt, pitch, mips; /* current contents (resource reused when dims match) */
    DXGI_FORMAT dxgi; /* decoded host format used by both pixel and vertex SRVs */
    u32 csum;           /* sparse checksum of the source bytes last uploaded */
    int cube;           /* resource is a 6-face cube, sampled as TextureCube */
    u32 key;            /* the ORIGINAL bound offset -- the cache lookup key.
                         * off is the RESOLVED source after TEX_OFF_BIAS/TEX_REMAP
                         * shift it, so keying on off never matches the raw offset
                         * a later draw arrives with, and every draw re-uploads. */
    int used;           /* referenced this frame */
    u64 last_use;       /* submission serial, for eviction across batches */
} VPTexSlot;
/* Distinct textures uploadable per frame. Four was enough for a UI/atlas-style
 * frame but not for a scene: this title binds ~20 distinct textures per frame,
 * so every draw past the fourth got slot -1 from vp_upload_tex_slot ("out of
 * slots") and drew UNTEXTURED. That reads as a correctly-lit but solid black
 * object, which is indistinguishable from a missing texture upload. */

/* Decompiled-VS cache: one entry per distinct vertex-program ucode seen at
 * draw time (hashed). Apps switch VPs between draws (gcm/cube: its MVP cube VP
 * vs dbgfont-gcm's text VP); compiling only the first left later draws
 * transformed by the wrong program (text offscreen). */
typedef struct {
    u32 hash;               /* FNV-1a of the ucode */
    u32 branch_base;        /* absolute RSX instruction slot used by branches */
    u32 vtex_mask, vtex_cube_mask, vtex_1d_mask;
    u32 input_mask, input_stride;
    ID3DBlob* vs;
    ID3DBlob* raw_vs;
    int uses_c03;
    u32 hash_next; /* stable slot+1, independent of realloc */
} VPVSEntry;
/* Queued draw records and PSOs retain these indices until execution. Grow
 * without eviction so compiling another program cannot change their meaning. */
#define VP_VS_INITIAL_CAPACITY 64

/* Compiled guest-FP pipeline cache: decompiled VS (by cache slot) + decompiled
 * PS (fragment ucode at fp_addr). */
typedef struct {
    u32 fp_addr;
    int vs_idx;             /* VPVSEntry slot this PSO's VS came from */
    u32 vs_hash;            /* validates the slot hasn't been evicted */
    u32 raw_vertex_fetch;
    u32 gen;
    u32 blend;              /* packed guest blend key (enable+factors+eq, PSO key) */
    u32 color_mask_mrt, blend_enable_mrt;
    u8 gamma_masks[RSX_TEXTURE_UNITS];
    int srgb_output;
    int depth_test;         /* guest depth-test enable (PSO key) */
    int depth_write;        /* guest depth write mask (PSO key) */
    u32 depth_func;         /* guest NV4097/GL comparison enum (PSO key) */
    u32 zmin_max_control;   /* NV4097_SET_ZMIN_MAX_CONTROL (PSO key) */
    RSXStencilState stencil;
    int nrt;                /* bound colour target count (PSO key)       */
    u32 rtfmt;              /* DXGI format of the colour targets (PSO key) */
    u32 dsvfmt;             /* DXGI format of the depth target (PSO key) */
    int exp32;              /* 32-bit-exports control bit (PSO key)       */
    u32 ucode_hash;         /* FNV-1a of the FP ucode: apps re-patch inline
                             * constants per frame (wave's stamp position/
                             * amplitude) -- address-only keying served the
                             * stale compile forever. */
    u32 cmask;              /* colour write mask (PSO key) */
    u32 cull;               /* packed face culling (PSO key) */
    int poly_fill;
    u32 poly_scale_bits, poly_bias_bits;
    u32 fog_mode, fog_param0_bits, fog_param1_bits;
    u32 shader_window;
    u32 texcoord_control_mask, anti_aliasing_control;
    int flat_shading, two_sided_lighting;
    u32 cube_mask;          /* which units are cube textures (PSO key): the HLSL
                             * declares those samplers as TextureCube, so a cube
                             * and a 2D variant of the same program are different
                             * pipelines and must not share a cache entry. */
    u32 shadow_mask;         /* depth units using Dref comparison sampling */
    u64 shadow_compare_key;  /* raw ZFUNC per unit; specialized in HLSL */
    rsx_fp_shadow_desc shadow[RSX_TEXTURE_UNITS];
    u32 depth_rgba_mask;    /* depth surfaces sampled as packed A8/D8R8G8B8 */
    D3D12FPSamplerLayout sampler_layout;
    u64 key_hash;
    u32 hash_next;          /* bucket-chain entry index + 1; zero terminates */
    ID3D12PipelineState* pso;
} VPFPEntry;
/* Keep generated pixel-shader bytecode separately from full PSOs. Different
 * blend/depth/raster variants often share exactly the same HLSL. */
typedef struct {
    u64 source_hash;
    u32 source_size;
    char* source;
    ID3DBlob* blob;
    u64 last_use;
} VPFPBlobEntry;
/* Persona 5's effects and scene materials use more than 1024 distinct
 * fragment-program/RSX-state PSO combinations. The old FIFO cache evicted
 * entries while the same scene was still drawing, causing thousands of
 * CreateGraphicsPipelineState misses every second. Keep the warmed set. */
#define VP_FP_CACHE 8192
#define VP_FP_HASH_BUCKETS 16384
#define VP_FP_BLOB_CACHE 256

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

typedef struct {
    /* Window */
    HWND hwnd;
    u32  width;
    u32  height;
    int  window_closed;
    int  raw_mouse_registered;

    /* D3D12 core */
    ID3D12Device*              device;
    ID3D12Device1*             device1; /* optional pipeline-library interface */
    ID3D12PipelineLibrary*     pipeline_library;
    void*                      pipeline_library_data; /* CreatePipelineLibrary retains this pointer */
    SIZE_T                     pipeline_library_data_size;
    CRITICAL_SECTION           pipeline_library_lock;
    int                        pipeline_library_lock_ready;
    char                       shader_cache_dir[MAX_PATH];
    char                       pipeline_library_path[MAX_PATH];
    volatile LONG              pipeline_library_dirty; /* mutation count since last save */
    ID3D12CommandQueue*        cmd_queue;
    IDXGISwapChain3*           swap_chain;
    ID3D12DescriptorHeap*      rtv_heap;
    u32                        rtv_descriptor_size;
    ID3D12Resource*            render_targets[FRAME_COUNT];
    ID3D12CommandAllocator*    cmd_allocators[FRAME_COUNT];
    ID3D12GraphicsCommandList* cmd_list;

    /* Synchronization */
    ID3D12Fence* fence;
    HANDLE       fence_event;
    u64          fence_values[FRAME_COUNT]; /* last submission using each swapchain buffer */
    u64          submission_fence_values[FRAME_COUNT]; /* retire point for each CPU/GPU slot */
    u64          next_fence_value;
    u64          last_fence_value;
    u32          frame_index;
    u32          present_buffer_id;   /* buffer the current flip names (for
                                       * the CPU-blit scanout fallback) */

    /* Pipeline */
    ID3D12RootSignature*  root_signature;
    ID3D12PipelineState*  pipeline_state;         /* triangle class — default */
    ID3D12PipelineState*  pipeline_state_lines;   /* line class */
    ID3D12PipelineState*  pipeline_state_points;  /* point class */

    /* Depth/stencil */
    ID3D12DescriptorHeap* dsv_heap;
    ID3D12Resource*       depth_buffer;
    u32                   dsv_inc;

    /* Dynamic vertex buffer (upload heap) */
    ID3D12Resource*       vertex_buffer;
    D3D12_VERTEX_BUFFER_VIEW vb_view;
    void*                 vb_mapped;      /* persistently mapped */
    u32                   vb_offset;      /* current write position */

    int                   pipeline_ready; /* 1 if root sig + PSO created */

    /* Debug: copy the presented backbuffer to disk as BMP for the first N
     * frames (enabled by env CELLMARK_DUMP). Lets us verify what actually
     * rendered without racing the window/process lifetime. */
    ID3D12Resource*       readback_buf;
    u32                   readback_pitch;
    int                   dump_frames_left;
    int                   dump_skip_left;   /* CELLMARK_DUMP_SKIP: presents to ignore first */

    /* Textured pipeline (dbgfont / 2D atlas quads). The font atlas is an 8-bit
     * coverage texture uploaded as R8_UNORM and sampled in the pixel shader. */
    ID3D12PipelineState*  pipeline_state_tex;   /* triangle + texture + blend */
    ID3D12DescriptorHeap* srv_heap;             /* shader-visible, 1 SRV slot  */
    ID3D12DescriptorHeap* sampler_heap[FRAME_COUNT]; /* in-flight-safe shader-visible tables */
    u32                   sampler_inc;
    u32                   sampler_table_count;
    u32                   sampler_table_last;
    u64                   sampler_table_hash[RSX_SAMPLER_TABLE_COUNT];
    D3D12_SAMPLER_DESC    sampler_table_descs[RSX_SAMPLER_TABLE_COUNT][RSX_SAMPLER_TABLE_SIZE];
    ID3D12Resource*       tex_resource[FRAME_COUNT]; /* atlas per in-flight slot */
    ID3D12Resource*       tex_upload[FRAME_COUNT];   /* slot-local staging buffers */
    u32                   tex_w, tex_h;         /* atlas dimensions              */
    u32                   tex_slot_w[FRAME_COUNT], tex_slot_h[FRAME_COUNT];
    int                   tex_ready[FRAME_COUNT]; /* 1 once that slot has an atlas */
    u32                   tex_src_offset;       /* guest RSX offset of atlas    */
    int                   tex_bound;            /* a texture is bound for draws */
    int                   tex_dirty[FRAME_COUNT]; /* each slot catches up on bind */

    /* Real RSX vertex-program path: the captured VP is decompiled to HLSL and
     * fed converted RSX attributes plus the vp_c[] constant bank. Exact input
     * analysis packs only referenced attributes; uncertain programs use all 16. */
    ID3D12RootSignature*  vp_root_sig;          /* CBV(b0) + SRV table + sampler */
    ID3D12PipelineState*  pipeline_state_vp;    /* decompiled VS + atlas PS      */
    ID3D12PipelineState*  pipeline_state_vp_color; /* decompiled VS + colour PS (untextured 3D) */
    ID3D12PipelineState*  pipeline_state_vp_z16;
    ID3D12PipelineState*  pipeline_state_vp_color_z16;
    ID3D12PipelineState*  pipeline_state_vp_no_rt;
    ID3D12PipelineState*  pipeline_state_vp_no_rt_z16;
    ID3D12PipelineState*  pipeline_state_vp_color_no_rt;
    ID3D12PipelineState*  pipeline_state_vp_color_no_rt_z16;
    ID3D12Resource*       vp_vb;                /* packed float4 VP inputs, per-frame */
    void*                 vp_vb_mapped;
    u32                   vp_vb_offset;
    u32                   vp_vb_vertex_count;
    u32                   vp_upload_mask, vp_upload_stride, vp_upload_count;
    ID3D12Resource*       vp_cb;                /* per-draw VP constant slots    */
    void*                 vp_cb_mapped;
    ID3D12Resource*       vp_fpcb;              /* per-draw FP texscale (b1)     */
    void*                 vp_fpcb_mapped;
    int                   vp_ready;             /* VS+PSO compiled ok            */
    u32                   vp_compiled_bytes;    /* ucode size when last compiled */
    ID3DBlob*             vp_vs_blob;           /* kept for guest-FP PSO builds  */
    u32                   vp_gen;               /* bumped per VP recompile       */
    int                   vp_uses_c03;          /* VS references vp_c[0..3]      */
    VPTexSlot             vp_tex[VP_TEX_SLOTS]; /* per-frame VP texture slots    */
    VPVSEntry*            vp_vs;   /* stable indices; storage may grow */
    int                   vp_vs_capacity;
    int                   vp_vs_n;
    u32                   vp_vs_heads[1024];
    u32                   vp_tex_heads[1024];
    u32                   vp_tex_next[VP_TEX_SLOTS], vp_tex_bucket[VP_TEX_SLOTS];
    VPFPEntry             vp_fp[VP_FP_CACHE];   /* guest-FP PSO cache            */
    int                   vp_fp_n;
    u32                   vp_fp_hash[VP_FP_HASH_BUCKETS];
    int                   vp_fp_next_replace;
    VPFPBlobEntry         vp_fp_blob[VP_FP_BLOB_CACHE];
    int                   vp_fp_blob_n;
    u64                   vp_fp_blob_clock;
    u32                   srv_inc;              /* CBV_SRV_UAV descriptor size   */
    /* VP path: latest texture bound per unit (t0-t15). */
    struct { u32 off, raw, w, h, fmt, ctrl1, mips; int cube; int set; } cur_texs[RSX_TEXTURE_UNITS];

    /* Render-to-texture: offscreen RT pool + their RTV heap. */
    OffRT                 off_rt[MAX_OFF_RTS];
    OffDepth              off_depth[MAX_OFF_DEPTHS];
    u64                   surface_write_serial;
    /* DRAW_ARRAYS batch merging (see d3d12_draw_arrays): the vertex index one
     * past the last batch, and whether the current BEGIN/END may still be
     * extended. */
    u32                   merge_first_end;
    int                   merge_prev_draw;
    ID3D12DescriptorHeap* rt_rtv_heap;          /* MAX_OFF_RTS RTVs (CPU only)   */

    /* Frame-parity double buffering for the per-draw upload streams (vp_vb
     * vertices, vp_cb constants, vp_fpcb texscales): records for frame N+1
     * are written while frame N's GPU work may still be reading -- writes go
     * to the other half. Toggled at the end of render_frame. */
    int                   vp_parity;

    /* Per-frame draw recording */
    int                   frame_recording; /* 1 if cmd list is open for recording */
    u32                   draw_count;      /* draws this frame */
    D3D12DrawRecord       draws[MAX_DRAWS];

    /* Pointer to current RSX state (set before draw calls) */
    const rsx_state*      current_rsx_state;

    /* Current frame state */
    float clear_color[4];  /* RGBA float */
    u32   clear_flags;      /* pending guest clear mask for frame start */
    float clear_depth;
    u8    clear_stencil;

    /* Stats */
    u64 frame_count;
    u64 last_fps_time;
    u32 fps;

    int initialized;
} D3D12State;

static D3D12State s_d3d;
/* Used by initialization before the frame-sync helper definitions. */
static int s_gpu_work_pending;
static int s_batch_slot_acquired;

static D3D12_COMPARISON_FUNC d3d12_shadow_sample_func(u32 raw_func);
char g_rsx_title_base[128] = "ps3recomp";
static u32 s_dbg_last_draws = 0;
static int is_display_target_key(u32 key);
static int display_rt_id_from_key(u32 key);
static u64 s_req_verts = 0, s_req_draws = 0, s_drop_draws = 0;
static u64 s_drop_vertices = 0, s_drop_input_elements = 0, s_drop_records = 0;

/* Keep the upstream opt-in drop report on this backend too. The renderer has
 * additional local fallbacks, but early rejection reasons and flip counts are
 * still useful when diagnosing a black frame. */
enum { D3D12_DROP_NOT_READY, D3D12_DROP_BAD_COUNT, D3D12_DROP_NO_VP_RESOURCES,
       D3D12_DROP_PRIM_UNWIRED, D3D12_DROP_UPLOAD_EMPTY, D3D12_DROP_RECORDS_FULL,
       D3D12_DROP_REASONS };
static const char* const s_d3d12_drop_names[D3D12_DROP_REASONS] = {
    "not_ready", "bad_count", "no_vp_resources", "prim_unwired", "upload_empty", "records_full" };
static struct {
    u64 calls, drop[D3D12_DROP_REASONS];
    u64 flips, flips_rendered, flips_deferred;
    u32 last_display_offset;
} s_d3d12_stats;
#define D3D12_DROP(reason) do { s_d3d12_stats.drop[reason]++; return; } while (0)

static void d3d12_print_stats(void)
{
    static long every = -1;
    if (every < 0) {
        const char* env = getenv("D3D12_DROP_STATS_EVERY");
        every = env ? atol(env) : 0;
    }
    if (every <= 0 || s_d3d12_stats.flips % (u64)every) return;

    u64 dropped = 0;
    for (int i = 0; i < D3D12_DROP_REASONS; i++) dropped += s_d3d12_stats.drop[i];
    if (!dropped && !s_d3d12_stats.flips_deferred) return;

    printf("[D3D12 drops] flips=%llu rendered=%llu deferred=%llu last_display_offset=0x%X draw_calls=%llu",
           (unsigned long long)s_d3d12_stats.flips,
           (unsigned long long)s_d3d12_stats.flips_rendered,
           (unsigned long long)s_d3d12_stats.flips_deferred,
           s_d3d12_stats.last_display_offset,
           (unsigned long long)s_d3d12_stats.calls);
    for (int i = 0; i < D3D12_DROP_REASONS; i++)
        printf(" %s=%llu", s_d3d12_drop_names[i],
               (unsigned long long)s_d3d12_stats.drop[i]);
    printf(" vp_guest_fp=0 vp_fallback_ps=0\n");
    fflush(stdout);
}

static int d3d12_draw_record_available(void)
{
    if (s_d3d.draw_count < MAX_DRAWS) {
        s_d3d.draws[s_d3d.draw_count].vp_raw_bytes = 0;
        s_d3d.draws[s_d3d.draw_count].vp_raw_index_offset = 0;
        return 1;
    }
    s_drop_records++;
    s_d3d12_stats.drop[D3D12_DROP_RECORDS_FULL]++;
    { static int n = 0;
      if (PS3_LOG_FIRST(n, 8))
          fprintf(stderr, "[DRAWCAP] draw-record capacity %u exhausted; dropped=%llu%c",
                  MAX_DRAWS, (unsigned long long)s_drop_records, 10); }
    return 0;
}
/* Raw bind offset of the duck's texture, identified by CONTENT (duck.tga is
 * overwhelmingly yellow, avg 243,191,23). VRAM offsets move between runs, so a
 * hard-coded offset in a filter silently matches nothing and the run looks like
 * "renders nothing" -- which cost real time. DRAW_KEEP_TEX=duck resolves here. */
/* Copy of the last completed frame, bound wherever a draw samples a
 * DISPLAY-SIZED texture. On RSX this title renders its reflection into a corner
 * of the render surface and then samples that surface as a texture; our backend
 * renders into a D3D backbuffer, so the guest memory behind that sampler is
 * never written and it reads empty -- which is why the water had no reflection.
 * Feeding it the previous frame costs one frame of latency, which for a water
 * reflection is not visible. */
static ID3D12Resource* s_screen_copy = NULL;
/* Sub-viewport render-to-texture. This title renders its reflection/refraction
 * pre-pass into a REGION of the same surface (vp 0,208 512x512 and 0,208
 * 1024x512, cmask=F) and then samples a texture of exactly that size. On
 * hardware the pass lands in the buffer that texture points at; our backend
 * renders into D3D resources, so that guest buffer is never written and every
 * fluid sampler reads zero. Capture each region into a matching resource and
 * bind it for samplers whose guest source is empty and whose size matches.
 *
 * SUBVP_RTT=1 to enable. OFF by default because it cannot be validated on this
 * title yet: the fluid draws that would consume these captures never rasterize
 * (see below), so binding them changes nothing observable. The capture and
 * binding themselves are confirmed working -- SUBVP_DBG shows the 512x512 and
 * 256x256 regions captured and handed to fp 0x5DB81/0x5EA81/0x5EF81/0x5E381. */
#define SUBVP_SLOTS 4
static struct { ID3D12Resource* res; u32 x, y, w, h; } s_subvp[SUBVP_SLOTS];
static int s_subvp_n = 0;

static void subvp_note(u32 x, u32 y, u32 w, u32 h)
{
    if (!w || !h || w > s_d3d.width || h > s_d3d.height) return;
    if (w == s_d3d.width && h == s_d3d.height) return;      /* full-screen pass */
    for (int i = 0; i < s_subvp_n; i++)
        if (s_subvp[i].w == w && s_subvp[i].h == h) { s_subvp[i].x = x; s_subvp[i].y = y; return; }
    if (s_subvp_n >= SUBVP_SLOTS) return;
    s_subvp[s_subvp_n].x = x; s_subvp[s_subvp_n].y = y;
    s_subvp[s_subvp_n].w = w; s_subvp[s_subvp_n].h = h;
    s_subvp[s_subvp_n].res = NULL;
    s_subvp_n++;
}

/* Copy each noted region out of the backbuffer. Called at the same moment as
 * screen_copy_capture -- after the reduced-viewport passes and before the
 * full-screen scene overwrites them. */
static void subvp_capture(u32 fi, int source_rt)
{
    static int en = -1;
    if (en < 0) { const char* e = getenv("SUBVP_RTT"); en = e ? atoi(e) : 0; }
    if (!en || !s_d3d.device || !s_subvp_n) return;
    ID3D12Resource* source = s_d3d.render_targets[fi];
    D3D12_RESOURCE_STATES source_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    u32 source_w = s_d3d.width, source_h = s_d3d.height;
    if (source_rt >= 0) {
        if (source_rt >= MAX_OFF_RTS) return;
        OffRT* r = &s_d3d.off_rt[source_rt];
        /* The screen-copy/subviewport path stores RGBA8. Do not copy a stale
         * swapchain image when the producer is an incompatible guest surface. */
        if (!r->res || r->dxgi != DXGI_FORMAT_R8G8B8A8_UNORM ||
            r->w < s_d3d.width || r->h < s_d3d.height)
            return;
        source = r->res;
        source_state = r->st;
        source_w = r->w;
        source_h = r->h;
    }
    for (int i = 0; i < s_subvp_n; i++) {
        if (s_subvp[i].x + s_subvp[i].w > source_w ||
            s_subvp[i].y + s_subvp[i].h > source_h)
            continue;
        if (!s_subvp[i].res) {
            D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC td = {0};
            td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width = s_subvp[i].w; td.Height = s_subvp[i].h;
            td.DepthOrArraySize = 1; td.MipLevels = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
            td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
                    s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &td,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, NULL,
                    &IID_ID3D12Resource, (void**)&s_subvp[i].res)))
                { s_subvp[i].res = NULL; continue; }
        }
        D3D12_RESOURCE_BARRIER bb[2] = {0};
        if (source_state != D3D12_RESOURCE_STATE_COPY_SOURCE) {
            bb[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bb[0].Transition.pResource   = source;
            bb[0].Transition.StateBefore = source_state;
            bb[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
            bb[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        bb[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bb[1].Transition.pResource   = s_subvp[i].res;
        bb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        bb[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        bb[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list,
            source_state != D3D12_RESOURCE_STATE_COPY_SOURCE ? 2 : 1,
            source_state != D3D12_RESOURCE_STATE_COPY_SOURCE ? bb : &bb[1]);

        D3D12_TEXTURE_COPY_LOCATION dstl = {0}, srcl = {0};
        dstl.pResource = s_subvp[i].res;
        dstl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dstl.SubresourceIndex = 0;
        srcl.pResource = source;
        srcl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; srcl.SubresourceIndex = 0;
        D3D12_BOX box; box.left = s_subvp[i].x; box.top = s_subvp[i].y;
        box.front = 0; box.right = s_subvp[i].x + s_subvp[i].w;
        box.bottom = s_subvp[i].y + s_subvp[i].h; box.back = 1;
        s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dstl, 0, 0, 0, &srcl, &box);

        bb[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bb[0].Transition.pResource   = s_subvp[i].res;
        bb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        bb[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        bb[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, bb);
        if (source_state != D3D12_RESOURCE_STATE_COPY_SOURCE) {
            D3D12_RESOURCE_BARRIER restore = {0};
            restore.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            restore.Transition.pResource = source;
            restore.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            restore.Transition.StateAfter = source_state;
            restore.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &restore);
        }
        { static int _n = 0; if (d3d12_setting_subvp_dbg() && PS3_LOG_FIRST(_n, 8))
            fprintf(stderr, "[SUBVP] captured %ux%u from (%u,%u)%c",
                    s_subvp[i].w, s_subvp[i].h, s_subvp[i].x, s_subvp[i].y, 10); }
    }
}
static void screen_copy_capture(u32 fi, int source_rt);   /* fwd */
static int  s_sc_dump_pending = 0;
static u32 s_duck_raw = 0;   /* as BOUND (what draw records carry) */
static u32 s_duck_off = 0;   /* as RESOLVED (what the uploader sees)   */
/* PERF=1: where a frame's CPU time actually goes. Guessing at this is how you
 * spend an afternoon optimising the wrong loop. */
#include "rsx_constant_snapshot_cache.h"
_Static_assert(VP_CB_STRIDE == RSX_CONSTANT_SNAPSHOT_BYTES, "constant cache stride");
static rsx_constant_snapshot_cache s_vp_cb_cache;
#include "rsx_d3d12_gpu_timer.h"
static rsx_d3d12_gpu_timer s_gpu_timer;
static double s_perf_fence_wait = 0.0;
static u32 s_perf_fence_calls = 0, s_perf_rt_recreates = 0, s_perf_rt_evicts = 0;
static double s_perf_tex = 0.0, s_perf_vtx = 0.0, s_perf_rf = 0.0;
static double s_perf_record = 0.0, s_perf_pre = 0.0, s_perf_srv = 0.0, s_perf_pso = 0.0;
static int s_perf_pso_calls = 0, s_perf_pso_miss = 0, s_perf_pso_hashbytes = 0;
static int s_perf_ps_blob_hits = 0, s_perf_ps_blob_misses = 0;
static int s_perf_disk_shader_hits = 0, s_perf_disk_shader_misses = 0;
static int s_perf_disk_pso_hits = 0, s_perf_disk_pso_misses = 0;
static u64    s_perf_nverts = 0;
static u64    s_perf_vtx_pool_items = 0, s_perf_vtx_expanded_primitives = 0;
static rsx_raw_upload_cache s_raw_upload_cache;
static u32 s_perf_raw_reused, s_perf_raw_indexed;
static u32* s_raw_vertex_ids;
static u32 s_raw_vertex_id_capacity;
static u32 s_perf_raw_draws;
static u64 s_perf_raw_bytes, s_perf_raw_saved;
static u32    s_perf_vtx_pool_calls = 0, s_perf_vtx_expanded_draws = 0;
static u64    s_perf_texbytes = 0;
static int    s_perf_ntex = 0;
static u64    s_vp_tex_clock = 0;
static u32    s_perf_tex_hits = 0, s_perf_tex_uploads = 0, s_perf_tex_allocs = 0;
/* Launch-time diagnostic flags: avoid CRT environment locks on every draw. */
static int d3d_flag_NO_ALPHATEST(void)
{
    static _Thread_local int value = -1;
    if (value < 0) value = getenv("NO_ALPHATEST") != NULL;
    return value;
}
static int d3d_flag_VP_FIXPROJ(void)
{
    static _Thread_local int value = -1;
    if (value < 0) value = getenv("VP_FIXPROJ") != NULL;
    return value;
}
static int d3d_flag_VP_NOFIXPROJ(void)
{
    static _Thread_local int value = -1;
    if (value < 0) value = getenv("VP_NOFIXPROJ") != NULL;
    return value;
}

static double perf_now(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
static int perf_on(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("PERF"); v = e ? atoi(e) : 0; }
    return v;
}
/* DBG_LOCK: running clip-space centroid of the tracked mesh (RSX_DBG_VTX's
 * texture), so the debug camera can follow it. The ducks are driven by the
 * physics sim and drift every frame, so a fixed DBG_CENTER loses them as soon
 * as the magnification is high enough to make one recognisable. */
static float s_lock_x = 0.0f, s_lock_y = 0.0f;
static int   s_lock_valid = 0;
/* Accumulated over one frame, published at the frame boundary. Averaging as the
 * frame is recorded makes the aim wander while it is being used, which at high
 * magnification walks the target off-screen entirely. */
static double s_lock_sx = 0.0, s_lock_sy = 0.0;
static u32    s_lock_n  = 0;

/* ---------------------------------------------------------------------------
 * Win32 window
 * -----------------------------------------------------------------------*/

static u16 d3d12_key_to_hid(WPARAM vk, LPARAM lp)
{
    if (vk >= 'A' && vk <= 'Z') return (u16)(CELL_KEYC_A + (vk - 'A'));
    if (vk >= '1' && vk <= '9') return (u16)(CELL_KEYC_1 + (vk - '1'));
    if (vk >= VK_F1 && vk <= VK_F12) return (u16)(0x3A + (vk - VK_F1));

    switch (vk) {
    case '0': return CELL_KEYC_0;
    case VK_RETURN: return CELL_KEYC_ENTER;
    case VK_ESCAPE: return CELL_KEYC_ESCAPE;
    case VK_BACK: return CELL_KEYC_BS;
    case VK_TAB: return CELL_KEYC_TAB;
    case VK_SPACE: return CELL_KEYC_SPACE;
    case VK_DELETE: return CELL_KEYC_DELETE;
    case VK_RIGHT: return CELL_KEYC_RIGHT_ARROW;
    case VK_LEFT: return CELL_KEYC_LEFT_ARROW;
    case VK_DOWN: return CELL_KEYC_DOWN_ARROW;
    case VK_UP: return CELL_KEYC_UP_ARROW;
    case VK_SHIFT:
        return (((lp >> 16) & 0xFF) == 0x36) ? 0xE5 : 0xE1;
    case VK_CONTROL:
        return (lp & (1u << 24)) ? 0xE4 : 0xE0;
    case VK_MENU:
        return (lp & (1u << 24)) ? 0xE6 : 0xE2;
    case VK_LWIN: return 0xE3;
    case VK_RWIN: return 0xE7;
    case VK_OEM_1: return 0x33;
    case VK_OEM_PLUS: return 0x2E;
    case VK_OEM_COMMA: return 0x36;
    case VK_OEM_MINUS: return 0x2D;
    case VK_OEM_PERIOD: return 0x37;
    case VK_OEM_2: return 0x38;
    case VK_OEM_3: return 0x35;
    case VK_OEM_4: return 0x2F;
    case VK_OEM_5: return 0x31;
    case VK_OEM_6: return 0x30;
    case VK_OEM_7: return 0x34;
    case VK_INSERT: return 0x49;
    case VK_HOME: return 0x4A;
    case VK_PRIOR: return 0x4B;
    case VK_END: return 0x4D;
    case VK_NEXT: return 0x4E;
    default: return CELL_KEYC_NO_EVENT;
    }
}

static void d3d12_release_keyboard(void)
{
    for (u16 key = 0; key < 256; ++key)
        cellKb_keyEvent(0, key, 0);
}

static void d3d12_mouse_buttons(USHORT flags)
{
    if (flags & RI_MOUSE_LEFT_BUTTON_DOWN)  cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_1, 1);
    if (flags & RI_MOUSE_LEFT_BUTTON_UP)    cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_1, 0);
    if (flags & RI_MOUSE_RIGHT_BUTTON_DOWN) cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_2, 1);
    if (flags & RI_MOUSE_RIGHT_BUTTON_UP)   cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_2, 0);
    if (flags & RI_MOUSE_MIDDLE_BUTTON_DOWN)cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_3, 1);
    if (flags & RI_MOUSE_MIDDLE_BUTTON_UP)  cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_3, 0);
    if (flags & RI_MOUSE_BUTTON_4_DOWN)     cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_4, 1);
    if (flags & RI_MOUSE_BUTTON_4_UP)       cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_4, 0);
    if (flags & RI_MOUSE_BUTTON_5_DOWN)     cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_5, 1);
    if (flags & RI_MOUSE_BUTTON_5_UP)       cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_5, 0);
}

static void d3d12_mouse_position(HWND hwnd, int x, int y)
{
    RECT client;
    if (!GetClientRect(hwnd, &client) || client.right <= 0 || client.bottom <= 0)
        return;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= client.right) x = client.right - 1;
    if (y >= client.bottom) y = client.bottom - 1;
    cellMouse_positionEvent(0,
        (s32)((u64)x * s_d3d.width / client.right),
        (s32)((u64)y * s_d3d.height / client.bottom));
}

static void d3d12_mouse_position_from_cursor(HWND hwnd)
{
    POINT point;
    if (GetCursorPos(&point) && ScreenToClient(hwnd, &point))
        d3d12_mouse_position(hwnd, point.x, point.y);
}

static LRESULT CALLBACK d3d12_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_CLOSE:
        s_d3d.window_closed = 1;
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        d3d12_release_keyboard();
        PostQuitMessage(0);
        return 0;
    case WM_KILLFOCUS:
        d3d12_release_keyboard();
        for (u8 button = CELL_MOUSE_BUTTON_1; button <= CELL_MOUSE_BUTTON_5; button <<= 1)
            cellMouse_buttonEvent(0, button, 0);
        return 0;
    case WM_SETCURSOR:
        if (LOWORD(lp) == HTCLIENT && cellMouse_isInitialized()) {
            SetCursor(NULL);
            return TRUE;
        }
        break;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYUP: {
        const u16 key = d3d12_key_to_hid(wp, lp);
        if (key != CELL_KEYC_NO_EVENT)
            cellKb_keyEvent(0, key, msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN);
        if (wp == VK_ESCAPE) {
            if (msg == WM_KEYDOWN) {
                s_d3d.window_closed = 1;
                DestroyWindow(hwnd);
            }
        }
        if (msg == WM_SYSKEYDOWN || msg == WM_SYSKEYUP)
            return DefWindowProcA(hwnd, msg, wp, lp);
        return 0;
    }
    case WM_INPUT: {
        RAWINPUT input;
        UINT size = sizeof(input);
        if (s_d3d.raw_mouse_registered &&
            GetRawInputData((HRAWINPUT)lp, RID_INPUT, &input, &size,
                            sizeof(RAWINPUTHEADER)) == sizeof(input) &&
            input.header.dwType == RIM_TYPEMOUSE) {
            const RAWMOUSE* mouse = &input.data.mouse;
            /* Motion comes from the Windows cursor in WM_MOUSEMOVE. Raw
             * device counts ignore cursor acceleration and window scaling,
             * so they cannot keep a software pointer at the host position. */
            if (mouse->usButtonFlags & (RI_MOUSE_LEFT_BUTTON_DOWN | RI_MOUSE_LEFT_BUTTON_UP |
                                        RI_MOUSE_RIGHT_BUTTON_DOWN | RI_MOUSE_RIGHT_BUTTON_UP |
                                        RI_MOUSE_MIDDLE_BUTTON_DOWN | RI_MOUSE_MIDDLE_BUTTON_UP |
                                        RI_MOUSE_BUTTON_4_DOWN | RI_MOUSE_BUTTON_4_UP |
                                        RI_MOUSE_BUTTON_5_DOWN | RI_MOUSE_BUTTON_5_UP))
                d3d12_mouse_position_from_cursor(hwnd);
            d3d12_mouse_buttons(mouse->usButtonFlags);
            if (mouse->usButtonFlags & RI_MOUSE_WHEEL)
                cellMouse_wheelEvent(0, (SHORT)mouse->usButtonData / WHEEL_DELTA);
            if (mouse->usButtonFlags & RI_MOUSE_HWHEEL)
                cellMouse_tiltEvent(0, (SHORT)mouse->usButtonData / WHEEL_DELTA);
        }
        return DefWindowProcA(hwnd, msg, wp, lp);
    }
    case WM_MOUSEMOVE: {
        d3d12_mouse_position(hwnd, (s16)(lp & 0xFFFF),
                             (s16)((lp >> 16) & 0xFFFF));
        return 0;
    }
    case WM_LBUTTONDOWN: case WM_LBUTTONUP:
        d3d12_mouse_position(hwnd, (s16)(lp & 0xFFFF),
                             (s16)((lp >> 16) & 0xFFFF));
        if (!s_d3d.raw_mouse_registered)
            cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_1, msg == WM_LBUTTONDOWN);
        return 0;
    case WM_RBUTTONDOWN: case WM_RBUTTONUP:
        d3d12_mouse_position(hwnd, (s16)(lp & 0xFFFF),
                             (s16)((lp >> 16) & 0xFFFF));
        if (!s_d3d.raw_mouse_registered)
            cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_2, msg == WM_RBUTTONDOWN);
        return 0;
    case WM_MBUTTONDOWN: case WM_MBUTTONUP:
        if (!s_d3d.raw_mouse_registered)
            cellMouse_buttonEvent(0, CELL_MOUSE_BUTTON_3, msg == WM_MBUTTONDOWN);
        return 0;
    case WM_XBUTTONDOWN: case WM_XBUTTONUP:
        if (!s_d3d.raw_mouse_registered) {
            const u8 button = HIWORD(wp) == XBUTTON1
                ? CELL_MOUSE_BUTTON_4 : CELL_MOUSE_BUTTON_5;
            cellMouse_buttonEvent(0, button, msg == WM_XBUTTONDOWN);
        }
        return TRUE;
    case WM_MOUSEWHEEL:
        if (!s_d3d.raw_mouse_registered)
            cellMouse_wheelEvent(0, (SHORT)HIWORD(wp) / WHEEL_DELTA);
        return 0;
    case WM_MOUSEHWHEEL:
        if (!s_d3d.raw_mouse_registered)
            cellMouse_tiltEvent(0, (SHORT)HIWORD(wp) / WHEEL_DELTA);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

static HICON s_game_icon_large;
static HICON s_game_icon_small;

static HICON load_game_icon(int width, int height)
{
    char icon_path[MAX_PATH];
    const DWORD length = GetModuleFileNameA(NULL, icon_path,
                                             (DWORD)sizeof(icon_path));
    if (length == 0 || length >= sizeof(icon_path)) return NULL;

    char* separator = strrchr(icon_path, '\\');
    char* forward_separator = strrchr(icon_path, '/');
    if (forward_separator && (!separator || forward_separator > separator))
        separator = forward_separator;
    if (!separator) return NULL;

    static const char icon_file[] = "ps3recomp.ico";
    const size_t directory_length = (size_t)(separator - icon_path) + 1;
    if (directory_length + sizeof(icon_file) > sizeof(icon_path)) return NULL;
    memcpy(icon_path + directory_length, icon_file, sizeof(icon_file));

    return (HICON)LoadImageA(NULL, icon_path, IMAGE_ICON, width, height,
                             LR_LOADFROMFILE | LR_DEFAULTCOLOR);
}

static HWND create_window(u32 width, u32 height, const char* title)
{
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = d3d12_wndproc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    if (!s_game_icon_large)
        s_game_icon_large = load_game_icon(GetSystemMetrics(SM_CXICON),
                                           GetSystemMetrics(SM_CYICON));
    if (!s_game_icon_small)
        s_game_icon_small = load_game_icon(GetSystemMetrics(SM_CXSMICON),
                                           GetSystemMetrics(SM_CYSMICON));
    wc.hIcon = s_game_icon_large;
    wc.hIconSm = s_game_icon_small;
    wc.lpszClassName = "ps3recomp_d3d12";
    RegisterClassExA(&wc);

    RECT wr = {0, 0, (LONG)width, (LONG)height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

    /* Keep the display awake (see rsx_null_backend.c). */
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    HWND hwnd = CreateWindowExA(
        0, "ps3recomp_d3d12",
        title ? title : "ps3recomp (D3D12)",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left, wr.bottom - wr.top,
        NULL, NULL, GetModuleHandle(NULL), NULL);
    if (hwnd) {
        RAWINPUTDEVICE mouse = {0x01, 0x02, 0, hwnd};
        s_d3d.raw_mouse_registered =
            RegisterRawInputDevices(&mouse, 1, sizeof(mouse)) != FALSE;
    }
    return hwnd;
}

/* ---------------------------------------------------------------------------
 * D3D12 initialization
 * -----------------------------------------------------------------------*/

static int init_d3d12(u32 width, u32 height)
{
    HRESULT hr;

    /* Enable debug layer in debug builds */
    /* Debug layer: on in debug builds, or in any build when D3D12_DBG is set
     * (so a Release run can capture exact PSO/validation errors). */
    if (
#ifdef NDEBUG
        getenv("D3D12_DBG")
#else
        !getenv("D3D12_NO_DBG") /* D3D12_NO_DBG=1 disables the debug layer in debug builds */
#endif
    ) {
        ID3D12Debug* debug_controller = NULL;
        hr = D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&debug_controller);
        if (SUCCEEDED(hr) && debug_controller) {
            debug_controller->lpVtbl->EnableDebugLayer(debug_controller);
            debug_controller->lpVtbl->Release(debug_controller);
            printf("[D3D12] Debug layer enabled\n");
        } else {
            printf("[D3D12] Debug layer requested but unavailable (0x%08lX) -- "
                   "install 'Graphics Tools' optional feature\n", hr);
        }
    }

    /* Create DXGI factory */
    IDXGIFactory4* factory = NULL;
    hr = CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory);
    if (FAILED(hr)) {
        printf("[D3D12] ERROR: CreateDXGIFactory1 failed (0x%08lX)\n", hr);
        return -1;
    }

    /* Create D3D12 device. On a dual-GPU laptop a NULL adapter usually lands on
     * the integrated GPU; explicitly pick the high-performance (discrete)
     * adapter via IDXGIFactory6. Set CELLMARK_IGPU to force the low-power one
     * (for A/B testing a suspected iGPU-driver stall). */
    {
        IDXGIFactory6* factory6 = NULL;
        if (SUCCEEDED(factory->lpVtbl->QueryInterface(
                factory, &IID_IDXGIFactory6, (void**)&factory6))) {
            DXGI_GPU_PREFERENCE pref = getenv("CELLMARK_IGPU")
                ? DXGI_GPU_PREFERENCE_MINIMUM_POWER
                : DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
            for (UINT ai = 0; ; ai++) {
                IDXGIAdapter1* adapter = NULL;
                hr = factory6->lpVtbl->EnumAdapterByGpuPreference(
                    factory6, ai, pref, &IID_IDXGIAdapter1, (void**)&adapter);
                if (FAILED(hr)) break;
                DXGI_ADAPTER_DESC1 ad;
                if (SUCCEEDED(adapter->lpVtbl->GetDesc1(adapter, &ad)) &&
                    !(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                    SUCCEEDED(D3D12CreateDevice((IUnknown*)adapter,
                        D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device,
                        (void**)&s_d3d.device))) {
                    printf("[D3D12] adapter: %ls (VRAM %llu MB, %s preference, "
                           "vendor=%04X device=%04X)\n", ad.Description,
                           (unsigned long long)(ad.DedicatedVideoMemory >> 20),
                           pref == DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE
                               ? "high-performance" : "minimum-power override",
                           ad.VendorId, ad.DeviceId);
                    adapter->lpVtbl->Release(adapter);
                    break;
                }
                adapter->lpVtbl->Release(adapter);
                adapter = NULL;
            }
            factory6->lpVtbl->Release(factory6);
        }
    }
    if (!s_d3d.device) {
        /* Older DXGI has no performance ordering. Prefer dedicated memory
         * among D3D12-capable hardware adapters rather than letting NULL
         * silently choose the display's integrated adapter. VRAM is only
         * a fallback heuristic; modern systems use DXGI's ranking above. */
        IDXGIAdapter1* best = NULL;
        DXGI_ADAPTER_DESC1 best_desc = {0};
        for (UINT ai = 0; ; ai++) {
            IDXGIAdapter1* adapter = NULL;
            if (FAILED(factory->lpVtbl->EnumAdapters1(factory, ai, &adapter))) break;
            DXGI_ADAPTER_DESC1 ad;
            if (SUCCEEDED(adapter->lpVtbl->GetDesc1(adapter, &ad)) &&
                !(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice((IUnknown*)adapter,
                    D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, NULL)) &&
                (!best || ad.DedicatedVideoMemory > best_desc.DedicatedVideoMemory)) {
                if (best) best->lpVtbl->Release(best);
                best = adapter;
                best_desc = ad;
            } else {
                adapter->lpVtbl->Release(adapter);
            }
        }
        hr = best ? D3D12CreateDevice((IUnknown*)best, D3D_FEATURE_LEVEL_11_0,
                                     &IID_ID3D12Device, (void**)&s_d3d.device)
                  : DXGI_ERROR_NOT_FOUND;
        if (best) best->lpVtbl->Release(best);
        if (FAILED(hr)) {
            printf("[D3D12] ERROR: D3D12CreateDevice failed (0x%08lX)\n", hr);
            factory->lpVtbl->Release(factory);
            return -1;
        }
        printf("[D3D12] adapter: %ls (VRAM %llu MB, hardware enumeration fallback)\n",
               best_desc.Description,
               (unsigned long long)(best_desc.DedicatedVideoMemory >> 20));
    }

    d3d12_pipeline_cache_open();

    /* Create command queue */
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = s_d3d.device->lpVtbl->CreateCommandQueue(
        s_d3d.device, &queue_desc, &IID_ID3D12CommandQueue, (void**)&s_d3d.cmd_queue);
    if (FAILED(hr)) {
        printf("[D3D12] ERROR: CreateCommandQueue failed\n");
        factory->lpVtbl->Release(factory);
        return -1;
    }

    /* Create swap chain */
    DXGI_SWAP_CHAIN_DESC1 sc_desc = {0};
    sc_desc.Width = width;
    sc_desc.Height = height;
    sc_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = FRAME_COUNT;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    IDXGISwapChain1* swap_chain1 = NULL;
    hr = factory->lpVtbl->CreateSwapChainForHwnd(
        factory, (IUnknown*)s_d3d.cmd_queue,
        s_d3d.hwnd, &sc_desc, NULL, NULL, &swap_chain1);
    if (FAILED(hr)) {
        printf("[D3D12] ERROR: CreateSwapChainForHwnd failed (0x%08lX)\n", hr);
        factory->lpVtbl->Release(factory);
        return -1;
    }

    /* Disable Alt+Enter fullscreen toggle */
    factory->lpVtbl->MakeWindowAssociation(factory, s_d3d.hwnd, DXGI_MWA_NO_ALT_ENTER);
    factory->lpVtbl->Release(factory);

    /* Query SwapChain3 interface */
    hr = swap_chain1->lpVtbl->QueryInterface(
        swap_chain1, &IID_IDXGISwapChain3, (void**)&s_d3d.swap_chain);
    swap_chain1->lpVtbl->Release(swap_chain1);
    if (FAILED(hr)) {
        printf("[D3D12] ERROR: QueryInterface for SwapChain3 failed\n");
        return -1;
    }

    s_d3d.frame_index = s_d3d.swap_chain->lpVtbl->GetCurrentBackBufferIndex(s_d3d.swap_chain);

    /* Create RTV descriptor heap */
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc = {0};
    rtv_heap_desc.NumDescriptors = FRAME_COUNT;
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hr = s_d3d.device->lpVtbl->CreateDescriptorHeap(
        s_d3d.device, &rtv_heap_desc, &IID_ID3D12DescriptorHeap, (void**)&s_d3d.rtv_heap);
    if (FAILED(hr)) return -1;

    s_d3d.rtv_descriptor_size = s_d3d.device->lpVtbl->GetDescriptorHandleIncrementSize(
        s_d3d.device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    /* Create RTVs for each frame */
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle;
    s_d3d.rtv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.rtv_heap, &rtv_handle);

    for (u32 i = 0; i < FRAME_COUNT; i++) {
        hr = s_d3d.swap_chain->lpVtbl->GetBuffer(
            s_d3d.swap_chain, i, &IID_ID3D12Resource, (void**)&s_d3d.render_targets[i]);
        if (FAILED(hr)) return -1;

        s_d3d.device->lpVtbl->CreateRenderTargetView(
            s_d3d.device, s_d3d.render_targets[i], NULL, rtv_handle);
        rtv_handle.ptr += s_d3d.rtv_descriptor_size;
    }

    /* ---------------------------------------------------------------
     * Depth/stencil buffer
     * 24-bit depth + 8-bit stencil (DXGI_FORMAT_D24_UNORM_S8_UINT).
     * Fallback for draws with no resolvable guest zeta surface; guest targets
     * get persistent resources in the per-surface depth pool.
     * ---------------------------------------------------------------*/
    {
        D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc = {0};
        dsv_heap_desc.NumDescriptors = 1 + 2 * MAX_OFF_DEPTHS;
        dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        hr = s_d3d.device->lpVtbl->CreateDescriptorHeap(
            s_d3d.device, &dsv_heap_desc, &IID_ID3D12DescriptorHeap,
            (void**)&s_d3d.dsv_heap);
        if (FAILED(hr)) {
            printf("[D3D12] DSV heap creation failed\n");
            return -1;
        }
        s_d3d.dsv_inc = s_d3d.device->lpVtbl->GetDescriptorHandleIncrementSize(
            s_d3d.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

        D3D12_HEAP_PROPERTIES heap_props = {0};
        heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC depth_desc = {0};
        depth_desc.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        /* Sized to cover the LARGEST render target, not just the window:
         * D3D12 clips rasterization to the smallest bound attachment, and
         * this DSV is shared by every pass -- with a window-sized depth
         * buffer, draws into larger offscreen RTs (wave's 1920x1080 input
         * image) were silently cropped to the window rect. */
        depth_desc.Width              = 2048;
        depth_desc.Height             = 2048;
        depth_desc.DepthOrArraySize   = 1;
        depth_desc.MipLevels          = 1;
        depth_desc.Format             = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depth_desc.SampleDesc.Count   = 1;
        depth_desc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        depth_desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        D3D12_CLEAR_VALUE depth_clear = {0};
        depth_clear.Format                       = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depth_clear.DepthStencil.Depth           = 1.0f;
        depth_clear.DepthStencil.Stencil         = 0;

        hr = s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &heap_props, D3D12_HEAP_FLAG_NONE,
            &depth_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &depth_clear,
            &IID_ID3D12Resource, (void**)&s_d3d.depth_buffer);
        if (FAILED(hr)) {
            printf("[D3D12] Depth buffer creation failed (0x%08lX)\n", hr);
            return -1;
        }

        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc = {0};
        dsv_desc.Format         = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsv_desc.ViewDimension  = D3D12_DSV_DIMENSION_TEXTURE2D;

        D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle;
        s_d3d.dsv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.dsv_heap, &dsv_handle);
        s_d3d.device->lpVtbl->CreateDepthStencilView(
            s_d3d.device, s_d3d.depth_buffer, &dsv_desc, dsv_handle);

        printf("[D3D12] Depth buffer created (%ux%u D24S8)\n", width, height);
    }

    /* Create command allocators and command list */
    for (u32 i = 0; i < FRAME_COUNT; i++) {
        hr = s_d3d.device->lpVtbl->CreateCommandAllocator(
            s_d3d.device, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void**)&s_d3d.cmd_allocators[i]);
        if (FAILED(hr)) return -1;
    }

    hr = s_d3d.device->lpVtbl->CreateCommandList(
        s_d3d.device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        s_d3d.cmd_allocators[0], NULL,
        &IID_ID3D12GraphicsCommandList, (void**)&s_d3d.cmd_list);
    if (FAILED(hr)) return -1;

    /* Close the command list (it starts in recording state) */
    s_d3d.cmd_list->lpVtbl->Close(s_d3d.cmd_list);

    /* Create fence */
    hr = s_d3d.device->lpVtbl->CreateFence(
        s_d3d.device, 0, D3D12_FENCE_FLAG_NONE,
        &IID_ID3D12Fence, (void**)&s_d3d.fence);
    if (FAILED(hr)) return -1;

    s_d3d.fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    memset(s_d3d.fence_values, 0, sizeof(s_d3d.fence_values));
    memset(s_d3d.submission_fence_values, 0, sizeof(s_d3d.submission_fence_values));
    s_d3d.next_fence_value = s_d3d.last_fence_value = 0;
    s_gpu_work_pending = 0;
    s_batch_slot_acquired = 0;

    /* ---------------------------------------------------------------
     * Create root signature with 16 root constants (one mat4 MVP at b0).
     * Visible only to the vertex shader — pixel shader doesn't need it.
     * ---------------------------------------------------------------*/
    {
        /* param 0: mat4 MVP as 16 root constants at b0 (vertex shader).
         * param 1: one SRV (t0) descriptor table for the atlas texture (pixel).
         * static sampler s0: linear clamp. The plain (untextured) PSO simply
         * never references t0/s0, so it ignores them. */
        D3D12_DESCRIPTOR_RANGE srv_range = {0};
        srv_range.RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srv_range.NumDescriptors     = 1;
        srv_range.BaseShaderRegister = 0;   /* t0 */
        srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        D3D12_ROOT_PARAMETER root_params[2] = {0};
        root_params[0].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        root_params[0].Constants.ShaderRegister = 0;   /* b0 */
        root_params[0].Constants.RegisterSpace  = 0;
        root_params[0].Constants.Num32BitValues = 16;  /* mat4 */
        root_params[0].ShaderVisibility         = D3D12_SHADER_VISIBILITY_VERTEX;
        root_params[1].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[1].DescriptorTable.NumDescriptorRanges = 1;
        root_params[1].DescriptorTable.pDescriptorRanges   = &srv_range;
        root_params[1].ShaderVisibility         = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_STATIC_SAMPLER_DESC samp = {0};
        samp.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samp.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samp.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samp.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samp.ShaderRegister   = 0;   /* s0 */
        samp.MaxLOD           = D3D12_FLOAT32_MAX;
        samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC rs_desc = {0};
        rs_desc.NumParameters     = 2;
        rs_desc.pParameters       = root_params;
        rs_desc.NumStaticSamplers = 1;
        rs_desc.pStaticSamplers   = &samp;
        rs_desc.Flags         = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                              | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS
                              | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
                              | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        ID3DBlob* signature_blob = NULL;
        ID3DBlob* error_blob = NULL;
        hr = D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                          &signature_blob, &error_blob);
        if (FAILED(hr)) {
            printf("[D3D12] Root signature serialization failed: %s\n",
                   error_blob ? (const char*)error_blob->lpVtbl->GetBufferPointer(error_blob) : "?");
            if (error_blob) error_blob->lpVtbl->Release(error_blob);
            return -1;
        }

        hr = s_d3d.device->lpVtbl->CreateRootSignature(
            s_d3d.device, 0,
            signature_blob->lpVtbl->GetBufferPointer(signature_blob),
            signature_blob->lpVtbl->GetBufferSize(signature_blob),
            &IID_ID3D12RootSignature, (void**)&s_d3d.root_signature);
        signature_blob->lpVtbl->Release(signature_blob);
        if (FAILED(hr)) {
            printf("[D3D12] Root signature creation failed\n");
            return -1;
        }
    }

    /* ---------------------------------------------------------------
     * Compile shaders and create PSO
     * ---------------------------------------------------------------*/
    {
        /* Basic vertex-colored shader.
         * The MVP matrix arrives via root constants as 4 vec4 columns
         * (PS3/OpenGL column-major convention). We multiply explicitly so
         * we don't depend on HLSL's matrix packing — matches PS3 semantics
         * `gl_Position = MVP * vec4(pos, 1.0)`. */
        static const char vs_hlsl[] =
            "cbuffer cb0 : register(b0) {\n"
            "    float4 mvp_col0;\n"
            "    float4 mvp_col1;\n"
            "    float4 mvp_col2;\n"
            "    float4 mvp_col3;\n"
            "};\n"
            "struct VSInput  { float3 pos : POSITION; float4 col : COLOR; };\n"
            "struct VSOutput { float4 pos : SV_POSITION; float4 col : COLOR; };\n"
            "VSOutput main(VSInput i) {\n"
            "    VSOutput o;\n"
            "    float4 p = float4(i.pos, 1.0);\n"
            "    o.pos = mvp_col0 * p.x + mvp_col1 * p.y + mvp_col2 * p.z + mvp_col3 * p.w;\n"
            "    o.col = i.col;\n"
            "    return o;\n"
            "}\n";
        static const char ps_hlsl[] =
            "struct PSInput { float4 pos : SV_POSITION; float4 col : COLOR; };\n"
            "float4 main(PSInput i) : SV_TARGET { return i.col; }\n";

        ID3DBlob* vs_blob = NULL;
        ID3DBlob* ps_blob = NULL;
        ID3DBlob* err = NULL;

        hr = d3d12_compile_cached(vs_hlsl, sizeof(vs_hlsl) - 1, "vs_basic", NULL, NULL,
                        "main", "vs_5_0", 0, 0, &vs_blob, &err);
        if (FAILED(hr)) {
            printf("[D3D12] VS compile failed: %s\n",
                   err ? (const char*)err->lpVtbl->GetBufferPointer(err) : "unknown");
            if (err) err->lpVtbl->Release(err);
        }

        hr = d3d12_compile_cached(ps_hlsl, sizeof(ps_hlsl) - 1, "ps_basic", NULL, NULL,
                        "main", "ps_5_0", 0, 0, &ps_blob, &err);
        if (FAILED(hr)) {
            printf("[D3D12] PS compile failed: %s\n",
                   err ? (const char*)err->lpVtbl->GetBufferPointer(err) : "unknown");
            if (err) err->lpVtbl->Release(err);
        }

        if (vs_blob && ps_blob) {
            D3D12_INPUT_ELEMENT_DESC input_layout[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
                 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12,
                 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 28,
                 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            };

            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {0};
            pso_desc.pRootSignature = s_d3d.root_signature;
            pso_desc.VS.pShaderBytecode = vs_blob->lpVtbl->GetBufferPointer(vs_blob);
            pso_desc.VS.BytecodeLength = vs_blob->lpVtbl->GetBufferSize(vs_blob);
            pso_desc.PS.pShaderBytecode = ps_blob->lpVtbl->GetBufferPointer(ps_blob);
            pso_desc.PS.BytecodeLength = ps_blob->lpVtbl->GetBufferSize(ps_blob);
            pso_desc.InputLayout.pInputElementDescs = input_layout;
            pso_desc.InputLayout.NumElements = 3;
            pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL;
            pso_desc.SampleMask = UINT_MAX;
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso_desc.NumRenderTargets = 1;
            pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            pso_desc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
            /* Depth test enabled, write enabled, LESS func.
             * Games with depth_test_enable=0 in RSX state can still render —
             * LESS just means new-z must be less than existing — but future
             * work should mirror RSX depth state into a PSO cache. */
            pso_desc.DepthStencilState.DepthEnable    = TRUE;
            pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            pso_desc.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_LESS_EQUAL;
            pso_desc.DepthStencilState.StencilEnable  = FALSE;
            pso_desc.SampleDesc.Count = 1;

            hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pso_desc,
                &IID_ID3D12PipelineState, (void**)&s_d3d.pipeline_state);
            if (SUCCEEDED(hr)) {
                s_d3d.pipeline_ready = 1;
                printf("[D3D12] Pipeline state created (triangle class)\n");
            } else {
                printf("[D3D12] PSO TRIANGLE creation failed (0x%08lX)\n", hr);
            }

            /* Line-class PSO — same shader, LINE topology type. */
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pso_desc,
                &IID_ID3D12PipelineState, (void**)&s_d3d.pipeline_state_lines);
            if (SUCCEEDED(hr)) printf("[D3D12] Pipeline state created (line class)\n");
            else printf("[D3D12] PSO LINE creation failed (0x%08lX)\n", hr);

            /* Point-class PSO. */
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pso_desc,
                &IID_ID3D12PipelineState, (void**)&s_d3d.pipeline_state_points);
            if (SUCCEEDED(hr)) printf("[D3D12] Pipeline state created (point class)\n");
            else printf("[D3D12] PSO POINT creation failed (0x%08lX)\n", hr);

            vs_blob->lpVtbl->Release(vs_blob);
            ps_blob->lpVtbl->Release(ps_blob);
        }

        /* -----------------------------------------------------------------
         * Textured triangle PSO (dbgfont / 2D atlas quads). Same MVP VS but
         * carrying UV; the PS samples the single-channel atlas as coverage
         * and modulates the vertex color, with straight alpha blending so
         * glyph edges composite over the framebuffer. Depth off (2D overlay).
         * ----------------------------------------------------------------*/
        {
            static const char vs_tex[] =
                "cbuffer cb0 : register(b0){float4 c0;float4 c1;float4 c2;float4 c3;};\n"
                "struct VSIn { float3 pos:POSITION; float4 col:COLOR; float2 uv:TEXCOORD; };\n"
                "struct VSOut{ float4 pos:SV_POSITION; float4 col:COLOR; float2 uv:TEXCOORD; };\n"
                "VSOut main(VSIn i){ VSOut o; float4 p=float4(i.pos,1.0);\n"
                "  o.pos=c0*p.x+c1*p.y+c2*p.z+c3*p.w; o.col=i.col; o.uv=i.uv; return o; }\n";
            static const char ps_tex[] =
                "Texture2D tex : register(t0);\n"
                "SamplerState smp : register(s0);\n"
                "struct PSIn{ float4 pos:SV_POSITION; float4 col:COLOR; float2 uv:TEXCOORD; };\n"
                "float4 main(PSIn i):SV_TARGET{\n"
                /* dbgfont texcoords are compressed ~10x (U) / ~8x (V) vs. the
                 * atlas; scale to recover glyph cells. Exact per-glyph offset
                 * still being calibrated. */
                "  float2 uv2 = float2(i.uv.x*10.0, i.uv.y*8.0 + 0.59);\n"
                "  float cov = tex.Sample(smp, uv2).r;\n"
                "  return float4(i.col.rgb, i.col.a * cov); }\n";

            ID3DBlob *vtb = NULL, *ptb = NULL, *e2 = NULL;
            hr = d3d12_compile_cached(vs_tex, sizeof(vs_tex) - 1, "vs_tex", NULL, NULL, "main", "vs_5_0", 0, 0, &vtb, &e2);
            if (FAILED(hr)) printf("[D3D12] VS(tex) compile failed: %s\n", e2 ? (const char*)e2->lpVtbl->GetBufferPointer(e2) : "?");
            hr = d3d12_compile_cached(ps_tex, sizeof(ps_tex) - 1, "ps_tex", NULL, NULL, "main", "ps_5_0", 0, 0, &ptb, &e2);
            if (FAILED(hr)) printf("[D3D12] PS(tex) compile failed: %s\n", e2 ? (const char*)e2->lpVtbl->GetBufferPointer(e2) : "?");

            if (vtb && ptb) {
                D3D12_INPUT_ELEMENT_DESC il[] = {
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };
                D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
                pd.pRootSignature = s_d3d.root_signature;
                pd.VS.pShaderBytecode = vtb->lpVtbl->GetBufferPointer(vtb);
                pd.VS.BytecodeLength  = vtb->lpVtbl->GetBufferSize(vtb);
                pd.PS.pShaderBytecode = ptb->lpVtbl->GetBufferPointer(ptb);
                pd.PS.BytecodeLength  = ptb->lpVtbl->GetBufferSize(ptb);
                pd.InputLayout.pInputElementDescs = il;
                pd.InputLayout.NumElements = 3;
                pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                pd.BlendState.RenderTarget[0].BlendEnable    = TRUE;
                pd.BlendState.RenderTarget[0].SrcBlend       = D3D12_BLEND_SRC_ALPHA;
                pd.BlendState.RenderTarget[0].DestBlend      = D3D12_BLEND_INV_SRC_ALPHA;
                pd.BlendState.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
                pd.BlendState.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
                pd.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                pd.BlendState.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
                pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
                pd.SampleMask = UINT_MAX;
                pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                pd.NumRenderTargets = 1;
                pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
                pd.DepthStencilState.DepthEnable = FALSE;
                pd.DepthStencilState.StencilEnable = FALSE;
                pd.SampleDesc.Count = 1;
                hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                    s_d3d.device, &pd, &IID_ID3D12PipelineState, (void**)&s_d3d.pipeline_state_tex);
                if (SUCCEEDED(hr)) printf("[D3D12] Pipeline state created (textured class)\n");
                else printf("[D3D12] PSO TEX creation failed (0x%08lX)\n", hr);
            }
            if (vtb) vtb->lpVtbl->Release(vtb);
            if (ptb) ptb->lpVtbl->Release(ptb);
        }

        /* SRV descriptor heap (shader-visible). VP texture uploads use the
         * cache slots, each in-flight slot has an atlas descriptor, offscreen
         * render targets are copied into per-draw windows, and every slot starts
         * as a null SRV. */
        {
            D3D12_DESCRIPTOR_HEAP_DESC hd = {0};
            hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            hd.NumDescriptors = SRV_HEAP_SIZE;
            hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            hr = s_d3d.device->lpVtbl->CreateDescriptorHeap(
                s_d3d.device, &hd, &IID_ID3D12DescriptorHeap, (void**)&s_d3d.srv_heap);
            if (FAILED(hr)) printf("[D3D12] SRV heap creation failed (0x%08lX)\n", hr);
            else {
                s_d3d.srv_inc = s_d3d.device->lpVtbl->GetDescriptorHandleIncrementSize(
                    s_d3d.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                D3D12_SHADER_RESOURCE_VIEW_DESC nv = {0};
                nv.Format = DXGI_FORMAT_R8_UNORM;
                nv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                nv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                nv.Texture2D.MipLevels = 1;
                D3D12_CPU_DESCRIPTOR_HANDLE hh;
                s_d3d.srv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &hh);
                for (int _i = 0; _i < SRV_HEAP_SIZE; _i++) {
                    s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, NULL, &nv, hh);
                    hh.ptr += s_d3d.srv_inc;
                }
            }
        }

        /* Guest pixel samplers vary by RSX texture unit. Keep one sixteen-
         * descriptor table per distinct state set in each in-flight slot. */
        {
            D3D12_DESCRIPTOR_HEAP_DESC hd = {0};
            hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            hd.NumDescriptors = RSX_SAMPLER_TABLE_COUNT * RSX_SAMPLER_TABLE_SIZE;
            hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            for (u32 i = 0; i < FRAME_COUNT; i++) {
                hr = s_d3d.device->lpVtbl->CreateDescriptorHeap(
                    s_d3d.device, &hd, &IID_ID3D12DescriptorHeap,
                    (void**)&s_d3d.sampler_heap[i]);
                if (FAILED(hr)) {
                    printf("[D3D12] sampler heap %u creation failed (0x%08lX)\n", i, hr);
                    break;
                }
            }
            s_d3d.sampler_inc = s_d3d.device->lpVtbl->GetDescriptorHandleIncrementSize(
                s_d3d.device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        }

        /* RTV heap for offscreen render targets (CPU-visible only). */
        {
            D3D12_DESCRIPTOR_HEAP_DESC hd = {0};
            hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            hd.NumDescriptors = MAX_OFF_RTS;
            hr = s_d3d.device->lpVtbl->CreateDescriptorHeap(
                s_d3d.device, &hd, &IID_ID3D12DescriptorHeap, (void**)&s_d3d.rt_rtv_heap);
            if (FAILED(hr)) printf("[D3D12] offscreen RTV heap creation failed (0x%08lX)\n", hr);
        }
    }

    /* ---------------------------------------------------------------
     * Create dynamic vertex buffer (upload heap, 4MB)
     * ---------------------------------------------------------------*/
    {
        D3D12_HEAP_PROPERTIES heap_props = {0};
        heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC buf_desc = {0};
        buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buf_desc.Width = (u64)FRAME_COUNT * MAX_VERTICES * VERTEX_STRIDE;
        buf_desc.Height = 1;
        buf_desc.DepthOrArraySize = 1;
        buf_desc.MipLevels = 1;
        buf_desc.SampleDesc.Count = 1;
        buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        hr = s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &heap_props, D3D12_HEAP_FLAG_NONE,
            &buf_desc, D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
            &IID_ID3D12Resource, (void**)&s_d3d.vertex_buffer);
        if (SUCCEEDED(hr)) {
            D3D12_RANGE read_range = {0, 0};
            s_d3d.vertex_buffer->lpVtbl->Map(
                s_d3d.vertex_buffer, 0, &read_range, &s_d3d.vb_mapped);
            s_d3d.vb_view.BufferLocation =
                s_d3d.vertex_buffer->lpVtbl->GetGPUVirtualAddress(s_d3d.vertex_buffer);
            s_d3d.vb_view.StrideInBytes = VERTEX_STRIDE;
            s_d3d.vb_view.SizeInBytes = MAX_VERTICES * VERTEX_STRIDE;
            printf("[D3D12] Vertex buffer created (%u KB x %u submission slots)\n",
                   (MAX_VERTICES * VERTEX_STRIDE) / 1024, FRAME_COUNT);
        }
    }

    /* ---------------------------------------------------------------
     * Real vertex-program path resources: root signature (CBV b0 for the
     * vp_c[] bank + SRV table + dynamic pixel-sampler table), a raw-float4
     * vertex buffer, and the constant-bank buffer. The PSO itself is built lazily once the
     * game uploads its VP microcode (render_frame).
     * ---------------------------------------------------------------*/
    {
        /* The fragment bank occupies t0-t15; vertex TXL uses t16-t19. A single
         * table exposes both banks to their respective shader stages. */
        D3D12_DESCRIPTOR_RANGE srv_range[2] = {{0}};
        srv_range[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srv_range[0].NumDescriptors = RSX_TEXTURE_UNITS;
        srv_range[0].BaseShaderRegister = 0;
        srv_range[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        srv_range[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srv_range[1].NumDescriptors = RSX_MAX_VERTEX_TEXTURES;
        srv_range[1].BaseShaderRegister = RSX_TEXTURE_UNITS;
        srv_range[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        D3D12_DESCRIPTOR_RANGE sampler_range = {0};
        sampler_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        sampler_range.NumDescriptors = RSX_SAMPLER_TABLE_SIZE;
        sampler_range.BaseShaderRegister = 0;
        sampler_range.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER rp[5] = {0};
        rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;   /* b0 = vp_c bank */
        rp[0].Descriptor.ShaderRegister = 0;
        rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rp[1].DescriptorTable.NumDescriptorRanges = 2;
        rp[1].DescriptorTable.pDescriptorRanges = srv_range;
        rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;   /* b1 = FP texscale */
        rp[2].Descriptor.ShaderRegister = 1;
        rp[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rp[3].DescriptorTable.NumDescriptorRanges = 1;
        rp[3].DescriptorTable.pDescriptorRanges = &sampler_range;
        rp[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        rp[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[4].Descriptor.ShaderRegister = 20;
        rp[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

        D3D12_STATIC_SAMPLER_DESC samp[RSX_MAX_VERTEX_TEXTURES] = {0};
        /* NV40 vertex TXL uses its own s0-s3 sampler registers. Static sampler
         * registers are vertex-only; pixel samplers are supplied by root table 3. */
        for (int _s = 0; _s < RSX_MAX_VERTEX_TEXTURES; _s++) {
            D3D12_STATIC_SAMPLER_DESC* vs = &samp[_s];
            vs->Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            vs->AddressU = vs->AddressV = vs->AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            vs->ShaderRegister = (UINT)_s;
            vs->MaxLOD = D3D12_FLOAT32_MAX;
            vs->ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        }

        D3D12_ROOT_SIGNATURE_DESC rd = {0};
        rd.NumParameters = 5; rd.pParameters = rp;
        rd.NumStaticSamplers = RSX_MAX_VERTEX_TEXTURES;
        rd.pStaticSamplers = samp;
        rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ID3DBlob* sig = NULL; ID3DBlob* err = NULL;
        hr = D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
        if (SUCCEEDED(hr)) {
            s_d3d.device->lpVtbl->CreateRootSignature(s_d3d.device, 0,
                sig->lpVtbl->GetBufferPointer(sig), sig->lpVtbl->GetBufferSize(sig),
                &IID_ID3D12RootSignature, (void**)&s_d3d.vp_root_sig);
            sig->lpVtbl->Release(sig);
        } else if (err) { printf("[D3D12] VP root sig: %s\n", (const char*)err->lpVtbl->GetBufferPointer(err)); err->lpVtbl->Release(err); }

        /* raw float4 vertex buffer + constant bank (both UPLOAD, persistently mapped) */
        D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd = {0};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_RANGE nr = {0,0};

        bd.Width = (u64)MAX_VERTICES * 256 * 2;  /* generic VP vertex: 16 float4 slots,
                                          * DOUBLE-buffered by frame parity so a
                                          * new frame's upload never overwrites
                                          * vertices the in-flight GPU frame is
                                          * still reading (was: torn/missing
                                          * triangles mixing stale+new verts) */
        /* Failure here leaves vp_vb_mapped NULL and every VP draw silently
         * declines to upload -- the scene just does not render, with no error
         * anywhere. Report the size too: this buffer is
         * MAX_VERTICES * 256 * 2 bytes and grows fast. */
        fprintf(stderr, "[VPVB] per-frame vertex buffer: %llu MB (%u verts, x2 parity)%c",
                (unsigned long long)(bd.Width >> 20), (unsigned)MAX_VERTICES, 10);
        if (SUCCEEDED(s_d3d.device->lpVtbl->CreateCommittedResource(s_d3d.device, &hp,
                D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                &IID_ID3D12Resource, (void**)&s_d3d.vp_vb)))
            s_d3d.vp_vb->lpVtbl->Map(s_d3d.vp_vb, 0, &nr, &s_d3d.vp_vb_mapped);
        else
            fprintf(stderr, "[VPVB] ALLOCATION FAILED -- no VP geometry will render%c", 10);

        bd.Width = (u64)VP_CB_STRIDE * MAX_DRAWS * 2;   /* per-draw constant slots, x2 parity */
        /* A failure here is silent and catastrophic: vp_record_cb returns early,
         * every draw loses its constants, and the whole scene transforms by
         * zeros -- indistinguishable from a broken vertex program. Say so. */
        fprintf(stderr, "[VPCB] per-draw constant buffer: %llu MB (%d draws)%c",
                (unsigned long long)(bd.Width >> 20), MAX_DRAWS, 10);
        if (SUCCEEDED(s_d3d.device->lpVtbl->CreateCommittedResource(s_d3d.device, &hp,
                D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                &IID_ID3D12Resource, (void**)&s_d3d.vp_cb)))
            s_d3d.vp_cb->lpVtbl->Map(s_d3d.vp_cb, 0, &nr, &s_d3d.vp_cb_mapped);

        bd.Width = (u64)VP_FPCB_STRIDE * MAX_DRAWS * 2;  /* per-draw FP slots (b1), x2 parity */
        if (SUCCEEDED(s_d3d.device->lpVtbl->CreateCommittedResource(s_d3d.device, &hp,
                D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                &IID_ID3D12Resource, (void**)&s_d3d.vp_fpcb)))
            s_d3d.vp_fpcb->lpVtbl->Map(s_d3d.vp_fpcb, 0, &nr, &s_d3d.vp_fpcb_mapped);

        printf("[D3D12] VP path resources: rootsig=%p vb=%p cb=%p\n",
               (void*)s_d3d.vp_root_sig, (void*)s_d3d.vp_vb, (void*)s_d3d.vp_cb);
    }

    printf("[D3D12] Initialization complete (%ux%u, %u buffers, pipeline=%s)\n",
           width, height, FRAME_COUNT,
           s_d3d.pipeline_ready ? "ready" : "NOT ready");
    return 0;
}

/* ---------------------------------------------------------------------------
 * Frame sync helpers
 * -----------------------------------------------------------------------*/

/* Queue submissions are serialized on the render thread. A completed drain
 * remains valid until another command list is submitted. */

static int wait_for_fence_value(u64 target, const char* context)
{
    if (!target || s_d3d.fence->lpVtbl->GetCompletedValue(s_d3d.fence) >= target)
        return 1;
    HRESULT hr = s_d3d.fence->lpVtbl->SetEventOnCompletion(
        s_d3d.fence, target, s_d3d.fence_event);
    if (FAILED(hr)) {
        printf("[D3D12] %s SetEventOnCompletion failed (0x%08lX)\n", context, hr);
        return 0;
    }
    for (int tries = 0; tries < 3; tries++) {
        if (WaitForSingleObject(s_d3d.fence_event, 2000) == WAIT_OBJECT_0)
            return 1;
        HRESULT rr = s_d3d.device->lpVtbl->GetDeviceRemovedReason(s_d3d.device);
        printf("[D3D12] %s STUCK %ds: want %llu got %llu removed=0x%08lX\n",
               context, 2 * (tries + 1), (unsigned long long)target,
               (unsigned long long)s_d3d.fence->lpVtbl->GetCompletedValue(s_d3d.fence),
               (long)rr);
        fflush(stdout);
        if (rr != 0) break;
    }
    return s_d3d.fence->lpVtbl->GetCompletedValue(s_d3d.fence) >= target;
}

#include "rsx_d3d12_retire.h"
static D3D12RetiredResource* s_retired_resources;
static void d3d12_collect_retired(int force) {
 rsx_d3d12_retire_collect(&s_retired_resources,force ? ~(u64)0 : s_d3d.fence->lpVtbl->GetCompletedValue(s_d3d.fence));
}
static int d3d12_retire_resource(ID3D12Resource** resource) {
 return rsx_d3d12_retire_add(&s_retired_resources,resource,s_d3d.next_fence_value+1u);
}

static void d3d12_signal_submission(u32 slot)
{
    const u64 value = ++s_d3d.next_fence_value;
    HRESULT hr = s_d3d.cmd_queue->lpVtbl->Signal(s_d3d.cmd_queue, s_d3d.fence, value);
    if (FAILED(hr)) {
        printf("[D3D12] queue fence signal failed (0x%08lX)\n", hr);
        return;
    }
    s_d3d.last_fence_value = value;
    s_d3d.fence_values[s_d3d.frame_index] = value;
    if (slot < FRAME_COUNT)
        s_d3d.submission_fence_values[slot] = value;
    s_gpu_work_pending = 1;
    d3d12_collect_retired(0);
}

static void wait_for_gpu_impl(void)
{
    if (!s_gpu_work_pending) return;
    if (wait_for_fence_value(s_d3d.last_fence_value, "wait_for_gpu"))
        s_gpu_work_pending = 0;
}

static void wait_for_gpu(void)
{
    if (!perf_on()) { wait_for_gpu_impl(); return; }
    double start = perf_now();
    wait_for_gpu_impl();
    s_perf_fence_wait += perf_now() - start;
    s_perf_fence_calls++;
}

static int d3d12_acquire_submission_slot(void)
{
    if (s_batch_slot_acquired) return 1;
    const u32 slot = (u32)s_d3d.vp_parity;
    const u64 target = s_d3d.submission_fence_values[slot];
    if (!wait_for_fence_value(target, "submission-slot")) return 0;
    if (target >= s_d3d.last_fence_value) s_gpu_work_pending = 0;
    s_vp_cb_cache.valid = 0;
    rsx_raw_upload_reset(&s_raw_upload_cache);
    s_batch_slot_acquired = 1;
    return 1;
}

static u8* d3d12_legacy_vertex_slot_base(void)
{
    return (u8*)s_d3d.vb_mapped +
        (u64)(u32)s_d3d.vp_parity * MAX_VERTICES * VERTEX_STRIDE;
}

static void move_to_next_frame(int presented)
{
    if (!presented) return;
    s_d3d.frame_index = s_d3d.swap_chain->lpVtbl->GetCurrentBackBufferIndex(s_d3d.swap_chain);

    const u64 target = s_d3d.fence_values[s_d3d.frame_index];
    if (wait_for_fence_value(target, "backbuffer-retire") &&
        target >= s_d3d.last_fence_value)
        s_gpu_work_pending = 0;
}

/* ---------------------------------------------------------------------------
 * Render a frame (clear + present)
 * -----------------------------------------------------------------------*/

/* Write the mapped readback buffer (R8G8B8A8, row pitch = readback_pitch) out
 * as a 24-bit bottom-up BMP. Debug-only. */
/* Name prefix for the next readback dump: lets the same writer emit the
 * backbuffer and the screen-copy texture under different filenames. */
static const char* s_dump_name = NULL;

static void dump_backbuffer_bmp(void)
{
    if (!s_d3d.readback_buf) return;
    void* mapped = NULL;
    D3D12_RANGE rr = {0, (SIZE_T)s_d3d.readback_pitch * s_d3d.height};
    if (FAILED(s_d3d.readback_buf->lpVtbl->Map(s_d3d.readback_buf, 0, &rr, &mapped)) || !mapped)
        return;

    static int idx = 0;
    char path[512];
    const char* dir = getenv("CELLMARK_DUMP_DIR");   /* default: current dir */
    snprintf(path, sizeof(path), "%s%s%s_%03d.bmp",
             dir ? dir : "", dir ? "/" : "", s_dump_name ? s_dump_name : "frame", idx++);
    FILE* f = fopen(path, "wb");
    if (f) {
        u32 w = s_d3d.width, h = s_d3d.height;
        u32 padded = (w * 3 + 3) & ~3u;
        u32 imgsz  = padded * h;
        u32 filesz = 54 + imgsz;
        unsigned char hdr[54] = {0};
        hdr[0] = 'B'; hdr[1] = 'M';
        hdr[2]=filesz&0xFF; hdr[3]=(filesz>>8)&0xFF; hdr[4]=(filesz>>16)&0xFF; hdr[5]=(filesz>>24)&0xFF;
        hdr[10]=54; hdr[14]=40;
        hdr[18]=w&0xFF; hdr[19]=(w>>8)&0xFF; hdr[20]=(w>>16)&0xFF; hdr[21]=(w>>24)&0xFF;
        hdr[22]=h&0xFF; hdr[23]=(h>>8)&0xFF; hdr[24]=(h>>16)&0xFF; hdr[25]=(h>>24)&0xFF;
        hdr[26]=1; hdr[28]=24;
        hdr[34]=imgsz&0xFF; hdr[35]=(imgsz>>8)&0xFF; hdr[36]=(imgsz>>16)&0xFF; hdr[37]=(imgsz>>24)&0xFF;
        fwrite(hdr, 1, 54, f);
        unsigned char* row = (unsigned char*)malloc(padded);
        if (row) {
            memset(row, 0, padded);
            for (int y = (int)h - 1; y >= 0; y--) {   /* BMP is bottom-up */
                unsigned char* src = (unsigned char*)mapped + (u32)y * s_d3d.readback_pitch;
                for (u32 x = 0; x < w; x++) {
                    row[x*3+0] = src[x*4+2]; /* B */
                    row[x*3+1] = src[x*4+1]; /* G */
                    row[x*3+2] = src[x*4+0]; /* R */
                }
                fwrite(row, 1, padded, f);
            }
            free(row);
        }
        fclose(f);
        printf("[D3D12] dumped %s (%ux%u)\n", path, s_d3d.width, s_d3d.height);
    }
    D3D12_RANGE wr = {0, 0};
    s_d3d.readback_buf->lpVtbl->Unmap(s_d3d.readback_buf, 0, &wr);
}

/* ---------------------------------------------------------------------------
 * RSX frame capture ("rsxcap") -- a purpose-built GPU debugger. RSX_CAP=start
 * [:count[:stride]] snapshots whole frames to <RSX_CAP_DIR>/frame_NN/: a text
 * manifest of every op (in submission order, with full surface/shader/texture
 * state) plus a BMP of the backbuffer and every offscreen colour RT used that
 * frame. One frame, captured atomically -- no cross-frame guessing.
 * -----------------------------------------------------------------------*/

/* Shared 24-bit BMP writer for readbacks. R8G8B8A8_UNORM straight; the half/
 * float RT formats are |v|-tonemapped so signed/HDR data is visible. */
static void cap_write_bmp(const char* path, const void* mapped, u32 w, u32 h,
                          u32 pitch, u32 dxgi)
{
    FILE* f = fopen(path, "wb");
    if (!f) return;
    u32 rowb = (w * 3 + 3) & ~3u, datasz = rowb * h;
    u8 hdr[54] = {0};
    hdr[0] = 'B'; hdr[1] = 'M';
    *(u32*)(hdr+2) = 54 + datasz; *(u32*)(hdr+10) = 54; *(u32*)(hdr+14) = 40;
    *(int*)(hdr+18) = (int)w; *(int*)(hdr+22) = (int)h;
    *(u16*)(hdr+26) = 1; *(u16*)(hdr+28) = 24; *(u32*)(hdr+34) = datasz;
    fwrite(hdr, 1, 54, f);
    u8* line = (u8*)malloc(rowb);
    if (line) for (int y = (int)h - 1; y >= 0; y--) {   /* BMP is bottom-up */
        const u8* srow = (const u8*)mapped + (u64)y * pitch;
        memset(line, 0, rowb);
        for (u32 x = 0; x < w; x++) {
            float rv, gv, bv;
            if (dxgi == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                const u16* hp = (const u16*)(srow + (u64)x * 8);
                float v[3];
                for (int c = 0; c < 3; c++) {
                    u16 hv = hp[c]; u32 s = (hv>>15)&1, e = (hv>>10)&0x1F, m = hv&0x3FF; float fv;
                    if (e == 0) fv = (float)m / 16777216.0f;
                    else { u32 fb = (s<<31)|((e-15+127)<<23)|(m<<13); memcpy(&fv,&fb,4); }
                    v[c] = fv;
                }
                rv = v[0]; gv = v[1]; bv = v[2];
            } else if (dxgi == DXGI_FORMAT_R32G32B32A32_FLOAT) {
                const float* fp = (const float*)(srow + (u64)x * 16);
                rv = fp[0]; gv = fp[1]; bv = fp[2];
            } else {
                const u8* p = srow + (u64)x * 4;
                rv = p[0]/255.0f; gv = p[1]/255.0f; bv = p[2]/255.0f;
            }
            float ar = rv<0?-rv:rv, ag = gv<0?-gv:gv, ab = bv<0?-bv:bv;
            if (ar>1) ar=1; if (ag>1) ag=1; if (ab>1) ab=1;
            line[x*3+0] = (u8)(ab*255.0f);
            line[x*3+1] = (u8)(ag*255.0f);
            line[x*3+2] = (u8)(ar*255.0f);
        }
        fwrite(line, 1, rowb, f);
    }
    free(line);
    fclose(f);
}

/* Copy one GPU resource into a fresh readback buffer and write it as a BMP.
 * Self-contained: waits, resets the frame command list, copies, executes,
 * waits, maps. `prior` is the resource's current tracked state (restored after
 * the copy so the caller's state tracking stays valid). */
static void cap_readback_write(ID3D12Resource* res, u32 w, u32 h, u32 dxgi,
                               D3D12_RESOURCE_STATES prior, u32 fi, const char* path)
{
    if (!res || !w || !h) return;
    u32 bpp = (dxgi == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 8 :
              (dxgi == DXGI_FORMAT_R32G32B32A32_FLOAT) ? 16 : 4;
    u32 pitch = (w * bpp + 255) & ~255u;

    ID3D12Resource* rb = NULL;
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {0};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = (u64)pitch * h; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COPY_DEST, NULL,
            &IID_ID3D12Resource, (void**)&rb)) || !rb)
        return;

    wait_for_gpu();
    s_d3d.cmd_allocators[fi]->lpVtbl->Reset(s_d3d.cmd_allocators[fi]);
    s_d3d.cmd_list->lpVtbl->Reset(s_d3d.cmd_list, s_d3d.cmd_allocators[fi], NULL);

    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    if (prior != D3D12_RESOURCE_STATE_COPY_SOURCE) {
        b.Transition.StateBefore = prior;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    }

    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = (DXGI_FORMAT)dxgi;
    dst.PlacedFootprint.Footprint.Width = w; dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = pitch;
    src.pResource = res; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dst, 0, 0, 0, &src, NULL);
        /* TEXUP_DBG=1: texture uploads per second. A cache that misses every
         * draw re-uploads megabytes per frame, which is the kind of GPU work
         * that trips a 2-second TDR while looking like nothing in a draw count. */
        { static int td = -1; if (td < 0) td = getenv("TEXUP_DBG") ? 1 : 0;
          if (td) { static long long n = 0, bytes = 0; static clock_t t0 = 0;
              n++; bytes += (long long)pitch * h;
              const clock_t now = clock();
              if (!t0) t0 = now;
              if (now - t0 >= CLOCKS_PER_SEC) {
                  fprintf(stderr, "[TEXUP] %lld uploads/s, %.1f MB/s\n",
                          n, bytes / 1048576.0);
                  n = 0; bytes = 0; t0 = now; } } }

    if (prior != D3D12_RESOURCE_STATE_COPY_SOURCE) {
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter  = prior;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    }

    s_d3d.cmd_list->lpVtbl->Close(s_d3d.cmd_list);
    ID3D12CommandList* cl[] = { (ID3D12CommandList*)s_d3d.cmd_list };
    s_d3d.cmd_queue->lpVtbl->ExecuteCommandLists(s_d3d.cmd_queue, 1, cl);
    d3d12_signal_submission(FRAME_COUNT);
    wait_for_gpu();

    void* mp = NULL; D3D12_RANGE rr = {0, (SIZE_T)pitch * h};
    if (SUCCEEDED(rb->lpVtbl->Map(rb, 0, &rr, &mp)) && mp) {
        cap_write_bmp(path, mp, w, h, pitch, dxgi);
        D3D12_RANGE wr = {0, 0};
        rb->lpVtbl->Unmap(rb, 0, &wr);
    }
    rb->lpVtbl->Release(rb);
}

static void rsx_capture_frame(u32 fi, u32 ndraws, u32 capidx)
{
    char dir[300]; const char* base = getenv("RSX_CAP_DIR");
    if (!base || !*base) base = "rsxcap";
    CreateDirectoryA(base, NULL);
    snprintf(dir, sizeof dir, "%s/frame_%02u", base, capidx);
    CreateDirectoryA(dir, NULL);
    char path[512];

    /* Keep the shaders and complete per-draw constants with the image. A
     * successful compilation alone cannot show whether a glyph samples the
     * intended coordinates or uses the intended transform. */
    for (int i = 0; i < s_d3d.vp_vs_n; i++) {
        for (u32 variant = 0; variant < 2; ++variant) {
        ID3DBlob* blob = variant ? s_d3d.vp_vs[i].raw_vs : s_d3d.vp_vs[i].vs;
        ID3DBlob* assembly = NULL;
        if (blob && SUCCEEDED(D3DDisassemble(blob->lpVtbl->GetBufferPointer(blob),
                blob->lpVtbl->GetBufferSize(blob), 0, NULL, &assembly))) {
            snprintf(path, sizeof path, "%s/vs_%03d%s.asm", dir, i, variant ? "_raw" : "");
            FILE* out = fopen(path, "wb");
            if (out) {
                fwrite(assembly->lpVtbl->GetBufferPointer(assembly), 1,
                       assembly->lpVtbl->GetBufferSize(assembly), out);
                fclose(out);
            }
            assembly->lpVtbl->Release(assembly);
        }
        }
    }
    for (int i = 0; i < s_d3d.vp_fp_blob_n; i++) {
        const char* source = s_d3d.vp_fp_blob[i].source;
        if (!source) continue;
        snprintf(path, sizeof path, "%s/ps_%03d.hlsl", dir, i);
        FILE* out = fopen(path, "wb");
        if (out) { fwrite(source, 1, strlen(source), out); fclose(out); }
    }
    for (u32 d = 0; d < ndraws && d < MAX_DRAWS; d++) {
        const D3D12DrawRecord* r = &s_d3d.draws[d];
        if (!r->is_vp || r->is_clear || r->cb_slot >= MAX_DRAWS) continue;
        const u64 cb = (u64)(s_d3d.vp_parity ^ 1) * MAX_DRAWS + r->vp_cb_slot;
        snprintf(path, sizeof path, "%s/cb_%03u.bin", dir, d);
        FILE* out = fopen(path, "wb");
        if (out) {
            fwrite((const char*)s_d3d.vp_cb_mapped + cb * VP_CB_STRIDE,
                   1, VP_CB_STRIDE, out);
            fclose(out);
        }
    }

    /* Manifest: every op in submission order with full state. */
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    FILE* mf = fopen(path, "w");
    if (mf) {
        fprintf(mf, "frame_count=%llu draws=%u backbuffer=%ux%u\n",
                (unsigned long long)s_d3d.frame_count, ndraws, s_d3d.width, s_d3d.height);
        for (int i = 0; i < MAX_OFF_RTS; i++)
            if (s_d3d.off_rt[i].res)
                fprintf(mf, "offrt[%d] off=0x%08X %ux%u dxgi=%u\n", i, s_d3d.off_rt[i].off,
                        s_d3d.off_rt[i].w, s_d3d.off_rt[i].h, s_d3d.off_rt[i].dxgi);
        fprintf(mf, "--- ops ---\n");
        for (u32 d = 0; d < ndraws && d < MAX_DRAWS; d++) {
            const D3D12DrawRecord* r = &s_d3d.draws[d];
            if (r->is_clear)
                fprintf(mf, "op%03u CLEAR rt=0x%08X zeta=0x%08X zpitch=0x%X mrt=0x%X/0x%X/0x%X flags=0x%02X depth=%.6f stencil=%u cc=(%.3f,%.3f,%.3f,%.3f)\n",
                        d, r->rt_off, r->zeta_off, r->zeta_pitch,
                        r->rt_mrt[0], r->rt_mrt[1], r->rt_mrt[2],
                        r->clear_flags, r->clear_depth, r->clear_stencil,
                        r->cc[0], r->cc[1], r->cc[2], r->cc[3]);
            else {
                fprintf(mf, "op%03u DRAW  rt=0x%08X zeta=0x%08X zpitch=0x%X mrt=0x%X/0x%X/0x%X vp=%u,%u,%ux%u fp=0x%08X vs=%d exp32=%d "
                            "n=%u cmask=%X blend=%d bk=0x%X t0=0x%08X t1=0x%08X t2=0x%08X t3=0x%08X\n",
                        d, r->rt_off, r->zeta_off, r->zeta_pitch,
                        r->rt_mrt[0], r->rt_mrt[1], r->rt_mrt[2],
                        r->vp_x, r->vp_y, r->vp_w, r->vp_h,
                        r->fp_addr, r->vs_idx, r->fp_exp32, r->vertex_count, r->cmask, r->blend, r->blend_key,
                        r->tex[0].raw, r->tex[1].raw, r->tex[2].raw, r->tex[3].raw);
                fprintf(mf, "      cull=0x%X alpha_ctl=0x%X topology=%u vb_offset=%u is_vp=%d\n",
                        r->cull, r->alpha_ctl, r->topology, r->vb_byte_offset, r->is_vp);
                fprintf(mf, "      depth test=%d write=%d func=0x%X stencil=%d\n",
                        r->depth_test, r->depth_write, r->depth_func, r->stencil.enable);
                const u32 vp_stride = r->vp_stride ? r->vp_stride : VP_VERT_STRIDE;
                const u32 vp_mask = r->vp_input_mask ? r->vp_input_mask : 0xFFFFu;
                if (r->is_vp && !r->vp_raw_bytes && s_d3d.vp_vb_mapped &&
                    (u64)r->vb_byte_offset + (u64)r->vertex_count * vp_stride <=
                        (u64)MAX_VERTICES * VP_VERT_STRIDE) {
                    const float* verts = (const float*)((const u8*)s_d3d.vp_vb_mapped +
                        (u64)(s_d3d.vp_parity ^ 1) * MAX_VERTICES * VP_VERT_STRIDE + r->vb_byte_offset);
                    for (u32 v = 0; v < r->vertex_count && v < 3; ++v) {
                        u32 packed = 0;
                        for (u32 a = 0; a < 16; ++a) {
                            if (!(vp_mask & (1u << a))) continue;
                            const float* x = (const float*)((const u8*)verts +
                                (u64)v * vp_stride + packed * 16u);
                            fprintf(mf, "      v%u a%u=(%.9g,%.9g,%.9g,%.9g)\n",
                                    v, a, x[0], x[1], x[2], x[3]);
                            packed++;
                        }
                    }
                }
                if (r->is_vp && r->vp_raw_bytes && s_d3d.vp_vb_mapped) {
                    fprintf(mf, "      raw_vertex_bytes=%u\n", r->vp_raw_bytes);
                    snprintf(path, sizeof(path), "%s/op%03u_vertices.raw", dir, d);
                    FILE* raw = fopen(path, "wb");
                    if (raw) {
                        const u8* vertices = (const u8*)s_d3d.vp_vb_mapped +
                            (u64)(s_d3d.vp_parity ^ 1) * MAX_VERTICES * VP_VERT_STRIDE +
                            r->vb_byte_offset;
                        fwrite(vertices, 1, r->vp_raw_bytes, raw);
                        fclose(raw);
                    }
                }
                for (int u = 0; u < RSX_TEXTURE_UNITS; u++) {
                    if (!r->tex[u].set) continue;
                    fprintf(mf, "      t%d off=0x%08X fmt=0x%02X (%s%s) %ux%u remap=0x%04X\n", u,
                            r->tex[u].off, r->tex[u].fmt,
                            (r->tex[u].fmt & 0x20) ? "LN" : "SZ",
                            (r->tex[u].fmt & 0x40) ? ",NR" : "",
                            r->tex[u].w, r->tex[u].h, r->tex[u].ctrl1 & 0xFFFF);
                    fprintf(mf, "      t%d sampler address=0x%08X filter=0x%08X control0=0x%08X border=0x%08X shadow=%u\n",
                            u, r->tex[u].address, r->tex[u].filter, r->tex[u].control0,
                            r->tex[u].border_color, (unsigned)r->tex_shadow[u].enabled);
                }
            }
        }
        /* Per-draw VP constant banks for the first few offscreen-RT geometry
         * draws: c[0..3] is the projection row-major for most CG VPs; also scan
         * for the largest non-zero slot so a mis-placed MVP is obvious. Records
         * live at slot==draw-index; parity is post-^1 here (capture is after the
         * frame's vp_parity flip), so read the other half. */
        if (s_d3d.vp_cb_mapped) {
            u32 par = (u32)(s_d3d.vp_parity ^ 1);
            fprintf(mf, "--- vp constants (first offscreen geometry draws) ---\n");
            int shown = 0;
            for (u32 d = 0; d < ndraws && d < MAX_DRAWS && shown < 4; d++) {
                const D3D12DrawRecord* r = &s_d3d.draws[d];
                if (r->is_clear || is_display_target_key(r->rt_off)) continue;
                const float* c = (const float*)((const char*)s_d3d.vp_cb_mapped
                    + ((u64)par * MAX_DRAWS + d) * VP_CB_STRIDE);
                int last_nz = -1; float maxabs = 0.0f;
                for (int i = 0; i < RSX_MAX_VERTEX_CONSTANTS * 4; i++) {
                    float a = c[i] < 0 ? -c[i] : c[i];
                    if (a > 1e-9f) last_nz = i;
                    if (a > maxabs) maxabs = a;
                }
                fprintf(mf, "op%03u vs=%d lastNZslot=%d maxabs=%.3f\n",
                        d, r->vs_idx, last_nz / 4, maxabs);
                if (shown == 0) {   /* full non-zero slot dump for the 1st draw */
                    for (int s = 0; s <= last_nz / 4; s++) {
                        const float* v = c + s * 4;
                        if ((v[0]||v[1]||v[2]||v[3]))
                            fprintf(mf, "   c[%3d] = %9.4f %9.4f %9.4f %9.4f\n",
                                    s, v[0], v[1], v[2], v[3]);
                    }
                }
                shown++;
            }
        }
        fclose(mf);
    }

    /* Backbuffer (in PRESENT state at capture time) + every used offscreen RT. */
    snprintf(path, sizeof path, "%s/backbuffer.bmp", dir);
    cap_readback_write(s_d3d.render_targets[fi], s_d3d.width, s_d3d.height,
                       DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_STATE_PRESENT, fi, path);
    for (int i = 0; i < MAX_OFF_RTS; i++) {
        OffRT* r = &s_d3d.off_rt[i];
        if (!r->res) continue;
        snprintf(path, sizeof path, "%s/rt_%08X.bmp", dir, r->off);
        cap_readback_write(r->res, r->w, r->h, r->dxgi, r->st, fi, path);
    }
    printf("[RSXCAP] frame %llu -> %s (%u ops)\n",
           (unsigned long long)s_d3d.frame_count, dir, ndraws);
}

/* RPCS3_REFERENCE_REVIEWED: vertex-texture enable/dimension decoding follows
 * the NV40 texture fields used by RPCS3's vertex-program and texture-cache
 * paths. The local D3D12 state mapping is independently written. */
/* Decompile the captured RSX vertex program to HLSL and build the VP PSO
 * (decompiled VS + atlas alpha-test PS). One-shot per program. */
static void d3d12_vtex_masks(const rsx_state* st, u32* mask,
                             u32* cube_mask, u32* one_d_mask)
{
    u32 m = 0, c = 0, d1 = 0;
    if (st) for (u32 u = 0; u < RSX_MAX_VERTEX_TEXTURES; u++) {
        const rsx_texture_state* t = &st->vertex_textures[u];
        const u32 w = (t->image_rect >> 16) & 0xFFFFu;
        const u32 h = t->image_rect & 0xFFFFu;
        const u32 dimension = (t->format >> 4) & 0xFu;
        if (!(t->control0 & 0x80000000u) || !w) continue;
        if (dimension == 1u) {
            m |= 1u << u;
            d1 |= 1u << u;
        } else if (dimension == 2u && h) {
            m |= 1u << u;
            if ((t->format & 4u) && w == h)
                c |= 1u << u;
        }
    }
    if (mask) *mask = m;
    if (cube_mask) *cube_mask = c;
    if (one_d_mask) *one_d_mask = d1;
}

static void compile_vp(void)
{
    extern int rsx_vp_decompile_textures(const uint8_t*, u32, u32, u32, u32,
                                         char*, u32);
    const rsx_state* st = s_d3d.current_rsx_state;
    if (!st || st->vp_ucode_bytes < 16 || !s_d3d.vp_root_sig) return;

    static char hlsl[64 * 1024];
    u32 vtex_mask, vtex_cube_mask, vtex_1d_mask;
    d3d12_vtex_masks(st, &vtex_mask, &vtex_cube_mask, &vtex_1d_mask);
    int ni = rsx_vp_decompile_textures(st->vp_ucode, st->vp_ucode_bytes,
        vtex_mask, vtex_cube_mask, vtex_1d_mask, hlsl, sizeof hlsl);
    /* Does this VP read vp_c[0..3]? Gates the garbage-projection fallback:
     * programs keeping their MVP elsewhere (gcm/cube: c[256..259]) must not
     * have c[0..3] stomped nor the viewport z-lane overridden. */
    s_d3d.vp_uses_c03 = (strstr(hlsl, "vp_c[0]") || strstr(hlsl, "vp_c[1]") ||
                         strstr(hlsl, "vp_c[2]") || strstr(hlsl, "vp_c[3]")) ? 1 : 0;
    if (ni <= 0) { printf("[VP] decompile failed (%d)\n", ni); s_d3d.vp_compiled_bytes = st->vp_ucode_bytes; return; }
    if (getenv("VP_DUMP")) {
        FILE* vf = fopen("vp_dump.hlsl", "w");
        if (vf) { fwrite(hlsl, 1, strlen(hlsl), vf); fclose(vf);
                  printf("[VP] dumped vp_dump.hlsl (%d instrs)\n", ni); }
        printf("[VPRAW] first 3 instrs (%u ucode bytes):\n", st->vp_ucode_bytes);
        for (u32 _q = 0; _q < 48 && _q < st->vp_ucode_bytes; _q += 16)
            printf("[VPRAW]  d0=%02X%02X%02X%02X d1=%02X%02X%02X%02X d2=%02X%02X%02X%02X d3=%02X%02X%02X%02X\n",
                st->vp_ucode[_q+0],st->vp_ucode[_q+1],st->vp_ucode[_q+2],st->vp_ucode[_q+3],
                st->vp_ucode[_q+4],st->vp_ucode[_q+5],st->vp_ucode[_q+6],st->vp_ucode[_q+7],
                st->vp_ucode[_q+8],st->vp_ucode[_q+9],st->vp_ucode[_q+10],st->vp_ucode[_q+11],
                st->vp_ucode[_q+12],st->vp_ucode[_q+13],st->vp_ucode[_q+14],st->vp_ucode[_q+15]);
    }

    /* Preserve the VSOutput varying signature and order in the PS. Keep the
     * system-generated face flag as a separate trailing entry-point argument
     * so it cannot disturb the user-varying signature or trigger compiler
     * signature-order diagnostics. */
    static const char ps[] =
        "Texture2D tex : register(t0); SamplerState smp : register(s0);\n"
        "struct PSIn{ float4 pos:SV_Position; float4 col0:COLOR0; float4 col1:COLOR1;\n"
        "  float4 col2:COLOR2; float4 col3:COLOR3; float4 fog:FOG;\n"
        "  float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;\n"
        "  float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7;\n"
        "  float4 t8:TEXCOORD8; float4 t9:TEXCOORD9; };\n"
        "float4 main(PSIn i, bool is_front_face:SV_IsFrontFace):SV_TARGET{ float cov = tex.Sample(smp, i.t0.xy).r;\n"
        "  if (cov <= 0.5 || isnan(i.fog.x)) discard;\n"
        "  float4 diffuse=is_front_face?i.col0:i.col2;\n"
        "  float4 specular=is_front_face?i.col1:i.col3;\n"
        "  return float4(diffuse.rgb+specular.rgb, 1); }\n";

    ID3DBlob *vb=NULL,*pb=NULL,*e=NULL;
    HRESULT hr = d3d12_compile_cached(hlsl, strlen(hlsl), "vp", NULL, NULL, "main", "vs_5_0", 0, 0, &vb, &e);
    if (FAILED(hr)) { printf("[VP] VS compile FAIL: %s\n", e?(const char*)e->lpVtbl->GetBufferPointer(e):"?"); if(e)e->lpVtbl->Release(e); s_d3d.vp_compiled_bytes=st->vp_ucode_bytes; return; }
    hr = d3d12_compile_cached(ps, sizeof(ps)-1, "vpps", NULL, NULL, "main", "ps_5_0", 0, 0, &pb, &e);
    if (FAILED(hr)) { printf("[VP] PS compile FAIL: %s\n", e?(const char*)e->lpVtbl->GetBufferPointer(e):"?"); if(e)e->lpVtbl->Release(e); if(vb)vb->lpVtbl->Release(vb); s_d3d.vp_compiled_bytes=st->vp_ucode_bytes; return; }

    /* This primary fallback pipeline keeps the full 16-slot stream. Per-draw
     * guest programs build their own input layouts from verified input masks. */
    D3D12_INPUT_ELEMENT_DESC il[16];
    for (int _e = 0; _e < 16; _e++) {
        il[_e].SemanticName = "ATTR";
        il[_e].SemanticIndex = _e;
        il[_e].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        il[_e].InputSlot = 0;
        /* 256-byte generic VP vertex: every ATTRi is a float4 slot at i*16
         * (read_vp_vertex converts each enabled RSX attrib by type; disabled
         * slots hold (0,0,0,1)). Covers tiny3d (a0/a3/a8), SDK gcm samples
         * (a0/a1/a2), dbgfont -- no aliasing. */
        il[_e].AlignedByteOffset = (UINT)(_e * 16);
        il[_e].InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        il[_e].InstanceDataStepRate = 0;
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_d3d.vp_root_sig;
    pd.VS.pShaderBytecode = vb->lpVtbl->GetBufferPointer(vb); pd.VS.BytecodeLength = vb->lpVtbl->GetBufferSize(vb);
    pd.PS.pShaderBytecode = pb->lpVtbl->GetBufferPointer(pb); pd.PS.BytecodeLength = pb->lpVtbl->GetBufferSize(pb);
    pd.InputLayout.pInputElementDescs = il; pd.InputLayout.NumElements = 16;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1; pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    /* Depth test ON (LESS_EQUAL, matching the guest's rsxtiny_DepthTestFunc 515 /
     * CELL_GCM_LEQUAL). Without it the cube's faces drew in submission order and
     * back faces overwrote the front -> you saw through the near face to the
     * darker interior. The depth buffer is bound + cleared to 1.0 each frame. */
    pd.DepthStencilState.DepthEnable = TRUE;
    pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    pd.DepthStencilState.StencilEnable = FALSE;
    pd.SampleDesc.Count = 1;
    hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(s_d3d.device, &pd, &IID_ID3D12PipelineState, (void**)&s_d3d.pipeline_state_vp);
    pd.DSVFormat = DXGI_FORMAT_D16_UNORM;
    s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
        s_d3d.device, &pd, &IID_ID3D12PipelineState,
        (void**)&s_d3d.pipeline_state_vp_z16);
    pd.NumRenderTargets = 0;
    pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
        s_d3d.device, &pd, &IID_ID3D12PipelineState,
        (void**)&s_d3d.pipeline_state_vp_no_rt);
    pd.DSVFormat = DXGI_FORMAT_D16_UNORM;
    s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
        s_d3d.device, &pd, &IID_ID3D12PipelineState,
        (void**)&s_d3d.pipeline_state_vp_no_rt_z16);
    pd.NumRenderTargets = 1;
    pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;

    /* Second PSO: same VS but a COLOUR-only PS (no texture sample, no alpha
     * discard) for untextured 3D geometry (vkcube's cube). The atlas PS above
     * samples a texture and discards where coverage<=0.5, which blanks any draw
     * with no atlas bound. */
    {
        static const char ps_col[] =
            "struct PSIn{ float4 pos:SV_Position; float4 col0:COLOR0; float4 col1:COLOR1;\n"
            "  float4 col2:COLOR2; float4 col3:COLOR3; float4 fog:FOG; };\n"
            "float4 main(PSIn i, bool is_front_face:SV_IsFrontFace):SV_TARGET{ if(isnan(i.fog.x)) discard;\n"
            "  float4 diffuse=is_front_face?i.col0:i.col2;\n"
            "  float4 specular=is_front_face?i.col1:i.col3;\n"
            "  return float4(diffuse.rgb+specular.rgb, 1); }\n";
        ID3DBlob *pcb=NULL,*ec=NULL;
        if (SUCCEEDED(d3d12_compile_cached(ps_col, sizeof(ps_col)-1, "vppsc", NULL, NULL,
                                 "main", "ps_5_0", 0, 0, &pcb, &ec)) && pcb) {
            pd.PS.pShaderBytecode = pcb->lpVtbl->GetBufferPointer(pcb);
            pd.PS.BytecodeLength  = pcb->lpVtbl->GetBufferSize(pcb);
            HRESULT hr2 = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pd, &IID_ID3D12PipelineState,
                (void**)&s_d3d.pipeline_state_vp_color);
            pd.DSVFormat = DXGI_FORMAT_D16_UNORM;
            s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pd, &IID_ID3D12PipelineState,
                (void**)&s_d3d.pipeline_state_vp_color_z16);
            pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
            pd.NumRenderTargets = 0;
            s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pd, &IID_ID3D12PipelineState,
                (void**)&s_d3d.pipeline_state_vp_color_no_rt);
            pd.DSVFormat = DXGI_FORMAT_D16_UNORM;
            s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
                s_d3d.device, &pd, &IID_ID3D12PipelineState,
                (void**)&s_d3d.pipeline_state_vp_color_no_rt_z16);
            pd.NumRenderTargets = 1;
            pd.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
            printf(SUCCEEDED(hr2) ? "[VP] colour pipeline ready\n"
                                  : "[VP] colour PSO FAIL (0x%08lX)\n", hr2);
            pcb->lpVtbl->Release(pcb);
        }
        if (ec) ec->lpVtbl->Release(ec);
    }
    /* Keep the VS bytecode for guest-FP PSO builds (vp_get_fp_pso); bump the
     * generation so cached FP PSOs built against the old VS are rebuilt. */
    if (s_d3d.vp_vs_blob) s_d3d.vp_vs_blob->lpVtbl->Release(s_d3d.vp_vs_blob);
    s_d3d.vp_vs_blob = vb;
    s_d3d.vp_gen++;
    pb->lpVtbl->Release(pb);
    s_d3d.vp_compiled_bytes = st->vp_ucode_bytes;
    if (SUCCEEDED(hr)) { s_d3d.vp_ready = 1; printf("[VP] pipeline ready (%d instrs)\n", ni); }
    else {
        printf("[VP] PSO creation FAIL (0x%08lX)\n", hr);
        /* Drain the debug layer's message queue to get the EXACT validation
         * reason (E_INVALIDARG is otherwise opaque). One-shot. */
        ID3D12InfoQueue* iq = NULL;
        if (SUCCEEDED(s_d3d.device->lpVtbl->QueryInterface(
                s_d3d.device, &IID_ID3D12InfoQueue, (void**)&iq)) && iq) {
            UINT64 n = iq->lpVtbl->GetNumStoredMessages(iq);
            for (UINT64 mi = 0; mi < n; mi++) {
                SIZE_T len = 0;
                iq->lpVtbl->GetMessage(iq, mi, NULL, &len);
                D3D12_MESSAGE* m = (D3D12_MESSAGE*)malloc(len);
                if (m && SUCCEEDED(iq->lpVtbl->GetMessage(iq, mi, m, &len)))
                    printf("[VP][DBG] %s\n", m->pDescription);
                free(m);
            }
            iq->lpVtbl->Release(iq);
        }
    }
}

/* Build (or fetch) a PSO pairing the current decompiled VS with the guest's
 * FRAGMENT program at fp_addr: read the FP ucode from guest memory, decompile
 * to HLSL (rsx_fp_decompiler), compile, and cache. This replaces the two
 * hardcoded pixel shaders for draws whose FP we can translate -- e.g.
 * gcm/cube's plasma FP `c = tex2D(t0, uv).x; out = (c, 0, c, 1)`. */
/* Hash + compile the CURRENT rsx_state vertex program into the VS cache;
 * returns the cache slot (or -1). Called at draw-record time so each draw
 * carries the VP that was loaded when it was submitted. */
static const char* d3d12_setting_vtx_dump(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("VTX_DUMP"); ready = 1; }
    return value;
}

static const char* d3d12_setting_vbfull(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("VBFULL"); ready = 1; }
    return value;
}

static const char* d3d12_setting_blenddbg(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("BLENDDBG"); ready = 1; }
    return value;
}

static const char* d3d12_setting_texfmtdbg(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("TEXFMTDBG"); ready = 1; }
    return value;
}

static const char* d3d12_setting_subvp_dbg(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("SUBVP_DBG"); ready = 1; }
    return value;
}

static const char* d3d12_setting_rtt_dump(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("RTT_DUMP"); ready = 1; }
    return value;
}

static const char* d3d12_setting_rtt_pass(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("RTT_PASS"); ready = 1; }
    return value;
}

static const char* d3d12_setting_rtt_viewrt(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("RTT_VIEWRT"); ready = 1; }
    return value;
}

static const char* d3d12_setting_mempeek(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("MEMPEEK"); ready = 1; }
    return value;
}

static const char* d3d12_setting_rtt_savert(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("RTT_SAVERT"); ready = 1; }
    return value;
}

static const char* d3d12_setting_rtt_savea(void)
{
    static __declspec(thread) const char* value;
    static __declspec(thread) int ready;
    if (!ready) { value = getenv("RTT_SAVEA"); ready = 1; }
    return value;
}

#include "rsx_fp_metadata_cache.h"
static rsx_fp_metadata_cache s_fp_metadata;

static u32 vp_hash_ucode(const u8* p, u32 n)
{
    u32 h = 2166136261u;
    /* Match the decompiler's consumed instructions. The rest of the resident
     * store belongs to other programs and may change independently. */
    for (u32 off = 0; off + 16 <= n; off += 16) {
        u32 d3;
        for (u32 i = 0; i < 16; i++) { h ^= p[off+i]; h *= 16777619u; }
        memcpy(&d3, p + off + 12, sizeof(d3));
        if (d3 & 1u) break;
    }
    return h ? h : 1u;
}

static int fp_buffered_constants(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* mode = getenv("FP_CONSTBUF");
        enabled = !mode || mode[0] != '0';
    }
    return enabled;
}

typedef struct {
    u32 bytes, available, input_mask, hash;
    u8 snapshot[1024 * 16];
} VPMetadata;
static VPMetadata s_vp_metadata[64];

static int d3d12_index_cache_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* e = getenv("RSX_D3D12_INDEX_CACHE");
        enabled = e && !strcmp(e, "1");
    }
    return enabled;
}
static int vp_get_vs(const rsx_state* st)
{
    extern int rsx_vp_decompile_compact_textures(
        const uint8_t*, u32, u32, u32, u32, u32, char*, u32);
    if (!st || st->vp_ucode_bytes < 16) return -1;
    /* Start at the instruction SET_TRANSFORM_PROGRAM_START selects, not at 0.
     * The microcode store holds every resident program -- this title keeps its
     * scene, fluid, caustics and droplet programs in it at once -- so
     * decompiling from 0 ran the scene's transform for every draw. That is why
     * the fluid's 114300 vertices rasterized nothing: they were being pushed
     * through the wrong program.
     * VP_START_OFF=1 restores the old always-from-zero behaviour. */
    u32 vstart = 0;
    { static int off = -1; if (off < 0) off = getenv("VP_START_OFF") ? 1 : 0;
      if (!off) vstart = st->transform_program_start * 16u; }
    if (vstart >= st->vp_ucode_bytes) vstart = 0;
    const u32 branch_base = vstart / 16u;
    const u8* vuc = st->vp_ucode + vstart;
    u32 vlen = st->vp_ucode_bytes - vstart;
    VPMetadata* metadata = &s_vp_metadata[branch_base & 63u];
    const int metadata_hit = metadata->bytes && metadata->available == vlen &&
        metadata->bytes <= vlen && !memcmp(metadata->snapshot, vuc, metadata->bytes);
    if (metadata_hit) vlen = metadata->bytes;
    else {
        extern u32 rsx_vp_program_size_instrs(const u8*, u32);
        const u32 instrs = rsx_vp_program_size_instrs(vuc, vlen);
        if (instrs) vlen = instrs * 16u;
    }
    u32 vtex_mask, vtex_cube_mask, vtex_1d_mask;
    d3d12_vtex_masks(st, &vtex_mask, &vtex_cube_mask, &vtex_1d_mask);
    u32 input_mask = metadata_hit ? metadata->input_mask : 0xFFFFu;
    u32 hash;
    if (metadata_hit) hash = metadata->hash;
    else {
        rsx_vp_input_analysis input_analysis = {0xFFFFu, 0};
        if (rsx_vp_analyze_inputs(vuc, vlen, &input_analysis) >= 0 &&
            input_analysis.exact && input_analysis.input_mask)
            input_mask = input_analysis.input_mask;
        hash = vp_hash_ucode(vuc, vlen);
        if (vlen <= sizeof(metadata->snapshot)) {
            memcpy(metadata->snapshot, vuc, vlen);
            metadata->available = st->vp_ucode_bytes - vstart;
            metadata->bytes = vlen;
            metadata->input_mask = input_mask;
            metadata->hash = hash;
        }
    }
    u32 input_count = 0;
    for (u32 bits = input_mask; bits; bits >>= 1) input_count += bits & 1u;
    const u32 input_stride = input_count * 16u;
    hash ^= branch_base;
    hash *= 16777619u;
    const u32 vs_bucket = (hash ^ (hash >> 16) ^ (vtex_mask * 31u) ^
        (vtex_cube_mask * 131u) ^ (vtex_1d_mask * 257u) ^ input_mask) & 1023u;
    const int indexed = d3d12_index_cache_enabled();
    for (u32 link = indexed ? s_d3d.vp_vs_heads[vs_bucket] : 1u;
         link && link <= (u32)s_d3d.vp_vs_n;
         link = indexed ? s_d3d.vp_vs[link - 1u].hash_next : link + 1u) {
        const int i = (int)(link - 1u);
        if (s_d3d.vp_vs[i].hash == hash &&
            s_d3d.vp_vs[i].branch_base == branch_base &&
            s_d3d.vp_vs[i].vtex_mask == vtex_mask &&
            s_d3d.vp_vs[i].vtex_cube_mask == vtex_cube_mask &&
            s_d3d.vp_vs[i].vtex_1d_mask == vtex_1d_mask &&
            s_d3d.vp_vs[i].input_mask == input_mask &&
            s_d3d.vp_vs[i].input_stride == input_stride) return i;
    }

    static char hlsl[262144];
    rsx_vp_set_branch_base(branch_base);
    int ni = rsx_vp_decompile_compact_textures(vuc, vlen, vtex_mask,
        vtex_cube_mask, vtex_1d_mask, input_mask, hlsl, sizeof hlsl);
    if (ni <= 0) return -1;
    if (getenv("VP_DUMP")) { static int _d=0; if (PS3_LOG_FIRST(_d, 4)) {
        FILE* f = fopen("vp2_dump.hlsl", _d==1 ? "w" : "a");
        if (f) { fprintf(f, "/* per-draw VS hash pending, %d instrs */%s%s", ni, hlsl, "\n"); fclose(f); } } }
    /* RSX_DBG_VP=<hex tex0 offset>: write the vertex program belonging to the draws
     * that bind that texture. vp2_dump.hlsl holds whichever four programs were
     * compiled first, which need not include the one under investigation -- and
     * reading the wrong program sends you chasing the wrong constants. */
    { static const char* dp = (const char*)1; static u32 wantp = 0; static int done = 0;
      if (dp == (const char*)1) { dp = getenv("RSX_DBG_VP");
                                  wantp = dp ? (u32)strtoul(dp, NULL, 16) : 0; }
      if (wantp && !done && s_d3d.cur_texs[0].raw == wantp) {
          done = 1;
          FILE* f = fopen("duck_vp.hlsl", "w");
          if (f) { fprintf(f, "/* duck VP, %d instrs, tex0=0x%X */%s", ni, wantp, hlsl);
                   fclose(f); }
          fprintf(stderr, "[DUCKVP] wrote duck_vp.hlsl (%d instrs)%c", ni, 10);
      } }
    /* VP_BYPASS=1: replace the guest transform with a direct map of attribute 0
     * into clip space. If geometry appears with this on, everything from the
     * input layout through raster/depth/present is sound and the fault is the
     * transform (constants or the decompiled program); if it stays blank, the
     * fault is upstream of the shader -- the attribute binding itself. Patched
     * in place, space-padded to the original length so offsets stay valid. */
    /* VP_W_ONE=1: keep the guest transform but force the clip-space w to 1.
     * Different from VP_BYPASS, which replaces the whole position: this tests
     * only the hypothesis that w is wrong. This title's VP ends
     *   { float4 _v = (float4)((vp_c[467]).xxxx); o[0].w = _v.w; }
     * with c467 = (-1, 1, 0, 0), so w = -1 and the clipper discards every
     * vertex. Patched in place, space-padded so offsets stay valid. */
    if (getenv("VP_W_ONE")) {
        static const char anchor[] = "o[0].w = _v.w;";
        static const char repl[]   = "o[0].w = 1.0;";
        char* at = hlsl;
        while ((at = strstr(at, anchor)) != NULL) {
            size_t n = sizeof(anchor) - 1, m = sizeof(repl) - 1;
            memcpy(at, repl, m);
            memset(at + m, ' ', n - m);
            at += n;
        }
    }
    if (getenv("VP_BYPASS")) {
        static const char anchor[] =
            "Out.pos = float4(_p.xyz * vp_posscale.xyz + _p.w * vp_posoffset.xyz, _p.w);";
        static const char repl[] = "Out.pos = float4(v[0].xy * 0.5, 0.5, 1.0);";
        char* at = hlsl;
        while ((at = strstr(at, anchor)) != NULL) {
            size_t n = sizeof(anchor) - 1, m = sizeof(repl) - 1;
            memcpy(at, repl, m);
            memset(at + m, ' ', n - m);
            at += n;
        }
    }
    ID3DBlob* vb = NULL; ID3DBlob* e = NULL;
    HRESULT hr = d3d12_compile_cached(hlsl, strlen(hlsl), "guest_vp2", NULL, NULL,
                            "main", "vs_5_0", 0, 0, &vb, &e);
    if (e) e->lpVtbl->Release(e);
    if (FAILED(hr) || !vb) {
        static int _e=0; if (PS3_LOG_FIRST(_e, 4)) printf("[VP] per-draw VS compile FAIL (hash=0x%08X)\n", hash);
        return -1;
    }
    if (s_d3d.vp_vs_n == s_d3d.vp_vs_capacity) {
        int capacity = s_d3d.vp_vs_capacity ? s_d3d.vp_vs_capacity * 2
                                          : VP_VS_INITIAL_CAPACITY;
        VPVSEntry* entries = (VPVSEntry*)realloc(s_d3d.vp_vs,
                                               sizeof(*entries) * (size_t)capacity);
        if (!entries) {
            vb->lpVtbl->Release(vb);
            fprintf(stderr, "[VP] cannot grow vertex program cache\n");
            return -1;
        }
        s_d3d.vp_vs = entries;
        s_d3d.vp_vs_capacity = capacity;
    }
    int slot = s_d3d.vp_vs_n++;
    s_d3d.vp_vs[slot].hash = hash;
    s_d3d.vp_vs[slot].branch_base = branch_base;
    s_d3d.vp_vs[slot].vtex_mask = vtex_mask;
    s_d3d.vp_vs[slot].vtex_cube_mask = vtex_cube_mask;
    s_d3d.vp_vs[slot].vtex_1d_mask = vtex_1d_mask;
    s_d3d.vp_vs[slot].input_mask = input_mask;
    s_d3d.vp_vs[slot].input_stride = input_stride;
    s_d3d.vp_vs[slot].hash_next = 0;
    if (indexed) {
        s_d3d.vp_vs[slot].hash_next = s_d3d.vp_vs_heads[vs_bucket];
        s_d3d.vp_vs_heads[vs_bucket] = (u32)slot + 1u;
    }
    s_d3d.vp_vs[slot].vs   = vb;
    s_d3d.vp_vs[slot].raw_vs = NULL;
    static char raw_hlsl[280 * 1024];
    if (rsx_raw_vertex_wrap_hlsl(hlsl, input_mask, raw_hlsl, sizeof(raw_hlsl))) {
        ID3DBlob* errors = NULL;
        HRESULT raw_hr = d3d12_compile_cached(raw_hlsl, strlen(raw_hlsl),
            "guest_vp_raw", NULL, NULL, "main", "vs_5_0", 0, 0,
            &s_d3d.vp_vs[slot].raw_vs, &errors);
        if (FAILED(raw_hr)) {
            static int failures = 0;
            if (PS3_LOG_FIRST(failures, 8))
                fprintf(stderr, "[VP] GPU fetch compile failed: %s\n",
                    errors ? (const char*)errors->lpVtbl->GetBufferPointer(errors) : "unknown");
        }
        if (errors) errors->lpVtbl->Release(errors);
    }
    s_d3d.vp_vs[slot].uses_c03 =
        (strstr(hlsl, "vp_c[0]") || strstr(hlsl, "vp_c[1]") ||
         strstr(hlsl, "vp_c[2]") || strstr(hlsl, "vp_c[3]")) ? 1 : 0;
    { static int _n=0; if (PS3_LOG_FIRST(_n, 6)) printf("[VP] per-draw VS cached (hash=0x%08X, %d instrs, slot %d)\n", hash, ni, slot); }
    /* Build the base VP pipeline here if it does not exist yet. render_frame's
     * trigger reads s_d3d.current_rsx_state, which by frame end no longer has
     * the microcode -- so for a title that only ever compiles per-draw VS the
     * base pipeline was never built, s_d3d.vp_ready stayed 0, and the VP draw
     * pass it gates dropped every recorded is_vp draw. Here we are at RECORD
     * time with a live state that definitely has microcode. */
    if (!s_d3d.vp_ready && st && st->vp_ucode_bytes >= 16) {
        const rsx_state* prev = s_d3d.current_rsx_state;
        s_d3d.current_rsx_state = st;
        compile_vp();
        s_d3d.current_rsx_state = prev;
    }
    return slot;
}

/* --- Guest blend state -> packed PSO key ----------------------------------
 * RSX blend factors/equation carry GL enums; each SFACTOR/DFACTOR/EQUATION
 * word packs (alpha<<16)|colour. The key packs D3D12 enums so the PSO cache
 * distinguishes real blend modes: bit0 enable, [5:1] srcC, [10:6] dstC,
 * [15:11] srcA, [20:16] dstA, [23:21] opC, [26:24] opA. DeferredShading's
 * light accumulation is additive (ONE,ONE) -- the old hardcoded straight-
 * alpha PSO turned each light pass into a screen-blanking flash. */
static u8 gl_blend_factor_d3d(u32 f, int is_alpha)
{
    switch (f & 0xFFFF) {
    case 0x0000: return D3D12_BLEND_ZERO;
    case 0x0001: return D3D12_BLEND_ONE;
    case 0x0300: return is_alpha ? D3D12_BLEND_SRC_ALPHA     : D3D12_BLEND_SRC_COLOR;
    case 0x0301: return is_alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
    case 0x0302: return D3D12_BLEND_SRC_ALPHA;
    case 0x0303: return D3D12_BLEND_INV_SRC_ALPHA;
    case 0x0304: return D3D12_BLEND_DEST_ALPHA;
    case 0x0305: return D3D12_BLEND_INV_DEST_ALPHA;
    case 0x0306: return is_alpha ? D3D12_BLEND_DEST_ALPHA     : D3D12_BLEND_DEST_COLOR;
    case 0x0307: return is_alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
    case 0x0308: return D3D12_BLEND_SRC_ALPHA_SAT;
    case 0x8001: return D3D12_BLEND_BLEND_FACTOR;       /* CONSTANT_COLOR   */
    case 0x8002: return D3D12_BLEND_INV_BLEND_FACTOR;
    case 0x8003: return D3D12_BLEND_ALPHA_FACTOR;       /* CONSTANT_ALPHA */
    case 0x8004: return D3D12_BLEND_INV_ALPHA_FACTOR;
    default:     return D3D12_BLEND_ONE;
    }
}
static u8 gl_blend_op_d3d(u32 e)
{
    switch (e & 0xFFFF) {
    case 0x8007: return D3D12_BLEND_OP_MIN;
    case 0x8008: return D3D12_BLEND_OP_MAX;
    case 0x800A: return D3D12_BLEND_OP_SUBTRACT;
    case 0x800B: return D3D12_BLEND_OP_REV_SUBTRACT;
    default:     return D3D12_BLEND_OP_ADD;   /* 0x8006 FUNC_ADD / unset */
    }
}
/* Pack the guest's face-culling state for the PSO key.
 * bit0 = cull enabled, bit1 = cull FRONT (else BACK), bit2 = front face is CCW.
 * RSX/GL enums: CULL_FACE FRONT=0x0404 BACK=0x0405 FRONT_AND_BACK=0x0408;
 * FRONT_FACE CW=0x0900 CCW=0x0901. rsx_commands seeds cull_face/front_face with
 * plain 1/0 before any register arrives, so treat those defaults as BACK/CW. */
static u32 rsx_cull_key(const rsx_state* st)
{
    if (!st) return 0;
    /* Keep the winding bit even when culling is disabled. D3D's
     * SV_IsFrontFace (used by two-sided lighting and guest shaders) still
     * depends on FrontCounterClockwise in that case. Returning 0 here made
     * every no-cull draw classify front faces as clockwise, unlike RSX and
     * RPCS3, whose captured pipelines retain CCW for those draws. */
    u32 k = (st->front_face == 0x0901u) ? 4u : 0u;
    if (!st->cull_face_enable) return k;
    k |= 1u;
    if (st->cull_face == 0x0404u || st->cull_face == 0x0408u) k |= 2u;  /* FRONT */
    return k;
}

static u32 rsx_blend_key(const rsx_state* st, int enable)
{
    { static int _bd = -1; if (_bd < 0) _bd = d3d12_setting_blenddbg() ? 1 : 0;
      if (_bd) { static u32 seen[32]; static int ns=0;
        u32 sk = (enable?0x80000000u:0u) | (st ? (st->blend_sfactor & 0xFFFFu) : 0u);
        int f=0; for (int k2=0;k2<ns;k2++) if (seen[k2]==sk) f=1;
        if (!f && ns<32) { seen[ns++]=sk;
            fprintf(stderr, "[BLEND] enable=%d sfactor=0x%X dfactor=0x%X%c", enable,
                    st?st->blend_sfactor:0, st?st->blend_dfactor:0, 10); } } }
    if (!enable) return 0;
    /* Factors never programmed: keep the legacy straight-alpha behaviour
     * (dbgfont-style text enables blending without setting factors). */
    if (!st || (st->blend_sfactor == 0 && st->blend_dfactor == 0))
        return 1u
             | ((u32)D3D12_BLEND_SRC_ALPHA     << 1)
             | ((u32)D3D12_BLEND_INV_SRC_ALPHA << 6)
             | ((u32)D3D12_BLEND_ONE           << 11)
             | ((u32)D3D12_BLEND_INV_SRC_ALPHA << 16)
             | ((u32)D3D12_BLEND_OP_ADD << 21) | ((u32)D3D12_BLEND_OP_ADD << 24);
    return 1u
         | ((u32)gl_blend_factor_d3d(st->blend_sfactor,       0) << 1)
         | ((u32)gl_blend_factor_d3d(st->blend_dfactor,       0) << 6)
         | ((u32)gl_blend_factor_d3d(st->blend_sfactor >> 16, 1) << 11)
         | ((u32)gl_blend_factor_d3d(st->blend_dfactor >> 16, 1) << 16)
         | ((u32)gl_blend_op_d3d(st->blend_equation)       << 21)
         | ((u32)gl_blend_op_d3d(st->blend_equation >> 16) << 24);
}

/* Bound colour target count for a draw record: 1 (target A) + the contiguous
 * prefix of MRT B/C/D offsets. */
static int dr_num_rts(const D3D12DrawRecord* dr)
{
    if (dr->rt_none) return 0;
    int n = 1;
    while (n < 4 && dr->rt_mrt[n - 1]) n++;
    return n;
}

/* Replace every occurrence of `find` with `repl` inside a NUL-terminated buffer,
 * shifting the tail. Used to retarget generated HLSL (2D sampler -> cube). */
static int hlsl_replace_all(char* buf, size_t cap, const char* find, const char* repl)
{
    size_t fl = strlen(find), rl = strlen(repl);
    if (!fl) return 0;
    char* at = buf;
    while ((at = strstr(at, find)) != NULL) {
        size_t used = strlen(buf) + 1;
        if (used + rl - fl > cap) return 0;
        memmove(at + rl, at + fl, used - (size_t)(at - buf) - fl);
        memcpy(at, repl, rl);
        at += rl;
    }
    return 1;
}

/* Add the pixel-shader system input used by D3D12 two-sided lighting. The
 * fragment decompiler only emits this member when its input body references
 * it, while the backend can introduce that reference after decompilation. */
static int d3d12_hlsl_ensure_frontface_input(char* source, size_t capacity)
{
    char* ps_input = strstr(source, "struct PSInput {");
    if (!ps_input) return 0;
    char* close = strstr(ps_input, "};");
    if (!close) return 0;
    char* existing = strstr(ps_input, "SV_IsFrontFace");
    if (existing && existing < close) return 1;

    static const char field[] = "    bool _rsx_front : SV_IsFrontFace;\n";
    const size_t field_size = sizeof(field) - 1;
    const size_t used = strlen(source);
    const size_t offset = (size_t)(close - source);
    if (used + field_size + 1 > capacity) return 0;
    memmove(source + offset + field_size, source + offset, used - offset + 1);
    memcpy(source + offset, field, field_size);
    return 1;
}

/* Raster-control prologues can access TC8/TC9 even when the translated guest
 * fragment body does not. Ensure the PS signature declares those varyings
 * before rsx_fp_apply_raster_controls injects the accesses. The decompiler
 * declares them itself when the guest body reads either varying. */
static int d3d12_hlsl_ensure_raster_texcoord_inputs(char* source,
                                                    size_t capacity,
                                                    u32 texcoord_control_mask)
{
    if (!(texcoord_control_mask & 0x300u)) return 1;

    char* ps_input = strstr(source, "struct PSInput {");
    if (!ps_input) return 0;
    char* close = strstr(ps_input, "};");
    if (!close) return 0;
    const char* tc8 = strstr(ps_input, "tc8:TEXCOORD8");
    const char* tc9 = strstr(ps_input, "tc9:TEXCOORD9");
    if (tc8 && tc9 && tc8 < close && tc9 < close) return 1;

    static const char fields[] =
        "    float4 tc8:TEXCOORD8; float4 tc9:TEXCOORD9;\n";
    const size_t field_size = sizeof(fields) - 1;
    const size_t used = strlen(source);
    const size_t offset = (size_t)(close - source);
    if (used + field_size + 1 > capacity) return 0;
    memmove(source + offset + field_size, source + offset, used - offset + 1);
    memcpy(source + offset, fields, field_size);
    return 1;
}

/* D3D shader model 5 exposes only sampler registers s0-s15. Pack the active
 * RSX texture-unit samplers into those registers, preserving each unit's
 * filter/address state. A rare shader needing more than sixteen sampler
 * objects keeps the former shared normal samplers and compare-function bank. */
static int d3d12_hlsl_remap_fp_samplers(char* source, size_t capacity,
                                       u64 shadow_compare_key,
                                       D3D12FPSamplerLayout* layout)
{
    static const char normal_decl[] = "SamplerState rsx_samp[16] : register(s0);\n";
    static const char shadow_decl16[] =
        "SamplerComparisonState rsx_shadow_samp[16] : register(s16);\n";
    static const char shadow_decl8[] =
        "SamplerComparisonState rsx_shadow_samp[8] : register(s16);\n";
    if (!source || !layout || !strstr(source, normal_decl)) return 0;

    memset(layout, 0, sizeof(*layout));
    memset(layout->normal_slot, 0xFF, sizeof(layout->normal_slot));
    memset(layout->shadow_slot, 0xFF, sizeof(layout->shadow_slot));
    for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
        char ref[40];
        snprintf(ref, sizeof(ref), "rsx_samp[%u]", unit);
        if (strstr(source, ref)) layout->normal_mask |= (u16)(1u << unit);
        snprintf(ref, sizeof(ref), "rsx_shadow_samp[%u]", unit);
        if (strstr(source, ref)) layout->shadow_mask |= (u16)(1u << unit);
    }

    u32 count = 0;
    for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
        if (layout->normal_mask & (1u << unit)) count++;
        if (layout->shadow_mask & (1u << unit)) count++;
    }
    layout->legacy_fallback = count > RSX_SAMPLER_TABLE_SIZE;

    char decls[1600];
    size_t used = 0;
    decls[0] = '\0';
#define APPEND_SAMPLER_DECL(...) do { \
        if (used >= sizeof(decls)) return 0; \
        int _n = snprintf(decls + used, sizeof(decls) - used, __VA_ARGS__); \
        if (_n < 0 || (size_t)_n >= sizeof(decls) - used) return 0; \
        used += (size_t)_n; \
    } while (0)

    if (!layout->legacy_fallback) {
        u8 next = 0;
        for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
            if (layout->normal_mask & (1u << unit)) {
                layout->normal_slot[unit] = next;
                APPEND_SAMPLER_DECL("SamplerState rsx_samp%u : register(s%u);\n", unit, next++);
            }
        }
        for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
            if (layout->shadow_mask & (1u << unit)) {
                layout->shadow_slot[unit] = next;
                APPEND_SAMPLER_DECL("SamplerComparisonState rsx_shadow_samp%u : register(s%u);\n",
                                    unit, next++);
            }
        }
    } else {
        u8 normal_declared[2] = {0, 0};
        u8 shadow_declared[RSX_SHADOW_COMPARE_FUNCS] = {0};
        for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
            if (!(layout->normal_mask & (1u << unit))) continue;
            const u32 slot = unit == 0 ? 0u : 1u;
            layout->normal_slot[unit] = (u8)slot;
            if (!normal_declared[slot]) {
                normal_declared[slot] = 1;
                APPEND_SAMPLER_DECL("SamplerState rsx_samp%u : register(s%u);\n", slot, slot);
            }
        }
        for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
            if (!(layout->shadow_mask & (1u << unit))) continue;
            const u32 func = (u32)((shadow_compare_key >> (4u * unit)) & 7u);
            const u32 slot = RSX_SHADOW_COMPARE_FIRST_REGISTER + func;
            layout->shadow_slot[unit] = (u8)slot;
            if (!shadow_declared[func]) {
                shadow_declared[func] = 1;
                APPEND_SAMPLER_DECL("SamplerComparisonState rsx_shadow_samp%u : register(s%u);\n",
                                    func, slot);
            }
        }
    }

    if (!hlsl_replace_all(source, capacity, normal_decl, decls)) return 0;
    if (strstr(source, shadow_decl16) &&
        !hlsl_replace_all(source, capacity, shadow_decl16, "")) return 0;
    if (strstr(source, shadow_decl8) &&
        !hlsl_replace_all(source, capacity, shadow_decl8, "")) return 0;

    for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
        char old_ref[40], new_ref[48];
        if (layout->normal_mask & (1u << unit)) {
            snprintf(old_ref, sizeof(old_ref), "rsx_samp[%u]", unit);
            const u32 variable = layout->legacy_fallback
                ? layout->normal_slot[unit] : unit;
            snprintf(new_ref, sizeof(new_ref), "rsx_samp%u", variable);
            if (!hlsl_replace_all(source, capacity, old_ref, new_ref)) return 0;
        }
        if (layout->shadow_mask & (1u << unit)) {
            snprintf(old_ref, sizeof(old_ref), "rsx_shadow_samp[%u]", unit);
            u32 variable = unit;
            if (layout->legacy_fallback)
                variable = (u32)((shadow_compare_key >> (4u * unit)) & 7u);
            snprintf(new_ref, sizeof(new_ref), "rsx_shadow_samp%u", variable);
            if (!hlsl_replace_all(source, capacity, old_ref, new_ref)) return 0;
        }
    }
#undef APPEND_SAMPLER_DECL
    return 1;
}

/* RPCS3's RSX depth-as-colour path reconstructs the guest's packed 24-bit
 * depth bytes after sampling an R24_UNORM depth SRV. A8R8G8B8/D8R8G8B8 reads
 * need that conversion; sampling the SRV's red channel directly loses the
 * other two bytes and breaks deferred passes that rebuild positions. */
static u32 rsx_remap_to_d3d(u32 c1, u32 basef);
static int d3d12_hlsl_unpack_depth_samples(char* source, size_t capacity,
                                          u32 unit_mask,
                                          const D3D12DrawRecord* draw)
{
    if (!unit_mask) return 1;
    const char* entry = strstr(source, "float4 main(");
    if (!entry) entry = strstr(source, "PSOutput main(");
    if (!entry) return 0;
    static const char helper[] =
        "float4 _rsx_depth_to_rgba(float d) {\n"
        "    float scaled = saturate(d) * 16777215.0;\n"
        "    uint v = (uint)scaled + (uint)(frac(scaled) >= 0.5);\n"
        "    return float4(((v >> 8) & 255u) / 255.0, "
        "(v & 255u) / 255.0, 1.0, ((v >> 16) & 255u) / 255.0);\n"
        "}\n";
    const size_t helper_len = sizeof(helper) - 1;
    size_t used = strlen(source) + 1;
    size_t offset = (size_t)(entry - source);
    if (used + helper_len > capacity) return 0;
    memmove(source + offset + helper_len, source + offset, used - offset);
    memcpy(source + offset, helper, helper_len);

    for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
        if (!(unit_mask & (1u << unit))) continue;
        const u32 mapping = rsx_remap_to_d3d(draw->tex[unit].ctrl1,
                                             draw->tex[unit].fmt & 0x9Fu);
        static const char* lanes[] = {"v.r", "v.g", "v.b", "v.a", "0.0", "1.0"};
        char unit_helper[256], prefix[48];
        u32 lane[4];
        for (u32 c = 0; c < 4; c++) {
            lane[c] = (mapping >> (3u * c)) & 7u;
            if (lane[c] > 5u) return 0;
        }
        const int helper_size = snprintf(unit_helper, sizeof unit_helper,
            "float4 _rsx_depth_unit%u(float d) { float4 v=_rsx_depth_to_rgba(d);"
            " return float4(%s,%s,%s,%s); }\n", unit,
            lanes[lane[0]], lanes[lane[1]], lanes[lane[2]], lanes[lane[3]]);
        char* main_entry = strstr(source, "float4 main(");
        if (!main_entry) main_entry = strstr(source, "PSOutput main(");
        if (!main_entry || helper_size < 0 || (size_t)helper_size >= sizeof unit_helper)
            return 0;
        used = strlen(source) + 1;
        if (used + (size_t)helper_size > capacity) return 0;
        memmove(main_entry + helper_size, main_entry,
                used - (size_t)(main_entry - source));
        memcpy(main_entry, unit_helper, (size_t)helper_size);
        snprintf(prefix, sizeof prefix, "_rsx_depth_unit%u(", unit);
        char needles[2][48];
        snprintf(needles[0], sizeof(needles[0]), "rsx_tex[%u].Sample", unit);
        snprintf(needles[1], sizeof(needles[1]), "rsx_tex%u.Sample", unit);
        for (int form = 0; form < 2; form++) {
            char* at = source;
            while ((at = strstr(at, needles[form])) != NULL) {
                char* open = strchr(at, '(');
                if (!open) return 0;
                int nesting = 0;
                char* close = NULL;
                for (char* p = open; *p; p++) {
                    if (*p == '(') nesting++;
                    else if (*p == ')' && --nesting == 0) { close = p; break; }
                }
                if (!close) return 0;
                const size_t prefix_len = strlen(prefix);
                used = strlen(source) + 1;
                if (used + prefix_len + 1 > capacity) return 0;
                size_t tail = used - (size_t)(close + 1 - source);
                memmove(close + 2, close + 1, tail);
                close[1] = ')';
                used++;
                memmove(at + prefix_len, at, used - (size_t)(at - source));
                memcpy(at, prefix, prefix_len);
                at += prefix_len + strlen(needles[form]);
            }
        }
    }
    return 1;
}

/* HLSL interpolation modifiers belong on the pixel-shader inputs. COLOR0..3
 * are the RSX diffuse/specular values; flat mode makes each value constant
 * across a primitive, selected from the D3D provoking vertex below. */
static int d3d12_hlsl_flatten_color_varyings(char* source, size_t capacity)
{
    static const char* const fields[] = {
        "float4 col0", "float4 col1", "float4 col2", "float4 col3"
    };
    static const char prefix[] = "nointerpolation ";
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
        char* field = strstr(source, fields[i]);
        if (!field) return 0;
        const size_t prefix_len = sizeof(prefix) - 1;
        if (field >= source + prefix_len &&
            memcmp(field - prefix_len, prefix, prefix_len) == 0)
            continue;
        const size_t used = strlen(source);
        if (used + prefix_len + 1 > capacity) return 0;
        memmove(field + prefix_len, field,
                used - (size_t)(field - source) + 1);
        memcpy(field, prefix, prefix_len);
    }
    return 1;
}

/* D3D's first-vertex provoking convention differs from the RSX's last-vertex
 * convention. Cyclically rotate complete vertices so the same geometric
 * triangle and winding survive, while the RSX provoking vertex becomes first. */
static void d3d12_select_rsx_provoking_vertex(void* vertices, u32 count,
                                              size_t stride)
{
    if (!vertices || stride == 0 || stride > 256u) return;
    u8* bytes = (u8*)vertices;
    u8 saved[256];
    for (u32 i = 0; i + 2 < count; i += 3) {
        u8* v0 = bytes + (size_t)i * stride;
        u8* v1 = v0 + stride;
        u8* v2 = v1 + stride;
        memcpy(saved, v2, stride);
        memmove(v2, v1, stride);
        memmove(v1, v0, stride);
        memcpy(v0, saved, stride);
    }
}

static void d3d12_apply_flat_vertex_order(const D3D12DrawRecord* dr)
{
    if (!dr || dr->vp_raw_bytes || !dr->flat_shading || dr->topology != RSX_TOPOLOGY_TRIANGLES)
        return;
    if (dr->is_vp) {
        u8* frame = (u8*)s_d3d.vp_vb_mapped +
                    (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE;
        d3d12_select_rsx_provoking_vertex(
        frame + dr->vb_byte_offset, dr->vertex_count,
        dr->vp_stride ? dr->vp_stride : VP_VERT_STRIDE);
    } else {
        d3d12_select_rsx_provoking_vertex(
            d3d12_legacy_vertex_slot_base() + dr->vb_byte_offset,
            dr->vertex_count, VERTEX_STRIDE);
    }
}

/* Which of a draw's texture units are cube textures, as a 16-bit mask. */
static u32 dr_cube_mask(const D3D12DrawRecord* dr)
{
    /* ON by default (RSX_CUBE_TEX=0 to disable). This was off while the cube path
     * cost 2262 PSO misses per 20 frames and 0.38 fps -- both caused by inline
     * fragment-program constants baking into the shader source. With those
     * constants hoisted into b1 the cube path costs 0 PSO misses, and with the
     * per-face stride fixed (each face is a whole mip pyramid, not one mip-0
     * image) all six faces decode as clean environment art. */
    static int en = -1;
    if (en < 0) { const char* e = getenv("RSX_CUBE_TEX"); en = e ? atoi(e) : 1; }
    if (!en) return 0;
    u32 m = 0;
    for (int u = 0; u < RSX_TEXTURE_UNITS; u++) if (dr->tex[u].set && dr->tex[u].cube) m |= 1u << u;
    return m;
}

/* RSX unnormalised (texel-addressed) coordinates need per-texture dimensions
 * folded into the PSO because this backend emits those scales as HLSL literals. */
static u32 dr_unnorm_info(const D3D12DrawRecord* dr, u32 dims[RSX_TEXTURE_UNITS][3])
{
    u32 mask = 0;
    memset(dims, 0, sizeof(u32) * RSX_TEXTURE_UNITS * 3u);
    if (!dr) return 0;
    for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++) {
        if (!dr->tex[u].set || !(dr->tex[u].fmt & 0x40u) || dr->tex[u].cube)
            continue;
        if (!dr->tex[u].w || !dr->tex[u].h) continue;
        dims[u][0] = dr->tex[u].w;
        dims[u][1] = dr->tex[u].h;
        dims[u][2] = 1;
        mask |= 1u << u;
    }
    return mask;
}

static void dr_capture_poly_offset(D3D12DrawRecord* dr, const rsx_state* st)
{
    dr->poly_fill = st ? st->poly_offset_fill : 0;
    dr->poly_scale_bits = dr->poly_bias_bits = 0;
    if (st && dr->poly_fill) {
        memcpy(&dr->poly_scale_bits, &st->poly_offset_scale, sizeof(u32));
        memcpy(&dr->poly_bias_bits, &st->poly_offset_bias, sizeof(u32));
    }
}

static DXGI_FORMAT rsx_zeta_dsv_format(u32 zeta_format)
{
    return zeta_format == 1u ? DXGI_FORMAT_D16_UNORM
                             : DXGI_FORMAT_D24_UNORM_S8_UINT;
}

static int rsx_zeta_format_supported(u32 zeta_format)
{
    return zeta_format == 1u || zeta_format == 2u;
}

static void dr_capture_zeta(D3D12DrawRecord* dr, const rsx_state* st)
{
    dr->zeta_off = st ? st->surface_zeta_offset : 0;
    dr->zeta_pitch = st ? st->surface_zeta_pitch : 0;
    dr->zeta_format = st ? ((st->surface_format >> 5) & 7u) : 0;
    dr->zeta_ea = 0xFFFFFFFFu;
    if (!st || !rsx_zeta_format_supported(dr->zeta_format) ||
        !dr->zeta_pitch)
        return;
    extern u32 cellGcmResolveLocated(int local, u32 offset);
    const int local = st->dma_context[6] != 0xFEED0001u;
    dr->zeta_ea = cellGcmResolveLocated(local, dr->zeta_off);
}

/* RPCS3_REFERENCE_REVIEWED: fog, WPOS, surface-window-offset, and polygon
 * offset behavior was cross-checked against RSXFragmentPrologue.glsl,
 * RSXDrawCommands.cpp, RSXThread.cpp, and VKDraw.cpp. This is a local HLSL /
 * D3D12 implementation; no RPCS3 function body was copied. See the license
 * audit for provenance limits. */
static void dr_capture_extended_rsx_state(D3D12DrawRecord* dr, const rsx_state* st)
{
    dr_capture_poly_offset(dr, st);
    dr_capture_zeta(dr, st);
    dr->rt_none = st && st->color_target == CELL_GCM_SURFACE_TARGET_NONE;
    dr->flat_shading = st && st->shade_mode == CELL_GCM_FLAT;
    dr->two_sided_lighting = st && st->two_sided_lighting;
    dr->depth_test = st ? (st->depth_test_enable != 0) : 0;
    dr->depth_write = st ? (st->depth_mask != 0) : 1;
    dr->depth_func = st ? st->depth_func : 0x0207u; /* GL_ALWAYS */
    dr->alpha_ref = st ? rsx_fp_alpha_ref(st->alpha_ref,
                                          st->surface_format & 0x1Fu) : 0.0f;
    dr->color_mask_mrt = st ? st->color_mask_mrt : 0u;
    dr->blend_enable_mrt = st ? st->blend_enable_mrt : 0u;
    dr->shader_packer = st ? st->shader_packer : 0u;
    dr->srgb_output = st ? rsx_fp_srgb_output_enabled(
        st->shader_packer, st->shader_control,
        st->surface_format & 0x1Fu) : 0;
    if (st && (st->surface_format & 0x1Fu) == 0x0Bu) {
        dr->blend_color[0] = rsx_fp_alpha_ref(st->blend_color, 0x0Bu);
        dr->blend_color[1] = rsx_fp_alpha_ref(st->blend_color >> 16, 0x0Bu);
        dr->blend_color[2] = rsx_fp_alpha_ref(st->blend_color2, 0x0Bu);
        dr->blend_color[3] = rsx_fp_alpha_ref(st->blend_color2 >> 16, 0x0Bu);
    } else if (st) {
        dr->blend_color[0] = (float)((st->blend_color >> 16) & 0xFFu) / 255.0f;
        dr->blend_color[1] = (float)((st->blend_color >> 8) & 0xFFu) / 255.0f;
        dr->blend_color[2] = (float)(st->blend_color & 0xFFu) / 255.0f;
        dr->blend_color[3] = (float)(st->blend_color >> 24) / 255.0f;
    } else {
        memset(dr->blend_color, 0, sizeof dr->blend_color);
    }
    dr->zmin_max_control = st ? st->zmin_max_control : 0x1u;
    dr->scissor_x = st ? st->scissor_x : 0u;
    dr->scissor_y = st ? st->scissor_y : 0u;
    dr->scissor_w = st ? st->scissor_w : 4096u;
    dr->scissor_h = st ? st->scissor_h : 4096u;
    memset(&dr->stencil, 0, sizeof dr->stencil);
    if (st) {
        dr->stencil.enable = st->stencil_test_enable ? 1u : 0u;
        dr->stencil.two_sided = st->two_sided_stencil ? 1u : 0u;
        dr->stencil.front_func = st->stencil_func;
        dr->stencil.front_ref = st->stencil_ref;
        dr->stencil.front_read_mask = st->stencil_func_mask;
        dr->stencil.front_write_mask = st->stencil_write_mask;
        dr->stencil.front_fail = st->stencil_op_fail;
        dr->stencil.front_zfail = st->stencil_op_zfail;
        dr->stencil.front_zpass = st->stencil_op_zpass;
        dr->stencil.back_func = st->back_stencil_func;
        dr->stencil.back_ref = st->back_stencil_ref;
        dr->stencil.back_read_mask = st->back_stencil_mask;
        dr->stencil.back_write_mask = st->back_stencil_write_mask;
        dr->stencil.back_fail = st->back_stencil_op_fail;
        dr->stencil.back_zfail = st->back_stencil_op_zfail;
        dr->stencil.back_zpass = st->back_stencil_op_zpass;
    }
    dr->texcoord_control_mask = 0;
    dr->anti_aliasing_control = st ? st->anti_aliasing_control : 0xFFFF0000u;
    dr->tex_shadow_mask = 0;
    dr->tex_shadow_compare_key = 0;
    memset(dr->tex_shadow, 0, sizeof dr->tex_shadow);
    memset(dr->gamma_masks, 0, sizeof dr->gamma_masks);
    for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++)
        dr->tex[u].filter = dr->tex[u].address = dr->tex[u].control0 =
            dr->tex[u].pitch = dr->tex[u].border_color = 0;
    if (st) for (u32 i = 0; i < 10; i++)
        dr->texcoord_control_mask |= (st->tex_coord_control[i] & 1u) << i;
    dr->vtex_mask = dr->vtex_cube_mask = dr->vtex_1d_mask = 0;
    memset(dr->vtex, 0, sizeof dr->vtex);
    for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++) dr->tex_depth[u] = -1;
    for (u32 u = 0; u < RSX_MAX_VERTEX_TEXTURES; u++) dr->vtex_depth[u] = -1;
    for (u32 u = 0; u < RSX_MAX_VERTEX_TEXTURES; u++)
        dr->vtex_rt[u] = -1;
    if (st) {
        extern u32 cellGcmResolveLocated(int local, u32 offset);
        for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++) {
            const rsx_texture_state* t = &st->textures[u];
            dr->tex[u].filter = t->filter;
            dr->tex[u].address = t->address;
            dr->tex[u].control0 = t->control0;
            dr->tex[u].pitch = t->control3 & 0xFFFFFu;
            dr->tex[u].border_color = t->border_color;
            dr->gamma_masks[u] = rsx_texture_gamma_mask(
                (t->format >> 8) & 0xFFu, t->control1, t->filter, t->address);
            const u32 fmt = ((t->format >> 8) & 0xFFu) & 0x9Fu;
            const u32 zfunc = (t->address >> 28) & 0xFu;
            const int enabled = (t->control0 >> 31) != 0;
            const int alpha_kill = ((t->control0 >> 2) & 1u) != 0;
            const int native_depth = (fmt == 0x90u || fmt == 0x92u);
            rsx_fp_shadow_desc_from_texture(&dr->tex_shadow[u],
                (t->format >> 8) & 0xFFu, (t->format >> 4) & 0xFu,
                (t->format >> 2) & 1u, (t->format >> 16) & 0xFFFFu,
                t->filter, t->address, t->control0);
            if (enabled && dr->tex[u].set && !dr->tex[u].cube && native_depth &&
                !alpha_kill && zfunc > 0u && zfunc < 7u) {
                dr->tex_shadow_mask |= 1u << u;
                dr->tex_shadow_compare_key |= (u64)zfunc << (4u * u);
            }
        }
        d3d12_vtex_masks(st, &dr->vtex_mask,
                         &dr->vtex_cube_mask, &dr->vtex_1d_mask);
        for (u32 u = 0; u < RSX_MAX_VERTEX_TEXTURES; u++) {
            if (!((dr->vtex_mask >> u) & 1u)) continue;
            const rsx_texture_state* t = &st->vertex_textures[u];
            const u32 raw = t->offset;
            u32 off = raw ? cellGcmResolveLocated((t->format & 3u) == 1u, raw) : 0u;
            if (off == 0xFFFFFFFFu) off = 0u;
            dr->vtex[u].raw = raw;
            dr->vtex[u].off = off;
            dr->vtex[u].w = (t->image_rect >> 16) & 0xFFFFu;
            dr->vtex[u].h = (dr->vtex_1d_mask & (1u << u))
                ? 1u : (t->image_rect & 0xFFFFu);
            dr->vtex[u].fmt = (t->format >> 8) & 0xFFu;
            dr->vtex[u].ctrl1 = t->control1;
            dr->vtex[u].pitch = t->control3 & 0xFFFFFu;
            dr->vtex[u].mips = t->format >> 16;
            dr->vtex[u].cube = (dr->vtex_cube_mask & (1u << u)) != 0;
            dr->vtex[u].is_1d = (dr->vtex_1d_mask & (1u << u)) != 0;
            dr->vtex[u].set = 1;
        }
    }
    dr->fog_mode = st ? st->fog_mode : 0x0800u;
    dr->fog_param0_bits = dr->fog_param1_bits = 0;
    if (st) {
        memcpy(&dr->fog_param0_bits, &st->fog_params[0], sizeof(u32));
        memcpy(&dr->fog_param1_bits, &st->fog_params[1], sizeof(u32));
        dr->shader_window = st->shader_window;
        dr->vp_min_depth = st->clip_min;
        dr->vp_max_depth = st->clip_max;
    } else {
        dr->shader_window = 0x1000;
        dr->vp_min_depth = 0.0f;
        dr->vp_max_depth = 1.0f;
    }
}

/* NV4097_SET_DEPTH_FUNC carries the GL comparison enum. D3D12 uses a distinct
 * enum numbering, so convert explicitly and match the other local backends'
 * behavior for an unrecognized value (ALWAYS). */
static D3D12_COMPARISON_FUNC d3d12_depth_func(u32 func)
{
    switch (func) {
    case 0x0200: return D3D12_COMPARISON_FUNC_NEVER;
    case 0x0201: return D3D12_COMPARISON_FUNC_LESS;
    case 0x0202: return D3D12_COMPARISON_FUNC_EQUAL;
    case 0x0203: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    case 0x0204: return D3D12_COMPARISON_FUNC_GREATER;
    case 0x0205: return D3D12_COMPARISON_FUNC_NOT_EQUAL;
    case 0x0206: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    case 0x0207: return D3D12_COMPARISON_FUNC_ALWAYS;
    default:     return D3D12_COMPARISON_FUNC_ALWAYS;
    }
}

/* GCM stores ZFUNC as 0..7 in TEXTURE_ADDRESS[31:28]. D3D SampleCmp compares
 * the shader reference value against the fetched depth, opposite the guest's
 * texel-vs-reference ordering for relational functions. */
static D3D12_COMPARISON_FUNC d3d12_shadow_sample_func(u32 raw_func)
{
    D3D12_COMPARISON_FUNC func = d3d12_depth_func(0x0200u + (raw_func & 7u));
    switch (func) {
    case D3D12_COMPARISON_FUNC_LESS:         return D3D12_COMPARISON_FUNC_GREATER;
    case D3D12_COMPARISON_FUNC_LESS_EQUAL:   return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    case D3D12_COMPARISON_FUNC_GREATER:      return D3D12_COMPARISON_FUNC_LESS;
    case D3D12_COMPARISON_FUNC_GREATER_EQUAL:return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    default:                                  return func;
    }
}

static D3D12_TEXTURE_ADDRESS_MODE d3d12_sampler_wrap(u32 wrap)
{
    switch (wrap & 0xFu) {
    case 1: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case 2: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case 3: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case 4: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case 5: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case 6: case 7: case 8: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}

static D3D12_SAMPLER_DESC d3d12_sampler_desc(u32 filter, u32 address,
                                              u32 control0, int comparison,
                                              u32 compare_func)
{
    D3D12_SAMPLER_DESC sd = {0};
    const u32 minf = (filter >> 16) & 7u;
    const u32 magf = (filter >> 24) & 7u;
    const D3D12_FILTER_TYPE mnf =
        (minf == 2u || minf == 4u || minf == 6u)
            ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mgf = magf == 2u
            ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_TYPE mpf = (minf == 5u || minf == 6u)
            ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
    const D3D12_FILTER_REDUCTION_TYPE reduction = comparison
        ? D3D12_FILTER_REDUCTION_TYPE_COMPARISON
        : D3D12_FILTER_REDUCTION_TYPE_STANDARD;
    sd.Filter = D3D12_ENCODE_BASIC_FILTER(mnf, mgf, mpf, reduction);
    sd.AddressU = d3d12_sampler_wrap(address);
    sd.AddressV = d3d12_sampler_wrap(address >> 8);
    sd.AddressW = d3d12_sampler_wrap(address >> 16);
    const u32 max_lod = (control0 >> 7) & 0xFFFu;
    const u32 min_lod = (control0 >> 19) & 0xFFFu;
    sd.MinLOD = (float)min_lod / 256.0f;
    sd.MaxLOD = minf >= 3u ? (float)max_lod / 256.0f : 0.0f;
    if (sd.MaxLOD < sd.MinLOD) sd.MaxLOD = sd.MinLOD;
    sd.MaxAnisotropy = 1;
    sd.ComparisonFunc = comparison
        ? d3d12_shadow_sample_func(compare_func)
        : D3D12_COMPARISON_FUNC_ALWAYS;
    /* Non-shadow and unsupported border values retain the legacy default. */
    return sd;
}

static void d3d12_sampler_default_table(D3D12_SAMPLER_DESC out[RSX_SAMPLER_TABLE_SIZE])
{
    D3D12_SAMPLER_DESC d = {0};
    d.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    d.AddressU = d.AddressV = d.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    d.MaxLOD = D3D12_FLOAT32_MAX;
    d.MaxAnisotropy = 1;
    d.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    for (u32 i = 0; i < RSX_SAMPLER_TABLE_SIZE; i++) out[i] = d;
    /* The generic VP text PSO samples its glyph atlas through s0. */
    out[0].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    out[0].AddressU = out[0].AddressV = out[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
}

static int d3d12_sampler_table_get(const D3D12_SAMPLER_DESC desc[RSX_SAMPLER_TABLE_SIZE],
                                   D3D12_GPU_DESCRIPTOR_HANDLE* out)
{
    const u32 slot = (u32)s_d3d.vp_parity;
    ID3D12DescriptorHeap* heap = s_d3d.sampler_heap[slot];
    if (!heap || !out || !s_d3d.sampler_inc) return 0;
    const u32 previous = s_d3d.sampler_table_last;
    if (previous < s_d3d.sampler_table_count &&
        !memcmp(s_d3d.sampler_table_descs[previous], desc, sizeof(s_d3d.sampler_table_descs[0]))) {
        heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(heap, out);
        out->ptr += (u64)previous * RSX_SAMPLER_TABLE_SIZE * s_d3d.sampler_inc;
        return 1;
    }
    u64 hash = 1469598103934665603ull;
    const u8* bytes = (const u8*)desc;
    for (size_t i = 0; i < sizeof(s_d3d.sampler_table_descs[0]); i++) {
        hash ^= bytes[i]; hash *= 1099511628211ull;
    }
    for (u32 i = 0; i < s_d3d.sampler_table_count; i++) {
        if (s_d3d.sampler_table_hash[i] == hash &&
            memcmp(s_d3d.sampler_table_descs[i], desc,
                   sizeof(s_d3d.sampler_table_descs[i])) == 0) {
            s_d3d.sampler_table_last = i;
            heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(heap, out);
            out->ptr += (u64)i * RSX_SAMPLER_TABLE_SIZE * s_d3d.sampler_inc;
            return 1;
        }
    }
    if (s_d3d.sampler_table_count >= RSX_SAMPLER_TABLE_COUNT) {
        static int warned = 0;
        if (!warned++) fprintf(stderr, "[D3D12] per-frame sampler table limit reached\n");
        return 0;
    }
    const u32 index = s_d3d.sampler_table_count++;
    s_d3d.sampler_table_last = index;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(heap, &cpu);
    cpu.ptr += (size_t)index * RSX_SAMPLER_TABLE_SIZE * s_d3d.sampler_inc;
    for (u32 i = 0; i < RSX_SAMPLER_TABLE_SIZE; i++) {
        s_d3d.device->lpVtbl->CreateSampler(s_d3d.device, &desc[i], cpu);
        cpu.ptr += s_d3d.sampler_inc;
    }
    memcpy(s_d3d.sampler_table_descs[index], desc,
           sizeof(s_d3d.sampler_table_descs[index]));
    s_d3d.sampler_table_hash[index] = hash;
    heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(heap, out);
    out->ptr += (u64)index * RSX_SAMPLER_TABLE_SIZE * s_d3d.sampler_inc;
    return 1;
}

static int d3d12_sampler_table_for_draw(const D3D12DrawRecord* dr,
                                       const D3D12FPSamplerLayout* layout,
                                       D3D12_GPU_DESCRIPTOR_HANDLE* out)
{
    D3D12_SAMPLER_DESC desc[RSX_SAMPLER_TABLE_SIZE];
    d3d12_sampler_default_table(desc);
    if (dr && layout) {
        if (layout->legacy_fallback) {
            for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
                if (layout->shadow_slot[unit] == 0xFFu) continue;
                const u32 func = (u32)((dr->tex_shadow_compare_key >> (4u * unit)) & 7u);
                D3D12_SAMPLER_DESC* sd = &desc[layout->shadow_slot[unit]];
                sd->Filter = D3D12_ENCODE_BASIC_FILTER(
                    D3D12_FILTER_TYPE_LINEAR, D3D12_FILTER_TYPE_LINEAR,
                    D3D12_FILTER_TYPE_LINEAR, D3D12_FILTER_REDUCTION_TYPE_COMPARISON);
                sd->AddressU = sd->AddressV = sd->AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                sd->MinLOD = 0.0f; sd->MaxLOD = D3D12_FLOAT32_MAX;
                sd->MaxAnisotropy = 1;
                sd->ComparisonFunc = d3d12_shadow_sample_func(func);
            }
        } else {
            for (u32 unit = 0; unit < RSX_TEXTURE_UNITS; unit++) {
                const u8 ns = layout->normal_slot[unit];
                if (ns != 0xFFu) {
                    const D3D12DrawRecord* r = dr;
                    desc[ns] = d3d12_sampler_desc(r->tex[unit].filter,
                        r->tex[unit].address, r->tex[unit].control0, 0, 0);
                    /* The manual comparison shader samples depth through a
                     * normal sampler. Border depth matters on this path too. */
                    rsx_d3d12_apply_shadow_border(&desc[ns],
                        dr->tex_shadow[unit].enabled == 1, dr->tex[unit].border_color);
                }
                const u8 cs = layout->shadow_slot[unit];
                if (cs != 0xFFu) {
                    const u32 func = (u32)((dr->tex_shadow_compare_key >> (4u * unit)) & 7u);
                    desc[cs] = d3d12_sampler_desc(dr->tex[unit].filter,
                        dr->tex[unit].address, dr->tex[unit].control0, 1, func);
                    rsx_d3d12_apply_shadow_border(&desc[cs],
                        dr->tex_shadow[unit].enabled == 1, dr->tex[unit].border_color);
                }
            }
        }
    }
    return d3d12_sampler_table_get(desc, out);
}

/* RPCS3_REFERENCE_REVIEWED: stencil compare/op enum mapping follows the NV40
 * GCM values used by RPCS3 GLDraw/VKDraw. The D3D12 PSO construction below is
 * expressed locally; its API has one reference/read/write mask for both faces. */
/* The stencil comparison and operation registers use the GL enum values. */
static D3D12_COMPARISON_FUNC d3d12_stencil_func(u32 func)
{
    return d3d12_depth_func(func);
}

static D3D12_STENCIL_OP d3d12_stencil_op(u32 op)
{
    switch (op) {
    case 0x0000: return D3D12_STENCIL_OP_ZERO;
    case 0x1E01: return D3D12_STENCIL_OP_REPLACE;
    case 0x1E02: return D3D12_STENCIL_OP_INCR_SAT;
    case 0x1E03: return D3D12_STENCIL_OP_DECR_SAT;
    case 0x150A: return D3D12_STENCIL_OP_INVERT;
    case 0x8507: return D3D12_STENCIL_OP_INCR;
    case 0x8508: return D3D12_STENCIL_OP_DECR;
    case 0x1E00: default: return D3D12_STENCIL_OP_KEEP;
    }
}

/* The reference is dynamic host state. It must not create another PSO. */
static int d3d12_stencil_pso_equal(const RSXStencilState* a,
                                    const RSXStencilState* b)
{
    return a->enable == b->enable && a->two_sided == b->two_sided &&
        a->front_func == b->front_func &&
        a->front_read_mask == b->front_read_mask &&
        a->front_write_mask == b->front_write_mask &&
        a->front_fail == b->front_fail && a->front_zfail == b->front_zfail &&
        a->front_zpass == b->front_zpass && a->back_func == b->back_func &&
        a->back_read_mask == b->back_read_mask &&
        a->back_write_mask == b->back_write_mask &&
        a->back_fail == b->back_fail && a->back_zfail == b->back_zfail &&
        a->back_zpass == b->back_zpass;
}

static u64 d3d12_pso_hash_bytes(u64 hash, const void* data, size_t size)
{
    const u8* bytes = (const u8*)data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* D3DCompile output is stable for the same HLSL, target, entry point, flags,
 * and compiler ABI. Keep shader blobs in the user's cache directory so the
 * first draw after a restart does not repeat HLSL compilation. */
typedef struct {
    char magic[8];
    u32 version;
    u32 header_size;
    u64 key_a;
    u64 key_b;
    u64 source_size;
    u64 bytecode_size;
    u64 bytecode_hash_a;
    u64 bytecode_hash_b;
} D3D12ShaderCacheHeader;

#define D3D12_SHADER_CACHE_MAX_SOURCE_BYTES   (64u * 1024u * 1024u)
#define D3D12_SHADER_CACHE_MAX_BYTECODE_BYTES (64u * 1024u * 1024u)

static void d3d12_shader_cache_digest(const void* bytecode, SIZE_T size,
                                      u64* hash_a, u64* hash_b)
{
    *hash_a = d3d12_pso_hash_bytes(14695981039346656037ull, bytecode, size);
    *hash_b = d3d12_pso_hash_bytes(7809847782465536322ull, bytecode, size);
}

static void d3d12_cache_mkdirs(char* path)
{
    if (!path || !*path) return;
    const size_t prefix = strlen(path) >= 3 && path[1] == ':' ? 3 : 0;
    for (char* p = path + prefix; *p; ++p) {
        if (*p != '\\' && *p != '/') continue;
        char saved = *p;
        *p = '\0';
        if (*path) CreateDirectoryA(path, NULL);
        *p = saved;
    }
    CreateDirectoryA(path, NULL);
}

static void d3d12_cache_directory_init(void)
{
    if (s_d3d.shader_cache_dir[0]) return;
    const char* override = getenv("PS3RECOMP_CACHE_DIR");
    if (override && *override) {
        snprintf(s_d3d.shader_cache_dir, sizeof(s_d3d.shader_cache_dir), "%s", override);
    } else {
        char root[MAX_PATH] = {0};
        DWORD n = GetEnvironmentVariableA("LOCALAPPDATA", root, (DWORD)sizeof(root));
        if (!n || n >= sizeof(root)) {
            n = GetTempPathA((DWORD)sizeof(root), root);
            if (!n || n >= sizeof(root)) return;
            if (root[n - 1] == '\\' || root[n - 1] == '/') root[n - 1] = '\0';
            snprintf(s_d3d.shader_cache_dir, sizeof(s_d3d.shader_cache_dir),
                     "%s\\ps3recomp\\cache", root);
        } else {
            snprintf(s_d3d.shader_cache_dir, sizeof(s_d3d.shader_cache_dir),
                     "%s\\ps3recomp\\cache", root);
        }
    }
    d3d12_cache_mkdirs(s_d3d.shader_cache_dir);
    snprintf(s_d3d.pipeline_library_path, sizeof(s_d3d.pipeline_library_path),
             "%s\\d3d12_pipeline_library_v1.bin", s_d3d.shader_cache_dir);
}

static void d3d12_shader_cache_key(const void* source, SIZE_T source_size,
                                  LPCSTR source_name, LPCSTR entry_point,
                                  LPCSTR target, UINT flags1, UINT flags2,
                                  u64* key_a, u64* key_b)
{
    u64 a = d3d12_pso_hash_bytes(14695981039346656037ull, source, source_size);
    u64 b = d3d12_pso_hash_bytes(7809847782465536322ull, source, source_size);
    const u64 length = (u64)source_size;
    a = d3d12_pso_hash_bytes(a, &length, sizeof(length));
    b = d3d12_pso_hash_bytes(b, &length, sizeof(length));
    const char* strings[] = {source_name ? source_name : "",
                             entry_point ? entry_point : "",
                             target ? target : ""};
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); ++i) {
        const size_t n = strlen(strings[i]) + 1;
        a = d3d12_pso_hash_bytes(a, strings[i], n);
        b = d3d12_pso_hash_bytes(b, strings[i], n);
    }
    a = d3d12_pso_hash_bytes(a, &flags1, sizeof(flags1));
    a = d3d12_pso_hash_bytes(a, &flags2, sizeof(flags2));
    b = d3d12_pso_hash_bytes(b, &flags2, sizeof(flags2));
    b = d3d12_pso_hash_bytes(b, &flags1, sizeof(flags1));
    *key_a = a;
    *key_b = b;
}

static int d3d12_shader_cache_read(const void* source, SIZE_T source_size,
                                   LPCSTR source_name, LPCSTR entry_point,
                                   LPCSTR target, UINT flags1, UINT flags2,
                                   ID3DBlob** code)
{
    d3d12_cache_directory_init();
    if (!s_d3d.shader_cache_dir[0] ||
        source_size > D3D12_SHADER_CACHE_MAX_SOURCE_BYTES) return 0;
    u64 key_a, key_b;
    d3d12_shader_cache_key(source, source_size, source_name, entry_point,
                           target, flags1, flags2, &key_a, &key_b);
    char path[MAX_PATH];
    snprintf(path, sizeof(path), "%s\\%016llX_%016llX.cso",
             s_d3d.shader_cache_dir, (unsigned long long)key_a,
             (unsigned long long)key_b);
    HANDLE file = CreateFileA(path, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER length;
    const u64 max_file_size = (u64)sizeof(D3D12ShaderCacheHeader) +
        (u64)source_size + D3D12_SHADER_CACHE_MAX_BYTECODE_BYTES;
    int valid = GetFileSizeEx(file, &length) &&
                length.QuadPart >= (LONGLONG)sizeof(D3D12ShaderCacheHeader) &&
                (u64)length.QuadPart <= max_file_size;
    BYTE* bytes = valid ? (BYTE*)malloc((size_t)length.QuadPart) : NULL;
    DWORD got = 0;
    if (!bytes || !ReadFile(file, bytes, (DWORD)length.QuadPart, &got, NULL) ||
        got != (DWORD)length.QuadPart) valid = 0;
    CloseHandle(file);
    if (!valid) { free(bytes); return 0; }
    D3D12ShaderCacheHeader header;
    memcpy(&header, bytes, sizeof(header));
    const u64 payload_size = (u64)length.QuadPart - sizeof(header);
    valid = memcmp(header.magic, "P3DXBC02", 8) == 0 && header.version == 2 &&
        header.header_size == sizeof(header) && header.key_a == key_a &&
        header.key_b == key_b && header.source_size == source_size &&
        header.source_size <= payload_size && header.bytecode_size > 0 &&
        header.bytecode_size <= D3D12_SHADER_CACHE_MAX_BYTECODE_BYTES &&
        header.bytecode_size == payload_size - header.source_size;
    if (valid)
        valid = memcmp(bytes + sizeof(header), source, source_size) == 0;
    const BYTE* bytecode = NULL;
    if (valid) {
        bytecode = bytes + sizeof(header) + (size_t)header.source_size;
        u64 hash_a, hash_b;
        d3d12_shader_cache_digest(bytecode, (SIZE_T)header.bytecode_size,
                                  &hash_a, &hash_b);
        valid = header.bytecode_hash_a == hash_a &&
                header.bytecode_hash_b == hash_b;
    }
    ID3DBlob* blob = NULL;
    if (valid && SUCCEEDED(D3DCreateBlob((SIZE_T)header.bytecode_size, &blob))) {
        memcpy(blob->lpVtbl->GetBufferPointer(blob), bytecode,
               (size_t)header.bytecode_size);
        *code = blob;
    } else {
        if (blob) blob->lpVtbl->Release(blob);
        valid = 0;
    }
    free(bytes);
    return valid;
}

static int d3d12_write_file_bytes(HANDLE file, const void* data, SIZE_T size)
{
    const BYTE* bytes = (const BYTE*)data;
    while (size) {
        DWORD part = size > 0x40000000u ? 0x40000000u : (DWORD)size;
        DWORD written = 0;
        if (!WriteFile(file, bytes, part, &written, NULL) || written != part)
            return 0;
        bytes += written;
        size -= written;
    }
    return 1;
}

static void d3d12_shader_cache_write(const void* source, SIZE_T source_size,
                                    LPCSTR source_name, LPCSTR entry_point,
                                    LPCSTR target, UINT flags1, UINT flags2,
                                    ID3DBlob* code)
{
    if (!code || source_size > D3D12_SHADER_CACHE_MAX_SOURCE_BYTES ||
        code->lpVtbl->GetBufferSize(code) == 0 ||
        code->lpVtbl->GetBufferSize(code) > D3D12_SHADER_CACHE_MAX_BYTECODE_BYTES)
        return;
    d3d12_cache_directory_init();
    if (!s_d3d.shader_cache_dir[0]) return;
    u64 key_a, key_b;
    d3d12_shader_cache_key(source, source_size, source_name, entry_point,
                           target, flags1, flags2, &key_a, &key_b);
    char path[MAX_PATH], temp[MAX_PATH];
    snprintf(path, sizeof(path), "%s\\%016llX_%016llX.cso",
             s_d3d.shader_cache_dir, (unsigned long long)key_a,
             (unsigned long long)key_b);
    snprintf(temp, sizeof(temp), "%s\\%016llX_%016llX.%lu.%lu.tmp",
             s_d3d.shader_cache_dir, (unsigned long long)key_a,
             (unsigned long long)key_b, GetCurrentProcessId(), GetCurrentThreadId());
    D3D12ShaderCacheHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "P3DXBC02", 8);
    header.version = 2;
    header.header_size = sizeof(header);
    header.key_a = key_a;
    header.key_b = key_b;
    header.source_size = source_size;
    header.bytecode_size = code->lpVtbl->GetBufferSize(code);
    d3d12_shader_cache_digest(code->lpVtbl->GetBufferPointer(code),
                              (SIZE_T)header.bytecode_size,
                              &header.bytecode_hash_a,
                              &header.bytecode_hash_b);
    HANDLE file = CreateFileA(temp, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY, NULL);
    if (file == INVALID_HANDLE_VALUE) return;
    int ok = d3d12_write_file_bytes(file, &header, sizeof(header)) &&
        d3d12_write_file_bytes(file, source, source_size) &&
        d3d12_write_file_bytes(file, code->lpVtbl->GetBufferPointer(code),
                               (SIZE_T)header.bytecode_size);
    if (ok) ok = FlushFileBuffers(file) != 0;
    CloseHandle(file);
    if (!ok || !MoveFileExA(temp, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        DeleteFileA(temp);
}

static HRESULT d3d12_compile_cached(
    LPCVOID source, SIZE_T source_size, LPCSTR source_name,
    const D3D_SHADER_MACRO* defines, ID3DInclude* include,
    LPCSTR entry_point, LPCSTR target, UINT flags1, UINT flags2,
    ID3DBlob** code, ID3DBlob** errors)
{
    if (code) *code = NULL;
    if (errors) *errors = NULL;
    if (!code || !source || !entry_point || !target) return E_INVALIDARG;
    if (!defines && !include) {
        if (d3d12_shader_cache_read(source, source_size, source_name,
                entry_point, target, flags1, flags2, code)) {
            s_perf_disk_shader_hits++;
            return S_OK;
        }
        s_perf_disk_shader_misses++;
    }
    HRESULT hr = D3DCompile(source, source_size, source_name, defines, include,
                            entry_point, target, flags1, flags2, code, errors);
    if (SUCCEEDED(hr) && *code && !defines && !include)
        d3d12_shader_cache_write(source, source_size, source_name, entry_point,
                                 target, flags1, flags2, *code);
    return hr;
}

static void d3d12_pipeline_cache_open(void)
{
    if (!s_d3d.device || s_d3d.pipeline_library) return;
    d3d12_cache_directory_init();
    if (!s_d3d.pipeline_library_path[0]) return;
    HRESULT hr = s_d3d.device->lpVtbl->QueryInterface(
        s_d3d.device, &s_iid_d3d12_device1, (void**)&s_d3d.device1);
    if (FAILED(hr) || !s_d3d.device1) {
        fprintf(stderr, "[D3D12] pipeline-library interface unavailable; using runtime PSO cache only\n");
        return;
    }
    InitializeCriticalSection(&s_d3d.pipeline_library_lock);
    s_d3d.pipeline_library_lock_ready = 1;
    void* saved = NULL;
    SIZE_T saved_size = 0;
    HANDLE file = CreateFileA(s_d3d.pipeline_library_path, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER length;
        if (GetFileSizeEx(file, &length) && length.QuadPart > 0 &&
            length.QuadPart <= 512ll * 1024 * 1024) {
            saved_size = (SIZE_T)length.QuadPart;
            saved = malloc(saved_size);
            DWORD got = 0;
            if (!saved || !ReadFile(file, saved, (DWORD)saved_size, &got, NULL) ||
                got != (DWORD)saved_size) {
                free(saved);
                saved = NULL;
                saved_size = 0;
            }
        }
        CloseHandle(file);
    }
    hr = s_d3d.device1->lpVtbl->CreatePipelineLibrary(
        s_d3d.device1, saved, saved_size, &s_iid_d3d12_pipeline_library,
        (void**)&s_d3d.pipeline_library);
    if (FAILED(hr) && saved) {
        fprintf(stderr, "[D3D12] serialized pipeline cache rejected (0x%08lX); starting a fresh cache\n", hr);
        free(saved);
        saved = NULL;
        saved_size = 0;
        hr = s_d3d.device1->lpVtbl->CreatePipelineLibrary(
            s_d3d.device1, NULL, 0, &s_iid_d3d12_pipeline_library,
            (void**)&s_d3d.pipeline_library);
    } else {
        if (SUCCEEDED(hr) && s_d3d.pipeline_library && saved && saved_size) {
            /* The D3D12 API retains the supplied serialized memory until the
             * pipeline-library interface is released. */
            s_d3d.pipeline_library_data = saved;
            s_d3d.pipeline_library_data_size = saved_size;
            saved = NULL;
        }
        free(saved);
    }
    if (FAILED(hr) || !s_d3d.pipeline_library) {
        fprintf(stderr, "[D3D12] pipeline library unavailable (0x%08lX); continuing without disk PSOs\n", hr);
        if (s_d3d.pipeline_library_lock_ready) {
            DeleteCriticalSection(&s_d3d.pipeline_library_lock);
            s_d3d.pipeline_library_lock_ready = 0;
        }
        s_d3d.device1->lpVtbl->Release(s_d3d.device1);
        s_d3d.device1 = NULL;
        return;
    }
    fprintf(stderr, "[D3D12] persistent D3D12 pipeline cache ready%s\n",
            saved_size ? " (loaded)" : " (new)");
}

static void d3d12_pipeline_cache_save(void)
{
    enum { MAX_PIPELINE_LIBRARY_BYTES = 512 * 1024 * 1024 };
    if (!s_d3d.pipeline_library ||
        InterlockedCompareExchange(&s_d3d.pipeline_library_dirty, 0, 0) == 0 ||
        !s_d3d.pipeline_library_path[0]) return;
    if (s_d3d.pipeline_library_lock_ready)
        EnterCriticalSection(&s_d3d.pipeline_library_lock);
    const LONG revision = InterlockedCompareExchange(&s_d3d.pipeline_library_dirty, 0, 0);
    SIZE_T size = s_d3d.pipeline_library->lpVtbl->GetSerializedSize(
        s_d3d.pipeline_library);
    if (!size || size > MAX_PIPELINE_LIBRARY_BYTES) {
        if (s_d3d.pipeline_library_lock_ready)
            LeaveCriticalSection(&s_d3d.pipeline_library_lock);
        return;
    }
    void* data = malloc(size);
    if (!data) {
        if (s_d3d.pipeline_library_lock_ready)
            LeaveCriticalSection(&s_d3d.pipeline_library_lock);
        return;
    }
    HRESULT hr = s_d3d.pipeline_library->lpVtbl->Serialize(
        s_d3d.pipeline_library, data, size);
    if (s_d3d.pipeline_library_lock_ready)
        LeaveCriticalSection(&s_d3d.pipeline_library_lock);
    if (FAILED(hr)) { free(data); return; }
    char temp[MAX_PATH];
    snprintf(temp, sizeof(temp), "%s.%lu.%lu.tmp",
             s_d3d.pipeline_library_path, GetCurrentProcessId(), GetCurrentThreadId());
    HANDLE file = CreateFileA(temp, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY, NULL);
    if (file != INVALID_HANDLE_VALUE) {
        int ok = d3d12_write_file_bytes(file, data, size);
        if (ok) ok = FlushFileBuffers(file) != 0;
        CloseHandle(file);
        if (ok && MoveFileExA(temp, s_d3d.pipeline_library_path,
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            /* Keep additions made after Serialize pending for the next save. */
            InterlockedCompareExchange(&s_d3d.pipeline_library_dirty, 0, revision);
            fprintf(stderr, "[D3D12] saved persistent PSO cache (%llu bytes)\n",
                    (unsigned long long)size);
        } else {
            DeleteFileA(temp);
        }
    }
    free(data);
}

static void d3d12_pipeline_library_name(
    const VPFPEntry* key, u32 input_mask,
    ID3DBlob* vsb, ID3DBlob* psb,
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc,
    wchar_t name[64])
{
    VPFPEntry stable = *key;
    /* VP cache slots and generation counters change with discovery order. The
     * bytecode and semantic RSX/PSO state below identify the reusable object. */
    stable.fp_addr = 0;
    stable.vs_idx = 0;
    stable.gen = 0;
    stable.key_hash = 0;
    stable.hash_next = 0;
    stable.pso = NULL;
    const size_t key_begin = offsetof(VPFPEntry, vs_idx);
    const size_t key_size = offsetof(VPFPEntry, pso) - key_begin;
    u64 a = d3d12_pso_hash_bytes(14695981039346656037ull,
                                 (const u8*)&stable + key_begin, key_size);
    u64 b = d3d12_pso_hash_bytes(7809847782465536322ull,
                                 (const u8*)&stable + key_begin, key_size);
#define D3D12_HASH_FIELD(field) do { \
    a = d3d12_pso_hash_bytes(a, &(field), sizeof(field)); \
    b = d3d12_pso_hash_bytes(b, &(field), sizeof(field)); \
} while (0)
    const void* vs_bytes = vsb->lpVtbl->GetBufferPointer(vsb);
    const SIZE_T vs_size = vsb->lpVtbl->GetBufferSize(vsb);
    const void* ps_bytes = psb->lpVtbl->GetBufferPointer(psb);
    const SIZE_T ps_size = psb->lpVtbl->GetBufferSize(psb);
    a = d3d12_pso_hash_bytes(a, vs_bytes, vs_size);
    b = d3d12_pso_hash_bytes(b, vs_bytes, vs_size);
    a = d3d12_pso_hash_bytes(a, ps_bytes, ps_size);
    b = d3d12_pso_hash_bytes(b, ps_bytes, ps_size);
    D3D12_HASH_FIELD(input_mask);
    D3D12_HASH_FIELD(desc->BlendState);
    D3D12_HASH_FIELD(desc->SampleMask);
    D3D12_HASH_FIELD(desc->RasterizerState);
    D3D12_HASH_FIELD(desc->DepthStencilState);
    D3D12_HASH_FIELD(desc->IBStripCutValue);
    D3D12_HASH_FIELD(desc->PrimitiveTopologyType);
    D3D12_HASH_FIELD(desc->NumRenderTargets);
    D3D12_HASH_FIELD(desc->RTVFormats);
    D3D12_HASH_FIELD(desc->DSVFormat);
    D3D12_HASH_FIELD(desc->SampleDesc);
    D3D12_HASH_FIELD(desc->NodeMask);
    D3D12_HASH_FIELD(desc->Flags);
#undef D3D12_HASH_FIELD
    char ascii[64];
    snprintf(ascii, sizeof(ascii), "rsx_fp_v1_%016llX_%016llX",
             (unsigned long long)a, (unsigned long long)b);
    size_t i = 0;
    for (; ascii[i] && i < 63; ++i) name[i] = (wchar_t)(unsigned char)ascii[i];
    name[i] = L'\0';
}

static HRESULT d3d12_pipeline_library_get_or_create(
    const wchar_t* name, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc,
    ID3D12PipelineState** pso)
{
    if (!s_d3d.device || !desc || !pso) return E_INVALIDARG;
    *pso = NULL;
    if (!s_d3d.pipeline_library)
        return s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
            s_d3d.device, desc, &IID_ID3D12PipelineState, (void**)pso);

    if (s_d3d.pipeline_library_lock_ready)
        EnterCriticalSection(&s_d3d.pipeline_library_lock);
    HRESULT load_hr = s_d3d.pipeline_library->lpVtbl->LoadGraphicsPipeline(
        s_d3d.pipeline_library, name, desc, &IID_ID3D12PipelineState,
        (void**)pso);
    if (SUCCEEDED(load_hr) && *pso) {
        s_perf_disk_pso_hits++;
        if (s_d3d.pipeline_library_lock_ready)
            LeaveCriticalSection(&s_d3d.pipeline_library_lock);
        return S_OK;
    }
    s_perf_disk_pso_misses++;
    if (s_d3d.pipeline_library_lock_ready)
        LeaveCriticalSection(&s_d3d.pipeline_library_lock);

    HRESULT hr = s_d3d.device->lpVtbl->CreateGraphicsPipelineState(
        s_d3d.device, desc, &IID_ID3D12PipelineState, (void**)pso);
    if (FAILED(hr) || !*pso) return FAILED(hr) ? hr : E_FAIL;

    /* Another render worker may have created and stored this same PSO while
     * we were compiling it. Serialize library calls and prefer that entry. */
    if (s_d3d.pipeline_library_lock_ready)
        EnterCriticalSection(&s_d3d.pipeline_library_lock);
    ID3D12PipelineState* raced = NULL;
    load_hr = s_d3d.pipeline_library->lpVtbl->LoadGraphicsPipeline(
        s_d3d.pipeline_library, name, desc, &IID_ID3D12PipelineState,
        (void**)&raced);
    if (SUCCEEDED(load_hr) && raced) {
        (*pso)->lpVtbl->Release(*pso);
        *pso = raced;
    } else {
        HRESULT store_hr = s_d3d.pipeline_library->lpVtbl->StorePipeline(
            s_d3d.pipeline_library, name, *pso);
        if (SUCCEEDED(store_hr))
            InterlockedIncrement(&s_d3d.pipeline_library_dirty);
    }
    if (s_d3d.pipeline_library_lock_ready)
        LeaveCriticalSection(&s_d3d.pipeline_library_lock);
    return S_OK;
}

static void d3d12_vp_fp_hash_remove(u32 entry_index)
{
    if (entry_index >= (u32)s_d3d.vp_fp_n)
        return;

    VPFPEntry* entry = &s_d3d.vp_fp[entry_index];
    const u32 bucket = (u32)entry->key_hash & (VP_FP_HASH_BUCKETS - 1u);
    u32* link = &s_d3d.vp_fp_hash[bucket];
    while (*link) {
        const u32 current = *link - 1u;
        if (current == entry_index) {
            *link = entry->hash_next;
            entry->hash_next = 0;
            return;
        }
        if (current >= (u32)s_d3d.vp_fp_n)
            return;
        link = &s_d3d.vp_fp[current].hash_next;
    }
}

static u64 d3d12_fp_ps_source_hash(const char* source, u32 size)
{
    u64 hash = 14695981039346656037ull;
    for (u32 i = 0; i < size; i++) {
        hash ^= (u8)source[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* The shader text is fully specialized before this call. Keying on the exact
 * final text makes the cache safe across all RSX state/debug rewrites while
 * allowing separate D3D12 PSOs to share compiled pixel bytecode. */
static ID3DBlob* d3d12_fp_ps_blob(const char* source, ID3DBlob** errors,
                                  HRESULT* result)
{
    const size_t source_size = strlen(source);
    if (source_size > 0xFFFFFFFFu) {
        if (result) *result = E_INVALIDARG;
        return NULL;
    }
    const u32 size = (u32)source_size;
    const u64 hash = d3d12_fp_ps_source_hash(source, size);
    if (errors) *errors = NULL;
    if (result) *result = S_OK;

    for (int i = 0; i < s_d3d.vp_fp_blob_n; i++) {
        VPFPBlobEntry* entry = &s_d3d.vp_fp_blob[i];
        if (entry->source_hash == hash && entry->source_size == size &&
            entry->source && memcmp(entry->source, source, size) == 0) {
            entry->last_use = ++s_d3d.vp_fp_blob_clock;
            entry->blob->lpVtbl->AddRef(entry->blob);
            s_perf_ps_blob_hits++;
            return entry->blob;
        }
    }

    s_perf_ps_blob_misses++;
    ID3DBlob* blob = NULL;
    HRESULT hr = d3d12_compile_cached(source, source_size, "guest_fp", NULL, NULL,
                            "main", "ps_5_0", 0, 0, &blob, errors);
    if (result) *result = hr;
    if (FAILED(hr) || !blob) {
        if (blob) blob->lpVtbl->Release(blob);
        return NULL;
    }

    char* source_copy = (char*)malloc(source_size + 1);
    if (!source_copy) return blob;
    memcpy(source_copy, source, source_size + 1);

    int slot;
    if (s_d3d.vp_fp_blob_n < VP_FP_BLOB_CACHE) {
        slot = s_d3d.vp_fp_blob_n++;
    } else {
        slot = 0;
        for (int i = 1; i < s_d3d.vp_fp_blob_n; i++)
            if (s_d3d.vp_fp_blob[i].last_use <
                s_d3d.vp_fp_blob[slot].last_use)
                slot = i;
        VPFPBlobEntry* old = &s_d3d.vp_fp_blob[slot];
        if (old->blob) old->blob->lpVtbl->Release(old->blob);
        free(old->source);
    }
    VPFPBlobEntry* entry = &s_d3d.vp_fp_blob[slot];
    entry->source_hash = hash;
    entry->source_size = size;
    entry->source = source_copy;
    entry->blob = blob;
    entry->last_use = ++s_d3d.vp_fp_blob_clock;
    blob->lpVtbl->AddRef(blob); /* cache owns one reference; caller owns one */
    return blob;
}

/* RSX FOGC is a generated fragment attribute: x is the interpolated fog
 * coordinate, y is the fixed-function fog factor, and z/w are zero.
 * RPCS3_REFERENCE_REVIEWED: fog input semantics and mode equations checked
 * against RSXFragmentPrologue.glsl; independently emitted HLSL below. */
static int d3d12_hlsl_apply_fog(char* source, size_t capacity, u32 mode,
                                  u32 param0_bits, u32 param1_bits)
{
    if (!strstr(source, "input.fog")) return 1;
    const int absolute = mode == 0x0802u || mode == 0x0803u || mode == 0x0804u;
    const int linear = mode == 0x2601u || mode == 0x0804u;
    const int exponential = mode == 0x0800u || mode == 0x0802u;
    const int squared = mode == 0x0801u || mode == 0x0803u;
    char prefix[768];
    int length = snprintf(prefix, sizeof(prefix),
        "\n    float _rsx_fog_coord = input.fog.x;\n"
        "    float _rsx_fog_distance = %s;\n"
        "    float _rsx_fog_term = asfloat(0x%08Xu) * _rsx_fog_distance"
        " + asfloat(0x%08Xu) - %s;\n",
        absolute ? "abs(_rsx_fog_coord)" : "_rsx_fog_coord",
        param1_bits, param0_bits, linear ? "1.0" : "1.5");
    if (length < 0 || (size_t)length >= sizeof(prefix)) return 0;
    const char* factor = linear ? "saturate(_rsx_fog_term)"
        : exponential ? "saturate(exp(11.084 * _rsx_fog_term))"
        : squared ? "saturate(exp(-22.174681 * _rsx_fog_term * _rsx_fog_term))"
        : "0.0";
    int suffix = snprintf(prefix + length, sizeof(prefix) - (size_t)length,
        "    input.fog = float4(_rsx_fog_coord, %s, 0.0, 0.0);\n", factor);
    if (suffix < 0 || (size_t)suffix >= sizeof(prefix) - (size_t)length) return 0;
    size_t inserted = (size_t)length + (size_t)suffix;
    char* entry = strstr(source, "main(PSInput input)");
    char* brace = entry ? strchr(entry, '{') : NULL;
    if (!brace || strlen(source) + inserted + 1 > capacity) return 0;
    char* body = brace + 1;
    memmove(body + inserted, body, strlen(body) + 1);
    memcpy(body, prefix, inserted);
    return 1;
}

static ID3D12PipelineState* vp_get_fp_pso(int vs_idx, u32 fp_addr, u32 blend, int nrt,
                                          DXGI_FORMAT rtfmt, DXGI_FORMAT dsvfmt,
                                          int exp32, u32 cmask, u32 color_mask_mrt,
                                          u32 blend_enable_mrt, u32 cull,
                                          int depth_test, int depth_write, u32 depth_func,
                                          u32 zmin_max_control,
                                           u32 cube_mask, u32 unnorm_mask,
                                           const u32 unnorm_dims[RSX_TEXTURE_UNITS][3],
                                           u32 shadow_mask,
                                          u64 shadow_compare_key,
                                          const rsx_fp_shadow_desc shadow[RSX_TEXTURE_UNITS],
                                          const u8 gamma_masks[RSX_TEXTURE_UNITS],
                                          int srgb_output,
                                          u32 depth_rgba_mask,
                                          int poly_fill,
                                          u32 poly_scale_bits, u32 poly_bias_bits,
                                          u32 fog_mode, u32 fog_param0_bits,
                                          u32 fog_param1_bits, u32 shader_window,
                                          u32 texcoord_control_mask, u32 anti_aliasing_control,
                                          int flat_shading, int two_sided_lighting,
                                          const RSXStencilState* stencil,
                                          D3D12FPSamplerLayout* sampler_layout_out,
                                          const D3D12DrawRecord* draw)
{
    static const RSXStencilState stencil_off = {0};
    static const u8 no_gamma[RSX_TEXTURE_UNITS] = {0};
    static const u32 no_unnorm_dims[RSX_TEXTURE_UNITS][3] = {{0}};
    if (!gamma_masks) gamma_masks = no_gamma;
    if (!unnorm_dims) unnorm_dims = no_unnorm_dims;
    if (sampler_layout_out) {
        memset(sampler_layout_out, 0, sizeof(*sampler_layout_out));
        memset(sampler_layout_out->normal_slot, 0xFF,
               sizeof(sampler_layout_out->normal_slot));
        memset(sampler_layout_out->shadow_slot, 0xFF,
               sizeof(sampler_layout_out->shadow_slot));
    }
    if (!stencil) stencil = &stencil_off;
    if (nrt < 0) nrt = 0; if (nrt > 4) nrt = 4;
    if (nrt > 0 && rtfmt == 0) rtfmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (dsvfmt != DXGI_FORMAT_D16_UNORM &&
        dsvfmt != DXGI_FORMAT_D24_UNORM_S8_UINT)
        dsvfmt = DXGI_FORMAT_D24_UNORM_S8_UINT;
    extern uint8_t* vm_base;
    extern u32 cellGcmResolveOffset(u32);
    if (!fp_addr || !vm_base) return NULL;
    { static int _off=-1; if(_off<0)_off=getenv("FP_OFF")?1:0; if(_off) return NULL; }
    /* Resolve the VS: the draw's cached per-VP blob, else the primary. */
    ID3DBlob* vsb = NULL; u32 vs_hash = 0;
    if (vs_idx >= 0 && vs_idx < s_d3d.vp_vs_n) {
        vsb = s_d3d.vp_vs[vs_idx].vs; vs_hash = s_d3d.vp_vs[vs_idx].hash;
    }
    if (!vsb) { vsb = s_d3d.vp_vs_blob; vs_idx = -1; }
    if (!vsb) return NULL;
    const int raw_fetch = draw && draw->vp_raw_bytes;
    if (raw_fetch) {
        if (vs_idx < 0 || !s_d3d.vp_vs[vs_idx].raw_vs) return NULL;
        vsb = s_d3d.vp_vs[vs_idx].raw_vs;
    }

    /* SET_SHADER_PROGRAM low bits = location+1 (same as textures): 1 = LOCAL
     * (VRAM), 2 = MAIN (gcm/cube emits 0x00B90001 for its VRAM-resident FP). */
    extern u32 cellGcmResolveLocated(int local, u32 offset);
    u32 off = cellGcmResolveLocated((fp_addr & 0x3u) == 1, fp_addr & ~0x3u);
    if (off == 0xFFFFFFFFu) return NULL;

    /* Content hash: inline constants are patched in place per frame (wave's
     * stamp), so identity is the BYTES, not the address. */
    /* Hash the CODE, not the constants: with fp_k[] hoisted into the per-draw
     * constant buffer the compiled shader is invariant under constant changes,
     * so hashing them made the pipeline key move every draw. */
    u32 uhash, sampled_units;
    {
        u32 usz = 0;
        if (!fp_buffered_constants()) {
            usz = rsx_fp_program_size(vm_base + off, 4096);
            if (usz == 0) usz = 64;
            uhash = 2166136261u;
            const u8* up = vm_base + off;
            for (u32 i = 0; i < usz; i++) { uhash ^= up[i]; uhash *= 16777619u; }
            sampled_units = rsx_fp_texture_units(vm_base + off, 4096);
        } else {
            const rsx_fp_metadata* metadata = rsx_fp_metadata_get(&s_fp_metadata, vm_base + off);
            if (metadata) {
                uhash = metadata->hash; usz = metadata->bytes; sampled_units = metadata->units;
            } else uhash = rsx_fp_code_hash_ex(vm_base + off, 4096, &usz, &sampled_units);
        }
        s_perf_pso_hashbytes += (int)(usz ? usz : 64u);
    }
    /* Only texture units actually read by this FP affect generated HLSL and
     * therefore belong in the shader cache key. */
    u32 active_unnorm_mask = unnorm_mask & sampled_units;
    active_unnorm_mask &= ~cube_mask;
    if (active_unnorm_mask) {
        uhash = (uhash ^ active_unnorm_mask) * 16777619u;
        for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++) {
            if (!(active_unnorm_mask & (1u << u))) continue;
            if (!unnorm_dims[u][0] || !unnorm_dims[u][1]) return NULL;
            uhash = (uhash ^ unnorm_dims[u][0]) * 16777619u;
            uhash = (uhash ^ unnorm_dims[u][1]) * 16777619u;
        }
    }

    s_perf_pso_calls++;
    for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++)
        if (depth_rgba_mask & (1u << u))
            uhash = (uhash ^ rsx_remap_to_d3d(draw->tex[u].ctrl1,
                        draw->tex[u].fmt & 0x9Fu)) * 16777619u;
    /* Keyed on the code, not on where it lives. The compiled shader reads its
     * inline constants from the draw's own fp_addr (fp_k[], refilled per draw),
     * so a program's pipeline is the same wherever the title puts it. God of
     * War HD uploads its fragment programs into a ring every frame: keyed on the
     * address, every draw of every frame missed and went through a full
     * decompile + D3DCompile (179 draws -> seconds per frame on its menu). */
    /* Hash the complete pipeline state, then verify the key on a bucket hit.
     * fp_addr is excluded: buffered constants are recorded separately per draw.
     * Zero initialization makes structure padding deterministic. */
    VPFPEntry key;
    memset(&key, 0, sizeof key);
    key.vs_idx = vs_idx;
    key.vs_hash = vs_hash;
    key.raw_vertex_fetch = raw_fetch;
    key.gen = s_d3d.vp_gen;
    key.blend = blend;
    key.color_mask_mrt = color_mask_mrt;
    key.blend_enable_mrt = blend_enable_mrt;
    memcpy(key.gamma_masks, gamma_masks, sizeof key.gamma_masks);
    key.srgb_output = srgb_output;
    key.depth_test = depth_test;
    key.depth_write = depth_write;
    key.depth_func = depth_func;
    key.zmin_max_control = zmin_max_control;
    key.stencil = *stencil;
    key.nrt = nrt;
    key.rtfmt = (u32)rtfmt;
    key.dsvfmt = (u32)dsvfmt;
    key.exp32 = exp32;
    key.ucode_hash = uhash;
    key.cmask = cmask;
    key.cull = cull;
    key.cube_mask = cube_mask;
    key.shadow_mask = shadow_mask;
    key.shadow_compare_key = shadow_compare_key;
    memcpy(key.shadow, shadow, sizeof key.shadow);
    key.depth_rgba_mask = depth_rgba_mask;
    key.poly_fill = poly_fill;
    key.poly_scale_bits = poly_scale_bits;
    key.poly_bias_bits = poly_bias_bits;
    key.fog_mode = fog_mode;
    key.fog_param0_bits = fog_param0_bits;
    key.fog_param1_bits = fog_param1_bits;
    key.shader_window = shader_window;
    key.texcoord_control_mask = texcoord_control_mask;
    key.anti_aliasing_control = anti_aliasing_control;
    key.flat_shading = flat_shading;
    key.two_sided_lighting = two_sided_lighting;
    const size_t key_begin = offsetof(VPFPEntry, vs_idx);
    const size_t key_size = offsetof(VPFPEntry, sampler_layout) - key_begin;
    const u64 key_hash = d3d12_pso_hash_bytes(14695981039346656037ull,
                                            (const u8*)&key + key_begin, key_size);
    const u32 bucket = (u32)key_hash & (VP_FP_HASH_BUCKETS - 1u);
    for (u32 link = s_d3d.vp_fp_hash[bucket]; link; ) {
        VPFPEntry* cached = &s_d3d.vp_fp[link - 1u];
        if (cached->key_hash == key_hash &&
            memcmp((const u8*)cached + key_begin,
                   (const u8*)&key + key_begin, key_size) == 0) {
            if (sampler_layout_out) *sampler_layout_out = cached->sampler_layout;
            return cached->pso;
        }
        link = cached->hash_next;
    }
    s_perf_pso_miss++;      /* falls through to a full decompile + D3DCompile */
    /* PSOMISSDBG=1: on a miss, name the key field that differs from an existing
     * entry for the same program. "the cache misses" is not actionable; "it
     * misses on ucode_hash" is. */
    /* HASHDBG=1: how many DISTINCT ucode hashes each program produces. Inline
     * constants are patched into the shader bytes, so a program whose constants
     * animate yields a new hash -- and a new pipeline -- every time. */
    { static int hd = -1;
      if (hd < 0) { const char* e = getenv("HASHDBG"); hd = e ? atoi(e) : 0; }
      if (hd) { static u32 fps[32]; static u32 cnt[32]; static int nf = 0;
        int idx = -1;
        for (int i = 0; i < nf; i++) if (fps[i] == fp_addr) idx = i;
        if (idx < 0 && nf < 32) { idx = nf++; fps[idx] = fp_addr; cnt[idx] = 0; }
        if (idx >= 0) { cnt[idx]++;
          if ((cnt[idx] % 200) == 0)
            fprintf(stderr, "[HASHDBG] fp=0x%X has produced %u distinct compiles%c",
                    fp_addr, cnt[idx], 10); } } }
    { static int md = -1;
      if (md < 0) { const char* e = getenv("PSOMISSDBG"); md = e ? atoi(e) : 0; }
      if (md) { static int n = 0;
        for (int i = 0; i < s_d3d.vp_fp_n && n < 12; i++) {
            VPFPEntry* q = &s_d3d.vp_fp[i];
            if (q->fp_addr != fp_addr) continue;
            n++;
            fprintf(stderr, "[PSOMISS] fp=0x%X differs:%s%s%s%s%s%s%s%s%c", fp_addr,
                    q->vs_idx != vs_idx        ? " vs_idx"     : "",
                    q->vs_hash != vs_hash      ? " vs_hash"    : "",
                    q->gen != s_d3d.vp_gen     ? " gen"        : "",
                    q->blend != blend          ? " blend"      : "",
                    q->nrt != nrt              ? " nrt"        : "",
                    q->rtfmt != (u32)rtfmt     ? " rtfmt"      : "",
                    q->ucode_hash != uhash     ? " ucode_hash" : "",
                    q->cube_mask != cube_mask  ? " cube_mask"  : "", 10);
            break;
        } } }
    /* CUBEKEYDBG=1: which (program, cube mask) pairs are being compiled. If one
     * program shows several masks, the mask is unstable and is the reason the
     * pipeline cache thrashes with cube support on. */
    { static int ck = -1;
      if (ck < 0) { const char* e = getenv("CUBEKEYDBG"); ck = e ? atoi(e) : 0; }
      if (ck) { static u32 seen[64][2]; static int ns = 0; int known = 0;
        for (int i = 0; i < ns; i++)
            if (seen[i][0] == fp_addr && seen[i][1] == cube_mask) known = 1;
        if (!known && ns < 64) { seen[ns][0] = fp_addr; seen[ns][1] = cube_mask; ns++;
            fprintf(stderr, "[CUBEKEY] fp=0x%X cube_mask=0x%X (distinct pairs=%d)%c",
                    fp_addr, cube_mask, ns, 10); } } }
    static char hlsl[RSX_FP_HLSL_BUFFER_SIZE];
    /* The third parameter is the SET_SHADER_CONTROL word; its 0x40 bit picks
     * the export registers (r0/r2/r3/r4 for 32-bit, h0/h4/h6/h8 for 16-bit).
     * Each draw records that bit (fp_exp32, also part of the pipeline key), so
     * pass it rather than letting the decompiler guess from "was r0 written".
     * The guess fails on a program that uses r0 as a temp and exports through
     * h0: God of War HD's 2x-AA resolve computes (B-A)/2 in r0 and the average
     * in h0, and with r0 exported every 3D frame came out black. The depth
     * export bits aren't recorded, so they stay off as before.
     * RSX_FP_CTRL_AUTO=1 restores the guess. */
    static int ctrl_auto = -1;
    if (ctrl_auto < 0) ctrl_auto = getenv("RSX_FP_CTRL_AUTO") ? 1 : 0;
    u32 fp_ctrl = ctrl_auto ? RSX_FP_CTRL_AUTO
                            : (exp32 ? CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS : 0u);
    u32 constant_count = 0;
    int n = fp_buffered_constants()
        ? rsx_fp_decompile_buffered_shadow_color_ex(vm_base + off, 4096,
            fp_ctrl, cube_mask, shadow, gamma_masks, hlsl, sizeof hlsl,
            &constant_count)
        : rsx_fp_decompile_shadow_desc_color_ex(vm_base + off, 4096,
            fp_ctrl, cube_mask, shadow, gamma_masks, hlsl, sizeof hlsl);
    if (constant_count > VP_FP_MAX_CONSTANTS) return NULL;
    if (n > 0 && !d3d12_hlsl_apply_fog(hlsl, sizeof hlsl,
            fog_mode, fog_param0_bits, fog_param1_bits)) return NULL;
    if (n > 0 && rsx_fp_apply_unnorm_scale(hlsl, sizeof hlsl,
            active_unnorm_mask, unnorm_dims, cube_mask, 0u) < 0) return NULL;
    if (n > 0 && srgb_output && rsx_fp_apply_srgb_output(hlsl, sizeof hlsl) < 0)
        return NULL;
    if (n <= 0) { static int _e=0; if(PS3_LOG_FIRST(_e, 16)) printf("[FP] decompile fail (fp=0x%08X)\n", fp_addr); return NULL; }
    /* FP_LIST=1: every program the title actually compiles, with its size. A
     * fragment program that hangs the GPU shows up here as an implausible
     * instruction count long before it shows up as a TDR. */
    { static int fl = -1; if (fl < 0) fl = getenv("FP_LIST") ? 1 : 0;
      if (fl) fprintf(stderr, "[FPLIST] fp=0x%08X instrs=%d hlsl=%u bytes%c",
                      fp_addr, n, (unsigned)strlen(hlsl), 10); }
    /* FP_DUMP=<n>: the first n compiled programs (1 = the old default of 4). */
    if (getenv("FP_DUMP")) { static int _d=0, _cap=-1;
      if (_cap < 0) { _cap = atoi(getenv("FP_DUMP")); if (_cap <= 1) _cap = 4; }
      if (PS3_LOG_FIRST(_d, _cap)) {
        FILE* f = fopen("fp_dump.hlsl", _d==1 ? "w" : "a");
        if (f) {
            const u8* uc = vm_base + off;
            fprintf(f, "/* fp_addr=0x%08X vmoff=0x%08X raw:", fp_addr, off);
            for (int _b = 0; _b < 32; _b++) fprintf(f, " %02X", uc[_b]);
            fprintf(f, " */\n%s\n", hlsl); fclose(f);
        } } }

    /* FP_PICK=<hex fp_addr>: write just that program, whatever order it compiles
     * in. fp_dump.hlsl keeps only the first four, same trap as vp2_dump.hlsl. */
    { static const char* fpk = (const char*)1; static u32 wantf = 0; static int done = 0;
      if (fpk == (const char*)1) { fpk = getenv("FP_PICK");
                                   wantf = fpk ? (u32)strtoul(fpk, NULL, 16) : 0; }
      if (wantf && !done && fp_addr == wantf) { done = 1;
          FILE* f = fopen("duck_fp.hlsl", "w");
          if (f) { fprintf(f, "/* fp_addr=0x%08X, %d instrs */%s", fp_addr, n, hlsl);
                   fclose(f); }
          fprintf(stderr, "[DUCKFP] wrote duck_fp.hlsl (fp=0x%08X, %d instrs)%c",
                  fp_addr, n, 10);
      } }

    /* Debug FP_ONE=<hex fp_addr>: force that program's colour output to
     * all-ones (e.g. wave's colour-detect -> full mask -> island borders
     * everywhere -> the water sim must visibly radiate if it works). */
    { const char* f1 = getenv("FP_ONE");
      if (f1 && (u32)strtoul(f1, NULL, 16) == fp_addr) {
          /* Match the assignment, not one exact spelling of it: the final
           * register is r[N] for some programs and h[N] for others, and the old
           * fixed "= r[0];" anchor silently did nothing for every h[] shader --
           * so this switch appeared to have no effect and was useless for
           * identifying which program draws what. Rewrite the expression with a
           * proper length-changing splice. */
          char* rp = strstr(hlsl, "_po.c0 = ");
          if (rp) {
              char* semi = strchr(rp, ';');
              if (semi) {
                  static const char rep[] = "_po.c0 = (1.0).xxxx";
                  size_t oldlen = (size_t)(semi - rp), newlen = sizeof(rep) - 1;
                  size_t tail = strlen(semi) + 1;
                  if ((rp - hlsl) + newlen + tail < sizeof(hlsl)) {
                      memmove(rp + newlen, semi, tail);
                      memcpy(rp, rep, newlen);
                  }
                  (void)oldlen;
              }
          }
      } }
    /* FP_TEX=1: make every program output its unit-0 texture sample verbatim.
     * Separates "the texture is wrong" from "the shading tints it" in one run,
     * without having to identify which program draws which surface first. */
    /* FP_SHOW=<hlsl expr>: replace a program's colour output with an arbitrary
     * expression (e.g. FP_SHOW=h[1]) so an intermediate register can be looked
     * at directly. Restrict with FP_TEX_FP=<hex>. Reading a 69-instruction
     * shader to guess which term carries a bad value is slower and less certain
     * than displaying the terms. */
    /* FP_KILL=<hex fp_addr>: make that program discard every fragment, so what
     * it covers becomes visible. Distinguishes "this surface shades black" from
     * "this surface is correct and something else is missing behind it". */
    { const char* kf = getenv("FP_KILL");
      if (kf && (u32)strtoul(kf, NULL, 16) == fp_addr) {
          char* rp = strstr(hlsl, "PSOut _po;");
          if (rp) {
              const char* ins = "discard; ";
              size_t nl = strlen(ins), tail = strlen(rp) + 1;
              if ((size_t)(rp - hlsl) + nl + tail < sizeof(hlsl)) {
                  memmove(rp + nl, rp, tail);
                  memcpy(rp, ins, nl);
                  fprintf(stderr, "[FPKILL] fp=0x%X discards%c", fp_addr, 10);
              }
          }
      } }
    { const char* sh = getenv("FP_SHOW");
      const char* only = getenv("FP_TEX_FP");
      if (sh && (!only || (u32)strtoul(only, NULL, 16) == fp_addr)) {
          char* rp = strstr(hlsl, "_po.c0 = ");
          if (rp) { char* semi = strchr(rp, ';');
            if (semi) {
                char rep[256];
                snprintf(rep, sizeof rep, "_po.c0 = float4((%s).xyz, 1.0)", sh);
                size_t newlen = strlen(rep), tail = strlen(semi) + 1;
                if ((size_t)(rp - hlsl) + newlen + tail < sizeof(hlsl)) {
                    memmove(rp + newlen, semi, tail);
                    memcpy(rp, rep, newlen);
                    fprintf(stderr, "[FPSHOW] fp=0x%X output replaced with %s%c",
                            fp_addr, sh, 10);
                }
            } }
      } }
    /* Retarget the samplers the guest bound as CUBE textures. The decompiler
     * always emits 2D samplers; a cube is sampled with a 3-component direction
     * and no texel-scale, and the emitted coordinate always ends in the fixed
     * tail ".xy * rsx_texscale[N].xy", so both edits are exact replacements. */
    if (cube_mask) {
        for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
            if (!(cube_mask & (1u << _u))) continue;
            char f1[64], r1[64], f2[64], r2[64];
            snprintf(f1, sizeof f1, "Texture2D    rsx_tex%d", _u);
            snprintf(r1, sizeof r1, "TextureCube  rsx_tex%d", _u);
            hlsl_replace_all(hlsl, sizeof hlsl, f1, r1);
            snprintf(f1, sizeof f1, "Texture2D rsx_tex%d", _u);
            snprintf(r1, sizeof r1, "TextureCube rsx_tex%d", _u);
            hlsl_replace_all(hlsl, sizeof hlsl, f1, r1);
            snprintf(f2, sizeof f2, ".xy * rsx_texscale[%d].xy", _u);
            snprintf(r2, sizeof r2, ".xyz");
            hlsl_replace_all(hlsl, sizeof hlsl, f2, r2);
        }
        { static int _n = 0; if (PS3_LOG_FIRST(_n, 4))
            fprintf(stderr, "[CUBE] fp=0x%X compiled with cube units mask 0x%X%c",
                    fp_addr, cube_mask, 10); }
    }
    if (n > 0 && !d3d12_hlsl_unpack_depth_samples(
            hlsl, sizeof(hlsl), depth_rgba_mask, draw)) n = -1;
    /* FP_IDCOLOR=1: give every fragment program a distinct flat colour derived
     * from its address. One frame then shows which program paints which surface,
     * instead of one run per candidate to test them by elimination. */
    if (getenv("FP_IDCOLOR")) {
        char* rp = strstr(hlsl, "_po.c0 = ");
        if (rp) {
            char* semi = strchr(rp, ';');
            if (semi) {
                /* Mix properly: every fp address in a title tends to share low
                 * bits (they all end 0x01/0x81 here), so a single multiply left
                 * one channel constant across all programs and made distinct
                 * shaders look like the same colour. */
                u32 hsh = fp_addr * 2654435761u;
                hsh ^= hsh >> 13; hsh *= 0x5BD1E995u; hsh ^= hsh >> 15;
                /* Force the top bit on in each channel: a hash that happens to
                 * land near black is indistinguishable from "this surface was
                 * never painted", which is the exact question the mode exists
                 * to answer. */
                float cr = (float)(((hsh >> 16) & 0x7F) | 0x80) / 255.0f;
                float cg = (float)(((hsh >>  8) & 0x7F) | 0x80) / 255.0f;
                float cb = (float)(((hsh      ) & 0x7F) | 0x80) / 255.0f;
                char rep[128];
                snprintf(rep, sizeof rep, "_po.c0 = float4(%.3f,%.3f,%.3f,1.0)", cr, cg, cb);
                size_t newlen = strlen(rep), tail = strlen(semi) + 1;
                if ((size_t)(rp - hlsl) + newlen + tail < sizeof(hlsl)) {
                    memmove(rp + newlen, semi, tail);
                    memcpy(rp, rep, newlen);
                    fprintf(stderr, "[FPID] fp=0x%X -> rgb(%.0f,%.0f,%.0f)%c",
                            fp_addr, cr*255, cg*255, cb*255, 10);
                }
            }
        }
    }
    /* FP_TEX=<unit>: output that unit's sample verbatim. FP_TEX_FP=<hex> limits
     * it to one program, so one surface can be inspected without repainting the
     * whole scene. FP_TEX_UV=1 samples with the same texcoord the program uses
     * for that unit rather than tc0. */
    if (getenv("FP_TEX") && (!getenv("FP_TEX_FP") ||
        (u32)strtoul(getenv("FP_TEX_FP"), NULL, 16) == fp_addr)) {
        int _un = atoi(getenv("FP_TEX"));
        if (_un < 0 || _un > 3) _un = 0;
        const int _sampler = _un == 0 ? 0 : 1;
        char* rp = strstr(hlsl, "_po.c0 = ");
        if (rp) {
            char* semi = strchr(rp, ';');
            if (semi) {
                char rep[160];
                /* FP_TEX_UV=1: sample at a fixed centre UV instead of the
                 * interpolated texcoord. If the texture is populated but the
                 * texcoords never arrive, the normal path samples texel (0,0)
                 * -- usually black -- and looks exactly like an empty texture.
                 * This tells the two apart. */
                if (getenv("FP_TEX_UV"))
                    snprintf(rep, sizeof rep,
                             "_po.c0 = rsx_tex%d.Sample(rsx_samp%d, float2(0.5, 0.5))",
                             _un, _sampler);
                else
                snprintf(rep, sizeof rep,
                         "_po.c0 = rsx_tex%d.Sample(rsx_samp%d, input.tc0.xy * rsx_texscale[%d].xy)",
                         _un, _sampler, _un);
                size_t newlen = strlen(rep), tail = strlen(semi) + 1;
                fprintf(stderr, "[FP_TEX] program 0x%X rewritten to sample unit %d%c",
                        fp_addr, _un, 10);
                if ((size_t)(rp - hlsl) + newlen + tail < sizeof(hlsl)) {
                    memmove(rp + newlen, semi, tail);
                    memcpy(rp, rep, newlen);
                }
            }
        }
    }
    /* FP_FORCE=1: replace the translated body with solid magenta -- isolates
     * geometry/transform problems from texture/blend problems. */
    if (getenv("FP_FORCE")) {
        char* rp = strstr(hlsl, "return r[0];");
        if (rp) memcpy(rp, "return float4(1,0,1,1)", 23);
    }
    if (fp_buffered_constants()) {
        int mrt = strstr(hlsl, "PSOutput main(") != NULL;
        if (!hlsl_replace_all(hlsl, sizeof(hlsl), " main(", " _rsx_fp_main("))
            return NULL;
        char wrapper[1024];
        int used = snprintf(wrapper, sizeof(wrapper),
            "\n%s main(PSInput input)%s{%s o=_rsx_fp_main(input);"
            "if(fp_alpha.y>0.5){float a=%s;float r=fp_alpha.x;"
            "int f=(int)fp_alpha.z;"
            "bool alpha_pass=f==7||(f==1&&a<r)||(f==2&&a==r)||"
            "(f==3&&a<=r)||(f==4&&a>r)||(f==5&&a!=r)||(f==6&&a>=r);"
            "if(!alpha_pass)discard;}return o;}\n",
            mrt ? "PSOutput" : "float4", mrt ? "" : " : SV_TARGET0",
            mrt ? "PSOutput" : "float4",
            mrt ? "o.t0.a" : "o.a");
        if (used < 0 || strlen(hlsl) + (size_t)used + 1 > sizeof(hlsl)) return NULL;
        strcat(hlsl, wrapper);
    }
    if (!d3d12_hlsl_ensure_raster_texcoord_inputs(hlsl, sizeof hlsl,
                                                texcoord_control_mask)) return NULL;
    if (rsx_fp_apply_raster_controls(hlsl, sizeof hlsl,
            texcoord_control_mask, anti_aliasing_control) < 0) return NULL;
    D3D12FPSamplerLayout sampler_layout;
    if (!d3d12_hlsl_remap_fp_samplers(hlsl, sizeof(hlsl), shadow_compare_key,
                                      &sampler_layout)) {
        static int _smerr = 0;
        if (_smerr++ < 8)
            printf("[FP] sampler-register remap failed (fp=0x%08X)\n", fp_addr);
        return NULL;
    }
    ID3DBlob* e = NULL;
    HRESULT hr = S_OK;
    ID3DBlob* pb = d3d12_fp_ps_blob(hlsl, &e, &hr);
    if (FAILED(hr) || !pb) {
        static int _e2=0; if (PS3_LOG_FIRST(_e2, 16))
            printf("[FP] PS compile FAIL (fp=0x%08X): %s\n", fp_addr,
                   e ? (const char*)e->lpVtbl->GetBufferPointer(e) : "?");
        if (e) e->lpVtbl->Release(e);
        return NULL;
    }
    if (e) e->lpVtbl->Release(e);

    D3D12_INPUT_ELEMENT_DESC il[16];
    u32 input_mask = 0xFFFFu;
    if (draw && draw->is_vp && draw->vp_input_mask)
        input_mask = draw->vp_input_mask;
    else if (vs_idx >= 0 && vs_idx < s_d3d.vp_vs_n)
        input_mask = s_d3d.vp_vs[vs_idx].input_mask;
    UINT il_count = 0, input_offset = 0;
    for (UINT attr = 0; attr < 16; attr++) {
        if (!(input_mask & (1u << attr))) continue;
        il[il_count].SemanticName = "ATTR";
        il[il_count].SemanticIndex = attr;
        il[il_count].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        il[il_count].InputSlot = 0;
        il[il_count].AlignedByteOffset = input_offset;
        il[il_count].InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        il[il_count].InstanceDataStepRate = 0;
        il_count++;
        input_offset += 16;
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {0};
    pd.pRootSignature = s_d3d.vp_root_sig;
    pd.VS.pShaderBytecode = vsb->lpVtbl->GetBufferPointer(vsb);
    pd.VS.BytecodeLength  = vsb->lpVtbl->GetBufferSize(vsb);
    pd.PS.pShaderBytecode = pb->lpVtbl->GetBufferPointer(pb);
    pd.PS.BytecodeLength  = pb->lpVtbl->GetBufferSize(pb);
    pd.InputLayout.pInputElementDescs = raw_fetch ? NULL : il;
    pd.InputLayout.NumElements = raw_fetch ? 0 : il_count;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    /* RPCS3 decodes the low nibble as depth-clip enable and the next nibble
     * as depth-clamp enable. D3D12's DepthClipEnable=false supplies the clamp
     * behavior. */
    pd.RasterizerState.DepthClipEnable =
        ((zmin_max_control & 0x0Fu) != 0u && (zmin_max_control & 0xF0u) == 0u)
            ? TRUE : FALSE;
    if (poly_fill) {
        float scale, bias;
        memcpy(&scale, &poly_scale_bits, sizeof(scale));
        memcpy(&bias, &poly_bias_bits, sizeof(bias));
        /* D3D12's UNORM depth-bias unit is the smallest representable depth
         * value (1/(2^24-1) for this backend's D24S8 buffer). RSX's constant
         * bias is already in those units; scaling it by 2^24 would shift depth
         * by the whole bias value and saturate the test scene. */
        double units = round((double)bias);
        if (units > 2147483647.0) units = 2147483647.0;
        if (units < -2147483648.0) units = -2147483648.0;
        pd.RasterizerState.DepthBias = (INT)units;
        pd.RasterizerState.SlopeScaledDepthBias = scale;
    }
    /* Guest face culling. CULL_OFF=1 forces double-sided, the old
     * unconditional behaviour -- geometry that vanishes under culling is a
     * winding/front-face problem, not a missing draw. */
    { static int _co = -1; if (_co < 0) _co = getenv("CULL_OFF") ? 1 : 0;
      pd.RasterizerState.CullMode = (!_co && (cull & 1u))
          ? ((cull & 2u) ? D3D12_CULL_MODE_FRONT : D3D12_CULL_MODE_BACK)
          : D3D12_CULL_MODE_NONE;
      /* CULL_FLIP=1: invert the front-face convention. Our VP path may hand
       * D3D the opposite winding from the guest's, in which case honouring
       * CULL_FACE culls exactly the faces it should keep. */
      static int _cfl = -1; if (_cfl < 0) _cfl = getenv("CULL_FLIP") ? 1 : 0;
      pd.RasterizerState.FrontCounterClockwise =
          (((cull & 4u) ? 1 : 0) ^ _cfl) ? TRUE : FALSE; }
    /* Blend per the guest's packed key (see rsx_blend_key): dbgfont text needs
     * straight alpha; demosaic's effect passes blend OFF; DeferredShading's
     * light accumulation is additive ONE,ONE. */
    pd.BlendState.RenderTarget[0].BlendEnable    = (blend & 1) ? TRUE : FALSE;
    pd.BlendState.RenderTarget[0].SrcBlend       = (blend & 1) ? (D3D12_BLEND)((blend >> 1)  & 0x1F) : D3D12_BLEND_ONE;
    pd.BlendState.RenderTarget[0].DestBlend      = (blend & 1) ? (D3D12_BLEND)((blend >> 6)  & 0x1F) : D3D12_BLEND_ZERO;
    pd.BlendState.RenderTarget[0].BlendOp        = (blend & 1) ? (D3D12_BLEND_OP)((blend >> 21) & 0x7) : D3D12_BLEND_OP_ADD;
    pd.BlendState.RenderTarget[0].SrcBlendAlpha  = (blend & 1) ? (D3D12_BLEND)((blend >> 11) & 0x1F) : D3D12_BLEND_ONE;
    pd.BlendState.RenderTarget[0].DestBlendAlpha = (blend & 1) ? (D3D12_BLEND)((blend >> 16) & 0x1F) : D3D12_BLEND_ZERO;
    pd.BlendState.RenderTarget[0].BlendOpAlpha   = (blend & 1) ? (D3D12_BLEND_OP)((blend >> 24) & 0x7) : D3D12_BLEND_OP_ADD;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.BlendState.IndependentBlendEnable = nrt > 1 ? TRUE : FALSE;
    pd.SampleMask = (anti_aliasing_control & 1u)
        ? anti_aliasing_control >> 16 : UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = (UINT)nrt;
    for (int _r = 0; _r < nrt; _r++) {
        pd.RTVFormats[_r] = rtfmt;
        pd.BlendState.RenderTarget[_r] = pd.BlendState.RenderTarget[0];
        if (_r > 0) {
            const u32 guest_mask = (color_mask_mrt >> (_r * 4)) & 0xFu;
            D3D12_RENDER_TARGET_BLEND_DESC* rt = &pd.BlendState.RenderTarget[_r];
            rt->BlendEnable = (blend_enable_mrt & (1u << _r)) ? TRUE : FALSE;
            rt->RenderTargetWriteMask =
                ((guest_mask & 0x2u) ? D3D12_COLOR_WRITE_ENABLE_RED   : 0) |
                ((guest_mask & 0x4u) ? D3D12_COLOR_WRITE_ENABLE_GREEN : 0) |
                ((guest_mask & 0x8u) ? D3D12_COLOR_WRITE_ENABLE_BLUE  : 0) |
                ((guest_mask & 0x1u) ? D3D12_COLOR_WRITE_ENABLE_ALPHA : 0);
        }
    }
    /* Guest colour write mask (RGBA nibble, already D3D-ordered). cmask is 0xF
     * by default (unset colour_mask register decodes to all-on); a literal 0
     * therefore means the guest EXPLICITLY masked every channel -- a depth-only
     * pass (e.g. DeferredShading's shadow-map generation). Honour it: forcing 0
     * back to 0xF splatters the depth pass's fragment colour onto the target. */
    /* CMASK_FORCE=1: ignore the guest colour mask (write RGBA on every draw).
     * A mis-decoded mask makes geometry rasterize correctly and write nothing,
     * which is indistinguishable from "the draw never happened". */
    { static int _cf = -1; if (_cf < 0) _cf = getenv("CMASK_FORCE") ? 1 : 0;
      if (_cf) {
          for (int _r = 0; _r < nrt; _r++)
              pd.BlendState.RenderTarget[_r].RenderTargetWriteMask =
                  D3D12_COLOR_WRITE_ENABLE_ALL;
      } else {
          pd.BlendState.RenderTarget[0].RenderTargetWriteMask = (UINT8)(cmask & 0xF);
      } }
    pd.DSVFormat = dsvfmt;
    pd.DepthStencilState.DepthEnable = depth_test ? TRUE : FALSE;
    pd.DepthStencilState.DepthWriteMask = depth_write
        ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    pd.DepthStencilState.DepthFunc = d3d12_depth_func(depth_func);
    pd.DepthStencilState.StencilEnable =
        (stencil->enable && dsvfmt == DXGI_FORMAT_D24_UNORM_S8_UINT)
            ? TRUE : FALSE;
    if (stencil->enable) {
        D3D12_DEPTH_STENCILOP_DESC front = {0}, back = {0};
        pd.DepthStencilState.StencilReadMask = (UINT8)(stencil->front_read_mask & 0xFFu);
        pd.DepthStencilState.StencilWriteMask = (UINT8)(stencil->front_write_mask & 0xFFu);
        front.StencilFunc = d3d12_stencil_func(stencil->front_func);
        front.StencilFailOp = d3d12_stencil_op(stencil->front_fail);
        front.StencilDepthFailOp = d3d12_stencil_op(stencil->front_zfail);
        front.StencilPassOp = d3d12_stencil_op(stencil->front_zpass);
        if (stencil->two_sided) {
            back.StencilFunc = d3d12_stencil_func(stencil->back_func);
            back.StencilFailOp = d3d12_stencil_op(stencil->back_fail);
            back.StencilDepthFailOp = d3d12_stencil_op(stencil->back_zfail);
            back.StencilPassOp = d3d12_stencil_op(stencil->back_zpass);
        } else {
            back = front;
        }
        pd.DepthStencilState.FrontFace = front;
        pd.DepthStencilState.BackFace = back;
    }
    /* DEPTH_OFF=1: drop the depth test for guest-FP draws, so submission order
     * alone decides what is on top. Paired with DRAW_LAST_TEX this puts one
     * object in front of everything and answers "is it merely occluded?" --
     * which reordering alone cannot, because a relocated draw still fails the
     * depth test against whatever is already in the buffer. Diagnostic. */
    { static int doff = -1;
      if (doff < 0) { const char* e = getenv("DEPTH_OFF"); doff = e ? atoi(e) : 0; }
      if (doff) { pd.DepthStencilState.DepthEnable = FALSE;
                  pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO; } }
    pd.SampleDesc.Count = 1;

    /* The layout is part of the persistent identity even though it is derived
     * from the generated HLSL rather than a PSO register. */
    key.sampler_layout = sampler_layout;
    wchar_t pipeline_name[64];
    d3d12_pipeline_library_name(&key, input_mask, vsb, pb, &pd, pipeline_name);
    ID3D12PipelineState* pso = NULL;
    hr = d3d12_pipeline_library_get_or_create(pipeline_name, &pd, &pso);
    pb->lpVtbl->Release(pb);
    if (FAILED(hr)) {
        static int _e3=0; if (PS3_LOG_FIRST(_e3, 16)) printf("[FP] PSO FAIL (fp=0x%08X, 0x%08lX)\n", fp_addr, hr);
        return NULL;
    }
    { static int _ok=0; if (PS3_LOG_FIRST(_ok, 32)) printf("[FP] guest FP pipeline ready (fp=0x%08X)\n", fp_addr); }

    /* Insert by direct index. Ring replacement keeps the cache bounded without
     * shifting every entry (and invalidates only the replaced hash-chain node). */
    u32 entry_index;
    if (s_d3d.vp_fp_n < VP_FP_CACHE) {
        entry_index = (u32)s_d3d.vp_fp_n++;
    } else {
        entry_index = (u32)s_d3d.vp_fp_next_replace;
        s_d3d.vp_fp_next_replace =
            (s_d3d.vp_fp_next_replace + 1) % VP_FP_CACHE;
        d3d12_vp_fp_hash_remove(entry_index);
        if (s_d3d.vp_fp[entry_index].pso)
            s_d3d.vp_fp[entry_index].pso->lpVtbl->Release(
                s_d3d.vp_fp[entry_index].pso);
    }
    VPFPEntry* entry = &s_d3d.vp_fp[entry_index];
    *entry = key;
    entry->fp_addr = fp_addr;
    entry->sampler_layout = sampler_layout;
    entry->pso = pso;
    entry->key_hash = key_hash;
    entry->hash_next = s_d3d.vp_fp_hash[bucket];
    s_d3d.vp_fp_hash[bucket] = entry_index + 1u;
    if (sampler_layout_out) *sampler_layout_out = sampler_layout;
    return pso;
}

/* Upload the guest texture for a VP draw into a per-frame texture slot
 * (re-uploaded every frame: gcm/cube's plasma animates in guest memory).
 * Returns the slot index (SRV at heap 1+slot) or -1. Must run while the
 * command list is open, before the draw passes. */
/* NV4097 TEXTURE_CONTROL1 component remap -> D3D12 Shader4ComponentMapping.
 * The crossbar decode itself is RSX semantics and lives in
 * rsx_texture_layout.c; all that is left here is packing the four selectors
 * into D3D12's word. Its force-zero/force-one encodings are 4 and 5, which is
 * what RSX_REMAP_ZERO/ONE already are, so the selectors pass straight through.
 *
 * out[] arrives in the crossbar's field order A,R,G,B; D3D12 wants
 * destR | destG<<3 | destB<<6 | destA<<9 | valid<<12. */
static u32 rsx_remap_to_d3d(u32 c1, u32 basef)
{
    u8 out[4];
    rsx_texture_component_remap(c1, basef, out);
    /* TEX_REMAP_ID=<n>: override the derived crossbar with a fixed mapping, to
     * test channel order directly. The resource holds the guest's A8R8G8B8
     * bytes straight through, so its components are (R=A, G=R, B=G, A=B) and
     * the shader needs destR=1, destG=2, destB=3, destA=0 to see real RGBA.
     *   1 = identity   2 = rotate (the ARGB fix)   3 = BGRA swap */
    { static int fixed = -1;
      if (fixed < 0) { const char* e = getenv("TEX_REMAP_ID"); fixed = e ? atoi(e) : 0; }
      if (fixed == 1) return 0u | (1u<<3) | (2u<<6) | (3u<<9) | (1u<<12);
      if (fixed == 2) return 1u | (2u<<3) | (3u<<6) | (0u<<9) | (1u<<12);
      if (fixed == 3) return 2u | (1u<<3) | (0u<<6) | (3u<<9) | (1u<<12); }
    /* D3D12 mapping: destR | destG<<3 | destB<<6 | destA<<9 | valid bit */
    { static int _rd = -1; if (_rd < 0) _rd = getenv("REMAPDBG") ? 1 : 0;
      if (_rd) { static u32 seen[32]; static int ns=0;
        u32 k = (basef << 16) | (c1 & 0xFFFFu); int f=0;
        for (int i2=0;i2<ns;i2++) if (seen[i2]==k) f=1;
        if (!f && ns<32) { seen[ns++]=k;
            fprintf(stderr, "[REMAP] basef=0x%02X c1=0x%04X -> destR=%u destG=%u destB=%u destA=%u%c",
                    basef, c1 & 0xFFFFu, out[1], out[2], out[3], out[0], 10); } } }
    return out[1] | (out[2] << 3) | (out[3] << 6) | (out[0] << 9) | (1u << 12);
}

static u32 d3d12_texture_component_mapping(u32 control1, u32 basef)
{
    /* Preserve the established B8 coverage behavior for the default crossbar
     * (R is replicated into alpha), but honor an explicit guest remap. */
    if ((basef & 0x9Fu) == 0x81u &&
        (control1 & 0xFFFFu) == 0xAAE4u && (control1 >> 16) == 0)
        return 0x1000u;
    if ((basef & 0x9Fu) == 0x81u && control1 == 0)
        return 0x1000u;
    return rsx_remap_to_d3d(control1, basef);
}

/* RPCS3_REFERENCE_REVIEWED: this swizzle mapping matches the named RPCS3
 * helper; targeted review did not establish a copied function body. See
 * docs/RPCS3_LICENSE_AUDIT.md. This is not license clearance.
 * Morton/Z-order texel offset for RSX swizzled textures (LN bit clear).
 * Interleaves x (even bit positions) and y (odd) until the smaller dimension's
 * bits run out, then the larger dimension's remaining bits ride above -- the
 * NV40 layout (matches RPCS3 convert_linear_swizzle). POT dims only, which is
 * what the hardware requires for swizzled textures. */

/* Sparse FNV-1a over a texture's source bytes -- enough to notice an animated
 * surface changing, cheap enough to run on every bind. Kept a function rather
 * than a statement-expression macro so the file still compiles under MSVC;
 * ps3recomp builds with clang-cl, but titles like Tokyo Jungle drive the build
 * through the Visual Studio generator. */
/* Sorted bucket chains preserve the old fallback's slot order. A failed
 * upload may leave an old hint; each lookup still checks resource and full
 * key. The old scan remains the authority for allocation and eviction. */
static void vp_tex_index_insert(u32 slot, u32 bucket)
{
    const u32 old = s_d3d.vp_tex_bucket[slot];
    if (old) {
        u32* link = &s_d3d.vp_tex_heads[old - 1u];
        while (*link && *link != slot + 1u) link = &s_d3d.vp_tex_next[*link - 1u];
        if (*link) *link = s_d3d.vp_tex_next[slot];
    }
    u32* link = &s_d3d.vp_tex_heads[bucket];
    while (*link && *link < slot + 1u) link = &s_d3d.vp_tex_next[*link - 1u];
    s_d3d.vp_tex_next[slot] = *link;
    *link = slot + 1u;
    s_d3d.vp_tex_bucket[slot] = bucket + 1u;
}
static u32 tex_csum(const u8* base, u32 nbytes)
{
    u32 h = 2166136261u;
    u32 step = nbytes > 4096u ? nbytes / 1024u : 4u;
    for (u32 i = 0; i + 3 < nbytes; i += step) {
        u32 w32; memcpy(&w32, base + i, sizeof w32);
        h ^= w32; h *= 16777619u;
    }
    return h;
}

/* TEX_UPLOAD_FAILDBG=<N>: report why a sampled guest texture could not get an
 * SRV. A NULL descriptor alone cannot distinguish pool pressure from a D3D12
 * allocation or upload failure. */
static void vp_tex_upload_fail(const char* reason, u32 off, u32 w, u32 h,
                               u32 fmt, u32 detail)
{
    static int cap = -1, n = 0;
    if (cap < 0) {
        const char* e = getenv("TEX_UPLOAD_FAILDBG");
        cap = e ? atoi(e) : 0;
    }
    if (cap > 0 && n < cap) {
        n++;
        fprintf(stderr, "[TEXUPLOAD-FAIL] reason=%s off=0x%08X %ux%u fmt=0x%02X detail=0x%X frame=%llu\n",
                reason, off, w, h, fmt & 0xFFu, detail,
                (unsigned long long)s_d3d.frame_count);
    }
}


/* The DXGI format a guest texture is uploaded as -- the DECODED image's, which
 * is not always the guest's: the packed 16-bit colour formats arrive unpacked
 * to RGBA8, and DEPTH24_D8 as a float. rsx_texture_layout.h says which class
 * each format byte lands in. The SRV over the upload must name the same one. */
static DXGI_FORMAT vp_tex_dxgi(u32 fmt)
{
    rsx_tex_layout tl;
    rsx_texture_layout(fmt, 4, 4, &tl);
    switch (tl.fmt) {
    case RSX_TEXFMT_R8G8B8A8:      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case RSX_TEXFMT_R8G8:          return DXGI_FORMAT_R8G8_UNORM;
    case RSX_TEXFMT_BC1:           return DXGI_FORMAT_BC1_UNORM;
    case RSX_TEXFMT_BC2:           return DXGI_FORMAT_BC2_UNORM;
    case RSX_TEXFMT_BC3:           return DXGI_FORMAT_BC3_UNORM;
    case RSX_TEXFMT_R16:           return DXGI_FORMAT_R16_UNORM;
    case RSX_TEXFMT_R16G16:        return DXGI_FORMAT_R16G16_UNORM;
    case RSX_TEXFMT_R16G16F:       return DXGI_FORMAT_R16G16_FLOAT;
    case RSX_TEXFMT_R16G16B16A16F: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case RSX_TEXFMT_R32F:          return DXGI_FORMAT_R32_FLOAT;
    case RSX_TEXFMT_R32G32B32A32F: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:                       return DXGI_FORMAT_R8_UNORM;
    }
}

static int vp_upload_tex_slot(u32 off, u32 w, u32 h, u32 fmt, u32 row_pitch, int cube, u32 mips)
{
    /* TEX_BUDGET=<n>: cap texture uploads per frame. Off by default -- it was
     * tried against the TDR and does not help, because PERF shows the upload
     * path costing 0.01s for 775 calls. Kept because it is the right lever if a
     * title ever does hit an upload-bound frame. */
    { static int budget = -1;
      if (budget < 0) { const char* e = getenv("TEX_BUDGET"); budget = e ? atoi(e) : 0; }
      if (budget > 0) {
          static u64 last_frame = ~0ull; static int used = 0;
          if (last_frame != (u64)s_d3d.frame_count) { last_frame = (u64)s_d3d.frame_count; used = 0; }
          if (++used > budget) {
              vp_tex_upload_fail("budget", off, w, h, fmt, (u32)budget);
              return -1;   /* try again next frame */
          }
      } }
    extern uint8_t* vm_base;

    if (!off || !w || !h || !vm_base || !s_d3d.srv_heap) {
        const u32 missing = (!off ? 1u : 0u) | (!w ? 2u : 0u) |
            (!h ? 4u : 0u) | (!vm_base ? 8u : 0u) | (!s_d3d.srv_heap ? 16u : 0u);
        vp_tex_upload_fail("invalid-args", off, w, h, fmt, missing);
        return -1;
    }
    /* Format classes (base = fmt & 0x9F). The LBP loading screen uses:
     * 0x85 A8R8G8B8 (swizzled UI art), 0x8B G8B8 (the 1024x2048 linear FONT
     * atlas -- without it no text renders at all), 0x86/87/88 DXT1/23/45
     * (512x512 detail/LUT layers bound at t1/t3 on every draw). */
    /* Format class, row sizes and swizzling are RSX semantics: rsx_texture_
     * layout.c owns them so the Metal backend can read a guest texture without
     * re-deriving any of it. Only the DXGI mapping below is ours. */
    rsx_tex_layout tl;
    rsx_texture_layout_pitched(fmt, w, h, row_pitch, &tl);
    u32 basef = fmt & 0x9F;              /* still needed for the SRV remap */
    int argb = (tl.fmt == RSX_TEXFMT_R8G8B8A8);
    int dxt  = tl.compressed;
    /* Compressed rows are counted in blocks, so keep bpp at 1 for the byte
     * arithmetic the diagnostics below do. */
    u32 bpp = tl.compressed ? 1u : tl.bytes_per_texel;
    /* The DECODED image's format, which is not always the guest's: the packed
     * 16-bit colour formats arrive here unpacked to RGBA8, and DEPTH24_D8 as a
     * float. rsx_texture_layout.h says which class each format byte lands in. */
    DXGI_FORMAT dxfmt = vp_tex_dxgi(fmt);
    /* DXT data is stored as linear 4x4 block rows (compressed formats are
     * never Morton-swizzled on RSX). Row of blocks = (w/4)*blocksize. */
    u32 blkrow  = dxt ? tl.row_bytes : 0;
    u32 blkrows = dxt ? tl.rows      : 0;
    const u32 key_off = off;         /* lookup key: the offset as bound */
    /* Sparse checksum of a texture's source bytes -- enough to notice an
     * animated surface changing, cheap enough to run on every bind. */
    /* (portable helper tex_csum() -- see above; a statement-expression macro
     * here was a GCC/Clang extension MSVC rejects.) */
    #define TEX_CSUM(base, nbytes) tex_csum((base), (nbytes))
    /* A hint avoids a linear resident-slot search on repeated binds. It never
     * establishes validity: eviction, guest writes, and hash collisions still
     * go through the complete key and source checksum checks below. Store
     * slot+1 so the zero-initialized table has no accidental valid entry. */
    static u32 slot_hint[1024][4];
    static u8 slot_hint_next[1024];
    u32 checked_off = 0, checked_size = 0, checked_csum = 0;
    int checked_valid = 0;
    const u32 hint_key = ((key_off >> 8) ^ (key_off >> 18) ^ w ^ (h << 3)
                         ^ fmt ^ row_pitch ^ (mips << 5) ^ (u32)cube) & 1023u;
    for (u32 way = 0; way < 4; ++way) {
        if (!slot_hint[hint_key][way]) continue;
        const u32 i = slot_hint[hint_key][way] - 1u;
        VPTexSlot* c = &s_d3d.vp_tex[i];
        if (c->res && c->key == key_off && c->w == w && c->h == h && c->fmt == fmt
            && c->pitch == row_pitch && c->mips == mips && c->cube == cube) {
            const u32 nb = cube ? rsx_texture_cube_face_stride(fmt, w, h, mips, row_pitch) * 6u
                                : tl.face_bytes;
            if (!checked_valid || checked_off != c->off || checked_size != nb) {
                checked_csum = TEX_CSUM(vm_base + c->off, nb);
                checked_off = c->off; checked_size = nb; checked_valid = 1;
            }
            if (checked_csum == c->csum) {
                c->used = 1;
                c->last_use = ++s_vp_tex_clock;
                s_perf_tex_hits++;
                return (int)i;
            }
        }
    }
    if (d3d12_index_cache_enabled()) {
        for (u32 link = s_d3d.vp_tex_heads[hint_key]; link; link = s_d3d.vp_tex_next[link - 1u]) {
            const u32 i = link - 1u;
            VPTexSlot* c = &s_d3d.vp_tex[i];
            if (!c->res || c->key != key_off || c->w != w || c->h != h ||
                c->fmt != fmt || c->pitch != row_pitch || c->mips != mips || c->cube != cube)
                continue;
            const u32 nb = cube ? rsx_texture_cube_face_stride(fmt, w, h, mips, row_pitch) * 6u
                                : tl.face_bytes;
            if (!checked_valid || checked_off != c->off || checked_size != nb) {
                checked_csum = TEX_CSUM(vm_base + c->off, nb);
                checked_off = c->off; checked_size = nb; checked_valid = 1;
            }
            if (checked_csum == c->csum) {
                c->used = 1;
                c->last_use = ++s_vp_tex_clock;
                s_perf_tex_hits++;
                slot_hint[hint_key][slot_hint_next[hint_key]++ & 3u] = i + 1u;
                return (int)i;
            }
            /* The old scan would select this stale unused slot; retain that
             * exact replacement choice instead of skipping to a later hit. */
            if (!c->used) break;
        }
    }
    int slot = -1, freeslot = -1;
    int empty_slot = -1;
    for (int i = 0; i < VP_TEX_SLOTS; i++) {
        VPTexSlot* c = &s_d3d.vp_tex[i];
        if (c->res && c->key == key_off && c->w == w && c->h == h && c->fmt == fmt
            && c->pitch == row_pitch && c->mips == mips && c->cube == cube) {
            /* A guest can rewrite a texture after an earlier draw sampled it
             * in this same frame. Reusing that slot without checking its bytes
             * makes the later draw see stale contents. If it did change, keep
             * the already-recorded draw's resource immutable and look for a
             * second slot below; never overwrite a resource referenced by an
             * earlier draw in the current command list. */
            u32 nb = cube ? rsx_texture_cube_face_stride(fmt, w, h, mips, row_pitch) * 6u
                          : tl.face_bytes;
            if (!checked_valid || checked_off != c->off || checked_size != nb) {
                checked_csum = TEX_CSUM(vm_base + c->off, nb);
                checked_off = c->off; checked_size = nb; checked_valid = 1;
            }
            u32 cs = checked_csum;
            if (cs == c->csum) {
                c->used = 1;
                c->last_use = ++s_vp_tex_clock;
                s_perf_tex_hits++;
                slot_hint[hint_key][slot_hint_next[hint_key]++ & 3u] = (u32)i + 1u;
                return i;
            }
            if (c->used) continue;
            { static int nocache = -1;            /* TEX_NOCACHE=1: always re-upload */
              if (nocache < 0) { const char* e = getenv("TEX_NOCACHE");
                                 nocache = e ? atoi(e) : 0; }
              if (nocache) { slot = i; break; } }
            /* Held from an earlier frame: re-upload only if the guest bytes
             * changed. Most of this scene's textures are static, and converting
             * every one of them every frame was ~70% of the frame's CPU time. */
            slot = i; break;                      /* stale: fall through and redo */
        }
        if (!c->res && empty_slot < 0) empty_slot = i;
        if (!c->used && (freeslot < 0 ||
                c->last_use < s_d3d.vp_tex[freeslot].last_use))
            freeslot = i;
    }
    /* Fill empty entries before evicting a resident texture. The previous
     * first-unused policy continually replaced early entries while leaving
     * later slots empty, causing allocation/decode churn on every batch. */
    if (empty_slot >= 0) freeslot = empty_slot;
    /* TEX_OFF_BIAS=1: sample the image one whole level-0 BELOW the bound offset.
     * Measured on the Rubber Ducky demo: at the bind offset every texture reads
     * as all-zero, while [off - w*h*bpp, off) holds the image (3800-4160 of 4298
     * sampled bytes non-zero, for four different textures at three sizes). So
     * the offset the guest programs points PAST level 0 rather than at it. */
    /* TEX_OFF_BIAS: 1 = one level-0 below the bound offset, 2 = one full MIP
     * CHAIN above it. The measured upload-to-bind deltas on the Rubber Ducky
     * demo are 0x2AAB00 for a 2MB level 0 and 0x555580 for a 4MB one -- both
     * exactly the mipmap pyramid total (L0 * 4/3), which is what the guest
     * reserves per texture. */
    { static int bias = -1;
      if (bias < 0) { const char* e = getenv("TEX_OFF_BIAS"); bias = e ? atoi(e) : 0; }
      if (bias == 1) { u32 sz = w * h * bpp; if (off > sz) off -= sz; }
      else if (bias >= 2) {
          u32 chain = 0;
          for (u32 mw = w, mh = h; mw && mh; mw >>= 1, mh >>= 1) {
              chain += mw * mh * bpp;
              if (mw == 1 && mh == 1) break;
          }
          /* bias 3: only shift when the bound offset really is empty, so a
           * texture that IS bound correctly keeps its own data. Applying the
           * chain unconditionally fixes the mis-bound majority but breaks any
           * correctly-bound minority, which shows up as objects going black. */
          int shift = 1;
          if (bias >= 3) {
              u32 nz = 0, sz = w * h * bpp;
              for (u32 i = 0; i < sz && i < 0x20000u; i += 97) if (vm_base[off + i]) nz++;
              shift = (nz == 0);
          }
          if (shift) off += chain;
      } }
    /* TEX_REMAP=1: if the bound offset holds nothing, look the image up by size
     * in the VRAM upload registry (cellGcmSys). For a title whose upload address
     * and SET_TEXTURE_OFFSET disagree this puts the real bytes under the sampler
     * without guessing a delta. */
    { static int remap = -1;
      if (remap < 0) { const char* e = getenv("TEX_REMAP"); remap = e ? atoi(e) : 0; }
      if (remap) {
        u32 nz = 0, sz = w * h * bpp;
        for (u32 i = 0; i < sz && i < 0x20000u; i += 97) if (vm_base[off + i]) nz++;
        if (!nz) {
            extern u32 rsx_find_vram_upload(u32);
            u32 alt = rsx_find_vram_upload(sz);
            if (alt) {
                static int _n = 0;
                if (PS3_LOG_FIRST(_n, 8)) fprintf(stderr, "[TEXREMAP] 0x%08X (%ux%u, %u bytes) -> 0x%08X%c",
                                      off, w, h, sz, alt, 10);
                off = alt;
            }
        } } }
    /* A stale cache entry (slot >= 0) re-uses its own slot; otherwise take a
     * free one. Without this the stale path fell through to freeslot and either
     * duplicated the texture into a second slot or bailed when none was free. */
    if (slot < 0 && freeslot < 0) {
        vp_tex_upload_fail("slots-exhausted", off, w, h, fmt, VP_TEX_SLOTS);
        return -1;      /* out of slots this frame */
    }
    /* TEX_SRCDBG=<N>: is the guest memory this slot uploads FROM actually
     * populated? An empty source and a broken sampler both render flat. */
    { static int cap = -1, n = 0;
      if (cap < 0) { const char* e = getenv("TEX_SRCDBG"); cap = e ? atoi(e) : 0; }
      if (cap && n < cap) { n++;
        u32 nz = 0, tot = 0;
        for (u32 i = 0; i < w * h * bpp && i < 0x40000u; i += 61) { tot++; if (vm_base[off + i]) nz++; }
        /* Several of this title's textures are bound at EXACTLY the end address
         * of an upload, so also sample the block immediately BELOW the bind
         * offset: if that holds the image, the bind is one whole texture high. */
        u32 nzb = 0, totb = 0;
        { u32 sz = w * h * bpp; u32 base = (off > sz) ? off - sz : 0;
          for (u32 i = 0; i < sz && i < 0x40000u; i += 61) { totb++; if (vm_base[base + i]) nzb++; } }
        long nearest = 0; u32 found = 0;
        if (!nz) {   /* empty: where IS the data? scan out from the bind offset */
            for (long d = 0x1000; d <= 0x1000000 && !found; d += 0x1000) {
                for (int s = 0; s < 2 && !found; s++) {
                    long a = (long)off + (s ? -d : d);
                    if (a < 0x1000) continue;
                    u32 c = 0;
                    for (u32 i = 0; i < 0x1000u; i += 61) if (vm_base[a + i]) c++;
                    if (c > 8) { nearest = (s ? -d : d); found = c; }
                }
            }
        }
        fprintf(stderr, "[TEXSRC] off=0x%08X %ux%u fmt=0x%02X bpp=%u nonzero=%u/%u"
                        "%s%+ld (page nonzero=%u)\n",
                off, w, h, fmt, bpp, nz, tot,
                nz ? "" : "  nearest data at ", nz ? 0L : nearest, found);
        fprintf(stderr, "[TEXSRC]    block BELOW bind (off-%u): nonzero=%u/%u%c",
                w * h * bpp, nzb, totb, 10);
        /* Walk DOWN in 4KB steps to the first populated page whose predecessor
         * is empty: that is where this texture's data actually begins, and the
         * delta from the bound offset is the rule we are missing. */
        { u32 step = 0x1000, prev_pop = 0; long start_delta = 0;
          for (long d = 0; d >= -(long)(w * h * bpp) * 2; d -= step) {
              u32 a = (u32)((long)off + d); u32 pop = 0;
              if (a < 0x1000) break;
              for (u32 i = 0; i < step; i += 53) if (vm_base[a + i]) pop++;
              if (!pop && prev_pop) { start_delta = d + step; break; }
              prev_pop = pop;
          }
          fprintf(stderr, "[TEXSRC]    data starts at off%+ld (level0=%u bytes)%c",
                  start_delta, w * h * bpp, 10); } } }
    if (slot < 0) slot = freeslot;
    VPTexSlot* t = &s_d3d.vp_tex[slot];
    /* The upload buffer's rows hold the DECODED image, so they are sized from
     * dst_row_bytes -- w * bpp is the source row, which is narrower for any
     * format rsx_texture_decode unpacks. */
    u32 pitch = (tl.dst_row_bytes + 255) & ~255u;
    int fresh = 0;

    /* Also recreate when the CUBE-ness changes: a slot holding a 2D texture of
     * the same dimensions would otherwise be reused with 1 array slice while the
     * cube path copies 6 subresources and writes a 6x upload buffer -- which
     * crashes. */
    if (t->res && (t->w != w || t->h != h || t->fmt != fmt || t->cube != cube)) {
        if (!d3d12_retire_resource(&t->res)) return -1;
        for (u32 i = 0; i < FRAME_COUNT; i++)
            if (!d3d12_retire_resource(&t->up[i])) return -1;
    }
    if (!t->res) {
        D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC td = {0};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = w; td.Height = h;
        td.DepthOrArraySize = cube ? 6 : 1;   /* cube = 6 array slices */
        td.MipLevels = 1;
        td.Format = dxfmt; td.SampleDesc.Count = 1;
        td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        HRESULT hr = s_d3d.device->lpVtbl->CreateCommittedResource(
                s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &td,
                D3D12_RESOURCE_STATE_COPY_DEST, NULL,
                &IID_ID3D12Resource, (void**)&t->res);
        if (FAILED(hr)) {
            vp_tex_upload_fail("texture-create", off, w, h, fmt, (u32)hr);
            return -1;
        }
        s_perf_tex_allocs++;
        fresh = 1;
    }

    const u32 upload_slot = (u32)s_d3d.vp_parity;
    if (!t->up[upload_slot]) {
        D3D12_HEAP_PROPERTIES hu = {0}; hu.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd = {0};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = (u64)pitch * (dxt ? blkrows : h) * (cube ? 6 : 1);
        bd.Height = 1; bd.DepthOrArraySize = 1;
        bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        HRESULT hr = s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &hu, D3D12_HEAP_FLAG_NONE, &bd,
            D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
            &IID_ID3D12Resource, (void**)&t->up[upload_slot]);
        if (FAILED(hr)) {
            vp_tex_upload_fail("upload-buffer-create", off, w, h, fmt, (u32)hr);
            if (fresh) { t->res->lpVtbl->Release(t->res); t->res = NULL; }
            return -1;
        }
    }

    void* mapped = NULL; D3D12_RANGE nr = {0,0};
    if (FAILED(t->up[upload_slot]->lpVtbl->Map(t->up[upload_slot], 0, &nr, &mapped)) || !mapped) return -1;
    { static int _tp=0; if (d3d12_setting_rtt_dump() && PS3_LOG_FIRST(_tp, 6)) {
        const u8* sp = vm_base + off;
        fprintf(stderr, "[TEXUP] off=0x%X fmt=0x%X row0:", off, fmt);
        for (int _b=0;_b<8;_b++) fprintf(stderr, " %02X", sp[_b]);
        fprintf(stderr, "  row240:");
        for (int _b=0;_b<8;_b++) fprintf(stderr, " %02X", sp[240*w*bpp+_b]);
        fprintf(stderr, "  mid:");
        for (int _b=0;_b<8;_b++) fprintf(stderr, " %02X", sp[240*w*bpp+320*bpp+_b]);
        fprintf(stderr, "%s", "\n");
    } }
    /* Swizzled (LN bit 0x20 clear): texels live in Morton/Z-order, not rows.
     * Uploading them as linear rows produced the diagonal-stripe garbage on
     * LBP's loading screen (every texture there is 0x85 swizzled). POT dims
     * only -- the hardware requires that for swizzled textures anyway. */
    int swz = tl.swizzled;
    /* Cube textures store their 6 faces consecutively. Convert each into its own
     * slice of the upload buffer; the conversion below is unchanged and simply
     * runs once per face with off/mapped rebased. */
    const u32 _face_rows  = dxt ? blkrows : h;
    /* The face stride is NOT one mip-0 image: RSX stores each cube face as its
     * own complete mip pyramid, so face f starts a whole pyramid (128-byte
     * aligned) in. Assuming mip-0-sized strides made face 1 land inside face 0's
     * mip chain -- which is exactly what the dumps showed, every face after the
     * first a progressively smaller copy of the first. */
    u32 _face_bytes = cube
        ? rsx_texture_cube_face_stride(fmt, w, h, mips, row_pitch)
        : tl.face_bytes;
    const u32 _nfaces     = cube ? 6u : 1u;
    const u32 _off0 = off; u8* const _map0 = (u8*)mapped;
    for (u32 _f = 0; _f < _nfaces; _f++) {
    off    = _off0 + _f * _face_bytes;
    if (cube && getenv("CUBEDBG")) { u32 _s=0; const u8* _p = vm_base + off;
        for (u32 _i = 0; _i < _face_bytes; _i += 97) _s = _s*131 + _p[_i];
        fprintf(stderr, "[CUBEFACE] f=%u off=0x%X bytes=%u bpp=%u swz=%d csum=%08X%c",
                _f, off, _face_bytes, bpp, swz, _s, 10); }
    mapped = _map0 + (u64)_f * pitch * _face_rows;
    /* Pixel conversion -- deswizzle, channel order, block copy -- is RSX
     * semantics and lives in rsx_texture_layout.c, so a second backend gets
     * it by calling this rather than porting the loops again. */
    rsx_texture_decode(mapped, pitch, vm_base + off, w, h, &tl,
                       rsx_texture_argb_is_rgba());

    /* The diagnostics below only apply to the linear 8-bit case (Bink video
     * planes); they used to sit inside that branch of the conversion. */
    if (!tl.compressed && tl.bytes_per_texel == 1 && !tl.swizzled) {
        /* MOVIE_FIND=1: when a 640x360 movie plane uploads ZERO, scan a wide
         * guest window for the REAL frame (a 640-wide region with content) so
         * we learn where the video decode actually wrote vs where the texture
         * points -- ends the frame-buffer-address guessing. */
        if (getenv("MOVIE_FIND") && w == 640 && h == 360) {
            extern uint8_t* vm_base;
            const u8* here = vm_base + off;
            u32 hz = 0; for (u32 i=0;i<w*h;i+=137) if (here[i]) { hz=1; break; }
            static int _mf = 0;
            if (!hz && PS3_LOG_FIRST(_mf, 3)) {
                fprintf(stderr, "[movie-find] bound plane ea=0x%X is ZERO; full scan...\n", off);
                /* Scan FULL VRAM + main heap in 0x8000 steps for a 640x360
                 * content block -- NO early cap (the previous found<6 stopped in
                 * the low-VRAM display buffers before ever reaching the movie
                 * region ~0x40E80000). Report the near-plane hits explicitly so
                 * ring-desync (frame present at a DIFFERENT slot) is separable
                 * from decode-never-ran (region entirely zero). */
                struct { u32 lo, hi; const char* tag; } rng[] = {
                    {0x40000000u, 0x41000000u, "VRAM"}, {0x00100000u, 0x10000000u, "MAIN"},
                    {0x48000000u, 0x4A000000u, "BINK"} };  /* Bink working set (handle ~0x4849E760, SPU I/O 0x4847/0x4945) */
                u32 plane_lo = (off > 0x400000u) ? off - 0x400000u : 0;
                u32 plane_hi = off + 0x400000u;
                int found = 0, nearc = 0;
                for (int r = 0; r < (int)(sizeof(rng)/sizeof(rng[0])); r++) {
                    u32 best_a = 0, best_nz = 0; u32 best_span = 0;
                    for (u32 a = rng[r].lo; a + w*h < rng[r].hi; a += 0x8000) {
                        const u8* p = vm_base + a;
                        u32 nz = 0; u8 mn2 = 255, mx2 = 0;
                        for (u32 i = 0; i < w*h; i += 257) { u8 v = p[i]; if (v) nz++; if (v<mn2) mn2=v; if (v>mx2) mx2=v; }
                        u32 span = mx2 - mn2;
                        if (nz > best_nz) { best_nz = nz; best_a = a; best_span = span; }
                        if (nz > 400 && span > 60) {  /* looks like image content */
                            int isnear = (a >= plane_lo && a <= plane_hi);
                            if (isnear) nearc++;
                            if (found < 40)
                                fprintf(stderr, "[movie-find]   CONTENT @0x%08X (%s) nz=%u span=%u%s\n",
                                        a, rng[r].tag, nz, span, isnear ? " <== NEAR bound plane" : "");
                            found++;
                        }
                    }
                    /* Always report the densest block in this range even below the
                     * content threshold -- the MM intro fades in from black, so the
                     * real first frames are low-span and would otherwise be invisible. */
                    fprintf(stderr, "[movie-find]   %s best @0x%08X nz=%u span=%u%s\n",
                            rng[r].tag, best_a, best_nz, best_span,
                            (best_a >= plane_lo && best_a <= plane_hi) ? " <== NEAR bound plane" : "");
                    /* Dump the densest block as a 640x360 grayscale BMP so the
                     * "is this the decoded movie frame" question is answered by
                     * eye. One dump per range per process. */
                    if (best_nz > 400 && best_span > 40) {
                        static int _bd[4] = {0,0,0,0};
                        if (r < 4 && !_bd[r]) { _bd[r] = 1;
                            const u8* sp = vm_base + best_a;
                            char pn[128];
                            snprintf(pn, sizeof(pn), "binkscan_%s_%08X.bmp", rng[r].tag, best_a);
                            FILE* f = fopen(pn, "wb");
                            if (f) {
                                u32 rowb = w*3, rowp = (rowb+3)&~3u, fsz = 54 + rowp*h;
                                u8 hd[54] = {'B','M'};
                                hd[2]=(u8)fsz; hd[3]=(u8)(fsz>>8); hd[4]=(u8)(fsz>>16); hd[5]=(u8)(fsz>>24);
                                hd[10]=54; hd[14]=40;
                                hd[18]=(u8)w; hd[19]=(u8)(w>>8); hd[22]=(u8)h; hd[23]=(u8)(h>>8);
                                hd[26]=1; hd[28]=24;
                                fwrite(hd,1,54,f);
                                for (int y=(int)h-1; y>=0; y--) {
                                    const u8* srow = sp + (u64)y*w;
                                    for (u32 x=0;x<w;x++){ u8 px[3]; px[0]=px[1]=px[2]=srow[x]; fwrite(px,1,3,f); }
                                    { u8 z[3]={0,0,0}; fwrite(z,1,rowp-rowb,f); }
                                }
                                fclose(f);
                                fprintf(stderr, "[movie-find]   dumped %s\n", pn);
                            }
                        }
                    }
                }
                fprintf(stderr, "[movie-find]   total=%d near-plane=%d (window 0x%X..0x%X)\n",
                        found, nearc, plane_lo, plane_hi);
                if (!found) fprintf(stderr, "[movie-find]   no 640x360 content block (>thresh) found anywhere\n");
            }
        }
        /* TEX_SAVE=1: also dump wide B8 uploads (Bink video planes are B8 --
         * the Y plane IS the movie frame in grayscale) + a content stat, so
         * "is the decoder producing pixels" is answerable by looking at a BMP. */
        if (getenv("TEX_SAVE") && w >= 256) {
            static int _bs = 0;
            const u8* sp = vm_base + off;
            u32 nz = 0; u8 mn = 255, mx = 0;
            for (u32 i = 0; i < w * h; i += 17) {
                u8 v = sp[i]; if (v) nz++; if (v < mn) mn = v; if (v > mx) mx = v;
            }
            fprintf(stderr, "[TEXB8] off=0x%X %ux%u nz=%u/%u min=%u max=%u\n",
                    off, w, h, nz, (w * h) / 17, mn, mx);
            if (_bs < 8 && mx > mn) { _bs++;
                char pn[128];
                snprintf(pn, sizeof(pn), "texb8_%08X_%ux%u_%d.bmp", off, w, h, _bs);
                FILE* f = fopen(pn, "wb");
                if (f) {
                    u32 rowb = w * 3, rowp = (rowb + 3) & ~3u;
                    u32 fsz = 54 + rowp * h;
                    u8 hd[54] = {'B','M'};
                    hd[2]=(u8)fsz; hd[3]=(u8)(fsz>>8); hd[4]=(u8)(fsz>>16); hd[5]=(u8)(fsz>>24);
                    hd[10]=54; hd[14]=40;
                    hd[18]=(u8)w; hd[19]=(u8)(w>>8); hd[22]=(u8)h; hd[23]=(u8)(h>>8);
                    hd[26]=1; hd[28]=24;
                    fwrite(hd, 1, 54, f);
                    for (int y = (int)h - 1; y >= 0; y--) {
                        const u8* srow = sp + (u64)y * w;
                        for (u32 x = 0; x < w; x++) {
                            u8 px[3]; px[0]=px[1]=px[2]=srow[x];
                            fwrite(px, 1, 3, f);
                        }
                        { u8 z[3] = {0,0,0}; fwrite(z, 1, rowp - rowb, f); }
                    }
                    fclose(f);
                    fprintf(stderr, "[TEX_SAVE] %s\n", pn);
                }
            }
        }
    }
    /* TEX_SAVE=1: dump the first few converted ARGB uploads as BMPs (rgb +
     * alpha channel separately) -- ground-truth for "is the guest texture
     * wrong or is the sampling wrong" questions (wave's hue palette). */
    if (argb && _f + 1 == _nfaces && getenv("TEX_SAVE")) { static int _ts = 0; if (_ts < 8) { _ts++;
        /* For a cube, write every face: the point of the dump is to check the
         * assumed 6-face layout against the game's cubemap art. */
        for (u32 _face = 0; _face < _nfaces; _face++)
        for (int pass = 0; pass < 2; pass++) {
            char pn[128];
            if (_nfaces > 1)
                snprintf(pn, sizeof(pn), "cube_%08X_%ux%u_f%u_%s.bmp", _off0, w, h,
                         _face, pass ? "a" : "rgb");
            else
                snprintf(pn, sizeof(pn), "tex_%08X_%ux%u_%s.bmp", off, w, h,
                     pass ? "a" : "rgb");
            FILE* f = fopen(pn, "wb");
            if (f) {
                u32 rowb = w * 3, rowp = (rowb + 3) & ~3u;
                u32 fsz = 54 + rowp * h;
                u8 hd[54] = {'B','M'};
                hd[2]=(u8)fsz; hd[3]=(u8)(fsz>>8); hd[4]=(u8)(fsz>>16); hd[5]=(u8)(fsz>>24);
                hd[10]=54; hd[14]=40;
                hd[18]=(u8)w; hd[19]=(u8)(w>>8); hd[22]=(u8)h; hd[23]=(u8)(h>>8);
                hd[26]=1; hd[28]=24;
                fwrite(hd, 1, 54, f);
                for (int y = (int)h - 1; y >= 0; y--) {
                    const u8* srow = _map0 + (u64)_face * pitch * _face_rows
                                     + (u64)y * pitch;
                    for (u32 x = 0; x < w; x++) {
                        u8 px[3];
                        if (pass) { px[0]=px[1]=px[2]=srow[x*4+3]; }
                        else { px[0]=srow[x*4+2]; px[1]=srow[x*4+1]; px[2]=srow[x*4+0]; }
                        fwrite(px, 1, 3, f);
                    }
                    { u8 z[3] = {0,0,0}; fwrite(z, 1, rowp - rowb, f); }
                }
                fclose(f);
                fprintf(stderr, "[TEX_SAVE] %s\n", pn);
            }
        }
    } }
    /* TEX_AVGDBG=1: one line per distinct bound texture offset with its average
     * source colour. duck.tga is overwhelmingly yellow (avg ~243,191,23), so this
     * says whether the duck's image is bound at all -- and if it is, the duck is
     * being drawn somewhere and the question is where, not whether. */
    { static int avgd = -1;
      if (avgd < 0) { const char* e = getenv("TEX_AVGDBG"); avgd = e ? atoi(e) : 0; }
      if (avgd && !dxt && bpp == 4) {
        static u32 seen[64]; static int nseen = 0;
        int known = 0;
        for (int i2 = 0; i2 < nseen; i2++) if (seen[i2] == off) { known = 1; break; }
        if (!known && nseen < 64) {
            seen[nseen++] = off;
            /* Report the source as R,G,B,A -- the layout TEX_RGBA uploads (PSGL
             * writes GL_RGBA/UNSIGNED_INT_8_8_8_8, so bytes land R,G,B,A). Alpha
             * matters as much as colour here: a blended surface whose alpha is a
             * flat 255 renders opaque no matter how correct the blend state is. */
            u64 sr = 0, sg = 0, sb = 0, sa = 0; u32 cnt = 0, amin = 255, amax = 0;
            for (u32 i2 = 0; i2 + 3 < w * h * 4u; i2 += 4 * 37) {
                const u8* q = vm_base + off + i2;
                sr += q[0]; sg += q[1]; sb += q[2]; sa += q[3]; cnt++;
                if (q[3] < amin) amin = q[3];
                if (q[3] > amax) amax = q[3];
            }
            if (cnt) fprintf(stderr, "[TEXAVG] raw=0x%08X %4ux%-4u fmt=0x%02X"
                                     " avgRGBA=(%3u,%3u,%3u,%3u) alpha[%u..%u]%c",
                             key_off, w, h, fmt, (u32)(sr/cnt), (u32)(sg/cnt),
                             (u32)(sb/cnt), (u32)(sa/cnt), amin, amax, 10);
        }
      } }
    /* Content-identify the duck's texture so filters can say "duck". */
    if (!dxt && bpp == 4 && w == 512 && h == 512 && s_duck_raw != key_off) {
        u64 ar = 0, ag = 0, ab = 0; u32 n = 0;
        for (u32 i2 = 0; i2 + 3 < w * h * 4u; i2 += 4 * 211) {
            const u8* q = vm_base + off + i2;
            ar += q[0]; ag += q[1]; ab += q[2]; n++;
        }
        if (n && ar/n > 200 && ag/n > 140 && ab/n < 90) {
            s_duck_off = key_off;
            { static u32 last = 0; if (last != key_off) { last = key_off;
                fprintf(stderr, "[DUCKTEX] resolved=0x%08X identified by content%c", key_off, 10); } }
        }
    }
    /* TEX_HILITE=1: paint the duck's texture pure red. duck.tga is overwhelmingly
     * yellow (avg 243,191,23); with the R,G,B,A source order that is bytes
     * s[0] high, s[1] mid, s[2] low. Flooding it with an unmistakable colour
     * answers "where on screen is the duck drawn" directly, which neither the
     * vertex data nor a pixel count can. Diagnostic only. */
    { static int hil = -1;
      if (hil < 0) { const char* e = getenv("TEX_HILITE"); hil = e ? atoi(e) : 0; }
      if (hil && !dxt && bpp == 4 && w == 512 && h == 512) {
        u64 sr = 0, sg = 0, sb = 0; u32 cnt = 0;
        for (u32 i2 = 0; i2 + 3 < w * h * 4u; i2 += 4 * 37) {
            const u8* q = vm_base + off + i2;
            sr += q[0]; sg += q[1]; sb += q[2]; cnt++;
        }
        if (cnt) {
            u32 ar = (u32)(sr/cnt), ag = (u32)(sg/cnt), ab = (u32)(sb/cnt);
            if (ar > 200 && ag > 140 && ab < 90) {
                for (u32 y = 0; y < h; y++) {
                    u8* drow = (u8*)mapped + (u64)y * pitch;
                    for (u32 x = 0; x < w; x++) {
                        drow[x*4+0] = 0xFF; drow[x*4+1] = 0x00;
                        drow[x*4+2] = 0x00; drow[x*4+3] = 0xFF;
                    }
                }
                { static int _n = 0; if (PS3_LOG_FIRST(_n, 4))
                    fprintf(stderr, "[TEXHILITE] duck texture off=0x%08X avg=(%u,%u,%u) -> red%c",
                            off, ar, ag, ab, 10); }
            }
        }
      } }
    /* TEX_MARKEMPTY=1: paint any texture whose guest source is entirely zero a
     * flat magenta instead of uploading the empty bytes. An unresolved texture
     * and a legitimately black surface are indistinguishable on screen, so this
     * says WHICH geometry is missing its image -- diagnostic only. */
    { static int mark = -1;
      if (mark < 0) { const char* e = getenv("TEX_MARKEMPTY"); mark = e ? atoi(e) : 0; }
      if (mark && !dxt) {
        u32 nz = 0, sz = w * h * bpp;
        for (u32 i2 = 0; i2 < sz && i2 < 0x40000u; i2 += 61) if (vm_base[off + i2]) nz++;
        if (!nz) {
            for (u32 y = 0; y < h; y++) {
                u8* drow = (u8*)mapped + (u64)y * pitch;
                for (u32 x = 0; x < w; x++) {
                    /* Pure RED. Magenta collided with this title's wall tint,
                     * so the marker was indistinguishable from real geometry. */
                    drow[x*4+0] = 0xFF; drow[x*4+1] = 0x00;
                    drow[x*4+2] = 0x00; drow[x*4+3] = 0xFF;
                }
            }
            { static int _n = 0; if (PS3_LOG_FIRST(_n, 8))
                fprintf(stderr, "[TEXEMPTY] off=0x%08X %ux%u fmt=0x%02X -> magenta%c",
                        off, w, h, fmt, 10); }
        }
      } }
    }   /* end per-face conversion */
    off = _off0; mapped = _map0;
    t->up[upload_slot]->lpVtbl->Unmap(t->up[upload_slot], 0, NULL);

    if (!fresh) {   /* reused resource: PSR -> COPY_DEST first */
        D3D12_RESOURCE_BARRIER b = {0};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = t->res;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    }
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = t->res; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = t->up[upload_slot]; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format   = dxfmt;
    src.PlacedFootprint.Footprint.Width    = w;
    src.PlacedFootprint.Footprint.Height   = h;
    src.PlacedFootprint.Footprint.Depth    = 1;
    src.PlacedFootprint.Footprint.RowPitch = pitch;
    for (u32 _f = 0; _f < _nfaces; _f++) {
        dst.SubresourceIndex        = _f;      /* cube face = array slice */
        src.PlacedFootprint.Offset  = (u64)_f * pitch * _face_rows;
        s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dst, 0, 0, 0, &src, NULL);
    }
    {
        D3D12_RESOURCE_BARRIER b = {0};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = t->res;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    }
    /* SRV at heap slot 1+slot */
    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {0};
    sv.Format = dxfmt;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    /* RSX B8 replicates the byte into all four channels (dbgfont's FP reads
     * coverage from .w); DXGI R8 defaults to (r,0,0,1), so swizzle (R,R,R,R)
     * = encoded 0x1000 (component 0 in all lanes + always-set bit). ARGB8
     * keeps the identity mapping. */
    sv.Shader4ComponentMapping = d3d12_texture_component_mapping(0, basef);
    sv.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE sh;
    s_d3d.srv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &sh);
    sh.ptr += (u64)(1 + slot) * s_d3d.srv_inc;
    if (cube) {
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        sv.TextureCube.MipLevels = 1;
        sv.TextureCube.MostDetailedMip = 0;
        sv.TextureCube.ResourceMinLODClamp = 0.0f;
    }
    s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, t->res, &sv, sh);

    t->off = off; t->key = key_off; t->w = w; t->h = h; t->fmt = fmt;
    t->pitch = row_pitch; t->mips = mips;
    t->dxgi = dxfmt;
    t->cube = cube; t->used = 1;
    t->last_use = ++s_vp_tex_clock;
    s_perf_tex_uploads++;
    { u32 nb = cube ? rsx_texture_cube_face_stride(fmt, w, h, mips, row_pitch) * 6u
                    : tl.face_bytes;
      t->csum = TEX_CSUM(vm_base + off, nb); }
    if (d3d12_index_cache_enabled()) vp_tex_index_insert((u32)slot, hint_key);
    slot_hint[hint_key][slot_hint_next[hint_key]++ & 3u] = (u32)slot + 1u;
    #undef TEX_CSUM
    return slot;
}

/* Formats implemented by rsx_texture_layout_pitched() and mapped to a typed
 * DXGI SRV in vp_upload_tex_slot(). Keep this explicit so unsupported RSX
 * encodings do not silently become a misleading R8 texture. */
static int vp_texture_format_supported(u32 fmt)
{
    switch (fmt & 0x9Fu) {
    case 0x81u:
    case 0x82u: case 0x83u: case 0x84u: case 0x85u:
    case 0x86u: case 0x87u: case 0x88u:
    case 0x8Bu: case 0x8Cu: case 0x8Du:
    case 0x90u: case 0x92u: case 0x94u: case 0x95u:
    case 0x97u: case 0x9Au: case 0x9Bu: case 0x9Cu:
    case 0x9Du: case 0x9Eu: case 0x9Fu:
        return 1;
    default:
        return 0;
    }
}

/* ---------------------------------------------------------------------------
 * Render-to-texture: offscreen RT pool.
 *
 * Draws/clears targeting a non-display surface (SET_SURFACE_COLOR_xOFFSET
 * not registered via cellGcmSetDisplayBuffer) render into a pooled RGBA8
 * texture keyed by the surface's resolved vm offset; a later draw binding a
 * texture at that offset samples the RT directly instead of guest memory.
 * RGBA8 for every RT (even half-float guest surfaces) keeps the existing
 * PSO/RTV formats -- values clamp to [0,1], which demosaic's RGB data fits.
 * -----------------------------------------------------------------------*/

/* Snapshot the VP constant bank + viewport epilogue for one draw into its
 * vp_cb slot. Runs at record time so every draw keeps the constants that
 * were live when the guest issued it. */
static void vp_record_cb(u32 slot, int vs_idx, D3D12DrawRecord* dr)
{
    const rsx_state* st = s_d3d.current_rsx_state;
    if (!s_d3d.vp_cb_mapped || !st || slot >= MAX_DRAWS) return;
    /* Snapshot the same runtime CONST layout emitted by the buffered FP
     * decompiler. Baking these values while hashing only instructions reused
     * stale camera/material/postprocess constants in cached pipelines. */
    if (s_d3d.vp_fpcb_mapped) {
        float* fp = (float*)((char*)s_d3d.vp_fpcb_mapped
            + ((u64)s_d3d.vp_parity * MAX_DRAWS + slot) * VP_FPCB_STRIDE);
        u32 count = 0;
        /* fp_k[]: the program's inline fragment constants, re-read from the
         * guest ucode every draw. The decompiler emits fp_k[i] lookups instead
         * of baking these in as literals, so the compiled shader no longer
         * changes when the title re-patches them -- which is what forced the
         * pipeline cache to key on a hash of the ucode bytes. */
        if (dr && dr->fp_addr) {
            extern uint8_t* vm_base;
            extern u32 cellGcmResolveLocated(int, u32);
            u32 off = cellGcmResolveLocated((dr->fp_addr & 3u) == 1,
                                             dr->fp_addr & ~3u);
            const rsx_fp_metadata* metadata = vm_base && off != 0xFFFFFFFFu
                ? rsx_fp_metadata_get(&s_fp_metadata, vm_base + off) : NULL;
            if (metadata && metadata->constants.count <= VP_FP_MAX_CONSTANTS) {
                count = metadata->constants.count;
                memcpy(fp, metadata->constants.values, count * 16u);
            }
        }
        float* alpha = fp + (count ? count : 1u) * 4u;
        alpha[0] = dr ? dr->alpha_ref : 0.0f;
        alpha[1] = dr && !d3d_flag_NO_ALPHATEST()
            ? (float)((dr->alpha_ctl >> 16) & 1u) : 0.0f;
        alpha[2] = dr ? (float)((dr->alpha_ctl >> 8) & 7u) : 7.0f;
    }
    _Alignas(16) u8 snapshot[VP_CB_STRIDE] = {0};
    char* dst = (char*)snapshot;
    memcpy(dst, st->vertex_constants, RSX_MAX_VERTEX_CONSTANTS * 16);
    /* VP_MVP=<N>: the constant bank as the shader will see it, for the first N
     * draws. A snapshot taken from a stale rsx_state looks exactly like a broken
     * vertex program from the outside -- both give zero fragments. */
    { static int cap = -1, seen = 0;
      if (cap < 0) { const char* e = getenv("VP_MVP"); cap = e ? atoi(e) : 0; }
      if (cap && seen < cap) { seen++;
        const float* c = (const float*)dst;
        int last_nz = -1;
        for (int i = 0; i < RSX_MAX_VERTEX_CONSTANTS * 4; i++)
            if (c[i] > 1e-9f || c[i] < -1e-9f) last_nz = i / 4;
        fprintf(stderr, "[VPMVP] slot=%u lastNZ=c%d  c260=(%g %g %g %g) c467=(%g %g %g %g)\n",
                slot, last_nz,
                c[260*4+0], c[260*4+1], c[260*4+2], c[260*4+3],
                c[467*4+0], c[467*4+1], c[467*4+2], c[467*4+3]); } }
    /* RSX_DBG_CB=<hex tex0 offset>: the constants as they land in the per-draw
     * constant buffer for that texture's draws -- i.e. exactly what the shader
     * samples, not what the CPU-side rsx_state holds. Those two agreeing is an
     * assumption worth checking directly when a draw transforms to nothing. */
    const float* vpx_dbg = (const float*)(dst + RSX_MAX_VERTEX_CONSTANTS * 16);
    { static const char* dc = (const char*)1; static u32 wantc = 0; static int n = 0;
      if (dc == (const char*)1) { dc = getenv("RSX_DBG_CB");
                                  wantc = dc ? (u32)strtoul(dc, NULL, 16) : 0; }
      if (wantc && dr && dr->tex[0].raw == wantc && n < 4) { n++;
          const float* c = (const float*)dst;
          fprintf(stderr, "[DUCKCB] slot=%u vs_idx=%d posscale=(%g %g %g) posoffset=(%g %g %g)%c",
                  slot, vs_idx, vpx_dbg[0], vpx_dbg[1], vpx_dbg[2],
                  vpx_dbg[4], vpx_dbg[5], vpx_dbg[6], 10);
          for (int r = 256; r <= 259; r++)
              fprintf(stderr, "[DUCKCB]   c%d=(%g %g %g %g)%c", r,
                      c[r*4+0], c[r*4+1], c[r*4+2], c[r*4+3], 10);
      } }

    /* Convert the guest window transform back to D3D clip coordinates.
     * D3D's viewport maps positive clip Y upward; RSX programs the sign
     * explicitly. Shadow passes can use positive Y scale while the main
     * scene uses negative Y scale, so an identity XY epilogue mirrors only
     * the shadow map relative to its consumers. */
    float* vpx = (float*)(dst + RSX_MAX_VERTEX_CONSTANTS * 16);
    const float* vs_ = st->viewport_scale;
    const float* vo_ = st->viewport_offset;
    vpx[0] = vpx[1] = vpx[3] = 1.0f;
    vpx[4] = vpx[5] = vpx[7] = 0.0f;
    {
        float tw = st->surface_clip_w ? (float)st->surface_clip_w : (float)s_d3d.width;
        float th = st->surface_clip_h ? (float)st->surface_clip_h : (float)s_d3d.height;
        float vx = 0.0f, vy = 0.0f;
        if (st->viewport_w >= 2 && st->viewport_h >= 2 &&
            (float)(st->viewport_x + st->viewport_w) <= tw + 0.5f &&
            (float)(st->viewport_y + st->viewport_h) <= th + 0.5f) {
            vx = (float)st->viewport_x; vy = (float)st->viewport_y;
            tw = (float)st->viewport_w; th = (float)st->viewport_h;
        }
        if (tw > 0.0f && th > 0.0f && vs_[0] != 0.0f && vs_[1] != 0.0f) {
            vpx[0] = 2.0f * vs_[0] / tw;
            vpx[1] = -2.0f * vs_[1] / th;
            vpx[4] = 2.0f * (vo_[0] - vx) / tw - 1.0f;
            vpx[5] = 1.0f - 2.0f * (vo_[1] - vy) / th;
        }
    }
    if (vs_[2] != 0.0f) { vpx[2] = vs_[2]; vpx[6] = vo_[2]; }
    else                { vpx[2] = 1.0f;   vpx[6] = 0.0f;   }
    /* DBG_ZOOM=<k> with DBG_CENTER="cx,cy": a debug camera applied in CLIP space,
     * not a framebuffer crop -- the geometry is re-rasterized at full resolution.
     * The shader epilogue already computes
     *     pos.xyz = _p.xyz * vp_posscale + _p.w * vp_posoffset
     * so recentring NDC point c and magnifying by k is exactly
     *     pos.xy = _p.xy * k + _p.w * (-c * k).
     * Rubber Ducky frames its ducks near the bottom edge of the viewport and its
     * camera only moves from the analog sticks, so without this there is no way
     * to look at the subject of the demo. It changes the view, nothing else:
     * same draws, same transform, same textures. */
    { static int z = -1; static float k = 1.0f, cx = 0.0f, cy = 0.0f;
      if (z < 0) { const char* e = getenv("DBG_ZOOM");
                   k = e ? (float)atof(e) : 1.0f;
                   z = (e && k > 0.0f) ? 1 : 0;
                   const char* ce = getenv("DBG_CENTER");
                   if (ce) { double a = 0, b = 0; sscanf(ce, "%lf,%lf", &a, &b);
                             cx = (float)a; cy = (float)b; } }
      if (z) { float ux = cx, uy = cy;
               static int lock = -1;
               if (lock < 0) { const char* le = getenv("DBG_LOCK"); lock = le ? atoi(le) : 0; }
               if (lock && s_lock_valid) { ux = s_lock_x; uy = s_lock_y; }
               vpx[0] = k; vpx[1] = k; vpx[4] = -ux * k; vpx[5] = -uy * k; } }

    /* The generic vkcube projection override is not an RSX rule. In particular,
     * c[0] can legitimately be negative for a game's coordinate convention;
     * replacing c[0..3] based on its sign destroys the guest transform. Keep
     * that old diagnostic available only when explicitly requested. */
    float* c = (float*)dst;
    float c00 = c[0];
    if (getenv("VP_FIXPROJ")) {
        { static int _fb = 0; if (PS3_LOG_FIRST(_fb, 6))
            printf("[VPFB] fallback proj on draw slot %u (c00=%g vs_idx=%d)\n",
                   slot, c00, vs_idx); }
        for (int _i = 0; _i < 16; _i++) c[_i] = 0.0f;
        c[0]=1.358f; c[5]=2.414f; c[11]=1.0f;
        c[10]=1.0f/99.0f; c[14]=-1.0f/99.0f;
        vpx[2] = 1.0f; vpx[6] = 0.0f;
    }
    /* NV4097_SET_TRANSFORM_BRANCH_BITS follows the 512 constants and two
     * viewport float4s in VPConst (offset 8224). */
    memcpy(dst + (RSX_MAX_VERTEX_CONSTANTS + 2u) * 16u,
           &st->transform_branch_bits, sizeof(st->transform_branch_bits));
    dr->vp_cb_slot = rsx_constant_snapshot_record(&s_vp_cb_cache,snapshot,slot,
        (u8*)s_d3d.vp_cb_mapped + (u64)s_d3d.vp_parity * MAX_DRAWS * VP_CB_STRIDE);

}

/* Give each registered PS3 scanout surface its own D3D12 render target. The
 * guest may register offset 0, so 0 cannot also mean "no render target" as it
 * did in the old swapchain-alias path. RSX local-memory offsets are far below
 * this reserved key range. */
static int display_rt_id_from_key(u32 key)
{
    if (key < DISPLAY_RT_KEY_BASE ||
        key >= DISPLAY_RT_KEY_BASE + CELL_GCM_MAX_DISPLAY_BUFFER_NUM)
        return -1;
    return (int)(key - DISPLAY_RT_KEY_BASE);
}

static u32 display_rt_key(u32 id)
{
    return DISPLAY_RT_KEY_BASE + id;
}

static u32 canonical_rt_key(u32 raw_off)
{
    if (display_rt_id_from_key(raw_off) >= 0) return raw_off;
    s32 id = cellGcmDisplayBufferIdFromOffset(raw_off);
    return id >= 0 ? display_rt_key((u32)id) : raw_off;
}

static int is_display_target_key(u32 key)
{
    /* Zero remains the legacy/direct-to-swapchain target for titles without a
     * registered display surface. Registered buffers use nonzero stable keys. */
    return key == 0 || display_rt_id_from_key(key) >= 0;
}

/* Which offscreen surface (if any) does the current RSX state render to?
 * Registered display buffers use stable synthetic keys, while other RTs keep
 * their raw RSX offsets. RTs are keyed by offset -- a texture bound at the same raw
 * offset is the same buffer (surface and texture registers share the offset
 * space; location bits differ but one title doesn't alias local vs main at
 * one offset), which sidesteps guessing the surface's context DMA location. */
static u32 current_rt_off(u32* out_w, u32* out_h, u32 out_mrt[3])
{
    const rsx_state* st = s_d3d.current_rsx_state;
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    if (out_mrt) out_mrt[0] = out_mrt[1] = out_mrt[2] = 0;
    if (!st) return 0;
    if (st->color_target == CELL_GCM_SURFACE_TARGET_NONE) {
        u32 w = st->surface_clip_w, h = st->surface_clip_h;
        if (w < 16 || w > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            h < 16 || h > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION) w = h = 0;
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0;
    }
    /* SET_SURFACE_COLOR_TARGET: 1 = A, 2 = B, 0x13 = MRT1 (A+B),
     * 0x17 = MRT2 (A+B+C), 0x1F = MRT3 (A+B+C+D). */
    int sel = (st->color_target == 2) ? 1 : 0;
    u32 raw = st->surface_color_offset[sel];
    if (out_mrt) {
        if (st->color_target >= 0x13) out_mrt[0] = st->surface_color_offset[1];
        if (st->color_target >= 0x17) out_mrt[1] = st->surface_color_offset[2];
        if (st->color_target >= 0x1F) out_mrt[2] = st->surface_color_offset[3];
        if (st->color_target >= 0x13) out_mrt[0] = canonical_rt_key(out_mrt[0]);
        if (st->color_target >= 0x17) out_mrt[1] = canonical_rt_key(out_mrt[1]);
        if (st->color_target >= 0x1F) out_mrt[2] = canonical_rt_key(out_mrt[2]);
    }
    const s32 display_id = cellGcmDisplayBufferIdFromOffset(raw);
    const u32 target_key = display_id >= 0 ? display_rt_key((u32)display_id) : raw;
    { static int _rs = -1; if (_rs < 0) _rs = getenv("RT_SEQDBG") ? 1 : 0;
      static u32 _last = 0; static int _n2 = 0;
      if (_rs && st->surface_color_offset[0] != _last && _n2 < 400000) {
          _last = st->surface_color_offset[0]; _n2++;
          fprintf(stderr, "[RTSEQ] #%d -> 0x%X%c", _n2, _last, 10); } }
    { static int _ro = -1; if (_ro < 0) _ro = getenv("RT_OFFDBG") ? 1 : 0;
      static u32 _seen[32]; static int _ns = 0;
      if (_ro) { u32 _k = st->surface_color_offset[0] ^ (st->color_target << 24)
                        ^ (st->surface_clip_w << 8) ^ st->surface_format;
        int _f = 0; for (int _i = 0; _i < _ns; _i++) if (_seen[_i] == _k) _f = 1;
        if (!_f && _ns < 32) { _seen[_ns++] = _k;
        fprintf(stderr, "[RTOFF] fmt=0x%X tgt=0x%X off[0]=0x%X off[1]=0x%X zeta=0x%X clip=%ux%u disp=%d -> 0x%X\n",
                st->surface_format, st->color_target, st->surface_color_offset[0], st->surface_color_offset[1],
                st->surface_zeta_offset, st->surface_clip_w, st->surface_clip_h,
                display_id >= 0, target_key); } } }
    if (display_id >= 0) {
        CellGcmDisplayInfo* info = cellGcmGetDisplayBufferByFlipIndex((u32)display_id);
        if (info) {
            if (out_w) *out_w = info->width;
            if (out_h) *out_h = info->height;
        }
        return target_key;
    }
    /* RT_DISPLAY_BY_SIZE=1: also treat a single-target surface whose clip
     * exactly matches the display resolution as the backbuffer.
     *
     * cellGcmSetDisplayBuffer only registers the buffers the FLIP may point at;
     * a guest is free to render into a different surface of the same size and
     * flip to it later (PSGL does). Matching offsets alone then classifies every
     * draw as offscreen, has_display stays 0, and render_frame() -- which is
     * what draws ALL recorded geometry -- is never called. The backbuffer shows
     * only the clear, which looks exactly like "nothing rasterizes". */
    { static int _bs = -1; if (_bs < 0) _bs = getenv("RT_DISPLAY_BY_SIZE") ? 1 : 0;
      if (_bs && st->color_target < 0x13 &&
          st->surface_clip_w == s_d3d.width && st->surface_clip_h == s_d3d.height)
          return 0; }
    /* SET_WINDOW_OFFSET moves the color surfaces within their pitched backing
     * allocations. Apply the register as a byte address adjustment so target
     * identity matches the address later used by a texture read. */
    {
        const u32 x = st->window_offset & 0xFFFFu;
        const u32 y = st->window_offset >> 16;
        if (x || y) {
            const u32 format = st->surface_format & 0x1Fu;
            const u32 bpp = (format == 3u) ? 2u :
                            (format == 11u) ? 8u :
                            (format == 12u) ? 16u : 4u;
            const u32 indices[4] = { (u32)sel, 1u, 2u, 3u };
            u32* offsets[4] = { &raw, out_mrt ? &out_mrt[0] : NULL,
                                out_mrt ? &out_mrt[1] : NULL,
                                out_mrt ? &out_mrt[2] : NULL };
            const u32 targets = !out_mrt ? 1u :
                                st->color_target >= 0x1Fu ? 4u :
                                st->color_target >= 0x17u ? 3u :
                                st->color_target >= 0x13u ? 2u : 1u;
            for (u32 i = 0; i < targets; ++i) {
                const u32 index = indices[i];
                const u32 pitch = st->surface_color_pitch[index]
                    ? st->surface_color_pitch[index]
                    : st->surface_clip_w * bpp;
                const u64 adjusted = (u64)*offsets[i] + (u64)y * pitch + (u64)x * bpp;
                if (*offsets[i] && adjusted <= 0xFFFFFFFFu)
                    *offsets[i] = (u32)adjusted;
            }
        }
    }
    /* Surface clip dims when sane; else the window size. Any size works --
     * passes draw normalized full-surface quads -- this only picks resolution. */
    u32 w = st->surface_clip_w, h = st->surface_clip_h;
    if (w < 16 || w > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        h < 16 || h > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION) { w = 0; h = 0; }
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return raw;
}

/* RSX surface colour format (SET_SURFACE_FORMAT bits [4:0]) -> DXGI. Float
 * targets matter: wave's water height maps store SIGNED values (F_W16..),
 * demosaic's differential planes likewise -- RGBA8 clamps them to zero. */
static DXGI_FORMAT rsx_surface_dxgi(u32 fmt)
{
    switch (fmt & 0x1F) {
    case 0x0B: return DXGI_FORMAT_R16G16B16A16_FLOAT; /* F_W16Z16Y16X16 */
    case 0x0C: return DXGI_FORMAT_R32G32B32A32_FLOAT; /* F_W32Z32Y32X32 */
    case 0x0D: return DXGI_FORMAT_R32_FLOAT;          /* F_X32          */
    default:   return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

static int off_rt_find(u32 off)
{
    off = canonical_rt_key(off);
    if (!off) return -1;
    static u32 hints[256];
    const u32 key=((off>>8)^(off>>16)^off)&255u;
    const u32 hint=hints[key];
    if(hint && s_d3d.off_rt[hint-1].res && s_d3d.off_rt[hint-1].off==off) return (int)hint-1;
    for (int i = 0; i < MAX_OFF_RTS; i++)
        if (s_d3d.off_rt[i].res && s_d3d.off_rt[i].off == off) {hints[key]=(u32)i+1;return i;}
    return -1;
}

/* Ensure an RT resource exists for this surface key; (re)creates the RTV at
 * rt_rtv_heap[i] and the SRV at srv_heap[RT_SRV_BASE+i]. */
static int off_rt_get(u32 off, u32 w, u32 h, u32 rsx_fmt)
{
    off = canonical_rt_key(off);
    if (!off || !s_d3d.rt_rtv_heap || !s_d3d.srv_heap) return -1;
    if (!w) w = s_d3d.width;
    if (!h) h = s_d3d.height;
    DXGI_FORMAT want_fmt = rsx_surface_dxgi(rsx_fmt);
    /* RT_FORCE_RGBA8=1: allocate every offscreen target as R8G8B8A8 rather than
     * its native format. A title whose final composite reads a half-float
     * surface cannot be shown with a plain CopyTextureRegion, and giving it an
     * 8-bit target costs HDR precision but makes the composite a same-format
     * copy. A stopgap for looking at the frame, not a rendering fix. */
    { static int f8 = -1;
      if (f8 < 0) { const char* e = getenv("RT_FORCE_RGBA8"); f8 = e ? atoi(e) : 0; }
      if (f8) want_fmt = DXGI_FORMAT_R8G8B8A8_UNORM; }
    int slot = off_rt_find(off);
    if (slot >= 0) {
        OffRT* r = &s_d3d.off_rt[slot];
        if (r->w == w && r->h == h && r->dxgi == (u32)want_fmt) { r->used = 1; return slot; }
        /* Retain the old version through the next submission fence, including
         * commands already recorded into the current open list. */
        if (perf_on()) s_perf_rt_recreates++;
        if (!d3d12_retire_resource(&r->res) ||
            !d3d12_retire_resource(&r->up) ||
            !d3d12_retire_resource(&r->feedback)) return -1;
    } else {
        for (int i = 0; i < MAX_OFF_RTS; i++)
            if (!s_d3d.off_rt[i].res) { slot = i; break; }
        if (slot < 0) {   /* evict an entry not used this frame */
            for (int i = 0; i < MAX_OFF_RTS; i++)
                if (!s_d3d.off_rt[i].used) { slot = i; break; }
            if (slot < 0) return -1;
            if (perf_on()) s_perf_rt_evicts++;
            if (!d3d12_retire_resource(&s_d3d.off_rt[slot].res) ||
                !d3d12_retire_resource(&s_d3d.off_rt[slot].feedback) ||
                !d3d12_retire_resource(&s_d3d.off_rt[slot].up)) return -1;
        }
    }
    OffRT* r = &s_d3d.off_rt[slot];
    D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = want_fmt; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv = {0};
    cv.Format = td.Format;
    if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &td,
            D3D12_RESOURCE_STATE_COPY_DEST, &cv,
            &IID_ID3D12Resource, (void**)&r->res)))
        return -1;
    r->off = off; r->w = w; r->h = h; r->dxgi = (u32)want_fmt;
    r->color_serial = 0;
    r->st = D3D12_RESOURCE_STATE_COPY_DEST;
    r->used = 1;
    r->contents_valid = 0;

    /* Populate the new RT from guest memory: titles CPU-initialise their
     * render-target buffers (wave fills the height fields with texels of
     * (0,0,0, 1.0f) -- the .w is 'inverse of mass'; from an all-zero GPU
     * resource the water simulation can never boot). Guest data is
     * big-endian and tightly packed. */
    {
        extern uint8_t* vm_base;
        extern u32 cellGcmResolveOffset(u32);
        u32 bpp = (want_fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 8u :
                  (want_fmt == DXGI_FORMAT_R32G32B32A32_FLOAT) ? 16u :
                  (want_fmt == DXGI_FORMAT_R32_FLOAT) ? 4u : 4u;
        u32 pitch = (w * bpp + 255) & ~255u;
        u32 src = cellGcmResolveOffset(off);
        if (vm_base && src != 0xFFFFFFFFu) {
            D3D12_HEAP_PROPERTIES hu = {0}; hu.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC bd = {0};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = (u64)pitch * h; bd.Height = 1; bd.DepthOrArraySize = 1;
            bd.MipLevels = 1; bd.SampleDesc.Count = 1;
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (SUCCEEDED(s_d3d.device->lpVtbl->CreateCommittedResource(
                    s_d3d.device, &hu, D3D12_HEAP_FLAG_NONE, &bd,
                    D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                    &IID_ID3D12Resource, (void**)&r->up))) {
                void* mp = NULL; D3D12_RANGE nr = {0,0};
                if (SUCCEEDED(r->up->lpVtbl->Map(r->up, 0, &nr, &mp)) && mp) {
                    const u8* sp = vm_base + src;
                    for (u32 y = 0; y < h; y++) {
                        const u8* srow = sp + (u64)y * w * bpp;
                        u8* drow = (u8*)mp + (u64)y * pitch;
                        if (want_fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
                            /* guest A8R8G8B8 (bytes A,R,G,B) -> R,G,B,A */
                            for (u32 x = 0; x < w; x++) {
                                drow[x*4+0] = srow[x*4+1];
                                drow[x*4+1] = srow[x*4+2];
                                drow[x*4+2] = srow[x*4+3];
                                drow[x*4+3] = srow[x*4+0];
                            }
                        } else if (want_fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                            for (u32 x = 0; x < w * 4; x++) {   /* u16 halves, BE */
                                drow[x*2+0] = srow[x*2+1];
                                drow[x*2+1] = srow[x*2+0];
                            }
                        } else {                                 /* u32 floats, BE */
                            u32 nw = (w * bpp) / 4;
                            for (u32 x = 0; x < nw; x++) {
                                drow[x*4+0] = srow[x*4+3];
                                drow[x*4+1] = srow[x*4+2];
                                drow[x*4+2] = srow[x*4+1];
                                drow[x*4+3] = srow[x*4+0];
                            }
                        }
                    }
                    r->up->lpVtbl->Unmap(r->up, 0, NULL);
                    D3D12_TEXTURE_COPY_LOCATION cdst = {0}, csrc = {0};
                    cdst.pResource = r->res;
                    cdst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    csrc.pResource = r->up;
                    csrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    csrc.PlacedFootprint.Footprint.Format   = want_fmt;
                    csrc.PlacedFootprint.Footprint.Width    = w;
                    csrc.PlacedFootprint.Footprint.Height   = h;
                    csrc.PlacedFootprint.Footprint.Depth    = 1;
                    csrc.PlacedFootprint.Footprint.RowPitch = pitch;
                    s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &cdst, 0, 0, 0, &csrc, NULL);
                }
            }
        }
        {
            D3D12_RESOURCE_BARRIER b = {0};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource   = r->res;
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
            r->st = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
    }

    D3D12_CPU_DESCRIPTOR_HANDLE rh;
    s_d3d.rt_rtv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.rt_rtv_heap, &rh);
    rh.ptr += (u64)slot * s_d3d.device->lpVtbl->GetDescriptorHandleIncrementSize(
        s_d3d.device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    s_d3d.device->lpVtbl->CreateRenderTargetView(s_d3d.device, r->res, NULL, rh);

    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {0};
    sv.Format = td.Format;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE sh;
    s_d3d.srv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &sh);
    sh.ptr += (u64)(RT_SRV_BASE + slot) * s_d3d.srv_inc;
    s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, r->res, &sv, sh);

    static int _log = 0;
    if (PS3_LOG_FIRST(_log, 8))
        printf("[D3D12] offscreen RT %d: off=0x%X %ux%u (render-to-texture)\n",
               slot, off, w, h);
    return slot;
}

static D3D12_CPU_DESCRIPTOR_HANDLE off_rt_rtv(int slot)
{
    D3D12_CPU_DESCRIPTOR_HANDLE rh;
    s_d3d.rt_rtv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.rt_rtv_heap, &rh);
    rh.ptr += (u64)slot * s_d3d.device->lpVtbl->GetDescriptorHandleIncrementSize(
        s_d3d.device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    return rh;
}

#include "rsx_d3d12_srv_cache.h"
static rsx_d3d12_srv_cache s_srv_cache;

/* Write a texture SRV (or a null SRV when res == NULL) at an absolute SRV
 * heap slot. Used to fill per-draw t0-t15 descriptor windows. */
static void srv_write_ex(u32 heap_slot, ID3D12Resource* res, DXGI_FORMAT fmt,
                         UINT mapping, int cube)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {0};
    sv.Format = fmt;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = mapping;
    sv.Texture2D.MipLevels = 1;
    if (cube) {   /* must match the TextureCube declaration in the HLSL */
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        sv.TextureCube.MipLevels = 1;
        sv.TextureCube.MostDetailedMip = 0;
        sv.TextureCube.ResourceMinLODClamp = 0.0f;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    s_d3d.srv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &h);
    h.ptr += (u64)heap_slot * s_d3d.srv_inc;
    rsx_d3d12_srv_cache_write(&s_srv_cache, s_d3d.device, s_d3d.srv_inc, h, res, &sv);
}

static void srv_write(u32 heap_slot, ID3D12Resource* res, DXGI_FORMAT fmt, UINT mapping)
{
    srv_write_ex(heap_slot, res, fmt, mapping, 0);
}

static int off_depth_find(u32 ea)
{
    if (!ea || ea == 0xFFFFFFFFu) return -1;
    static u32 hints[256];
    const u32 key=((ea>>8)^(ea>>16)^ea)&255u;
    const u32 hint=hints[key];
    if(hint && s_d3d.off_depth[hint-1].res && s_d3d.off_depth[hint-1].ea==ea) return (int)hint-1;
    for(int i=0;i<MAX_OFF_DEPTHS;++i)
        if(s_d3d.off_depth[i].res && s_d3d.off_depth[i].ea==ea) {hints[key]=(u32)i+1;return i;}
    return -1;
}

static u64 d3d12_next_surface_write_serial(void)
{
    if (++s_d3d.surface_write_serial == 0)
        ++s_d3d.surface_write_serial;
    return s_d3d.surface_write_serial;
}

static void d3d12_mark_color_surface(u32 key, u64 serial)
{
    const int slot = off_rt_find(key);
    if (slot >= 0) s_d3d.off_rt[slot].color_serial = serial;
}

/* Record the guest's ordered writes while the per-draw SRVs are prepared.
 * This mirrors the shared draw engine's color/depth write ordering: a texture
 * alias reads whichever surface was written most recently. */
static void d3d12_record_surface_writes(const D3D12DrawRecord* dr)
{
    if (!dr || !dr->is_vp) return;
    if (dr->is_clear) {
        if (dr->clear_flags & RSX_CLEAR_COLOR_MASK) {
            const u64 serial = d3d12_next_surface_write_serial();
            d3d12_mark_color_surface(dr->rt_off, serial);
            for (int m = 0; m < 3; m++)
                d3d12_mark_color_surface(dr->rt_mrt[m], serial);
        }
        if (dr->clear_flags & (RSX_CLEAR_DEPTH_MASK | RSX_CLEAR_STENCIL_MASK)) {
            const int slot = off_depth_find(dr->zeta_ea);
            if (slot >= 0)
                s_d3d.off_depth[slot].depth_serial = d3d12_next_surface_write_serial();
        }
        return;
    }

    if (!dr->rt_none && dr->cmask) {
        const u64 serial = d3d12_next_surface_write_serial();
        d3d12_mark_color_surface(dr->rt_off, serial);
        for (int m = 0; m < 3; m++)
            d3d12_mark_color_surface(dr->rt_mrt[m], serial);
    }
    if (dr->depth_write) {
        const int slot = off_depth_find(dr->zeta_ea);
        if (slot >= 0)
            s_d3d.off_depth[slot].depth_serial = d3d12_next_surface_write_serial();
    }
}

static void d3d12_select_texture_surface(D3D12DrawRecord* dr, u32 unit)
{
    dr->tex_rt[unit] = -1;
    dr->tex_depth[unit] = -1;
    if (!dr->tex[unit].set) return;

    const int rt = off_rt_find(dr->tex[unit].raw);
    const int depth = off_depth_find(dr->tex[unit].off);
    if (rt >= 0 && depth >= 0) {
        if (s_d3d.off_depth[depth].depth_serial >
            s_d3d.off_rt[rt].color_serial)
            dr->tex_depth[unit] = depth;
        else
            dr->tex_rt[unit] = rt;
    } else if (depth >= 0) {
        dr->tex_depth[unit] = depth;
    } else if (rt >= 0) {
        dr->tex_rt[unit] = rt;
    }
}

static void d3d12_select_vertex_texture_surface(D3D12DrawRecord* dr, u32 unit)
{
    dr->vtex_rt[unit] = -1;
    dr->vtex_depth[unit] = -1;
    if (!dr->vtex[unit].set || dr->vtex[unit].cube) return;

    const int rt = off_rt_find(dr->vtex[unit].raw);
    const int depth = off_depth_find(dr->vtex[unit].off);
    if (rt >= 0 && depth >= 0) {
        if (s_d3d.off_depth[depth].depth_serial >
            s_d3d.off_rt[rt].color_serial)
            dr->vtex_depth[unit] = depth;
        else
            dr->vtex_rt[unit] = rt;
    } else if (depth >= 0) {
        dr->vtex_depth[unit] = depth;
    } else if (rt >= 0) {
        dr->vtex_rt[unit] = rt;
    }
}

/* D3D12 can sample the currently bound depth surface only with a read-only
 * DSV and DEPTH_READ|PIXEL_SHADER_RESOURCE state. Never enable this if the
 * guest draw writes depth or stencil. */
static int dr_reads_active_zeta(const D3D12DrawRecord* dr)
{
    if (!dr || dr->is_clear || dr->depth_write || dr->stencil.enable ||
        dr->zeta_ea == 0 || dr->zeta_ea == 0xFFFFFFFFu)
        return 0;
    const int zeta = off_depth_find(dr->zeta_ea);
    if (zeta < 0) return 0;
    for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++)
        if (dr->tex[u].set && dr->tex_depth[u] == zeta &&
            dr->tex[u].off == dr->zeta_ea)
            return 1;
    return 0;
}

/* RPCS3's depth-read emulation is selected only for RSX colour formats that
 * reinterpret Z24S8 bytes as A8R8G8B8/D8R8G8B8. */
static u32 dr_depth_rgba_mask(const D3D12DrawRecord* dr)
{
    if (!dr) return 0;
    const int active_zeta_readable = dr_reads_active_zeta(dr);
    u32 mask = 0;
    for (u32 u = 0; u < RSX_TEXTURE_UNITS; u++) {
        const int depth = dr->tex_depth[u];
        if (!dr->tex[u].set || depth < 0 || depth >= MAX_OFF_DEPTHS ||
            !s_d3d.off_depth[depth].res ||
            s_d3d.off_depth[depth].zeta_format != 2u)
            continue;
        const u32 fmt = dr->tex[u].fmt & 0x9Fu;
        if (fmt != 0x85u && fmt != 0x9Eu) continue;
        if (dr->tex[u].off == dr->zeta_ea && !active_zeta_readable)
            continue;
        mask |= 1u << u;
    }
    return mask;
}

static D3D12_CPU_DESCRIPTOR_HANDLE off_depth_dsv(int slot)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    s_d3d.dsv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.dsv_heap, &h);
    h.ptr += (u64)(slot + 1) * s_d3d.dsv_inc; /* slot zero is the fallback DSV */
    return h;
}

static D3D12_CPU_DESCRIPTOR_HANDLE off_depth_dsv_readonly(int slot)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    s_d3d.dsv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.dsv_heap, &h);
    h.ptr += (u64)(slot + 1 + MAX_OFF_DEPTHS) * s_d3d.dsv_inc;
    return h;
}

static void off_depth_transition(int slot, D3D12_RESOURCE_STATES to)
{
    OffDepth* d = &s_d3d.off_depth[slot];
    if (!d->res || d->st == to) return;
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = d->res;
    b.Transition.StateBefore = d->st;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    d->st = to;
}

static void off_depth_ensure_initialized(int slot)
{
    OffDepth* d = &s_d3d.off_depth[slot];
    if (!d->res || d->initialized) return;
    off_depth_transition(slot, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    D3D12_CLEAR_FLAGS flags = D3D12_CLEAR_FLAG_DEPTH;
    if (d->zeta_format == 2u)
        flags = (D3D12_CLEAR_FLAGS)(flags | D3D12_CLEAR_FLAG_STENCIL);
    s_d3d.cmd_list->lpVtbl->ClearDepthStencilView(
        s_d3d.cmd_list, off_depth_dsv(slot), flags, 1.0f, 0, 0, NULL);
    d->initialized = 1;
}

static int off_depth_get(u32 ea, u32 w, u32 h, u32 pitch, u32 zeta_format)
{
    if (!ea || ea == 0xFFFFFFFFu || !rsx_zeta_format_supported(zeta_format) ||
        !s_d3d.dsv_heap || !s_d3d.srv_heap)
        return -1;
    if (!w) w = s_d3d.width;
    if (!h) h = s_d3d.height;
    const u32 bpp = zeta_format == 1u ? 2u : 4u;
    if (pitch >= w * bpp && pitch / bpp > w)
        w = pitch / bpp;
    if (!w || !h || w > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        h > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION) return -1;

    const DXGI_FORMAT resource_fmt = zeta_format == 1u
        ? DXGI_FORMAT_R16_TYPELESS : DXGI_FORMAT_R24G8_TYPELESS;
    const DXGI_FORMAT dsv_fmt = rsx_zeta_dsv_format(zeta_format);
    const DXGI_FORMAT srv_fmt = zeta_format == 1u
        ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    int slot = off_depth_find(ea);
    if (slot >= 0) {
        OffDepth* d = &s_d3d.off_depth[slot];
        if (d->w == w && d->h == h && d->pitch == pitch &&
            d->zeta_format == zeta_format) {
            d->used = 1;
            return slot;
        }
        if (!d3d12_retire_resource(&d->res)) return -1;
    } else {
        for (int i = 0; i < MAX_OFF_DEPTHS; i++)
            if (!s_d3d.off_depth[i].res) { slot = i; break; }
        if (slot < 0) {
            for (int i = 0; i < MAX_OFF_DEPTHS; i++)
                if (!s_d3d.off_depth[i].used) { slot = i; break; }
            if (slot < 0) return -1;
            if (!d3d12_retire_resource(&s_d3d.off_depth[slot].res)) return -1;
        }
    }

    OffDepth* d = &s_d3d.off_depth[slot];
    D3D12_HEAP_PROPERTIES hp = {0};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {0};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h;
    td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = resource_fmt; td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    /* Shader access is permitted by default; D3D12 has no ALLOW_SHADER_RESOURCE
     * flag, only an opt-out DENY_SHADER_RESOURCE bit. */
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE cv = {0};
    cv.Format = dsv_fmt;
    cv.DepthStencil.Depth = 1.0f;
    cv.DepthStencil.Stencil = 0;
    if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &td,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv,
            &IID_ID3D12Resource, (void**)&d->res)))
        return -1;
    d->ea = ea; d->w = w; d->h = h; d->pitch = pitch;
    d->zeta_format = zeta_format;
    d->depth_serial = 0;
    d->resource_format = resource_fmt;
    d->dsv_format = dsv_fmt;
    d->srv_format = srv_fmt;
    d->st = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    d->used = 1;
    d->initialized = 0;

    D3D12_DEPTH_STENCIL_VIEW_DESC dv = {0};
    dv.Format = dsv_fmt;
    dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    s_d3d.device->lpVtbl->CreateDepthStencilView(
        s_d3d.device, d->res, &dv, off_depth_dsv(slot));
    dv.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    s_d3d.device->lpVtbl->CreateDepthStencilView(
        s_d3d.device, d->res, &dv, off_depth_dsv_readonly(slot));
    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {0};
    sv.Format = srv_fmt;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE sh;
    s_d3d.srv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &sh);
    sh.ptr += (u64)(RT_DEPTH_SRV_BASE + slot) * s_d3d.srv_inc;
    s_d3d.device->lpVtbl->CreateShaderResourceView(s_d3d.device, d->res, &sv, sh);
    return slot;
}

static void off_rt_transition(int slot, D3D12_RESOURCE_STATES to)
{
    OffRT* r = &s_d3d.off_rt[slot];
    if (r->st == to) return;
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = r->res;
    b.Transition.StateBefore = r->st;
    b.Transition.StateAfter  = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    r->st = to;
}

/* Set by the present entry point: 1 = this batch ends with a swapchain
 * Present, 0 = execute the recorded draws only. An OFFSCREEN-only batch (a
 * render-to-texture pass) must still run -- it is what fills the texture a
 * later batch samples -- but presenting it would show a half-built frame. */
static ID3D12Resource* off_rt_feedback_resource(int slot)
{
    if (slot < 0 || slot >= MAX_OFF_RTS) return NULL;
    OffRT* r = &s_d3d.off_rt[slot];
    if (!r->res) return NULL;
    if (!r->feedback) {
        D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC td = {0};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = r->w; td.Height = r->h;
        td.DepthOrArraySize = td.MipLevels = 1;
        td.Format = (DXGI_FORMAT)r->dxgi; td.SampleDesc.Count = 1;
        if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
                s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &td,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, NULL,
                &IID_ID3D12Resource, (void**)&r->feedback))) return NULL;
    }
    return r->feedback;
}

/* Copy in draw order, after the producer and before the consumer. A host
 * backbuffer snapshot contains the previous composite, not this surface. */
static void off_rt_feedback_capture(int slot)
{
    OffRT* r = &s_d3d.off_rt[slot];
    if (!r->feedback) return;
    D3D12_RESOURCE_STATES prior = r->st;
    off_rt_transition(slot, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r->feedback;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    s_d3d.cmd_list->lpVtbl->CopyResource(s_d3d.cmd_list, r->feedback, r->res);
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &b);
    off_rt_transition(slot, prior);
}

static int s_present_this_frame = 1;
static u32 s_present_display_id = 0;


/* Resolve a DRAW_*_TEX value: a hex raw offset, or the literal "duck" for the
 * content-identified duck texture. */
static u32 draw_filter_tex(const char* e)
{
    if (!e) return 0;
    if (e[0] == 'd' && e[1] == 'u') return s_duck_raw;
    return (u32)strtoul(e, NULL, 16);
}

/* Capture the surface that just received the reduced-viewport pass. Registered
 * display buffers are separate OffRT resources; copying the swapchain here
 * would feed the previous frame to screen-space effects and leave visible
 * trails. A direct-to-swapchain frame still uses the backbuffer. */
static void screen_copy_capture(u32 fi, int source_rt)
{
    static int en = -1;
    /* Replacing every display-sized guest texture with a frame snapshot is a
     * diagnostic workaround, not RSX texture semantics. Keep it opt-in until
     * the exact producer/consumer pair is identified. */
    if (en < 0) { const char* e = getenv("SCREEN_AS_TEX"); en = e ? atoi(e) : 0; }
    if (!en || !s_d3d.device) return;
            if (!s_screen_copy) {
                D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC td = {0};
                td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                td.Width = s_d3d.width; td.Height = s_d3d.height;
                td.DepthOrArraySize = 1; td.MipLevels = 1;
                td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
                td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
                if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
                        s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &td,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, NULL,
                        &IID_ID3D12Resource, (void**)&s_screen_copy)))
                    s_screen_copy = NULL;
            }
            if (s_screen_copy) {
                ID3D12Resource* source = s_d3d.render_targets[fi];
                D3D12_RESOURCE_STATES source_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
                if (source_rt >= 0) {
                    if (source_rt >= MAX_OFF_RTS) return;
                    OffRT* r = &s_d3d.off_rt[source_rt];
                    /* CopyResource requires matching dimensions and format. */
                    if (!r->res || r->w != s_d3d.width || r->h != s_d3d.height ||
                        r->dxgi != DXGI_FORMAT_R8G8B8A8_UNORM)
                        return;
                    source = r->res;
                    source_state = r->st;
                    off_rt_transition(source_rt, D3D12_RESOURCE_STATE_COPY_SOURCE);
                } else {
                    D3D12_RESOURCE_BARRIER to_source = {0};
                    to_source.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    to_source.Transition.pResource = source;
                    to_source.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    to_source.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                    to_source.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &to_source);
                }

                D3D12_RESOURCE_BARRIER to_dest = {0};
                to_dest.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                to_dest.Transition.pResource = s_screen_copy;
                to_dest.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                to_dest.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                to_dest.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &to_dest);
                s_d3d.cmd_list->lpVtbl->CopyResource(s_d3d.cmd_list, s_screen_copy, source);
                to_dest.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                to_dest.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &to_dest);

                if (source_rt >= 0) {
                    off_rt_transition(source_rt, source_state);
                    { static int n = 0;
                      if (getenv("SCREEN_COPY_DBG") && PS3_LOG_FIRST(n, 12))
                          fprintf(stderr, "[SCREEN_COPY] source RT=%d off=0x%08X %ux%u%c",
                                  source_rt, s_d3d.off_rt[source_rt].off,
                                  s_d3d.off_rt[source_rt].w,
                                  s_d3d.off_rt[source_rt].h, 10); }
                } else {
                    D3D12_RESOURCE_BARRIER restore = {0};
                    restore.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    restore.Transition.pResource = source;
                    restore.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    restore.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    restore.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
                    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &restore);
                }

                /* SCREENCOPY_DUMP=1: stage the captured image into the readback
                 * buffer so it can be written out and LOOKED AT. Claiming a
                 * capture happened is not the same as showing what is in it. */
                { static int sd = -1;
                  if (sd < 0) { const char* e = getenv("SCREENCOPY_DUMP"); sd = e ? atoi(e) : 0; }
                  if (sd && s_d3d.readback_buf) {
                      D3D12_RESOURCE_BARRIER t = {0};
                      t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                      t.Transition.pResource   = s_screen_copy;
                      t.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                      t.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
                      t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                      s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &t);

                      D3D12_TEXTURE_COPY_LOCATION cd = {0}, cs = {0};
                      cd.pResource = s_d3d.readback_buf;
                      cd.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                      cd.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
                      cd.PlacedFootprint.Footprint.Width    = s_d3d.width;
                      cd.PlacedFootprint.Footprint.Height   = s_d3d.height;
                      cd.PlacedFootprint.Footprint.Depth    = 1;
                      cd.PlacedFootprint.Footprint.RowPitch = s_d3d.readback_pitch;
                      cs.pResource = s_screen_copy;
                      cs.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                      s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &cd, 0, 0, 0, &cs, NULL);

                      t.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                      t.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                      s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &t);
                      s_sc_dump_pending = 1;
                  } }
            }

}

/* rsx_topology -> D3D_PRIMITIVE_TOPOLOGY. Spelled out rather than relying on
 * the two enums happening to agree: the draw record carries the neutral value
 * so a second backend can read it, and this is the one place it becomes a D3D
 * one. */
static D3D12_PRIMITIVE_TOPOLOGY topo_to_d3d(u32 t)
{
    switch (t) {
    case RSX_TOPOLOGY_POINTS:         return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case RSX_TOPOLOGY_LINES:          return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case RSX_TOPOLOGY_LINE_STRIP:     return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case RSX_TOPOLOGY_TRIANGLE_STRIP: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    case RSX_TOPOLOGY_TRIANGLES:      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    default:                          return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    }
}

/* GCM_GUEST_FB=<off>,<w>,<h>,<pitch> -- present the guest display buffer.
 *
 * A title that composites its final image with the RSX 2D engine assembles it
 * in GUEST memory: nv3089_blit is a CPU copy and never touches D3D12, and the
 * draw pass below only runs when a recorded draw targets the display surface,
 * which for such a title never happens. The frame is finished, in the wrong
 * memory, and the swapchain shows only the clear. Copy it across.
 *
 * Off unless the offset and geometry are given, so nothing else changes. */
/* GPU-side composite of an offscreen RT onto the backbuffer.
 *
 * A title that finishes its frame with the RSX 2D engine blits a rendered
 * surface into the display buffer. That surface lives in guest RAM, which the
 * D3D12 path never writes -- the scene is on the GPU -- so the CPU copy in
 * nv3089_blit faithfully moves zeros. The backend already holds the same
 * surface as a texture, so do the composite here instead. Triggered from
 * cellGcmSys under GCM_COMPOSITE_RT. */
static u32 s_composite_src = 0;
static int s_last_drawn_slot = -1;
static u32 s_last_drawn_off  = 0;

void rsx_d3d12_composite_from_offscreen(u32 raw_off) { s_composite_src = raw_off; }

static void composite_present(u32 fi)
{
    u32 want = s_composite_src;
    s_composite_src = 0;
    if (!want) return;
    /* want == 2 means "whatever this frame's draws targeted" -- the one thing
     * guaranteed to hold geometry, and immune to the offset collision above. */
    int i;
    if (want == 2u) { i = s_last_drawn_slot; want = s_last_drawn_off; }
    else            { i = off_rt_find(want); }
    if (i < 0 || !s_d3d.off_rt[i].res) return;
    OffRT* r = &s_d3d.off_rt[i];
    /* Same-format copy only; a half-float source would need a converting draw. */
    if (r->dxgi != DXGI_FORMAT_R8G8B8A8_UNORM) {
        static int w1 = 0;
        if (PS3_LOG_FIRST(w1, 2))
            fprintf(stderr, "[COMPOSITE] RT 0x%08X is dxgi=%u, not R8G8B8A8 -- needs a draw%c",
                    want, r->dxgi, 10);
        return;
    }
    u32 cw = r->w < s_d3d.width  ? r->w : s_d3d.width;
    u32 ch = r->h < s_d3d.height ? r->h : s_d3d.height;
    if (!cw || !ch) return;
    { static int n = 0;
      if (PS3_LOG_FIRST(n, 3)) fprintf(stderr, "[COMPOSITE] RT 0x%08X %ux%u -> backbuffer%c",
                           want, cw, ch, 10); }

    D3D12_RESOURCE_BARRIER b[2] = {0};
    b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b[0].Transition.pResource   = r->res;
    b[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b[0].Transition.StateBefore = r->st;
    b[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b[1] = b[0];
    b[1].Transition.pResource   = s_d3d.render_targets[fi];
    b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 2, b);

    D3D12_TEXTURE_COPY_LOCATION dl = {0}, sl = {0};
    dl.pResource = s_d3d.render_targets[fi];
    dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dl.SubresourceIndex = 0;
    sl.pResource = r->res;
    sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; sl.SubresourceIndex = 0;
    D3D12_BOX box = {0};
    box.left = 0; box.top = 0; box.front = 0;
    box.right = cw; box.bottom = ch; box.back = 1;
    s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dl, 0, 0, 0, &sl, &box);

    b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b[0].Transition.StateAfter  = r->st;
    b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 2, b);
}

/* Copy the display surface selected by the guest's flip to the DXGI
 * swapchain. Display buffers are rendered into persistent OffRT resources so
 * guest buffer 0 and buffer 1 never alias the same host image. */
static int scanout_display_buffer(u32 fi, u32 display_id)
{
    if (display_id >= CELL_GCM_MAX_DISPLAY_BUFFER_NUM) return 0;
    const u32 key = display_rt_key(display_id);
    const int slot = off_rt_find(key);
    if (slot < 0 || !s_d3d.off_rt[slot].res) {
        static int missing = 0;
        if (missing++ < 8)
            fprintf(stderr, "[D3D12] flip selected display buffer %u but its RT was not allocated%c",
                    display_id, 10);
        return 0;
    }

    OffRT* r = &s_d3d.off_rt[slot];
    if (r->dxgi != DXGI_FORMAT_R8G8B8A8_UNORM) {
        static int warned = 0;
        if (warned++ < 4)
            fprintf(stderr, "[D3D12] scanout buffer %u has unsupported format %u (RT key=0x%08X)%c",
                    display_id, (unsigned)r->dxgi, key, 10);
        return 0;
    }

    const u32 cw = r->w < s_d3d.width ? r->w : s_d3d.width;
    const u32 ch = r->h < s_d3d.height ? r->h : s_d3d.height;
    if (!cw || !ch) return 0;

    const D3D12_RESOURCE_STATES prior = r->st;
    off_rt_transition(slot, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_RESOURCE_BARRIER to_copy = {0};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = s_d3d.render_targets[fi];
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &to_copy);

    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = s_d3d.render_targets[fi];
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = r->res;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX box = {0, 0, 0, cw, ch, 1};
    s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dst, 0, 0, 0, &src, &box);

    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &to_copy);
    off_rt_transition(slot, prior);

    { static int logged = 0;
      if (logged < 12) {
          CellGcmDisplayInfo* info = cellGcmGetDisplayBufferByFlipIndex(display_id);
          fprintf(stderr, "[D3D12] scanout buffer=%u offset=0x%08X target=%ux%u -> backbuffer%c",
                  display_id, info ? info->offset : 0, r->w, r->h, 10);
          logged++;
      } }
    return 1;
}


static void guest_fb_present(u32 fi)
{
    static int inited = 0;
    static u32 off, gw, gh, gpitch, up_pitch;
    static ID3D12Resource* up = NULL;
    if (!inited) {
        inited = 1;
        const char* e = getenv("GCM_GUEST_FB");
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (!e || sscanf(e, "%i,%u,%u,%u", &a, &b, &c, &d) != 4 || !b || !c) return;
        off = a; gw = b; gh = c; gpitch = d;
        up_pitch = (gw * 4u + 255u) & ~255u;      /* D3D12 wants 256-byte rows */
        D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd = {0};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width  = (u64)up_pitch * gh; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
                s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &bd,
                D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                &IID_ID3D12Resource, (void**)&up)))
            up = NULL;
        fprintf(stderr, "[GUESTFB] presenting guest 0x%08X %ux%u pitch %u -> backbuffer%c",
                off, gw, gh, gpitch, 10);
    }
    if (!up) return;
    { extern u8* vm_base;
      if (!vm_base) return;
      /* RSX local memory is mapped at guest 0xC0000000. */
      const u8* srow = vm_base + 0xC0000000u + off;
      void* m = NULL; D3D12_RANGE nr = {0, 0};
      if (FAILED(up->lpVtbl->Map(up, 0, &nr, &m)) || !m) return;
      for (u32 y = 0; y < gh; y++) {
          const u8* s = srow + (size_t)y * gpitch;
          u8* dp = (u8*)m + (size_t)y * up_pitch;
          for (u32 x = 0; x < gw; x++) {   /* guest B,G,R,A -> backbuffer R,G,B,A */
              dp[x*4+0] = s[x*4+2]; dp[x*4+1] = s[x*4+1];
              dp[x*4+2] = s[x*4+0]; dp[x*4+3] = 0xFF;
          }
      }
      D3D12_RANGE wr = {0, (SIZE_T)up_pitch * gh};
      up->lpVtbl->Unmap(up, 0, &wr); }

    D3D12_RESOURCE_BARRIER gb = {0};
    gb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    gb.Transition.pResource   = s_d3d.render_targets[fi];
    gb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    gb.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    gb.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &gb);

    D3D12_TEXTURE_COPY_LOCATION dl = {0}, sl = {0};
    dl.pResource = s_d3d.render_targets[fi];
    dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dl.SubresourceIndex = 0;
    sl.pResource = up; sl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sl.PlacedFootprint.Offset = 0;
    sl.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    sl.PlacedFootprint.Footprint.Width    = gw;
    sl.PlacedFootprint.Footprint.Height   = gh;
    sl.PlacedFootprint.Footprint.Depth    = 1;
    sl.PlacedFootprint.Footprint.RowPitch = up_pitch;
    s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dl, 0, 0, 0, &sl, NULL);

    gb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    gb.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &gb);
}

/* SCANOUT_FB=0 disables. Automatic CPU-blit scanout fallback: when the frame
 * recorded no RSX work at all (s_dbg_last_draws == 0) but the flip names a
 * registered VRAM display buffer, the title renders on the CPU and writes its
 * display buffer straight into guest VRAM (TanakaDOOM-cGcm: software 320x200
 * renderer, zero RSX commands, ever). The normal path presents a never-rendered
 * surface = black. Copy the guest bytes to the backbuffer instead,
 * A,R,G,B -> R,G,B,A (the guest is big-endian; its 32-bit pixels land as
 * A,R,G,B bytes -- verified against RPCS3 on TanakaDOOM: byte-as-is reads
 * alpha as red, R/B swap reads blue as red). Skipped when
 * GCM_GUEST_FB is set (that debug path pins its own fixed offset). */
static void scanout_auto_present(u32 fi)
{
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("SCANOUT_FB");
        on = !(e && e[0] == '0') && !getenv("GCM_GUEST_FB");
    }
    if (!on || s_dbg_last_draws != 0) return;
    /* The display-buffer registry that matters is cellGcmSys's own: the
     * ticker-driven backend path never brings the draw engine up, so the
     * engine's table stays empty. cellGcm tracks the buffer the current flip
     * names (updated on every SetFlip path). */
    extern int cellGcm_current_scanout(u32*, u32*, u32*, u32*);
    u32 off = 0, gpitch = 0, gw = 0, gh = 0;
    const int got = cellGcm_current_scanout(&off, &gpitch, &gw, &gh);
    { static int dbg = -1;
      if (dbg < 0) dbg = getenv("SCANOUT_FB_DBG") ? 1 : 0;
      if (dbg) { static int n = 0;
                 if (n < 12) { n++;
                     fprintf(stderr, "[scanout-fb] last_draws=%u got=%d "
                                     "off=0x%X pitch=%u %ux%u\n",
                             s_dbg_last_draws, got, off, gpitch, gw, gh); } } }
    if (!got || !gpitch || !gw || !gh)
        return;
    extern u8* vm_base;
    if (!vm_base) return;
    const u32 up_pitch = (gw * 4u + 255u) & ~255u;   /* 256-byte rows */
    static ID3D12Resource* up = NULL;
    static u32 up_w = 0, up_h = 0;
    if (!up || up_w != gw || up_h != gh) {
        if (up) { up->lpVtbl->Release(up); up = NULL; }
        D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd = {0};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width  = (u64)up_pitch * gh; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(s_d3d.device->lpVtbl->CreateCommittedResource(
                s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &bd,
                D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                &IID_ID3D12Resource, (void**)&up)))
            return;
        up_w = gw; up_h = gh;
    }
    /* RSX local memory is mapped at guest 0xC0000000. The guest is a
     * big-endian PPC title: its 32-bit pixels land as A,R,G,B bytes, so
     * route R,G,B to the backbuffer and force alpha (verified against
     * RPCS3: byte-as-is copy reads alpha as red; a R/B swap reads blue
     * as red -- TanakaDOOM title screen). */
    { const u8* srow = vm_base + 0xC0000000u + off;
      void* m = NULL; D3D12_RANGE nr = {0, 0};
      if (FAILED(up->lpVtbl->Map(up, 0, &nr, &m)) || !m) return;
      for (u32 y = 0; y < gh; y++) {
          const u8* s = srow + (size_t)y * gpitch;
          u8* dp = (u8*)m + (size_t)y * up_pitch;
          for (u32 x = 0; x < gw; x++) {
              dp[x*4+0] = s[x*4+1];
              dp[x*4+1] = s[x*4+2];
              dp[x*4+2] = s[x*4+3];
              dp[x*4+3] = 0xFF;
          }
      }
      D3D12_RANGE wr = {0, (SIZE_T)up_pitch * gh};
      up->lpVtbl->Unmap(up, 0, &wr); }

    D3D12_RESOURCE_BARRIER gb = {0};
    gb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    gb.Transition.pResource   = s_d3d.render_targets[fi];
    gb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    gb.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    gb.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &gb);

    D3D12_TEXTURE_COPY_LOCATION dl = {0}, sl = {0};
    dl.pResource = s_d3d.render_targets[fi];
    dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dl.SubresourceIndex = 0;
    sl.pResource = up; sl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sl.PlacedFootprint.Offset = 0;
    sl.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    sl.PlacedFootprint.Footprint.Width    = gw;
    sl.PlacedFootprint.Footprint.Height   = gh;
    sl.PlacedFootprint.Footprint.Depth    = 1;
    sl.PlacedFootprint.Footprint.RowPitch = up_pitch;
    s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dl, 0, 0, 0, &sl, NULL);

    gb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    gb.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &gb);
}

static void render_frame_impl(void)
{
    u32 fi = s_d3d.frame_index;
    u32 slot = (u32)s_d3d.vp_parity;
    int frame_writes_selected_display = 0;
    D3D12_GPU_DESCRIPTOR_HANDLE default_sampler_table = {0};

    /* Reuse only this slot's command allocator, upload ranges, and descriptors
     * after its fence retires. The other slot can remain queued on the GPU. */
    if (!d3d12_acquire_submission_slot()) return;
    rsx_d3d12_srv_cache_reset(&s_srv_cache);
    if (s_d3d.vertex_buffer)
        s_d3d.vb_view.BufferLocation =
            s_d3d.vertex_buffer->lpVtbl->GetGPUVirtualAddress(s_d3d.vertex_buffer)
            + (u64)slot * MAX_VERTICES * VERTEX_STRIDE;
    s_d3d.sampler_table_count = 0;
    D3D12_SAMPLER_DESC default_sampler_desc[RSX_SAMPLER_TABLE_SIZE];
    d3d12_sampler_default_table(default_sampler_desc);
    const int have_default_sampler_table =
        d3d12_sampler_table_get(default_sampler_desc, &default_sampler_table);

    /* Compile the real vertex program once its microcode is captured, and keep
     * the constant bank uploaded for the VS. */
    /* The base VP pipeline gates the ENTIRE vertex-program draw pass below
     * (`if (s_d3d.vp_ready ...)`). The old trigger also required the microcode
     * size to differ from the last compile, so a title whose first captured
     * program has the same byte count as vp_compiled_bytes never built the base
     * pipeline -- and then every is_vp draw record was silently dropped: draws
     * recorded, constants uploaded, per-draw VS compiled and cached, and not one
     * of them submitted. While !vp_ready there is nothing to be stale against,
     * so only the "have microcode" test belongs here. */
    if (s_d3d.current_rsx_state && !s_d3d.vp_ready &&
        s_d3d.current_rsx_state->vp_ucode_bytes >= 16)
        compile_vp();
    /* Per-draw VP constants are snapshotted at record time (vp_record_cb). */

    /* Lazily create the readback buffer the first time a dump is requested. */
    if (s_d3d.dump_frames_left > 0 && !s_d3d.readback_buf) {
        s_d3d.readback_pitch = (s_d3d.width * 4 + 255) & ~255u;
        D3D12_HEAP_PROPERTIES hp = {0};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rd = {0};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = (u64)s_d3d.readback_pitch * s_d3d.height;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        s_d3d.device->lpVtbl->CreateCommittedResource(
            s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COPY_DEST, NULL,
            &IID_ID3D12Resource, (void**)&s_d3d.readback_buf);
    }

    /* Reset command allocator and list */
    s_d3d.cmd_allocators[slot]->lpVtbl->Reset(s_d3d.cmd_allocators[slot]);
    s_d3d.cmd_list->lpVtbl->Reset(s_d3d.cmd_list, s_d3d.cmd_allocators[slot], NULL);
    if (perf_on() && rsx_gpu_timer_init(&s_gpu_timer,s_d3d.device,s_d3d.cmd_queue))
        rsx_gpu_timer_begin(&s_gpu_timer,s_d3d.fence,s_d3d.cmd_list,slot);

    /* Upload the bound font atlas into this slot's R8_UNORM texture and create its SRV.
     * The atlas is a linear 8-bit coverage map, so a straight row copy (no
     * deswizzle) suffices. It is a DYNAMIC glyph cache -- the game rasterizes
     * new glyphs into it over time -- so re-upload whenever that slot is dirty.
     * Slot-local resources avoid overwriting the atlas still sampled by the GPU. */
    if (s_d3d.tex_bound && (!s_d3d.tex_ready[slot] || s_d3d.tex_dirty[slot]) &&
        s_d3d.srv_heap && s_d3d.pipeline_state_tex) {
        extern uint8_t* vm_base;
        u32 w = s_d3d.tex_w, h = s_d3d.tex_h;
        u32 pitch = (w + 255) & ~255u;   /* D3D12 requires 256-byte row pitch */

        const int fresh = !s_d3d.tex_resource[slot] ||
            s_d3d.tex_slot_w[slot] != w || s_d3d.tex_slot_h[slot] != h;
        if (fresh) s_d3d.tex_ready[slot] = 0;
        if (fresh && s_d3d.tex_resource[slot]) {
            s_d3d.tex_resource[slot]->lpVtbl->Release(s_d3d.tex_resource[slot]);
            s_d3d.tex_resource[slot] = NULL;
        }
        if (fresh && s_d3d.tex_upload[slot]) {
            s_d3d.tex_upload[slot]->lpVtbl->Release(s_d3d.tex_upload[slot]);
            s_d3d.tex_upload[slot] = NULL;
        }

        HRESULT thr = S_OK;
        if (!s_d3d.tex_resource[slot]) {
            D3D12_HEAP_PROPERTIES hp_def = {0}; hp_def.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC td = {0};
            td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width = w; td.Height = h; td.DepthOrArraySize = 1; td.MipLevels = 1;
            td.Format = DXGI_FORMAT_R8_UNORM; td.SampleDesc.Count = 1;
            td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            thr = s_d3d.device->lpVtbl->CreateCommittedResource(
                s_d3d.device, &hp_def, D3D12_HEAP_FLAG_NONE, &td,
                D3D12_RESOURCE_STATE_COPY_DEST, NULL,
                &IID_ID3D12Resource, (void**)&s_d3d.tex_resource[slot]);
        }

        D3D12_HEAP_PROPERTIES hp_up = {0}; hp_up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC ud = {0};
        ud.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        ud.Width = (u64)pitch * h; ud.Height = 1; ud.DepthOrArraySize = 1; ud.MipLevels = 1;
        ud.SampleDesc.Count = 1; ud.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (SUCCEEDED(thr) && !s_d3d.tex_upload[slot])
            thr = s_d3d.device->lpVtbl->CreateCommittedResource(
                s_d3d.device, &hp_up, D3D12_HEAP_FLAG_NONE, &ud,
                D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                &IID_ID3D12Resource, (void**)&s_d3d.tex_upload[slot]);

        if (SUCCEEDED(thr) && vm_base) {
            void* mapped = NULL;
            D3D12_RANGE nr = {0, 0};
            if (SUCCEEDED(s_d3d.tex_upload[slot]->lpVtbl->Map(
                    s_d3d.tex_upload[slot], 0, &nr, &mapped)) && mapped) {
                const u8* srcbase = vm_base + s_d3d.tex_src_offset;
                for (u32 y = 0; y < h; y++)
                    memcpy((u8*)mapped + (u64)y * pitch, srcbase + (u64)y * w, w);
                s_d3d.tex_upload[slot]->lpVtbl->Unmap(s_d3d.tex_upload[slot], 0, NULL);

                D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
                if (!fresh) {
                    D3D12_RESOURCE_BARRIER old_state = {0};
                    old_state.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    old_state.Transition.pResource = s_d3d.tex_resource[slot];
                    old_state.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    old_state.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                    old_state.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &old_state);
                }
                dst.pResource = s_d3d.tex_resource[slot];
                dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dst.SubresourceIndex = 0;
                src.pResource = s_d3d.tex_upload[slot];
                src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint.Offset = 0;
                src.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8_UNORM;
                src.PlacedFootprint.Footprint.Width    = w;
                src.PlacedFootprint.Footprint.Height   = h;
                src.PlacedFootprint.Footprint.Depth    = 1;
                src.PlacedFootprint.Footprint.RowPitch = pitch;
                s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dst, 0, 0, 0, &src, NULL);

                D3D12_RESOURCE_BARRIER tb = {0};
                tb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                tb.Transition.pResource   = s_d3d.tex_resource[slot];
                tb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                tb.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                tb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &tb);

                D3D12_SHADER_RESOURCE_VIEW_DESC sv = {0};
                sv.Format = DXGI_FORMAT_R8_UNORM;
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                sv.Texture2D.MipLevels = 1;
                D3D12_CPU_DESCRIPTOR_HANDLE sh;
                s_d3d.srv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &sh);
                sh.ptr += (u64)(ATLAS_SRV_BASE + slot) * s_d3d.srv_inc;
                s_d3d.device->lpVtbl->CreateShaderResourceView(
                    s_d3d.device, s_d3d.tex_resource[slot], &sv, sh);

                s_d3d.tex_ready[slot] = 1;
                s_d3d.tex_dirty[slot] = 0;   /* content now in sync with guest atlas */
                { static int _au = 0; if (PS3_LOG_FIRST(_au, 3))
                    printf("[D3D12] atlas uploaded (%ux%u R8) -> textured\n", w, h); }
            }
        }
    }

    /* Frames with no ops (init/boot presents) don't count against the cap --
     * PNG-decode-heavy titles burn hundreds of empty presents before the first
     * real draw. */
    if (d3d12_setting_rtt_dump() && s_d3d.draw_count > 0) { static int _f=0; int _cap = atoi(d3d12_setting_rtt_dump()); if (_cap < 2) _cap = 14; if (PS3_LOG_FIRST(_f, _cap)) {
        fprintf(stderr, "[RTT] frame %d: %u ops\n", _f, s_d3d.draw_count);
        for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
            D3D12DrawRecord* r = &s_d3d.draws[_d];
            fprintf(stderr, "[RTT]  op%02u %s rt=0x%X mrt=0x%X/0x%X/0x%X t0=0x%X fp=0x%X n=%u cmask=%X vp=%u,%u %ux%u blend=%d\n",
                _d, r->is_clear?"CLR ":(r->is_vp?"draw":"leg "), r->rt_off,
                r->rt_mrt[0], r->rt_mrt[1], r->rt_mrt[2],
                r->tex[0].raw, r->fp_addr, r->vertex_count,
                r->cmask, r->vp_x, r->vp_y, r->vp_w, r->vp_h, r->blend);
        }
    } }

    /* Debug: RTT_PASS=N shows pass N's output directly on screen (drops later
     * ops, retargets pass N to the backbuffer). */
    { const char* rp = d3d12_setting_rtt_pass();
      if (rp) {
        int keep = atoi(rp), seen = 0; u32 cut = s_d3d.draw_count;
        int viewrt = d3d12_setting_rtt_viewrt() != NULL;
        for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
            D3D12DrawRecord* r = &s_d3d.draws[_d];
            if (r->is_clear || !r->is_vp) continue;
            if (seen == keep) {
                /* Alone: retarget op N to the backbuffer. With RTT_VIEWRT:
                 * keep op N intact and pull the display composite forward so
                 * the chosen RT is shown as of this point in the chain. */
                if (!viewrt) { r->rt_off = 0; r->rt_mrt[0] = r->rt_mrt[1] = r->rt_mrt[2] = 0; }
                cut = _d + 1;
                break;
            }
            seen++;
        }
        if (viewrt) {
            for (u32 _j = cut; _j < s_d3d.draw_count && _j < MAX_DRAWS; _j++) {
                D3D12DrawRecord* r = &s_d3d.draws[_j];
                if (r->is_vp && !r->is_clear && is_display_target_key(r->rt_off)) {
                    if (cut < MAX_DRAWS) s_d3d.draws[cut++] = *r;
                    break;
                }
            }
        }
        s_d3d.draw_count = (cut < s_d3d.draw_count) ? cut : s_d3d.draw_count;
      } }

    /* DRAW_LIMIT=<N>: keep only the first N recorded ops this frame. The duck's
     * draws are ops 00..~03 of every frame, so a small limit shows the ducks
     * alone -- separating "the ducks never rasterize" from "later geometry is
     * painted over them". Diagnostic only. */
    { static int lim = -1;
      if (lim < 0) { const char* e = getenv("DRAW_LIMIT"); lim = e ? atoi(e) : 0; }
      if (lim > 0 && s_d3d.draw_count > (u32)lim) s_d3d.draw_count = (u32)lim; }

    /* DRAW_SKIP_TEX=<hex raw offset>: drop every op binding that texture on
     * unit 0. The ducks sit behind the water's slab geometry, so removing the
     * occluder is the way to see whether the duck itself renders. Diagnostic. */
    { static const char* sk = (const char*)1; static u32 want = 0;
      if (sk == (const char*)1) sk = getenv("DRAW_SKIP_TEX");
      want = draw_filter_tex(sk);
      if (want) {
        u32 keep = 0;
        for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
            if (!s_d3d.draws[_d].is_clear && s_d3d.draws[_d].tex[0].raw == want) continue;
            if (keep != _d) s_d3d.draws[keep] = s_d3d.draws[_d];
            keep++;
        }
        s_d3d.draw_count = keep;
      } }

    /* DRAW_KEEP_TEX=<hex raw offset>: the inverse -- keep only the ops binding
     * that texture (plus clears). Isolates one object from everything drawn over
     * it, which is what finally shows the duck on its own. Diagnostic. */
    { static const char* kp = (const char*)1; static u32 want = 0;
      if (kp == (const char*)1) kp = getenv("DRAW_KEEP_TEX");
      want = draw_filter_tex(kp);
      /* DRAW_KEEP_NOCLEAR=1: also drop the clears. Keeping them preserves their
       * ORIGINAL position in the batch, so a clear that originally ran after the
       * kept draws still runs after them -- and wipes the very geometry being
       * isolated. That makes an object that renders fine look like it renders
       * nothing. */
      static int noclr = -1;
      if (noclr < 0) { const char* e = getenv("DRAW_KEEP_NOCLEAR"); noclr = e ? atoi(e) : 0; }
      /* DRAW_KEEP_BLEND=0|1: further restrict DRAW_KEEP_TEX to draws with that
       * blend flag. A mesh drawn twice -- once opaque, once blended -- is two
       * different passes (body vs reflection shell) that a texture filter alone
       * cannot separate. */
      static int kblend = -2;
      if (kblend == -2) { const char* e = getenv("DRAW_KEEP_BLEND");
                          kblend = e ? atoi(e) : -1; }
      if (want) {
        /* Emit the CLEARS FIRST, then the kept draws. Preserving the original
         * order means a clear that ran after them still runs after them and
         * wipes the isolated geometry; dropping the clears entirely (noclr)
         * leaves stale DEPTH that rejects most of its fragments. Neither shows
         * the object properly -- clear, then draw, does. */
        static D3D12DrawRecord kept[MAX_DRAWS];
        u32 nk = 0;
        if (!noclr)
            for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++)
                if (s_d3d.draws[_d].is_clear && nk < MAX_DRAWS) kept[nk++] = s_d3d.draws[_d];
        for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++)
            if (!s_d3d.draws[_d].is_clear && s_d3d.draws[_d].tex[0].raw == want
                && (kblend < 0 || s_d3d.draws[_d].blend == kblend)
                && nk < MAX_DRAWS)
                kept[nk++] = s_d3d.draws[_d];
        for (u32 _k = 0; _k < nk; _k++) s_d3d.draws[_k] = kept[_k];
        s_d3d.draw_count = nk;
      } }

    /* DRAW_LAST_TEX=<hex raw offset>: move the ops binding that texture to the
     * END of the batch so nothing is drawn over them. The duck's draws are ops
     * 00..03 of every frame and the tub wall is drawn afterwards across the same
     * screen area, so anything that defeats the depth sort hides the duck
     * completely. Reordering separates "occluded" from "not rendered". */
    { static const char* lt = (const char*)1; static u32 want = 0;
      if (lt == (const char*)1) lt = getenv("DRAW_LAST_TEX");
      want = draw_filter_tex(lt);
      if (want && s_d3d.draw_count > 1) {
        static D3D12DrawRecord moved[MAX_DRAWS];
        u32 nm = 0, keep = 0;
        for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
            if (!s_d3d.draws[_d].is_clear && s_d3d.draws[_d].tex[0].raw == want) {
                if (nm < MAX_DRAWS) moved[nm++] = s_d3d.draws[_d];
                continue;
            }
            if (keep != _d) s_d3d.draws[keep] = s_d3d.draws[_d];
            keep++;
        }
        for (u32 _m = 0; _m < nm && keep < MAX_DRAWS; _m++)
            s_d3d.draws[keep++] = moved[_m];
        s_d3d.draw_count = keep;
      } }

    /* Display buffers and ordinary render-to-texture surfaces both use the
     * OffRT path. Remember whether this batch writes any scanout buffer so the
     * selected guest buffer can be copied to the swapchain after replay. */
    for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
        D3D12DrawRecord* dr = &s_d3d.draws[_d];
        const u32 selected_key = display_rt_key(s_present_display_id);
        if (dr->rt_off == selected_key) frame_writes_selected_display = 1;
        for (int _m = 0; _m < 3; _m++)
            if (dr->rt_mrt[_m] == selected_key) frame_writes_selected_display = 1;
    }

    /* Pre-create every guest surface targeted this frame so draws that sample
     * it can resolve to the same resource, whatever the op order. */
    double _pre0 = perf_on() ? perf_now() : 0.0;
    for (int _i = 0; _i < MAX_OFF_RTS; _i++) s_d3d.off_rt[_i].used = 0;
    for (int _i = 0; _i < MAX_OFF_DEPTHS; _i++) s_d3d.off_depth[_i].used = 0;
    for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
        D3D12DrawRecord* dr = &s_d3d.draws[_d];
        for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
            int slot = dr->tex[_u].set ? off_depth_find(dr->tex[_u].off) : -1;
            if (slot >= 0) s_d3d.off_depth[slot].used = 1;
        }
        for (int _u = 0; _u < RSX_MAX_VERTEX_TEXTURES; _u++) {
            int slot = dr->vtex[_u].set ? off_depth_find(dr->vtex[_u].off) : -1;
            if (slot >= 0) s_d3d.off_depth[slot].used = 1;
        }
    }

    /* Collect depth surfaces once. The old code searched all earlier and later
     * draws for every draw that named a zeta surface, making this pass O(n^2)
     * in scenes with many batches. Keep the first format and maximum dimensions
     * and pitch for each address, matching the old selection rules. */
    typedef struct {
        u32 ea, w, h, pitch, zeta_format;
        u32 created;
    } DepthFrameGroup;
    enum { DEPTH_FRAME_HASH_SIZE = MAX_DRAWS * 2 };
    static DepthFrameGroup depth_groups[MAX_DRAWS];
    static u32 depth_group_for_draw[MAX_DRAWS];
    static u32 depth_group_hash[DEPTH_FRAME_HASH_SIZE];
    memset(depth_group_hash, 0, sizeof(depth_group_hash));
    u32 depth_group_count = 0;
    for (u32 d = 0; d < s_d3d.draw_count && d < MAX_DRAWS; d++) {
        D3D12DrawRecord* dr = &s_d3d.draws[d];
        depth_group_for_draw[d] = 0xFFFFFFFFu;
        if (!dr->is_vp || dr->zeta_ea == 0xFFFFFFFFu || dr->zeta_ea == 0)
            continue;

        u32 hash = (dr->zeta_ea * 2654435761u) & (DEPTH_FRAME_HASH_SIZE - 1u);
        while (depth_group_hash[hash] != 0 &&
               depth_groups[depth_group_hash[hash] - 1u].ea != dr->zeta_ea)
            hash = (hash + 1u) & (DEPTH_FRAME_HASH_SIZE - 1u);

        u32 group_index;
        if (depth_group_hash[hash] == 0) {
            group_index = depth_group_count++;
            depth_group_hash[hash] = group_index + 1u;
            depth_groups[group_index].ea = dr->zeta_ea;
            depth_groups[group_index].w = dr->rt_w;
            depth_groups[group_index].h = dr->rt_h;
            depth_groups[group_index].pitch = dr->zeta_pitch;
            depth_groups[group_index].zeta_format = dr->zeta_format;
            depth_groups[group_index].created = 0;
        } else {
            group_index = depth_group_hash[hash] - 1u;
            DepthFrameGroup* group = &depth_groups[group_index];
            if (dr->rt_w > group->w) group->w = dr->rt_w;
            if (dr->rt_h > group->h) group->h = dr->rt_h;
            if (dr->zeta_pitch > group->pitch) group->pitch = dr->zeta_pitch;
        }
        depth_group_for_draw[d] = group_index;
    }

    for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
        D3D12DrawRecord* dr = &s_d3d.draws[_d];
        if (dr->is_vp && dr->rt_off)
            off_rt_get(dr->rt_off, dr->rt_w, dr->rt_h, dr->rt_fmt);
        const u32 depth_group = depth_group_for_draw[_d];
        if (depth_group != 0xFFFFFFFFu &&
            !depth_groups[depth_group].created &&
            rsx_zeta_format_supported(depth_groups[depth_group].zeta_format)) {
            DepthFrameGroup* group = &depth_groups[depth_group];
            group->created = 1;
            off_depth_get(group->ea, group->w, group->h, group->pitch,
                          group->zeta_format);
        }
        for (int _m = 0; _m < 3; _m++)
            if (dr->is_vp && dr->rt_mrt[_m])
                off_rt_get(dr->rt_mrt[_m], dr->rt_w, dr->rt_h, dr->rt_fmt);
    }

    /* Per-frame VP textures + guest-FP pipelines: for each VP draw, upload the
     * texture it had bound at submit time into a slot (SRV heap 1+slot; plasma
     * animates so contents re-upload every frame) and pre-build its FP PSO.
     * A texture whose offset matches an offscreen RT samples the RT directly
     * (tex_slot 1000+idx) -- no guest-memory upload. */
    for (int _i = 0; _i < VP_TEX_SLOTS; _i++) s_d3d.vp_tex[_i].used = 0;
    for (u32 _d = 0; _d < s_d3d.draw_count && _d < MAX_DRAWS; _d++) {
        D3D12DrawRecord* dr = &s_d3d.draws[_d];
        if (!dr->is_vp) continue;
        if (dr->is_clear) {
            d3d12_record_surface_writes(dr);
            continue;
        }
        /* Debug: RTT_VIEWRT=<hex raw offset> makes display draws sample that
         * offscreen RT at t0 (the composite blit then shows it fullscreen). */
        { static const char* vr = (const char*)1;     /* hoisted: this runs per DRAW */
          if (vr == (const char*)1) vr = d3d12_setting_rtt_viewrt();
          if (vr && is_display_target_key(dr->rt_off)) {
              dr->tex[0].raw = (u32)strtoul(vr, NULL, 16);
              dr->tex[0].off = 0;
              dr->tex[0].set = 1;
          } }
        u32 sampled_mask = 0xFFFFu;
        static int allow_texture_pruning = -1;
        if (allow_texture_pruning < 0)
            allow_texture_pruning = !getenv("FP_OFF") && !getenv("VP_BYPASS") &&
                !getenv("RTT_VIEWRT") && !getenv("SCREEN_AS_TEX");
        if (allow_texture_pruning && dr->fp_addr) {
            extern u8* vm_base;
            const u32 ea = cellGcmResolveLocated((dr->fp_addr & 3u)==1,dr->fp_addr & ~3u);
            const rsx_fp_metadata* metadata = ea != 0xFFFFFFFFu && vm_base
                ? rsx_fp_metadata_get(&s_fp_metadata,vm_base+ea) : NULL;
            if (metadata) sampled_mask = metadata->units | 1u;
        }
        /* Select aliased color/depth surfaces using the writes that precede
         * this draw in guest command order. */
        for (u32 _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
            dr->tex_feedback[_u] = -1;
            if (sampled_mask & (1u << _u)) d3d12_select_texture_surface(dr, _u);
            else { dr->tex_rt[_u] = -1; dr->tex_depth[_u] = -1; }
        }
        for (u32 _u = 0; _u < RSX_MAX_VERTEX_TEXTURES; _u++)
            d3d12_select_vertex_texture_surface(dr, _u);
        /* SCREENTEX=1: report draws that sample a DISPLAY-SIZED texture. On RSX
         * a title renders a reflection/refraction into a surface and then
         * samples it; our backend renders into a D3D backbuffer, so the guest
         * memory that sampler reads is never written and the texture comes back
         * empty. Finding which geometry does it is the first step to feeding it
         * the rendered image instead. */
        /* EMPTYTEX=1: which geometry binds a texture whose source is all zero.
         * An unresolved texture renders untextured, so this names the surfaces
         * affected instead of leaving it to guesswork. */
        { static int ed = -1;
          if (ed < 0) { const char* e = getenv("EMPTYTEX"); ed = e ? atoi(e) : 0; }
          extern uint8_t* vm_base;
          if (ed && vm_base) for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
              if (!dr->tex[_u].set || !dr->tex[_u].off) continue;
              u32 nz = 0, sz = dr->tex[_u].w * dr->tex[_u].h * 4u;
              for (u32 i2 = 0; i2 < sz && i2 < 0x20000u; i2 += 997)
                  if (vm_base[dr->tex[_u].off + i2]) { nz = 1; break; }
              if (!nz) {
                  static u32 seen[24]; static int ns = 0; int known = 0;
                  for (int k = 0; k < ns; k++) if (seen[k] == dr->fp_addr) known = 1;
                  if (!known && ns < 24) { seen[ns++] = dr->fp_addr;
                      /* Scan a window around the resolved address: if the pass
                       * that fills this surface ran but landed elsewhere, the
                       * data is nearby; if the whole window is zero it was never
                       * produced. */
                      /* Same offset through BOTH context DMAs: a texture the
                       * guest put in main memory but that we resolve as LOCAL
                       * reads as untouched VRAM, i.e. all zeros. */
                      { extern u32 cellGcmResolveLocated(int, u32);
                        u32 la = cellGcmResolveLocated(1, dr->tex[_u].raw);
                        u32 ma = cellGcmResolveLocated(0, dr->tex[_u].raw);
                        u32 ln = 0, mn = 0;
                        for (u32 i4 = 0; i4 < 0x20000u; i4 += 997) {
                            if (vm_base[la + i4]) ln++;
                            if (vm_base[ma + i4]) mn++;
                        }
                        fprintf(stderr, "[TEXLOC] raw=0x%08X LOCAL@0x%08X nz=%u  MAIN@0x%08X nz=%u%c",
                                dr->tex[_u].raw, la, ln, ma, mn, 10); }
                      { u32 base = dr->tex[_u].off & ~0xFFFFFu;
                        for (u32 w = 0; w < 0x400000u; w += 0x100000u) {
                            u32 nzc = 0;
                            for (u32 i3 = 0; i3 < 0x100000u; i3 += 1021)
                                if (vm_base[base + w + i3]) nzc++;
                            fprintf(stderr, "[EMPTYSCAN] 0x%08X..+1MB nonzero %u/1027%c",
                                    base + w, nzc, 10);
                        } }
                      fprintf(stderr, "[EMPTYTEX] fp=0x%X unit=%d raw=0x%08X res=0x%08X %ux%u"
                                      " fmt=0x%02X verts=%u vp=%u,%u %ux%u%c",
                              dr->fp_addr, _u, dr->tex[_u].raw, dr->tex[_u].off, dr->tex[_u].w,
                              dr->tex[_u].h, dr->tex[_u].fmt, dr->vertex_count,
                              dr->vp_x, dr->vp_y, dr->vp_w, dr->vp_h, 10); }
              }
          } }
        { static int sd = -1;
          if (sd < 0) { const char* e = getenv("SCREENTEX"); sd = e ? atoi(e) : 0; }
          if (sd) for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++)
              if (dr->tex[_u].set && dr->tex[_u].w == s_d3d.width &&
                  dr->tex[_u].h == s_d3d.height) {
                  static u32 seen[16]; static int ns = 0; int known = 0;
                  for (int k = 0; k < ns; k++) if (seen[k] == dr->fp_addr) known = 1;
                  if (!known && ns < 16) { seen[ns++] = dr->fp_addr;
                      fprintf(stderr, "[SCREENTEX] unit=%d raw=0x%08X %ux%u fmt=0x%02X"
                                      " fp=0x%X verts=%u vp=%u,%u %ux%u%c",
                              _u, dr->tex[_u].raw, dr->tex[_u].w, dr->tex[_u].h,
                              dr->tex[_u].fmt, dr->fp_addr, dr->vertex_count,
                              dr->vp_x, dr->vp_y, dr->vp_w, dr->vp_h, 10); }
              } }
        /* Fill this draw's t0-t15 SRV window: each fragment unit
         * resolves to an offscreen RT (sampled directly), an uploaded guest
         * texture, or a null SRV. */
        double _sv0 = perf_on() ? perf_now() : 0.0;
        for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
            u32 wslot = DRAW_SRV_SLOT(slot, _d, _u);
            if (!(sampled_mask & (1u << _u))) {
                srv_write(wslot,NULL,DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                continue;
            }

            /* Keep the surface selected above. Clearing tex_rt here sends
             * render-to-texture consumers through the guest-memory upload
             * path, whose bytes do not contain the GPU's preceding draws. */
            /* Display-sized sampler source -> the rendered frame. */
            if (dr->tex[_u].set && s_screen_copy &&
                dr->tex[_u].w == s_d3d.width && dr->tex[_u].h == s_d3d.height) {
                static int en = -1;
                if (en < 0) { const char* e = getenv("SCREEN_AS_TEX"); en = e ? atoi(e) : 0; }
                if (en) {
                    { static int _n = 0; if (PS3_LOG_FIRST(_n, 3))
                        fprintf(stderr, "[SCREENTEX] bound frame copy at unit %d for"
                                        " fp=0x%X (%ux%u)%c", _u, dr->fp_addr,
                                dr->tex[_u].w, dr->tex[_u].h, 10); }
                    srv_write(wslot, s_screen_copy, DXGI_FORMAT_R8G8B8A8_UNORM,
                              D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                    continue;
                }
            }
            /* A sampler whose guest buffer was never written, at the size of a
             * reduced-viewport pass we captured: that pass IS its producer. */
            if (dr->tex[_u].set && dr->tex[_u].off && s_subvp_n) {
                int hit = -1;
                for (int i = 0; i < s_subvp_n; i++)
                    if (s_subvp[i].res && s_subvp[i].w == dr->tex[_u].w &&
                        s_subvp[i].h == dr->tex[_u].h) { hit = i; break; }
                if (hit >= 0 && dr->tex_depth[_u] < 0 && dr->tex_rt[_u] < 0) {
                    extern uint8_t* vm_base;
                    u32 nz = 0, sz = dr->tex[_u].w * dr->tex[_u].h * 4u;
                    for (u32 i5 = 0; i5 < sz && i5 < 0x20000u; i5 += 997)
                        if (vm_base[dr->tex[_u].off + i5]) { nz = 1; break; }
                    if (!nz) {
                        { static int _n = 0; if (d3d12_setting_subvp_dbg() && PS3_LOG_FIRST(_n, 8))
                            fprintf(stderr, "[SUBVP] unit %d fp=0x%X <- captured %ux%u%c",
                                    _u, dr->fp_addr, dr->tex[_u].w, dr->tex[_u].h, 10); }
                        srv_write(wslot, s_subvp[hit].res, DXGI_FORMAT_R8G8B8A8_UNORM,
                                  D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                        continue;
                    }
                }
            }
            if (dr->tex[_u].set) {
                int depth = dr->tex_depth[_u];
                if (depth >= 0) {
                    if (depth == off_depth_find(dr->zeta_ea) &&
                        !dr_reads_active_zeta(dr))
                        srv_write(wslot, NULL, DXGI_FORMAT_R8G8B8A8_UNORM,
                                  D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                    else
                        srv_write(wslot, s_d3d.off_depth[depth].res,
                                  s_d3d.off_depth[depth].srv_format,
                                  D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                    continue;
                }
                int rt = dr->tex_rt[_u];
                if (rt >= 0) {
                    ID3D12Resource* source = s_d3d.off_rt[rt].res;
                    int feedback = dr->rt_off && rt == off_rt_find(dr->rt_off);
                    for (int m = 0; m < 3 && !feedback; m++)
                        feedback = dr->rt_mrt[m] && rt == off_rt_find(dr->rt_mrt[m]);
                    if (feedback) {
                        /* An SRV cannot read the color attachment being
                         * written by this draw. Allocate its snapshot now;
                         * record the actual copy immediately before the draw,
                         * after all preceding producers have executed. */
                        source = off_rt_feedback_resource(rt);
                        if (source) dr->tex_feedback[_u] = rt;
                    }
                    srv_write(wslot, source, (DXGI_FORMAT)s_d3d.off_rt[rt].dxgi,
                              D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                    continue;
                }
                u32 _bf = dr->tex[_u].fmt & 0x9F;
                /* Use the same format table as vertex textures. The old
                 * fragment-only list omitted D8R8G8B8 and native depth
                 * formats, even though vp_upload_tex_slot/texture_layout
                 * already decode them. P5's shadow/depth samplers then
                 * reached the null-SRV fallback. */
                if (dr->tex[_u].off && vp_texture_format_supported(_bf)) {
                    double _tt = perf_on() ? perf_now() : 0.0;
                    int _cube = dr->tex[_u].cube && (dr_cube_mask(dr) & (1u << _u));
                    int ts = vp_upload_tex_slot(dr->tex[_u].off, dr->tex[_u].w,
                                                dr->tex[_u].h, dr->tex[_u].fmt,
                                                dr->tex[_u].pitch, _cube,
                                                dr->tex[_u].mips);
                    if (perf_on()) { s_perf_tex += perf_now() - _tt; s_perf_ntex++;
                        s_perf_texbytes += (u64)dr->tex[_u].w * dr->tex[_u].h * 4u; }
                    /* Both forms of the offset are in hand here; the filters
                     * compare the BOUND one. */
                    if (s_duck_off && dr->tex[_u].off == s_duck_off)
                        s_duck_raw = dr->tex[_u].raw;
                    if (ts >= 0) {
                        DXGI_FORMAT sf = s_d3d.vp_tex[ts].dxgi;
                        srv_write_ex(wslot, s_d3d.vp_tex[ts].res, sf,
                                  d3d12_texture_component_mapping(dr->tex[_u].ctrl1, _bf),
                                  _cube);
                        continue;
                    }
                }
            }
            /* TEX_BINDDBG=<N>: a unit that ends up with a NULL SRV. The draw is
             * still recorded, lit and transformed -- it just samples nothing,
             * which renders as a solid black object. */
            { static int cap = -1, n = 0;
              if (cap < 0) { const char* e = getenv("TEX_BINDDBG"); cap = e ? atoi(e) : 0; }
              if (cap && n < cap && dr->tex[_u].set) { n++;
                fprintf(stderr, "[TEXBIND] null SRV unit=%d raw=0x%X off=0x%X %ux%u fmt=0x%02X%c",
                        _u, dr->tex[_u].raw, dr->tex[_u].off, dr->tex[_u].w,
                        dr->tex[_u].h, dr->tex[_u].fmt, 10); } }
            srv_write(wslot, NULL, DXGI_FORMAT_R8G8B8A8_UNORM,
                      D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
        }
        /* Fill the draw's t16-t19 window for NV40 vertex TXL. Vertex and
         * fragment textures share the upload cache, while their descriptors
         * remain in separate shader-register ranges. */
        for (int _u = 0; _u < RSX_MAX_VERTEX_TEXTURES; _u++) {
            const u32 wslot = DRAW_SRV_SLOT(slot, _d,
                                            RSX_TEXTURE_UNITS + (u32)_u);
            if (!dr->vtex[_u].set) {
                srv_write(wslot, NULL, DXGI_FORMAT_R8G8B8A8_UNORM,
                          D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                continue;
            }
            int depth = dr->vtex_depth[_u];
            if (depth >= 0) {
                if (depth == off_depth_find(dr->zeta_ea))
                    srv_write(wslot, NULL, DXGI_FORMAT_R8G8B8A8_UNORM,
                              D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                else
                    srv_write(wslot, s_d3d.off_depth[depth].res,
                              s_d3d.off_depth[depth].srv_format,
                              D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                continue;
            }
            int rt = dr->vtex_rt[_u];
            if (rt >= 0) {
                srv_write(wslot, s_d3d.off_rt[rt].res,
                          (DXGI_FORMAT)s_d3d.off_rt[rt].dxgi,
                          D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                continue;
            }
            const u32 bf = dr->vtex[_u].fmt & 0x9Fu;
            if (vp_texture_format_supported(bf)) {
                int ts = vp_upload_tex_slot(dr->vtex[_u].off,
                    dr->vtex[_u].w, dr->vtex[_u].h, dr->vtex[_u].fmt,
                    dr->vtex[_u].pitch, dr->vtex[_u].cube, dr->vtex[_u].mips);
                if (ts >= 0) {
                    const DXGI_FORMAT sf = s_d3d.vp_tex[ts].dxgi;
                    const u32 mapping = d3d12_texture_component_mapping(
                        dr->vtex[_u].ctrl1, bf);
                    srv_write_ex(wslot, s_d3d.vp_tex[ts].res, sf, mapping,
                                 dr->vtex[_u].cube);
                    continue;
                }
            }
            srv_write_ex(wslot, NULL, DXGI_FORMAT_R8G8B8A8_UNORM,
                         D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                         dr->vtex[_u].cube);
        }
        if (perf_on()) s_perf_srv += perf_now() - _sv0;
        d3d12_record_surface_writes(dr);
    }

    /* Transition render target to RENDER_TARGET state */
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s_d3d.render_targets[fi];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &barrier);

    /* Get RTV handle for current frame */
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle;
    s_d3d.rtv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.rtv_heap, &rtv_handle);
    rtv_handle.ptr += fi * s_d3d.rtv_descriptor_size;

    /* Get the fallback DSV handle (guest zeta targets use the depth pool). */
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle;
    s_d3d.dsv_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_d3d.dsv_heap, &dsv_handle);

    if (perf_on()) s_perf_pre += perf_now() - _pre0;
    /* Set render target + depth */
    s_d3d.cmd_list->lpVtbl->OMSetRenderTargets(s_d3d.cmd_list, 1, &rtv_handle, FALSE, &dsv_handle);

    /* Apply the guest's pending clear mask and values. Unrequested depth or
     * colour clears must preserve the previous contents. */
    if (s_d3d.clear_flags & RSX_CLEAR_COLOR_MASK)
        s_d3d.cmd_list->lpVtbl->ClearRenderTargetView(
            s_d3d.cmd_list, rtv_handle, s_d3d.clear_color, 0, NULL);
    D3D12_CLEAR_FLAGS ds_flags = (D3D12_CLEAR_FLAGS)0;
    if (s_d3d.clear_flags & RSX_CLEAR_DEPTH_MASK)
        ds_flags = (D3D12_CLEAR_FLAGS)(ds_flags | D3D12_CLEAR_FLAG_DEPTH);
    if (s_d3d.clear_flags & RSX_CLEAR_STENCIL_MASK)
        ds_flags = (D3D12_CLEAR_FLAGS)(ds_flags | D3D12_CLEAR_FLAG_STENCIL);
    if (ds_flags)
        s_d3d.cmd_list->lpVtbl->ClearDepthStencilView(
            s_d3d.cmd_list, dsv_handle, ds_flags,
            s_d3d.clear_depth, s_d3d.clear_stencil, 0, NULL);

    /* Set viewport and scissor */
    D3D12_VIEWPORT viewport = {0, 0, (float)s_d3d.width, (float)s_d3d.height, 0.0f, 1.0f};
    D3D12_RECT scissor = {0, 0, (LONG)s_d3d.width, (LONG)s_d3d.height};
    s_d3d.cmd_list->lpVtbl->RSSetViewports(s_d3d.cmd_list, 1, &viewport);
    s_d3d.cmd_list->lpVtbl->RSSetScissorRects(s_d3d.cmd_list, 1, &scissor);

    /* Bind pipeline state and push MVP if anything to draw */
    if (s_d3d.pipeline_ready && s_d3d.draw_count > 0) {
        s_d3d.cmd_list->lpVtbl->SetGraphicsRootSignature(s_d3d.cmd_list, s_d3d.root_signature);
        s_d3d.cmd_list->lpVtbl->IASetVertexBuffers(s_d3d.cmd_list, 0, 1, &s_d3d.vb_view);

        /* Make the atlas SRV heap current and bind its table (t0) for textured
         * draws. Safe to set even if no draw is textured. */
        if (s_d3d.tex_ready[slot] && s_d3d.srv_heap) {
            ID3D12DescriptorHeap* heaps[] = { s_d3d.srv_heap };
            s_d3d.cmd_list->lpVtbl->SetDescriptorHeaps(s_d3d.cmd_list, 1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE gh;
            s_d3d.srv_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &gh);
            gh.ptr += (u64)(ATLAS_SRV_BASE + slot) * s_d3d.srv_inc;
            s_d3d.cmd_list->lpVtbl->SetGraphicsRootDescriptorTable(s_d3d.cmd_list, 1, gh);
        }

        /* Push the MVP matrix from RSX vertex constants slots 0..3.
         * If the game hasn't written any constants (e.g. placeholder data
         * already in clip space), fall back to identity. */
        float mvp[16];
        const rsx_state* st = s_d3d.current_rsx_state;
        int have_mvp = 0;
        if (st) {
            for (u32 r = 0; r < 4; r++) {
                for (u32 c = 0; c < 4; c++) {
                    float v = st->vertex_constants[r][c];
                    mvp[r * 4 + c] = v;
                    if (v != 0.0f) have_mvp = 1;
                }
            }
        }
        if (!have_mvp) {
            memset(mvp, 0, sizeof(mvp));
            mvp[0] = mvp[5] = mvp[10] = mvp[15] = 1.0f; /* identity */
        }
        s_d3d.cmd_list->lpVtbl->SetGraphicsRoot32BitConstants(
            s_d3d.cmd_list, 0 /*root param 0*/, 16, mvp, 0);

        /* D3D12_IQ=1: dump exactly what the GPU is about to see -- the MVP root
         * constants and the first uploaded host vertices. The legacy path draws
         * with no validation errors yet produces zero pixels, so the geometry
         * must be degenerate/off-screen post-VS. */
        { static int _vd = -1;
          if (_vd < 0) { const char* e = getenv("D3D12_IQ"); _vd = e ? 1 : 0; }
          static int _n = 0;
          if (_vd && _n < 3) {
            _n++;
            fprintf(stderr, "[D3D12-DBG] have_mvp=%d draw_count=%u vb_offset=%u\n",
                    have_mvp, s_d3d.draw_count, s_d3d.vb_offset);
            fprintf(stderr, "[D3D12-DBG] mvp rows: [%.2f %.2f %.2f %.2f][%.2f %.2f %.2f %.2f]"
                            "[%.2f %.2f %.2f %.2f][%.2f %.2f %.2f %.2f]\n",
                    mvp[0],mvp[1],mvp[2],mvp[3], mvp[4],mvp[5],mvp[6],mvp[7],
                    mvp[8],mvp[9],mvp[10],mvp[11], mvp[12],mvp[13],mvp[14],mvp[15]);
            if (s_d3d.vb_mapped) {
                /* host vertex = 9 floats (pos3 + col4 + uv2), VERTEX_STRIDE=36 */
            const float* bv = (const float*)d3d12_legacy_vertex_slot_base();
                for (int k = 0; k < 3; k++) {
                    const float* v = bv + k * 9;
                    fprintf(stderr, "[D3D12-DBG]  hostvert[%d] pos=(%.3f,%.3f,%.3f) col=(%.2f,%.2f,%.2f,%.2f)\n",
                            k, v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
                }
            }
            for (u32 d = 0; d < s_d3d.draw_count && d < 3; d++) {
                const D3D12DrawRecord* dr = &s_d3d.draws[d];
                fprintf(stderr, "[D3D12-DBG]  draw[%u] is_vp=%d is_clear=%d topo=%u cnt=%u vbofs=%u startv=%u rt=%u\n",
                        d, dr->is_vp, dr->is_clear, dr->topology, dr->vertex_count,
                        dr->vb_byte_offset, dr->vb_byte_offset / VERTEX_STRIDE, dr->rt_off);
            }
            fflush(stderr);
          } }

        /* Replay each recorded draw with its own primitive topology and
         * the matching PSO class (triangle / line / point). The PSO class
         * must match the topology or D3D12 rejects the draw. */
        u32 last_topo = 0xFFFFFFFFu;
        ID3D12PipelineState* last_pso = NULL;
        u32 draws = s_d3d.draw_count;
        if (draws > MAX_DRAWS) draws = MAX_DRAWS;
        for (u32 d = 0; d < draws; d++) {
            const D3D12DrawRecord* dr = &s_d3d.draws[d];
            if (dr->is_vp) continue; /* drawn by the VP pass below */

            /* Select PSO: textured triangles (dbgfont) use the atlas PSO;
             * otherwise pick by topology class. */
            ID3D12PipelineState* target_pso = s_d3d.pipeline_state; /* default triangle */
            if (dr->textured && s_d3d.tex_ready[slot] && s_d3d.pipeline_state_tex) {
                target_pso = s_d3d.pipeline_state_tex;
            } else if (dr->topology == RSX_TOPOLOGY_POINTS) {
                target_pso = s_d3d.pipeline_state_points
                             ? s_d3d.pipeline_state_points : s_d3d.pipeline_state;
            } else if (dr->topology == RSX_TOPOLOGY_LINES ||
                       dr->topology == RSX_TOPOLOGY_LINE_STRIP) {
                target_pso = s_d3d.pipeline_state_lines
                             ? s_d3d.pipeline_state_lines : s_d3d.pipeline_state;
            }
            if (target_pso != last_pso) {
                s_d3d.cmd_list->lpVtbl->SetPipelineState(s_d3d.cmd_list, target_pso);
                last_pso = target_pso;
            }
            if (dr->topology != last_topo) {
                s_d3d.cmd_list->lpVtbl->IASetPrimitiveTopology(s_d3d.cmd_list, topo_to_d3d(dr->topology));
                last_topo = dr->topology;
            }
            s_d3d.cmd_list->lpVtbl->OMSetStencilRef(
                s_d3d.cmd_list, dr->stencil.front_ref & 0xFFu);
            s_d3d.cmd_list->lpVtbl->OMSetBlendFactor(s_d3d.cmd_list,
                                                     dr->blend_color);
            d3d12_apply_flat_vertex_order(dr);
            u32 start_vert = dr->vb_byte_offset / VERTEX_STRIDE;
            s_d3d.cmd_list->lpVtbl->DrawInstanced(
                s_d3d.cmd_list, dr->vertex_count, 1, start_vert, 0);
        }
    }

    /* VP pass: real decompiled vertex program + atlas alpha-test PS. Feeds raw
     * float4 attrib0 from vp_vb and the vp_c[] constant bank. */
    if (s_d3d.vp_ready && s_d3d.draw_count > 0) {
        int any = 0;
        for (u32 d = 0; d < s_d3d.draw_count && d < MAX_DRAWS; d++)
            if (s_d3d.draws[d].is_vp) { any = 1; break; }
        /* Textured geometry (dbgfont atlas) uses the sampling PS; untextured 3D
         * (vkcube) uses the colour-only PS. Fall back to whichever exists. */
        ID3D12PipelineState* vpso =
            (s_d3d.tex_ready[slot] && s_d3d.pipeline_state_vp) ? s_d3d.pipeline_state_vp
                                                         : s_d3d.pipeline_state_vp_color;
        if (!vpso) vpso = s_d3d.pipeline_state_vp;
        { static int cap = -1, n = 0;
          if (cap < 0) { const char* e = getenv("VP_SUBMIT"); cap = e ? atoi(e) : 0; }
          if (cap && n < cap) { n++;
            u32 nvp = 0, nclr = 0;
            for (u32 d2 = 0; d2 < s_d3d.draw_count && d2 < MAX_DRAWS; d2++) {
                if (s_d3d.draws[d2].is_vp)    nvp++;
                if (s_d3d.draws[d2].is_clear) nclr++;
            }
            fprintf(stderr, "[VPPASS] records=%u is_vp=%u clears=%u any=%d vpso=%p rootsig=%p\n",
                    s_d3d.draw_count, nvp, nclr, any,
                    (void*)vpso, (void*)s_d3d.vp_root_sig); } }
        if (any && vpso && have_default_sampler_table) {
            s_d3d.cmd_list->lpVtbl->SetGraphicsRootSignature(s_d3d.cmd_list, s_d3d.vp_root_sig);
            s_d3d.cmd_list->lpVtbl->SetPipelineState(s_d3d.cmd_list, vpso);
            s_d3d.cmd_list->lpVtbl->IASetPrimitiveTopology(s_d3d.cmd_list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            s_d3d.cmd_list->lpVtbl->SetGraphicsRootConstantBufferView(s_d3d.cmd_list, 0,
                s_d3d.vp_cb->lpVtbl->GetGPUVirtualAddress(s_d3d.vp_cb));
            ID3D12DescriptorHeap* heaps[] = { s_d3d.srv_heap, s_d3d.sampler_heap[slot] };
            s_d3d.cmd_list->lpVtbl->SetDescriptorHeaps(s_d3d.cmd_list, 2, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE gh;
            s_d3d.srv_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &gh);
            s_d3d.cmd_list->lpVtbl->SetGraphicsRootDescriptorTable(s_d3d.cmd_list, 1, gh);
            s_d3d.cmd_list->lpVtbl->SetGraphicsRootDescriptorTable(
                s_d3d.cmd_list, 3, default_sampler_table);
            D3D12_VERTEX_BUFFER_VIEW vbv;
            vbv.BufferLocation = s_d3d.vp_vb->lpVtbl->GetGPUVirtualAddress(s_d3d.vp_vb)
                               + (u64)s_d3d.vp_parity * MAX_VERTICES * 256;
            vbv.SizeInBytes    = MAX_VERTICES * 256;
            vbv.StrideInBytes  = 256;   /* 16 float4 attrib slots */
            s_d3d.cmd_list->lpVtbl->IASetVertexBuffers(s_d3d.cmd_list, 0, 1, &vbv);
            D3D12_GPU_DESCRIPTOR_HANDLE gh_base;
            s_d3d.srv_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_d3d.srv_heap, &gh_base);
            int cur_rt = -1;                       /* target A: -1 = backbuffer */
            int cur_none = 0;                      /* no color attachment */
            int cur_ds = -1;                       /* zeta: -1 = fallback DSV */
            int cur_ds_readonly = 0;
            D3D12_CPU_DESCRIPTOR_HANDLE active_dsv = dsv_handle;
            float vp_last_w = 0, vp_last_h = 0;    /* for DRAWARGS */
            float dvp_x = 0, dvp_y = 0, dvp_w = 0, dvp_h = 0;  /* per-draw, for DRAWARGS */
            int cur_m[3] = {-1, -1, -1};           /* MRT B/C/D: -1 = unbound   */
            double _rec0 = perf_on() ? perf_now() : 0.0;
            int seen_small_vp = 0, captured = 0;
            for (u32 d = 0; d < s_d3d.draw_count && d < MAX_DRAWS; d++) {
                const D3D12DrawRecord* dr = &s_d3d.draws[d];
                if (!dr->is_vp) continue;
                /* The reduced-viewport passes (this title's reflection, drawn
                 * into a region of the same target) finish here -- snapshot
                 * before the full-screen scene overwrites them. */
                if (!dr->is_clear) {
                    if (dr->vp_w && dr->vp_w < s_d3d.width) {
                        seen_small_vp = 1;
                        subvp_note(dr->vp_x, dr->vp_y, dr->vp_w, dr->vp_h);
                    }
                    else if (seen_small_vp && !captured &&
                             dr->vp_w == s_d3d.width && is_display_target_key(dr->rt_off)) {
                        captured = 1;
                        screen_copy_capture(fi, cur_rt);
                        subvp_capture(fi, cur_rt);
                        s_d3d.cmd_list->lpVtbl->OMSetRenderTargets(
                            s_d3d.cmd_list, 1, &rtv_handle, FALSE, &active_dsv);
                    }
                }
                /* Retarget color and guest depth surfaces together. */
                int want  = dr->rt_off  ? off_rt_find(dr->rt_off)  : -1;
                int want_none = dr->rt_none;
                int want_ds = (dr->zeta_ea != 0xFFFFFFFFu && dr->zeta_ea != 0 &&
                               rsx_zeta_format_supported(dr->zeta_format))
                                  ? off_depth_find(dr->zeta_ea) : -1;
                int want_ds_readonly = !dr->is_clear && want_ds >= 0 &&
                                       dr_reads_active_zeta(dr);
                /* Remember where geometry actually went: off_rt_find() matches
                 * on offset alone and slots share offsets at different sizes,
                 * so the composite cannot just look the source up by address. */
                if (want >= 0) { s_last_drawn_slot = want; s_last_drawn_off = dr->rt_off; }
                int wantm[3];
                for (int _m = 0; _m < 3; _m++)
                    wantm[_m] = dr->rt_mrt[_m] ? off_rt_find(dr->rt_mrt[_m]) : -1;
                if (want != cur_rt || want_none != cur_none || want_ds != cur_ds ||
                    want_ds_readonly != cur_ds_readonly ||
                    wantm[0] != cur_m[0] ||
                    wantm[1] != cur_m[1] || wantm[2] != cur_m[2]) {
                    D3D12_CPU_DESCRIPTOR_HANDLE rh[4];
                    UINT nrt = want_none ? 0u : 1u;
                    if (nrt) rh[0] = rtv_handle;
                    D3D12_VIEWPORT vp = {0, 0, (float)s_d3d.width, (float)s_d3d.height, 0.0f, 1.0f};
                    if (want_none) {
                        vp.Width = dr->rt_w ? (float)dr->rt_w : (float)s_d3d.width;
                        vp.Height = dr->rt_h ? (float)dr->rt_h : (float)s_d3d.height;
                    } else if (want >= 0) {
                        off_rt_transition(want, D3D12_RESOURCE_STATE_RENDER_TARGET);
                        rh[0] = off_rt_rtv(want);
                        vp.Width  = (float)s_d3d.off_rt[want].w;
                        vp.Height = (float)s_d3d.off_rt[want].h;
                    }
                    for (int _m = 0; !want_none && _m < 3 && wantm[_m] >= 0; _m++) {
                        off_rt_transition(wantm[_m], D3D12_RESOURCE_STATE_RENDER_TARGET);
                        rh[nrt++] = off_rt_rtv(wantm[_m]);
                    }
                    if (want_ds >= 0) {
                        off_depth_ensure_initialized(want_ds);
                        const D3D12_RESOURCE_STATES ds_state = want_ds_readonly
                            ? (D3D12_RESOURCE_STATE_DEPTH_READ |
                               D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
                            : D3D12_RESOURCE_STATE_DEPTH_WRITE;
                        off_depth_transition(want_ds, ds_state);
                        active_dsv = want_ds_readonly
                            ? off_depth_dsv_readonly(want_ds) : off_depth_dsv(want_ds);
                    } else {
                        active_dsv = dsv_handle;
                    }
                    D3D12_RECT sc = {0, 0, (LONG)vp.Width, (LONG)vp.Height};
                    vp_last_w = vp.Width; vp_last_h = vp.Height;
                    s_d3d.cmd_list->lpVtbl->OMSetRenderTargets(
                        s_d3d.cmd_list, nrt, nrt ? rh : NULL, FALSE, &active_dsv);
                    s_d3d.cmd_list->lpVtbl->RSSetViewports(s_d3d.cmd_list, 1, &vp);
                    s_d3d.cmd_list->lpVtbl->RSSetScissorRects(s_d3d.cmd_list, 1, &sc);
                    /* RT_CLEARDBG=2: magenta-clear the surface the DRAWS bind,
                     * at the moment they bind it. The guest's own clears may go
                     * to a different surface entirely (they do here: clears to
                     * 0xAB0000, draws to 0x1BE0000), so clearing on the clear
                     * path tests the wrong target. Anything not magenta
                     * afterwards was rasterised. */
                    { static int cd2 = -1;
                      if (cd2 < 0) { const char* e = getenv("RT_CLEARDBG"); cd2 = e ? atoi(e) : 0; }
                      if (cd2 >= 2 && want >= 0) {
                          const float mag[4] = {1.0f, 0.0f, 1.0f, 1.0f};
                          s_d3d.cmd_list->lpVtbl->ClearRenderTargetView(
                              s_d3d.cmd_list, off_rt_rtv(want), mag, 0, NULL);
                      } }
                    /* VP_TESTTRI=1: our own triangle, through the game's exact
                     * pipeline -- same vpso, same root signature, same vertex
                     * buffer, same RTV just bound. Written at vertex 1000 so it
                     * cannot collide with the guest's data. Pair with
                     * VP_BYPASS=1 so the shader is a trivial passthrough.
                     *
                     * If this renders, the GPU does read vp_vb and the fault is
                     * in the guest draws' own state. If it does not, nothing
                     * this backend draws reaches a render target at all. */
                    { static int tt = -1;
                      if (tt < 0) { const char* e = getenv("VP_TESTTRI"); tt = e ? atoi(e) : 0; }
                      if (tt && want >= 0 && s_d3d.vp_vb_mapped) {
                          float* vtx = (float*)((char*)s_d3d.vp_vb_mapped
                              + (u64)s_d3d.vp_parity * MAX_VERTICES * 256u
                              + (u64)1000 * 256u);
                          static const float tri[3][4] = {
                              {-2.0f, -2.0f, 0.0f, 1.0f},
                              { 2.0f, -2.0f, 0.0f, 1.0f},
                              { 0.0f,  2.0f, 0.0f, 1.0f} };
                          for (int _v = 0; _v < 3; _v++) {
                              float* dstv = vtx + (size_t)_v * (256u / 4u);
                              dstv[0] = tri[_v][0]; dstv[1] = tri[_v][1];
                              dstv[2] = tri[_v][2]; dstv[3] = tri[_v][3];
                          }
                          s_d3d.cmd_list->lpVtbl->DrawInstanced(s_d3d.cmd_list, 3, 1, 1000, 0);
                          { static int _n = 0;
                            if (PS3_LOG_FIRST(_n, 3)) fprintf(stderr, "[TESTTRI] issued on rt=%d off=0x%08X%c",
                                                  want, s_d3d.off_rt[want].off, 10); }
                      } }
                    cur_rt = want;
                    cur_none = want_none;
                    cur_ds = want_ds;
                    cur_ds_readonly = want_ds_readonly;
                    for (int _m = 0; _m < 3; _m++) cur_m[_m] = wantm[_m];
                }
                if (dr->is_clear) {
                    D3D12_CPU_DESCRIPTOR_HANDLE rh =
                        (cur_rt >= 0) ? off_rt_rtv(cur_rt) : rtv_handle;
                    /* RT_CLEARDBG=1: clear offscreen targets to magenta rather
                     * than the guest colour. "0.000% nonzero" cannot tell an
                     * untouched surface from one written black, and that
                     * ambiguity invalidates every read of an offscreen target.
                     * If the surface comes back magenta the clears land and the
                     * draws are writing black; if it stays black, nothing
                     * reaches it at all. */
                    /* Depth/stencil part: one depth buffer serves every
                     * target, and it is reset whenever the pass switches
                     * targets; a clear inside a pass has to land here. */
                    if (dr->clear_mask & 0x3u) {
                        D3D12_CLEAR_FLAGS zf = (D3D12_CLEAR_FLAGS)
                            (((dr->clear_mask & 0x1u) ? D3D12_CLEAR_FLAG_DEPTH : 0) |
                             ((dr->clear_mask & 0x2u) ? D3D12_CLEAR_FLAG_STENCIL : 0));
                        s_d3d.cmd_list->lpVtbl->ClearDepthStencilView(s_d3d.cmd_list, active_dsv,
                            zf, dr->clear_z, dr->clear_s, 0, NULL);
                    }
                    if (!(dr->clear_mask & 0xF0u))
                        continue;
                    { static int cd = -1;
                      if (cd < 0) { const char* e = getenv("RT_CLEARDBG"); cd = e ? atoi(e) : 0; }
                      if (cd && cur_rt >= 0) {
                          const float mag[4] = {1.0f, 0.0f, 1.0f, 1.0f};
                          s_d3d.cmd_list->lpVtbl->ClearRenderTargetView(s_d3d.cmd_list, rh, mag, 0, NULL);
                      } else
                          s_d3d.cmd_list->lpVtbl->ClearRenderTargetView(s_d3d.cmd_list, rh, dr->cc, 0, NULL); }
                    for (int _m = 0; _m < 3; _m++)
                        if (cur_m[_m] >= 0) {
                            D3D12_CPU_DESCRIPTOR_HANDLE rhm = off_rt_rtv(cur_m[_m]);
                            s_d3d.cmd_list->lpVtbl->ClearRenderTargetView(s_d3d.cmd_list, rhm, dr->cc, 0, NULL);
                        }
                    continue;
                }
                if (cur_ds >= 0)
                    off_depth_ensure_initialized(cur_ds);
                const DXGI_FORMAT draw_dsvfmt = cur_ds >= 0
                    ? s_d3d.off_depth[cur_ds].dsv_format
                    : DXGI_FORMAT_D24_UNORM_S8_UINT;
                /* Per-draw pipeline: prefer the guest's own compiled FP; fall
                 * back to the hardcoded atlas/colour PS pair. */
                D3D12FPSamplerLayout draw_sampler_layout = {0};
                u32 unnorm_dims[RSX_TEXTURE_UNITS][3];
                const u32 unnorm_mask = dr_unnorm_info(dr, unnorm_dims);
                const double pso_begin = perf_on() ? perf_now() : 0.0;
                ID3D12PipelineState* dpso =
                    dr->fp_addr ? vp_get_fp_pso(dr->vs_idx, dr->fp_addr, dr->blend_key,
                                                dr_num_rts(dr),
                                                dr->rt_off ? rsx_surface_dxgi(dr->rt_fmt)
                                                           : DXGI_FORMAT_R8G8B8A8_UNORM,
                                                draw_dsvfmt,
                                                dr->fp_exp32, dr->cmask,
                                                dr->color_mask_mrt,
                                                dr->blend_enable_mrt, dr->cull,
                                                dr->depth_test, dr->depth_write, dr->depth_func,
                                                dr->zmin_max_control,
                                                dr_cube_mask(dr),
                                                 unnorm_mask, unnorm_dims,
                                                dr->tex_shadow_mask,
                                                dr->tex_shadow_compare_key,
                                                dr->tex_shadow,
                                                dr->gamma_masks, dr->srgb_output,
                                                dr_depth_rgba_mask(dr),
                                                dr->poly_fill,
                                                dr->poly_scale_bits, dr->poly_bias_bits,
                                                dr->fog_mode, dr->fog_param0_bits, dr->fog_param1_bits,
                                                dr->shader_window, dr->texcoord_control_mask,
                                                dr->anti_aliasing_control,
                                                dr->flat_shading, dr->two_sided_lighting,
                                                &dr->stencil,
                                                &draw_sampler_layout, dr) : NULL;
                if (perf_on()) s_perf_pso += perf_now() - pso_begin;
                /* The generic fallback PSO uses the legacy 16-slot/256-byte
                 * input stream. It cannot safely consume an analyzed compact
                 * stream if this draw's guest pixel shader failed to compile. */
                if (!dpso && (dr->vp_raw_bytes ||
                    (dr->vp_stride && dr->vp_stride != VP_VERT_STRIDE))) {
                    static int skipped_compact_fallback = 0;
                    if (PS3_LOG_FIRST(skipped_compact_fallback, 16))
                        fprintf(stderr,
                                "[VP] skip compact draw fp=0x%08X mask=0x%04X stride=%u: no compatible guest PSO%c",
                                dr->fp_addr, dr->vp_input_mask, dr->vp_stride, 10);
                    continue;
                }
                ID3D12PipelineState* draw_vpso = vpso;
                const int use_tex_vp = s_d3d.tex_ready[slot] && s_d3d.pipeline_state_vp;
                if (dr->rt_none) {
                    if (draw_dsvfmt == DXGI_FORMAT_D16_UNORM)
                        draw_vpso = use_tex_vp ? s_d3d.pipeline_state_vp_no_rt_z16
                                               : s_d3d.pipeline_state_vp_color_no_rt_z16;
                    else
                        draw_vpso = use_tex_vp ? s_d3d.pipeline_state_vp_no_rt
                                               : s_d3d.pipeline_state_vp_color_no_rt;
                } else if (draw_dsvfmt == DXGI_FORMAT_D16_UNORM) {
                    draw_vpso = use_tex_vp ? s_d3d.pipeline_state_vp_z16
                                           : s_d3d.pipeline_state_vp_color_z16;
                }
                if (!draw_vpso) draw_vpso = vpso;
                s_d3d.cmd_list->lpVtbl->SetPipelineState(s_d3d.cmd_list,
                                                         dpso ? dpso : draw_vpso);
                /* Per-draw viewport: the guest rect when sane, else the
                 * full target. Scissor tracks the same rect. */
                {
                    float tw = cur_none && dr->rt_w ? (float)dr->rt_w
                        : (cur_rt >= 0 ? (float)s_d3d.off_rt[cur_rt].w : (float)s_d3d.width);
                    float th = cur_none && dr->rt_h ? (float)dr->rt_h
                        : (cur_rt >= 0 ? (float)s_d3d.off_rt[cur_rt].h : (float)s_d3d.height);
                    D3D12_VIEWPORT dvp = {0, 0, tw, th, 0.0f, 1.0f};
                    if (isfinite(dr->vp_min_depth) && isfinite(dr->vp_max_depth) &&
                        dr->vp_min_depth >= 0.0f && dr->vp_max_depth <= 1.0f &&
                        dr->vp_min_depth <= dr->vp_max_depth) {
                        dvp.MinDepth = dr->vp_min_depth;
                        dvp.MaxDepth = dr->vp_max_depth;
                    }
                    if (dr->vp_w >= 2 && dr->vp_h >= 2 &&
                        (float)(dr->vp_x + dr->vp_w) <= tw + 0.5f &&
                        (float)(dr->vp_y + dr->vp_h) <= th + 0.5f) {
                        dvp.TopLeftX = (float)dr->vp_x;
                        dvp.TopLeftY = (float)dr->vp_y;
                        dvp.Width    = (float)dr->vp_w;
                        dvp.Height   = (float)dr->vp_h;
                    }
                    /* RSX scissor is independent of the viewport. Clip the
                     * guest rectangle to the active target before submitting
                     * it; using the viewport here leaks fragments outside the
                     * guest's clipping region. */
                    const u32 target_w = tw > 0.0f ? (u32)tw : 0u;
                    const u32 target_h = th > 0.0f ? (u32)th : 0u;
                    const u64 sx1 = (u64)dr->scissor_x + dr->scissor_w;
                    const u64 sy1 = (u64)dr->scissor_y + dr->scissor_h;
                    u32 sx0c = dr->scissor_x < target_w ? dr->scissor_x : target_w;
                    u32 sy0c = dr->scissor_y < target_h ? dr->scissor_y : target_h;
                    u32 sx1c = sx1 < target_w ? (u32)sx1 : target_w;
                    u32 sy1c = sy1 < target_h ? (u32)sy1 : target_h;
                    if (sx1c < sx0c) sx1c = sx0c;
                    if (sy1c < sy0c) sy1c = sy0c;
                    D3D12_RECT dsc = {(LONG)sx0c, (LONG)sy0c,
                                      (LONG)sx1c, (LONG)sy1c};
                    dvp_x = dvp.TopLeftX; dvp_y = dvp.TopLeftY;
                    dvp_w = dvp.Width;    dvp_h = dvp.Height;
                    s_d3d.cmd_list->lpVtbl->RSSetViewports(s_d3d.cmd_list, 1, &dvp);
                    s_d3d.cmd_list->lpVtbl->RSSetScissorRects(s_d3d.cmd_list, 1, &dsc);
                }
                /* Per-draw descriptor table: fragment SRVs occupy t0-t15,
                 * vertex SRVs t16-t19. Transition sampled offscreen targets
                 * for the shader stage that reads each texture. */
                u64 feedback_captured = 0;
                for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
                    int feedback_rt = dr->tex_feedback[_u];
                    if (feedback_rt >= 0 && !(feedback_captured & (1ull << feedback_rt))) {
                        off_rt_feedback_capture(feedback_rt);
                        feedback_captured |= 1ull << feedback_rt;
                    }
                    int ds = dr->tex_depth[_u];
                    if (ds >= 0 && ds != cur_ds)
                        off_depth_transition(ds, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                    int rt = dr->tex_rt[_u];
                    if (rt >= 0 && rt != cur_rt &&
                        rt != cur_m[0] && rt != cur_m[1] && rt != cur_m[2])
                        off_rt_transition(rt, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                }
                for (int _u = 0; _u < RSX_MAX_VERTEX_TEXTURES; _u++) {
                    int ds = dr->vtex_depth[_u];
                    if (ds >= 0 && ds != cur_ds)
                        off_depth_transition(ds,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    int rt = dr->vtex_rt[_u];
                    if (rt >= 0 && rt != cur_rt &&
                        rt != cur_m[0] && rt != cur_m[1] && rt != cur_m[2])
                        off_rt_transition(rt,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
                D3D12_GPU_DESCRIPTOR_HANDLE gh = gh_base;
                gh.ptr += (u64)DRAW_SRV_SLOT(slot, d, 0) * s_d3d.srv_inc;
                s_d3d.cmd_list->lpVtbl->SetGraphicsRootDescriptorTable(s_d3d.cmd_list, 1, gh);
                D3D12_GPU_DESCRIPTOR_HANDLE draw_sampler_table = default_sampler_table;
                if (dpso && !d3d12_sampler_table_for_draw(
                        dr, &draw_sampler_layout, &draw_sampler_table))
                    draw_sampler_table = default_sampler_table;
                s_d3d.cmd_list->lpVtbl->SetGraphicsRootDescriptorTable(
                    s_d3d.cmd_list, 3, draw_sampler_table);
                /* Per-draw constants: this draw's vp_cb + FP texscale slots. */
                s_d3d.cmd_list->lpVtbl->SetGraphicsRootConstantBufferView(s_d3d.cmd_list, 0,
                    s_d3d.vp_cb->lpVtbl->GetGPUVirtualAddress(s_d3d.vp_cb)
                    + ((u64)s_d3d.vp_parity * MAX_DRAWS + dr->vp_cb_slot) * VP_CB_STRIDE);
                if (s_d3d.vp_fpcb)
                    s_d3d.cmd_list->lpVtbl->SetGraphicsRootConstantBufferView(s_d3d.cmd_list, 2,
                        s_d3d.vp_fpcb->lpVtbl->GetGPUVirtualAddress(s_d3d.vp_fpcb)
                        + ((u64)s_d3d.vp_parity * MAX_DRAWS + dr->cb_slot) * VP_FPCB_STRIDE);
                /* DRAWARGS=<N>: the arguments actually handed to the GPU, plus
                 * the target they are aimed at. A zero vertex count, or a start
                 * vertex past the buffer, rasterises nothing while every other
                 * check upstream still says the draw is fine. */
                { static int dcap = -1, dn = 0;
                  if (dcap < 0) { const char* e = getenv("DRAWARGS"); dcap = e ? atoi(e) : 0; }
                  if (dcap && dn < dcap) { dn++;
                      D3D12_CPU_DESCRIPTOR_HANDLE _rh =
                          (cur_rt >= 0) ? off_rt_rtv(cur_rt) : rtv_handle;
                      fprintf(stderr, "[DRAWARGS] verts=%u start=0 vbbyte=%u rt=%d rec=0x%08X bound=0x%08X"
                              " tex0=0x%08X texrt=%d dvp=%.0f,%.0f %.0fx%.0f%c",
                              dr->vertex_count, dr->vb_byte_offset, cur_rt,
                              dr->rt_off,
                              (cur_rt >= 0) ? s_d3d.off_rt[cur_rt].off : 0u,
                              dr->tex[0].raw, dr->tex_rt[0],
                              (double)dvp_x, (double)dvp_y,
                              (double)dvp_w, (double)dvp_h, 10); } }
                /* RT_REBIND=1: re-issue OMSetRenderTargets immediately before
                 * every draw instead of only when the target changes. A clear
                 * proves the RTV descriptor is good but not that the binding is
                 * still live at draw time -- ClearRenderTargetView writes
                 * through the descriptor and never consults the OM. If anything
                 * between the retarget and here drops the binding, this restores
                 * it. */
                { static int rb = -1;
                  if (rb < 0) { const char* e = getenv("RT_REBIND"); rb = e ? atoi(e) : 0; }
                  if (rb) {
                      D3D12_CPU_DESCRIPTOR_HANDLE brh[4];
                    UINT bn = cur_none ? 0u : 1u;
                    if (bn) brh[0] = (cur_rt >= 0) ? off_rt_rtv(cur_rt) : rtv_handle;
                    for (int _m = 0; !cur_none && _m < 3; _m++)
                          if (cur_m[_m] >= 0) brh[bn++] = off_rt_rtv(cur_m[_m]);
                      s_d3d.cmd_list->lpVtbl->OMSetRenderTargets(
                        s_d3d.cmd_list, bn, bn ? brh : NULL, FALSE, &active_dsv);
                  } }
                s_d3d.cmd_list->lpVtbl->OMSetStencilRef(
                    s_d3d.cmd_list, dr->stencil.front_ref & 0xFFu);
                s_d3d.cmd_list->lpVtbl->OMSetBlendFactor(s_d3d.cmd_list,
                                                         dr->blend_color);
                d3d12_apply_flat_vertex_order(dr);
                const u32 draw_stride = dr->vp_stride ? dr->vp_stride : VP_VERT_STRIDE;
                D3D12_VERTEX_BUFFER_VIEW draw_vbv;
                draw_vbv.BufferLocation = s_d3d.vp_vb->lpVtbl->GetGPUVirtualAddress(s_d3d.vp_vb)
                    + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE
                    + dr->vb_byte_offset;
                draw_vbv.SizeInBytes = dr->vertex_count * draw_stride;
                draw_vbv.StrideInBytes = draw_stride;
                if (dr->vp_raw_bytes) {
                    s_d3d.cmd_list->lpVtbl->SetGraphicsRootShaderResourceView(
                        s_d3d.cmd_list, 4, draw_vbv.BufferLocation);
                    s_d3d.cmd_list->lpVtbl->IASetVertexBuffers(
                        s_d3d.cmd_list, 0, 0, NULL);
                } else {
                    s_d3d.cmd_list->lpVtbl->IASetVertexBuffers(
                        s_d3d.cmd_list, 0, 1, &draw_vbv);
                }
                if (dr->vp_raw_index_offset) {
                    D3D12_INDEX_BUFFER_VIEW ibv;
                    ibv.BufferLocation = draw_vbv.BufferLocation + dr->vp_raw_index_offset;
                    ibv.SizeInBytes = dr->vertex_count * 4u;
                    ibv.Format = DXGI_FORMAT_R32_UINT;
                    s_d3d.cmd_list->lpVtbl->IASetIndexBuffer(s_d3d.cmd_list, &ibv);
                    s_d3d.cmd_list->lpVtbl->DrawIndexedInstanced(s_d3d.cmd_list,
                        dr->vertex_count, 1, 0, 0, 0);
                } else {
                    s_d3d.cmd_list->lpVtbl->DrawInstanced(s_d3d.cmd_list,
                        dr->vertex_count, 1, 0, 0);
                }
                /* VP_SUBMIT=<N>: prove the VP pass actually reaches the GPU.
                 * "records exist" and "draws were submitted" are different
                 * claims, and every blank-output investigation conflates them. */
                { static int cap = -1, n = 0;
                  if (cap < 0) { const char* e = getenv("VP_SUBMIT"); cap = e ? atoi(e) : 0; }
                  if (cap && n < cap) { n++;
                    fprintf(stderr, "[VPSUBMIT] draw[%u] verts=%u pso=%s vp=%ux%u tex0=0x%X(set=%d) tex1=0x%X slot=%d fp=0x%X%c",
                            d, dr->vertex_count, dpso ? "guest-fp" : "fallback",
                            dr->vp_w, dr->vp_h, dr->tex[0].raw, dr->tex[0].set,
                            dr->tex[1].raw, dr->tex_slot, dr->fp_addr, 10);
                    fprintf(stderr, "            vs_idx=%d cb_slot=%u vbbyte=%u stride=%u mask=0x%04X cull=%u blend=%d%c",
                            dr->vs_idx, dr->cb_slot, dr->vb_byte_offset,
                            draw_stride, dr->vp_input_mask,
                            dr->cull, dr->blend, 10); } }
            }
            if (perf_on()) s_perf_record += perf_now() - _rec0;
            /* Leave the backbuffer bound for the dump/present epilogue. */
            if (cur_rt >= 0 || cur_none) {
                s_d3d.cmd_list->lpVtbl->OMSetRenderTargets(s_d3d.cmd_list, 1, &rtv_handle, FALSE, &dsv_handle);
                D3D12_VIEWPORT vp = {0, 0, (float)s_d3d.width, (float)s_d3d.height, 0.0f, 1.0f};
                D3D12_RECT sc = {0, 0, (LONG)s_d3d.width, (LONG)s_d3d.height};
                s_d3d.cmd_list->lpVtbl->RSSetViewports(s_d3d.cmd_list, 1, &vp);
                s_d3d.cmd_list->lpVtbl->RSSetScissorRects(s_d3d.cmd_list, 1, &sc);
            }
        }
    }

    /* MEMPEEK=<hex addr>:<n>: n big-endian floats at a guest address each frame,
     * with a min/max so a buffer of constants is distinguishable from real data. */
    { const char* mp = d3d12_setting_mempeek();
      if (mp) { extern uint8_t* vm_base; static int _n = 0;
        u32 addr = 0; int cnt = 8; sscanf(mp, "%x:%d", &addr, &cnt);
        if (vm_base && addr && PS3_LOG_FIRST(_n, 6)) {
            float lo = 1e30f, hi = -1e30f; u32 nz = 0;
            for (int i = 0; i < cnt; i++) {
                const u8* q = vm_base + addr + i * 4;
                u32 bits = ((u32)q[0]<<24)|((u32)q[1]<<16)|((u32)q[2]<<8)|q[3];
                float f; memcpy(&f, &bits, 4);
                if (f == f && f < 1e30f && f > -1e30f) { if (f < lo) lo = f; if (f > hi) hi = f; }
                if (bits) nz++;
            }
            fprintf(stderr, "[MEMPEEK] 0x%08X n=%d nonzero=%u range[%g..%g]%c",
                    addr, cnt, nz, lo, hi, 10);
        } } }
    /* VRAMSCAN=1: which 64 KB blocks of guest memory change between frames.
     * "nothing writes the buffer we read" is only half an answer -- this says
     * where the producer IS writing, so the two can be matched up. */
    { static int vs = -1; if (vs < 0) vs = getenv("VRAMSCAN") ? 1 : 0;
      if (vs) { extern uint8_t* vm_base;
        static u32 prev[512]; static int have = 0; static int reported = 0;
        u32 base = 0xC6000000u; u32 nblk = 512;          /* 32 MB window */
        const char* bs = getenv("VRAMSCAN_BASE");
        if (bs) base = (u32)strtoul(bs, NULL, 0);
        if (vm_base && reported < 6) {
            u32 changed = 0; char list[512]; int ln = 0;
            for (u32 b = 0; b < nblk; b++) {
                const u8* p = vm_base + base + b * 0x10000u;
                u32 h = 2166136261u;
                for (u32 i = 0; i < 0x10000u; i += 61) h = (h ^ p[i]) * 16777619u;
                if (have && h != prev[b]) { changed++;
                    if (ln < 400) ln += snprintf(list + ln, sizeof(list) - ln,
                                                 " +0x%X", b * 0x10000u); }
                prev[b] = h;
            }
            if (have && changed) { reported++;
                fprintf(stderr, "[VRAMSCAN] base=0x%08X %u/512 blocks changed:%s%c",
                        base, changed, list, 10); }
            have = 1;
        } } }
    /* FRAME_BUDGET=1: geometry a frame asked for vs what the batch can hold.
     * Reported from render_frame so it works under RSX_ACCUM_FRAME too -- it
     * used to live in the clear-boundary present path, which accum bypasses, so
     * it silently produced nothing in exactly the configuration being used. */
    { static int fb = -1;
      if (fb < 0) { const char* e = getenv("FRAME_BUDGET"); fb = e ? atoi(e) : 0; }
      if (fb && (s_req_draws || s_drop_records || s_drop_vertices || s_drop_input_elements)) { static int n = 0; if (PS3_LOG_FIRST(n, 40))
        fprintf(stderr, "[BUDGET] frame: requested %llu input elements / %llu draws; "
                        "vertex capacity=%u, clipped output vertices=%llu, rejected input elements=%llu, "
                        "truncated chunks=%llu, dropped draw records=%llu%c",
                (unsigned long long)s_req_verts, (unsigned long long)s_req_draws,
                MAX_VERTICES, (unsigned long long)s_drop_vertices,
                (unsigned long long)s_drop_input_elements,
                (unsigned long long)s_drop_draws, (unsigned long long)s_drop_records, 10); }
      s_req_verts = 0; s_req_draws = 0; s_drop_draws = 0;
      s_drop_vertices = 0; s_drop_input_elements = 0; s_drop_records = 0; }
    { static int dt = -1;
      if (dt < 0) { const char* e = getenv("RSX_DBG_TRACK"); dt = e ? atoi(e) : 0; }
      if (dt) { static int fr = 0; fr++;
        if (s_lock_n)
            fprintf(stderr, "[RSX_DBG_TRACK] frame %d: %u duck draws, ndc=(%.3f,%.3f)%c",
                    fr, s_lock_n, (float)(s_lock_sx/s_lock_n), (float)(s_lock_sy/s_lock_n), 10);
        else if (fr % 20 == 0)
            fprintf(stderr, "[RSX_DBG_TRACK] frame %d: NO duck draws%c", fr, 10);
      } }
    if (s_lock_n) {                       /* publish this frame's centroid */
        s_lock_x = (float)(s_lock_sx / s_lock_n);
        s_lock_y = (float)(s_lock_sy / s_lock_n);
        s_lock_valid = 1;
        s_lock_sx = s_lock_sy = 0.0; s_lock_n = 0;
    }
    s_dbg_last_draws = s_d3d.draw_count;
    s_d3d.vb_offset  = 0; /* reset for next frame */
    s_d3d.vp_vb_offset = 0;
    rsx_raw_upload_reset(&s_raw_upload_cache);
    s_d3d.vp_vb_vertex_count = 0;
    s_d3d.draw_count = 0;
    s_d3d.vp_parity ^= 1;  /* next frame's records go to the other half */
    s_batch_slot_acquired = 0;

    /* Debug: RTT_SAVERT=<hex raw offset>[:frame] copies that offscreen RT
     * into a readback buffer this frame and writes rt_save.bmp (half-float
     * RTs are tonemapped |v| -> byte). */
    static ID3D12Resource* s_rtsave_buf = NULL;
    static u32 s_rtsave_state = 0;   /* 1 = copy queued this frame */
    static u32 s_rtsave_w, s_rtsave_h, s_rtsave_pitch, s_rtsave_dxgi;
    { const char* sv = d3d12_setting_rtt_savert();
      static int _done = 0;
      static u32 _skip = 0xFFFFFFFFu;
      if (_skip == 0xFFFFFFFFu) { const char* sk = getenv("RTT_SAVERT_SKIP");
          _skip = sk ? (u32)strtoul(sk, NULL, 0) : 0; }   /* frame 60 is far too
          early for a title that takes a minute to reach its menu */
      /* RTT_SAVERT_AFTER=<seconds> triggers on wall time instead of a frame
       * count. This title runs its menu at a few frames a second, so a frame
       * number is not a usable way to say "once it has settled". */
      static u32 _after = 0xFFFFFFFFu;
      if (_after == 0xFFFFFFFFu) { const char* sa = getenv("RTT_SAVERT_AFTER");
          _after = sa ? (u32)strtoul(sa, NULL, 0) : 0; }
      static clock_t _t0 = 0;
      if (!_t0) _t0 = clock();
      const int _due = _after
          ? ((u32)((clock() - _t0) / CLOCKS_PER_SEC) >= _after)
          : (s_d3d.frame_count > 60 + _skip);
      { static int _p1 = 0;
        if (sv && _due && !_p1) { _p1 = 1;
            printf("[RTT_SAVERT] due: want=0x%lX found_rt=%d frame=%u\n",
                   strtoul(sv, NULL, 16), off_rt_find((u32)strtoul(sv, NULL, 16)),
                   (unsigned)s_d3d.frame_count); fflush(stdout); } }
      if (sv && !_done && _due) {
        int rt = off_rt_find((u32)strtoul(sv, NULL, 16));
        fprintf(stderr, "[RTT_SAVERT] rt=%d res=%p\n", rt, rt >= 0 ? (void*)s_d3d.off_rt[rt].res : NULL);
        if (rt >= 0 && s_d3d.off_rt[rt].res) {
            OffRT* r = &s_d3d.off_rt[rt];
            u32 bpp = (r->dxgi == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 8 :
                      (r->dxgi == DXGI_FORMAT_R32G32B32A32_FLOAT) ? 16 : 4;
            u32 pitch = (r->w * bpp + 255) & ~255u;
            fprintf(stderr, "[RTT_SAVERT] rt=%d w=%u h=%u bpp=%u pitch=%u dxgi=%d bytes=%llu\n",
                    rt, r->w, r->h, bpp, pitch, (int)r->dxgi,
                    (unsigned long long)((u64)pitch * r->h));
            if (!s_rtsave_buf) {
                D3D12_HEAP_PROPERTIES hp = {0}; hp.Type = D3D12_HEAP_TYPE_READBACK;
                D3D12_RESOURCE_DESC rd = {0};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                rd.Width = (u64)pitch * r->h; rd.Height = 1; rd.DepthOrArraySize = 1;
                rd.MipLevels = 1; rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                s_d3d.device->lpVtbl->CreateCommittedResource(
                    s_d3d.device, &hp, D3D12_HEAP_FLAG_NONE, &rd,
                    D3D12_RESOURCE_STATE_COPY_DEST, NULL,
                    &IID_ID3D12Resource, (void**)&s_rtsave_buf);
            }
            if (!s_rtsave_buf) fprintf(stderr, "[RTT_SAVERT] readback buffer alloc FAILED\n");
            if (s_rtsave_buf) {
                off_rt_transition(rt, D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION cdst = {0}, csrc = {0};
                cdst.pResource = s_rtsave_buf;
                cdst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                cdst.PlacedFootprint.Footprint.Format   = (DXGI_FORMAT)r->dxgi;
                cdst.PlacedFootprint.Footprint.Width    = r->w;
                cdst.PlacedFootprint.Footprint.Height   = r->h;
                cdst.PlacedFootprint.Footprint.Depth    = 1;
                cdst.PlacedFootprint.Footprint.RowPitch = pitch;
                csrc.pResource = r->res;
                csrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &cdst, 0, 0, 0, &csrc, NULL);
                off_rt_transition(rt, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                s_rtsave_state = 1;
                s_rtsave_w = r->w; s_rtsave_h = r->h;
                s_rtsave_pitch = pitch; s_rtsave_dxgi = r->dxgi;
                fprintf(stderr, "[RTT_SAVERT] copy queued: rt=%d %ux%u pitch=%u\n",
                        rt, r->w, r->h, pitch);
                _done = 1;
            }
        }
      } }

    /* CELLMARK_DUMP_MINDRAWS=<N>: only consider frames carrying at least N draw
     * records. Most render_frame calls come from the guest flip with a batch the
     * clear boundary already drained, so they present an empty backbuffer -- and
     * sampling those wastes every dump on a black image while the frames that do
     * hold the scene go uncaptured. */
    { static int mind = -1;
      if (mind < 0) { const char* e = getenv("CELLMARK_DUMP_MINDRAWS");
                      mind = e ? atoi(e) : 0; }
      /* draw_count is already reset by here; s_dbg_last_draws holds the
       * count this frame actually rendered. */
      if (mind > 0 && s_dbg_last_draws < (u32)mind) goto skip_dump_consider; }
    if (s_d3d.dump_skip_left > 0 && s_d3d.dump_frames_left > 0) s_d3d.dump_skip_left--;
skip_dump_consider: ;
    guest_fb_present(fi);
    /* A guest frame may be drained across several offscreen batches. Only
     * fall back to CPU scanout on a real flip, after checking all batches in
     * that frame rather than just the final (often empty) batch. */
    {
        static u64 draws_since_present = 0;
        draws_since_present += s_dbg_last_draws;
        if (s_present_this_frame) {
            if (!draws_since_present) scanout_auto_present(fi);
            draws_since_present = 0;
        }
    }
    composite_present(fi);
    /* The guest display surface persists across flips. An empty guest flip
     * still selects the same buffer, but the swapchain rotates to a different
     * backbuffer; copy the last known guest image again or that backbuffer can
     * expose stale contents from two presents ago. Only reuse surfaces that
     * have already been scanned out after a draw. */
    {
        const u32 display_key = display_rt_key(s_present_display_id);
        const int display_slot = off_rt_find(display_key);
        if (display_slot >= 0 &&
            (frame_writes_selected_display || s_d3d.off_rt[display_slot].contents_valid)) {
            if (scanout_display_buffer(fi, s_present_display_id))
                s_d3d.off_rt[display_slot].contents_valid = 1;
        }
    }
    int dumping = (s_d3d.dump_frames_left > 0 && s_d3d.readback_buf
                   && s_d3d.dump_skip_left == 0);
    { static int mind2 = -1;
      if (mind2 < 0) { const char* e = getenv("CELLMARK_DUMP_MINDRAWS");
                       mind2 = e ? atoi(e) : 0; }
      if (mind2 > 0 && s_dbg_last_draws < (u32)mind2) dumping = 0; }
    /* CELLMARK_DUMP_EVERY=N: after each dump, skip N-1 frames -- samples the whole
     * run instead of one contiguous window, so a short burst of real content
     * can't fall between the dumped frames. */
    if (dumping) {
        static int s_every = -1;
        if (s_every < 0) { const char* e = getenv("CELLMARK_DUMP_EVERY");
                           s_every = e ? atoi(e) : 0; }
        if (s_every > 1) s_d3d.dump_skip_left = s_every - 1;
    }
    if (dumping) {
        /* RT -> COPY_SOURCE, copy into the readback buffer, then -> PRESENT. */
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &barrier);

        D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
        dst.pResource = s_d3d.readback_buf;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Offset = 0;
        dst.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
        dst.PlacedFootprint.Footprint.Width    = s_d3d.width;
        dst.PlacedFootprint.Footprint.Height   = s_d3d.height;
        dst.PlacedFootprint.Footprint.Depth    = 1;
        dst.PlacedFootprint.Footprint.RowPitch = s_d3d.readback_pitch;
        src.pResource = s_d3d.render_targets[fi];
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        s_d3d.cmd_list->lpVtbl->CopyTextureRegion(s_d3d.cmd_list, &dst, 0, 0, 0, &src, NULL);

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &barrier);
    } else {
        /* End-of-frame fallback snapshot: used when the frame never contained a
         * reduced-viewport pass to capture mid-frame. */
        screen_copy_capture(fi, -1);
        /* Transition render target to PRESENT state */
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        s_d3d.cmd_list->lpVtbl->ResourceBarrier(s_d3d.cmd_list, 1, &barrier);
    }

    /* Close and execute */
    rsx_gpu_timer_end(&s_gpu_timer,s_d3d.cmd_list,slot);
    s_d3d.cmd_list->lpVtbl->Close(s_d3d.cmd_list);
    ID3D12CommandList* cmd_lists[] = {(ID3D12CommandList*)s_d3d.cmd_list};
    s_d3d.cmd_queue->lpVtbl->ExecuteCommandLists(s_d3d.cmd_queue, 1, cmd_lists);
    d3d12_signal_submission(slot);
    if(s_gpu_timer.active) {s_gpu_timer.pending[slot]=s_d3d.last_fence_value;s_gpu_timer.active=0;}

    /* D3D12_IQ=1: drain the debug layer's message queue after submitting the
     * frame. The legacy (is_vp=0) DrawInstanced path silently produced ZERO
     * pixels in flOw's render injection while ClearRenderTargetView worked, and
     * every input (verts, offsets, layout, MVP, PSO, root sig, depth, cull) was
     * verified correct -- so the only remaining explanation is a validation
     * error the debug layer is swallowing. Off unless the env var is set. */
    { static int _iq = -1;
      if (_iq < 0) { const char* e = getenv("D3D12_IQ"); _iq = e ? 1 : 0; }
      if (_iq && s_d3d.device) {
        ID3D12InfoQueue* iq = NULL;
        if (SUCCEEDED(s_d3d.device->lpVtbl->QueryInterface(
                s_d3d.device, &IID_ID3D12InfoQueue, (void**)&iq)) && iq) {
            UINT64 n = iq->lpVtbl->GetNumStoredMessages(iq);
            static int _printed = 0;
            for (UINT64 mi = 0; mi < n && _printed < 80; mi++) {
                SIZE_T len = 0;
                iq->lpVtbl->GetMessage(iq, mi, NULL, &len);
                D3D12_MESSAGE* m = (D3D12_MESSAGE*)malloc(len);
                if (m && SUCCEEDED(iq->lpVtbl->GetMessage(iq, mi, m, &len))) {
                    fprintf(stderr, "[D3D12-IQ][sev=%d id=%d] %s\n",
                            (int)m->Severity, (int)m->ID, m->pDescription);
                    _printed++;
                }
                free(m);
            }
            if (n) iq->lpVtbl->ClearStoredMessages(iq);
            iq->lpVtbl->Release(iq);
            fflush(stderr);
        }
      } }

    if (s_sc_dump_pending) {
        s_sc_dump_pending = 0;
        wait_for_gpu();
        s_dump_name = "screencopy";
        dump_backbuffer_bmp();
        s_dump_name = NULL;
    }
    if (dumping) {
        wait_for_gpu();            /* ensure the copy finished before mapping */
        dump_backbuffer_bmp();
        s_d3d.dump_frames_left--;
    }

    if (s_rtsave_state) {
        fprintf(stderr, "[RTT_SAVERT] writing rt_save.bmp %ux%u\n", s_rtsave_w, s_rtsave_h);
        if (!dumping) { double _g = perf_on() ? perf_now() : 0.0;
                        wait_for_gpu();
                        if (perf_on()) s_perf_record += perf_now() - _g; }
        void* mp = NULL; D3D12_RANGE rr = {0, (SIZE_T)s_rtsave_pitch * s_rtsave_h};
        if (SUCCEEDED(s_rtsave_buf->lpVtbl->Map(s_rtsave_buf, 0, &rr, &mp)) && mp) {
            FILE* f = fopen("rt_save.bmp", "wb");
            if (f) {
                u32 w = s_rtsave_w, h = s_rtsave_h;
                u32 rowb = (w * 3 + 3) & ~3u;
                u32 datasz = rowb * h;
                u8 hdr[54] = {0};
                hdr[0]='B'; hdr[1]='M';
                *(u32*)(hdr+2) = 54 + datasz; *(u32*)(hdr+10) = 54;
                *(u32*)(hdr+14) = 40; *(int*)(hdr+18) = (int)w; *(int*)(hdr+22) = (int)h;
                *(u16*)(hdr+26) = 1; *(u16*)(hdr+28) = 24; *(u32*)(hdr+34) = datasz;
                fwrite(hdr, 1, 54, f);
                u8* line = (u8*)malloc(rowb);
                for (int y = (int)h - 1; y >= 0; y--) {
                    const u8* srow = (const u8*)mp + (u64)y * s_rtsave_pitch;
                    memset(line, 0, rowb);
                    for (u32 x = 0; x < w; x++) {
                        float rv, gv, bv;
                        if (s_rtsave_dxgi == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                            const u16* hp16 = (const u16*)(srow + (u64)x * 8);
                            /* crude half->float: sign|exp|mant */
                            float v[3];
                            for (int c2 = 0; c2 < 3; c2++) {
                                u16 hv = hp16[c2];
                                u32 sign = (hv >> 15) & 1, exp = (hv >> 10) & 0x1F, man = hv & 0x3FF;
                                float fv;
                                if (exp == 0) fv = (float)man / 16777216.0f;
                                else { u32 fb = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
                                       memcpy(&fv, &fb, 4); }
                                v[c2] = fv;
                            }
                            rv = v[0]; gv = v[1]; bv = v[2];
                        } else if (s_rtsave_dxgi == DXGI_FORMAT_R32G32B32A32_FLOAT) {
                            const float* fp32 = (const float*)(srow + (u64)x * 16);
                            rv = fp32[0]; gv = fp32[1]; bv = fp32[2];
                        } else {
                            const u8* p8 = srow + (u64)x * 4;
                            rv = p8[0] / 255.0f; gv = p8[1] / 255.0f; bv = p8[2] / 255.0f;
                        }
                        /* RTT_SAVEA=1: show the alpha lane in RED (gates
                         * like wave's mask.w live there). */
                        if (d3d12_setting_rtt_savea()) {
                            float av;
                            if (s_rtsave_dxgi == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                                const u16* hp16 = (const u16*)(srow + (u64)x * 8);
                                u16 hv = hp16[3];
                                u32 sign = (hv >> 15) & 1, exp = (hv >> 10) & 0x1F, man = hv & 0x3FF;
                                if (exp == 0) av = (float)man / 16777216.0f;
                                else { u32 fb = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
                                       memcpy(&av, &fb, 4); }
                            } else if (s_rtsave_dxgi == DXGI_FORMAT_R32G32B32A32_FLOAT) {
                                av = ((const float*)(srow + (u64)x * 16))[3];
                            } else {
                                av = (srow + (u64)x * 4)[3] / 255.0f;
                            }
                            rv = av;
                        }
                        /* |v| tonemap so signed heights are visible */
                        float ar = rv < 0 ? -rv : rv, ag = gv < 0 ? -gv : gv, ab = bv < 0 ? -bv : bv;
                        if (ar > 1) ar = 1; if (ag > 1) ag = 1; if (ab > 1) ab = 1;
                        line[x*3+0] = (u8)(ab * 255.0f);
                        line[x*3+1] = (u8)(ag * 255.0f);
                        line[x*3+2] = (u8)(ar * 255.0f);
                    }
                    fwrite(line, 1, rowb, f);
                }
                free(line);
                fclose(f);
                printf("[D3D12] wrote rt_save.bmp (%ux%u dxgi=%u)\n", w, h, s_rtsave_dxgi);
            }
            s_rtsave_buf->lpVtbl->Unmap(s_rtsave_buf, 0, NULL);
        }
        s_rtsave_state = 0;
    }

    /* rsxcap: snapshot whole frames (backbuffer + every offscreen RT + a draw
     * manifest) at RSX_CAP=start[:count[:stride]] frame_counts. Runs before the
     * present transition would matter -- the backbuffer is in PRESENT here and
     * is restored to PRESENT after each readback. */
    { const char* cap = getenv("RSX_CAP");
      if (cap) {
        static int s_cap_parsed = 0;
        static u32 s_cap_start = 0, s_cap_count = 1, s_cap_stride = 1, s_cap_done = 0;
        static u32 s_cap_min_draws = 1;
        if (!s_cap_parsed) {
            s_cap_parsed = 1;
            u32 a = 0, b = 1, c = 1;
            int n = sscanf(cap, "%u:%u:%u", &a, &b, &c);
            s_cap_start  = (n >= 1 && a) ? a : 200;
            s_cap_count  = (n >= 2 && b) ? b : 1;
            s_cap_stride = (n >= 3 && c) ? c : 1;
            const char* min_draws = getenv("RSX_CAP_MIN_DRAWS");
            if (min_draws) s_cap_min_draws = (u32)strtoul(min_draws, NULL, 10);
        }
        if (s_cap_done < s_cap_count) {
            u32 target = s_cap_start + s_cap_done * s_cap_stride;
            if (s_d3d.frame_count >= target && s_dbg_last_draws > 0 &&
                s_dbg_last_draws >= s_cap_min_draws) {
                rsx_capture_frame(fi, s_dbg_last_draws, s_cap_done);
                s_cap_done++;
            }
        }
      } }

    /* RSX_CAP_TRIGGER=<file>: capture the next submitted frame after the
     * marker file appears. This is useful for scenes reached through user
     * input, where a fixed frame number is unreliable. The one-shot capture
     * uses the same RSX_CAP_DIR and includes the backbuffer, offscreen color
     * targets, draw manifest, and vertex constants. */
    { static int s_cap_trigger_done = 0;
      const char* trigger = getenv("RSX_CAP_TRIGGER");
      const char* min_draws_env = getenv("RSX_CAP_TRIGGER_MIN_DRAWS");
      unsigned min_draws = min_draws_env ? (unsigned)strtoul(min_draws_env, NULL, 0) : 1u;
      if (!min_draws) min_draws = 1u;
      if (s_present_this_frame && !s_cap_trigger_done && trigger && *trigger && s_dbg_last_draws >= min_draws) {
          FILE* marker = fopen(trigger, "rb");
          if (marker) {
              fclose(marker);
              remove(trigger);
              rsx_capture_frame(fi, s_dbg_last_draws, 0);
              s_cap_trigger_done = 1;
              const char* dir = getenv("RSX_CAP_DIR");
              fprintf(stderr, "[RSX_CAP] captured frame=%llu draws=%u dir=%s/frame_00\n",
                      (unsigned long long)s_d3d.frame_count, s_dbg_last_draws,
                      dir && *dir ? dir : "rsxcap");
          }
      } }

    /* Present */
    if (s_present_this_frame)
        s_d3d.swap_chain->lpVtbl->Present(s_d3d.swap_chain, 1, 0); /* vsync */   /* skipped for an offscreen-only batch */

    move_to_next_frame(s_present_this_frame);

    s_d3d.frame_count++;

    /* RPCS3-style titlebar stats, refreshed once a second: presented FPS,
     * draw count of the last frame, and the backbuffer size. */
    {
        static ULONGLONG s_tt0 = 0;
        static u64 s_tframes = 0;
        static u32 s_pending_draws = 0, s_presented_draws = 0;
        s_pending_draws += s_dbg_last_draws;
        if (s_present_this_frame) {
            s_tframes++;
            s_presented_draws = s_pending_draws;
            s_pending_draws = 0;
        }
        ULONGLONG tnow = GetTickCount64();
        if (s_tt0 == 0) s_tt0 = tnow;
        if (tnow - s_tt0 >= 1000 && s_d3d.hwnd) {
            extern char g_rsx_title_base[128];
            char tb[256];
            snprintf(tb, sizeof(tb), "%s | FPS: %.2f | draws: %u | %ux%u",
                     g_rsx_title_base,
                     s_tframes * 1000.0 / (double)(tnow - s_tt0),
                     s_presented_draws, s_d3d.width, s_d3d.height);
            SetWindowTextA(s_d3d.hwnd, tb);
            s_tframes = 0;
            s_tt0 = tnow;
        }
    }
}

/* PERF must cover the complete render operation. The earlier timer stopped
 * inside render_frame_impl before command-list submission, Present, and fence
 * advancement, incorrectly counting that tail as "outside-render". */
static void render_frame(void)
{
    if (!perf_on()) {
        render_frame_impl();
        return;
    }

    static double window_start = 0.0;
    static int window_batches = 0;
    const double start = perf_now();
    if (window_batches == 0)
        window_start = start;

    render_frame_impl();
    const double end = perf_now();
    s_perf_rf += end - start;

    if (++window_batches == 20) {
        double fr = end - window_start;
        if (fr <= 0.0) fr = 1.0;
        double outside = fr - s_perf_rf;
        if (outside < 0.0) outside = 0.0;
        fprintf(stderr, "[PERF] %.2f batches/s | tex %.2fs (%.0f%%, %d calls) | vtx %.2fs"
                        " (%.0f%%, %.0fk verts) | render_frame %.2fs (%.0f%%)"
                        " | prepass %.2fs (%.0f%%) [srv %.2fs %.0f%%] | record CPU %.2fs (%.0f%%) [pso %.2fs %.0f%% %d calls %d MISS %dKB hashed | PS cache %d hit/%d miss | disk shader %d/%d PSO %d/%d] | outside-render %.2fs (%.0f%%) | texture cache %u hit/%u upload/%u alloc%c",
                20.0 / fr,
                s_perf_tex, 100.0 * s_perf_tex / fr, s_perf_ntex,
                s_perf_vtx, 100.0 * s_perf_vtx / fr,
                (double)s_perf_nverts / 1000.0,
                s_perf_rf, 100.0 * s_perf_rf / fr,
                s_perf_pre, 100.0 * s_perf_pre / fr,
                s_perf_srv, 100.0 * s_perf_srv / fr,
                s_perf_record, 100.0 * s_perf_record / fr,
                s_perf_pso, 100.0 * s_perf_pso / fr,
                s_perf_pso_calls, s_perf_pso_miss, s_perf_pso_hashbytes / 1024,
                s_perf_ps_blob_hits, s_perf_ps_blob_misses,
                s_perf_disk_shader_hits, s_perf_disk_shader_misses,
                s_perf_disk_pso_hits, s_perf_disk_pso_misses,
                outside, 100.0 * outside / fr,
                s_perf_tex_hits, s_perf_tex_uploads, s_perf_tex_allocs, 10);
        if(s_gpu_timer.samples) {
            fprintf(stderr,"[PERF-GPU] %.3f ms/batch avg | %.3f ms max | %u completed batches\n",
                s_gpu_timer.total_ms/s_gpu_timer.samples,s_gpu_timer.max_ms,s_gpu_timer.samples);
            s_gpu_timer.total_ms=s_gpu_timer.max_ms=0;s_gpu_timer.samples=0;
        }
        fprintf(stderr, "[PERF-SYNC] fence %.3fs %u calls | RT recreate %u evict %u\n",
                s_perf_fence_wait, s_perf_fence_calls, s_perf_rt_recreates, s_perf_rt_evicts);
        fprintf(stderr, "[PERF-VTX] pool %u dispatches %llu items | expanded %u draws %llu primitives\n",
                s_perf_vtx_pool_calls, (unsigned long long)s_perf_vtx_pool_items,
                s_perf_vtx_expanded_draws, (unsigned long long)s_perf_vtx_expanded_primitives);
        fprintf(stderr, "[PERF-GPUFETCH] %u draws | %llu raw bytes | %llu upload bytes saved\n",
                s_perf_raw_draws, (unsigned long long)s_perf_raw_bytes,
                (unsigned long long)s_perf_raw_saved);
        fprintf(stderr, "[PERF-RAWREUSE] %u reused packets | %u indexed draws\n",
                s_perf_raw_reused, s_perf_raw_indexed);
        s_perf_raw_reused = s_perf_raw_indexed = 0;
        s_perf_raw_draws = 0; s_perf_raw_bytes = s_perf_raw_saved = 0;
        s_perf_vtx_pool_calls = s_perf_vtx_expanded_draws = 0;
        s_perf_vtx_pool_items = s_perf_vtx_expanded_primitives = 0;
        fprintf(stderr, "[PERF-SRV] %llu creates | %llu cached | %llu copies\n",
                (unsigned long long)s_srv_cache.creates,
                (unsigned long long)s_srv_cache.hits,
                (unsigned long long)s_srv_cache.copies);
        s_srv_cache.creates = s_srv_cache.hits = s_srv_cache.copies = 0;
        s_perf_fence_wait = 0.0;
        s_perf_fence_calls = s_perf_rt_recreates = s_perf_rt_evicts = 0;
        s_perf_tex = 0.0; s_perf_vtx = 0.0; s_perf_rf = 0.0;
        s_perf_record = 0.0; s_perf_pre = 0.0; s_perf_srv = 0.0; s_perf_pso = 0.0;
        s_perf_pso_calls = 0; s_perf_pso_miss = 0; s_perf_pso_hashbytes = 0;
        s_perf_ps_blob_hits = 0; s_perf_ps_blob_misses = 0;
        s_perf_disk_shader_hits = s_perf_disk_shader_misses = 0;
        s_perf_disk_pso_hits = s_perf_disk_pso_misses = 0;
        s_perf_ntex = 0; s_perf_texbytes = 0; s_perf_nverts = 0;
        s_perf_tex_hits = s_perf_tex_uploads = s_perf_tex_allocs = 0;
        window_batches = 0;
        window_start = 0.0;
    }
}

/* ---------------------------------------------------------------------------
 * RSX backend callbacks
 * -----------------------------------------------------------------------*/

static void d3d12_wait_idle(void* ud)
{
    (void)ud;
    if (!s_d3d.initialized || !s_d3d.cmd_queue || !s_d3d.fence)
        return;

    /* RSX draws are buffered in CPU-side records until a present boundary.
     * Submit that queue without presenting, then wait on the D3D12 fence so
     * WAIT_FOR_IDLE orders later guest work after the completed draws. */
    if (s_d3d.draw_count) {
        const int saved_present = s_present_this_frame;
        s_present_this_frame = 0;
        render_frame();
        s_present_this_frame = saved_present;
    }
    wait_for_gpu();
}

static int d3d12_init(void* ud, u32 width, u32 height)
{
    (void)ud;
    printf("[D3D12] Backend init(%ux%u)\n", width, height);
    return 0;
}

static void d3d12_shutdown(void* ud)
{
    (void)ud;
    printf("[D3D12] Backend shutdown\n");
}

static void d3d12_begin_frame(void* ud)
{
    (void)ud;
}

static void d3d12_end_frame(void* ud)
{
    (void)ud;
}

static u32 s_dbg_clears_since_present = 0;   /* CELLMARK_BLINKDBG */
static u32 s_clear_presents = 0;   /* presents issued at clear (frame boundary) */
/* Accumulated-frame mode defers clear-boundary presents to the guest flip.
 * Still remember that a real display frame has begun so the periodic fallback
 * present cannot expose a partially accumulated backbuffer between flips. */
static volatile LONG s_accum_display_clear_seen = 0;

/* FRAME_BUDGET=1: how much geometry a guest frame actually asks for, versus
 * what the per-frame vertex buffer can hold. A frame that overflows is silently
 * truncated -- the tail of the scene never renders -- which reads as "the object
 * is missing" rather than "the batch is too small". */

static int blink_dbg(void)
{
    static int v = -1;
    if (v < 0) v = getenv("CELLMARK_BLINKDBG") ? 1 : 0;
    return v;
}

static void d3d12_present(void* ud, u32 buffer_id)
{
    (void)ud;
    s_present_display_id = buffer_id < CELL_GCM_MAX_DISPLAY_BUFFER_NUM
                         ? buffer_id : cellGcmGetCurrentDisplayBufferId();

    if (blink_dbg())
        printf("[PRESENT] draws=%u clears_since_last=%u\n",
               s_d3d.draw_count, s_dbg_clears_since_present);
    s_dbg_clears_since_present = 0;

    /* An accumulated batch with draws but NONE targeting a display buffer is
     * offscreen pass work only (demosaic flips once per effect pass): showing
     * it would strobe the bare backbuffer clear. Keep accumulating -- the
     * composite draw that targets the display presents the whole chain, in
     * order, in one command list. Empty batches still present (boot/idle). */
    int flip_has_display = (s_d3d.draw_count == 0);
    for (u32 _i = 0; _i < s_d3d.draw_count && _i < MAX_DRAWS; _i++)
        if (is_display_target_key(s_d3d.draws[_i].rt_off)) {
            flip_has_display = 1;
            break;
        }

    s_d3d12_stats.flips++;
    const CellGcmDisplayInfo* display = cellGcmGetDisplayBufferByFlipIndex(buffer_id);
    s_d3d12_stats.last_display_offset = display ? display->offset : 0;
    if (s_d3d.initialized && flip_has_display) s_d3d12_stats.flips_rendered++;
    else if (s_d3d.initialized && s_d3d.draw_count) s_d3d12_stats.flips_deferred++;

    if (s_d3d.initialized && flip_has_display)
        render_frame();
    d3d12_print_stats();

    /* FPS tracking */
    ULONGLONG now = GetTickCount64();
    if (now - s_d3d.last_fps_time >= 1000) {
        static u64 s_last_fps_frame = 0;
        s_d3d.fps = (u32)(s_d3d.frame_count - s_last_fps_frame);
        s_last_fps_frame = s_d3d.frame_count;
        s_d3d.last_fps_time = now;
    }
}

static void d3d12_clear(void* ud, u32 flags, u32 color, float depth, u8 stencil)
{
    (void)ud;

    /* Convert RSX ARGB u32 to float[4] RGBA */
    float cc[4];
    cc[0] = ((color >> 16) & 0xFF) / 255.0f; /* R */
    cc[1] = ((color >> 8) & 0xFF) / 255.0f;  /* G */
    cc[2] = (color & 0xFF) / 255.0f;          /* B */
    cc[3] = ((color >> 24) & 0xFF) / 255.0f;  /* A */
    /* CLEAR_RGB=r,g,b: override the guest clear colour. Black holes (geometry
     * that never rasterized) and black pixels a shader really wrote look
     * identical against a black clear; this separates them in one run. */
    { static int _cinit = 0; static float _co[3]; static int _con = 0;
      if (!_cinit) { _cinit = 1; const char* e = getenv("CLEAR_RGB");
          if (e && sscanf(e, "%f,%f,%f", &_co[0], &_co[1], &_co[2]) == 3) _con = 1; }
      if (_con && (flags & RSX_CLEAR_COLOR_MASK)) {
          cc[0] = _co[0]; cc[1] = _co[1]; cc[2] = _co[2]; } }

    u32 rt_w = 0, rt_h = 0, mrt[3] = {0, 0, 0};
    u32 rt = current_rt_off(&rt_w, &rt_h, mrt);
    const int rt_none = s_d3d.current_rsx_state &&
        s_d3d.current_rsx_state->color_target == CELL_GCM_SURFACE_TARGET_NONE;
    if (rt_none) flags &= ~RSX_CLEAR_COLOR_MASK;

    extern unsigned cellGcm_flip_request_count(void);
    static int accum_frame = -1;
    if (accum_frame < 0) {
        const char* value = getenv("RSX_ACCUM_FRAME");
        accum_frame = value && atoi(value) != 0;
    }
    const int flip_driven = cellGcm_flip_request_count() != 0;
    u32 recorded_clear_flags = 0;

    /* An OFFSCREEN clear is just an ordered op in the current frame's pass
     * chain (demosaic clears each effect pass's surface) -- record it, don't
     * touch the frame boundary. So is a clear with no colour bits, wherever
     * it lands: God of War HD clears depth (mask 0x1/0x3) between compositing
     * its scene into the final surface and drawing the HUD over it, and
     * clearing colour there wiped the scene -- the menu came out black. It is
     * not a frame boundary either. */
    if (rt != 0 || !(flags & RSX_CLEAR_COLOR_MASK) || flip_driven || accum_frame) {
        if (d3d12_draw_record_available()) {
            D3D12DrawRecord* dr = &s_d3d.draws[s_d3d.draw_count++];
            memset(dr, 0, sizeof(*dr));
            dr->is_vp = 1; dr->is_clear = 1; dr->tex_slot = -1;
            dr->rt_off = rt; memcpy(dr->rt_mrt, mrt, sizeof(mrt));
            dr->rt_none = rt_none;
            dr->rt_w = rt_w; dr->rt_h = rt_h;
            dr->rt_fmt = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->surface_format : 0;
            dr_capture_zeta(dr, s_d3d.current_rsx_state);
            dr->clear_flags = flags;
            recorded_clear_flags = dr->clear_flags;
            dr->clear_depth = depth;
            dr->clear_stencil = stencil;
            memcpy(dr->cc, cc, sizeof(cc));
            dr->clear_mask = flags;
            dr->clear_z = depth;
            dr->clear_s = stencil;
        }
        if (rt != 0) return;
    }

    if (flags & RSX_CLEAR_COLOR_MASK)
        memcpy(s_d3d.clear_color, cc, sizeof(cc));
    s_d3d.clear_flags = flags & ~recorded_clear_flags;
    s_d3d.clear_depth = depth;
    s_d3d.clear_stencil = stencil;

    /* This mode deliberately waits for the guest flip. Keep all clear phases
     * ordered above and never let the heuristic split the batch at a clear. */
    if (accum_frame) {
        s_dbg_clears_since_present++;
        return;
    }

    /* Flip-driven titles are presented by the runner after the FIFO drain
     * reaches the flip marker. The clear is only a command-stream event; it is
     * not a safe frame boundary because later draws in this flip may still be
     * pending. Leave the ordered clear and draws queued for that completed flip. */
    if (flip_driven) {
        s_dbg_clears_since_present++;
        return;
    }

    /* A DISPLAY clear marks the start of a new visible frame. If a completed
     * frame is still accumulated (the drain gulped across a frame boundary --
     * guaranteed at the FIFO ring wrap, where rest-of-frame-N + clear-N+1
     * arrive in one batch), PRESENT it now instead of discarding it. Only a
     * batch that actually contains DISPLAY draws is a completed frame: a
     * batch of offscreen pass work (render-to-texture) must keep accumulating
     * until its composite draw arrives, or the screen strobes intermediates. */
    int have_display_draws = 0;
    for (u32 i = 0; i < s_d3d.draw_count && i < MAX_DRAWS; i++)
        if (!s_d3d.draws[i].is_clear && is_display_target_key(s_d3d.draws[i].rt_off)) {
            have_display_draws = 1;
            break;
        }
    if (!have_display_draws)
        return;   /* keep accumulating the in-progress frame */

    /* A title that DOUBLE-BUFFERS the display (DeferredShading clears both
     * 0x0 and 0x440000 per frame, plus a HUD pass) issues several display
     * clears per real frame -- treating each as a boundary presented the
     * 1-2-draw intermediates, strobing black between the full 146-draw
     * frames. Anchor the boundary to the FLIP instead: only present once a
     * cellGcmSetFlip has landed since the last present. Titles that flip once
     * per frame (wave/cellmark/gcmcube) are unaffected -- their single
     * display clear still follows their single flip. Fall back to the old
     * clear-only heuristic for titles that never flip (fc stays 0). */
    static u32 s_frame_draws_max = 0;   /* running max frame size (typical full frame) */
    if (s_d3d.draw_count > s_frame_draws_max) s_frame_draws_max = s_d3d.draw_count;
    {
        extern unsigned cellGcm_flip_request_count(void);
        static unsigned s_last_present_flip = 0;
        unsigned fc = cellGcm_flip_request_count();
        if (fc != 0) {
            if (fc == s_last_present_flip)
                return;   /* no flip since last present -> not a real boundary */
            /* A double-buffered title issues several display clears per flip
             * (DeferredShading: clear back-buffer, HUD pass, ...). They can
             * arrive in either order, so a small batch here may be an
             * intermediate that precedes the full frame's clear. Don't present
             * (or consume the flip) until the batch is a substantial fraction
             * of a full frame -- keep accumulating so the complete frame lands
             * in one present. Gauged against the running MAX frame size (not
             * the last, which a leaked tiny batch would poison), so it
             * self-scales per title with no fixed threshold. */
            if (s_frame_draws_max > 16 && s_d3d.draw_count < s_frame_draws_max / 4)
                return;   /* intermediate: keep accumulating, flip stays pending */
            s_last_present_flip = fc;
        }
    }

    if (s_d3d.initialized) {
        if (blink_dbg())
            printf("[CLEAR] presenting %u accumulated draws at frame boundary\n",
                   s_d3d.draw_count);
        render_frame();
        s_clear_presents++;
    }
    s_dbg_clears_since_present++;
    s_d3d.draw_count   = 0;
    s_d3d.vb_offset    = 0;
    s_d3d.vp_vb_offset = 0;
    rsx_raw_upload_reset(&s_raw_upload_cache);
    s_d3d.vp_vb_vertex_count = 0;
}

static void d3d12_set_render_target(void* ud, const rsx_state* state)
{
    (void)ud;
    s_d3d.current_rsx_state = state;
    /* Log only the first few; set_render_target is called every frame and
     * floods the log otherwise. */
    static int s_count = 0;
    if (s_count < 5) {
        printf("[D3D12] set_render_target(%ux%u)\n",
               state->surface_clip_w, state->surface_clip_h);
        s_count++;
    }
}

static void d3d12_set_viewport(void* ud, const rsx_state* state)
{
    (void)ud;
    /* TODO: update D3D12 viewport from RSX state */
    (void)state;
}

/* Our host vertex layout for the fallback path: position (xyz) + color (rgba)
 * + texcoord (uv), 36 bytes. (The real VP path feeds raw float4 attrib0.) */
typedef struct { float x, y, z; float r, g, b, a; float u, v; } BasicVertex;

/* Vertex conversion is CPU-only once a draw's RSX state is fixed. Large draws
 * can be decoded in parallel into disjoint ranges of the mapped upload buffer;
 * draw submission and all D3D12 calls remain on the render thread. */
#define RSX_VERTEX_PARALLEL_MIN 512u
#define RSX_VERTEX_MIN_CHUNK 128u
typedef void (*RSXVertexWorkFn)(void*, u32);
static SRWLOCK s_vtx_work_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_vtx_work_ready = CONDITION_VARIABLE_INIT;
static CONDITION_VARIABLE s_vtx_work_done = CONDITION_VARIABLE_INIT;
static HANDLE* s_vtx_workers;
static DWORD s_vtx_worker_count;
static LONG s_vtx_work_generation;
static LONG s_vtx_work_pending;
static LONG s_vtx_work_stopping;
static volatile LONG s_vtx_work_next;
static DWORD s_vtx_work_active_workers;
static u32 s_vtx_work_chunk_size;
static u32 s_vtx_work_count;
static RSXVertexWorkFn s_vtx_work_fn;
static void* s_vtx_work_context;

static void rsx_vertex_run_range(RSXVertexWorkFn fn, void* context,
                                 u32 begin, u32 end)
{
    for (u32 i = begin; i < end; ++i)
        fn(context, i);
}

static void rsx_vertex_run_queued(RSXVertexWorkFn fn, void* context,
                                 u32 count, u32 chunk_size)
{
    for (;;) {
        const LONG begin = InterlockedExchangeAdd(&s_vtx_work_next,
                                                  (LONG)chunk_size);
        if (begin < 0 || (u32)begin >= count) break;
        const u32 end = (u32)begin + chunk_size < count
            ? (u32)begin + chunk_size : count;
        for (u32 i = (u32)begin; i < end; ++i)
            fn(context, i);
    }
}

static DWORD WINAPI rsx_vertex_worker(void* parameter)
{
    const DWORD worker_id = (DWORD)(uintptr_t)parameter;
    LONG seen_generation = 0;
    for (;;) {
        AcquireSRWLockExclusive(&s_vtx_work_lock);
        while (!s_vtx_work_stopping && s_vtx_work_generation == seen_generation)
            SleepConditionVariableSRW(&s_vtx_work_ready, &s_vtx_work_lock, INFINITE, 0);
        if (s_vtx_work_stopping) {
            ReleaseSRWLockExclusive(&s_vtx_work_lock);
            return 0;
        }

        seen_generation = s_vtx_work_generation;
        const u32 count = s_vtx_work_count;
        const DWORD workers = s_vtx_work_active_workers;
        const u32 chunk_size = s_vtx_work_chunk_size;
        if (worker_id >= workers) {
            ReleaseSRWLockExclusive(&s_vtx_work_lock);
            continue;
        }
        RSXVertexWorkFn fn = s_vtx_work_fn;
        void* context = s_vtx_work_context;
        const u32 begin = worker_id * chunk_size;
        const u32 end = begin < count && chunk_size < count - begin
            ? begin + chunk_size : count;
        ReleaseSRWLockExclusive(&s_vtx_work_lock);

        rsx_vertex_run_range(fn, context, begin, end);
        rsx_vertex_run_queued(fn, context, count, chunk_size);

        AcquireSRWLockExclusive(&s_vtx_work_lock);
        if (InterlockedDecrement(&s_vtx_work_pending) == 0)
            WakeAllConditionVariable(&s_vtx_work_done);
        ReleaseSRWLockExclusive(&s_vtx_work_lock);
    }
}

static void rsx_vertex_workers_start(void)
{
    if (s_vtx_worker_count) return;
    const char* configured = getenv("RSX_VERTEX_THREADS");
    long requested = -1;  /* unset/invalid: retain the existing auto policy */
    if (configured) {
        char* end = NULL;
        const long parsed = strtol(configured, &end, 10);
        if (end != configured && *end == '\0' && parsed >= 0)
            requested = parsed;
    }
    if (requested == 0) return;
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (info.dwNumberOfProcessors < 2) return;
    DWORD wanted = info.dwNumberOfProcessors - 1;
    if (requested > 0 && (unsigned long)requested < wanted)
        wanted = (DWORD)requested;
    HANDLE* workers = (HANDLE*)calloc(wanted, sizeof(*workers));
    if (!workers) return;

    AcquireSRWLockExclusive(&s_vtx_work_lock);
    s_vtx_work_generation = 0;
    s_vtx_work_pending = 0;
    s_vtx_work_stopping = 0;
    s_vtx_work_next = 0;
    s_vtx_work_active_workers = 0;
    s_vtx_work_chunk_size = 0;
    s_vtx_work_count = 0;
    s_vtx_work_fn = NULL;
    s_vtx_work_context = NULL;
    ReleaseSRWLockExclusive(&s_vtx_work_lock);

    for (DWORD i = 0; i < wanted; ++i) {
        HANDLE thread = CreateThread(NULL, 0, rsx_vertex_worker,
                                     (void*)(uintptr_t)i, 0, NULL);
        if (!thread) break;
        workers[s_vtx_worker_count++] = thread;
    }
    s_vtx_workers = workers;
    if (!s_vtx_worker_count) {
        free(s_vtx_workers);
        s_vtx_workers = NULL;
    }
}

static void rsx_vertex_workers_stop(void)
{
    const DWORD count = s_vtx_worker_count;
    if (!count) return;
    AcquireSRWLockExclusive(&s_vtx_work_lock);
    s_vtx_work_stopping = 1;
    WakeAllConditionVariable(&s_vtx_work_ready);
    ReleaseSRWLockExclusive(&s_vtx_work_lock);
    for (DWORD first = 0; first < count; first += MAXIMUM_WAIT_OBJECTS) {
        DWORD batch = count - first;
        if (batch > MAXIMUM_WAIT_OBJECTS) batch = MAXIMUM_WAIT_OBJECTS;
        WaitForMultipleObjects(batch, s_vtx_workers + first, TRUE, INFINITE);
    }
    for (DWORD i = 0; i < count; ++i) {
        CloseHandle(s_vtx_workers[i]);
    }
    free(s_vtx_workers);
    s_vtx_workers = NULL;
    s_vtx_worker_count = 0;
}

/* Returns nonzero when dispatched to the persistent pool. Small work stays on
 * the caller to avoid waking threads for draws where synchronization costs
 * more than vertex decoding. */
static int rsx_vertex_parallel_for(RSXVertexWorkFn fn, void* context, u32 count)
{
    const DWORD workers = s_vtx_worker_count;
    if (!workers || count < RSX_VERTEX_PARALLEL_MIN) return 0;

    DWORD participants = count / RSX_VERTEX_MIN_CHUNK;
    if (participants > workers + 1u) participants = workers + 1u;
    if (participants < 2u) return 0;
    const DWORD active_workers = participants - 1u;
    u32 chunk_size = count / (participants * 4u);
    if (chunk_size < 64u) chunk_size = 64u;
    if (chunk_size > 512u) chunk_size = 512u;
    ++s_perf_vtx_pool_calls;
    s_perf_vtx_pool_items += count;

    AcquireSRWLockExclusive(&s_vtx_work_lock);
    s_vtx_work_fn = fn;
    s_vtx_work_context = context;
    s_vtx_work_count = count;
    s_vtx_work_chunk_size = chunk_size;
    InterlockedExchange(&s_vtx_work_next, (LONG)(participants * chunk_size));
    s_vtx_work_active_workers = active_workers;
    InterlockedExchange(&s_vtx_work_pending, (LONG)active_workers);
    ++s_vtx_work_generation;
    WakeAllConditionVariable(&s_vtx_work_ready);
    ReleaseSRWLockExclusive(&s_vtx_work_lock);

    const u32 begin = active_workers * chunk_size;
    const u32 end = begin < count && chunk_size < count - begin
        ? begin + chunk_size : count;
    rsx_vertex_run_range(fn, context, begin, end);
    rsx_vertex_run_queued(fn, context, count, chunk_size);

    AcquireSRWLockExclusive(&s_vtx_work_lock);
    while (s_vtx_work_pending)
        SleepConditionVariableSRW(&s_vtx_work_done, &s_vtx_work_lock, INFINITE, 0);
    s_vtx_work_fn = NULL;
    s_vtx_work_context = NULL;
    s_vtx_work_count = 0;
    ReleaseSRWLockExclusive(&s_vtx_work_lock);
    return 1;
}


/* Read one RSX vertex (by absolute vertex index) from guest memory into our
 * host layout. Position is attrib 0 (float3+), color is attrib 3 (ubyte4 or
 * float4); missing attribs default to opaque white. RSX stores each 32-bit
 * component big-endian, so every lane is byte-swapped. */
static void read_rsx_vertex(const rsx_state* state, u32 vindex, BasicVertex* out)
{
    extern uint8_t* vm_base;
    extern u32 cellGcmResolveOffset(u32);

    out->x = out->y = out->z = 0.0f;
    out->r = out->g = out->b = out->a = 1.0f;
    out->u = out->v = 0.0f;
    if (!state || !vm_base) return;

    const rsx_vertex_attrib* pos = &state->vertex_attribs[0];
    if (pos->enabled && pos->type == 2 /* float */ && pos->size >= 2) {
        u8* src = vm_base + cellGcmResolveOffset(pos->offset + vindex * pos->stride);
        out->x = rsx_rd_bef(src);
        out->y = rsx_rd_bef(src + 4);
        if (pos->size >= 3) out->z = rsx_rd_bef(src + 8);

        /* dbgfont (and similar 2D overlays) store screen positions biased far
         * outside clip space; their vertex program folds them back to clip
         * space using transform constants we don't execute. When a position is
         * well outside NDC, recover the fractional part as a normalized [0,1]
         * screen coord and map it to NDC (screen Y-down -> clip Y-up). In this
         * layout the attribute is [posX, posY, U, V] (size 4), so the last two
         * components are the atlas texcoords, not depth.
         * TODO: execute the real vertex-program / RSX viewport transform so
         * this is not needed. */
        if (out->x > 2.0f || out->x < -2.0f || out->y > 2.0f || out->y < -2.0f) {
            float sx = out->x - floorf(out->x);
            float sy = out->y - floorf(out->y);
            out->x = sx * 2.0f - 1.0f;
            out->y = 1.0f - sy * 2.0f;
            out->z = 0.0f;
            if (pos->size >= 4) {
                out->u = rsx_rd_bef(src + 8);   /* U */
                out->v = rsx_rd_bef(src + 12);  /* V */
            }
        }
    }

    const rsx_vertex_attrib* col = &state->vertex_attribs[3];
    if (col->enabled && col->size >= 3) {
        u8* src = vm_base + cellGcmResolveOffset(col->offset + vindex * col->stride);
        if (col->type == 4 /* ubyte */) {
            out->r = src[0] / 255.0f;
            out->g = src[1] / 255.0f;
            out->b = src[2] / 255.0f;
            out->a = (col->size >= 4) ? src[3] / 255.0f : 1.0f;
        } else if (col->type == 2 /* float */) {
            u32 fr, fg, fb, fa;
            memcpy(&fr, src,     4); fr = ((fr>>24)&0xFF)|((fr>>8)&0xFF00)|((fr<<8)&0xFF0000)|((fr<<24)&0xFF000000);
            memcpy(&fg, src + 4, 4); fg = ((fg>>24)&0xFF)|((fg>>8)&0xFF00)|((fg<<8)&0xFF0000)|((fg<<24)&0xFF000000);
            memcpy(&fb, src + 8, 4); fb = ((fb>>24)&0xFF)|((fb>>8)&0xFF00)|((fb<<8)&0xFF0000)|((fb<<24)&0xFF000000);
            memcpy(&out->r, &fr, 4); memcpy(&out->g, &fg, 4); memcpy(&out->b, &fb, 4);
            if (col->size >= 4) {
                memcpy(&fa, src + 12, 4); fa = ((fa>>24)&0xFF)|((fa>>8)&0xFF00)|((fa<<8)&0xFF0000)|((fa<<24)&0xFF000000);
                memcpy(&out->a, &fa, 4);
            }
        }
    }
}

typedef struct {
    const rsx_state* state;
    u32 first;
    BasicVertex* out;
} RSXBasicVertexTask;

static void rsx_decode_basic_vertex(void* opaque, u32 index)
{
    RSXBasicVertexTask* task = (RSXBasicVertexTask*)opaque;
    read_rsx_vertex(task->state, task->first + index, &task->out[index]);
}

typedef struct {
    const rsx_state* state;
    u32 first;
    BasicVertex* out;
} RSXQuadVertexTask;

static void rsx_decode_quad(void* opaque, u32 index)
{
    RSXQuadVertexTask* task = (RSXQuadVertexTask*)opaque;
    BasicVertex corner[4];
    const u32 first = task->first + index * 4;
    for (u32 k = 0; k < 4; ++k)
        read_rsx_vertex(task->state, first + k, &corner[k]);
    BasicVertex* out = &task->out[index * 6];
    out[0] = corner[0]; out[1] = corner[1]; out[2] = corner[2];
    out[3] = corner[0]; out[4] = corner[2]; out[5] = corner[3];
}

/* Upload `count` sequential vertices [first, first+count). Returns the count
 * actually written (clamped to the remaining per-frame buffer). */
static u32 upload_vertices_from_rsx(u32 first, u32 count)
{
    BasicVertex* verts = (BasicVertex*)(d3d12_legacy_vertex_slot_base() + s_d3d.vb_offset);
    u32 max_verts = (MAX_VERTICES * VERTEX_STRIDE - s_d3d.vb_offset) / sizeof(BasicVertex);
    if (count > max_verts) {
        s_drop_vertices += count - max_verts; s_drop_draws++;
        count = max_verts;
    }
    const rsx_state* state = s_d3d.current_rsx_state;
    RSXBasicVertexTask task = { state, first, verts };
    if (!rsx_vertex_parallel_for(rsx_decode_basic_vertex, &task, count))
        for (u32 i = 0; i < count; i++)
            read_rsx_vertex(state, first + i, &verts[i]);
    s_d3d.vb_offset += count * sizeof(BasicVertex);
    return count;
}

/* Upload RSX QUADS (prim 8) as a triangle list: each 4-vertex quad v0..v3
 * (perimeter winding) splits into triangles (v0,v1,v2) and (v0,v2,v3).
 * D3D12 has no quad topology, so this expansion is how quads render at all.
 * Returns the number of triangle-list vertices emitted (6 per quad). */
static u32 upload_quads_from_rsx(u32 first, u32 count)
{
    const rsx_state* state = s_d3d.current_rsx_state;
    u32 quads = count / 4;
    u32 max_verts = (MAX_VERTICES * VERTEX_STRIDE - s_d3d.vb_offset) / sizeof(BasicVertex);
    if (quads * 6 > max_verts) {
        const u32 kept = max_verts / 6;
        s_drop_vertices += quads * 6 - kept * 6; s_drop_draws++;
        quads = kept;
    }
    BasicVertex* verts = (BasicVertex*)(d3d12_legacy_vertex_slot_base() + s_d3d.vb_offset);
    RSXQuadVertexTask task = { state, first, verts };
    u32 o = quads * 6;
    if (!rsx_vertex_parallel_for(rsx_decode_quad, &task, quads)) {
        o = 0;
        for (u32 q = 0; q < quads; q++) {
            BasicVertex c[4];
            for (u32 k = 0; k < 4; k++)
                read_rsx_vertex(state, first + q * 4 + k, &c[k]);
            verts[o++] = c[0]; verts[o++] = c[1]; verts[o++] = c[2];
            verts[o++] = c[0]; verts[o++] = c[2]; verts[o++] = c[3];
        }
    }
    s_d3d.vb_offset += o * sizeof(BasicVertex);
    return o;
}

/* Expand RSX quads to a triangle list and upload converted VP inputs. The mask
 * is proven by the shader analyzer; uncertain programs retain all 16 float4
 * slots. Enabled attributes are fetched using their RSX types, while disabled
 * slots use the guest's current constant value. */
typedef struct { float v[4]; } VPSlot;

#include "rsx_vertex_decode_cache.h"
static _Thread_local rsx_vertex_decode_cache s_vp_decode_cache;
static u64 s_vp_decode_epoch;

static u64 vp_next_decode_epoch(void)
{
    if (++s_vp_decode_epoch == 0) ++s_vp_decode_epoch;
    return s_vp_decode_epoch;
}

static u32 vp_mask_count(u32 mask)
{
    u32 count = 0;
    for (mask &= 0xFFFFu; mask; mask >>= 1) count += mask & 1u;
    return count;
}

static u32 vp_mask_slot(u32 mask, u32 attr)
{
    if (attr >= 16 || !(mask & (1u << attr))) return 0;
    return vp_mask_count(mask & ((1u << attr) - 1u));
}

static rsx_vertex_fetch_prepared s_vertex_fetch_prepared;

static void vp_select_upload_layout(int vs_idx, const rsx_state* state)
{
    static int compact_enabled = -1;
    if (compact_enabled < 0)
        compact_enabled = !getenv("FP_OFF") && !getenv("VP_BYPASS");
    u32 mask = 0xFFFFu;
    u32 stride = VP_VERT_STRIDE;
    /* With no guest FP there is no per-VP guest PSO to carry the compact input
     * layout, so retain the legacy 16-slot stream for the generic fallback. */
    if (compact_enabled && state && state->shader_program &&
        vs_idx >= 0 && vs_idx < s_d3d.vp_vs_n) {
        const VPVSEntry* vs = &s_d3d.vp_vs[vs_idx];
        const u32 count = vp_mask_count(vs->input_mask);
        if (vs->input_mask && count && vs->input_stride == count * sizeof(VPSlot)) {
            mask = vs->input_mask;
            stride = vs->input_stride;
        }
    }
    s_d3d.vp_upload_mask = mask;
    s_d3d.vp_upload_count = vp_mask_count(mask);
    s_d3d.vp_upload_stride = stride;
}


void rsx_vtx_pos_dbg(const rsx_state* state, const float* v, u32 n);

/* Fill the analyzed input attributes for one vertex in packed order. Unknown
 * or unsupported VP analysis selects all 16 slots, preserving the old layout. */
static void read_vp_vertex_impl(const rsx_state* state, u32 vi,
                               VPSlot* out_packed, int trace_position)
{
    /* Finish conversion in normal cached CPU memory. The destination may
     * be write-combined D3D12 upload memory: publish each packed vertex only
     * once, and keep diagnostic reads off the mapped destination. */
    VPSlot decoded[16];
    u32 slot = 0;
    u32 remaining = s_d3d.vp_upload_mask & 0xFFFFu;
    while (remaining) {
        const u32 i = (u32)__builtin_ctz(remaining);
        remaining &= remaining - 1u;
        if (state->vertex_attribs[i].enabled)
            rsx_fetch_attrib_prepared(&s_vertex_fetch_prepared, (int)i, vi, decoded[slot].v);
        else
            memcpy(&decoded[slot], state->vertex_data4f[i], sizeof(decoded[slot]));
        if (trace_position && i == 0) {
            const rsx_vertex_attrib* a = &state->vertex_attribs[0];
            u32 n = a->size ? a->size : 4; if (n > 4) n = 4;
            rsx_vtx_pos_dbg(state, decoded[slot].v, n);
        }
        slot++;
    }
    memcpy(out_packed, decoded, slot * sizeof(*out_packed));
}

static void read_vp_vertex(const rsx_state* state, u32 vi, VPSlot* out_packed)
{
    read_vp_vertex_impl(state, vi, out_packed, 1);
}

static void read_vp_vertex_cached(const rsx_state* state, u32 vi,
                                  VPSlot* out, u64 epoch, int trace_position)
{
    VPSlot* cached = (VPSlot*)rsx_vertex_decode_cache_find(
        &s_vp_decode_cache, epoch, vi);
    if (!cached) {
        cached = (VPSlot*)rsx_vertex_decode_cache_store(&s_vp_decode_cache, epoch, vi);
        read_vp_vertex_impl(state, vi, cached, trace_position);
    } else if (trace_position && (s_d3d.vp_upload_mask & 1u)) {
        const rsx_vertex_attrib* attr = &state->vertex_attribs[0];
        u32 n = attr->size ? attr->size : 4u;
        if (n > 4u) n = 4u;
        rsx_vtx_pos_dbg(state, cached[0].v, n);
    }
    memcpy(out, cached, s_d3d.vp_upload_count * sizeof(*out));
}

static u32 resolve_indexed_vertex(const rsx_state* st, const u32* immediate,
                                  u32 index);

typedef struct {
    const rsx_state* state;
    u32 first;
    const u32* immediate;
    VPSlot* out;
    u32 slots_per_vertex;
    int indexed;
    u64 epoch;
} RSXVPVertexTask;

static void rsx_decode_vp_vertex(void* opaque, u32 index)
{
    RSXVPVertexTask* task = (RSXVPVertexTask*)opaque;
    const u32 vertex = task->indexed
        ? resolve_indexed_vertex(task->state, task->immediate, task->first + index)
        : task->first + index;
    VPSlot* out = &task->out[index * task->slots_per_vertex];
    if (task->indexed)
        read_vp_vertex_cached(task->state, vertex, out, task->epoch, 0);
    else
        read_vp_vertex_impl(task->state, vertex, out, 0);
}

/* Preserve the first-vertex diagnostic on the render thread, then decode the
 * remaining vertices into disjoint mapped-buffer ranges on the worker pool. */
static void rsx_decode_vp_vertices(const rsx_state* state, u32 first, u32 count,
                                   const u32* immediate, int indexed, VPSlot* out)
{
    const u32 slots = s_d3d.vp_upload_count;
    const u64 epoch = indexed ? vp_next_decode_epoch() : 0;
    if (count > RSX_VERTEX_PARALLEL_MIN && s_vtx_worker_count) {
        const u32 vertex = indexed
            ? resolve_indexed_vertex(state, immediate, first) : first;
        read_vp_vertex(state, vertex, out);
        RSXVPVertexTask task = {
            state, first + 1, immediate, out + slots, slots, indexed, epoch
        };
        (void)rsx_vertex_parallel_for(rsx_decode_vp_vertex, &task, count - 1);
        return;
    }
    for (u32 k = 0; k < count; ++k) {
        const u32 vertex = indexed
            ? resolve_indexed_vertex(state, immediate, first + k) : first + k;
        if (indexed)
            read_vp_vertex_cached(state, vertex, &out[k * slots], epoch, 1);
        else
            read_vp_vertex(state, vertex, &out[k * slots]);
    }
}

/* VTX_POS=<N>: print the first fetched position of the first N draws. A guest
 * vertex array that resolves to the wrong memory reads as garbage/denormals, and
 * that is indistinguishable from "the shader is wrong" without seeing the input. */
void rsx_vtx_pos_dbg(const rsx_state* state, const float* v, u32 n)
{
    static int cap = -1, seen = 0;
    if (cap < 0) { const char* e = getenv("VTX_POS"); cap = e ? atoi(e) : 0; }
    if (!cap || seen >= cap) return;
    seen++;
    const rsx_vertex_attrib* a = &state->vertex_attribs[0];
    fprintf(stderr, "[VTXPOS] a0 off=0x%X stride=%u size=%u type=%u -> (%g, %g, %g, %g)\n",
            a->offset, a->stride, a->size, a->type,
            v[0], v[1], v[2], v[3]);   /* all four: masking by `size` hid whether w survived */
}

static void vp_attrs_dbg(const rsx_state* state)
{
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("VP_ATTRS") != NULL;
    if (!enabled) return;
    /* VP_ATTRS_FP=<hex shader_program>: dump only the draws that use one
     * fragment program. Without it the first six draws are whatever the frame
     * happens to start with, which is never the mesh being investigated. */
    { static const char* e = (const char*)1; static u32 want = 0;
      if (e == (const char*)1) { e = getenv("VP_ATTRS_FP");
          want = e ? (u32)strtoul(e, NULL, 0) : 0; }
      if (want && (!state || state->shader_program != want)) return; }
    /* Dedupe by the set of enabled attributes rather than printing the first
     * few draws: one fragment program can be used by meshes with different
     * vertex layouts, and a mesh missing an attribute takes the constant
     * register instead -- which is exactly the case worth seeing. */
    { u32 mask = 0;
      for (int i = 0; i < 16; i++) if (state->vertex_attribs[i].enabled) mask |= 1u << i;
      static u32 seen[16]; static int ns = 0;
      for (int i = 0; i < ns; i++) if (seen[i] == mask) return;
      if (ns >= 16) return;
      seen[ns++] = mask; }
    fprintf(stderr, "[VPATTR] --- fp=0x%X ---%c", state->shader_program, 10);
    fprintf(stderr, "[VPATTR] divider_op=0x%08X\n", state->frequency_divider_op);
    for (int i = 0; i < 16; i++) {
        const rsx_vertex_attrib* a = &state->vertex_attribs[i];
        if (a->enabled) fprintf(stderr, "[VPATTR] a%d off=0x%X stride=%u size=%u type=%u freq=%u fmt=0x%08X\n",
                                i, a->offset, a->stride, a->size, a->type, a->frequency, a->format);
    }
    /* VP_ATTRS_DUMP=<attr>:<n>: read n entries of that attribute straight out of
     * guest memory. Answers "did the guest write these values, or did our fetch
     * mangle them" without inferring it from the rendered image. */
    { const char* e = getenv("VP_ATTRS_DUMP"); if (!e) return;
      int ai = 0, cnt = 8; sscanf(e, "%d:%d", &ai, &cnt);
      const rsx_vertex_attrib* a = &state->vertex_attribs[ai];
      if (!a->enabled || !a->stride) return;
      extern uint8_t* vm_base;
      extern u32 cellGcmResolveLocated(int, u32);
      u32 nbad = 0, nzero = 0;
      for (int v = 0; v < cnt; v++) {
          u32 aoff = (a->offset & 0x7FFFFFFFu) + (u32)v * a->stride;
          const u8* q = vm_base + ((a->offset & 0x80000000u)
                        ? cellGcmResolveLocated(0, aoff) : cellGcmResolveLocated(1, aoff));
          /* Also read the same offset through the OTHER context-DMA. An array
           * the guest put in main memory but whose offset we resolve as LOCAL
           * reads as untouched VRAM, i.e. all zeros -- indistinguishable from
           * "the producer never ran". */
          if (v == 0) {
              u32 la = cellGcmResolveLocated(1, aoff), ma = cellGcmResolveLocated(0, aoff);
              const u8* lp = vm_base + la; const u8* mp = vm_base + ma;
              const u8* rp = vm_base + (a->offset & 0x7FFFFFFFu);   /* raw, unresolved */
              fprintf(stderr, "[VPDUMP] a%d off=0x%X  LOCAL@0x%X=(%g,%g,%g)  MAIN@0x%X=(%g,%g,%g)"
                              "  RAW@0x%X=(%g,%g,%g)%c",
                      ai, aoff, la, rsx_rd_bef(lp), rsx_rd_bef(lp+4), rsx_rd_bef(lp+8),
                      ma, rsx_rd_bef(mp), rsx_rd_bef(mp+4), rsx_rd_bef(mp+8),
                      (a->offset & 0x7FFFFFFFu), rsx_rd_bef(rp), rsx_rd_bef(rp+4), rsx_rd_bef(rp+8), 10);
          }
          u32 nc = a->size ? a->size : 3; if (nc > 3) nc = 3;
          float f[3] = {0,0,0};
          for (u32 k = 0; k < nc; k++) f[k] = rsx_rd_bef(q + k * 4);
          if (v < 12)
              fprintf(stderr, "[VPDUMP] a%d[%d] = (%g, %g, %g)%c", ai, v, f[0], f[1], f[2], 10);
          int allz = 1; for (u32 k = 0; k < nc; k++) if (f[k] != 0.f) allz = 0;
          if (allz) nzero++;
          else if (f[2] < -0.9f && f[0] > -0.1f && f[0] < 0.1f) nbad++;
      }
      fprintf(stderr, "[VPDUMP] a%d over %d verts: %u zero, %u near (0,0,-1)%c",
              ai, cnt, nzero, nbad, 10);
      /* How many entries differ from the first? An all-identical position array
       * is a degenerate mesh: it rasterizes nothing, which looks the same as
       * "the draw never happened". */
      { float f0[3]; u32 o0 = (a->offset & 0x7FFFFFFFu);
        const u8* q0 = vm_base + ((a->offset & 0x80000000u)
                       ? cellGcmResolveLocated(0, o0) : cellGcmResolveLocated(1, o0));
        for (int k = 0; k < 3; k++) f0[k] = rsx_rd_bef(q0 + k * 4);
        u32 diff = 0;
        for (int v = 1; v < cnt; v++) {
            u32 ao = o0 + (u32)v * a->stride;
            const u8* q = vm_base + ((a->offset & 0x80000000u)
                          ? cellGcmResolveLocated(0, ao) : cellGcmResolveLocated(1, ao));
            for (int k = 0; k < 3; k++)
                if (rsx_rd_bef(q + k * 4) != f0[k]) { diff++; break; }
        }
        fprintf(stderr, "[VPDUMP] a%d: %u/%d entries differ from entry 0 (%g,%g,%g)%c",
                ai, diff, cnt, f0[0], f0[1], f0[2], 10);
        /* Extent: an isosurface with little fluid is a speck, and a speck that
         * rasterizes nothing is correct, not a bug. */
        { float lo[3] = {1e30f,1e30f,1e30f}, hi[3] = {-1e30f,-1e30f,-1e30f};
          for (int v = 0; v < cnt; v++) {
              u32 ao = o0 + (u32)v * a->stride;
              const u8* q = vm_base + ((a->offset & 0x80000000u)
                            ? cellGcmResolveLocated(0, ao) : cellGcmResolveLocated(1, ao));
              for (int k = 0; k < 3; k++) { float f = rsx_rd_bef(q + k * 4);
                  if (f < lo[k]) lo[k] = f; if (f > hi[k]) hi[k] = f; } }
          u32 nnan = 0, nfin = 0;
          for (int v = 0; v < cnt; v++) {
              u32 ao = o0 + (u32)v * a->stride;
              const u8* q = vm_base + ((a->offset & 0x80000000u)
                            ? cellGcmResolveLocated(0, ao) : cellGcmResolveLocated(1, ao));
              int bad = 0;
              for (int k = 0; k < 3; k++) { float f = rsx_rd_bef(q + k * 4);
                  if (!(f == f) || f > 1e30f || f < -1e30f) bad = 1; }
              if (bad) nnan++; else nfin++;
          }
          fprintf(stderr, "[VPDUMP] a%d extent x[%g..%g] y[%g..%g] z[%g..%g]"
                          "  finite=%u nan/inf=%u%c",
                  ai, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], nfin, nnan, 10); } } }
}

static u32 read_index_source(const rsx_state* st, const u32* immediate,
                             u32 index);

typedef struct {
    const rsx_state* state;
    const u32* immediate;
    VPSlot* out;
    u64 epoch;
    u32 first, primitive_base, slots;
    int indexed, quad, fan, trace;
} RSXVPExpandedTask;

static void rsx_decode_vp_expanded_primitive(void* opaque, u32 index)
{
    RSXVPExpandedTask* task = (RSXVPExpandedTask*)opaque;
    const u32 primitive = task->primitive_base + index;
    u32 indices[4];
    const u32 inputs = task->quad ? 4u : 3u;
    if (task->quad) {
        for (u32 k = 0; k < 4; ++k)
            indices[k] = primitive * 4u + k;
    } else {
        indices[0] = task->fan ? 0 : primitive;
        indices[1] = primitive + 1u;
        indices[2] = primitive + 2u;
        if (!task->fan && (primitive & 1u)) {
            const u32 swap = indices[1];
            indices[1] = indices[2];
            indices[2] = swap;
        }
    }
    VPSlot decoded[4][16];
    VPSlot* out = task->out + primitive * (task->quad ? 6u : 3u) * task->slots;
    for (u32 k = 0; k < inputs; ++k) {
        const u32 source = task->first + indices[k];
        const u32 vertex = !task->indexed ? source : task->quad
            ? resolve_indexed_vertex(task->state, task->immediate, source)
            : read_index_source(task->state, task->immediate, source);
        read_vp_vertex_cached(task->state, vertex,
            task->quad ? decoded[k] : out + k * task->slots,
            task->epoch, task->trace);
    }
    if (task->quad) {
        static const u32 winding[6] = {0, 1, 2, 0, 2, 3};
        for (u32 k = 0; k < 6; ++k)
            memcpy(out + k * task->slots, decoded[winding[k]],
                   task->slots * sizeof(VPSlot));
    }
}

static void rsx_decode_vp_expanded(const rsx_state* state, u32 first,
    u32 primitives, const u32* immediate, int indexed, int quad, int fan,
    VPSlot* out)
{
    if (!primitives) return;
    ++s_perf_vtx_expanded_draws;
    s_perf_vtx_expanded_primitives += primitives;
    const double start = perf_on() ? perf_now() : 0.0;
    RSXVPExpandedTask task = {state, immediate, out, vp_next_decode_epoch(),
        first, 0, s_d3d.vp_upload_count, indexed, quad, fan, 1};
    rsx_decode_vp_expanded_primitive(&task, 0);
    task.primitive_base = 1;
    if (primitives > RSX_VERTEX_PARALLEL_MIN && s_vtx_worker_count) {
        task.trace = 0;
        if (!rsx_vertex_parallel_for(rsx_decode_vp_expanded_primitive,
                                    &task, primitives - 1u)) {
            task.trace = 1;
            for (u32 i = 0; i < primitives - 1u; ++i)
                rsx_decode_vp_expanded_primitive(&task, i);
        }
    } else {
        for (u32 i = 0; i < primitives - 1u; ++i)
            rsx_decode_vp_expanded_primitive(&task, i);
    }
    if (perf_on()) {
        s_perf_vtx += perf_now() - start;
        s_perf_nverts += (u64)primitives * (quad ? 6u : 3u);
    }
}

static u32 upload_quads_vp(const rsx_state* state, u32 first, u32 count)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    vp_attrs_dbg(state);
    u32 quads = count / 4;
    u32 maxv = MAX_VERTICES - s_d3d.vp_vb_vertex_count;
    if (quads * 6 > maxv) {
        const u32 kept = maxv / 6;
        s_drop_vertices += quads * 6 - kept * 6;
        s_drop_draws++;
        quads = kept;
    }
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    if (!d3d12_setting_vtx_dump()) {
        rsx_decode_vp_expanded(state, first, quads, NULL, 0, 1, 0, out);
        const u32 emitted = quads * 6u;
        s_d3d.vp_vb_offset += emitted * s_d3d.vp_upload_stride;
        s_d3d.vp_vb_vertex_count += emitted;
        return emitted;
    }
    u32 o = 0;
    for (u32 q = 0; q < quads; q++) {
        VPSlot c[4][16];
        for (u32 k = 0; k < 4; k++)
            read_vp_vertex(state, first + q*4 + k, c[k]);
        /* quad -> two triangles (perimeter winding) */
        static const int idx[6] = {0,1,2, 0,2,3};
        for (int t = 0; t < 6; t++) {
            memcpy(&out[o * s_d3d.vp_upload_count], c[idx[t]],
                   s_d3d.vp_upload_count * sizeof(VPSlot));
            o++;
        }
        if (d3d12_setting_vtx_dump()) { static int _n=0; if (PS3_LOG_FIRST(_n, 4)) {
            FILE* f = fopen("vtx_dump.txt", _n==1 ? "w" : "a");
            if (f) { for (u32 k = 0; k < 4; k++)
                fprintf(f, "q%02d v%u a0=(%.3f,%.3f,%.3f,%.3f) a8=(%.3f,%.3f)\n", _n, k,
                    c[k][0].v[0],c[k][0].v[1],c[k][0].v[2],c[k][0].v[3],
                    c[k][vp_mask_slot(s_d3d.vp_upload_mask, 8)].v[0],
                    c[k][vp_mask_slot(s_d3d.vp_upload_mask, 8)].v[1]);
              fclose(f); } } }
    }
    s_d3d.vp_vb_offset += o * s_d3d.vp_upload_stride;
    s_d3d.vp_vb_vertex_count += o;
    return o;
}

/* Straight triangle-list upload through the VP path (gcm/cube's cube draws
 * TRIANGLES, prim 5 -- no expansion needed). */
static u32 upload_tris_vp(const rsx_state* state, u32 first, u32 count)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    vp_attrs_dbg(state);
    u32 maxv = MAX_VERTICES - s_d3d.vp_vb_vertex_count;
    if (count > maxv) {
        const u32 kept = maxv - (maxv % 3);
        s_drop_vertices += count - kept; s_drop_draws++;
        { static int _n = 0; if (d3d12_setting_vbfull() && PS3_LOG_FIRST(_n, 12))
            fprintf(stderr, "[VBFULL] fp=0x%X wanted %u verts, room for %u%c",
                    state->shader_program, count, maxv, 10); }
        count = kept; }
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    { double _tv = perf_on() ? perf_now() : 0.0;
      rsx_decode_vp_vertices(state, first, count, NULL, 0, out);
      if (perf_on()) { s_perf_vtx += perf_now() - _tv; s_perf_nverts += count; } }
    /* IDXDBG=<hex shader_program>: vertex range for that program's draws. */
    { static const char* ie = (const char*)1; static u32 iw = 0;
      if (ie == (const char*)1) { ie = getenv("IDXDBG");
          iw = ie ? (u32)strtoul(ie, NULL, 16) : 0; }
      if (iw && state->shader_program == iw) { static int _n = 0; if (PS3_LOG_FIRST(_n, 12))
          fprintf(stderr, "[IDXDBG] tris first=%u count=%u a0off=0x%X stride=%u%c",
                  first, count, state->vertex_attribs[0].offset,
                  state->vertex_attribs[0].stride, 10); } }
    /* RSX_DBG_VTX=<hex tex0 offset>: dump attribute-0 positions for the draws that
     * bind that texture. The duck's texture resolves and its 9960 draws all
     * target the backbuffer, yet none of its texels reach the screen -- so the
     * question is whether its vertices are where they should be. */
    { static const char* dv = (const char*)1; static u32 want = 0; static int n = 0;
      if (dv == (const char*)1) { dv = getenv("RSX_DBG_VTX");
                                  want = dv ? (u32)strtoul(dv, NULL, 16) : 0; }
      if (want && s_d3d.cur_texs[0].raw == want && n < 6) { n++;
        fprintf(stderr, "[DUCKVTX] first=%u count=%u", first, count);
        for (u32 k = 0; k < count && k < 4; k++)
            fprintf(stderr, "  v%u=(%.2f,%.2f,%.2f)", k,
                    out[k*s_d3d.vp_upload_count].v[0], out[k*s_d3d.vp_upload_count].v[1], out[k*s_d3d.vp_upload_count].v[2]);
        { int lo = -1, hi = -1, nz = 0;
          for (int r = 0; r < RSX_MAX_VERTEX_CONSTANTS; r++) {
              const float* m = state->vertex_constants[r];
              if (m[0] || m[1] || m[2] || m[3]) { if (lo < 0) lo = r; hi = r; nz++; }
          }
          fprintf(stderr, "  | constants: %d non-zero, slots %d..%d", nz, lo, hi); }
        fprintf(stderr, "%c", 10);
      } }
    if (d3d12_setting_vtx_dump() && state->surface_color_offset[0] == 0xCC0000) {
        static int _n=0; if (PS3_LOG_FIRST(_n, 1)) {
        FILE* f = fopen("vtx_dump.txt", "w");
        if (f) { fprintf(f, "G-buffer draw surf0=0x%X count=%u first=%u\n",
                         state->surface_color_offset[0], count, first);
          for (u32 k = 0; k < count && k < 30; k++)
            fprintf(f, "v%02u a0=(%.3f,%.3f,%.3f) a2=(%.3f,%.3f,%.3f) a9=(%.4f,%.4f,%.4f,%.4f)\n", k,
                out[k*s_d3d.vp_upload_count].v[0],out[k*s_d3d.vp_upload_count].v[1],out[k*s_d3d.vp_upload_count].v[2],
                out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 2)].v[0],out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 2)].v[1],out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 2)].v[2],
                out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 9)].v[0],out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 9)].v[1],out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 9)].v[2],out[k*s_d3d.vp_upload_count+vp_mask_slot(s_d3d.vp_upload_mask, 9)].v[3]);
          fclose(f); } } }
    s_d3d.vp_vb_offset += count * s_d3d.vp_upload_stride;
    s_d3d.vp_vb_vertex_count += count;
    return count;
}

/* Non-indexed TRIANGLE_STRIP (fan=0) / TRIANGLE_FAN (fan=1) through the VP
 * path, expanded to a triangle list. LBP's Bink movie draws its YUV quad as a
 * non-indexed 4-vertex strip (prim 6); without this it fell to the fixed-
 * function fallback with is_vp=0 (no vertex-program transform -> the quad
 * collapsed to a line) and textured=0 (the Y/U/V planes never sampled). */
static u32 upload_strip_vp(const rsx_state* state, u32 first, u32 count, int fan)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    if (count < 3) return 0;
    u32 tris = count - 2;
    u32 maxv = MAX_VERTICES - s_d3d.vp_vb_vertex_count;
    if (tris * 3 > maxv) {
        const u32 kept = maxv / 3;
        s_drop_vertices += tris * 3 - kept * 3; s_drop_draws++;
        tris = kept;
    }
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    rsx_decode_vp_expanded(state, first, tris, NULL, 0, 0, fan, out);
    const u32 o = tris * 3u;
    s_d3d.vp_vb_offset += o * s_d3d.vp_upload_stride;
    s_d3d.vp_vb_vertex_count += o;
    return o;
}

static u32 upload_quad_strip_vp(const rsx_state* state, u32 first, u32 count)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    if (count < 4) return 0;
    u32 quads = (count - 2) / 2;
    u32 maxv = (MAX_VERTICES * VP_VERT_STRIDE - s_d3d.vp_vb_offset) / VP_VERT_STRIDE;
    if (quads * 6 > maxv) quads = maxv / 6;
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    u32 o = 0;
    for (u32 q = 0; q < quads; q++) {
        u32 b = first + q * 2;
        VPSlot v0[16], v1[16], v2[16], v3[16];
        read_vp_vertex(state, b + 0, v0);
        read_vp_vertex(state, b + 1, v1);
        read_vp_vertex(state, b + 2, v2);
        read_vp_vertex(state, b + 3, v3);
        memcpy(&out[(o + 0)*16], v0, sizeof(v0));
        memcpy(&out[(o + 1)*16], v1, sizeof(v1));
        memcpy(&out[(o + 2)*16], v3, sizeof(v3));
        memcpy(&out[(o + 3)*16], v0, sizeof(v0));
        memcpy(&out[(o + 4)*16], v3, sizeof(v3));
        memcpy(&out[(o + 5)*16], v2, sizeof(v2));
        o += 6;
    }
    s_d3d.vp_vb_offset += o * VP_VERT_STRIDE;
    return o;
}

/* Fetch index k from the guest index array (SET_INDEX_ARRAY_ADDRESS/_DMA:
 * dma [3:0] = location (0 local, 1 main), [7:4] = type (0 u32, 1 u16)).
 * Indices are big-endian in guest memory. */
static u32 read_guest_index(const rsx_state* st, u32 k)
{
    extern uint8_t* vm_base;
    extern u32 cellGcmResolveLocated(int local, u32 offset);
    int local = ((st->index_array_dma & 0xF) == 0);
    u32 base = cellGcmResolveLocated(local, st->index_array_offset);
    const u8* p = vm_base + base;
    if (((st->index_array_dma >> 4) & 0xF) == 1) {
        p += (u64)k * 2;
        return ((u32)p[0] << 8) | p[1];
    }
    p += (u64)k * 4;
    return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3];
}

/* RPCS3_REFERENCE_REVIEWED: immediate indices are already decoded guest
 * u32 values; ordinary indexed draws still read the guest index buffer. */
static u32 read_index_source(const rsx_state* st, const u32* immediate,
                             u32 index)
{
    return immediate ? immediate[index] : read_guest_index(st, index);
}

/* RSX adds the element base after reading the raw index. Keep restart checks
 * on read_index_source(), since the restart marker is compared before rebasing. */
static u32 resolve_indexed_vertex(const rsx_state* st, const u32* immediate,
                                  u32 index)
{
    return (read_index_source(st, immediate, index) +
            st->vertex_data_base_index) & 0x000FFFFFu;
}

/* Indexed variants of the VP-path uploads. */
static u32 upload_quads_vp_indexed(const rsx_state* state, u32 first, u32 count,
                                   const u32* immediate)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    u32 quads = count / 4;
    u32 maxv = MAX_VERTICES - s_d3d.vp_vb_vertex_count;
    if (quads * 6 > maxv) {
        const u32 kept = maxv / 6;
        s_drop_vertices += quads * 6 - kept * 6;
        s_drop_draws++;
        quads = kept;
    }
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    const u64 epoch = vp_next_decode_epoch();
    if (!d3d12_setting_vtx_dump()) {
        rsx_decode_vp_expanded(state, first, quads, immediate, 1, 1, 0, out);
        const u32 emitted = quads * 6u;
        s_d3d.vp_vb_offset += emitted * s_d3d.vp_upload_stride;
        s_d3d.vp_vb_vertex_count += emitted;
        return emitted;
    }
    u32 o = 0;
    for (u32 q = 0; q < quads; q++) {
        VPSlot c[4][16];
        for (u32 k = 0; k < 4; k++)
            read_vp_vertex_cached(state, resolve_indexed_vertex(state, immediate,
                first + q*4 + k), c[k], epoch, 1);
        static const int idx[6] = {0,1,2, 0,2,3};
        for (int t = 0; t < 6; t++) {
            memcpy(&out[o * s_d3d.vp_upload_count], c[idx[t]],
                   s_d3d.vp_upload_count * sizeof(VPSlot));
            o++;
        }
        if (d3d12_setting_vtx_dump()) { static int _n=0; if (PS3_LOG_FIRST(_n, 6)) {
            FILE* f = fopen("vtx_dump.txt", _n==1 ? "w" : "a");
            if (f) {
                const float* vs_ = state->viewport_scale;
                const float* vo_ = state->viewport_offset;
                fprintf(f, "iq%02d vp_scale=(%.1f,%.1f,%.3f) vp_off=(%.1f,%.1f,%.3f)\n", _n,
                        vs_[0], vs_[1], vs_[2], vo_[0], vo_[1], vo_[2]);
                for (int _c = 464; _c <= 467; _c++)
                    fprintf(f, "  c[%d]=(%.3f,%.3f,%.3f,%.3f)\n", _c,
                            state->vertex_constants[_c][0], state->vertex_constants[_c][1],
                            state->vertex_constants[_c][2], state->vertex_constants[_c][3]);
                for (u32 k = 0; k < 4; k++)
                    fprintf(f, "  v%u i%u a0=(%.3f,%.3f,%.3f,%.3f) a8=(%.3f,%.3f)\n", k,
                        read_index_source(state, immediate, first + q*4 + k),
                        c[k][0].v[0],c[k][0].v[1],c[k][0].v[2],c[k][0].v[3],
                        c[k][vp_mask_slot(s_d3d.vp_upload_mask, 8)].v[0],
                        c[k][vp_mask_slot(s_d3d.vp_upload_mask, 8)].v[1]);
                fclose(f);
            } } }
    }
    s_d3d.vp_vb_offset += o * s_d3d.vp_upload_stride;
    s_d3d.vp_vb_vertex_count += o;
    return o;
}

static u32 upload_tris_vp_indexed(const rsx_state* state, u32 first, u32 count,
                                  const u32* immediate)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    vp_attrs_dbg(state);
    u32 maxv = MAX_VERTICES - s_d3d.vp_vb_vertex_count;
    if (count > maxv) {
        const u32 kept = maxv - (maxv % 3);
        s_drop_vertices += count - kept; s_drop_draws++;
        { static int _n = 0; if (d3d12_setting_vbfull() && PS3_LOG_FIRST(_n, 12))
            fprintf(stderr, "[VBFULL] fp=0x%X wanted %u verts, room for %u%c",
                    state->shader_program, count, maxv, 10); }
        count = kept; }
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    { double _tv = perf_on() ? perf_now() : 0.0;
      rsx_decode_vp_vertices(state, first, count, immediate, 1, out);
      if (perf_on()) { s_perf_vtx += perf_now() - _tv; s_perf_nverts += count; } }
    /* IDXDBG=<hex shader_program>: index range for that program's draws. An
     * index past the vertex array reads unmapped guest memory as zero, which
     * shows up as a spike triangle with a zero texcoord, not as a missing draw. */
    { static const char* ie = (const char*)1; static u32 iw = 0;
      if (ie == (const char*)1) { ie = getenv("IDXDBG");
          iw = ie ? (u32)strtoul(ie, NULL, 16) : 0; }
      if (iw && state->shader_program == iw) { static int _n = 0; if (PS3_LOG_FIRST(_n, 12)) {
          u32 lo = 0xFFFFFFFFu, hi = 0;
          for (u32 k = 0; k < count; k++) { u32 ix = read_index_source(state, immediate, first + k);
              if (ix < lo) lo = ix; if (ix > hi) hi = ix; }
          fprintf(stderr, "[IDXDBG] first=%u count=%u idx=[%u..%u] dma=0x%X a0off=0x%X stride=%u%c",
                  first, count, lo, hi, state->index_array_dma,
                  state->vertex_attribs[0].offset, state->vertex_attribs[0].stride, 10); } } }
    /* DBG_LOCK tracking: projected centroid of this mesh, updated every draw. */
    { static const char* dl = (const char*)1; static u32 wantl = 0; static int mvpb = 256;
      if (dl == (const char*)1) { dl = getenv("RSX_DBG_VTX");
                                  wantl = dl ? (u32)strtoul(dl, NULL, 16) : 0;
                                  const char* mb = getenv("MVP_BASE");
                                  if (mb) mvpb = atoi(mb); }
      /* RSX_DBG_TRACK=1 tracks the content-identified duck without needing its
       * offset passed in (VRAM offsets move between runs). Re-read every call:
       * s_duck_raw is filled by the texture upload, which happens AFTER the
       * first draws, so caching it once left the tracker watching offset 0. */
      { static int dtrk = -1;
        if (dtrk < 0) { const char* e = getenv("RSX_DBG_TRACK"); dtrk = e ? atoi(e) : 0; }
        if (dtrk && s_duck_raw) wantl = s_duck_raw; }
      /* RSX_DBG_PICK=<first>: track ONLY the draw with that starting index. The
       * mesh is one big vertex buffer sliced into 256-index chunks, so averaging
       * across all of them converges on the centre of the whole field -- which is
       * where there is nothing in particular. A single slice is a stable target
       * and can be magnified without drifting off it. */
      static int pick = -1;
      if (pick == -1) { const char* e = getenv("RSX_DBG_PICK"); pick = e ? atoi(e) : -2; }
      if (wantl && s_d3d.cur_texs[0].raw == wantl && count
          && (pick == -2 || (int)first == pick)) {
          double sx = 0, sy = 0, sz = 0;
          for (u32 k = 0; k < count; k++) {
              sx += out[k*s_d3d.vp_upload_count].v[0]; sy += out[k*s_d3d.vp_upload_count].v[1]; sz += out[k*s_d3d.vp_upload_count].v[2];
          }
          float v[4] = { (float)(sx/count), (float)(sy/count), (float)(sz/count), 1.0f };
          float clip[4];
          for (int r = 0; r < 4; r++) {
              const float* m = state->vertex_constants[mvpb + r];
              clip[r] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3]*v[3];
          }
          if (clip[3] > 1e-6f) {
              float nx = clip[0]/clip[3], ny = clip[1]/clip[3];
              if (nx > -2.0f && nx < 2.0f && ny > -2.0f && ny < 2.0f) {
                  s_lock_sx += nx; s_lock_sy += ny; s_lock_n++;
                  if (!s_lock_valid) {        /* seed so frame 1 is usable */
                      s_lock_x = nx; s_lock_y = ny; s_lock_valid = 1;
                  }
              }
          }
      } }
    /* RSX_DBG_SCALE=<f>: multiply attribute-0 positions for the draws binding
     * RSX_DBG_VTX's texture. The duck's object-space bbox is ~0.2 units and its
     * draws rasterize only a few dozen pixels per frame -- sub-pixel. Growing it
     * about its own origin tests the rest of the chain (index buffer, texture,
     * per-draw transform): if a duck-shaped, duck-coloured object appears, only
     * the scale is wrong. Diagnostic. */
    { static const char* ds = (const char*)1; static float sc = 0.0f;
      if (ds == (const char*)1) { ds = getenv("RSX_DBG_SCALE");
                                  sc = ds ? (float)atof(ds) : 0.0f; }
      static const char* dv2 = (const char*)1; static u32 want2 = 0;
      if (dv2 == (const char*)1) { dv2 = getenv("RSX_DBG_VTX");
                                   want2 = dv2 ? (u32)strtoul(dv2, NULL, 16) : 0; }
      if (sc > 0.0f && want2 && s_d3d.cur_texs[0].raw == want2 && count) {
        /* Magnify about a FIXED point captured from the first tracked draw, not
         * about each chunk's own centre -- recentring per chunk would stack every
         * chunk of the mesh on top of the others. Combined with VP_BYPASS (which
         * maps attribute 0 straight to clip space) this renders the mesh's real
         * geometry and texture at a readable size. */
        /* RSX_DBG_RECENTER=1: recentre EVERY chunk on its own centroid, overlaying
         * them. If each chunk is one instance of the same mesh they align into a
         * single silhouette; if they are arbitrary slices of one big mesh they
         * smear. Either way the answer is visible at a glance. */
        static int recen = -1;
        if (recen < 0) { const char* e = getenv("RSX_DBG_RECENTER"); recen = e ? atoi(e) : 0; }
        static int have_c = 0; static float ox = 0, oy = 0, oz = 0;
        if (!have_c || recen) {
            double sx = 0, sy = 0, sz = 0;
            for (u32 k = 0; k < count; k++) {
                sx += out[k*s_d3d.vp_upload_count].v[0]; sy += out[k*s_d3d.vp_upload_count].v[1]; sz += out[k*s_d3d.vp_upload_count].v[2];
            }
            ox = (float)(sx/count); oy = (float)(sy/count); oz = (float)(sz/count);
            have_c = 1;
        }
        for (u32 k = 0; k < count; k++) {
            /* RSX_DBG_SHIFT="dx,dy": nudge after scaling. VP_BYPASS applies the
             * guest's posoffset, so a recentred mesh does not land at screen
             * centre -- without this the magnified geometry sits clipped against
             * the top edge. */
            static int shf = -1; static float dx = 0.0f, dy = 0.0f;
            if (shf < 0) { const char* e = getenv("RSX_DBG_SHIFT");
                           if (e) { double a=0,b=0; sscanf(e, "%lf,%lf", &a, &b);
                                    dx = (float)a; dy = (float)b; }
                           shf = 1; }
            out[k*s_d3d.vp_upload_count].v[0] = (out[k*s_d3d.vp_upload_count].v[0] - ox) * sc + dx;
            out[k*s_d3d.vp_upload_count].v[1] = (out[k*s_d3d.vp_upload_count].v[1] - oy) * sc + dy;
            /* Leave z alone: under VP_BYPASS the object z maps straight to clip
             * z, so scaling it walks the geometry out of the depth range and
             * everything clips away. */
            (void)oz;
        }
      } }
    /* RSX_DBG_VTX=<hex tex0 offset>: attribute-0 positions for the draws binding
     * that texture (see upload_tris_vp). The duck's meshes are indexed, so this
     * is the copy that actually fires for it. */
    { static const char* dv = (const char*)1; static u32 want = 0; static int n = 0;
      if (dv == (const char*)1) { dv = getenv("RSX_DBG_VTX");
                                  want = dv ? (u32)strtoul(dv, NULL, 16) : 0; }
      if (want && s_d3d.cur_texs[0].raw == want && n < 8) { n++;
        float mnx=1e30f,mny=1e30f,mnz=1e30f,mxx=-1e30f,mxy=-1e30f,mxz=-1e30f;
        for (u32 k = 0; k < count; k++) {
            float x=out[k*s_d3d.vp_upload_count].v[0], y=out[k*s_d3d.vp_upload_count].v[1], z=out[k*s_d3d.vp_upload_count].v[2];
            if (x<mnx)mnx=x; if (y<mny)mny=y; if (z<mnz)mnz=z;
            if (x>mxx)mxx=x; if (y>mxy)mxy=y; if (z>mxz)mxz=z;
        }
        const rsx_vertex_attrib* _a3 = &state->vertex_attribs[3];
        fprintf(stderr, "[DUCKVTX] a3(en=%d type=%u size=%u stride=%u off=0x%08X freq=%u)"
                        " -> col0 v0=(%.3f %.3f %.3f %.3f)%c",
                _a3->enabled, _a3->type, _a3->size, _a3->stride, _a3->offset,
                _a3->frequency,
                out[vp_mask_slot(s_d3d.vp_upload_mask, 3)].v[0],
                out[vp_mask_slot(s_d3d.vp_upload_mask, 3)].v[1],
                out[vp_mask_slot(s_d3d.vp_upload_mask, 3)].v[2],
                out[vp_mask_slot(s_d3d.vp_upload_mask, 3)].v[3], 10);
        const rsx_vertex_attrib* _a0 = &state->vertex_attribs[0];
        fprintf(stderr, "[DUCKVTX] first=%u count=%u a0(type=%u size=%u stride=%u off=0x%08X)"
                        " obj-bbox x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]%c",
                first, count, _a0->type, _a0->size, _a0->stride, _a0->offset,
                mnx,mxx, mny,mxy, mnz,mxz, 10);
        /* Project the object-space bbox corners through the first four transform
         * constants (the conventional MVP rows) and report NDC. Sub-pixel output
         * with correct vertices means the placement transform is the problem, and
         * the NDC extent says by how much. */
        { float cx=(mnx+mxx)*0.5f, cy=(mny+mxy)*0.5f, cz=(mnz+mxz)*0.5f;
          /* This title's Cg programs keep their MVP at c[256..259], not c[0..3]
           * (see the vp_uses_c03 note); projecting through c0 reported zeros. */
          static int MVP = -1;
          if (MVP < 0) { const char* e = getenv("MVP_BASE"); MVP = e ? atoi(e) : 256; }
          float v[4] = { cx, cy, cz, 1.0f };
          float clip[4];
          for (int r = 0; r < 4; r++) {
              const float* m = state->vertex_constants[MVP + r];
              clip[r] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3]*v[3];
          }
          float ex[4];
          { float e[4] = { mxx-cx, mxy-cy, mxz-cz, 0.0f };
            for (int r = 0; r < 4; r++) {
                const float* m = state->vertex_constants[MVP + r];
                ex[r] = m[0]*e[0] + m[1]*e[1] + m[2]*e[2];
            } }
          { int lo = -1, hi = -1, nz = 0;
            for (int r = 0; r < RSX_MAX_VERTEX_CONSTANTS; r++) {
                const float* m = state->vertex_constants[r];
                if (m[0] || m[1] || m[2] || m[3]) { if (lo < 0) lo = r; hi = r; nz++; }
            }
            fprintf(stderr, "[DUCKVTX]   constants: %d non-zero, slots %d..%d;", nz, lo, hi);
            for (int r = (lo < 0 ? 0 : lo); r < (lo < 0 ? 0 : lo) + 4 && r < RSX_MAX_VERTEX_CONSTANTS; r++)
                fprintf(stderr, " c%d=(%.3f %.3f %.3f %.3f)", r,
                        state->vertex_constants[r][0], state->vertex_constants[r][1],
                        state->vertex_constants[r][2], state->vertex_constants[r][3]);
            fprintf(stderr, "%c", 10); }
          fprintf(stderr, "[DUCKVTX]   mvp0=(%.3f %.3f %.3f %.3f) clip=(%.3f %.3f %.3f w=%.3f)",
                  state->vertex_constants[MVP][0], state->vertex_constants[MVP][1],
                  state->vertex_constants[MVP][2], state->vertex_constants[MVP][3],
                  clip[0], clip[1], clip[2], clip[3]);
          if (clip[3] > 1e-6f)
              fprintf(stderr, " ndc=(%.4f %.4f) half-extent-px=(%.2f %.2f)",
                      clip[0]/clip[3], clip[1]/clip[3],
                      ex[0]/clip[3]*640.0f, ex[1]/clip[3]*360.0f);
          fprintf(stderr, "%c", 10); }
      } }
    /* IDX_DBG=<N>: the first indices and the position they resolve to. An index
     * buffer read with the wrong element size or base yields huge indices and
     * degenerate triangles -- an INDEXED mesh vanishes while non-indexed
     * geometry in the same scene renders fine. */
    { static int cap = -1, n = 0;
      if (cap < 0) { const char* e = getenv("IDX_DBG"); cap = e ? atoi(e) : 0; }
      if (cap && n < cap && count >= 3) { n++;
        extern u32 cellGcmResolveOffset(u32);
        const rsx_vertex_attrib* a0 = &state->vertex_attribs[0];
        u32 vbase = cellGcmResolveOffset(a0->offset & 0x7FFFFFFFu);
        u32 nz = 0; for (u32 q = 0; q < 0x1000u; q += 29) if (vm_base[vbase + q]) nz++;
        fprintf(stderr, "[IDX] idxoff=0x%X idx: %u %u %u | a0 off=0x%08X(->0x%08X nz=%u/142)"
                        " stride=%u size=%u type=%u  pos0=(%g %g %g)%c",
                state->index_array_offset,
                read_index_source(state, immediate, first),
                read_index_source(state, immediate, first+1),
                read_index_source(state, immediate, first+2),
                a0->offset, vbase, nz, a0->stride, a0->size, a0->type,
                out[0].v[0], out[0].v[1], out[0].v[2], 10); } }
    s_d3d.vp_vb_offset += count * s_d3d.vp_upload_stride;
    s_d3d.vp_vb_vertex_count += count;
    return count;
}

/* Indexed TRIANGLE_STRIP / TRIANGLE_FAN, expanded CPU-side into a triangle
 * list (the whole VP path draws lists). DeferredShading's spline tube meshes
 * -- the octopus tentacles -- are indexed strips. Odd strip triangles swap
 * two vertices, as in upload_strip_vp: guest culling is honoured now, and
 * without the swap every second triangle comes out back-facing (God of War
 * HD's fullscreen 4-index quads lost their lower-right half). */
static u32 upload_strip_vp_indexed(const rsx_state* state, u32 first, u32 count, int fan,
                                    const u32* immediate)
{
    extern uint8_t* vm_base;
    if (!state || !vm_base || !s_d3d.vp_vb_mapped) return 0;
    if (!state->vertex_attribs[0].enabled) return 0;
    if (count < 3) return 0;
    u32 tris = count - 2;
    u32 maxv = MAX_VERTICES - s_d3d.vp_vb_vertex_count;
    if (tris * 3 > maxv) {
        const u32 kept = maxv / 3;
        s_drop_vertices += tris * 3 - kept * 3; s_drop_draws++;
        tris = kept;
    }
    VPSlot* out = (VPSlot*)((u8*)s_d3d.vp_vb_mapped
        + (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset);
    rsx_decode_vp_expanded(state, first, tris, immediate, 1, 0, fan, out);
    const u32 o = tris * 3u;
    s_d3d.vp_vb_offset += o * s_d3d.vp_upload_stride;
    s_d3d.vp_vb_vertex_count += o;
    return o;
}

/* Expand index-restart segments independently. Triangle strips must restart
 * their winding at each marker; concatenating the segments would connect the
 * end of one strip to the beginning of the next. */
static u32 upload_indexed_with_restart(const rsx_state* state, u32 primitive,
                                       u32 first, u32 count,
                                       const u32* immediate)
{
    if (!state || !state->restart_index_enable)
        return 0;
    u32 restart = state->restart_index;
    if (!immediate && ((state->index_array_dma >> 4) & 0xFu) == 1u)
        restart &= 0xFFFFu;
    u32 emitted = 0, segment = 0;
    for (u32 i = 0; i <= count; ++i) {
        const int boundary = i == count ||
            read_index_source(state, immediate, first + i) == restart;
        if (!boundary)
            continue;
        const u32 length = i - segment;
        if (primitive == RSX_PRIMITIVE_QUADS)
            emitted += upload_quads_vp_indexed(state, first + segment,
                                                length & ~3u, immediate);
        else if (primitive == RSX_PRIMITIVE_TRIANGLES)
            emitted += upload_tris_vp_indexed(state, first + segment,
                                               length - (length % 3u), immediate);
        else if (primitive == RSX_PRIMITIVE_TRIANGLE_STRIP && length >= 3u)
            emitted += upload_strip_vp_indexed(state, first + segment,
                                                length, 0, immediate);
        else if (primitive == RSX_PRIMITIVE_TRIANGLE_FAN && length >= 3u)
            emitted += upload_strip_vp_indexed(state, first + segment,
                                                length, 1, immediate);
        segment = i + 1u;
    }
    return emitted;
}


static u32 upload_vertices_raw(const rsx_state* state, u32 primitive, u32 first, u32 count,
    const u32* immediate, int indexed, int vs_idx, u32* raw_bytes,
    u32* packet_offset, u32* index_offset)
{
    extern u8* vm_base;
    extern u32 cellGcmResolveLocated(int local, u32 offset);
    static int enabled = -1;
    if (enabled < 0) {
        const char* setting = getenv("RSX_GPU_VERTEX_FETCH");
        enabled = (!setting || setting[0] != '0') && !getenv("VP_NOFREQ") &&
            !d3d12_setting_vtx_dump() && !getenv("RSX_DBG_VTX") && !getenv("IDX_DBG");
    }
    *raw_bytes = 0;
    *index_offset = 0;
    const u32 output_count = rsx_raw_vertex_count(primitive, count);
    if (!enabled || !state || !vm_base || !output_count ||
        state->restart_index_enable ||
        !state->shader_program || !state->vertex_attribs[0].enabled ||
        vs_idx < 0 || vs_idx >= s_d3d.vp_vs_n ||
        !s_d3d.vp_vs[vs_idx].raw_vs || s_d3d.vp_upload_stride == VP_VERT_STRIDE ||
        output_count > MAX_VERTICES - s_d3d.vp_vb_vertex_count) return 0;
    if ((u64)output_count * s_d3d.vp_upload_stride <=
        RSX_RAW_VERTEX_HEADER_BYTES + (u64)output_count * 4 + 16) return 0;
    if (indexed && state->vertex_data_base_index) {
        for (u32 a = 0; a < 16; ++a) {
            const rsx_vertex_attrib* attr = &state->vertex_attribs[a];
            if ((s_d3d.vp_upload_mask & (1u << a)) && attr->enabled &&
                (attr->frequency > 1 ||
                 (attr->frequency == 1 && (state->frequency_divider_op & (1u << a)))))
                return 0;
        }
    }
    if (primitive == RSX_PRIMITIVE_TRIANGLES && s_d3d.merge_prev_draw && s_d3d.draw_count && s_d3d.merge_first_end == first) {
        const D3D12DrawRecord* previous = &s_d3d.draws[s_d3d.draw_count - 1];
        if (previous->begin_epoch == state->begin_epoch && previous->vertex_count % 3)
            return 0;
    }
    if (output_count * 2u > s_raw_vertex_id_capacity) {
        u32 capacity = s_raw_vertex_id_capacity ? s_raw_vertex_id_capacity : 256;
        while (capacity < output_count * 2u) capacity *= 2;
        u32* ids = (u32*)realloc(s_raw_vertex_ids, (size_t)capacity * sizeof(u32));
        if (!ids) return 0;
        s_raw_vertex_ids = ids;
        s_raw_vertex_id_capacity = capacity;
    }
    const double start = perf_on() ? perf_now() : 0.0;
    const u8* index_data = NULL;
    const u32 index_size = ((state->index_array_dma >> 4) & 0xFu) == 1 ? 2u : 4u;
    if (indexed && !immediate)
        index_data = vm_base + cellGcmResolveLocated(
            (state->index_array_dma & 0xFu) == 0, state->index_array_offset);
    for (u32 i = 0; i < output_count; ++i) {
        const u32 source = first + rsx_raw_vertex_source(primitive, i,
            state->shade_mode == CELL_GCM_FLAT);
        u32 vertex = source;
        if (indexed) {
            if (immediate) vertex = immediate[source];
            else {
                const u8* p = index_data + (u64)source * index_size;
                vertex = index_size == 2 ? ((u32)p[0] << 8) | p[1]
                    : ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3];
            }
            vertex = (vertex + state->vertex_data_base_index) & 0x000FFFFFu;
        }
        s_raw_vertex_ids[i] = vertex;
    }
    u32 capacity = MAX_VERTICES * VP_VERT_STRIDE - s_d3d.vp_vb_offset;
    const u32 converted_bytes = output_count * s_d3d.vp_upload_stride;
    if (capacity > converted_bytes) capacity = converted_bytes;
    rsx_raw_vertex_plan plan;
    static int indexed_enabled = -1, reuse_enabled = -1;
    if (indexed_enabled < 0) {
        const char* setting = getenv("RSX_RAW_INDEXED");
        indexed_enabled = !setting || setting[0] != '0';
        setting = getenv("RSX_RAW_UPLOAD_REUSE");
        reuse_enabled = !setting || setting[0] != '0';
    }
    u32* dense = s_raw_vertex_ids + output_count;
    const int native_indexed = indexed_enabled && rsx_raw_vertex_indexed_prepare(state,
        s_d3d.vp_upload_mask, s_raw_vertex_ids, output_count, dense,
        capacity, cellGcmResolveLocated, &plan);
    if (native_indexed) {
        *index_offset = plan.bytes;
    } else if (!rsx_raw_vertex_prepare(state, s_d3d.vp_upload_mask, s_raw_vertex_ids,
            output_count, capacity, cellGcmResolveLocated, &plan)) {
        if (perf_on()) s_perf_vtx += perf_now() - start;
        return 0;
    }
    const u32 packet_bytes = plan.bytes + (native_indexed ? ((output_count * 4u + 15u) & ~15u) : 0u);
    u8* destination = (u8*)s_d3d.vp_vb_mapped +
        (u64)s_d3d.vp_parity * MAX_VERTICES * VP_VERT_STRIDE + s_d3d.vp_vb_offset;
    int reused = 0;
    if (reuse_enabled) reused = rsx_raw_upload_reuse(&s_raw_upload_cache, &plan,
        native_indexed ? dense : s_raw_vertex_ids, s_raw_vertex_ids,
        native_indexed ? output_count : 0, vm_base, s_d3d.vp_vb_offset,
        destination, packet_offset);
    else {
        rsx_raw_vertex_write(&plan, native_indexed ? dense : s_raw_vertex_ids, vm_base, destination);
        if (native_indexed) memcpy(destination + plan.bytes, s_raw_vertex_ids, output_count * 4u);
        if (packet_bytes > plan.bytes + (native_indexed ? output_count * 4u : 0u))
            memset(destination + plan.bytes + output_count * 4u, 0, packet_bytes - plan.bytes - output_count * 4u);
        *packet_offset = s_d3d.vp_vb_offset;
    }
    if (!reused) s_d3d.vp_vb_offset += packet_bytes;
    else ++s_perf_raw_reused;
    if (native_indexed) ++s_perf_raw_indexed;
    s_d3d.vp_vb_vertex_count += output_count;
    *raw_bytes = packet_bytes;
    ++s_perf_raw_draws;
    s_perf_raw_bytes += reused ? 0 : packet_bytes;
    s_perf_raw_saved += converted_bytes - (reused ? 0 : packet_bytes);
    if (perf_on()) { s_perf_vtx += perf_now() - start; s_perf_nverts += output_count; }
    return output_count;
}

static void d3d12_draw_arrays(void* ud, u32 primitive, u32 first, u32 count)
{
    /* RSX_LIVE_FEED_DBG=1: pair with the [feed] method counts so "the FIFO
     * carries no draw batches" can be checked against "the backend is told
     * to draw" in the same run. */
    { static int dbg = -1; if (dbg < 0) dbg = getenv("RSX_LIVE_FEED_DBG") ? 1 : 0;
      if (dbg) { static long long n = 0;
          if (((++n) % 20000) == 0)
              printf("[backend] arrays callbacks=%lld (prim=%u count=%u)\n", n, primitive, count), fflush(stdout); } }
    (void)ud;
    /* Log the first 20 calls in detail, then every 1000th to show liveness
     * without flooding. */
    static u64 s_total = 0;
    static int trace_draws = -1;
    if (trace_draws < 0) trace_draws = getenv("RSX_DRAW_TRACE") != NULL;
    if (trace_draws && (s_total < 20 || (s_total % 1000) == 0)) {
        printf("[D3D12] draw_arrays #%llu prim=%u first=%u count=%u\n",
               (unsigned long long)s_total, primitive, first, count);
    }
    s_total++;

    s_d3d12_stats.calls++;
    if (!s_d3d.pipeline_ready || !s_d3d.vb_mapped)
        D3D12_DROP(D3D12_DROP_NOT_READY);
    if (count == 0) D3D12_DROP(D3D12_DROP_BAD_COUNT);
    s_req_verts += count; s_req_draws++;
    if (count > MAX_VERTICES) {
        s_drop_input_elements += count; s_drop_draws++;
        D3D12_DROP(D3D12_DROP_BAD_COUNT);
    }
    if (!d3d12_acquire_submission_slot()) return;

    /* QUADS (prim 8) have no D3D12 topology; expand to a triangle list.
     * dbgfont draws all text as quads. Prefer the real vertex-program path:
     * upload the analyzed guest inputs into vp_vb and
     * mark the draw is_vp; render_frame compiles the VP and draws it. Fall
     * back to the frac-approximation textured path if VP resources are absent. */
    if (primitive == RSX_PRIMITIVE_QUADS) {
        /* Prefer the real vertex-program path whenever VP resources exist --
         * NOT only when a texture is bound. Requiring tex_bound routed vkcube's
         * UNtextured cube to the fixed-function fallback, which never applies the
         * VP's MVP transform, so the cube was drawn in object space and clipped
         * off-screen. Untextured VP draws now render via the colour PSO. */
        if (s_d3d.vp_vb_mapped && s_d3d.vp_root_sig) {
            const int vp_vs_idx = vp_get_vs(s_d3d.current_rsx_state);
            vp_select_upload_layout(vp_vs_idx, s_d3d.current_rsx_state);
            u32 rec = s_d3d.vp_vb_offset;
            u32 raw_bytes = 0, raw_index_offset = 0;
            u32 emitted = upload_vertices_raw(s_d3d.current_rsx_state, primitive,
                first, count, NULL, 0, vp_vs_idx, &raw_bytes, &rec, &raw_index_offset);
            if (!emitted) {
                rsx_vertex_fetch_prepare(&s_vertex_fetch_prepared, s_d3d.current_rsx_state);
                emitted = upload_quads_vp(s_d3d.current_rsx_state, first, count);
            }
            if (emitted && d3d12_draw_record_available()) {
                D3D12DrawRecord* dr = &s_d3d.draws[s_d3d.draw_count];
                dr->vb_byte_offset = rec;
                dr->vertex_count   = emitted;
                dr->topology       = RSX_TOPOLOGY_TRIANGLES;
                dr->textured       = s_d3d.tex_bound;
                dr->is_vp          = 1;
                dr->fp_addr = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->shader_program : 0;
                dr->fp_exp32 = s_d3d.current_rsx_state ?
                    ((s_d3d.current_rsx_state->shader_control & 0x40) != 0) : 1;
                dr->cull = rsx_cull_key(s_d3d.current_rsx_state);
        dr->begin_epoch = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->begin_epoch : 0;
            dr->begin_epoch = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->begin_epoch : 0;
                dr->cmask = 0xF;
                if (s_d3d.current_rsx_state) {
                    u32 _cm = s_d3d.current_rsx_state->color_mask;
                    dr->cmask = ((_cm & 0x00010000) ? 1u : 0u)   /* R */
                              | ((_cm & 0x00000100) ? 2u : 0u)   /* G */
                              | ((_cm & 0x00000001) ? 4u : 0u)   /* B */
                              | ((_cm & 0x01000000) ? 8u : 0u);  /* A */
                }
                dr->alpha_ctl = 0;
                if (s_d3d.current_rsx_state)
                    dr->alpha_ctl = ((s_d3d.current_rsx_state->alpha_test_enable ? 1u : 0u) << 16)
                                  | ((s_d3d.current_rsx_state->alpha_func & 0xFFu) << 8)
                                  | (s_d3d.current_rsx_state->alpha_ref & 0xFFu);
                for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
                    dr->tex[_u].off = s_d3d.cur_texs[_u].off;
                    dr->tex[_u].raw = s_d3d.cur_texs[_u].raw;
                    dr->tex[_u].w   = s_d3d.cur_texs[_u].w;
                    dr->tex[_u].h   = s_d3d.cur_texs[_u].h;
                    dr->tex[_u].fmt = s_d3d.cur_texs[_u].fmt;
                    dr->tex[_u].ctrl1 = s_d3d.cur_texs[_u].ctrl1;
                    dr->tex[_u].cube  = s_d3d.cur_texs[_u].cube;
                    dr->tex[_u].mips  = s_d3d.cur_texs[_u].mips;
                    dr->tex[_u].set = s_d3d.cur_texs[_u].set;
                    dr->tex_rt[_u]  = -1;
                }
                dr->tex_slot = -1;
                dr->vs_idx = vp_vs_idx;
                dr->vp_input_mask = s_d3d.vp_upload_mask;
                dr->vp_stride = s_d3d.vp_upload_stride;
                dr->vp_raw_bytes = raw_bytes;
                dr->vp_raw_index_offset = raw_index_offset;
                dr->is_clear = 0;
                dr->blend = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->blend_enable : 1;
                dr->blend_key = rsx_blend_key(s_d3d.current_rsx_state, dr->blend);
                dr_capture_extended_rsx_state(dr, s_d3d.current_rsx_state);
                if (d3d12_setting_blenddbg()) { static u32 seen[32]; static int ns=0;
                    u32 k = (dr->blend ? 0x80000000u : 0u) | dr->blend_key;
                    int f=0; for (int i=0;i<ns;i++) if (seen[i]==k) f=1;
                    if (!f && ns<32) { seen[ns++]=k;
                        fprintf(stderr, "[BLEND] enable=%d key=0x%X%c",
                                dr->blend, dr->blend_key, 10); } }
                dr->rt_off = current_rt_off(&dr->rt_w, &dr->rt_h, dr->rt_mrt);
                dr->rt_fmt = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->surface_format : 0;
                if (s_d3d.current_rsx_state) {
                    dr->vp_x = s_d3d.current_rsx_state->viewport_x;
                    dr->vp_y = s_d3d.current_rsx_state->viewport_y;
                    dr->vp_w = s_d3d.current_rsx_state->viewport_w;
                    dr->vp_h = s_d3d.current_rsx_state->viewport_h;
                } else { dr->vp_x = dr->vp_y = dr->vp_w = dr->vp_h = 0; }
                dr->cb_slot = s_d3d.draw_count;
                vp_record_cb(s_d3d.draw_count, dr->vs_idx, dr);
                s_d3d.draw_count++;
            }
            return;
        }
        u32 record_offset = s_d3d.vb_offset;
        u32 emitted = upload_quads_from_rsx(first, count);
        if (emitted == 0) return;
        if (d3d12_draw_record_available()) {
            s_d3d.draws[s_d3d.draw_count].vb_byte_offset = record_offset;
            s_d3d.draws[s_d3d.draw_count].vertex_count   = emitted;
            s_d3d.draws[s_d3d.draw_count].topology       = RSX_TOPOLOGY_TRIANGLES;
            s_d3d.draws[s_d3d.draw_count].textured       = s_d3d.tex_bound;
            s_d3d.draws[s_d3d.draw_count].is_vp          = 0;
            s_d3d.draws[s_d3d.draw_count].is_clear       = 0;
            s_d3d.draws[s_d3d.draw_count].rt_off         = 0;
            s_d3d.draws[s_d3d.draw_count].flat_shading =
                s_d3d.current_rsx_state &&
                s_d3d.current_rsx_state->shade_mode == CELL_GCM_FLAT;
            s_d3d.draws[s_d3d.draw_count].two_sided_lighting =
                s_d3d.current_rsx_state &&
                s_d3d.current_rsx_state->two_sided_lighting;
            s_d3d.draw_count++;
        }
        return;
    }

    /* TRIANGLES (prim 5): same VP-path preference as quads -- the guest's
     * vertex program does the MVP transform (gcm/cube draws its cube this way);
     * the fixed-function fallback below applies no transform, so 3D geometry
     * ends up in object space (invisible/garbage). */
    /* Route TRIANGLES/STRIP/FAN through the VP path (sagemono: LBP draws
     * non-indexed strips/fans), but ONLY when the guest ACTUALLY HAS a vertex
     * program (flOw). The VP path transforms via the guest's VP microcode; with
     * no microcode loaded it transforms nothing and the draw silently produces
     * zero pixels (it also uploads to vp_vb, leaving the fixed-function vb empty).
     * Geometry already in clip space with no VP -- flOw's injected scene -- must
     * go down the fixed-function passthrough below instead. */
    if ((primitive == RSX_PRIMITIVE_TRIANGLES || primitive == RSX_PRIMITIVE_TRIANGLE_STRIP
         || primitive == RSX_PRIMITIVE_TRIANGLE_FAN || primitive == RSX_PRIMITIVE_POLYGON
         || primitive == RSX_PRIMITIVE_QUAD_STRIP) &&
        s_d3d.vp_vb_mapped && s_d3d.vp_root_sig &&
        s_d3d.current_rsx_state && s_d3d.current_rsx_state->vp_ucode_bytes >= 16) {
        const int vp_vs_idx = vp_get_vs(s_d3d.current_rsx_state);
        vp_select_upload_layout(vp_vs_idx, s_d3d.current_rsx_state);
        u32 rec = s_d3d.vp_vb_offset;
        u32 raw_bytes = 0, raw_index_offset = 0;
        u32 emitted = upload_vertices_raw(s_d3d.current_rsx_state, primitive, first, count,
            NULL, 0, vp_vs_idx, &raw_bytes, &rec, &raw_index_offset);
        if (!emitted) {
            rsx_vertex_fetch_prepare(&s_vertex_fetch_prepared, s_d3d.current_rsx_state);
            emitted = (primitive == RSX_PRIMITIVE_TRIANGLES)
                ? upload_tris_vp(s_d3d.current_rsx_state, first, count)
                : upload_strip_vp(s_d3d.current_rsx_state, first, count, primitive == RSX_PRIMITIVE_TRIANGLE_FAN);
        }
        /* RSX splits one primitive stream across several DRAW_ARRAYS entries
         * inside a single BEGIN/END (each carries at most 256 vertices), and the
         * hardware concatenates them before assembling primitives. Recording a
         * draw per batch instead regroups the triangles: this title's tub mesh
         * arrives as 128 then 256,256,..., neither a multiple of 3, so every
         * batch after the first was assembled one vertex out of phase -- which
         * rendered as spike triangles with a zero texcoord along the tub rim.
         * The batches land contiguously in the vertex buffer, so extending the
         * previous record is all that is needed to restore the stream. */
        if (emitted && s_d3d.merge_prev_draw && s_d3d.draw_count > 0 &&
            s_d3d.merge_first_end == first) {
            D3D12DrawRecord* pv = &s_d3d.draws[s_d3d.draw_count - 1];
            u32 fpnow = s_d3d.current_rsx_state
                        ? s_d3d.current_rsx_state->shader_program : 0;
            if (!raw_bytes && !pv->vp_raw_bytes && pv->is_vp && pv->topology == RSX_TOPOLOGY_TRIANGLES &&
                pv->vs_idx == vp_vs_idx && pv->fp_addr == fpnow &&
                primitive == RSX_PRIMITIVE_TRIANGLES &&
                pv->flat_shading == (s_d3d.current_rsx_state &&
                    s_d3d.current_rsx_state->shade_mode == CELL_GCM_FLAT) &&
                pv->two_sided_lighting == (s_d3d.current_rsx_state &&
                    s_d3d.current_rsx_state->two_sided_lighting) &&
                pv->begin_epoch == (s_d3d.current_rsx_state
                                    ? s_d3d.current_rsx_state->begin_epoch : 0) &&
                pv->vp_input_mask == s_d3d.vp_upload_mask &&
                pv->vp_stride == s_d3d.vp_upload_stride &&
                pv->vb_byte_offset + pv->vertex_count * pv->vp_stride == rec) {
                pv->vertex_count += emitted;
                s_d3d.merge_first_end = first + count;
                return;
            }
        }
        if (emitted && d3d12_draw_record_available()) {
            D3D12DrawRecord* dr = &s_d3d.draws[s_d3d.draw_count];
            dr->vb_byte_offset = rec;
            dr->vertex_count   = emitted;
            dr->topology       = RSX_TOPOLOGY_TRIANGLES;
            dr->textured       = s_d3d.tex_bound;
            dr->is_vp          = 1;
            dr->fp_addr = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->shader_program : 0;
            dr->fp_exp32 = s_d3d.current_rsx_state ?
                ((s_d3d.current_rsx_state->shader_control & 0x40) != 0) : 1;
            dr->cull = rsx_cull_key(s_d3d.current_rsx_state);
        dr->begin_epoch = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->begin_epoch : 0;
            dr->begin_epoch = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->begin_epoch : 0;
            dr->cmask = 0xF;
            if (s_d3d.current_rsx_state) {
                u32 _cm = s_d3d.current_rsx_state->color_mask;
                dr->cmask = ((_cm & 0x00010000) ? 1u : 0u)   /* R */
                          | ((_cm & 0x00000100) ? 2u : 0u)   /* G */
                          | ((_cm & 0x00000001) ? 4u : 0u)   /* B */
                          | ((_cm & 0x01000000) ? 8u : 0u);  /* A */
            }
            dr->alpha_ctl = 0;
            if (s_d3d.current_rsx_state)
                dr->alpha_ctl = ((s_d3d.current_rsx_state->alpha_test_enable ? 1u : 0u) << 16)
                              | ((s_d3d.current_rsx_state->alpha_func & 0xFFu) << 8)
                              | (s_d3d.current_rsx_state->alpha_ref & 0xFFu);
            for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
                dr->tex[_u].off = s_d3d.cur_texs[_u].off;
                dr->tex[_u].raw = s_d3d.cur_texs[_u].raw;
                dr->tex[_u].w   = s_d3d.cur_texs[_u].w;
                dr->tex[_u].h   = s_d3d.cur_texs[_u].h;
                dr->tex[_u].fmt = s_d3d.cur_texs[_u].fmt;
                    dr->tex[_u].ctrl1 = s_d3d.cur_texs[_u].ctrl1;
                    dr->tex[_u].cube  = s_d3d.cur_texs[_u].cube;
                    dr->tex[_u].mips  = s_d3d.cur_texs[_u].mips;
                dr->tex[_u].set = s_d3d.cur_texs[_u].set;
                dr->tex_rt[_u]  = -1;
            }
            dr->tex_slot = -1;
            dr->vs_idx = vp_vs_idx;
            dr->vp_input_mask = s_d3d.vp_upload_mask;
            dr->vp_stride = s_d3d.vp_upload_stride;
            dr->vp_raw_bytes = raw_bytes;
                dr->vp_raw_index_offset = raw_index_offset;
            dr->is_clear = 0;
            dr->blend = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->blend_enable : 1;
            dr->blend_key = rsx_blend_key(s_d3d.current_rsx_state, dr->blend);
            dr_capture_extended_rsx_state(dr, s_d3d.current_rsx_state);
            dr->rt_off = current_rt_off(&dr->rt_w, &dr->rt_h, dr->rt_mrt);
            dr->rt_fmt = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->surface_format : 0;
            if (s_d3d.current_rsx_state) {
                dr->vp_x = s_d3d.current_rsx_state->viewport_x;
                dr->vp_y = s_d3d.current_rsx_state->viewport_y;
                dr->vp_w = s_d3d.current_rsx_state->viewport_w;
                dr->vp_h = s_d3d.current_rsx_state->viewport_h;
            } else { dr->vp_x = dr->vp_y = dr->vp_w = dr->vp_h = 0; }
            dr->cb_slot = s_d3d.draw_count;
                vp_record_cb(s_d3d.draw_count, dr->vs_idx, dr);
            s_d3d.draw_count++;
            /* Anchor for merging the next DRAW_ARRAYS batch of this stream. */
            s_d3d.merge_prev_draw  = (primitive == RSX_PRIMITIVE_TRIANGLES);
            s_d3d.merge_first_end  = first + count;
        }
        return;
    }
    s_d3d.merge_prev_draw = 0;   /* any other primitive breaks the stream */

    u32 topo = (u32)rsx_primitive_topology(primitive);
    if (topo == RSX_TOPOLOGY_UNSUPPORTED) {
        /* Other primitives still needing index-buffer conversion
         * (line loops, triangle fans) — skip rather than draw wrong. */
        static int s_skipped_nontri = 0;
        if (s_skipped_nontri < 3) {
            printf("[D3D12] draw_arrays: skipping prim=%u (needs index conversion)\n",
                   primitive);
            s_skipped_nontri++;
        }
        return;
    }

    u32 record_offset = s_d3d.vb_offset;
    u32 actual_count  = upload_vertices_from_rsx(first, count);
    if (actual_count == 0) return;

    if (d3d12_draw_record_available()) {
        s_d3d.draws[s_d3d.draw_count].vb_byte_offset = record_offset;
        s_d3d.draws[s_d3d.draw_count].vertex_count   = actual_count;
        s_d3d.draws[s_d3d.draw_count].topology       = topo;
        s_d3d.draws[s_d3d.draw_count].textured       = 0;
        s_d3d.draws[s_d3d.draw_count].is_vp          = 0;
        s_d3d.draws[s_d3d.draw_count].is_clear       = 0;
        s_d3d.draws[s_d3d.draw_count].rt_off         = 0;
        s_d3d.draws[s_d3d.draw_count].flat_shading =
            s_d3d.current_rsx_state &&
            s_d3d.current_rsx_state->shade_mode == CELL_GCM_FLAT;
        s_d3d.draws[s_d3d.draw_count].two_sided_lighting =
            s_d3d.current_rsx_state &&
            s_d3d.current_rsx_state->two_sided_lighting;
        s_d3d.draw_count++;
    }
}

static void d3d12_draw_indexed_source(void* ud, u32 primitive, u32 first,
                                     u32 count, const u32* immediate)
{
    /* RSX_LIVE_FEED_DBG=1: pair with the [feed] method counts so "the FIFO
     * carries no draw batches" can be checked against "the backend is told
     * to draw" in the same run. */
    { static int dbg = -1; if (dbg < 0) dbg = getenv("RSX_LIVE_FEED_DBG") ? 1 : 0;
      if (dbg) { static long long n = 0;
          if (((++n) % 20000) == 0)
              printf("[backend] indexed callbacks=%lld (prim=%u count=%u)\n", n, primitive, count), fflush(stdout); } }
    (void)ud;
    static int log_count = 0;
    static int trace_draws = -1;
    if (trace_draws < 0) trace_draws = getenv("RSX_DRAW_TRACE") != NULL;
    if (trace_draws && log_count < 8) {
        printf("[D3D12] draw_indexed%s(prim=%u, first=%u, count=%u)\n",
               immediate ? "_immediate" : "", primitive, first, count);
        log_count++;
    }
    s_d3d12_stats.calls++;
    if (!s_d3d.pipeline_ready) D3D12_DROP(D3D12_DROP_NOT_READY);
    if (count == 0) D3D12_DROP(D3D12_DROP_BAD_COUNT);
    s_req_verts += count; s_req_draws++;
    if (count > MAX_VERTICES) {
        s_drop_input_elements += count; s_drop_draws++;
        D3D12_DROP(D3D12_DROP_BAD_COUNT);
    }
    if (!s_d3d.vp_vb_mapped || !s_d3d.vp_root_sig)
        D3D12_DROP(D3D12_DROP_NO_VP_RESOURCES);
    if (!d3d12_acquire_submission_slot()) return;

    /* Raw GPU attribute fetch and the CPU fallback share triangle-list ordering. */
    const int vp_vs_idx = vp_get_vs(s_d3d.current_rsx_state);
    vp_select_upload_layout(vp_vs_idx, s_d3d.current_rsx_state);
    u32 emitted = 0;
    u32 rec = s_d3d.vp_vb_offset;
    u32 raw_bytes = 0, raw_index_offset = 0;
    emitted = upload_vertices_raw(s_d3d.current_rsx_state, primitive, first, count,
        immediate, 1, vp_vs_idx, &raw_bytes, &rec, &raw_index_offset);
    if (!emitted) {
        rsx_vertex_fetch_prepare(&s_vertex_fetch_prepared, s_d3d.current_rsx_state);
        if (s_d3d.current_rsx_state && s_d3d.current_rsx_state->restart_index_enable)
            emitted = upload_indexed_with_restart(
                s_d3d.current_rsx_state, primitive, first, count, immediate);
        else if (primitive == RSX_PRIMITIVE_QUADS)
            emitted = upload_quads_vp_indexed(
                s_d3d.current_rsx_state, first, count, immediate);
        else if (primitive == RSX_PRIMITIVE_TRIANGLES)
            emitted = upload_tris_vp_indexed(
                s_d3d.current_rsx_state, first, count, immediate);
        else if (primitive == RSX_PRIMITIVE_TRIANGLE_STRIP)
            emitted = upload_strip_vp_indexed(
                s_d3d.current_rsx_state, first, count, 0, immediate);
        else if (primitive == RSX_PRIMITIVE_TRIANGLE_FAN)
            emitted = upload_strip_vp_indexed(
                s_d3d.current_rsx_state, first, count, 1, immediate);
        else {
            static int _skip = 0;
            if (PS3_LOG_FIRST(_skip, 3))
                printf("[D3D12] draw_indexed: skipping prim=%u (not wired)\n", primitive);
            return;
        }
    }
    /* Same BEGIN/END batch concatenation as the non-indexed path: this title's
     * fluid arrives as ~3000 DRAW_INDEX_ARRAY batches of 256, and 256 is not a
     * multiple of 3, so every batch after the first assembled out of phase. */
    if (emitted && !immediate && s_d3d.merge_prev_draw && s_d3d.draw_count > 0 &&
        s_d3d.merge_first_end == first && primitive == RSX_PRIMITIVE_TRIANGLES) {
        D3D12DrawRecord* pv = &s_d3d.draws[s_d3d.draw_count - 1];
        u32 fpnow = s_d3d.current_rsx_state
                    ? s_d3d.current_rsx_state->shader_program : 0;
        if (!raw_bytes && !pv->vp_raw_bytes && pv->is_vp && pv->topology == RSX_TOPOLOGY_TRIANGLES &&
            pv->vs_idx == vp_vs_idx && pv->fp_addr == fpnow &&
            pv->flat_shading == (s_d3d.current_rsx_state &&
                s_d3d.current_rsx_state->shade_mode == CELL_GCM_FLAT) &&
            pv->two_sided_lighting == (s_d3d.current_rsx_state &&
                s_d3d.current_rsx_state->two_sided_lighting) &&
            pv->begin_epoch == (s_d3d.current_rsx_state
                                ? s_d3d.current_rsx_state->begin_epoch : 0) &&
            pv->vp_input_mask == s_d3d.vp_upload_mask &&
            pv->vp_stride == s_d3d.vp_upload_stride &&
            pv->vb_byte_offset + pv->vertex_count * pv->vp_stride == rec) {
            pv->vertex_count += emitted;
            s_d3d.merge_first_end = first + count;
            return;
        }
    }
    if (emitted && d3d12_draw_record_available()) {
        D3D12DrawRecord* dr = &s_d3d.draws[s_d3d.draw_count];
        dr->vb_byte_offset = rec;
        dr->vertex_count   = emitted;
        dr->topology       = RSX_TOPOLOGY_TRIANGLES;
        dr->textured       = s_d3d.tex_bound;
        dr->is_vp          = 1;
        dr->fp_addr = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->shader_program : 0;
        dr->fp_exp32 = s_d3d.current_rsx_state ?
            ((s_d3d.current_rsx_state->shader_control & 0x40) != 0) : 1;
        dr->cull = rsx_cull_key(s_d3d.current_rsx_state);
        dr->begin_epoch = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->begin_epoch : 0;
        dr->cmask = 0xF;
        if (s_d3d.current_rsx_state) {
            u32 _cm = s_d3d.current_rsx_state->color_mask;
            dr->cmask = ((_cm & 0x00010000) ? 1u : 0u)   /* R */
                      | ((_cm & 0x00000100) ? 2u : 0u)   /* G */
                      | ((_cm & 0x00000001) ? 4u : 0u)   /* B */
                      | ((_cm & 0x01000000) ? 8u : 0u);  /* A */
        }
        dr->alpha_ctl = 0;
        if (s_d3d.current_rsx_state)
            dr->alpha_ctl = ((s_d3d.current_rsx_state->alpha_test_enable ? 1u : 0u) << 16)
                          | ((s_d3d.current_rsx_state->alpha_func & 0xFFu) << 8)
                          | (s_d3d.current_rsx_state->alpha_ref & 0xFFu);
        for (int _u = 0; _u < RSX_TEXTURE_UNITS; _u++) {
            dr->tex[_u].off = s_d3d.cur_texs[_u].off;
            dr->tex[_u].raw = s_d3d.cur_texs[_u].raw;
            dr->tex[_u].w   = s_d3d.cur_texs[_u].w;
            dr->tex[_u].h   = s_d3d.cur_texs[_u].h;
            dr->tex[_u].fmt = s_d3d.cur_texs[_u].fmt;
                    dr->tex[_u].ctrl1 = s_d3d.cur_texs[_u].ctrl1;
                    dr->tex[_u].cube  = s_d3d.cur_texs[_u].cube;
                    dr->tex[_u].mips  = s_d3d.cur_texs[_u].mips;
            dr->tex[_u].set = s_d3d.cur_texs[_u].set;
            dr->tex_rt[_u]  = -1;
        }
        dr->tex_slot = -1;
        dr->vs_idx = vp_vs_idx;
        dr->vp_input_mask = s_d3d.vp_upload_mask;
        dr->vp_stride = s_d3d.vp_upload_stride;
            dr->vp_raw_bytes = raw_bytes;
                dr->vp_raw_index_offset = raw_index_offset;
        dr->is_clear = 0;
        dr->blend = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->blend_enable : 1;
        dr->blend_key = rsx_blend_key(s_d3d.current_rsx_state, dr->blend);
        dr_capture_extended_rsx_state(dr, s_d3d.current_rsx_state);
        dr->rt_off = current_rt_off(&dr->rt_w, &dr->rt_h, dr->rt_mrt);
        dr->rt_fmt = s_d3d.current_rsx_state ? s_d3d.current_rsx_state->surface_format : 0;
        if (s_d3d.current_rsx_state) {
            dr->vp_x = s_d3d.current_rsx_state->viewport_x;
            dr->vp_y = s_d3d.current_rsx_state->viewport_y;
            dr->vp_w = s_d3d.current_rsx_state->viewport_w;
            dr->vp_h = s_d3d.current_rsx_state->viewport_h;
        } else { dr->vp_x = dr->vp_y = dr->vp_w = dr->vp_h = 0; }
        dr->cb_slot = s_d3d.draw_count;
                vp_record_cb(s_d3d.draw_count, dr->vs_idx, dr);
        s_d3d.draw_count++;
        /* Anchor for merging the next batch of this indexed stream. */
        s_d3d.merge_prev_draw = !immediate &&
            (primitive == RSX_PRIMITIVE_TRIANGLES);
        s_d3d.merge_first_end = first + count;
    }
}

static void d3d12_draw_indexed(void* ud, u32 primitive, u32 first, u32 count)
{
    d3d12_draw_indexed_source(ud, primitive, first, count, NULL);
}

static void d3d12_draw_indexed_immediate(void* ud, u32 primitive,
                                         const u32* indices, u32 count)
{
    if (!indices || !count) return;
    d3d12_draw_indexed_source(ud, primitive, 0, count, indices);
}

static void d3d12_bind_texture(void* ud, u32 unit, const rsx_texture_state* tex)
{
    (void)ud;
    extern uint8_t* vm_base;
    extern u32 cellGcmResolveOffset(u32);

    u32 width  = (tex->image_rect >> 16) & 0xFFFF;
    u32 height = tex->image_rect & 0xFFFF;
    u32 format = (tex->format >> 8) & 0xFF;
    u32 offset = tex->offset;
    /* TEXFMT_ALL=1: every distinct texture a draw binds, whatever its format.
     * TEXFMTDBG logs only the formats this backend binds; a format it drops
     * (the draw then samples nothing) never shows up there. */
    { static int ta = -1;
      if (ta < 0) ta = getenv("TEXFMT_ALL") ? 1 : 0;
      if (ta) {
          static u32 seen[256]; static int ns = 0; int known = 0;
          const u32 key = offset ^ (format << 24) ^ unit;
          for (int k = 0; k < ns; k++) if (seen[k] == key) known = 1;
          if (!known && ns < 256) { seen[ns++] = key;
              fprintf(stderr, "[TEXFMT_ALL] unit=%u fmt=0x%02X %ux%u pitch=%u off=0x%08X "
                              "remap=0x%08X ctrl0=0x%08X filter=0x%08X%c",
                      unit, format, (tex->image_rect >> 16) & 0xFFFF, tex->image_rect & 0xFFFF,
                      tex->control3 & 0xFFFFF, offset, tex->control1, tex->control0, tex->filter, 10); }
      } }
    /* CUBEDBG=1: NV4097_SET_TEXTURE_FORMAT bit 2 is the cubemap flag, and the
     * line above throws it away with the rest of the low byte. Nothing in this
     * backend handles cube textures, so a cubemap bound for an environment
     * reflection is sampled as a plain 2D image -- which is what the chrome
     * faucet's black patches and banded escutcheons look like. */
    { static int cd = -1;
      if (cd < 0) { const char* e = getenv("CUBEDBG"); cd = e ? atoi(e) : 0; }
      if (cd && (tex->format & 4)) {
          static u32 seen[8]; static int ns = 0; int known = 0;
          for (int k = 0; k < ns; k++) if (seen[k] == offset) known = 1;
          if (!known && ns < 8) { seen[ns++] = offset;
              fprintf(stderr, "[CUBE] unit=%u offset=0x%08X %ux%u fmt=0x%02X"
                              " raw_format=0x%08X (cubemap bit SET)%c",
                      unit, offset, width, height, format, tex->format, 10); }
      } }

    static int log_count = 0;
    if (log_count < 10) {
        printf("[D3D12] bind_texture(unit=%u, offset=0x%X, fmt=0x%02X, %ux%u)\n",
               unit, offset, format, width, height);
        log_count++;
    }
    /* MOVIE_BIND=1: trace movie-plane binds (640x360 Y / 320x180 U/V) with the
     * resolved EA + a content probe -- used to diagnose the Bink frame-buffer
     * ring mismatch (the draw binds a cleared buffer 0x4D80 before the one the
     * decoder actually fills). */
    static int movie_bind_trace = -1;
    if (movie_bind_trace < 0) movie_bind_trace = getenv("MOVIE_BIND") != NULL;
    if (movie_bind_trace &&
        ((width == 640 && height == 360) || (width == 320 && height == 180))) {
        extern u32 cellGcmResolveLocated(int local, u32 offset);
        static int _mv = 0; if (PS3_LOG_FIRST(_mv, 24)) {
            /* Resolve the SAME raw offset both ways -- as LOCAL (VRAM) and as MAIN
             * (IO-mapped) -- and count nonzero bytes over the whole plane for each.
             * If the plane the game declares (loc bits fmt&3) is zero but the OTHER
             * pool has content, the decoder wrote to a different memory space than
             * our texture resolve picked (the local-vs-main mismatch). */
            u32 loc = tex->format & 3;               /* 1=LOCAL, 2=MAIN */
            u32 ea_loc  = cellGcmResolveLocated(1, offset);
            u32 ea_main = cellGcmResolveLocated(0, offset);
            u32 plane = width * height;
            u32 nz_loc = 0, nz_main = 0;
            if (vm_base) for (u32 i = 0; i < plane; i += 137) {
                if (vm_base[ea_loc  + i]) nz_loc++;
                if (vm_base[ea_main + i]) nz_main++;
            }
            fprintf(stderr, "[mv-bind] unit=%u fmt=0x%02X loc=%u %ux%u off=0x%X | LOCAL ea=0x%08X nz=%u | MAIN ea=0x%08X nz=%u\n",
                    unit, format, loc, width, height, offset,
                    ea_loc, nz_loc, ea_main, nz_main);
        }
    }

    if (!vm_base || width == 0 || height == 0) return;

    /* Record the currently-bound atlas so subsequent quad draws sample it. The
     * actual GPU upload happens in render_frame (we have no open command list
     * here). Only the 8-bit single-channel font atlas (B8, RSX fmt base 0x81 /
     * as-seen 0xA1 with the LN flag) is wired up so far; other formats fall
     * back to untextured. */
    u32 base_fmt = format & 0x9F;   /* strip LN(0x20)/UN(0x40) flags */
    /* VP path: record the latest bound texture (any supported format) so draws
     * can carry it per-draw. Location bits (format[1:0]): 1 = LOCAL, 2 = MAIN. */
    extern u32 cellGcmResolveLocated(int local, u32 offset);
    if (unit < RSX_TEXTURE_UNITS && vp_texture_format_supported(base_fmt)) {
        /* TEX_RESOLVE_AUTO=1: resolve through the page tables (local-page map
         * then IO table) instead of trusting the format's location bits. A
         * texture the guest built in main memory but tagged local resolves to
         * untouched VRAM and samples as all-zero -- geometry renders, flat. */
        { static int _ra = -1; if (_ra < 0) _ra = getenv("TEX_RESOLVE_AUTO") ? 1 : 0;
          extern u32 cellGcmResolveIO(u32);
          u32 _r = 0;
          if (_ra) _r = cellGcmResolveIO(offset);          /* IO table first */
          if (!_r) _r = cellGcmResolveLocated((tex->format & 3) == 1, offset);
          /* TEX_PROBE=1: the location bits are not always right. This title
           * binds its menu textures with a location that resolves into main
           * memory, where the bytes are all zero, while the same raw offset
           * resolved as LOCAL lands on real data -- so 100k menu draws sample
           * black and the menu never appears. When the chosen resolution is
           * empty over a sample and the other one is not, take the other. */
          { static int _tp = -1;
            if (_tp < 0) _tp = getenv("TEX_PROBE") ? 1 : 0;
            if (_tp && vm_base && _r) {
                u32 alt = cellGcmResolveLocated((tex->format & 3) != 1, offset);
                if (alt && alt != _r && alt < 0xE0000000u && _r < 0xE0000000u) {
                    u32 span = width * height / 2;
                    if (span > 4096) span = 4096;
                    u32 nz_r = 0, nz_a = 0;
                    for (u32 i = 0; i < span; i += 7) {
                        if (vm_base[_r  + i]) nz_r++;
                        if (vm_base[alt + i]) nz_a++;
                    }
                    if (nz_r == 0 && nz_a > 0) {
                        static u32 seen[32]; static int ns = 0; int known = 0;
                        for (int k = 0; k < ns; k++) if (seen[k] == offset) known = 1;
                        if (!known && ns < 32) { seen[ns++] = offset;
                            fprintf(stderr, "[TEX_PROBE] 0x%08X: %s empty, using %s"
                                            " (%u non-zero)%c", offset,
                                    ((tex->format & 3) == 1) ? "local" : "main",
                                    ((tex->format & 3) == 1) ? "main" : "local",
                                    nz_a, 10); }
                        _r = alt;
                    }
                }
            } }
          s_d3d.cur_texs[unit].off = _r; }
        s_d3d.cur_texs[unit].raw = offset;
        s_d3d.cur_texs[unit].w = width; s_d3d.cur_texs[unit].h = height;
        s_d3d.cur_texs[unit].fmt = format;   /* full byte: LN(0x20)/UN(0x40) kept */
        s_d3d.cur_texs[unit].ctrl1 = tex->control1;
        s_d3d.cur_texs[unit].cube  = (tex->format & 4) ? 1 : 0;
        s_d3d.cur_texs[unit].mips  = (tex->format >> 16) & 0xFFFFu;
        if (d3d12_setting_texfmtdbg()) { static u32 seen[64]; static int ns=0;
            u32 k = (unit<<24) | (((tex->format>>8) & 0xFFu)<<16) | (width & 0xFFFFu);
            int f=0; for (int i=0;i<ns;i++) if (seen[i]==k) f=1;
            if (!f && ns < 64) { seen[ns++]=k;
                fprintf(stderr, "[TEXFMT] unit=%u fmt=0x%02X %ux%u mips=%u cube=%d off=0x%X%c",
                        unit, (tex->format >> 8) & 0xFFu, width, height,
                        (tex->format>>16)&0xFFFFu, (tex->format&4)?1:0, offset, 10); } }
        s_d3d.cur_texs[unit].set = 1;
    }
    if (base_fmt == 0x81 /* B8 */) {
        s_d3d.tex_src_offset = cellGcmResolveLocated((tex->format & 3) == 1, offset);
        if (s_d3d.tex_w != width || s_d3d.tex_h != height) {
            /* dims changed -> resource must be (re)created in render_frame */
            for (u32 i = 0; i < FRAME_COUNT; i++) s_d3d.tex_ready[i] = 0;
        }
        s_d3d.tex_w     = width;
        s_d3d.tex_h     = height;
        s_d3d.tex_bound = 1;
        for (u32 i = 0; i < FRAME_COUNT; i++) s_d3d.tex_dirty[i] = 1;
    } else {
        s_d3d.tex_bound = 0;
    }
}

static void d3d12_set_vertex_attribs(void* ud, const rsx_state* state)
{
    (void)ud;
    s_d3d.current_rsx_state = state;

    /* Log enabled vertex attributes for debugging. RSX_VTXDBG=<N> raises the
     * cap: the fixed 5 were all consumed by boot-time setup, so the layouts the
     * real draws use were never printed. */
    static int log_count = 0, log_cap = -1;
    if (log_cap < 0) { const char* e = getenv("RSX_VTXDBG"); log_cap = e ? atoi(e) : 5; }
    if (log_count < log_cap) {
        printf("[D3D12] set_vertex_attribs:\n");
        for (int i = 0; i < 16; i++) {
            const rsx_vertex_attrib* a = &state->vertex_attribs[i];
            if (a->enabled) {
                const char* type_name = "?";
                switch (a->type) {
                case 1: type_name = "snorm16"; break;
                case 2: type_name = "float32"; break;
                case 3: type_name = "float16"; break;
                case 4: type_name = "ubyte"; break;
                case 5: type_name = "s16"; break;
                case 7: type_name = "ubyte256"; break;
                }
                printf("  attrib[%d]: %s x%u, stride=%u, offset=0x%X\n",
                       i, type_name, a->size, a->stride, a->offset);
            }
        }
        log_count++;
    }
}

static void d3d12_set_shader(void* ud, const rsx_state* state)
{
    (void)ud;
    s_d3d.current_rsx_state = state;

    static int log_count = 0;
    if (log_count < 5) {
        printf("[D3D12] set_shader: fp_addr=0x%08X, vp_load=%u, output_mask=0x%08X\n",
               state->fragment_program_addr, state->transform_program_load,
               state->vertex_attrib_output_mask);
        log_count++;
    }

    /* TODO: Look up or compile a PSO matching this shader combination.
     * For now we use the basic vertex-colored PSO for everything. */
}

static void d3d12_set_blend(void* ud, const rsx_state* state)
{
    (void)ud;
    /* TODO: modify PSO blend state or use dynamic state.
     * D3D12 requires PSO recreation for blend state changes,
     * so we'd need a PSO cache keyed by blend configuration. */
    static int log_count = 0;
    if (log_count < 5) {
        printf("[D3D12] set_blend(enable=%d, sfactor=0x%X, dfactor=0x%X)\n",
               state->blend_enable, state->blend_sfactor, state->blend_dfactor);
        log_count++;
    }
}

static void d3d12_set_depth_stencil(void* ud, const rsx_state* state)
{
    (void)ud;
    static int log_count = 0;
    if (log_count < 5) {
        printf("[D3D12] set_depth_stencil(depth=%d, stencil=%d, func=0x%X)\n",
               state->depth_test_enable, state->stencil_test_enable,
               state->depth_func);
        log_count++;
    }
}

/* ---------------------------------------------------------------------------
 * Backend registration
 * -----------------------------------------------------------------------*/

static rsx_backend s_d3d12_backend = {0};

/* ---------------------------------------------------------------------------
 * Public API
 * -----------------------------------------------------------------------*/

int rsx_d3d12_backend_init(u32 width, u32 height, const char* title)
{
    memset(&s_d3d, 0, sizeof(s_d3d));
    s_present_display_id = 0;
    s_d3d.width = width;
    s_d3d.height = height;
    s_d3d.clear_color[0] = 0.0f;
    s_d3d.clear_color[1] = 0.0f;
    s_d3d.clear_color[2] = 0.1f;
    s_d3d.clear_color[3] = 1.0f;
    s_d3d.clear_flags = RSX_CLEAR_COLOR_MASK | RSX_CLEAR_DEPTH_MASK | RSX_CLEAR_STENCIL_MASK;
    s_d3d.clear_depth = 1.0f;
    s_d3d.clear_stencil = 0;

    /* Debug: dump the first N presented frames to BMP if CELLMARK_DUMP is set
     * (its numeric value when > 1, else 24). */
    {
        const char* dv = getenv("CELLMARK_DUMP");
        int n = dv ? atoi(dv) : 0;
        s_d3d.dump_frames_left = dv ? (n > 1 ? n : 24) : 0;
        /* CELLMARK_DUMP_SKIP=N: ignore the first N presents before dumping.
         * The opening frames are the loading screen; the interesting content
         * only appears once the sim SPUs have advanced the scene. */
        { const char* sv = getenv("CELLMARK_DUMP_SKIP");
          s_d3d.dump_skip_left = sv ? atoi(sv) : 0; }
    }

    /* Create window */
    {
        extern char g_rsx_title_base[128];
        snprintf(g_rsx_title_base, sizeof(g_rsx_title_base), "%s",
                 title ? title : "ps3recomp");
    }
    s_d3d.hwnd = create_window(width, height, title);
    if (!s_d3d.hwnd) {
        printf("[D3D12] ERROR: Window creation failed\n");
        return -1;
    }

    /* Initialize D3D12 */
    if (init_d3d12(width, height) != 0) {
        printf("[D3D12] ERROR: D3D12 initialization failed\n");
        return -1;
    }

    /* Set up backend callbacks */
    s_d3d12_backend.userdata          = &s_d3d;
    s_d3d12_backend.init              = d3d12_init;
    s_d3d12_backend.shutdown          = d3d12_shutdown;
    s_d3d12_backend.begin_frame       = d3d12_begin_frame;
    s_d3d12_backend.end_frame         = d3d12_end_frame;
    s_d3d12_backend.present           = d3d12_present;
    s_d3d12_backend.clear             = d3d12_clear;
    s_d3d12_backend.set_render_target = d3d12_set_render_target;
    s_d3d12_backend.set_viewport      = d3d12_set_viewport;
    s_d3d12_backend.set_blend         = d3d12_set_blend;
    s_d3d12_backend.set_depth_stencil = d3d12_set_depth_stencil;
    s_d3d12_backend.set_shader        = d3d12_set_shader;
    s_d3d12_backend.set_vertex_attribs = d3d12_set_vertex_attribs;
    s_d3d12_backend.draw_arrays       = d3d12_draw_arrays;
    s_d3d12_backend.draw_indexed      = d3d12_draw_indexed;
    s_d3d12_backend.draw_indexed_immediate = d3d12_draw_indexed_immediate;
    s_d3d12_backend.wait_idle         = d3d12_wait_idle;
    s_d3d12_backend.bind_texture      = d3d12_bind_texture;

    rsx_set_backend(&s_d3d12_backend);

    s_d3d.initialized = 1;
    s_d3d.last_fps_time = GetTickCount64();
    rsx_vertex_workers_start();

    printf("[D3D12] Backend ready: %ux%u (vertex workers=%lu)\n",
           width, height, (unsigned long)s_vtx_worker_count);
    return 0;
}

void rsx_d3d12_backend_shutdown(void)
{
    rsx_vertex_workers_stop();
    rsx_raw_upload_destroy(&s_raw_upload_cache);
    free(s_raw_vertex_ids);
    s_raw_vertex_ids = NULL; s_raw_vertex_id_capacity = 0;
    for (int i = 0; i < s_d3d.vp_vs_n; ++i) {
        if (s_d3d.vp_vs[i].vs) s_d3d.vp_vs[i].vs->lpVtbl->Release(s_d3d.vp_vs[i].vs);
        if (s_d3d.vp_vs[i].raw_vs) s_d3d.vp_vs[i].raw_vs->lpVtbl->Release(s_d3d.vp_vs[i].raw_vs);
    }
    free(s_d3d.vp_vs);
    s_d3d.vp_vs = NULL;
    s_d3d.vp_vs_n = s_d3d.vp_vs_capacity = 0;
    memset(s_d3d.vp_vs_heads, 0, sizeof(s_d3d.vp_vs_heads));
    memset(s_d3d.vp_tex_heads, 0, sizeof(s_d3d.vp_tex_heads));
    memset(s_d3d.vp_tex_next, 0, sizeof(s_d3d.vp_tex_next));
    memset(s_d3d.vp_tex_bucket, 0, sizeof(s_d3d.vp_tex_bucket));
    if (!s_d3d.initialized) return;

    wait_for_gpu();
    d3d12_collect_retired(1);
    rsx_gpu_timer_destroy(&s_gpu_timer);

    d3d12_pipeline_cache_save();
    rsx_d3d12_srv_cache_shutdown(&s_srv_cache);

    /* Release D3D12 resources */
    for (int i = 0; i < MAX_OFF_RTS; ++i) {
        OffRT* r = &s_d3d.off_rt[i];
        if (r->feedback) r->feedback->lpVtbl->Release(r->feedback);
        if (r->up) r->up->lpVtbl->Release(r->up);
        if (r->res) r->res->lpVtbl->Release(r->res);
        memset(r, 0, sizeof(*r));
    }
    for (int i = 0; i < s_d3d.vp_fp_n; i++) {
        if (s_d3d.vp_fp[i].pso)
            s_d3d.vp_fp[i].pso->lpVtbl->Release(s_d3d.vp_fp[i].pso);
        s_d3d.vp_fp[i].pso = NULL;
    }
    s_d3d.vp_fp_n = 0;
    memset(s_d3d.vp_fp_hash, 0, sizeof(s_d3d.vp_fp_hash));
    s_d3d.vp_fp_next_replace = 0;
    for (int i = 0; i < s_d3d.vp_fp_blob_n; i++) {
        if (s_d3d.vp_fp_blob[i].blob)
            s_d3d.vp_fp_blob[i].blob->lpVtbl->Release(s_d3d.vp_fp_blob[i].blob);
        free(s_d3d.vp_fp_blob[i].source);
        s_d3d.vp_fp_blob[i].blob = NULL;
        s_d3d.vp_fp_blob[i].source = NULL;
    }
    s_d3d.vp_fp_blob_n = 0;
    if (s_d3d.vertex_buffer) {
        s_d3d.vertex_buffer->lpVtbl->Unmap(s_d3d.vertex_buffer, 0, NULL);
        s_d3d.vertex_buffer->lpVtbl->Release(s_d3d.vertex_buffer);
    }
    if (s_d3d.pipeline_state)        s_d3d.pipeline_state->lpVtbl->Release(s_d3d.pipeline_state);
    if (s_d3d.pipeline_state_lines)  s_d3d.pipeline_state_lines->lpVtbl->Release(s_d3d.pipeline_state_lines);
    if (s_d3d.pipeline_state_points) s_d3d.pipeline_state_points->lpVtbl->Release(s_d3d.pipeline_state_points);
    if (s_d3d.pipeline_state_vp_z16) s_d3d.pipeline_state_vp_z16->lpVtbl->Release(s_d3d.pipeline_state_vp_z16);
    if (s_d3d.pipeline_state_vp_color_z16) s_d3d.pipeline_state_vp_color_z16->lpVtbl->Release(s_d3d.pipeline_state_vp_color_z16);
    if (s_d3d.pipeline_state_vp_no_rt) s_d3d.pipeline_state_vp_no_rt->lpVtbl->Release(s_d3d.pipeline_state_vp_no_rt);
    if (s_d3d.pipeline_state_vp_no_rt_z16) s_d3d.pipeline_state_vp_no_rt_z16->lpVtbl->Release(s_d3d.pipeline_state_vp_no_rt_z16);
    if (s_d3d.pipeline_state_vp_color_no_rt) s_d3d.pipeline_state_vp_color_no_rt->lpVtbl->Release(s_d3d.pipeline_state_vp_color_no_rt);
    if (s_d3d.pipeline_state_vp_color_no_rt_z16) s_d3d.pipeline_state_vp_color_no_rt_z16->lpVtbl->Release(s_d3d.pipeline_state_vp_color_no_rt_z16);
    for (int i = 0; i < MAX_OFF_DEPTHS; i++)
        if (s_d3d.off_depth[i].res)
            s_d3d.off_depth[i].res->lpVtbl->Release(s_d3d.off_depth[i].res);
    if (s_d3d.depth_buffer) s_d3d.depth_buffer->lpVtbl->Release(s_d3d.depth_buffer);
    if (s_d3d.dsv_heap)     s_d3d.dsv_heap->lpVtbl->Release(s_d3d.dsv_heap);
    for (u32 i = 0; i < FRAME_COUNT; i++) {
        if (s_d3d.sampler_heap[i]) s_d3d.sampler_heap[i]->lpVtbl->Release(s_d3d.sampler_heap[i]);
        if (s_d3d.tex_resource[i]) s_d3d.tex_resource[i]->lpVtbl->Release(s_d3d.tex_resource[i]);
        if (s_d3d.tex_upload[i]) s_d3d.tex_upload[i]->lpVtbl->Release(s_d3d.tex_upload[i]);
    }
    for (u32 i = 0; i < VP_TEX_SLOTS; i++) {
        VPTexSlot* t = &s_d3d.vp_tex[i];
        if (t->res) t->res->lpVtbl->Release(t->res);
        for (u32 slot = 0; slot < FRAME_COUNT; slot++)
            if (t->up[slot]) t->up[slot]->lpVtbl->Release(t->up[slot]);
    }
    if (s_d3d.root_signature) s_d3d.root_signature->lpVtbl->Release(s_d3d.root_signature);
    if (s_d3d.fence) s_d3d.fence->lpVtbl->Release(s_d3d.fence);
    if (s_d3d.fence_event) CloseHandle(s_d3d.fence_event);
    if (s_d3d.cmd_list) s_d3d.cmd_list->lpVtbl->Release(s_d3d.cmd_list);
    for (u32 i = 0; i < FRAME_COUNT; i++) {
        if (s_d3d.cmd_allocators[i]) s_d3d.cmd_allocators[i]->lpVtbl->Release(s_d3d.cmd_allocators[i]);
        if (s_d3d.render_targets[i]) s_d3d.render_targets[i]->lpVtbl->Release(s_d3d.render_targets[i]);
    }
    if (s_d3d.rtv_heap) s_d3d.rtv_heap->lpVtbl->Release(s_d3d.rtv_heap);
    if (s_d3d.swap_chain) s_d3d.swap_chain->lpVtbl->Release(s_d3d.swap_chain);
    if (s_d3d.cmd_queue) s_d3d.cmd_queue->lpVtbl->Release(s_d3d.cmd_queue);
    if (s_d3d.pipeline_library)
        s_d3d.pipeline_library->lpVtbl->Release(s_d3d.pipeline_library);
    s_d3d.pipeline_library = NULL;
    free(s_d3d.pipeline_library_data);
    s_d3d.pipeline_library_data = NULL;
    s_d3d.pipeline_library_data_size = 0;
    if (s_d3d.pipeline_library_lock_ready) {
        DeleteCriticalSection(&s_d3d.pipeline_library_lock);
        s_d3d.pipeline_library_lock_ready = 0;
    }
    if (s_d3d.device1) s_d3d.device1->lpVtbl->Release(s_d3d.device1);
    if (s_d3d.device) s_d3d.device->lpVtbl->Release(s_d3d.device);

    if (s_d3d.hwnd) DestroyWindow(s_d3d.hwnd);

    rsx_set_backend(NULL);
    s_d3d.initialized = 0;

    printf("[D3D12] Backend shut down\n");
}

int rsx_d3d12_backend_pump_messages(void)
{
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            rsx_d3d12_backend_shutdown();
            return -1;
        }
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    if (s_d3d.window_closed) {
        rsx_d3d12_backend_shutdown();
        return -1;
    }
    return 0;
}

void rsx_d3d12_backend_present(void)
{
    s_present_display_id = cellGcmGetCurrentDisplayBufferId();
    if (blink_dbg())
        printf("[PRESENT] draws=%u clears_since_last=%u clear_presents=%u\n",
               s_d3d.draw_count, s_dbg_clears_since_present, s_clear_presents);
    s_dbg_clears_since_present = 0;

    /* Once frame-boundary presents are active (d3d12_clear presents each
     * completed frame as the drain crosses into the next one), the ticker
     * present would only ever show the partially-accumulated NEXT frame --
     * that partial present right after the FIFO ring recycle was the visible
     * blink. Keep the ticker present solely as the boot-time fallback (before
     * the first framed clear arrives). */
    extern unsigned cellGcm_flip_request_count(void);
    const unsigned flip_count = cellGcm_flip_request_count();
    if (flip_count == 0 &&
        (s_clear_presents > 0 ||
         InterlockedCompareExchange(&s_accum_display_clear_seen, 0, 0) != 0))
        return;

    /* Same display gate as d3d12_present: a batch of offscreen pass work only
     * (render-to-texture) keeps accumulating until its composite arrives.
     * Empty batches present only until the first real frame -- after that an
     * empty present is a flip/drain race and wipes the screen for a frame
     * (wave: black flashes and layout flicker between frames). A display clear
     * alone is also not a completed frame: double-buffered titles clear the
     * next buffer before drawing it, and presenting that clear exposed a blank
     * frame between completed scenes. */
    static int s_seen_content = 0;
    int has_display = (s_d3d.draw_count == 0 && !s_seen_content);
    int has_display_draw = 0;
    int has_display_clear = 0;
    for (u32 _i = 0; _i < s_d3d.draw_count && _i < MAX_DRAWS; _i++)
        if (is_display_target_key(s_d3d.draws[_i].rt_off)) {
            if (s_d3d.draws[_i].is_clear) {
                has_display_clear = 1;
                continue;
            }
            has_display = 1;
            has_display_draw = 1;
            break;
        }
    /* The guest-framebuffer present needs the pass to run at all. */
    { static int gfb = -1;
      if (gfb < 0) { const char* e = getenv("GCM_GUEST_FB"); gfb = e ? 1 : 0; }
      if (gfb) { has_display = 1; has_display_draw = 1; } }
    if (s_composite_src) { has_display = 1; has_display_draw = 1; }
    if (has_display_draw) s_seen_content = 1;

    /* RSX_PRESENT_GUARD_DBG=1: confirm that a clear-only flip is suppressed
     * after a real display frame, without logging the per-frame clear stream. */
    { static int dbg = -1; static unsigned skipped = 0;
      if (dbg < 0) dbg = getenv("RSX_PRESENT_GUARD_DBG") ? 1 : 0;
      if (dbg && has_display_clear && !has_display && s_seen_content && skipped < 8) {
          ++skipped;
          fprintf(stderr, "[D3D12] skip clear-only display batch after content (%u)\n", skipped);
      } }

    /* VP_SUBMIT=<N>: has_display gates render_frame() entirely, so a batch whose
     * records all target an offscreen RT presents without ever running the draw
     * pass -- the backbuffer then shows only the clear, which
     * reads as "nothing rasterizes" from every downstream check. */
    { static int cap = -1, n = 0;
      if (cap < 0) { const char* e = getenv("VP_SUBMIT"); cap = e ? atoi(e) : 0; }
      if (cap && n < cap && s_d3d.draw_count) { n++;
        u32 onscreen = 0, offscreen = 0, clears = 0;
        for (u32 _i = 0; _i < s_d3d.draw_count && _i < MAX_DRAWS; _i++) {
            if (s_d3d.draws[_i].is_clear) { clears++; continue; }
            if (is_display_target_key(s_d3d.draws[_i].rt_off)) onscreen++; else offscreen++;
        }
        { /* Which surfaces the batch's draws actually target -- "offscreen=57"
           * does not say whether they all go to one RT or scatter. */
          u32 offs[8]; u32 cnt[8]; int nd = 0;
          for (u32 _i = 0; _i < s_d3d.draw_count && _i < MAX_DRAWS; _i++) {
              if (s_d3d.draws[_i].is_clear) continue;
              u32 o = s_d3d.draws[_i].rt_off; int f = -1;
              for (int _j = 0; _j < nd; _j++) if (offs[_j] == o) { f = _j; break; }
              if (f < 0) { if (nd < 8) { offs[nd] = o; cnt[nd] = 1; nd++; } }
              else cnt[f]++;
          }
          { u32 vp = 0, novp = 0;
            for (u32 _i = 0; _i < s_d3d.draw_count && _i < MAX_DRAWS; _i++) {
                if (s_d3d.draws[_i].is_clear) continue;
                if (s_d3d.draws[_i].is_vp) vp++; else novp++;
            }
            /* The execution loop skips anything without is_vp, so a batch full
             * of !is_vp records is counted, gated and then never drawn. */
            fprintf(stderr, "[PRESENTGATE] is_vp=%u not_vp=%u%c", vp, novp, 10); }
          fprintf(stderr, "[PRESENTGATE] targets:");
          for (int _j = 0; _j < nd; _j++)
              fprintf(stderr, " 0x%08X x%u", offs[_j], cnt[_j]);
          fprintf(stderr, "%c", 10); }
        fprintf(stderr, "[PRESENTGATE] records=%u onscreen=%u offscreen=%u clears=%u"
                        " has_display=%d seen_content=%d -> render_frame=%s\n",
                s_d3d.draw_count, onscreen, offscreen, clears, has_display,
                s_seen_content, (s_d3d.initialized && has_display) ? "YES" : "SKIPPED"); } }

    /* Execute the batch whenever it has draws. Gating the whole call on
     * has_display meant a render-to-texture pass -- every draw targeting an
     * offscreen surface -- was DISCARDED rather than deferred, so the texture
     * it produces was never written and whatever sampled it later read an
     * empty resource. Only the Present needs onscreen content. */
    if (s_d3d.initialized && (has_display || s_d3d.draw_count > 0)) {
        { extern void rsx_reset_upload_claims(void);
          static int remap = -1;
          if (remap < 0) { const char* e = getenv("TEX_REMAP"); remap = e ? atoi(e) : 0; }
          if (remap) rsx_reset_upload_claims(); }
        s_present_this_frame = has_display;
        render_frame();
        s_present_this_frame = 1;
    }
}

#else /* !_WIN32 */

#include <ps3emu/ps3types.h>   /* u32 (header includes above are inside the _WIN32 guard) */
#include <stdio.h>

/* Stub for non-Windows — D3D12 is Windows-only */
int rsx_d3d12_backend_init(u32 w, u32 h, const char* t)
{
    (void)w; (void)h; (void)t;
    printf("[D3D12] Not available on this platform (use Vulkan backend)\n");
    return -1;
}
void rsx_d3d12_backend_shutdown(void) {}
int rsx_d3d12_backend_pump_messages(void) { return 0; }
void rsx_d3d12_backend_present(void) {}

#endif /* _WIN32 */
