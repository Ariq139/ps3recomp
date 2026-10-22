/*
 * ps3recomp - RSX Command Buffer Processor
 *
 * The PS3's RSX (Reality Synthesizer) is an NV47-class GPU that receives
 * rendering commands through a ring buffer. Games write NV47xx method
 * calls to this buffer via cellGcm, and the RSX processes them.
 *
 * In recompilation, we intercept the command buffer and translate
 * NV47xx methods to the host graphics API (D3D12 on Windows, Vulkan
 * on Linux/macOS).
 *
 * Architecture:
 *   Game code → cellGcmSys (state tracking) → command buffer → RSX processor
 *                                                                  ↓
 *                                                          Host GPU backend
 *
 * Reference: NV47xx method registers documented in envytools/rnndb
 * and RPCS3's RSX module (rpcs3/Emu/RSX/).
 */

#ifndef PS3RECOMP_RSX_COMMANDS_H
#define PS3RECOMP_RSX_COMMANDS_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * NV47xx Method Register Addresses
 *
 * These are the GPU method addresses that games write to the command buffer.
 * Each method has a 32-bit address and one or more 32-bit data words.
 *
 * Naming: NV4097_SET_* for 3D methods, NV3062_SET_* for 2D methods
 * -----------------------------------------------------------------------*/

/* Surface / render target configuration (real NV4097 numbering -- RPCS3 /
 * nouveau. The original table here was shuffled: only COLOR_AOFFSET was
 * right, so SET_SURFACE_COLOR_TARGET (MRT selection) was never decoded and
 * clip dims read the format register). */
/* Which context DMA a colour surface's offset is an offset INTO: handle
 * 0xFEED0000 is RSX local memory and 0xFEED0001 is IO-mapped main memory.
 * RSX offsets are two overlapping number spaces, so this is what tells a
 * surface at offset X from a texture at offset X in the other one. Decoded
 * by rsx_dispatch (rsx_dsp_surface.color_location); rsx_state has no field
 * for it and rsx_process_method ignores it. */
#define NV4097_SET_CONTEXT_DMA_COLOR_A         0x00000194
/* ...and the same for colour target B, which sits BELOW A in the register
 * file rather than after it (C and D are at 0x1B4 and 0x1B8). An MRT set
 * whose members disagree about which memory they live in is one the draw
 * engine cannot match a texture unit against, since it keys surfaces by
 * (location, offset). */
#define NV4097_SET_CONTEXT_DMA_COLOR_B         0x0000018C
/* ...and the same question for the depth buffer, which a texture unit coming
 * back to sample it has to agree with. */
#define NV4097_SET_CONTEXT_DMA_ZETA            0x00000198
#define CELL_GCM_CONTEXT_DMA_MEMORY_FRAME_BUFFER 0xFEED0000
#define CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER  0xFEED0001
#define NV4097_SET_CONTEXT_DMA_REPORT           0x000001A8

#define NV4097_SET_SURFACE_CLIP_HORIZONTAL     0x00000200
#define NV4097_SET_SURFACE_CLIP_VERTICAL       0x00000204
#define NV4097_SET_SURFACE_FORMAT              0x00000208
#define NV4097_SET_SURFACE_PITCH_A             0x0000020C
#define NV4097_SET_SURFACE_COLOR_AOFFSET       0x00000210
#define NV4097_SET_SURFACE_ZETA_OFFSET         0x00000214
#define NV4097_SET_SURFACE_COLOR_BOFFSET       0x00000218
#define NV4097_SET_SURFACE_PITCH_B             0x0000021C
#define NV4097_SET_SURFACE_COLOR_TARGET        0x00000220
#define NV4097_SET_SURFACE_PITCH_Z             0x0000022C
#define NV4097_SET_SURFACE_PITCH_C             0x00000280
#define NV4097_SET_SURFACE_PITCH_D             0x00000284
#define NV4097_SET_SURFACE_COLOR_COFFSET       0x00000288
#define NV4097_SET_SURFACE_COLOR_DOFFSET       0x0000028C

/* SET_SURFACE_COLOR_TARGET values */
#define CELL_GCM_SURFACE_TARGET_NONE  0x00
#define CELL_GCM_SURFACE_TARGET_0     0x01
#define CELL_GCM_SURFACE_TARGET_1     0x02
#define CELL_GCM_SURFACE_TARGET_MRT1  0x13   /* A + B     */
#define CELL_GCM_SURFACE_TARGET_MRT2  0x17   /* A + B + C */
#define CELL_GCM_SURFACE_TARGET_MRT3  0x1F   /* A+B+C+D   */

/* Viewport (real NV4097 numbering -- the old 0x300/0x304 values actually
 * belong to the dither/alpha-test block, so viewport rects read garbage). */
#define NV4097_SET_VIEWPORT_HORIZONTAL         0x00000A00
#define NV4097_SET_VIEWPORT_VERTICAL           0x00000A04
#define NV4097_SET_CLIP_MIN                    0x00000394
#define NV4097_SET_CLIP_MAX                    0x00000398

/* Scissor */
#define NV4097_SET_SCISSOR_HORIZONTAL          0x000008C0
#define NV4097_SET_SCISSOR_VERTICAL            0x000008C4

/* RPCS3 behavior reference (GPL-2.0): gcm_enums.h / NV47/HW/nv4097.cpp.
 * Register/value facts only; the local method handler is independently written. */
#define NV4097_NOTIFY                           0x00000104
#define CELL_GCM_CONTEXT_DMA_NOTIFY_MAIN_0      0x6660420Fu
#define NV4097_WAIT_FOR_IDLE                    0x00000110
#define NV406E_SET_REFERENCE                    0x00000050
#define NV406E_SEMAPHORE_OFFSET                0x00000064
#define NV406E_SEMAPHORE_ACQUIRE               0x00000068
#define NV406E_SEMAPHORE_RELEASE               0x0000006C
#define NV4097_SET_WINDOW_OFFSET               0x000002B8
#define NV4097_SET_CONTEXT_DMA_COLOR_C         0x000001B4
#define NV4097_SET_CONTEXT_DMA_COLOR_D         0x000001B8
#define NV4097_CLEAR_ZCULL_SURFACE             0x00001438
#define NV4097_CLEAR_REPORT_VALUE              0x000017C8
#define NV4097_SET_ZPASS_PIXEL_COUNT_ENABLE    0x000017CC
#define NV4097_GET_REPORT                      0x00001800
#define NV4097_SET_ZCULL_STATS_ENABLE          0x00001804
#define NV4097_SET_SEMAPHORE_OFFSET             0x00001D6C
#define NV4097_BACK_END_WRITE_SEMAPHORE_RELEASE 0x00001D70
#define NV4097_TEXTURE_READ_SEMAPHORE_RELEASE  0x00001D74
#define NV4097_SET_ZCULL_EN                    0x00001D84
#define NV4097_SET_RENDER_ENABLE               0x00001E98
#define CELL_GCM_ZPASS_PIXEL_CNT                1u
#define CELL_GCM_ZCULL_STATS                    2u
#define CELL_GCM_ZCULL_STATS1                   3u
#define CELL_GCM_ZCULL_STATS2                   4u
#define CELL_GCM_ZCULL_STATS3                   5u

/* RPCS3 behavior reference (GPL-2.0): gcm_enums.h / rsx_methods.h.
 * Register facts only; implementation below is independently written. */
#define NV4097_SET_TEX_COORD_CONTROL           0x00000B40
#define NV4097_INVALIDATE_VERTEX_CACHE_FILE    0x00001710
#define NV4097_INVALIDATE_VERTEX_FILE          0x00001714
#define NV4097_SET_ANTI_ALIASING_CONTROL       0x00001D7C
#define NV4097_SET_TRANSFORM_TIMEOUT           0x00001EF8

/* Clear */
#define NV4097_SET_COLOR_CLEAR_VALUE           0x00001D90
#define NV4097_SET_ZSTENCIL_CLEAR_VALUE        0x00001D8C
#define NV4097_CLEAR_SURFACE                   0x00001D94

/* Draw commands */
#define NV4097_SET_BEGIN_END                    0x00001808
#define NV4097_ARRAY_ELEMENT16                  0x0000180C
#define NV4097_ARRAY_ELEMENT32                  0x00001810
#define NV4097_DRAW_ARRAYS                     0x00001814
#define NV4097_INLINE_ARRAY                    0x00001818
#define NV4097_SET_INDEX_ARRAY_ADDRESS         0x0000181C
#define NV4097_SET_INDEX_ARRAY_DMA             0x00001820
#define NV4097_DRAW_INDEX_ARRAY                0x00001824

/* Vertex attributes */
#define NV4097_SET_VERTEX_DATA_ARRAY_FORMAT     0x00001740
#define NV4097_SET_VERTEX_DATA_ARRAY_OFFSET     0x00001680
#define NV4097_SET_VERTEX_DATA_BASE_OFFSET      0x00001738
#define NV4097_SET_VERTEX_DATA_BASE_INDEX       0x0000173C
#define NV4097_SET_VERTEX_DATA3F_M              0x00001500
#define NV4097_SET_VERTEX_DATA2F_M              0x00001880
#define NV4097_SET_VERTEX_DATA2S_M              0x00001900
#define NV4097_SET_VERTEX_DATA4UB_M             0x00001940
#define NV4097_SET_VERTEX_DATA4S_M              0x00001980
#define NV4097_SET_VERTEX_DATA_SCALED4S_M       0x00000A80
#define NV4097_SET_VERTEX_DATA4F_M              0x00001C00
#define NV4097_SET_VERTEX_DATA1F_M              0x00001E40

/* Texture */
#define NV4097_SET_TEXTURE_CONTROL2             0x00000B00 /* 16 units, 4 bytes each */
#define NV4097_SET_TEXTURE_OFFSET               0x00001A00
#define NV4097_SET_TEXTURE_FORMAT               0x00001A04
#define NV4097_SET_TEXTURE_ADDRESS              0x00001A08
#define NV4097_SET_TEXTURE_CONTROL0             0x00001A0C
#define NV4097_SET_TEXTURE_CONTROL1             0x00001A10
#define NV4097_SET_TEXTURE_FILTER               0x00001A14
#define NV4097_SET_TEXTURE_IMAGE_RECT           0x00001A18
#define NV4097_SET_TEXTURE_BORDER_COLOR         0x00001A1C
/* CONTROL3 sits in its own block, 4 bytes per unit rather than 0x20, because
 * it was added to the register file after the per-unit blocks were laid out.
 * Its low 20 bits are the row pitch of a linear texture and the rest is the
 * depth of a 3D one. */
#define NV4097_SET_TEXTURE_CONTROL3             0x00001840

/* Vertex textures: four units of eight words at 0x0900 + unit*0x20. The block
 * is laid out like a fragment unit's with one substitution -- CONTROL3 sits
 * where CONTROL1 does, so a vertex unit has a row pitch and no component
 * crossbar. A transform program's TXL instruction samples these. */
#define NV4097_SET_VERTEX_TEXTURE_OFFSET        0x00000900
#define NV4097_SET_VERTEX_TEXTURE_FORMAT        0x00000904
#define NV4097_SET_VERTEX_TEXTURE_ADDRESS       0x00000908
#define NV4097_SET_VERTEX_TEXTURE_CONTROL0      0x0000090C
#define NV4097_SET_VERTEX_TEXTURE_CONTROL3      0x00000910
#define NV4097_SET_VERTEX_TEXTURE_FILTER        0x00000914
#define NV4097_SET_VERTEX_TEXTURE_IMAGE_RECT    0x00000918
#define NV4097_SET_VERTEX_TEXTURE_BORDER_COLOR  0x0000091C

/* Shader programs */
#define NV4097_SET_SHADER_PROGRAM               0x000008E4
#define NV4097_SET_SHADER_CONTROL               0x00001D60
#define NV4097_SET_SHADER_PACKER                0x00001FEC
/* shader_control bit: fragment colour exports come from r0/r2/r3/r4 (32-bit)
 * instead of h0/h4/h6/h8 (half). */
#define CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS 0x00000040
#define NV4097_SET_VERTEX_ATTRIB_OUTPUT_MASK    0x00001FF4
#define NV4097_SET_TRANSFORM_PROGRAM_LOAD       0x00001E9C
/* Which INSTRUCTION the vertex program starts executing at. Separate from
 * _LOAD, which only says where following microcode words are written. A title
 * that keeps several programs resident in the 512-instruction store selects
 * between them with this, so ignoring it runs whichever program happens to sit
 * at instruction 0 for every draw. */
#define NV4097_SET_TRANSFORM_PROGRAM_START      0x00001EA0
#define NV4097_SET_TRANSFORM_PROGRAM            0x00000B80
#define NV4097_SET_TRANSFORM_CONSTANT_LOAD      0x00001EFC
/* Vertex constants are written as up to 32 dwords (8 vec4s) per command,
 * 0x1F00..0x1F7C. Slot = transform_constant_load + (reg/4); lane = reg%4.
 * 0x1F80..0x1FFC are other registers (below), not more constants.
 * The hardware supports 512 vec4 constants total in the vertex register file. */
#define NV4097_SET_TRANSFORM_CONSTANT           0x00001F00
#define NV4097_SET_TRANSFORM_CONSTANT_DWORDS    32
#define NV4097_INVALIDATE_L2                    0x00001FD8
#define NV4097_SET_VERTEX_ATTRIB_INPUT_MASK     0x00001FF0
#define NV4097_SET_TRANSFORM_BRANCH_BITS        0x00001FF8
#define RSX_MAX_VERTEX_CONSTANTS                512

/* Color mask */
#define NV4097_SET_COLOR_MASK                   0x00000324
#define NV4097_SET_COLOR_MASK_MRT               0x00000370

/* Alpha test */
#define NV4097_SET_ALPHA_TEST_ENABLE            0x00000304
#define NV4097_SET_ALPHA_FUNC                   0x00000308
#define NV4097_SET_ALPHA_REF                    0x0000030C

/* Flat shading applies to the front/back diffuse and specular vertex outputs. */
#define NV4097_SET_SHADE_MODE                   0x00000368
#define CELL_GCM_FLAT                           0x00001D00
#define CELL_GCM_SMOOTH                         0x00001D01
#define NV4097_SET_TWO_SIDE_LIGHT_EN            0x0000142C

/* Blending */
#define NV4097_SET_BLEND_ENABLE                 0x00000310
#define NV4097_SET_BLEND_ENABLE_MRT             0x0000036C
#define NV4097_SET_BLEND_FUNC_SFACTOR           0x00000314
#define NV4097_SET_BLEND_FUNC_DFACTOR           0x00000318
#define NV4097_SET_BLEND_EQUATION               0x00000320
#define NV4097_SET_BLEND_COLOR                  0x0000031C
#define NV4097_SET_BLEND_COLOR2                 0x0000037C

/* Depth / stencil */
#define NV4097_SET_DEPTH_TEST_ENABLE            0x00000A74
#define NV4097_SET_DEPTH_FUNC                   0x00000A6C
#define NV4097_SET_DEPTH_MASK                   0x00000A70
#define NV4097_SET_ZMIN_MAX_CONTROL              0x00001D78
#define NV4097_SET_STENCIL_TEST_ENABLE          0x00000328
#define NV4097_SET_STENCIL_MASK                 0x0000032C
#define NV4097_SET_STENCIL_FUNC                 0x00000330
#define NV4097_SET_STENCIL_FUNC_REF             0x00000334
#define NV4097_SET_STENCIL_FUNC_MASK            0x00000338
#define NV4097_SET_STENCIL_OP_FAIL              0x0000033C
#define NV4097_SET_STENCIL_OP_ZFAIL             0x00000340
#define NV4097_SET_STENCIL_OP_ZPASS             0x00000344
#define NV4097_SET_TWO_SIDED_STENCIL_TEST_ENABLE 0x00000348
#define NV4097_SET_BACK_STENCIL_MASK            0x0000034C
#define NV4097_SET_BACK_STENCIL_FUNC            0x00000350
#define NV4097_SET_BACK_STENCIL_FUNC_REF        0x00000354
#define NV4097_SET_BACK_STENCIL_FUNC_MASK       0x00000358
#define NV4097_SET_BACK_STENCIL_OP_FAIL         0x0000035C
#define NV4097_SET_BACK_STENCIL_OP_ZFAIL        0x00000360
#define NV4097_SET_BACK_STENCIL_OP_ZPASS        0x00000364
#define NV4097_SET_DEPTH_BOUNDS_TEST_ENABLE     0x00000380
#define NV4097_SET_DEPTH_BOUNDS_MIN             0x00000384  /* float */
#define NV4097_SET_DEPTH_BOUNDS_MAX             0x00000388  /* float */

/* Culling */
#define NV4097_SET_CULL_FACE_ENABLE             0x0000183C
#define NV4097_SET_CULL_FACE                    0x00001830
#define NV4097_SET_FRONT_FACE                   0x00001834

/* Primitive types */
#define RSX_PRIMITIVE_POINTS             1
#define RSX_PRIMITIVE_LINES              2
#define RSX_PRIMITIVE_LINE_LOOP          3
#define RSX_PRIMITIVE_LINE_STRIP         4
#define RSX_PRIMITIVE_TRIANGLES          5
#define RSX_PRIMITIVE_TRIANGLE_STRIP     6
#define RSX_PRIMITIVE_TRIANGLE_FAN       7
#define RSX_PRIMITIVE_QUADS              8
#define RSX_PRIMITIVE_QUAD_STRIP         9
#define RSX_PRIMITIVE_POLYGON            10

/* ---------------------------------------------------------------------------
 * RSX State — tracked as methods are processed
 * -----------------------------------------------------------------------*/

#define RSX_MAX_TEXTURES          16
#define RSX_MAX_VERTEX_TEXTURES    4
#define RSX_MAX_VERTEX_ATTRIBS    16
#define RSX_MAX_RENDER_TARGETS     4
#define RSX_MAX_INLINE_INDEX_ELEMENTS 16384u

typedef struct rsx_texture_state {
    u32 offset;
    u32 format;
    u32 address;
    u32 control0;
    u32 control1;
    u32 control2;    /* separate contiguous block at 0x0B00 + unit*4 */
    u32 filter;
    u32 image_rect;
    u32 border_color;
    u32 control3;    /* row pitch [19:0], depth [31:20] -- a separate block */
    int dirty;
} rsx_texture_state;

typedef struct rsx_vertex_attrib {
    u32 format;
    u32 offset;
    u32 stride;
    u32 size;
    u32 type;
    u32 frequency;   /* format bits [31:16]: vertex frequency divisor (instancing) */
    int enabled;
} rsx_vertex_attrib;

typedef struct rsx_state {
    /* Render targets */
    u32 surface_format;
    u32 surface_color_offset[RSX_MAX_RENDER_TARGETS];
    u32 surface_color_pitch[RSX_MAX_RENDER_TARGETS];
    u32 surface_zeta_offset;
    u32 surface_zeta_pitch;
    u32 surface_clip_x, surface_clip_y;
    u32 surface_clip_w, surface_clip_h;
    u32 color_target;

    /* Viewport */
    u32 viewport_x, viewport_y, viewport_w, viewport_h;
    float clip_min, clip_max;

    /* Additional NV4097 render state used by RSX fragment processing. */
    u32 window_offset;
    u32 shader_window;
    u32 fog_mode;
    float fog_params[2];
    int poly_offset_point, poly_offset_line, poly_offset_fill;
    float poly_offset_scale, poly_offset_bias;
    u32 control0, zcull_enable, zcull_stats_enable, zcull_control1;
    int zpass_pixel_count_enable;
    u32 zmin_max_control; /* depth clip/clamp/ignore-W control */
    u32 dma_context[15]; /* NV4097 context DMA registers 0x0180..0x01B8 */
    u32 restart_index_enable, restart_index;

    /* Scissor */
    u32 scissor_x, scissor_y, scissor_w, scissor_h;

    /* Clear */
    u32 color_clear_value;
    u32 zstencil_clear_value;

    /* Blend */
    int blend_enable;
    u32 blend_sfactor, blend_dfactor;
    u32 blend_equation;
    u32 blend_color;
    u32 blend_color2;

    /* Depth */
    int depth_test_enable;
    u32 depth_func;
    int depth_mask;

    /* Stencil */
    int stencil_test_enable;
    int stencil_two_sided_enable;
    u32 stencil_func, stencil_ref, stencil_func_mask, stencil_write_mask;
    u32 stencil_op_fail, stencil_op_zfail, stencil_op_zpass;
    u32 back_stencil_func, back_stencil_ref, back_stencil_func_mask, back_stencil_write_mask;
    u32 back_stencil_op_fail, back_stencil_op_zfail, back_stencil_op_zpass;

    /* Culling */
    int cull_face_enable;
    u32 cull_face;
    u32 front_face;

    /* Vertex colour interpretation. Flat mode selects the last vertex in
     * each RSX primitive; two-sided lighting selects o1/o2 vs o3/o4 by face. */
    u32 shade_mode;
    int two_sided_lighting;

    /* Color mask — bits: [0] B, [8] G, [16] R, [24] A */
    u32 color_mask;
    /* MRT1..3 write masks, four bits per target (A,R,G,B). */
    u32 color_mask_mrt;
    int color_mask_dirty;
    /* MRT1..3 blend-enable bits are bits 1..3. */
    u32 blend_enable_mrt;

    /* Alpha test */
    int alpha_test_enable;
    u32 alpha_func;
    u32 alpha_ref;
    int alpha_dirty;

    /* Textures */
    rsx_texture_state textures[RSX_MAX_TEXTURES];
    /* Vertex textures reuse rsx_texture_state deliberately: a vertex unit's
     * registers are a fragment unit's with CONTROL3 in CONTROL1's slot, so
     * decoding one into the same struct with control1 left at zero -- which
     * reads as the identity crossbar -- lets a backend resolve, upload and
     * cache both kinds through one path. */
    rsx_texture_state vertex_textures[RSX_MAX_VERTEX_TEXTURES];

    /* Vertex attributes */
    rsx_vertex_attrib vertex_attribs[RSX_MAX_VERTEX_ATTRIBS];
    /* Constant ("current") vertex attributes -- NV4097_SET_VERTEX_DATA4F_M and
     * friends. When an attribute array is DISABLED, the hardware feeds every
     * vertex this register rather than zero, exactly like glColor4f with the
     * colour array switched off. Default (0,0,0,1) matches the RSX reset state. */
    float vertex_data4f[RSX_MAX_VERTEX_ATTRIBS][4];
    /* NV4097_SET_FREQUENCY_DIVIDER_OPERATION (0x1FC0): per-attribute bitmask
     * selecting how the frequency divisor is applied -- bit set = MODULO
     * (index = vertex % freq, repeats a mesh), clear = DIVIDE (index =
     * vertex / freq, advances once per instance). Instancing needs both. */
    u32 frequency_divider_op;
    /* NV4097 vertex array address modifiers. Base offset is added to each
     * enabled attribute offset in the 28-bit RSX address domain; base index
     * is added only to indexed draw elements in the 20-bit index domain. */
    u32 vertex_data_base_offset;
    /* NV4097_SET_VERTEX_DATA_BASE_INDEX: added to indexed draw elements in
     * the RSX's 20-bit vertex-index domain before vertex fetch. */
    u32 vertex_data_base_index;

    /* Current draw state */
    u32 primitive_type;
    int in_begin_end;  /* between BEGIN_END(type) and BEGIN_END(0) */
    int conditional_render_enabled;
    u32 conditional_render_offset;

    /* Index array (SET_INDEX_ARRAY_ADDRESS/_DMA): offset is a raw RSX offset;
     * dma bits [3:0] = location context (0=local 1=main via CELL_GCM_DMA ids),
     * bits [7:4] = index type (0 = u32, 1 = u16). */
    /* Incremented on every SET_BEGIN_END(begin). One primitive stream may be
     * split across several DRAW_ARRAYS/DRAW_INDEX_ARRAY entries inside one
     * BEGIN/END; the backend concatenates those and must not merge across a
     * boundary, where the transform constants can change. */
    u32 begin_epoch;
    u32 index_array_offset;
    u32 index_array_dma;

    /* Shader state */
    /* RSX viewport transform (SET_VIEWPORT_OFFSET 0x0A20 / _SCALE 0x0A30,
     * 4 floats each): window = ndc * scale + offset (pixels; z [0,1]). */
    float viewport_offset[4];
    float viewport_scale[4];
    u32 shader_program;       /* fragment program address (offset | location in bits [0:1]) */
    u32 shader_control;       /* SET_SHADER_CONTROL (0x40 = 32-bit colour exports) */
    u32 shader_packer;        /* SET_SHADER_PACKER: nonzero enables integer sRGB output */
    u32 fragment_program_addr;
    u32 vertex_attrib_output_mask;
    u32 vertex_attrib_input_mask;      /* 0x1FF0 */
    u32 transform_branch_bits;         /* 0x1FF8: VP static branch condition bits */
    u32 transform_program_load;
    u32 transform_program_start; /* vertex program load slot index */
    u32 transform_constant_load;
    int shader_dirty;
    u32 tex_coord_control[10]; /* bit 0: force texture coordinate z=0, w=clip W */
    u32 anti_aliasing_control; /* enable[0], A2C[4], alpha-one[8], mask[31:16] */
    u32 transform_timeout;    /* hardware scheduling hint; no host watchdog */

    /* Vertex transform constants — written by NV4097_SET_TRANSFORM_CONSTANT.
     * Each constant is a vec4 (4 floats). Games typically place the MVP
     * matrix in the first few slots. Used by the backend as the vertex
     * shader constant buffer when no RSX vertex program is translated. */
    float vertex_constants[RSX_MAX_VERTEX_CONSTANTS][4];
    /* Lowest and highest slot touched since last reset — used to push only
     * the dirty range to the host CBV. */
    u32 vertex_constants_lo;
    u32 vertex_constants_hi;
    int vertex_constants_dirty;

    /* Captured RSX vertex program microcode (NV40 ISA), filled by
     * NV4097_SET_TRANSFORM_PROGRAM. Words are host-endian (as delivered by the
     * FIFO parser), i.e. RPCS3 order, ready for rsx_vp_decompile. */
    u8  vp_ucode[1024 * 16];  /* up to 1024 instructions (Tiny3D's VP exceeds 128;
                               * truncation dropped the end bit + HPOS write) */
    u32 vp_ucode_write;       /* byte write cursor (from SET_TRANSFORM_PROGRAM_LOAD) */
    u32 vp_ucode_bytes;       /* highest byte written + 1 */
    int vp_dirty;             /* program changed since last translate */

    /* Dirty flags for incremental state updates */
    int surface_dirty;
    int viewport_dirty;
    int blend_dirty;
    int depth_dirty;
    int stencil_dirty;
    int texture_dirty;
    int vertex_dirty;
    /* Immediate ARRAY_ELEMENT16/32 words for one BEGIN_END group. RPCS3's
     * immediate-index path stages both methods as u32 indices. */
    u32 inline_index_values[RSX_MAX_INLINE_INDEX_ELEMENTS];
    u32 inline_index_count;
    u32 inline_index_dropped;
    int inline_index_overflow;
    int explicit_draw_command;
} rsx_state;

/* ---------------------------------------------------------------------------
 * RSX Command Processor API
 * -----------------------------------------------------------------------*/

/* Initialize RSX state to defaults */
void rsx_state_init(rsx_state* state);

/* Process a single NV47xx method call.
 * method: register address (e.g., NV4097_SET_BLEND_ENABLE)
 * data:   32-bit value to write
 * Returns 0 on success, -1 if method is unrecognized. */
int rsx_process_method(rsx_state* state, u32 method, u32 data);
/* Replay sync methods found while recovering a skipped FIFO range. */
int rsx_replay_skipped_sync_method(u32 subchannel, u32 method, u32 data);

/* Process a command buffer segment.
 * buf:  pointer to command buffer data (host memory)
 * size: size in bytes
 * Returns number of methods processed. */
int rsx_process_command_buffer(rsx_state* state, const u32* buf, u32 size);

/* ---------------------------------------------------------------------------
 * Backend callbacks — implemented per-platform (D3D12, Vulkan, null)
 * -----------------------------------------------------------------------*/

typedef struct rsx_backend {
    void* userdata;

    /* Lifecycle */
    int  (*init)(void* userdata, u32 width, u32 height);
    void (*shutdown)(void* userdata);

    /* Frame */
    void (*begin_frame)(void* userdata);
    void (*end_frame)(void* userdata);
    void (*present)(void* userdata, u32 buffer_id);

    /* State changes (called when dirty flags are set) */
    void (*set_render_target)(void* userdata, const rsx_state* state);
    void (*set_viewport)(void* userdata, const rsx_state* state);
    void (*set_blend)(void* userdata, const rsx_state* state);
    void (*set_depth_stencil)(void* userdata, const rsx_state* state);
    void (*set_color_mask)(void* userdata, const rsx_state* state);
    void (*set_alpha_test)(void* userdata, const rsx_state* state);
    void (*set_shader)(void* userdata, const rsx_state* state);
    void (*set_vertex_attribs)(void* userdata, const rsx_state* state);

    /* Clear */
    void (*clear)(void* userdata, u32 flags, u32 color, float depth, u8 stencil);

    /* Draw */
    void (*draw_arrays)(void* userdata, u32 primitive, u32 first, u32 count);
    void (*draw_indexed)(void* userdata, u32 primitive, u32 index_offset, u32 count);
    void (*wait_idle)(void* userdata); /* NV4097_WAIT_FOR_IDLE */

    /* Texture */
    void (*bind_texture)(void* userdata, u32 unit, const rsx_texture_state* tex);

    /* Immediate indexed draw from ARRAY_ELEMENT16/32. Optional; called at
     * END with a temporary stream valid for the duration of the call. */
    void (*draw_indexed_immediate)(void* userdata, u32 primitive,
                                   const u32* indices, u32 count);
} rsx_backend;

/* Register a backend. Call before any RSX processing. */
void rsx_set_backend(rsx_backend* backend);

/* Get the current backend (or NULL if none set). */
rsx_backend* rsx_get_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* PS3RECOMP_RSX_COMMANDS_H */
