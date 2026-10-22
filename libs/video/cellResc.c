/*
 * ps3recomp - cellResc HLE implementation
 *
 * Resolution scaling stub. Since the recompiled game renders through
 * the host GPU, actual rescaling is handled by the host graphics API.
 * This module tracks state so game code that queries RESC works correctly.
 */

#include "cellResc.h"
#include "../../runtime/ppu/ppu_memory.h"   /* GUEST_PTR, vm_write*: translate + byte-swap */
#include "ps3emu/guest_call.h"   /* ps3_invoke_guest: handlers are guest OPDs */
#include <stdio.h>
#include <string.h>
#include "ps3emu/log_first.h"

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

static int s_initialized = 0;
static CellRescInitConfig s_config;
static u32 s_display_mode = CELL_RESC_1280x720;
static CellRescSrc s_src[8]; /* up to 8 color buffers */
static CellRescDsts s_dsts[4]; /* one per display mode */
/* Guest OPD addresses, not host function pointers: what a title passes is a
 * guest function descriptor, and calling it as a host pointer is a crash
 * waiting for the first title that sets one. Invoked through ps3_invoke_guest,
 * the same way cellGcmSetFlipHandler's is. */
static u32 s_flip_handler_opd = 0;
static u32 s_vblank_handler_opd = 0;
static u32 s_flip_target = 0;   /* display buffer the next convert-and-flip presents */
static s32 s_flip_status = 0;
static u64 s_last_flip_time = 0;
static float s_aspect_h = 1.0f;
static float s_aspect_v = 1.0f;
static float s_pal_ratio = 0.5f;

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 cellRescInit(const CellRescInitConfig* initConfig)
{
    printf("[cellResc] Init()\n");

    if (s_initialized)
        return (s32)CELL_RESC_ERROR_REINITIALIZED;

    if (!initConfig)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    const u32 config_ea = (u32)(uintptr_t)initConfig;
    s_config.size = vm_read32(config_ea + 0);
    if (s_config.size > sizeof(CellRescInitConfig))
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    s_config.resourcePolicy = vm_read32(config_ea + 4);
    s_config.supportModes = vm_read32(config_ea + 8);
    s_config.ratioMode = vm_read32(config_ea + 12);
    s_config.palTemporalMode = vm_read32(config_ea + 16);
    s_config.interlaceMode = vm_read32(config_ea + 20);
    s_config.flipMode = vm_read32(config_ea + 24);
    memset(s_src, 0, sizeof(s_src));
    memset(s_dsts, 0, sizeof(s_dsts));
    s_flip_handler_opd = 0;
    s_vblank_handler_opd = 0;
    s_flip_target = 0;
    s_flip_status = 0;
    s_last_flip_time = 0;
    s_aspect_h = 1.0f;
    s_aspect_v = 1.0f;
    s_initialized = 1;
    return CELL_OK;
}

void cellRescExit(void)
{
    printf("[cellResc] Exit()\n");
    s_initialized = 0;
}

s32 cellRescSetDisplayMode(u32 displayMode)
{
    printf("[cellResc] SetDisplayMode(0x%x)\n", displayMode);

    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;

    if (!displayMode || displayMode > CELL_RESC_1920x1080 ||
        !(s_config.supportModes & displayMode))
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    if (displayMode == CELL_RESC_720x576) {
        const u32 pal_mode = s_config.palTemporalMode;
        const u32 flip_mode = s_config.flipMode;
        if (((pal_mode >= CELL_RESC_PAL_60_INTERPOLATE &&
              pal_mode <= CELL_RESC_PAL_60_INTERPOLATE_DROP_FLEXIBLE) ||
             pal_mode == CELL_RESC_PAL_60_DROP) &&
            flip_mode == CELL_RESC_DISPLAY_HSYNC)
            return (s32)CELL_RESC_ERROR_BAD_COMBINATION;
        if (pal_mode == CELL_RESC_PAL_60_FOR_HSYNC &&
            flip_mode == CELL_RESC_DISPLAY_VSYNC)
            return (s32)CELL_RESC_ERROR_BAD_COMBINATION;
    }

    s_display_mode = displayMode;
    return CELL_OK;
}

s32 cellRescGetNumColorBuffers(u32 displayMode, u32 palTemporalMode, s32 reserved)
{
    /* Behavior and ABI follow RPCS3 cellResc.cpp at 2ba2f75bc. */
    if (reserved != 0)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    if (displayMode == CELL_RESC_720x576) {
        if (palTemporalMode >= CELL_RESC_PAL_60_INTERPOLATE &&
            palTemporalMode <= CELL_RESC_PAL_60_INTERPOLATE_DROP_FLEXIBLE)
            return 6;
        if (palTemporalMode == CELL_RESC_PAL_60_DROP)
            return 3;
    }
    return 2;
}

s32 cellRescGetBufferSize(u32* colorBufSize, u32* vertexBufSize, u32* fragmentBufSize)
{
    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;

    /* Provide reasonable buffer sizes for state tracking */
    if (colorBufSize)
        vm_write32((uint32_t)(uintptr_t)colorBufSize, 1920 * 1080 * 4); /* RGBA 1080p */
    if (vertexBufSize)
        vm_write32((uint32_t)(uintptr_t)vertexBufSize, 64 * 1024); /* 64KB vertex buffer */
    if (fragmentBufSize)
        vm_write32((uint32_t)(uintptr_t)fragmentBufSize, 256 * 1024); /* 256KB fragment shader */

    return CELL_OK;
}

s32 cellRescSetBufferAddress(void* colorBuf, void* vertexBuf, void* fragmentBuf)
{
    (void)colorBuf;
    (void)vertexBuf;
    (void)fragmentBuf;

    printf("[cellResc] SetBufferAddress()\n");

    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;

    if (!colorBuf || !vertexBuf || !fragmentBuf)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    if (((uintptr_t)colorBuf & 127u) || ((uintptr_t)vertexBuf & 3u) ||
        ((uintptr_t)fragmentBuf & 63u))
        return (s32)CELL_RESC_ERROR_BAD_ALIGNMENT;

    /* In host-GPU mode we don't use these buffers directly */
    return CELL_OK;
}

s32 cellRescSetSrc(s32 index, const CellRescSrc* src)
{
    printf("[cellResc] SetSrc(index=%d)\n", index);

    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;

    if (!src || index < 0 || index >= 8)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    /* Decoded to host order: the guest struct is big-endian. */
    const u32 ea = (u32)(uintptr_t)src;
    s_src[index].format = vm_read32(ea + 0);
    s_src[index].pitch  = vm_read32(ea + 4);
    s_src[index].width  = vm_read16(ea + 8);
    s_src[index].height = vm_read16(ea + 10);
    s_src[index].offset = vm_read32(ea + 12);
    if (!s_src[index].width || !s_src[index].height || s_src[index].height > 4096)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    printf("[cellResc] SetSrc(%d): offset=0x%X pitch=%u %ux%u fmt=%u\n", index,
           s_src[index].offset, s_src[index].pitch, s_src[index].width,
           s_src[index].height, s_src[index].format);
    return CELL_OK;
}

s32 cellRescSetDsts(u32 displayMode, const CellRescDsts* dsts)
{
    printf("[cellResc] SetDsts(mode=0x%x)\n", displayMode);

    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;

    if (!dsts)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    if (!displayMode || displayMode > CELL_RESC_1920x1080)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    /* Map display mode to index */
    int idx = 0;
    if (displayMode & CELL_RESC_720x576) idx = 1;
    else if (displayMode & CELL_RESC_1280x720) idx = 2;
    else if (displayMode & CELL_RESC_1920x1080) idx = 3;

    s_dsts[idx] = *GUEST_PTR(dsts, const CellRescDsts*);
    return CELL_OK;
}

s32 cellRescSetConvertAndFlip(void* context, s32 index)
{
    (void)context;

    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;
    if (index < 0 || index >= 8)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    /* This is a real flip, not bookkeeping. A title that presents through RESC
     * -- Virtua Fighter 5 does -- renders into an off-screen surface and asks
     * RESC to scale it into a display buffer and show it. Tracking a counter
     * here and returning CELL_OK meant no flip ever reached cellGcm, so the
     * title's frame loop never closed: nothing retired, and it kept appending
     * to a command buffer it could never reclaim.
     *
     * ponytail: the convert half is deliberately not emitted -- the backend
     * already presents at the output resolution, so RESC's scale is a no-op for
     * us. What must happen is the flip. Rotate through the display buffers the
     * title registered, the way RESC's own double/triple buffering does. Emit a
     * real conversion draw if a title ever needs RESC's PAL/interlace modes
     * rather than a plain scale. */
    extern s32 cellGcmSetFlipCommand(u32 bufferId);
    extern u32 cellGcm_display_buffer_count(void);
    u32 nbuf = cellGcm_display_buffer_count();
    if (nbuf) {
        cellGcmSetFlipCommand(s_flip_target % nbuf);
        s_flip_target = (u32)((s_flip_target + 1) % nbuf);
    }

    s_flip_status = 0;      /* flip complete */
    s_last_flip_time++;

    { static int n = 0;
      if (PS3_LOG_FIRST(n, 4))
          printf("[cellResc] SetConvertAndFlip(src=%d) -> flip, %u display buffer(s)\n",
                 index, nbuf); }

    if (s_flip_handler_opd)
        ps3_invoke_guest(s_flip_handler_opd, 1, 0, 0, 0, 0, 0, 0, 0);

    return CELL_OK;
}

/* ABI/validation mirror of RPCS3 cellResc.cpp at 2ba2f75bc. These are local
 * implementations; no RPCS3 function bodies are copied. */
s32 cellRescSetWaitFlip(void* context)
{
    (void)context;
    return CELL_OK;
}

s32 cellRescGcmSurface2RescSrc(const void* gcmSurface, CellRescSrc* rescSrc)
{
    (void)gcmSurface;
    (void)rescSrc;
    if (!gcmSurface || !rescSrc)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    return CELL_OK;
}

s32 cellRescVideoOutResolutionId2RescBufferMode(u32 resolutionId, u32* bufferMode)
{
    (void)bufferMode;
    if (!bufferMode || !resolutionId || resolutionId > CELL_VIDEO_OUT_RESOLUTION_576)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    return CELL_OK;
}

s32 cellRescGetRegisterCount(void)
{
    return 0;
}

void cellRescSetRegisterCount(s32 registerCount)
{
    (void)registerCount;
}

s32 cellRescSetFlipHandler(void (*handler)(u32))
{
    s_flip_handler_opd = (u32)(uintptr_t)handler;
    printf("[cellResc] SetFlipHandler(opd=0x%08X)\n", s_flip_handler_opd);
    return CELL_OK;
}

s32 cellRescSetVBlankHandler(void (*handler)(u32))
{
    s_vblank_handler_opd = (u32)(uintptr_t)handler;
    printf("[cellResc] SetVBlankHandler(opd=0x%08X)\n", s_vblank_handler_opd);
    return CELL_OK;
}

s32 cellRescGetDisplayMode(u32* displayMode)
{
    if (!displayMode)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    vm_write32((uint32_t)(uintptr_t)displayMode, s_display_mode);
    return CELL_OK;
}

u64 cellRescGetLastFlipTime(void)
{
    return s_last_flip_time;
}

void cellRescResetFlipStatus(void)
{
    s_flip_status = 1; /* waiting for next flip */
}

s32 cellRescGetFlipStatus(void)
{
    /* Auto-complete pending flips when polled, so games don't stall
     * waiting for events in the null renderer. */
    if (s_flip_status == 1) {
        s_flip_status = 0;
        s_last_flip_time++;
        if (s_flip_handler_opd)
            ps3_invoke_guest(s_flip_handler_opd, 1, 0, 0, 0, 0, 0, 0, 0);
        if (s_vblank_handler_opd)
            ps3_invoke_guest(s_vblank_handler_opd, 1, 0, 0, 0, 0, 0, 0, 0);
    }
    return s_flip_status;
}

s32 cellRescSetPalInterpolateDropFlexRatio(float ratio)
{
    printf("[cellResc] SetPalInterpolateDropFlexRatio(%.2f)\n", ratio);
    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;
    if (ratio < 0.0f || ratio > 1.0f)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    s_pal_ratio = ratio;
    return CELL_OK;
}

s32 cellRescCreateInterlaceTable(void* table, float srcHeight, s32 depth, s32 length)
{
    /* RPCS3 currently validates these arguments and returns success without
     * touching the output table; preserve that behavior and its ABI. */
    if (!table || srcHeight <= 0.0f || depth > CELL_RESC_ELEMENT_FLOAT || length <= 0)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;
    return CELL_OK;
}

s32 cellRescAdjustAspectRatio(float horizontal, float vertical)
{
    printf("[cellResc] AdjustAspectRatio(h=%.2f, v=%.2f)\n", horizontal, vertical);

    if (!s_initialized)
        return (s32)CELL_RESC_ERROR_NOT_INITIALIZED;

    if (horizontal < 0.5f || horizontal > 2.0f ||
        vertical < 0.5f || vertical > 2.0f)
        return (s32)CELL_RESC_ERROR_BAD_ARGUMENT;

    s_aspect_h = horizontal;
    s_aspect_v = vertical;
    return CELL_OK;
}
