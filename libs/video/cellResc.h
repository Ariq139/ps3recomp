/*
 * ps3recomp - cellResc HLE
 *
 * Resolution scaling/conversion module. Handles scaling of render targets
 * to display resolution (e.g., rendering at 720p, output at 1080p).
 */

#ifndef PS3RECOMP_CELL_RESC_H
#define PS3RECOMP_CELL_RESC_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Error codes
 * -----------------------------------------------------------------------*/
#define CELL_RESC_ERROR_NOT_INITIALIZED    0x80210301
#define CELL_RESC_ERROR_REINITIALIZED      0x80210302
#define CELL_RESC_ERROR_BAD_ALIGNMENT      0x80210303
#define CELL_RESC_ERROR_BAD_ARGUMENT       0x80210304
#define CELL_RESC_ERROR_LESS_MEMORY        0x80210305
#define CELL_RESC_ERROR_INSUFFICIENT_BUFFER CELL_RESC_ERROR_LESS_MEMORY
#define CELL_RESC_ERROR_GCM_FLIP_QUEUE_FULL 0x80210306
#define CELL_RESC_ERROR_BAD_COMBINATION    0x80210307
#define CELL_RESC_ERROR_UNKNOWN_308        0x80210308

/* ---------------------------------------------------------------------------
 * Display buffer modes
 * -----------------------------------------------------------------------*/
#define CELL_RESC_720x480                  0x01
#define CELL_RESC_720x576                  0x02
#define CELL_RESC_1280x720                 0x04
#define CELL_RESC_1920x1080                0x08

/* ---------------------------------------------------------------------------
 * Conversion modes
 * -----------------------------------------------------------------------*/
#define CELL_RESC_FULLSCREEN               0
#define CELL_RESC_LETTERBOX                1
#define CELL_RESC_PANSCAN                  2

/* ---------------------------------------------------------------------------
 * Pal temporal mode
 * -----------------------------------------------------------------------*/
#define CELL_RESC_PAL_50                   0
#define CELL_RESC_PAL_60_DROP              1
#define CELL_RESC_PAL_60_INTERPOLATE       2
#define CELL_RESC_PAL_60_INTERPOLATE_30_DROP   3
#define CELL_RESC_PAL_60_INTERPOLATE_DROP_FLEXIBLE 4
#define CELL_RESC_PAL_60_FOR_HSYNC             5

#define CELL_RESC_DISPLAY_VSYNC                 0
#define CELL_RESC_DISPLAY_HSYNC                 1

/* cellRescCreateInterlaceTable element-depth values. */
#define CELL_RESC_ELEMENT_HALF_FLOAT           0
#define CELL_RESC_ELEMENT_FLOAT                1

/* cellVideoOut resolution IDs accepted by cellResc's conversion helper. */
#define CELL_VIDEO_OUT_RESOLUTION_1080          1
#define CELL_VIDEO_OUT_RESOLUTION_720           2
#define CELL_VIDEO_OUT_RESOLUTION_480           4
#define CELL_VIDEO_OUT_RESOLUTION_576           5

/* ---------------------------------------------------------------------------
 * Ratio conversion mode
 * -----------------------------------------------------------------------*/
#define CELL_RESC_INTERLACE_FILTER         0
#define CELL_RESC_NORMAL_BILINEAR          1
#define CELL_RESC_ELEMENT_BILINEAR         2

/* ---------------------------------------------------------------------------
 * Types
 * -----------------------------------------------------------------------*/

typedef struct CellRescInitConfig {
    u32 size;
    u32 resourcePolicy;
    u32 supportModes;
    u32 ratioMode;
    u32 palTemporalMode;
    u32 interlaceMode;
    u32 flipMode;
} CellRescInitConfig;

typedef struct CellRescSrc {   /* SDK resc.h: 16 bytes, width/height are u16 */
    u32 format;
    u32 pitch;
    u16 width;
    u16 height;
    u32 offset;
} CellRescSrc;

typedef struct CellRescDsts {
    u32 format;
    u32 pitch;
    u32 heightAlign;
} CellRescDsts;

/* ---------------------------------------------------------------------------
 * Functions
 * -----------------------------------------------------------------------*/

s32 cellRescInit(const CellRescInitConfig* initConfig);
void cellRescExit(void);

s32 cellRescSetDisplayMode(u32 displayMode);
s32 cellRescGetNumColorBuffers(u32 displayMode, u32 palTemporalMode, s32 reserved);

s32 cellRescGetBufferSize(u32* colorBufSize, u32* vertexBufSize, u32* fragmentBufSize);
s32 cellRescSetBufferAddress(void* colorBuf, void* vertexBuf, void* fragmentBuf);

s32 cellRescSetSrc(s32 index, const CellRescSrc* src);
s32 cellRescSetDsts(u32 displayMode, const CellRescDsts* dsts);
s32 cellRescSetConvertAndFlip(void* context, s32 index);
s32 cellRescSetWaitFlip(void* context);
s32 cellRescGcmSurface2RescSrc(const void* gcmSurface, CellRescSrc* rescSrc);
s32 cellRescVideoOutResolutionId2RescBufferMode(u32 resolutionId, u32* bufferMode);
s32 cellRescGetRegisterCount(void);
void cellRescSetRegisterCount(s32 registerCount);

s32 cellRescSetFlipHandler(void (*handler)(u32));
s32 cellRescSetVBlankHandler(void (*handler)(u32));

s32 cellRescGetDisplayMode(u32* displayMode);
u64 cellRescGetLastFlipTime(void);
void cellRescResetFlipStatus(void);
s32 cellRescGetFlipStatus(void);

s32 cellRescSetPalInterpolateDropFlexRatio(float ratio);
s32 cellRescCreateInterlaceTable(void* table, float srcHeight, s32 depth, s32 length);
s32 cellRescAdjustAspectRatio(float horizontal, float vertical);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_RESC_H */
