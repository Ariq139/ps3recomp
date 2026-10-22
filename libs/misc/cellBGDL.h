/*
 * ps3recomp - cellBGDL HLE
 *
 * Contains RPCS3-compatible cellBGDLUtility exports plus the pre-existing
 * lowercase local download-management interface.
 */

#ifndef PS3RECOMP_CELL_BGDL_H
#define PS3RECOMP_CELL_BGDL_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define CELL_BGDL_ERROR_NOT_INITIALIZED     0x80410901
#define CELL_BGDL_ERROR_ALREADY_INITIALIZED 0x80410902
#define CELL_BGDL_ERROR_INVALID_ARGUMENT    0x80410903
#define CELL_BGDL_ERROR_NOT_FOUND           0x80410904
#define CELL_BGDL_ERROR_BUSY                0x80410905

/* Download status */
#define CELL_BGDL_STATUS_IDLE         0
#define CELL_BGDL_STATUS_DOWNLOADING  1
#define CELL_BGDL_STATUS_PAUSED       2
#define CELL_BGDL_STATUS_COMPLETE     3
#define CELL_BGDL_STATUS_ERROR        4

/* RPCS3's cellBGDLUtility ABI. Its current handlers are TODO success paths. */
#define CELL_BGDL_UTIL_ERROR_BUSY          0x8002CE01u
#define CELL_BGDL_UTIL_ERROR_INTERNAL      0x8002CE02u
#define CELL_BGDL_UTIL_ERROR_PARAM         0x8002CE03u
#define CELL_BGDL_UTIL_ERROR_ACCESS_ERROR  0x8002CE04u
#define CELL_BGDL_UTIL_ERROR_INITIALIZE    0x8002CE05u

typedef enum CellBGDLState {
    CELL_BGDL_STATE_ERROR = 0,
    CELL_BGDL_STATE_PAUSE,
    CELL_BGDL_STATE_READY,
    CELL_BGDL_STATE_RUN,
    CELL_BGDL_STATE_COMPLETE
} CellBGDLState;

typedef enum CellBGDLMode {
    CELL_BGDL_MODE_AUTO = 0,
    CELL_BGDL_MODE_ALWAYS_ALLOW
} CellBGDLMode;

/* Guest ABI layout from RPCS3 cellBgdl.h. Fields are stored big-endian in
 * guest memory; reserved is a 32-bit guest pointer. */
typedef struct CellBGDLInfo {
    u64 received_size;
    u64 content_size;
    s32 state;
    u32 reserved;
} CellBGDLInfo;

/* Types */
typedef u32 CellBgdlId;

typedef struct CellBgdlInfo {
    CellBgdlId id;
    s32 status;
    u64 totalSize;
    u64 downloadedSize;
    char contentId[48];
    u32 reserved[4];
} CellBgdlInfo;

/* Functions */
s32 cellBgdlInit(void);
s32 cellBgdlTerm(void);

s32 cellBgdlGetInfo(CellBgdlId id, CellBgdlInfo* info);
s32 cellBgdlGetList(CellBgdlInfo* list, u32 maxEntries, u32* numEntries);

s32 cellBgdlStartDownload(const char* url, const char* contentId,
                            CellBgdlId* id);
s32 cellBgdlPauseDownload(CellBgdlId id);
s32 cellBgdlResumeDownload(CellBgdlId id);
s32 cellBgdlCancelDownload(CellBgdlId id);

/* NIDs exported by RPCS3's cellBGDLUtility module. */
s32 cellBGDLGetInfo(const char* contentId, CellBGDLInfo* info, s32 num);
s32 cellBGDLGetInfo2(const char* serviceId, CellBGDLInfo* info, s32 num);
s32 cellBGDLSetMode(CellBGDLMode mode);
s32 cellBGDLGetMode(CellBGDLMode* mode);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_BGDL_H */
