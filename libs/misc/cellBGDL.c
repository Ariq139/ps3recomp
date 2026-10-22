/*
 * ps3recomp - cellBGDL HLE implementation
 *
 * Canonical cellBGDLUtility exports match RPCS3's current TODO-success
 * behavior. A separate lowercase local download-management API is retained
 * for compatibility and has no RPCS3 counterpart.
 */

#include "cellBGDL.h"
#include <stdio.h>
#include <string.h>
#include "../../runtime/ppu/ppu_memory.h"   /* vm_write*: guest EA -> host, byte-swapped */

/* Internal state */

static int s_initialized = 0;

/* API */

s32 cellBgdlInit(void)
{
    printf("[cellBGDL] Init()\n");
    if (s_initialized)
        return (s32)CELL_BGDL_ERROR_ALREADY_INITIALIZED;
    s_initialized = 1;
    return CELL_OK;
}

s32 cellBgdlTerm(void)
{
    printf("[cellBGDL] Term()\n");
    s_initialized = 0;
    return CELL_OK;
}

s32 cellBgdlGetInfo(CellBgdlId id, CellBgdlInfo* info)
{
    (void)id; (void)info;
    if (!s_initialized) return (s32)CELL_BGDL_ERROR_NOT_INITIALIZED;
    return (s32)CELL_BGDL_ERROR_NOT_FOUND;
}

s32 cellBgdlGetList(CellBgdlInfo* list, u32 maxEntries, u32* numEntries)
{
    (void)list; (void)maxEntries;
    if (!s_initialized) return (s32)CELL_BGDL_ERROR_NOT_INITIALIZED;
    if (!numEntries) return (s32)CELL_BGDL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)numEntries, (u32)0); /* no downloads */
    return CELL_OK;
}

s32 cellBgdlStartDownload(const char* url, const char* contentId,
                            CellBgdlId* id)
{
    (void)url; (void)contentId; (void)id;
    printf("[cellBGDL] StartDownload() - stub\n");
    if (!s_initialized) return (s32)CELL_BGDL_ERROR_NOT_INITIALIZED;
    return (s32)CELL_BGDL_ERROR_BUSY;
}

s32 cellBgdlPauseDownload(CellBgdlId id)
{
    (void)id;
    if (!s_initialized) return (s32)CELL_BGDL_ERROR_NOT_INITIALIZED;
    return (s32)CELL_BGDL_ERROR_NOT_FOUND;
}

s32 cellBgdlResumeDownload(CellBgdlId id)
{
    (void)id;
    if (!s_initialized) return (s32)CELL_BGDL_ERROR_NOT_INITIALIZED;
    return (s32)CELL_BGDL_ERROR_NOT_FOUND;
}

s32 cellBgdlCancelDownload(CellBgdlId id)
{
    (void)id;
    if (!s_initialized) return (s32)CELL_BGDL_ERROR_NOT_INITIALIZED;
    return (s32)CELL_BGDL_ERROR_NOT_FOUND;
}

/* RPCS3_REFERENCE_REVIEWED (RPCS3 commit 2ba2f75bc): these are the four
 * cellBGDLUtility exports registered by cellBgdl.cpp. Each current RPCS3
 * handler is a TODO that returns CELL_OK without reading arguments or
 * modifying output memory, so preserve that observable behavior. */
s32 cellBGDLGetInfo(const char* contentId, CellBGDLInfo* info, s32 num)
{
    (void)contentId;
    (void)info;
    (void)num;
    return CELL_OK;
}

s32 cellBGDLGetInfo2(const char* serviceId, CellBGDLInfo* info, s32 num)
{
    (void)serviceId;
    (void)info;
    (void)num;
    return CELL_OK;
}

s32 cellBGDLSetMode(CellBGDLMode mode)
{
    (void)mode;
    return CELL_OK;
}

s32 cellBGDLGetMode(CellBGDLMode* mode)
{
    (void)mode;
    return CELL_OK;
}
