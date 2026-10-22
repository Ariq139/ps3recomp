/*
 * ps3recomp - cellRemotePlay HLE
 *
 * RPCS3_REFERENCE_REVIEWED (RPCS3 commit 2ba2f75bc): the currently registered
 * calls are TODO handlers returning CELL_OK, except GetComparativeVolume,
 * which writes 1.0 when its output pointer is non-null. This file expresses
 * those observed contracts through ps3recomp's guest-memory API. No RPCS3
 * function body or host streaming implementation is copied.
 */

#include "cellRemotePlay.h"
#include "../../runtime/ppu/ppu_memory.h"

s32 cellRemotePlayGetStatus(void)
{
    return CELL_OK;
}

s32 cellRemotePlaySetComparativeVolume(float comparative_audio_volume)
{
    (void)comparative_audio_volume;
    return CELL_OK;
}

s32 cellRemotePlayGetPeerInfo(void)
{
    return CELL_OK;
}

s32 cellRemotePlayGetSharedMemory(void)
{
    return CELL_OK;
}

s32 cellRemotePlayEncryptAllData(void)
{
    return CELL_OK;
}

s32 cellRemotePlayStopPeerVideoOut(void)
{
    return CELL_OK;
}

s32 cellRemotePlayGetComparativeVolume(float* volume)
{
    if (volume)
        vm_write32((u32)(uintptr_t)volume, 0x3F800000u); /* 1.0f */
    return CELL_OK;
}

s32 cellRemotePlayBreak(void)
{
    return CELL_OK;
}
