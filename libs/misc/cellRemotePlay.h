/*
 * ps3recomp - cellRemotePlay HLE
 *
 * Remote Play export surface. The current RPCS3 implementation treats these
 * calls as successful TODO handlers and reports comparative volume as 1.0.
 */

#ifndef PS3RECOMP_CELL_REMOTE_PLAY_H
#define PS3RECOMP_CELL_REMOTE_PLAY_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error code and constants from RPCS3's cellRemotePlay.h. */
#define CELL_REMOTEPLAY_ERROR_INTERNAL 0x80029830

#define CELL_REMOTEPLAY_STATUS_LOADING    0x00000000u
#define CELL_REMOTEPLAY_STATUS_WAIT       0x00000001u
#define CELL_REMOTEPLAY_STATUS_RUNNING    0x00000002u
#define CELL_REMOTEPLAY_STATUS_UNLOADING  0x00000003u
#define CELL_REMOTEPLAY_STATUS_FATALERROR 0x00000004u
#define CELL_REMOTEPLAY_STATUS_PREMOEND   0x00000005u

#define CELL_REMOTEPLAY_M4V_MEMORY_CONTAINER_SIZE   (8u * 1024u * 1024u)
#define CELL_REMOTEPLAY_AVC_MEMORY_CONTAINER_SIZE   (10u * 1024u * 1024u)
#define CELL_REMOTEPLAY_480P_MEMORY_CONTAINER_SIZE  (10u * 1024u * 1024u)

/* Functions */
s32 cellRemotePlayGetStatus(void);
s32 cellRemotePlaySetComparativeVolume(float comparative_audio_volume);
s32 cellRemotePlayGetPeerInfo(void);
s32 cellRemotePlayGetSharedMemory(void);
s32 cellRemotePlayEncryptAllData(void);
s32 cellRemotePlayStopPeerVideoOut(void);
s32 cellRemotePlayGetComparativeVolume(float* volume);
s32 cellRemotePlayBreak(void);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_REMOTE_PLAY_H */
