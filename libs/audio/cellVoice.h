/*
 * ps3recomp - cellVoice public HLE surface.
 *
 * Guest layouts and API contracts follow the PS3 cellVoice ABI. The runtime
 * does not provide a microphone or voice codec; those APIs therefore report
 * the same unavailable-device behavior as the local RPCS3 reference.
 */

#ifndef PS3RECOMP_CELL_VOICE_H
#define PS3RECOMP_CELL_VOICE_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* cellVoice error values from the public PS3 ABI. */
#define CELL_VOICE_ERROR_NOT_INITIALIZED      0x80310801u
#define CELL_VOICE_ERROR_ALREADY_INITIALIZED  0x80310802u
#define CELL_VOICE_ERROR_GENERAL              0x80310803u
#define CELL_VOICE_ERROR_PORT_INVALID         0x80310804u
#define CELL_VOICE_ERROR_ARGUMENT_INVALID     0x80310805u
#define CELL_VOICE_ERROR_CONTAINER_INVALID    0x80310806u
#define CELL_VOICE_ERROR_TOPOLOGY             0x80310807u
#define CELL_VOICE_ERROR_RESOURCE_INSUFFICIENT 0x80310808u
#define CELL_VOICE_ERROR_NOT_IMPLEMENTED       0x80310809u
#define CELL_VOICE_ERROR_ADDRESS_INVALID       0x8031080Au
#define CELL_VOICE_ERROR_SERVICE_DETACHED      0x8031080Bu
#define CELL_VOICE_ERROR_SERVICE_ATTACHED      0x8031080Cu
#define CELL_VOICE_ERROR_SERVICE_NOT_FOUND     0x8031080Du
#define CELL_VOICE_ERROR_SHAREDMEMORY          0x8031080Eu
#define CELL_VOICE_ERROR_EVENT_QUEUE           0x8031080Fu
#define CELL_VOICE_ERROR_SERVICE_HANDLE        0x80310810u
#define CELL_VOICE_ERROR_EVENT_DISPATCH        0x80310811u
#define CELL_VOICE_ERROR_DEVICE_NOT_PRESENT    0x80310812u

/* Compatibility spellings used by the original local implementation. */
#define CELL_VOICE_ERROR_INVALID_ARGUMENT CELL_VOICE_ERROR_ARGUMENT_INVALID
#define CELL_VOICE_ERROR_PORT_NOT_FOUND   CELL_VOICE_ERROR_TOPOLOGY
#define CELL_VOICE_ERROR_OUT_OF_MEMORY    CELL_VOICE_ERROR_RESOURCE_INSUFFICIENT

#define CELL_VOICE_PORT_MAX 128

#define CELL_VOICE_BITRATE_3850   3850u
#define CELL_VOICE_BITRATE_4650   4650u
#define CELL_VOICE_BITRATE_5700   5700u
#define CELL_VOICE_BITRATE_7300   7300u
#define CELL_VOICE_BITRATE_14400  14400u
#define CELL_VOICE_BITRATE_16000  16000u
#define CELL_VOICE_BITRATE_22533  22533u

#define CELL_VOICE_PORT_IN_MIC        0u
#define CELL_VOICE_PORT_IN_PCMAUDIO   1u
#define CELL_VOICE_PORT_IN_VOICE     2u
#define CELL_VOICE_PORT_OUT_PCMAUDIO 3u
#define CELL_VOICE_PORT_OUT_VOICE    4u
#define CELL_VOICE_PORT_OUT_SECONDARY 5u

#define CELL_VOICE_EVENT_DATA_ERROR         (1u << 0)
#define CELL_VOICE_EVENT_PORT_ATTACHED      (1u << 1)
#define CELL_VOICE_EVENT_PORT_DETACHED      (1u << 2)
#define CELL_VOICE_EVENT_SERVICE_ATTACHED   (1u << 3)
#define CELL_VOICE_EVENT_SERVICE_DETACHED   (1u << 4)
#define CELL_VOICE_EVENT_PORT_WEAK_ATTACHED (1u << 5)
#define CELL_VOICE_EVENT_PORT_WEAK_DETACHED (1u << 6)

#define CELL_VOICE_PCM_FLOAT                  0u
#define CELL_VOICE_PCM_FLOAT_LITTLE_ENDIAN    1u
#define CELL_VOICE_PCM_SHORT                  2u
#define CELL_VOICE_PCM_SHORT_LITTLE_ENDIAN    3u
#define CELL_VOICE_PCM_INTEGER                4u
#define CELL_VOICE_PCM_INTEGER_LITTLE_ENDIAN  5u

#define CELL_VOICE_ATTR_ENERGY_LEVEL       1000u
#define CELL_VOICE_ATTR_VAD                1001u
#define CELL_VOICE_ATTR_DTX                1002u
#define CELL_VOICE_ATTR_AUTO_RESAMPLE      1003u
#define CELL_VOICE_ATTR_LATENCY            1004u
#define CELL_VOICE_ATTR_SILENCE_THRESHOLD  1005u

typedef u32 CellVoicePortId;

/* Opaque guest ABI records. Their fields are read/written by byte offset so
 * guest endianness and the 64-byte CellVoicePortParam alignment are explicit. */
typedef struct CellVoiceInitParam { u8 bytes[32]; } CellVoiceInitParam;
typedef struct CellVoicePortParam { u8 bytes[64]; } CellVoicePortParam;
typedef struct CellVoiceStartParam { u8 bytes[32]; } CellVoiceStartParam;
typedef struct CellVoiceBasePortInfo { u8 bytes[24]; } CellVoiceBasePortInfo;

s32 cellVoiceInit(const CellVoiceInitParam* param);
s32 cellVoiceInitEx(const CellVoiceInitParam* param);
s32 cellVoiceEnd(void);

s32 cellVoiceCreateNotifyEventQueue(u32* queueId, u64* key);
s32 cellVoiceRemoveNotifyEventQueue(u64 key);
s32 cellVoiceSetNotifyEventQueue(u64 key, u64 source);

s32 cellVoiceCreatePort(CellVoicePortId* portId, const CellVoicePortParam* param);
s32 cellVoiceDeletePort(CellVoicePortId portId);
s32 cellVoiceConnectIPortToOPort(CellVoicePortId inPortId, CellVoicePortId outPortId);
s32 cellVoiceDisconnectIPortFromOPort(CellVoicePortId inPortId, CellVoicePortId outPortId);
s32 cellVoiceUpdatePort(CellVoicePortId portId, const CellVoicePortParam* param);
s32 cellVoiceGetPortInfo(CellVoicePortId portId, CellVoiceBasePortInfo* info);

s32 cellVoiceStart(void);
s32 cellVoiceStartEx(const CellVoiceStartParam* param);
s32 cellVoiceStop(void);
s32 cellVoicePausePort(CellVoicePortId portId);
s32 cellVoicePausePortAll(void);
s32 cellVoiceResumePort(CellVoicePortId portId);
s32 cellVoiceResumePortAll(void);
s32 cellVoiceResetPort(CellVoicePortId portId);

s32 cellVoiceGetBitRate(CellVoicePortId portId, u32* bitrate);
s32 cellVoiceSetBitRate(CellVoicePortId portId, u32 bitrate);
s32 cellVoiceGetMuteFlag(CellVoicePortId portId, u16* muted);
s32 cellVoiceSetMuteFlag(CellVoicePortId portId, u16 muted);
s32 cellVoiceSetMuteFlagAll(u16 muted);
s32 cellVoiceGetVolume(CellVoicePortId portId, float* volume);
s32 cellVoiceSetVolume(CellVoicePortId portId, float volume);
s32 cellVoiceGetPortAttr(CellVoicePortId portId, u32 attr, void* value);
s32 cellVoiceSetPortAttr(CellVoicePortId portId, u32 attr, const void* value);
s32 cellVoiceGetSignalState(CellVoicePortId portId, u32 attr, void* value);

s32 cellVoiceWriteToIPort(CellVoicePortId portId, const void* data, u32* size);
s32 cellVoiceWriteToIPortEx(CellVoicePortId portId, const void* data, u32* size,
                            u32 numFrameLost);
s32 cellVoiceWriteToIPortEx2(CellVoicePortId portId, const void* data, u32* size,
                             s16 frameGaps);
s32 cellVoiceReadFromOPort(CellVoicePortId portId, void* data, u32* size);
s32 cellVoiceDebugTopology(void);


#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_VOICE_H */
