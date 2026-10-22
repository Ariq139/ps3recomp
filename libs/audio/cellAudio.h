/*
 * ps3recomp - cellAudio HLE
 *
 * Audio output: port management, mixing thread, event notification.
 * Backend: WASAPI on Windows, SDL2 audio elsewhere.
 */

#ifndef PS3RECOMP_CELL_AUDIO_H
#define PS3RECOMP_CELL_AUDIO_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Constants
 * -----------------------------------------------------------------------*/

#define CELL_AUDIO_PORT_MAX                 8
#define CELL_AUDIO_BLOCK_SAMPLES            256
#define CELL_AUDIO_BLOCK_8                  8
#define CELL_AUDIO_BLOCK_16                 16
#define CELL_AUDIO_BLOCK_32                 32

/* Channel count */
#define CELL_AUDIO_PORT_2CH                 2
#define CELL_AUDIO_PORT_8CH                 8

/* Sample rate (always 48 kHz on PS3) */
#define CELL_AUDIO_SAMPLE_RATE              48000
#define CELL_AUDIO_PERSONAL_DEVICE_PRIMARY  0x00008000

/* Maximum event queues that can be registered */
#define CELL_AUDIO_MAX_NOTIFY_EVENT_QUEUES  8

/* Audio port status */
#define CELL_AUDIO_STATUS_CLOSE             0
#define CELL_AUDIO_STATUS_READY             1
#define CELL_AUDIO_STATUS_RUN               2

/* Event queue creation and port initialization attributes. */
#define CELL_AUDIO_CREATEEVENTFLAG_SPU      0x00000001u
#define CELL_AUDIO_PORTATTR_INITLEVEL        0x0000000000001000ull
#define CELL_AUDIO_EVENT_MIX                 0u
#define CELL_AUDIO_EVENTFLAG_BEFOREMIX       0x80000000u
#define CELL_AUDIO_EVENTFLAG_DECIMATE_2      0x08000000u
#define CELL_AUDIO_EVENTFLAG_DECIMATE_4      0x10000000u
#define CELL_AUDIO_EVENTFLAG_HEADPHONE       0x20000000u
#define CELL_AUDIO_EVENTFLAG_NOMIX           0x40000000u
#define CELL_AUDIO_EVENTFLAG_VALID_MASK      0xF8000000u

/* Audio period in microseconds (~5.333ms for 256 samples @ 48kHz) */
#define CELL_AUDIO_PERIOD_US                5333

/* ---------------------------------------------------------------------------
 * Structures
 * -----------------------------------------------------------------------*/

/* Parameters used to open an audio port */
typedef struct CellAudioPortParam {
    u64 nChannel;       /* CELL_AUDIO_PORT_2CH or CELL_AUDIO_PORT_8CH */
    u64 nBlock;         /* number of blocks (8, 16, or 32) */
    u64 attr;           /* attribute flags */
    float level;        /* volume level (0.0 .. 1.0) */
} CellAudioPortParam;

/* Returned after opening a port -- describes the allocated resources */
typedef struct CellAudioPortConfig {
    u64 readIndexAddr;      /* guest address of the read index counter */
    u32 status;             /* port status */
    u64 nChannel;           /* channels */
    u64 nBlock;             /* blocks */
    u32 portSize;           /* total port buffer size in bytes */
    u64 portAddr;           /* guest address of the port audio buffer */
} CellAudioPortConfig;

/* ---------------------------------------------------------------------------
 * Functions
 * -----------------------------------------------------------------------*/

/* NID: 0x0B168F92 */
s32 cellAudioInit(void);

/* NID: 0xCA5AC370 */
s32 cellAudioQuit(void);

/* NID: 0xCD7BC431 */
s32 cellAudioPortOpen(const CellAudioPortParam* param, u32* portNum);

/* NID: 0x4129FE2D */
s32 cellAudioPortClose(u32 portNum);

/* NID: 0x89BE28F2 */
s32 cellAudioPortStart(u32 portNum);

/* NID: 0x5B1E2C73 */
s32 cellAudioPortStop(u32 portNum);

/* NID: 0xDAB029AA. `volume` is passed in FPR1 in the PPU ABI and therefore
 * the runtime uses a context-aware bridge for this entry point. */
s32 cellAudioAddData(u32 portNum, const float* data, u32 samples, float volume);
s32 cellAudioAdd2chData(u32 portNum, const float* data, u32 samples, float volume);
s32 cellAudioAdd6chData(u32 portNum, const float* data, float volume);

/* NID: 0x04AF134E */
s32 cellAudioCreateNotifyEventQueue(u32* id, u64* key);
s32 cellAudioCreateNotifyEventQueueEx(u32* id, u64* key, u32 flags);

/* NID: 0x377E0CD9 */
s32 cellAudioSetNotifyEventQueue(u64 key);
s32 cellAudioSetNotifyEventQueueEx(u64 key, u32 flags);

/* NID: 0xFF3626FD */
s32 cellAudioRemoveNotifyEventQueue(u64 key);
s32 cellAudioRemoveNotifyEventQueueEx(u64 key, u32 flags);

/* NID: 0x17D1213B */
s32 cellAudioSendAck(u64 data3);

/* NID: 0x31211F6B; volume is passed in FPR1 by the PPU ABI. */
s32 cellAudioMiscSetAccessoryVolume(u32 devNum, float volume);

/* NID: 0x74A66AF0 */
s32 cellAudioGetPortConfig(u32 portNum, CellAudioPortConfig* config);

/* NID: 0xB8EF8006 */
s32 cellAudioPortGetStatus(u32 portNum, u32* status);
s32 cellAudioSetPortLevel(u32 portNum, float level);

/* NID: 0xE4046AFE */
s32 cellAudioGetPortBlockTag(u32 portNum, u64 blockNo, u64* tag);

/* NID: 0x4109D08C */
s32 cellAudioGetPortTimestamp(u32 portNum, u64 tag, u64* stamp);

/* NID: 0x5676F81C */
s32 cellAudioSetPersonalDevice(s32 iPersonalStream, s32 iDevice);

/* NID: 0x28BC1409 */
s32 cellAudioUnsetPersonalDevice(s32 iPersonalStream);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_AUDIO_H */
