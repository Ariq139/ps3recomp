/*
 * ps3recomp - cellVoice HLE implementation.
 *
 * Port lifecycle, validation, and notification behavior were independently
 * implemented from the PS3 ABI and compared with RPCS3 cellVoice.cpp
 * (GPL-2.0, checkout commit 2ba2f75bc). No RPCS3 implementation body is copied.
 * The runtime has no microphone capture or PS3 voice codec, so data transfer
 * is intentionally unavailable/empty rather than simulated as working audio.
 */

#include "cellVoice.h"
#include <string.h>
#include <stdint.h>
#include "../../runtime/ppu/ppu_memory.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
static SRWLOCK s_voice_lock = SRWLOCK_INIT;
static void voice_lock(void)   { AcquireSRWLockExclusive(&s_voice_lock); }
static void voice_unlock(void) { ReleaseSRWLockExclusive(&s_voice_lock); }
#else
#  include <pthread.h>
static pthread_mutex_t s_voice_lock = PTHREAD_MUTEX_INITIALIZER;
static void voice_lock(void)   { pthread_mutex_lock(&s_voice_lock); }
static void voice_unlock(void) { pthread_mutex_unlock(&s_voice_lock); }
#endif

/* Implemented by the shared LV2 event runtime. */
extern int sys_event_queue_push_by_id(uint32_t queue_id, uint64_t source,
                                      uint64_t data1, uint64_t data2,
                                      uint64_t data3);
extern uint32_t sys_event_queue_create_direct(uint64_t key, int32_t size);
extern uint32_t sys_event_find_queue_by_key(uint64_t key);

#define VOICE_MAX_QUEUES 128
#define VOICE_MAX_SOURCES_PER_QUEUE 128
#define VOICE_QUEUE_KEY_BASE 0x80004d494f323285ull
#define VOICE_EVENT_SOURCE_BASE 0x0e000000u

typedef struct VoicePort {
    int active;
    u16 id;
    u32 type;
    u16 threshold;
    u16 muted;
    float volume;
    u32 bitrate;
    u32 buffer_size;
    u8 channels;
    u8 sample_alignment;
    u32 data_type;
    u32 sample_rate;
    u32 player_id;
} VoicePort;

typedef struct VoiceNotifyQueue {
    int active;
    u64 key;
    u64 sources[VOICE_MAX_SOURCES_PER_QUEUE];
    u32 source_count;
} VoiceNotifyQueue;

static VoicePort s_ports[CELL_VOICE_PORT_MAX];
static VoiceNotifyQueue s_notify_queues[VOICE_MAX_QUEUES];
static int s_initialized;
static int s_service_started;
static u8 s_id_counter;
static u32 s_event_source_counter;

static u32 guest_ea(const void* p)
{
    return (u32)(uintptr_t)p;
}

static VoicePort* voice_port_find(u32 id)
{
    const u16 guest_id = (u16)id; /* The public ABI ignores the upper 16 bits. */
    for (u32 i = 0; i < CELL_VOICE_PORT_MAX; ++i) {
        if (s_ports[i].active && s_ports[i].id == guest_id)
            return &s_ports[i];
    }
    return NULL;
}

static VoicePort* voice_port_slot(void)
{
    for (u32 i = 0; i < CELL_VOICE_PORT_MAX; ++i) {
        if (!s_ports[i].active)
            return &s_ports[i];
    }
    return NULL;
}

static int voice_valid_bitrate(u32 bitrate)
{
    return bitrate == CELL_VOICE_BITRATE_3850 ||
           bitrate == CELL_VOICE_BITRATE_4650 ||
           bitrate == CELL_VOICE_BITRATE_5700 ||
           bitrate == CELL_VOICE_BITRATE_7300 ||
           bitrate == CELL_VOICE_BITRATE_14400 ||
           bitrate == CELL_VOICE_BITRATE_16000 ||
           bitrate == CELL_VOICE_BITRATE_22533;
}

static void voice_read_param(u32 ea, VoicePort* port)
{
    port->type = vm_read32(ea + 0x00);
    port->threshold = vm_read16(ea + 0x04);
    port->muted = vm_read16(ea + 0x06);
    port->volume = vm_read_f32(ea + 0x08);
    port->bitrate = 0;
    port->buffer_size = 0;
    port->channels = 0;
    port->sample_alignment = 0;
    port->data_type = 0;
    port->sample_rate = 0;
    port->player_id = 0;

    switch (port->type) {
    case CELL_VOICE_PORT_IN_VOICE:
    case CELL_VOICE_PORT_OUT_VOICE:
        port->bitrate = vm_read32(ea + 0x0c);
        break;
    case CELL_VOICE_PORT_IN_PCMAUDIO:
    case CELL_VOICE_PORT_OUT_PCMAUDIO:
        port->buffer_size = vm_read32(ea + 0x0c);
        port->channels = vm_read8(ea + 0x10);
        port->sample_alignment = vm_read8(ea + 0x11);
        port->data_type = vm_read32(ea + 0x12);
        port->sample_rate = vm_read32(ea + 0x16);
        break;
    case CELL_VOICE_PORT_OUT_SECONDARY:
        port->player_id = vm_read32(ea + 0x0c);
        break;
    default:
        break;
    }
}

static int voice_valid_param(const VoicePort* port)
{
    switch (port->type) {
    case CELL_VOICE_PORT_IN_MIC:
    case CELL_VOICE_PORT_OUT_SECONDARY:
        return 1;
    case CELL_VOICE_PORT_IN_PCMAUDIO:
    case CELL_VOICE_PORT_OUT_PCMAUDIO:
        return port->data_type <= CELL_VOICE_PCM_INTEGER_LITTLE_ENDIAN;
    case CELL_VOICE_PORT_IN_VOICE:
    case CELL_VOICE_PORT_OUT_VOICE:
        return voice_valid_bitrate(port->bitrate);
    default:
        return 0;
    }
}

static void voice_write_service_event(u64 data)
{
    for (u32 i = 0; i < VOICE_MAX_QUEUES; ++i) {
        VoiceNotifyQueue* q = &s_notify_queues[i];
        if (!q->active)
            continue;
        const u32 qid = sys_event_find_queue_by_key(q->key);
        if (!qid)
            continue;
        for (u32 j = 0; j < q->source_count; ++j)
            (void)sys_event_queue_push_by_id(qid, q->sources[j], data, 0, 0);
    }
}

static s32 voice_start_locked(void)
{
    if (!s_service_started) {
        s_service_started = 1;
        voice_write_service_event(CELL_VOICE_EVENT_SERVICE_ATTACHED);
    }
    return CELL_OK;
}

static s32 voice_stop_locked(void)
{
    if (s_service_started) {
        s_service_started = 0;
        voice_write_service_event(CELL_VOICE_EVENT_SERVICE_DETACHED);
    }
    return CELL_OK;
}

static VoiceNotifyQueue* voice_notify_queue_find(u64 key)
{
    for (u32 i = 0; i < VOICE_MAX_QUEUES; ++i) {
        if (s_notify_queues[i].active && s_notify_queues[i].key == key)
            return &s_notify_queues[i];
    }
    return NULL;
}

s32 cellVoiceInit(const CellVoiceInitParam* param)
{
    voice_lock();
    if (s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ALREADY_INITIALIZED;
    }
    if (!param) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    memset(s_ports, 0, sizeof(s_ports));
    memset(s_notify_queues, 0, sizeof(s_notify_queues));
    s_id_counter = 0;
    s_event_source_counter = 0;
    s_service_started = 0;
    s_initialized = 1;
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceInitEx(const CellVoiceInitParam* param)
{
    return cellVoiceInit(param);
}

s32 cellVoiceEnd(void)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    (void)voice_stop_locked();
    memset(s_ports, 0, sizeof(s_ports));
    memset(s_notify_queues, 0, sizeof(s_notify_queues));
    s_initialized = 0;
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceCreateNotifyEventQueue(u32* queueId, u64* key)
{
    if (!queueId || !key)
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;

    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }

    for (u32 i = 0; i < 10; ++i) {
        const u64 candidate = VOICE_QUEUE_KEY_BASE + i;
        if (sys_event_find_queue_by_key(candidate))
            continue;
        const u32 id = sys_event_queue_create_direct(candidate, 0x40);
        if (!id)
            continue;
        vm_write32(guest_ea(queueId), id);
        vm_write64(guest_ea(key), candidate);
        voice_unlock();
        return CELL_OK;
    }
    voice_unlock();
    return (s32)CELL_VOICE_ERROR_EVENT_QUEUE;
}

s32 cellVoiceRemoveNotifyEventQueue(u64 key)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoiceNotifyQueue* q = voice_notify_queue_find(key);
    if (!q) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_EVENT_QUEUE;
    }
    memset(q, 0, sizeof(*q));
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceSetNotifyEventQueue(u64 key, u64 source)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    if (!sys_event_find_queue_by_key(key)) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_EVENT_QUEUE;
    }

    VoiceNotifyQueue* q = voice_notify_queue_find(key);
    if (!q) {
        for (u32 i = 0; i < VOICE_MAX_QUEUES; ++i) {
            if (!s_notify_queues[i].active) {
                q = &s_notify_queues[i];
                q->active = 1;
                q->key = key;
                break;
            }
        }
    }
    if (!q) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_RESOURCE_INSUFFICIENT;
    }
    if (q->source_count >= VOICE_MAX_SOURCES_PER_QUEUE) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_RESOURCE_INSUFFICIENT;
    }
    if (!source) {
        source = ((u64)2 << 32) |
                 (u64)(VOICE_EVENT_SOURCE_BASE + s_event_source_counter++);
    }
    q->sources[q->source_count++] = source;
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceCreatePort(CellVoicePortId* portId, const CellVoicePortParam* param)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    if (!portId || !param) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    VoicePort candidate;
    memset(&candidate, 0, sizeof(candidate));
    voice_read_param(guest_ea(param), &candidate);
    if (!voice_valid_param(&candidate)) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    VoicePort* slot = voice_port_slot();
    if (!slot) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_RESOURCE_INSUFFICIENT;
    }

    s_id_counter = (u8)((s_id_counter + 1) % 0xa0);
    for (u32 high = 1; high <= CELL_VOICE_PORT_MAX; ++high) {
        const u16 id = (u16)((high << 8) | s_id_counter);
        if (voice_port_find(id))
            continue;
        candidate.active = 1;
        candidate.id = id;
        *slot = candidate;
        vm_write32(guest_ea(portId), id);
        voice_unlock();
        return CELL_OK;
    }
    voice_unlock();
    return (s32)CELL_VOICE_ERROR_RESOURCE_INSUFFICIENT;
}

s32 cellVoiceDeletePort(CellVoicePortId portId)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    memset(port, 0, sizeof(*port));
    voice_unlock();
    return CELL_OK;
}

static s32 voice_validate_connection(u32 inPortId, u32 outPortId)
{
    VoicePort* in = voice_port_find(inPortId);
    VoicePort* out = voice_port_find(outPortId);
    if (!in || in->type >= CELL_VOICE_PORT_OUT_PCMAUDIO ||
        !out || out->type <= CELL_VOICE_PORT_IN_VOICE)
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    return CELL_OK;
}

s32 cellVoiceConnectIPortToOPort(CellVoicePortId inPortId, CellVoicePortId outPortId)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    const s32 result = voice_validate_connection(inPortId, outPortId);
    voice_unlock();
    return result;
}

s32 cellVoiceDisconnectIPortFromOPort(CellVoicePortId inPortId, CellVoicePortId outPortId)
{
    return cellVoiceConnectIPortToOPort(inPortId, outPortId);
}

s32 cellVoiceUpdatePort(CellVoicePortId portId, const CellVoicePortParam* param)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    if (!param) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    const u32 ea = guest_ea(param);
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    port->muted = vm_read16(ea + 0x06);
    port->volume = vm_read_f32(ea + 0x08);
    port->threshold = vm_read16(ea + 0x04);
    if (port->type == CELL_VOICE_PORT_IN_VOICE ||
        port->type == CELL_VOICE_PORT_OUT_VOICE)
        port->bitrate = vm_read32(ea + 0x0c);
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceGetPortInfo(CellVoicePortId portId, CellVoiceBasePortInfo* info)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    if (!s_service_started) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_SERVICE_DETACHED;
    }
    if (!info) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    vm_write32(guest_ea(info), port->type);
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceStart(void)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    const s32 result = voice_start_locked();
    voice_unlock();
    return result;
}

s32 cellVoiceStartEx(const CellVoiceStartParam* param)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    if (!param) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    const s32 result = voice_start_locked(); /* Container validation is not modeled by this VM. */
    voice_unlock();
    return result;
}

s32 cellVoiceStop(void)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    const s32 result = voice_stop_locked();
    voice_unlock();
    return result;
}

static s32 voice_validate_port(CellVoicePortId portId)
{
    if (!s_initialized)
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    return voice_port_find(portId) ? CELL_OK : (s32)CELL_VOICE_ERROR_TOPOLOGY;
}

s32 cellVoicePausePort(CellVoicePortId portId)
{
    voice_lock(); const s32 result = voice_validate_port(portId); voice_unlock(); return result;
}
s32 cellVoiceResumePort(CellVoicePortId portId)
{
    voice_lock(); const s32 result = voice_validate_port(portId); voice_unlock(); return result;
}
s32 cellVoiceResetPort(CellVoicePortId portId)
{
    voice_lock(); const s32 result = voice_validate_port(portId); voice_unlock(); return result;
}
s32 cellVoicePausePortAll(void)
{
    voice_lock(); const s32 result = s_initialized ? CELL_OK : (s32)CELL_VOICE_ERROR_NOT_INITIALIZED; voice_unlock(); return result;
}
s32 cellVoiceResumePortAll(void)
{
    voice_lock(); const s32 result = s_initialized ? CELL_OK : (s32)CELL_VOICE_ERROR_NOT_INITIALIZED; voice_unlock(); return result;
}

s32 cellVoiceGetBitRate(CellVoicePortId portId, u32* bitrate)
{
    if (!bitrate)
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    const u32 out_ea = guest_ea(bitrate);
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    vm_write32(out_ea, 0x4f323285u);
    VoicePort* port = voice_port_find(portId);
    if (!port || (port->type != CELL_VOICE_PORT_IN_VOICE &&
                  port->type != CELL_VOICE_PORT_OUT_VOICE)) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    vm_write32(out_ea, port->bitrate);
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceSetBitRate(CellVoicePortId portId, u32 bitrate)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port || (port->type != CELL_VOICE_PORT_IN_VOICE &&
                  port->type != CELL_VOICE_PORT_OUT_VOICE)) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    if (!voice_valid_bitrate(bitrate)) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    }
    port->bitrate = bitrate;
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceGetMuteFlag(CellVoicePortId portId, u16* muted)
{
    if (!muted)
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    vm_write16(guest_ea(muted), port->muted);
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceSetMuteFlag(CellVoicePortId portId, u16 muted)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    port->muted = muted;
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceSetMuteFlagAll(u16 muted)
{
    (void)muted; /* RPCS3 accepts this call without changing per-port mute state. */
    voice_lock();
    const s32 result = s_initialized ? CELL_OK : (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    voice_unlock();
    return result;
}

s32 cellVoiceGetVolume(CellVoicePortId portId, float* volume)
{
    if (!volume)
        return (s32)CELL_VOICE_ERROR_ARGUMENT_INVALID;
    const u32 out_ea = guest_ea(volume);
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        vm_write32(out_ea, 0x4f323285u);
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    vm_write_f32(out_ea, port->volume);
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceSetVolume(CellVoicePortId portId, float volume)
{
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    port->volume = volume;
    voice_unlock();
    return CELL_OK;
}

static s32 voice_device_attr_result(CellVoicePortId portId)
{
    if (!s_initialized)
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    return voice_port_find(portId) ?
        (s32)CELL_VOICE_ERROR_DEVICE_NOT_PRESENT : (s32)CELL_VOICE_ERROR_TOPOLOGY;
}

s32 cellVoiceGetPortAttr(CellVoicePortId portId, u32 attr, void* value)
{
    (void)attr; (void)value;
    voice_lock(); const s32 result = voice_device_attr_result(portId); voice_unlock(); return result;
}
s32 cellVoiceSetPortAttr(CellVoicePortId portId, u32 attr, const void* value)
{
    (void)attr; (void)value;
    voice_lock(); const s32 result = voice_device_attr_result(portId); voice_unlock(); return result;
}
s32 cellVoiceGetSignalState(CellVoicePortId portId, u32 attr, void* value)
{
    (void)attr; (void)value;
    voice_lock(); const s32 result = voice_device_attr_result(portId); voice_unlock(); return result;
}

static s32 voice_validate_input_port(CellVoicePortId portId)
{
    if (!s_initialized)
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    VoicePort* port = voice_port_find(portId);
    return (!port || port->type >= CELL_VOICE_PORT_OUT_PCMAUDIO) ?
        (s32)CELL_VOICE_ERROR_TOPOLOGY : CELL_OK;
}

s32 cellVoiceWriteToIPort(CellVoicePortId portId, const void* data, u32* size)
{
    (void)data; (void)size;
    voice_lock(); const s32 result = voice_validate_input_port(portId); voice_unlock(); return result;
}
s32 cellVoiceWriteToIPortEx(CellVoicePortId portId, const void* data, u32* size,
                            u32 numFrameLost)
{
    (void)data; (void)size; (void)numFrameLost;
    voice_lock(); const s32 result = voice_validate_input_port(portId); voice_unlock(); return result;
}
s32 cellVoiceWriteToIPortEx2(CellVoicePortId portId, const void* data, u32* size,
                             s16 frameGaps)
{
    (void)data; (void)size; (void)frameGaps;
    voice_lock(); const s32 result = voice_validate_input_port(portId); voice_unlock(); return result;
}

s32 cellVoiceReadFromOPort(CellVoicePortId portId, void* data, u32* size)
{
    (void)data;
    const u32 size_ea = size ? guest_ea(size) : 0;
    voice_lock();
    if (!s_initialized) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    }
    VoicePort* port = voice_port_find(portId);
    if (!port || port->type <= CELL_VOICE_PORT_IN_VOICE) {
        voice_unlock();
        return (s32)CELL_VOICE_ERROR_TOPOLOGY;
    }
    if (size)
        vm_write32(size_ea, 0);
    voice_unlock();
    return CELL_OK;
}

s32 cellVoiceDebugTopology(void)
{
    voice_lock();
    const s32 result = s_initialized ? (s32)CELL_VOICE_ERROR_NOT_IMPLEMENTED :
                                      (s32)CELL_VOICE_ERROR_NOT_INITIALIZED;
    voice_unlock();
    return result;
}
