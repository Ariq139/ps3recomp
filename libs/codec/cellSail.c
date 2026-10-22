/*
 * ps3recomp - cellSail HLE implementation
 *
 * Stub. Player lifecycle and state management work.
 * No actual media playback occurs — games typically show
 * a black screen for cutscenes, which is acceptable for
 * initial recompilation testing.
 */

#include "ps3emu/guest_call.h"
#include "cellSail.h"
#include "../../runtime/ppu/ppu_memory.h"   /* vm_read32 / vm_write32 */
#include "../../runtime/memory/vm.h"        /* ppu_vm_size */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>   /* getenv -- Clang errors on the implicit declaration MSVC allows */
#include "../guest_struct.h"   /* GUEST_EA, guest_struct_load/store */

/* Internal state */

static int s_initialized = 0;
static int sail_guest_range(u32 ea, u32 n);

typedef struct {
    int in_use;
    s32 state;
    CellSailPlayerCallback callback;
    void* callbackArg;
} SailPlayer;

static SailPlayer s_players[CELL_SAIL_PLAYER_MAX];

/* Players registered by cellSailPlayerInitialize2, keyed by the guest struct EA
 * the title passed as `pSelf`. Declared here because the adapter setters below
 * need it -- they take that same pointer, not an index into s_players[]. */
static struct { u32 self_ea, allocator_ea, callback_ea, arg_ea; int in_use; int state; }
    s_sail_players[CELL_SAIL_PLAYER_MAX];

/* RPCS3_REFERENCE_REVIEWED: CellSailPlayer field layout and state enum checked
 * against RPCS3's cellSail.h/cellSail.cpp. The guest-memory writes below are
 * local code; no RPCS3 function body was copied. Pointer fields are 32-bit
 * guest addresses, so the host struct layout must never be used here. */
#define SAIL_PLAYER_ABI_SIZE       0x100u
#define SAIL_PLAYER_ALLOCATOR      0x00u
#define SAIL_PLAYER_CALLBACK       0x08u
#define SAIL_PLAYER_CALLBACK_ARG   0x0Cu
#define SAIL_PLAYER_ATTRIBUTE      0x10u
#define SAIL_PLAYER_RESOURCE       0x30u
#define SAIL_PLAYER_PLAYBACK_COMMAND 0x40u
#define SAIL_PLAYER_REPEAT_MODE    0x44u
#define SAIL_PLAYER_PAUSED         0x54u
#define SAIL_PLAYER_BOOTED         0x55u
#define SAIL_PLAYER_SOUND_ADAPTER  0x58u
#define SAIL_PLAYER_GRAPHICS_ADAPTER 0x5Cu

/* A CellSailPlayer is a GUEST POINTER, not a small integer handle.
 *
 * Every player entry point below used to validate its first argument as an
 * index (`handle >= CELL_SAIL_PLAYER_MAX`) into the handle-keyed s_players[]
 * table, which rejects every real call: a guest pointer is always larger than
 * the table. cellSailPlayerInitialize2 has always registered the player by its
 * `pSelf` EA, so the two halves of this file disagreed about what a handle is.
 *
 * That is not a harmless error return. Tokyo Jungle sets its sound adapter
 * during audio init; on failure it abandons the path, closes a resource it
 * never opened ("sgxResClose unknown ID (0)"), terminates its SGX sound system
 * and then blocks forever joining an audio thread that will not exit.
 *
 * One lookup, used by all of them, so the disagreement cannot come back. */
static int sail_player_slot(u32 self_ea)
{
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++)
        if (s_sail_players[i].in_use && s_sail_players[i].self_ea == self_ea)
            return i;
    return -1;
}

/* Lifecycle */

s32 cellSailInit(void)
{
    printf("[cellSail] Init()\n");
    if (s_initialized)
        return (s32)CELL_SAIL_ERROR_ALREADY_INITIALIZED;
    memset(s_players, 0, sizeof(s_players));
    s_initialized = 1;
    return CELL_OK;
}

s32 cellSailTerm(void)
{
    printf("[cellSail] Term()\n");
    s_initialized = 0;
    return CELL_OK;
}

/* Player management */

s32 cellSailPlayerCreate(const CellSailPlayerAttribute* attr,
                           const CellSailPlayerResource* resource,
                           CellSailPlayerCallback callback,
                           void* callbackArg,
                           CellSailPlayerHandle* handle)
{
    (void)attr; (void)resource;
    printf("[cellSail] PlayerCreate()\n");

    if (!s_initialized) return (s32)CELL_SAIL_ERROR_NOT_INITIALIZED;
    if (!handle) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) {
        if (!s_players[i].in_use) {
            memset(&s_players[i], 0, sizeof(SailPlayer));
            s_players[i].in_use = 1;
            s_players[i].state = CELL_SAIL_PLAYER_STATE_INITIALIZED;
            s_players[i].callback = callback;
            s_players[i].callbackArg = callbackArg;
            vm_write32((u32)(uintptr_t)handle, (u32)i);   /* guest out-param */
            return CELL_OK;
        }
    }
    return (s32)CELL_SAIL_ERROR_OUT_OF_MEMORY;
}

s32 cellSailPlayerDestroy(CellSailPlayerHandle handle)
{
    printf("[cellSail] PlayerDestroy(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].in_use = 0;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_CLOSED;
    return CELL_OK;
}

/* CellSailEvent is {be32 major; be32 minor} -- one 64-bit register, major in
 * the high half. Async player calls report completion as major=2
 * (CALL_COMPLETED) with minor = the CellSailPlayerCall id that finished; a
 * title that drives SAIL asynchronously blocks until its handler sees that.
 * We run every call synchronously, so notify right after each one. */
#define SAIL_EV_CALL_COMPLETED 2
enum { SAIL_CALL_BOOT = 1, SAIL_CALL_OPEN_STREAM = 2, SAIL_CALL_CLOSE_STREAM = 3,
       SAIL_CALL_START = 10, SAIL_CALL_STOP = 11 };

static void sail_notify(int slot, u32 major, u32 minor, u64 arg0, u64 arg1)
{
    if (slot < 0 || !g_ps3_guest_caller) return;
    const u32 cb = s_sail_players[slot].callback_ea;
    if (!cb) return;
    g_ps3_guest_caller(cb, s_sail_players[slot].arg_ea,
                       ((u64)major << 32) | minor, arg0, arg1, 0, 0, 0, 0);
}

/* The player also reports every state it passes through: major=3
 * (PLAYER_STATE_CHANGED), minor 0, arg0 = the new state in the SDK's
 * numbering. Midway Arcade Origins maps arg0 through a
 * nine-entry table (CLOSED=2, OPENED=4, RUNNING=6, ...) and opens nothing until
 * boot has reported CLOSED; without these events it boots the player and then
 * waits on a black screen for good. */
#define SAIL_EV_STATE_CHANGED 3
enum { SAIL_ST_BOOT_TRANSITION = 1, SAIL_ST_CLOSED = 2, SAIL_ST_OPEN_TRANSITION = 3,
       SAIL_ST_OPENED = 4, SAIL_ST_START_TRANSITION = 5, SAIL_ST_RUNNING = 6,
       SAIL_ST_STOP_TRANSITION = 7, SAIL_ST_CLOSE_TRANSITION = 8 };

/* One synchronous call: transition state, completion, settled state.
 * The call token belongs to completion arg1, not either state notification. */
static void sail_call(int slot, u32 call, u32 via, u32 to, u64 userParam)
{
    sail_notify(slot, SAIL_EV_STATE_CHANGED, 0, via, 0);
    sail_notify(slot, SAIL_EV_CALL_COMPLETED, call, 0, userParam);
    sail_notify(slot, SAIL_EV_STATE_CHANGED, 0, to, 0);
}

s32 cellSailPlayerBoot(CellSailPlayerHandle handle, u64 userParam)
{
    printf("[cellSail] PlayerBoot(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_CLOSED;
    vm_write8(s_sail_players[_sp].self_ea + SAIL_PLAYER_BOOTED, 1);
    sail_call(_sp, SAIL_CALL_BOOT, SAIL_ST_BOOT_TRANSITION, SAIL_ST_CLOSED, userParam);
    return CELL_OK;
}

s32 cellSailPlayerOpenStream(CellSailPlayerHandle handle, const char* path)
{
    /* path is a GUEST address; printing it as a host char* faulted here the
     * same way it did in CreateDescriptor. Copy it out of guest memory. */
    char path_host[512];
    path_host[0] = '\0';
    if (path) {
        u32 ea = (u32)(uintptr_t)path;
        size_t n = 0;
        while (n < sizeof(path_host) - 1) {
            char c = (char)vm_read8(ea + (u32)n);
            if (!c) break;
            path_host[n++] = c;
        }
        path_host[n] = '\0';
    }
    printf("[cellSail] PlayerOpenStream(%u, \"%s\") - stub\n", handle,
           path ? path_host : "null");
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    /* Stub: don't actually open anything */
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_OPENED;
    sail_call(_sp, SAIL_CALL_OPEN_STREAM, SAIL_ST_OPEN_TRANSITION, SAIL_ST_OPENED, 0);
    return CELL_OK;
}

s32 cellSailPlayerCloseStream(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_CLOSED;
    sail_call(_sp, SAIL_CALL_CLOSE_STREAM, SAIL_ST_CLOSE_TRANSITION, SAIL_ST_CLOSED, 0);
    return CELL_OK;
}

s32 cellSailPlayerStart(CellSailPlayerHandle handle)
{
    printf("[cellSail] PlayerStart(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    /* No media backend is attached, so model the completed stream and settle
     * back to OPENED after reporting its transition. */
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_OPENED;
    sail_call(_sp, SAIL_CALL_START, SAIL_ST_START_TRANSITION, SAIL_ST_RUNNING, 0);
    /* Nothing decodes the stream, so it has already ended: a finished stream
     * drops the player back to OPENED, which is what a title waiting out a
     * movie (Midway's logo AVIs) watches for. */
    sail_notify(_sp, SAIL_EV_STATE_CHANGED, 0, SAIL_ST_STOP_TRANSITION, 0);
    sail_notify(_sp, SAIL_EV_STATE_CHANGED, 0, SAIL_ST_OPENED, 0);
    return CELL_OK;
}

s32 cellSailPlayerStop(CellSailPlayerHandle handle)
{
    printf("[cellSail] PlayerStop(%u)\n", handle);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_OPENED;
    sail_call(_sp, SAIL_CALL_STOP, SAIL_ST_STOP_TRANSITION, SAIL_ST_OPENED, 0);
    return CELL_OK;
}

s32 cellSailPlayerPause(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write8(s_sail_players[_sp].self_ea + SAIL_PLAYER_PAUSED, 1);
    return CELL_OK;
}

s32 cellSailPlayerGetState(CellSailPlayerHandle handle, s32* state)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!state) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)state, (u32)s_sail_players[_sp].state);
    return CELL_OK;
}

s32 cellSailPlayerGetStreamNum(CellSailPlayerHandle handle, u32* streamNum)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!streamNum) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)streamNum, 0);   /* no streams in stub */
    return CELL_OK;
}

s32 cellSailPlayerGetStreamInfo(CellSailPlayerHandle handle, u32 streamIndex,
                                  CellSailStreamInfo* info)
{
    (void)streamIndex;
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!info) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return (s32)CELL_SAIL_ERROR_NOT_FOUND;
}

/* r3 is a CellSailPlayer* -- a GUEST POINTER, the same `pSelf` that
 * cellSailPlayerInitialize2 registered -- not a small integer handle. Validating
 * it as an index (`handle >= CELL_SAIL_PLAYER_MAX`) rejects every real call,
 * because a guest pointer is always larger than the table size.
 *
 * That is not a harmless error return: Tokyo Jungle sets its sound adapter
 * during audio init, and on failure abandons the whole path -- it closes a
 * resource it never opened ("sgxResClose unknown ID (0)"), terminates its SGX
 * sound system, and then blocks forever joining an audio thread that is not
 * going to exit. Look the player up the way Initialize2 stored it. */
s32 cellSailPlayerSetSoundAdapter(CellSailPlayerHandle handle, u32 index, void* adapter)
{
    u32 self_ea = (u32)handle;
    const int slot = sail_player_slot(self_ea);
    const u32 adapter_ea = (u32)(uintptr_t)adapter;
    if (slot < 0 || !sail_guest_range(self_ea, SAIL_PLAYER_ABI_SIZE) ||
        !sail_guest_range(adapter_ea, 0x30u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (index > vm_read32(self_ea + SAIL_PLAYER_ATTRIBUTE + 4u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32(self_ea + SAIL_PLAYER_SOUND_ADAPTER, adapter_ea);
    vm_write32(adapter_ea + 20u, index);
    vm_write8(adapter_ea + 1u, 1);
    return CELL_OK;
}

/* r3 is a CellSailPlayer* -- a GUEST POINTER, the same `pSelf` that
 * cellSailPlayerInitialize2 registered -- not a small integer handle. Validating
 * it as an index (`handle >= CELL_SAIL_PLAYER_MAX`) rejects every real call,
 * because a guest pointer is always larger than the table size.
 *
 * That is not a harmless error return: Tokyo Jungle sets its sound adapter
 * during audio init, and on failure abandons the whole path -- it closes a
 * resource it never opened ("sgxResClose unknown ID (0)"), terminates its SGX
 * sound system, and then blocks forever joining an audio thread that is not
 * going to exit. Look the player up the way Initialize2 stored it. */
s32 cellSailPlayerSetGraphicsAdapter(CellSailPlayerHandle handle, u32 index, void* adapter)
{
    u32 self_ea = (u32)handle;
    const int slot = sail_player_slot(self_ea);
    const u32 adapter_ea = (u32)(uintptr_t)adapter;
    if (slot < 0 || !sail_guest_range(self_ea, SAIL_PLAYER_ABI_SIZE) ||
        !sail_guest_range(adapter_ea, 0x40u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (index > vm_read32(self_ea + SAIL_PLAYER_ATTRIBUTE + 8u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32(self_ea + SAIL_PLAYER_GRAPHICS_ADAPTER, adapter_ea);
    vm_write32(adapter_ea + 60u, index);
    vm_write8(adapter_ea + 1u, 1);
    return CELL_OK;
}

s32 cellSailPlayerCancel(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_sail_players[_sp].state = CELL_SAIL_PLAYER_STATE_OPENED;
    return CELL_OK;
}

s32 cellSailPlayerIsPaused(CellSailPlayerHandle handle)
{
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return 0;
    return vm_read8(s_sail_players[_sp].self_ea + SAIL_PLAYER_PAUSED) != 0;
}

s32 cellSailPlayerSetRepeatMode(CellSailPlayerHandle handle, s32 repeatMode, void* command)
{
    printf("[cellSail] SetRepeatMode(%u, %d)\n", handle, repeatMode);
    const int _sp = sail_player_slot((u32)handle);
    if (_sp < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    const u32 self_ea = s_sail_players[_sp].self_ea;
    vm_write32(self_ea + SAIL_PLAYER_PLAYBACK_COMMAND,
               (u32)(uintptr_t)command);
    vm_write32(self_ea + SAIL_PLAYER_REPEAT_MODE, (u32)repeatMode);
    /* Real ABI returns the (now-set) repeat mode in r3, not an error code. */
    return repeatMode;
}

/* ---------------------------------------------------------------------------
 * Descriptor management
 * -----------------------------------------------------------------------*/

#define MAX_DESCRIPTORS 16

typedef struct {
    int in_use;
    s32 streamType;
    CellSailPlayerHandle player;
    char uri[512];
} SailDescriptor;

static SailDescriptor s_descs[MAX_DESCRIPTORS];

s32 cellSailPlayerCreateDescriptor(CellSailPlayerHandle handle,
                                     s32 streamType, void* mediaInfo,
                                     char* uri,
                                     CellSailDescriptorHandle* desc)
{
    (void)mediaInfo;

    /* uri is a GUEST address. Printing it as a host char* faulted the moment
     * the title opened a movie -- the log line stopped mid-token at
     * 'uri="' and the process died with 0xC0000005. Copy it out of guest
     * memory first. The descriptor has always had a uri[512] field; nothing
     * ever filled it, so the URI was discarded even when it did not crash. */
    char uri_host[512];
    uri_host[0] = '\0';
    if (uri) {
        u32 ea = (u32)(uintptr_t)uri;
        size_t n = 0;
        while (n < sizeof(uri_host) - 1) {
            char c = (char)vm_read8(ea + (u32)n);
            if (!c) break;
            uri_host[n++] = c;
        }
        uri_host[n] = '\0';
    }

    printf("[cellSail] CreateDescriptor(player=%u, type=%d, uri=\"%s\")\n", handle,
           streamType, uri ? uri_host : "null");
    if (!desc) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    for (int i = 0; i < MAX_DESCRIPTORS; i++) {
        if (!s_descs[i].in_use) {
            memset(&s_descs[i], 0, sizeof(SailDescriptor));
            s_descs[i].in_use = 1;
            s_descs[i].streamType = streamType;
            s_descs[i].player = handle;
            snprintf(s_descs[i].uri, sizeof(s_descs[i].uri), "%s", uri_host);
            vm_write32((u32)(uintptr_t)desc, (u32)i);   /* guest out-param */
            return CELL_OK;
        }
    }
    return (s32)CELL_SAIL_ERROR_OUT_OF_MEMORY;
}

s32 cellSailPlayerDestroyDescriptor(CellSailPlayerHandle handle,
                                      CellSailDescriptorHandle desc)
{
    (void)handle;
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    s_descs[desc].in_use = 0;
    return CELL_OK;
}

s32 cellSailPlayerAddDescriptor(CellSailPlayerHandle handle,
                                  CellSailDescriptorHandle desc)
{
    (void)handle;
    printf("[cellSail] AddDescriptor(player=%u, desc=%u)\n", handle, desc);
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailPlayerRemoveDescriptor(CellSailPlayerHandle handle,
                                     CellSailDescriptorHandle desc)
{
    (void)handle;
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorCreateDatabase(CellSailDescriptorHandle desc,
                                       void* dbAddr, u32 dbSize, u64 arg)
{
    (void)dbAddr; (void)dbSize; (void)arg;
    printf("[cellSail] DescriptorCreateDatabase(desc=%u)\n", desc);
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorDestroyDatabase(CellSailDescriptorHandle desc)
{
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorGetStreamType(CellSailDescriptorHandle desc, s32* type)
{
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use || !type)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)type, (u32)s_descs[desc].streamType);
    return CELL_OK;
}

s32 cellSailDescriptorSetAutoSelection(CellSailDescriptorHandle desc, s32 enable)
{
    (void)enable;
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 cellSailDescriptorGetUri(CellSailDescriptorHandle desc, char* uri, u32 maxLen)
{
    if (desc >= MAX_DESCRIPTORS || !s_descs[desc].in_use || !uri)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    strncpy(GUEST_PTR(uri, char*), s_descs[desc].uri, maxLen - 1);
    uri[maxLen - 1] = '\0';
    return CELL_OK;
}

/* Memory allocator */

/* cellSailPlayerInitialize2(pSelf, pAllocator, pCallback, pUserParam,
 *                          pAttribute, pResource)  -- NID 0x23654375
 *
 * This was unimplemented, so it took the generic "return success and touch
 * nothing" path. That is the worst answer for an initialiser: the guest
 * believes its player struct is set up and reads whatever happened to be in
 * that memory. Tokyo Jungle then dereferenced one of those stale words and
 * faulted reading guest 0x10F000B4, well past the end of main memory.
 *
 * The struct fields are not documented here, and guessing offsets to write
 * would just trade one wrong value for another. What IS safe and sufficient is
 * to ZERO it: every pointer the guest reads out is then NULL, which callers
 * check, instead of a garbage address that faults. The handles we were given
 * are kept on our side, keyed by the struct EA, so no guest layout is assumed.
 */
#define SAIL_PLAYER_CLEAR   SAIL_PLAYER_ABI_SIZE
#define SAIL_ADAPTER_CLEAR  0x100u


static void sail_zero_guest(u32 ea, u32 n)
{
    for (u32 i = 0; i < n; i += 4) vm_write32(ea + i, 0);
}

static int sail_guest_range(u32 ea, u32 n)
{
    return VM_READY && ea && n <= UINT32_MAX - ea &&
           (!ppu_vm_size || (u64)ea + n <= ppu_vm_size);
}

s32 cellSailPlayerInitialize2(u32 pSelf, u32 pAllocator, u32 pCallback,
                              u32 pUserParam, u32 pAttribute, u32 pResource)
{
    printf("[cellSail] PlayerInitialize2(self=0x%08X alloc=0x%08X cb=0x%08X)\n",
           pSelf, pAllocator, pCallback);
    if (!sail_guest_range(pSelf, SAIL_PLAYER_ABI_SIZE) ||
        !sail_guest_range(pAllocator, 8u) || !pCallback ||
        !sail_guest_range(pAttribute, 32u) ||
        !sail_guest_range(pResource, 16u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    /* PS3_NO_SAIL=1: report that the player cannot be created. On hardware
     * this call spawns the SAIL player's own worker threads, and a title that
     * then waits for one of them to signal readiness will wait forever against
     * a stub that spawns nothing. Failing cleanly lets a title take its
     * "no video" path instead of deadlocking during setup. */
    if (getenv("PS3_NO_SAIL")) {
        printf("[cellSail] PS3_NO_SAIL -- reporting player creation failure\n");
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    }

    if (sail_player_slot(pSelf) >= 0)
        return (s32)CELL_SAIL_ERROR_ALREADY_INITIALIZED;
    int slot = -1;
    for (int i = 0; i < CELL_SAIL_PLAYER_MAX; i++) {
        if (!s_sail_players[i].in_use) {
            slot = i;
            break;
        }
    }
    if (slot < 0)
        return (s32)CELL_SAIL_ERROR_OUT_OF_MEMORY;

    sail_zero_guest(pSelf, SAIL_PLAYER_CLEAR);
    /* Copy the exact guest ABI fields RPCS3's CellSailPlayerInitialize2
     * initializes: allocator (8), callback and arg (8), attributes (32),
     * resource (16), then paused=true and booted=false. */
    for (u32 off = 0; off < 8u; off += 4u)
        vm_write32(pSelf + SAIL_PLAYER_ALLOCATOR + off, vm_read32(pAllocator + off));
    vm_write32(pSelf + SAIL_PLAYER_CALLBACK, pCallback);
    vm_write32(pSelf + SAIL_PLAYER_CALLBACK_ARG, pUserParam);
    for (u32 off = 0; off < 32u; off += 4u)
        vm_write32(pSelf + SAIL_PLAYER_ATTRIBUTE + off, vm_read32(pAttribute + off));
    for (u32 off = 0; off < 16u; off += 4u)
        vm_write32(pSelf + SAIL_PLAYER_RESOURCE + off, vm_read32(pResource + off));
    vm_write8(pSelf + SAIL_PLAYER_PAUSED, 1);
    s_sail_players[slot].in_use       = 1;
    s_sail_players[slot].self_ea      = pSelf;
    s_sail_players[slot].allocator_ea = pAllocator;
    s_sail_players[slot].callback_ea  = pCallback;
    s_sail_players[slot].arg_ea       = pUserParam;
    s_sail_players[slot].state        = CELL_SAIL_PLAYER_STATE_INITIALIZED;
    sail_notify(slot, SAIL_EV_STATE_CHANGED, 0,
                CELL_SAIL_PLAYER_STATE_INITIALIZED, 0);
    return CELL_OK;
}

/* PlayerFinalize releases the local bookkeeping and opaque guest storage
 * created by Initialize2. Descriptors owned by this player are invalidated so
 * a later player cannot accidentally observe them. */
s32 cellSailPlayerFinalize(u32 pSelf)
{
    const int slot = sail_player_slot(pSelf);
    if (slot < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    const u32 sound_adapter = vm_read32(pSelf + SAIL_PLAYER_SOUND_ADAPTER);
    const u32 graphics_adapter = vm_read32(pSelf + SAIL_PLAYER_GRAPHICS_ADAPTER);
    if (sail_guest_range(sound_adapter, 2u)) vm_write8(sound_adapter + 1u, 0);
    if (sail_guest_range(graphics_adapter, 2u)) vm_write8(graphics_adapter + 1u, 0);

    for (int i = 0; i < MAX_DESCRIPTORS; ++i) {
        if (s_descs[i].in_use && s_descs[i].player == pSelf)
            memset(&s_descs[i], 0, sizeof(s_descs[i]));
    }
    memset(&s_sail_players[slot], 0, sizeof(s_sail_players[slot]));
    sail_zero_guest(pSelf, SAIL_PLAYER_CLEAR);
    return CELL_OK;
}

/* Store the guest ABI pause flag. RPCS3 currently leaves this API as a
 * success-only TODO, so this models the documented player field without
 * claiming that media playback is paused by a local decoder. */
s32 cellSailPlayerSetPaused(u32 pSelf, u8 paused)
{
    const int slot = sail_player_slot(pSelf);
    if (slot < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!sail_guest_range(pSelf, SAIL_PLAYER_ABI_SIZE))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    vm_write8(pSelf + SAIL_PLAYER_PAUSED, paused ? 1 : 0);
    return CELL_OK;
}

/* This API returns the count directly (it has no output parameter). */
s32 cellSailPlayerGetDescriptorCount(u32 pSelf)
{
    if (sail_player_slot(pSelf) < 0)
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;

    s32 count = 0;
    for (int i = 0; i < MAX_DESCRIPTORS; ++i)
        if (s_descs[i].in_use && s_descs[i].player == pSelf)
            ++count;
    return count;
}

/* cellSailSoundAdapterInitialize(pSelf, pCallbacks, pArg) -- NID 0x3D0D3B72.
 * Store the SDK callback descriptors and initialized state in the guest ABI. */
s32 cellSailSoundAdapterInitialize(u32 pSelf, u32 pCallbacks, u32 pArg)
{
    printf("[cellSail] SoundAdapterInitialize(self=0x%08X funcs=0x%08X)\n",
           pSelf, pCallbacks);
    if (!sail_guest_range(pSelf, SAIL_ADAPTER_CLEAR) ||
        !sail_guest_range(pCallbacks, 12u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (vm_read8(pSelf) || vm_read8(pSelf + 1u))
        return (s32)CELL_SAIL_ERROR_INVALID_STATE;
    sail_zero_guest(pSelf, SAIL_ADAPTER_CLEAR);
    vm_write8(pSelf, 1);
    vm_write32(pSelf + 4u, vm_read32(pCallbacks));
    vm_write32(pSelf + 8u, vm_read32(pCallbacks + 4u));
    vm_write32(pSelf + 12u, vm_read32(pCallbacks + 8u));
    vm_write32(pSelf + 16u, pArg);
    return CELL_OK;
}

/* cellSailGraphicsAdapterInitialize(pSelf, pCallbacks, pArg)
 * The video-side twin of the sound adapter initializer. */
s32 cellSailGraphicsAdapterInitialize(u32 pSelf, u32 pCallbacks, u32 pArg)
{
    printf("[cellSail] GraphicsAdapterInitialize(self=0x%08X funcs=0x%08X)\n",
           pSelf, pCallbacks);
    if (!sail_guest_range(pSelf, SAIL_ADAPTER_CLEAR) ||
        !sail_guest_range(pCallbacks, 20u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (vm_read8(pSelf) || vm_read8(pSelf + 1u))
        return (s32)CELL_SAIL_ERROR_INVALID_STATE;
    sail_zero_guest(pSelf, SAIL_ADAPTER_CLEAR);
    vm_write8(pSelf, 1);
    vm_write8(pSelf + 1u, 1);
    for (u32 i = 0; i < 5; ++i)
        vm_write32(pSelf + 4u + i * 4u, vm_read32(pCallbacks + i * 4u));
    vm_write32(pSelf + 24u, pArg);
    return CELL_OK;
}

/* RPCS3_REFERENCE_REVIEWED: adapter state and format offsets follow the
 * CellSail adapter ABI. This local implementation only manages adapter
 * metadata; it does not copy RPCS3 code or provide a decoder/renderer. */
s32 cellSailSoundAdapterFinalize(u32 pSelf)
{
    if (!sail_guest_range(pSelf, SAIL_ADAPTER_CLEAR))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!vm_read8(pSelf) || vm_read8(pSelf + 1u))
        return (s32)CELL_SAIL_ERROR_INVALID_STATE;
    sail_zero_guest(pSelf, SAIL_ADAPTER_CLEAR);
    return CELL_OK;
}

s32 cellSailSoundAdapterGetFormat(u32 pSelf, u32 pFormat)
{
    if (!sail_guest_range(pSelf, 0x30u) ||
        !sail_guest_range(pFormat, 0x18u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    memcpy(GUEST_PTR((void*)(uintptr_t)pFormat, void*),
           GUEST_PTR((void*)(uintptr_t)(pSelf + 0x18u), const void*), 0x18u);
    return CELL_OK;
}

s32 cellSailGraphicsAdapterFinalize(u32 pSelf)
{
    if (!sail_guest_range(pSelf, SAIL_ADAPTER_CLEAR))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (!vm_read8(pSelf) || vm_read8(pSelf + 1u))
        return (s32)CELL_SAIL_ERROR_INVALID_STATE;
    sail_zero_guest(pSelf, SAIL_ADAPTER_CLEAR);
    return CELL_OK;
}

s32 cellSailGraphicsAdapterGetFormat(u32 pSelf, u32 pFormat)
{
    if (!sail_guest_range(pSelf, 0x3Cu) ||
        !sail_guest_range(pFormat, 0x20u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    memcpy(GUEST_PTR((void*)(uintptr_t)pFormat, void*),
           GUEST_PTR((void*)(uintptr_t)(pSelf + 0x1Cu), const void*), 0x20u);
    return CELL_OK;
}

/* Save the caller's preferred format in the adapter's guest ABI structure.
 * No decoder is attached, but format get/set still has a complete local path. */
s32 cellSailGraphicsAdapterSetPreferredFormat(u32 pSelf, u32 pFormat)
{
    printf("[cellSail] GraphicsAdapterSetPreferredFormat(self=0x%08X fmt=0x%08X)\n",
           pSelf, pFormat);
    if (!sail_guest_range(pSelf, 0x3Cu) || !sail_guest_range(pFormat, 0x20u))
        return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    memcpy(GUEST_PTR((void*)(uintptr_t)(pSelf + 0x1Cu), void*),
           GUEST_PTR((void*)(uintptr_t)pFormat, const void*), 0x20u);
    return CELL_OK;
}

s32 cellSailMemAllocatorInitialize(CellSailMemAllocator* allocator,
                                     CellSailMemAllocatorFuncs* pFuncs)
{
    /* Both parameters are GUEST addresses, and both structs live in GUEST
     * memory with 32-bit big-endian pointers. Dereferencing them as host
     * structs was wrong twice over: the address was never translated, and the
     * host struct has 64-bit fields, so even the offsets did not line up. The
     * values stored here are guest OPD addresses that SAIL calls back into. */
    u32 alloc_ea = (u32)(uintptr_t)allocator;
    u32 funcs_ea = (u32)(uintptr_t)pFuncs;
    printf("[cellSail] MemAllocatorInitialize(allocator=0x%08X funcs=0x%08X)\n",
           alloc_ea, funcs_ea);
    if (!alloc_ea) return (s32)CELL_SAIL_ERROR_INVALID_ARGUMENT;
    if (funcs_ea) {
        vm_write32(alloc_ea + 0, vm_read32(funcs_ea + 0));   /* pAlloc */
        vm_write32(alloc_ea + 4, vm_read32(funcs_ea + 4));   /* pFree  */
    }
    return CELL_OK;
}
