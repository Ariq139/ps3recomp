/*
 * ps3recomp - sceNpTus HLE implementation
 *
 * Local storage for TUS variables and data. Games can set/get
 * leaderboard scores and binary blobs, stored in memory only
 * (no PSN connectivity required).
 */

#include "sceNpTus.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "../../runtime/ppu/ppu_memory.h"   /* vm_write*: guest EA -> host, byte-swapped */
#include "../guest_struct.h"   /* GUEST_EA, vm_read/vm_write: guest EA -> host */

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

static int s_initialized = 0;

typedef struct {
    int in_use;
    SceNpId owner;
} TusTitleContext;

typedef struct {
    int in_use;
    u32 title_slot;
} TusTransactionContext;

static TusTitleContext s_title_contexts[SCE_NP_TUS_MAX_TITLE_CTX];
static TusTransactionContext s_transaction_contexts[SCE_NP_TUS_MAX_TRANSACTION_CTX];
static int tus_guest_range(u32 ea, u32 size);
static int tus_current_tick(u64* tick);
static int tus_legacy_ctx_valid(SceNpTusCtx ctx);
static int tus_compare(s32 operation, s64 current, s64 expected);
static int tus_change_outputs(u32 option_ea, u32 expected_size,
                              u32 date_offset, u32 author_offset,
                              const SceNpId* owner, u64 timestamp,
                              int write_outputs);

typedef struct {
    int in_use;
} TusCtxSlot;

static TusCtxSlot s_ctx[SCE_NP_TUS_MAX_CTX];

/* Local variable storage */
typedef struct {
    int valid;
    s64 value;
    s64 old_value;
    u64 timestamp;
} TusVarSlot;

static TusVarSlot s_variables[SCE_NP_TUS_MAX_SLOT_NUM];

/* Local data storage */
typedef struct {
    int valid;
    u8* data;
    u32 dataSize;
    u64 timestamp;
    SceNpTusDataInfo info;
} TusDataSlot;

static TusDataSlot s_data[SCE_NP_TUS_MAX_SLOT_NUM];

/* ---------------------------------------------------------------------------
 * Lifecycle
 * -----------------------------------------------------------------------*/

s32 sceNpTusInit(s32 prio)
{
    (void)prio;
    printf("[sceNpTus] Init()\n");

    if (s_initialized)
        return (s32)SCE_NP_TUS_ERROR_ALREADY_INITIALIZED;

    memset(s_ctx, 0, sizeof(s_ctx));
    memset(s_title_contexts, 0, sizeof(s_title_contexts));
    memset(s_transaction_contexts, 0, sizeof(s_transaction_contexts));
    memset(s_variables, 0, sizeof(s_variables));
    memset(s_data, 0, sizeof(s_data));
    s_initialized = 1;
    return CELL_OK;
}

s32 sceNpTusTerm(void)
{
    printf("[sceNpTus] Term()\n");

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;

    /* Free allocated data blobs */
    for (int i = 0; i < SCE_NP_TUS_MAX_SLOT_NUM; i++) {
        if (s_data[i].valid && s_data[i].data) {
            free(s_data[i].data);
            s_data[i].data = NULL;
        }
    }

    memset(s_title_contexts, 0, sizeof(s_title_contexts));
    memset(s_transaction_contexts, 0, sizeof(s_transaction_contexts));
    s_initialized = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Context management
 * -----------------------------------------------------------------------*/

s32 sceNpTusCreateCtx(SceNpTusCtx* ctx)
{
    printf("[sceNpTus] CreateCtx()\n");

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_guest_range(GUEST_EA(ctx), 4u))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    for (int i = 0; i < SCE_NP_TUS_MAX_CTX; i++) {
        if (!s_ctx[i].in_use) {
            s_ctx[i].in_use = 1;
            vm_write32((u32)(uintptr_t)ctx, (u32)i);
            return CELL_OK;
        }
    }
    return (s32)SCE_NP_TUS_ERROR_OUT_OF_MEMORY;
}

s32 sceNpTusDestroyCtx(SceNpTusCtx ctx)
{
    printf("[sceNpTus] DestroyCtx(ctx=%u)\n", ctx);

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    s_ctx[ctx].in_use = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Variable operations
 * -----------------------------------------------------------------------*/

s32 sceNpTusSetVariable(SceNpTusCtx ctx, u32 slotId, s64 variable,
                          void* option)
{
    (void)option;

    printf("[sceNpTus] SetVariable(slot=%u, val=%lld)\n",
           slotId, (long long)variable);

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx) || slotId >= SCE_NP_TUS_MAX_SLOT_NUM)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    u64 changed_tick;
    if (!tus_current_tick(&changed_tick))
        return (s32)CELL_EFAULT;
    s_variables[slotId].old_value = s_variables[slotId].valid
        ? s_variables[slotId].value : 0;
    s_variables[slotId].valid = 1;
    s_variables[slotId].value = variable;
    s_variables[slotId].timestamp = changed_tick;
    return CELL_OK;
}

static int tus_guest_range(u32 ea, u32 size)
{
    return vm_base && ea && size <= UINT32_MAX - ea;
}

/* CellRtcTick is microseconds since the PS3 epoch. Use the host's actual UTC
 * wall clock at its available second resolution; never synthesize a timestamp
 * by incrementing a counter. */
static int tus_current_tick(u64* tick)
{
    if (!tick)
        return 0;
    const time_t now = time(NULL);
    if (now == (time_t)-1 || (int64_t)now < 0)
        return 0;
    const u64 seconds = (u64)now;
    if (seconds > UINT64_MAX / 1000000u - CELL_RTC_UNIX_EPOCH_OFFSET)
        return 0;
    *tick = (CELL_RTC_UNIX_EPOCH_OFFSET + seconds) * 1000000u;
    return 1;
}

static int tus_legacy_ctx_valid(SceNpTusCtx ctx)
{
    return ctx < SCE_NP_TUS_MAX_CTX && s_ctx[ctx].in_use;
}

static int tus_change_outputs(u32 option_ea, u32 expected_size,
                              u32 date_offset, u32 author_offset,
                              const SceNpId* owner, u64 timestamp,
                              int write_outputs)
{
    if (!option_ea)
        return 1;
    if (!tus_guest_range(option_ea, expected_size) ||
        vm_read32(option_ea) != expected_size)
        return 0;
    const u32 date_ea = vm_read32(option_ea + date_offset);
    const u32 author_ea = vm_read32(option_ea + author_offset);
    if ((date_ea && !tus_guest_range(date_ea, sizeof(CellRtcTick))) ||
        (author_ea && !tus_guest_range(author_ea, sizeof(SceNpId))))
        return 0;
    if (write_outputs) {
        if (date_ea)
            vm_write64(date_ea, timestamp);
        if (author_ea && owner)
            memcpy(vm_base + author_ea, owner, sizeof(*owner));
    }
    return 1;
}

static TusTransactionContext* tus_transaction(s32 id)
{
    if (id <= 0 || (u32)id > SCE_NP_TUS_MAX_TRANSACTION_CTX)
        return NULL;
    TusTransactionContext* trans = &s_transaction_contexts[id - 1];
    if (!trans->in_use || trans->title_slot >= SCE_NP_TUS_MAX_TITLE_CTX ||
        !s_title_contexts[trans->title_slot].in_use)
        return NULL;
    return trans;
}

static int tus_target_is_local(const TusTitleContext* title, u32 target_ea,
                               int virtual_user)
{
    if (!title || !tus_guest_range(target_ea, virtual_user ? 20u : 36u))
        return 0;
    return memcmp(vm_base + target_ea, &title->owner,
                  virtual_user ? 20u : sizeof(title->owner)) == 0;
}

/* Cell's ABI layout is 120 bytes: ownerId, hasData, tick, pad, authorId,
 * variable, oldVariable, and reserved bytes. Write fields explicitly so the
 * host's endianness/alignment never changes the guest structure. */
static void tus_variable_store(u32 ea, const SceNpId* owner,
                               const TusVarSlot* slot)
{
    if (!tus_guest_range(ea, sizeof(SceNpTusVariable)))
        return;
    memset(vm_base + ea, 0, sizeof(SceNpTusVariable));
    if (owner) {
        memcpy(vm_base + ea, owner, sizeof(*owner));
        memcpy(vm_base + ea + 52u, owner, sizeof(*owner));
    }
    vm_write32(ea + 36u, slot && slot->valid ? 1u : 0u);
    if (slot) {
        vm_write64(ea + 40u, slot->timestamp);
        vm_write64(ea + 88u, (u64)slot->value);
        vm_write64(ea + 96u, (u64)slot->old_value);
    }
}

s32 sceNpTusGetVariable(SceNpTusCtx ctx, const void* npId, u32 slotId,
                          SceNpTusVariable* outVariable, void* option)
{
    (void)npId; (void)option;

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx) || slotId >= SCE_NP_TUS_MAX_SLOT_NUM ||
        !outVariable || !tus_guest_range(GUEST_EA(outVariable), sizeof(SceNpTusVariable)))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    if (!s_variables[slotId].valid)
        return (s32)SCE_NP_TUS_ERROR_DATA_NOT_FOUND;

    tus_variable_store(GUEST_EA(outVariable), NULL, &s_variables[slotId]);
    return CELL_OK;
}

s32 sceNpTusTryAndSetVariable(SceNpTusCtx ctx, u32 slotId,
                                s32 opType, s64 variable,
                                SceNpTusVariable* outVariable, void* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx) || slotId >= SCE_NP_TUS_MAX_SLOT_NUM)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    TusVarSlot* slot = &s_variables[slotId];
    const s64 current = slot->valid ? slot->value : 0;
    const u32 option_ea = GUEST_EA(option);
    if (option_ea) {
        if (!tus_guest_range(option_ea, sizeof(SceNpTusTryAndSetVariableOptParam)) ||
            vm_read32(option_ea) != sizeof(SceNpTusTryAndSetVariableOptParam))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        const u32 compare_ea = vm_read32(option_ea + 12u);
        if (opType < SCE_NP_TUS_OPETYPE_EQUAL ||
            opType > SCE_NP_TUS_OPETYPE_LESS_OR_EQUAL ||
            !tus_guest_range(compare_ea, 8u))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        if (tus_compare(opType, current, (s64)vm_read64(compare_ea))) {
            u64 changed_tick;
            if (!tus_current_tick(&changed_tick))
                return (s32)CELL_EFAULT;
            slot->old_value = current;
            slot->value = variable;
            slot->valid = 1;
            slot->timestamp = changed_tick;
        }
    } else {
        if (opType != 0)
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        u64 changed_tick;
        if (!tus_current_tick(&changed_tick))
            return (s32)CELL_EFAULT;
        slot->old_value = current;
        slot->value = variable;
        slot->valid = 1;
        slot->timestamp = changed_tick;
    }
    if (outVariable) {
        if (!tus_guest_range(GUEST_EA(outVariable), sizeof(SceNpTusVariable)))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        tus_variable_store(GUEST_EA(outVariable), NULL, slot);
    }
    return CELL_OK;
}

s32 sceNpTusAddAndGetVariable(SceNpTusCtx ctx, u32 slotId,
                                s64 inVariable, SceNpTusVariable* outVariable,
                                void* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx) || slotId >= SCE_NP_TUS_MAX_SLOT_NUM)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    const u64 current = s_variables[slotId].valid
        ? (u64)s_variables[slotId].value : 0;
    const s64 updated = (s64)(current + (u64)inVariable);
    return sceNpTusTryAndSetVariable(ctx, slotId, 0, updated,
                                      outVariable, option);
}

s32 sceNpTusDeleteMultiSlotVariable(SceNpTusCtx ctx, const u32* slotIds,
                                      u32 slotNum, void* option)
{
    (void)option;
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    const u32 ids_ea = GUEST_EA(slotIds);
    if (!tus_legacy_ctx_valid(ctx) || slotNum == 0 ||
        slotNum > SCE_NP_TUS_MAX_SLOT_NUM ||
        !tus_guest_range(ids_ea, slotNum * 4u))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    for (u32 i = 0; i < slotNum; i++) {
        const s32 slot_id = (s32)vm_read32(ids_ea + i * 4u);
        if (slot_id < 0 || slot_id >= SCE_NP_TUS_MAX_SLOT_NUM)
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    }
    for (u32 i = 0; i < slotNum; i++)
        s_variables[(s32)vm_read32(ids_ea + i * 4u)].valid = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Data operations
 * -----------------------------------------------------------------------*/

s32 sceNpTusSetData(SceNpTusCtx ctx, u32 slotId,
                      const void* data, u32 dataSize,
                      const SceNpTusDataInfo* info, void* option)
{
    (void)option;

    printf("[sceNpTus] SetData(slot=%u, size=%u)\n", slotId, dataSize);

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx) || slotId >= SCE_NP_TUS_MAX_SLOT_NUM ||
        (dataSize && !data))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    if (dataSize && !tus_guest_range(GUEST_EA(data), dataSize))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    if (info && !tus_guest_range(GUEST_EA(info), sizeof(SceNpTusDataInfo)))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    if (info && vm_read32(GUEST_EA(info)) > SCE_NP_TUS_DATA_INFO_MAX_SIZE)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    u64 changed_tick;
    if (!tus_current_tick(&changed_tick))
        return (s32)CELL_EFAULT;
    u8* next_data = dataSize ? (u8*)malloc(dataSize) : NULL;
    if (dataSize && !next_data)
        return (s32)SCE_NP_TUS_ERROR_OUT_OF_MEMORY;

    if (dataSize)
        memcpy(next_data, GUEST_PTR(data, const void*), dataSize);
    free(s_data[slotId].data);
    s_data[slotId].data = next_data;
    s_data[slotId].dataSize = dataSize;
    s_data[slotId].valid = 1;
    s_data[slotId].timestamp = changed_tick;

    if (info) {
        s_data[slotId].info.infoSize = vm_read32(GUEST_EA(info));
        memcpy(s_data[slotId].info.data,
               vm_base + GUEST_EA(info) + offsetof(SceNpTusDataInfo, data),
               sizeof(s_data[slotId].info.data));
    } else
        memset(&s_data[slotId].info, 0, sizeof(SceNpTusDataInfo));

    return CELL_OK;
}

s32 sceNpTusGetData(SceNpTusCtx ctx, const void* npId, u32 slotId,
                      void* data, u32 dataSize, u32* outDataSize,
                      SceNpTusDataInfo* outInfo, void* option)
{
    (void)npId; (void)option;

    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_legacy_ctx_valid(ctx) || slotId >= SCE_NP_TUS_MAX_SLOT_NUM ||
        (dataSize && !data) ||
        (dataSize && !tus_guest_range(GUEST_EA(data), dataSize)) ||
        (outDataSize && !tus_guest_range(GUEST_EA(outDataSize), 4u)) ||
        (outInfo && !tus_guest_range(GUEST_EA(outInfo), sizeof(SceNpTusDataInfo))))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    if (!s_data[slotId].valid)
        return (s32)SCE_NP_TUS_ERROR_DATA_NOT_FOUND;

    u32 copySize = s_data[slotId].dataSize;
    if (copySize > dataSize)
        copySize = dataSize;

    if (data && copySize > 0) {
        if (!tus_guest_range(GUEST_EA(data), copySize))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        memcpy(GUEST_PTR(data, void*), s_data[slotId].data, copySize);
    }

    if (outDataSize) {
        if (!tus_guest_range(GUEST_EA(outDataSize), 4))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        vm_write32((u32)(uintptr_t)outDataSize, (u32)s_data[slotId].dataSize);
    }

    if (outInfo) {
        if (!tus_guest_range(GUEST_EA(outInfo), sizeof(SceNpTusDataInfo)))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        vm_write32(GUEST_EA(outInfo), s_data[slotId].info.infoSize);
        memset(vm_base + GUEST_EA(outInfo) + 4u, 0, 4u);
        memcpy(vm_base + GUEST_EA(outInfo) + offsetof(SceNpTusDataInfo, data),
               s_data[slotId].info.data, sizeof(s_data[slotId].info.data));
    }

    return CELL_OK;
}

s32 sceNpTusDeleteMultiSlotData(SceNpTusCtx ctx, const u32* slotIds,
                                  u32 slotNum, void* option)
{
    (void)option;
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    const u32 ids_ea = GUEST_EA(slotIds);
    if (!tus_legacy_ctx_valid(ctx) || slotNum == 0 ||
        slotNum > SCE_NP_TUS_MAX_SLOT_NUM ||
        !tus_guest_range(ids_ea, slotNum * 4u))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    for (u32 i = 0; i < slotNum; i++) {
        const s32 slot_id = (s32)vm_read32(ids_ea + i * 4u);
        if (slot_id < 0 || slot_id >= SCE_NP_TUS_MAX_SLOT_NUM)
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    }
    for (u32 i = 0; i < slotNum; i++) {
        const s32 slot_id = (s32)vm_read32(ids_ea + i * 4u);
        free(s_data[slot_id].data);
        s_data[slot_id].data = NULL;
        s_data[slot_id].dataSize = 0;
        s_data[slot_id].valid = 0;
    }
    return CELL_OK;
}

/* RPCS3_REFERENCE_REVIEWED: SDK guest structure sizes and argument layouts
 * were checked against the local RPCS3 headers. The handlers below use the
 * local in-memory store; no RPCS3 implementation body is copied.
 *
 * SDK title/transaction APIs used by this EBOOT.
 *
 * These handlers back one initialized local profile with the in-memory TUS
 * store above. Online peers and chunked transfers remain unavailable; those
 * cases return an explicit error instead of claiming network success.
 * -----------------------------------------------------------------------*/

s32 sceNpTusCreateTitleCtx(const SceNpCommunicationId* communicationId,
                            const SceNpCommunicationPassphrase* passphrase,
                            const SceNpId* selfNpId)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    const u32 comm_ea = GUEST_EA(communicationId);
    const u32 pass_ea = GUEST_EA(passphrase);
    const u32 owner_ea = GUEST_EA(selfNpId);
    if (!tus_guest_range(comm_ea, sizeof(SceNpCommunicationId)) ||
        !tus_guest_range(pass_ea, sizeof(SceNpCommunicationPassphrase)) ||
        !tus_guest_range(owner_ea, sizeof(SceNpId)))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    for (u32 i = 0; i < SCE_NP_TUS_MAX_TITLE_CTX; ++i) {
        if (!s_title_contexts[i].in_use) {
            memcpy(&s_title_contexts[i].owner, vm_base + owner_ea,
                   sizeof(SceNpId));
            s_title_contexts[i].in_use = 1;
            return (s32)(i + 1u);
        }
    }
    return (s32)SCE_NP_TUS_ERROR_OUT_OF_MEMORY;
}

s32 sceNpTusDestroyTitleCtx(s32 titleCtxId)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (titleCtxId <= 0 || (u32)titleCtxId > SCE_NP_TUS_MAX_TITLE_CTX ||
        !s_title_contexts[titleCtxId - 1].in_use)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    const u32 title_slot = (u32)titleCtxId - 1u;
    for (u32 i = 0; i < SCE_NP_TUS_MAX_TRANSACTION_CTX; ++i)
        if (s_transaction_contexts[i].in_use &&
            s_transaction_contexts[i].title_slot == title_slot)
            memset(&s_transaction_contexts[i], 0,
                   sizeof(s_transaction_contexts[i]));
    memset(&s_title_contexts[title_slot], 0,
           sizeof(s_title_contexts[title_slot]));
    return CELL_OK;
}

s32 sceNpTusCreateTransactionCtx(s32 titleCtxId)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (titleCtxId <= 0 || (u32)titleCtxId > SCE_NP_TUS_MAX_TITLE_CTX ||
        !s_title_contexts[titleCtxId - 1].in_use)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    for (u32 i = 0; i < SCE_NP_TUS_MAX_TRANSACTION_CTX; ++i) {
        if (!s_transaction_contexts[i].in_use) {
            s_transaction_contexts[i].title_slot = (u32)titleCtxId - 1u;
            s_transaction_contexts[i].in_use = 1;
            return (s32)(i + 1u);
        }
    }
    return (s32)SCE_NP_TUS_ERROR_OUT_OF_MEMORY;
}

s32 sceNpTusDestroyTransactionCtx(s32 transId)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    if (!trans)
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    memset(trans, 0, sizeof(*trans));
    return CELL_OK;
}

s32 sceNpTusGetMultiSlotVariable(s32 transId, const SceNpId* targetNpId,
                                  const SceNpTusSlotId* slotIdArray,
                                  SceNpTusVariable* variableArray,
                                  u32 variableArraySize, s32 arrayNum,
                                  void* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    const u32 target_ea = GUEST_EA(targetNpId);
    const u32 slots_ea = GUEST_EA(slotIdArray);
    const u32 out_ea = GUEST_EA(variableArray);
    if (!trans || !target_ea || !slots_ea || !out_ea || option ||
        arrayNum <= 0 || arrayNum > SCE_NP_TUS_MAX_SLOT_NUM_PER_TRANS ||
        variableArraySize != (u32)arrayNum * (u32)sizeof(SceNpTusVariable) ||
        !tus_guest_range(slots_ea, (u32)arrayNum * 4u) ||
        !tus_guest_range(out_ea, variableArraySize))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    TusTitleContext* title = &s_title_contexts[trans->title_slot];
    if (!tus_target_is_local(title, target_ea, 0))
        return (s32)SCE_NP_TUS_ERROR_NOT_CONNECTED;

    for (s32 i = 0; i < arrayNum; ++i) {
        const s32 slot_id = (s32)vm_read32(slots_ea + (u32)i * 4u);
        if (slot_id < 0 || slot_id >= SCE_NP_TUS_MAX_SLOT_NUM)
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    }
    for (s32 i = 0; i < arrayNum; ++i) {
        const s32 slot_id = (s32)vm_read32(slots_ea + (u32)i * 4u);
        const TusVarSlot* slot = &s_variables[slot_id];
        tus_variable_store(out_ea + (u32)i * (u32)sizeof(SceNpTusVariable),
                           &title->owner, slot);
    }
    return CELL_OK;
}

s32 sceNpTusSetMultiSlotVariable(s32 transId, const SceNpId* targetNpId,
                                  const SceNpTusSlotId* slotIdArray,
                                  const s64* variableArray, s32 arrayNum,
                                  void* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    const u32 target_ea = GUEST_EA(targetNpId);
    const u32 slots_ea = GUEST_EA(slotIdArray);
    const u32 values_ea = GUEST_EA(variableArray);
    if (!trans || !target_ea || !slots_ea || !values_ea || option ||
        arrayNum <= 0 || arrayNum > SCE_NP_TUS_MAX_SLOT_NUM_PER_TRANS ||
        !tus_guest_range(slots_ea, (u32)arrayNum * 4u) ||
        !tus_guest_range(values_ea, (u32)arrayNum * 8u))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    TusTitleContext* title = &s_title_contexts[trans->title_slot];
    if (!tus_target_is_local(title, target_ea, 0))
        return (s32)SCE_NP_TUS_ERROR_NOT_CONNECTED;
    for (s32 i = 0; i < arrayNum; ++i) {
        const s32 slot_id = (s32)vm_read32(slots_ea + (u32)i * 4u);
        if (slot_id < 0 || slot_id >= SCE_NP_TUS_MAX_SLOT_NUM)
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    }
    u64 changed_tick;
    if (!tus_current_tick(&changed_tick))
        return (s32)CELL_EFAULT;
    for (s32 i = 0; i < arrayNum; ++i) {
        const s32 slot_id = (s32)vm_read32(slots_ea + (u32)i * 4u);
        TusVarSlot* slot = &s_variables[slot_id];
        slot->old_value = slot->valid ? slot->value : 0;
        slot->value = (s64)vm_read64(values_ea + (u32)i * 8u);
        slot->valid = 1;
        slot->timestamp = changed_tick;
    }
    return CELL_OK;
}

static int tus_compare(s32 operation, s64 current, s64 expected)
{
    switch (operation) {
    case SCE_NP_TUS_OPETYPE_EQUAL:             return current == expected;
    case SCE_NP_TUS_OPETYPE_NOT_EQUAL:         return current != expected;
    case SCE_NP_TUS_OPETYPE_GREATER_THAN:      return current > expected;
    case SCE_NP_TUS_OPETYPE_GREATER_OR_EQUAL:  return current >= expected;
    case SCE_NP_TUS_OPETYPE_LESS_THAN:         return current < expected;
    case SCE_NP_TUS_OPETYPE_LESS_OR_EQUAL:     return current <= expected;
    default:                                   return 0;
    }
}

s32 sceNpTusTryAndSetVariableVUser(s32 transId,
                                    const SceNpTusVirtualUserId* targetVirtualUserId,
                                    SceNpTusSlotId slotId, s32 opeType,
                                    s64 variable, SceNpTusVariable* resultVariable,
                                    u32 resultVariableSize,
                                    SceNpTusTryAndSetVariableOptParam* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    const u32 target_ea = GUEST_EA(targetVirtualUserId);
    const u32 out_ea = GUEST_EA(resultVariable);
    const u32 option_ea = GUEST_EA(option);
    if (!trans || !target_ea || !out_ea || slotId < 0 ||
        slotId >= SCE_NP_TUS_MAX_SLOT_NUM ||
        resultVariableSize != sizeof(SceNpTusVariable) ||
        !tus_guest_range(out_ea, sizeof(SceNpTusVariable)) ||
        (option_ea && !tus_guest_range(option_ea, sizeof(SceNpTusTryAndSetVariableOptParam))))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    TusTitleContext* title = &s_title_contexts[trans->title_slot];
    if (!tus_target_is_local(title, target_ea, 1))
        return (s32)SCE_NP_TUS_ERROR_NOT_CONNECTED;

    TusVarSlot* slot = &s_variables[slotId];
    const s64 current = slot->valid ? slot->value : 0;
    if (option_ea) {
        if (!tus_change_outputs(option_ea,
                sizeof(SceNpTusTryAndSetVariableOptParam), 4u, 8u,
                &title->owner, slot->timestamp, 0))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        const u32 compare_ea = vm_read32(option_ea + 12u);
        if (opeType < SCE_NP_TUS_OPETYPE_EQUAL ||
            opeType > SCE_NP_TUS_OPETYPE_LESS_OR_EQUAL || !compare_ea ||
            !tus_guest_range(compare_ea, 8u))
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        if (tus_compare(opeType, current, (s64)vm_read64(compare_ea))) {
            u64 changed_tick;
            if (!tus_current_tick(&changed_tick))
                return (s32)CELL_EFAULT;
            slot->old_value = current;
            slot->value = variable;
            slot->valid = 1;
            slot->timestamp = changed_tick;
        }
    } else {
        if (opeType != 0)
            return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
        u64 changed_tick;
        if (!tus_current_tick(&changed_tick))
            return (s32)CELL_EFAULT;
        slot->old_value = current;
        slot->value = variable;
        slot->valid = 1;
        slot->timestamp = changed_tick;
    }
    if (option_ea && !tus_change_outputs(option_ea,
            sizeof(SceNpTusTryAndSetVariableOptParam), 4u, 8u,
            &title->owner, slot->timestamp, 1))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    tus_variable_store(out_ea, &title->owner, slot);
    return CELL_OK;
}

s32 sceNpTusAddAndGetVariableVUser(s32 transId,
                                    const SceNpTusVirtualUserId* targetVirtualUserId,
                                    SceNpTusSlotId slotId, s64 inVariable,
                                    SceNpTusVariable* outVariable,
                                    u32 outVariableSize,
                                    SceNpTusAddAndGetVariableOptParam* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    const u32 target_ea = GUEST_EA(targetVirtualUserId);
    const u32 out_ea = GUEST_EA(outVariable);
    const u32 option_ea = GUEST_EA(option);
    if (!trans || !target_ea || !out_ea || slotId < 0 ||
        slotId >= SCE_NP_TUS_MAX_SLOT_NUM ||
        outVariableSize != sizeof(SceNpTusVariable) ||
        !tus_guest_range(out_ea, sizeof(SceNpTusVariable)) ||
        (option_ea && !tus_guest_range(option_ea, sizeof(SceNpTusAddAndGetVariableOptParam))))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    TusTitleContext* title = &s_title_contexts[trans->title_slot];
    if (!tus_target_is_local(title, target_ea, 1))
        return (s32)SCE_NP_TUS_ERROR_NOT_CONNECTED;
    if (option_ea && !tus_change_outputs(option_ea,
            sizeof(SceNpTusAddAndGetVariableOptParam), 4u, 8u,
            &title->owner, s_variables[slotId].timestamp, 0))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;

    TusVarSlot* slot = &s_variables[slotId];
    u64 changed_tick;
    if (!tus_current_tick(&changed_tick))
        return (s32)CELL_EFAULT;
    const u64 current = slot->valid ? (u64)slot->value : 0;
    slot->old_value = (s64)current;
    slot->value = (s64)(current + (u64)inVariable);
    slot->valid = 1;
    slot->timestamp = changed_tick;
    if (option_ea && !tus_change_outputs(option_ea,
            sizeof(SceNpTusAddAndGetVariableOptParam), 4u, 8u,
            &title->owner, slot->timestamp, 1))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    tus_variable_store(out_ea, &title->owner, slot);
    return CELL_OK;
}

s32 sceNpTusSetDataAsync(s32 transId, const SceNpId* targetNpId,
                          SceNpTusSlotId slotId, u32 totalSize, u32 sendSize,
                          const void* data, const SceNpTusDataInfo* info,
                          u32 infoStructSize, SceNpTusSetDataOptParam* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    const u32 target_ea = GUEST_EA(targetNpId);
    const u32 info_ea = GUEST_EA(info);
    const u32 option_ea = GUEST_EA(option);
    if (!trans || !target_ea || slotId < 0 || slotId >= SCE_NP_TUS_MAX_SLOT_NUM ||
        !data || !totalSize || sendSize != totalSize ||
        (info_ea && infoStructSize != sizeof(SceNpTusDataInfo)) ||
        (info_ea && !tus_guest_range(info_ea, sizeof(SceNpTusDataInfo))) ||
        !tus_guest_range(GUEST_EA(data), totalSize) ||
        (info_ea && vm_read32(info_ea) > SCE_NP_TUS_DATA_INFO_MAX_SIZE) ||
        (option_ea && !tus_change_outputs(option_ea,
             sizeof(SceNpTusSetDataOptParam), 4u, 8u,
             NULL, s_data[slotId].timestamp, 0)))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    TusTitleContext* title = &s_title_contexts[trans->title_slot];
    if (!tus_target_is_local(title, target_ea, 0))
        return (s32)SCE_NP_TUS_ERROR_NOT_CONNECTED;

    u64 changed_tick;
    if (!tus_current_tick(&changed_tick))
        return (s32)CELL_EFAULT;

    u8* next_data = (u8*)malloc(totalSize);
    if (!next_data)
        return (s32)SCE_NP_TUS_ERROR_OUT_OF_MEMORY;
    memcpy(next_data, GUEST_PTR(data, const void*), totalSize);
    free(s_data[slotId].data);
    s_data[slotId].data = next_data;
    s_data[slotId].dataSize = totalSize;
    s_data[slotId].valid = 1;
    s_data[slotId].timestamp = changed_tick;
    if (info_ea) {
        s_data[slotId].info.infoSize = vm_read32(info_ea);
        memcpy(s_data[slotId].info.data,
               vm_base + info_ea + offsetof(SceNpTusDataInfo, data),
               sizeof(s_data[slotId].info.data));
    } else {
        memset(&s_data[slotId].info, 0, sizeof(s_data[slotId].info));
    }
    if (option_ea && !tus_change_outputs(option_ea,
            sizeof(SceNpTusSetDataOptParam), 4u, 8u,
            &title->owner, s_data[slotId].timestamp, 1))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    return CELL_OK;
}

s32 sceNpTusGetDataAsync(s32 transId, const SceNpId* targetNpId,
                          SceNpTusSlotId slotId, SceNpTusDataStatus* dataStatus,
                          u32 dataStatusSize, void* data, u32 recvSize,
                          void* option)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    TusTransactionContext* trans = tus_transaction(transId);
    const u32 target_ea = GUEST_EA(targetNpId);
    const u32 status_ea = GUEST_EA(dataStatus);
    const u32 data_ea = GUEST_EA(data);
    if (!trans || !target_ea || !status_ea || slotId < 0 ||
        slotId >= SCE_NP_TUS_MAX_SLOT_NUM || option ||
        dataStatusSize != sizeof(SceNpTusDataStatus) ||
        !tus_guest_range(status_ea, sizeof(SceNpTusDataStatus)) ||
        (recvSize && (!data_ea || !tus_guest_range(data_ea, recvSize))))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    TusTitleContext* title = &s_title_contexts[trans->title_slot];
    if (!tus_target_is_local(title, target_ea, 0))
        return (s32)SCE_NP_TUS_ERROR_NOT_CONNECTED;

    memset(vm_base + status_ea, 0, sizeof(SceNpTusDataStatus));
    if (!s_data[slotId].valid)
        return (s32)SCE_NP_TUS_ERROR_DATA_NOT_FOUND;

    u32 copied_size = s_data[slotId].dataSize;
    if (copied_size > recvSize)
        copied_size = recvSize;
    if (copied_size)
        memcpy(vm_base + data_ea, s_data[slotId].data, copied_size);
    memcpy(vm_base + status_ea, &title->owner, sizeof(SceNpId));
    vm_write32(status_ea + 36u, 1u);
    vm_write64(status_ea + 40u, s_data[slotId].timestamp);
    memcpy(vm_base + status_ea + 48u, &title->owner, sizeof(SceNpId));
    vm_write32(status_ea + 84u, data_ea);
    vm_write32(status_ea + 88u, s_data[slotId].dataSize);
    const u32 info_ea = status_ea + 96u;
    vm_write32(info_ea, s_data[slotId].info.infoSize);
    memcpy(vm_base + info_ea + 8u, s_data[slotId].info.data,
           sizeof(s_data[slotId].info.data));
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Async polling (TUS operations execute synchronously in the local store)
 * -----------------------------------------------------------------------*/

s32 sceNpTusPollAsync(s32 transId, s32* result)
{
    if (!s_initialized)
        return (s32)SCE_NP_TUS_ERROR_NOT_INITIALIZED;
    if (!tus_transaction(transId) || !result ||
        !tus_guest_range(GUEST_EA(result), 4u))
        return (s32)SCE_NP_TUS_ERROR_INVALID_ARGUMENT;
    vm_write32(GUEST_EA(result), 0u); /* local operations are already complete */
    return CELL_OK;
}

s32 sceNpTusWaitAsync(s32 transId, s32* result)
{
    return sceNpTusPollAsync(transId, result);
}

s32 sceNpTusAbortRequest(SceNpTusRequestId reqId)
{
    (void)reqId;
    return CELL_OK;
}
