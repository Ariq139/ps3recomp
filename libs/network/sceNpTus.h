/*
 * ps3recomp - sceNpTus HLE
 *
 * NP Title User Storage: local profile-backed storage for game data
 * (leaderboards, stats, and data blobs). No PSN service is contacted.
 */

#ifndef PS3RECOMP_SCE_NP_TUS_H
#define PS3RECOMP_SCE_NP_TUS_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"
#include "sceNp.h"
#include "../system/cellRtc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Error codes
 * -----------------------------------------------------------------------*/
#define SCE_NP_TUS_ERROR_NOT_INITIALIZED       0x80022E01
#define SCE_NP_TUS_ERROR_ALREADY_INITIALIZED   0x80022E02
#define SCE_NP_TUS_ERROR_INVALID_ARGUMENT      0x80022E03
#define SCE_NP_TUS_ERROR_OUT_OF_MEMORY         0x80022E04
#define SCE_NP_TUS_ERROR_NOT_CONNECTED         0x80022E05
#define SCE_NP_TUS_ERROR_DATA_NOT_FOUND        0x80022E06
#define SCE_NP_TUS_ERROR_TIMEOUT               0x80022E07

/* ---------------------------------------------------------------------------
 * Constants
 * -----------------------------------------------------------------------*/
#define SCE_NP_TUS_MAX_SLOT_NUM           64
#define SCE_NP_TUS_DATA_INFO_MAX_SIZE     384
#define SCE_NP_TUS_MAX_CTX               8
#define SCE_NP_TUS_MAX_TITLE_CTX         8
#define SCE_NP_TUS_MAX_TRANSACTION_CTX   16
#define SCE_NP_TUS_MAX_SLOT_NUM_PER_TRANS 64

/* TryAndSet comparison operations used by the SDK. */
#define SCE_NP_TUS_OPETYPE_EQUAL              1
#define SCE_NP_TUS_OPETYPE_NOT_EQUAL          2
#define SCE_NP_TUS_OPETYPE_GREATER_THAN       3
#define SCE_NP_TUS_OPETYPE_GREATER_OR_EQUAL  4
#define SCE_NP_TUS_OPETYPE_LESS_THAN          5
#define SCE_NP_TUS_OPETYPE_LESS_OR_EQUAL      6

/* ---------------------------------------------------------------------------
 * Types
 * -----------------------------------------------------------------------*/
typedef u32 SceNpTusCtx;
typedef u32 SceNpTusRequestId;
typedef s32 SceNpTusSlotId;
typedef SceNpOnlineId SceNpTusVirtualUserId;

typedef struct SceNpTusVariable {
    SceNpId ownerId;                    /* 0x00, 36 bytes */
    s32 hasData;                        /* 0x24 */
    CellRtcTick lastChangedDate;        /* 0x28 */
    u8 pad[4];                          /* 0x30 */
    SceNpId lastChangedAuthorId;        /* 0x34 */
    s64 variable;                       /* 0x58 */
    s64 oldVariable;                    /* 0x60 */
    u8 reserved[16];                    /* 0x68; total 120 bytes */
} SceNpTusVariable;

typedef struct SceNpTusDataInfo {
    u32 infoSize;
    u8  pad[4];
    u8  data[SCE_NP_TUS_DATA_INFO_MAX_SIZE];
} SceNpTusDataInfo;

typedef struct SceNpTusDataStatus {
    SceNpId ownerId;
    s32 hasData;
    CellRtcTick lastChangedDate;
    SceNpId lastChangedAuthorId;
    u32 data;                            /* guest address */
    u32 dataSize;
    u8 pad[4];
    SceNpTusDataInfo info;
} SceNpTusDataStatus;

typedef struct SceNpTusAddAndGetVariableOptParam {
    u32 size;
    u32 isLastChangedDate;               /* guest address */
    u32 isLastChangedAuthorId;           /* guest address */
} SceNpTusAddAndGetVariableOptParam;

typedef struct SceNpTusSetDataOptParam {
    u32 size;
    u32 isLastChangedDate;               /* guest address */
    u32 isLastChangedAuthorId;           /* guest address */
} SceNpTusSetDataOptParam;

typedef struct SceNpTusTryAndSetVariableOptParam {
    u32 size;
    u32 isLastChangedDate;               /* guest address */
    u32 isLastChangedAuthorId;           /* guest address */
    u32 compareValue;                    /* guest address to s64 */
} SceNpTusTryAndSetVariableOptParam;

/* ---------------------------------------------------------------------------
 * Functions
 * -----------------------------------------------------------------------*/

/* Lifecycle */
s32 sceNpTusInit(s32 prio);
s32 sceNpTusTerm(void);

/* SDK title/transaction contexts and TUS calls imported by this EBOOT.
 * The local implementation serves the initialized profile from in-memory
 * storage; it does not contact PSN. */
s32 sceNpTusCreateTitleCtx(const SceNpCommunicationId* communicationId,
                            const SceNpCommunicationPassphrase* passphrase,
                            const SceNpId* selfNpId);
s32 sceNpTusDestroyTitleCtx(s32 titleCtxId);
s32 sceNpTusCreateTransactionCtx(s32 titleCtxId);
s32 sceNpTusDestroyTransactionCtx(s32 transId);
s32 sceNpTusGetMultiSlotVariable(s32 transId, const SceNpId* targetNpId,
                                  const SceNpTusSlotId* slotIdArray,
                                  SceNpTusVariable* variableArray,
                                  u32 variableArraySize, s32 arrayNum,
                                  void* option);
s32 sceNpTusSetMultiSlotVariable(s32 transId, const SceNpId* targetNpId,
                                  const SceNpTusSlotId* slotIdArray,
                                  const s64* variableArray, s32 arrayNum,
                                  void* option);
s32 sceNpTusTryAndSetVariableVUser(s32 transId,
                                    const SceNpTusVirtualUserId* targetVirtualUserId,
                                    SceNpTusSlotId slotId, s32 opeType,
                                    s64 variable, SceNpTusVariable* resultVariable,
                                    u32 resultVariableSize,
                                    SceNpTusTryAndSetVariableOptParam* option);
s32 sceNpTusAddAndGetVariableVUser(s32 transId,
                                    const SceNpTusVirtualUserId* targetVirtualUserId,
                                    SceNpTusSlotId slotId, s64 inVariable,
                                    SceNpTusVariable* outVariable,
                                    u32 outVariableSize,
                                    SceNpTusAddAndGetVariableOptParam* option);
s32 sceNpTusSetDataAsync(s32 transId, const SceNpId* targetNpId,
                          SceNpTusSlotId slotId, u32 totalSize, u32 sendSize,
                          const void* data, const SceNpTusDataInfo* info,
                          u32 infoStructSize, SceNpTusSetDataOptParam* option);
s32 sceNpTusGetDataAsync(s32 transId, const SceNpId* targetNpId,
                          SceNpTusSlotId slotId, SceNpTusDataStatus* dataStatus,
                          u32 dataStatusSize, void* data, u32 recvSize,
                          void* option);

/* Context management */
s32 sceNpTusCreateCtx(SceNpTusCtx* ctx);
s32 sceNpTusDestroyCtx(SceNpTusCtx ctx);

/* Variable operations (leaderboard scores, etc.) */
s32 sceNpTusSetVariable(SceNpTusCtx ctx, u32 slotId, s64 variable,
                          void* option);
s32 sceNpTusGetVariable(SceNpTusCtx ctx, const void* npId, u32 slotId,
                          SceNpTusVariable* outVariable, void* option);
s32 sceNpTusTryAndSetVariable(SceNpTusCtx ctx, u32 slotId,
                                s32 opType, s64 variable,
                                SceNpTusVariable* outVariable, void* option);
s32 sceNpTusAddAndGetVariable(SceNpTusCtx ctx, u32 slotId,
                                s64 inVariable, SceNpTusVariable* outVariable,
                                void* option);
s32 sceNpTusDeleteMultiSlotVariable(SceNpTusCtx ctx, const u32* slotIds,
                                      u32 slotNum, void* option);

/* Data operations (binary blobs) */
s32 sceNpTusSetData(SceNpTusCtx ctx, u32 slotId,
                      const void* data, u32 dataSize,
                      const SceNpTusDataInfo* info, void* option);
s32 sceNpTusGetData(SceNpTusCtx ctx, const void* npId, u32 slotId,
                      void* data, u32 dataSize, u32* outDataSize,
                      SceNpTusDataInfo* outInfo, void* option);
s32 sceNpTusDeleteMultiSlotData(SceNpTusCtx ctx, const u32* slotIds,
                                  u32 slotNum, void* option);

/* Poll for async completion */
s32 sceNpTusPollAsync(s32 transId, s32* result);
s32 sceNpTusWaitAsync(s32 transId, s32* result);
s32 sceNpTusAbortRequest(SceNpTusRequestId reqId);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_SCE_NP_TUS_H */
