/*
 * ps3recomp - cellCrossController guest ABI
 *
 * RPCS3 currently validates the request and displays a dialog; its Vita
 * connection path is still a TODO. This local API models that current surface.
 */
#ifndef PS3RECOMP_CELL_CROSS_CONTROLLER_H
#define PS3RECOMP_CELL_CROSS_CONTROLLER_H

#include "ps3emu/ps3types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CELL_CROSS_CONTROLLER_ERROR_CANCEL                ((s32)0x8002CD80u)
#define CELL_CROSS_CONTROLLER_ERROR_NETWORK               ((s32)0x8002CD81u)
#define CELL_CROSS_CONTROLLER_ERROR_OUT_OF_MEMORY         ((s32)0x8002CD90u)
#define CELL_CROSS_CONTROLLER_ERROR_FATAL                 ((s32)0x8002CD91u)
#define CELL_CROSS_CONTROLLER_ERROR_INVALID_PKG_FILENAME  ((s32)0x8002CD92u)
#define CELL_CROSS_CONTROLLER_ERROR_INVALID_SIG_FILENAME  ((s32)0x8002CD93u)
#define CELL_CROSS_CONTROLLER_ERROR_INVALID_ICON_FILENAME ((s32)0x8002CD94u)
#define CELL_CROSS_CONTROLLER_ERROR_INVALID_VALUE         ((s32)0x8002CD95u)
#define CELL_CROSS_CONTROLLER_ERROR_PKG_FILE_OPEN         ((s32)0x8002CD96u)
#define CELL_CROSS_CONTROLLER_ERROR_SIG_FILE_OPEN         ((s32)0x8002CD97u)
#define CELL_CROSS_CONTROLLER_ERROR_ICON_FILE_OPEN        ((s32)0x8002CD98u)
#define CELL_CROSS_CONTROLLER_ERROR_INVALID_STATE         ((s32)0x8002CD99u)
#define CELL_CROSS_CONTROLLER_ERROR_INVALID_PKG_FILE      ((s32)0x8002CD9Au)
#define CELL_CROSS_CONTROLLER_ERROR_INTERNAL              ((s32)0x8002CDA0u)

enum {
    CELL_CROSS_CONTROLLER_STATUS_INITIALIZED = 1,
    CELL_CROSS_CONTROLLER_STATUS_FINALIZED = 2
};

enum {
    CELL_CROSS_CONTROLLER_PKG_APP_VER_LEN = 6,
    CELL_CROSS_CONTROLLER_PKG_TITLE_ID_LEN = 10,
    CELL_CROSS_CONTROLLER_PKG_TITLE_LEN = 52,
    CELL_CROSS_CONTROLLER_PARAM_FILE_NAME_LEN = 255
};

/* Pointer members are guest big-endian EAs, not host pointers. */
typedef struct CellCrossControllerParam {
    u32 pPackageFileName;
    u32 pSignatureFileName;
    u32 pIconFileName;
    u32 option;
} CellCrossControllerParam;

typedef struct CellCrossControllerPackageInfo {
    u32 pTitle;
    u32 pTitleId;
    u32 pAppVer;
} CellCrossControllerPackageInfo;

typedef void (*CellCrossControllerCallback)(s32 status, s32 errorCode,
                                            u32 option, u32 userdata);

s32 cellCrossControllerInitialize(u32 pParam, u32 pPkgInfo,
                                  u32 callbackOpd, u32 userdata);

#ifdef __cplusplus
}
#endif

#endif
