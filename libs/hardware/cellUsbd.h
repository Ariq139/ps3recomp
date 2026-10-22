/*
 * ps3recomp - cellUsbd HLE
 *
 * USB device access. Stub — init/term work, device enumeration
 * returns empty, transfers return NOT_FOUND.
 */

#ifndef PS3RECOMP_CELL_USBD_H
#define PS3RECOMP_CELL_USBD_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error values from RPCS3's public CellUsbdError ABI. */
#define CELL_USBD_ERROR_NOT_INITIALIZED       0x80110001u
#define CELL_USBD_ERROR_ALREADY_INITIALIZED   0x80110002u
#define CELL_USBD_ERROR_NO_MEMORY             0x80110003u
#define CELL_USBD_ERROR_INVALID_PARAM         0x80110004u
#define CELL_USBD_ERROR_INVALID_TRANSFER_TYPE 0x80110005u
#define CELL_USBD_ERROR_LDD_ALREADY_REGISTERED 0x80110006u
#define CELL_USBD_ERROR_LDD_NOT_ALLOCATED     0x80110007u
#define CELL_USBD_ERROR_LDD_NOT_RELEASED      0x80110008u
#define CELL_USBD_ERROR_LDD_NOT_FOUND         0x80110009u
#define CELL_USBD_ERROR_DEVICE_NOT_FOUND      0x8011000Au
#define CELL_USBD_ERROR_PIPE_NOT_ALLOCATED    0x8011000Bu
#define CELL_USBD_ERROR_PIPE_NOT_RELEASED     0x8011000Cu
#define CELL_USBD_ERROR_PIPE_NOT_FOUND        0x8011000Du
#define CELL_USBD_ERROR_IOREQ_NOT_ALLOCATED   0x8011000Eu
#define CELL_USBD_ERROR_IOREQ_NOT_RELEASED    0x8011000Fu
#define CELL_USBD_ERROR_IOREQ_NOT_FOUND       0x80110010u
#define CELL_USBD_ERROR_CANNOT_GET_DESCRIPTOR 0x80110011u
#define CELL_USBD_ERROR_FATAL                 0x801100FFu

/* Legacy local spellings retained for existing callers. */
#define CELL_USBD_ERROR_INVALID_ARGUMENT CELL_USBD_ERROR_INVALID_PARAM
#define CELL_USBD_ERROR_NO_DEVICE        CELL_USBD_ERROR_DEVICE_NOT_FOUND
#define CELL_USBD_ERROR_TRANSFER         CELL_USBD_ERROR_FATAL

/* Constants */
#define CELL_USBD_MAX_PIPES    16

/* Transfer types */
#define CELL_USBD_TRANSFER_CONTROL     0
#define CELL_USBD_TRANSFER_ISOCHRONOUS 1
#define CELL_USBD_TRANSFER_BULK        2
#define CELL_USBD_TRANSFER_INTERRUPT   3

/* Types */
typedef u32 CellUsbdDeviceId;
typedef u32 CellUsbdPipeId;

typedef struct CellUsbdDeviceInfo {
    CellUsbdDeviceId deviceId;
    u16 vendorId;
    u16 productId;
    u8  deviceClass;
    u8  deviceSubClass;
    u8  deviceProtocol;
    u8  reserved;
} CellUsbdDeviceInfo;

typedef void (*CellUsbdLddOpsAttachCb)(CellUsbdDeviceId deviceId);
typedef void (*CellUsbdLddOpsDetachCb)(CellUsbdDeviceId deviceId);

typedef struct CellUsbdLddOps {
    const char* name;
    CellUsbdLddOpsAttachCb attach;
    CellUsbdLddOpsDetachCb detach;
} CellUsbdLddOps;

/* Functions */
s32 cellUsbdInit(void);
s32 cellUsbdEnd(void);

s32 cellUsbdRegisterLdd(const CellUsbdLddOps* ops);
s32 cellUsbdUnregisterLdd(const CellUsbdLddOps* ops);

s32 cellUsbdGetDeviceList(CellUsbdDeviceInfo* list, u32 maxDevices,
                            u32* numDevices);

s32 cellUsbdOpenPipe(CellUsbdDeviceId deviceId, u32 endpoint,
                       CellUsbdPipeId* pipeId);
s32 cellUsbdClosePipe(CellUsbdPipeId pipeId);

s32 cellUsbdBulkTransfer(CellUsbdPipeId pipeId, void* data,
                           u32 length, u32* transferred);
s32 cellUsbdControlTransfer(CellUsbdPipeId pipeId, u8 requestType,
                              u8 request, u16 value, u16 index,
                              void* data, u32 length, u32* transferred);

/* RPCS3 currently registers these entry points as explicit no-op TODOs that
 * return CELL_OK. Keep them visible to the NID generator, but do not describe
 * them as implemented USB behavior. */
s32 cellUsbdSetThreadPriority(void);
s32 cellUsbdSetThreadPriority2(void);
s32 cellUsbdGetThreadPriority(void);
s32 cellUsbdRegisterCompositeLdd(void);
s32 cellUsbdRegisterExtraLdd(void);
s32 cellUsbdRegisterExtraLdd2(void);
s32 cellUsbdUnregisterCompositeLdd(void);
s32 cellUsbdUnregisterExtraLdd(void);
s32 cellUsbdInterruptTransfer(void);
s32 cellUsbdIsochronousTransfer(void);
s32 cellUsbdHSIsochronousTransfer(void);
s32 cellUsbdScanStaticDescriptor(void);
s32 cellUsbdGetDeviceSpeed(void);
s32 cellUsbdGetDeviceLocation(void);
s32 cellUsbdSetPrivateData(void);
s32 cellUsbdGetPrivateData(void);
s32 cellUsbdAllocateMemory(void);
s32 cellUsbdAllocateMemoryFromContainer(void);
s32 cellUsbdAllocateSharedMemory(void);
s32 cellUsbdFreeMemory(void);
s32 cellUsbdResetDevice(void);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_USBD_H */
