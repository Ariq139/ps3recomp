/*
 * ps3recomp - cellVideoUpload HLE implementation
 *
 * Stub. Init/term work, upload returns NOT_SUPPORTED.
 */

#include "cellVideoUpload.h"
#include "../system/cellSysutil.h"
#include "../../runtime/syscalls/sys_ppu_thread.h"
#include <stdio.h>
#include <string.h>
#include "../../runtime/ppu/ppu_memory.h"   /* vm_write*: guest EA -> host, byte-swapped */

/* Internal state */

static int s_initialized = 0;
static u32 s_result_url_ea;

/* API */

s32 cellVideoUploadInit(void)
{
    printf("[cellVideoUpload] Init()\n");
    if (s_initialized)
        return (s32)CELL_VIDEO_UPLOAD_ERROR_ALREADY_INITIALIZED;
    s_initialized = 1;
    return CELL_OK;
}

s32 cellVideoUploadTerm(void)
{
    printf("[cellVideoUpload] Term()\n");
    s_initialized = 0;
    return CELL_OK;
}

/* RPCS3 cellVideoUploadInitialize queues INITIALIZED and FINALIZED callback
 * notifications through its sysutil callback pump. The local implementation
 * queues the same guest callback arguments for the next
 * cellSysutilCheckCallback(). Uploading itself remains unavailable. */
s32 cellVideoUploadInitialize(const CellVideoUploadParam* param,
                              CellVideoUploadCallback callback, void* userdata)
{
    (void)param;
    const u32 callback_ea = (u32)(uintptr_t)callback;
    const u32 userdata_ea = (u32)(uintptr_t)userdata;
    if (!callback_ea)
        return (s32)CELL_VIDEO_UPLOAD_ERROR_INVALID_VALUE;

    if (!s_result_url_ea) {
        s_result_url_ea = vm_stack_allocate(&g_vm_stack_alloc, 128);
        if (!s_result_url_ea)
            return (s32)CELL_VIDEO_UPLOAD_ERROR_OUT_OF_MEMORY;
        memset(vm_base + s_result_url_ea, 0, 128);
    }

    u64 args[8] = {0};
    args[0] = CELL_VIDEO_UPLOAD_STATUS_INITIALIZED;
    args[1] = CELL_OK;
    args[2] = s_result_url_ea;
    args[3] = userdata_ea;
    s32 result = cellSysutilQueueGuestCallbackArgs(callback_ea, args);
    if (result != CELL_OK)
        return result;

    args[0] = CELL_VIDEO_UPLOAD_STATUS_FINALIZED;
    return cellSysutilQueueGuestCallbackArgs(callback_ea, args);
}

s32 cellVideoUploadStart(const CellVideoUploadParam* param,
                           CellVideoUploadCallback callback, void* arg)
{
    (void)param; (void)callback; (void)arg;
    printf("[cellVideoUpload] Start() - not supported\n");
    if (!s_initialized) return (s32)CELL_VIDEO_UPLOAD_ERROR_NOT_INITIALIZED;
    return (s32)CELL_VIDEO_UPLOAD_ERROR_NOT_SUPPORTED;
}

s32 cellVideoUploadCancel(void)
{
    if (!s_initialized) return (s32)CELL_VIDEO_UPLOAD_ERROR_NOT_INITIALIZED;
    return CELL_OK;
}

s32 cellVideoUploadGetStatus(s32* status)
{
    if (!s_initialized) return (s32)CELL_VIDEO_UPLOAD_ERROR_NOT_INITIALIZED;
    if (!status) return (s32)CELL_VIDEO_UPLOAD_ERROR_INVALID_ARGUMENT;
    vm_write32((u32)(uintptr_t)status, (u32)0); /* idle */
    return CELL_OK;
}
