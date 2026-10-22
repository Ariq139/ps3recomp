/*
 * ps3recomp - cellCrossController HLE
 *
 * RPCS3 behavior reference: cellCrossController.cpp/.h at the local RPCS3
 * checkout. This independent implementation uses ps3recomp guest-memory,
 * dialog, and deferred-callback helpers; no RPCS3 function body is copied.
 */
#include "cellCrossController.h"
#include "../system/cellMsgDialog.h"
#include "../system/cellSysutil.h"
#include "../../runtime/ppu/ppu_memory.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

static atomic_int s_cross_controller_status = ATOMIC_VAR_INIT(0);
static u32 s_cross_controller_callback_opd;
static u32 s_cross_controller_userdata;
extern u32 ppu_vm_size;

/* POSIX reports the allocated guest-VM bound in ppu_vm_size. Windows maps the
 * whole 32-bit EA space and leaves that value zero, which means unchecked in
 * the runtime's own VM accessors. */
static int cross_guest_range(u32 ea, u32 size)
{
    const u64 limit = ppu_vm_size ? (u64)ppu_vm_size : 0x100000000ull;
    return vm_base && ea != 0 && size != 0 &&
           (u64)ea + (u64)size <= limit;
}

static int cross_guest_string_valid(u32 ea, u32 max_chars)
{
    if (!ea) return 0;
    for (u32 i = 0; i <= max_chars; ++i) {
        const u64 pos = (u64)ea + i;
        if (pos >= 0x100000000ull) return 0;
        if (!cross_guest_range((u32)pos, 1)) return 0;
        if (vm_read8((u32)pos) == 0) return 1;
    }
    return 0;
}

static void cross_guest_string_copy(u32 ea, char* dst, u32 capacity)
{
    if (!dst || capacity == 0) return;
    u32 n = 0;
    while (n + 1 < capacity && vm_read8(ea + n) != 0) {
        dst[n] = (char)vm_read8(ea + n);
        ++n;
    }
    dst[n] = '\0';
}

static void cross_queue_callback(s32 status, s32 error_code)
{
    const u64 args[8] = {
        (u64)(s64)status,
        (u64)(s64)error_code,
        0,
        s_cross_controller_userdata,
        0, 0, 0, 0
    };
    s32 rc = cellSysutilQueueGuestCallbackArgs(s_cross_controller_callback_opd,
                                                args);
    if (rc != CELL_OK)
        fprintf(stderr, "[cellCrossController] callback enqueue failed: 0x%08X\n",
                (u32)rc);
}

/* RPCS3 finalizes this service only when the user cancels its prompt. A
 * headless auto-dismiss result is not interpreted as a user cancellation. */
static void cross_dialog_finished(s32 button_type, void* userdata)
{
    (void)userdata;
    if (button_type != CELL_MSGDIALOG_BUTTON_ESCAPE) return;

    int expected = CELL_CROSS_CONTROLLER_STATUS_INITIALIZED;
    if (!atomic_compare_exchange_strong(&s_cross_controller_status, &expected,
                                        CELL_CROSS_CONTROLLER_STATUS_FINALIZED))
        return;

    cross_queue_callback(CELL_CROSS_CONTROLLER_STATUS_FINALIZED,
                         CELL_CROSS_CONTROLLER_ERROR_CANCEL);
}

/* RPCS3_REFERENCE_REVIEWED: validation order, string limits, state rejection,
 * initialized callback, and cancel/finalized callback follow RPCS3's current
 * cellCrossController implementation. Its PS Vita connection loop is itself
 * a TODO, so this HLE does not invent network behavior. */
s32 cellCrossControllerInitialize(u32 pParam, u32 pPkgInfo,
                                  u32 callbackOpd, u32 userdata)
{
    if (atomic_load(&s_cross_controller_status) ==
        CELL_CROSS_CONTROLLER_STATUS_INITIALIZED)
        return CELL_CROSS_CONTROLLER_ERROR_INVALID_STATE;

    if (!cross_guest_range(pParam, 16) || !cross_guest_range(pPkgInfo, 12))
        return CELL_CROSS_CONTROLLER_ERROR_INVALID_VALUE;

    const u32 package_file = vm_read32(pParam + 0x0);
    const u32 signature_file = vm_read32(pParam + 0x4);
    const u32 icon_file = vm_read32(pParam + 0x8);
    const u32 title = vm_read32(pPkgInfo + 0x0);
    const u32 title_id = vm_read32(pPkgInfo + 0x4);
    const u32 app_ver = vm_read32(pPkgInfo + 0x8);

    if (!cross_guest_string_valid(package_file,
            CELL_CROSS_CONTROLLER_PARAM_FILE_NAME_LEN))
        return CELL_CROSS_CONTROLLER_ERROR_INVALID_PKG_FILENAME;
    if (!cross_guest_string_valid(signature_file,
            CELL_CROSS_CONTROLLER_PARAM_FILE_NAME_LEN))
        return CELL_CROSS_CONTROLLER_ERROR_INVALID_SIG_FILENAME;
    if (!cross_guest_string_valid(icon_file,
            CELL_CROSS_CONTROLLER_PARAM_FILE_NAME_LEN))
        return CELL_CROSS_CONTROLLER_ERROR_INVALID_ICON_FILENAME;
    if (!cross_guest_string_valid(app_ver, CELL_CROSS_CONTROLLER_PKG_APP_VER_LEN) ||
        !cross_guest_string_valid(title_id, CELL_CROSS_CONTROLLER_PKG_TITLE_ID_LEN) ||
        !cross_guest_string_valid(title, CELL_CROSS_CONTROLLER_PKG_TITLE_LEN) ||
        !callbackOpd)
        return CELL_CROSS_CONTROLLER_ERROR_INVALID_VALUE;

    s_cross_controller_callback_opd = callbackOpd;
    s_cross_controller_userdata = userdata;

    char title_text[CELL_CROSS_CONTROLLER_PKG_TITLE_LEN + 1];
    char message[128];
    cross_guest_string_copy(title, title_text, sizeof(title_text));
    snprintf(message, sizeof(message), "Cross-controller: %s", title_text);

    /* The RPCS3 UI opens a no-button dialog and then reports INITIALIZED via
     * sysutil. The local dialog helper preserves that deferred order and gives
     * the host a real cancel callback while retaining headless auto-dismiss. */
    const s32 dialog_rc = ps3_cell_msg_dialog_open_host(
        CELL_MSGDIALOG_TYPE_DISABLE_CANCEL_OFF, message,
        cross_dialog_finished, NULL, 1);
    atomic_store(&s_cross_controller_status,
                 CELL_CROSS_CONTROLLER_STATUS_INITIALIZED);
    cross_queue_callback(CELL_CROSS_CONTROLLER_STATUS_INITIALIZED,
        dialog_rc == CELL_OK ? CELL_OK : CELL_CROSS_CONTROLLER_ERROR_INTERNAL);

    return CELL_OK;
}
