/*
 * ps3recomp - cellMouse HLE implementation
 *
 * Mouse input. Tracks mouse state from host events and provides data
 * in PS3 CellMouseData format.
 *
 * The host input layer should call cellMouse_moveEvent(), cellMouse_buttonEvent(),
 * and cellMouse_wheelEvent() when mouse events occur.
 *
 * RPCS3_REFERENCE_REVIEWED (RPCS3 commit 2ba2f75bc, cellMouse.cpp): guest
 * structures and mouse status behavior were checked; event buffering and the
 * Win32/SDL host event translations are local, with no RPCS3 body copied.
 */

#include "cellMouse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../runtime/platform/win32_compat.h"

/* HLE struct args are GUEST EAs -- write through the vm accessors (also
 * declared above cellMouseGetInfo; harmless duplicates). */
extern void vm_write8 (unsigned long long a, unsigned char  v);
extern void vm_write16(unsigned long long a, unsigned short v);
extern void vm_write32(unsigned long long a, unsigned int   v);

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

#define MOUSE_PENDING_CAPACITY 64u

typedef struct {
    u8 buttons;
    s32 acc_dx;
    s32 acc_dy;
    s32 acc_wheel;
    s32 acc_tilt;
    int updated;
} MousePending;

typedef struct {
    int connected;
    u32 mode;
    MousePending pending;
    MousePending queue[MOUSE_PENDING_CAPACITY];
    u32 queue_head;
    u32 queue_count;
    u8 reported_buttons;
    int position_valid;
    s32 position_x;
    s32 position_y;
} MousePortState;

static int            s_mouse_initialized = 0;
static u32            s_mouse_max_connect = 0;
static MousePortState s_mouse_ports[CELL_MOUSE_MAX_MICE];
static SRWLOCK        s_mouse_lock = SRWLOCK_INIT;
static unsigned       s_mouse_trace_poll_count;

static int mouse_trace_enabled(void)
{
    const char* value = getenv("PS3_MOUSE_TRACE");
    return value && value[0] && value[0] != '0';
}

#define MOUSE_UNLOCK_RETURN(value) do { ReleaseSRWLockExclusive(&s_mouse_lock); return (value); } while (0)
#define MOUSE_UNLOCK_VOID()        do { ReleaseSRWLockExclusive(&s_mouse_lock); return; } while (0)

/* ---------------------------------------------------------------------------
 * Host event injection
 * -----------------------------------------------------------------------*/

void cellMouse_moveEvent(u32 port, s32 dx, s32 dy)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized || port >= CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_VOID();

    MousePortState* ms = &s_mouse_ports[port];
    if (!ms->connected) ms->connected = 1;

    ms->pending.acc_dx += dx;
    ms->pending.acc_dy += dy;
    ms->pending.updated = 1;
    ReleaseSRWLockExclusive(&s_mouse_lock);
}

static void mouse_queue_pending(MousePortState* ms)
{
    if (!ms->pending.updated) return;
    if (ms->queue_count == MOUSE_PENDING_CAPACITY) {
        MousePending old = ms->queue[ms->queue_head];
        ms->queue_head = (ms->queue_head + 1u) % MOUSE_PENDING_CAPACITY;
        --ms->queue_count;
        MousePending* next = &ms->queue[ms->queue_head];
        next->acc_dx += old.acc_dx;
        next->acc_dy += old.acc_dy;
        next->acc_wheel += old.acc_wheel;
        next->acc_tilt += old.acc_tilt;
    }
    ms->queue[(ms->queue_head + ms->queue_count) % MOUSE_PENDING_CAPACITY] = ms->pending;
    ++ms->queue_count;
    u8 buttons = ms->pending.buttons;
    memset(&ms->pending, 0, sizeof(ms->pending));
    ms->pending.buttons = buttons;
}

void cellMouse_buttonEvent(u32 port, u8 button_mask, int pressed)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized || port >= CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_VOID();

    MousePortState* ms = &s_mouse_ports[port];
    if (!ms->connected) ms->connected = 1;

    u8 next_buttons = pressed ? (ms->pending.buttons | button_mask)
                              : (ms->pending.buttons & (u8)~button_mask);
    if (next_buttons != ms->pending.buttons)
        mouse_queue_pending(ms);
    if (pressed)
        ms->pending.buttons |= button_mask;
    else
        ms->pending.buttons &= ~button_mask;
    ms->pending.updated = 1;
    if (mouse_trace_enabled()) {
        s_mouse_trace_poll_count = 0;
        fprintf(stderr, "[mouse-trace] host button port=%u mask=0x%02X pressed=%d state=0x%02X pos=%d,%d\n",
                port, button_mask, pressed, ms->pending.buttons,
                ms->position_x, ms->position_y);
    }
    ReleaseSRWLockExclusive(&s_mouse_lock);
}

void cellMouse_wheelEvent(u32 port, s32 wheel)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized || port >= CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_VOID();

    MousePortState* ms = &s_mouse_ports[port];
    if (!ms->connected) ms->connected = 1;

    ms->pending.acc_wheel += wheel;
    ms->pending.updated = 1;
    ReleaseSRWLockExclusive(&s_mouse_lock);
}

void cellMouse_tiltEvent(u32 port, s32 tilt)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized || port >= CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_VOID();

    MousePortState* ms = &s_mouse_ports[port];
    if (!ms->connected) ms->connected = 1;

    ms->pending.acc_tilt += tilt;
    ms->pending.updated = 1;
    ReleaseSRWLockExclusive(&s_mouse_lock);
}

int cellMouse_isInitialized(void)
{
    AcquireSRWLockShared(&s_mouse_lock);
    int initialized = s_mouse_initialized;
    ReleaseSRWLockShared(&s_mouse_lock);
    return initialized;
}

void cellMouse_positionEvent(u32 port, s32 x, s32 y)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized || port >= s_mouse_max_connect)
        MOUSE_UNLOCK_VOID();
    MousePortState* ms = &s_mouse_ports[port];
    ms->connected = 1;
    ms->pending.acc_dx += x - ms->position_x;
    ms->pending.acc_dy += y - ms->position_y;
    ms->position_x = x;
    ms->position_y = y;
    ms->position_valid = 1;
    ms->pending.updated = 1;
    ReleaseSRWLockExclusive(&s_mouse_lock);
}

void cellMouse_syncPosition(u32 port, s32 x, s32 y)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized || port >= s_mouse_max_connect)
        MOUSE_UNLOCK_VOID();
    MousePortState* ms = &s_mouse_ports[port];
    if (ms->position_valid) {
        s32 queued_dx = 0, queued_dy = 0;
        for (u32 i = 0; i < ms->queue_count; ++i) {
            const MousePending* sample = &ms->queue[(ms->queue_head + i) % MOUSE_PENDING_CAPACITY];
            queued_dx += sample->acc_dx;
            queued_dy += sample->acc_dy;
        }
        ms->pending.acc_dx = ms->position_x - x - queued_dx;
        ms->pending.acc_dy = ms->position_y - y - queued_dy;
        if (ms->pending.acc_dx || ms->pending.acc_dy) ms->pending.updated = 1;
    }
    ReleaseSRWLockExclusive(&s_mouse_lock);
}

/* Preserve movement beyond one signed-byte report; dropping it makes slow
 * polling and fast host movement leave the guest cursor permanently behind. */
static s8 mouse_take_axis(s32* pending)
{
    s32 value = *pending;
    if (value > 127) value = 127;
    if (value < -128) value = -128;
    *pending -= value;
    return (s8)value;
}

static CellMouseData mouse_take_data(MousePortState* ms, int buffered)
{
    MousePending* sample = ms->queue_count ? &ms->queue[ms->queue_head] : &ms->pending;
    CellMouseData d;
    d.update  = sample->updated ? 1 : 0;
    d.buttons = sample->buttons;

    d.x_axis = mouse_take_axis(&sample->acc_dx);
    d.y_axis = mouse_take_axis(&sample->acc_dy);
    /* A software cursor sums X/Y across a list but reads its last wheel
     * sample. Keep wheel input on the final movement fragment. */
    if (buffered && (sample->acc_dx || sample->acc_dy)) {
        d.wheel = d.tilt = 0;
    } else {
        d.wheel = mouse_take_axis(&sample->acc_wheel);
        d.tilt  = mouse_take_axis(&sample->acc_tilt);
    }
    sample->updated = sample->acc_dx || sample->acc_dy || sample->acc_wheel || sample->acc_tilt;
    if (ms->queue_count && !sample->updated) {
        ms->queue_head = (ms->queue_head + 1u) % MOUSE_PENDING_CAPACITY;
        --ms->queue_count;
    }
    ms->reported_buttons = d.buttons;
    return d;
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 cellMouseInit(u32 max_connect)
{
    printf("[cellMouse] Init(max_connect=%u)\n", max_connect);
    AcquireSRWLockExclusive(&s_mouse_lock);

    if (s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_ALREADY_INITIALIZED);

    if (max_connect == 0 || max_connect > CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_INVALID_PARAMETER);

    s_mouse_initialized = 1;
    s_mouse_max_connect = max_connect;
    memset(s_mouse_ports, 0, sizeof(s_mouse_ports));

    /* No mouse is connected until the host says one is (cellMouse_moveEvent
     * and friends set ports[n].connected). This used to assert port 0 was
     * always present, which reports hardware that is not there: a title is
     * then entitled to open a mouse-driven path and wait on input that can
     * never arrive. Report what actually exists -- an unplugged port is a
     * perfectly normal thing for a title to see. */

    MOUSE_UNLOCK_RETURN(CELL_OK);
}

s32 cellMouseEnd(void)
{
    printf("[cellMouse] End()\n");
    AcquireSRWLockExclusive(&s_mouse_lock);

    if (!s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_UNINITIALIZED);

    s_mouse_initialized = 0;
    MOUSE_UNLOCK_RETURN(CELL_OK);
}

/* Guest CellMouseData layout (6 bytes): +0 update u8, +1 buttons
 * u8, +2 x_axis s8, +3 y_axis s8, +4 wheel s8, +5 tilt s8. The
 * arg is a GUEST EA -- write via vm accessors (see cellMouseGetInfo below);
 * dereferencing it as a host pointer is an instant AV. */
static void mouse_write_data(unsigned long long ea, u8 update, u8 buttons,
                             s8 x, s8 y, s8 wheel, s8 tilt)
{
    vm_write8(ea + 0, update);
    vm_write8(ea + 1, buttons);
    vm_write8(ea + 2, (u8)x);
    vm_write8(ea + 3, (u8)y);
    vm_write8(ea + 4, (u8)wheel);
    vm_write8(ea + 5, (u8)tilt);
}

s32 cellMouseGetData(u32 port_no, CellMouseData* data)
{
    unsigned long long ea = (unsigned int)(uintptr_t)data;
    AcquireSRWLockExclusive(&s_mouse_lock);

    if (!s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_UNINITIALIZED);

    if (port_no >= s_mouse_max_connect || !data)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_INVALID_PARAMETER);

    MousePortState* ms = &s_mouse_ports[port_no];

    if (!ms->connected) {
        mouse_write_data(ea, 0, 0, 0, 0, 0, 0);
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_NO_DEVICE);
    }

    CellMouseData d = mouse_take_data(ms, 0);
    mouse_write_data(ea, d.update, d.buttons,
                     d.x_axis, d.y_axis, d.wheel, d.tilt);
    if (mouse_trace_enabled() && (d.buttons || s_mouse_trace_poll_count < 100))
        fprintf(stderr, "[mouse-trace] guest data port=%u ea=0x%08X update=%u buttons=0x%02X delta=%d,%d\n",
                port_no, (unsigned)ea, d.update, d.buttons, d.x_axis, d.y_axis);
    if (mouse_trace_enabled() && s_mouse_trace_poll_count < 100)
        s_mouse_trace_poll_count++;

    MOUSE_UNLOCK_RETURN(CELL_OK);
}

/* Guest CellMouseDataList layout: +0x00 list_num u32, +0x04 CellMouseData[8]. */
s32 cellMouseGetDataList(u32 port_no, CellMouseDataList* data)
{
    unsigned long long ea = (unsigned int)(uintptr_t)data;
    AcquireSRWLockExclusive(&s_mouse_lock);

    if (!s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_UNINITIALIZED);

    if (port_no >= s_mouse_max_connect || !data)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_INVALID_PARAMETER);

    MousePortState* ms = &s_mouse_ports[port_no];

    if (!ms->connected) {
        vm_write32(ea + 0x00, 0);
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_NO_DEVICE);
    }

    u32 count = 0;
    int transition_reported = 0;
    while ((ms->queue_count || ms->pending.updated) && count < CELL_MOUSE_MAX_DATA_LIST_NUM) {
        const MousePending* next = ms->queue_count ? &ms->queue[ms->queue_head] : &ms->pending;
        /* GFD consumes the last button state in a list. Keep a second edge
         * for the next poll so a complete tap cannot cancel itself. */
        if (transition_reported && next->buttons != ms->reported_buttons) break;
        u8 previous_buttons = ms->reported_buttons;
        CellMouseData d = mouse_take_data(ms, 1);
        if (d.buttons != previous_buttons) transition_reported = 1;
        mouse_write_data(ea + 0x04 + count * sizeof(CellMouseData),
                         d.update, d.buttons, d.x_axis, d.y_axis, d.wheel, d.tilt);
        if (mouse_trace_enabled() && (d.buttons || s_mouse_trace_poll_count < 100))
            fprintf(stderr, "[mouse-trace] guest list port=%u ea=0x%08X index=%u update=%u buttons=0x%02X delta=%d,%d\n",
                    port_no, (unsigned)ea, count, d.update, d.buttons,
                    d.x_axis, d.y_axis);
        if (mouse_trace_enabled() && s_mouse_trace_poll_count < 100)
            s_mouse_trace_poll_count++;
        count++;
    }
    vm_write32(ea, count);

    MOUSE_UNLOCK_RETURN(CELL_OK);
}

/* CellMouseInfo lives in GUEST memory and is BIG-ENDIAN. Layout is the SDK's
 * (cell/mouse/mouse_codes.h, CELL_MAX_MICE = 127):
 *   +0x000 u32 max_connect
 *   +0x004 u32 now_connect
 *   +0x008 u32 info
 *   +0x00C u16 vendor_id[127]    (ends +0x10A)
 *   +0x10A u16 product_id[127]   (ends +0x208)
 *   +0x208 u8  status[127]       (ends +0x287 -- 647 bytes total)
 *
 * This took a CellMouseInfo* and memset/assigned through it, which dereferences
 * the raw guest EA as a host pointer (instant AV), and the local struct declares
 * u32 arrays where hardware has u16/u16/u8, so even a translated pointer would
 * lay the fields out wrong. Neither showed up because the NID was never
 * registered: gen_hle_nids.py was handed the PRX name "sys_io", but the file is
 * cellMouse.c, so it warned once into the build noise and left every mouse NID
 * unresolved -- and the dispatcher's unresolved path answers CELL_OK, so LBP
 * polled a mouse that silently never existed.
 */
extern void vm_write8 (unsigned long long a, unsigned char  v);
extern void vm_write16(unsigned long long a, unsigned short v);
extern void vm_write32(unsigned long long a, unsigned int   v);

#define MOUSE_INFO_VENDOR    0x00Cu
#define MOUSE_INFO_PRODUCT   0x10Au
#define MOUSE_INFO_STATUS    0x208u
#define MOUSE_INFO_SIZE      0x287u
#define MOUSE_INFO_MAX_MICE  127u

s32 cellMouseGetInfo(u32 info_ea)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_UNINITIALIZED);

    if (!info_ea)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_INVALID_PARAMETER);

    for (u32 i = 0; i < MOUSE_INFO_SIZE; i++)
        vm_write8((unsigned long long)info_ea + i, 0);

    vm_write32((unsigned long long)info_ea + 0, s_mouse_max_connect);

    u32 connected = 0;
    for (u32 i = 0; i < s_mouse_max_connect && i < MOUSE_INFO_MAX_MICE
                    && i < CELL_MOUSE_MAX_MICE; i++) {
        if (!s_mouse_ports[i].connected)
            continue;
        vm_write8 ((unsigned long long)info_ea + MOUSE_INFO_STATUS  + i,     CELL_MOUSE_STATUS_CONNECTED);
        vm_write16((unsigned long long)info_ea + MOUSE_INFO_VENDOR  + i * 2, 0x054C); /* Sony */
        vm_write16((unsigned long long)info_ea + MOUSE_INFO_PRODUCT + i * 2, 0x0001);
        connected++;
    }
    vm_write32((unsigned long long)info_ea + 4, connected);

    MOUSE_UNLOCK_RETURN(CELL_OK);
}

s32 cellMouseSetTabletMode(u32 port_no, u32 mode)
{
    printf("[cellMouse] SetTabletMode(port=%u, mode=%u)\n", port_no, mode);
    AcquireSRWLockExclusive(&s_mouse_lock);

    if (!s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_UNINITIALIZED);

    if (port_no >= CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_INVALID_PARAMETER);

    s_mouse_ports[port_no].mode = mode;
    MOUSE_UNLOCK_RETURN(CELL_OK);
}

s32 cellMouseClearBuf(u32 port_no)
{
    AcquireSRWLockExclusive(&s_mouse_lock);
    if (!s_mouse_initialized)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_UNINITIALIZED);

    if (port_no >= CELL_MOUSE_MAX_MICE)
        MOUSE_UNLOCK_RETURN(CELL_MOUSE_ERROR_INVALID_PARAMETER);

    MousePortState* ms = &s_mouse_ports[port_no];
    ms->queue_count = ms->queue_head = 0;
    ms->reported_buttons = ms->pending.buttons;
    ms->pending.acc_dx = 0;
    ms->pending.acc_dy = 0;
    ms->pending.acc_wheel = 0;
    ms->pending.acc_tilt = 0;
    ms->pending.updated = 0;

    MOUSE_UNLOCK_RETURN(CELL_OK);
}

#undef MOUSE_UNLOCK_RETURN
#undef MOUSE_UNLOCK_VOID
