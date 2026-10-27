/*
 * ps3recomp - cellPad HLE implementation
 *
 * Reads real gamepad input from the host and translates to PS3 pad format.
 *
 * Backend selection:
 *   - Windows default: XInput, loaded at run time from System32 (XInput1_4.dll,
 *     XInput1_3.dll, XInput9_1_0.dll); no XInput DLL means no pads
 *   - Everywhere else / if PS3RECOMP_PAD_USE_SDL2 is defined: SDL2 GameController
 *
 * Define PS3RECOMP_PAD_USE_SDL2 to force SDL2 backend on Windows.
 *
 * RPCS3_REFERENCE_REVIEWED (RPCS3 commit 2ba2f75bc, cellPad.cpp and
 * xinput_pad_handler.cpp): CellPad ABI, XInput DLL fallback order, rumble,
 * SCP extension polling, and sensor behavior were checked. Host event and
 * guest-API glue are expressed with ps3recomp interfaces. SCP extension field
 * layouts are transcribed from RPCS3's xinput_pad_handler.h and are marked in
 * docs/RPCS3_LICENSE_AUDIT.md for provenance/license review.
 */

#include <time.h>
#include "cellPad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../runtime/platform/win32_compat.h"   /* GetTickCount64 on POSIX */
#include "../../runtime/ppu/ppu_memory.h"   /* GUEST_PTR, vm_write*: guest EA -> host */
#include "ps3emu/log_first.h"

/* PAD_CLOCK=flips: PAD_SCRIPT and PAD_STICK times count guest flips instead of
 * wall-clock seconds. A loaded machine runs a title at a third of its usual
 * frame rate, and a wall-clock schedule then presses START on the logos; a
 * flip schedule replays the same screens however slow the run is. */
extern u32 cellGcm_flip_request_count(void);
static double pad_clock(unsigned long long t0_ms)
{
    static int flips = -1;
    if (flips < 0) { const char* e = getenv("PAD_CLOCK"); flips = e && !strcmp(e, "flips"); }
    return flips ? (double)cellGcm_flip_request_count()
                 : (double)(GetTickCount64() - t0_ms) / 1000.0;
}

/* ---------------------------------------------------------------------------
 * Backend selection
 * -----------------------------------------------------------------------*/

#if defined(PS3RECOMP_PAD_USE_SDL2)
  #define PAD_BACKEND_SDL2  1
  #define PAD_BACKEND_XINPUT 0
#elif defined(_WIN32)
  #define PAD_BACKEND_SDL2  0
  #define PAD_BACKEND_XINPUT 1
#else
  #define PAD_BACKEND_SDL2  1
  #define PAD_BACKEND_XINPUT 0
#endif

#if PAD_BACKEND_XINPUT
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  /* Types only. The exe must not import an XInput DLL: MinGW's libxinput is
   * xinput1_3.dll, which only the DirectX June 2010 redistributable installs,
   * so on a stock Windows 10/11 the exe did not start (0xC0000135). The DLL is
   * loaded at run time instead, see pad_xinput_load(). */
  #include <xinput.h>
#endif

#if PAD_BACKEND_SDL2
  #if defined(_WIN32)
    #include <SDL.h>
  #else
    #include <SDL2/SDL.h>
  #endif
#endif

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

#define PAD_MAX_HOST_PORTS  4  /* XInput supports max 4; SDL may support more */

typedef struct {
    int  connected;
    int  keyboard_fallback;
    u16  buttons;           /* CELL_PAD_CTRL_* bitmask */
    int  ps_button;         /* PS button lives in digital1's high byte */
    u8   analog_lx;         /* 0-255, center=128 */
    u8   analog_ly;
    u8   analog_rx;
    u8   analog_ry;
    u8   trigger_l2;        /* 0-255 */
    u8   trigger_r2;        /* 0-255 */
    /* Pressure-sensitive face buttons (0-255) */
    u8   press_right;
    u8   press_left;
    u8   press_up;
    u8   press_down;
    u8   press_triangle;
    u8   press_circle;
    u8   press_cross;
    u8   press_square;
    u8   press_l1;
    u8   press_r1;
    u16  sensor_x;
    u16  sensor_y;
    u16  sensor_z;
    u16  sensor_g;
} PadHostState;

static int           s_pad_initialized = 0;
static u32           s_max_connect = 0;
static u32           s_port_setting[CELL_PAD_MAX_PORT_NUM];
static PadHostState  s_host_state[PAD_MAX_HOST_PORTS];
static SRWLOCK       s_pad_lock = SRWLOCK_INIT;
/* Per-port "a report has arrived since your last read" flag; see cellPadGetData. */
static int s_data_fresh[PAD_MAX_HOST_PORTS];

#define PAD_UNLOCK_RETURN(value) do { ReleaseSRWLockExclusive(&s_pad_lock); return (value); } while (0)

static void pad_reset_host_state(PadHostState* hs)
{
    memset(hs, 0, sizeof(*hs));
    hs->analog_lx = hs->analog_ly = 128;
    hs->analog_rx = hs->analog_ry = 128;
    hs->sensor_x = 512;
    hs->sensor_y = 399;
    hs->sensor_z = 512;
    hs->sensor_g = 512;
}

#if PAD_BACKEND_SDL2
/* PS3 port i's controller, by SDL instance id (-1 = none); the port keeps the
 * GUID of the last one it had so that pad gets the same port back. */
static SDL_GameController* s_sdl_controllers[PAD_MAX_HOST_PORTS];
static SDL_JoystickID      s_sdl_ids[PAD_MAX_HOST_PORTS] = { -1, -1, -1, -1 };
static SDL_JoystickGUID    s_sdl_last_guid[PAD_MAX_HOST_PORTS];
static int                 s_sdl_had[PAD_MAX_HOST_PORTS];
static unsigned long long  s_sdl_scan_ms;   /* last device-list scan (pad_now_ms) */
static int s_sdl_inited = 0;
#endif

/* A port with no device behind it: nothing pressed and both sticks CENTRED.
 * Zero is not neutral for a stick -- 0 is full deflection up / left. Port 0 is
 * served even with no device (see cellPadGetData), and with the SDL2 backend
 * and no controller plugged in its state was the zeroed array: every title
 * read both sticks pushed to the top-left corner. Twisted Metal's car drove
 * itself in circles and its chase camera pitched up at the sky, so the
 * player's own vehicle left the screen within seconds of a match starting.
 * A pad unplugged mid-game kept its last report the same way. */
static void pad_host_release(PadHostState* hs)
{
    memset(hs, 0, sizeof(*hs));
    hs->analog_lx = hs->analog_ly = 128;
    hs->analog_rx = hs->analog_ry = 128;
}

/* One poll at a time: the guest polls from several threads, and a poll that
 * opens or closes a controller must not race another one using it.
 * cellPadSetActDirect takes it too (it talks to the controller). */
#ifdef _WIN32
  #define pad_yield() Sleep(0)
#else
  #include <sched.h>
  #define pad_yield() sched_yield()
#endif
static volatile char s_poll_lock;
static int  pad_lock_try(void) { return !__atomic_test_and_set(&s_poll_lock, __ATOMIC_ACQUIRE); }
static void pad_lock(void)     { while (!pad_lock_try()) pad_yield(); }
static void pad_unlock(void)   { __atomic_clear(&s_poll_lock, __ATOMIC_RELEASE); }

/* CELL_PAD_STATUS_ASSIGN_CHANGES: a port whose reported connection changed
 * since the title last asked (cellPadGetInfo/GetInfo2/PeriphGetInfo), as
 * libpad reports a controller plugged in or taken away. */
static int s_port_reported[PAD_MAX_HOST_PORTS];
static int s_port_changed[PAD_MAX_HOST_PORTS];
static void pad_track_assign(void)
{
    for (int i = 0; i < PAD_MAX_HOST_PORTS; i++) {
        const int on = s_host_state[i].connected || i == 0;   /* port 0: the virtual pad */
        if (on != s_port_reported[i]) { s_port_reported[i] = on; s_port_changed[i] = 1; }
    }
}
static u32 pad_status_take(u32 port)
{
    if (port >= PAD_MAX_HOST_PORTS) return 0;
    const int ch = __atomic_exchange_n(&s_port_changed[port], 0, __ATOMIC_RELAXED);
    return ch ? CELL_PAD_STATUS_ASSIGN_CHANGES : 0;
}

/* ---------------------------------------------------------------------------
 * Rumble (cellPadSetActDirect)
 *
 * A DualShock 3 has two motors: the small one (motor[0]) is on or off, the
 * large one (motor[1]) takes 0-255. A title sets them and they stay that way
 * until it sets them again; some titles only call when the value changes.
 * SDL's rumble runs for a duration instead, so the last request per port is
 * kept and renewed from the pad poll while it is non-zero. If the title stops
 * polling (a hang, a closed window), the motors stop within
 * PAD_RUMBLE_HOLD_MS. Large -> the host's low-frequency motor, small -> the
 * high-frequency one, which is how DualShock 4, DualSense and Xbox pads are
 * laid out too.
 * -------------------------------------------------------------------------*/

#define PAD_RUMBLE_HOLD_MS     1000   /* one SDL request lasts this long ... */
#define PAD_RUMBLE_RENEW_MS     400   /* ... and is renewed this often */

typedef struct {
    u16 lo, hi;                 /* what the host motors were last asked for */
    unsigned long long sent_ms; /* when (pad_now_ms), 0 = nothing to renew */
} PadRumble;
static PadRumble s_rumble[PAD_MAX_HOST_PORTS];

static unsigned long long pad_now_ms(void);

/* The two host motor levels (0-65535) for a PS3 actuator request. */
static void pad_rumble_levels(u8 small_on, u8 large, u16* lo, u16* hi)
{
    *lo = (u16)(large * 257u);
    *hi = small_on ? 0xFFFFu : 0u;
}

/* ---------------------------------------------------------------------------
 * XInput backend
 * -----------------------------------------------------------------------*/

#if PAD_BACKEND_XINPUT

/* RPCS3_PROVENANCE_REVIEW_REQUIRED: these optional ScpToolkit-compatible
 * extension layouts were transcribed from RPCS3 xinput_pad_handler.h. They
 * expose analog DualShock pressure buttons and motion sensors that ordinary
 * XInput omits; see the license audit before redistribution. */
typedef struct {
    float up, right, down, left;
    float lx, ly;
    float l1, l2, l3;
    float rx, ry;
    float r1, r2, r3;
    float triangle, circle, cross, square;
    float select, start, ps;
} PadScpExtendedState;

typedef struct {
    WORD x, z, y, gyro;
} PadScpAccel;

typedef DWORD (WINAPI *PadXInputGetStateFn)(DWORD, XINPUT_STATE*);
typedef DWORD (WINAPI *PadXInputSetStateFn)(DWORD, XINPUT_VIBRATION*);
typedef DWORD (WINAPI *PadXInputGetExtendedFn)(DWORD, PadScpExtendedState*);
typedef DWORD (WINAPI *PadXInputGetCustomDataFn)(DWORD, DWORD, void*);

static HMODULE s_xinput_library;
static PadXInputGetStateFn s_xinput_get_state;
static PadXInputSetStateFn s_xinput_set_state;
static PadXInputGetExtendedFn s_xinput_get_extended;
static PadXInputGetCustomDataFn s_xinput_get_custom_data;

/* Deadzone for analog sticks (same as XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) */
#define PAD_LEFT_STICK_DEADZONE  XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE
#define PAD_RIGHT_STICK_DEADZONE XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE
#define PAD_TRIGGER_THRESHOLD 30
#define PAD_XINPUT_GUIDE_BUTTON 0x0400

/* XInput is resolved at run time. Only XInputGetState and XInputSetState are
 * called, and XInput9_1_0.dll exports both, so any of the three DLLs serves.
 * Loaded once, from System32 only (not the exe folder, the working directory
 * or PATH), and kept loaded until exit. No DLL, or one without the two entry
 * points, means every slot reads as unplugged. */
typedef DWORD (WINAPI *pad_xi_get_state_fn)(DWORD, XINPUT_STATE*);
typedef DWORD (WINAPI *pad_xi_set_state_fn)(DWORD, XINPUT_VIBRATION*);
static pad_xi_get_state_fn p_xi_get_state;
static pad_xi_set_state_fn p_xi_set_state;
static int s_xi_tried;

static int pad_xinput_load(void)
{
    static const wchar_t* const names[] = { L"XInput1_4.dll", L"XInput1_3.dll", L"XInput9_1_0.dll" };
    if (s_xi_tried) return p_xi_get_state != NULL;
    s_xi_tried = 1;
    for (size_t k = 0; k < sizeof(names) / sizeof(names[0]); k++) {
        HMODULE m = LoadLibraryExW(names[k], NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
        pad_xi_get_state_fn g;
        pad_xi_set_state_fn st;
        if (!m) continue;
        g  = (pad_xi_get_state_fn)(void (*)(void))GetProcAddress(m, "XInputGetState");
        st = (pad_xi_set_state_fn)(void (*)(void))GetProcAddress(m, "XInputSetState");
        if (!g || !st) { FreeLibrary(m); continue; }
        p_xi_get_state = g;
        p_xi_set_state = st;
        printf("[cellPad] XInput from %ls\n", names[k]);
        return 1;
    }
    printf("[cellPad] no XInput DLL (XInput1_4, XInput1_3, XInput9_1_0): no XInput pads\n");
    return 0;
}

static u8 pad_xinput_stick_to_u8(short raw, short deadzone)
{
    float normalized;
    if (raw > deadzone)
        normalized = (float)(raw - deadzone) / (float)(32767 - deadzone);
    else if (raw < -deadzone)
        normalized = (float)(raw + deadzone) / (float)(32767 - deadzone);
    else
        normalized = 0.0f;

    /* Map -1.0..1.0 to 0..255 with center at 128 */
    int val = (int)(normalized * 127.0f) + 128;
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (u8)val;
}

static u8 pad_scp_pressure_to_u8(float value)
{
    if (!(value > 0.0f)) return 0; /* also maps NaN to zero */
    if (value >= 1.0f) return 255;
    return (u8)(value * 255.0f + 0.5f);
}

static u8 pad_scp_axis_to_u8(float value, int invert)
{
    if (!(value == value)) value = 0.0f; /* NaN -> stick center */
    if (value < -1.0f) value = -1.0f;
    if (value > 1.0f) value = 1.0f;
    if (invert) value = -value;
    int result = (int)((value + 1.0f) * 127.5f + 0.5f);
    if (result < 0) result = 0;
    if (result > 255) result = 255;
    return (u8)result;
}

static void pad_poll_scp_extended(PadHostState* hs, const PadScpExtendedState* scp)
{
    const u8 up = pad_scp_pressure_to_u8(scp->up);
    const u8 right = pad_scp_pressure_to_u8(scp->right);
    const u8 down = pad_scp_pressure_to_u8(scp->down);
    const u8 left = pad_scp_pressure_to_u8(scp->left);
    const u8 l1 = pad_scp_pressure_to_u8(scp->l1);
    const u8 l2 = pad_scp_pressure_to_u8(scp->l2);
    const u8 l3 = pad_scp_pressure_to_u8(scp->l3);
    const u8 r1 = pad_scp_pressure_to_u8(scp->r1);
    const u8 r2 = pad_scp_pressure_to_u8(scp->r2);
    const u8 r3 = pad_scp_pressure_to_u8(scp->r3);
    const u8 triangle = pad_scp_pressure_to_u8(scp->triangle);
    const u8 circle = pad_scp_pressure_to_u8(scp->circle);
    const u8 cross = pad_scp_pressure_to_u8(scp->cross);
    const u8 square = pad_scp_pressure_to_u8(scp->square);
    const u8 select = pad_scp_pressure_to_u8(scp->select);
    const u8 start = pad_scp_pressure_to_u8(scp->start);

    hs->buttons = 0;
    if (select > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_SELECT;
    if (l3 > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_L3;
    if (r3 > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_R3;
    if (start > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_START;
    if (up > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_UP;
    if (right > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_RIGHT;
    if (down > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_DOWN;
    if (left > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_LEFT;
    if (l2 > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_L2;
    if (r2 > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_R2;
    if (l1 > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_L1;
    if (r1 > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_R1;
    if (triangle > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_TRIANGLE;
    if (circle > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_CIRCLE;
    if (cross > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_CROSS;
    if (square > PAD_TRIGGER_THRESHOLD) hs->buttons |= CELL_PAD_CTRL_SQUARE;
    hs->ps_button = pad_scp_pressure_to_u8(scp->ps) > PAD_TRIGGER_THRESHOLD;

    hs->analog_lx = pad_scp_axis_to_u8(scp->lx, 0);
    hs->analog_ly = pad_scp_axis_to_u8(scp->ly, 1);
    hs->analog_rx = pad_scp_axis_to_u8(scp->rx, 0);
    hs->analog_ry = pad_scp_axis_to_u8(scp->ry, 1);
    hs->trigger_l2 = l2;
    hs->trigger_r2 = r2;
    hs->press_up = up;
    hs->press_right = right;
    hs->press_down = down;
    hs->press_left = left;
    hs->press_l1 = l1;
    hs->press_r1 = r1;
    hs->press_triangle = triangle;
    hs->press_circle = circle;
    hs->press_cross = cross;
    hs->press_square = square;
}

#endif /* PAD_BACKEND_XINPUT */

/* Keyboard fallback is used on port 0 only when no host controller is
 * connected. Bindings use SDL-style key names in PAD_KEY_<ACTION>, for example
 * PAD_KEY_CROSS=SDLK_z. Set PAD_NO_KEYBOARD=1 to disable the fallback. */
#ifdef _WIN32
static const struct {
    const char* name;
    const char* default_key;
    u16 button;
} s_pad_keyboard_bindings[] = {
    { "UP", "Up", CELL_PAD_CTRL_UP },
    { "RIGHT", "Right", CELL_PAD_CTRL_RIGHT },
    { "DOWN", "Down", CELL_PAD_CTRL_DOWN },
    { "LEFT", "Left", CELL_PAD_CTRL_LEFT },
    { "L2", "1", CELL_PAD_CTRL_L2 },
    { "R2", "2", CELL_PAD_CTRL_R2 },
    { "L1", "q", CELL_PAD_CTRL_L1 },
    { "R1", "w", CELL_PAD_CTRL_R1 },
    { "TRIANGLE", "s", CELL_PAD_CTRL_TRIANGLE },
    { "CIRCLE", "x", CELL_PAD_CTRL_CIRCLE },
    { "CROSS", "z", CELL_PAD_CTRL_CROSS },
    { "SQUARE", "a", CELL_PAD_CTRL_SQUARE },
    { "SELECT", "Tab", CELL_PAD_CTRL_SELECT },
    { "L3", "none", CELL_PAD_CTRL_L3 },
    { "R3", "none", CELL_PAD_CTRL_R3 },
    { "START", "Return", CELL_PAD_CTRL_START },
};

static int s_pad_keyboard_keys[sizeof s_pad_keyboard_bindings / sizeof s_pad_keyboard_bindings[0]];
static int s_pad_keyboard_keys_initialized;

static int pad_ascii_equal_ci(const char* a, const char* b)
{
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca + ('a' - 'A'));
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb + ('a' - 'A'));
        if (ca != cb) return 0;
    }
    return *a == *b;
}

static int pad_ascii_starts_with_ci(const char* value, const char* prefix)
{
    if (!value || !prefix) return 0;
    while (*prefix) {
        unsigned char a = (unsigned char)*value++;
        unsigned char b = (unsigned char)*prefix++;
        if (a >= 'A' && a <= 'Z') a = (unsigned char)(a + ('a' - 'A'));
        if (b >= 'A' && b <= 'Z') b = (unsigned char)(b + ('a' - 'A'));
        if (a != b) return 0;
    }
    return 1;
}

static int pad_keyboard_virtual_key(const char* name)
{
    if (!name || !*name || pad_ascii_equal_ci(name, "none")) return 0;
    if (pad_ascii_starts_with_ci(name, "SDLK_")) name += 5;

    const size_t length = strlen(name);
    if (length == 1) {
        unsigned char ch = (unsigned char)name[0];
        if (ch >= 'a' && ch <= 'z') return ch - ('a' - 'A');
        if ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) return ch;
    }

    if (length == 4 && pad_ascii_equal_ci(name, "KP_0")) return VK_NUMPAD0;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_1")) return VK_NUMPAD1;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_2")) return VK_NUMPAD2;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_3")) return VK_NUMPAD3;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_4")) return VK_NUMPAD4;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_5")) return VK_NUMPAD5;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_6")) return VK_NUMPAD6;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_7")) return VK_NUMPAD7;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_8")) return VK_NUMPAD8;
    if (length == 4 && pad_ascii_equal_ci(name, "KP_9")) return VK_NUMPAD9;
    if (length >= 2 && (name[0] == 'F' || name[0] == 'f')) {
        char* end = NULL;
        long function_number = strtol(name + 1, &end, 10);
        if (end != name + 1 && *end == '\0' && function_number >= 1 && function_number <= 24)
            return VK_F1 + (int)function_number - 1;
    }

    static const struct { const char* name; int key; } named_keys[] = {
        { "up", VK_UP }, { "uparrow", VK_UP },
        { "right", VK_RIGHT }, { "rightarrow", VK_RIGHT },
        { "down", VK_DOWN }, { "downarrow", VK_DOWN },
        { "left", VK_LEFT }, { "leftarrow", VK_LEFT },
        { "return", VK_RETURN }, { "enter", VK_RETURN },
        { "escape", VK_ESCAPE }, { "esc", VK_ESCAPE },
        { "tab", VK_TAB }, { "space", VK_SPACE },
        { "backspace", VK_BACK }, { "delete", VK_DELETE },
        { "insert", VK_INSERT }, { "home", VK_HOME }, { "end", VK_END },
        { "pageup", VK_PRIOR }, { "pagedown", VK_NEXT },
        { "capslock", VK_CAPITAL }, { "numlockclear", VK_NUMLOCK },
        { "scrolllock", VK_SCROLL }, { "printscreen", VK_SNAPSHOT }, { "pause", VK_PAUSE },
        { "leftshift", VK_LSHIFT }, { "rightshift", VK_RSHIFT }, { "shift", VK_SHIFT },
        { "leftctrl", VK_LCONTROL }, { "rightctrl", VK_RCONTROL }, { "ctrl", VK_CONTROL },
        { "leftalt", VK_LMENU }, { "rightalt", VK_RMENU }, { "alt", VK_MENU },
        { "minus", VK_OEM_MINUS }, { "equals", VK_OEM_PLUS },
        { "leftbracket", VK_OEM_4 }, { "rightbracket", VK_OEM_6 },
        { "backslash", VK_OEM_5 }, { "semicolon", VK_OEM_1 },
        { "apostrophe", VK_OEM_7 }, { "quote", VK_OEM_7 },
        { "comma", VK_OEM_COMMA }, { "period", VK_OEM_PERIOD },
        { "slash", VK_OEM_2 }, { "grave", VK_OEM_3 },
        { "-", VK_OEM_MINUS }, { "=", VK_OEM_PLUS },
        { "[", VK_OEM_4 }, { "]", VK_OEM_6 }, { "\\", VK_OEM_5 },
        { ";", VK_OEM_1 }, { "'", VK_OEM_7 }, { ",", VK_OEM_COMMA },
        { ".", VK_OEM_PERIOD }, { "/", VK_OEM_2 }, { "`", VK_OEM_3 },
    };
    for (unsigned i = 0; i < sizeof named_keys / sizeof named_keys[0]; ++i)
        if (pad_ascii_equal_ci(name, named_keys[i].name)) return named_keys[i].key;
    return 0;
}

static int pad_host_window_focused(void)
{
    HWND fg = GetForegroundWindow();
    if (!fg) return 0;
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    return pid == GetCurrentProcessId();
}

static void pad_poll_keyboard(void)
{
    static int off = -1;
    if (off < 0) {
        const char* disabled = getenv("PAD_NO_KEYBOARD");
        off = disabled && *disabled && strcmp(disabled, "0") != 0 &&
              !pad_ascii_equal_ci(disabled, "false") && !pad_ascii_equal_ci(disabled, "off")
            ? 1 : 0;
    }
    if (off) return;

    if (!s_pad_keyboard_keys_initialized) {
        for (unsigned i = 0; i < sizeof s_pad_keyboard_bindings / sizeof s_pad_keyboard_bindings[0]; ++i) {
            char variable[40];
            snprintf(variable, sizeof variable, "PAD_KEY_%s", s_pad_keyboard_bindings[i].name);
            const char* key_name = getenv(variable);
            if (!key_name || !*key_name) key_name = s_pad_keyboard_bindings[i].default_key;
            s_pad_keyboard_keys[i] = pad_keyboard_virtual_key(key_name);
        }
        s_pad_keyboard_keys_initialized = 1;
    }

    PadHostState* hs = &s_host_state[0];
    if (!pad_host_window_focused()) {
        /* Release everything on focus loss, or a key held while alt-tabbing
         * would stay down forever. */
        pad_reset_host_state(hs);
        hs->connected = 1;
        return;
    }

    u16 btns = 0;
    for (unsigned i = 0; i < sizeof s_pad_keyboard_bindings / sizeof s_pad_keyboard_bindings[0]; ++i)
        if (s_pad_keyboard_keys[i] && (GetAsyncKeyState(s_pad_keyboard_keys[i]) & 0x8000))
            btns |= s_pad_keyboard_bindings[i].button;

    hs->buttons   = btns;
    hs->ps_button = 0;
    hs->connected = 1;
    hs->keyboard_fallback = 1;
    hs->analog_lx = (u8)((btns & CELL_PAD_CTRL_LEFT) ? 0 :
                         (btns & CELL_PAD_CTRL_RIGHT) ? 255 : 128);
    hs->analog_ly = (u8)((btns & CELL_PAD_CTRL_UP) ? 0 :
                         (btns & CELL_PAD_CTRL_DOWN) ? 255 : 128);
    hs->analog_rx = hs->analog_ry = 128;
    hs->trigger_l2 = (u8)((btns & CELL_PAD_CTRL_L2) ? 255 : 0);
    hs->trigger_r2 = (u8)((btns & CELL_PAD_CTRL_R2) ? 255 : 0);
    hs->press_up = (u8)((btns & CELL_PAD_CTRL_UP) ? 255 : 0);
    hs->press_right = (u8)((btns & CELL_PAD_CTRL_RIGHT) ? 255 : 0);
    hs->press_down = (u8)((btns & CELL_PAD_CTRL_DOWN) ? 255 : 0);
    hs->press_left = (u8)((btns & CELL_PAD_CTRL_LEFT) ? 255 : 0);
    hs->press_l1 = (u8)((btns & CELL_PAD_CTRL_L1) ? 255 : 0);
    hs->press_r1 = (u8)((btns & CELL_PAD_CTRL_R1) ? 255 : 0);
    hs->press_triangle = (u8)((btns & CELL_PAD_CTRL_TRIANGLE) ? 255 : 0);
    hs->press_circle = (u8)((btns & CELL_PAD_CTRL_CIRCLE) ? 255 : 0);
    hs->press_cross = (u8)((btns & CELL_PAD_CTRL_CROSS) ? 255 : 0);
    hs->press_square = (u8)((btns & CELL_PAD_CTRL_SQUARE) ? 255 : 0);

    { static int said = 0;
      if (!said && btns) { said = 1;
          printf("[cellPad] keyboard fallback active on port 0 (no XInput device)\n");
          fflush(stdout); } }
}
#endif

/* Controller button remapping is applied after a backend has normalized its
 * layout to PS3 actions. A means Cross on Xbox and DualSense, B means Circle,
 * X means Square, and Y means Triangle. SDL GameController uses that same
 * standard layout for supported devices. PAD_MAP_CROSS=B, for example, makes
 * the physical B/Circle button send PS3 Cross. */
static const struct {
    const char* name;
    u16 button;
} s_pad_controller_actions[] = {
    { "UP", CELL_PAD_CTRL_UP },
    { "RIGHT", CELL_PAD_CTRL_RIGHT },
    { "DOWN", CELL_PAD_CTRL_DOWN },
    { "LEFT", CELL_PAD_CTRL_LEFT },
    { "L2", CELL_PAD_CTRL_L2 },
    { "R2", CELL_PAD_CTRL_R2 },
    { "L1", CELL_PAD_CTRL_L1 },
    { "R1", CELL_PAD_CTRL_R1 },
    { "TRIANGLE", CELL_PAD_CTRL_TRIANGLE },
    { "CIRCLE", CELL_PAD_CTRL_CIRCLE },
    { "CROSS", CELL_PAD_CTRL_CROSS },
    { "SQUARE", CELL_PAD_CTRL_SQUARE },
    { "SELECT", CELL_PAD_CTRL_SELECT },
    { "L3", CELL_PAD_CTRL_L3 },
    { "R3", CELL_PAD_CTRL_R3 },
    { "START", CELL_PAD_CTRL_START },
};

static int pad_controller_name_equal_ci(const char* a, const char* b)
{
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca + ('a' - 'A'));
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb + ('a' - 'A'));
        if (ca != cb) return 0;
    }
    return *a == *b;
}

static int pad_controller_action_from_name(const char* name)
{
    if (!name || !*name || pad_controller_name_equal_ci(name, "none") ||
        pad_controller_name_equal_ci(name, "off") || pad_controller_name_equal_ci(name, "disabled"))
        return -1;

    for (unsigned i = 0; i < sizeof s_pad_controller_actions / sizeof s_pad_controller_actions[0]; ++i)
        if (pad_controller_name_equal_ci(name, s_pad_controller_actions[i].name)) return (int)i;

    static const struct { const char* alias; int action; } aliases[] = {
        { "DPAD_UP", 0 }, { "DPAD_RIGHT", 1 }, { "DPAD_DOWN", 2 }, { "DPAD_LEFT", 3 },
        { "LT", 4 }, { "LEFT_TRIGGER", 4 }, { "LEFTTRIGGER", 4 },
        { "RT", 5 }, { "RIGHT_TRIGGER", 5 }, { "RIGHTTRIGGER", 5 },
        { "LB", 6 }, { "LEFT_BUMPER", 6 }, { "LEFTBUMPER", 6 },
        { "RB", 7 }, { "RIGHT_BUMPER", 7 }, { "RIGHTBUMPER", 7 },
        { "Y", 8 }, { "B", 9 }, { "A", 10 }, { "X", 11 },
        { "BACK", 12 }, { "VIEW", 12 }, { "SHARE", 12 },
        { "LEFT_STICK", 13 }, { "LEFTSTICK", 13 },
        { "RIGHT_STICK", 14 }, { "RIGHTSTICK", 14 },
        { "MENU", 15 },
    };
    for (unsigned i = 0; i < sizeof aliases / sizeof aliases[0]; ++i)
        if (pad_controller_name_equal_ci(name, aliases[i].alias)) return aliases[i].action;
    return -1;
}

static u8 pad_controller_action_pressure(const PadHostState* hs, int action)
{
    u8 pressure = 0;
    switch (action) {
        case 0: pressure = hs->press_up; break;
        case 1: pressure = hs->press_right; break;
        case 2: pressure = hs->press_down; break;
        case 3: pressure = hs->press_left; break;
        case 4: pressure = hs->trigger_l2; break;
        case 5: pressure = hs->trigger_r2; break;
        case 6: pressure = hs->press_l1; break;
        case 7: pressure = hs->press_r1; break;
        case 8: pressure = hs->press_triangle; break;
        case 9: pressure = hs->press_circle; break;
        case 10: pressure = hs->press_cross; break;
        case 11: pressure = hs->press_square; break;
        default: break;
    }
    if (!pressure && action >= 0 && action < (int)(sizeof s_pad_controller_actions / sizeof s_pad_controller_actions[0]) &&
        (hs->buttons & s_pad_controller_actions[action].button))
        pressure = 255;
    return pressure;
}

static void pad_set_controller_action_pressure(PadHostState* hs, int action, u8 pressure)
{
    switch (action) {
        case 0: hs->press_up = pressure; break;
        case 1: hs->press_right = pressure; break;
        case 2: hs->press_down = pressure; break;
        case 3: hs->press_left = pressure; break;
        case 4: hs->trigger_l2 = pressure; break;
        case 5: hs->trigger_r2 = pressure; break;
        case 6: hs->press_l1 = pressure; break;
        case 7: hs->press_r1 = pressure; break;
        case 8: hs->press_triangle = pressure; break;
        case 9: hs->press_circle = pressure; break;
        case 10: hs->press_cross = pressure; break;
        case 11: hs->press_square = pressure; break;
        default: break;
    }
}

static void pad_apply_controller_mapping(PadHostState* hs)
{
    enum { ACTION_COUNT = sizeof s_pad_controller_actions / sizeof s_pad_controller_actions[0] };
    static int initialized;
    static u16 source_button[ACTION_COUNT];
    if (!hs->connected || hs->keyboard_fallback) return;

    if (!initialized) {
        for (int target = 0; target < ACTION_COUNT; ++target) {
            source_button[target] = s_pad_controller_actions[target].button;
            char variable[40];
            snprintf(variable, sizeof variable, "PAD_MAP_%s", s_pad_controller_actions[target].name);
            const char* value = getenv(variable);
            if (value && *value) {
                int source = pad_controller_action_from_name(value);
                source_button[target] = source >= 0 ? s_pad_controller_actions[source].button : 0;
            }
        }
        initialized = 1;
    }

    const u16 original_buttons = hs->buttons;
    const PadHostState original_state = *hs;
    u16 mapped_buttons = 0;
    u8 mapped_pressure[ACTION_COUNT];
    memset(mapped_pressure, 0, sizeof mapped_pressure);
    for (int target = 0; target < ACTION_COUNT; ++target) {
        const u16 source = source_button[target];
        if (!source || !(original_buttons & source)) continue;
        mapped_buttons |= s_pad_controller_actions[target].button;
        for (int action = 0; action < ACTION_COUNT; ++action) {
            if (source == s_pad_controller_actions[action].button) {
                mapped_pressure[target] = pad_controller_action_pressure(&original_state, action);
                break;
            }
        }
    }
    hs->buttons = mapped_buttons;
    for (int target = 0; target < ACTION_COUNT; ++target)
        pad_set_controller_action_pressure(hs, target, mapped_pressure[target]);
}

#if PAD_BACKEND_XINPUT

/* XInputGetState on a slot with nothing plugged in costs about 140 us (2 us on
 * a connected one), and the backend is polled at least once a frame: three
 * empty slots beside one pad were 0.6 ms of every frame on God of War HD's
 * main thread. An empty slot is looked at again once a second, which is how
 * soon a pad plugged in there is seen; the slots are staggered so one poll
 * never pays for all of them. */
#define PAD_EMPTY_RECHECK_MS 1000
static unsigned long long s_pad_empty_until[PAD_MAX_HOST_PORTS];

static void pad_poll_xinput(void)
{
    const unsigned long long now = pad_now_ms();
    for (int i = 0; i < PAD_MAX_HOST_PORTS; i++) {
        PadHostState* hs = &s_host_state[i];
        pad_reset_host_state(hs);

        PadScpExtendedState scp;
        memset(&scp, 0, sizeof(scp));
        if (s_xinput_get_extended &&
            s_xinput_get_extended((DWORD)i, &scp) == ERROR_SUCCESS) {
            hs->connected = 1;
            pad_poll_scp_extended(hs, &scp);

            if (s_xinput_get_custom_data) {
                PadScpAccel sensors;
                memset(&sensors, 0, sizeof(sensors));
                if (s_xinput_get_custom_data((DWORD)i, 0, &sensors) == ERROR_SUCCESS) {
                    hs->sensor_x = sensors.x;
                    hs->sensor_y = sensors.y;
                    hs->sensor_z = sensors.z;
                    hs->sensor_g = sensors.gyro;
                }
            }
            continue;
        }

        if (s_pad_empty_until[i] && now < s_pad_empty_until[i]) {
            s_host_state[i].connected = 0;
            continue;
        }
        XINPUT_STATE state;
        memset(&state, 0, sizeof(state));

        DWORD result = p_xi_get_state ? p_xi_get_state((DWORD)i, &state)
                                      : ERROR_DEVICE_NOT_CONNECTED;
        if (result != ERROR_SUCCESS) {
            pad_host_release(&s_host_state[i]);
            s_pad_empty_until[i] = now + PAD_EMPTY_RECHECK_MS
                                 + (unsigned long long)i * (PAD_EMPTY_RECHECK_MS / PAD_MAX_HOST_PORTS);
            continue;
        }
        s_pad_empty_until[i] = 0;

        hs->connected = 1;
        XINPUT_GAMEPAD* gp = &state.Gamepad;

        /* Map XInput buttons to PS3 CELL_PAD_CTRL_* */
        u16 btns = 0;
        if (gp->wButtons & XINPUT_GAMEPAD_BACK)           btns |= CELL_PAD_CTRL_SELECT;
        if (gp->wButtons & XINPUT_GAMEPAD_LEFT_THUMB)     btns |= CELL_PAD_CTRL_L3;
        if (gp->wButtons & XINPUT_GAMEPAD_RIGHT_THUMB)    btns |= CELL_PAD_CTRL_R3;
        if (gp->wButtons & XINPUT_GAMEPAD_START)          btns |= CELL_PAD_CTRL_START;
        if (gp->wButtons & XINPUT_GAMEPAD_DPAD_UP)        btns |= CELL_PAD_CTRL_UP;
        if (gp->wButtons & XINPUT_GAMEPAD_DPAD_RIGHT)     btns |= CELL_PAD_CTRL_RIGHT;
        if (gp->wButtons & XINPUT_GAMEPAD_DPAD_DOWN)      btns |= CELL_PAD_CTRL_DOWN;
        if (gp->wButtons & XINPUT_GAMEPAD_DPAD_LEFT)      btns |= CELL_PAD_CTRL_LEFT;
        if (gp->bLeftTrigger > PAD_TRIGGER_THRESHOLD)     btns |= CELL_PAD_CTRL_L2;
        if (gp->bRightTrigger > PAD_TRIGGER_THRESHOLD)    btns |= CELL_PAD_CTRL_R2;
        if (gp->wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER)  btns |= CELL_PAD_CTRL_L1;
        if (gp->wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER) btns |= CELL_PAD_CTRL_R1;
        if (gp->wButtons & XINPUT_GAMEPAD_Y)              btns |= CELL_PAD_CTRL_TRIANGLE;
        if (gp->wButtons & XINPUT_GAMEPAD_B)              btns |= CELL_PAD_CTRL_CIRCLE;
        if (gp->wButtons & XINPUT_GAMEPAD_A)              btns |= CELL_PAD_CTRL_CROSS;
        if (gp->wButtons & XINPUT_GAMEPAD_X)              btns |= CELL_PAD_CTRL_SQUARE;

        hs->buttons = btns;
        hs->ps_button = (gp->wButtons & PAD_XINPUT_GUIDE_BUTTON) != 0;

        /* Analog sticks. PS3 Y axis is inverted vs XInput (up = 0). Reflect about
         * 128 (256 - x), NOT 255 - x: the latter turns a centered stick into 127,
         * whose bits (0x7F) alias SELECT+START and can make the guest self-exit.
         * (via sagemono, PR #42) */
        hs->analog_lx = pad_xinput_stick_to_u8(gp->sThumbLX, PAD_LEFT_STICK_DEADZONE);
        hs->analog_ly = (u8)(256 - pad_xinput_stick_to_u8(gp->sThumbLY, PAD_LEFT_STICK_DEADZONE));
        hs->analog_rx = pad_xinput_stick_to_u8(gp->sThumbRX, PAD_RIGHT_STICK_DEADZONE);
        hs->analog_ry = (u8)(256 - pad_xinput_stick_to_u8(gp->sThumbRY, PAD_RIGHT_STICK_DEADZONE));

        /* Triggers */
        hs->trigger_l2 = gp->bLeftTrigger;
        hs->trigger_r2 = gp->bRightTrigger;

        /* Pressure-sensitive buttons: XInput has digital only, so 0 or 255 */
        hs->press_up       = (btns & CELL_PAD_CTRL_UP)       ? 255 : 0;
        hs->press_down     = (btns & CELL_PAD_CTRL_DOWN)     ? 255 : 0;
        hs->press_left     = (btns & CELL_PAD_CTRL_LEFT)     ? 255 : 0;
        hs->press_right    = (btns & CELL_PAD_CTRL_RIGHT)    ? 255 : 0;
        hs->press_triangle = (btns & CELL_PAD_CTRL_TRIANGLE) ? 255 : 0;
        hs->press_circle   = (btns & CELL_PAD_CTRL_CIRCLE)   ? 255 : 0;
        hs->press_cross    = (btns & CELL_PAD_CTRL_CROSS)    ? 255 : 0;
        hs->press_square   = (btns & CELL_PAD_CTRL_SQUARE)   ? 255 : 0;
        hs->press_l1       = (btns & CELL_PAD_CTRL_L1)       ? 255 : 0;
        hs->press_r1       = (btns & CELL_PAD_CTRL_R1)       ? 255 : 0;
    }
}

static void pad_init_backend(void)
{
    /* XInput needs no explicit init, only the run-time load of the DLL */
    pad_xinput_load();
}

static void pad_shutdown_backend(void)
{
    if (s_xinput_library) {
        if (s_xinput_set_state) {
            XINPUT_VIBRATION stop = {0, 0};
            for (DWORD i = 0; i < PAD_MAX_HOST_PORTS; ++i)
                if (s_host_state[i].connected)
                    s_xinput_set_state(i, &stop);
        }
        FreeLibrary(s_xinput_library);
    }
    s_xinput_library = NULL;
    s_xinput_get_state = NULL;
    s_xinput_set_state = NULL;
    s_xinput_get_extended = NULL;
    s_xinput_get_custom_data = NULL;
}

#endif /* PAD_BACKEND_XINPUT */

/* ---------------------------------------------------------------------------
 * SDL2 backend
 * -----------------------------------------------------------------------*/

#if PAD_BACKEND_SDL2

static u8 pad_sdl_axis_to_u8(int raw)
{
    /* SDL axis: -32768..32767 -> 0..255 with center at 128 */
    int val = ((raw + 32768) * 255) / 65535;
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (u8)val;
}

static u8 pad_sdl_trigger_to_u8(int raw)
{
    /* SDL trigger: 0..32767 -> 0..255 */
    int val = (raw * 255) / 32767;
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (u8)val;
}

/* Send port i's rumble state to its controller, with a fresh duration. */
static void pad_sdl_rumble_send(int i)
{
    SDL_GameController* gc = s_sdl_controllers[i];
    if (!gc) return;
    PadRumble* r = &s_rumble[i];
    const int on = r->lo || r->hi;
    SDL_GameControllerRumble(gc, r->lo, r->hi, on ? PAD_RUMBLE_HOLD_MS : 0);
    r->sent_ms = on ? pad_now_ms() : 0;
}

/* ---- hot-plug ------------------------------------------------------------
 * A DualSense over Bluetooth drops off when it is switched off or idles, and
 * comes back as a NEW SDL device (a new instance id, maybe another index).
 * This used to open controllers by device index == PS3 port and ask
 * SDL_IsGameController(port) again on every poll. With Homebrew's SDL2, which
 * is sdl2-compat over SDL3, a device index is resolved through a list that is
 * refreshed only by SDL_NumJoysticks() or by the SDL2 event filter (installed
 * with an SDL window, which the Metal build never makes): the list kept the
 * switched-off pad's instance id, SDL_IsGameController(0) stayed false, and
 * the pad that came back was never opened until the game was restarted.
 *
 * Now each poll drains SDL's joystick/controller events (nothing else here
 * reads them, and without a window nothing else does either, so they piled up
 * in the queue), closes a controller SDL says is gone, by its own handle, and
 * on a device event -- or every PAD_SDL_RESCAN_MS while a port is free, in
 * case a window's event loop ate the event -- walks the device list from
 * SDL_NumJoysticks() and opens every game controller that is not open yet:
 * into the port its GUID had last (the same pad reconnecting gets its port
 * back), else the lowest free one. A new controller starts with its motors
 * off and no rumble left over. The port reads connected again on the next
 * poll, and cellPadGetInfo/GetInfo2/PeriphGetInfo flag the change once with
 * CELL_PAD_STATUS_ASSIGN_CHANGES (port 0 is the virtual pad and never reads
 * disconnected). */
#define PAD_SDL_RESCAN_MS 1000

static int pad_sdl_port_of(SDL_JoystickID id)
{
    for (int p = 0; p < PAD_MAX_HOST_PORTS; p++)
        if (s_sdl_controllers[p] && s_sdl_ids[p] == id) return p;
    return -1;
}

/* Close port p's controller (gone, or shutting down). */
static void pad_sdl_close(int p, int gone)
{
    SDL_GameController* gc = s_sdl_controllers[p];
    if (!gc) return;
    if (gone) fprintf(stderr, "[cellPad] SDL controller on port %d disconnected (instance %d)\n", p, (int)s_sdl_ids[p]);
    else      SDL_GameControllerRumble(gc, 0, 0, 0);
    SDL_GameControllerClose(gc);
    s_sdl_controllers[p] = NULL;
    s_sdl_ids[p] = -1;
    memset(&s_rumble[p], 0, sizeof s_rumble[p]);
    pad_host_release(&s_host_state[p]);
}

/* Open every game controller SDL lists that no port has yet. */
static void pad_sdl_scan(void)
{
    const int n = SDL_NumJoysticks();   /* also refreshes sdl2-compat's index list */
    for (int idx = 0; idx < n; idx++) {
        if (!SDL_IsGameController(idx)) continue;
        const SDL_JoystickID id = SDL_JoystickGetDeviceInstanceID(idx);
        if (id < 0 || pad_sdl_port_of(id) >= 0) continue;
        const SDL_JoystickGUID guid = SDL_JoystickGetDeviceGUID(idx);
        int port = -1, back = 0;
        for (int p = 0; p < PAD_MAX_HOST_PORTS && port < 0; p++)
            if (!s_sdl_controllers[p] && s_sdl_had[p] && !memcmp(&s_sdl_last_guid[p], &guid, sizeof guid)) { port = p; back = 1; }
        for (int p = 0; p < PAD_MAX_HOST_PORTS && port < 0; p++)
            if (!s_sdl_controllers[p]) port = p;
        if (port < 0) return;                 /* every port has a controller */
        SDL_GameController* gc = SDL_GameControllerOpen(idx);
        if (!gc) {
            fprintf(stderr, "[cellPad] SDL controller %d: open failed: %s\n", idx, SDL_GetError());
            continue;
        }
        s_sdl_controllers[port] = gc;
        s_sdl_ids[port] = id;
        s_sdl_last_guid[port] = guid;
        s_sdl_had[port] = 1;
        memset(&s_rumble[port], 0, sizeof s_rumble[port]);   /* nothing left over from before */
        SDL_GameControllerRumble(gc, 0, 0, 0);
        const char* name = SDL_GameControllerName(gc);
        fprintf(stderr, "[cellPad] SDL controller %s on port %d: %s (instance %d)%s\n",
                back ? "reconnected" : "connected", port, name ? name : "?", (int)id,
                SDL_GameControllerHasRumble(gc) ? "" : " (no rumble)");
    }
}

/* Device changes since the last poll; force = scan now (init). */
static void pad_sdl_hotplug(int force)
{
    int changed = force;
    SDL_Event ev[32];
    int got;
    /* SDL_JOYAXISMOTION (0x600) up to the touch events (0x700): every
     * joystick and game controller event, which only this file uses. */
    while ((got = SDL_PeepEvents(ev, 32, SDL_GETEVENT, SDL_JOYAXISMOTION, SDL_JOYAXISMOTION + 0xFF)) > 0)
        for (int i = 0; i < got; i++)
            if (ev[i].type == SDL_JOYDEVICEADDED || ev[i].type == SDL_JOYDEVICEREMOVED ||
                ev[i].type == SDL_CONTROLLERDEVICEADDED || ev[i].type == SDL_CONTROLLERDEVICEREMOVED)
                changed = 1;
    for (int p = 0; p < PAD_MAX_HOST_PORTS; p++)
        if (s_sdl_controllers[p] && !SDL_GameControllerGetAttached(s_sdl_controllers[p])) {
            pad_sdl_close(p, 1);
            changed = 1;
        }
    int free_port = 0;
    for (int p = 0; p < PAD_MAX_HOST_PORTS; p++) if (!s_sdl_controllers[p]) free_port = 1;
    if (!free_port) return;
    const unsigned long long now = pad_now_ms();
    if (!changed && s_sdl_scan_ms && now - s_sdl_scan_ms < PAD_SDL_RESCAN_MS) return;
    s_sdl_scan_ms = now;
    pad_sdl_scan();
}

static void pad_poll_sdl2(void)
{
    SDL_GameControllerUpdate();
    pad_sdl_hotplug(0);

    for (int i = 0; i < PAD_MAX_HOST_PORTS; i++) {
        SDL_GameController* gc = s_sdl_controllers[i];
        if (!gc) {
            pad_host_release(&s_host_state[i]);
            memset(&s_rumble[i], 0, sizeof s_rumble[i]);
            continue;
        }

        s_host_state[i].connected = 1;
        s_host_state[i].keyboard_fallback = 0;

        /* Keep a rumble the title set going (see PAD_RUMBLE_HOLD_MS). */
        if (s_rumble[i].sent_ms && pad_now_ms() - s_rumble[i].sent_ms >= PAD_RUMBLE_RENEW_MS)
            pad_sdl_rumble_send(i);

        u16 btns = 0;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_BACK))          btns |= CELL_PAD_CTRL_SELECT;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_LEFTSTICK))     btns |= CELL_PAD_CTRL_L3;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_RIGHTSTICK))    btns |= CELL_PAD_CTRL_R3;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_START))         btns |= CELL_PAD_CTRL_START;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_DPAD_UP))       btns |= CELL_PAD_CTRL_UP;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_DPAD_RIGHT))    btns |= CELL_PAD_CTRL_RIGHT;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_DPAD_DOWN))     btns |= CELL_PAD_CTRL_DOWN;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_DPAD_LEFT))     btns |= CELL_PAD_CTRL_LEFT;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_LEFTSHOULDER))  btns |= CELL_PAD_CTRL_L1;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) btns |= CELL_PAD_CTRL_R1;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_Y))             btns |= CELL_PAD_CTRL_TRIANGLE;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_B))             btns |= CELL_PAD_CTRL_CIRCLE;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_A))             btns |= CELL_PAD_CTRL_CROSS;
        if (SDL_GameControllerGetButton(gc, SDL_CONTROLLER_BUTTON_X))             btns |= CELL_PAD_CTRL_SQUARE;

        /* Triggers via axis */
        int lt = SDL_GameControllerGetAxis(gc, SDL_CONTROLLER_AXIS_TRIGGERLEFT);
        int rt = SDL_GameControllerGetAxis(gc, SDL_CONTROLLER_AXIS_TRIGGERRIGHT);
        if (lt > 3000) btns |= CELL_PAD_CTRL_L2;
        if (rt > 3000) btns |= CELL_PAD_CTRL_R2;

        s_host_state[i].buttons = btns;

        /* Analog sticks */
        s_host_state[i].analog_lx = pad_sdl_axis_to_u8(SDL_GameControllerGetAxis(gc, SDL_CONTROLLER_AXIS_LEFTX));
        s_host_state[i].analog_ly = pad_sdl_axis_to_u8(SDL_GameControllerGetAxis(gc, SDL_CONTROLLER_AXIS_LEFTY));
        s_host_state[i].analog_rx = pad_sdl_axis_to_u8(SDL_GameControllerGetAxis(gc, SDL_CONTROLLER_AXIS_RIGHTX));
        s_host_state[i].analog_ry = pad_sdl_axis_to_u8(SDL_GameControllerGetAxis(gc, SDL_CONTROLLER_AXIS_RIGHTY));

        /* Triggers */
        s_host_state[i].trigger_l2 = pad_sdl_trigger_to_u8(lt);
        s_host_state[i].trigger_r2 = pad_sdl_trigger_to_u8(rt);

        /* Pressure: SDL has digital buttons, so 0 or 255 */
        s_host_state[i].press_up       = (btns & CELL_PAD_CTRL_UP)       ? 255 : 0;
        s_host_state[i].press_down     = (btns & CELL_PAD_CTRL_DOWN)     ? 255 : 0;
        s_host_state[i].press_left     = (btns & CELL_PAD_CTRL_LEFT)     ? 255 : 0;
        s_host_state[i].press_right    = (btns & CELL_PAD_CTRL_RIGHT)    ? 255 : 0;
        s_host_state[i].press_triangle = (btns & CELL_PAD_CTRL_TRIANGLE) ? 255 : 0;
        s_host_state[i].press_circle   = (btns & CELL_PAD_CTRL_CIRCLE)   ? 255 : 0;
        s_host_state[i].press_cross    = (btns & CELL_PAD_CTRL_CROSS)    ? 255 : 0;
        s_host_state[i].press_square   = (btns & CELL_PAD_CTRL_SQUARE)   ? 255 : 0;
        s_host_state[i].press_l1       = (btns & CELL_PAD_CTRL_L1)       ? 255 : 0;
        s_host_state[i].press_r1       = (btns & CELL_PAD_CTRL_R1)       ? 255 : 0;
    }
}

static void pad_init_backend(void)
{
    if (!s_sdl_inited) {
        if (SDL_WasInit(SDL_INIT_GAMECONTROLLER) == 0) {
            /* The game is not always the focused app: on macOS its window
             * can belong to another process (the launcher in
             * tools/host_window), and then this process is never in the
             * foreground. Controller input must not depend on focus. The
             * hint stops SDL dropping joystick events while it has no
             * focused window and lets Apple's GameController framework
             * deliver them to a background process
             * (GCController.shouldMonitorBackgroundEvents). The HIDAPI
             * drivers (DualShock 4, DualSense) read IOKit directly and never
             * cared. SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=0 in the
             * environment still wins over this. */
            SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
            SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER);
        }
        s_sdl_inited = 1;
    }
    memset(s_sdl_controllers, 0, sizeof(s_sdl_controllers));
    for (int p = 0; p < PAD_MAX_HOST_PORTS; p++) s_sdl_ids[p] = -1;

    /* Open any controllers already connected */
    pad_sdl_hotplug(1);
}

static void pad_shutdown_backend(void)
{
    for (int i = 0; i < PAD_MAX_HOST_PORTS; i++) pad_sdl_close(i, 0);
    memset(s_rumble, 0, sizeof s_rumble);
}

#endif /* PAD_BACKEND_SDL2 */

/* ---------------------------------------------------------------------------
 * Poll dispatcher
 * -----------------------------------------------------------------------*/

/* Re-poll the host at most every PAD_POLL_INTERVAL_MS; serve cached state in
 * between.
 *
 * XInputGetState is a driver call, and on a slot with nothing plugged in it is
 * an expensive one: measured at 97.6 us for a sweep of the 7 host ports on a
 * machine with no controller attached. cellPadGetData ran that sweep on EVERY
 * call, and a guest that polls the pad from a spin loop -- The Simpsons Arcade
 * Game does, in its input stage -- therefore spends its whole main thread
 * inside the input driver.
 *
 * 4 ms is 250 Hz, well above the 60 Hz a title can actually observe (and above
 * a real DualShock 3's own report rate), so no game can tell the difference,
 * including one sampling for button-press edges. */
#define PAD_POLL_INTERVAL_MS 4

static unsigned long long pad_now_ms(void)
{
#ifdef _WIN32
    return (unsigned long long)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000ull +
           (unsigned long long)ts.tv_nsec / 1000000ull;
#endif
}

static void pad_poll_backend_locked(void)
{
    static unsigned long long s_last_ms = 0;
    unsigned long long now = pad_now_ms();
    if (s_last_ms && now - s_last_ms < PAD_POLL_INTERVAL_MS) return;
    if (!pad_lock_try()) return;   /* another thread is polling: its result serves this read */
    s_last_ms = now;
    /* A real pad streams reports at roughly this rate whether or not anything
     * changed, and libpad buffers them; a fresh sweep is a fresh report for
     * every port. cellPadGetData hands each one out exactly once. */
    for (int _i = 0; _i < PAD_MAX_HOST_PORTS; _i++) s_data_fresh[_i] = 1;

#if PAD_BACKEND_XINPUT
    pad_poll_xinput();
#elif PAD_BACKEND_SDL2
    pad_poll_sdl2();
#endif
    for (int i = 0; i < PAD_MAX_HOST_PORTS; ++i)
        pad_apply_controller_mapping(&s_host_state[i]);
#ifdef _WIN32
    int any_controller_connected = 0;
    for (int i = 0; i < PAD_MAX_HOST_PORTS; ++i)
        any_controller_connected |= s_host_state[i].connected;
    if (!any_controller_connected) pad_poll_keyboard();
#endif
    pad_track_assign();
    pad_unlock();
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 cellPadInit(u32 max_connect)
{
    printf("[cellPad] Init(max_connect=%u)\n", max_connect);
    AcquireSRWLockExclusive(&s_pad_lock);

    if (s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_ALREADY_OPENED);

    /* The bound is the 127-slot pad table, not the 7 ports: larger requests
     * are accepted and clamped. Resistance: Fall of Man asks for 8; rejected,
     * the library stayed closed, every GetInfo failed and the title never
     * polled the pad -- stuck on "Press the START button". */
    if (max_connect == 0 || max_connect > 127)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    s_pad_initialized = 1;
    s_max_connect = max_connect > CELL_PAD_MAX_PORT_NUM ? CELL_PAD_MAX_PORT_NUM : max_connect;
    memset(s_port_setting, 0, sizeof(s_port_setting));
    for (int i = 0; i < PAD_MAX_HOST_PORTS; i++)
        pad_host_release(&s_host_state[i]);

    pad_lock();
    pad_init_backend();
    pad_unlock();

    /* Do an initial poll to detect connected controllers */
    pad_poll_backend_locked();
    /* What is there at start is the baseline; only later changes are
     * flagged CELL_PAD_STATUS_ASSIGN_CHANGES. */
    for (int i = 0; i < PAD_MAX_HOST_PORTS; i++) {
        s_port_reported[i] = s_host_state[i].connected || i == 0;
        s_port_changed[i] = 0;
    }

    PAD_UNLOCK_RETURN(CELL_OK);
}

s32 cellPadEnd(void)
{
    printf("[cellPad] End()\n");
    AcquireSRWLockExclusive(&s_pad_lock);

    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    pad_lock();
    pad_shutdown_backend();
    pad_unlock();

    s_pad_initialized = 0;
    s_max_connect = 0;
    PAD_UNLOCK_RETURN(CELL_OK);
}

void cellPad_poll(void)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (s_pad_initialized) {
        pad_poll_backend_locked();
    }
    ReleaseSRWLockExclusive(&s_pad_lock);
}

/* HLE args are GUEST effective addresses; guest structs are BIG-ENDIAN. The
 * vm_write helpers from ppu_memory.h translate + byte-swap. (cellPad was
 * written for host pointers and was never actually invoked until its NIDs
 * were fixed.) These used to be redeclared here with a 64-bit address
 * parameter, which conflicts with the header now that it is included. */

s32 cellPadGetData(u32 port_no, CellPadData* data_guest)
{
    { static int _once = 0;
      if (!_once++) printf("[cellPad] GetData polling begins (port %u)\n", port_no); }
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    if (port_no >= s_max_connect || !data_guest)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    /* Build in a local host struct, then copy to guest memory big-endian. */
    CellPadData _d; CellPadData* data = &_d;
    memset(data, 0, sizeof(CellPadData));

    /* Poll fresh state */
    pad_poll_backend_locked();

    /* YDKJ_INJECT_PAD: the game's wait loop polls the pad (cellPadSetActDirect /
     * GetData) -- if it's parked on a "press button" prompt after loading, no
     * host input means it waits forever. Inject a real button PULSE (CROSS+START+
     * CIRCLE) so it advances. Legit input simulation, not forged pixels. */
    /* NOTE: cache getenv() -- this runs on every poll from multiple threads, and the
     * uncached version crashed deterministically (varying fault address = race). Every
     * other probe in this tree uses the cached `static int x=-1` pattern; match it. */
    static int  s_inj   = -1;
    static u16  s_btn   = 0x0008u;   /* default START */
    static long s_delay = 3000;
    if (s_inj < 0) {
        s_inj = getenv("YDKJ_INJECT_PAD") ? 1 : 0;
        const char* be = getenv("YDKJ_PAD_BTN");   if (be) s_btn   = (u16)strtoul(be,0,0);
        const char* de = getenv("YDKJ_PAD_DELAY"); if (de) s_delay = strtol(de,0,0);
    }
    /* Gate on WALL-CLOCK, not poll count: the game polls the pad hundreds of thousands
     * of times inside an early wait loop, so any count-based delay fires during init
     * (crash: guest ctr=0x0005F8A0 r3=0 -- button handler on an unconstructed object).
     * YDKJ_PAD_DELAY is now SECONDS to wait before the first injected press. */
    if (s_inj && port_no < PAD_MAX_HOST_PORTS) {
        static long _pc = 0; _pc++;
        /* Cheap counter gate FIRST: clock() is a slow call and this poll loop is hammered
         * by the guest from multiple threads; sampling it every poll perturbs timing. Only
         * consult the clock once every 256 polls, and never after arming. */
        static int     s_armed = 0;
        static clock_t s_t0    = 0;
        if (!s_armed) {
            if ((_pc & 0xFF) == 0) {
                if (!s_t0) s_t0 = clock();
                else if ((double)(clock() - s_t0) / (double)CLOCKS_PER_SEC >= (double)s_delay) s_armed = 1;
            }
            goto skip_inject;
        }
        u16 btn = s_btn; long delay = 0;
        /* Delay past init, then a single clean pulse every ~180 calls (press 15, release 165). */
        if (_pc > delay) {
            long ph = (_pc - delay) % 180;
            if (ph < 15) s_host_state[port_no].buttons |= btn;
            else         s_host_state[port_no].buttons &= ~btn;
            s_host_state[port_no].connected = 1;
        }
    }
skip_inject: ;

    if (port_no >= PAD_MAX_HOST_PORTS || (!s_host_state[port_no].connected && port_no != 0)) {
        goto emit;   /* data stays zeroed -> len=0 */
    }
    /* port 0 is always a valid (virtual) pad; s_host_state[0] is neutral if no
     * physical device, which yields no-buttons-pressed data below. */

    {
    PadHostState* hs = &s_host_state[port_no];
    u32 setting = s_port_setting[port_no];

    /* Determine data length based on port settings */
    s32 len = CELL_PAD_LEN_CHANGE_DEFAULT;
    if (setting & CELL_PAD_SETTING_SENSOR_ON)
        len = CELL_PAD_LEN_CHANGE_SENSOR_ON;
    else if (setting & CELL_PAD_SETTING_PRESS_ON)
        len = CELL_PAD_LEN_CHANGE_PRESS_ON;

    data->len = len;
    /* Finalize this call's packet freshness before any injected hold budget
     * advances. len=0 drain reads still build the ordinary payload, but do
     * not consume a PAD_FILE report. See the emission contract below. */
    {
        static int always = -1;
        if (always < 0) always = getenv("PAD_ALWAYS_CHANGE") ? 1 : 0;
        if (!always && port_no < PAD_MAX_HOST_PORTS && data->len) {
            if (!s_data_fresh[port_no]) data->len = 0;   /* nothing new: end the drain */
            else                        s_data_fresh[port_no] = 0;
        }
    }
    /* button[0] is reserved and 0 on hardware; button[1] carries the pad mode
     * in its high nibble (0x7) and len/2 in its low one. GH3 drops any packet
     * whose button[0] is non-zero -- with len (8) there, every press was
     * received and thrown away, and "Press any button" never advanced. */
    data->button[0] = 0;
    data->button[1] = (u16)(0x70 | ((len / 2) & 0xF));

    /* Digital buttons. hs->buttons packs both halves into one 16-bit mask
     * (SELECT=bit0..LEFT=bit7 = DIGITAL1; L2=bit8..SQUARE=bit15 = DIGITAL2).
     * Split correctly — writing the whole value into DIGITAL1 with DIGITAL2=0
     * left every face button (cross/circle/triangle/square, L1/L2/R1/R2) dead.
     * (via sagemono, PR #42) */
    data->button[CELL_PAD_BTN_OFFSET_DIGITAL1] = (u16)((hs->buttons & 0xFF) |
        (hs->ps_button ? CELL_PAD_CTRL_PS : 0));
    data->button[CELL_PAD_BTN_OFFSET_DIGITAL2] = (u16)((hs->buttons >> 8) & 0xFF);

    /* PAD_AUTOPRESS: headless bring-up input -- pulse CROSS then START every few
     * seconds of polling so boot screens that wait for input advance without a
     * human at the pad. Env-gated test scaffolding, off by default. */
    { static int s_ap = -1; static unsigned s_apn = 0;
      if (s_ap < 0) s_ap = getenv("PAD_AUTOPRESS") ? 1 : 0;
      if (s_ap && port_no == 0) {
          unsigned ph = s_apn++ % 240;
          if (ph < 12)
              data->button[CELL_PAD_BTN_OFFSET_DIGITAL2] |= 0x40;   /* CROSS */
          else if (ph >= 120 && ph < 132)
              data->button[CELL_PAD_BTN_OFFSET_DIGITAL1] |= 0x08;   /* START */
      } }

    /* PAD_SCRIPT="<sec>:<mask>,<sec>:<mask>,..." -- press a NAMED button at a
     * given wall-clock second (or flip, under PAD_CLOCK=flips), each held ~250 ms. PAD_AUTOPRESS only pulses
     * CROSS and START, which is enough to clear a "press start" screen but not
     * to navigate a menu: You Don't Know Jack needs DOWN to move off its name
     * field before CROSS means anything. Masks are CELL_PAD_CTRL_* packed as
     * (DIGITAL2 << 8) | DIGITAL1: START 0x0008, UP 0x0010, DOWN 0x0040,
     * LEFT 0x0080, RIGHT 0x0020, CROSS 0x4000, CIRCLE 0x2000, TRIANGLE 0x1000.
     * Legit input simulation on the same footing as PAD_AUTOPRESS/PAD_STICK. */
    { static int s_sc = -1;
      static struct { double t; unsigned mask; } ev[256]; static int n_ev = 0;
      static unsigned long long t0 = 0;   /* not ULONGLONG: this builds on POSIX too */
      if (s_sc < 0) {
          s_sc = 0;
          const char* e = getenv("PAD_SCRIPT");
          if (e && *e) {
              s_sc = 1; t0 = GetTickCount64();
              const char* p = e;
              while (*p && n_ev < 256) {
                  double t = strtod(p, (char**)&p);
                  if (*p == ':') p++;
                  unsigned m = (unsigned)strtoul(p, (char**)&p, 0);
                  ev[n_ev].t = t; ev[n_ev].mask = m; n_ev++;
                  while (*p == ',' || *p == ' ') p++;
              }
              printf("[cellPad] PAD_SCRIPT: %d event(s)%s\n", n_ev,
                     *p ? " -- TRUNCATED at 256, the rest are ignored" : "");
          }
      }
      if (s_sc && port_no == 0) {
          double now = pad_clock(t0);
          /* Latch, do not window. A fixed time window is missed whenever the
           * title stops polling the pad for longer than the window (it does
           * exactly that on a loading screen), so the press silently never
           * happens. Fire on the first poll at or after the event time and
           * hold it for a fixed number of polls, which is what a real press
           * looks like regardless of poll rate. */
          static int fired[256], held[256];
          for (int i = 0; i < n_ev; i++) {
              if (!fired[i] && now >= ev[i].t) { fired[i] = 1; held[i] = 40; }
              if (held[i] > 0) {
                  held[i]--;
                  data->button[CELL_PAD_BTN_OFFSET_DIGITAL1] |= (u16)(ev[i].mask & 0xFF);
                  data->button[CELL_PAD_BTN_OFFSET_DIGITAL2] |= (u16)((ev[i].mask >> 8) & 0xFF);
                  static int said[256];
                  if (!said[i]) { said[i] = 1;
                      printf("[cellPad] PAD_SCRIPT t=%.1fs press 0x%04X\n", ev[i].t, ev[i].mask); }
              }
          }
      } }

    /* PAD_FILE=<path> -- closed-loop input. Each poll, if the file holds a
     * button mask (same encoding as PAD_SCRIPT), press it for a fixed number
     * of positive guest reports and empty the file on admission. A driver
     * outside the process can look at the log or a frame dump, decide which
     * screen the title is on, and
     * write the next button -- which a fixed time schedule cannot do, because
     * a menu system with several screens wanders when driven blind.
     *
     *   echo 0x0008 > pad.txt    # START
     *   echo 0x4000 > pad.txt    # CROSS
     *   echo 0x0040 4 > pad.txt  # DOWN for 4 reports (menus auto-repeat a
     *                            # long hold; the default is 40)
     */
    { static int s_pf = -1; static char pf_path[512];
      static unsigned held_mask = 0; static int held_n = 0;
      if (s_pf < 0) { const char* e = getenv("PAD_FILE");
                      s_pf = (e && *e) ? 1 : 0;
                      if (s_pf) { strncpy(pf_path, e, sizeof pf_path - 1); pf_path[sizeof pf_path - 1] = 0; } }
      if (s_pf && port_no == 0) {
          if (held_n <= 0) {
              FILE* f = fopen(pf_path, "rb");
              if (f) {
                  char buf[64]; buf[0] = 0;
                  if (fgets(buf, sizeof buf, f)) {
                      char* rest;
                      unsigned m = (unsigned)strtoul(buf, &rest, 0);
                      int n = (int)strtol(rest, 0, 0);
                      if (m) { held_mask = m; held_n = n > 0 ? n : 40;
                               printf("[cellPad] PAD_FILE press 0x%04X\n", m); fflush(stdout); }
                  }
                  fclose(f);
                  if (held_n > 0) { FILE* w = fopen(pf_path, "wb"); if (w) fclose(w); }
              }
          }
          if (held_n > 0) {
              if (data->len > 0) held_n--;
              data->button[CELL_PAD_BTN_OFFSET_DIGITAL1] |= (u16)(held_mask & 0xFF);
              data->button[CELL_PAD_BTN_OFFSET_DIGITAL2] |= (u16)((held_mask >> 8) & 0xFF); }
      } }

    /* Analog sticks */
    /* PAD_STICK="lx,ly,rx,ry" (0-255, 128 = centred): hold the analog sticks at
     * fixed positions. Rubber Ducky drives its camera from the sticks and never
     * moves on its own, so with no host pad the view is frozen wherever it
     * started -- which can leave the subject of the demo off-screen or too far
     * away to make out. Legit input simulation, same footing as YDKJ_INJECT_PAD. */
    { static int s_st = -1;
      static unsigned char s_v[4] = {128,128,128,128};
      /* Optional 5th/6th fields: the seconds the hold starts and ends, so the menus
       * before gameplay are not driven by it (PAD_STICK="128,0,128,128,300,320"). */
      static double s_from = 0, s_until = 1e30; static unsigned long long s_t0 = 0;
      if (!s_t0) s_t0 = GetTickCount64();
      if (s_st < 0) { const char* e = getenv("PAD_STICK");
        s_st = e ? 1 : 0;
        if (e) { int a=128,b=128,c=128,d=128;
                 sscanf(e, "%d,%d,%d,%d,%lf,%lf", &a,&b,&c,&d,&s_from,&s_until);
                 s_v[0]=(unsigned char)a; s_v[1]=(unsigned char)b;
                 s_v[2]=(unsigned char)c; s_v[3]=(unsigned char)d; } }
      if (s_st && pad_clock(s_t0) >= s_from && pad_clock(s_t0) < s_until) { hs->analog_lx = s_v[0]; hs->analog_ly = s_v[1];
                  hs->analog_rx = s_v[2]; hs->analog_ry = s_v[3];
                  hs->connected = 1; }
      /* PAD_SWEEP=<seconds per step>: walk the sticks through a fixed set of
       * deflections instead of holding one. Guessing a single stick position and
       * re-running costs ~5 minutes a try; a sweep answers "can input move the
       * camera anywhere useful" in one run, and the frame dumps say which step
       * paid off. Steps: centre, each left-stick direction, then each right. */
      { static int sw = -1; static clock_t t0 = 0;
        if (sw < 0) { const char* e = getenv("PAD_SWEEP");
                      sw = e ? atoi(e) : 0; if (sw < 0) sw = 0; }
        if (sw) {
            static const unsigned char steps[9][4] = {
                {128,128,128,128},
                {128,  0,128,128}, {128,255,128,128},
                {  0,128,128,128}, {255,128,128,128},
                {128,128,128,  0}, {128,128,128,255},
                {128,128,  0,128}, {128,128,255,128},
            };
            if (!t0) t0 = clock();
            long el = (long)((double)(clock() - t0) / (double)CLOCKS_PER_SEC);
            int k = (int)((el / (sw > 0 ? sw : 1)) % 9);
            hs->analog_lx = steps[k][0]; hs->analog_ly = steps[k][1];
            hs->analog_rx = steps[k][2]; hs->analog_ry = steps[k][3];
            hs->connected = 1;
            { static int lastk = -1;
              if (k != lastk) { lastk = k;
                fprintf(stderr, "[PADSWEEP] step %d: l=(%u,%u) r=(%u,%u)%c", k,
                        steps[k][0], steps[k][1], steps[k][2], steps[k][3], 10); } }
        } } }
    data->button[CELL_PAD_BTN_OFFSET_ANALOG_RIGHT_X] = hs->analog_rx;
    data->button[CELL_PAD_BTN_OFFSET_ANALOG_RIGHT_Y] = hs->analog_ry;
    data->button[CELL_PAD_BTN_OFFSET_ANALOG_LEFT_X]  = hs->analog_lx;
    data->button[CELL_PAD_BTN_OFFSET_ANALOG_LEFT_Y]  = hs->analog_ly;

    /* Pressure-sensitive buttons (only meaningful if PRESS_ON) */
    if (setting & CELL_PAD_SETTING_PRESS_ON) {
        data->button[CELL_PAD_BTN_OFFSET_PRESS_RIGHT]    = hs->press_right;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_LEFT]     = hs->press_left;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_UP]       = hs->press_up;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_DOWN]     = hs->press_down;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_TRIANGLE] = hs->press_triangle;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_CIRCLE]   = hs->press_circle;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_CROSS]    = hs->press_cross;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_SQUARE]   = hs->press_square;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_L1]       = hs->press_l1;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_R1]       = hs->press_r1;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_L2]       = hs->trigger_l2;
        data->button[CELL_PAD_BTN_OFFSET_PRESS_R2]       = hs->trigger_r2;
        /* A button whose digital bit is set reads full pressure. Injected
         * input (PAD_FILE/PAD_SCRIPT/PAD_AUTOPRESS, keyboard) only sets the
         * digital bits, and the host values above zeroed any pressure it had
         * written -- so a title keying its menus off the pressure words (GH3,
         * Tornado Outbreak's title screen) never saw the press. */
        static const struct { u8 word; u8 bit; u8 off; } pm[] = {
            { CELL_PAD_BTN_OFFSET_DIGITAL1, 0x20, CELL_PAD_BTN_OFFSET_PRESS_RIGHT },
            { CELL_PAD_BTN_OFFSET_DIGITAL1, 0x80, CELL_PAD_BTN_OFFSET_PRESS_LEFT },
            { CELL_PAD_BTN_OFFSET_DIGITAL1, 0x10, CELL_PAD_BTN_OFFSET_PRESS_UP },
            { CELL_PAD_BTN_OFFSET_DIGITAL1, 0x40, CELL_PAD_BTN_OFFSET_PRESS_DOWN },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x10, CELL_PAD_BTN_OFFSET_PRESS_TRIANGLE },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x20, CELL_PAD_BTN_OFFSET_PRESS_CIRCLE },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x40, CELL_PAD_BTN_OFFSET_PRESS_CROSS },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x80, CELL_PAD_BTN_OFFSET_PRESS_SQUARE },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x04, CELL_PAD_BTN_OFFSET_PRESS_L1 },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x08, CELL_PAD_BTN_OFFSET_PRESS_R1 },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x01, CELL_PAD_BTN_OFFSET_PRESS_L2 },
            { CELL_PAD_BTN_OFFSET_DIGITAL2, 0x02, CELL_PAD_BTN_OFFSET_PRESS_R2 } };
        for (unsigned k = 0; k < sizeof pm / sizeof pm[0]; k++)
            if ((data->button[pm[k].word] & pm[k].bit) && data->button[pm[k].off] < 255)
                data->button[pm[k].off] = 255;
    }

    /* Sensor data (only meaningful if SENSOR_ON) */
    if (setting & CELL_PAD_SETTING_SENSOR_ON) {
        /* Default sensor values: accelerometer at rest (512 = 1g center) */
        data->button[CELL_PAD_BTN_OFFSET_SENSOR_X] = hs->sensor_x;
        data->button[CELL_PAD_BTN_OFFSET_SENSOR_Y] = hs->sensor_y;
        data->button[CELL_PAD_BTN_OFFSET_SENSOR_Z] = hs->sensor_z;
        data->button[CELL_PAD_BTN_OFFSET_SENSOR_G] = hs->sensor_g;
    }
    }  /* end connected block */

emit:
    /* A non-zero len means "here is a CHANGE packet" -- the constant is literally
     * named CELL_PAD_LEN_CHANGE_DEFAULT. Hardware reports len = 0 once the port
     * has nothing new, and titles DRAIN on that: The Simpsons Arcade Game's
     * input stage is
     *
     *     while (cellPadGetData(port, &d) == CELL_OK && d.len > 0) { ... }
     *
     * (func_0013CFA8, loop 0x13D648 -> 0x13D040). Reporting a change on every
     * call made that loop infinite: the main thread pegged one core inside
     * libpad and the title never got past its first rendered screen.
     *
     * Freshness was finalized before input injection above. Hand out one
     * packet per host report and report len = 0 for any read
     * after that until the next one arrives (pad_poll_backend sets the flag when
     * it actually re-polls, every PAD_POLL_INTERVAL_MS). That is what the
     * hardware does: a pad streams reports at a fixed rate whether or not
     * anything changed, and libpad buffers them.
     *
     * Reporting only on CHANGE instead is wrong and was tried: a HELD button
     * produces exactly one packet, so a title that reads button[] only when
     * len > 0 sees the press for a single frame and then sees nothing. In this
     * game that came out as jump being the only control that seemed to work and
     * attack firing once.
     *
     * PAD_ALWAYS_CHANGE=1 restores the old always-a-packet behaviour. */
    {
        unsigned int ea = (unsigned int)(uintptr_t)data_guest;
        vm_write32((unsigned long long)ea + 0, (unsigned int)data->len);   /* len (s32) */
        for (int _i = 0; _i < CELL_PAD_MAX_CODES; _i++)                    /* button[] u16 */
            vm_write16((unsigned long long)ea + 4 + (unsigned int)_i * 2, data->button[_i]);
    }
    PAD_UNLOCK_RETURN(CELL_OK);
}

/* cellPadGetInfo (v1 API, NID 0x3AAAD464) — used by PSL1GHT's ioPadGetInfo.
 * Layout (RPCS3 cellPad.h CellPadInfo, all big-endian):
 *   +0x00 u32 max_connect        +0x04 u32 now_connect   +0x08 u32 system_info
 *   +0x0C u16 vendor_id[127]     +0x10A u16 product_id[127] +0x208 u8 status[127]
 *   (CELL_MAX_PADS / PSL1GHT MAX_PADS = 127 -- NOT 7; see below)
 * Leaving this unimplemented let the guest read stack garbage as pad state and
 * run off into the weeds before ever reaching gcm init. */
s32 cellPadGetInfo(CellPadInfo2* info)
{
    uint32_t gaddr = (uint32_t)(uintptr_t)info;
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);
    if (!gaddr)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    pad_poll_backend_locked();

    /* Present a virtual pad on port 0 when nothing physical is attached, the
     * same as cellPadGetInfo2 does.
     *
     * The two entry points disagreed: GetInfo2 carried the fallback and GetInfo
     * did not, so which one a title imports decided whether it believed a
     * controller existed. The Orange Box's chooser imports GetInfo (v1) and
     * never GetInfo2, so it saw zero pads, opened "SIXAXIS wireless controller
     * not detected", took the auto-OK, and opened it again -- forever, before
     * it ever finished booting. Both entry points describe the same hardware
     * and have to answer the same way. */
    u32 connected = 0;
    for (u32 i = 0; i < s_max_connect && i < PAD_MAX_HOST_PORTS; i++)
        if (s_host_state[i].connected || i == 0) connected++;

    vm_write32(gaddr + 0x00, s_max_connect);
    vm_write32(gaddr + 0x04, connected);
    vm_write32(gaddr + 0x08, 0);                       /* system_info */
    /* CellPadInfo (v1) sizes its arrays by CELL_MAX_PADS = 127, not by the 7
     * ports: vendor_id[127] @0x0C, product_id[127] @0x10A, status[127] @0x208.
     * Packed for 7 ports, status landed at 0x28 and GH3 -- which reads
     * info+0x208+port -- saw port 0 disconnected and never polled the pad. */
    for (u32 i = 0; i < CELL_PAD_MAX_PORT_NUM; i++) {
        int on = (i < s_max_connect && i < PAD_MAX_HOST_PORTS &&
                  (s_host_state[i].connected || i == 0));
        vm_write16(gaddr + 0x00C + i * 2, on ? 0x054C : 0);   /* vendor: Sony  */
        vm_write16(gaddr + 0x10A + i * 2, on ? 0x0268 : 0);   /* product: DS3  */
        vm_write8(gaddr + 0x208 + i, (u8)((on ? CELL_PAD_STATUS_CONNECTED : 0) | pad_status_take(i)));
    }
    PAD_UNLOCK_RETURN(CELL_OK);
}

s32 cellPadGetInfo2(CellPadInfo2* info_guest)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    if (!info_guest)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    /* Poll to get latest connection state */
    pad_poll_backend_locked();

    /* Build in a local host struct, then copy to guest memory big-endian. */
    CellPadInfo2 _in; CellPadInfo2* info = &_in;
    memset(info, 0, sizeof(CellPadInfo2));
    info->max_connect = s_max_connect;

    u32 connected = 0;
    for (u32 i = 0; i < s_max_connect && i < PAD_MAX_HOST_PORTS; i++) {
        /* Always present a virtual pad on port 0 (standard emulator behavior) so
         * boot flows that block until a controller is connected can proceed even
         * with no physical device attached. */
        if (s_host_state[i].connected || i == 0) {
            info->port_status[i]       = CELL_PAD_STATUS_CONNECTED | pad_status_take(i);
            info->port_setting[i]      = s_port_setting[i];
            info->device_capability[i] = CELL_PAD_CAPABILITY_PS3_CONFORMITY
                                       | CELL_PAD_CAPABILITY_PRESS_MODE
                                       | CELL_PAD_CAPABILITY_SENSOR_MODE
                                       | CELL_PAD_CAPABILITY_HP_ANALOG_STICK
                                       | CELL_PAD_CAPABILITY_ACTUATOR;
            info->device_type[i]       = CELL_PAD_DEV_TYPE_STANDARD;
            connected++;
        } else {
            info->port_status[i] = CELL_PAD_STATUS_DISCONNECTED | pad_status_take(i);
        }
    }
    info->now_connect = connected;

    /* copy to guest big-endian (CellPadInfo2 is all u32 fields) */
    {
        unsigned int ea = (unsigned int)(uintptr_t)info_guest;
        for (unsigned int _o = 0; _o < sizeof(CellPadInfo2); _o += 4)
            vm_write32((unsigned long long)ea + _o, *(u32*)((char*)info + _o));
    }
    PAD_UNLOCK_RETURN(CELL_OK);
}

/* cellPadPeriphGetInfo -- the peripheral-class view of the pad ports (NID
 * 0x4CC9B68D). LBP polls this during boot; leaving it unresolved returned
 * CELL_OK with the caller's CellPadPeriphInfo left as uninitialised stack
 * garbage, so the game read junk port_status/pclass_type for all 7 ports.
 *
 * Layout (SDK cell/pad/pad_codes.h) -- all u32, so the whole struct marshals
 * big-endian in one sweep like cellPadGetInfo2:
 *   max_connect, now_connect, system_info,
 *   port_status[7], port_setting[7], device_capability[7],
 *   device_type[7], pclass_type[7], pclass_profile[7]
 * We report a standard DUALSHOCK-class pad (pclass_type STANDARD = 0, no
 * profile bits) mirroring the ports cellPadGetInfo2 already reports. */
s32 cellPadPeriphGetInfo(CellPadPeriphInfo* info_guest)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);
    if (!info_guest)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    pad_poll_backend_locked();

    CellPadPeriphInfo _in; CellPadPeriphInfo* info = &_in;
    memset(info, 0, sizeof(CellPadPeriphInfo));
    info->max_connect = s_max_connect;
    info->system_info = 0;

    u32 connected = 0;
    for (u32 i = 0; i < s_max_connect && i < CELL_PAD_MAX_PORT_NUM; i++) {
        /* Same virtual-pad-on-port-0 rule as cellPadGetInfo2, so both views of
         * the ports agree (a game that cross-checks them must see one truth). */
        if ((i < PAD_MAX_HOST_PORTS && s_host_state[i].connected) || i == 0) {
            info->port_status[i]       = CELL_PAD_STATUS_CONNECTED | pad_status_take(i);
            info->port_setting[i]      = s_port_setting[i];
            info->device_capability[i] = CELL_PAD_CAPABILITY_PS3_CONFORMITY
                                       | CELL_PAD_CAPABILITY_PRESS_MODE
                                       | CELL_PAD_CAPABILITY_SENSOR_MODE
                                       | CELL_PAD_CAPABILITY_HP_ANALOG_STICK
                                       | CELL_PAD_CAPABILITY_ACTUATOR;
            info->device_type[i]       = CELL_PAD_DEV_TYPE_STANDARD;
            info->pclass_type[i]       = CELL_PAD_PCLASS_TYPE_STANDARD;
            info->pclass_profile[i]    = 0;
            connected++;
        } else {
            info->port_status[i] = CELL_PAD_STATUS_DISCONNECTED | pad_status_take(i);
        }
    }
    info->now_connect = connected;

    { unsigned int ea = (unsigned int)(uintptr_t)info_guest;
      for (unsigned int _o = 0; _o < sizeof(CellPadPeriphInfo); _o += 4)
          vm_write32((unsigned long long)ea + _o, *(u32*)((char*)info + _o)); }

    { static int _n = 0;
      if (PS3_LOG_FIRST(_n, 2))
          printf("[cellPad] PeriphGetInfo(max=%u now=%u) -> STANDARD class\n",
                 info->max_connect, connected); }
    PAD_UNLOCK_RETURN(CELL_OK);
}

s32 cellPadSetPortSetting(u32 port_no, u32 port_setting)
{
    printf("[cellPad] SetPortSetting(port=%u, setting=0x%X)\n",
           port_no, port_setting);

    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    if (port_no >= CELL_PAD_MAX_PORT_NUM)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    s_port_setting[port_no] = port_setting;
    PAD_UNLOCK_RETURN(CELL_OK);
}

/* cellPadInfoPressMode (NID 0x0E2DFAAD) / cellPadSetPressMode (NID 0xF83F8182)
 *
 * Press mode makes the digital buttons report an analog 0..255 pressure in the
 * extended part of CellPadData. A DS3 supports it, and the virtual pad reported
 * by cellPadGetInfo/GetInfo2 already advertises CELL_PAD_CAPABILITY_PRESS_MODE,
 * so these have to agree with that claim rather than fall through to the
 * unresolved-NID path.
 *
 * Info returns 1 when the port supports press mode; Set toggles it, recorded in
 * the same s_port_setting word cellPadSetPortSetting uses so a title that reads
 * the setting back sees what it asked for. */
s32 cellPadInfoPressMode(u32 port_no)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);
    if (port_no >= CELL_PAD_MAX_PORT_NUM)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);
    PAD_UNLOCK_RETURN(1);                      /* supported */
}

s32 cellPadSetPressMode(u32 port_no, u32 mode)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);
    if (port_no >= CELL_PAD_MAX_PORT_NUM)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);
    if (mode)
        s_port_setting[port_no] |= CELL_PAD_SETTING_PRESS_ON;
    else
        s_port_setting[port_no] &= ~(u32)CELL_PAD_SETTING_PRESS_ON;
    PAD_UNLOCK_RETURN(CELL_OK);
}

/* cellPadInfoSensorMode (NID 0x78200559) / cellPadSetSensorMode (NID 0xBE5BE3BA)
 *
 * The same pair for the SIXAXIS sensors, which the virtual pad also advertises
 * (CELL_PAD_CAPABILITY_SENSOR_MODE). Unresolved, Info answered 0 -- "no
 * sensors" from a pad whose capability word says otherwise -- and Set never
 * turned the sensor block of CellPadData on. With SENSOR_ON the packet grows to
 * CELL_PAD_LEN_CHANGE_SENSOR_ON and carries the at-rest readings (see
 * cellPadGetData), which is what a DS3 lying still reports. */
s32 cellPadInfoSensorMode(u32 port_no)
{
    if (!s_pad_initialized)
        return CELL_PAD_ERROR_NOT_OPENED;
    if (port_no >= CELL_PAD_MAX_PORT_NUM)
        return CELL_PAD_ERROR_INVALID_PARAMETER;
    return 1;                                  /* supported */
}

s32 cellPadSetSensorMode(u32 port_no, u32 mode)
{
    if (!s_pad_initialized)
        return CELL_PAD_ERROR_NOT_OPENED;
    if (port_no >= CELL_PAD_MAX_PORT_NUM)
        return CELL_PAD_ERROR_INVALID_PARAMETER;
    const u32 was = s_port_setting[port_no] & CELL_PAD_SETTING_SENSOR_ON;
    if (mode)
        s_port_setting[port_no] |= CELL_PAD_SETTING_SENSOR_ON;
    else
        s_port_setting[port_no] &= ~(u32)CELL_PAD_SETTING_SENSOR_ON;
    if (was != (s_port_setting[port_no] & CELL_PAD_SETTING_SENSOR_ON))
        printf("[cellPad] SetSensorMode(port %u) -> %s\n", port_no, mode ? "on" : "off");
    return CELL_OK;
}

s32 cellPadGetCapabilityInfo(u32 port_no, CellPadCapabilityInfo* info)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    if (port_no >= CELL_PAD_MAX_PORT_NUM || !info)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    u32 info_ea = (u32)(uintptr_t)info;
    for (u32 i = 0; i < CELL_PAD_MAX_CODES; i++)
        vm_write32(info_ea + i * 4, 0);

    /* Report standard DualShock 3 capabilities */
    vm_write32(info_ea, CELL_PAD_CAPABILITY_PS3_CONFORMITY
                        | CELL_PAD_CAPABILITY_PRESS_MODE
                        | CELL_PAD_CAPABILITY_SENSOR_MODE
                        | CELL_PAD_CAPABILITY_HP_ANALOG_STICK
                        | CELL_PAD_CAPABILITY_ACTUATOR);

    PAD_UNLOCK_RETURN(CELL_OK);
}

s32 cellPadSetActDirect(u32 port_no, CellPadActParam* param)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    if (port_no >= CELL_PAD_MAX_PORT_NUM || !param)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    param = GUEST_PTR(param, CellPadActParam*);

    if (port_no >= PAD_MAX_HOST_PORTS)
        return CELL_OK;

    u16 lo, hi;
    pad_rumble_levels(param->motor[CELL_PAD_ACTUATOR_PARAM_SMALL],
                      param->motor[CELL_PAD_ACTUATOR_PARAM_LARGE], &lo, &hi);
    s_rumble[port_no].lo = lo;
    s_rumble[port_no].hi = hi;

#if PAD_BACKEND_XINPUT
    /* XInput retains vibration until the next update; no renewal is needed. */
    if (s_xinput_set_state && s_host_state[port_no].connected) {
        XINPUT_VIBRATION vib;
        vib.wLeftMotorSpeed  = (WORD)lo;
        vib.wRightMotorSpeed = (WORD)hi;
        if (p_xi_set_state) p_xi_set_state((DWORD)port_no, &vib);
    }
#endif

#if PAD_BACKEND_SDL2
    pad_lock();
    pad_sdl_rumble_send((int)port_no);
    pad_unlock();
#endif

    PAD_UNLOCK_RETURN(CELL_OK);
}

s32 cellPadClearBuf(u32 port_no)
{
    AcquireSRWLockExclusive(&s_pad_lock);
    if (!s_pad_initialized)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_NOT_OPENED);

    if (port_no >= CELL_PAD_MAX_PORT_NUM)
        PAD_UNLOCK_RETURN(CELL_PAD_ERROR_INVALID_PARAMETER);

    /* Nothing to clear in our implementation -- state is polled fresh */
    PAD_UNLOCK_RETURN(CELL_OK);
}

#undef PAD_UNLOCK_RETURN
