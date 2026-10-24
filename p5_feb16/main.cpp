/*
 * ps3recomp game project -- entry point template.
 *
 * This is the runner. It boots a lifted title through the toolkit's PPU
 * scaffold: allocate the flat guest VM, load the PPU ELF named on the command
 * line, register the lifted function table and the HLE NID handlers, start a
 * frame clock, and dispatch the entry OPD.
 *
 * The scaffold does the work; nothing below reimplements it. ppu_load_elf,
 * ppu_recomp_register, ppu_hle_init, ppu_sysprx_register, ppu_fs_register,
 * lv2_init_syscalls and ppu_run all come from the toolkit, compiled into this
 * project from PS3RECOMP_DIR (see CMakeLists.txt). What is left here is the
 * part a port owns: which backend to present through, how the frame clock is
 * paced, and whatever diagnostics the title turns out to need.
 *
 * runtime/ppu/tests/boot_main.cpp in the toolkit is the same boot in its
 * fully-instrumented form -- crash filter, hang watchdog, guest-PC sampling
 * profiler. Read it when a boot goes wrong; copy from it what the title
 * needs. It is deliberately not what a fresh project starts with.
 *
 * The one part that IS duplicated from it is the frame clock below, because a
 * port is expected to change its pacing and a shared one would be the wrong
 * shape for that. It was copied faithfully, comments and all. If a pacing bug
 * is fixed in one of the two, look at the other.
 *
 * This builds on Windows, macOS and Linux. CI builds and runs it against the
 * boot smoke title on the last two, and compile-checks it against clang-cl on
 * the first (tools/check_ppu_scaffold.py).
 */

/* The lifter's generated header, which declares ppu_context and the function
 * table. It comes first: everything below is written against that struct. */
#include "ppu_recomp.h"
#include "../runtime/cache/boot_program_dump.h"
#include "../runtime/ps3_console.h"

/* The Win32 names -- Sleep, GetTickCount64, CreateThread, InterlockedIncrement,
 * VirtualAlloc, the scalar typedefs -- from one place. On Windows this is a
 * passthrough to <windows.h>; off Windows the toolkit's shim supplies the same
 * names over pthreads and mmap, so the call sites below are written once. */
#include "win32_compat.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
/* timeBeginPeriod: <windows.h> arrives with WIN32_LEAN_AND_MEAN set, which
 * excludes the multimedia timer API, so it is asked for by name -- and after
 * windows.h, since timeapi.h uses UINT. No shim exists off Windows because
 * there is nothing to shim: the 15.6 ms default timer granularity this works
 * around is a Windows problem. */
#include <timeapi.h>
#endif

/* ---------------------------------------------------------------------------
 * What the scaffold gives this file
 * -----------------------------------------------------------------------*/
extern "C" {
uint32_t ppu_load_elf(const char* path);      /* ELF -> guest RAM, returns entry OPD */
uint32_t ppu_load_elf_image(const uint8_t* image, size_t size, const char* name);
void     ppu_recomp_register(void);           /* generated: lifted table -> address map */
void     ppu_hle_init(void);                  /* firmware import NID -> HLE handler */
void     ppu_sysprx_register(void);           /* boot-critical CRT (sys_initialize_tls, ...) */
void     ppu_fs_register(void);               /* cellFs over the real game directory */
void     lv2_init_syscalls(void);             /* the lv2 syscall table */
int      ppu_run(uint32_t entry_opd, uint32_t stack_top);

/* This project's own: load whatever real system PRX modules the title needs
 * into guest RAM and register their exports. stubs.cpp has the empty version
 * a game with no lifted PRX wants. */
void     ps3_load_prx_modules(void);
void     ps3_register_unimplemented_imports(void);

extern const char* ppu_vfs_root;              /* host dir the PS3 mount points map into */
extern const char* ppu_hdd0_root;             /* optional host root for /dev_hdd0 */
void cellGame_init_from_paramsfo(const char* sfo_path);

/* The status the guest handed to sys_process_exit. A title that exits that way
 * never returns through ppu_run, so this is only read on the path where the
 * entry function unwound instead. */
extern int     g_sys_process_exit_called;
extern int32_t g_sys_process_exit_code;

/* Host-provided symbols the runtime and the HLE libraries link against. */
uint8_t* vm_base = nullptr;
extern uint32_t ppu_vm_size;                  /* ppu_loader.cpp: the OOB guard */

/* Guest-callback dispatch. g_ps3_guest_caller is the hook the HLE runtime
 * calls back into recompiled code through -- cellSysutil events and the GCM
 * vblank/flip handlers. ppu_guest_call does the OPD -> dispatch. */
typedef void (*ps3_guest_caller_fn)(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t, uint64_t);
extern ps3_guest_caller_fn g_ps3_guest_caller;
uint64_t ppu_guest_call(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t,
                        uint64_t, uint64_t, uint64_t, uint64_t);

/* The GCM half of the frame clock. */
void     cellGcmTickVBlank(void);
void     cellGcmTickFlip(void);
int      cellGcm_take_flip_pending(void);
void     cellGcm_rsx_process_fifo(void);      /* drain get -> put */
void     cellGcm_fifo_kick_wait(unsigned ms); /* wake immediately for a dry fence queue */
unsigned cellGcm_flip_request_count(void);
}

/* ---------------------------------------------------------------------------
 * RSX present backend
 * -----------------------------------------------------------------------*/
/* D3D12 on Windows, Metal on Apple, and the null backend's headless software
 * path anywhere else -- the same selection runtime/host/host_posix.c and the
 * toolkit's boot harness make. All three expose the same three entry points,
 * so the frame clock below is backend-agnostic. */
#if defined(_WIN32)
extern "C" int  rsx_d3d12_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_d3d12_backend_present(void);
extern "C" int  rsx_d3d12_backend_pump_messages(void);
#  define rsx_backend_init    rsx_d3d12_backend_init
#  define rsx_backend_present rsx_d3d12_backend_present
#  define rsx_backend_pump    rsx_d3d12_backend_pump_messages
#  define RSX_BACKEND_NAME    "D3D12"
#elif defined(__APPLE__)
extern "C" int  rsx_metal_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_metal_backend_present(void);
extern "C" int  rsx_metal_backend_pump_messages(void);
#  define rsx_backend_init    rsx_metal_backend_init
#  define rsx_backend_present rsx_metal_backend_present
#  define rsx_backend_pump    rsx_metal_backend_pump_messages
#  define RSX_BACKEND_NAME    "Metal"
#else
extern "C" int  rsx_null_backend_init(uint32_t w, uint32_t h, const char* title);
extern "C" void rsx_null_backend_present(void);
extern "C" int  rsx_null_backend_pump_messages(void);
#  define rsx_backend_init    rsx_null_backend_init
#  define rsx_backend_present rsx_null_backend_present
#  define rsx_backend_pump    rsx_null_backend_pump_messages
#  define RSX_BACKEND_NAME    "null (headless software)"
#endif

/* ---------------------------------------------------------------------------
 * Configuration
 * -----------------------------------------------------------------------*/
/* The flat VM must span every region the PS3 memory map uses, not just the
 * game image: sys_ppu_thread_create puts thread stacks at 0xD0000000, so a
 * smaller arena makes every spawned thread's stack access out of bounds. */
#define VM_SIZE       0x100010000ull   /* the full 32-bit guest space + a 64K guard */
#define STACK_TOP     0x0FF00000u      /* main-thread stack, below the 0x10000000 segment */
#define WINDOW_WIDTH  1280
#define WINDOW_HEIGHT 720

/* ---------------------------------------------------------------------------
 * Frame clock
 * -----------------------------------------------------------------------*/
/* On real hardware the RSX raises a vblank interrupt about 60 times a second
 * and that is what drives the game's frame loop. With no RSX it is synthesized
 * here: a host thread calls cellGcmTickVBlank/TickFlip, which invoke the
 * handlers the guest registered, and drains the GCM FIFO. Without this the
 * title initialises, registers its handlers and then waits forever.
 *
 * The ticks are driven off real elapsed time rather than off how long present()
 * took. A hidden or occluded window makes present block hard, and pacing the
 * ticks behind it paces the whole game behind it. */
static volatile LONG g_frames_presented = 0;
static volatile LONG g_frame_phase = 0;
static DWORD WINAPI frame_probe(LPVOID)
{
    for (unsigned i = 0; i < 15; ++i) {
        Sleep(2000);
        fprintf(stderr, "[frame-probe] phase=%ld guest-presents=%ld\n",
                g_frame_phase, g_frames_presented);
    }
    return 0;
}

/* Frames handed to the backend at a guest FLIP boundary -- one per frame the
 * guest actually finished. The presents made before the guest's first flip, so
 * a fresh window is not left blank through a long boot, carry no guest frame
 * and are deliberately not counted. */
extern "C" unsigned ppu_boot_frames_presented(void)
{
    return (unsigned)g_frames_presented;
}

static void present_guest_frame(void)
{
    g_frame_phase = 3;
    rsx_backend_present();
    /* This thread increments and guest threads read, so interlocked rather
     * than a volatile ++, which on arm64 is neither atomic nor a fence. */
    InterlockedIncrement(&g_frames_presented);
}

static uint32_t window_dimension_from_environment(const char* name,
                                                 uint32_t fallback,
                                                 uint32_t minimum,
                                                 uint32_t maximum)
{
    const char* value = getenv(name);
    if (!value || !*value) return fallback;

    char* end = nullptr;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum)
        return fallback;
    return (uint32_t)parsed;
}

static DWORD WINAPI frame_clock(LPVOID)
{
    const char* title = getenv("PS3_TITLE");
    if (!title || !*title) title = "ps3recomp";

    const uint32_t window_width = window_dimension_from_environment("RSX_W", WINDOW_WIDTH, 320, 7680);
    const uint32_t window_height = window_dimension_from_environment("RSX_H", WINDOW_HEIGHT, 240, 4320);
    printf("[rsx] window size: %ux%u\n", window_width, window_height);
    int rsx_ok = (rsx_backend_init(window_width, window_height, title) == 0);
    fprintf(stderr, "[rsx] %s backend init %s\n", RSX_BACKEND_NAME,
            rsx_ok ? "OK -- window open" : "FAILED");

    unsigned  last_flip = 0;
    ULONGLONG next_tick = GetTickCount64();

    for (;;) {
        cellGcm_fifo_kick_wait(4);
        ULONGLONG now = GetTickCount64();

        int fired = 0;
        while ((long long)(now - next_tick) >= 0 && fired < 240) {
            g_frame_phase = 1;
            cellGcmTickVBlank();
            g_frame_phase = 2;
            cellGcmTickFlip();
            /* Present a pending flip BEFORE draining any further. The flip
             * fires at a get==put frame boundary on the guest thread, so the
             * batch held right now is exactly the completed frame; presenting
             * after the drain races the guest's next-frame writes and shows a
             * mixed one. */
            if (rsx_ok && cellGcm_take_flip_pending()) {
                present_guest_frame();
                last_flip = cellGcm_flip_request_count();
            }
            /* Drain the FIFO every tick. This is what writes the RSX sync-fence
             * labels the game's per-frame logic blocks on, so it has to keep
             * advancing at 60 Hz even while present() throttles. */
            if (rsx_ok) { g_frame_phase = 4; cellGcm_rsx_process_fifo(); }
            next_tick += 16;             /* ~60 Hz */
            fired++;
        }
        if (fired >= 240) next_tick = now;   /* fell too far behind -- resync */

        /* Drain at the outer cadence too, not only on the 16 ms tick: titles
         * fence every render pass on a label the drain writes, and one wave of
         * those at 16 ms apiece paces the guest into single-figure frame rates.
         * The real RSX writes them in microseconds. */
        if (rsx_ok) {
            if (cellGcm_take_flip_pending()) {
                present_guest_frame();
                last_flip = cellGcm_flip_request_count();
            }
            g_frame_phase = 5;
            cellGcm_rsx_process_fifo();

            g_frame_phase = 6;
            if (rsx_backend_pump() != 0) {
                rsx_ok = 0;              /* window closed */
                continue;
            }
            /* Present on a guest flip. A present on a fixed clock can catch the
             * drain mid-frame and flash a partial one. Before the first flip
             * present freely, so the window is not blank during boot. */
            unsigned fc = cellGcm_flip_request_count();
            if (fc != last_flip) {
                present_guest_frame();
                last_flip = fc;
            } else if (fc == 0) {
                g_frame_phase = 3;
                rsx_backend_present();
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * VFS root
 * -----------------------------------------------------------------------*/
/* Derive the directory holding PS3_GAME from the EBOOT path
 * <root>/PS3_GAME/USRDIR/EBOOT.elf -> <root>. $PS3_VFS_ROOT overrides. */
static char s_vfs_root[1024];
static char s_hdd0_root[1024];

static void derive_vfs_root(const char* eboot)
{
    const char* env = getenv("PS3_VFS_ROOT");
    if (env && *env) { ppu_vfs_root = env; return; }
    if (!eboot || !*eboot) { ppu_vfs_root = "."; return; }

    strncpy(s_vfs_root, eboot, sizeof s_vfs_root - 1);
    for (char* p = s_vfs_root; *p; p++) if (*p == '\\') *p = '/';
    /* strip EBOOT.elf, USRDIR and PS3_GAME */
    for (int i = 0; i < 3; i++) { char* s = strrchr(s_vfs_root, '/'); if (s) *s = 0; }
    if (!s_vfs_root[0]) strcpy(s_vfs_root, ".");
    ppu_vfs_root = s_vfs_root;
}

static int path_chars_equal_ci(const char* a, const char* b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        char ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return 1;
}

/* If the disc root itself lives under RPCS3's dev_hdd0/game/<title> layout,
 * map /dev_hdd0 to its real parent. Otherwise preserve the configured root or
 * the VFS-root fallback used by standalone disc dumps. */
static void derive_hdd0_root(void)
{
    const char* configured = getenv("PS3_HDD0_ROOT");
    if (configured && *configured) {
        ppu_hdd0_root = configured;
        return;
    }

    char normalized[1024];
    strncpy(normalized, ppu_vfs_root ? ppu_vfs_root : "", sizeof normalized - 1);
    normalized[sizeof normalized - 1] = '\0';
    for (char* p = normalized; *p; p++) if (*p == '\\') *p = '/';

    static const char marker[] = "dev_hdd0/game/";
    for (char* p = normalized; *p; p++) {
        if (p != normalized && p[-1] != '/') continue;
        if (strlen(p) < sizeof marker - 1) break;
        if (!path_chars_equal_ci(p, marker, sizeof marker - 1)) continue;
        size_t root_len = (size_t)(p - normalized) + sizeof("dev_hdd0") - 1;
        if (root_len >= sizeof s_hdd0_root) break;
        memcpy(s_hdd0_root, normalized, root_len);
        s_hdd0_root[root_len] = '\0';
        ppu_hdd0_root = s_hdd0_root;
        printf("[boot] /dev_hdd0 root: %s (derived from disc path)\n", s_hdd0_root);
        return;
    }
    ppu_hdd0_root = nullptr;
}

static void init_cell_game_paramsfo(void)
{
    static const char* const layouts[] = {
        "%s/PS3_GAME/PARAM.SFO", /* disc root */
        "%s/../PARAM.SFO",       /* USRDIR root */
        "%s/PARAM.SFO",          /* title directory */
    };
    char sfo[1100];
    int found = 0;
    for (size_t i = 0; i < sizeof layouts / sizeof layouts[0]; i++) {
        snprintf(sfo, sizeof sfo, layouts[i], ppu_vfs_root);
        FILE* f = fopen(sfo, "rb");
        if (f) { fclose(f); found = 1; break; }
    }
    if (!found)
        snprintf(sfo, sizeof sfo, "%s/PS3_GAME/PARAM.SFO", ppu_vfs_root);
    cellGame_init_from_paramsfo(sfo);
}

/* ---------------------------------------------------------------------------
 * Guest VM
 * -----------------------------------------------------------------------*/
#ifdef _WIN32
/* Demand-paging for the flat VM: reserve the whole 4 GB guest space, which
 * costs no commit, and commit each 64 KB page on first access. Every 32-bit
 * guest offset is then valid, so a garbage guest pointer reads zero instead of
 * killing the process. Faults outside the arena fall through untouched. */
static LONG WINAPI vm_commit_veh(EXCEPTION_POINTERS* ep)
{
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        ULONG_PTR fault = ep->ExceptionRecord->ExceptionInformation[1];
        uintptr_t base  = (uintptr_t)vm_base;
        if (vm_base && fault >= base && fault < base + VM_SIZE) {
            void* page = (void*)(fault & ~(uintptr_t)0xFFFF);
            if (VirtualAlloc(page, 0x10000, MEM_COMMIT, PAGE_READWRITE))
                return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static bool alloc_guest_vm(void)
{
#ifdef _WIN32
    AddVectoredExceptionHandler(1, vm_commit_veh);
    vm_base = (uint8_t*)VirtualAlloc(NULL, VM_SIZE, MEM_RESERVE, PAGE_READWRITE);
    ppu_vm_size = 0;              /* the whole space is backed; no OOB guard needed */
#else
    /* No vectored exception handlers off Windows, so the arena is committed up
     * front and lazily backed by the OS: only pages the title touches cost
     * anything. It stops below the 0xE0000000 mark, so ppu_vm_size arms the
     * loader's bounds check for what is left. */
    vm_base = (uint8_t*)calloc(1, 0xE0000000u);
    ppu_vm_size = 0xE0000000u;
#endif
    return vm_base != nullptr;
}

/* ---------------------------------------------------------------------------
 * Main
 * -----------------------------------------------------------------------*/
static void harness_guest_caller(uint32_t opd, uint64_t a0, uint64_t a1,
                                 uint64_t a2, uint64_t a3, uint64_t a4,
                                 uint64_t a5, uint64_t a6, uint64_t a7)
{
    ppu_guest_call(opd, a0, a1, a2, a3, a4, a5, a6, a7);
}

#ifdef PS3RECOMP_EMBED_PPU_ELF
static uint32_t ppu_load_embedded_elf(void)
{
    HRSRC resource = FindResourceA(NULL, "PS3RECOMP_BOOT_PPU_ELF", RT_RCDATA);
    if (!resource) {
        fprintf(stderr, "[ppu] embedded boot image resource is missing\n");
        return 0;
    }
    HGLOBAL loaded = LoadResource(NULL, resource);
    const uint8_t* image = (const uint8_t*)LockResource(loaded);
    DWORD size = SizeofResource(NULL, resource);
    if (!image || !size) {
        fprintf(stderr, "[ppu] embedded boot image resource is empty\n");
        return 0;
    }
    return ppu_load_elf_image(image, (size_t)size, "embedded PPU ELF");
}
#endif

int main(int argc, char** argv)
{
    const char* elf_path = (argc >= 2) ? argv[1] : NULL;
    if (!elf_path || !*elf_path) elf_path = getenv("PS3_PPU_ELF");
#ifdef PS3RECOMP_DEFAULT_PPU_ELF
    if (!elf_path || !*elf_path) elf_path = PS3RECOMP_DEFAULT_PPU_ELF;
#endif
#if !defined(PS3RECOMP_EMBED_PPU_ELF)
    if (!elf_path || !*elf_path) {
        printf("usage: %s [PPU ELF] (or set PS3_PPU_ELF / configure PORT_PPU_ELF)\n", argv[0]);
        return 2;
    }
#endif

    ps3_console_init();

#ifdef _WIN32
#pragma comment(lib, "winmm.lib")
    /* 1 ms timer resolution. The default granularity is about 15.6 ms, which
     * inflates every shorter wait the title makes and throttles the whole
     * thing. POSIX timers are already fine-grained. */
    timeBeginPeriod(1);
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: do not lose prints on a kill */
#endif

    printf("=== ps3recomp game runner ===\n");

    if (!alloc_guest_vm()) {
        fprintf(stderr, "ERROR: could not allocate the guest address space\n");
        ps3_console_shutdown();
        return 1;
    }

    uint32_t entry = 0;
    if (elf_path && *elf_path) entry = ppu_load_elf(elf_path);
#ifdef PS3RECOMP_EMBED_PPU_ELF
    else entry = ppu_load_embedded_elf();
#endif
    if (!entry) {
        fprintf(stderr, "ERROR: could not load %s\n",
                (elf_path && *elf_path) ? elf_path : "embedded PPU ELF");
        ps3_console_shutdown();
        return 1;
    }

    derive_vfs_root(elf_path);
    printf("[boot] VFS root: %s\n", ppu_vfs_root);
    derive_hdd0_root();
    init_cell_game_paramsfo();
    ps3_boot_collect_programs(ppu_vfs_root, elf_path, argv[0]);

    ppu_recomp_register();   /* lifted function table -> address map */
    ps3_load_prx_modules();  /* this game's lifted system PRX, if it has any */
    ppu_hle_init();          /* firmware import NID -> HLE handlers */
    ppu_sysprx_register();   /* boot-critical CRT */
    ppu_fs_register();       /* cellFs over the game directory */
    lv2_init_syscalls();     /* the lv2 syscall table */
    ps3_register_unimplemented_imports(); /* explicit failures for unresolved EBOOT imports */

    /* Install the guest-callback hook, then start the frame clock. It no-ops
     * until the title registers its vblank and flip handlers during init. */
    g_ps3_guest_caller = harness_guest_caller;
    CreateThread(NULL, 4u * 1024 * 1024, frame_clock, NULL, 0, NULL);
    if (getenv("GFD_FRAME_TRACE")) CreateThread(NULL, 0, frame_probe, NULL, 0, NULL);

    printf("\n[boot] dispatching entry OPD 0x%08X (stack top 0x%08X)\n\n",
           entry, STACK_TOP);

    /* The profiler is enabled only when PS3_SAMPLE is set. Start it on the
     * boot host thread so PS3_SAMPLE_MAIN selects the game's main thread. */
    extern void ps3_sampler_start(void);
    ps3_sampler_start();
    int rc = ppu_run(entry, STACK_TOP);
    printf("\n[boot] ppu_run returned %d (entry function unwound)\n", rc);

    /* A title that called sys_process_exit never reaches this line: that path
     * ends in the host exit(). Getting here means the entry function returned
     * instead, so hand back the status the guest published if it published
     * one. Returning a hardcoded 0 would report success for a run that never
     * got anywhere. */
    const int exit_code = g_sys_process_exit_called ? (int)g_sys_process_exit_code : 0;
    ps3_console_shutdown();
    return exit_code;
}
