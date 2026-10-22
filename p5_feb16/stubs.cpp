/*
 * ps3recomp game project -- game-specific hooks and overrides.
 *
 * main.cpp is the runner and is game-agnostic. This file is where a port puts
 * everything that is not: the PRX loading hook it must answer, overrides for
 * firmware functions the HLE library gets wrong for this title, and patches
 * over recompiled code.
 *
 * It starts out as the empty version of all of that, which is what a game with
 * no lifted PRX and no overrides needs.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"
#include "../runtime/ppu/ppu_context.h"
#include "prx_loader.h"
#include "vm.h"
#include "../libs/codec/cellSail.h"
#include "../libs/input/cellMouse.h"

extern "C" int ps3_hle_has(uint32_t nid);
extern "C" uint8_t vm_read8(uint64_t ea);
extern "C" uint16_t vm_read16(uint64_t ea);
extern "C" uint32_t vm_read32(uint64_t ea);
extern "C" void ps3_hle_register_ctx(uint32_t nid, const char* name,
                                      void (*fn)(ppu_context*));

/* These EBOOT imports have no active implementation in the local HLE table.
 * Most mirror RPCS3's CELL_OK TODO behavior; special cases preserve the void
 * ABI or reproduce the return checks RPCS3 implements. Existing handlers are
 * kept. No RPCS3 implementation body is copied here. */
static uint8_t* p5_guest_bytes(uint32_t ea, uint32_t size)
{
    if (!ea || (uint64_t)ea + size > VM_TOTAL_SIZE ||
        (ppu_vm_size && (uint64_t)ea + size > ppu_vm_size))
        return nullptr;
    return vm_base ? vm_base + ea : reinterpret_cast<uint8_t*>(uintptr_t(ea));
}

/* RPCS3's counterpart is void/TODO: preserve r3, since there is no return
 * value to synthesize. Its side effects remain unknown. */
static void p5_todo_cellGcmFunc15(ppu_context* ctx)
{
    (void)ctx;
    static int warned = 0;
    if (!warned++)
        std::fprintf(stderr, "[hle] TODO _cellGcmFunc15 (NID 0x3A33C1FD); no return value\n");
}

/* RPCS3 checks the adapter state and sample limit before reporting success. */
static void p5_cellSailSoundAdapterGetFrame(ppu_context* ctx)
{
    const uint32_t self_ea = (uint32_t)ctx->gpr[3];
    const uint32_t samples = (uint32_t)ctx->gpr[4];
    uint8_t* self = p5_guest_bytes(self_ea, 2);
    int32_t rc = CELL_OK;
    if (!self) {
        rc = CELL_SAIL_ERROR_INVALID_ARGUMENT;
    } else if (!self[0] || self[1]) {
        rc = CELL_SAIL_ERROR_INVALID_STATE;
    } else if (samples > 2048) {
        rc = CELL_SAIL_ERROR_INVALID_ARGUMENT;
    }
    ctx->gpr[3] = (uint64_t)(int64_t)rc;
}

/* RPCS3 associates a valid export-library descriptor with its owning PRX.
 * Model that association for PRXs loaded by the local PRX loader; as RPCS3
 * does, an unmatchable but valid descriptor still returns CELL_OK. */
static void p5_partial_sys_prx_register_library(ppu_context* ctx)
{
    const uint32_t entry_ea = (uint32_t)ctx->gpr[3];
    uint8_t* entry = p5_guest_bytes(entry_ea, 0x1Cu);
    int32_t rc = entry ? CELL_OK : CELL_EFAULT;
    if (entry && ((((uint32_t)entry[4] << 8) | entry[5]) & 1u)) {
        const char* library_name = nullptr;
        if (prx_register_library_descriptor(entry, &library_name)) {
            std::fprintf(stderr, "[prx] registered external export library '%s'\n",
                         library_name ? library_name : "?");
        } else {
            std::fprintf(stderr, "[prx] could not associate export library at 0x%08X\n",
                         entry_ea);
        }
    }
    ctx->gpr[3] = (uint64_t)(int64_t)rc;
}

#define P5_UNIMPLEMENTED_IMPORTS(X) \
    X(cellSailGraphicsAdapterGetFrame,          0x0247C69Eu) \
    X(cellSailDescriptorSetEs,                  0x0D0C2F0Cu) \
    X(cellSailGraphicsAdapterPtsToTimePosition, 0x1872331Bu) \
    X(cellSailGraphicsAdapterUpdateAvSync,      0x44A20E79u) \
    X(cellSailPlayerReplaceEventHandler,        0x47632810u) \
    X(cellSailSoundAdapterPtsToTimePosition,    0x4AE979DFu) \
    X(cellSailSourceNotifyInputEos,             0x54C53688u) \
    X(cellSailSourceNotifyCallCompleted,        0x764EC2D2u) \
    X(cellSailSourceNotifyStreamOut,            0x95EE1695u) \
    X(cellSailSourceInitialize,                 0x9D30BDCEu) \
    X(cellSailDescriptorOpen,                   0xC044FAB1u) \
    X(cellSailSourceFinalize,                   0xEE724C99u) \
    X(cellSailSoundAdapterUpdateAvSync,         0xEEC22809u)

#define DECLARE_P5_UNIMPLEMENTED_IMPORT(name, nid) \
    static void p5_unimplemented_##name(ppu_context* ctx) \
    { \
        static int warned = 0; \
        if (!warned) { \
            warned = 1; \
            std::fprintf(stderr, "[hle] STUB %s (NID 0x%08X) -> CELL_OK; " \
                         "ABI-specific outputs are not modeled\n", #name, (unsigned)(nid)); \
        } \
        ctx->gpr[3] = (uint64_t)(int64_t)(int32_t)CELL_OK; \
    }
P5_UNIMPLEMENTED_IMPORTS(DECLARE_P5_UNIMPLEMENTED_IMPORT)
#undef DECLARE_P5_UNIMPLEMENTED_IMPORT

struct P5UnimplementedImport {
    uint32_t nid;
    const char* name;
    void (*handler)(ppu_context*);
};
#define P5_UNIMPLEMENTED_ENTRY(name, nid) { nid, #name, p5_unimplemented_##name },
static const P5UnimplementedImport kP5UnimplementedImports[] = {
    P5_UNIMPLEMENTED_IMPORTS(P5_UNIMPLEMENTED_ENTRY)
    { 0x3A33C1FDu, "_cellGcmFunc15", p5_todo_cellGcmFunc15 },
    { 0x7EB8D6B5u, "cellSailSoundAdapterGetFrame", p5_cellSailSoundAdapterGetFrame },
    { 0x42B23552u, "sys_prx_register_library", p5_partial_sys_prx_register_library }
};
#undef P5_UNIMPLEMENTED_ENTRY
#undef P5_UNIMPLEMENTED_IMPORTS

static void p5_mouse_get_data_list(ppu_context* ctx)
{
    const uint32_t port = (uint32_t)ctx->gpr[3];
    /* gfdMouse's cumulative-position path (B38C4C) starts its integration
     * from the two s16 arrays below. Supply feedback at its mouse-read call
     * so resets and edge clamps cannot offset the host and guest pointers.
     * Other callers, releases and gfdMouse's relative mode keep normal input. */
    if (ctx->lr == 0x00B38CE8u && ctx->gpr[2] == 0x011ECE08u && port < 2) {
        const uint32_t mode = vm_read32(ctx->gpr[2] - 0x6544u);
        const uint32_t x = vm_read32(ctx->gpr[2] - 0x4544u) + port * 2u;
        const uint32_t y = vm_read32(ctx->gpr[2] - 0x4540u) + port * 2u;
        if (p5_guest_bytes(mode, 9) && vm_read8(mode + 8) &&
            p5_guest_bytes(x, 2) && p5_guest_bytes(y, 2)) {
            cellMouse_syncPosition(port, (int16_t)vm_read16(x),
                                         (int16_t)vm_read16(y));
        }
    }
    ctx->gpr[3] = (uint64_t)(int64_t)cellMouseGetDataList(
        port, (CellMouseDataList*)(uintptr_t)(uint32_t)ctx->gpr[4]);
}

extern "C" void ps3_register_unimplemented_imports(void)
{
    unsigned registered = 0;
    for (const auto& entry : kP5UnimplementedImports) {
        if (ps3_hle_has(entry.nid))
            continue; /* retain any active implementation or existing handler */
        ps3_hle_register_ctx(entry.nid, entry.name, entry.handler);
        ++registered;
    }
    std::fprintf(stderr, "[hle] registered %u game-import compatibility handlers\n", registered);
    ps3_hle_register_ctx(0x6BD131F0u, "cellMouseGetDataList", p5_mouse_get_data_list);
}

/* ---------------------------------------------------------------------------
 * PRX loading
 *
 * main.cpp calls this once, after the lifted function table is registered and
 * vm_base is live, before the game runs. A title that lifts a real system PRX
 * -- libsre, which is cellSpurs and cellSync -- loads its image into guest RAM
 * here and registers its exports. Most ports never need to: the HLE library
 * answers those modules instead.
 * -----------------------------------------------------------------------*/

#ifdef P5_HAS_LIBSRE_PRX
extern "C" const prx_func_entry libsre_function_table[];
extern "C" const uint64_t libsre_function_table_count;
extern "C" const prx_export libsre_exports[];
extern "C" const uint32_t libsre_export_count;
extern "C" const prx_library_descriptor libsre_libraries[];
extern "C" const uint32_t libsre_library_count;
extern "C" void ppu_register_function(uint64_t addr,
                                      void (*fn)(ppu_context*));

static void p5_register_prx_function(uint32_t addr, void (*host)(void*))
{
    ppu_register_function(addr,
        reinterpret_cast<void (*)(ppu_context*)>(host));
}
#endif

extern "C" void ps3_load_prx_modules(void)
{
#ifdef P5_HAS_LIBSRE_PRX
    constexpr uint32_t kLibsreBase = 0x02000000u;
    const char* path = std::getenv("PS3_LIBSRE_IMAGE");
    if (!path || !*path)
        path = P5_LIBSRE_IMAGE_PATH;

    uint32_t image_size = 0;
    uint8_t* image = prx_image_load_file(path, &image_size);
    if (!image) {
        std::fprintf(stderr, "[prx] libsre unavailable; GFD will use HLE "
                             "(image: %s)\n", path);
        return;
    }

    const uint64_t image_end = (uint64_t)kLibsreBase + image_size;
    if (image_end > VM_TOTAL_SIZE) {
        std::fprintf(stderr, "[prx] libsre image exceeds guest VM: "
                             "base=0x%08X size=0x%X\n",
                     kLibsreBase, image_size);
        std::free(image);
        return;
    }

    const int32_t commit_rc = vm_commit(kLibsreBase, image_size);
    if (commit_rc != CELL_OK) {
        std::fprintf(stderr, "[prx] cannot commit libsre guest pages: "
                             "rc=0x%08X base=0x%08X size=0x%X\n",
                     (unsigned)commit_rc, kLibsreBase, image_size);
        std::free(image);
        return;
    }

    const prx_module module = {
        "cellSre_Library",
        kLibsreBase,
        image,
        image_size,
        libsre_function_table,
        libsre_function_table_count,
        libsre_exports,
        libsre_export_count,
        libsre_libraries,
        libsre_library_count
    };
    const prx_load_result result =
        prx_load_module(&module, p5_register_prx_function);
    std::free(image);
    if (!result.ok)
        std::fprintf(stderr, "[prx] libsre registration failed\n");
#endif
}

/* ---------------------------------------------------------------------------
 * Example: Override a specific NID
 *
 * The module system resolves function imports by NID.  To replace a default
 * HLE stub with custom logic, register your function at init time.
 * -----------------------------------------------------------------------*/

/*
 * Example: override cellGcmSetFlipMode (NID 0xA53D12AE) to force vsync.
 *
 *   static void my_cellGcmSetFlipMode(u32 mode)
 *   {
 *       printf("[hook] cellGcmSetFlipMode: forcing VSYNC (was %u)\n", mode);
 *       // Force vsync regardless of what the game requested
 *       cellGcmSetFlipMode(CELL_GCM_DISPLAY_VSYNC);
 *   }
 *
 * Then in your init code:
 *
 *   ps3::modules::override_nid(0xA53D12AE,
 *       reinterpret_cast<void*>(my_cellGcmSetFlipMode));
 */

/* ---------------------------------------------------------------------------
 * Example: Hook into module loading
 *
 * You can intercept cellSysmoduleLoadModule to perform custom actions
 * when a particular module is loaded.
 * -----------------------------------------------------------------------*/

/*
 *   static s32 my_loadmodule_hook(u16 id)
 *   {
 *       printf("[hook] Module 0x%04X loaded\n", id);
 *
 *       if (id == CELL_SYSMODULE_AUDIO) {
 *           // Perform custom audio backend initialization
 *           init_my_audio_backend();
 *       }
 *
 *       // Call the original implementation
 *       return cellSysmoduleLoadModule(id);
 *   }
 */

/* ---------------------------------------------------------------------------
 * Game-specific patches
 *
 * Sometimes you need to patch recompiled code at specific addresses.
 * Use the patch system to modify behavior without editing generated code.
 * -----------------------------------------------------------------------*/

/*
 *   // Skip an integrity check at guest address 0x00812FA0
 *   ps3::patches::nop_range(0x00812FA0, 0x00812FB0);
 *
 *   // Replace a function call with a direct return
 *   ps3::patches::force_return(0x009A1000, 0);  // return 0
 */
