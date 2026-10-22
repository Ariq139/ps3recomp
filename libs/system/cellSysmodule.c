/*
 * ps3recomp - cellSysmodule HLE implementation
 */

#include "cellSysmodule.h"
#include "../../runtime/syscalls/sys_memory.h"
#include <stdio.h>
#include <stdatomic.h>
#include <limits.h>

/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

/* This runtime has a static HLE surface rather than RPCS3's firmware PRXs.
 * Track load references so repeated loads and balanced unloads keep the same
 * visible state, without claiming that the firmware image was dynamically
 * loaded. */
#define SYSMODULE_INTERNAL_BASE  0xF000u
#define SYSMODULE_INTERNAL_COUNT 87u /* IDs 0xF000 through 0xF056 */
#define SYSMODULE_CORE_COUNT     0x005Du
#define SYSMODULE_INVALID_ID     0xFFFFu

static atomic_uint s_module_load_count[SYSMODULE_CORE_COUNT + SYSMODULE_INTERNAL_COUNT];
static atomic_uint s_debug_mode;
static atomic_uint s_memcontainer_id;
static atomic_uint s_use_memcontainer;

static int sysmodule_slot(u16 id)
{
    if (id < SYSMODULE_CORE_COUNT && id != 0x0058u) return (int)id;
    if ((id & 0xFF00u) == SYSMODULE_INTERNAL_BASE &&
        (id & 0x00FFu) < SYSMODULE_INTERNAL_COUNT)
        return (int)(SYSMODULE_CORE_COUNT + (id & 0x00FFu));
    return -1;
}

static int sysmodule_internal_allowed(u16 id, int extended)
{
    static const u16 regular_ids[] = {
        0xF00A, 0xF010, 0xF019, 0xF01B, 0xF01D, 0xF01E,
        0xF023, 0xF028, 0xF029, 0xF02A, 0xF02B, 0xF02C,
        0xF02E, 0xF02F, 0xF030, 0xF034, 0xF035, 0xF044, 0xF054
    };
    static const u16 extended_ids[] = {
        0xF005, 0xF00A, 0xF010, 0xF013, 0xF019, 0xF01B, 0xF01D, 0xF01E,
        0xF020, 0xF021, 0xF022, 0xF023, 0xF024, 0xF026, 0xF028, 0xF029,
        0xF02A, 0xF02B, 0xF02C, 0xF02E, 0xF02F, 0xF030, 0xF032, 0xF033,
        0xF034, 0xF035, 0xF036, 0xF039, 0xF03A, 0xF03D, 0xF03E, 0xF03F,
        0xF040, 0xF041, 0xF043, 0xF044, 0xF045, 0xF046, 0xF047, 0xF048,
        0xF049, 0xF04A, 0xF04D, 0xF04E, 0xF04F, 0xF050, 0xF051, 0xF052,
        0xF054, 0xF055, 0xF056
    };
    const u16* ids = extended ? extended_ids : regular_ids;
    size_t count = extended
        ? sizeof(extended_ids) / sizeof(extended_ids[0])
        : sizeof(regular_ids) / sizeof(regular_ids[0]);
    for (size_t i = 0; i < count; ++i)
        if (ids[i] == id) return 1;
    return 0;
}

static s32 sysmodule_ref(u16 id, int increment)
{
    int slot = sysmodule_slot(id);
    if (slot < 0) return CELL_SYSMODULE_ERROR_UNKNOWN;

    unsigned count = atomic_load_explicit(&s_module_load_count[slot], memory_order_relaxed);
    if (increment) {
        do {
            if (count >= INT_MAX) return CELL_SYSMODULE_ERROR_FATAL;
        } while (!atomic_compare_exchange_weak_explicit(
            &s_module_load_count[slot], &count, count + 1,
            memory_order_relaxed, memory_order_relaxed));
        return CELL_OK;
    }

    while (count != 0) {
        if (atomic_compare_exchange_weak_explicit(
                &s_module_load_count[slot], &count, count - 1,
                memory_order_relaxed, memory_order_relaxed))
            return CELL_OK;
    }
    return CELL_SYSMODULE_ERROR_UNLOADED;
}

static s32 sysmodule_loaded(u16 id)
{
    int slot = sysmodule_slot(id);
    if (slot < 0) return CELL_SYSMODULE_ERROR_UNKNOWN;
    return atomic_load_explicit(&s_module_load_count[slot], memory_order_relaxed)
        ? CELL_OK : CELL_SYSMODULE_ERROR_UNLOADED;
}

static const char* sysmodule_id_to_name(u16 id)
{
    switch (id) {
    case CELL_SYSMODULE_NET:           return "CELL_SYSMODULE_NET";
    case CELL_SYSMODULE_HTTP:          return "CELL_SYSMODULE_HTTP";
    case CELL_SYSMODULE_HTTP_UTIL:     return "CELL_SYSMODULE_HTTP_UTIL";
    case CELL_SYSMODULE_SSL:           return "CELL_SYSMODULE_SSL";
    case CELL_SYSMODULE_HTTPS:         return "CELL_SYSMODULE_HTTPS";
    case CELL_SYSMODULE_VDEC:          return "CELL_SYSMODULE_VDEC";
    case CELL_SYSMODULE_ADEC:          return "CELL_SYSMODULE_ADEC";
    case CELL_SYSMODULE_DMUX:          return "CELL_SYSMODULE_DMUX";
    case CELL_SYSMODULE_VPOST:         return "CELL_SYSMODULE_VPOST";
    case CELL_SYSMODULE_RTC:           return "CELL_SYSMODULE_RTC";
    case CELL_SYSMODULE_SPURS:         return "CELL_SYSMODULE_SPURS";
    case CELL_SYSMODULE_OVIS:          return "CELL_SYSMODULE_OVIS";
    case CELL_SYSMODULE_SHEAP:         return "CELL_SYSMODULE_SHEAP";
    case CELL_SYSMODULE_SYNC:          return "CELL_SYSMODULE_SYNC";
    case CELL_SYSMODULE_SYNC2:         return "CELL_SYSMODULE_SYNC2";
    case CELL_SYSMODULE_FS:            return "CELL_SYSMODULE_FS";
    case CELL_SYSMODULE_JPGDEC:        return "CELL_SYSMODULE_JPGDEC";
    case CELL_SYSMODULE_GCM_SYS:       return "CELL_SYSMODULE_GCM_SYS";
    case CELL_SYSMODULE_AUDIO:         return "CELL_SYSMODULE_AUDIO";
    case CELL_SYSMODULE_PAMF:          return "CELL_SYSMODULE_PAMF";
    case CELL_SYSMODULE_ATRAC3PLUS:    return "CELL_SYSMODULE_ATRAC3PLUS";
    case CELL_SYSMODULE_NETCTL:        return "CELL_SYSMODULE_NETCTL";
    case CELL_SYSMODULE_SYSUTIL:       return "CELL_SYSMODULE_SYSUTIL";
    case CELL_SYSMODULE_SYSUTIL_NP:    return "CELL_SYSMODULE_SYSUTIL_NP";
    case CELL_SYSMODULE_IO:            return "CELL_SYSMODULE_IO";
    case CELL_SYSMODULE_PNGDEC:        return "CELL_SYSMODULE_PNGDEC";
    case CELL_SYSMODULE_FONT:          return "CELL_SYSMODULE_FONT";
    case CELL_SYSMODULE_FREETYPE:      return "CELL_SYSMODULE_FREETYPE";
    case CELL_SYSMODULE_USBD:          return "CELL_SYSMODULE_USBD";
    case CELL_SYSMODULE_SAIL:          return "CELL_SYSMODULE_SAIL";
    case CELL_SYSMODULE_L10N:          return "CELL_SYSMODULE_L10N";
    case CELL_SYSMODULE_RESC:          return "CELL_SYSMODULE_RESC";
    case CELL_SYSMODULE_SYSUTIL_SAVEDATA: return "CELL_SYSMODULE_SYSUTIL_SAVEDATA";
    case CELL_SYSMODULE_SYSUTIL_GAME:  return "CELL_SYSMODULE_SYSUTIL_GAME";
    case CELL_SYSMODULE_FIBER:         return "CELL_SYSMODULE_FIBER";
    case CELL_SYSMODULE_GEM:           return "CELL_SYSMODULE_GEM";
    case CELL_SYSMODULE_CAMERA:        return "CELL_SYSMODULE_CAMERA";
    case CELL_SYSMODULE_LV2DBG:        return "CELL_SYSMODULE_LV2DBG";
    case CELL_SYSMODULE_SYSUTIL_USERINFO: return "CELL_SYSMODULE_SYSUTIL_USERINFO";
    case CELL_SYSMODULE_SYSUTIL_NP2:   return "CELL_SYSMODULE_SYSUTIL_NP2";
    case CELL_SYSMODULE_SYSUTIL_GAME_EXEC: return "CELL_SYSMODULE_SYSUTIL_GAME_EXEC";
    case CELL_SYSMODULE_SYSUTIL_NP_TROPHY: return "CELL_SYSMODULE_SYSUTIL_NP_TROPHY";
    case CELL_SYSMODULE_MIC:           return "CELL_SYSMODULE_MIC";
    case CELL_SYSMODULE_FONTFT:        return "CELL_SYSMODULE_FONTFT";
    case CELL_SYSMODULE_AVCONF_EXT:    return "CELL_SYSMODULE_AVCONF_EXT";
    case CELL_SYSMODULE_USBPSPCM:      return "CELL_SYSMODULE_USBPSPCM";
    case CELL_SYSMODULE_SPURS_JQ:      return "CELL_SYSMODULE_SPURS_JQ";
    case CELL_SYSMODULE_SYSUTIL_NP_TUS: return "CELL_SYSMODULE_SYSUTIL_NP_TUS";
    default:                           return "UNKNOWN";
    }
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

/* NID: 0x26A6E12B */
/* NID: 0x63FF6FF9 -- library init/teardown are no-ops in HLE. */
s32 cellSysmoduleInitialize(void) { return CELL_OK; }
s32 cellSysmoduleFinalize(void)   { return CELL_OK; }

s32 cellSysmoduleLoadModule(u16 id)
{
    printf("[cellSysmodule] LoadModule(id=0x%04X '%s')\n",
           id, sysmodule_id_to_name(id));

    if (id >= SYSMODULE_INTERNAL_BASE && !sysmodule_internal_allowed(id, 0))
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_ref(id, 1);
}

/* NID: 0x112A5EE9 */
s32 cellSysmoduleUnloadModule(u16 id)
{
    printf("[cellSysmodule] UnloadModule(id=0x%04X '%s')\n",
           id, sysmodule_id_to_name(id));

    if (id >= SYSMODULE_INTERNAL_BASE && !sysmodule_internal_allowed(id, 0))
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_ref(id, 0);
}

/* NID: 0x5A59E258 */
s32 cellSysmoduleIsLoaded(u16 id)
{
    printf("[cellSysmodule] IsLoaded(id=0x%04X '%s')\n",
           id, sysmodule_id_to_name(id));

    if (id >= SYSMODULE_INTERNAL_BASE && !sysmodule_internal_allowed(id, 0))
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_loaded(id);
}

/* RPCS3_REFERENCE_REVIEWED: the internal-ID range and loadable sets are ABI
 * facts transcribed into local tables; state is tracked through the static
 * HLE model, with no RPCS3 function body copied. */
s32 cellSysmoduleLoadModuleInternal(u16 id)
{
    if (id == SYSMODULE_INVALID_ID || id < SYSMODULE_INTERNAL_BASE ||
        id >= SYSMODULE_INTERNAL_BASE + SYSMODULE_INTERNAL_COUNT)
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_ref(id, 1);
}

s32 cellSysmoduleUnloadModuleInternal(u16 id)
{
    if (id == SYSMODULE_INVALID_ID || id < SYSMODULE_INTERNAL_BASE ||
        id >= SYSMODULE_INTERNAL_BASE + SYSMODULE_INTERNAL_COUNT)
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_ref(id, 0);
}

s32 cellSysmoduleSetMemcontainer(u32 container_id)
{
    atomic_store_explicit(&s_use_memcontainer, 0, memory_order_relaxed);
    atomic_store_explicit(&s_memcontainer_id, UINT32_MAX, memory_order_relaxed);
    if (container_id == UINT32_MAX) {
        return CELL_OK;
    }

    if (container_id == 0 || container_id > SYS_MEMORY_CONTAINER_MAX)
        return CELL_SYSMODULE_ERROR_INVALID_MEMCONTAINER;

    const sys_mem_container_info* container = &g_sys_mem_containers[container_id - 1];
    if (!container->active || container->total_size < container->used_size ||
        container->total_size - container->used_size < 0x60000u) {
        atomic_store_explicit(&s_use_memcontainer, 0, memory_order_relaxed);
        return CELL_SYSMODULE_ERROR_INVALID_MEMCONTAINER;
    }

    atomic_store_explicit(&s_memcontainer_id, container_id, memory_order_relaxed);
    atomic_store_explicit(&s_use_memcontainer, 1, memory_order_relaxed);
    return CELL_OK;
}

s32 cellSysmoduleSetDebugmode(u32 debug_mode)
{
    atomic_store_explicit(&s_debug_mode, debug_mode == 1, memory_order_relaxed);
    return CELL_OK;
}

s32 cellSysmoduleSetInternalmode(void)
{
    return CELL_OK;
}

/* NID: 0x70E319B5. Extended-ID query only; shares the existing HLE module
 * tracker and does not load firmware or imply that a module's APIs exist.
 * The internal namespace is 0xF000..0xF056 (87 module IDs).
 * ABI reference: RPCS3 rpcs3/Emu/Cell/Modules/cellSysmodule.cpp,
 * cellSysmoduleIsLoadedEx and INTERNAL_MODULES_COUNT. */
s32 cellSysmoduleIsLoadedEx(u16 id)
{
    if (id < 0xF000 || id >= 0xF057)
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_loaded(id);
}

/* RPCS3 LOADABLE_INTERNAL_MODULES_EX policy shared by the Ex entry points.
 * The permission-dependent IDs are valid for unload; only loading them needs
 * the process-PARAM.SFO permission check that this HLE does not yet provide. */
static int sysmodule_ex_permitted(u16 id)
{
    switch (id) {
    case 0xF005: case 0xF00A:
    case 0xF010: case 0xF013: case 0xF019: case 0xF01B:
    case 0xF01D: case 0xF01E:
    case 0xF021: case 0xF022: case 0xF023: case 0xF026:
    case 0xF028: case 0xF029: case 0xF02A: case 0xF02B:
    case 0xF02C: case 0xF02E: case 0xF02F:
    case 0xF030: case 0xF032: case 0xF033: case 0xF034:
    case 0xF035: case 0xF036: case 0xF039: case 0xF03A:
    case 0xF03D: case 0xF03F:
    case 0xF040: case 0xF041: case 0xF043: case 0xF044:
    case 0xF045: case 0xF046: case 0xF047: case 0xF048:
    case 0xF049: case 0xF04A: case 0xF04D: case 0xF04E: case 0xF04F:
    case 0xF050: case 0xF051: case 0xF052: case 0xF054:
    case 0xF055: case 0xF056:
    case 0xF020: case 0xF024: case 0xF03E:
        return 1;
    default:
        return 0;
    }
}

/* NID: 0x1A267F98. Preserve the existing HLE tracker's idempotent load
 * semantics. This records module state; it does not load firmware. */
s32 cellSysmoduleLoadModuleEx(u16 id)
{
    if (!sysmodule_ex_permitted(id) || id == 0xF020 || id == 0xF024 || id == 0xF03E)
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_ref(id, 1);
}

/* NID: 0xFD0E1CAC. One u16 argument; extra PPU GPRs are not output pointers.
 * Reference: RPCS3 cellSysmodule.cpp, cellSysmoduleUnloadModuleEx and
 * LOADABLE_INTERNAL_MODULES_EX. Reuse the HLE tracker and preserve its
 * UNKNOWN / UNLOADED errors. No PRX unloading or module finalizer is implied. */
s32 cellSysmoduleUnloadModuleEx(u16 id)
{
    if (!sysmodule_ex_permitted(id))
        return CELL_SYSMODULE_ERROR_UNKNOWN;
    return sysmodule_ref(id, 0);
}
