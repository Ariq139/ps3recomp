#include "event_editor_june_api.h"
#include "generated/june_image_info.h"
#include "cellGcmSys.h"
#include "win32_compat.h"
#include "../../runtime/cpk_event_fallback.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

#undef vm_read8
#undef vm_read16
#undef vm_read32
#undef vm_read64
#undef vm_write8
#undef vm_write16
#undef vm_write32
#undef vm_write64
#undef ps3_indirect_call
#undef ps3_indirect_call_to
#undef ps3_hle_call
#undef ppu_stwcx32

struct JuneDataMapEntry {
    uint32_t source;
    uint32_t target;
};

#include "june_shared_data_map.inc"
#include "generated/june_static_data_refs.inc"

extern "C" {
int32_t ps3_guest_memory_alloc(uint32_t size, uint32_t alignment, uint32_t* out_addr);
int32_t ps3_guest_memory_alloc_spill(uint32_t size, uint32_t alignment, uint32_t* out_addr);
int32_t ps3_guest_memory_free(uint32_t addr);
void ppu_register_function(uint64_t addr, void (*fn)(ppu_context*));
extern uint32_t g_barrier_sync_watch;
extern int vm_inline_st_ok;
void ps3_indirect_call(ppu_context* ctx);
void ps3_indirect_call_to(ppu_context* ctx, uint32_t target);
void ps3_hle_call(uint32_t nid, ppu_context* ctx);
int ppu_stwcx32(uint64_t addr, uint32_t expected, uint32_t value);
extern uint32_t ppu_vm_size;
extern const char* ppu_vfs_root;
extern PPU_THREAD_LOCAL void (*g_trampoline_fn)(void*);
}

void func_00B0E348(ppu_context* ctx);
void func_00AEC70C(ppu_context* ctx);
void func_00B3A140(ppu_context* ctx);
void func_00A81AF8(ppu_context* ctx);
void func_00A7EF4C(ppu_context* ctx);
void func_00A7EF5C(ppu_context* ctx);
void func_00A7EF6C(ppu_context* ctx);
void func_00A80220(ppu_context* ctx);
void func_00AE51FC(ppu_context* ctx);
void func_00AE5328(ppu_context* ctx);
void func_00B3E5A4(ppu_context* ctx);
void func_00AB8248(ppu_context* ctx);
void func_00A7B7F4(ppu_context* ctx);
void func_00A7B82C(ppu_context* ctx);
void func_00B0E908(ppu_context* ctx);
void func_00AB8620(ppu_context* ctx);
void func_00AB8740(ppu_context* ctx);
void func_00AB88C0(ppu_context* ctx);
void func_00AB8BEC(ppu_context* ctx);
void func_00AB8CD8(ppu_context* ctx);
void func_00AB9358(ppu_context* ctx);
void june_evt_func_008F705C(ppu_context* ctx);

static int june_initialize_rsx_memory(ppu_context* ctx,
                                      uint32_t local_address,
                                      uint32_t local_size)
{
    if (!ctx || !local_address || local_size <= 0x200u)
        return 0;

    const uint32_t manager_source = vm_read32(ctx->gpr[2] - 0x7810u);
    const uint32_t manager = june_evt_translate_ea(manager_source);
    const uint32_t platform_source = vm_read32(ctx->gpr[2] - 0x7800u);
    const uint32_t platform = june_evt_translate_ea(platform_source);
    const uint32_t allocation_file_source = vm_read32(ctx->gpr[2] - 0x7808u);
    const uint32_t allocation_file = june_evt_translate_ea(allocation_file_source);
    if (!manager || !platform || !allocation_file ||
        (ppu_vm_size && ((uint64_t)manager + 0x10u > ppu_vm_size ||
                         (uint64_t)platform + 0x24u > ppu_vm_size)))
        return 0;

    ppu_context saved = *ctx;
    ctx->gpr[3] = 0x14u;
    ctx->gpr[4] = 0xFF000002u;
    ctx->gpr[5] = allocation_file;
    ctx->gpr[6] = 0x109u;
    ctx->lr = 0x0081588Cu;
    june_evt_func_00884198(ctx);

    const uint32_t node = (uint32_t)ctx->gpr[3];
    if (!node || (ppu_vm_size && (uint64_t)node + 0x14u > ppu_vm_size)) {
        *ctx = saved;
        return 0;
    }

    vm_write32(node + 0x10u, 0);
    vm_write32(node + 0x0Cu, 0);
    vm_write16(node, 0);
    vm_write32(node + 0x08u, local_size - 0x200u);
    vm_write32(node + 0x04u, local_address);
    vm_write32(manager + 0x04u, node);
    vm_write32(manager, node);

    ctx->gpr[3] = manager + 0x0Cu;
    ctx->lr = 0x008158C4u;
    june_evt_func_00835034(ctx);
    vm_write32(manager + 0x08u, local_address);

    ctx->gpr[3] = platform;
    ctx->gpr[4] = 0x24u;
    ctx->lr = 0x008158D8u;
    june_evt_func_00834944(ctx);
    vm_write32(platform + 0x20u, local_size - 0x200u);

    const int initialized = vm_read32(manager + 0x0Cu) != 0;
    *ctx = saved;
    return initialized;
}

static uint32_t june_overlay_base;
static int june_editor_installed;
static constexpr uint32_t JUNE_GFD_ROOT = 0x016390D0u;
static constexpr uint32_t JUNE_GFD_LIST_OFFSET = 0x1254u;
static constexpr uint32_t JUNE_GFD_MUTEX_OFFSET = 0x124Cu;
static constexpr uint32_t JUNE_GFD_SYSTEM_FONT_OFFSET = 0x1290u;
static constexpr uint32_t JUNE_GFD_DBG_FONT_TEXTURE_TOC_OFFSET = 0x4298u;
static constexpr uint32_t FEB16_GFD_ROOT_TOC_OFFSET = 0x5108u;
static constexpr uint32_t FEB16_GFD_SYSTEM_FONT_OFFSET = 0x1318u;
static constexpr uint32_t FEB16_GFD_DBG_FONT_TEXTURE_TOC_OFFSET = 0x4D98u;
static constexpr uint32_t FEB16_GFD_ROOT = 0x01768FE0u;
static constexpr uint32_t GFD_RENDER_LIST0_OFFSET = 0xA0u;
static constexpr uint32_t GFD_RENDER_LIST1_OFFSET = 0xA4u;
static constexpr uint32_t GFD_RENDER_PRIORITY_MAX_OFFSET = 0xB8u;
static constexpr uint32_t GFD_RENDER_FRAME_ID_OFFSET = 0xC4u;
static std::once_flag june_gfd_init_once;
static int june_gfd_init_ready;
static void (*june_original_evt_update_dispatch)(ppu_context*);

struct JuneFrame {
    ppu_context* ctx;
    uint64_t saved_r2;
    uint64_t saved_r13;
    uint64_t saved_cia;
    uint64_t native_r13;
};

struct JuneThreadState {
    uint32_t tls_base;
    unsigned depth;
    JuneFrame frames[64];
};

static thread_local JuneThreadState june_thread_state;

struct JuneFunctionSentinel {
    uint32_t address;
    uint8_t bytes[16];
};

static const JuneFunctionSentinel june_native_sentinels[] = {
    {0x00AB8248u, {0x81,0x62,0x9B,0x28,0x7C,0x69,0x1B,0x78,0x80,0x0B,0x10,0x70,0x2F,0x80,0x00,0x00}},
    {0x00AB8620u, {0xF8,0x21,0xFF,0x91,0x7C,0x08,0x02,0xA6,0xF8,0x01,0x00,0x80,0x80,0x03,0x00,0x00}},
    {0x00AB8740u, {0xF8,0x21,0xFF,0x71,0x7C,0x08,0x02,0xA6,0xFB,0xC1,0x00,0x80,0xF8,0x01,0x00,0xA0}},
    {0x00AB88C0u, {0xF8,0x21,0xFF,0x71,0x7C,0x08,0x02,0xA6,0xFB,0xC1,0x00,0x80,0xF8,0x01,0x00,0xA0}},
    {0x00AB8BECu, {0xF8,0x21,0xFF,0x71,0x7C,0x08,0x02,0xA6,0xFB,0x81,0x00,0x70,0x83,0x82,0x9B,0x28}},
    {0x00AB8CD8u, {0x2F,0x89,0x00,0x00,0x7C,0x08,0x02,0xA6,0xF8,0x21,0xFF,0x21,0xFA,0x61,0x00,0x78}},
    {0x00AB9358u, {0xF8,0x21,0xFF,0x71,0x7C,0x08,0x02,0xA6,0xFB,0xC1,0x00,0x80,0xF8,0x01,0x00,0xA0}},
};

static uint32_t read_be32(const uint8_t* p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

static void write_be32(uint8_t* p, uint32_t value)
{
    p[0] = (uint8_t)(value >> 24);
    p[1] = (uint8_t)(value >> 16);
    p[2] = (uint8_t)(value >> 8);
    p[3] = (uint8_t)value;
}

static uint64_t june_fnv1a64(const uint8_t* bytes, size_t size)
{
    uint64_t value = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i)
        value = (value ^ bytes[i]) * 1099511628211ull;
    return value;
}

static const JuneEventDispatchEntry* june_find_function(uint32_t address)
{
    size_t first = 0;
    size_t last = (size_t)june_evt_dispatch_table_count;
    while (first < last) {
        const size_t mid = first + (last - first) / 2;
        const uint32_t at = june_evt_dispatch_table[mid].source_ea;
        if (at < address) first = mid + 1;
        else last = mid;
    }
    if (first < (size_t)june_evt_dispatch_table_count &&
        june_evt_dispatch_table[first].source_ea == address)
        return &june_evt_dispatch_table[first];
    return nullptr;
}

static const JuneDataMapEntry* june_find_data_pair(uint32_t address)
{
    size_t first = 0;
    size_t last = sizeof(june_data_map) / sizeof(june_data_map[0]);
    while (first < last) {
        const size_t mid = first + (last - first) / 2;
        if (june_data_map[mid].source < address) first = mid + 1;
        else last = mid;
    }
    if (first < sizeof(june_data_map) / sizeof(june_data_map[0]) &&
        june_data_map[first].source == address)
        return &june_data_map[first];
    return nullptr;
}

static uint32_t june_translate_stdout(uint32_t address)
{
    uint32_t source = address;
    if (june_overlay_base && address >= june_overlay_base &&
        (uint64_t)address < (uint64_t)june_overlay_base + JUNE_IMAGE_SIZE)
        source = JUNE_IMAGE_BASE + (address - june_overlay_base);
    if (source >= 0x01478BF0u && source < 0x01478C38u)
        return 0x015E6F68u + (source - 0x01478BF0u);
    return address;
}

static int june_is_shared_target_data(uint32_t address)
{
    if (address >= 0x015E6F68u && address < 0x015E6FB0u) return 1;
    for (size_t i = 0; i < sizeof(june_data_map) / sizeof(june_data_map[0]); ++i)
        if (june_data_map[i].target == address) return 1;
    return 0;
}

static int june_is_static_data_reference(uint32_t address)
{
    size_t first = 0;
    size_t last = sizeof(june_static_data_refs) / sizeof(june_static_data_refs[0]);
    while (first < last) {
        const size_t mid = first + (last - first) / 2;
        if (june_static_data_refs[mid] < address) first = mid + 1;
        else last = mid;
    }
    return first < sizeof(june_static_data_refs) / sizeof(june_static_data_refs[0]) &&
           june_static_data_refs[first] == address;
}

static unsigned char june_ascii_lower(unsigned char ch)
{
    return ch >= 'A' && ch <= 'Z' ? (unsigned char)(ch + ('a' - 'A')) : ch;
}

static int compare_prefix_case_insensitive(const char* text,
                                           const char* prefix,
                                           size_t length)
{
    if (!text || !prefix) return text == prefix ? 0 : 1;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char a = (unsigned char)text[i];
        const unsigned char b = (unsigned char)prefix[i];
        if (!a || june_ascii_lower(a) != june_ascii_lower(b))
            return (int)june_ascii_lower(a) - (int)june_ascii_lower(b);
    }
    return 0;
}

static int june_is_event_path(const char* path)
{
    if (!path) return 0;
    while (*path == '/' || *path == '\\') ++path;
    if (compare_prefix_case_insensitive(path, "app_home", 8) == 0) {
        path += 8;
        while (*path == '/' || *path == '\\') ++path;
    }
    return compare_prefix_case_insensitive(path, "event", 5) == 0;
}

static void june_normalize_guest_event_path(uint32_t path_address,
                                            const char* source)
{
    if (!path_address || (ppu_vm_size && path_address >= ppu_vm_size)) return;
    const size_t available = ppu_vm_size ? ppu_vm_size - path_address : 0x400u;
    const size_t limit = available < 0x400u ? available : 0x400u;
    char guest_path[0x400];
    size_t length = 0;
    while (length < limit) {
        const char ch = (char)vm_read8(path_address + (uint32_t)length);
        guest_path[length] = ch;
        if (!ch) break;
        ++length;
    }
    if (!length || length >= sizeof(guest_path) || length >= limit) return;
    guest_path[length] = '\0';
    if (!june_is_event_path(guest_path)) return;

    char canonical_path[0x400];
    if (!ps3recomp_canonicalize_cpk_event_path(ppu_vfs_root, guest_path,
                                               canonical_path, sizeof(canonical_path))) {
        if (getenv("PS3_FSLOG"))
            fprintf(stderr, "[event-editor] no matching CPK path at %s: '%s'\n",
                    source, guest_path);
        return;
    }
    if (std::strcmp(guest_path, canonical_path) == 0) return;

    const size_t canonical_length = std::strlen(canonical_path);
    for (size_t i = 0; i <= canonical_length; ++i)
        vm_write8(path_address + (uint32_t)i, (uint8_t)canonical_path[i]);
    if (getenv("PS3_FSLOG"))
        fprintf(stderr, "[event-editor] CPK path at %s '%s' -> '%s'\n",
                source, guest_path, canonical_path);
}

void june_evt_func_001625F4_raw(ppu_context* ctx);
void june_evt_func_00162BB8_raw(ppu_context* ctx);

static bool june_copy_canonical_event_path(uint32_t path_address,
                                           const char* source,
                                           char* guest_path,
                                           size_t guest_capacity,
                                           char* canonical_path,
                                           size_t canonical_capacity)
{
    if (!path_address || !guest_path || !guest_capacity ||
        !canonical_path || !canonical_capacity ||
        (ppu_vm_size && path_address >= ppu_vm_size))
        return false;

    size_t limit = ppu_vm_size ? ppu_vm_size - path_address : guest_capacity;
    if (limit > guest_capacity) limit = guest_capacity;
    size_t length = 0;
    while (length < limit) {
        const char ch = (char)vm_read8(path_address + (uint32_t)length);
        guest_path[length] = ch;
        if (!ch) break;
        ++length;
    }
    if (!length || length >= limit || length >= guest_capacity) return false;
    guest_path[length] = '\0';
    if (!june_is_event_path(guest_path)) return false;

    if (!ps3recomp_canonicalize_cpk_event_path(ppu_vfs_root, guest_path,
                                                canonical_path,
                                                canonical_capacity)) {
        if (getenv("PS3_FSLOG"))
            fprintf(stderr, "[event-editor] no matching CPK path at %s: '%s'\n",
                    source, guest_path);
        return false;
    }
    return true;
}

static void june_evt_call_pak_with_canonical_path(
    ppu_context* ctx, void (*raw_function)(ppu_context*), const char* source)
{
    if (!ctx || !raw_function) return;

    char guest_path[0x400];
    char canonical_path[0x400];
    if (!june_copy_canonical_event_path((uint32_t)ctx->gpr[3], source,
                                        guest_path, sizeof(guest_path),
                                        canonical_path, sizeof(canonical_path)) ||
        std::strcmp(guest_path, canonical_path) == 0) {
        raw_function(ctx);
        return;
    }

    uint32_t canonical_guest_address = 0;
    const uint32_t canonical_size = (uint32_t)std::strlen(canonical_path) + 1u;
    if (ps3_guest_memory_alloc(canonical_size, 16, &canonical_guest_address) != 0 ||
        !canonical_guest_address) {
        if (getenv("PS3_FSLOG"))
            fprintf(stderr, "[event-editor] could not allocate canonical path for %s\n",
                    source);
        raw_function(ctx);
        return;
    }

    for (uint32_t i = 0; i < canonical_size; ++i)
        vm_write8(canonical_guest_address + i, (uint8_t)canonical_path[i]);
    if (getenv("PS3_FSLOG"))
        fprintf(stderr, "[event-editor] PAK %s '%s' -> '%s'\n",
                source, guest_path, canonical_path);

    ctx->gpr[3] = canonical_guest_address;
    raw_function(ctx);
    ps3_guest_memory_free(canonical_guest_address);
}

void june_evt_func_001625F4(ppu_context* ctx)
{
    june_evt_call_pak_with_canonical_path(ctx, june_evt_func_001625F4_raw,
                                          "readReqPakFile");
}

void june_evt_func_00162BB8(ppu_context* ctx)
{
    june_evt_call_pak_with_canonical_path(ctx, june_evt_func_00162BB8_raw,
                                          "existPakFile");
}

static void june_normalize_event_path(ppu_context* ctx)
{
    static thread_local char previous_path[0x400];
    if (!ctx || !ctx->gpr[3]) return;
    const uint32_t object = (uint32_t)ctx->gpr[3];
    const uint32_t path_address = object + 0x44u;
    if (ppu_vm_size && (uint64_t)path_address + 0x400u > ppu_vm_size) return;

    char guest_path[0x400];
    size_t length = 0;
    while (length + 1 < sizeof(guest_path)) {
        const char ch = (char)vm_read8(path_address + (uint32_t)length);
        guest_path[length] = ch;
        if (!ch) break;
        ++length;
    }
    guest_path[sizeof(guest_path) - 1] = '\0';
    if (length < 10 || !june_is_event_path(guest_path) ||
        std::strcmp(previous_path, guest_path) == 0)
        return;
    std::strncpy(previous_path, guest_path, sizeof(previous_path) - 1);
    previous_path[sizeof(previous_path) - 1] = '\0';

    june_normalize_guest_event_path(path_address, "editor");
}

extern "C" void june_evt_normalize_pak_filename(ppu_context* ctx)
{
    if (!ctx) return;
    june_normalize_guest_event_path((uint32_t)ctx->gpr[3], "PAK lookup");
}

static void june_evt_update_file_dispatch(ppu_context* ctx)
{
    june_normalize_event_path(ctx);
    if (june_original_evt_update_dispatch)
        june_original_evt_update_dispatch(ctx);
}

static int june_is_overlay_ea(uint32_t address)
{
    return june_overlay_base && address >= june_overlay_base &&
           (uint64_t)address < (uint64_t)june_overlay_base + JUNE_IMAGE_SIZE;
}

static int june_is_source_image_ea(uint32_t address)
{
    return address >= JUNE_IMAGE_BASE &&
           (uint64_t)address < (uint64_t)JUNE_IMAGE_BASE + JUNE_IMAGE_SIZE;
}

static int june_event_trace_enabled(void)
{
    static const int enabled = [] {
        const char* value = getenv("JUNE_EVT_TRACE");
        return value && *value;
    }();
    return enabled;
}

static void june_trace_unmapped_code(uint32_t address)
{
    static thread_local unsigned count;
    if (!june_event_trace_enabled() || count >= 64 ||
        address < JUNE_IMAGE_BASE ||
        (uint64_t)address >= (uint64_t)JUNE_IMAGE_BASE + JUNE_IMAGE_SIZE)
        return;
    ++count;
    fprintf(stderr, "[event-editor-trace] unmapped code=0x%08X\n", address);
}

static void june_trace_dispatch(ppu_context* ctx, uint32_t source_ea, unsigned depth)
{
    static thread_local unsigned count;
    if (!june_event_trace_enabled() || count >= 16384) return;
    ++count;
    fprintf(stderr,
            "[event-editor-trace] dispatch tid=%llu depth=%u source=0x%08X lr=0x%08X cia=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X\n",
            (unsigned long long)ctx->thread_id, depth, source_ea, (uint32_t)ctx->lr,
            ctx->cia, (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
            (uint32_t)ctx->gpr[5]);
}

static void june_trace_leave(ppu_context* ctx, unsigned depth)
{
    static thread_local unsigned count;
    if (!june_event_trace_enabled() || count >= 16384) return;
    ++count;
    fprintf(stderr,
            "[event-editor-trace] leave tid=%llu depth=%u source=0x%08X lr=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X\n",
            (unsigned long long)ctx->thread_id, depth, (uint32_t)ctx->cia,
            (uint32_t)ctx->lr, (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
            (uint32_t)ctx->gpr[5]);
}

static int june_gfd_trace_enabled(void)
{
    static const int enabled = [] {
        const char* value = getenv("JUNE_GFD_TRACE");
        return value && *value;
    }();
    return enabled;
}

extern "C" void june_evt_gcm_context_probe(ppu_context* ctx, uint32_t callsite)
{
    static thread_local unsigned count;
    if (!june_gfd_trace_enabled() || !ctx || count >= 32) return;
    ++count;

    const uint32_t context = (uint32_t)ctx->gpr[3];
    if (ppu_vm_size && (uint64_t)context + 0x10u > ppu_vm_size) {
        fprintf(stderr,
                "[june-gcm-callback] callsite=0x%08X context=0x%08X size=%u invalid_context=1 sp=0x%08X\n",
                callsite, context, (uint32_t)ctx->gpr[4],
                (uint32_t)ctx->gpr[1]);
        return;
    }

    const uint32_t begin = vm_read32(context);
    const uint32_t end = vm_read32(context + 4u);
    const uint32_t current = vm_read32(context + 8u);
    const uint32_t callback = vm_read32(context + 0xCu);
    const int callback_valid = callback >= 4u &&
        (!ppu_vm_size || (uint64_t)callback + 8u <= ppu_vm_size);
    const uint32_t callback_code = callback_valid ? vm_read32(callback) : 0;
    const uint32_t callback_toc = callback_valid ? vm_read32(callback + 4u) : 0;
    const uint32_t remaining = current <= end ? end - current : 0xFFFFFFFFu;
    const uint32_t sp = (uint32_t)ctx->gpr[1];
    const uint32_t backchain = (!ppu_vm_size || (uint64_t)sp + 8u <= ppu_vm_size)
        ? vm_read32(sp + 4u) : 0;
    const uint32_t caller_lr = (!ppu_vm_size || (uint64_t)sp + 0x98u <= ppu_vm_size)
        ? vm_read32(sp + 0x94u) : 0;
    const uint32_t tail0 = current >= begin && current - begin >= 8u
        ? vm_read32(current - 8u) : 0;
    const uint32_t tail1 = current >= begin && current - begin >= 8u
        ? vm_read32(current - 4u) : 0;

    fprintf(stderr,
            "[june-gcm-callback] callsite=0x%08X context=0x%08X size=%u required_bytes=%u begin=0x%08X end=0x%08X current=0x%08X remaining=0x%08X callback=0x%08X code=0x%08X toc=0x%08X sp=0x%08X backchain=0x%08X caller_lr=0x%08X tail=%08X,%08X\n",
            callsite, context, (uint32_t)ctx->gpr[4],
            (uint32_t)ctx->gpr[4] * 4u, begin, end, current, remaining,
            callback, callback_code, callback_toc, sp, backchain, caller_lr,
            tail0, tail1);
}

extern "C" void june_evt_trace_point(ppu_context* ctx, uint32_t caller,
                                     uint32_t target, int after)
{
    if (june_gfd_trace_enabled() && ctx &&
        ((caller == 0x008E732Cu && target == 0x00884FE0u) ||
         (caller == 0x008E78DCu && target == 0x00884FE0u) ||
         (caller == 0x00884FE0u && target == 0x008350F0u) ||
         (caller == 0x008350F0u && target == 0x00000066u))) {
        const uint32_t root = june_overlay_base + (JUNE_GFD_ROOT - JUNE_IMAGE_BASE);
        const uint32_t manager = vm_read32(root + JUNE_GFD_LIST_OFFSET);
        const uint32_t mutex_id = vm_read32(root + JUNE_GFD_MUTEX_OFFSET);
        fprintf(stderr,
                "[gfd-bridge] %s caller=0x%08X target=0x%08X r3=0x%08X june_root=0x%08X list=0x%08X mutex_id=0x%08X\n",
                after ? "return" : "call", caller, target,
                (uint32_t)ctx->gpr[3], root, manager, mutex_id);
    }
    static thread_local unsigned count;
    if (!june_event_trace_enabled() || !june_thread_state.depth || count >= 32768)
        return;
    ++count;
    fprintf(stderr,
            "[event-editor-trace] step %s tid=%llu caller=0x%08X target=0x%08X lr=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X\n",
            after ? "return" : "call", (unsigned long long)ctx->thread_id,
            caller, target, (uint32_t)ctx->lr, (uint32_t)ctx->gpr[3],
            (uint32_t)ctx->gpr[4], (uint32_t)ctx->gpr[5]);
}

extern "C" void june_evt_trace_loop(ppu_context* ctx, uint32_t site)
{
    static thread_local uint32_t counts[4];
    static const int enabled = [] {
        const char* value = getenv("JUNE_EVT_LOOP_TRACE");
        return value && *value;
    }();
    if (!enabled || !ctx) return;
    unsigned index;
    unsigned interval;
    switch (site) {
    case 0x002762C4u: index = 0; interval = 8; break;
    case 0x002762E8u: index = 1; interval = 100; break;
    case 0x002773A4u: index = 2; interval = 10; break;
    case 0x00277840u: index = 3; interval = 1; break;
    default: return;
    }
    const uint32_t count = ++counts[index];
    if (index != 3 && count > 5 && count % interval != 0) return;
    fprintf(stderr,
            "[event-editor-loop] site=0x%08X count=%u r3=0x%08X r4=0x%08X r5=0x%08X r23=0x%08X r28=0x%08X r29=0x%08X r30=0x%08X r31=0x%08X\n",
            site, count, (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
            (uint32_t)ctx->gpr[5], (uint32_t)ctx->gpr[23],
            (uint32_t)ctx->gpr[28], (uint32_t)ctx->gpr[29],
            (uint32_t)ctx->gpr[30], (uint32_t)ctx->gpr[31]);
}

static void june_trace_native_task(ppu_context* ctx, uint32_t caller, uint32_t target, int after)
{
    static thread_local unsigned count;
    if (!june_event_trace_enabled() || !june_thread_state.depth || count >= 256)
        return;
    ++count;
    fprintf(stderr,
            "[event-editor-trace] native %s tid=%llu caller=0x%08X target=0x%08X lr=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X\n",
            after ? "return" : "call", (unsigned long long)ctx->thread_id,
            caller, target, (uint32_t)ctx->lr,
            (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
            (uint32_t)ctx->gpr[5]);
}

extern "C" void june_evt_cycle_probe(ppu_context* ctx, uint32_t source_ea,
                                      uint32_t object, uint32_t field,
                                      uint32_t limit)
{
    static thread_local uint32_t counts[10];
    static const int enabled = [] {
        const char* value = getenv("JUNE_EVT_CYCLE_TRACE");
        return value && *value && *value != '0';
    }();
    if (!enabled || !ctx) return;
    unsigned index;
    switch (source_ea) {
    case 0x00824AC0u: index = 0; break;
    case 0x00824F18u: index = 1; break;
    case 0x008C20D8u: index = 2; break;
    case 0x008EE0F4u: index = 3; break;
    case 0x00824AFCu: index = 4; break;
    case 0x00824B14u: index = 5; break;
    case 0x00824B48u: index = 6; break;
    case 0x008ED278u: index = 7; break;
    case 0x008ED2C8u: index = 8; break;
    case 0x008ED2D8u: index = 9; break;
    default: return;
    }
    const uint32_t count = ++counts[index];
    if (count > 12 && count % 128 != 0) return;
    fprintf(stderr,
            "[june-cycle] ea=%08X n=%u tid=%llu lr=%08X sp=%08X toc=%08X r0=%08X r3=%08X r4=%08X r5=%08X r9=%08X r11=%08X r29=%08X r30=%08X r31=%08X cr=%08X aux=%08X,%08X,%08X\n",
            source_ea, count, (unsigned long long)ctx->thread_id,
            (uint32_t)ctx->lr, (uint32_t)ctx->gpr[1], (uint32_t)ctx->gpr[2],
            (uint32_t)ctx->gpr[0], (uint32_t)ctx->gpr[3],
            (uint32_t)ctx->gpr[4], (uint32_t)ctx->gpr[5],
            (uint32_t)ctx->gpr[9], (uint32_t)ctx->gpr[11],
            (uint32_t)ctx->gpr[29], (uint32_t)ctx->gpr[30],
            (uint32_t)ctx->gpr[31], ctx->cr, object, field, limit);
}

extern "C" void june_evt_failure_probe(ppu_context* ctx, uint32_t source_ea)
{
    static thread_local unsigned count;
    static const int enabled = [] {
        const char* value = getenv("JUNE_EVT_FAILURE_TRACE");
        return value && *value && *value != '0';
    }();
    if (!enabled || !ctx || count >= 128) return;
    if (source_ea == 0x008B9850u && (uint32_t)ctx->gpr[3] >= 0x1000u) return;
    uint32_t probe_pointer = 0;
    uint32_t probe_value = 0;
    uint32_t manager = 0;
    uint32_t manager_mutex = 0;
    if (source_ea == 0x00815904u && (uint32_t)ctx->lr != 0x00826D94u) return;
    if (source_ea == 0x008350F0u && (uint32_t)ctx->lr != 0x00815990u) return;
    if (source_ea == 0x008350F0u) {
        probe_pointer = (uint32_t)ctx->gpr[3];
        if (!ppu_vm_size || (uint64_t)probe_pointer + 4u <= ppu_vm_size)
            probe_value = vm_read32(probe_pointer);
    }
    if (source_ea == 0x00815904u || source_ea == 0x008350F0u) {
        const uint64_t toc = ctx->gpr[2];
        if (toc >= 0x7810u && (!ppu_vm_size || toc - 0x7810u + 4u <= ppu_vm_size)) {
            manager = vm_read32(toc - 0x7810u);
            if (manager && (!ppu_vm_size || (uint64_t)manager + 0x10u <= ppu_vm_size))
                manager_mutex = vm_read32(manager + 0x0Cu);
        }
        if (source_ea == 0x008350F0u && probe_value != 0 && manager_mutex != 0) return;
    }
    const uint32_t n = ++count;
    fprintf(stderr,
            "[june-fail] ea=%08X n=%u tid=%llu lr=%08X sp=%08X toc=%08X r3=%08X r4=%08X r5=%08X r6=%08X r7=%08X r8=%08X r9=%08X r22=%08X r24=%08X r25=%08X r26=%08X r29=%08X r30=%08X r31=%08X ptr=%08X val=%08X mgr=%08X mgr_mutex=%08X\n",
            source_ea, n, (unsigned long long)ctx->thread_id,
            (uint32_t)ctx->lr, (uint32_t)ctx->gpr[1], (uint32_t)ctx->gpr[2],
            (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
            (uint32_t)ctx->gpr[5], (uint32_t)ctx->gpr[6],
            (uint32_t)ctx->gpr[7], (uint32_t)ctx->gpr[8],
            (uint32_t)ctx->gpr[9], (uint32_t)ctx->gpr[22],
            (uint32_t)ctx->gpr[24], (uint32_t)ctx->gpr[25],
            (uint32_t)ctx->gpr[26], (uint32_t)ctx->gpr[29],
            (uint32_t)ctx->gpr[30], (uint32_t)ctx->gpr[31], probe_pointer, probe_value, manager, manager_mutex);
}
static uint32_t june_overlay_ea(uint32_t source_ea)
{
    if (!june_overlay_base || source_ea < JUNE_IMAGE_BASE ||
        (uint64_t)source_ea >= (uint64_t)JUNE_IMAGE_BASE + JUNE_IMAGE_SIZE)
        return source_ea;
    return june_overlay_base + (source_ea - JUNE_IMAGE_BASE);
}

extern "C" uint32_t june_evt_overlay_toc(void)
{
    return june_overlay_ea(JUNE_TOC);
}

static uint32_t june_translate_code(uint32_t address)
{
    if (june_is_overlay_ea(address)) return address;
    if (june_find_function(address)) return june_overlay_ea(address);
    june_trace_unmapped_code(address);
    return address;
}

static uint64_t june_translate_pointer(uint64_t value)
{
    const uint32_t address = (uint32_t)value;
    const uint32_t high = (uint32_t)(value >> 32);
    if (high && high != 0xFFFFFFFFu) return value;
    if (june_is_shared_target_data(address)) return value;
    const uint32_t stream = june_translate_stdout(address);
    if (stream != address) return stream;
    if (june_is_overlay_ea(address)) return june_evt_translate_ea(address);
    if (june_find_function(address)) return june_overlay_ea(address);
    if (june_is_source_image_ea(address)) return june_overlay_ea(address);
    if (!june_find_data_pair(address) && !june_is_static_data_reference(address))
        return value;
    return june_evt_translate_ea(address);
}

extern "C" uint32_t june_evt_translate_ea(uint32_t address)
{
    if (!june_overlay_base) return address;
    const uint32_t renderer_source = june_is_overlay_ea(address)
        ? JUNE_IMAGE_BASE + (address - june_overlay_base) : address;
    if (renderer_source == JUNE_GFD_ROOT + GFD_RENDER_FRAME_ID_OFFSET)
        return FEB16_GFD_ROOT + GFD_RENDER_FRAME_ID_OFFSET;
    if (renderer_source >= 0x0163A370u && renderer_source < 0x0163A372u)
        return 0x0176A314u + (renderer_source - 0x0163A370u);
    if (renderer_source >= 0x0163A372u && renderer_source < 0x0163A374u)
        return 0x0176A312u + (renderer_source - 0x0163A372u);
    uint32_t renderer_target_offset = 0;
    switch (renderer_source) {
    case 0x016347F0u + 0x98u: renderer_target_offset = 0x94u; break;
    case 0x016347F0u + 0xF4u: renderer_target_offset = 0xF0u; break;
    case 0x016347F0u + 0xF8u: renderer_target_offset = 0xF4u; break;
    case 0x016347F0u + 0xFCu: renderer_target_offset = 0xF8u; break;
    case 0x016347F0u + 0x1B8u: renderer_target_offset = 0x1B4u; break;
    case 0x016347F0u + 0x208u: renderer_target_offset = 0x204u; break;
    case 0x016347F0u + 0x20Cu: renderer_target_offset = 0x208u; break;
    default: break;
    }
    if (renderer_target_offset) {
        const uint32_t renderer = vm_read32(FEB16_TOC - 0x7330u);
        if (renderer && (!ppu_vm_size ||
            (uint64_t)renderer + renderer_target_offset + 4u <= ppu_vm_size))
            return renderer + renderer_target_offset;
    }
    const uint32_t stream = june_translate_stdout(address);
    if (stream != address) return stream;
    if (june_is_overlay_ea(address)) {
        const uint32_t source_ea = JUNE_IMAGE_BASE + (address - june_overlay_base);
        const JuneDataMapEntry* pair = june_find_data_pair(source_ea);
        return pair ? pair->target : address;
    }
    if (june_is_shared_target_data(address)) return address;
    const JuneDataMapEntry* pair = june_find_data_pair(address);
    if (pair) return pair->target;
    if (june_is_source_image_ea(address)) return june_overlay_ea(address);
    if (june_is_static_data_reference(address)) return june_overlay_ea(address);
    return address;
}

extern "C" uint32_t june_evt_source_ea(uint32_t address)
{
    if (!june_is_overlay_ea(address)) return address;
    return JUNE_IMAGE_BASE + (address - june_overlay_base);
}

extern "C" uint64_t june_evt_rebase_loaded_pointer(uint32_t load_ea, uint64_t value)
{
    if (!june_is_overlay_ea(load_ea)) return value;
    const uint32_t address = (uint32_t)value;
    const uint32_t high = (uint32_t)(value >> 32);
    if (high && high != 0xFFFFFFFFu) return value;
    if (june_is_shared_target_data(address)) return value;
    const uint32_t stream = june_translate_stdout(address);
    if (stream != address) return stream;
    if (june_find_function(address)) return value;
    if (june_is_source_image_ea(address))
        return june_overlay_ea(address);
    return june_translate_pointer(value);
}

extern "C" uint64_t june_evt_rebase_stored_pointer(uint32_t store_ea, uint64_t value)
{
    return june_is_shared_target_data(store_ea) ? june_translate_pointer(value) : value;
}

static int june_ensure_tls(uint32_t* out_base)
{
    JuneThreadState& state = june_thread_state;
    if (!state.tls_base) {
        if (JUNE_TLS_FILE_SIZE > JUNE_TLS_MEM_SIZE || JUNE_TLS_MEM_SIZE > 0x1000u)
            return 0;
        uint32_t base = 0;
        if (ps3_guest_memory_alloc(0x1000u, 0x1000u, &base) != 0 || !base)
            return 0;
        memset(vm_base + base, 0, 0x1000u);
        memcpy(vm_base + base,
               vm_base + june_overlay_base + (JUNE_TLS_VADDR - JUNE_IMAGE_BASE),
               JUNE_TLS_FILE_SIZE);
        memset(vm_base + base + JUNE_TLS_FILE_SIZE, 0,
               JUNE_TLS_MEM_SIZE - JUNE_TLS_FILE_SIZE);
        state.tls_base = base;
    }
    *out_base = state.tls_base;
    return 1;
}

static uint32_t june_input_word(uint32_t address)
{
    address = june_evt_translate_ea(address);
    if (address < 4u || (ppu_vm_size && (uint64_t)address + 4u > ppu_vm_size)) return 0;
    return vm_read32(address);
}

static void june_toolbar_event_probe(ppu_context* ctx)
{
    const char* enabled = getenv("PS3_MOUSE_TRACE");
    if (!enabled || !*enabled || *enabled == '0' || ctx->gpr[4] != 2u) return;
    uint32_t widget = (uint32_t)ctx->gpr[3];
    const uint32_t parent = june_input_word(widget + 0x48u);
    if (parent) widget = parent;
    const uint32_t work = june_input_word(widget + 0x88u);
    const uint32_t core = work ? june_input_word(work + 0x0Cu) : 0;
    const uint32_t predicate = work ? june_input_word(work + 0x20u) : 0;
    fprintf(stderr, "[gui-toolbar-event] widget=%08X type=%u lparam=%08X wparam=%08X kind=%u id=%u work=%08X core=%08X core_state=%u predicate=%08X source_fps=%u native_fps=%u\n",
            widget, (uint32_t)ctx->gpr[4], (uint32_t)ctx->gpr[5],
            (uint32_t)ctx->gpr[6], (uint32_t)ctx->gpr[7], (uint32_t)ctx->gpr[8],
            work, core, core ? june_input_word(core + 4u) : 0, predicate,
            june_input_word(JUNE_GFD_ROOT + 0x2Cu),
            vm_read32(FEB16_GFD_ROOT + 0x2Cu));
}

extern "C" int june_evt_context_enter(ppu_context* ctx, uint32_t source_ea)
{
    if (!june_editor_installed || !ctx) return 0;
    if (source_ea == 0x0025F990u) june_toolbar_event_probe(ctx);
    if (source_ea == 0x002735D4u) {
        const char* trace = getenv("PS3_MOUSE_TRACE");
        if (trace && trace[0] && trace[0] != '0')
            fprintf(stderr,
                    "[event-editor-input] action callback tid=%llu lr=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X\n",
                    (unsigned long long)ctx->thread_id, (uint32_t)ctx->lr,
                    (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
                    (uint32_t)ctx->gpr[5]);
    }
    if (source_ea == 0x00D303B8u) {
        static int armed;
        const char* watch = getenv("PPU_EVENT_VTABLE_WATCH");
        if (!armed && watch && *watch && *watch != '0') {
            armed = 1;
            g_barrier_sync_watch = june_overlay_ea(0x00FD56BCu);
            vm_inline_st_ok = 0;
            fprintf(stderr, "[event-editor] vtable write watch armed at 0x%08X\n",
                    g_barrier_sync_watch + 0x40u);
        }
    }
    JuneThreadState& state = june_thread_state;
    if (!june_find_function(source_ea)) {
        june_trace_unmapped_code(source_ea);
        return 0;
    }
    june_trace_dispatch(ctx, source_ea, state.depth);
    if (state.depth >= sizeof(state.frames) / sizeof(state.frames[0]))
        return 0;
    uint32_t tls_base = 0;
    if (!june_ensure_tls(&tls_base)) return 0;
    JuneFrame& frame = state.frames[state.depth];
    frame.ctx = ctx;
    frame.saved_r2 = ctx->gpr[2];
    frame.saved_r13 = ctx->gpr[13];
    frame.saved_cia = ctx->cia;
    frame.native_r13 = ctx->gpr[13];
    if (state.depth && state.frames[state.depth - 1].ctx == ctx)
        frame.native_r13 = state.frames[state.depth - 1].native_r13;
    state.depth++;
    ctx->gpr[2] = june_overlay_ea(JUNE_TOC);
    ctx->gpr[13] = tls_base + 0x7000u;
    ctx->cia = source_ea;
    std::call_once(june_gfd_init_once, [ctx, tls_base] {
        ppu_context saved = *ctx;
        june_evt_func_008F705C(ctx);
        *ctx = saved;
        const uint32_t root = june_overlay_base + (JUNE_GFD_ROOT - JUNE_IMAGE_BASE);
        const uint32_t list = vm_read32(root + JUNE_GFD_LIST_OFFSET);
        const uint32_t mutex_id = vm_read32(root + JUNE_GFD_MUTEX_OFFSET);
        const uint32_t target_root = vm_read32(FEB16_TOC - FEB16_GFD_ROOT_TOC_OFFSET);
        const uint32_t target_root_state =
            target_root == FEB16_GFD_ROOT ? vm_read32(target_root) : 0;
        const uint32_t target_root_3c =
            target_root == FEB16_GFD_ROOT ? vm_read32(target_root + 0x3Cu) : 0;
        const uint32_t target_render_list0 =
            target_root == FEB16_GFD_ROOT
                ? vm_read32(target_root + GFD_RENDER_LIST0_OFFSET)
                : 0;
        const uint32_t target_render_list1 =
            target_root == FEB16_GFD_ROOT
                ? vm_read32(target_root + GFD_RENDER_LIST1_OFFSET)
                : 0;
        const uint32_t target_priority_max =
            target_root == FEB16_GFD_ROOT
                ? vm_read32(target_root + GFD_RENDER_PRIORITY_MAX_OFFSET)
                : 0;
        const uint32_t target_frame_id =
            target_root == FEB16_GFD_ROOT
                ? vm_read32(target_root + GFD_RENDER_FRAME_ID_OFFSET)
                : 0xFFFFFFFFu;
        const uint32_t target_system_font =
            target_root == FEB16_GFD_ROOT
                ? vm_read32(target_root + FEB16_GFD_SYSTEM_FONT_OFFSET)
                : 0;
        const uint32_t june_font_texture_slot =
            june_overlay_ea(JUNE_TOC) - JUNE_GFD_DBG_FONT_TEXTURE_TOC_OFFSET;
        const uint32_t june_font_texture_global =
            june_evt_translate_ea(vm_read32(june_font_texture_slot));
        const uint32_t target_font_texture_global =
            vm_read32(FEB16_TOC - FEB16_GFD_DBG_FONT_TEXTURE_TOC_OFFSET);
        const int june_font_texture_global_in_range = june_font_texture_global &&
            (!ppu_vm_size || (uint64_t)june_font_texture_global + 4u <= ppu_vm_size);
        const int target_font_texture_global_in_range = target_font_texture_global &&
            (!ppu_vm_size || (uint64_t)target_font_texture_global + 4u <= ppu_vm_size);
        const uint32_t target_font_texture = target_font_texture_global_in_range
            ? vm_read32(target_font_texture_global)
            : 0;
        const int target_font_texture_in_range = target_font_texture &&
            (!ppu_vm_size || (uint64_t)target_font_texture + 0x20u <= ppu_vm_size);
        const int debug_font_bridged = june_font_texture_global_in_range &&
            target_font_texture_in_range;
        if (debug_font_bridged)
            vm_write32(june_font_texture_global, target_font_texture);
        const uint64_t render_list_bytes = (uint64_t)target_priority_max * 0x20u;
        const int render_lists_in_range = target_priority_max > 0x49u &&
            target_priority_max <= 0x100u && target_frame_id < 2u &&
            target_render_list0 && target_render_list1 &&
            (!ppu_vm_size ||
             ((uint64_t)target_render_list0 + render_list_bytes <= ppu_vm_size &&
              (uint64_t)target_render_list1 + render_list_bytes <= ppu_vm_size));
        const int list_in_range = list &&
            (!ppu_vm_size || (uint64_t)list + 0x20u <= ppu_vm_size);
        const int root_3c_in_range = target_root_3c &&
            (!ppu_vm_size || (uint64_t)target_root_3c + 0x20u <= ppu_vm_size);
        const int font_in_range = target_system_font &&
            (!ppu_vm_size || (uint64_t)target_system_font + 0x20u <= ppu_vm_size);
        if (target_root == FEB16_GFD_ROOT)
            vm_write32(root, target_root_state);
        if (render_lists_in_range) {
            vm_write32(root + GFD_RENDER_LIST0_OFFSET, target_render_list0);
            vm_write32(root + GFD_RENDER_LIST1_OFFSET, target_render_list1);
            vm_write32(root + GFD_RENDER_PRIORITY_MAX_OFFSET, target_priority_max);
            vm_write32(root + GFD_RENDER_FRAME_ID_OFFSET, target_frame_id);
        }
        if (root_3c_in_range)
            vm_write32(root + 0x3Cu, target_root_3c);
        if (font_in_range)
            vm_write32(root + JUNE_GFD_SYSTEM_FONT_OFFSET, target_system_font);
        const uint32_t config_ea = tls_base + 0xF00u;
        const int config_in_range = !ppu_vm_size ||
            (uint64_t)config_ea + sizeof(CellGcmConfig) <= ppu_vm_size;
        const int config_status = config_in_range
            ? cellGcmGetConfiguration(
                  reinterpret_cast<CellGcmConfig*>(static_cast<uintptr_t>(config_ea)))
            : -1;
        const uint32_t local_address = config_status == 0
            ? vm_read32(config_ea)
            : 0;
        const uint32_t local_size = config_status == 0
            ? vm_read32(config_ea + 8u)
            : 0;
        const int rsx_memory_ready = config_status == 0 &&
            june_initialize_rsx_memory(ctx, local_address, local_size);
        const uint32_t list_mutex = list_in_range ? vm_read32(list + 0x1Cu) : 0;
        const uint32_t rsx_manager = june_evt_translate_ea(vm_read32(ctx->gpr[2] - 0x7810u));
        const uint32_t rsx_mutex_id = rsx_manager
            ? vm_read32(rsx_manager + 0x0Cu)
            : 0;
        june_gfd_init_ready = list_in_range && mutex_id && list_mutex &&
            root_3c_in_range && font_in_range && render_lists_in_range &&
            rsx_memory_ready;
        fprintf(stderr,
                "[event-editor] June GFD setup %s root=0x%08X list=0x%08X mutex=0x%08X list_mutex=0x%08X root_state=0x%08X root_3c=0x%08X ot_lists=0x%08X/0x%08X prio_max=0x%08X frame_id=0x%08X system_font=0x%08X debug_font_global=0x%08X/0x%08X texture=0x%08X bridged=%d rsx_config=%d local=0x%08X size=0x%08X rsx_mutex=0x%08X\n",
                june_gfd_init_ready ? "ready" : "failed", root, list,
                mutex_id, list_mutex, target_root_state, target_root_3c,
                target_render_list0, target_render_list1, target_priority_max,
                target_frame_id, target_system_font, june_font_texture_global,
                target_font_texture_global, target_font_texture, debug_font_bridged,
                config_status,
                local_address, local_size, rsx_mutex_id);
    });
    if (!june_gfd_init_ready) {
        ctx->gpr[2] = frame.saved_r2;
        ctx->gpr[13] = frame.saved_r13;
        ctx->cia = frame.saved_cia;
        state.depth--;
        return 0;
    }
    if (source_ea == 0x00276118u) {
        const char* bypass = getenv("JUNE_EVT_TEST_READY_BYPASS");
        if (bypass && bypass[0] && bypass[0] != '0') {
            const uint32_t root = june_overlay_base + (JUNE_GFD_ROOT - JUNE_IMAGE_BASE);
            const uint32_t previous_state = vm_read32(root);
            const uint32_t previous_ready = vm_read32(root + 0x3Cu);
            const uint32_t state = previous_state | 1u;
            if (state != previous_state) vm_write32(root, state);
            if (!previous_ready) vm_write32(root + 0x3Cu, 1);
            fprintf(stderr,
                    "[event-editor] readiness probe at entry 0x%08X: root state %u -> %u, ready %u -> 1, callbacks=%08X/%08X\n",
                    source_ea, previous_state, state, previous_ready,
                    vm_read32(root + 0x1220), vm_read32(root + 0x1224));
        }
    }
    return 1;
}

extern "C" void june_evt_context_leave(ppu_context* ctx)
{
    JuneThreadState& state = june_thread_state;
    if (!state.depth || !ctx) return;
    JuneFrame& frame = state.frames[state.depth - 1];
    if (frame.ctx != ctx) return;
    june_trace_leave(ctx, state.depth - 1);
    ctx->gpr[2] = frame.saved_r2;
    ctx->gpr[13] = frame.saved_r13;
    ctx->cia = frame.saved_cia;
    state.depth--;
}

static void june_window_frame_stage(ppu_context* ctx, uint32_t source_ea)
{
    if (!ctx || !june_editor_installed) return;
    const uint32_t manager = june_translate_pointer(
        vm_read32(june_overlay_ea(JUNE_TOC - 0x6270u)));
    if (!manager || !vm_read32(manager + 4u)) return;
    const ppu_context saved = *ctx;
    if (!june_evt_context_enter(ctx, source_ea)) return;
    if (source_ea == 0x0087199Cu)
        june_evt_func_0087199C(ctx);
    else
        june_evt_func_00872724(ctx);
    while (g_trampoline_fn) {
        void (*continuation)(void*) = g_trampoline_fn;
        g_trampoline_fn = nullptr;
        continuation(ctx);
    }
    june_evt_context_leave(ctx);
    *ctx = saved;
}

extern "C" void june_evt_window_update(ppu_context* ctx)
{
    june_window_frame_stage(ctx, 0x00872724u);
}

extern "C" void june_evt_window_draw(ppu_context* ctx)
{
    june_window_frame_stage(ctx, 0x0087199Cu);
}

extern "C" void june_evt_indirect_call(ppu_context* ctx)
{
    if (!ctx) return;
    const uint32_t callsite = (uint32_t)ctx->lr;
    const bool formatter_call = callsite == 0x00D30B80u;
    if (june_event_trace_enabled() &&
        (callsite == 0x00D02740u || callsite == 0x00D02778u ||
         callsite == 0x00D02894u || formatter_call)) {
        static thread_local unsigned trace_count;
        const uint32_t target = (uint32_t)ctx->ctr;
        if (trace_count++ < 64 || (formatter_call && target == 0)) {
            const uint32_t object = (uint32_t)ctx->gpr[31];
            const int object_in_range = object >= 4u &&
                (!ppu_vm_size || (uint64_t)object + 0x20u <= ppu_vm_size);
            const uint32_t caller_lr = (uint32_t)vm_read64(ctx->gpr[1] + 0xC0u);
            const uint32_t outer_lr = (uint32_t)vm_read64(ctx->gpr[1] + 0x150u);
            const uint32_t outer_object = (uint32_t)vm_read64(ctx->gpr[1] + 0x120u);
            uint32_t fields[8] = {};
            if (object_in_range)
                for (unsigned i = 0; i < 8; ++i)
                    fields[i] = vm_read32(object + i * 4u);
            fprintf(stderr,
                    "[event-editor-indirect] lr=0x%08X caller_lr=0x%08X outer_lr=0x%08X outer_object=0x%08X cia=0x%08X ctr=0x%08X object=0x%08X descriptor=0x%08X code=0x%08X toc=0x%08X r3=0x%08X r4=0x%08X fields=%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X\n",
                    callsite, caller_lr, outer_lr, outer_object,
                    (uint32_t)ctx->cia, (uint32_t)ctx->ctr, object,
                    (uint32_t)ctx->gpr[9], (uint32_t)ctx->gpr[0],
                    (uint32_t)ctx->gpr[2], (uint32_t)ctx->gpr[3],
                    (uint32_t)ctx->gpr[4], fields[0], fields[1], fields[2],
                    fields[3], fields[4], fields[5], fields[6], fields[7]);
            if (formatter_call) {
                const uint32_t sp = (uint32_t)ctx->gpr[1];
                const uint32_t descriptor = vm_read32(sp + 0xE0u);
                const int descriptor_in_range = descriptor >= 4u &&
                    (!ppu_vm_size || (uint64_t)descriptor + 8u <= ppu_vm_size);
                const uint32_t descriptor_code = descriptor_in_range ?
                    vm_read32(descriptor) : 0;
                const uint32_t descriptor_toc = descriptor_in_range ?
                    vm_read32(descriptor + 4u) : 0;
                const uint32_t saved_toc = vm_read32(sp + 0x2Cu);
                const uint32_t toc_slot = saved_toc + 0x51D0u;
                const uint32_t toc_value = saved_toc &&
                    (!ppu_vm_size || (uint64_t)toc_slot + 4u <= ppu_vm_size) ?
                    vm_read32(toc_slot) : 0;
                const uint64_t overlay_end =
                    (uint64_t)june_overlay_base + JUNE_IMAGE_SIZE;
                const int stack_overlaps_overlay = june_overlay_base &&
                    (uint64_t)sp < overlay_end &&
                    (uint64_t)june_overlay_base < (uint64_t)sp + 0x178u;
                fprintf(stderr,
                        "[event-editor-format-call] target=0x%08X sp=0x%08X backchain=0x%08X saved_lr=0x%08X saved_toc=0x%08X toc_slot=0x%08X toc_value=0x%08X descriptor=0x%08X descriptor_code=0x%08X descriptor_toc=0x%08X local_e4=0x%08X overlay=0x%08X-0x%08llX stack_overlap=%d r3=0x%08X r4=0x%08X r5=0x%08X\n",
                        target, sp, vm_read32(sp + 4u),
                        vm_read32(sp + 0x174u), saved_toc, toc_slot, toc_value,
                        descriptor, descriptor_code, descriptor_toc,
                        vm_read32(sp + 0xE4u),
                        june_overlay_base, (unsigned long long)overlay_end,
                        stack_overlaps_overlay,
                        (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
                        (uint32_t)ctx->gpr[5]);
            }
        }
    }
    ps3_indirect_call_to(ctx, june_translate_code((uint32_t)ctx->ctr));
}

extern "C" void june_evt_indirect_call_to(ppu_context* ctx, uint32_t target)
{
    if (!ctx) return;
    ps3_indirect_call_to(ctx, june_translate_code(target));
}

static uint64_t june_translate_stack_value(uint64_t value)
{
    return june_translate_pointer(value);
}

extern "C" void june_evt_hle_call(uint32_t nid, ppu_context* ctx)
{
    if (!ctx) return;
    for (int reg = 3; reg <= 10; ++reg)
        ctx->gpr[reg] = june_translate_pointer(ctx->gpr[reg]);
    uint64_t stack_args[2];
    const uint32_t stack_base = (uint32_t)ctx->gpr[1] + 0x70u;
    stack_args[0] = vm_read64(stack_base);
    stack_args[1] = vm_read64(stack_base + 8u);
    const uint64_t translated[2] = {
        june_translate_stack_value(stack_args[0]),
        june_translate_stack_value(stack_args[1])
    };
    for (unsigned i = 0; i < 2; ++i) {
        if (translated[i] != stack_args[i])
            vm_write64(stack_base + i * 8u, translated[i]);
    }
    ps3_hle_call(nid, ctx);
    for (unsigned i = 0; i < 2; ++i) {
        const uint32_t address = stack_base + i * 8u;
        if (translated[i] != stack_args[i] && vm_read64(address) == translated[i])
            vm_write64(address, stack_args[i]);
    }
}

extern "C" int june_evt_stwcx32(uint64_t address, uint32_t expected, uint32_t value)
{
    return ppu_stwcx32(june_evt_translate_ea((uint32_t)address), expected, value);
}

static void june_native_task_call(ppu_context* ctx, uint32_t target,
                                  void (*function)(ppu_context*), int stack_args,
                                  uint16_t pointer_argument_registers = 0x07F8u)
{
    JuneThreadState& state = june_thread_state;
    if (!ctx || !state.depth || state.frames[state.depth - 1].ctx != ctx)
        return;
    const uint64_t old_r2 = ctx->gpr[2];
    const uint64_t old_r13 = ctx->gpr[13];
    const uint64_t old_cia = ctx->cia;
    const uint32_t caller = (uint32_t)old_cia;
    const uint32_t old_stack_base = (uint32_t)ctx->gpr[1] + 0x70u;
    uint64_t old_stack[2] = {0, 0};
    const uint32_t raw_r3 = (uint32_t)ctx->gpr[3];
    const uint32_t raw_r4 = (uint32_t)ctx->gpr[4];
    const uint32_t raw_r5 = (uint32_t)ctx->gpr[5];
    const uint32_t raw_r6 = (uint32_t)ctx->gpr[6];
    for (int reg = 3; reg <= 10; ++reg) {
        if (pointer_argument_registers & (1u << reg))
            ctx->gpr[reg] = june_translate_pointer(ctx->gpr[reg]);
    }
    if (target == 0x00AB6554u && june_gfd_trace_enabled()) {
        static thread_local unsigned trace_count;
        if (trace_count++ < 64)
            fprintf(stderr,
                    "[gfd-mem-alloc] caller=0x%08X raw=%08X,%08X,%08X,%08X native=%08X,%08X,%08X,%08X\n",
                    caller, raw_r3, raw_r4, raw_r5, raw_r6,
                    (uint32_t)ctx->gpr[3], (uint32_t)ctx->gpr[4],
                    (uint32_t)ctx->gpr[5], (uint32_t)ctx->gpr[6]);
    }
    if (stack_args) {
        old_stack[0] = vm_read64(old_stack_base);
        old_stack[1] = vm_read64(old_stack_base + 8u);
        vm_write64(old_stack_base, june_translate_stack_value(old_stack[0]));
        vm_write64(old_stack_base + 8u, june_translate_stack_value(old_stack[1]));
    }
    ctx->gpr[2] = FEB16_TOC;
    ctx->gpr[13] = state.frames[state.depth - 1].native_r13;
    ctx->cia = target;
    june_trace_native_task(ctx, caller, target, 0);
    function(ctx);
    while (g_trampoline_fn) {
        void (*continuation)(void*) = g_trampoline_fn;
        g_trampoline_fn = nullptr;
        continuation(ctx);
    }
    june_trace_native_task(ctx, caller, target, 1);
    ctx->gpr[2] = old_r2;
    ctx->gpr[13] = old_r13;
    ctx->cia = old_cia;
    if (stack_args) {
        vm_write64(old_stack_base, old_stack[0]);
        vm_write64(old_stack_base + 8u, old_stack[1]);
    }
}

static void june_native_00AB6554(ppu_context* ctx)
{
    ps3_indirect_call_to(ctx, 0x00AB6554u);
}

extern "C" void june_evt_bind_render_resource(ppu_context* ctx, uint32_t kind)
{
    static thread_local unsigned counts[2];
    if (!ctx || kind > 1u) return;
    const uint32_t object = (uint32_t)ctx->gpr[3];
    if (counts[kind]++ < 8u)
        fprintf(stderr, "[june-resource-bind] kind=%u object=%08X lr=%08X\n",
                kind, object, (uint32_t)ctx->lr);
    june_native_task_call(ctx, kind ? 0x00A7B82Cu : 0x00A7B7F4u,
                         kind ? func_00A7B82C : func_00A7B7F4, 0, 0x0008u);
}

static void june_trace_render_node_state(const char* phase, uint32_t priority,
                                         uint32_t node)
{
    const uint32_t root = vm_read32(FEB16_TOC - FEB16_GFD_ROOT_TOC_OFFSET);
    const int root_valid = root == FEB16_GFD_ROOT &&
        (!ppu_vm_size || (uint64_t)root + 0xD0u <= ppu_vm_size);
    const uint32_t frame_id = root_valid ? vm_read32(root + GFD_RENDER_FRAME_ID_OFFSET) : 0xFFFFFFFFu;
    const uint32_t priority_max = root_valid ? vm_read32(root + GFD_RENDER_PRIORITY_MAX_OFFSET) : 0;
    const uint32_t render_list0 = root_valid ? vm_read32(root + GFD_RENDER_LIST0_OFFSET) : 0;
    const uint32_t render_list1 = root_valid ? vm_read32(root + GFD_RENDER_LIST1_OFFSET) : 0;
    uint32_t words[16] = {};
    const int node_valid = node >= 4u &&
        (!ppu_vm_size || (uint64_t)node + sizeof(words) <= ppu_vm_size);
    if (node_valid)
        for (unsigned i = 0; i < 16; ++i)
            words[i] = vm_read32(node + i * 4u);
    fprintf(stderr,
            "[june-render-node-state] %s priority=%u node=%08X root=%08X frame=%u max=%u lists=%08X/%08X valid=%d words=%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X\n",
            phase, priority, node, root, frame_id, priority_max,
            render_list0, render_list1, node_valid,
            words[0], words[1], words[2], words[3], words[4], words[5],
            words[6], words[7], words[8], words[9], words[10], words[11],
            words[12], words[13], words[14], words[15]);
}

extern "C" void june_evt_register_render_node(ppu_context* ctx)
{
    static thread_local unsigned reports;
    if (!ctx) return;
    const uint32_t priority = (uint32_t)ctx->gpr[3];
    const uint32_t node = (uint32_t)ctx->gpr[4];
    const bool trace = reports++ < 128u;
    if (trace)
        fprintf(stderr, "[june-render-node] priority=%u node=%08X caller=%08X packet=%08X\n",
                priority, node, (uint32_t)ctx->lr, vm_read32(node));
    if (trace && june_gfd_trace_enabled() && node >= 4u &&
        (!ppu_vm_size || (uint64_t)node + 0x40u <= ppu_vm_size))
        fprintf(stderr,
                "[june-render-packet] node=%08X adjacent=%08X,%08X,%08X,%08X,%08X,%08X,%08X,%08X\n",
                node, vm_read32(node + 0x20u), vm_read32(node + 0x24u),
                vm_read32(node + 0x28u), vm_read32(node + 0x2Cu),
                vm_read32(node + 0x30u), vm_read32(node + 0x34u),
                vm_read32(node + 0x38u), vm_read32(node + 0x3Cu));
    if (trace && june_gfd_trace_enabled())
        june_trace_render_node_state("before", priority, node);
    june_native_task_call(ctx, 0x00B0E908u, func_00B0E908, 0, 0x0010u);
    if (trace && june_gfd_trace_enabled())
        june_trace_render_node_state("after", priority, node);
}
void june_evt_func_00884414(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB6554u, june_native_00AB6554, 0, 0x0020u);
}

static void june_native_00AB66A8(ppu_context* ctx)
{
    ps3_indirect_call_to(ctx, 0x00AB66A8u);
}
void june_evt_func_00884558(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB66A8u, june_native_00AB66A8, 0);
}

static void june_native_free(ppu_context* ctx)
{
    ps3_indirect_call_to(ctx, 0x00AB6338u);
}
void june_evt_func_00884214(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB6338u, june_native_free, 0);
}

void june_evt_func_008869F0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB8248u, func_00AB8248, 0);
}

void june_evt_func_00886DC8(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB8620u, func_00AB8620, 0);
}

void june_evt_func_00886EE8(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB8740u, func_00AB8740, 0);
}

void june_evt_func_00887068(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB88C0u, func_00AB88C0, 0);
}

void june_evt_func_00887394(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB8BECu, func_00AB8BEC, 0);
}

void june_evt_func_00887480(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB8CD8u, func_00AB8CD8, 1);
}

void june_evt_func_00887B00(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB9358u, func_00AB9358, 0);
}

static int june_load_image(const uint8_t** out_image, size_t* out_size, uint8_t** out_owned)
{
    *out_image = nullptr;
    *out_size = 0;
    *out_owned = nullptr;
#ifdef _WIN32
    HRSRC resource = FindResourceA(nullptr, MAKEINTRESOURCEA(102), RT_RCDATA);
    if (!resource) return 0;
    HGLOBAL loaded = LoadResource(nullptr, resource);
    const uint8_t* image = (const uint8_t*)LockResource(loaded);
    const DWORD size = SizeofResource(nullptr, resource);
    if (!image || !size) return 0;
    *out_image = image;
    *out_size = (size_t)size;
    return 1;
#else
    const char* path = getenv("PS3_JUNE_EVENT_EDITOR_IMAGE");
#ifndef PS3RECOMP_JUNE_EVENT_EDITOR_IMAGE_PATH
#define PS3RECOMP_JUNE_EVENT_EDITOR_IMAGE_PATH "event_editor_june/june_image.bin"
#endif
    if (!path || !*path) path = PS3RECOMP_JUNE_EVENT_EDITOR_IMAGE_PATH;
    FILE* file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return 0; }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return 0; }
    uint8_t* bytes = (uint8_t*)malloc((size_t)length);
    if (!bytes) { fclose(file); return 0; }
    const size_t read = fread(bytes, 1, (size_t)length, file);
    fclose(file);
    if (read != (size_t)length) { free(bytes); return 0; }
    *out_image = bytes;
    *out_size = read;
    *out_owned = bytes;
    return 1;
#endif
}

static int june_target_matches(void)
{
    static const uint32_t expected_row[5] = {
        0x00F327CCu, 0x00000000u, 0x011C94A8u, 0x011C94B0u, 0x011C94B8u,
    };
    for (unsigned i = 0; i < 5; ++i)
        if (vm_read32(FEB16_EVENT_MENU_ROW + i * 4u) != expected_row[i])
            return 0;
    if (vm_read32(FEB16_TOC - FEB16_GFD_ROOT_TOC_OFFSET) != FEB16_GFD_ROOT)
        return 0;
    for (size_t i = 0; i < sizeof(june_native_sentinels) / sizeof(june_native_sentinels[0]); ++i) {
        const JuneFunctionSentinel& sentinel = june_native_sentinels[i];
        for (unsigned j = 0; j < sizeof(sentinel.bytes); ++j)
            if (vm_read8(sentinel.address + j) != sentinel.bytes[j])
                return 0;
    }
    return 1;
}

static int june_source_menu_matches(const uint8_t* image, size_t size)
{
    const uint64_t row_offset = (uint64_t)JUNE_EVENT_MENU_ROW - JUNE_IMAGE_BASE;
    if (row_offset + 20 > size) return 0;
    const uint8_t* row = image + row_offset;
    static const uint32_t expected[5] = {
        0x00DF5B28u, 0x01000000u, 0x01093EE8u, 0x01093EF0u, 0x01093EF8u,
    };
    for (unsigned i = 0; i < 5; ++i)
        if (read_be32(row + i * 4u) != expected[i]) return 0;
    const uint32_t label = expected[0];
    if (label < JUNE_IMAGE_BASE || (uint64_t)label + 13u > JUNE_IMAGE_BASE + size)
        return 0;
    if (memcmp(image + (label - JUNE_IMAGE_BASE), "EVENT EDITOR", 13) != 0)
        return 0;
    for (unsigned i = 2; i < 5; ++i) {
        const uint32_t opd = expected[i];
        if (opd < JUNE_OPD_BASE || (uint64_t)opd + 8 > (uint64_t)JUNE_OPD_BASE + JUNE_OPD_SIZE)
            return 0;
        const size_t opd_offset = (size_t)(opd - JUNE_IMAGE_BASE);
        if (opd_offset + 8 > size || !june_find_function(read_be32(image + opd_offset)))
            return 0;
    }
    return 1;
}

extern "C" int event_editor_june_init(void)
{
    if (june_editor_installed) return 1;
    if (!vm_base || !june_target_matches()) {
        fprintf(stderr, "[event-editor] February target signature mismatch; editor row left unchanged\n");
        return 0;
    }
    const uint8_t* image = nullptr;
    size_t image_size = 0;
    uint8_t* owned_image = nullptr;
    if (!june_load_image(&image, &image_size, &owned_image) ||
        image_size != JUNE_IMAGE_SIZE || june_fnv1a64(image, image_size) != JUNE_IMAGE_FNV1A64 ||
        !june_source_menu_matches(image, image_size)) {
        if (owned_image) free(owned_image);
        fprintf(stderr, "[event-editor] June image signature mismatch; editor row left unchanged\n");
        return 0;
    }
    uint32_t overlay = 0;
    if (ps3_guest_memory_alloc_spill(JUNE_IMAGE_SIZE, 0x10000u, &overlay) != 0 || !overlay ||
        (ppu_vm_size && (uint64_t)overlay + JUNE_IMAGE_SIZE > ppu_vm_size)) {
        if (overlay) ps3_guest_memory_free(overlay);
        if (owned_image) free(owned_image);
        fprintf(stderr, "[event-editor] could not allocate the June overlay; editor row left unchanged\n");
        return 0;
    }
    june_overlay_base = overlay;
    memcpy(vm_base + overlay, image, image_size);
    if (owned_image) free(owned_image);
    uint32_t tls_base = 0;
    if (!june_ensure_tls(&tls_base)) {
        ps3_guest_memory_free(overlay);
        june_overlay_base = 0;
        fprintf(stderr, "[event-editor] could not allocate editor TLS; editor row left unchanged\n");
        return 0;
    }
    const uint32_t june_toc = june_overlay_ea(JUNE_TOC);
    for (uint32_t offset = 0; offset < JUNE_OPD_SIZE; offset += 8u) {
        const uint32_t opd_ea = JUNE_OPD_BASE + offset;
        const uint32_t code = vm_read32(june_overlay_ea(opd_ea));
        if (!june_find_function(code)) continue;
        vm_write32(june_overlay_ea(opd_ea), june_overlay_ea(code));
        vm_write32(june_overlay_ea(opd_ea + 4u), june_toc);
    }
    for (uint64_t i = 0; i < june_evt_dispatch_table_count; ++i) {
        const JuneEventDispatchEntry& entry = june_evt_dispatch_table[i];
        if (entry.source_ea == 0x00275C6Cu) {
            june_original_evt_update_dispatch = entry.func;
            ppu_register_function(june_overlay_ea(entry.source_ea), june_evt_update_file_dispatch);
        } else {
            ppu_register_function(june_overlay_ea(entry.source_ea), entry.func);
        }
    }
    const uint32_t source_row = june_overlay_ea(JUNE_EVENT_MENU_ROW);
    uint32_t row[5];
    for (unsigned i = 0; i < 5; ++i)
        row[i] = vm_read32(source_row + i * 4u);
    row[0] = june_overlay_ea(row[0]);
    row[2] = june_overlay_ea(row[2]);
    row[3] = june_overlay_ea(row[3]);
    row[4] = june_overlay_ea(row[4]);
    for (unsigned i = 0; i < 5; ++i)
        vm_write32(FEB16_EVENT_MENU_ROW + i * 4u, row[i]);
    june_editor_installed = 1;
    fprintf(stderr, "[event-editor] June editor overlay installed at 0x%08X (%llu functions)\n",
            june_overlay_base, (unsigned long long)june_evt_dispatch_table_count);
    return 1;
}

void june_evt_func_0092BE30(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B3E5A4u, func_00B3E5A4, 0, 0x0048u);
}

void june_evt_func_008B9724(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AE51FCu, func_00AE51FC, 0, 0x0020u);
}

void june_evt_func_008B9850(ppu_context* ctx)
{
    ctx->gpr[7] &= 1u;
    june_native_task_call(ctx, 0x00AE5328u, func_00AE5328, 0, 0x0020u);
}

void june_evt_func_00828220(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A80220u, func_00A80220, 0, 0x0008u);
}

void june_evt_func_00827264(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A7EF4Cu, func_00A7EF4C, 0, 0x0000u);
}

void june_evt_func_00827274(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A7EF5Cu, func_00A7EF5C, 0, 0x0000u);
}

void june_evt_func_00827284(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A7EF6Cu, func_00A7EF6C, 0, 0x0008u);
}

void june_evt_func_008ED738(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B0E348u, func_00B0E348, 0, 0x0000u);
}

void june_evt_func_008C2B64(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEC70Cu, func_00AEC70C, 0, 0x0010u);
}

void june_evt_func_0082A05C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A81AF8u, func_00A81AF8, 0, 0x0000u);
}

void june_evt_func_00827194(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B3A140u, func_00B3A140, 0, 0x0000u);
}

extern void func_00AEB118(ppu_context* ctx);
extern void func_00AEC6B4(ppu_context* ctx);
extern void func_00A7F3F0(ppu_context* ctx);
extern void func_00AE50C8(ppu_context* ctx);
extern void func_00AEC560(ppu_context* ctx);

void june_evt_func_008C0E90(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEB118u, func_00AEB118, 0, 0x0000u);
}

void june_evt_func_008C2B0C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEC6B4u, func_00AEC6B4, 0, 0x0000u);
}

void june_evt_func_00827B8C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A7F3F0u, func_00A7F3F0, 0, 0x0010u);
}

void june_evt_func_008B95F0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AE50C8u, func_00AE50C8, 0, 0x0020u);
}

void june_evt_func_008C29B8(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEC560u, func_00AEC560, 0, 0x0020u);
}

extern void func_00B39AA8(ppu_context* ctx);

void june_evt_func_00826A30(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B39AA8u, func_00B39AA8, 0, 0x0010u);
}

extern void func_00A6BE6C(ppu_context* ctx);

void june_evt_func_00815904(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A6BE6Cu, func_00A6BE6C, 0, 0x0000u);
}

extern void func_00A6C1D0(ppu_context* ctx);

void june_evt_func_00815C68(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A6C1D0u, func_00A6C1D0, 0, 0x0008u);
}

extern void func_00B39C94(ppu_context* ctx);

void june_evt_func_00826CD0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B39C94u, func_00B39C94, 0, 0x0000u);
}

extern void func_00B39C50(ppu_context* ctx);

void june_evt_func_00826C8C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B39C50u, func_00B39C50, 0, 0x0008u);
}

extern void func_00B39C10(ppu_context* ctx);

void june_evt_func_00826C4C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B39C10u, func_00B39C10, 0, 0x0008u);
}

extern void func_00B131AC(ppu_context* ctx);

void june_evt_func_008F2C10(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B131ACu, func_00B131AC, 0, 0x0008u);
}

extern void func_00B139F8(ppu_context* ctx);

void june_evt_func_008F3664(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B139F8u, func_00B139F8, 0, 0x0008u);
}

extern void func_00A98EC8(ppu_context* ctx);

void june_evt_func_008616F8(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A98EC8u, func_00A98EC8, 0, 0x02F0u);
}

extern void func_00A99140(ppu_context* ctx);

void june_evt_func_00861970(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A99140u, func_00A99140, 0, 0x0070u);
}

extern void func_00A99324(ppu_context* ctx);

void june_evt_func_00861B54(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A99324u, func_00A99324, 0, 0x0070u);
}

extern void func_00B60BD0(ppu_context* ctx);

void june_evt_func_00868814(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B60BD0u, func_00B60BD0, 0, 0x0300u);
}

extern void func_00ADFAB8(ppu_context* ctx);

void june_evt_func_008B07D0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00ADFAB8u, func_00ADFAB8, 0, 0x0000u);
}

extern void func_00ADFAC8(ppu_context* ctx);

void june_evt_func_008B07F0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00ADFAC8u, func_00ADFAC8, 0, 0x0008u);
}

extern void func_00ADFAD8(ppu_context* ctx);

void june_evt_func_008B0800(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00ADFAD8u, func_00ADFAD8, 0, 0x0018u);
}

extern void func_00ADFAE4(ppu_context* ctx);

void june_evt_func_008B080C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00ADFAE4u, func_00ADFAE4, 0, 0x0018u);
}

extern void func_00ADFB30(ppu_context* ctx);

void june_evt_func_008B0858(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00ADFB30u, func_00ADFB30, 0, 0x0010u);
}

extern void func_00AE024C(ppu_context* ctx);

void june_evt_func_008B15CC(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AE024Cu, func_00AE024C, 0, 0x0008u);
}

extern void func_00AE1174(ppu_context* ctx);

void june_evt_func_008B284C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AE1174u, func_00AE1174, 0, 0x0018u);
}

extern void func_00AE1430(ppu_context* ctx);

void june_evt_func_008B2B08(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AE1430u, func_00AE1430, 0, 0x0018u);
}

extern void func_00AEBF04(ppu_context* ctx);

void june_evt_func_008C1F70(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEBF04u, func_00AEBF04, 0, 0x0000u);
}

extern void func_00AEBFB8(ppu_context* ctx);

void june_evt_func_008C2024(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEBFB8u, func_00AEBFB8, 0, 0x0000u);
}

extern void func_00AEC06C(ppu_context* ctx);

void june_evt_func_008C20D8(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEC06Cu, func_00AEC06C, 0, 0x0000u);
}

extern void func_00AEC464(ppu_context* ctx);

void june_evt_func_008C2768(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AEC464u, func_00AEC464, 0, 0x0000u);
}

extern void func_00AF900C(ppu_context* ctx);

void june_evt_func_008D5034(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AF900Cu, func_00AF900C, 0, 0x0008u);
}

extern void func_00AE48F8(ppu_context* ctx);

void june_evt_func_008B8CB0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AE48F8u, func_00AE48F8, 0, 0x0020u);
}

extern void func_00AB7AFC(ppu_context* ctx);

void june_evt_func_008861F0(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB7AFCu, func_00AB7AFC, 0, 0);
}

extern void func_00AB7B08(ppu_context* ctx);

void june_evt_func_008861FC(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB7B08u, func_00AB7B08, 0, 0);
}

extern void func_00A909F4(ppu_context* ctx);

void june_evt_func_008427CC(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00A909F4u, func_00A909F4, 0, 0x0008u);
}

extern void func_00B3A064(ppu_context* ctx);

void june_evt_func_0082702C(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B3A064u, func_00B3A064, 0, 0x0008u);
}

extern "C" void june_evt_button_probe(ppu_context* ctx)
{
    const char* enabled = getenv("PS3_MOUSE_TRACE");
    if (!enabled || !*enabled || *enabled == '0' || !(vm_read16(0x0176444Au) & 1u)) return;
    const uint32_t part = (uint32_t)ctx->gpr[4];
    const uint32_t clip = (uint32_t)ctx->gpr[5];
    float rect[4], bounds[4];
    for (unsigned i = 0; i < 4; ++i) {
        const uint32_t r = june_input_word(part + i * 4u);
        const uint32_t b = june_input_word(clip + i * 4u);
        memcpy(&rect[i], &r, 4); memcpy(&bounds[i], &b, 4);
    }
    fprintf(stderr, "[gui-button-hit] widget=%08X part=%08X id=%u callback=%08X mouse=%d,%d rect=%.2f,%.2f,%.2f,%.2f clip=%.2f,%.2f,%.2f,%.2f\n",
            (uint32_t)ctx->gpr[3], part, june_input_word(part + 0x24u),
            (uint32_t)ctx->gpr[7], (int16_t)vm_read16(0x0174838Cu),
            (int16_t)vm_read16(0x0174838Eu), rect[0], rect[1], rect[2], rect[3],
            bounds[0], bounds[1], bounds[2], bounds[3]);
}

extern void func_00B1767C(ppu_context* ctx);

void june_evt_func_008F6ECC(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00B1767Cu, func_00B1767C, 0, 0);
}
