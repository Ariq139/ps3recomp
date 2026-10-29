#include "event_editor_june_api.h"
#include "generated/june_image_info.h"
#include "win32_compat.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
int32_t ps3_guest_memory_free(uint32_t addr);
void ppu_register_function(uint64_t addr, void (*fn)(ppu_context*));
extern uint32_t g_barrier_sync_watch;
extern int vm_inline_st_ok;
void ps3_indirect_call(ppu_context* ctx);
void ps3_indirect_call_to(ppu_context* ctx, uint32_t target);
void ps3_hle_call(uint32_t nid, ppu_context* ctx);
int ppu_stwcx32(uint64_t addr, uint32_t expected, uint32_t value);
extern uint32_t ppu_vm_size;
}

void func_00AB8248(ppu_context* ctx);
void func_00AB8620(ppu_context* ctx);
void func_00AB8740(ppu_context* ctx);
void func_00AB88C0(ppu_context* ctx);
void func_00AB8BEC(ppu_context* ctx);
void func_00AB8CD8(ppu_context* ctx);
void func_00AB9358(ppu_context* ctx);

static uint32_t june_overlay_base;
static int june_editor_installed;

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

static int june_is_shared_target_data(uint32_t address)
{
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

extern "C" void june_evt_trace_point(ppu_context* ctx, uint32_t caller,
                                     uint32_t target, int after)
{
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

extern "C" int june_evt_context_enter(ppu_context* ctx, uint32_t source_ea)
{
    if (!june_editor_installed || !ctx) return 0;
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

extern "C" void june_evt_indirect_call(ppu_context* ctx)
{
    if (!ctx) return;
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
    vm_write64(stack_base, june_translate_stack_value(stack_args[0]));
    vm_write64(stack_base + 8u, june_translate_stack_value(stack_args[1]));
    ps3_hle_call(nid, ctx);
    vm_write64(stack_base, stack_args[0]);
    vm_write64(stack_base + 8u, stack_args[1]);
}

extern "C" int june_evt_stwcx32(uint64_t address, uint32_t expected, uint32_t value)
{
    return ppu_stwcx32(june_evt_translate_ea((uint32_t)address), expected, value);
}

static void june_native_task_call(ppu_context* ctx, uint32_t target,
                                  void (*function)(ppu_context*), int stack_args)
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
    for (int reg = 3; reg <= 10; ++reg)
        ctx->gpr[reg] = june_translate_pointer(ctx->gpr[reg]);
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
void june_evt_func_00884414(ppu_context* ctx)
{
    june_native_task_call(ctx, 0x00AB6554u, june_native_00AB6554, 0);
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
    if (ps3_guest_memory_alloc(JUNE_IMAGE_SIZE, 0x10000u, &overlay) != 0 || !overlay ||
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
        ppu_register_function(june_overlay_ea(entry.source_ea), entry.func);
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
