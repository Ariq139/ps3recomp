/*
 * prx_loader.c — see prx_loader.h for the design.
 *
 * Two steps per module: place the relocated image into guest RAM at its load
 * base, then register every lifted function in the host dispatch table.
 */
#include "prx_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The runtime's guest memory: host_ptr = vm_base + guest_addr. Declared by
 * runtime/memory/vm.h; we only need the symbol, so extern it directly to keep
 * the loader independent of the memory backend's init path. */
extern uint8_t* vm_base;

/* ---- export registry ---------------------------------------------------- *
 * NID -> guest-address table, filled as modules load. The HLE dispatcher
 * calls prx_resolve_export on ordinary imports, including misses, so index
 * this table instead of scanning all loaded exports on every call. */
typedef struct {
    uint32_t nid;
    uint32_t guest_addr;
    uint32_t library_index;
} prx_export_reg_entry;
#define PRX_EXPORT_REG_CAP 2048u
#define PRX_EXPORT_HASH_BITS 12u
#define PRX_EXPORT_HASH_SIZE (1u << PRX_EXPORT_HASH_BITS)
#define PRX_EXPORT_HASH_MASK (PRX_EXPORT_HASH_SIZE - 1u)
/* NID zero is not published by prx_load_module. Keep the original maximum
 * export count and a load factor of at most one half. Module loading is a
 * startup operation; this does not change the registry's threading contract. */
static prx_export_reg_entry s_export_reg[PRX_EXPORT_HASH_SIZE];
static uint32_t             s_export_reg_n = 0;

static prx_export_reg_entry* prx_export_slot(uint32_t nid)
{
    uint32_t i = (nid * 2654435761u) >> (32u - PRX_EXPORT_HASH_BITS);
    while (s_export_reg[i].nid && s_export_reg[i].nid != nid)
        i = (i + 1u) & PRX_EXPORT_HASH_MASK;
    return &s_export_reg[i];
}

typedef struct {
    uint8_t descriptor[0x1C];
    const char* name;
    uint32_t module_base;
    int externally_registered;
} prx_loaded_library;
#define PRX_LIBRARY_REG_CAP 256
static prx_loaded_library s_loaded_libraries[PRX_LIBRARY_REG_CAP];
static uint32_t s_loaded_libraries_n = 0;

static void prx_export_register(uint32_t nid, uint32_t guest_addr,
                                uint32_t library_index)
{
    /* Last writer wins if two modules export the same NID (matches the PS3
     * loader's link order). Overwrite an existing entry rather than duplicate. */
    if (!nid) return;
    prx_export_reg_entry* e = prx_export_slot(nid);
    if (e->nid) {
        e->guest_addr = guest_addr;
        e->library_index = library_index;
        return;
    }
    if (s_export_reg_n < PRX_EXPORT_REG_CAP) {
        *e = (prx_export_reg_entry){ nid, guest_addr, library_index };
        ++s_export_reg_n;
    } else
        fprintf(stderr, "[prx] export registry full, dropping NID 0x%08X\n", nid);
}

static uint32_t prx_library_register_loaded(const prx_module* m,
                                            const prx_library_descriptor* lib)
{
    if (!m || !lib || !vm_base ||
        (uint64_t)lib->vaddr + sizeof(s_loaded_libraries[0].descriptor) > m->image_size ||
        s_loaded_libraries_n >= PRX_LIBRARY_REG_CAP)
        return UINT32_MAX;

    prx_loaded_library* loaded = &s_loaded_libraries[s_loaded_libraries_n];
    memcpy(loaded->descriptor, vm_base + m->base + lib->vaddr,
           sizeof(loaded->descriptor));
    loaded->name = lib->name;
    loaded->module_base = m->base;
    loaded->externally_registered = 0;
    return s_loaded_libraries_n++;
}

uint32_t prx_resolve_export(uint32_t nid)
{
    if (!nid || !s_export_reg_n) return 0;
    const prx_export_reg_entry* e = prx_export_slot(nid);
    return e->nid ? e->guest_addr : 0;
}

uint32_t prx_export_registry_count(void) { return s_export_reg_n; }

int prx_register_library_descriptor(const uint8_t descriptor[0x1C],
                                    const char** out_library_name)
{
    if (out_library_name)
        *out_library_name = NULL;
    if (!descriptor)
        return 0;

    for (uint32_t i = 0; i < s_loaded_libraries_n; ++i) {
        prx_loaded_library* lib = &s_loaded_libraries[i];
        if (memcmp(descriptor, lib->descriptor, sizeof(lib->descriptor)) != 0)
            continue;
        lib->externally_registered = 1;
        if (out_library_name)
            *out_library_name = lib->name;
        return 1;
    }
    return 0;
}

int prx_export_library_is_registered(uint32_t nid)
{
    if (!nid || !s_export_reg_n) return 0;
    const prx_export_reg_entry* e = prx_export_slot(nid);
    if (!e->nid || e->library_index == UINT32_MAX ||
        e->library_index >= s_loaded_libraries_n)
        return 0;
    return s_loaded_libraries[e->library_index].externally_registered;
}

prx_load_result prx_load_module(const prx_module* m, prx_register_fn reg)
{
    prx_load_result r;
    memset(&r, 0, sizeof(r));

    if (!m || !m->image || m->image_size == 0) {
        fprintf(stderr, "[prx] load failed: null/empty module\n");
        return r;
    }
    if (!vm_base) {
        fprintf(stderr, "[prx] load failed: vm_base not initialized "
                        "(call vm_init before loading PRX modules)\n");
        return r;
    }

    r.base       = m->base;
    r.image_size = m->image_size;

    uint32_t* library_indices = NULL;
    if (m->library_count && m->libraries) {
        library_indices = (uint32_t*)malloc((size_t)m->library_count * sizeof(uint32_t));
        if (!library_indices) {
            fprintf(stderr, "[prx] out of memory tracking export libraries\n");
            return r;
        }
    }

    /* Step A — place the relocated image into guest RAM at the load base.
     * The bytes already carry base-applied addresses (prx_relocate.py), so the
     * module's data, OPDs and pointer tables are immediately consistent with
     * the lifted code and the OPD-walking dispatcher. */
    memcpy(vm_base + m->base, m->image, m->image_size);

    /* Step B — register every lifted function at its guest address. The lift
     * baked the load base in (--base B), so e->addr is already B + offset and
     * needs no adjustment here. A null host function (rare: a table hole) is
     * skipped rather than registered. */
    if (reg) {
        for (uint64_t i = 0; i < m->func_count; i++) {
            const prx_func_entry* e = &m->funcs[i];
            if (!e->func)
                continue;
            reg((uint32_t)e->addr, e->func);
            r.funcs_registered++;
        }
    }

    if (m->library_count && m->libraries) {
        for (uint32_t i = 0; i < m->library_count; ++i)
            library_indices[i] = prx_library_register_loaded(m, &m->libraries[i]);
    }

    /* Step C — publish exports as NID -> (base + vaddr) so a title's import
     * resolver can dispatch into this module by NID. Function exports point at
     * an OPD in the freshly-placed image; ps3_indirect_call walks it. */
    uint32_t exports_published = 0;
    if (m->exports) {
        for (uint32_t i = 0; i < m->export_count; i++) {
            const prx_export* e = &m->exports[i];
            if (e->nid == 0)
                continue;
            uint32_t library_index = UINT32_MAX;
            if (library_indices && e->library_index < m->library_count)
                library_index = library_indices[e->library_index];
            prx_export_register(e->nid, m->base + e->vaddr, library_index);
            exports_published++;
        }
    }
    free(library_indices);

    r.ok = 1;
    fprintf(stderr,
            "[prx] loaded %-12s base=0x%08X size=%u (0x%X) funcs=%u/%llu exports=%u\n",
            m->name ? m->name : "?", m->base, m->image_size, m->image_size,
            r.funcs_registered, (unsigned long long)m->func_count,
            exports_published);
    return r;
}

uint8_t* prx_image_load_file(const char* path, uint32_t* out_size)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[prx] cannot open image '%s'\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n <= 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    uint8_t* buf = (uint8_t*)malloc((size_t)n);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "[prx] short read on '%s'\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (out_size)
        *out_size = (uint32_t)n;
    return buf;
}
