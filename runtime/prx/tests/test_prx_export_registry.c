/* Synthetic PRX export registry test: no proprietary module or game data. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../prx_loader.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static uint8_t memory[4096];
uint8_t* vm_base = memory;

int main(void)
{
    uint8_t image[64] = {0};
    image[0] = 0x11;
    image[32] = 0x22;
    const prx_library_descriptor libs[] = {{0, "first"}, {32, "second"}};
    /* NIDs 2 and 2586 collide in the proposed high-bit multiplicative hash.
     * NID zero must not publish even when present in a module's exports. */
    const prx_export exports[] = {{2, 28, 0}, {2586, 60, 1}, {0, 4, 0}};
    const prx_module m = {
        .name = "fixture", .base = 0x100, .image = image,
        .image_size = sizeof image, .exports = exports, .export_count = 3,
        .libraries = libs, .library_count = 2
    };
    assert(prx_resolve_export(2) == 0 && prx_export_registry_count() == 0);
    assert(prx_load_module(&m, NULL).ok);
    assert(prx_export_registry_count() == 2);
    assert(prx_resolve_export(0) == 0);
    assert(prx_resolve_export(2) == 0x11c);
    assert(prx_resolve_export(2586) == 0x13c);
    assert(prx_resolve_export(123) == 0);
    assert(!prx_export_library_is_registered(2));
    assert(!prx_export_library_is_registered(2586));
    const char* name = NULL;
    assert(prx_register_library_descriptor(memory + 0x100, &name));
    assert(name && !strcmp(name, "first"));
    assert(prx_export_library_is_registered(2));
    assert(!prx_export_library_is_registered(2586));

    /* A later module replaces an existing export, including its library. */
    const prx_export replacement[] = {{2, 60, 0}};
    const prx_module later = {
        .name = "later", .base = 0x200, .image = image,
        .image_size = sizeof image, .exports = replacement, .export_count = 1,
        .libraries = libs + 1, .library_count = 1
    };
    assert(prx_load_module(&later, NULL).ok);
    assert(prx_export_registry_count() == 2);
    assert(prx_resolve_export(2) == 0x23c);
    assert(!prx_export_library_is_registered(2));
    assert(prx_resolve_export(2586) == 0x13c);

    /* Fill to the existing capacity, then reject a new NID but still permit
     * duplicate replacement. NID lookup must retain exact 32-bit keys. */
    static prx_export fill[2046];
    for (unsigned i = 0; i < 2046; ++i)
        fill[i] = (prx_export){0x80000000u + i, i % 16u, UINT32_MAX};
    const prx_module full = {
        .name = "fill", .base = 0x300, .image = image,
        .image_size = sizeof image, .exports = fill, .export_count = 2046
    };
    assert(prx_load_module(&full, NULL).ok);
    assert(prx_export_registry_count() == 2048);
    for (unsigned i = 0; i < 2046; ++i) {
        assert(prx_resolve_export(fill[i].nid) == 0x300u + i % 16u);
        assert(!prx_export_library_is_registered(fill[i].nid));
    }
    const prx_export overflow[] = {{0x90000000u, 4, UINT32_MAX},
                                   {2586, 8, UINT32_MAX}};
    const prx_module extra = {
        .name = "overflow", .base = 0x400, .image = image,
        .image_size = sizeof image, .exports = overflow, .export_count = 2
    };
    assert(prx_load_module(&extra, NULL).ok);
    assert(prx_export_registry_count() == 2048);
    assert(prx_resolve_export(0x90000000u) == 0);
    assert(prx_resolve_export(2586) == 0x408);
    assert(!prx_export_library_is_registered(2586));
    assert(prx_resolve_export(2) == 0x23c);
    puts("PASS: PRX registry collisions, misses, replacement, library metadata and capacity");
    return 0;
}
