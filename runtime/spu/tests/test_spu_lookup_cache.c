#include "../spu_interp.h"
#include <assert.h>
#include <stdio.h>

extern void spu_begin_image(int);
extern void spu_register_function(uint32_t, spu_lifted_fn);
extern uint32_t test_spu_lookup_cache_hash(uint32_t, int);
uint8_t* vm_base;
uint32_t ppu_vm_size;
static spu_context context;
static void entry_a(spu_context* ctx) { (void)ctx; }
static void entry_b(spu_context* ctx) { (void)ctx; }

int main(void)
{
    /* 0x80 and 0x480 collide in the original 256-entry index. The optional
     * mixer should spread this representative pair while preserving exact
     * key checks in either mode. */
#if defined(PS3RECOMP_SPU_LOOKUP_CACHE_HASH_MIX) && \
    PS3RECOMP_SPU_LOOKUP_CACHE_HASH_MIX
    assert((test_spu_lookup_cache_hash(0x80, 7) & 255u) !=
           (test_spu_lookup_cache_hash(0x480, 7) & 255u));
#else
    assert((test_spu_lookup_cache_hash(0x80, 7) & 255u) ==
           (test_spu_lookup_cache_hash(0x480, 7) & 255u));
#endif
    context.image_id = 7;
    assert(!spu_lifted_lookup(&context, 0x80));
    spu_begin_image(7);
    spu_register_function(0x80, entry_a);
    for (int i = 0; i < 1000; ++i) assert(spu_lifted_lookup(&context, 0x80) == entry_a);
    context.image_id = 8;
    assert(!spu_lifted_lookup(&context, 0x80));
    spu_begin_image(8);
    spu_register_function(0x80, entry_b);
    assert(spu_lifted_lookup(&context, 0x80) == entry_b);
    context.image_id = 7;
    assert(spu_lifted_lookup(&context, 0x80) == entry_a);
    assert(!spu_lifted_lookup(&context, 0x480)); /* unregistered miss */
    assert(spu_lifted_lookup(&context, 0x80) == entry_a);
    spu_begin_image(0);
    spu_register_function(0x84, entry_b);
    assert(spu_lifted_lookup(&context, 0x84) == entry_b);
    assert(spu_lifted_lookup(NULL, 0x80) == entry_a);
    spu_begin_image(7);
    spu_register_function(0x84, entry_a);
    assert(spu_lifted_lookup(&context, 0x84) == entry_b); /* retain registry ordering */
    puts("SPU lookup cache: late registration, images, wildcards, collisions, ordering passed");
    return 0;
}
