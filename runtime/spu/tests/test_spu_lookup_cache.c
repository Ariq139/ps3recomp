#include "../spu_interp.h"
#include <assert.h>
#include <stdio.h>

extern void spu_begin_image(int);
extern void spu_register_function(uint32_t, spu_lifted_fn);
uint8_t* vm_base;
uint32_t ppu_vm_size;
static spu_context context;
static void entry_a(spu_context* ctx) { (void)ctx; }
static void entry_b(spu_context* ctx) { (void)ctx; }

int main(void)
{
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
    assert(!spu_lifted_lookup(&context, 0x480)); /* same cache slot */
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
