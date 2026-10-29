#include "rsx_vertex_decode_cache.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static rsx_vertex_decode_cache worker[2];

int main(void)
{
    unsigned char expected[256];
    for (unsigned i = 0; i < sizeof expected; i++) expected[i] = (unsigned char)(i * 37u);
    assert(!rsx_vertex_decode_cache_find(&worker[0], 1, 7));
    memcpy(rsx_vertex_decode_cache_store(&worker[0], 1, 7), expected, sizeof expected);
    assert(memcmp(rsx_vertex_decode_cache_find(&worker[0], 1, 7), expected, sizeof expected) == 0);
    assert(!rsx_vertex_decode_cache_find(&worker[1], 1, 7));
    assert(!rsx_vertex_decode_cache_find(&worker[0], 2, 7));
    assert(!rsx_vertex_decode_cache_find(&worker[0], 0, 7));
    const uint32_t collision = 7 + RSX_VERTEX_DECODE_CACHE_SLOTS;
    memset(rsx_vertex_decode_cache_store(&worker[0], 1, collision), 0xA5, sizeof expected);
    assert(!rsx_vertex_decode_cache_find(&worker[0], 1, 7));
    assert(rsx_vertex_decode_cache_find(&worker[0], 1, collision));
    memcpy(rsx_vertex_decode_cache_store(&worker[0], 2, 7), expected, 32);
    assert(memcmp(rsx_vertex_decode_cache_find(&worker[0], 2, 7), expected, 32) == 0);
    assert(!rsx_vertex_decode_cache_find(&worker[0], 1, 7));
    puts("vertex decode cache: payload, collision, draw invalidation and worker isolation passed");
    return 0;
}
