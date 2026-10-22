#include "../rsx_vertex_fetch.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static u8 memory[128];
u8* vm_base = memory;
static rsx_state state;
u32 cellGcmResolveOffset(u32 off) { return off; }
u32 cellGcmResolveLocated(int local, u32 off) { (void)local; return off; }

static void put32(u32 off, u32 value)
{
    for (u32 k = 0; k < 4; ++k) memory[off + k] = (u8)(value >> (24 - 8*k));
}

int main(void)
{
    float out[4];
    state.vertex_data4f[0][0] = 9.0f;
    state.vertex_data4f[0][2] = 7.0f;
    state.vertex_data4f[0][3] = 6.0f;
    rsx_fetch_attrib(&state, 0, 17, out);
    assert(out[0] == 9.0f && out[2] == 7.0f && out[3] == 6.0f);

    state.vertex_attribs[0].enabled = 1;
    state.vertex_attribs[0].type = 2;
    state.vertex_attribs[0].size = 2;
    state.vertex_attribs[0].offset = 16;
    state.vertex_attribs[0].stride = 0;
    put32(16, 0x3F800000u); put32(20, 0x40000000u);
    rsx_fetch_attrib(&state, 0, 999, out);
    assert(out[0] == 1.0f && out[1] == 2.0f && out[2] == 0.0f && out[3] == 1.0f);

    state.vertex_attribs[0].type = 6;
    state.vertex_attribs[0].size = 1;
    put32(16, (511u << 22) | (1024u << 11) | 1023u);
    rsx_fetch_attrib(&state, 0, 0, out);
    assert(out[0] == 32736.0f/32767.0f);
    assert(out[1] == -32768.0f/32767.0f);
    assert(out[2] == 32704.0f/32767.0f && out[3] == 1.0f);

    state.vertex_attribs[0].type = 1;
    state.vertex_attribs[0].size = 3;
    const u8 snorm[] = {0x80,0, 0x7F,0xFF, 0,0};
    memcpy(memory + 16, snorm, sizeof(snorm));
    rsx_fetch_attrib(&state, 0, 0, out);
    assert(out[0] == -1.0f && out[1] == 1.0f);
    assert(out[2] == 0.5f/32767.5f && out[3] == 1.0f);

    /* All binary16 patterns, against the mathematical value of the format. */
    for (u32 bits = 0; bits <= 65535u; ++bits) {
        u8 bytes[2] = {(u8)(bits >> 8), (u8)bits};
        float value = rsx_rd_half_be(bytes);
        u32 exponent = (bits >> 10) & 31u, fraction = bits & 1023u;
        if (exponent == 31u) {
            assert(fraction ? isnan(value) : isinf(value));
        } else {
            float expected = exponent ? ldexpf((float)(1024u + fraction), (int)exponent - 25)
                                      : ldexpf((float)fraction, -24);
            if (bits & 32768u) expected = -expected;
            assert(value == expected && !!signbit(value) == !!signbit(expected));
        }
    }
    puts("vertex fetch: defaults, zero stride, CMP, SNORM16, all 65536 half patterns passed");
    return 0;
}
