#include "rsx_fp_decompiler.h"
#include "rsx_fp_metadata_cache.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static u32 random_state = 0x825F13u;
static u32 next_random(void)
{
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

static void put_word(u8* p, u32 word)
{
    const u32 swapped = (word << 16) | (word >> 16);
    p[0] = (u8)(swapped >> 24); p[1] = (u8)(swapped >> 16);
    p[2] = (u8)(swapped >> 8); p[3] = (u8)swapped;
}

static rsx_fp_metadata_cache metadata_cache;

static void check_metadata_changes(void)
{
    u8 code[4096] = {0};
    put_word(code, 0x01000000u);   /* MOV + CONST payload, no END */
    put_word(code + 4, 2u);
    put_word(code + 16, 0x3f800000u);
    put_word(code + 32, 0x17000001u | (5u << 17)); /* texture op + END */
    const rsx_fp_metadata* m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->bytes == 48 && m->constants.count == 1 && m->offsets[0] == 16);
    const u32 original_hash = m->hash;
    assert(m->constants.values[0][0] == 0x3f800000u);
    put_word(code + 16, 0x7fc00000u);
    m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->hash == original_hash && m->constants.values[0][0] == 0);
    put_word(code + 32, 0x01000001u);
    m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->bytes == 48 && m->hash == rsx_fp_code_hash(code, 48));
    assert(m->units == rsx_fp_texture_units(code, 48));
    put_word(code, 0x01000001u); /* earlier END still includes payload */
    m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->bytes == 32 && m->constants.count == 1);
    put_word(code + 4, 0u);       /* changed CONST operand layout */
    m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->bytes == 16 && m->constants.count == 0);
    memset(code, 0, sizeof(code)); /* no END in the entire bound */
    assert(!rsx_fp_metadata_get(&metadata_cache, code));

    for (u32 i = 0; i < RSX_FP_METADATA_OFFSETS; ++i) {
        put_word(code + i * 32u, 0x01000000u | (i + 1u == RSX_FP_METADATA_OFFSETS));
        put_word(code + i * 32u + 4u, 2u);
    }
    m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->bytes == sizeof(code) && m->constants.count == RSX_FP_METADATA_OFFSETS);
    assert(m->offsets[m->constants.count - 1u] == sizeof(code) - 16u);
    put_word(code + sizeof(code) - 4u, 0x40000000u);
    m = rsx_fp_metadata_get(&metadata_cache, code);
    assert(m && m->constants.values[m->constants.count - 1u][3] == 0x40000000u);
}

int main(void)
{
    check_metadata_changes();
    const u32 ops[] = {1, 0x17, 0x18, 0x19, 0x2F, 0x31, 0x33, 0x34};
    for (u32 trial = 0; trial < 3000; trial++) {
        u8 code[4096] = {0};
        u32 offsets[64], instructions = 1 + next_random() % 64u, size = 0;
        for (u32 i = 0; i < instructions; i++) {
            const u32 branch = (next_random() >> 16) & 1u;
            const u32 constant = (next_random() >> 16) & 1u;
            const u32 opcode = ops[next_random() % (sizeof ops / sizeof ops[0])];
            const u32 w0 = (opcode << 24) | ((next_random() % 16u) << 17) |
                (i + 1u == instructions ? 1u : 0u);
            offsets[i] = size;
            put_word(code + size, w0);
            put_word(code + size + 4, constant ? 2u : 0u);
            put_word(code + size + 8, branch ? 0x80000000u : 0u);
            size += 16;
            if (constant && !branch) {
                for (u32 component = 0; component < 4; component++)
                    put_word(code + size + component * 4u, next_random());
                size += 16;
            }
        }
        u32 expected_hash = 2166136261u;
        for (u32 i = 0; i < instructions; i++)
            for (u32 byte = 0; byte < 16; byte++)
                expected_hash = (expected_hash ^ code[offsets[i] + byte]) * 16777619u;
        u32 bytes = 0, units = 0;
        assert(rsx_fp_code_hash_ex(code, size, &bytes, &units) == expected_hash);
        assert(bytes == size && bytes == rsx_fp_program_size(code, size));
        assert(units == rsx_fp_texture_units(code, size));
        const rsx_fp_metadata* metadata=rsx_fp_metadata_get(&metadata_cache,code);
        assert(metadata && metadata->hash==expected_hash && metadata->bytes==size && metadata->units==units);
        rsx_fp_constant_block expected_constants;
        assert(rsx_fp_collect_constants(code,size,&expected_constants)>=0);
        assert(metadata->constants.count==expected_constants.count);
        assert(!memcmp(metadata->constants.values,expected_constants.values,expected_constants.count*16u));
        assert(rsx_fp_metadata_get(&metadata_cache,code)==metadata);
        for (u32 instruction=0;instruction<instructions;++instruction) {
            const u32 next=instruction+1<instructions ? offsets[instruction+1] : size;
            if (next-offsets[instruction]!=32u) continue;
            const u32 k=offsets[instruction]+16u;
            code[k]^=0x55u;
            metadata=rsx_fp_metadata_get(&metadata_cache,code);
            assert(metadata && metadata->hash==expected_hash);
            assert(rsx_fp_collect_constants(code,size,&expected_constants)>=0);
            assert(!memcmp(metadata->constants.values,expected_constants.values,expected_constants.count*16u));
            code[k]^=0x55u;
            metadata=rsx_fp_metadata_get(&metadata_cache,code);
            assert(metadata && metadata->hash==expected_hash);
            break;
        }


        rsx_fp_texture_usage usage;
        assert(rsx_fp_analyze_texture_usage(code, size, &usage));
        assert(usage.sampled_units == units);
        u8 masks[16];
        for (u32 unit = 0; unit < 16; unit++) masks[unit] = (u8)(next_random() >> 24);
        u32 gamma = 0, expand = 0;
        for (u32 unit = 0; unit < 16; unit++) if (masks[unit]) {
            gamma |= usage.gamma_candidates & (1u << unit);
            expand |= usage.expand_candidates & (1u << unit);
        }
        assert(gamma == rsx_fp_texture_gamma_units(code, size, masks));
        assert(expand == rsx_fp_texture_expand_units(code, size, masks));
        (void)rsx_fp_code_hash_ex(code, size - 1u, &bytes, &units);
        assert(bytes == 0 && units == ~0u);
    }
    puts("FP runtime analysis: 3000 programs matched code hashes, sizes, texture masks and truncation handling");
    return 0;
}
