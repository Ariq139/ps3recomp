/* Exercise every low control-bit combination and hostile upper bits against
 * the original scalar ISA definitions. SIMD must preserve SPU byte order. */
#include "../spu_helpers.h"
#include <stdio.h>

static uint32_t rng = 0x19937u;
static uint32_t next(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static unsigned checks;
static int same(const char* name, u128 got, u128 want, uint32_t control) {
    ++checks;
    if (!memcmp(&got, &want, 16)) return 1;
    printf("%s mismatch control=%08X\n", name, control); return 0;
}
#define CHECK(op, a, c) do { if (!same(#op, op(a, c), op##_ref(a, c), control)) return 1; } while (0)

int main(void) {
    for (unsigned trial = 0; trial < 512; ++trial) {
        u128 a;
        for (unsigned lane = 0; lane < 16; ++lane)
            a._u8[lane] = trial ? (uint8_t)next() : (uint8_t)(lane + 1);
        for (uint32_t low = 0; low < 256; ++low) {
            /* Bit-indexed forms use bits 3..7. Upper bits and other lanes
             * must not affect a byte permutation. Avoid signed-negation UB
             * in the original reference for the single INT_MIN value. */
            uint32_t control = trial % 2 ? (next() & 0x7FFFFF00u) | low : low;
            if (trial % 3 == 0) control |= 0x80000100u;
            u128 b = {{0}};
            for (unsigned lane = 0; lane < 4; ++lane) b._u32[lane] = next();
            b._u32[0] = control;
            CHECK(spu_rotqby, a, b); CHECK(spu_shlqby, a, b);
            CHECK(spu_rotqmby, a, b); CHECK(spu_rotqbybi, a, b);
            CHECK(spu_shlqbybi, a, b); CHECK(spu_rotqmbybi, a, b);
            int immediate = (int)low - 128;
            CHECK(spu_rotqbyi, a, immediate); CHECK(spu_shlqbyi, a, immediate);
            CHECK(spu_rotqmbyi, a, immediate);
        }
    }
    printf("SPU byte permutations: %u full-vector comparisons passed\n", checks);
    return 0;
}
