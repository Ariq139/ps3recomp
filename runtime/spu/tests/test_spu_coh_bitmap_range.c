/* Differential tests for the production reservation bitmap range scanner.
 * No guest memory is needed: the bitmap covers all 32-bit guest addresses. */
#include "../spu_coh_bitmap.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static unsigned char bitmap[1u << (32 - SPU_COH_LINE_SHIFT - 3)];

static void mark(uint32_t addr)
{
    const uint32_t line = addr >> SPU_COH_LINE_SHIFT;
    __atomic_fetch_or(&bitmap[line >> 3], (unsigned char)(1u << (line & 7u)),
                      __ATOMIC_RELEASE);
}

static uint32_t scalar_next(uint32_t first, uint32_t last)
{
    if (first > last) return UINT32_MAX;
    uint32_t a = first & ~127u;
    const uint32_t end = last & ~127u;
    for (;;) {
        if (spu_coh_line_reserved(a)) return a;
        if (a == end) return UINT32_MAX;
        a += 128u;
    }
}

static void check_range(uint32_t first, uint32_t last)
{
    /* Compare the complete ordered sequence, not just its first match. */
    for (;;) {
        const uint32_t expected = scalar_next(first, last);
        const uint32_t actual = spu_coh_next_reserved(first, last);
        assert(actual == expected);
        if (actual == UINT32_MAX || actual == (last & ~127u)) return;
        first = actual + 128u;
    }
}

int main(void)
{
    assert(spu_coh_next_reserved(0, UINT32_MAX) == UINT32_MAX);
    __atomic_store_n(&g_spu_coh_bits, bitmap, __ATOMIC_RELEASE);
    check_range(0x10000u, 0x14000u);
    check_range(UINT32_MAX, 0);
    check_range(0xFFFFFF00u, UINT32_MAX);

    /* Every range boundary within two adjacent bytes, for every bit pattern. */
    const uint32_t base = 0x10000u;
    const uint32_t b = base >> (SPU_COH_LINE_SHIFT + 3);
    for (unsigned pattern = 0; pattern < 65536u; ++pattern) {
        __atomic_store_n(&bitmap[b], (unsigned char)pattern, __ATOMIC_RELEASE);
        __atomic_store_n(&bitmap[b + 1], (unsigned char)(pattern >> 8), __ATOMIC_RELEASE);
        for (unsigned lo = 0; lo < 16; ++lo)
            for (unsigned hi = lo; hi < 16; ++hi) {
                const uint32_t first = base + lo * 128u + 17u;
                const uint32_t last = base + hi * 128u + 23u;
                assert(spu_coh_next_reserved(first, last) == scalar_next(first, last));
            }
    }
    memset(bitmap, 0, sizeof bitmap);

    /* Sparse and dense marks, partial first/last lines, long DMA ranges,
     * and the final representable guest line. */
    mark(0); mark(base - 128u); mark(base + 128u);
    mark(base + 7u * 128u); mark(base + 8u * 128u);
    mark(base + 128u * 128u); mark(0xFFFFFF80u);
    check_range(0, 0);
    check_range(base, base + 16383u);
    check_range(base + 127u, base + 127u + 16383u);
    check_range(base + 129u, base + 7u * 128u - 1u);
    check_range(0xFFFFF800u, UINT32_MAX);
    check_range(UINT32_MAX, UINT32_MAX);

    /* A mark published after a clean first scan must be visible on rescan. */
    const uint32_t fresh = base + 0x8000u;
    assert(spu_coh_next_reserved(fresh, fresh + 16383u) == UINT32_MAX);
    mark(fresh + 8192u);
    assert(spu_coh_next_reserved(fresh, fresh + 16383u) == fresh + 8192u);

    for (unsigned i = 0; i < 129u; ++i) mark(fresh + i * 128u);
    check_range(fresh + 127u, fresh + 127u + 16383u);
    puts("PASS: reservation bitmap range scanner matches scalar lookup");
    return 0;
}
