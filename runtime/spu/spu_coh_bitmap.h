/*
 * ps3recomp - the SPU reserved-line bitmap, for inline store paths
 *
 * Every guest store asks whether an SPU has reserved its 128-byte line
 * (spu_coherency.h). The PPU store paths test the bitmap here, inline, rather
 * than through a call into spu_coherency.c.
 *
 * g_spu_coh_bits is NULL until an SPU reserves its first line, so a title that
 * runs no SPU code pays one load and branch per store. It is one merged
 * definition per program (selectany / weak): a target that links the store
 * paths without spu_coherency.c, such as the PPU loader fixtures, sees NULL
 * and never a reserved line.
 */

#ifndef SPU_COH_BITMAP_H
#define SPU_COH_BITMAP_H

#include <stdint.h>

/* selectany (a COMDAT "any": one definition kept, every object sees the same
 * variable) under MSVC and MinGW. MinGW's weak definition is a PE weak external
 * whose default is a per-object `.weak.<name>.<object>` symbol; linking C++
 * objects that reach it through the .refptr indirection ended with the symbol
 * undefined. */
#if defined(_MSC_VER) || defined(__MINGW32__)
#  define SPU_COH_SHARED __declspec(selectany)
#else
#  define SPU_COH_SHARED __attribute__((weak))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPU_COH_LINE_SHIFT  7   /* 128-byte lines */

/* Atomically add one reservation bit without clobbering a different bit set
 * concurrently in the same byte. The CAS loop avoids a write when this bit
 * is already present, which keeps the repeated-reservation fast path read
 * only. */
static inline void spu_coh_bitmap_mark_bit(unsigned char* byte, unsigned char bit)
{
    unsigned char old = __atomic_load_n(byte, __ATOMIC_RELAXED);
    while (!(old & bit) &&
           !__atomic_compare_exchange_n(byte, &old, (unsigned char)(old | bit),
                                        1, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
    }
}

/* One bit per line over the 32-bit guest address space; spu_coherency.c
 * publishes it on the first reservation and only ever sets bits. */
SPU_COH_SHARED const unsigned char* g_spu_coh_bits = 0;

static inline int spu_coh_line_reserved(uint32_t addr)
{
    const unsigned char* bits = __atomic_load_n(&g_spu_coh_bits, __ATOMIC_ACQUIRE);
    if (!bits) return 0;
    const uint32_t line = addr >> SPU_COH_LINE_SHIFT;
    return (__atomic_load_n(&bits[line >> 3], __ATOMIC_ACQUIRE) >> (line & 7)) & 1;
}

/* First marked line intersecting the inclusive guest address range.
 * UINT32_MAX is not a line-aligned address, so it is an unambiguous sentinel.
 * Match the byte-sized atomic stores in spu_coherency.c: casting to a wider
 * word would overlap separately published atomic objects. Marks never clear.
 * A caller must scan again after an unlocked guest write, not reuse a result
 * from before the write. */
static inline uint32_t spu_coh_next_reserved(uint32_t first, uint32_t last)
{
    if (first > last) return UINT32_MAX;
    const unsigned char* bits = __atomic_load_n(&g_spu_coh_bits, __ATOMIC_ACQUIRE);
    if (!bits) return UINT32_MAX;
    const uint32_t lo = first >> SPU_COH_LINE_SHIFT;
    const uint32_t hi = last >> SPU_COH_LINE_SHIFT;
    const uint32_t b0 = lo >> 3, b1 = hi >> 3;
    for (uint32_t b = b0; b <= b1; ++b) {
        unsigned marked = __atomic_load_n(&bits[b], __ATOMIC_ACQUIRE);
        if (b == b0) marked &= 0xFFu << (lo & 7u);
        if (b == b1) marked &= 0xFFu >> (7u - (hi & 7u));
        if (marked)
            return ((b << 3) + (uint32_t)__builtin_ctz(marked)) << SPU_COH_LINE_SHIFT;
    }
    return UINT32_MAX;
}

#ifdef __cplusplus
}
#endif

#endif /* SPU_COH_BITMAP_H */
