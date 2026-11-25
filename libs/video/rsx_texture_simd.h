#ifndef PS3RECOMP_RSX_TEXTURE_SIMD_H
#define PS3RECOMP_RSX_TEXTURE_SIMD_H
#include <stdint.h>
#if !defined(RSX_TEXTURE_FORCE_SCALAR) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define RSX_TEXTURE_SSE2 1
#include <emmintrin.h>

/* Exact floors, not bit replication: 255/31 = 8 + 7/31,
 * 255/63 = 4 + 3/63. The rounded-up Q16 factors are exact
 * over the complete 5-bit and 6-bit input domains. */
static inline __m128i rsx_expand5x8(__m128i v)
{
    return _mm_add_epi16(_mm_slli_epi16(v, 3),
        _mm_mulhi_epu16(v, _mm_set1_epi16(14799)));
}
static inline __m128i rsx_expand6x8(__m128i v)
{
    return _mm_add_epi16(_mm_slli_epi16(v, 2),
        _mm_mulhi_epu16(v, _mm_set1_epi16(3121)));
}
/* Exactly eight guest big-endian pixels; caller checks the format and row
 * extent. All loads and stores permit unaligned source/destination. */
static inline void rsx_decode_packed16x8(uint8_t* dst, const uint8_t* src, uint32_t fmt)
{
    const __m128i v0 = _mm_loadu_si128((const __m128i*)src);
    const __m128i v = _mm_or_si128(_mm_slli_epi16(v0, 8), _mm_srli_epi16(v0, 8));
    const __m128i m5 = _mm_set1_epi16(31), m4 = _mm_set1_epi16(15);
    const __m128i one = _mm_set1_epi16(1), full = _mm_set1_epi16(255);
    __m128i r, g, b, a = full;
    if (fmt == 0x83u) {
        const __m128i r4 = _mm_and_si128(_mm_srli_epi16(v, 8), m4);
        const __m128i g4 = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
        const __m128i b4 = _mm_and_si128(v, m4);
        const __m128i a4 = _mm_srli_epi16(v, 12);
        r = _mm_or_si128(r4, _mm_slli_epi16(r4, 4));
        g = _mm_or_si128(g4, _mm_slli_epi16(g4, 4));
        b = _mm_or_si128(b4, _mm_slli_epi16(b4, 4));
        a = _mm_or_si128(a4, _mm_slli_epi16(a4, 4));
    } else if (fmt == 0x84u) {
        r = rsx_expand5x8(_mm_srli_epi16(v, 11));
        g = rsx_expand6x8(_mm_and_si128(_mm_srli_epi16(v, 5), _mm_set1_epi16(63)));
        b = rsx_expand5x8(_mm_and_si128(v, m5));
    } else if (fmt == 0x97u) {
        r = rsx_expand5x8(_mm_srli_epi16(v, 11));
        g = rsx_expand5x8(_mm_and_si128(_mm_srli_epi16(v, 6), m5));
        b = rsx_expand5x8(_mm_and_si128(_mm_srli_epi16(v, 1), m5));
        a = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(v, one), one), full);
    } else {
        r = rsx_expand5x8(_mm_and_si128(_mm_srli_epi16(v, 10), m5));
        g = rsx_expand5x8(_mm_and_si128(_mm_srli_epi16(v, 5), m5));
        b = rsx_expand5x8(_mm_and_si128(v, m5));
        if (fmt == 0x82u) a = _mm_and_si128(_mm_srai_epi16(v, 15), full);
    }
    const __m128i rg = _mm_unpacklo_epi8(_mm_packus_epi16(r, r), _mm_packus_epi16(g, g));
    const __m128i ba = _mm_unpacklo_epi8(_mm_packus_epi16(b, b), _mm_packus_epi16(a, a));
    _mm_storeu_si128((__m128i*)dst, _mm_unpacklo_epi16(rg, ba));
    _mm_storeu_si128((__m128i*)(dst + 16), _mm_unpackhi_epi16(rg, ba));
}
/* Lowest Morton bits are x then y, so one 16-byte 2x2 tile contains two
 * top-row pixels followed by two bottom-row pixels. */
static inline void rsx_decode_argb_tile2x2(uint8_t* top, uint8_t* bottom,
    const uint8_t* src, int already_rgba, int opaque)
{
    __m128i v = _mm_loadu_si128((const __m128i*)src);
    if (!already_rgba) v = _mm_or_si128(_mm_srli_epi32(v, 8), _mm_slli_epi32(v, 24));
    if (opaque) v = _mm_or_si128(v, _mm_set1_epi32((int)0xFF000000u));
    _mm_storel_epi64((__m128i*)top, v);
    _mm_storel_epi64((__m128i*)bottom, _mm_srli_si128(v, 8));
}
#else
#define RSX_TEXTURE_SSE2 0
#endif
#endif
