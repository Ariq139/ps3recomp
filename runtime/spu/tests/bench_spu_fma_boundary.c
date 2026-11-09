/* Optional observation-only benchmark: frozen previous TwoSum AVX2 helper vs
 * current lazy boundary-exactness helper. Baseline is valid under RN only.
 * Same immutable corpus per pair, alternating order; output bytes must agree.
 * Corpora: ordinary balanced normals; zeros/unit coefficients; exact integers;
 * arbitrary bits. Synthetic timing does not establish game FPS improvement. */
#define _POSIX_C_SOURCE 200809L
#include "../spu_helpers.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#if !defined(__AVX2__) || defined(SPU_SCALAR_HELPERS)
int main(void) {
  puts("SKIP: benchmark requires AVX2");
  return 0;
}
#else
SPU__X8_INLINE __m128i baseline_trunc0(__m256d d, __m128i *ok, __m128i *zm,
                                       const __m256d *err) {
  const __m256i u = _mm256_castpd_si256(d);
  const __m256i az =
      _mm256_cmpeq_epi64(_mm256_slli_epi64(u, 1), _mm256_setzero_si256());
  const __m256i e = _mm256_sub_epi64(
      _mm256_and_si256(_mm256_srli_epi64(u, 52), _mm256_set1_epi64x(0x7FF)),
      _mm256_set1_epi64x(1023));
  __m256i good = _mm256_andnot_si256(
      _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_set1_epi64x(-126), e),
                      _mm256_cmpgt_epi64(e, _mm256_set1_epi64x(127))),
      _mm256_set1_epi64x(-1));
  if (err) {
    const __m256i low = _mm256_and_si256(u, _mm256_set1_epi64x(0x1FFFFFFF));
    const __m256i bnd = _mm256_cmpeq_epi64(low, _mm256_setzero_si256());
    const __m256i rnd = _mm256_castpd_si256(
        _mm256_cmp_pd(*err, _mm256_setzero_pd(), _CMP_NEQ_UQ));
    good = _mm256_andnot_si256(_mm256_and_si256(bnd, rnd), good);
  }
  *ok = spu__x8_narrow(_mm256_or_si256(good, az));
  *zm = spu__x8_narrow(az);
  const __m256d t = _mm256_castsi256_pd(
      _mm256_and_si256(u, _mm256_set1_epi64x(~0x1FFFFFFFll)));
  return _mm_andnot_si128(_mm_and_si128(*zm, _mm_set1_epi32((int)0x80000000u)),
                          _mm_castps_si128(_mm256_cvtpd_ps(t)));
}
SPU__X8_INLINE u128 baseline_fma(u128 a, u128 b, u128 c, uint32_t negp,
                                 uint32_t negc) {
  const __m128i x = _mm_xor_si128(spu__x8_ld(a), _mm_set1_epi32((int)negp)),
                y = spu__x8_ld(b);
  const __m128i z = _mm_xor_si128(spu__x8_ld(c), _mm_set1_epi32((int)negc));
  const __m256d p = _mm256_mul_pd(spu__x8_wide(x), spu__x8_wide(y)),
                zz = spu__x8_wide(z);
  const __m256d sum = _mm256_add_pd(p, zz), bb = _mm256_sub_pd(sum, p);
  const __m256d err = _mm256_add_pd(_mm256_sub_pd(p, _mm256_sub_pd(sum, bb)),
                                    _mm256_sub_pd(zz, bb));
  __m128i ok, zm;
  const __m128i r = baseline_trunc0(sum, &ok, &zm, &err);
  ok = _mm_and_si128(
      ok, _mm_and_si128(_mm_and_si128(spu__x8_plain0(x), spu__x8_plain0(y)),
                        spu__x8_plain0(z)));
  return spu__x8_all4(ok)
             ? spu__x8_st(r)
             : spu__x8_fma_fix(spu__x8_st(r), ok, a, b, c, negp, negc);
}
static uint64_t s_rng = 0x2545F4914F6CDD1Dull;
static inline uint32_t rnd32(void) {
  s_rng ^= s_rng << 13;
  s_rng ^= s_rng >> 7;
  s_rng ^= s_rng << 17;
  return (uint32_t)(s_rng >> 16);
}

static inline uint32_t word(uint32_t sign, uint32_t exp, uint32_t frac) {
  return sign << 31 | exp << 23 | frac;
}
#define N 2048
static u128 av[N], bv[N], cv[N], out[N];
static volatile unsigned sink;
static double seconds(void) {
#ifdef _WIN32
  LARGE_INTEGER h, t;
  QueryPerformanceFrequency(&h);
  QueryPerformanceCounter(&t);
  return (double)t.QuadPart / h.QuadPart;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}
#define BENCH(name, EXPR)                                                      \
  __attribute__((noinline)) static double name(int rounds) {                   \
    double t0 = seconds();                                                     \
    for (int r = 0; r < rounds; r++) {                                         \
      __asm__ __volatile__("" ::: "memory");                                   \
      for (int i = 0; i < N; i++) {                                            \
        u128 a = av[i], b = bv[i], c = cv[i];                                  \
        out[i] = (EXPR);                                                       \
      }                                                                        \
    }                                                                          \
    double t1 = seconds();                                                     \
    unsigned h = 0;                                                            \
    for (int i = 0; i < N; i++)                                                \
      h ^= out[i]._u32[0];                                                     \
    sink = h;                                                                  \
    return (t1 - t0) * 1e9 / (rounds * (double)N);                             \
  }
BENCH(oldbench, baseline_fma(a, b, c, 0, 0))
BENCH(newbench, spu_fma(a, b, c))
static void fill(int kind) {
  for (int i = 0; i < N; i++)
    for (int k = 0; k < 4; k++) {
      unsigned x = word(rnd32() & 1, 123 + rnd32() % 9, rnd32() & 0x7fffff),
               y = word(rnd32() & 1, 123 + rnd32() % 9, rnd32() & 0x7fffff),
               z = word(rnd32() & 1, 123 + rnd32() % 9, rnd32() & 0x7fffff);
      if (kind == 1) {
        if (rnd32() % 4 == 0)
          x = 0;
        if (rnd32() % 4 == 0)
          y = 0x3f800000;
        if (rnd32() % 4 == 0)
          z = 0;
      }
      if (kind == 2) {
        float f = (int)(rnd32() % 17) - 8;
        memcpy(&x, &f, 4);
        f = (int)(rnd32() % 17) - 8;
        memcpy(&y, &f, 4);
        f = (int)(rnd32() % 17) - 8;
        memcpy(&z, &f, 4);
      }
      if (kind == 3) {
        x = rnd32();
        y = rnd32();
        z = rnd32();
      }
      av[i]._u32[k] = x;
      bv[i]._u32[k] = y;
      cv[i]._u32[k] = z;
    }
}

int main(void) {
  unsigned saved = _mm_getcsr();
  _mm_setcsr(saved & ~(0x6000u | 0x8040u));
  static u128 first_out[N];
  for (int kind = 0; kind < 4; kind++) {
    fill(kind);
    for (int pair = 0; pair < 6; pair++) {
      double old, current;
      if (pair & 1) {
        current = newbench(8192);
        memcpy(first_out, out, sizeof(out));
        old = oldbench(8192);
      } else {
        old = oldbench(8192);
        memcpy(first_out, out, sizeof(out));
        current = newbench(8192);
      }
      assert(memcmp(first_out, out, sizeof(out)) == 0);
      printf("corpus=%d pair=%d old=%.3fns current=%.3fns ratio=%.4f\n", kind,
             pair, old, current, current / old);
      fflush(stdout);
    }
  }
  _mm_setcsr(saved);
  return 0;
}
#endif
