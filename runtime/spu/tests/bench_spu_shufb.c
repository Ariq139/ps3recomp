/* Optional paired throughput benchmark. Frozen pre-LUT SSE helper vs current
 * helper, with the same evolving 2048-vector corpus and alternating order.
 * Run correctness separately; timings are observations, never pass/fail gates.
 * Build with SSE4.1/AVX2 so both implementations are available. */
#define _POSIX_C_SOURCE 200809L
#include "../spu_helpers.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static inline u128 baseline_shufb(u128 a, u128 b, u128 c) {
  __m128i A, B, C;
  memcpy(&A, &a, 16);
  memcpy(&B, &b, 16);
  memcpy(&C, &c, 16);
  const __m128i idx = _mm_xor_si128(C, _mm_set1_epi8(3));
  /* BLENDV reads each byte's sign bit; shifting 16-bit lanes by three
   * puts that byte's bit 4 there without interference from its neighbor. */
  const __m128i fromb = _mm_slli_epi16(C, 3);
  __m128i r = _mm_blendv_epi8(_mm_shuffle_epi8(A, idx),
                              _mm_shuffle_epi8(B, idx), fromb);
  /* 10x is already zero. Fill 110 with FF, then override 111 with 80. */
  const __m128i mC0 = _mm_cmpeq_epi8(
      _mm_and_si128(C, _mm_set1_epi8((char)0xC0)), _mm_set1_epi8((char)0xC0));
  const __m128i mE0 = _mm_cmpeq_epi8(
      _mm_and_si128(C, _mm_set1_epi8((char)0xE0)), _mm_set1_epi8((char)0xE0));
  r = _mm_blendv_epi8(_mm_or_si128(r, mC0), _mm_set1_epi8((char)0x80), mE0);
  u128 out;
  memcpy(&out, &r, 16);
  return out;
}
static unsigned rndstate = 0x12345678;
static unsigned rnd(void) {
  rndstate ^= rndstate << 13;
  rndstate ^= rndstate >> 17;
  rndstate ^= rndstate << 5;
  return rndstate;
}
static u128 randv(void) {
  u128 x;
  for (int i = 0; i < 4; i++)
    x._u32[i] = rnd();
  return x;
}
static volatile unsigned sink;
#define N 2048
static u128 initial[N], input[N], bs[N], cs[N];
#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif
static double bench_now(void) {
#ifdef _WIN32
  LARGE_INTEGER hz, q;
  QueryPerformanceFrequency(&hz);
  QueryPerformanceCounter(&q);
  return (double)q.QuadPart / hz.QuadPart;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}
#define BENCH(name, expr)                                                      \
  NOINLINE static double name(int same, int rounds) {                          \
    memcpy(input, initial, sizeof input);                                      \
    double t0 = bench_now();                                                   \
    for (int r = 0; r < rounds; r++)                                           \
      for (int i = 0; i < N; i++) {                                            \
        u128 a = input[i], b = same ? a : bs[i], c = cs[i];                    \
        input[i] = (expr);                                                     \
      }                                                                        \
    double t1 = bench_now();                                                   \
    unsigned h = 0;                                                            \
    for (int i = 0; i < N; i++)                                                \
      h ^= input[i]._u32[0];                                                   \
    sink = h;                                                                  \
    return 1e9 * (t1 - t0) / (N * (double)rounds);                             \
  }
BENCH(oldbench, baseline_shufb(a, b, c))
BENCH(newbench, spu_shufb(a, b, c))
int main(void) {
  for (int i = 0; i < N; i++) {
    initial[i] = randv();
    bs[i] = randv();
    cs[i] = randv();
  }
  for (int same = 0; same < 2; same++) {
    for (int pair = 0; pair < 6; pair++) {
      double old, newv;
      if (pair & 1) {
        newv = newbench(same, 65536);
        old = oldbench(same, 65536);
      } else {
        old = oldbench(same, 65536);
        newv = newbench(same, 65536);
      }
      printf("same=%d pair=%d old=%.4fns lut=%.4fns ratio=%.4f\n", same, pair,
             old, newv, newv / old);
    }
  }
  return 0;
}
