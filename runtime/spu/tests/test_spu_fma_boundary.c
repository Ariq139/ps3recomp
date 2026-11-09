/* AVX2 FMA/FMS/FNMS against integer SPU references in every MXCSR rounding
 * mode, with FTZ/DAZ both off and on. Include the directed-rounding boundary
 * regression and zeros, extended values, cancellation and wide exponent gaps.
 */
#include "../spu_helpers.h"
#include <stdio.h>
#include <string.h>
static uint64_t state = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (uint32_t)(state >> 16);
}
static const uint32_t edges[] = {
    0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
    0x00800001, 0x3f000000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0xbf800000,
    0x7f000000, 0x7f7fffff, 0x7f800000, 0x7fffffff, 0xff800000, 0xffffffff};
static unsigned long long checks;
static unsigned bad;
static void check(u128 a, u128 b, u128 c) {
  for (int op = 0; op < 3; op++) {
    u128 got = op == 0   ? spu_fma(a, b, c)
               : op == 1 ? spu_fms(a, b, c)
                         : spu_fnms(a, b, c);
    u128 ref = op == 0   ? spu_fma_ref(a, b, c)
               : op == 1 ? spu_fms_ref(a, b, c)
                         : spu_fnms_ref(a, b, c);
    if (memcmp(&got, &ref, 16) && bad++ < 8)
      fprintf(stderr,
              "FMA mismatch op=%d a=%08x b=%08x c=%08x got=%08x ref=%08x\n", op,
              a._u32[0], b._u32[0], c._u32[0], got._u32[0], ref._u32[0]);
    ++checks;
  }
}
int main(void) {
#if defined(__AVX2__) && !defined(SPU_SCALAR_HELPERS)
  unsigned saved = _mm_getcsr();
  const unsigned modes[] = {_MM_ROUND_NEAREST, _MM_ROUND_DOWN, _MM_ROUND_UP,
                            _MM_ROUND_TOWARD_ZERO};
  for (unsigned m = 0; m < 4; m++)
    for (unsigned ftz = 0; ftz < 4; ftz++) {
      _mm_setcsr((saved & ~(0x6000u | 0x8040u)) | modes[m] |
                 ((ftz & 1) ? 0x8000u : 0) | ((ftz & 2) ? 0x40u : 0));
      u128 a, b, c;
      for (int k = 0; k < 4; k++) {
        a._u32[k] = 0x9c5cb436;
        b._u32[k] = 0xbec63589;
        c._u32[k] = 0x55000000;
      }
      check(a, b,
            c); /* old TwoSum accepted D5000000 instead of D4FFFFFF under RD */
      for (unsigned x = 0; x < sizeof(edges) / sizeof(*edges); x++)
        for (unsigned y = 0; y < sizeof(edges) / sizeof(*edges); y++)
          for (unsigned z = 0; z < sizeof(edges) / sizeof(*edges); z++) {
            for (int k = 0; k < 4; k++) {
              a._u32[k] = edges[(x + k) % 18];
              b._u32[k] = edges[(y + 3 * k) % 18];
              c._u32[k] = edges[(z + 5 * k) % 18];
            }
            check(a, b, c);
          }
      for (unsigned i = 0; i < 500000; i++) {
        for (int k = 0; k < 4; k++) {
          a._u32[k] = rnd();
          b._u32[k] = rnd();
          c._u32[k] = rnd();
          if (i % 4 == 1) {
            a._u32[k] = (rnd() & 0x807fffff) | ((1 + rnd() % 254) << 23);
            b._u32[k] = (rnd() & 0x807fffff) | ((1 + rnd() % 254) << 23);
            c._u32[k] = (rnd() & 0x807fffff) | ((1 + rnd() % 254) << 23);
          }
          if (i % 4 == 2) {
            b._u32[k] = 0x3f800000;
            c._u32[k] = (a._u32[k] ^ 0x80000000) + rnd() % 3 - 1;
          }
          if (i % 4 == 3) {
            a._u32[k] = (rnd() & 0x80000000) | ((1 + rnd() % 254) << 23);
            b._u32[k] = (rnd() & 0x80000000) | ((1 + rnd() % 254) << 23);
          }
        }
        check(a, b, c);
      }
      printf("FMA MXCSR mode=%u FTZ/DAZ=%u cumulative=%llu vectors bad=%u\n", m,
             ftz, checks, bad);
      fflush(stdout);
    }
  _mm_setcsr(saved);
  return bad ? 1 : 0;
#else
  puts("SKIP: compile with AVX2 to test the vector FMA path");
  return 0;
#endif
}
