
#include "ppu_recomp.h"
#include <stdio.h>
#include <string.h>
#include <math.h>


static inline uint32_t ppu_f2i32(double v, int rn)
{
    if (v != v) return 0x80000000u;
    double r = rn ? nearbyint(v) : trunc(v);
    if (r >= 2147483647.0)  return 0x7FFFFFFFu;
    if (r <= -2147483648.0) return 0x80000000u;
    return (uint32_t)(int32_t)r;
}
static inline uint64_t ppu_f2i64(double v, int rn)
{
    if (v != v) return 0x8000000000000000ULL;
    double r = rn ? nearbyint(v) : trunc(v);
    if (r >= 9223372036854775807.0)  return 0x7FFFFFFFFFFFFFFFULL;
    if (r <= -9223372036854775808.0) return 0x8000000000000000ULL;
    return (uint64_t)(int64_t)r;
}


static inline double ppu_fp_quiet(double x)
{
    uint64_t b; memcpy(&b, &x, 8);
    b |= 0x0008000000000000ULL;
    memcpy(&x, &b, 8);
    return x;
}
static inline double ppu_fp_default_qnan(void)
{
    uint64_t q = 0x7FF8000000000000ULL; double d;
    memcpy(&d, &q, 8);
    return d;
}

#ifndef PPU_FAST_FP
static inline double ppu_fp_bin(double a, double b, double r)
{
    if (a != a) return ppu_fp_quiet(a);
    if (b != b) return ppu_fp_quiet(b);
    if (r != r) return ppu_fp_default_qnan();
    return r;
}
#else
static inline double ppu_fp_bin(double a, double b, double r) { (void)a; (void)b; return r; }
#endif
static inline double ppu_fadd(double a, double b) { return ppu_fp_bin(a, b, a + b); }
static inline double ppu_fsub(double a, double b) { return ppu_fp_bin(a, b, a - b); }
static inline double ppu_fmul(double a, double b) { return ppu_fp_bin(a, b, a * b); }
static inline double ppu_fdiv(double a, double b) { return ppu_fp_bin(a, b, a / b); }

static inline double ppu_fmadd_core(double a, double c, double b, int neg_b, int neg_res)
{
#ifndef PPU_FAST_FP
    if (a != a) return ppu_fp_quiet(a);
    if (b != b) return ppu_fp_quiet(b);
    if (c != c) return ppu_fp_quiet(c);
    double r = fma(a, c, neg_b ? -b : b);
    if (r != r) return ppu_fp_default_qnan();
#else
    double r = fma(a, c, neg_b ? -b : b);
#endif
    return neg_res ? -r : r;
}

static inline double ppu_fp_single(double r)
{
    return (double)(float)r;
}

static inline double ppu_frsp(double b)
{
    if (b != b) {
        double q = ppu_fp_quiet(b);
        uint64_t u; memcpy(&u, &q, 8);
        u &= ~((1ull << 29) - 1);
        memcpy(&q, &u, 8);
        return q;
    }
    return (double)(float)b;
}

static inline uint32_t ppu_fpscr_sum(uint32_t f)
{
    f &= ~0x60000000u;
    if (f & 0x01F80700u) f |= 0x20000000u;
    if ((f >> 22) & f & 0xF8u) f |= 0x40000000u;
    return f;
}
static inline void ppu_mtfsb(ppu_context* ctx, int bt, int set)
{
    uint32_t m = 0x80000000u >> bt;
    if (bt == 1 || bt == 2) return;
    if (set) {
        
        if (!(ctx->fpscr & m) && (m & 0x1FF80700u)) ctx->fpscr |= 0x80000000u;
        ctx->fpscr |= m;
    } else {
        ctx->fpscr &= ~m;
    }
    ctx->fpscr = ppu_fpscr_sum(ctx->fpscr);
}
static inline void ppu_mtfsf(ppu_context* ctx, uint32_t flm, uint32_t v)
{
    uint32_t m = 0;
    for (int i = 0; i < 8; i++) if (flm & (0x80u >> i)) m |= 0xF0000000u >> (4 * i);
    ctx->fpscr = ppu_fpscr_sum((ctx->fpscr & ~m) | (v & m));
}
static inline uint32_t ppu_mcrfs(ppu_context* ctx, int bfa)
{
    uint32_t sh = 28 - 4 * bfa, field = (ctx->fpscr >> sh) & 0xF;
    
    ctx->fpscr = ppu_fpscr_sum(ctx->fpscr & ~(0x9FF80700u & (0xFu << sh)));
    return field;
}


static inline uint32_t ppu_vbswap32(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}
static inline void ppu_vldf4(const void* v, float o[4]) {
    const uint8_t* p = (const uint8_t*)v;
    for (int i = 0; i < 4; i++) { uint32_t b; memcpy(&b, p + i*4, 4); b = ppu_vbswap32(b); memcpy(&o[i], &b, 4); }
}
static inline void ppu_vstf4(void* v, const float in[4]) {
    uint8_t* p = (uint8_t*)v;
    for (int i = 0; i < 4; i++) { uint32_t b; memcpy(&b, &in[i], 4); b = ppu_vbswap32(b); memcpy(p + i*4, &b, 4); }
}
static inline void ppu_vldu4(const void* v, uint32_t o[4]) {
    const uint8_t* p = (const uint8_t*)v;
    for (int i = 0; i < 4; i++) { uint32_t b; memcpy(&b, p + i*4, 4); o[i] = ppu_vbswap32(b); }
}
static inline void ppu_vstu4(void* v, const uint32_t in[4]) {
    uint8_t* p = (uint8_t*)v;
    for (int i = 0; i < 4; i++) { uint32_t b = ppu_vbswap32(in[i]); memcpy(p + i*4, &b, 4); }
}


#ifdef __cplusplus
extern "C" uint64_t ppu_timebase_now(void);
#else
extern uint64_t ppu_timebase_now(void);
#endif


#ifdef _MSC_VER
#include <intrin.h>

#ifndef __clang__
static inline int __builtin_clz(unsigned int x) {
    
    unsigned long idx;
    if (!x) return 32;
    _BitScanReverse(&idx, x);
    return 31 - (int)idx;
}
static inline int __builtin_clzll(unsigned long long x) {
    unsigned long idx;
    if (!x) return 64;
    _BitScanReverse64(&idx, x);
    return 63 - (int)idx;
}
#endif
static inline int64_t ppc_mulhd(int64_t a, int64_t b) {
    int64_t hi;
    _mul128(a, b, &hi);
    return hi;
}
static inline uint64_t ppc_mulhdu(uint64_t a, uint64_t b) {
    uint64_t hi;
    _umul128(a, b, &hi);
    return hi;
}
#else
static inline int64_t ppc_mulhd(int64_t a, int64_t b) {
    return (int64_t)((__int128)(int64_t)a * (__int128)(int64_t)b >> 64);
}
static inline uint64_t ppc_mulhdu(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * (unsigned __int128)b >> 64);
}
#endif

#include "ppu_vmx.h"   


extern "C" uint8_t* vm_base;


extern "C" void ps3_indirect_call(ppu_context* ctx);

extern "C" void ps3_indirect_call_to(ppu_context* ctx, uint32_t target);


extern "C" void ps3_hle_call(unsigned int nid, ppu_context* ctx);


extern "C" PPU_THREAD_LOCAL void (*g_trampoline_fn)(void*);


#define DRAIN_TRAMPOLINE(ctx) do { \
    while (g_trampoline_fn) { \
        void(*_tf)(void*) = g_trampoline_fn; \
        g_trampoline_fn = 0; \
        _tf((void*)(ctx)); \
    } \
} while(0)


#define PPU_MM_SNAP const ppu_mm _mm = ppu_mm_take(); (void)_mm;
#ifndef PPU_NO_INLINE_LOADS
#undef vm_read8
#undef vm_read16
#undef vm_read32
#undef vm_read64
#define vm_read8(a)  vm_read8_mm(&_mm, a)
#define vm_read16(a) vm_read16_mm(&_mm, a)
#define vm_read32(a) vm_read32_mm(&_mm, a)
#define vm_read64(a) vm_read64_mm(&_mm, a)
#endif
#if !defined(PPU_NO_INLINE_LOADS) && !defined(PPU_NO_INLINE_STORES)
#undef vm_write8
#undef vm_write16
#undef vm_write32
#undef vm_write64
#define vm_write8(a, v)  vm_write8_mm(&_mm, a, v)
#define vm_write16(a, v) vm_write16_mm(&_mm, a, v)
#define vm_write32(a, v) vm_write32_mm(&_mm, a, v)
#define vm_write64(a, v) vm_write64_mm(&_mm, a, v)
#endif



static inline uint64_t ppc_rlw64(uint64_t rs, int sh, int mb, int me, uint64_t ra, int insert) {
    uint32_t w = (uint32_t)rs; sh &= 31;
    uint32_t rw = sh ? ((w << sh) | (w >> (32 - sh))) : w;
    uint64_t r = ((uint64_t)rw << 32) | rw;
    int x = mb + 32, y = me + 32;
    uint64_t m = (x <= y) ? ((~0ull >> x) & (~0ull << (63 - y)))
                          : ((~0ull >> x) | (~0ull << (63 - y)));
    return insert ? ((r & m) | (ra & ~m)) : (r & m);
}
static inline uint32_t ppc_rlwinm(uint32_t rs, int sh, int mb, int me) {
    sh &= 31;
    uint32_t rotated = sh ? ((rs << sh) | (rs >> (32 - sh))) : rs;
    uint32_t mask;
    if (mb <= me) {
        mask = ((uint32_t)-1 >> mb) & ((uint32_t)-1 << (31 - me));
    } else {
        mask = ((uint32_t)-1 >> mb) | ((uint32_t)-1 << (31 - me));
    }
    return rotated & mask;
}

static inline uint32_t ppc_rlwimi(uint32_t ra, uint32_t rs, int sh, int mb, int me) {
    sh &= 31;
    uint32_t rotated = sh ? ((rs << sh) | (rs >> (32 - sh))) : rs;
    uint32_t mask;
    if (mb <= me) {
        mask = ((uint32_t)-1 >> mb) & ((uint32_t)-1 << (31 - me));
    } else {
        mask = ((uint32_t)-1 >> mb) | ((uint32_t)-1 << (31 - me));
    }
    return (rotated & mask) | (ra & ~mask);
}


static inline uint64_t ppc_rotl64(uint64_t v, int n) {
    n &= 63;
    return n ? ((v << n) | (v >> (64 - n))) : v;
}
static inline uint64_t ppc_mask64(int mb, int me) {
    uint64_t mask;
    if (mb <= me) {
        mask = (mb == 0) ? ~0ULL : (~0ULL >> mb);
        mask &= (me == 63) ? ~0ULL : (~0ULL << (63 - me));
    } else {
        mask = (~0ULL >> mb) | (~0ULL << (63 - me));
    }
    return mask;
}
static inline uint64_t ppc_rldicl(uint64_t rs, int sh, int mb) {
    return ppc_rotl64(rs, sh) & ppc_mask64(mb, 63);
}
static inline uint64_t ppc_rldicr(uint64_t rs, int sh, int me) {
    return ppc_rotl64(rs, sh) & ppc_mask64(0, me);
}
static inline uint64_t ppc_rldic(uint64_t rs, int sh, int mb) {
    int me = 63 - sh;
    return ppc_rotl64(rs, sh) & ppc_mask64(mb, me);
}
static inline uint64_t ppc_rldimi(uint64_t ra, uint64_t rs, int sh, int mb) {
    int me = 63 - sh;
    uint64_t mask = ppc_mask64(mb, me);
    return (ppc_rotl64(rs, sh) & mask) | (ra & ~mask);
}

static inline int64_t ppc_sraw(uint32_t* xer, int32_t rs, int n) {
    int sh = n & 0x3F;
    uint32_t so = (sh == 0) ? 0u : ((sh >= 32) ? (uint32_t)rs : ((uint32_t)rs & (((uint32_t)1u << sh) - 1u)));
    uint32_t ca = (rs < 0 && so != 0u) ? 1u : 0u;
    *xer = (*xer & ~(1u << 29)) | (ca << 29);
    return (int64_t)((sh >= 32) ? (rs >> 31) : (rs >> sh));
}
static inline int64_t ppc_srad(uint32_t* xer, int64_t rs, int n) {
    int sh = n & 0x7F;
    uint64_t so = (sh == 0) ? 0ull : ((sh >= 64) ? (uint64_t)rs : ((uint64_t)rs & (((uint64_t)1ull << sh) - 1ull)));
    uint32_t ca = (rs < 0 && so != 0ull) ? 1u : 0u;
    *xer = (*xer & ~(1u << 29)) | (ca << 29);
    return (sh >= 64) ? (rs >> 63) : (rs >> sh);
}


void june_evt_func_00840698(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_21 = ctx->gpr[21];
    uint64_t _cs_22 = ctx->gpr[22];
    uint64_t _cs_23 = ctx->gpr[23];
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[28]);
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[23] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[22] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[29] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    vm_write32(ctx->gpr[1] + 0xF8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    ctx->gpr[30] = vm_read32(ctx->gpr[3] + 0x88);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840904;
loc_00840704:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840774;
loc_0084070C:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00840728; june_evt_func_008714AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xF8);
    ctx->gpr[21] = _cs_21;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[22] = _cs_22;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_00840774:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084091C;
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008408C0;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0xE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840A40;
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0xE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084097C;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00840A44;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0xD; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840E30;
loc_008407A4:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084070C;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[31] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840DD4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840E68;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x237);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D40);
    ctx->lr = 0x008407E0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008407E4:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[5] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00840EB0;
    ctx->gpr[9] = ctx->gpr[5] + ctx->gpr[8];
    ctx->gpr[0] = ctx->gpr[4] - ctx->gpr[8];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(1);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[10] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[0] = vm_read8(ctx->gpr[6] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00840834;
    goto loc_00840EB8;
loc_00840828:
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840844;
loc_00840834:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840828;
loc_00840844:
    ctx->gpr[7] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[10] = ctx->gpr[5] + ctx->gpr[10];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = ctx->gpr[9] | ctx->gpr[9];
    vm_write8(ctx->gpr[9] + 0x1, ctx->gpr[0]);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008408A0;
    ctx->gpr[0] = ctx->gpr[7] - ctx->gpr[8];
    ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->ctr = ctx->gpr[0];
    ;
loc_00840878:
    ctx->gpr[9] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[8] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840878;
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[5];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_008408A0:
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write8(ctx->gpr[6] + 0x0, ctx->gpr[31]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840BE8;
loc_008408AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[9]);
    goto loc_00840944;
loc_008408C0:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840AE0;
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840A2C;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)8; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840D2C;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0xA; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840B20;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008407A4;
loc_008408E8:
    ctx->gpr[25] = (int64_t)(int32_t)(0);
loc_008408EC:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0084070C;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00840944;
loc_00840904:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2C9);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D54);
    ctx->lr = 0x00840914; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00840704;
loc_0084091C:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084070C;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[11]);
loc_00840944:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->lr = 0x00840950; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[31], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    goto loc_0084070C;
loc_0084097C:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0x201; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840C14;
    ctx->gpr[0] = ctx->gpr[29] ^ ((uint64_t)0xFF00 << 16);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840DE8;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0x7F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008407A4;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[8];
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00840A1C;
    ctx->gpr[11] = ctx->gpr[8] + ctx->gpr[6];
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840A1C;
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[8];
    ctx->gpr[7] = ctx->gpr[11] | ctx->gpr[11];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->ctr = ctx->gpr[0];
    goto loc_008409F4;
loc_008409E8:
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840A1C;
loc_008409F4:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[7] = ctx->gpr[11] | ctx->gpr[11];
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_008409E8;
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[8];
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
loc_00840A1C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    goto loc_00840944;
loc_00840A2C:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840D0C;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008407A4;
    goto loc_008408E8;
loc_00840A40:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
loc_00840A44:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[6] = (int64_t)(int32_t)(0x306);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 1, 0, 30, 0, 0);
    ctx->lr = 0x00840A64; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00840A7C; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00840C0C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
loc_00840A90:
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6C98);
    ctx->lr = 0x00840AB0; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00840ABC; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00840BF0;
loc_00840AD0:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00840AD8; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008408EC;
loc_00840AE0:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084070C;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084070C;
    ctx->gpr[0] = ctx->gpr[11] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[0]);
    goto loc_00840944;
loc_00840B20:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[5] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840ECC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084070C;
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00840B78;
    goto loc_00840F50;
loc_00840B64:
    ctx->gpr[10] = ctx->gpr[6] + ctx->gpr[7];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840B84;
loc_00840B78:
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840B64;
    ctx->gpr[10] = ctx->gpr[6] + ctx->gpr[5];
loc_00840B84:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[9] + 0x1, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840BD4;
    ctx->gpr[0] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[7], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
    ctx->ctr = ctx->gpr[9];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00840F68;
loc_00840BB0:
    ctx->gpr[9] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[8] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840BB0;
    ctx->gpr[10] = ctx->gpr[6] + ctx->gpr[7];
loc_00840BD4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008408AC;
loc_00840BE8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00840944;
loc_00840BF0:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00840C00; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[27]);
    goto loc_00840AD0;
loc_00840C0C:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[1]));
    goto loc_00840A90;
loc_00840C14:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x00840C1C; june_evt_func_0086E8C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084070C;
    ctx->lr = 0x00840C30; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6C90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840C60;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6C94);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
loc_00840C60:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0xC);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840F48;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840F48;
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->ctr = ctx->gpr[0];
    goto loc_00840CB8;
loc_00840CA4:
    ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840E90;
loc_00840CB8:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840CA4;
    ctx->gpr[9] = ctx->gpr[8] | ctx->gpr[8];
loc_00840CC4:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084070C;
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[0]);
    goto loc_00840944;
loc_00840D0C:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084070C;
loc_00840D1C:
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[0]);
    goto loc_00840944;
loc_00840D2C:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084070C;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[8] = ctx->gpr[10] + (int64_t)(-1);
    ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[7] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00840DB4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840DB4;
    ctx->gpr[9] = ctx->gpr[7] - ctx->gpr[10];
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[10];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(-1);
    ctx->ctr = ctx->gpr[9];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840D9C;
    goto loc_00840F74;
loc_00840D8C:
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x2);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840E88;
loc_00840D9C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840D8C;
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[10];
loc_00840DB4:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[9]);
    goto loc_00840944;
loc_00840DD4:
    ctx->gpr[0] = ctx->gpr[29] + (int64_t)(-32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x5E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084070C;
    goto loc_008407E4;
loc_00840DE8:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840E98;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[11] + 0x40);
loc_00840E00:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->lr = 0x00840E28; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    goto loc_0084070C;
loc_00840E30:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840F34;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->lr = 0x00840E4C; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00840E50:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840E98;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x3C);
    goto loc_00840E00;
loc_00840E68:
    ctx->gpr[0] = ctx->gpr[29] + (int64_t)(-48);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008407E4;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084070C;
    goto loc_008407E4;
loc_00840E88:
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[8];
    goto loc_00840DB4;
loc_00840E90:
    ctx->gpr[9] = ctx->gpr[11] + (int64_t)(1);
    goto loc_00840CC4;
loc_00840E98:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0x8000 << 16);
    ctx->lr = 0x00840EA8; june_evt_func_0086DE20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084070C;
loc_00840EB0:
    ctx->gpr[9] = ctx->gpr[5] + ctx->gpr[8];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[9], 0, 32);
loc_00840EB8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = ctx->gpr[8] | ctx->gpr[8];
    vm_write8(ctx->gpr[9] + 0x1, ctx->gpr[0]);
    goto loc_008408A0;
loc_00840ECC:
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840F18;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[5];
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    goto loc_00840EF4;
loc_00840EE8:
    ctx->gpr[0] = vm_read8(ctx->gpr[8] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840F14;
loc_00840EF4:
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[6];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[11] = vm_read8(ctx->gpr[8] + 0x0);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840EE8;
loc_00840F14:
    ctx->gpr[11] = ctx->gpr[8] | ctx->gpr[8];
loc_00840F18:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00840D1C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00840944;
loc_00840F34:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->lr = 0x00840F40; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00840E50;
loc_00840F48:
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    goto loc_00840CC4;
loc_00840F50:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(0x2D);
    vm_write8(ctx->gpr[11] + 0x1, ctx->gpr[0]);
    vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    goto loc_00840944;
loc_00840F68:
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[11];
    goto loc_00840BB0;
loc_00840F74:
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[9];
    goto loc_00840D9C;
}

void june_evt_func_0083FF00(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x88);
    ctx->gpr[4] = (int64_t)(int32_t)(0x34D);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0083FF5C;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->lr = 0x0083FF2C; june_evt_func_00887B00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x0083FF38; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083FF44; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_0083FF5C:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D54);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->lr = 0x0083FF68; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->lr = 0x0083FF74; june_evt_func_00887B00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x0083FF80; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083FF8C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0083FFA4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00840564;
loc_0083FFD4:
    ctx->gpr[27] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084057C;
loc_0083FFE0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6CA0);
    ctx->gpr[9] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 25, 25, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008401D4;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008402CC;
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[0]);
loc_00840010:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00840018; june_evt_func_00885B74(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840160;
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
loc_00840030:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->lr = 0x0084003C; june_evt_func_00885AD4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008402D4;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008403C4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840408;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x237);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D40);
    ctx->lr = 0x00840078; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0084007C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008403EC;
loc_00840094:
    ctx->gpr[9] = ctx->gpr[4] + ctx->gpr[8];
    ctx->gpr[0] = ctx->gpr[6] - ctx->gpr[8];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(1);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[7] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[0] = vm_read8(ctx->gpr[5] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008400CC;
    goto loc_008403F4;
loc_008400C0:
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008400E0;
loc_008400CC:
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_008400C0;
loc_008400E0:
    ctx->gpr[10] = ctx->gpr[4] + ctx->gpr[7];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = ctx->gpr[9] | ctx->gpr[9];
    vm_write8(ctx->gpr[9] + 0x1, ctx->gpr[0]);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084013C;
    ctx->gpr[0] = ctx->gpr[7] - ctx->gpr[8];
    ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->ctr = ctx->gpr[0];
    ;
loc_00840110:
    ctx->gpr[9] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[8] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840110;
    ctx->gpr[0] = ctx->gpr[4] + ctx->gpr[7];
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_0084013C:
    vm_write8(ctx->gpr[5] + 0x0, ctx->gpr[30]);
    if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00840150;
loc_00840144:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[9]);
loc_00840150:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
loc_00840154:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00840030;
loc_00840160:
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_008401A8;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->lr = 0x0084017C; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
loc_008401A8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_008401D4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084028C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 23, 23, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00840450;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 22, 22, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840594;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[8] + ctx->gpr[6];
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084027C;
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[8];
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084027C;
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[8];
    ctx->gpr[7] = ctx->gpr[11] | ctx->gpr[11];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->ctr = ctx->gpr[0];
    goto loc_00840254;
loc_00840248:
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084027C;
loc_00840254:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[7] = ctx->gpr[11] | ctx->gpr[11];
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840248;
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[8];
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
loc_0084027C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    goto loc_00840010;
loc_0084028C:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008402CC;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008402CC;
    ctx->gpr[0] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[0]);
    goto loc_00840010;
loc_008402CC:
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    goto loc_00840010;
loc_008402D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00840154;
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[3], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x63; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084042C;
loc_008402EC:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x43; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084042C;
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00840154;
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = ctx->gpr[11] + (int64_t)(-1);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008404F8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840154;
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084034C;
    goto loc_00840670;
loc_00840338:
    ctx->gpr[10] = ctx->gpr[6] + ctx->gpr[8];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840654;
loc_0084034C:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840338;
    ctx->gpr[10] = ctx->gpr[5] + ctx->gpr[6];
    ctx->gpr[7] = ctx->gpr[5] | ctx->gpr[5];
loc_0084035C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[9] + 0x1, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008403AC;
    ctx->gpr[0] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[7], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
    ctx->ctr = ctx->gpr[9];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084068C;
loc_00840388:
    ctx->gpr[9] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[8] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840388;
    ctx->gpr[10] = ctx->gpr[7] + ctx->gpr[6];
loc_008403AC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00840144;
    goto loc_00840150;
loc_008403C4:
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x5E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00840154;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00840094;
loc_008403EC:
    ctx->gpr[9] = ctx->gpr[4] + ctx->gpr[8];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[9], 0, 32);
loc_008403F4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[8]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[9] + 0x1, ctx->gpr[0]);
    goto loc_0084013C;
loc_00840408:
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-48);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084007C;
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[3], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x2E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084007C;
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x63; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008402EC;
loc_0084042C:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840154;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[11]);
    goto loc_00840154;
loc_00840450:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008402CC;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[8] = ctx->gpr[10] + (int64_t)(-1);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[7] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[11] = ctx->gpr[8] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008404D8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008404D8;
    ctx->gpr[9] = ctx->gpr[7] - ctx->gpr[10];
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[6] + ctx->gpr[10];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(-1);
    ctx->ctr = ctx->gpr[9];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008404C0;
    goto loc_00840680;
loc_008404B0:
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x2);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084064C;
loc_008404C0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_008404B0;
    ctx->gpr[11] = ctx->gpr[10] + ctx->gpr[6];
loc_008404D8:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[9]);
    goto loc_00840010;
loc_008404F8:
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840544;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[5];
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    goto loc_00840520;
loc_00840514:
    ctx->gpr[0] = vm_read8(ctx->gpr[8] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840540;
loc_00840520:
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[6];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[11] = vm_read8(ctx->gpr[8] + 0x0);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00840514;
loc_00840540:
    ctx->gpr[11] = ctx->gpr[8] | ctx->gpr[8];
loc_00840544:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00840150;
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-1);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[0]);
    goto loc_00840150;
loc_00840564:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6CA8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x102);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6CA4);
    ctx->lr = 0x00840574; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0083FFD4;
loc_0084057C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x364);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D54);
    ctx->lr = 0x0084058C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0083FFE0;
loc_00840594:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6C9C);
    ctx->gpr[9] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 21, 21, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840610;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084065C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x008405C4; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008405C8:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00840630;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00840608; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    goto loc_00840010;
loc_00840610:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 20, 20, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008402CC;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x38);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008402CC;
loc_00840630:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0x8000 << 16);
    ctx->lr = 0x00840644; june_evt_func_0086DE20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00840010;
loc_0084064C:
    ctx->gpr[11] = ctx->gpr[8] + ctx->gpr[6];
    goto loc_008404D8;
loc_00840654:
    ctx->gpr[7] = ctx->gpr[8] | ctx->gpr[8];
    goto loc_0084035C;
loc_0084065C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x00840668; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008405C8;
loc_00840670:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[10]);
    vm_write8(ctx->gpr[11] + 0x1, ctx->gpr[0]);
    goto loc_00840150;
loc_00840680:
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[9];
    goto loc_008404C0;
loc_0084068C:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[0];
    goto loc_00840388;
}

void june_evt_func_00842A88(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_23 = ctx->gpr[23];
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x150, ctx->gpr[1]); ctx->gpr[1] += -0x150;
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[29] = ctx->gpr[8] | ctx->gpr[8];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842B40;
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[8], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842E0C;
loc_00842AE8:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00842B04; june_evt_func_008714AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_00842B40:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[8], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00842AE8;
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[30] = (int64_t)(int32_t)(4);
    ctx->lr = 0x00842B60; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842B90;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x19E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BD0);
    ctx->lr = 0x00842B8C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842B90:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[30] = (int64_t)(int32_t)(4);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00842BB0; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084302C;
loc_00842BC0:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842BE0;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1A7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->lr = 0x00842BDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842BE0:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842C04;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC4);
    ctx->lr = 0x00842C00; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842C04:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00842C0C; june_evt_func_00853B10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084305C;
loc_00842C1C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842C2C; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[1];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00842C48;
    ctx->fpr[31] = ctx->fpr[0];
loc_00842C48:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->lr = 0x00842C54; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00843044;
loc_00842C64:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842C84;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1A7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->lr = 0x00842C80; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842C84:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842CA8;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC4);
    ctx->lr = 0x00842CA4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842CA8:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00842CB0; june_evt_func_00853B10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842FFC;
loc_00842CC0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842CD0; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    { double a = ctx->fpr[31]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00842CE4;
    ctx->fpr[31] = ctx->fpr[1];
loc_00842CE4:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->lr = 0x00842CF0; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842FE4;
loc_00842D00:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842D20;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1A7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->lr = 0x00842D1C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842D20:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842D44;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC4);
    ctx->lr = 0x00842D40; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842D44:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00842D4C; june_evt_func_00853B10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00843014;
loc_00842D5C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842D6C; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    { double a = ctx->fpr[1]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00842D80;
    ctx->fpr[31] = ctx->fpr[1];
loc_00842D80:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[30] = (int64_t)(int32_t)(4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6BBC);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00842DA4; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00842DB4; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1C);
    ctx->lr = 0x00842DC4; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->lr = 0x00842DD4; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1C);
    ctx->lr = 0x00842DE4; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->lr = 0x00842DF4; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1C);
    ctx->lr = 0x00842E04; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842AE8;
loc_00842E0C:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[8], 0, 32);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00842E2C; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842E5C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1C8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BD0);
    ctx->lr = 0x00842E58; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842E5C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00842E7C; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842EAC;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1CD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->lr = 0x00842EA8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842EAC:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842ED0;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1CF);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC4);
    ctx->lr = 0x00842ECC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842ED0:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00842ED8; june_evt_func_00853B10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842F8C;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00842EF4; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842F34;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842F24;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1D8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->lr = 0x00842F20; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842F24:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->lr = 0x00842F30; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842F34:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00842F8C;
loc_00842F3C:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->lr = 0x00842F48; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842F78;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1D8);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->lr = 0x00842F74; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842F78:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->lr = 0x00842F84; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842AE8;
loc_00842F8C:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->lr = 0x00842F98; june_evt_func_0086E8F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842FD8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842FC8;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1D8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC8);
    ctx->lr = 0x00842FC4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842FC8:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->lr = 0x00842FD4; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842FD8:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842AE8;
    goto loc_00842F3C;
loc_00842FE4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1A6);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BCC);
    ctx->lr = 0x00842FF4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842D00;
loc_00842FFC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC0);
    ctx->lr = 0x0084300C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842CC0;
loc_00843014:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC0);
    ctx->lr = 0x00843024; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842D5C;
loc_0084302C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1A6);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BCC);
    ctx->lr = 0x0084303C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842BC0;
loc_00843044:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1A6);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BCC);
    ctx->lr = 0x00843054; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842C64;
loc_0084305C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BC0);
    ctx->lr = 0x0084306C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842C1C;
}

void june_evt_func_00842A10(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = vm_read32(ctx->gpr[3] + 0x88);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1EF);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842A54;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x88);
    ctx->lr = 0x00842A3C; june_evt_func_00887B00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_00842A54:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BDC);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->lr = 0x00842A60; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x88);
    ctx->lr = 0x00842A70; june_evt_func_00887B00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0084297C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x102);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_008429FC;
loc_0084299C:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[4] = (int64_t)(int32_t)(0x358);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008429CC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x008429B4; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_008429CC:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BDC);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->lr = 0x008429D8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x008429E4; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_008429FC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BD8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BD4);
    ctx->lr = 0x00842A08; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084299C;
}

void june_evt_func_00843074(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x102);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00843110;
loc_00843094:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[4] = (int64_t)(int32_t)(0x201);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00843124;
loc_008430A4:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x40);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008430D8;
loc_008430C0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = ctx->gpr[0];
    return;
loc_008430D8:
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008430C0;
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0xC);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0x8000 << 16);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008430C0;
    ctx->lr = 0x00843104; june_evt_func_0086DE20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(-1);
    goto loc_008430C0;
loc_00843110:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BD8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BD4);
    ctx->lr = 0x0084311C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00843094;
loc_00843124:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BDC);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->lr = 0x00843130; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008430A4;
}

void june_evt_func_0084544C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00845B2C;
loc_00845480:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845B14;
loc_0084548C:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x40);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008454D4;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_008454D4:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->lr = 0x008454DC; june_evt_func_008C31C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845B94;
loc_008454EC:
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008459F4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[27] = ctx->gpr[8] + (int64_t)(0x90);
    ctx->lr = 0x00845504; june_evt_func_008E1AC0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ctx->gpr[3] + (int64_t)(0x10);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845B44;
loc_00845514:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845530;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845AE4;
loc_00845530:
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
loc_00845538:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x7C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084580C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008459E8;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x252);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6AC4);
    ctx->lr = 0x0084555C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00845824;
loc_00845570:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[25];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00845890;
    ;
loc_00845588:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008455A0;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0084559C; june_evt_func_008E8F38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008455A0:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845610;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ctx->gpr[27] + (int64_t)(0x10);
    ctx->gpr[10] = ctx->gpr[27] + (int64_t)(0x20);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x1C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x20, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x24, tmp); }
loc_00845610:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00845A74;
loc_0084561C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xB4);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845640;
    ctx->lr = 0x00845630; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008459A4;
loc_00845640:
    ctx->gpr[9] = ctx->gpr[2] + (int64_t)(-27620);
    { uint64_t ea = (ctx->gpr[29]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x80);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[13], 16); memcpy(&ctx->vr[13], tmp + 12, 16); }
    { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[11] = (int64_t)(int32_t)(0x20);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    { uint64_t ea = ctx->gpr[9]; uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[0]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[13], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[13])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    { uint64_t ea = ((ctx->gpr[28] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = (ctx->gpr[28]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[9], &ctx->vr[0], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[10]; uint32_t val = ((uint32_t*)&ctx->vr[13])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[28] + ctx->gpr[11])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = ((ctx->gpr[28] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[12], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[11], &ctx->vr[13], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x008456B8; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x008456C0; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008456CC; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6ABC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[9]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00845864;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
loc_00845714:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[8]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845748;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_00845748:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00845764;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[9]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00845874;
loc_00845764:
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(-242);
    ctx->gpr[9] = ctx->gpr[3] + (int64_t)(-340);
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 1);
    ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_008457A4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x40);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0xC0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0xC0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->lr = 0x008457D8; june_evt_func_0086F9D4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_0084580C:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[30] + 0xAC, ctx->gpr[0]);
loc_00845814:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00845570;
loc_00845824:
    ctx->gpr[4] = ctx->gpr[31] + (int64_t)(0x8C);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = (int64_t)(int32_t)(0x21);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00845844; june_evt_func_008436A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845588;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[25];
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00845588;
    goto loc_00845890;
loc_00845864:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00845714;
loc_00845874:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00845764;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008457A4;
    goto loc_00845764;
loc_00845890:
    ctx->gpr[0] = vm_read8(ctx->gpr[30] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845970;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008458BC;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845B4C;
loc_008458BC:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x58); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x4, tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[30] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845588;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
loc_008458E4:
    ctx->gpr[0] = ctx->gpr[27] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x5C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845B60;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x60); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845B60;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845B60;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845B60;
loc_0084592C:
    ctx->gpr[0] = ctx->gpr[27] + (int64_t)(0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x6C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845BAC;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845BAC;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845BAC;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00845588;
loc_00845970:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xA4);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845640;
    ctx->lr = 0x00845984; june_evt_func_0086DF64(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845640;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->lr = 0x0084599C; june_evt_func_008D0C8C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00845640;
loc_008459A4:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0xB4);
    ctx->gpr[7] = (int64_t)(int32_t)(-1);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xB0);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = vm_read32(ctx->gpr[30] + 0xB8);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0xBC);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x008459D8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x008459E0; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00845640;
loc_008459E8:
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[30] + 0xAC, ctx->gpr[0]);
    goto loc_00845814;
loc_008459F4:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845BC8;
loc_00845A00:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x6ACC);
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[27] = ctx->gpr[8] | ctx->gpr[8];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x4, tmp); }
    goto loc_00845514;
loc_00845A74:
    ctx->gpr[9] = ctx->gpr[27] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x50, tmp); }
    ctx->gpr[11] = ctx->gpr[27] + (int64_t)(0x20);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x54, tmp); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x58, tmp); }
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x40);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x5C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x60, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x64, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x68, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x6C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x70, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x74, tmp); }
    ctx->lr = 0x00845ADC; june_evt_func_0086E5D0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084561C;
loc_00845AE4:
    ctx->gpr[11] = (int64_t)(int32_t)(0x10);
    ctx->gpr[10] = (int64_t)(int32_t)(0x20);
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[11])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[10])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = ((ctx->gpr[29] + ctx->gpr[11])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint64_t ea = ((ctx->gpr[29] + ctx->gpr[10])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint64_t ea = (ctx->gpr[29]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    goto loc_00845538;
loc_00845B14:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x227);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BDC);
    ctx->lr = 0x00845B24; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084548C;
loc_00845B2C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BD8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x102);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BD4);
    ctx->lr = 0x00845B3C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00845480;
loc_00845B44:
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x6ACC);
    goto loc_00845514;
loc_00845B4C:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x58); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008458E4;
    goto loc_008458BC;
loc_00845B60:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x60); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[30] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845588;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    goto loc_0084592C;
loc_00845B94:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x22D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6B74);
    ctx->lr = 0x00845BA4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008454EC;
loc_00845BAC:
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    goto loc_00845588;
loc_00845BC8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x239);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6AC8);
    ctx->lr = 0x00845BD8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x84);
    goto loc_00845A00;
}

void june_evt_func_0086E8C4(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x0086E8E0; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    return;
}

void june_evt_func_008436A8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_14 = ctx->gpr[14];
    uint64_t _cs_15 = ctx->gpr[15];
    uint64_t _cs_16 = ctx->gpr[16];
    uint64_t _cs_17 = ctx->gpr[17];
    uint64_t _cs_18 = ctx->gpr[18];
    uint64_t _cs_19 = ctx->gpr[19];
    uint64_t _cs_20 = ctx->gpr[20];
    uint64_t _cs_21 = ctx->gpr[21];
    uint64_t _cs_22 = ctx->gpr[22];
    uint64_t _cs_23 = ctx->gpr[23];
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[0] = (int64_t)(int32_t)(0x2B0);
    vm_write64(ctx->gpr[1] + -0x460, ctx->gpr[1]); ctx->gpr[1] += -0x460;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[20]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2C0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[21]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2D0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[22]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2E0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[23]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2F0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[24]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x300);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[25]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x310);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[26]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x320);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[27]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x330);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[28]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x340);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[29]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x350);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[30]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x360);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[31]); }
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x408, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x410, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x418, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x420, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x428, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x430, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x438, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x440, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x448, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x450, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x458, tmp); }
    vm_write64(ctx->gpr[1] + 0x378, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x380, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x388, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x390, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x398, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x3A0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x3A8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x3B0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x3B8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x3C0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x3E0, ctx->gpr[27]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x3C8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x3F8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x400, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x3D0, ctx->gpr[25]);
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0x3D8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x3E8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x3F0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x470, ctx->gpr[0]);
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    vm_write32(ctx->gpr[1] + 0x1B0, ctx->gpr[6]);
    ctx->lr = 0x008437A0; june_evt_func_008C31C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write32(ctx->gpr[1] + 0x1AC, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845160;
loc_008437B4:
    ctx->lr = 0x008437B8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008437E8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B70);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_008437E8:
    ctx->lr = 0x008437EC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084381C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B6C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[9] = ppu_frsp(ctx->fpr[0]);
loc_0084381C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[24], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[24], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[11]));
    { double a = ctx->fpr[10]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008441E4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_0084385C:
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008441D4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
loc_00843870:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xFC, tmp); }
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x100, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x108, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x10C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF4, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF8, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x104, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008440E8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008449C0;
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0x1AC);
    { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6B54);
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(0x10);
    ctx->gpr[19] = vm_read32(ctx->gpr[2] + -0x6B4C);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6B50);
    ctx->gpr[8] = (int64_t)(int32_t)(0x160);
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vm_write64(ctx->gpr[1] + 0x208, ctx->gpr[11]);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t ea = (ctx->gpr[19]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    vmx_vaddfp(&ctx->vr[8], &ctx->vr[13], &ctx->vr[13], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[7]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[6]; uint32_t val = ((uint32_t*)&ctx->vr[13])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[20] = vm_read32(ctx->gpr[2] + -0x6B48);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    ctx->gpr[11] = ctx->gpr[2] + (int64_t)(-27620);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 12, 16); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B44);
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x1B0);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[10]);
    { uint64_t ea = (ctx->gpr[20]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[10] = (int64_t)(int32_t)(0x150);
    { uint64_t ea = ctx->gpr[11]; uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[1]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
    ctx->gpr[11] = (int64_t)(int32_t)(0x140);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[5], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vm_write64(ctx->gpr[1] + 0x250, ctx->gpr[5]);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[7], &ctx->vr[9], &ctx->vr[8], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[8]; uint8_t* b = (uint8_t*)&ctx->vr[8]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[12]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[8]; uint8_t* b = (uint8_t*)&ctx->vr[8]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[0], tmp + 4, 16); }
    vmx_vnmsubfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[3], &ctx->vr[12], ctx->vscr);
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[9], &ctx->vr[6], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint32_t* d = (uint32_t*)&ctx->vr[16]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[17]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[15]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vnmsubfp(&ctx->vr[6], &ctx->vr[10], &ctx->vr[4], &ctx->vr[6], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[19]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[3], &ctx->vr[9], &ctx->vr[16], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[10], &ctx->vr[13], &ctx->vr[7], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[12], &ctx->vr[8], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[6]; uint64_t* c=(uint64_t*)&ctx->vr[11]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[6]; uint64_t* b=(uint64_t*)&ctx->vr[13]; uint64_t* c=(uint64_t*)&ctx->vr[11]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[10]; uint64_t* c=(uint64_t*)&ctx->vr[11]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[13]; uint64_t* c=(uint64_t*)&ctx->vr[5]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[10]; uint64_t* c=(uint64_t*)&ctx->vr[5]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[6]; uint64_t* c=(uint64_t*)&ctx->vr[5]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[1], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[11], 16); memcpy(tmp+16, &ctx->vr[11], 16); memcpy(&ctx->vr[11], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[1], tmp + 4, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[0], tmp + 4, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[11], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[11], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[6]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[2]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[5]; uint32_t val = ((uint32_t*)&ctx->vr[1])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[11])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[3], &ctx->vr[9], &ctx->vr[6], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[11])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[10])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x20);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[3], &ctx->vr[9], &ctx->vr[8], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[18]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[12], &ctx->vr[9], &ctx->vr[17], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[10]; uint32_t val = ((uint32_t*)&ctx->vr[11])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[12], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[8])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[12], &ctx->vr[9], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[9], &ctx->vr[13], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[11])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[9], &ctx->vr[2], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[1])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[2]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vmx_vmaddfp(&ctx->vr[19], &ctx->vr[0], &ctx->vr[16], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[8], &ctx->vr[13], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[15], &ctx->vr[4], &ctx->vr[17], &ctx->vr[15], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[4], &ctx->vr[7], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[0], &ctx->vr[6], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[4], &ctx->vr[5], &ctx->vr[9], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[3], &ctx->vr[2], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[4], &ctx->vr[12], &ctx->vr[18], ctx->vscr);
    vmx_vaddfp(&ctx->vr[19], &ctx->vr[19], &ctx->vr[15], ctx->vscr);
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[10], ctx->vscr);
    vmx_vaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[9], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[4], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[19]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[11])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[10])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[8])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
loc_00843A3C:
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    ctx->gpr[29] = (int64_t)(int32_t)(0xB0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[5] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    vm_write64(ctx->gpr[1] + 0x298, ctx->gpr[9]);
    vm_write64(ctx->gpr[1] + 0x290, ctx->gpr[10]);
    ctx->lr = 0x00843A70; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[17] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x270, ctx->gpr[0]);
    ctx->fpr[25] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[23]));
    ctx->gpr[0] = (int64_t)(int32_t)(0x140);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[17] | ctx->gpr[17];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x110);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0xA0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x00843ABC; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[13], ctx->fpr[0], 0, 0));
    ctx->fpr[0] = (double)(float)sqrt(ctx->fpr[0]);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    ctx->fpr[29] = ctx->fpr[0];
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00843B10;
    ctx->fpr[29] = ctx->fpr[10];
loc_00843B10:
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x140);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x270);
    ctx->gpr[29] = (int64_t)(int32_t)(0xB0);
    ctx->gpr[0] = ctx->gpr[5] + (int64_t)(0x10);
    ctx->gpr[28] = (int64_t)(int32_t)(0xA0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0x120);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    vm_write64(ctx->gpr[1] + 0x268, ctx->gpr[5]);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    vm_write64(ctx->gpr[1] + 0x218, ctx->gpr[7]);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[28])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x00843B60; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[0], ctx->fpr[11], 0, 0));
    ctx->fpr[11] = (double)(float)sqrt(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    { double a = ctx->fpr[29]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00843BAC;
    ctx->fpr[29] = ctx->fpr[11];
loc_00843BAC:
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x268);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = ctx->gpr[5] + (int64_t)(0x20);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0x130);
    vm_write64(ctx->gpr[1] + 0x210, ctx->gpr[7]);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vmx_vaddfp(&ctx->vr[1], &ctx->vr[0], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[28])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    ctx->lr = 0x00843BE4; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xEC, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[0], ctx->fpr[11], 0, 0));
    ctx->fpr[11] = (double)(float)sqrt(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00843C30;
    ctx->fpr[29] = ctx->fpr[11];
loc_00843C30:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    vm_write64(ctx->gpr[1] + 0x248, ctx->gpr[27]);
    ctx->lr = 0x00843C48; june_evt_func_008C29B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[20] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00843C54; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x60);
    vm_write32(ctx->gpr[1] + 0x190, ctx->gpr[20]);
    ctx->lr = 0x00843C68; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6B3C);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6B28);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0xD8);
    ctx->gpr[16] = vm_read32(ctx->gpr[2] + -0x6B14);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x110);
    ctx->gpr[15] = vm_read32(ctx->gpr[2] + -0x6B0C);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[14] = vm_read32(ctx->gpr[2] + -0x6B08);
    vm_write32(ctx->gpr[1] + 0x18C, ctx->gpr[5]);
    vm_write64(ctx->gpr[1] + 0x278, ctx->gpr[7]);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6B24);
    ctx->gpr[18] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x6B20);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x8C);
    vm_write64(ctx->gpr[1] + 0x280, ctx->gpr[9]);
    vm_write64(ctx->gpr[1] + 0x288, ctx->gpr[10]);
    vm_write64(ctx->gpr[1] + 0x220, ctx->gpr[11]);
    vm_write64(ctx->gpr[1] + 0x228, ctx->gpr[5]);
    vm_write64(ctx->gpr[1] + 0x230, ctx->gpr[7]);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B1C);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6B18);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6B10);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6B04);
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x6B00);
    vm_write32(ctx->gpr[1] + 0x180, ctx->gpr[3]);
    vm_write32(ctx->gpr[1] + 0x1A4, ctx->gpr[3]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x238, ctx->gpr[9]);
    vm_write64(ctx->gpr[1] + 0x240, ctx->gpr[10]);
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[11]);
    vm_write64(ctx->gpr[1] + 0x1E8, ctx->gpr[5]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[7]);
loc_00843CF0:
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x278);
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0x190);
    ctx->gpr[31] = vm_read32(ctx->gpr[24] + 0x20);
    ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[11];
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0x18C);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[27] = vm_read32(ctx->gpr[11] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[26], ctx->fpr[12], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[27], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844814;
loc_00843D44:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x180);
    { uint64_t* d = (uint64_t*)&ctx->vr[31]; uint64_t* a = (uint64_t*)&ctx->vr[31]; uint64_t* b = (uint64_t*)&ctx->vr[31]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)1; ppu_vstu4(&ctx->vr[28],v); }
    ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x280);
    ctx->fpr[30] = ctx->fpr[31];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x288);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)2; ppu_vstu4(&ctx->vr[25],v); }
    vm_write32(ctx->gpr[1] + 0x188, ctx->gpr[7]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(0xA0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    vm_write32(ctx->gpr[9] + 0x1C, ctx->gpr[27]);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[27]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    ctx->lr = 0x00843DA0; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x110);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00843DB0; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00843DBC; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x200);
    ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00843DD0; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[21]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[22]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[20], 4, 0, 27, 0, 0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[20] = ctx->gpr[20] + (int64_t)(1);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x280);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)-1; ppu_vstu4(&ctx->vr[0],v); }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[20], 4, 27, 27, 0, 0);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6B54);
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x6B50);
    vmx_vslw(&ctx->vr[29], &ctx->vr[0], &ctx->vr[0]);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x188);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[23] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6B28);
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[27], ctx->fpr[0]));
    ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[5];
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[26], ctx->fpr[0]));
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[26], ea); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[27], ea); }
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    ctx->gpr[29] = ctx->gpr[3] + (int64_t)(0x1E0);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[30], ea); }
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[30], 16); memcpy(tmp+16, &ctx->vr[30], 16); memcpy(&ctx->vr[22], tmp + 4, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[30]; uint8_t* b = (uint8_t*)&ctx->vr[30]; uint8_t* c = (uint8_t*)&ctx->vr[27]; uint8_t* d = (uint8_t*)&ctx->vr[23]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[30]; uint8_t* b = (uint8_t*)&ctx->vr[30]; uint8_t* c = (uint8_t*)&ctx->vr[26]; uint8_t* d = (uint8_t*)&ctx->vr[24]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    goto loc_00843E88;
loc_00843E7C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x20);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
loc_00843E88:
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(0xB0);
    { uint8_t b = (uint8_t)((ctx->gpr[28]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[0])[i] = b + i; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { uint64_t ea = (ctx->gpr[28]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x228);
    { uint8_t* a = (uint8_t*)&ctx->vr[11]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x230);
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x238);
    { uint64_t ea = (ctx->gpr[16]) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    { uint64_t ea = (ctx->gpr[15]) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[11])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x1E8);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[5], ea); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[31], &ctx->vr[20], ctx->vscr);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B18);
    { uint64_t ea = (ctx->gpr[14]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
    { uint64_t ea = (ctx->gpr[19]) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1F0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[19], ea); }
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[18], ea); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[11], &ctx->vr[31], &ctx->vr[13], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[20]; uint64_t* b=(uint64_t*)&ctx->vr[13]; uint64_t* c=(uint64_t*)&ctx->vr[29]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[0], ctx->vscr);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)3; ppu_vstu4(&ctx->vr[0],v); }
    vmx_vctsxs(&ctx->vr[13], &ctx->vr[13], 0, &ctx->vscr);
    vmx_vcfsx(&ctx->vr[1], &ctx->vr[13], 0);
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[0]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[12]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[28]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[7]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[25]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { const uint8_t* a=(const uint8_t*)&ctx->vr[13]; const uint8_t* b=(const uint8_t*)&ctx->vr[28]; uint8_t d[16]; for(int i=0;i<4;i++){ uint32_t x=0,y=0; for(int k=0;k<4;k++){ x=(uint32_t)((x<<8)|a[i*4+k]); y=(uint32_t)((y<<8)|b[i*4+k]); } uint32_t r=(uint32_t)(x+y); for(int k=0;k<4;k++) d[i*4+k]=(uint8_t)(r>>(8*(4-1-k))); } memcpy(&ctx->vr[13], d, 16); }
    (void)vmx_vcmpequw(&ctx->vr[12], &ctx->vr[12], &ctx->vr[31]);
    (void)vmx_vcmpequw(&ctx->vr[7], &ctx->vr[7], &ctx->vr[31]);
    vmx_vnmsubfp(&ctx->vr[6], &ctx->vr[1], &ctx->vr[11], &ctx->vr[6], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B10);
    { uint64_t* d=(uint64_t*)&ctx->vr[9]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[28]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[25]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    (void)vmx_vcmpequw(&ctx->vr[9], &ctx->vr[9], &ctx->vr[31]);
    (void)vmx_vcmpequw(&ctx->vr[13], &ctx->vr[13], &ctx->vr[31]);
    vmx_vnmsubfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[6], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[31], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[0], &ctx->vr[10], &ctx->vr[11], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[0], &ctx->vr[8], &ctx->vr[11], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[0], &ctx->vr[31], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[3], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[2], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[4], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[1], &ctx->vr[11], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[12]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[8]; uint64_t* c=(uint64_t*)&ctx->vr[12]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[10]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[8]; uint64_t* c=(uint64_t*)&ctx->vr[9]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[29]; uint64_t* b = (uint64_t*)&ctx->vr[12]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[29]; uint64_t* b = (uint64_t*)&ctx->vr[10]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[12]; uint64_t* c=(uint64_t*)&ctx->vr[7]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[10]; uint64_t* c=(uint64_t*)&ctx->vr[13]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[21], &ctx->vr[31], &ctx->vr[0], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[1]; uint64_t* c=(uint64_t*)&ctx->vr[18]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[31], &ctx->vr[30], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[11], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[10]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[26]; uint8_t* d = (uint8_t*)&ctx->vr[9]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[27]; uint8_t* d = (uint8_t*)&ctx->vr[12]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[10], &ctx->vr[31], &ctx->vr[30], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[13], &ctx->vr[22], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[13], 16); memcpy(&ctx->vr[13], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[9], &ctx->vr[1], &ctx->vr[23], ctx->vscr);
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[11], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[1], &ctx->vr[12], &ctx->vr[1], &ctx->vr[24], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[31], &ctx->vr[0], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[27]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[26]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[13], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[10], &ctx->vr[11], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[9], &ctx->vr[0], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[12], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[31], 16); memcpy(&ctx->vr[12], tmp + 4, 16); }
    vmx_vaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[19], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[26])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[12]); }
    ctx->lr = 0x00843FE0; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[27]);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0x10);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[28], ctx->fpr[0], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write32(ctx->gpr[9] + 0x8, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[28], ctx->fpr[0], ctx->fpr[12], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    ctx->lr = 0x00844040; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x00844058; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00843E7C;
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0x190);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[3] = ctx->gpr[23] + (int64_t)(0x100);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(8);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    vm_write32(ctx->gpr[1] + 0x190, ctx->gpr[7]);
    ctx->lr = 0x00844088; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x18C);
    ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x248);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(4);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[6] = (int64_t)(int32_t)(0x11);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[1] + 0x18C, ctx->gpr[9]);
    ctx->lr = 0x008440B0; june_evt_func_008B9724(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x248);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[6] = (int64_t)(int32_t)(0x20);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x008440CC; june_evt_func_008B9724(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008441F4;
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x180);
    ctx->gpr[5] = ctx->gpr[5] + (int64_t)(0x20);
    vm_write32(ctx->gpr[1] + 0x180, ctx->gpr[5]);
    goto loc_00843CF0;
loc_008440E8:
    ctx->gpr[5] = (int64_t)(int32_t)(0);
loc_008440EC:
    ctx->gpr[0] = (int64_t)(int32_t)(0x2B0);
    ctx->gpr[14] = _cs_14;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[15] = _cs_15;
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2C0);
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2D0);
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[22], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2E0);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[23], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x2F0);
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x408); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x410); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x418); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[24], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x300);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x420); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x428); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x430); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x438); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[25], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x310);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x440); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x448); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x450); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x458); memcpy(&ctx->fpr[31], &tmp, 8); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[26], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x320);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[27], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x330);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[28], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x340);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[29], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x350);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[30], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x360);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[31], ea); }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x470);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x460);
    ctx->lr = ctx->gpr[0];
    return;
loc_008441D4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    goto loc_00843870;
loc_008441E4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    goto loc_0084385C;
loc_008441F4:
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x1A4);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x248);
    ctx->gpr[6] = (int64_t)(int32_t)(6);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0084420C; june_evt_func_008B9724(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845064;
loc_0084421C:
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x1A8, ctx->gpr[11]);
loc_00844224:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[7] = vm_read8(ctx->gpr[1] + 0x1A8);
    ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[7];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008440EC;
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    ctx->gpr[29] = (int64_t)(int32_t)(0x110);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x110);
    ctx->gpr[5] = ctx->gpr[18] | ctx->gpr[18];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x00844254; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x140);
    ctx->gpr[9] = (int64_t)(int32_t)(0xB0);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x84);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0xA0);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x00844288; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->fpr[0] = (double)(float)sqrt(ctx->fpr[0]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[28] = ctx->fpr[0];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008442CC;
    ctx->fpr[28] = ctx->fpr[9];
loc_008442CC:
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x218);
    ctx->gpr[29] = (int64_t)(int32_t)(0x110);
    ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x290);
    ctx->gpr[28] = (int64_t)(int32_t)(0xA0);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x298);
    ctx->gpr[27] = (int64_t)(int32_t)(0xB0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[28])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[27])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x00844310; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[10], ctx->fpr[13], 0, 0));
    ctx->fpr[12] = (double)(float)sqrt(ctx->fpr[13]);
    { double a = ctx->fpr[28]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084434C;
    ctx->fpr[28] = ctx->fpr[12];
loc_0084434C:
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x210);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[29])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[28])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[27])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->lr = 0x00844374; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[10], ctx->fpr[13], 0, 0));
    ctx->fpr[12] = (double)(float)sqrt(ctx->fpr[13]);
    { double a = ctx->fpr[12]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008443B0;
    ctx->fpr[28] = ctx->fpr[12];
loc_008443B0:
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[23]));
    ctx->lr = 0x008443BC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844AB0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B38);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844AB0;
loc_008443E0:
    ctx->gpr[0] = (int64_t)(int32_t)(0xA0);
    ctx->gpr[20] = vm_read32(ctx->gpr[2] + -0x6B48);
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[1]; uint64_t* b = (uint64_t*)&ctx->vr[1]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x1A0, ctx->gpr[5]);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vm_write64(ctx->gpr[1] + 0x198, ctx->gpr[7]);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[0], tmp + 4, 16); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
loc_0084440C:
    ctx->lr = 0x00844410; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x198);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x210);
    ctx->gpr[18] = ctx->gpr[0] | ctx->gpr[0];
    ctx->lr = 0x00844428; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x5555 << 16);
    { uint64_t ea = (ctx->gpr[20]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[11] = ctx->gpr[18] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[8] = ctx->gpr[8] | 0x5556;
    ctx->gpr[7] = ctx->gpr[18] + (int64_t)(2);
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 0x1F);
    ctx->gpr[5] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[7], 0x1F);
    ctx->gpr[6] = (int64_t)(int32_t)(0xA0);
    vm_write32(ctx->gpr[1] + 0x184, ctx->gpr[3]);
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[1]; uint64_t* b = (uint64_t*)&ctx->vr[1]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[6])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[11] * (int64_t)(int32_t)ctx->gpr[8] >> 32));
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[8] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[7] * (int64_t)(int32_t)ctx->gpr[8] >> 32));
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 1, 0, 30, 0, 0);
    ctx->gpr[8] = ctx->gpr[8] - ctx->gpr[5];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x198);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 2, 0, 29, 0, 0);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 2, 0, 29, 0, 0);
    ctx->gpr[28] = (uint64_t)((ctx->gpr[18] & 0x20) ? 0u : ((uint32_t)ctx->gpr[0] << (ctx->gpr[18] & 0x1F)));
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 1, 0, 30, 0, 0);
    ctx->gpr[11] = ctx->gpr[17] + ctx->gpr[11];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[8];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[17];
    ctx->gpr[8] = ctx->gpr[7] - ctx->gpr[0];
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 2, 0, 29, 0, 0);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[8], 0, 32);
    vm_write32(ctx->gpr[1] + 0x1F8, ctx->gpr[8]);
    { uint8_t b = (uint8_t)(16 - ((ctx->gpr[27]) & 0xF)); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[13])[i] = b + i; }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[13]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t* d=(uint64_t*)&ctx->vr[12]; uint64_t* a=(uint64_t*)&ctx->vr[12]; uint64_t* b=(uint64_t*)&ctx->vr[1]; uint64_t* c=(uint64_t*)&ctx->vr[0]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[6])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[12]); }
    goto loc_00844558;
loc_008444DC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B38);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844668;
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
loc_008444FC:
    ctx->gpr[0] = ctx->gpr[28] & ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00844538;
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[25]));
    vm_write32(ctx->gpr[1] + 0x1A0, ctx->gpr[18]);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[26]));
    ctx->fpr[12] = fabs(ctx->fpr[12]);
    ctx->fpr[13] = fabs(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[13], ctx->fpr[0], 0, 0));
    ctx->fpr[0] = (double)(float)sqrt(ctx->fpr[0]);
    ctx->fpr[22] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[23] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
loc_00844538:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0x1F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[26] = ctx->fpr[10];
    ctx->fpr[25] = ctx->fpr[11];
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844AF4;
loc_00844550:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
loc_00844558:
    ctx->fpr[1] = ctx->fpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00844568; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[23] + 0x0, tmp); }
    ctx->fpr[1] = ctx->fpr[29];
    ctx->lr = 0x0084457C; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(0xA0);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x0, tmp); }
    { uint8_t b = (uint8_t)(16 - ((ctx->gpr[27]) & 0xF)); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[12])[i] = b + i; }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[20]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t* d = (uint64_t*)&ctx->vr[10]; uint64_t* a = (uint64_t*)&ctx->vr[10]; uint64_t* b = (uint64_t*)&ctx->vr[10]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x140);
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x110);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[11])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[10]; uint64_t* c=(uint64_t*)&ctx->vr[0]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x150);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[1])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x160);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[10], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0xB0);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[11], &ctx->vr[9], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[12], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[11])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    ctx->lr = 0x008445FC; june_evt_func_008D221C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[28]));
    ctx->fpr[10] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[27], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[27], ctx->fpr[13], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    vm_write32(ctx->gpr[9] + 0x8, ctx->gpr[30]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00844AD4;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0x1F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844AC8;
loc_00844654:
    ctx->lr = 0x00844658; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008444DC;
loc_00844668:
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00844ABC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[3] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[26]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[31]));
    vm_write32(ctx->gpr[1] + 0x2A0, ctx->gpr[30]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[3], ctx->fpr[9]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[13], ctx->fpr[0], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x2A0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008446E8;
    ctx->fpr[5] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[5]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x2A0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[2], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008446E8;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00844FEC;
loc_008446E8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x100); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[5], ctx->fpr[6]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x104); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[8]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[3], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[9], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[12]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844740;
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[6] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[1]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[9], ctx->fpr[6], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844740;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00845014;
    ;
loc_00844740:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x108); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[5]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x10C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[7]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[3], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[9], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[12]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844798;
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->fpr[5] = ppu_fp_single(ppu_fsub(ctx->fpr[5], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[1]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[9], ctx->fpr[5], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844798;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084503C;
    ;
loc_00844798:
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[6]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[3], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[9], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[12]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844538;
    ctx->fpr[5] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[6] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[5]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[9], ctx->fpr[6], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844538;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00844538;
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[5]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[6], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844538;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00844538;
loc_00844808:
    ctx->gpr[11] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[28]);
    goto loc_008444FC;
loc_00844814:
    ctx->lr = 0x00844818; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00844A8C;
loc_00844828:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[0] = (uint64_t)((ctx->gpr[20] & 0x20) ? 0u : ((uint32_t)ctx->gpr[0] << (ctx->gpr[20] & 0x1F)));
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->fpr[5] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[31]));
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->gpr[9] = ctx->gpr[11] & ~ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[9]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[5], ctx->fpr[13], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008448B4;
    ctx->fpr[8] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[3], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008448B4;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008451F8;
loc_008448B4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x100); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[6]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x104); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[7]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[5], ctx->fpr[13], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[11]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844908;
    ctx->fpr[7] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[6] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[7]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[6], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844908;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008451D0;
loc_00844908:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x108); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[8]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x10C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[13]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[5], ctx->fpr[10], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[11]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084495C;
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[1]));
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[2]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[9], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084495C;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008451A8;
loc_0084495C:
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[6]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[7]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[5], ctx->fpr[13], ctx->fpr[0], 1, 0));
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008449A8;
    ctx->fpr[8] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[1]));
    ctx->fpr[7] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[7], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[6]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008449A8;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[6]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00845178;
loc_008449A8:
    ctx->gpr[5] = ctx->gpr[0] & ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00843D44;
    ctx->gpr[27] = (int64_t)(int32_t)((uint32_t)0xFFFF << 16);
    ctx->gpr[27] = ctx->gpr[27] | 0xFF;
    goto loc_00843D44;
loc_008449C0:
    ctx->gpr[11] = ctx->gpr[2] + (int64_t)(-27620);
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B60);
    ctx->gpr[0] = (int64_t)(int32_t)(0x150);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[1], tmp + 12, 16); }
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x1B0);
    ctx->gpr[8] = (int64_t)(int32_t)(0xA0);
    ctx->gpr[19] = vm_read32(ctx->gpr[2] + -0x6B4C);
    { uint64_t* d = (uint64_t*)&ctx->vr[11]; uint64_t* a = (uint64_t*)&ctx->vr[11]; uint64_t* b = (uint64_t*)&ctx->vr[11]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = ctx->gpr[11]; uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[0]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
    ctx->gpr[10] = (int64_t)(int32_t)(0x170);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6B58);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x160);
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    vm_write64(ctx->gpr[1] + 0x250, ctx->gpr[5]);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[1], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[1])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[8])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B5C);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[11], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[11], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[11] = (int64_t)(int32_t)(0x20);
    { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[1])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[10])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x140);
    { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[11])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6B54);
    { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[10], &ctx->vr[1], ctx->vscr);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B64);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[12], &ctx->vr[11], ctx->vscr);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6B50);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[11]);
    vm_write64(ctx->gpr[1] + 0x208, ctx->gpr[5]);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[10])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[8])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    goto loc_00843A3C;
loc_00844A8C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B38);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844828;
    ctx->gpr[0] = (uint64_t)((ctx->gpr[20] & 0x20) ? 0u : ((uint32_t)ctx->gpr[31] << (ctx->gpr[20] & 0x1F)));
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_008449A8;
loc_00844AB0:
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[11]);
    goto loc_008443E0;
loc_00844ABC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_008444FC;
loc_00844AC8:
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    goto loc_00844654;
loc_00844AD4:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0x1F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[26] = ctx->fpr[10];
    ctx->fpr[25] = ctx->fpr[11];
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00844550;
loc_00844AF4:
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x184);
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = ctx->gpr[5] + (int64_t)(0x200);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00844B0C; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x18);
    ctx->gpr[5] = ctx->gpr[28] & ctx->gpr[0];
    ctx->gpr[7] = (int64_t)(int32_t)((uint32_t)0xFFFF << 16);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[7] = ctx->gpr[7] | 0xFF;
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00844B38;
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0x1F8);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B3C);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    ctx->gpr[7] = vm_read32((ctx->gpr[9] + ctx->gpr[0]));
loc_00844B38:
    ctx->gpr[10] = (int64_t)(int32_t)(0x21);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[10];
    ;
loc_00844B48:
    ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[22];
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x10);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[7]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00844B48;
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x198);
    ctx->gpr[5] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x248);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[6] = (int64_t)(int32_t)(0x21);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    vm_write64(ctx->gpr[1] + 0x198, ctx->gpr[11]);
    ctx->lr = 0x00844B80; june_evt_func_008B9724(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x198);
    { int64_t a = (int64_t)ctx->gpr[5]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084440C;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x18);
    ctx->gpr[7] = vm_read8(ctx->gpr[1] + 0x1A8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[7];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008440EC;
    ctx->lr = 0x00844BA8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read8(ctx->gpr[1] + 0x1A8);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008440EC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B38);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008440EC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AF8);
    ctx->fpr[31] = ctx->fpr[24];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844BF0;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AF4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_00844BF0:
    ctx->lr = 0x00844BF4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844C24;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AF0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
loc_00844C24:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00844C30; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00844C60;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AEC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
loc_00844C60:
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->fpr[0] = (double)(float)sqrt(ctx->fpr[0]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008440EC;
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x1A0);
    ctx->gpr[7] = (int64_t)(int32_t)((uint32_t)0xAAAA << 16);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6AE8);
    ctx->gpr[6] = (int64_t)(int32_t)(0xA0);
    ctx->gpr[9] = ctx->gpr[5] + (int64_t)(2);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6AE4);
    ctx->gpr[7] = ctx->gpr[7] | 0xAAAB;
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x1B0);
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x6ADC);
    { uint64_t* d = (uint64_t*)&ctx->vr[4]; uint64_t* a = (uint64_t*)&ctx->vr[4]; uint64_t* b = (uint64_t*)&ctx->vr[4]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[19]) & ~0xFULL; vm_raw_ld16(&ctx->vr[25], ea); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[22], ctx->fpr[11]));
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)-1; ppu_vstu4(&ctx->vr[15],v); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)1; ppu_vstu4(&ctx->vr[31],v); }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6AE0);
    vmx_vslw(&ctx->vr[15], &ctx->vr[15], &ctx->vr[15]);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6AD8);
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[5], ea); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x228);
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[19], ea); }
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x220);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[23], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x288);
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[18], ea); }
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x240);
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x230);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[24], ea); }
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x238);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[16], ea); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[23], ea); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x208);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x1E0);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[22], ea); }
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0x1AC);
    { uint64_t ea = (ctx->gpr[16]) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[30], ea); }
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[14], ea); }
    ctx->gpr[10] = ctx->gpr[11] + (int64_t)(0x10);
    { uint64_t ea = (ctx->gpr[15]) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    ctx->gpr[7] = (uint32_t)(((uint64_t)(uint32_t)ctx->gpr[9] * (uint64_t)(uint32_t)ctx->gpr[7]) >> 32);
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[7], 31, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 1, 0, 30, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[7];
    ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x268);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 4, 0, 27, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[7];
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0x1D0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[29], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[6])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[29]); }
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x250);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x250);
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x20);
    ctx->gpr[11] = (int64_t)(int32_t)(0x10);
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1C0);
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[9]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[11]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[9]; uint8_t* d = (uint8_t*)&ctx->vr[9]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[11]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[29])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t* a = (uint8_t*)&ctx->vr[9]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[3]; uint8_t* d = (uint8_t*)&ctx->vr[12]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[3]; uint8_t* d = (uint8_t*)&ctx->vr[3]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    { uint8_t* a = (uint8_t*)&ctx->vr[9]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[9]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[3], &ctx->vr[4], &ctx->vr[12], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[29])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[29]; uint32_t val = ((uint32_t*)&ctx->vr[29])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[9], &ctx->vr[4], &ctx->vr[3], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[11], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[4], &ctx->vr[12], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[3], 16); memcpy(tmp+16, &ctx->vr[3], 16); memcpy(&ctx->vr[3], tmp + 8, 16); }
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[5], &ctx->vr[4], &ctx->vr[9], ctx->vscr);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0x1D0);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[4], &ctx->vr[12], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[5], &ctx->vr[4], &ctx->vr[3], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[19]; uint8_t* d = (uint8_t*)&ctx->vr[1]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[8]; uint8_t* b = (uint8_t*)&ctx->vr[8]; uint8_t* c = (uint8_t*)&ctx->vr[19]; uint8_t* d = (uint8_t*)&ctx->vr[8]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[7]; uint8_t* b = (uint8_t*)&ctx->vr[7]; uint8_t* c = (uint8_t*)&ctx->vr[19]; uint8_t* d = (uint8_t*)&ctx->vr[7]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vnmsubfp(&ctx->vr[2], &ctx->vr[9], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[1], &ctx->vr[5], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[10]; uint8_t* c = (uint8_t*)&ctx->vr[19]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[8], 16); memcpy(tmp+16, &ctx->vr[8], 16); memcpy(&ctx->vr[17], tmp + 8, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[11]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[19]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[7], 16); memcpy(tmp+16, &ctx->vr[7], 16); memcpy(&ctx->vr[27], tmp + 8, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[6]; uint8_t* b = (uint8_t*)&ctx->vr[6]; uint8_t* c = (uint8_t*)&ctx->vr[19]; uint8_t* d = (uint8_t*)&ctx->vr[6]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[19], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[11], 16); memcpy(tmp+16, &ctx->vr[11], 16); memcpy(&ctx->vr[28], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[5], &ctx->vr[4], &ctx->vr[11], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[6], 16); memcpy(tmp+16, &ctx->vr[6], 16); memcpy(&ctx->vr[26], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[9], &ctx->vr[2], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[5], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[13], &ctx->vr[5], &ctx->vr[13], &ctx->vr[28], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[12], &ctx->vr[2], &ctx->vr[8], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[3], &ctx->vr[0], &ctx->vr[7], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[13], 16); memcpy(&ctx->vr[13], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[17], &ctx->vr[12], &ctx->vr[8], &ctx->vr[17], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[10], &ctx->vr[12], &ctx->vr[13], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[3], &ctx->vr[0], &ctx->vr[27], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[3], &ctx->vr[17], &ctx->vr[11], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[19], &ctx->vr[12], &ctx->vr[10], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[12], &ctx->vr[0], &ctx->vr[6], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[11], &ctx->vr[28], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[9], &ctx->vr[19], &ctx->vr[7], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[6], &ctx->vr[26], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[5], &ctx->vr[4], &ctx->vr[3], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[7], &ctx->vr[27], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[5])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[5])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[5])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[5]; uint32_t val = ((uint32_t*)&ctx->vr[5])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[5], &ctx->vr[5], &ctx->vr[13], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[5], ctx->vscr);
    vmx_vrefp(&ctx->vr[1], &ctx->vr[0], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[25], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], &ctx->vr[1], ctx->vscr);
    (void)vmx_vcmpeqfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[0]);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[13]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B54);
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[4], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x6B00);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[3], &ctx->vr[0], ctx->vscr);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)2; ppu_vstu4(&ctx->vr[3],v); }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[12], &ctx->vr[29], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[6])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[9]); }
    { uint8_t b = (uint8_t)((ctx->gpr[8]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[0])[i] = b + i; }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[1]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B08);
    { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[12])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t* a = (uint8_t*)&ctx->vr[12]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)3; ppu_vstu4(&ctx->vr[0],v); }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[7], tmp + 12, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[12]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[30]; uint8_t* d = (uint8_t*)&ctx->vr[6]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[4], &ctx->vr[24], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[18], &ctx->vr[1], &ctx->vr[4], &ctx->vr[18], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[24]; uint64_t* a=(uint64_t*)&ctx->vr[24]; uint64_t* b=(uint64_t*)&ctx->vr[18]; uint64_t* c=(uint64_t*)&ctx->vr[15]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vaddfp(&ctx->vr[18], &ctx->vr[18], &ctx->vr[24], ctx->vscr);
    vmx_vctsxs(&ctx->vr[18], &ctx->vr[18], 0, &ctx->vscr);
    vmx_vcfsx(&ctx->vr[13], &ctx->vr[18], 0);
    { uint64_t* d=(uint64_t*)&ctx->vr[18]; uint64_t* a=(uint64_t*)&ctx->vr[18]; uint64_t* b=(uint64_t*)&ctx->vr[0]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[18]; uint64_t* b=(uint64_t*)&ctx->vr[31]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[10]; uint64_t* a=(uint64_t*)&ctx->vr[18]; uint64_t* b=(uint64_t*)&ctx->vr[3]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { const uint8_t* a=(const uint8_t*)&ctx->vr[18]; const uint8_t* b=(const uint8_t*)&ctx->vr[31]; uint8_t d[16]; for(int i=0;i<4;i++){ uint32_t x=0,y=0; for(int k=0;k<4;k++){ x=(uint32_t)((x<<8)|a[i*4+k]); y=(uint32_t)((y<<8)|b[i*4+k]); } uint32_t r=(uint32_t)(x+y); for(int k=0;k<4;k++) d[i*4+k]=(uint8_t)(r>>(8*(4-1-k))); } memcpy(&ctx->vr[18], d, 16); }
    (void)vmx_vcmpequw(&ctx->vr[11], &ctx->vr[11], &ctx->vr[4]);
    (void)vmx_vcmpequw(&ctx->vr[10], &ctx->vr[10], &ctx->vr[4]);
    vmx_vnmsubfp(&ctx->vr[23], &ctx->vr[13], &ctx->vr[1], &ctx->vr[23], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[31]; uint64_t* a=(uint64_t*)&ctx->vr[18]; uint64_t* b=(uint64_t*)&ctx->vr[31]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[18]; uint64_t* a=(uint64_t*)&ctx->vr[18]; uint64_t* b=(uint64_t*)&ctx->vr[3]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B04);
    (void)vmx_vcmpequw(&ctx->vr[31], &ctx->vr[31], &ctx->vr[4]);
    (void)vmx_vcmpequw(&ctx->vr[18], &ctx->vr[18], &ctx->vr[4]);
    vmx_vnmsubfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[23], &ctx->vr[22], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[13], &ctx->vr[4], &ctx->vr[13], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[0], &ctx->vr[21], &ctx->vr[16], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[14], &ctx->vr[0], &ctx->vr[20], &ctx->vr[14], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[0], &ctx->vr[4], &ctx->vr[13], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[16], &ctx->vr[3], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B54);
    vmx_vmaddfp(&ctx->vr[14], &ctx->vr[14], &ctx->vr[3], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[16], &ctx->vr[25], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[14], &ctx->vr[14], &ctx->vr[13], &ctx->vr[1], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[16]; uint64_t* b=(uint64_t*)&ctx->vr[14]; uint64_t* c=(uint64_t*)&ctx->vr[11]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[16]; uint64_t* a=(uint64_t*)&ctx->vr[16]; uint64_t* b=(uint64_t*)&ctx->vr[14]; uint64_t* c=(uint64_t*)&ctx->vr[31]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[15]; uint64_t* b = (uint64_t*)&ctx->vr[11]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[15]; uint64_t* a = (uint64_t*)&ctx->vr[15]; uint64_t* b = (uint64_t*)&ctx->vr[16]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[11]; uint64_t* c=(uint64_t*)&ctx->vr[10]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[15]; uint64_t* a=(uint64_t*)&ctx->vr[15]; uint64_t* b=(uint64_t*)&ctx->vr[16]; uint64_t* c=(uint64_t*)&ctx->vr[18]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[4], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t* d=(uint64_t*)&ctx->vr[9]; uint64_t* a=(uint64_t*)&ctx->vr[9]; uint64_t* b=(uint64_t*)&ctx->vr[15]; uint64_t* c=(uint64_t*)&ctx->vr[0]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[9])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[9], &ctx->vr[4], &ctx->vr[12], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[1], tmp + 12, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[9]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[3]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[9]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[30]; uint8_t* d = (uint8_t*)&ctx->vr[30]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[4], &ctx->vr[12], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[11], &ctx->vr[7], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[11], &ctx->vr[7], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[13], &ctx->vr[9], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[3], &ctx->vr[8], &ctx->vr[6], ctx->vscr);
    vmx_vsubfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vmx_vnmsubfp(&ctx->vr[30], &ctx->vr[30], &ctx->vr[2], &ctx->vr[5], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[30]; uint64_t* a=(uint64_t*)&ctx->vr[30]; uint64_t* b=(uint64_t*)&ctx->vr[1]; uint64_t* c=(uint64_t*)&ctx->vr[0]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[30]); }
    goto loc_008440EC;
loc_00844FEC:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[5]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[2], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008446E8;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008446E8;
    goto loc_00844808;
loc_00845014:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[1]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[6], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844740;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00844740;
    goto loc_00844808;
loc_0084503C:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[1]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[3], ctx->fpr[5], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844798;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[7]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00844798;
    goto loc_00844808;
loc_00845064:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008440E8;
    ctx->lr = 0x00845074; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845158;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B38);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845158;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AF8);
    ctx->fpr[31] = ctx->fpr[24];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008450B4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6AF4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_008450B4:
    ctx->lr = 0x008450B8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008450E8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AF0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
loc_008450E8:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x008450F4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00845124;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AEC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
loc_00845124:
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[11], ctx->fpr[11], ctx->fpr[0], 0, 0));
    ctx->fpr[10] = (double)(float)sqrt(ctx->fpr[0]);
    { double a = ctx->fpr[10]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00845220;
loc_00845148:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[1] + 0x1A8, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    goto loc_00844224;
loc_00845158:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    goto loc_0084421C;
loc_00845160:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x39D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6B74);
    ctx->lr = 0x00845170; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008437B4;
loc_00845178:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[7], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[6]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008449A8;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[6]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008449A8;
loc_0084519C:
    ctx->gpr[11] = ctx->gpr[9] | ctx->gpr[0];
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[11]);
    goto loc_008449A8;
loc_008451A8:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[2]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[9], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084495C;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084495C;
    goto loc_0084519C;
loc_008451D0:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[7]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[6], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00844908;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00844908;
    goto loc_0084519C;
loc_008451F8:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[3], ctx->fpr[0], 1, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008448B4;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[2]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008448B4;
    goto loc_0084519C;
loc_00845220:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x250);
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    ctx->gpr[11] = (int64_t)(int32_t)(0x20);
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x1B0);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6AE8);
    { uint64_t* d = (uint64_t*)&ctx->vr[4]; uint64_t* a = (uint64_t*)&ctx->vr[4]; uint64_t* b = (uint64_t*)&ctx->vr[4]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x6AE4);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[10]));
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x6ADC);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[4], 16); memcpy(tmp+16, &ctx->vr[4], 16); memcpy(&ctx->vr[31], tmp + 0, 16); }
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xC0);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[31] = (int64_t)(int32_t)(0xA0);
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6AE0);
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6AD8);
    { uint8_t* a = (uint8_t*)&ctx->vr[11]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[1]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[30] = vm_read32(ctx->gpr[1] + 0x188);
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[11]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[3]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x288);
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x18);
    { uint8_t* a = (uint8_t*)&ctx->vr[11]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint8_t* a = (uint8_t*)&ctx->vr[11]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t ea = (ctx->gpr[19]) & ~0xFULL; vm_raw_ld16(&ctx->vr[14], ea); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[5], 16); memcpy(tmp+16, &ctx->vr[5], 16); memcpy(&ctx->vr[6], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[2], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[10], &ctx->vr[4], &ctx->vr[3], ctx->vscr);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[11], &ctx->vr[4], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[11], &ctx->vr[4], &ctx->vr[3], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[11], &ctx->vr[4], &ctx->vr[2], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[5], &ctx->vr[4], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[6], &ctx->vr[4], &ctx->vr[3], ctx->vscr);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[10]));
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[9]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[9]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[8]; uint8_t* b = (uint8_t*)&ctx->vr[8]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[8]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[7]; uint8_t* b = (uint8_t*)&ctx->vr[7]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[7]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[10]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[6]; uint8_t* b = (uint8_t*)&ctx->vr[6]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[6]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[13], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[8], 16); memcpy(tmp+16, &ctx->vr[8], 16); memcpy(&ctx->vr[16], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[19], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[6], 16); memcpy(tmp+16, &ctx->vr[6], 16); memcpy(&ctx->vr[17], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[18], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[7], 16); memcpy(tmp+16, &ctx->vr[7], 16); memcpy(&ctx->vr[15], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[12], &ctx->vr[5], &ctx->vr[4], &ctx->vr[13], ctx->vscr);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    vmx_vnmsubfp(&ctx->vr[13], &ctx->vr[11], &ctx->vr[4], &ctx->vr[13], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[11], &ctx->vr[4], &ctx->vr[6], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[5], &ctx->vr[12], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[11], &ctx->vr[13], &ctx->vr[0], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[1], &ctx->vr[11], &ctx->vr[1], &ctx->vr[17], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[3], &ctx->vr[12], &ctx->vr[10], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[1], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[2], &ctx->vr[0], &ctx->vr[8], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[9], &ctx->vr[3], &ctx->vr[1], &ctx->vr[9], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[19], &ctx->vr[3], &ctx->vr[10], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[2], &ctx->vr[0], &ctx->vr[16], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[18], &ctx->vr[3], &ctx->vr[9], &ctx->vr[18], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[2], &ctx->vr[19], &ctx->vr[6], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[3], &ctx->vr[0], &ctx->vr[7], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[5], &ctx->vr[18], &ctx->vr[8], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[6], &ctx->vr[17], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[7], &ctx->vr[15], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[5], &ctx->vr[5], &ctx->vr[8], &ctx->vr[16], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[4], &ctx->vr[2], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[11])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[11])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[11])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[11])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[13], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[11], ctx->vscr);
    vmx_vrefp(&ctx->vr[1], &ctx->vr[0], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[14], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], &ctx->vr[1], ctx->vscr);
    (void)vmx_vcmpeqfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[0]);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[13]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[5], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[2], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[3], &ctx->vr[4], &ctx->vr[1], ctx->vscr);
loc_008453AC:
    ctx->gpr[0] = ctx->gpr[5] + ctx->gpr[6];
    ctx->gpr[9] = (uint64_t)((ctx->gpr[7] & 0x20) ? 0u : ((uint32_t)ctx->gpr[4] << (ctx->gpr[7] & 0x1F)));
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = ctx->gpr[9] & ctx->gpr[11];
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0x10);
    ctx->gpr[5] = ctx->gpr[5] + (int64_t)(8);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_008453E0;
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008453AC;
    goto loc_00845148;
loc_008453E0:
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x1AC);
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t b = (uint8_t)((ctx->gpr[3]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[1])[i] = b + i; }
    { uint64_t ea = (ctx->gpr[3]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[31], &ctx->vr[0], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[13])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[12], &ctx->vr[31], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[11], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[10], &ctx->vr[0], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[31])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008453AC;
    goto loc_00845148;
}

void june_evt_func_008D0C8C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_23 = ctx->gpr[23];
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D88);
    vm_write64(ctx->gpr[1] + -0x2B0, ctx->gpr[1]); ctx->gpr[1] += -0x2B0;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[20]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x180);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[21]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x190);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[22]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1A0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[23]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1B0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[24]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1C0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[25]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1D0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[26]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1E0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[27]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1F0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[28]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x200);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[29]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x210);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[30]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x220);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[31]); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x278, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x2C0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x280, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x288, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x290, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x298, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x2A0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x2A8, tmp); }
    vm_write64(ctx->gpr[1] + 0x238, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x240, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x248, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x250, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x258, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x260, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x268, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x270, ctx->gpr[30]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D170C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_008D0D50:
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[1]));
    ctx->lr = 0x008D0D58; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D16E4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_008D0D6C:
    ctx->fpr[27] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->lr = 0x008D0D74; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D0D98;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D74);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D1740;
loc_008D0D98:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[28] = ctx->fpr[30];
loc_008D0DA0:
    ctx->lr = 0x008D0DA4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D0DC8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D74);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D17A0;
loc_008D0DC8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[29] = ctx->fpr[31];
    ctx->fpr[0] = ctx->fpr[31];
loc_008D0DD4:
    { double a = ctx->fpr[27]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D0DE4;
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D1714;
loc_008D0DE4:
    ctx->gpr[29] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[25] = vm_read32(ctx->gpr[2] + -0x4D64);
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    ctx->gpr[24] = vm_read32(ctx->gpr[2] + -0x4D60);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[23] = vm_read32(ctx->gpr[2] + -0x4D58);
    ctx->gpr[11] = (int64_t)(int32_t)(0x20);
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x4D5C);
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x4D54);
    { uint64_t* d = (uint64_t*)&ctx->vr[2]; uint64_t* a = (uint64_t*)&ctx->vr[2]; uint64_t* b = (uint64_t*)&ctx->vr[2]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[29]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t* d = (uint64_t*)&ctx->vr[4]; uint64_t* a = (uint64_t*)&ctx->vr[4]; uint64_t* b = (uint64_t*)&ctx->vr[4]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[25]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[11])) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint64_t ea = (ctx->gpr[24]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    { uint64_t ea = (ctx->gpr[28]) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x4D50);
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[9]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[11]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[11]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[1]; uint64_t* b = (uint64_t*)&ctx->vr[1]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[23]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[17], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[19]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[0], &ctx->vr[1], &ctx->vr[19], ctx->vscr);
    { uint64_t ea = (ctx->gpr[26]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[18], tmp + 8, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[1]; uint64_t* b = (uint64_t*)&ctx->vr[1]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[10], &ctx->vr[2], &ctx->vr[0], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[7], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[13], &ctx->vr[2], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[13], &ctx->vr[2], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[4], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[13], &ctx->vr[4], &ctx->vr[18], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[12]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[6]; uint8_t* d = (uint8_t*)&ctx->vr[12]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[10], &ctx->vr[9], &ctx->vr[12], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[11], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[12], &ctx->vr[13], &ctx->vr[9], &ctx->vr[12], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[6]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[5]; uint8_t* c = (uint8_t*)&ctx->vr[6]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[2]; uint8_t* b = (uint8_t*)&ctx->vr[2]; uint8_t* c = (uint8_t*)&ctx->vr[6]; uint8_t* d = (uint8_t*)&ctx->vr[2]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[13], &ctx->vr[9], &ctx->vr[0], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[16], tmp + 8, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[7]; uint8_t* b = (uint8_t*)&ctx->vr[7]; uint8_t* c = (uint8_t*)&ctx->vr[6]; uint8_t* d = (uint8_t*)&ctx->vr[7]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[2], 16); memcpy(tmp+16, &ctx->vr[2], 16); memcpy(&ctx->vr[31], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[5], 16); memcpy(tmp+16, &ctx->vr[5], 16); memcpy(&ctx->vr[15], tmp + 8, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[4]; uint8_t* b = (uint8_t*)&ctx->vr[4]; uint8_t* c = (uint8_t*)&ctx->vr[6]; uint8_t* d = (uint8_t*)&ctx->vr[4]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[10], &ctx->vr[3], &ctx->vr[11], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[7], 16); memcpy(tmp+16, &ctx->vr[7], 16); memcpy(&ctx->vr[14], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[13], &ctx->vr[12], &ctx->vr[11], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[13], &ctx->vr[1], &ctx->vr[7], ctx->vscr);
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[1])[i] = b + i; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[4], 16); memcpy(tmp+16, &ctx->vr[4], 16); memcpy(&ctx->vr[12], tmp + 8, 16); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t* a = (uint8_t*)&ctx->vr[17]; uint8_t* b = (uint8_t*)&ctx->vr[17]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[17]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vnmsubfp(&ctx->vr[9], &ctx->vr[13], &ctx->vr[9], &ctx->vr[16], ctx->vscr);
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[1])[i] = b + i; }
    { uint32_t* d = (uint32_t*)&ctx->vr[17]; uint32_t val = ((uint32_t*)&ctx->vr[17])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[19], &ctx->vr[3], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[11], 16); memcpy(tmp+16, &ctx->vr[11], 16); memcpy(&ctx->vr[11], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[8], &ctx->vr[13], &ctx->vr[8], &ctx->vr[14], ctx->vscr);
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[3]; uint8_t* b = (uint8_t*)&ctx->vr[3]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[3]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vnmsubfp(&ctx->vr[11], &ctx->vr[18], &ctx->vr[11], &ctx->vr[2], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[9], tmp + 8, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[3]; uint32_t val = ((uint32_t*)&ctx->vr[3])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vnmsubfp(&ctx->vr[16], &ctx->vr[19], &ctx->vr[0], &ctx->vr[16], ctx->vscr);
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[0])[i] = b + i; }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[18], &ctx->vr[9], &ctx->vr[5], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[6]; uint8_t* b = (uint8_t*)&ctx->vr[6]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[6]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[8], 16); memcpy(tmp+16, &ctx->vr[8], 16); memcpy(&ctx->vr[8], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[18], &ctx->vr[11], &ctx->vr[31], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[6]; uint32_t val = ((uint32_t*)&ctx->vr[6])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vnmsubfp(&ctx->vr[5], &ctx->vr[19], &ctx->vr[8], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[18], &ctx->vr[16], &ctx->vr[7], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[9], &ctx->vr[18], &ctx->vr[9], &ctx->vr[15], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[19], &ctx->vr[11], &ctx->vr[4], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[15], &ctx->vr[19], &ctx->vr[5], &ctx->vr[15], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[18], &ctx->vr[18], &ctx->vr[7], &ctx->vr[14], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[4], &ctx->vr[10], &ctx->vr[9], &ctx->vr[4], ctx->vscr);
    { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    vmx_vnmsubfp(&ctx->vr[19], &ctx->vr[19], &ctx->vr[11], &ctx->vr[12], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[10], &ctx->vr[15], &ctx->vr[2], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[9], &ctx->vr[18], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[10], &ctx->vr[4], &ctx->vr[12], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[2], &ctx->vr[31], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[13])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[13])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[11], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = (ctx->gpr[27]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    vmx_vrefp(&ctx->vr[1], &ctx->vr[0], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[13], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], &ctx->vr[1], ctx->vscr);
    (void)vmx_vcmpeqfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[0]);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[13]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    vmx_vmaddfp(&ctx->vr[31], &ctx->vr[10], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[18], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[31], &ctx->vr[0], &ctx->vr[17], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[9]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[19], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[17], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[9], &ctx->vr[0], &ctx->vr[6], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[9]); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[1], &ctx->vr[0], &ctx->vr[3], ctx->vscr);
    vmx_vaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[3], ctx->vscr);
    vmx_vaddfp(&ctx->vr[24], &ctx->vr[12], &ctx->vr[6], ctx->vscr);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D1214;
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D1214;
loc_008D1000:
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    { uint64_t ea = (ctx->gpr[25]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t ea = (ctx->gpr[24]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint8_t* a = (uint8_t*)&ctx->vr[31]; uint8_t* b = (uint8_t*)&ctx->vr[24]; uint8_t* c = (uint8_t*)&ctx->vr[13]; uint8_t* d = (uint8_t*)&ctx->vr[11]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[11] = (int64_t)(int32_t)(0x20);
    { uint8_t* a = (uint8_t*)&ctx->vr[31]; uint8_t* b = (uint8_t*)&ctx->vr[24]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[8]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t ea = (ctx->gpr[28]) & ~0xFULL; vm_raw_ld16(&ctx->vr[16], ea); }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint64_t ea = (ctx->gpr[26]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    { uint64_t ea = (ctx->gpr[27]) & ~0xFULL; vm_raw_ld16(&ctx->vr[31], ea); }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[9]; uint8_t* c = (uint8_t*)&ctx->vr[13]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t* d = (uint64_t*)&ctx->vr[9]; uint64_t* a = (uint64_t*)&ctx->vr[9]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[23]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[8]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[2]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[1]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[8]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[8]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[11]; uint8_t* c = (uint8_t*)&ctx->vr[12]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[1], &ctx->vr[9], &ctx->vr[2], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[3], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[8], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[8], 16); memcpy(tmp+16, &ctx->vr[8], 16); memcpy(&ctx->vr[7], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[13], &ctx->vr[9], &ctx->vr[8], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[13], &ctx->vr[9], &ctx->vr[2], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[13], &ctx->vr[9], &ctx->vr[3], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[9], &ctx->vr[2], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[10]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[1]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[6]; uint8_t* b = (uint8_t*)&ctx->vr[6]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[6]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[4]; uint8_t* b = (uint8_t*)&ctx->vr[4]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[4]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vnmsubfp(&ctx->vr[18], &ctx->vr[8], &ctx->vr[9], &ctx->vr[10], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[12], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[13], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[19], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[10], &ctx->vr[13], &ctx->vr[9], &ctx->vr[10], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[7]; uint8_t* b = (uint8_t*)&ctx->vr[7]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[7]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[6], 16); memcpy(tmp+16, &ctx->vr[6], 16); memcpy(&ctx->vr[17], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[4], 16); memcpy(tmp+16, &ctx->vr[4], 16); memcpy(&ctx->vr[14], tmp + 8, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[5]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[13], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[7], 16); memcpy(tmp+16, &ctx->vr[7], 16); memcpy(&ctx->vr[15], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[18], &ctx->vr[8], &ctx->vr[18], &ctx->vr[12], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[13], &ctx->vr[0], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[13], &ctx->vr[10], &ctx->vr[12], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[5], 16); memcpy(tmp+16, &ctx->vr[5], 16); memcpy(&ctx->vr[10], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[11], &ctx->vr[13], &ctx->vr[11], &ctx->vr[15], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[2], &ctx->vr[18], &ctx->vr[1], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[12], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[3], &ctx->vr[0], &ctx->vr[6], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[12], &ctx->vr[3], &ctx->vr[12], &ctx->vr[4], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[11], 16); memcpy(tmp+16, &ctx->vr[11], 16); memcpy(&ctx->vr[11], tmp + 8, 16); }
    vmx_vnmsubfp(&ctx->vr[19], &ctx->vr[2], &ctx->vr[1], &ctx->vr[19], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[6], &ctx->vr[2], &ctx->vr[11], &ctx->vr[6], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[3], &ctx->vr[0], &ctx->vr[17], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[3], &ctx->vr[12], &ctx->vr[14], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[3], &ctx->vr[19], &ctx->vr[7], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[2], &ctx->vr[6], &ctx->vr[17], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[2], &ctx->vr[12], &ctx->vr[5], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[5], &ctx->vr[8], &ctx->vr[0], &ctx->vr[5], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[7], &ctx->vr[15], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[8], &ctx->vr[17], &ctx->vr[4], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[12], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[8], &ctx->vr[5], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[9], &ctx->vr[3], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[4], &ctx->vr[14], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[13])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[12]; uint32_t val = ((uint32_t*)&ctx->vr[13])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[12], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[13], ctx->vscr);
    vmx_vrefp(&ctx->vr[1], &ctx->vr[0], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[31], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], &ctx->vr[1], ctx->vscr);
    (void)vmx_vcmpeqfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[0]);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[13]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[10]); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    { uint64_t ea = (ctx->gpr[29]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[3]); }
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[11])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[8]); }
    { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
loc_008D1168:
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x180);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x190);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x280); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x288); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x290); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x298); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[22], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1A0);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2A0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2A8); memcpy(&ctx->fpr[31], &tmp, 8); }
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[23], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1B0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[24], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1C0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[25], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1D0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[26], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1E0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[27], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1F0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[28], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x200);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[29], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x210);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[30], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x220);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[31], ea); }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x2C0);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x2B0);
    ctx->lr = ctx->gpr[0];
    return;
loc_008D1214:
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x80);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[24], &ctx->vr[0], &ctx->vr[24], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[24], 16); memcpy(tmp+16, &ctx->vr[24], 16); memcpy(&ctx->vr[0], tmp + 4, 16); }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[13], &ctx->vr[0], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[1], tmp + 8, 16); }
    vmx_vaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[1] + ctx->gpr[0]); ea &= ~3ULL; { uint32_t _e; memcpy(&_e, (const uint8_t*)&ctx->vr[1] + (ea & 15), 4); VM_RAW_ST(uint32_t, ea, _e); } }
    { uint32_t tmp = vm_read32((ctx->gpr[1] + ctx->gpr[0])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[13] = (double)(float)sqrt(ctx->fpr[1]);
    { double a = ctx->fpr[13]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D1730;
loc_008D1244:
    ctx->fpr[0] = -ctx->fpr[13];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D4C);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4D48);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)-1; ppu_vstu4(&ctx->vr[19],v); }
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4D44);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    vmx_vslw(&ctx->vr[19], &ctx->vr[19], &ctx->vr[19]);
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x4D3C);
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x4D14);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)3; ppu_vstu4(&ctx->vr[2],v); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)1; ppu_vstu4(&ctx->vr[5],v); }
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4D34);
    { uint32_t* d = (uint32_t*)&ctx->vr[26]; uint32_t val = ((uint32_t*)&ctx->vr[31])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D40);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[7])[i] = b + i; }
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[1])[i] = b + i; }
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4D24);
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[0])[i] = b + i; }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)2; ppu_vstu4(&ctx->vr[1],v); }
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[17], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4D1C);
    { uint8_t* a = (uint8_t*)&ctx->vr[12]; uint8_t* b = (uint8_t*)&ctx->vr[12]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[12]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[25], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D10);
    { uint8_t* a = (uint8_t*)&ctx->vr[3]; uint8_t* b = (uint8_t*)&ctx->vr[3]; uint8_t* c = (uint8_t*)&ctx->vr[7]; uint8_t* d = (uint8_t*)&ctx->vr[3]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[0], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[12]; uint32_t val = ((uint32_t*)&ctx->vr[12])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[11] = ctx->gpr[2] + (int64_t)(-19976);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[16], ea); }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x100);
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x120);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[0], &ctx->vr[9], ctx->vscr);
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x4D18);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
    { uint64_t ea = ctx->gpr[11]; uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[22]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
    { uint32_t* d = (uint32_t*)&ctx->vr[3]; uint32_t val = ((uint32_t*)&ctx->vr[3])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[22]; uint32_t val = ((uint32_t*)&ctx->vr[22])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[18], ea); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[13], &ctx->vr[0], &ctx->vr[10], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[12], &ctx->vr[0], &ctx->vr[10], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[9]; uint64_t* b=(uint64_t*)&ctx->vr[11]; uint64_t* c=(uint64_t*)&ctx->vr[19]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[0], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[31])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t* d=(uint64_t*)&ctx->vr[9]; uint64_t* a=(uint64_t*)&ctx->vr[9]; uint64_t* b=(uint64_t*)&ctx->vr[10]; uint64_t* c=(uint64_t*)&ctx->vr[19]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x110);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[31])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[30], tmp + 12, 16); }
    vmx_vctsxs(&ctx->vr[11], &ctx->vr[11], 0, &ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[31])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[30], 16); memcpy(tmp+16, &ctx->vr[22], 16); memcpy(&ctx->vr[30], tmp + 4, 16); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    vmx_vctsxs(&ctx->vr[10], &ctx->vr[10], 0, &ctx->vscr);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    { uint8_t b = (uint8_t)((ctx->gpr[31]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[0])[i] = b + i; }
    { uint64_t ea = (ctx->gpr[31]) & ~0xFULL; vm_raw_ld16(&ctx->vr[15], ea); }
    { uint8_t* a = (uint8_t*)&ctx->vr[15]; uint8_t* b = (uint8_t*)&ctx->vr[15]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[15]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint32_t* d = (uint32_t*)&ctx->vr[23]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vcfsx(&ctx->vr[7], &ctx->vr[11], 0);
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[2]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint32_t* d = (uint32_t*)&ctx->vr[15]; uint32_t val = ((uint32_t*)&ctx->vr[15])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xC0);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t* d=(uint64_t*)&ctx->vr[28]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[1]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[9]); }
    vmx_vcfsx(&ctx->vr[9], &ctx->vr[10], 0);
    { uint64_t* d=(uint64_t*)&ctx->vr[10]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[2]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[2]; uint64_t* a = (uint64_t*)&ctx->vr[2]; uint64_t* b = (uint64_t*)&ctx->vr[2]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x150);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[31], &ctx->vr[2], &ctx->vr[3], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[13], &ctx->vr[7], &ctx->vr[13], &ctx->vr[4], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[14]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[5]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[31]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[5]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint64_t* d=(uint64_t*)&ctx->vr[27]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[1]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    (void)vmx_vcmpequw(&ctx->vr[14], &ctx->vr[14], &ctx->vr[2]);
    vmx_vnmsubfp(&ctx->vr[4], &ctx->vr[9], &ctx->vr[12], &ctx->vr[4], ctx->vscr);
    { uint64_t* d = (uint64_t*)&ctx->vr[12]; uint64_t* a = (uint64_t*)&ctx->vr[12]; uint64_t* b = (uint64_t*)&ctx->vr[12]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { const uint8_t* a=(const uint8_t*)&ctx->vr[10]; const uint8_t* b=(const uint8_t*)&ctx->vr[5]; uint8_t d[16]; for(int i=0;i<4;i++){ uint32_t x=0,y=0; for(int k=0;k<4;k++){ x=(uint32_t)((x<<8)|a[i*4+k]); y=(uint32_t)((y<<8)|b[i*4+k]); } uint32_t r=(uint32_t)(x+y); for(int k=0;k<4;k++) d[i*4+k]=(uint8_t)(r>>(8*(4-1-k))); } memcpy(&ctx->vr[10], d, 16); }
    { const uint8_t* a=(const uint8_t*)&ctx->vr[11]; const uint8_t* b=(const uint8_t*)&ctx->vr[5]; uint8_t d[16]; for(int i=0;i<4;i++){ uint32_t x=0,y=0; for(int k=0;k<4;k++){ x=(uint32_t)((x<<8)|a[i*4+k]); y=(uint32_t)((y<<8)|b[i*4+k]); } uint32_t r=(uint32_t)(x+y); for(int k=0;k<4;k++) d[i*4+k]=(uint8_t)(r>>(8*(4-1-k))); } memcpy(&ctx->vr[11], d, 16); }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    vmx_vaddfp(&ctx->vr[3], &ctx->vr[24], &ctx->vr[3], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[13], &ctx->vr[25], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[29]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[5]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[10]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[1]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint64_t* d=(uint64_t*)&ctx->vr[5]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[5]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint32_t* d = (uint32_t*)&ctx->vr[24]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[1]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    vmx_vnmsubfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[4], &ctx->vr[25], ctx->vscr);
    (void)vmx_vcmpequw(&ctx->vr[10], &ctx->vr[10], &ctx->vr[12]);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xF0);
    (void)vmx_vcmpequw(&ctx->vr[5], &ctx->vr[5], &ctx->vr[12]);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    (void)vmx_vcmpequw(&ctx->vr[11], &ctx->vr[11], &ctx->vr[12]);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[7], &ctx->vr[2], &ctx->vr[7], ctx->vscr);
    (void)vmx_vcmpequw(&ctx->vr[31], &ctx->vr[31], &ctx->vr[12]);
    (void)vmx_vcmpequw(&ctx->vr[27], &ctx->vr[27], &ctx->vr[12]);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    (void)vmx_vcmpequw(&ctx->vr[28], &ctx->vr[28], &ctx->vr[12]);
    (void)vmx_vcmpequw(&ctx->vr[29], &ctx->vr[29], &ctx->vr[12]);
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[9], &ctx->vr[2], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t* d = (uint32_t*)&ctx->vr[25]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xE0);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x4D38);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x150);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[4], &ctx->vr[0], &ctx->vr[8], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x4D38);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x4D30);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[2], &ctx->vr[0], &ctx->vr[8], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x140);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[2], &ctx->vr[0], &ctx->vr[17], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x4D30);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x160);
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[4], &ctx->vr[0], &ctx->vr[17], ctx->vscr);
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[2], &ctx->vr[0], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[4], &ctx->vr[0], &ctx->vr[7], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D2C);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x150);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], &ctx->vr[4], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[1], &ctx->vr[2], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x140);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D28);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x140);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[1], &ctx->vr[2], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[17], &ctx->vr[1], &ctx->vr[4], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x150);
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[17], &ctx->vr[7], &ctx->vr[12], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D50);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x140);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[13], &ctx->vr[4], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[13], &ctx->vr[2], ctx->vscr);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x4D20);
    { uint64_t* d = (uint64_t*)&ctx->vr[2]; uint64_t* a = (uint64_t*)&ctx->vr[2]; uint64_t* b = (uint64_t*)&ctx->vr[2]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[13], &ctx->vr[9], &ctx->vr[4], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    { uint64_t* d=(uint64_t*)&ctx->vr[31]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[17]; uint64_t* c=(uint64_t*)&ctx->vr[31]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[17]; uint64_t* c=(uint64_t*)&ctx->vr[5]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[13]; uint64_t* a = (uint64_t*)&ctx->vr[19]; uint64_t* b = (uint64_t*)&ctx->vr[31]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[14]; uint64_t* a=(uint64_t*)&ctx->vr[8]; uint64_t* b=(uint64_t*)&ctx->vr[1]; uint64_t* c=(uint64_t*)&ctx->vr[14]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[8]; uint64_t* a=(uint64_t*)&ctx->vr[8]; uint64_t* b=(uint64_t*)&ctx->vr[1]; uint64_t* c=(uint64_t*)&ctx->vr[29]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[31]; uint64_t* c=(uint64_t*)&ctx->vr[28]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[19]; uint64_t* b = (uint64_t*)&ctx->vr[14]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[12]; uint64_t* a = (uint64_t*)&ctx->vr[19]; uint64_t* b = (uint64_t*)&ctx->vr[8]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[19]; uint64_t* a = (uint64_t*)&ctx->vr[19]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[14]; uint64_t* c=(uint64_t*)&ctx->vr[27]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[12]; uint64_t* a=(uint64_t*)&ctx->vr[12]; uint64_t* b=(uint64_t*)&ctx->vr[8]; uint64_t* c=(uint64_t*)&ctx->vr[10]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[10]; uint64_t* a = (uint64_t*)&ctx->vr[10]; uint64_t* b = (uint64_t*)&ctx->vr[10]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[19]; uint64_t* a=(uint64_t*)&ctx->vr[19]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[11]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[10], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x4D50);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[9], &ctx->vr[10], &ctx->vr[13], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[12]; uint64_t* c=(uint64_t*)&ctx->vr[21]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[19]; uint64_t* c=(uint64_t*)&ctx->vr[21]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[1])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[11], tmp + 12, 16); }
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[1], &ctx->vr[10], &ctx->vr[13], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[13])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[13], 16); memcpy(&ctx->vr[7], tmp + 12, 16); }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[10], &ctx->vr[13], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[18]; uint8_t* d = (uint8_t*)&ctx->vr[4]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[10]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[18]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[11], &ctx->vr[5], &ctx->vr[7], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[5], &ctx->vr[7], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[9], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x100);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[12], tmp + 8, 16); }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[8], &ctx->vr[4], ctx->vscr);
    vmx_vsubfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[12], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[10], &ctx->vr[13], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[11]; uint64_t* c=(uint64_t*)&ctx->vr[21]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vaddfp(&ctx->vr[1], &ctx->vr[0], &ctx->vr[0], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[12]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[0]; uint8_t* c = (uint8_t*)&ctx->vr[18]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[13], &ctx->vr[2], &ctx->vr[1], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[16]; uint8_t* d = (uint8_t*)&ctx->vr[16]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint8_t* a = (uint8_t*)&ctx->vr[1]; uint8_t* b = (uint8_t*)&ctx->vr[1]; uint8_t* c = (uint8_t*)&ctx->vr[18]; uint8_t* d = (uint8_t*)&ctx->vr[18]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vnmsubfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[9], &ctx->vr[16], ctx->vscr);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[16], &ctx->vr[2], &ctx->vr[12], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[12], &ctx->vr[18], &ctx->vr[11], &ctx->vr[12], ctx->vscr);
    vmx_vnmsubfp(&ctx->vr[18], &ctx->vr[0], &ctx->vr[13], &ctx->vr[18], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[16], &ctx->vr[1], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[18]; uint64_t* b=(uint64_t*)&ctx->vr[12]; uint64_t* c=(uint64_t*)&ctx->vr[6]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[18]; uint64_t* c=(uint64_t*)&ctx->vr[6]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[6]; uint64_t* a=(uint64_t*)&ctx->vr[12]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[6]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[0]; uint64_t* c=(uint64_t*)&ctx->vr[20]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[12]; uint64_t* c=(uint64_t*)&ctx->vr[20]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[6]; uint64_t* a=(uint64_t*)&ctx->vr[6]; uint64_t* b=(uint64_t*)&ctx->vr[18]; uint64_t* c=(uint64_t*)&ctx->vr[20]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[13], 16); memcpy(&ctx->vr[13], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[1], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[6], 16); memcpy(tmp+16, &ctx->vr[6], 16); memcpy(&ctx->vr[6], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[13], 16); memcpy(tmp+16, &ctx->vr[2], 16); memcpy(&ctx->vr[13], tmp + 4, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[1], 16); memcpy(tmp+16, &ctx->vr[2], 16); memcpy(&ctx->vr[1], tmp + 4, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[6], 16); memcpy(tmp+16, &ctx->vr[2], 16); memcpy(&ctx->vr[6], tmp + 4, 16); }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[13], &ctx->vr[2], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xB0);
    vmx_vmaddfp(&ctx->vr[23], &ctx->vr[13], &ctx->vr[2], &ctx->vr[23], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[1], &ctx->vr[2], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xC0);
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[1], &ctx->vr[2], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xD0);
    vmx_vmaddfp(&ctx->vr[24], &ctx->vr[30], &ctx->vr[23], &ctx->vr[24], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[2], &ctx->vr[9], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[26], &ctx->vr[6], &ctx->vr[11], &ctx->vr[26], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x120);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[2], &ctx->vr[9], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[25], &ctx->vr[6], &ctx->vr[12], &ctx->vr[25], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xE0);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[30], &ctx->vr[0], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xF0);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[6], &ctx->vr[1], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    vmx_vmaddfp(&ctx->vr[30], &ctx->vr[30], &ctx->vr[13], &ctx->vr[1], ctx->vscr);
    vmx_vaddfp(&ctx->vr[26], &ctx->vr[26], &ctx->vr[0], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[25], &ctx->vr[24], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[15], &ctx->vr[26], &ctx->vr[2], &ctx->vr[15], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[26], 16); memcpy(tmp+16, &ctx->vr[26], 16); memcpy(&ctx->vr[31], tmp + 0, 16); }
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[6], &ctx->vr[30], ctx->vscr);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    vmx_vaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[15], ctx->vscr);
    { uint64_t ea = (ctx->gpr[6]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[3], 16); memcpy(tmp+16, &ctx->vr[3], 16); memcpy(&ctx->vr[3], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[3], 16); memcpy(tmp+16, &ctx->vr[22], 16); memcpy(&ctx->vr[24], tmp + 4, 16); }
    goto loc_008D1000;
loc_008D16E4:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D7C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    goto loc_008D0D6C;
loc_008D170C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4E08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_008D0D50;
loc_008D1714:
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D0DE4;
    { double a = ctx->fpr[28]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D0DE4;
    { double a = ctx->fpr[30]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D0DE4;
    goto loc_008D1168;
loc_008D1730:
    ctx->lr = 0x008D1734; june_evt_func_00AED020(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ctx->fpr[1];
    goto loc_008D1244;
loc_008D1740:
    ctx->lr = 0x008D1744; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D17E0;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_008D1758:
    ctx->lr = 0x008D175C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D1794;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D6C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4DA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
loc_008D1794:
    ctx->fpr[28] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[30]));
    ctx->fpr[30] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    goto loc_008D0DA0;
loc_008D17A0:
    ctx->lr = 0x008D17A4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D1830;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
loc_008D17B8:
    ctx->lr = 0x008D17BC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D1808;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_008D17D0:
    ctx->fpr[29] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[26]));
    goto loc_008D0DD4;
loc_008D17E0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D70);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[30] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    goto loc_008D1758;
loc_008D1808:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D6C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    goto loc_008D17D0;
loc_008D1830:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4D70);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[26] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    goto loc_008D17B8;
}

void june_evt_func_008D221C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[0] = (int64_t)(int32_t)(0xF0);
    vm_write64(ctx->gpr[1] + -0x1E0, ctx->gpr[1]); ctx->gpr[1] += -0x1E0;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[20]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x100);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[21]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x110);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[22]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x120);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[23]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x130);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[24]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x140);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[25]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x150);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[26]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x160);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[27]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[28]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x180);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[29]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x190);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[30]); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1A0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[31]); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x1D8, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x1C0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x1C8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x1D0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->lr = 0x008D22A8; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008D22B4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D264C;
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4C94);
    ctx->gpr[8] = (int64_t)(int32_t)(0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x190); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)(0x30);
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)-1; ppu_vstu4(&ctx->vr[11],v); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x10);
    vmx_vslw(&ctx->vr[11], &ctx->vr[11], &ctx->vr[11]);
    { uint64_t* d = (uint64_t*)&ctx->vr[10]; uint64_t* a = (uint64_t*)&ctx->vr[10]; uint64_t* b = (uint64_t*)&ctx->vr[10]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[8])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[10])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint32_t* d=(uint32_t*)&ctx->vr[12]; uint32_t* a=(uint32_t*)&ctx->vr[0]; uint32_t* b=(uint32_t*)&ctx->vr[13]; uint32_t t[4]; t[0]=a[0]; t[1]=b[0]; t[2]=a[1]; t[3]=b[1]; memcpy(d, t, 16); }
    { uint64_t* d = (uint64_t*)&ctx->vr[11]; uint64_t* a = (uint64_t*)&ctx->vr[11]; uint64_t* b = (uint64_t*)&ctx->vr[1]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C98);
    { uint32_t* d=(uint32_t*)&ctx->vr[0]; uint32_t* a=(uint32_t*)&ctx->vr[0]; uint32_t* b=(uint32_t*)&ctx->vr[13]; uint32_t t[4]; t[0]=a[2]; t[1]=b[2]; t[2]=a[3]; t[3]=b[3]; memcpy(d, t, 16); }
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[11] = ctx->gpr[31] + (int64_t)(0x10);
    { uint32_t* d=(uint32_t*)&ctx->vr[9]; uint32_t* a=(uint32_t*)&ctx->vr[12]; uint32_t* b=(uint32_t*)&ctx->vr[8]; uint32_t t[4]; t[0]=a[0]; t[1]=b[0]; t[2]=a[1]; t[3]=b[1]; memcpy(d, t, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[6]; uint32_t val = ((uint32_t*)&ctx->vr[11])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[5]; uint32_t val = ((uint32_t*)&ctx->vr[11])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[9] = ctx->gpr[2] + (int64_t)(-19976);
    { uint8_t* a = (uint8_t*)&ctx->vr[0]; uint8_t* b = (uint8_t*)&ctx->vr[8]; uint8_t* c = (uint8_t*)&ctx->vr[13]; uint8_t* d = (uint8_t*)&ctx->vr[0]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[9], &ctx->vr[10], &ctx->vr[6], ctx->vscr);
    { uint8_t* a = (uint8_t*)&ctx->vr[12]; uint8_t* b = (uint8_t*)&ctx->vr[8]; uint8_t* c = (uint8_t*)&ctx->vr[1]; uint8_t* d = (uint8_t*)&ctx->vr[12]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[11])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[9], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[8], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[12], 16); memcpy(tmp+16, &ctx->vr[12], 16); memcpy(&ctx->vr[7], tmp + 12, 16); }
    { uint64_t ea = ctx->gpr[9]; uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[1]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[9], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[7], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[7], tmp + 4, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[8], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[8], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[15]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[6], &ctx->vr[5], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[2]; uint32_t val = ((uint32_t*)&ctx->vr[9])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[16]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[19]; uint32_t val = ((uint32_t*)&ctx->vr[7])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[18]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[17]; uint32_t val = ((uint32_t*)&ctx->vr[8])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[5]; uint32_t val = ((uint32_t*)&ctx->vr[9])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[6], &ctx->vr[10], &ctx->vr[2], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[4]; uint32_t val = ((uint32_t*)&ctx->vr[7])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[6], &ctx->vr[10], &ctx->vr[17], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[3]; uint32_t val = ((uint32_t*)&ctx->vr[8])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[19], &ctx->vr[6], &ctx->vr[10], &ctx->vr[19], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[12], &ctx->vr[11], ctx->vscr);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[15], &ctx->vr[11], &ctx->vr[10], &ctx->vr[15], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[10])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[11], &ctx->vr[10], &ctx->vr[16], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[18], &ctx->vr[11], &ctx->vr[10], &ctx->vr[18], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[13], &ctx->vr[2], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[13], &ctx->vr[17], &ctx->vr[3], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[13], &ctx->vr[19], &ctx->vr[4], ctx->vscr);
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[0], 16); memcpy(&ctx->vr[0], tmp + 12, 16); }
    { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[0], 16); memcpy(tmp+16, &ctx->vr[1], 16); memcpy(&ctx->vr[0], tmp + 4, 16); }
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[12]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[14]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[10], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[8])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[6], &ctx->vr[10], &ctx->vr[12], ctx->vscr);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[1], &ctx->vr[15], &ctx->vr[9], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[1], &ctx->vr[16], &ctx->vr[7], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[1], &ctx->vr[18], &ctx->vr[8], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[11], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[6], &ctx->vr[14], ctx->vscr);
    vmx_vaddfp(&ctx->vr[17], &ctx->vr[9], &ctx->vr[5], ctx->vscr);
    vmx_vaddfp(&ctx->vr[16], &ctx->vr[7], &ctx->vr[4], ctx->vscr);
    vmx_vaddfp(&ctx->vr[15], &ctx->vr[8], &ctx->vr[3], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[13], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008D2688;
loc_008D2424:
    ctx->gpr[9] = ctx->gpr[6] + (int64_t)(0x50);
    { uint32_t* d = (uint32_t*)&ctx->vr[2]; uint32_t val = ((uint32_t*)&ctx->vr[17])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x10);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[17])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[16])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[16])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t* d = (uint64_t*)&ctx->vr[12]; uint64_t* a = (uint64_t*)&ctx->vr[12]; uint64_t* b = (uint64_t*)&ctx->vr[12]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint32_t* d = (uint32_t*)&ctx->vr[3]; uint32_t val = ((uint32_t*)&ctx->vr[17])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint32_t* d = (uint32_t*)&ctx->vr[10]; uint32_t val = ((uint32_t*)&ctx->vr[17])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[4], &ctx->vr[12], &ctx->vr[2], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[4], &ctx->vr[12], &ctx->vr[1], ctx->vscr);
    ctx->gpr[0] = (int64_t)(int32_t)(0x30);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[7], &ctx->vr[12], &ctx->vr[0], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[18]; uint32_t val = ((uint32_t*)&ctx->vr[15])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[7], &ctx->vr[12], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x20);
    { uint32_t* d = (uint32_t*)&ctx->vr[19]; uint32_t val = ((uint32_t*)&ctx->vr[16])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[5]; uint32_t val = ((uint32_t*)&ctx->vr[13])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[11]; uint32_t val = ((uint32_t*)&ctx->vr[16])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[17]; uint32_t val = ((uint32_t*)&ctx->vr[15])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[12], &ctx->vr[8], ctx->vscr);
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    vmx_vmaddfp(&ctx->vr[5], &ctx->vr[7], &ctx->vr[12], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[4], &ctx->vr[4], &ctx->vr[12], &ctx->vr[18], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[18]; uint32_t val = ((uint32_t*)&ctx->vr[15])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[13], &ctx->vr[2], &ctx->vr[10], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[2]; uint32_t val = ((uint32_t*)&ctx->vr[15])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[3], &ctx->vr[6], &ctx->vr[0], &ctx->vr[3], ctx->vscr);
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[13], &ctx->vr[1], &ctx->vr[11], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[19], &ctx->vr[6], &ctx->vr[9], &ctx->vr[19], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[12], &ctx->vr[17], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[29]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[13], &ctx->vr[8], &ctx->vr[1], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[6], &ctx->vr[5], &ctx->vr[9], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[5]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[4], &ctx->vr[2], ctx->vscr);
    vmx_vaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[3], ctx->vscr);
    vmx_vaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[6], &ctx->vr[6], &ctx->vr[7], &ctx->vr[18], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[9], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[10], &ctx->vr[10], &ctx->vr[12], &ctx->vr[8], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[12], &ctx->vr[7], ctx->vscr);
    vmx_vaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[6], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[11], &ctx->vr[5], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[13], &ctx->vr[10], &ctx->vr[0], ctx->vscr);
    vmx_vaddfp(&ctx->vr[1], &ctx->vr[13], &ctx->vr[1], ctx->vscr);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[1])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t _c6 = vmx_vcmpgtfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[0]); ctx->cr = (ctx->cr & ~0xF0u) | (_c6 << 4); }
    if ((((ctx->cr >> 7) & 1) == 1)) goto loc_008D2890;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4DA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(0xAC);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t ea = (ctx->gpr[1] + ctx->gpr[11]); ea &= ~3ULL; { uint32_t _e; memcpy(&_e, (const uint8_t*)&ctx->vr[1] + (ea & 15), 4); VM_RAW_ST(uint32_t, ea, _e); } }
    { uint32_t tmp = vm_read32((ctx->gpr[1] + ctx->gpr[11])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[9]);
    ctx->fpr[8] = ppu_frsp(ctx->fpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = fabs(ctx->fpr[9]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008D2630;
    ctx->gpr[0] = (int64_t)(int32_t)(0x94);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4DA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4E08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[12]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[8], ctx->fpr[12]));
    { uint64_t ea = (ctx->gpr[1] + ctx->gpr[0]); ea &= ~3ULL; { uint32_t _e; memcpy(&_e, (const uint8_t*)&ctx->vr[1] + (ea & 15), 4); VM_RAW_ST(uint32_t, ea, _e); } }
    { uint32_t tmp = vm_read32((ctx->gpr[1] + ctx->gpr[0])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint64_t ea = (ctx->gpr[1] + ctx->gpr[9]); ea &= ~3ULL; { uint32_t _e; memcpy(&_e, (const uint8_t*)&ctx->vr[1] + (ea & 15), 4); VM_RAW_ST(uint32_t, ea, _e); } }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[9]));
    { uint32_t tmp = vm_read32((ctx->gpr[1] + ctx->gpr[9])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[12], 0, 0));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
loc_008D25AC:
    ctx->gpr[0] = (int64_t)(int32_t)(0xF0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x100);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x110);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[22], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x120);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[23], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x130);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[24], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x140);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[25], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x150);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[26], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x160);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[27], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x170);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[28], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x180);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[29], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x190);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[30], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x1A0);
    { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[31], ea); }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1F0);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1E0);
    ctx->lr = ctx->gpr[0];
    return;
loc_008D2630:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4DA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[8], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
    goto loc_008D25AC;
loc_008D264C:
    ctx->gpr[9] = ctx->gpr[31] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x190); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[17], ea); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x20);
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[16], ea); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[15], ea); }
    { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008D2424;
loc_008D2688:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4DE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4C90);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4C8C);
    { uint64_t* d = (uint64_t*)&ctx->vr[10]; uint64_t* a = (uint64_t*)&ctx->vr[10]; uint64_t* b = (uint64_t*)&ctx->vr[10]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x4C88);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)-1; ppu_vstu4(&ctx->vr[11],v); }
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x4C7C);
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)1; ppu_vstu4(&ctx->vr[6],v); }
    { uint32_t* d = (uint32_t*)&ctx->vr[25]; uint32_t val = ((uint32_t*)&ctx->vr[17])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[12], ea); }
    vmx_vslw(&ctx->vr[11], &ctx->vr[11], &ctx->vr[11]);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)2; ppu_vstu4(&ctx->vr[23],v); }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[28], ea); }
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x4C80);
    { uint32_t* d = (uint32_t*)&ctx->vr[27]; uint32_t val = ((uint32_t*)&ctx->vr[16])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4C78);
    { uint32_t* d = (uint32_t*)&ctx->vr[24]; uint32_t val = ((uint32_t*)&ctx->vr[17])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint8_t b = (uint8_t)((ctx->gpr[9]) & 0xF); for (int i = 0; i < 16; i++) ((uint8_t*)&ctx->vr[0])[i] = b + i; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xC0);
    { uint8_t* a = (uint8_t*)&ctx->vr[13]; uint8_t* b = (uint8_t*)&ctx->vr[13]; uint8_t* c = (uint8_t*)&ctx->vr[0]; uint8_t* d = (uint8_t*)&ctx->vr[13]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C64);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[15])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4C60);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4C68);
    { uint32_t* d = (uint32_t*)&ctx->vr[26]; uint32_t val = ((uint32_t*)&ctx->vr[16])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[13]; uint32_t val = ((uint32_t*)&ctx->vr[13])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint32_t* d = (uint32_t*)&ctx->vr[0]; uint32_t val = ((uint32_t*)&ctx->vr[15])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[22], ea); }
    vmx_vmaddfp(&ctx->vr[12], &ctx->vr[13], &ctx->vr[10], &ctx->vr[12], ctx->vscr);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C5C);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[20], ea); }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[21], ea); }
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4C58);
    { uint32_t* d = (uint32_t*)&ctx->vr[31]; uint32_t val = ((uint32_t*)&ctx->vr[17])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C84);
    { uint32_t* d = (uint32_t*)&ctx->vr[30]; uint32_t val = ((uint32_t*)&ctx->vr[16])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[29]; uint32_t val = ((uint32_t*)&ctx->vr[15])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[19]; uint32_t val = ((uint32_t*)&ctx->vr[0])[0]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[14]; uint32_t val = ((uint32_t*)&ctx->vr[0])[1]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[18]; uint32_t val = ((uint32_t*)&ctx->vr[0])[3]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)3; ppu_vstu4(&ctx->vr[0],v); }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[12]; uint64_t* c=(uint64_t*)&ctx->vr[11]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint32_t* d = (uint32_t*)&ctx->vr[17]; uint32_t val = ((uint32_t*)&ctx->vr[17])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[16]; uint32_t val = ((uint32_t*)&ctx->vr[16])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint32_t* d = (uint32_t*)&ctx->vr[15]; uint32_t val = ((uint32_t*)&ctx->vr[15])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    vmx_vaddfp(&ctx->vr[12], &ctx->vr[12], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
    vmx_vctsxs(&ctx->vr[12], &ctx->vr[12], 0, &ctx->vscr);
    vmx_vcfsx(&ctx->vr[1], &ctx->vr[12], 0);
    { uint64_t* d=(uint64_t*)&ctx->vr[12]; uint64_t* a=(uint64_t*)&ctx->vr[12]; uint64_t* b=(uint64_t*)&ctx->vr[0]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint32_t* d = (uint32_t*)&ctx->vr[2]; uint32_t val = ((uint32_t*)&ctx->vr[0])[2]; d[0] = d[1] = d[2] = d[3] = val; }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C74);
    { const uint8_t* a=(const uint8_t*)&ctx->vr[12]; const uint8_t* b=(const uint8_t*)&ctx->vr[6]; uint8_t d[16]; for(int i=0;i<4;i++){ uint32_t x=0,y=0; for(int k=0;k<4;k++){ x=(uint32_t)((x<<8)|a[i*4+k]); y=(uint32_t)((y<<8)|b[i*4+k]); } uint32_t r=(uint32_t)(x+y); for(int k=0;k<4;k++) d[i*4+k]=(uint8_t)(r>>(8*(4-1-k))); } memcpy(&ctx->vr[5], d, 16); }
    { uint64_t* d=(uint64_t*)&ctx->vr[7]; uint64_t* a=(uint64_t*)&ctx->vr[12]; uint64_t* b=(uint64_t*)&ctx->vr[6]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[12]; uint64_t* a=(uint64_t*)&ctx->vr[12]; uint64_t* b=(uint64_t*)&ctx->vr[23]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[6]; uint64_t* a=(uint64_t*)&ctx->vr[5]; uint64_t* b=(uint64_t*)&ctx->vr[6]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    (void)vmx_vcmpequw(&ctx->vr[7], &ctx->vr[7], &ctx->vr[10]);
    vmx_vnmsubfp(&ctx->vr[28], &ctx->vr[1], &ctx->vr[13], &ctx->vr[28], ctx->vscr);
    { uint64_t ea = (ctx->gpr[7]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    { uint64_t* d=(uint64_t*)&ctx->vr[5]; uint64_t* a=(uint64_t*)&ctx->vr[5]; uint64_t* b=(uint64_t*)&ctx->vr[23]; for(int i=0;i<2;i++) d[i]=a[i]&b[i]; }
    (void)vmx_vcmpequw(&ctx->vr[6], &ctx->vr[6], &ctx->vr[10]);
    (void)vmx_vcmpequw(&ctx->vr[12], &ctx->vr[12], &ctx->vr[10]);
    (void)vmx_vcmpequw(&ctx->vr[5], &ctx->vr[5], &ctx->vr[10]);
    vmx_vnmsubfp(&ctx->vr[1], &ctx->vr[1], &ctx->vr[28], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[1], &ctx->vr[10], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[0], &ctx->vr[13], &ctx->vr[9], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C70);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[0], &ctx->vr[13], &ctx->vr[8], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[13], &ctx->vr[0], &ctx->vr[10], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[28], ea); }
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x4C6C);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[28], &ctx->vr[0], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[28], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC0);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[28], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[9], &ctx->vr[9], &ctx->vr[20], &ctx->vr[0], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[8], &ctx->vr[8], &ctx->vr[1], &ctx->vr[13], ctx->vscr);
    { uint64_t* d=(uint64_t*)&ctx->vr[7]; uint64_t* a=(uint64_t*)&ctx->vr[9]; uint64_t* b=(uint64_t*)&ctx->vr[8]; uint64_t* c=(uint64_t*)&ctx->vr[7]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[9]; uint64_t* a=(uint64_t*)&ctx->vr[9]; uint64_t* b=(uint64_t*)&ctx->vr[8]; uint64_t* c=(uint64_t*)&ctx->vr[6]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[1]; uint64_t* a = (uint64_t*)&ctx->vr[11]; uint64_t* b = (uint64_t*)&ctx->vr[7]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d = (uint64_t*)&ctx->vr[13]; uint64_t* a = (uint64_t*)&ctx->vr[11]; uint64_t* b = (uint64_t*)&ctx->vr[9]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[1]; uint64_t* a=(uint64_t*)&ctx->vr[1]; uint64_t* b=(uint64_t*)&ctx->vr[7]; uint64_t* c=(uint64_t*)&ctx->vr[12]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[13]; uint64_t* a=(uint64_t*)&ctx->vr[13]; uint64_t* b=(uint64_t*)&ctx->vr[9]; uint64_t* c=(uint64_t*)&ctx->vr[5]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d = (uint64_t*)&ctx->vr[11]; uint64_t* a = (uint64_t*)&ctx->vr[11]; uint64_t* b = (uint64_t*)&ctx->vr[1]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[13]; uint64_t* c=(uint64_t*)&ctx->vr[22]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[10]; uint64_t* b=(uint64_t*)&ctx->vr[11]; uint64_t* c=(uint64_t*)&ctx->vr[22]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t* d=(uint64_t*)&ctx->vr[0]; uint64_t* a=(uint64_t*)&ctx->vr[0]; uint64_t* b=(uint64_t*)&ctx->vr[1]; uint64_t* c=(uint64_t*)&ctx->vr[21]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint64_t* d=(uint64_t*)&ctx->vr[11]; uint64_t* a=(uint64_t*)&ctx->vr[11]; uint64_t* b=(uint64_t*)&ctx->vr[13]; uint64_t* c=(uint64_t*)&ctx->vr[21]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
    vmx_vmaddfp(&ctx->vr[19], &ctx->vr[0], &ctx->vr[10], &ctx->vr[19], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[25], &ctx->vr[0], &ctx->vr[10], &ctx->vr[25], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[27], &ctx->vr[0], &ctx->vr[10], &ctx->vr[27], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[0], &ctx->vr[0], &ctx->vr[10], &ctx->vr[1], ctx->vscr);
    { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    vmx_vmaddfp(&ctx->vr[14], &ctx->vr[11], &ctx->vr[10], &ctx->vr[14], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[24], &ctx->vr[11], &ctx->vr[10], &ctx->vr[24], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[26], &ctx->vr[11], &ctx->vr[10], &ctx->vr[26], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[11], &ctx->vr[11], &ctx->vr[10], &ctx->vr[1], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[19], &ctx->vr[4], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[17], &ctx->vr[17], &ctx->vr[25], &ctx->vr[4], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[16], &ctx->vr[16], &ctx->vr[27], &ctx->vr[4], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[15], &ctx->vr[15], &ctx->vr[0], &ctx->vr[4], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[18], &ctx->vr[18], &ctx->vr[14], &ctx->vr[3], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[31], &ctx->vr[31], &ctx->vr[24], &ctx->vr[3], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[30], &ctx->vr[30], &ctx->vr[26], &ctx->vr[3], ctx->vscr);
    vmx_vmaddfp(&ctx->vr[29], &ctx->vr[29], &ctx->vr[11], &ctx->vr[3], ctx->vscr);
    vmx_vaddfp(&ctx->vr[0], &ctx->vr[2], &ctx->vr[18], ctx->vscr);
    vmx_vaddfp(&ctx->vr[17], &ctx->vr[17], &ctx->vr[31], ctx->vscr);
    vmx_vaddfp(&ctx->vr[16], &ctx->vr[16], &ctx->vr[30], ctx->vscr);
    vmx_vaddfp(&ctx->vr[15], &ctx->vr[15], &ctx->vr[29], ctx->vscr);
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    goto loc_008D2424;
loc_008D2890:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4DA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
    goto loc_008D25AC;
}
