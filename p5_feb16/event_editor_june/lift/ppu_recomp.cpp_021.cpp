
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

void june_evt_func_0085EF60(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[24]);
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->gpr[23] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[29]);
    ctx->gpr[25] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[31]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->lr = 0x0085EFF0; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { double a = ctx->fpr[24]; double b = ctx->fpr[25]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085F3A4;
    ctx->fpr[24] = ctx->fpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
loc_0085F030:
    { double a = ctx->fpr[27]; double b = ctx->fpr[25]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085F39C;
    ctx->fpr[27] = ctx->fpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
loc_0085F040:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x1C);
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[4], 0, 32);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085F06C;
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[29], 0, 0));
loc_0085F06C:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[26]));
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x65C4);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->fpr[22] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[28]));
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[21], 0, 32);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[23] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[28]));
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[28]));
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[28]));
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F0DC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x10);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F108; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x20);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F134; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x30);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F160; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x40);
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F198; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x50);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F1C4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x60);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F1FC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x70);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F228; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[24]; double b = ctx->fpr[25]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085F240;
    { double a = ctx->fpr[27]; double b = ctx->fpr[25]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085F3B4;
    ;
loc_0085F240:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F350;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F41C;
loc_0085F254:
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085F2E4;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x18);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085F3AC;
loc_0085F2A4:
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x20);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[21] | ctx->gpr[21];
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[24], ctx->fpr[12]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[12], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085F2E0; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085F2E4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_0085F350:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x10);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[11] = vm_read32(ctx->gpr[25] + 0x14);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[11]);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[29] = ppu_frsp(ctx->fpr[0]);
    goto loc_0085F254;
loc_0085F39C:
    ctx->fpr[21] = ppu_fp_single(ppu_fsub(ctx->fpr[27], ctx->fpr[25]));
    goto loc_0085F040;
loc_0085F3A4:
    ctx->fpr[20] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[25]));
    goto loc_0085F030;
loc_0085F3AC:
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    goto loc_0085F2A4;
loc_0085F3B4:
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F42C;
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[7] = vm_read32(ctx->gpr[25] + 0x34);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085F414; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    goto loc_0085F240;
loc_0085F41C:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    goto loc_0085F254;
loc_0085F42C:
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x80);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085F460; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085F240;
}

void june_evt_func_0085F468(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->lr = 0x0085F484; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085F490; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_0085F638(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[26]);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[31]);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[26] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[29]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F6C8;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 29, 1, 31, 0, 0);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
loc_0085F6C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F7E0;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FC00;
loc_0085F6DC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F784;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[11]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[0], ctx->fpr[13], 0, 0));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[26] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[23]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[29] = ppu_frsp(ctx->fpr[13]);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085F848;
    { double a = ctx->fpr[29]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085F804;
loc_0085F748:
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[26]));
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0085F774; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085F858;
loc_0085F784:
    ctx->gpr[30] = (int64_t)(int32_t)(0);
loc_0085F788:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[30];
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_0085F7E0:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[28] = ppu_frsp(ctx->fpr[0]);
    goto loc_0085F6DC;
loc_0085F804:
    ctx->fpr[29] = ctx->fpr[11];
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[26]));
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0085F834; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F784;
    goto loc_0085F858;
loc_0085F848:
    { double a = ctx->fpr[29]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ctx->fpr[11];
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085F748;
    goto loc_0085F804;
loc_0085F858:
    ctx->lr = 0x0085F85C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F888;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65AC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085F888:
    ctx->lr = 0x0085F88C; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085F898; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085F8D4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F900;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65A8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[25] = ppu_frsp(ctx->fpr[0]);
loc_0085F900:
    ctx->lr = 0x0085F904; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085F910; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085F938;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FC20;
loc_0085F938:
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
loc_0085F940:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FD3C;
loc_0085F94C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FBA4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FD10;
loc_0085F960:
    { double a = ctx->fpr[28]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FA00;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[13]);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { double a = ctx->fpr[29]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085FC18;
loc_0085F9B0:
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[27], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[31]));
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[12]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[23], ctx->fpr[12], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0085F9EC; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FC10;
    ;
loc_0085FA00:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0085FAA0;
loc_0085FA0C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FAC8;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FA38;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[11]);
    ;
loc_0085FA38:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FB04;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FDE4;
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FAE8;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FDEC;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FE7C;
loc_0085FA78:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FB04;
loc_0085FA8C:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_0085FB18;
loc_0085FAA0:
    ctx->lr = 0x0085FAA4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FAC8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65A0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FA0C;
loc_0085FAC8:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FA38;
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    goto loc_0085FA38;
loc_0085FAE8:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FEEC;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x149);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65B0);
    ctx->lr = 0x0085FB00; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085FB04:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085F788;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
loc_0085FB18:
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FB48;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FDB8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FF1C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x162);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65B0);
    ctx->lr = 0x0085FB44; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085FB48:
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F788;
    ctx->lr = 0x0085FB54; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F788;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[26], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(9);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x0085FB94; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0085FB9C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085F788;
loc_0085FBA4:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085FCF8;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085FCB0;
loc_0085FBC0:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[11], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[28] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[27] = ppu_frsp(ctx->fpr[12]);
    goto loc_0085F960;
loc_0085FC00:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    goto loc_0085F6DC;
loc_0085FC10:
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    goto loc_0085FA00;
loc_0085FC18:
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    goto loc_0085F9B0;
loc_0085FC20:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085F938;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085F938;
    ctx->lr = 0x0085FC78; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FF90;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65A4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FF90;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    goto loc_0085F940;
loc_0085FCB0:
    ctx->gpr[11] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[10];
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[9]);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[11], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[28] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[27] = ppu_frsp(ctx->fpr[12]);
    goto loc_0085F960;
loc_0085FCF8:
    ctx->gpr[11] = ctx->gpr[10] + (int64_t)(-1);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[11]);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085FBC0;
    goto loc_0085FCB0;
loc_0085FD10:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085FE6C;
loc_0085FD20:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085FE60;
loc_0085FD2C:
    ctx->fpr[28] = ctx->fpr[10];
    ctx->fpr[31] = ctx->fpr[11];
    ctx->fpr[27] = ctx->fpr[0];
    goto loc_0085F960;
loc_0085FD3C:
    ctx->lr = 0x0085FD40; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FD9C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65A0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FD9C;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[27]));
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FF38;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FFEC;
loc_0085FD80:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = (int64_t)(int32_t)(2);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[11]);
    goto loc_0085F94C;
loc_0085FD9C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[11]);
    goto loc_0085F94C;
loc_0085FDB8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FE28;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FEB4;
loc_0085FDD4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0085FB48;
loc_0085FDE4:
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[28];
    goto loc_0085FB18;
loc_0085FDEC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FE98;
loc_0085FDF8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6590); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x90);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FA8C;
    goto loc_0085FB04;
loc_0085FE28:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FED0;
loc_0085FE34:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6590); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    goto loc_0085FB48;
loc_0085FE60:
    ctx->fpr[0] = ctx->fpr[11];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085FD2C;
loc_0085FE6C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x659C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085FD20;
loc_0085FE7C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x12E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6598);
    ctx->lr = 0x0085FE8C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_0085FA78;
loc_0085FE98:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x136);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6594);
    ctx->lr = 0x0085FEA8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_0085FDF8;
loc_0085FEB4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x156);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6598);
    ctx->lr = 0x0085FEC4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_0085FDD4;
loc_0085FED0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x159);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6594);
    ctx->lr = 0x0085FEE0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_0085FE34;
loc_0085FEEC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FFB4;
loc_0085FEF8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085FB04;
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[29]);
    goto loc_0085FB18;
loc_0085FF1C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FFD0;
loc_0085FF28:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    goto loc_0085FB48;
loc_0085FF38:
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FF9C;
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { double a = ctx->fpr[31]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085FFA0;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[28], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    ctx->gpr[11] = ctx->gpr[9] + ctx->gpr[0];
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[11]);
loc_0085FF7C:
    { double a = ctx->fpr[11]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085FD80;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0xC);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[11]);
    goto loc_0085FD80;
loc_0085FF90:
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0085F940;
loc_0085FF9C:
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
loc_0085FFA0:
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FF7C;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x10);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[11]);
    goto loc_0085FF7C;
loc_0085FFB4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x140);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6594);
    ctx->lr = 0x0085FFC4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_0085FEF8;
loc_0085FFD0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x15E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6594);
    ctx->lr = 0x0085FFE0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_0085FF28;
loc_0085FFEC:
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860034;
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { double a = ctx->fpr[31]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860038;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[28], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
loc_00860020:
    { double a = ctx->fpr[31]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085FD80;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085FD80;
loc_00860034:
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
loc_00860038:
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00860020;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_00860020;
}

void june_evt_func_00860050(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[4]);
    return;
}

void june_evt_func_00860058(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x2C);
    return;
}

void june_evt_func_00861F88(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[9] = (int64_t)(int32_t)(-1);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862080;
    ctx->lr = 0x0086201C; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x651C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[3] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6514); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x84);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0086207C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862080:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086216C;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6510); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x650C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x84);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6508); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_008620B4;
loc_008620A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862130;
loc_008620B4:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008620A8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[30]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x008620F8; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6504);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00862120; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008620B4;
loc_00862130:
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086216C;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x74);
    ;
loc_00862148:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x0086215C; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862148;
loc_0086216C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
}

void june_evt_func_008621A4(ppu_context* ctx) {
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
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0x100, ctx->gpr[1]); ctx->gpr[1] += -0x100;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[23] = ctx->gpr[25] | ctx->gpr[25];
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    vm_write32(ctx->gpr[1] + 0x108, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862294;
loc_00862240:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x108);
    ctx->gpr[23] = _cs_23;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[24] = _cs_24;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
    return;
loc_00862294:
    ctx->gpr[3] = vm_read64(ctx->gpr[4] + 0x18);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862778;
loc_008622A0:
    ctx->lr = 0x008622A4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625F4;
loc_008622B8:
    ctx->lr = 0x008622BC; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008622C8; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00862304; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625D4;
loc_00862314:
    ctx->lr = 0x00862318; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00862324; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00862A70;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6508); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    goto loc_008623C4;
loc_00862374:
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008624E4;
    ;
loc_008623B8:
    ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008624E4;
loc_008623C4:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008623B8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862374;
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862430;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862430;
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862430;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008624E4;
    ;
loc_00862430:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    ctx->gpr[30] = vm_read32(ctx->gpr[29] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862980;
loc_00862444:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[11];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[0];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008623B8;
loc_008624E4:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862540;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862514;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086271C;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00862510; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862514:
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00862538;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862614;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00862534; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862538:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[31]);
loc_00862540:
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086259C;
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x70);
loc_00862558:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x0086256C; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862580;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
loc_00862580:
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862558;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0086259C;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_0086259C:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008625CC;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008625C0;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625CC;
loc_008625C0:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008627F8;
loc_008625CC:
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[24];
    goto loc_00862240;
loc_008625D4:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64F8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00862314;
loc_008625F4:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64FC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_008622B8;
loc_00862614:
    ctx->gpr[30] = vm_read32(ctx->gpr[29] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862A08;
loc_00862620:
    ctx->gpr[3] = vm_read64(ctx->gpr[27] + 0x18);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086299C;
loc_00862630:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->lr = 0x00862638; june_evt_func_0086DDD0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862658;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x15E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64EC);
    ctx->lr = 0x00862654; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862658:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64E8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(4);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64E4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    ctx->fpr[0] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->lr = 0x008626C0; june_evt_func_0086DE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x88);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[9] + 0x88, ctx->gpr[0]);
    ctx->lr = 0x008626DC; june_evt_func_00870C34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x98);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008626F0; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862A20;
loc_00862700:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00862710; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[31]);
    goto loc_00862540;
loc_0086271C:
    ctx->gpr[30] = vm_read32(ctx->gpr[3] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862A58;
loc_00862728:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->lr = 0x00862734; june_evt_func_0086DDD0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862758;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[0] = vm_read64(ctx->gpr[9] + 0x90);
    vm_write64(ctx->gpr[27] + 0x18, ctx->gpr[0]);
    ctx->lr = 0x00862750; june_evt_func_00834970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x20, tmp); }
loc_00862758:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862A3C;
loc_00862764:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00862770; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00862514;
loc_00862778:
    ctx->lr = 0x0086277C; june_evt_func_0086E280(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008622A0;
    ctx->lr = 0x00862790; june_evt_func_00834970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6500); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008622A0;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008627BC; june_evt_func_0086DE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008627C8; june_evt_func_0086DDD0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008627EC;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008627DC; june_evt_func_0086F880(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008627E8; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008627EC:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[27] + 0x18, ctx->gpr[0]);
    goto loc_008622A0;
loc_008627F8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->lr = 0x00862808; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008625CC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64E0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008625CC;
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625CC;
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[28]));
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625CC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625CC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008625CC;
    ctx->gpr[10] = (int64_t)(int32_t)((uint32_t)0x80 << 16);
    ctx->gpr[30] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[10] = ctx->gpr[10] | 0x1;
    goto loc_00862890;
loc_00862880:
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008628A0;
    ctx->gpr[30] = ctx->gpr[9] | ctx->gpr[9];
loc_00862890:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x7C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862880;
loc_008628A0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x008628AC; june_evt_func_0086FE44(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008628D0;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x18D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64DC);
    ctx->lr = 0x008628CC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008628D0:
    ctx->lr = 0x008628D4; june_evt_func_0086E0AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008628F4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x400);
    ctx->lr = 0x008628F0; june_evt_func_0086DE20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008628F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008625CC;
    ctx->lr = 0x00862904; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008625CC;
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862930;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x8);
    ctx->gpr[28] = (int64_t)(int8_t)ctx->gpr[0];
loc_00862930:
    ctx->gpr[29] = vm_read32(ctx->gpr[27] + 0x24);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086293C; june_evt_func_0086BEF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086296C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00862974; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[24];
    goto loc_00862240;
loc_00862980:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x124);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64F4);
    ctx->lr = 0x00862990; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    goto loc_00862444;
loc_0086299C:
    ctx->lr = 0x008629A0; june_evt_func_0086E280(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862630;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008629D0;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008629CC; june_evt_func_0086DE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008629D0:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008629D8; june_evt_func_0086DDD0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008629FC;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008629EC; june_evt_func_0086F880(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008629F8; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008629FC:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[27] + 0x18, ctx->gpr[0]);
    goto loc_00862630;
loc_00862A08:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x14E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64F4);
    ctx->lr = 0x00862A18; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00862620;
loc_00862A20:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x167);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64F0);
    ctx->lr = 0x00862A30; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_00862700;
loc_00862A3C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x142);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64F0);
    ctx->lr = 0x00862A4C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x0);
    goto loc_00862764;
loc_00862A58:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0x13A);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64F4);
    ctx->lr = 0x00862A68; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00862728;
loc_00862A70:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    goto loc_008624E4;
}

void june_evt_func_00862A78(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862AC8;
    ;
loc_00862AA8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->lr = 0x00862AB8; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862AA8;
loc_00862AC8:
    ctx->gpr[3] = vm_read64(ctx->gpr[31] + 0x18);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862B20;
loc_00862AD4:
    ctx->gpr[3] = vm_read64(ctx->gpr[31] + 0x10);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x00862AE0; june_evt_func_0086E280(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862AF8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00862AF4; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862AF8:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00862B00; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00862B20:
    ctx->lr = 0x00862B24; june_evt_func_0086E280(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862AD4;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862B40; june_evt_func_0086DE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862B4C; june_evt_func_0086DDD0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862B70;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862B60; june_evt_func_0086F880(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862B6C; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862B70:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    goto loc_00862AD4;
}

void june_evt_func_008632E8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x24);
    return;
}

void june_evt_func_008632F0(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863338;
    ;
loc_00863318:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->lr = 0x00863328; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863318;
loc_00863338:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x00863340; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_00864058(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x24, ctx->gpr[4]);
    return;
}

void june_evt_func_00864060(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x24);
    return;
}

void june_evt_func_00864664(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1A0, ctx->gpr[1]); ctx->gpr[1] += -0x1A0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[21]);
    ctx->gpr[18] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    ctx->gpr[21] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x190, tmp); }
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    ctx->lr = 0x008646E4; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[18] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[18] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00864B10;
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00864B10;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = ctx->gpr[3] + (int64_t)(4);
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[15] | ctx->gpr[15];
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00864758; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864B7C;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008647EC;
loc_00864794:
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x008647C8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864E0C;
loc_008647D8:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->lr = 0x008647E0; june_evt_func_008555F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
loc_008647EC:
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[1]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[5] = ctx->gpr[15] | ctx->gpr[15];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00864820; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864AA0;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864E38;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00864858; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086486C; june_evt_func_00855600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
loc_00864874:
    ctx->gpr[10] = vm_read32(ctx->gpr[27] + 0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864AA0;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6480); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[16] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = ctx->gpr[17] + (int64_t)(0xC);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[14] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[25] = ctx->gpr[20] | ctx->gpr[20];
    vm_write32(ctx->gpr[1] + 0xD4, ctx->gpr[0]);
    ;
loc_008648B0:
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864D04;
loc_008648C0:
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x008648D0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086494C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[16]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0xC, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x10, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864930;
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0xC, tmp); }
loc_00864930:
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[20] | ctx->gpr[20];
    ctx->lr = 0x0086493C; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
loc_0086494C:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864A80;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[13] = ctx->fpr[28];
    ctx->fpr[0] = ctx->fpr[29];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864A78;
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    goto loc_00864994;
loc_0086497C:
    ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864A78;
loc_00864988:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_00864994:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[30];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x008649B4; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x008649CC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086497C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x10, tmp); }
    ctx->lr = 0x00864A28; june_evt_func_00855600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 0, 31, 31, 0, 0);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864A5C;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0xC, tmp); }
loc_00864A5C:
    ctx->lr = 0x00864A60; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0x24);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0xC, tmp); }
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x10, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864988;
loc_00864A78:
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_00864A80:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008648B0;
loc_00864AA0:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864BE0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864B10;
loc_00864ABC:
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xD1);
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864DE4;
loc_00864ADC:
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xD0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864B10;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xB0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x00864B0C; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00864B10:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->gpr[14] = _cs_14;
    ctx->gpr[15] = _cs_15;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x190); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1A0);
    return;
loc_00864B7C:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write8(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00864DB0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00864D90;
loc_00864BA0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6484); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write8(ctx->gpr[1] + 0xD1, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864794;
    goto loc_008647EC;
loc_00864BE0:
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[17] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    { uint32_t tmp = vm_read32(ctx->gpr[18] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[18] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x647C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00864C50; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    ctx->lr = 0x00864C84; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    ctx->lr = 0x00864CB8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    ctx->lr = 0x00864CEC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864ABC;
    goto loc_00864B10;
loc_00864D04:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864D78;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
loc_00864D2C:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[3] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x00864D4C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008648C0;
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0xD4);
    ctx->gpr[5] = ctx->gpr[14] | ctx->gpr[14];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->lr = 0x00864D70; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008648C0;
loc_00864D78:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    goto loc_00864D2C;
loc_00864D90:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    vm_write8(ctx->gpr[1] + 0xD1, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864794;
    goto loc_008647EC;
loc_00864DB0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6484); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    vm_write8(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00864BA0;
    goto loc_00864D90;
loc_00864DE4:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0xC);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xB0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x00864E04; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00864ADC;
loc_00864E0C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00864E30; june_evt_func_008556D0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008647D8;
loc_00864E38:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    goto loc_00864874;
}

void june_evt_func_00864E48(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00864E70; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x64A0);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x2D);
    ctx->gpr[3] = (int64_t)(int32_t)(0x28);
    ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00864E94; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x2E);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864FC4;
loc_00864EB0:
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    ctx->lr = 0x00864EBC; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00864F88;
loc_00864EDC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864F60;
    ctx->lr = 0x00864F00; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x646C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6470); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(2);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x648C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[11]);
    ctx->lr = 0x00864F40; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[3]);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00864F58; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
loc_00864F60:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00864F88:
    ctx->lr = 0x00864F8C; june_evt_func_008558BC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    ctx->lr = 0x00864FA0; june_evt_func_008555B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x36);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00864EDC;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6474);
    ctx->lr = 0x00864FBC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00864EDC;
loc_00864FC4:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6478);
    ctx->lr = 0x00864FCC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00864EB0;
}

void june_evt_func_00865054(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1A0, ctx->gpr[1]); ctx->gpr[1] += -0x1A0;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[31]);
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[20] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x190, tmp); }
    ctx->gpr[14] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    ctx->lr = 0x008650D8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[1] + 0xCC, ctx->gpr[20]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865110;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6468);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00865110:
    ctx->lr = 0x00865114; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00865120; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0086515C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865188;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6464);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00865188:
    ctx->lr = 0x0086518C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086519C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    vm_write32(ctx->gpr[1] + 0xA4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->fpr[26] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086520C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 28, 1, 31, 0, 0);
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[0]);
loc_0086520C:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865244;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00865228; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->fpr[31] = ctx->fpr[1];
    ctx->lr = 0x00865238; june_evt_func_008555F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->fpr[30] = ctx->fpr[1];
loc_00865244:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865330;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->fpr[9] = ctx->fpr[31];
    ctx->fpr[11] = ctx->fpr[30];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865298;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865D58;
    ;
loc_00865278:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865278;
    ctx->fpr[9] = ctx->fpr[31];
loc_00865298:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865994;
loc_008652A4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x10, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x14, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008659AC;
loc_008652BC:
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008659AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00865C80;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[12]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write8(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00865CB8;
loc_008652E8:
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0xC1, ctx->gpr[11]);
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0xC);
loc_008652F4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x647C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x648C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    vm_write32(ctx->gpr[11] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x14, tmp); }
loc_00865330:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[17] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0x70);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086538C;
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[4] = ctx->gpr[17] | ctx->gpr[17];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0086537C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865A90;
loc_0086538C:
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ctx->gpr[17] | ctx->gpr[17];
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[5] = ctx->gpr[15] | ctx->gpr[15];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x008653C8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865718;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865AD0;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00865400; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00865414; june_evt_func_00855600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00865418:
    ctx->gpr[25] = vm_read32(ctx->gpr[27] + 0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865718;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x90);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[16] = ppc_rlw64(ctx->gpr[3], 0, 31, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0xB0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[18] = ppc_rldicl(ctx->gpr[23], 0, 32);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865A7C;
loc_00865464:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
loc_00865470:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0086548C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008654E8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008654E8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008654E8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008654E8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865CE8;
    ;
loc_008654E8:
    ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x008654F8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865584;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[16]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0xC, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x10, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865558;
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0xC, tmp); }
loc_00865558:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[5] = ctx->gpr[18] | ctx->gpr[18];
    ctx->lr = 0x00865564; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865578;
    ctx->gpr[19] = (int64_t)(int32_t)(1);
loc_00865578:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
loc_00865584:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865A3C;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[13] = ctx->fpr[28];
    ctx->fpr[0] = ctx->fpr[29];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865A34;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    goto loc_008656B8;
loc_008655B4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865A2C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865A2C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865600;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865600;
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[25]);
loc_00865600:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    ctx->lr = 0x00865648; june_evt_func_00855600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865670;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
loc_00865670:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->lr = 0x00865680; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865694;
    ctx->gpr[19] = (int64_t)(int32_t)(1);
loc_00865694:
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
loc_0086569C:
    ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865A34;
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_008656B8:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->lr = 0x008656D4; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x008656EC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086569C;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008655B4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_00865600;
loc_00865718:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
loc_00865720:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865774;
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xC1);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008659F8;
loc_00865750:
    ctx->gpr[31] = (int64_t)(int32_t)(0);
loc_00865754:
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xC0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008659BC;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[31];
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00865798;
loc_00865774:
    ctx->gpr[4] = ctx->gpr[17] | ctx->gpr[17];
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0xCC);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = 0x00865784; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086580C;
loc_00865798:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[11];
    ctx->gpr[14] = _cs_14;
    ctx->gpr[15] = _cs_15;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x170); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x190); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1A0);
    return;
loc_0086580C:
    ctx->lr = 0x00865810; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865C60;
loc_00865824:
    ctx->lr = 0x00865828; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00865834; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00865870; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865C1C;
    ctx->lr = 0x00865884; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00865890; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865AE0;
loc_008658A4:
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x18);
loc_008658A8:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865798;
    ctx->lr = 0x008658B8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008658E0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6460);
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865798;
loc_008658E0:
    ctx->lr = 0x008658E4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086590C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x645C);
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865798;
loc_0086590C:
    ctx->lr = 0x00865910; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865938;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x645C);
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865798;
loc_00865938:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6458);
    ctx->gpr[11] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865CF0;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865CF0;
loc_0086595C:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0xC8);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00865970; june_evt_func_008644AC(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[14]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086598C;
    ctx->lr = 0x0086597C; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865D14;
loc_0086598C:
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    goto loc_00865798;
loc_00865994:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008652A4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6488); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008652BC;
loc_008659AC:
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0xC1, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0xC0, ctx->gpr[11]);
    goto loc_00865330;
loc_008659BC:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->gpr[5] = ctx->gpr[17] | ctx->gpr[17];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->lr = 0x008659D4; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086598C;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[31];
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00865774;
    goto loc_00865798;
loc_008659F8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0xC);
    ctx->gpr[5] = ctx->gpr[17] | ctx->gpr[17];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->lr = 0x00865A10; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865750;
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    goto loc_00865754;
loc_00865A2C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_00865600;
loc_00865A34:
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_00865A3C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[25] = vm_read32(ctx->gpr[26] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = ctx->fpr[0];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865720;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865464;
loc_00865A7C:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    goto loc_00865470;
loc_00865A90:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[14], 0, 32);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00865AB8; june_evt_func_00855C2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086538C;
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    goto loc_0086538C;
loc_00865AD0:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    goto loc_00865418;
loc_00865AE0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008658A4;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008658A4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008658A4;
    ctx->lr = 0x00865B4C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865B80;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6460);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865B80;
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xC4);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865D90;
loc_00865B80:
    ctx->lr = 0x00865B84; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865BA8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x645C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865BD0;
loc_00865BA8:
    ctx->lr = 0x00865BAC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008658A4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x645C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008658A4;
loc_00865BD0:
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xC4);
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0xC8);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008658A8;
    ctx->gpr[4] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = ctx->gpr[0] | ctx->gpr[0];
    ctx->lr = 0x00865BF4; june_evt_func_008644AC(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[14]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865C10;
    ctx->lr = 0x00865C00; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865DFC;
loc_00865C10:
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x18);
    goto loc_008658A8;
loc_00865C1C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6464);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x00865C3C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00865C48; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008658A4;
    goto loc_00865AE0;
loc_00865C60:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6468);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00865824;
loc_00865C80:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6484); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write8(ctx->gpr[1] + 0xC0, ctx->gpr[10]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { double a = ctx->fpr[12]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008652E8;
loc_00865CB8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6484); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0xC);
    vm_write8(ctx->gpr[1] + 0xC1, ctx->gpr[10]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    goto loc_008652F4;
loc_00865CE8:
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[25]);
    goto loc_008654E8;
loc_00865CF0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 26, 26, 0, 0);
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865798;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086595C;
    goto loc_00865798;
loc_00865D14:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[14], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x24);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(6);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00865D44; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00865D4C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    goto loc_00865798;
loc_00865D58:
    ctx->fpr[9] = ctx->fpr[31];
    ctx->fpr[11] = ctx->fpr[30];
loc_00865D60:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00865D78;
    ctx->fpr[9] = ctx->fpr[0];
loc_00865D78:
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x28);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00865D60;
    goto loc_00865298;
loc_00865D90:
    { int64_t a = (int32_t)ctx->gpr[14]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008658A8;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[19];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00865798;
    ctx->lr = 0x00865DA8; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008658A4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[14], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x24);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(6);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00865DE8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00865DF0; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x18);
    goto loc_008658A8;
loc_00865DFC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[14], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x24);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(6);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00865E30; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00865E38; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x18);
    goto loc_008658A8;
}

void june_evt_func_008660F8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->lr = 0x00866130; june_evt_func_008644AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00866140; june_evt_func_008644AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[30] = vm_read32(ctx->gpr[27] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866280;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866244;
    ;
loc_00866168:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866264;
loc_00866174:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866274;
loc_00866188:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    vm_write32(ctx->gpr[9] + 0x2C, ctx->gpr[0]);
loc_00866194:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008661AC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008661A8; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008661AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008661D8;
loc_008661B8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->lr = 0x008661C8; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008661B8;
loc_008661D8:
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008661E4; june_evt_func_00878FA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866214;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00866210; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_00866214:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086621C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866280;
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866168;
loc_00866244:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00866254; june_evt_func_008644AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866174;
loc_00866264:
    vm_write32(ctx->gpr[27] + 0x20, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866188;
loc_00866274:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    vm_write32(ctx->gpr[27] + 0x1C, ctx->gpr[0]);
    goto loc_00866194;
loc_00866280:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866298;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00866294; june_evt_func_00855824(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00866298:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008662B0;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008662AC; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008662B0:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008662C8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008662C4; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008662C8:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x008662D0; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_008681FC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008683A4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0086822C; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00868354;
loc_00868248:
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00868380;
loc_00868260:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00868298;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00868274; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00868388;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x8);
    ;
loc_00868298:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008682C4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->lr = 0x008682B4; june_evt_func_00882C2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
loc_008682C4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086834C;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[0]);
loc_008682E8:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[0]);
loc_008682F0:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00868310;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0086830C; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00868310:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00868328;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00868324; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00868328:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00868330; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_0086834C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    goto loc_008682E8;
loc_00868354:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[0]);
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00868260;
loc_00868380:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    goto loc_008682F0;
loc_00868388:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x8);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[0]);
    goto loc_00868298;
loc_008683A4:
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    goto loc_00868248;
}

void june_evt_func_0086C7C4(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086C7EC;
    ctx->lr = 0x0086C7E8; june_evt_func_00866A34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086C7EC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_0086C7FC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[30]);
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[27]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086C84C; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 27, 27, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086C8C4;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CAFC;
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
loc_0086C8B0:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
loc_0086C8C4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086C908;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CAC0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086CAC8;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
loc_0086C908:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x94);
    ctx->lr = 0x0086C920; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CA3C;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CA70;
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 23, 23, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CA68;
loc_0086C980:
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x8);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[9], 1, 63);
loc_0086C98C:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086C9D4;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CAF0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62C4);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(0x20);
loc_0086C9AC:
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086C9D0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086C9D4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CA3C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->lr = 0x0086CA38; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086CA3C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_0086CA68:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x8);
    goto loc_0086C98C;
loc_0086CA70:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CAE4;
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x62C4);
loc_0086CA80:
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->lr = 0x0086CAA4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 23, 23, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CA68;
    goto loc_0086C980;
loc_0086CAC0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ;
loc_0086CAC8:
    ctx->fpr[0] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    goto loc_0086C908;
loc_0086CAE4:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62C4);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(0x10);
    goto loc_0086CA80;
loc_0086CAF0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62C4);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(0x30);
    goto loc_0086C9AC;
loc_0086CAFC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_0086C8B0;
}

void june_evt_func_0086CB04(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    ctx->gpr[9] = vm_read16(ctx->gpr[4] + 0x0);
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[26]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 25, 25, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[29]);
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[27] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CB74;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 25, 1, 31, 0, 0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 7, 0, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
    vm_write16(ctx->gpr[4] + 0x0, ctx->gpr[0]);
loc_0086CB74:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 16, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CCB0;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
loc_0086CBA8:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = 0x0086CBBC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CD18;
loc_0086CBCC:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
loc_0086CBD0:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CF8C;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0086CE70;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x14);
loc_0086CBEC:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CF9C;
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CFDC;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x62AC);
    ctx->gpr[4] = (int64_t)(int32_t)(0xD8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62A8);
    ctx->lr = 0x0086CC0C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086CC10:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CE98;
    ctx->lr = 0x0086CC1C; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CE98;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = vm_read8(ctx->gpr[31] + 0x8);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(7);
    ctx->lr = 0x0086CC68; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[5] = vm_read8(ctx->gpr[31] + 0x8);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(7);
    ctx->lr = 0x0086CCA0; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0086CCA8; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086CE98;
loc_0086CCB0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CBA8;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CBA8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CBA8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x0086CD04; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CBCC;
    ;
loc_0086CD18:
    ctx->lr = 0x0086CD1C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CBCC;
    ctx->lr = 0x0086CD30; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086D068;
loc_0086CD44:
    ctx->lr = 0x0086CD48; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086CD54; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0086CD90; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086D048;
loc_0086CDA0:
    ctx->lr = 0x0086CDA4; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086CDB0; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CBCC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CBCC;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[29], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CBCC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CBCC;
    ctx->lr = 0x0086CE30; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CBCC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62B0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CBCC;
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x8);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 1, 63);
    vm_write8(ctx->gpr[31] + 0x8, ctx->gpr[9]);
    goto loc_0086CBD0;
loc_0086CE70:
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CFAC;
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D000;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x62AC);
    ctx->gpr[4] = (int64_t)(int32_t)(0xC4);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62A8);
    ctx->lr = 0x0086CE94; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086CE98:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CF08;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0086CEF4; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CF48;
loc_0086CF08:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_0086CF48:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_0086CF8C:
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0086CC10;
    goto loc_0086CE98;
loc_0086CF9C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x8);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0086CC10;
loc_0086CFAC:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[11] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CE98;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x8, ctx->gpr[11]);
loc_0086CFD0:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CBEC;
    goto loc_0086CC10;
loc_0086CFDC:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D030;
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    goto loc_0086CC10;
loc_0086D000:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0086D088;
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086CE98;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x8, ctx->gpr[9]);
    goto loc_0086CFD0;
loc_0086D030:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ctx->gpr[11] & ~ctx->gpr[0];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0086CC10;
loc_0086D048:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62B4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[29] = ppu_frsp(ctx->fpr[0]);
    goto loc_0086CDA0;
loc_0086D068:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62B8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0086CD44;
loc_0086D088:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086CE98;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    goto loc_0086CFD0;
}

void june_evt_func_0086D0A4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D0D0;
    ctx->lr = 0x0086D0CC; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086D0D0:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086D0D8; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0086D2A0(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[26]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    ctx->gpr[9] = vm_read16(ctx->gpr[4] + 0x8);
    ctx->gpr[25] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[29]);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 26, 26, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[26] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[28]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D3C4;
loc_0086D34C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_0086D3C4:
    ctx->lr = 0x0086D3C8; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[25] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[23] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->fpr[24] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0xC);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D40C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D6D0;
loc_0086D40C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62A0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(0xC0);
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(0x180);
    ctx->gpr[10] = ctx->gpr[9] + (int64_t)(0x30);
    ctx->gpr[8] = ctx->gpr[9] + (int64_t)(0x60);
    ctx->gpr[7] = ctx->gpr[9] + (int64_t)(0x90);
    ctx->gpr[5] = ctx->gpr[9] + (int64_t)(0xF0);
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0x120);
    ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x150);
    ctx->gpr[21] = ctx->gpr[0] + ctx->gpr[6];
    ctx->gpr[17] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[24] = ctx->gpr[0] + ctx->gpr[10];
    ctx->gpr[23] = ctx->gpr[0] + ctx->gpr[8];
    ctx->gpr[22] = ctx->gpr[0] + ctx->gpr[7];
    ctx->gpr[20] = ctx->gpr[0] + ctx->gpr[5];
    ctx->gpr[19] = ctx->gpr[0] + ctx->gpr[4];
    ctx->gpr[18] = ctx->gpr[0] + ctx->gpr[3];
    ctx->gpr[6] = ctx->gpr[0] + ctx->gpr[9];
loc_0086D458:
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[22]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x629C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[23], ctx->fpr[21]));
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[23], ctx->fpr[31]));
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[29] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[31]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[31]));
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D4C4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D4F0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D51C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D548; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6298); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[21], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[22], ctx->fpr[30]));
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0086D58C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D5B8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D5F0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D61C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086D654; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6294); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[22], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x4);
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[21], ctx->fpr[31]));
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D6A8;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0086D6A4; june_evt_func_00866C10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086D6A8:
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D34C;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0086D6C8; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086D34C;
loc_0086D6D0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x62A0);
    ctx->gpr[17] = ctx->gpr[9] + (int64_t)(0x180);
    ctx->gpr[6] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[24] = ctx->gpr[9] + (int64_t)(0x30);
    ctx->gpr[23] = ctx->gpr[9] + (int64_t)(0x60);
    ctx->gpr[22] = ctx->gpr[9] + (int64_t)(0x90);
    ctx->gpr[21] = ctx->gpr[9] + (int64_t)(0xC0);
    ctx->gpr[20] = ctx->gpr[9] + (int64_t)(0xF0);
    ctx->gpr[19] = ctx->gpr[9] + (int64_t)(0x120);
    ctx->gpr[18] = ctx->gpr[9] + (int64_t)(0x150);
    goto loc_0086D458;
}

void june_evt_func_0086D6FC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D730;
    ctx->lr = 0x0086D72C; june_evt_func_008681FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086D730:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D748;
    ctx->lr = 0x0086D744; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086D748:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0086D750; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0086D860(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x100, ctx->gpr[1]); ctx->gpr[1] += -0x100;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
    ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 25, 25, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D8D8;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 25, 1, 31, 0, 0);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 7, 0, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
    vm_write16(ctx->gpr[31] + 0x8, ctx->gpr[0]);
loc_0086D8D8:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 26, 26, 0, 0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D928;
loc_0086D8E8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
    return;
loc_0086D928:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x0086D93C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D8E8;
    ctx->lr = 0x0086D954; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DBC4;
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DBC4;
    ctx->lr = 0x0086D974; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D9A4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6284);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0086D9A4:
    ctx->lr = 0x0086D9A8; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086D9B4; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086D9C0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6290); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D9F0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6280);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0086D9F0:
    ctx->lr = 0x0086D9F4; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086DA00; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DBC4;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[24];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[25];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086DC84;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086DC84;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[23];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086DC84;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086DC84;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DD0C;
    ctx->lr = 0x0086DABC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DCE0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x627C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DCE0;
loc_0086DAE0:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
loc_0086DAEC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x629C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6294); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[9]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DB58;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->lr = 0x0086DB48; june_evt_func_0086780C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DC40;
loc_0086DB58:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D8E8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0086DB80; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086D8E8;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[29];
    goto loc_0086D8E8;
loc_0086DB9C:
    ctx->lr = 0x0086DBA0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DD44;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6278);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DD44;
loc_0086DBC4:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0086DAEC;
    ctx->gpr[11] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DAEC;
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DC48;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DAEC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6274);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 21, 21, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DAEC;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DAEC;
    ctx->lr = 0x0086DC1C; june_evt_func_0083F098(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DAEC;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    goto loc_0086DAEC;
loc_0086DC40:
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    goto loc_0086DB58;
loc_0086DC48:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DC64;
    ctx->lr = 0x0086DC54; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DCA0;
loc_0086DC64:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DAEC;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    goto loc_0086DAEC;
loc_0086DC84:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DBC4;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    goto loc_0086DAEC;
loc_0086DCA0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x0086DCD0; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0086DCD8; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086DC64;
loc_0086DCE0:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DDA0;
loc_0086DCF0:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DBC4;
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    goto loc_0086DAEC;
loc_0086DD0C:
    ctx->lr = 0x0086DD10; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DD34;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6278);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DBC4;
loc_0086DD34:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DB9C;
loc_0086DD44:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DCF0;
    ctx->lr = 0x0086DD50; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DCF0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x0086DD90; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0086DD98; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086DCF0;
loc_0086DDA0:
    ctx->lr = 0x0086DDA4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DCF0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x627C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086DCF0;
    goto loc_0086DAE0;
}

void june_evt_func_0093A8F0(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xD0, tmp); }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093A930; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x0093A938; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AD54;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AAE8;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x84);
loc_0093A98C:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093A9B0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AB70;
loc_0093A9C0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0093A9E0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093ABC8;
loc_0093A9F0:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0093AA1C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AC30;
loc_0093AA2C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0093AA4C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AC88;
loc_0093AA5C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[30]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0093AAA4; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093ACB4;
loc_0093AAB8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_0093AAE8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[30]));
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0093AB20; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0093AB5C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A9C0;
    ;
loc_0093AB70:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2EA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093AB94; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0093ABB8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A9F0;
loc_0093ABC8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2EA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093ABEC; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0093AC1C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AA2C;
    ;
loc_0093AC30:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2EA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093AC54; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0093AC78; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AA5C;
loc_0093AC88:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2EA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093ACAC; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093AA5C;
loc_0093ACB4:
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    ctx->lr = 0x0093ACFC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AAB8;
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0093AD20; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_0093AD54:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093AD74; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_0093A98C;
}

void june_evt_func_0084DA30(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[15] = ctx->gpr[7] | ctx->gpr[7];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[31]);
    ctx->gpr[18] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[29]);
    ctx->gpr[24] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[19] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[21] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[20] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[17] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084DB5C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[16] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[5] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[24] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084DBB8;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DB58;
loc_0084DAF0:
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DB58;
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ;
loc_0084DB18:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[8] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0084DB3C; june_evt_func_0084DA30(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DB4C;
    ctx->gpr[16] = (int64_t)(int32_t)(1);
loc_0084DB4C:
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084DB18;
loc_0084DB58:
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[16];
loc_0084DB5C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[15] = _cs_15;
    ctx->gpr[16] = _cs_16;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_0084DBB8:
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x1014);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    ctx->lr = 0x0084DBC4; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DCF8;
loc_0084DBE4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DD14;
loc_0084DBF4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DCEC;
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[16] = (int64_t)(int32_t)(0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ;
loc_0084DC28:
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[27], 2, 0, 29, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[27], 6, 0, 25, 0, 0);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[10];
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0xC);
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[7] = ctx->gpr[7] + ctx->gpr[0];
    ctx->gpr[8] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[10] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[29] = ctx->gpr[27] + (int64_t)(1);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0084DC98; june_evt_func_00942D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DCAC;
    ctx->gpr[16] = (int64_t)(int32_t)(1);
loc_0084DCAC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084DC28;
loc_0084DCB8:
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[24] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DB58;
    goto loc_0084DAF0;
loc_0084DCEC:
    ctx->gpr[16] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_0084DCB8;
loc_0084DCF8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x9F2);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68C4);
    ctx->lr = 0x0084DD08; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_0084DBE4;
loc_0084DD14:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x9F3);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68C0);
    ctx->lr = 0x0084DD24; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084DBF4;
}

void june_evt_func_0084EB88(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1B0, ctx->gpr[1]); ctx->gpr[1] += -0x1B0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x1C0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[14] = ctx->gpr[4] | ctx->gpr[4];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[31]);
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[15] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    ctx->gpr[30] = ctx->gpr[9] | ctx->gpr[9];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x190, tmp); }
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    ctx->gpr[25] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1A0, tmp); }
    ctx->gpr[21] = ctx->gpr[10] | ctx->gpr[10];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    ctx->gpr[19] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[29]);
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[8]);
    ctx->gpr[26] = ctx->gpr[6] | ctx->gpr[6];
    vm_write32(ctx->gpr[1] + 0xA4, ctx->gpr[7]);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0xAC, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0xB0, ctx->gpr[10]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084EE78;
    ctx->gpr[20] = vm_read32(ctx->gpr[5] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F52C;
loc_0084EC60:
    ctx->lr = 0x0084EC64; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F22C;
loc_0084EC78:
    ctx->lr = 0x0084EC7C; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0084EC88; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0084ECC4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F20C;
loc_0084ECD4:
    ctx->lr = 0x0084ECD8; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0084ECE4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x102C);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[24] = vm_read32(ctx->gpr[31] + 0x1030);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[3]);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[28] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F204;
loc_0084ED2C:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[17] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write32(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[18] = (uint64_t)((int64_t)(int32_t)ctx->gpr[24] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->lr = 0x0084ED90; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EF00;
loc_0084EDA0:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[27]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EDEC;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F24C;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
loc_0084EDD0:
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[25], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
loc_0084EDEC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EEF8;
    ctx->gpr[5] = vm_read32(ctx->gpr[28] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EEF8;
    ctx->gpr[31] = vm_read32(ctx->gpr[1] + 0x224);
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->gpr[26] = vm_read32(ctx->gpr[1] + 0xA4);
    ctx->gpr[27] = vm_read32(ctx->gpr[1] + 0xA8);
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0xAC);
    ctx->gpr[30] = vm_read32(ctx->gpr[1] + 0xB0);
    ;
loc_0084EE28:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = ctx->gpr[30] | ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[31]);
    ctx->lr = 0x0084EE54; june_evt_func_0084EB88(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EE64;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_0084EE64:
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EE28;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[23];
    ;
loc_0084EE78:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1C0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[14] = _cs_14;
    ctx->gpr[15] = _cs_15;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x170); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x190); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1B0);
    return;
loc_0084EEF8:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    goto loc_0084EE78;
loc_0084EF00:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EDA0;
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[19]);
    ctx->lr = 0x0084EF50; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EDA0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6840);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EDA0;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[0] = ctx->gpr[0] | 0x2493;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[26], ctx->fpr[1]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 0x1F);
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[11] * (int64_t)(int32_t)ctx->gpr[0] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 3);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[9];
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0084EFE4; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[24];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[30]));
    ctx->lr = 0x0084F02C; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->fpr[24] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[26], ctx->fpr[0]));
    ctx->lr = 0x0084F044; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = ctx->gpr[24] + ctx->gpr[0];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[30]));
    ctx->lr = 0x0084F07C; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0084F0AC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F550;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F550;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F550;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F550;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F550;
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
loc_0084F0FC:
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[17] | ctx->gpr[17];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0084F124; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F560;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F560;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F560;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F560;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F560;
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
loc_0084F174:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x683C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[17] | ctx->gpr[17];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0084F1A8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084EDA0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084EDA0;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = (int64_t)(int32_t)(4);
    vm_write32(ctx->gpr[31] + 0x1034, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[9]);
    goto loc_0084EDA0;
loc_0084F204:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    goto loc_0084ED2C;
loc_0084F20C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6844);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0084ECD4;
loc_0084F22C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6848);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0084EC78;
loc_0084F24C:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[5] = ctx->gpr[17] | ctx->gpr[17];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write32(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[25]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0084F290; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F51C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F51C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F51C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F51C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F51C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F51C;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x102C);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[19]);
    ctx->gpr[0] = ctx->gpr[0] | 0x2493;
    ctx->gpr[4] = vm_read32(ctx->gpr[11] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 0x1F);
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[11] * (int64_t)(int32_t)ctx->gpr[0] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 3);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[9];
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0084F364; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
    ctx->gpr[21] = vm_read32(ctx->gpr[3] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F398;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x81A);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6898);
    ctx->lr = 0x0084F390; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x14);
loc_0084F398:
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F544;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x224);
    ctx->gpr[20] = (int64_t)(int32_t)ctx->gpr[24];
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[0]);
loc_0084F3B4:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->lr = 0x0084F3C0; june_evt_func_008946F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[25], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F504;
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[20]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[27] = ctx->gpr[0] | ctx->gpr[0];
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[13]);
loc_0084F418:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[25] = ppc_rlw64(ctx->gpr[29], 2, 0, 29, 0, 0);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->gpr[9] = ctx->gpr[25] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[26], ctx->fpr[1]));
    ctx->lr = 0x0084F438; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[31]));
    ctx->lr = 0x0084F46C; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[30], 0, 0));
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0084F488; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F4F4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F4F4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F4F4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F4F4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[31] = vm_read64(ctx->gpr[1] + 0xB8);
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F4F4;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[27]);
    ctx->gpr[9] = ctx->gpr[25] + ctx->gpr[9];
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[23]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[19]);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
loc_0084F4F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[27] = ctx->gpr[27] + ctx->gpr[24];
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084F418;
loc_0084F504:
    ctx->gpr[23] = ctx->gpr[23] + (int64_t)(1);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint32_t)ctx->gpr[23]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084F544;
    ctx->gpr[21] = ctx->gpr[21] + (int64_t)(0x30);
    goto loc_0084F3B4;
loc_0084F51C:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_0084EDD0;
loc_0084F52C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x7BA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68B8);
    ctx->lr = 0x0084F53C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084EC60;
loc_0084F544:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_0084EDD0;
loc_0084F550:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F0FC;
    goto loc_0084F174;
loc_0084F560:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F174;
    goto loc_0084EDA0;
}

void june_evt_func_0084F79C(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x150, ctx->gpr[1]); ctx->gpr[1] += -0x150;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[24]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[18], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[19], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F858;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[12], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x0, tmp); }
loc_0084F858:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBCC;
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBCC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->fpr[18] = ctx->fpr[27];
loc_0084F878:
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F8B4;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[27], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084F8B4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBC0;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBC0;
    ctx->fpr[22] = ctx->fpr[18];
    ctx->fpr[19] = ctx->fpr[18];
loc_0084F8D4:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F910;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[27], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084F910:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBB4;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBB4;
    ctx->fpr[26] = ctx->fpr[22];
loc_0084F92C:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[22] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F968;
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[26], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084F968:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBA8;
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FBA8;
    ctx->fpr[23] = ctx->fpr[19];
    ctx->fpr[20] = ctx->fpr[19];
loc_0084F988:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084F9C4;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[26], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084F9C4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB9C;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB9C;
    ctx->fpr[25] = ctx->fpr[23];
loc_0084F9E0:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FA1C;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[25], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084FA1C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB90;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB90;
    ctx->fpr[24] = ctx->fpr[20];
    ctx->fpr[21] = ctx->fpr[20];
loc_0084FA3C:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FA78;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[25], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084FA78:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB84;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB84;
    ctx->fpr[29] = ctx->fpr[24];
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->fpr[28] = ctx->fpr[21];
loc_0084FA9C:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FAD8;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[29], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084FAD8:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB78;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB78;
    ctx->fpr[31] = ctx->fpr[28];
    ;
loc_0084FAF8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB34;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
loc_0084FB34:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB6C;
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FB6C;
    ;
loc_0084FB50:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0084FB60; june_evt_func_0084F79C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FB50;
loc_0084FB6C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FAF8;
loc_0084FB78:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FA9C;
loc_0084FB84:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FA3C;
loc_0084FB90:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F9E0;
loc_0084FB9C:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F988;
loc_0084FBA8:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F92C;
loc_0084FBB4:
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F8D4;
loc_0084FBC0:
    ctx->gpr[3] = vm_read32(ctx->gpr[20] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084F878;
loc_0084FBCC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[18], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[19], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
}

void june_evt_func_0084FC44(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[4] + 0x1C);
    ctx->gpr[7] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[24]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008503AC;
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[11] = ctx->gpr[11] | 0x2;
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[11]);
loc_0084FCB0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085039C;
    ctx->gpr[11] = ctx->gpr[11] | 0x4;
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[11]);
loc_0084FCCC:
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850388;
loc_0084FCE0:
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850348;
loc_0084FCEC:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[21] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850334;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
loc_0084FD10:
    ctx->gpr[9] = vm_read32(ctx->gpr[21] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850320;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
loc_0084FD30:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[21] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085030C;
loc_0084FD44:
    ctx->gpr[10] = vm_read32(ctx->gpr[21] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502F8;
loc_0084FD50:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[22] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502E4;
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[22] + 0x0, ctx->gpr[0]);
loc_0084FD74:
    ctx->gpr[9] = vm_read32(ctx->gpr[22] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[22] + 0x0, ctx->gpr[0]);
loc_0084FD94:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[22] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502BC;
loc_0084FDA8:
    ctx->gpr[10] = vm_read32(ctx->gpr[22] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502A8;
loc_0084FDB4:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850294;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
loc_0084FDD8:
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850280;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
loc_0084FDF8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[23] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085026C;
loc_0084FE0C:
    ctx->gpr[10] = vm_read32(ctx->gpr[23] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850258;
loc_0084FE18:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850244;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[0]);
loc_0084FE3C:
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850230;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[0]);
loc_0084FE5C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085021C;
loc_0084FE70:
    ctx->gpr[10] = vm_read32(ctx->gpr[24] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850208;
loc_0084FE7C:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008501E0;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[25] + 0x0, ctx->gpr[0]);
loc_0084FEA0:
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008501CC;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[25] + 0x0, ctx->gpr[0]);
loc_0084FEC0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[25] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008501F4;
loc_0084FED4:
    ctx->gpr[10] = vm_read32(ctx->gpr[25] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008501B8;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085015C;
loc_0084FEF8:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850180;
loc_0084FF18:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008501A4;
loc_0084FF38:
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x30);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[20], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850134;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008500D8;
loc_0084FF60:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008500FC;
loc_0084FF80:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850120;
loc_0084FFA0:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008500B0;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850054;
loc_0084FFC4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850078;
loc_0084FFE4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085009C;
loc_00850004:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085002C;
loc_00850010:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00850020; june_evt_func_0084FC44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850010;
loc_0085002C:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008500B0;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FFC4;
loc_00850054:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FFE4;
loc_00850078:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850004;
loc_0085009C:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850004;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_00850004;
loc_008500B0:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850134;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FF60;
loc_008500D8:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FF80;
loc_008500FC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FFA0;
loc_00850120:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FFA0;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_0084FFA0;
loc_00850134:
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008501B8;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FEF8;
loc_0085015C:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FF18;
loc_00850180:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084FF38;
loc_008501A4:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FF38;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_0084FF38;
loc_008501B8:
    ctx->gpr[10] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850208;
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    goto loc_0084FE7C;
loc_008501CC:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[25] + 0x0, ctx->gpr[0]);
    goto loc_0084FEC0;
loc_008501E0:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[25] + 0x0, ctx->gpr[0]);
    goto loc_0084FEA0;
loc_008501F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FED4;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_0084FED4;
loc_00850208:
    ctx->gpr[10] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850258;
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    goto loc_0084FE18;
loc_0085021C:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FE70;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_0084FE70;
loc_00850230:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    goto loc_0084FE5C;
loc_00850244:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    goto loc_0084FE3C;
loc_00850258:
    ctx->gpr[10] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502A8;
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    goto loc_0084FDB4;
loc_0085026C:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FE0C;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_0084FE0C;
loc_00850280:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    goto loc_0084FDF8;
loc_00850294:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    goto loc_0084FDD8;
loc_008502A8:
    ctx->gpr[10] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008502F8;
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    goto loc_0084FD50;
loc_008502BC:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FDA8;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[10]);
    goto loc_0084FDA8;
loc_008502D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[22] + 0x0, ctx->gpr[0]);
    goto loc_0084FD94;
loc_008502E4:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[22] + 0x0, ctx->gpr[0]);
    goto loc_0084FD74;
loc_008502F8:
    ctx->gpr[4] = vm_read32(ctx->gpr[21] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850348;
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x1014);
    goto loc_0084FCEC;
loc_0085030C:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FD44;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[4]);
    goto loc_0084FD44;
loc_00850320:
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
    goto loc_0084FD30;
loc_00850334:
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
    goto loc_0084FD10;
loc_00850348:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
    return;
loc_00850388:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084FCE0;
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[7]);
    goto loc_0084FCE0;
loc_0085039C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    goto loc_0084FCCC;
loc_008503AC:
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 30, 1, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[11]);
    goto loc_0084FCB0;
}

void june_evt_func_008555C4(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6780); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) return;
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    ctx->ctr = ctx->gpr[0];
loc_008555E0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_008555E0;
    return;
}

void june_evt_func_00859374(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xB0, tmp); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859410;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->lr = 0x008593D8; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859410;
    ;
loc_008593F0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00859400; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008593F0;
loc_00859410:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00859458;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00859424; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859458;
    ;
loc_00859438:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00859448; june_evt_func_0086BF4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859438;
loc_00859458:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[28];
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859618;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00859618;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859490;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00859490; june_evt_func_008588D4(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00859490:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859618;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[0];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859544;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008594D8;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
loc_008594D8:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859500;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_00859500:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859544;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859544;
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x78);
loc_00859520:
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x00859538; june_evt_func_00858E50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859520;
loc_00859544:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859650;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00859558; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x2C);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00859568; june_evt_func_008555F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_00859578:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008595FC;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008595FC;
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859618;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859618;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x70);
    goto loc_008595DC;
loc_008595D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859618;
loc_008595DC:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->lr = 0x008595F0; june_evt_func_00858A98(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008595D0;
loc_008595FC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00859618;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x34);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    ;
loc_00859618:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xC8);
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00859650:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00859578;
}

void june_evt_func_00889EB4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
    ctx->fpr[31] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[26]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[31]);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889F7C;
loc_00889EF0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889F70;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889F70;
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->lr = 0x00889F18; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[28] = vm_read32(ctx->gpr[27] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889F70;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ;
loc_00889F38:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x1C);
    ctx->gpr[4] = ppc_rlw64(ctx->gpr[31], 6, 0, 25, 0, 0);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->gpr[3] = ctx->gpr[30] + ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] + ctx->gpr[0];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0x30);
    ctx->lr = 0x00889F64; june_evt_func_008982F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00889F38;
loc_00889F70:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00889EF0;
loc_00889F7C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
}

void june_evt_func_00889FA8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
    ctx->fpr[31] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[25]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[31]);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A0B4;
loc_00889FE8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0088A0E4;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0088A0E4;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A0A8;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A0A8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0088A028; june_evt_func_008F3D50(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0xC);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->lr = 0x0088A03C; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[27] = vm_read32(ctx->gpr[26] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A0A8;
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_0088A058:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rlw64(ctx->gpr[30], 6, 0, 25, 0, 0);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x1C);
    ctx->gpr[3] = ctx->gpr[29] + ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[28] + ctx->gpr[4];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x30);
    ctx->lr = 0x0088A084; june_evt_func_0089800C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[0];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x0088A09C; june_evt_func_008F3E04(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088A058;
loc_0088A0A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00889FE8;
loc_0088A0B4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_0088A0E4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A0A8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0088A0FC; june_evt_func_008F3D50(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00889FE8;
    goto loc_0088A0B4;
}

void june_evt_func_008933E4(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[3] + 0x4);
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00893554;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x5AAC);
    ctx->gpr[6] = (int64_t)(int32_t)(0xC55);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 5, 0, 26, 0, 0);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5AA8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 29, 0, 0);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[22] = ctx->gpr[10] + (int64_t)(-1);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[30] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x00893478; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[3], 0, 32);
loc_00893480:
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00893548;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    goto loc_008934AC;
loc_00893498:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[26] = ctx->gpr[26] + ctx->gpr[30];
    { uint64_t a = (uint32_t)ctx->gpr[22]; uint64_t b = (uint32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[25] = ctx->gpr[25] + ctx->gpr[30];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0089353C;
loc_008934AC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 2, 0, 29, 0, 0);
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0xC);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00893498;
    ctx->gpr[29] = vm_read32(ctx->gpr[23] + 0x8);
    ctx->gpr[25] = ctx->gpr[25] + ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[28] = ctx->gpr[29] + ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[25];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[26] = ctx->gpr[26] + ctx->gpr[30];
    ctx->lr = 0x00893504; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x00893518; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0089352C; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t a = (uint32_t)ctx->gpr[22]; uint64_t b = (uint32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008934AC;
loc_0089353C:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00893480;
loc_00893548:
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->lr = 0x00893550; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00893554:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xC8);
    ctx->gpr[22] = _cs_22;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[23] = _cs_23;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
}

void june_evt_func_00AECE40(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0xB0);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00AECE60; june_evt_func_00AED78C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { uint64_t a = (uint32_t)ctx->gpr[3]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00AECE9C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0xB48); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00AECEB4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0xB4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_00AECE8C:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00AECE98; june_evt_func_00AED678(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00AECE9C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_00AECEB4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0xB4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    goto loc_00AECE8C;
}

void june_evt_func_00852F30(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[31] = vm_read32(ctx->gpr[3] + 0x28);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008530B8;
loc_00852F64:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008530AC;
    ;
loc_00852F78:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852F78;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_00852F90:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[9]);
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852FD8;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853040;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853040;
loc_00852FC0:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00852FCC; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[4]);
loc_00852FD8:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853060;
loc_00852FE8:
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853014;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
loc_00852FF8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00853008; june_evt_func_00852A50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852FF8;
loc_00853014:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->lr = 0x0085301C; june_evt_func_008578C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00853040:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5BD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6818);
    ctx->lr = 0x00853050; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    goto loc_00852FC0;
loc_00853060:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852FE8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->lr = 0x00853078; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852FE8;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[30] + 0x10);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->lr = 0x008530A4; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00852FE8;
loc_008530AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_00852F90;
loc_008530B8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5A8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x681C);
    ctx->lr = 0x008530C8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00852F64;
}

void june_evt_func_00853238(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x28);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085327C;
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0085327C; june_evt_func_00852F30(ctx); DRAIN_TRAMPOLINE(ctx);
loc_0085327C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085329C;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6810);
    ctx->gpr[4] = (int64_t)(int32_t)(0x58A);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->lr = 0x00853298; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085329C:
    ctx->gpr[28] = vm_read32(ctx->gpr[27] + 0x2C);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[27] + 0x30, ctx->gpr[30]);
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[27]);
    vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[28]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853344;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085336C;
loc_008532C8:
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008532F4;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
loc_008532D8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008532E8; june_evt_func_00852A50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008532D8;
loc_008532F4:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x1C);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->lr = 0x00853300; june_evt_func_0085A110(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853320;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->lr = 0x0085331C; june_evt_func_00889FA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00853320:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_00853344:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853364;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->lr = 0x0085335C; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[28]);
loc_00853364:
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[28]);
    goto loc_008532C8;
loc_0085336C:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085339C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->lr = 0x00853388; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008533A8;
loc_0085339C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[0]);
    goto loc_008532C8;
loc_008533A8:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[30] + 0x10);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->lr = 0x008533C0; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085339C;
}

void june_evt_func_00853880(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x8);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008539C4;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085393C;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008538EC;
loc_008538D0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x34);
    ctx->lr = 0x008538E0; june_evt_func_00853438(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008538D0;
loc_008538EC:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[29]);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[29]);
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[29]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853928;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853A20;
loc_00853914:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00853920; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[29]);
loc_00853928:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00853930; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[0]);
loc_0085393C:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[28] + 0x8, ctx->gpr[27]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008539C4;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008539C4;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853984;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x42F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6814);
    ctx->lr = 0x0085397C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x102C);
loc_00853984:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->lr = 0x00853990; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008539E8;
loc_008539A4:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x102C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853A08;
loc_008539B4:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->lr = 0x008539C0; june_evt_func_00889FA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008539C4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_008539E8:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[30] + 0x10);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->lr = 0x00853A00; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008539A4;
loc_00853A08:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6808);
    ctx->gpr[4] = (int64_t)(int32_t)(0x437);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->lr = 0x00853A18; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008539B4;
loc_00853A20:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x00853A30; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00853914;
}

void june_evt_func_0083FE10(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[25]);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[26] = ctx->gpr[8] | ctx->gpr[8];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    ctx->lr = 0x0083FE54; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[26], 0, 48);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->lr = 0x0083FE74; june_evt_func_0083F0D4(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1EB);
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0x88);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0083FEBC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[31]);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_0083FEBC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D54);
    ctx->lr = 0x0083FEC8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[31]);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
}

void june_evt_func_00866310(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0xC);
    return;
}

void june_evt_func_00871840(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6268);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xB0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xB8, tmp); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[26] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    ctx->lr = 0x0087188C; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00871898; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[11] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008718E8;
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087195C;
    ctx->gpr[3] = ctx->gpr[28] - ctx->gpr[3];
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008718E8;
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[3];
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
loc_008718E8:
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087192C;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00871964;
    ctx->fpr[0] = ctx->fpr[30];
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[9] + ctx->gpr[26];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[0] = ctx->gpr[10] - ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087192C;
    ctx->gpr[0] = ctx->gpr[6] - ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
loc_0087192C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_0087195C:
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[3]);
    goto loc_008718E8;
loc_00871964:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
}

void june_evt_func_00D2BE80(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00D2BAE8; return; }
}

void june_evt_func_00D2D29C(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00D2CF54; return; }
}

void june_evt_func_008555B8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    return;
}

void june_evt_func_008555F8(ppu_context* ctx) {
    PPU_MM_SNAP
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x677C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    return;
}

void june_evt_func_00855668(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0xC);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x12D);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008556A4;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6774);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6778);
    ctx->lr = 0x008556A0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008556A4:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 4, 0, 27, 0, 0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->lr = ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    return;
}

void june_evt_func_00856178(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x180, ctx->gpr[1]); ctx->gpr[1] += -0x180;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->fpr[27] = ctx->fpr[2];
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    ctx->fpr[29] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->gpr[25] = ctx->gpr[9] | ctx->gpr[9];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x160, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    ctx->gpr[21] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[18] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[20] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[17] = ctx->gpr[9] | ctx->gpr[9];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085629C;
loc_0085622C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x190);
    ctx->gpr[15] = _cs_15;
    ctx->gpr[16] = _cs_16;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x160); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x170); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x180);
    return;
loc_0085629C:
    ctx->gpr[22] = (int64_t)(int8_t)ctx->gpr[10];
    ctx->lr = 0x008562A4; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008562B0; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[19] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[16] = ctx->gpr[23] | ctx->gpr[23];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00856770;
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write32(ctx->gpr[15] + 0x0, ctx->gpr[0]);
loc_008562D0:
    ctx->gpr[27] = vm_read32(ctx->gpr[28] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085688C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[25]));
loc_008562E8:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008567D4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00856300; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[1]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00856780;
    ;
loc_00856318:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x88);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085646C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6724); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[28]));
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085698C;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008569D8;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->lr = 0x00856398; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008569A4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x671C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
loc_008563AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[26], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->lr = 0x008563E0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00856954;
loc_008563F0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085646C;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x672C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6718); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6714); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856870;
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6710);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ctx->gpr[15] | ctx->gpr[15];
loc_00856458:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00856468; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085646C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    ctx->lr = 0x0085648C; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008567DC;
loc_0085649C:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856580;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856580;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00856580;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x78);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_008564F4;
loc_008564EC:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x2C);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_008564F4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[30];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00856514; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00856550;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_00856550:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856580;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
    ctx->lr = 0x0085656C; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008564EC;
    ;
loc_00856580:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008565D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008567CC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008565C8;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856788;
loc_008565C8:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_008565D0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085622C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856688;
    ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008566DC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00856A48;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[31] = (int64_t)(int8_t)ctx->gpr[21];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008566A8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6724); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x0085664C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856688;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] + (int64_t)(0x18);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00856684; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00856688:
    ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0x30);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[31] = (int64_t)(int8_t)ctx->gpr[21];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008566DC;
    ;
loc_008566A8:
    ctx->fpr[1] = ctx->fpr[29];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008566D0; june_evt_func_00856178(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008566A8;
loc_008566DC:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085622C;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085622C;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085622C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6724); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xA8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[25]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x00856730; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085622C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] + (int64_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x00856768; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085622C;
loc_00856770:
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[0] = (int64_t)(int32_t)(-160);
    vm_write32(ctx->gpr[15] + 0x0, ctx->gpr[0]);
    goto loc_008562D0;
loc_00856780:
    ctx->fpr[11] = ctx->fpr[0];
    goto loc_00856318;
loc_00856788:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6724); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x008567B4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00856A8C;
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_008567CC:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_008565C8;
loc_008567D4:
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_00856318;
loc_008567DC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x670C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 62, 63);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00856984;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x672C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0085681C:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[11]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6710);
    ctx->gpr[6] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6704); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[15] | ctx->gpr[15];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->gpr[6] = ctx->gpr[6] + ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->lr = 0x00856868; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085649C;
loc_00856870:
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6710);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x10);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ctx->gpr[15] | ctx->gpr[15];
    goto loc_00856458;
loc_0085688C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[25]));
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x2C);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856A5C;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
loc_008568B8:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x8);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->lr = 0x008568E4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008562E8;
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0xC);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ctx->gpr[11] | ctx->gpr[11];
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0x73);
    ctx->gpr[0] = vm_read8(ctx->gpr[1] + 0x77);
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[0] = ppc_rldicr(ctx->gpr[9], 7, 56);
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[9], 15, 48);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[10] = ppc_rldicr(ctx->gpr[0], 16, 47);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[10];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 32, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 25, 7, 31, 0, 0);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    ctx->lr = 0x0085694C; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008562E8;
loc_00856954:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] + (int64_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0085697C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008563F0;
loc_00856984:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6708); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085681C;
loc_0085698C:
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0xA8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x98);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x671C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x88);
    goto loc_008563AC;
loc_008569A4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] + (int64_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x008569CC; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x671C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    goto loc_008563AC;
loc_008569D8:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0xA8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x671C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->lr = 0x00856A08; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008563AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] + (int64_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x00856A40; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008563AC;
loc_00856A48:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[31] = (int64_t)(int8_t)ctx->gpr[21];
    goto loc_008566A8;
loc_00856A5C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856AC8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856AFC;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5D3);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6734);
    ctx->lr = 0x00856A84; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008568B8;
loc_00856A8C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] + (int64_t)(0x18);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00856AB4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_008565C8;
loc_00856AC8:
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x672C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008568B8;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    goto loc_008568B8;
loc_00856AFC:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x672C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    goto loc_008568B8;
}

void june_evt_func_00857994(ppu_context* ctx) {
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
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B78;
loc_008579E4:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B6C;
loc_008579FC:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B60;
loc_00857A14:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B54;
loc_00857A2C:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B48;
loc_00857A44:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B3C;
loc_00857A5C:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B30;
loc_00857A74:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B24;
loc_00857A8C:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857AD4;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B18;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    goto loc_00857AB8;
loc_00857AAC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857B18;
loc_00857AB8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00857AC8; june_evt_func_00857994(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857AAC;
loc_00857AD4:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
loc_00857AD8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
    return;
loc_00857B18:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857A8C;
loc_00857B24:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857A74;
loc_00857B30:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857A5C;
loc_00857B3C:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857A44;
loc_00857B48:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857A2C;
loc_00857B54:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857A14;
loc_00857B60:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008579FC;
loc_00857B6C:
    ctx->gpr[3] = vm_read32(ctx->gpr[21] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008579E4;
loc_00857B78:
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    goto loc_00857AD8;
}

void june_evt_func_0085A3B8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[3]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[3]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x30);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[14]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A620;
loc_0085A420:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x30);
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A5FC;
loc_0085A438:
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[19] + 0x30);
    ctx->gpr[18] = vm_read32(ctx->gpr[19] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A5D4;
loc_0085A44C:
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[20] + 0x30);
    ctx->gpr[17] = vm_read32(ctx->gpr[20] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A5AC;
loc_0085A460:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[21] + 0x30);
    ctx->gpr[16] = vm_read32(ctx->gpr[21] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A584;
loc_0085A474:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x30);
    ctx->gpr[15] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A55C;
loc_0085A488:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x30);
    ctx->gpr[14] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A534;
    ctx->gpr[25] = (int64_t)(int32_t)(0);
loc_0085A4A0:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x30);
    ctx->gpr[26] = vm_read32(ctx->gpr[27] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A510;
    ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_0085A4B8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A4EC;
    ;
loc_0085A4D0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x34);
    ctx->lr = 0x0085A4E0; june_evt_func_0085A3B8(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A4D0;
loc_0085A4EC:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[28]);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[28]);
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[28]);
    ctx->lr = 0x0085A500; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A510;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    goto loc_0085A4B8;
loc_0085A510:
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[25]);
    vm_write32(ctx->gpr[27] + 0x30, ctx->gpr[25]);
    vm_write32(ctx->gpr[27] + 0x34, ctx->gpr[25]);
    ctx->lr = 0x0085A524; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A534;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    goto loc_0085A4A0;
loc_0085A534:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    vm_write32(ctx->gpr[24] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[24] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[24] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0085A54C; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[14]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A55C;
    ctx->gpr[3] = ctx->gpr[14] | ctx->gpr[14];
    goto loc_0085A488;
loc_0085A55C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    vm_write32(ctx->gpr[23] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[23] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[23] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0085A574; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[15]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A584;
    ctx->gpr[3] = ctx->gpr[15] | ctx->gpr[15];
    goto loc_0085A474;
loc_0085A584:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    vm_write32(ctx->gpr[21] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[21] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[21] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0085A59C; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[16]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A5AC;
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    goto loc_0085A460;
loc_0085A5AC:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    vm_write32(ctx->gpr[20] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[20] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[20] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0085A5C4; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[17]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A5D4;
    ctx->gpr[3] = ctx->gpr[17] | ctx->gpr[17];
    goto loc_0085A44C;
loc_0085A5D4:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[19] | ctx->gpr[19];
    vm_write32(ctx->gpr[19] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[19] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[19] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0085A5EC; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A5FC;
    ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
    goto loc_0085A438;
loc_0085A5FC:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    vm_write32(ctx->gpr[22] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[22] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[22] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0085A614; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A420;
loc_0085A620:
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[14] = _cs_14;
    ctx->gpr[15] = _cs_15;
    ctx->gpr[16] = _cs_16;
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[17] = _cs_17;
    ctx->gpr[18] = _cs_18;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    { g_trampoline_fn = (void(*)(void*))june_evt_func_0085A250; return; }
}
