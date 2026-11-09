
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

void june_evt_func_00853B08(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x2C);
    return;
}

void june_evt_func_0093A8E8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x48);
    return;
}

void june_evt_func_0086C740(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0xC);
    return;
}

void june_evt_func_0086343C(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    ctx->lr = 0x0086348C; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086364C;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008635A0;
    ctx->gpr[4] = ctx->gpr[27] + (int64_t)(0x28);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x64C4);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[7] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00863510; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00863514:
    ctx->gpr[29] = vm_read32(ctx->gpr[27] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863544;
loc_00863520:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->lr = 0x00863534; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863520;
loc_00863544:
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008636E4;
loc_00863554:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008637A0;
loc_00863560:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
    return;
loc_008635A0:
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[29] = (int64_t)(int32_t)(-1);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[29]);
    ctx->lr = 0x008635DC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[10] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x28); ctx->gpr[10] += 0x28;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x8C);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    ctx->gpr[8] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0x90);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[8]);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[7] + 0x0, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x8F, ctx->gpr[29]);
    vm_write8(ctx->gpr[1] + 0x87, ctx->gpr[29]);
    ctx->lr = 0x00863644; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00863514;
loc_0086364C:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x00863664; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863560;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863544;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ;
loc_00863688:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    ctx->lr = 0x008636BC; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863688;
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863554;
loc_008636E4:
    ctx->lr = 0x008636E8; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[29] = ctx->gpr[3] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00863730; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00863760; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0086378C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    goto loc_00863864;
loc_008637A0:
    ctx->lr = 0x008637A4; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[28] = ctx->gpr[3] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x14);
    ctx->lr = 0x008637F4; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00863828; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00863854; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
loc_00863864:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00863880; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
    return;
}

void june_evt_func_00866C10(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x130, ctx->gpr[1]); ctx->gpr[1] += -0x130;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[26]);
    vm_write8(ctx->gpr[1] + 0x87, ctx->gpr[0]);
    ctx->gpr[10] = vm_read16(ctx->gpr[4] + 0x0);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[28]);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[29]);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 0, 27, 27, 0, 0);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[27]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    vm_write8(ctx->gpr[1] + 0x84, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x85, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x86, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866EA0;
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[4] | ctx->gpr[4];
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[9]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; } ctx->gpr[9] += 0xC;
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00866EA0;
    { double a = ctx->fpr[10]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00866EA0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866EDC;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866DD0;
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867344;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00866D30; june_evt_func_00827578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x00866D54; june_evt_func_00827580(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[8] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[12]));
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
loc_00866D88:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[9]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[8], ctx->fpr[10]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00867238;
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[10], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6420); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[11], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ;
loc_00866DD0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866DE8;
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x1F);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867120;
loc_00866DE8:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008671AC;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867080;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(1);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00866E20; june_evt_func_008C2768(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x84);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00866E48; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[6] = (int64_t)(int32_t)(4);
    ctx->gpr[7] = (int64_t)(int32_t)(5);
    ctx->lr = 0x00866E64; june_evt_func_008C2768(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00866E68:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866EA0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00867184;
loc_00866E94:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_008670AC;
loc_00866EA0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x130);
    return;
loc_00866EDC:
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086710C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00866EF4; june_evt_func_00827578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x00866F18; june_evt_func_00827580(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->fpr[8] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[9] = ppu_fp_single(ppu_fmul(ctx->fpr[8], ctx->fpr[13]));
loc_00866F44:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00867174;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867294;
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
loc_00866F64:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
loc_00866F74:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866FA0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 23, 23, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866FA0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0086730C;
    ;
loc_00866FA0:
    { double a = ctx->fpr[9]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00867168;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866FF0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008672F0;
loc_00866FD4:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
loc_00866FF0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
loc_00867000:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866DD0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 22, 22, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866DD0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00866DD0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00867060;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
loc_00867060:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    goto loc_00866DD0;
loc_00867080:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x008670A4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00866E68;
loc_008670AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x008670CC; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x130);
    return;
loc_0086710C:
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[10] = ctx->fpr[31];
    ctx->fpr[9] = ctx->fpr[8];
    goto loc_00866F44;
loc_00867120:
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] + (int64_t)(0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x00867160; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00866DE8;
loc_00867168:
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008672D0;
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    goto loc_00867000;
loc_00867174:
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00867270;
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_00866F74;
loc_00867184:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA8);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x008671A4; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00866E94;
loc_008671AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866E68;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866E68;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008673DC;
    ctx->gpr[28] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x108);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866E68;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x10C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866E68;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = (int64_t)(int32_t)(0x3FE);
    ctx->lr = 0x0086721C; june_evt_func_008A0B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867350;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6414); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x30, tmp); }
    goto loc_00866E68;
loc_00867238:
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[9], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6420); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[8], ctx->fpr[31]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[11], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    goto loc_00866DD0;
loc_00867270:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6420); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    goto loc_00866F74;
loc_00867294:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    goto loc_00866F64;
loc_008672D0:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6420); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    goto loc_00867000;
loc_008672F0:
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    goto loc_00866FD4;
loc_0086730C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    goto loc_00866FA0;
loc_00867344:
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00866D88;
loc_00867350:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(1);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0086736C; june_evt_func_008C2768(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[2] = ctx->fpr[30];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x108);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x10C);
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6418);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x641C);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0xC);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x008673A4; june_evt_func_008EF7F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[6] = (int64_t)(int32_t)(4);
    ctx->gpr[7] = (int64_t)(int32_t)(5);
    ctx->lr = 0x008673C0; june_evt_func_008C2768(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->lr = 0x008673CC; june_evt_func_008F6EB4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x30, tmp); }
    goto loc_00866E68;
loc_008673DC:
    ctx->lr = 0x008673E0; june_evt_func_008F6EB4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x30, tmp); }
    goto loc_00866E68;
}

void june_evt_func_0086780C(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x100, ctx->gpr[1]); ctx->gpr[1] += -0x100;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x8);
    ctx->gpr[22] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867878;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00867C6C;
loc_00867878:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
loc_00867880:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 26, 26, 0, 0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678A0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 26, 1, 31, 0, 0);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 6, 0, 31, 0, 0);
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
loc_008678A0:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086790C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[27];
    ctx->lr = 0x008678B4; june_evt_func_008F21F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086790C;
    ;
loc_008678C8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[30];
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[24] = _cs_24;
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
loc_0086790C:
    ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867AC8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00868050;
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00868050;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867D88;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00868080;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00867978; june_evt_func_00827578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0086799C; june_evt_func_00827580(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[12]));
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[30] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
loc_008679C4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008679E0;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 23, 23, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008680BC;
loc_008679E0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
loc_008679F4:
    { double a = ctx->fpr[12]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00868058;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
loc_00867A0C:
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[28]);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[31]));
    ctx->lr = 0x00867A34; june_evt_func_00AEC4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[1]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[28]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[31]));
    ctx->lr = 0x00867A64; june_evt_func_00AEC4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[25];
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[1]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x14, tmp); }
    vm_write8(ctx->gpr[31] + 0x3C, ctx->gpr[25]);
    vm_write8(ctx->gpr[31] + 0x3D, ctx->gpr[26]);
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0086808C;
loc_00867A84:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00867AC8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00867AB0; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867AC8;
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[27];
    ;
loc_00867AC8:
    ctx->lr = 0x00867ACC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867AF0;
loc_00867ADC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6400);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867E9C;
loc_00867AF0:
    ctx->lr = 0x00867AF4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867D04;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x63F4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867D04;
loc_00867B18:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = 0x00867B28; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    ctx->lr = 0x00867B3C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00868100;
loc_00867B50:
    ctx->lr = 0x00867B54; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00867B60; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00867B9C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008680E0;
loc_00867BAC:
    ctx->lr = 0x00867BB0; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00867BBC; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    ctx->lr = 0x00867C44; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    goto loc_00868024;
loc_00867C6C:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = (ctx->gpr[9] & 64) ? 0 : (ctx->gpr[0] << (ctx->gpr[9] & 63));
    ctx->gpr[9] = (int64_t)(int32_t)(0x6C);
    ctx->gpr[30] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int64_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867D30;
    ctx->gpr[27] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867DB0;
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = vm_read32(ctx->gpr[4] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6408); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x18, tmp); }
    vm_write32(ctx->gpr[4] + 0xC, ctx->gpr[27]);
    vm_write32(ctx->gpr[4] + 0x10, ctx->gpr[27]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x14, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867E94;
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867CD0;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->lr = 0x00867CC8; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[30]);
loc_00867CD0:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867CEC;
    ctx->gpr[3] = ctx->gpr[29] + (int64_t)(8);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[29]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00867CE8; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00867CEC:
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008680C8;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00867D78;
loc_00867D04:
    ctx->lr = 0x00867D08; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x63F4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    goto loc_00867B18;
loc_00867D30:
    ctx->gpr[10] = vm_read32(ctx->gpr[4] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867E94;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->lr = 0x00867D44; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 28, 28, 0, 0);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[10]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867D78;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[0]);
    ;
loc_00867D78:
    ctx->gpr[0] = ctx->gpr[8] | 0x20;
    ctx->gpr[8] = ctx->gpr[0] | ctx->gpr[0];
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00867880;
loc_00867D88:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[27];
    vm_write8(ctx->gpr[31] + 0x3D, ctx->gpr[0]);
    vm_write8(ctx->gpr[31] + 0x3C, ctx->gpr[0]);
    ctx->lr = 0x00867D9C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00867ADC;
    goto loc_00867AF0;
loc_00867DB0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867878;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00868120;
    ctx->gpr[30] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00867DDC; june_evt_func_008A17E4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[31] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00867DF4; june_evt_func_008A1598(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00868184;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[30] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008681F4;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867E30;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00867E28; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[27]);
loc_00867E30:
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867E4C;
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(8);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[30]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00867E48; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00867E4C:
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867E6C;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
loc_00867E6C:
    ctx->gpr[0] = ctx->gpr[8] | 0x20;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00867E7C; june_evt_func_008F6EB4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x30, tmp); }
    goto loc_00867880;
loc_00867E94:
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    goto loc_00867D78;
loc_00867E9C:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = 0x00867EAC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    ctx->lr = 0x00867EC0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867EF0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x63FC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00867EF0:
    ctx->lr = 0x00867EF4; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00867F00; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6410); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00867F3C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867F68;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x63F8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_00867F68:
    ctx->lr = 0x00867F6C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00867F78; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008678C8;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    ctx->lr = 0x00868000; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008678C8;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(3);
loc_00868024:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00868040; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00868048; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008678C8;
loc_00868050:
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    goto loc_008678C8;
loc_00868058:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 22, 22, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867A0C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6424); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    goto loc_00867A0C;
loc_00868080:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_008679C4;
loc_0086808C:
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x008680A4; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867A84;
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    goto loc_00867A84;
loc_008680BC:
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_008679F4;
loc_008680C8:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[27]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[27]);
    goto loc_00867D78;
loc_008680E0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x63F8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    goto loc_00867BAC;
loc_00868100:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x63FC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00867B50;
loc_00868120:
    ctx->gpr[3] = vm_read32(ctx->gpr[4] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008681DC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00868134; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 28, 28, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008681E4;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_00868160:
    ctx->gpr[0] = ctx->gpr[8] | 0x20;
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x0086816C; june_evt_func_008F6EB4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x30, tmp); }
    goto loc_00867880;
loc_00868184:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008681EC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->lr = 0x00868198; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 28, 28, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008681C4;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
loc_008681C4:
    ctx->gpr[0] = ctx->gpr[8] | 0x20;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6414); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x30, tmp); }
    ctx->gpr[8] = ctx->gpr[0] | ctx->gpr[0];
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00867880;
loc_008681DC:
    ctx->gpr[8] = vm_read16(ctx->gpr[4] + 0x0);
    goto loc_00868160;
loc_008681E4:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00868160;
loc_008681EC:
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    goto loc_008681C4;
loc_008681F4:
    ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x0);
    goto loc_00867E6C;
}

void june_evt_func_008638C4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xD0, tmp); }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863934;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 29, 1, 31, 0, 0);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
loc_00863934:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863AB8;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863AEC;
loc_00863980:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863C94;
loc_0086398C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863A3C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863A04;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ;
loc_008639B8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x008639CC; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008639E0;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
loc_008639E0:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008639B8;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[28];
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00863A04;
    ;
loc_00863A00:
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[27];
loc_00863A04:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_00863A3C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863A04;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ;
loc_00863A58:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    ctx->lr = 0x00863A8C; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863AA0;
    ctx->gpr[27] = (int64_t)(int32_t)(1);
loc_00863AA0:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863A58;
    goto loc_00863A00;
loc_00863AB8:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x00863AC8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[27];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863A04;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863980;
loc_00863AEC:
    ctx->lr = 0x00863AF0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863ECC;
loc_00863B04:
    ctx->lr = 0x00863B08; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00863B14; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00863B50; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863E88;
    ctx->lr = 0x00863B64; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00863B70; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863DE8;
loc_00863B84:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
loc_00863B88:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863F80;
loc_00863B94:
    ctx->gpr[9] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); } ctx->gpr[9] += 0x10;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[8] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[8]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863DD4;
    ctx->fpr[6] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863DD4;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863DD4;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[7], ctx->fpr[5]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863DD4;
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00863C3C;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00863C3C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00863C64;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[7]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[5]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[5] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[5]));
    { float ftmp = (float)ctx->fpr[5]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
loc_00863C64:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00863C78;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[9]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00863C78:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[7], ctx->fpr[5]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00864050;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[7]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    goto loc_00863980;
loc_00863C94:
    ctx->lr = 0x00863C98; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863F48;
loc_00863CAC:
    ctx->lr = 0x00863CB0; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00863CBC; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00863CF8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863EEC;
    ctx->lr = 0x00863D0C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00863D18; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863DBC;
loc_00863D44:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863DBC;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[11]));
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863DBC;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[10]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863DBC;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_0086398C;
loc_00863DBC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863F68;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0086398C;
loc_00863DD4:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00863980;
loc_00863DE8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863B84;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863B84;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863B84;
    ctx->lr = 0x00863E54; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863B84;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64B0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863B84;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00863B88;
loc_00863E88:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64B4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x00863EA8; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00863EB4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863B84;
    goto loc_00863DE8;
loc_00863ECC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64B8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00863B04;
loc_00863EEC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64B4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x00863F0C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00863F18; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863DBC;
    goto loc_00863D44;
loc_00863F48:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64B8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00863CAC;
loc_00863F68:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 30, 1, 31, 0, 0);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_0086398C;
loc_00863F80:
    ctx->lr = 0x00863F84; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864030;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64AC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864030;
    ctx->lr = 0x00863FAC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863FE4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64A8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
loc_00863FE4:
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    ctx->lr = 0x00863FEC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00864024;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x64A4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
loc_00864024:
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    goto loc_00863B94;
loc_00864030:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[9]);
    goto loc_00863B94;
loc_00864050:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00863980;
}

void june_evt_func_00861D48(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x28);
    return;
}

void june_evt_func_00826B90(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[4] + (int64_t)(8);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 18, 18, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00826BE0;
loc_00826BC4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00826BE0:
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x73F8);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xF8);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0xFC);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xF4);
    vm_write32(ctx->gpr[31] + 0x44, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[9]);
    ctx->lr = 0x00826BFC; june_evt_func_00D0321C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0xF8, ctx->gpr[29]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00826BC4;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xF4);
    ctx->lr = 0x00826C1C; june_evt_func_00D0A1E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xF4);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00826C2C; june_evt_func_00D0A308(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}



void june_evt_func_008A0B54(ppu_context* ctx) {
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
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[30] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x104);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[24]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[28] = ctx->gpr[8] | ctx->gpr[8];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[25] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[26] = ctx->gpr[8] | ctx->gpr[8];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008A0E6C;
loc_008A0BC8:
    ctx->gpr[30] = vm_read32(ctx->gpr[29] + 0x100);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008A0DB8;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x104);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5864); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008A0DB8;
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008A0E0C;
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[28]);
loc_008A0C10:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t iv = ppu_f2i64(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x80);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008A0C34;
    { uint64_t a = (uint32_t)ctx->gpr[30]; uint64_t b = (uint32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008A0DB8;
loc_008A0C34:
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; } ctx->gpr[29] += 0x8;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t _b=(uint32_t)ctx->gpr[30]; ctx->gpr[0] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[9]/_b); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[0] * (int64_t)(int32_t)ctx->gpr[30]);
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[29] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[29], ctx->fpr[31]));
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[29], ctx->fpr[0]));
    ctx->lr = 0x008A0C8C; june_evt_func_008A2E68(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; } ctx->gpr[29] += 0x10;
    ctx->fpr[25] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[30], 0, 0));
    ctx->lr = 0x008A0CA8; june_evt_func_008A2E68(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[26] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[31], 0, 0));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x5C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; } ctx->gpr[29] += 0x5C;
    ctx->lr = 0x008A0CC4; june_evt_func_008A2E68(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[30] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[27], 0, 0));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; } ctx->gpr[29] += 0x64;
    ctx->lr = 0x008A0CE0; june_evt_func_008A2E68(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[28], 0, 0));
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(4);
    ctx->fpr[1] = ctx->fpr[29];
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008A0D00; june_evt_func_009477DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x58);
    ctx->fpr[28] = ctx->fpr[1];
    ctx->fpr[1] = ctx->fpr[29];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008A0D18; june_evt_func_009477DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5874); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x586C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[1], ctx->fpr[0], 0, 0));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0xC, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x8, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[26] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[26], ctx->fpr[28], ctx->fpr[13], 0, 0));
    ctx->fpr[25] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[28], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; } ctx->gpr[29] += 0xB0;
    ctx->lr = 0x008A0D60; june_evt_func_008A2E68(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[28] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[31], 0, 0));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; } ctx->gpr[29] += 0xB8;
    ctx->lr = 0x008A0D7C; june_evt_func_008A2E68(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[30], 0, 0));
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0xAC);
    ctx->fpr[1] = ctx->fpr[29];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[28]));
    ctx->lr = 0x008A0D9C; june_evt_func_009477DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5868); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[31], ctx->fpr[28], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[24] + 0x0, tmp); }
    goto loc_008A0DBC;
loc_008A0DB8:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
loc_008A0DBC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_008A0E0C:
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[0] = ctx->gpr[25] * (uint64_t)(int64_t)(0x1977);
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[26] * (int64_t)(int32_t)ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x008A0E20; june_evt_func_008A2E40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x104);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    { uint32_t _b=(uint32_t)ctx->gpr[30]; ctx->gpr[0] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[3]/_b); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[0] * (int64_t)(int32_t)ctx->gpr[30]);
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[0];
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    goto loc_008A0C10;
loc_008A0E6C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x588C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xE3);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5870);
    ctx->lr = 0x008A0E7C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008A0BC8;
}

void june_evt_func_008EF7F8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xB0, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->fpr[30] = ctx->fpr[1];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xB8, tmp); }
    ctx->fpr[31] = ctx->fpr[2];
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[25]);
    ctx->gpr[24] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[25] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008EF848; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x90);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->lr = 0x008EF85C; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[10] = ctx->gpr[3] | ctx->gpr[3];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[7] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[8] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = vm_read8(ctx->gpr[28] + 0x3);
    ctx->gpr[11] = vm_read8(ctx->gpr[28] + 0x2);
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = vm_read8(ctx->gpr[28] + 0x1);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 8, 0, 23, 0, 0);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x24, tmp); } ctx->gpr[10] += 0x24;
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
    ctx->gpr[6] = ppc_rlw64(ctx->gpr[6], 16, 0, 15, 0, 0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x8, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 24, 0, 7, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x6C, tmp); } ctx->gpr[7] += 0x6C;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x48, tmp); } ctx->gpr[8] += 0x48;
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x8, tmp); }
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[6];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0xC, tmp); }
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0xC, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
    ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    vm_write32(ctx->gpr[3] + 0x10, ctx->gpr[11]);
    vm_write32(ctx->gpr[10] + 0x10, ctx->gpr[11]);
    vm_write32(ctx->gpr[8] + 0x10, ctx->gpr[11]);
    vm_write32(ctx->gpr[7] + 0x10, ctx->gpr[11]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x18, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x18, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x1C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x20, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x1C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x20, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x20, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x20, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x1C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x1C, tmp); }
    ctx->lr = 0x008EF994; june_evt_func_008C29B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x134);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x008EF9A8; june_evt_func_008C29B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(4);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x008EF9C4; june_evt_func_008B95F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x008EF9D8; june_evt_func_008C29B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
}

void june_evt_func_0086DF2C(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6270);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086DF4C;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_0086DF4C:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    return;
}

void june_evt_func_0086DF54(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6270);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    return;
}

void june_evt_func_0086E3AC(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x0086E3BC; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    return;
}

void june_evt_func_0086E44C(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x0086E45C; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    return;
}

void june_evt_func_008A1598(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0x24);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008A1644;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[1]));
    { uint64_t iv = ppu_f2i64(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + -0x10, tmp); }
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + -0x10);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008A164C;
loc_008A15DC:
    ctx->gpr[10] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->gpr[8] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t _b=(uint32_t)ctx->gpr[11]; ctx->gpr[0] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[9]/_b); }
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[0] * (int64_t)(int32_t)ctx->gpr[11]);
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    { uint32_t _b=(uint32_t)ctx->gpr[10]; ctx->gpr[9] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[0]/_b); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[10]);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[11]);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[4] + 0x4, tmp); }
loc_008A1644:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[8];
    return;
loc_008A164C:
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008A15DC;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[8];
    return;
}

void june_evt_func_008A17E4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0x10);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[4] = (int64_t)(int32_t)(0x86);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008A18F4;
loc_008A1810:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x87);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008A18DC;
loc_008A1824:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[28] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->lr = 0x008A1834; june_evt_func_00827578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->lr = 0x008A1844; june_evt_func_00827580(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[7] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[8]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->lr = ctx->gpr[0];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[7]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[13], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[10]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { uint32_t _b=(uint32_t)ctx->gpr[7]; ctx->gpr[3] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[3]/_b); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[3]);
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    ctx->gpr[28] = _cs_28;
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { uint32_t _b=(uint32_t)ctx->gpr[8]; ctx->gpr[29] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[29]/_b); }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[10]);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[29]);
    ctx->gpr[29] = (uint64_t)((int64_t)(int32_t)ctx->gpr[29] * (int64_t)(int32_t)ctx->gpr[3]);
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[11]));
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[29]);
    ctx->gpr[29] = _cs_29;
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_008A18DC:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5844);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5850);
    ctx->lr = 0x008A18E8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x10);
    goto loc_008A1824;
loc_008A18F4:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x584C);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5850);
    ctx->lr = 0x008A1900; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x10);
    goto loc_008A1810;
}

void june_evt_func_00AEC4C0(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x98, tmp); }
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->fpr[31] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x00AEC4E4; june_evt_func_00AED678(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00AEC50C;
loc_00AEC4F4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_00AEC50C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00AEC4F4;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0xB8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->lr = ctx->gpr[0];
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_008A2E40(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0xFD << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x43FD;
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(int32_t)((uint32_t)0xC4 << 16);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-24893);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 8, 31, 0, 0);
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[9]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[9], 0, 40);
    return;
}

void june_evt_func_008A2E68(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0xFD << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5820); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[0] | 0x43FD;
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(int32_t)((uint32_t)0xC4 << 16);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-24893);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 8, 31, 0, 0);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[9]);
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    return;
}

void june_evt_func_009477DC(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->fpr[12] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ctx->fpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[1] = ctx->fpr[0];
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[1]));
    { uint64_t iv = ppu_f2i64(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + -0x10, tmp); }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x10);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[9] = ctx->gpr[11] | ctx->gpr[11];
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x1E);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[3] = ctx->gpr[8] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[8] = ctx->gpr[8] + ctx->gpr[9];
    ctx->fpr[9] = ppu_frsp(ctx->fpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00947870;
    ctx->gpr[0] = vm_read16(ctx->gpr[3] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
loc_00947870:
    if ((((ctx->cr >> 6) & 1) == 0)) goto loc_0094788C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[9], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[1]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[11], 0, 0));
    return;
loc_0094788C:
    ctx->gpr[0] = vm_read16(ctx->gpr[8] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x10); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[10] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[9], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[1]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[11], 0, 0));
    return;
}
