
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

void june_evt_func_0093996C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[4] + 0x0);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[31] = ctx->gpr[7] | ctx->gpr[7];
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009399DC;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_009399D8;
    ctx->lr = 0x009399B0; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x19);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009399FC;
loc_009399D8:
    ctx->gpr[3] = (int64_t)(int32_t)(1);
loc_009399DC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_009399FC:
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00939A14; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00939A1C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_0093957C(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[0]);
    ctx->gpr[25] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    ctx->lr = 0x009395E8; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x009395F4; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    { double a = ctx->fpr[24]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00939954;
    { double a = ctx->fpr[30]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->fpr[24] = ctx->fpr[23];
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00939960;
loc_00939638:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->fpr[30] = ctx->fpr[23];
    ctx->fpr[1] = ctx->fpr[22];
loc_00939644:
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[25]));
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2F18);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x2F14);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[26]));
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[26]));
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[26]));
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[26]));
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009396AC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x10);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009396D8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x20);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939704; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x30);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939730; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x40);
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939768; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x50);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939794; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x60);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009397CC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x70);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009397F8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[24] + (int64_t)(4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093982C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009398F8;
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x10);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0x80);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0093989C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F00); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[7] = (int64_t)(int32_t)((uint32_t)0x3333 << 16);
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[24], ctx->fpr[1], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F0C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[26]));
    ctx->gpr[7] = ctx->gpr[7] | 0x33FF;
    ctx->gpr[8] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2F04);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    { uint64_t iv = ppu_f2i64(ctx->fpr[31], 0); memcpy(&ctx->fpr[31], &iv, 8); }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[13]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[23]));
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->lr = 0x009398F4; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_009398F8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x130);
    return;
loc_00939954:
    { double a = ctx->fpr[30]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[21] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[23]));
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00939638;
loc_00939960:
    ctx->fpr[22] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    goto loc_00939644;
}


void june_evt_func_00939540(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x14, ctx->gpr[4]);
    return;
}


void june_evt_func_00939548(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x14);
    return;
}


void june_evt_func_00849304(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00849330; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x27);
    ctx->gpr[3] = (int64_t)(int32_t)(0x78);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00849358; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849478;
loc_00849374:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0x78);
    ctx->lr = 0x00849380; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[9] = vm_read16(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x2F);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x8, tmp); }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    vm_write16(ctx->gpr[31] + 0x4, ctx->gpr[9]);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849468;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849468;
loc_008493C4:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = (int64_t)(int32_t)(8);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008493D8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849408;
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
loc_00849408:
    vm_write16(ctx->gpr[31] + 0x10, ctx->gpr[26]);
    ctx->lr = 0x00849410; june_evt_func_0086335C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write16(ctx->gpr[31] + 0x40, ctx->gpr[26]);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00849424; june_evt_func_0086335C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(0x74);
    vm_write32(ctx->gpr[31] + 0x58, ctx->gpr[3]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x00849438; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_00849468:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x691C);
    ctx->lr = 0x00849470; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008493C4;
loc_00849478:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6924);
    ctx->lr = 0x00849480; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849374;
}


void june_evt_func_00848E50(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[3] + (int64_t)(0x74);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = (int64_t)(int32_t)(0x22);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[9], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00848F38;
loc_00848E88:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00848E90; june_evt_func_00834F80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00848F1C;
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00848EB4; june_evt_func_00849090(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00848EC4; june_evt_func_00848F4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x28);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[9] + 0x20, ctx->gpr[31]);
    ctx->lr = 0x00848ED4; june_evt_func_008632F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00848EEC; june_evt_func_00849090(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00848EFC; june_evt_func_00848F4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x58);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[9] + 0x20, ctx->gpr[31]);
    ctx->lr = 0x00848F0C; june_evt_func_008632F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00848F18; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00848F1C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00848F38:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x693C);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6940);
    ctx->lr = 0x00848F44; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00848E88;
}


void june_evt_func_0084AD44(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x170, ctx->gpr[1]); ctx->gpr[1] += -0x170;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x160, tmp); }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[31]);
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[30]);
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[20] = ctx->gpr[7] | ctx->gpr[7];
    ctx->lr = 0x0084ADDC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[25] = ctx->gpr[26] | ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[22] = ctx->gpr[24] | ctx->gpr[24];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AE14;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6910);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0084AE14:
    ctx->lr = 0x0084AE18; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0084AE24; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0084AE60; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AE8C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x690C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0084AE8C:
    ctx->lr = 0x0084AE90; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0084AE9C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read16(ctx->gpr[26] + 0x4);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B014;
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B540;
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
loc_0084AEEC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF3C;
    ctx->gpr[28] = ctx->gpr[25] + (int64_t)(0x34);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x80);
loc_0084AF04:
    ctx->gpr[9] = ctx->gpr[28] + (int64_t)(-36);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B49C;
loc_0084AF18:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x30);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x10);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084AF04;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
loc_0084AF3C:
    ctx->gpr[9] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = vm_read16(ctx->gpr[26] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x2C, tmp); } ctx->gpr[9] += 0x2C;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B294;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B5E8;
loc_0084AF74:
    ctx->gpr[9] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = vm_read16(ctx->gpr[26] + 0x40);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x5C, tmp); } ctx->gpr[9] += 0x5C;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B090;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B5BC;
loc_0084AFAC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x180);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 63);
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x160); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x170);
    return;
loc_0084B014:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B64C;
loc_0084B02C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
loc_0084B034:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B7D8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0084B048:
    ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_0084B04C:
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[12], ctx->fpr[28], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[25]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    goto loc_0084AEEC;
loc_0084B090:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(8);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0xAC);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[11], 0); memcpy(&ctx->fpr[11], &iv, 8); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(0xC);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(4);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[3], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[11], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B120;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0084B108; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B7C0;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
loc_0084B120:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x58);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->lr = 0x0084B140; june_evt_func_008638C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B630;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0xB0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B198;
loc_0084B164:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0084B17C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B7C8;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
    ;
loc_0084B198:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0084B1A0; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B1C8;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0084B1BC; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BA40;
loc_0084B1C8:
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0084B1D8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AFAC;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x48);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->lr = 0x0084B1F4; june_evt_func_0086E098(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084B204; june_evt_func_0086E098(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B760;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B760;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B760;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B760;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AFAC;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B27C;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B27C;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AFAC;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AFAC;
loc_0084B27C:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x0084B288; june_evt_func_00871268(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[26] + 0x4C, ctx->gpr[31]);
    goto loc_0084AFAC;
loc_0084B294:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(8);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0xAC);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[11], 0); memcpy(&ctx->fpr[11], &iv, 8); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(0xC);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(4);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[3], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[11], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B328;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0084B30C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B7D0;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ;
loc_0084B328:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x28);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->lr = 0x0084B348; june_evt_func_008638C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B614;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0xB0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B3A0;
loc_0084B36C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0084B384; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B7B8;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ;
loc_0084B3A0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0084B3A8; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B3D0;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0084B3C4; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BA54;
loc_0084B3D0:
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0084B3E0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF74;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->lr = 0x0084B3FC; june_evt_func_0086E098(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084B40C; june_evt_func_0086E098(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B78C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B78C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B78C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B78C;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF74;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B484;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B484;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF74;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF74;
loc_0084B484:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x0084B490; june_evt_func_00871268(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[26] + 0x1C, ctx->gpr[31]);
    goto loc_0084AF74;
loc_0084B49C:
    ctx->gpr[29] = ctx->gpr[27] + ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[0]));
    ctx->lr = 0x0084B4B4; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[1]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->lr = 0x0084B4D8; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[28] + (int64_t)(-12);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[6] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[29] = vm_read32(ctx->gpr[11] + 0x20);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0xA0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF18;
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[31] = (int64_t)(int32_t)ctx->gpr[7];
loc_0084B510:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0084B52C; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B510;
    goto loc_0084AF18;
loc_0084B540:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B6D4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
loc_0084B560:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B908;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0084B574:
    ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_0084B578:
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[12], ctx->fpr[28], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[25]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    goto loc_0084AEEC;
loc_0084B5BC:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x44);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = 0x0084B5D4; june_evt_func_0084AD44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AFAC;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    goto loc_0084AFAC;
loc_0084B5E8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->lr = 0x0084B600; june_evt_func_0084AD44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF74;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    goto loc_0084AF74;
loc_0084B614:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0xB0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B36C;
    goto loc_0084B3A0;
loc_0084B630:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0xB0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B164;
    goto loc_0084B198;
loc_0084B64C:
    { double a = ctx->fpr[30]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B02C;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[28]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B02C;
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[12], ctx->fpr[29], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084BA68;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B034;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6908);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->lr = 0x0084B6A0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BB4C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6904);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BB4C;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[9]);
    goto loc_0084B034;
loc_0084B6D4:
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[12], ctx->fpr[30], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    { double a = ctx->fpr[26]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084BA38;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B560;
    { double a = ctx->fpr[29]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B560;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[27]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B560;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6908);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x0084B72C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BBB4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6904);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BBB4;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[9]);
    goto loc_0084B560;
loc_0084B760:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084AFAC;
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AFAC;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x48);
    ctx->lr = 0x0084B780; june_evt_func_00871268(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[26] + 0x4C, ctx->gpr[31]);
    goto loc_0084AFAC;
loc_0084B78C:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084AF74;
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AF74;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->lr = 0x0084B7AC; june_evt_func_00871268(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[26] + 0x1C, ctx->gpr[31]);
    goto loc_0084AF74;
loc_0084B7B8:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    goto loc_0084B3A0;
loc_0084B7C0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
    goto loc_0084B120;
loc_0084B7C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x48);
    goto loc_0084B198;
loc_0084B7D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    goto loc_0084B328;
loc_0084B7D8:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6908);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x0084B7E8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B8E8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6900);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B8E8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->lr = 0x0084B818; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B848;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68FC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
loc_0084B848:
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[29], ctx->fpr[30], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[25], ctx->fpr[10]));
    ctx->fpr[30] = ctx->fpr[13];
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[10]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084B874;
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084B874;
    ctx->fpr[30] = ctx->fpr[0];
loc_0084B874:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[12]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[18] = vm_read32(ctx->gpr[1] + 0xA0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[19] = vm_read32(ctx->gpr[1] + 0xA0);
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int32_t)ctx->gpr[19]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BAD0;
    ctx->gpr[27] = ctx->gpr[25] + (int64_t)(0x10);
    ctx->gpr[23] = (int64_t)(int32_t)(0);
loc_0084B8AC:
    ctx->gpr[0] = vm_read16(ctx->gpr[27] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BA70;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BAD8;
loc_0084B8C0:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x30);
    ctx->gpr[23] = ctx->gpr[23] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B8AC;
    ctx->fpr[12] = ctx->fpr[30];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x8, tmp); }
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_0084B04C;
loc_0084B8E8:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 30, 1, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[9]);
    goto loc_0084B04C;
loc_0084B908:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6908);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x0084B918; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BA18;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6900);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BA18;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->lr = 0x0084B948; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B978;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68F8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
loc_0084B978:
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[29], ctx->fpr[30], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[25], ctx->fpr[10]));
    ctx->fpr[30] = ctx->fpr[13];
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[10]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084B9A4;
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084B9A4;
    ctx->fpr[30] = ctx->fpr[0];
loc_0084B9A4:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[12]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[19] = vm_read32(ctx->gpr[1] + 0xA0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[23] = vm_read32(ctx->gpr[1] + 0xA0);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int32_t)ctx->gpr[23]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BBBC;
    ctx->gpr[27] = ctx->gpr[25] + (int64_t)(0x10);
    ctx->gpr[18] = (int64_t)(int32_t)(0);
loc_0084B9DC:
    ctx->gpr[0] = vm_read16(ctx->gpr[27] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BB54;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BBC4;
loc_0084B9F0:
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x30);
    ctx->gpr[18] = ctx->gpr[18] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B9DC;
    ctx->fpr[12] = ctx->fpr[30];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x8, tmp); }
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_0084B578;
loc_0084BA18:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 30, 1, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[9]);
    goto loc_0084B578;
loc_0084BA38:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    goto loc_0084B560;
loc_0084BA40:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x0084BA4C; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084AFAC;
loc_0084BA54:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x0084BA60; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084AF74;
loc_0084BA68:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    goto loc_0084B034;
loc_0084BA70:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[7] = ctx->gpr[19] - ctx->gpr[18];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x20);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BA90;
    ctx->gpr[5] = ctx->gpr[18] - ctx->gpr[19];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
loc_0084BA90:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B8C0;
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[7];
loc_0084BAA0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084BABC; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084BAA0;
    goto loc_0084B8C0;
loc_0084BAD0:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    goto loc_0084B048;
loc_0084BAD8:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[29] = ctx->gpr[19] - ctx->gpr[18];
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BAF4;
    ctx->gpr[28] = ctx->gpr[18] - ctx->gpr[19];
    ctx->gpr[29] = (int64_t)(int32_t)(0);
loc_0084BAF4:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BC54;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BCDC;
loc_0084BB18:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BC38;
loc_0084BB24:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B8C0;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x44);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[7] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0084BB48; june_evt_func_0084A1FC(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084B8C0;
loc_0084BB4C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    goto loc_0084B034;
loc_0084BB54:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[6] = ctx->gpr[23] - ctx->gpr[19];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x20);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BB74;
    ctx->gpr[4] = ctx->gpr[19] - ctx->gpr[23];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
loc_0084BB74:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084B9F0;
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[6];
loc_0084BB84:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0084BBA0; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084BB84;
    goto loc_0084B9F0;
loc_0084BBB4:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    goto loc_0084B560;
loc_0084BBBC:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    goto loc_0084B574;
loc_0084BBC4:
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[29] = ctx->gpr[23] - ctx->gpr[19];
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BBE0;
    ctx->gpr[28] = ctx->gpr[19] - ctx->gpr[23];
    ctx->gpr[29] = (int64_t)(int32_t)(0);
loc_0084BBE0:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BD18;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BD5C;
loc_0084BC04:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084BCC0;
loc_0084BC10:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084B9F0;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x44);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0084BC34; june_evt_func_0084A1FC(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084B9F0;
loc_0084BC38:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->lr = 0x0084BC50; june_evt_func_0084A1FC(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084BB24;
loc_0084BC54:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[29];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084BDCC;
loc_0084BCAC:
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084BCB8;
    ctx->fpr[0] = ctx->fpr[28];
loc_0084BCB8:
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    goto loc_0084BB18;
loc_0084BCC0:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0084BCD8; june_evt_func_0084A1FC(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084BC10;
loc_0084BCDC:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[25]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084BCAC;
    ctx->fpr[0] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    goto loc_0084BB18;
loc_0084BD18:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[25]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084BDC0;
loc_0084BD48:
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084BD54;
    ctx->fpr[0] = ctx->fpr[28];
loc_0084BD54:
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    goto loc_0084BC04;
loc_0084BD5C:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[29];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084BD48;
    ctx->fpr[0] = ctx->fpr[10];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    goto loc_0084BC04;
loc_0084BDC0:
    ctx->fpr[0] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    goto loc_0084BC04;
loc_0084BDCC:
    ctx->fpr[0] = ctx->fpr[10];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
    goto loc_0084BB18;
}


void june_evt_func_008497F4(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x2A0, ctx->gpr[1]); ctx->gpr[1] += -0x2A0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x1E8, ctx->gpr[27]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[14], 8); vm_write64(ctx->gpr[1] + 0x210, tmp); }
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[15], 8); vm_write64(ctx->gpr[1] + 0x218, tmp); }
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[16], 8); vm_write64(ctx->gpr[1] + 0x220, tmp); }
    ctx->gpr[9] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[17], 8); vm_write64(ctx->gpr[1] + 0x228, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[18], 8); vm_write64(ctx->gpr[1] + 0x230, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[19], 8); vm_write64(ctx->gpr[1] + 0x238, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0x240, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x248, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x250, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x258, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x260, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x268, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x270, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x278, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x280, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x288, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x290, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x298, tmp); }
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x188, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x198, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x1A0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x1A8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x1B8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x1C0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x1C8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x1D0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x1D8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[4] + 0x4);
    vm_write64(ctx->gpr[1] + 0x1F8, ctx->gpr[29]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x208, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849A5C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849AAC;
loc_008498B4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[29] = ctx->fpr[31];
    ctx->fpr[28] = ctx->fpr[30];
    ctx->gpr[30] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->fpr[27] = ctx->fpr[31];
    ctx->gpr[31] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x110);
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0xF0);
loc_008498DC:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008499A4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008499C0;
loc_008498F0:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[0] = ctx->gpr[22] + (int64_t)(0x20);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008498DC;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x2B0);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x210); memcpy(&ctx->fpr[14], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x218); memcpy(&ctx->fpr[15], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x220); memcpy(&ctx->fpr[16], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x228); memcpy(&ctx->fpr[17], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x230); memcpy(&ctx->fpr[18], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x238); memcpy(&ctx->fpr[19], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x240); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x248); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x250); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x258); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x260); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x268); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x270); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x278); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x280); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x288); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x290); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x298); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x2A0);
    return;
loc_008499A4:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x008499B8; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008498F0;
loc_008499C0:
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849B6C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849BDC;
loc_008499E4:
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xD0);
    ctx->fpr[25] = ctx->fpr[28];
    ctx->gpr[28] = ctx->gpr[10] + (int64_t)(0x10);
    ctx->fpr[24] = ctx->fpr[27];
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[9]);
    ctx->fpr[26] = ctx->fpr[29];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->fpr[21] = ctx->fpr[27];
    ctx->gpr[29] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849A40;
loc_00849A14:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849AFC;
loc_00849A1C:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x10);
    ctx->gpr[0] = ctx->gpr[19] + (int64_t)(0x20);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008498F0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849A14;
loc_00849A40:
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x18);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00849A54; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849A1C;
loc_00849A5C:
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[11], ctx->fpr[12], ctx->fpr[0], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_008498B4;
loc_00849AAC:
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[12], ctx->fpr[0], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    goto loc_008498B4;
loc_00849AFC:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849D08;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849D78;
loc_00849B20:
    ctx->fpr[19] = ctx->fpr[21];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->fpr[18] = ctx->fpr[25];
    ctx->gpr[24] = ctx->gpr[10] + (int64_t)(0x10);
    ctx->fpr[23] = ctx->fpr[24];
    ctx->gpr[25] = ctx->gpr[15] | ctx->gpr[15];
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[9]);
loc_00849B3C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849BC0;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849C30;
loc_00849B54:
    ctx->gpr[25] = ctx->gpr[25] + (int64_t)(0x10);
    ctx->gpr[0] = ctx->gpr[15] + (int64_t)(0x20);
    ctx->gpr[24] = ctx->gpr[24] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849B3C;
    goto loc_00849A1C;
loc_00849B6C:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x110, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x118, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x114, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[30], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x128, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x120, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x11C, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x124, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x12C, tmp); }
    goto loc_008499E4;
loc_00849BC0:
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x18);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->lr = 0x00849BD4; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849B54;
loc_00849BDC:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x114, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x110, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[30], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x11C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x124, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x12C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x118, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x120, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x128, tmp); }
    goto loc_008499E4;
loc_00849C30:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849CB4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849E50;
loc_00849C54:
    ctx->fpr[17] = ctx->fpr[19];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[14] = ftmp; }
    ctx->gpr[23] = ctx->gpr[0] | ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[9]);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x130);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x140);
    ctx->gpr[21] = ctx->gpr[10] + (int64_t)(0x10);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[9]);
loc_00849C80:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849D5C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849DCC;
loc_00849C98:
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[23] = ctx->gpr[23] + (int64_t)(0x10);
    ctx->gpr[21] = ctx->gpr[21] + (int64_t)(0x30);
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[23]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849C80;
    goto loc_00849B54;
loc_00849CB4:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[18], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[23]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xEC, tmp); }
    goto loc_00849C54;
loc_00849D08:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[30], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x108, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x100, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xFC, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x104, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x10C, tmp); }
    goto loc_00849B20;
loc_00849D5C:
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x18);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->lr = 0x00849D70; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849C98;
loc_00849D78:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[30], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xFC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x104, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x10C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF8, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x100, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x108, tmp); }
    goto loc_00849B20;
loc_00849DCC:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849F08;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849F7C;
loc_00849DF0:
    ctx->fpr[16] = ctx->fpr[14];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x158);
    ctx->gpr[18] = ctx->gpr[10] + (int64_t)(0x10);
    ctx->gpr[20] = ctx->gpr[0] | ctx->gpr[0];
loc_00849E00:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849E34;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849EA4;
loc_00849E18:
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x158);
    ctx->gpr[20] = ctx->gpr[20] + (int64_t)(0x10);
    ctx->gpr[18] = ctx->gpr[18] + (int64_t)(0x30);
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[20]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849E00;
    goto loc_00849C98;
loc_00849E34:
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x18);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->lr = 0x00849E48; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849E18;
loc_00849E50:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[18], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xEC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[23]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    goto loc_00849C54;
loc_00849EA4:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = vm_read16(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A038;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A090;
loc_00849EC8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    ctx->gpr[16] = ctx->gpr[10] + (int64_t)(0x10);
    ctx->gpr[17] = ctx->gpr[0] | ctx->gpr[0];
loc_00849ED4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849F60;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849FD4;
loc_00849EEC:
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x160);
    ctx->gpr[17] = ctx->gpr[17] + (int64_t)(0x10);
    ctx->gpr[16] = ctx->gpr[16] + (int64_t)(0x30);
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[17]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849ED4;
    goto loc_00849E18;
loc_00849F08:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[0], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[17]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[17]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    goto loc_00849DF0;
loc_00849F60:
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x18);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->lr = 0x00849F74; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849EEC;
loc_00849F7C:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[10], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[17]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[17]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    goto loc_00849DF0;
loc_00849FD4:
    ctx->gpr[14] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[14] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A128;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A1A0;
loc_00849FF4:
    ctx->gpr[0] = vm_read16(ctx->gpr[14] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A108;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A184;
loc_0084A008:
    ctx->gpr[0] = vm_read16(ctx->gpr[14] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A0E8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849EEC;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[14] + 0x44);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0084A034; june_evt_func_008497F4(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_00849EEC;
loc_0084A038:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[0], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[16]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[16]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    goto loc_00849EC8;
loc_0084A090:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[10], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[16]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[16]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    goto loc_00849EC8;
loc_0084A0E8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[14] + 0x58);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0084A100; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00849EEC;
loc_0084A108:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x168);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[14] + 0x28);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0084A120; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084A008;
loc_0084A128:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x138, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x130, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x134, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[14] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[0], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x148, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x140, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x13C, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x144, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x14C, tmp); }
    goto loc_00849FF4;
loc_0084A184:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x168);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[14] + 0x14);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0084A19C; june_evt_func_008497F4(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084A008;
loc_0084A1A0:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x134, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x130, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[14] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[10], 1, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x13C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x144, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x14C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x138, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x140, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x148, tmp); }
    goto loc_00849FF4;
}


void june_evt_func_00848DD0(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x70, ctx->gpr[4]);
    return;
}


void june_evt_func_00848DD8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x70);
    return;
}


void june_evt_func_0084A1FC(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1F0, ctx->gpr[1]); ctx->gpr[1] += -0x1F0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[26]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[14], 8); vm_write64(ctx->gpr[1] + 0x160, tmp); }
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[15], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    ctx->gpr[25] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[16], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[26] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[17], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    ctx->gpr[11] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[18], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[19], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0x190, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x1A0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x1C8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x1D0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x1D8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1E0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[0]);
    ctx->gpr[9] = vm_read16(ctx->gpr[3] + 0x4);
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[31]);
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A5A4;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A618;
loc_0084A2CC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->fpr[30] = ctx->fpr[31];
    ctx->gpr[31] = ctx->gpr[11] + (int64_t)(0x10);
    ctx->fpr[27] = ctx->fpr[31];
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->fpr[21] = ctx->fpr[31];
    ctx->gpr[16] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->gpr[15] = (int64_t)(int32_t)ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->gpr[14] = (int64_t)(int32_t)ctx->gpr[24];
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[0]);
loc_0084A2FC:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A3B8;
loc_0084A308:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x30);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A2FC;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x200);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x160); memcpy(&ctx->fpr[14], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[15], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x170); memcpy(&ctx->fpr[16], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[17], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[18], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[19], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x190); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A0); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A8); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B8); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1F0);
    return;
loc_0084A3B8:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A684;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A754;
loc_0084A3E0:
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[26];
    ctx->fpr[28] = ctx->fpr[27];
    ctx->gpr[30] = ctx->gpr[10] + (int64_t)(0x10);
    ctx->fpr[29] = ctx->fpr[22];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[9]);
    ctx->fpr[24] = ctx->fpr[27];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->fpr[25] = ctx->fpr[21];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->fpr[16] = ctx->fpr[27];
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[24];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A444;
loc_0084A424:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0x30);
    ctx->gpr[23] = ctx->gpr[23] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A308;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A424;
loc_0084A444:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A7C0;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A6EC;
loc_0084A46C:
    ctx->gpr[27] = ctx->gpr[10] + (int64_t)(0x10);
    ctx->fpr[26] = ctx->fpr[24];
    ctx->fpr[17] = ctx->fpr[24];
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->fpr[20] = ctx->fpr[16];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A4B0;
loc_0084A490:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x30);
    ctx->gpr[19] = ctx->gpr[19] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A424;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A490;
loc_0084A4B0:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A888;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A960;
loc_0084A4D8:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->fpr[23] = ctx->fpr[17];
    ctx->gpr[22] = ctx->gpr[10] + (int64_t)(0x10);
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[9]);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[26];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[25];
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[9]);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[28];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[9]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[24];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A540;
loc_0084A518:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x7C);
    ctx->gpr[22] = ctx->gpr[22] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A490;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A518;
loc_0084A540:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AA2C;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AAA0;
loc_0084A568:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[19] = ftmp; }
    ctx->gpr[20] = ctx->gpr[10] + (int64_t)(0x10);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[9]);
loc_0084A578:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A824;
loc_0084A588:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x78);
    ctx->gpr[20] = ctx->gpr[20] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A578;
    goto loc_0084A518;
loc_0084A5A4:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[5];
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[12] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084A678;
loc_0084A604:
    { double a = ctx->fpr[0]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A610;
    ctx->fpr[0] = ctx->fpr[9];
loc_0084A610:
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
    goto loc_0084A2CC;
loc_0084A618:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[4];
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[12] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084A604;
loc_0084A678:
    ctx->fpr[0] = ctx->fpr[8];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
    goto loc_0084A2CC;
loc_0084A684:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[16]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[15]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[10] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AA20;
    { double a = ctx->fpr[12]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A6E4;
    ctx->fpr[12] = ctx->fpr[31];
loc_0084A6E4:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A3E0;
loc_0084A6EC:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[14]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[12] = ppu_frsp(ctx->fpr[8]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[13]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AB64;
    { double a = ctx->fpr[11]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A74C;
    ctx->fpr[11] = ctx->fpr[28];
loc_0084A74C:
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A46C;
loc_0084A754:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[14]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[30]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[12] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[13]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AA94;
    { double a = ctx->fpr[11]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A7B8;
    ctx->fpr[11] = ctx->fpr[30];
loc_0084A7B8:
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A3E0;
loc_0084A7C0:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[16]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[25]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[15]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AB58;
    { double a = ctx->fpr[12]; double b = ctx->fpr[25]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A81C;
    ctx->fpr[12] = ctx->fpr[25];
loc_0084A81C:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A46C;
loc_0084A824:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A8EC;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AB70;
loc_0084A84C:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[14] = ftmp; }
    ctx->gpr[17] = ctx->gpr[10] + (int64_t)(0x10);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
loc_0084A85C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084A9C0;
loc_0084A86C:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[17] = ctx->gpr[17] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A85C;
    goto loc_0084A588;
loc_0084A888:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[16]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[9], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[20]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[15]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[20], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084A954;
    { double a = ctx->fpr[12]; double b = ctx->fpr[20]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A8E4;
    ctx->fpr[12] = ctx->fpr[20];
loc_0084A8E4:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A4D8;
loc_0084A8EC:
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084A948;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A94C;
loc_0084A948:
    ctx->fpr[12] = ctx->fpr[9];
loc_0084A94C:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A84C;
loc_0084A954:
    ctx->fpr[12] = ctx->fpr[9];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A4D8;
loc_0084A960:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[14]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[8], &tmp, 8); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[26]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084A954;
    { double a = ctx->fpr[12]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A8E4;
    ctx->fpr[12] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A4D8;
loc_0084A9C0:
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084ABD0;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AC38;
loc_0084A9E8:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[18] = ctx->gpr[10] + (int64_t)(0x10);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
loc_0084A9F4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AB00;
loc_0084AA04:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[18] = ctx->gpr[18] + (int64_t)(0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084A9F4;
    goto loc_0084A86C;
loc_0084AA20:
    ctx->fpr[12] = ctx->fpr[8];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A3E0;
loc_0084AA2C:
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[8], &tmp, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AA88;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084AA8C;
loc_0084AA88:
    ctx->fpr[12] = ctx->fpr[9];
loc_0084AA8C:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A568;
loc_0084AA94:
    ctx->fpr[11] = ctx->fpr[8];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A3E0;
loc_0084AAA0:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[12], &tmp, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AA88;
    { double a = ctx->fpr[12]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084AA8C;
    ctx->fpr[12] = ctx->fpr[23];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A568;
loc_0084AB00:
    ctx->gpr[21] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[9] = vm_read16(ctx->gpr[21] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084ACB4;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AD1C;
loc_0084AB24:
    ctx->gpr[0] = vm_read16(ctx->gpr[21] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084AC98;
loc_0084AB30:
    ctx->gpr[0] = vm_read16(ctx->gpr[21] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084AA04;
    ctx->gpr[3] = vm_read32(ctx->gpr[21] + 0x44);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[24];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[7] = (int64_t)(int32_t)ctx->gpr[26];
    ctx->lr = 0x0084AB54; june_evt_func_0084A1FC(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084AA04;
loc_0084AB58:
    ctx->fpr[12] = ctx->fpr[9];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A46C;
loc_0084AB64:
    ctx->fpr[11] = ctx->fpr[9];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A46C;
loc_0084AB70:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[12], &tmp, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[19]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[19], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084A948;
    { double a = ctx->fpr[12]; double b = ctx->fpr[19]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084A94C;
    ctx->fpr[12] = ctx->fpr[19];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A84C;
loc_0084ABD0:
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[8], &tmp, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AC2C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084AC30;
loc_0084AC2C:
    ctx->fpr[12] = ctx->fpr[9];
loc_0084AC30:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A9E8;
loc_0084AC38:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[12], &tmp, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[14]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[14], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AC2C;
    { double a = ctx->fpr[12]; double b = ctx->fpr[14]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084AC30;
    ctx->fpr[12] = ctx->fpr[14];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_0084A9E8;
loc_0084AC98:
    ctx->gpr[3] = vm_read32(ctx->gpr[21] + 0x14);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[24];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[7] = (int64_t)(int32_t)ctx->gpr[26];
    ctx->lr = 0x0084ACB0; june_evt_func_0084A1FC(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084AB30;
loc_0084ACB4:
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[8], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[8], 8); ctx->fpr[8] = (double)iv; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[8]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[9]);
loc_0084ACD8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6914); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6918); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084AD10;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6920); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084AD14;
loc_0084AD10:
    ctx->fpr[12] = ctx->fpr[9];
loc_0084AD14:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[21] + 0x8, tmp); }
    goto loc_0084AB24;
loc_0084AD1C:
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[21] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[10] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    goto loc_0084ACD8;
}


void june_evt_func_00938978(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2F70);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[30]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(0x38);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x4D);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->lr = 0x009389D0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938CBC;
loc_009389E0:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x38);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x009389F4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->lr = 0x00938A00; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->lr = 0x00938A20; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0xD4);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00938A3C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    ctx->lr = 0x00938A50; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x00938A60; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x2F68);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[11]);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x00938A84; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2F64);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00938A9C; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2F60);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00938AB4; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938C9C;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x00938AD8; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_00938AE4:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(6);
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = (int64_t)(int32_t)(0xE);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = (int64_t)(int32_t)(0xA);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[4]);
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0xE0, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0xDC, ctx->gpr[8]);
    vm_write32(ctx->gpr[1] + 0xE4, ctx->gpr[7]);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0xBC, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0xB8, ctx->gpr[8]);
    vm_write32(ctx->gpr[1] + 0xC0, ctx->gpr[7]);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[25], 0, 32);
    vm_write32(ctx->gpr[1] + 0xEC, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xE8, ctx->gpr[28]);
    ctx->lr = 0x00938B48; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->lr = 0x00938B60; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[3]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    ctx->gpr[26] = (int64_t)(int32_t)(3);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xC0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F5C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0xE4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[28]);
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[26]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xEC, tmp); }
    ctx->lr = 0x00938BB0; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->lr = 0x00938BC8; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[3]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xC0);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[26]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xEC, tmp); }
    vm_write32(ctx->gpr[1] + 0xE4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[28]);
    ctx->lr = 0x00938C10; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->lr = 0x00938C28; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2F58);
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
    vm_write32(ctx->gpr[1] + 0x94, ctx->gpr[0]);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x8C);
    vm_write16(ctx->gpr[1] + 0x92, ctx->gpr[28]);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[0]);
    ctx->lr = 0x00938C4C; june_evt_func_0094218C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x8C);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[3]);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_00938C9C:
    ctx->gpr[0] = (int64_t)(int32_t)(5);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    vm_write32(ctx->gpr[1] + 0x8C, ctx->gpr[4]);
    goto loc_00938AE4;
loc_00938CBC:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2F6C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4E);
    ctx->lr = 0x00938CCC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_009389E0;
}


void june_evt_func_009388CC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x28);
    ctx->lr = 0x009388E8; june_evt_func_00942088(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->lr = 0x009388F4; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x18);
    ctx->lr = 0x00938900; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x24);
    ctx->lr = 0x0093890C; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->lr = 0x00938918; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x14);
    ctx->lr = 0x00938924; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x20);
    ctx->lr = 0x00938930; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->lr = 0x0093893C; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->lr = 0x00938948; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->lr = 0x00938954; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00938960; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_009382E8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[22] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938390;
loc_00938348:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_00938390:
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[29]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x8);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009383F0; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938420; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093843C;
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[25], 0, 56);
    ctx->gpr[0] = (uint64_t)0 - ctx->gpr[0];
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 1, 63);
loc_0093843C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[31] = ctx->fpr[30];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938468; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093847C;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_0093847C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009384C0; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009384D4;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_009384D4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009384F8; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093850C;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_0093850C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938530; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938544;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_00938544:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[29]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938588; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093859C;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_0093859C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009385C0; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009385D4;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_009385D4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009385F8; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093860C;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_0093860C:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938658; june_evt_func_00941AC0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093866C;
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[24];
loc_0093866C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009388C4;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009386CC;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009386CC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009386CC;
    vm_write32(ctx->gpr[10] + 0x14, ctx->gpr[9]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[11] + 0x14, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_009386CC:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938720;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938720;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938720;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x14, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x14, tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_00938720:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938778;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938778;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938778;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x14, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x14, tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ;
loc_00938778:
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009388AC;
loc_00938790:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938808;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938808;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938808;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938808;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938808;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938808;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938348;
    ;
loc_00938808:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938848;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[7] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x14);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0xC);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[7] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
loc_00938848:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[6], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938348;
    ctx->lr = 0x00938860; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938348;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0x1B);
    ctx->lr = 0x0093889C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x009388A4; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00938348;
loc_009388AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938790;
    vm_write32(ctx->gpr[10] + 0xC, ctx->gpr[0]);
    goto loc_00938790;
loc_009388C4:
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x28);
    goto loc_00938790;
}


void june_evt_func_00938CD4(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[28]);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[30]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    ctx->gpr[20] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00938DCC;
loc_00938D5C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->lr = ctx->gpr[0];
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
loc_00938DCC:
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[18] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2FA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x00938E20; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938E48; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x20);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938E6C; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[24]));
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938EB0; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938ED4; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938EF8; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938F3C; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938F60; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x24);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938F84; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[25]));
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x28);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00938FCC; june_evt_func_00941420(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x00938FD4; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[25]));
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[19] = ctx->gpr[3] | ctx->gpr[3];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00939018; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00939114;
loc_00939028:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ctx->gpr[19] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[25]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[24]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x00939080; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009390AC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x009390E0; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093910C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00938D5C;
loc_00939114:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[1] = ctx->fpr[22];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2F50);
    ctx->lr = 0x0093912C; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00939398;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00939028;
    ctx->lr = 0x0093914C; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x60);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093915C; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x24);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[26] = ctx->gpr[3] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0xC);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ctx->gpr[26] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[26], 0, 32);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F48); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x50);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { uint64_t iv = ppu_f2i64(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[12]));
    ctx->fpr[1] = ctx->fpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[7]));
    ctx->fpr[31] = ppu_frsp(ctx->fpr[10]);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint64_t iv = ppu_f2i64(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[10]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[25]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[11]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[0]);
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[24]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[9]));
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[8]));
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[9]));
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
    ctx->gpr[29] = ppc_rldicr(ctx->gpr[9], 7, 56);
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[9], 15, 48);
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[11];
    ctx->gpr[0] = ppc_rldicr(ctx->gpr[29], 16, 47);
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[0];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[9];
    ctx->lr = 0x0093927C; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[29], 0, 0));
    ctx->fpr[1] = ctx->fpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 32, 32);
    ctx->gpr[29] = ppc_rlw64(ctx->gpr[29], 25, 7, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x0, tmp); }
    ctx->lr = 0x0093929C; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[30], 0, 0));
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x8, tmp); }
    ctx->fpr[1] = ctx->fpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[29]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x4, tmp); }
    ctx->lr = 0x009392C0; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[29], 0, 0));
    ctx->fpr[1] = ctx->fpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
    ctx->lr = 0x009392D8; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[30], 0, 0));
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x8, tmp); }
    ctx->fpr[1] = ctx->fpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[25] + 0xC, ctx->gpr[29]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x4, tmp); }
    ctx->lr = 0x009392FC; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[29], 0, 0));
    ctx->fpr[1] = ctx->fpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[24] + 0x0, tmp); }
    ctx->lr = 0x00939314; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[30], 0, 0));
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[24] + 0x8, tmp); }
    ctx->fpr[1] = ctx->fpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[24] + 0xC, ctx->gpr[29]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[24] + 0x4, tmp); }
    ctx->lr = 0x00939338; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[29], 0, 0));
    ctx->fpr[1] = ctx->fpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x0, tmp); }
    ctx->lr = 0x00939350; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[1], ctx->fpr[30], 0, 0));
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x8, tmp); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[29]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x4, tmp); }
    ctx->lr = 0x00939374; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[4] = (int64_t)(int32_t)(5);
    ctx->gpr[6] = (int64_t)(int32_t)(6);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00939390; june_evt_func_008B9724(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00939028;
loc_00939398:
    ctx->lr = 0x0093939C; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x320);
    ctx->fpr[26] = ctx->fpr[22];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x009393B0; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x24);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[24] = ctx->gpr[3] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[28] = ctx->gpr[24] | ctx->gpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = ctx->gpr[3] + (int64_t)(0x310);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F48); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2F40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[6] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint64_t iv = ppu_f2i64(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[5]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[4]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint64_t iv = ppu_f2i64(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[8]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[24]));
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[6]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[0]);
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[25]));
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[7]));
    ctx->fpr[30] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[9]));
    ctx->gpr[11] = (uint64_t)((int64_t)(int32_t)ctx->gpr[11] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[7]));
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
    ctx->gpr[0] = ppc_rldicr(ctx->gpr[11], 7, 56);
    ctx->gpr[10] = ppc_rldicr(ctx->gpr[11], 15, 48);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[10];
    ctx->gpr[9] = ppc_rldicr(ctx->gpr[0], 16, 47);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 32, 32);
    ctx->gpr[27] = ppc_rlw64(ctx->gpr[0], 25, 7, 31, 0, 0);
    ;
loc_009394B8:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x10);
    ctx->lr = 0x009394CC; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[1], ctx->fpr[28], 0, 0));
    ctx->fpr[1] = ctx->fpr[31];
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
    ctx->lr = 0x009394E8; june_evt_func_00AEDC14(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[1], ctx->fpr[29], 0, 0));
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[27]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009394B8;
    ctx->gpr[3] = ctx->gpr[24] + (int64_t)(0x300);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x00939518; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(5);
    ctx->gpr[6] = (int64_t)(int32_t)(0x32);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00939534; june_evt_func_008B9724(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00939028;
}


void june_evt_func_009382D8(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x34, ctx->gpr[4]);
    return;
}


void june_evt_func_009382E0(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x34);
    return;
}


void june_evt_func_00944330(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[9] = (int64_t)(int32_t)(-1);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[23]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[23] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[18] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[31]);
    vm_write8(ctx->gpr[1] + 0x76, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x7F, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x75, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x77, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x7B, ctx->gpr[9]);
    ctx->gpr[19] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_009445B8;
    ctx->fpr[29] = ctx->fpr[8];
    ctx->gpr[20] = (int64_t)(int32_t)(0xFF);
loc_009443D4:
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[9], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[9], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[20]; ctx->gpr[28] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009444E8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x7C);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[27] = ctx->gpr[21] | ctx->gpr[21];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[8], ctx->fpr[0]));
    ctx->fpr[31] = ctx->fpr[30];
    ;
loc_00944470:
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    vm_write8(ctx->gpr[1] + 0x7A, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x79, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[9]);
    ctx->lr = 0x009444BC; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00944470;
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_009444E8:
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[20]; ctx->gpr[9] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[20]);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00944564;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x7C);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write8(ctx->gpr[1] + 0x7A, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x79, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x78);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944560; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00944564:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[18] = _cs_18;
    ctx->gpr[19] = _cs_19;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_009445B8:
    ctx->fpr[0] = ctx->fpr[29];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[20] = vm_read32(ctx->gpr[1] + 0x90);
    goto loc_009443D4;
}


void june_evt_func_009445D0(ppu_context* ctx) {
    PPU_MM_SNAP
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
    ctx->gpr[9] = vm_read32(ctx->gpr[7] + 0x18);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    ctx->gpr[11] = vm_read32(ctx->gpr[7] + 0x30);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x14);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[16] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    vm_write16(ctx->gpr[1] + 0x72, ctx->gpr[9]);
    ctx->lr = 0x0094466C; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x7C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[17] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[21] = ctx->gpr[19] | ctx->gpr[19];
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00944894;
    ctx->fpr[29] = ctx->fpr[0];
    ctx->gpr[18] = (int64_t)(int32_t)(0x20);
loc_009446A0:
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[9], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[9], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[18]; ctx->gpr[28] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009447B0;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x84);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[31] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[31] = ctx->fpr[30];
    ;
loc_00944730:
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x0094475C; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944784; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00944730;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_009447B0:
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[18]; ctx->gpr[9] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[18]);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00944838;
    ctx->gpr[0] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00944800; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x84);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944834; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00944838:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_00944894:
    ctx->fpr[0] = ctx->fpr[29];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[18] = vm_read32(ctx->gpr[1] + 0xA0);
    goto loc_009446A0;
}


void june_evt_func_009448AC(ppu_context* ctx) {
    PPU_MM_SNAP
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
    ctx->gpr[11] = vm_read32(ctx->gpr[7] + 0x18);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    ctx->gpr[10] = vm_read32(ctx->gpr[7] + 0x24);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write16(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x14);
    ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[16] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    ctx->lr = 0x00944948; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x7C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[17] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[21] = ctx->gpr[19] | ctx->gpr[19];
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00944B6C;
    ctx->fpr[29] = ctx->fpr[0];
    ctx->gpr[18] = (int64_t)(int32_t)(0x40);
loc_0094497C:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[9], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[9], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[18]; ctx->gpr[28] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00944A88;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x84);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[31] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[31] = ctx->fpr[30];
loc_00944A08:
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    vm_write16(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->lr = 0x00944A34; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944A5C; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00944A08;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_00944A88:
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[18]; ctx->gpr[9] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[18]);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00944B10;
    ctx->gpr[0] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write16(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->lr = 0x00944AD8; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x84);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944B0C; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00944B10:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_00944B6C:
    ctx->fpr[0] = ctx->fpr[29];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[18] = vm_read32(ctx->gpr[1] + 0xA0);
    goto loc_0094497C;
}


void june_evt_func_00944B84(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[19]);
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[0] = (int64_t)(int32_t)(0x80);
    ctx->gpr[9] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[16] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    vm_write16(ctx->gpr[1] + 0x72, ctx->gpr[9]);
    ctx->lr = 0x00944C18; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x7C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[17] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[21] = ctx->gpr[19] | ctx->gpr[19];
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00944E3C;
    ctx->fpr[29] = ctx->fpr[0];
    ctx->gpr[18] = (int64_t)(int32_t)(0x10);
loc_00944C4C:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[9], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[9], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[18]; ctx->gpr[28] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00944D58;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x84);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[31] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[31] = ctx->fpr[30];
loc_00944CD8:
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x00944D04; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944D2C; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[23] + 0x0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00944CD8;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_00944D58:
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[10], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[18]; ctx->gpr[9] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[18]);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00944DE0;
    ctx->gpr[0] = (int64_t)(int32_t)(0x167);
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x00944DA8; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x84);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944DDC; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00944DE0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[16] = _cs_16;
    ctx->gpr[17] = _cs_17;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_00944E3C:
    ctx->fpr[0] = ctx->fpr[29];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[18] = vm_read32(ctx->gpr[1] + 0xA0);
    goto loc_00944C4C;
}


void june_evt_func_00879E0C(ppu_context* ctx) {
    PPU_MM_SNAP
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00879AF8; return; }
}

