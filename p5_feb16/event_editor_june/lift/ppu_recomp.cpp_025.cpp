
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

void june_evt_func_00842258(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x4D0, ctx->gpr[1]); ctx->gpr[1] += -0x4D0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x4B0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x4E0, ctx->gpr[0]);
    ctx->gpr[28] = vm_read32(ctx->gpr[3] + 0x88);
    vm_write64(ctx->gpr[1] + 0x478, ctx->gpr[21]);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x480, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x490, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x498, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x4A0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x4A8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x4C0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x488, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x4B8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x4C8, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[22] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[25] = ctx->gpr[8] | ctx->gpr[8];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084249C;
loc_008422B8:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842330;
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008423D8;
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842430;
loc_008422D4:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->lr = 0x008422F0; june_evt_func_008714AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x4E0);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x4D0);
    return;
loc_00842330:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[23] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008425B0;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x30);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008422D4;
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x14);
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->lr = 0x00842374; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[29] + 0x34, ctx->gpr[30]);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[23]);
    ctx->lr = 0x00842388; june_evt_func_00841F3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842394; june_evt_func_00842034(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008426A8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[29] + 0x40);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x008423D0; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    goto loc_008422D4;
loc_008423D8:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->lr = 0x008423F0; june_evt_func_00853B10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842558;
loc_00842400:
    ctx->gpr[0] = vm_read8(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084252C;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[3] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
    ctx->gpr[0] = ctx->gpr[9] | 0x60;
    vm_write16(ctx->gpr[3] + 0x8, ctx->gpr[0]);
    goto loc_008422D4;
loc_00842430:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842570;
    { uint64_t a = (uint32_t)ctx->gpr[25]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008424B4;
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842724;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x28);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00842748;
loc_0084247C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008422D4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x14);
    ctx->lr = 0x00842494; june_evt_func_00853B3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008422D4;
loc_0084249C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6C7C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2DB);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6C64);
    ctx->lr = 0x008424AC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008422B8;
loc_008424B4:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008424D0;
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0xFF00 << 16);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
loc_008424D0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x48);
    vm_write32(ctx->gpr[3] + 0x34, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008426A8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[3] + 0x44);
loc_008424F8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->lr = 0x00842524; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    goto loc_008422D4;
loc_0084252C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read16(ctx->gpr[3] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008422D4;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 26, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 6, 0, 31, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] | 0x40;
    vm_write16(ctx->gpr[3] + 0x8, ctx->gpr[0]);
    goto loc_008422D4;
loc_00842558:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BF8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x35F);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6C7C);
    ctx->lr = 0x00842568; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842400;
loc_00842570:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    ctx->lr = 0x0084258C; june_evt_func_00841F3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842598; june_evt_func_00842034(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008426A8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x40);
    goto loc_008424F8;
loc_008425B0:
    ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0x14);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6BEC);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008425C0; june_evt_func_00D2F758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008426C4;
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x008425D4; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008425E8; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008425F4; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00842668;
    ctx->gpr[11] = ctx->gpr[3] + (int64_t)(-1);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00842758;
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[11];
    ctx->gpr[9] = ctx->gpr[1] + ctx->gpr[10];
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x70);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008426F4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x5C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->ctr = ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00842658;
    goto loc_008426F4;
loc_00842638:
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[11];
    ctx->gpr[9] = ctx->gpr[1] + ctx->gpr[10];
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x70);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x5C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008426F4;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_008426F4;
loc_00842658:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(-1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00842638;
loc_00842660:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
loc_00842668:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->lr = 0x0084267C; june_evt_func_008F7154(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084268C; june_evt_func_00841158(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008422D4;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6C44);
    ctx->lr = 0x008426A4; june_evt_func_00841158(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_008422D4;
loc_008426A8:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0x8000 << 16);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->lr = 0x008426BC; june_evt_func_0086DE20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008422D4;
loc_008426C4:
    ctx->lr = 0x008426C8; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00842704;
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008426EC; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842668;
loc_008426F4:
    ctx->gpr[9] = ctx->gpr[1] + ctx->gpr[10];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[9] + 0x70, ctx->gpr[0]);
    goto loc_00842668;
loc_00842704:
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6BF4);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6BF0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084271C; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842668;
loc_00842724:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read16(ctx->gpr[3] + 0x28);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008422D4;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842744; june_evt_func_00842034(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_008422D4;
loc_00842748:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00842754; june_evt_func_00842034(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0084247C;
loc_00842758:
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00842668;
    goto loc_00842660;
}

void june_evt_func_00841F3C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x490, ctx->gpr[1]); ctx->gpr[1] += -0x490;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x488, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x4A0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x10);
    vm_write64(ctx->gpr[1] + 0x480, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x478, ctx->gpr[29]);
    ctx->lr = 0x00841F60; june_evt_func_00853B10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AD);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842020;
loc_00841F74:
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(0x18);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00841F84; june_evt_func_00878FA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x00841F8C; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00841FFC;
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x00841FAC; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6BF0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6BF4);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00841FCC; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00841FDC; june_evt_func_008791AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x4A0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x490);
    return;
loc_00841FFC:
    ctx->lr = 0x00842000; june_evt_func_008791AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x4A0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x490);
    return;
loc_00842020:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BF8);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6C7C);
    ctx->lr = 0x0084202C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00841F74;
}

void june_evt_func_00842034(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x4C0, ctx->gpr[1]); ctx->gpr[1] += -0x4C0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x498, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x4D0, ctx->gpr[0]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x8);
    vm_write64(ctx->gpr[1] + 0x488, ctx->gpr[25]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x490, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x4A0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x4A8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x4B0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x4B8, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842158;
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084221C;
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[29] = vm_read32(ctx->gpr[4] + 0x14);
    ctx->lr = 0x00842080; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x6BF0);
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x6BF4);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008420A8; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x008420B0; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842184;
    ctx->lr = 0x008420C4; june_evt_func_008F6F00(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x008420E0; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[27] + (int64_t)(0x18);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008420F4; june_evt_func_008791AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00842100; june_evt_func_00882388(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t a = (uint32_t)ctx->gpr[3]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008421AC;
loc_00842110:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = (int64_t)(int32_t)(0x2D);
    ctx->gpr[0] = (ctx->gpr[3] & 64) ? 0 : (ctx->gpr[0] << (ctx->gpr[3] & 63));
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008421AC;
    ;
loc_00842130:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00842140; june_evt_func_00867480(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00842158;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00842154; june_evt_func_00882BF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00842158:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x4D0);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x4C0);
    return;
loc_00842184:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00842190; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0084219C; june_evt_func_00882388(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t a = (uint32_t)ctx->gpr[3]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00842110;
loc_008421AC:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x008421C4; june_evt_func_008258D4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008421D4; june_evt_func_008259E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008421E4; june_evt_func_008259A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842204; june_evt_func_00883708(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00842214; june_evt_func_00825954(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00842130;
loc_0084221C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00842228; june_evt_func_00867480(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x4D0);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x4C0);
    return;
}

void june_evt_func_00853B10(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->lr = 0x00853B24; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    return;
}

void june_evt_func_00853B3C(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->lr = 0x00853B50; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_00867480(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x8);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086758C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867584;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x008674BC; june_evt_func_008F369C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086757C;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x8);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[0]);
loc_008674F0:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867514;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00867508; june_evt_func_00882C2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
loc_00867514:
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867530;
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(0x30);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[30]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0086752C; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00867530:
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00867554;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x18, ctx->gpr[0]);
loc_00867554:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = _cs_30;
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[0]);
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = ctx->gpr[0];
    return;
loc_0086757C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_008674F0;
loc_00867584:
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    goto loc_008674F0;
loc_0086758C:
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[0] = ctx->gpr[10] | 0x20;
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_00882388(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->lr = 0x008823A4; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008823C8;
    ctx->gpr[7] = ctx->gpr[31] + ctx->gpr[3];
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823E4;
loc_008823C8:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
loc_008823CC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 61);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = ctx->gpr[0];
    return;
loc_008823E4:
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-3);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x5EF0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[8] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[8], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_008824E8;
    ctx->gpr[0] = ctx->gpr[8] | ctx->gpr[8];
loc_0088241C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x64; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008824F0;
loc_00882424:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_00882568;
loc_00882428:
    ctx->gpr[0] = ctx->gpr[8] | ctx->gpr[8];
loc_0088242C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x67; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882570;
loc_00882434:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_008825E0;
loc_00882438:
    ctx->gpr[0] = ctx->gpr[8] | ctx->gpr[8];
loc_0088243C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x65; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008825E8;
loc_00882444:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_00882658;
loc_00882448:
    ctx->gpr[0] = ctx->gpr[8] | ctx->gpr[8];
loc_0088244C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x65; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882660;
loc_00882454:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_008826D0;
loc_00882458:
    ctx->gpr[0] = ctx->gpr[8] | ctx->gpr[8];
loc_0088245C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x67; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008826D8;
loc_00882464:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_00882748;
loc_00882468:
    ctx->gpr[0] = ctx->gpr[8] | ctx->gpr[8];
loc_0088246C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x65; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008823C8;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-2);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008824A4;
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
loc_008824A4:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x70; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008823C8;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[3], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882798;
loc_008824D8:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0x64; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(6);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823CC;
    goto loc_008823C8;
loc_008824E8:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x20);
    goto loc_0088241C;
loc_008824F0:
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-2);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882750;
loc_0088251C:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x64; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882424;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882554;
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
loc_00882554:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x73; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823CC;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00882428;
    ;
loc_00882568:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x20);
    goto loc_0088242C;
loc_00882570:
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-2);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882758;
loc_0088259C:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x66; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882434;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882770;
loc_008825D0:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x73; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823CC;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00882438;
loc_008825E0:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x20);
    goto loc_0088243C;
loc_008825E8:
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-2);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882760;
loc_00882614:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x70; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882444;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882780;
loc_00882648:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823CC;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00882448;
loc_00882658:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x20);
    goto loc_0088244C;
loc_00882660:
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-2);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882768;
loc_0088268C:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x70; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882454;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882788;
loc_008826C0:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(4);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823CC;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00882458;
loc_008826D0:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x20);
    goto loc_0088245C;
loc_008826D8:
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-2);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882778;
loc_00882704:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x6D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882464;
    ctx->gpr[9] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00882790;
loc_00882738:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0x64; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(5);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008823CC;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00882468;
loc_00882748:
    ctx->gpr[0] = ctx->gpr[8] + (int64_t)(0x20);
    goto loc_0088246C;
loc_00882750:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_0088251C;
loc_00882758:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_0088259C;
loc_00882760:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_00882614;
loc_00882768:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_0088268C;
loc_00882770:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_008825D0;
loc_00882778:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_00882704;
loc_00882780:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_00882648;
loc_00882788:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_008826C0;
loc_00882790:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x20);
    goto loc_00882738;
loc_00882798:
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x20);
    goto loc_008824D8;
}

void june_evt_func_00882BF4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x00882C0C; june_evt_func_0087694C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00882ABC; return; }
}

void june_evt_func_00883708(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[5] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[4] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[6] = ctx->gpr[8] | ctx->gpr[8];
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->lr = 0x0088374C; june_evt_func_00882C64(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 4, 0, 27, 0, 0);
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x5EF4);
    ctx->gpr[29] = ppc_rlw64(ctx->gpr[29], 2, 0, 29, 0, 0);
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[29] + ctx->gpr[26];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008837F0;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0x8000 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x91);
    vm_write32(ctx->gpr[11] + 0x10, ctx->gpr[9]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088381C;
loc_008837A4:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 4, 0, 27, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 29, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[26];
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[11] + 0x4);
    ctx->lr = 0x008837DC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 1, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[3]);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[0]);
loc_008837F0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_0088381C:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5F08);
    ctx->lr = 0x00883824; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    ctx->lr = 0x00883840; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[28]);
    goto loc_008837A4;
}

void june_evt_func_00882C64(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { uint64_t a = (uint32_t)ctx->gpr[3]; uint64_t b = (uint64_t)0x6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[4] = (int64_t)(int32_t)(0x66);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00882DCC;
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5F08);
loc_00882CB0:
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[6] = (int64_t)(int32_t)(0x68);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x3C);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00882CCC; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x3C);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x00882CE8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[27] + (int64_t)(0x1C);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[30]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[28]);
    vm_write32(ctx->gpr[27] + 0x14, ctx->gpr[25]);
    vm_write32(ctx->gpr[27] + 0x18, ctx->gpr[24]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[28]);
    vm_write32(ctx->gpr[9] + 0x8, ctx->gpr[28]);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[28]);
    ctx->lr = 0x00882D1C; june_evt_func_008791AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x5C);
    ctx->gpr[5] = (int64_t)(int32_t)(0x2F);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00882D30; june_evt_func_00878E08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00882D3C; june_evt_func_00878F3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[27] + (int64_t)(0x30);
    vm_write32(ctx->gpr[27] + 0x2C, ctx->gpr[3]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x00882D50; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x5EEC);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(-148);
    ctx->lr = 0x00882D64; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[27] + 0x34, ctx->gpr[28]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x90);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882DE4;
    vm_write32(ctx->gpr[27] + 0x38, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x90);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[27]);
loc_00882D84:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x5F14);
    ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x94);
    vm_write32(ctx->gpr[9] + 0x90, ctx->gpr[31]);
    ctx->lr = 0x00882D94; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_00882DCC:
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5F08);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5F00);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00882DDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00882CB0;
loc_00882DE4:
    vm_write32(ctx->gpr[27] + 0x38, ctx->gpr[0]);
    goto loc_00882D84;
}

void june_evt_func_00878E08(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[4];
    goto loc_00878E44;
loc_00878E30:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00878E68;
loc_00878E44:
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00878E30;
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[5]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00878E44;
loc_00878E68:
    return;
}

void june_evt_func_00878F3C(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00878F74;
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x8);
loc_00878F60:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
loc_00878F74:
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00878F60;
    ctx->lr = 0x00878F88; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
    return;
}
