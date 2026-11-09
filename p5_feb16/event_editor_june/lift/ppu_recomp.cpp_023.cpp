
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

void june_evt_func_00882ABC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[3] + (int64_t)(0x30);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882BB0;
loc_00882AF8:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00882B00; june_evt_func_00834F80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882B2C;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00882B2C:
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x5EEC);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00882B38; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882B54;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_00882B54:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882BEC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[9] + 0x38, ctx->gpr[11]);
loc_00882B68:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00882B70; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882BC8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x5EDC);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->lr = 0x00882B8C; june_evt_func_00876CB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00882BB0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5EE4);
    ctx->gpr[4] = (int64_t)(int32_t)(0x22);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5EE0);
    ctx->lr = 0x00882BC0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00882AF8;
loc_00882BC8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00882990; return; }
loc_00882BEC:
    vm_write32(ctx->gpr[30] + -0x4, ctx->gpr[11]);
    goto loc_00882B68;
}

void june_evt_func_00D2BAE8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x150, ctx->gpr[1]); ctx->gpr[1] += -0x150;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x180);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[25]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    vm_write32(ctx->gpr[1] + 0x180, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[29] = ctx->gpr[0] | ctx->gpr[0];
    ctx->lr = 0x00D2BB3C; june_evt_func_00D2CCC4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BB54;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
loc_00D2BB54:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 28, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BC24;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BD54;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BC04;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BC14;
loc_00D2BB7C:
    { uint64_t tmp = vm_read64(ctx->gpr[2] + 0x5128); memcpy(&ctx->fpr[0], &tmp, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
loc_00D2BB84:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BBCC;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->fpr[1] = -ctx->fpr[0];
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_00D2BBCC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[1], &tmp, 8); }
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_00D2BC04:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x5134);
    { uint64_t tmp = vm_read64(ctx->gpr[9] + 0x0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00D2BB84;
loc_00D2BC14:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x5138);
    { uint64_t tmp = vm_read64(ctx->gpr[9] + 0x0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00D2BB84;
loc_00D2BC24:
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x180);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->lr = 0x00D2BC40; june_evt_func_00D2C230(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BB7C;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x7C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BB7C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0xEE6B << 16);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2800;
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[1] = (double)iv; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x401C);
    ctx->gpr[9] = ppc_rldicr(ctx->gpr[9], 32, 31);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x88);
    vm_write32(ctx->gpr[1] + 0xB8, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[9]);
    ctx->lr = 0x00D2BC8C; june_evt_func_00D389F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2BD28;
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(4);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0xD0);
loc_00D2BCAC:
    ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x88);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(4);
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0xB8);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->lr = 0x00D2BCCC; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[1] = (double)iv; }
    ctx->lr = 0x00D2BCF0; june_evt_func_00D389F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0xA8);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x98);
    ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->lr = 0x00D2BD0C; june_evt_func_00D38CF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xD0);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xD8);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2BCAC;
loc_00D2BD28:
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x78);
    ctx->gpr[4] = ctx->gpr[26] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2BE48;
loc_00D2BD38:
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x88);
    ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->lr = 0x00D2BD48; june_evt_func_00D38FA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00D2BB84;
loc_00D2BD54:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x180);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->lr = 0x00D2BD6C; june_evt_func_00D2D600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2BDB4;
    { uint64_t tmp = vm_read64(ctx->gpr[2] + 0x5128); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x70);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
loc_00D2BD84:
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x78);
    ctx->gpr[3] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->lr = 0x00D2BD94; june_evt_func_00D29898(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[26];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[1], &tmp, 8); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->lr = 0x00D2BDA8; june_evt_func_00D386E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00D2BB84;
loc_00D2BDB4:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[3], 2, 0, 29, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x5130);
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-1);
    ctx->gpr[9] = ctx->gpr[8] + ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[12] = (double)iv; }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D2BE74;
loc_00D2BDF0:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(-1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(8);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) == 0)) goto loc_00D2BE40;
loc_00D2BE00:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[3], 2, 0, 29, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[8] + ctx->gpr[9];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[0];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2BDF0;
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(-1);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[13] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[10] + 0x0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(8);
    ctx->fpr[12] = ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[12], 0, 0);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D2BE00;
loc_00D2BE40:
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write64(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00D2BD84;
loc_00D2BE48:
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xE0);
    ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x88);
    ctx->lr = 0x00D2BE5C; june_evt_func_00D38D90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xE0);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xE8);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    goto loc_00D2BD38;
loc_00D2BE74:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[0];
    goto loc_00D2BDF0;
}

void june_evt_func_00D2CF54(ppu_context* ctx) {
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
    ctx->gpr[12] = ctx->cr;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
    ctx->gpr[22] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[21] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[16] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[18] = ctx->gpr[4] | ctx->gpr[4];
    vm_write32(ctx->gpr[1] + 0xF8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D2CFCC;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[6] + 0x0, ctx->gpr[0]);
loc_00D2CFCC:
    ctx->gpr[17] = vm_read32(ctx->gpr[2] + 0x5178);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[28], 1, 23, 30, 0, 0);
    ctx->gpr[10] = vm_read32(ctx->gpr[17] + 0x0);
    goto loc_00D2CFF4;
loc_00D2CFE4:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[28] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[28], 1, 23, 30, 0, 0);
loc_00D2CFF4:
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[10];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0x144);
    ctx->gpr[9] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CFE4;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D1DC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D1DC;
    ctx->gpr[28] = (int64_t)(int32_t)(0x2B);
loc_00D2D028:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 7) & 1) == 1)) goto loc_00D2D040;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D040;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0x24; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2D0B0;
loc_00D2D040:
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D054;
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[16]);
loc_00D2D054:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xF8);
    ctx->gpr[16] = _cs_16;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[17] = _cs_17;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_00D2D0B0:
    if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D2D234;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D1E4;
loc_00D2D0BC:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[20] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[0] = vm_read8(ctx->gpr[4] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D100;
loc_00D2D0D8:
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
loc_00D2D0DC:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D0DC;
    ctx->gpr[20] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[31] = ctx->gpr[29] | ctx->gpr[29];
loc_00D2D100:
    ctx->gpr[23] = vm_read32(ctx->gpr[2] + 0x517C);
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    goto loc_00D2D134;
loc_00D2D118:
    ctx->gpr[26] = ctx->gpr[3] - ctx->gpr[30];
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->gpr[19] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[25] * (int64_t)(int32_t)ctx->gpr[27]);
    ctx->gpr[25] = ctx->gpr[0] + ctx->gpr[9];
loc_00D2D134:
    ctx->gpr[4] = vm_read8(ctx->gpr[4] + 0x0);
    ctx->gpr[30] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[0] = vm_read32(ctx->gpr[17] + 0x0);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[4], 1, 0, 30, 0, 0);
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D168;
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x20);
loc_00D2D168:
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->lr = 0x00D2D170; june_evt_func_00D251DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D118;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[20]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D040;
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[27];
    ctx->gpr[11] = ctx->gpr[31] - ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D2D220;
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2D204;
loc_00D2D1A8:
    ctx->lr = 0x00D2D1AC; june_evt_func_00D23CA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x8001 << 16);
    ctx->gpr[0] = ctx->gpr[0] | 0x1C;
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D2D294;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[25] = (int64_t)(int32_t)(-1);
    vm_write32(ctx->gpr[22] + 0x0, ctx->gpr[0]);
loc_00D2D1CC:
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D054;
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[31]);
    goto loc_00D2D054;
loc_00D2D1DC:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    goto loc_00D2D028;
loc_00D2D1E4:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->gpr[0] = vm_read8(ctx->gpr[4] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D274;
loc_00D2D1F8:
    ctx->gpr[29] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[20] = ctx->gpr[31] | ctx->gpr[31];
    goto loc_00D2D100;
loc_00D2D204:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->gpr[0] = ctx->gpr[25] - ctx->gpr[0];
    { uint64_t a = (uint32_t)ctx->gpr[25]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D2D1A8;
    { uint32_t _b=(uint32_t)ctx->gpr[27]; ctx->gpr[0] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[0]/_b); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[19]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D1A8;
loc_00D2D220:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D1CC;
    ctx->gpr[25] = (uint64_t)0 - ctx->gpr[25];
    goto loc_00D2D1CC;
loc_00D2D234:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[27] = (int64_t)(int32_t)(0xA);
    ctx->gpr[5] = (int64_t)(int32_t)(0xA);
    ctx->gpr[0] = vm_read8(ctx->gpr[4] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D1F8;
    ctx->gpr[0] = vm_read8(ctx->gpr[4] + 0x1);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D268;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = (int64_t)(int32_t)(8);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D0D8;
loc_00D2D268:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(2);
    ctx->gpr[27] = (int64_t)(int32_t)(0x10);
    goto loc_00D2D0BC;
loc_00D2D274:
    ctx->gpr[0] = vm_read8(ctx->gpr[4] + 0x1);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D28C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D0D8;
loc_00D2D28C:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(2);
    goto loc_00D2D0BC;
loc_00D2D294:
    ctx->gpr[25] = (int64_t)(int32_t)(-1);
    goto loc_00D2D1CC;
}

void june_evt_func_00882990(ppu_context* ctx) {
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
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x10);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x18E);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00882A84;
    ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x161);
    { uint64_t a = (uint32_t)ctx->gpr[31]; uint64_t b = (uint64_t)0x6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00882AA8;
loc_008829E0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 4, 0, 27, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x5EF4);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 2, 0, 29, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882A24;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00882A20; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_00882A24:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00882A40;
    ctx->lr = 0x00882A3C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00882A40:
    ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x1C);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[27]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00882A50; june_evt_func_00878FA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00882A5C; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_00882A84:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5EE8);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5F08);
    ctx->lr = 0x00882A90; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[31]; uint64_t b = (uint64_t)0x6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = (int64_t)(int32_t)(0x161);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008829E0;
    ;
loc_00882AA8:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5F00);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5F08);
    ctx->lr = 0x00882AB4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008829E0;
}

void june_evt_func_00D2C230(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 3, 0, 28, 0, 0);
    vm_write64(ctx->gpr[1] + -0x8, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[0] + ctx->gpr[7];
    vm_write64(ctx->gpr[1] + -0x30, ctx->gpr[26]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + -0x40, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + -0x38, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + -0x18, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + -0x28, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + -0x20, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[30]);
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[25] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2C278;
    ctx->gpr[31] = (int64_t)(int32_t)(0x2D);
loc_00D2C278:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[5] + 0x4, ctx->gpr[9]);
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[9]);
    ctx->gpr[8] = vm_read8(ctx->gpr[4] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C2A4;
loc_00D2C294:
    ctx->gpr[8] = vm_read8(ctx->gpr[4] + 0x1); ctx->gpr[4] += 0x1;
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C294;
    ctx->gpr[6] = (int64_t)(int32_t)(1);
loc_00D2C2A4:
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x514C);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[8], 1, 23, 30, 0, 0);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[30] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C300;
    goto loc_00D2C344;
loc_00D2C2D4:
    vm_write8(ctx->gpr[9] + 0x30, ctx->gpr[10]);
    ctx->gpr[8] = vm_read8(ctx->gpr[4] + 0x1); ctx->gpr[4] += 0x1;
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[8], 1, 23, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C340;
loc_00D2C300:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(-160);
    ctx->gpr[10] = ctx->gpr[8] + (int64_t)(-48);
    ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[0];
    if ((((ctx->cr >> 7) & 1) == 0)) goto loc_00D2C2D4;
    ctx->gpr[9] = vm_read32(ctx->gpr[5] + 0x0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[9]);
    ctx->gpr[8] = vm_read8(ctx->gpr[4] + 0x1); ctx->gpr[4] += 0x1;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[8], 1, 23, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C300;
loc_00D2C340:
    ctx->gpr[6] = (int64_t)(int32_t)(1);
loc_00D2C344:
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x5150);
    ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[8];
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x2C);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C70C;
loc_00D2C360:
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C390;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C390;
    ctx->gpr[9] = vm_read32(ctx->gpr[5] + 0x0);
loc_00D2C378:
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[9]);
    ctx->gpr[8] = vm_read8(ctx->gpr[4] + 0x1); ctx->gpr[4] += 0x1;
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C378;
    ctx->gpr[6] = (int64_t)(int32_t)(1);
loc_00D2C390:
    ctx->gpr[10] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 23, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C400;
loc_00D2C3B0:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(-160);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(-48);
    ctx->gpr[11] = ctx->gpr[9] + ctx->gpr[0];
    if ((((ctx->cr >> 7) & 1) == 1)) goto loc_00D2C3DC;
    ctx->gpr[9] = vm_read32(ctx->gpr[5] + 0x0);
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    vm_write8(ctx->gpr[11] + 0x30, ctx->gpr[10]);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[9]);
loc_00D2C3DC:
    ctx->gpr[10] = vm_read8(ctx->gpr[4] + 0x1); ctx->gpr[4] += 0x1;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 23, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C3B0;
    ctx->gpr[6] = (int64_t)(int32_t)(1);
loc_00D2C400:
    if ((((ctx->cr >> 7) & 1) == 0)) goto loc_00D2C430;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(-160);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[11] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x30);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D2C6E0;
loc_00D2C420:
    ctx->gpr[9] = vm_read32(ctx->gpr[5] + 0x0);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[9]);
loc_00D2C430:
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2C724;
    ctx->gpr[8] = ctx->gpr[7] + (int64_t)(-1);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(-160);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[8];
    ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[0];
    ctx->gpr[11] = vm_read8(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C49C;
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[5] + 0x0);
    ctx->ctr = ctx->gpr[7];
    goto loc_00D2C47C;
loc_00D2C464:
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(-160);
    ctx->gpr[9] = ctx->gpr[7] + ctx->gpr[0];
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C700;
    ctx->gpr[8] = ctx->gpr[11] | ctx->gpr[11];
loc_00D2C47C:
    ctx->gpr[11] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[11];
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D2C464;
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[10]);
loc_00D2C490:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[1] + -0x70, ctx->gpr[0]);
loc_00D2C49C:
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
    if ((((ctx->cr >> 25) & 1) == 1)) goto loc_00D2C69C;
    ctx->gpr[6] = (int64_t)(int32_t)((uint32_t)0x38E3 << 16);
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[7], 0x1F);
    ctx->gpr[6] = ctx->gpr[6] | 0x8E39;
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[7] * (int64_t)(int32_t)ctx->gpr[6] >> 32));
    ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 1);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[11];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 3, 0, 28, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[0] = ctx->gpr[7] - ctx->gpr[0];
    { uint64_t _a = ctx->gpr[0]; uint64_t _b = (uint64_t)(int64_t)(9); ctx->gpr[10] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
    ctx->gpr[8] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[10], 0x1F);
    ctx->gpr[9] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[10] * (int64_t)(int32_t)ctx->gpr[6] >> 32));
    ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 1);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[8];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 3, 0, 28, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[0] = ctx->gpr[10] - ctx->gpr[0];
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
    ctx->gpr[9] = ctx->gpr[11] ^ ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[11] - ctx->gpr[9];
    ctx->gpr[28] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2C5A4;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[28], 2, 0, 29, 0, 0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[12] = ctx->gpr[0] + ctx->gpr[29];
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    ctx->ctr = ctx->gpr[7];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(-112);
    ctx->gpr[31] = ctx->gpr[12] | ctx->gpr[12];
    goto loc_00D2C550;
loc_00D2C524:
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(4);
    ctx->gpr[0] = vm_read8(ctx->gpr[7] + 0x0);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[12] = ctx->gpr[12] + (int64_t)(4);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    ctx->gpr[6] = ctx->gpr[6] + (int64_t)(1);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) == 0)) goto loc_00D2C5A4;
loc_00D2C54C:
    ctx->gpr[8] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[10], 0x1F);
loc_00D2C550:
    ctx->gpr[0] = ctx->gpr[6] + ctx->gpr[29];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[12], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[10] * (int64_t)(int32_t)ctx->gpr[27] >> 32));
    ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 1);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[8];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 3, 0, 28, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C524;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    ctx->gpr[9] = vm_read8(ctx->gpr[7] + 0x0);
    ctx->gpr[6] = ctx->gpr[6] + (int64_t)(1);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[9];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D2C54C;
loc_00D2C5A4:
    ctx->gpr[0] = vm_read8(ctx->gpr[4] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x65; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C5BC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x45; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C6A0;
loc_00D2C5BC:
    ctx->gpr[10] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[9] = vm_read8(ctx->gpr[10] + 0x1); ctx->gpr[10] += 0x1;
    ctx->gpr[7] = (int64_t)(int8_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0x2B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C714;
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C714;
    ctx->gpr[7] = (int64_t)(int32_t)(0x2B);
loc_00D2C5DC:
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C678;
loc_00D2C604:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 3, 0, 28, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[8], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x5F5 << 16);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[11] = vm_read8(ctx->gpr[10] + 0x1); ctx->gpr[10] += 0x1;
    ctx->gpr[0] = ctx->gpr[0] | 0xE0FF;
    ctx->gpr[8] = ctx->gpr[9] + (int64_t)(-48);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C674;
loc_00D2C644:
    if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D2C604;
    ctx->gpr[11] = vm_read8(ctx->gpr[10] + 0x1); ctx->gpr[10] += 0x1;
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x5F5 << 16);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] | 0xE0FF;
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C644;
loc_00D2C674:
    ctx->gpr[6] = (int64_t)(int32_t)(1);
loc_00D2C678:
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C71C;
loc_00D2C680:
    ctx->gpr[0] = vm_read32(ctx->gpr[5] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[0];
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C6A0;
    ctx->gpr[4] = ctx->gpr[10] | ctx->gpr[10];
    goto loc_00D2C6A0;
loc_00D2C69C:
    ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_00D2C6A0:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2C6B0;
    if ((((ctx->cr >> 25) & 1) == 1)) goto loc_00D2C6D8;
loc_00D2C6AC:
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[4]);
loc_00D2C6B0:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    return;
loc_00D2C6D8:
    ctx->gpr[4] = ctx->gpr[24] | ctx->gpr[24];
    goto loc_00D2C6AC;
loc_00D2C6E0:
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(-1);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(-160);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[0];
    ctx->gpr[9] = vm_read8(ctx->gpr[11] + 0x30);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write8(ctx->gpr[11] + 0x30, ctx->gpr[9]);
    goto loc_00D2C420;
loc_00D2C700:
    ctx->gpr[7] = ctx->gpr[8] | ctx->gpr[8];
    vm_write32(ctx->gpr[5] + 0x0, ctx->gpr[10]);
    goto loc_00D2C49C;
loc_00D2C70C:
    ctx->gpr[8] = vm_read8(ctx->gpr[4] + 0x1); ctx->gpr[4] += 0x1;
    goto loc_00D2C360;
loc_00D2C714:
    ctx->gpr[9] = vm_read8(ctx->gpr[10] + 0x1); ctx->gpr[10] += 0x1;
    goto loc_00D2C5DC;
loc_00D2C71C:
    ctx->gpr[8] = (uint64_t)0 - ctx->gpr[8];
    goto loc_00D2C680;
loc_00D2C724:
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2C49C;
    goto loc_00D2C490;
}

void june_evt_func_00D2CCC4(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x5174);
    ctx->gpr[6] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[8] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[7] = vm_read32(ctx->gpr[9] + 0x0);
    goto loc_00D2CCE0;
loc_00D2CCDC:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
loc_00D2CCE0:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read8(ctx->gpr[10] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 1, 23, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[7];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0x144);
    ctx->gpr[9] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CCDC;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CE30;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x2B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CE40;
loc_00D2CD20:
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CDBC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CDBC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x69; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CD60;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x49; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CD60;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CE84;
loc_00D2CD50:
    ctx->gpr[3] = ctx->gpr[3] | 0x1;
loc_00D2CD54:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write32(ctx->gpr[6] + 0x0, ctx->gpr[8]);
    return;
loc_00D2CD60:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CD80;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA0;
loc_00D2CD80:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x66; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CEA8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x46; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CEA8;
loc_00D2CDA0:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
loc_00D2CDA4:
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CD54;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[8]);
    vm_write32(ctx->gpr[6] + 0x0, ctx->gpr[8]);
    return;
loc_00D2CDBC:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x61; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CDDC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x41; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA0;
loc_00D2CDDC:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CDFC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA0;
loc_00D2CDFC:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[10] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[11] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x28; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CE4C;
loc_00D2CE18:
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[3] = (int64_t)(int32_t)(4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x29; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA4;
    ctx->gpr[8] = ctx->gpr[10] + (int64_t)(1);
    goto loc_00D2CDA4;
loc_00D2CE30:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[3] = (int64_t)(int32_t)(8);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    goto loc_00D2CD20;
loc_00D2CE40:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    goto loc_00D2CD20;
loc_00D2CE4C:
    ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 1, 23, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x5F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[7];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0x232);
    ctx->gpr[9] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CE4C;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00D2CE4C;
    goto loc_00D2CE18;
loc_00D2CE84:
    ctx->gpr[0] = vm_read8(ctx->gpr[10] + 0x1);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CE9C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CD50;
loc_00D2CE9C:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(2);
    ctx->gpr[3] = ctx->gpr[3] | 0x2;
    goto loc_00D2CD54;
loc_00D2CEA8:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[3] = ctx->gpr[3] | 0x3;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x69; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CECC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x49; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA4;
loc_00D2CECC:
    ctx->gpr[11] = ctx->gpr[8] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CEEC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA4;
loc_00D2CEEC:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x69; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CF0C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x49; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA4;
loc_00D2CF0C:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CF2C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x54; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA4;
loc_00D2CF2C:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x79; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2CF4C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x59; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2CDA4;
loc_00D2CF4C:
    ctx->gpr[8] = ctx->gpr[11] + (int64_t)(1);
    goto loc_00D2CDA4;
}

void june_evt_func_00D2D600(ppu_context* ctx) {
    PPU_MM_SNAP
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
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[4], 0, 32);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[21] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[19] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0xE8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 3, 0, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[25] = ctx->gpr[0] - ctx->gpr[7];
    ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D694;
loc_00D2D67C:
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D67C;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_00D2D694:
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + 0x5188);
    ctx->gpr[22] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[27] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[22] = ctx->gpr[22] | 0x2493;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ctx->gpr[26] | ctx->gpr[26];
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
    goto loc_00D2D718;
loc_00D2D6BC:
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[31] * (int64_t)(int32_t)ctx->gpr[22] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[31];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 2);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[31] - ctx->gpr[9];
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D6F4;
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
loc_00D2D6F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = vm_read8(ctx->gpr[10] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[9];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
loc_00D2D70C:
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[23] = (int64_t)(int32_t)(1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
loc_00D2D718:
    ctx->gpr[4] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x16);
    ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[4];
    ctx->lr = 0x00D2D72C; june_evt_func_00D251DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[31], 0x1F);
    ctx->gpr[0] = ctx->gpr[3] - ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[10] = ctx->gpr[0] + ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D7A0;
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00D2D6BC;
    ctx->gpr[0] = ctx->gpr[3] - ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[8] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[3] = ctx->gpr[0] + ctx->gpr[29];
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_00D2D70C;
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x18);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[31] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2D70C;
    ctx->gpr[31] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    goto loc_00D2D70C;
loc_00D2D7A0:
    if ((((ctx->cr >> 14) & 1) == 0)) goto loc_00D2D7A8;
    ctx->gpr[31] = ctx->gpr[25] | ctx->gpr[25];
loc_00D2D7A8:
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + 0x518C);
    ctx->gpr[11] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[11];
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x2C);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DB54;
loc_00D2D7C8:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D800;
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D800;
    ctx->gpr[8] = vm_read32(ctx->gpr[24] + 0x0);
loc_00D2D7E0:
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x30; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D7E0;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_00D2D800:
    ctx->gpr[4] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = (int64_t)(int32_t)(0x16);
    ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[4];
    ctx->lr = 0x00D2D814; june_evt_func_00D251DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->gpr[0] = ctx->gpr[3] - ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] | 0x2493;
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[31], 0x1F);
    ctx->gpr[8] = ctx->gpr[0] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D8E8;
loc_00D2D840:
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[23] = (int64_t)(int32_t)(1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    if ((((ctx->cr >> 7) & 1) == 0)) goto loc_00D2D800;
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[31] * (int64_t)(int32_t)ctx->gpr[9] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[31];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 2);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[31] - ctx->gpr[9];
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D884;
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(4);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
loc_00D2D884:
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[0] = vm_read8(ctx->gpr[8] + 0x18);
    ctx->gpr[5] = (int64_t)(int32_t)(0x16);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 4, 0, 27, 0, 0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[8] = ctx->gpr[11] + (int64_t)(-1);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
    ctx->gpr[4] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[4];
    ctx->lr = 0x00D2D8BC; june_evt_func_00D251DC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->gpr[0] = ctx->gpr[3] - ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] | 0x2493;
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[31], 0x1F);
    ctx->gpr[8] = ctx->gpr[0] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[27], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D840;
loc_00D2D8E8:
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[31] * (int64_t)(int32_t)ctx->gpr[9] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[31];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 2);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[31] - ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DA50;
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { uint64_t _a = ctx->gpr[0]; uint64_t _b = (uint64_t)(int64_t)(7); ctx->gpr[9] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 2, 0, 29, 0, 0);
    ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[9];
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[0] = (uint64_t)((ctx->gpr[11] & 0x20) ? 0u : ((uint32_t)ctx->gpr[0] << (ctx->gpr[11] & 0x1F)));
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    ctx->gpr[8] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[8] = ctx->gpr[8] - ctx->gpr[9];
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
loc_00D2D934:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D2D9A8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D9A8;
    ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0x2492 << 16);
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(-7);
    ctx->gpr[9] = ctx->gpr[9] | 0x4925;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (uint32_t)(((uint64_t)(uint32_t)ctx->gpr[0] * (uint64_t)(uint32_t)ctx->gpr[9]) >> 32);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 1, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 30, 2, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->ctr = ctx->gpr[9];
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D2D990;
    goto loc_00D2DB74;
loc_00D2D984:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2D9A8;
loc_00D2D990:
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(-4);
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(7);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(-7);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D2D984;
loc_00D2D9A8:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 2, 0, 29, 0, 0);
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[8]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D9D0;
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x70; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DA60;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x50; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DA60;
loc_00D2D9D0:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2D9E4;
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DA58;
loc_00D2D9E0:
    vm_write32(ctx->gpr[21] + 0x0, ctx->gpr[30]);
loc_00D2D9E4:
    ctx->gpr[3] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xE8);
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[31], 0x1F);
    ctx->gpr[19] = _cs_19;
    ctx->gpr[3] = ctx->gpr[3] | 0x2493;
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
    ctx->gpr[22] = _cs_22;
    ctx->gpr[23] = _cs_23;
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[3] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[31] * (int64_t)(int32_t)ctx->gpr[3] >> 32));
    ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[31];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[3] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[3], 2);
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = ctx->gpr[0];
    return;
loc_00D2DA50:
    ctx->gpr[8] = vm_read32(ctx->gpr[24] + 0x0);
    goto loc_00D2D934;
loc_00D2DA58:
    ctx->gpr[30] = ctx->gpr[20] | ctx->gpr[20];
    goto loc_00D2D9E0;
loc_00D2DA60:
    ctx->gpr[7] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0x2B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DB60;
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DB60;
    ctx->gpr[4] = (int64_t)(int32_t)(0x2B);
loc_00D2DA84:
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x5190);
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[5] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[5];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DB38;
loc_00D2DAB4:
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[6], 3, 0, 28, 0, 0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[6], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x5F5 << 16);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[10];
    ctx->gpr[10] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0xE0FF;
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(-48);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[5];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DB34;
loc_00D2DAFC:
    if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D2DAB4;
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x5F5 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[0] = ctx->gpr[0] | 0xE0FF;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[10] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[5];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2DAFC;
loc_00D2DB34:
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_00D2DB38:
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0x2D; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2DB6C;
loc_00D2DB40:
    ctx->gpr[0] = ctx->gpr[6] + ctx->gpr[8];
    ctx->gpr[20] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[30] = ctx->gpr[7] | ctx->gpr[7];
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    goto loc_00D2D9D0;
loc_00D2DB54:
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[30], 0, 32);
    goto loc_00D2D7C8;
loc_00D2DB60:
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[7], 0, 32);
    goto loc_00D2DA84;
loc_00D2DB6C:
    ctx->gpr[6] = (uint64_t)0 - ctx->gpr[6];
    goto loc_00D2DB40;
loc_00D2DB74:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->ctr = ctx->gpr[0];
    goto loc_00D2D990;
}

void june_evt_func_00D386E8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->fpr[13] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38720;
    { uint64_t tmp = vm_read64(ctx->gpr[2] + 0x52D8); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3873C;
loc_00D38720:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->fpr[1] = ctx->fpr[13];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_00D3873C:
    ctx->lr = 0x00D38740; june_evt_func_00D389F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x70);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x78);
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = 0x00D38758; june_evt_func_00D38D90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x88);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    ctx->lr = 0x00D3877C; june_evt_func_00D38FA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_00D38CF4(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + 0x48, ctx->gpr[6]);
    vm_write64(ctx->gpr[1] + 0x38, ctx->gpr[4]);
    ctx->gpr[6] = vm_read32(ctx->gpr[1] + 0x38);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x48);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x40, ctx->gpr[5]);
    ctx->gpr[0] = ctx->gpr[6] - ctx->gpr[0];
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0x44);
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0x40);
    ctx->gpr[7] = (ctx->gpr[0] & 64) ? 0 : (ctx->gpr[7] >> (ctx->gpr[0] & 63));
    vm_write64(ctx->gpr[1] + 0x50, ctx->gpr[7]);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x54);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x50);
    ctx->gpr[10] = ctx->gpr[9] + ctx->gpr[10];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 32, 32);
    ctx->gpr[11] = (int64_t)(int32_t)(-1);
    ctx->gpr[7] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)(0x40);
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[11], 0, 0);
    ctx->ctr = ctx->gpr[0];
    goto loc_00D38D58;
loc_00D38D50:
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) == 0)) goto loc_00D38D68;
loc_00D38D58:
    ctx->gpr[0] = (ctx->gpr[8] & 64) ? 0 : (ctx->gpr[11] >> (ctx->gpr[8] & 63));
    ctx->gpr[0] = ctx->gpr[0] & ctx->gpr[7];
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38D50;
loc_00D38D68:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint64_t _a = ctx->gpr[8]; uint64_t _b = (uint64_t)(int64_t)(0x20); ctx->gpr[10] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
    ctx->gpr[9] = ctx->gpr[6] - ctx->gpr[8];
    ctx->gpr[11] = (ctx->gpr[10] & 64) ? 0 : (ctx->gpr[11] >> (ctx->gpr[10] & 63));
    ctx->gpr[0] = (ctx->gpr[8] & 64) ? 0 : (ctx->gpr[7] << (ctx->gpr[8] & 63));
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x20);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[11];
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[9]);
    vm_write64(ctx->gpr[3] + 0x8, ctx->gpr[0]);
    return;
}
