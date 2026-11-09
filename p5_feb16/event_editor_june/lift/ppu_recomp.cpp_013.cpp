
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

void june_evt_func_00853B00(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[4]);
        return;
}

void june_evt_func_00853D38(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x67E4);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[3] = (int64_t)(int32_t)(0x30);
        ctx->gpr[6] = (int64_t)(int32_t)(0x2A);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
        ctx->lr = 0x00853D78; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[28] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853F74;
loc_00853D8C:
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0x30);
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->lr = 0x00853D9C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x4);
        ctx->gpr[11] = vm_read16(ctx->gpr[30] + 0x0);
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[10]);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x20);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
        vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x28);
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[11]);
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853F4C;
        { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00853F14;
loc_00853DD8:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853F40;
loc_00853DE4:
        ctx->gpr[29] = ctx->gpr[28] + (int64_t)(8);
        ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x74);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x00853DF8; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->lr = 0x00853E0C; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x00853E24; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x00853E30; june_evt_func_0085D6AC(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = (int64_t)(int32_t)(0x40);
        vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00853E44; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00853EAC;
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x13);
        vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
        vm_write16(ctx->gpr[1] + 0x88, ctx->gpr[9]);
        ctx->lr = 0x00853E70; june_evt_func_008673F0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00853E7C; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
        ctx->gpr[28] = _cs_28;
        vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[3]);
        ctx->lr = ctx->gpr[0];
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[27] = _cs_27;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
        return;
loc_00853EAC:
        ctx->lr = 0x00853EB0; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[10] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x67D4);
        ctx->gpr[0] = (int64_t)(int32_t)(2);
        ctx->gpr[9] = (int64_t)(int32_t)(1);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
        vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[10]);
        vm_write16(ctx->gpr[1] + 0x78, ctx->gpr[9]);
        vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[8]);
        ctx->lr = 0x00853EE4; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
        vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[3]);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[27] = _cs_27;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
        return;
loc_00853F14:
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853F60;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x67DC);
        ctx->gpr[4] = (int64_t)(int32_t)(0x41);
        ctx->lr = 0x00853F2C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00853DE4;
        ;
loc_00853F40:
        ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x67D8);
        vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[0]);
        goto loc_00853DE4;
loc_00853F4C:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x18);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x1C);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[9]);
        goto loc_00853DD8;
loc_00853F60:
        { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x8, tmp); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
        goto loc_00853DD8;
loc_00853F74:
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x67E0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x2B);
        ctx->lr = 0x00853F84; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_00853D8C;
}

void june_evt_func_0085B920(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[4]);
        return;
}

void june_evt_func_0085BBD8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x66C4);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[6] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[3] = (int64_t)(int32_t)(0x14);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->lr = 0x0085BC10; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[4] = (int64_t)(int32_t)(0x2C);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BC64;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(0x14);
        ctx->lr = 0x0085BC38; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
        ctx->gpr[30] = _cs_30;
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_0085BC64:
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66B8);
        ctx->lr = 0x0085BC6C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(0x14);
        ctx->lr = 0x0085BC7C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
        ctx->gpr[30] = _cs_30;
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
}

void june_evt_func_008632E0(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x24, ctx->gpr[4]);
        return;
}

void june_evt_func_00866300(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x24, ctx->gpr[4]);
        return;
}

void june_evt_func_00866AA8(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x40, ctx->gpr[4]);
        return;
}

void june_evt_func_0086C738(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[4]);
        return;
}

void june_evt_func_0086D0F0(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
        ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x62AC);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[3] = (int64_t)(int32_t)(0x1C);
        ctx->gpr[6] = (int64_t)(int32_t)(0x38);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
        ctx->lr = 0x0086D130; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[28] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D278;
loc_0086D144:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0x1C);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x0086D154; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        vm_write16(ctx->gpr[30] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x14);
        vm_write8(ctx->gpr[30] + 0x8, ctx->gpr[9]);
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086D1E4;
loc_0086D178:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x1C);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[0]);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x20);
        vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[9]);
        vm_write32(ctx->gpr[30] + 0x18, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D1BC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D1FC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086D230;
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62A8);
        ctx->gpr[4] = (int64_t)(int32_t)(0x53);
        ctx->lr = 0x0086D1B8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_0086D1BC:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
        return;
loc_0086D1E4:
        ctx->gpr[3] = ctx->gpr[28] + (int64_t)(4);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x0086D1F0; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[3]);
        goto loc_0086D178;
loc_0086D1FC:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
        vm_write8(ctx->gpr[30] + 0x8, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
        return;
loc_0086D230:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[0] = ctx->gpr[3] & ctx->gpr[0];
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[11] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[11] - ctx->gpr[9];
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        vm_write8(ctx->gpr[30] + 0x8, ctx->gpr[9]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
        return;
loc_0086D278:
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62A4);
        ctx->gpr[4] = (int64_t)(int32_t)(0x39);
        ctx->lr = 0x0086D288; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0086D144;
}

void june_evt_func_0086D290(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x10, ctx->gpr[4]);
        return;
}

void june_evt_func_0093A8E0(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x48, ctx->gpr[4]);
        return;
}

void june_evt_func_0093B73C(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[28]);
        ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[31]);
        ctx->lr = 0x0093B770; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093BAA0;
        ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x2E98);
loc_0093B788:
        ctx->gpr[4] = (int64_t)(int32_t)(0xB);
        ctx->gpr[3] = (int64_t)(int32_t)(0x4C);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[6] = (int64_t)(int32_t)(0x32);
        ctx->lr = 0x0093B7A0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093BA88;
loc_0093B7B0:
        ctx->gpr[25] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0x4C);
        ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
        ctx->lr = 0x0093B7C0; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093BA28;
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
        ctx->gpr[31] = (int64_t)(int32_t)(0);
loc_0093B7E0:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B7E0;
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093BA28;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 4, 0, 27, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 2, 0, 29, 0, 0);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB);
        ctx->gpr[27] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[3] = ctx->gpr[27] + (int64_t)(0x14);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[6] = (int64_t)(int32_t)(0x3D);
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 0, 0, 29, 0, 0);
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        ctx->lr = 0x0093B82C; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(0);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[5] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->gpr[12] = (int64_t)(int32_t)(1);
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[3]);
loc_0093B850:
        ctx->gpr[9] = ctx->gpr[6] + ctx->gpr[5];
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[4];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(0x14);
        ctx->gpr[7] = ctx->gpr[7] + (int64_t)(0xC);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[12]);
        ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x8);
        ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x0);
        ctx->gpr[8] = vm_read32(ctx->gpr[11] + 0x4);
        vm_write32(ctx->gpr[9] + 0x10, ctx->gpr[0]);
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[10]);
        vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[8]);
        vm_write16(ctx->gpr[9] + 0x8, ctx->gpr[29]);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_0093B850;
        ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
loc_0093B890:
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[5];
        ctx->gpr[29] = ctx->gpr[30] + (int64_t)(0x14);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0xFF << 16);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[0] = ctx->gpr[0] | 0xFFFF;
        ctx->gpr[27] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
        vm_write32(ctx->gpr[9] + 0x10, ctx->gpr[27]);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[27]);
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[27]);
        vm_write16(ctx->gpr[9] + 0x8, ctx->gpr[27]);
        ctx->lr = 0x0093B8C8; june_evt_func_00861E0C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x38, tmp); }
        ctx->lr = 0x0093B8D8; june_evt_func_00861D50(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x3C, tmp); }
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093BA6C;
loc_0093B8EC:
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
        ctx->gpr[4] = (int64_t)(int32_t)(0x10);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[31]);
        vm_write32(ctx->gpr[30] + 0x30, ctx->gpr[0]);
        ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xC);
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x10);
        vm_write32(ctx->gpr[30] + 0x40, ctx->gpr[9]);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[0]);
        ctx->lr = 0x0093B920; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x0);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)(2);
        vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B954;
        ctx->gpr[0] = (int64_t)(int32_t)(3);
        vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
loc_0093B954:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x2C);
        ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 29, 0, 0);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
        ctx->lr = 0x0093B988; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[3]);
        ctx->gpr[4] = (int64_t)(int32_t)(0x40);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x0093B99C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        vm_write16(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->lr = 0x0093B9AC; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EA0);
        vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        vm_write16(ctx->gpr[1] + 0x94, ctx->gpr[27]);
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
        { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
        ctx->lr = 0x0093B9E4; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
        vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[3]);
        vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
        ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
        ctx->gpr[24] = _cs_24;
        ctx->gpr[25] = _cs_25;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
        return;
loc_0093BA28:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E64);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3B);
        ctx->gpr[31] = (int64_t)(int32_t)(0);
        ctx->lr = 0x0093BA3C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = (int64_t)(int32_t)(0xB);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[3] = (int64_t)(int32_t)(0x14);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[6] = (int64_t)(int32_t)(0x3D);
        ctx->lr = 0x0093BA58; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = (int64_t)(int32_t)(0);
        ctx->gpr[5] = ctx->gpr[3] | ctx->gpr[3];
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[3]);
        goto loc_0093B890;
loc_0093BA6C:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E60);
        ctx->gpr[4] = (int64_t)(int32_t)(0x50);
        ctx->lr = 0x0093BA7C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
        goto loc_0093B8EC;
loc_0093BA88:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E7C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x33);
        ctx->lr = 0x0093BA98; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0093B7B0;
loc_0093BAA0:
        ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x2E98);
        ctx->gpr[4] = (int64_t)(int32_t)(0x30);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E68);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x0093BAB4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0093B788;
        ;
         return;
}
