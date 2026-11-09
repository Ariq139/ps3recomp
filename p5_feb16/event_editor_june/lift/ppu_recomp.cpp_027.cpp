
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

void june_evt_func_00944E54(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[30]);
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[26]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(0x50);
    ctx->gpr[6] = (int64_t)(int32_t)(0x50);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    ctx->lr = 0x00944E9C; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945208;
loc_00944EB0:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x50);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[29] = ctx->gpr[31] + (int64_t)(4);
    ctx->lr = 0x00944EC4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x00944ECC; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[4] = ctx->gpr[31] + (int64_t)(8);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00944EF8; june_evt_func_00879E0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(8);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00944F10; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00944F3C;
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write8(ctx->gpr[1] + 0x77, ctx->gpr[0]);
loc_00944F3C:
    ctx->gpr[0] = (int64_t)(int32_t)(0x2A);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x00944F4C; june_evt_func_0086335C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00944F68; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[10] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x2C24);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[11]);
    vm_write16(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[10]);
    ctx->lr = 0x00944F94; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C20);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[0]);
    ctx->lr = 0x00944FAC; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C1C);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[0]);
    ctx->lr = 0x00944FC4; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[3]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009451EC;
loc_00944FDC:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[27] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00944FF4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0094500C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[1] + 0xAC, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[9] = (int64_t)(int32_t)(8);
    ctx->gpr[0] = (int64_t)(int32_t)(0x167);
    vm_write32(ctx->gpr[1] + 0xB8, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0x94, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[11] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = (int64_t)(int32_t)(6);
    ctx->gpr[9] = (int64_t)(int32_t)(0xE);
    ctx->gpr[11] = (int64_t)(int16_t)ctx->gpr[11];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[1] + 0x8C, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0xB0, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[11]);
    ctx->lr = 0x00945068; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C14);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[11] + 0x34, ctx->gpr[28]);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x98);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    vm_write32(ctx->gpr[1] + 0xBC, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[0]);
    ctx->lr = 0x0094509C; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[3]);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[27]);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xA);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[0]);
    ctx->lr = 0x009450BC; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C10);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[11] + 0x34, ctx->gpr[28]);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x98);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    vm_write32(ctx->gpr[1] + 0xBC, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[0]);
    ctx->lr = 0x009450F0; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[3]);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[27]);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[0]);
    ctx->lr = 0x00945110; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C0C);
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x34, ctx->gpr[28]);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x98);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xBC, ctx->gpr[9]);
    ctx->lr = 0x00945144; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[3]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009451AC;
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[27]);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[0]);
    ctx->lr = 0x00945170; june_evt_func_0085F4A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C08);
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x3C, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    vm_write32(ctx->gpr[11] + 0x34, ctx->gpr[28]);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x98);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xBC, ctx->gpr[9]);
    ctx->lr = 0x009451A4; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x40, ctx->gpr[3]);
loc_009451AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
    vm_write32(ctx->gpr[31] + 0x44, ctx->gpr[9]);
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_009451EC:
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2C18);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[0]);
    ctx->lr = 0x009451FC; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x38, ctx->gpr[3]);
    goto loc_00944FDC;
loc_00945208:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2C28);
    ctx->gpr[4] = (int64_t)(int32_t)(0x51);
    ctx->lr = 0x00945218; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00944EB0;
}


void june_evt_func_00944230(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x28);
    ctx->lr = 0x00944254; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->lr = 0x00944260; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->lr = 0x0094426C; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00944288;
    ctx->lr = 0x00944284; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00944288:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->lr = 0x00944290; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->lr = 0x0094429C; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->lr = 0x009442A8; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x3C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009442C4;
    ctx->lr = 0x009442C0; june_evt_func_0085F468(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_009442C4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->lr = 0x009442CC; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->lr = 0x009442D8; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->lr = 0x009442E4; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00944300;
    ctx->lr = 0x009442FC; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00944300:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->lr = 0x00944308; june_evt_func_008632F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00944314; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}


void june_evt_func_00945220(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
    ctx->gpr[0] = ctx->lr;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[23]);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[29]);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[7], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[31]);
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x18);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[30]);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->lr = 0x009452C0; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[22] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x009452F8; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945314;
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[28], 0, 56);
    ctx->gpr[0] = (uint64_t)0 - ctx->gpr[0];
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 1, 63);
loc_00945314:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0094533C; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945350;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_00945350:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945AFC;
loc_00945360:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0094539C; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009453B0;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_009453B0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x009453D4; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009453E8;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_009453E8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0094540C; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945420;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_00945420:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945AC0;
loc_00945430:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C6C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00945478; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094548C;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_0094548C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[31] = ctx->fpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x009454B4; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009454C8;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_009454C8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x009454EC; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945500;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_00945500:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945B38;
loc_00945510:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945B88;
loc_00945524:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945BB8;
loc_00945534:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945BE8;
loc_00945544:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945564;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945D44;
loc_00945564:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945C08;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945C28;
loc_00945584:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945C48;
loc_00945594:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009455B4;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945CD0;
loc_009455B4:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    ctx->gpr[7] = vm_read32(ctx->gpr[9] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945A9C;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x24);
loc_009455D4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write16(ctx->gpr[31] + 0x8, ctx->gpr[7]);
    ctx->gpr[3] = ctx->gpr[22] + (int64_t)(8);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    vm_write16(ctx->gpr[31] + 0xA, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write16(ctx->gpr[31] + 0xC, ctx->gpr[9]);
    ctx->lr = 0x00945608; june_evt_func_0087AC30(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0094560C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094562C;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    vm_write8(ctx->gpr[11] + 0x2B, ctx->gpr[0]);
loc_0094562C:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[28] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x28);
    vm_write32(ctx->gpr[28] + 0x4, ctx->gpr[0]); ctx->gpr[28] += 0x4;
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00945680; june_evt_func_008638C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945694;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
loc_00945694:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945DC8;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[29];
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00945908;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0094575C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00945D64;
loc_009456CC:
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[11] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945CE4;
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x4);
    ctx->gpr[7] = vm_read8(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945E9C;
loc_00945700:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[31] + 0x4, ctx->gpr[7]);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x1);
    vm_write8(ctx->gpr[31] + 0x5, ctx->gpr[0]);
    ctx->gpr[11] = vm_read8(ctx->gpr[9] + 0x2);
    vm_write8(ctx->gpr[31] + 0x6, ctx->gpr[11]);
loc_00945720:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945748;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[11] = vm_read8(ctx->gpr[31] + 0x7);
    ctx->gpr[9] = vm_read8(ctx->gpr[9] + 0x3);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945840;
    ;
loc_00945748:
    ctx->gpr[7] = (int64_t)(int8_t)ctx->gpr[29];
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00945848;
    goto loc_009459E8;
loc_0094575C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00945788;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1BF);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2C00);
    ctx->lr = 0x00945778; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    ;
loc_00945788:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[7] = vm_read8(ctx->gpr[1] + 0x93);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945D84;
loc_009457B8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[31] + 0x4, ctx->gpr[7]);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[7], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    vm_write8(ctx->gpr[31] + 0x5, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[7], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    vm_write8(ctx->gpr[31] + 0x6, ctx->gpr[0]);
loc_009457FC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945748;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00945E80;
loc_00945814:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[11] = vm_read8(ctx->gpr[31] + 0x7);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x90);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945748;
loc_00945840:
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x7, ctx->gpr[9]);
loc_00945848:
    ctx->gpr[4] = ctx->gpr[22] + (int64_t)(8);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00945858; june_evt_func_00879E0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[11] + 0x14, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xA);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xA);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[11] + 0x14, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write32(ctx->gpr[11] + 0x14, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->lr = 0x009458C4; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->lr = 0x009458D0; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->lr = 0x009458DC; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945E3C;
loc_009458F0:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945990;
loc_00945908:
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945A30;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00945E1C;
loc_00945924:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x4);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x5);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x6);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945C5C;
    ;
loc_00945990:
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009459E8;
    ctx->lr = 0x0094599C; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009459E8;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x4C);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0xA);
    ctx->lr = 0x009459D8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x009459E0; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ;
loc_009459E8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[30];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_00945A30:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00945E00;
loc_00945A3C:
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[11] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945C8C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x4);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x5);
    vm_write8(ctx->gpr[11] + 0x1, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x6);
    vm_write8(ctx->gpr[10] + 0x2, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945990;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
    vm_write8(ctx->gpr[9] + 0x3, ctx->gpr[0]);
    goto loc_00945990;
loc_00945A9C:
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xA);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945DE0;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x24);
    goto loc_009455D4;
loc_00945AC0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00945AE4; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945430;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    goto loc_00945430;
loc_00945AFC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00945B20; june_evt_func_0085F638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945360;
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    goto loc_00945360;
loc_00945B38:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00945B5C; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945510;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945524;
    ;
loc_00945B88:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->lr = 0x00945BA0; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945534;
    ;
loc_00945BB8:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->lr = 0x00945BD0; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945544;
    ;
loc_00945BE8:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->lr = 0x00945C00; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00945544;
loc_00945C08:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945584;
loc_00945C28:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945594;
loc_00945C48:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    goto loc_00945594;
loc_00945C5C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00945E64;
loc_00945C68:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    goto loc_00945990;
loc_00945C8C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x4);
    vm_write8(ctx->gpr[9] + 0x3, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x5);
    vm_write8(ctx->gpr[11] + 0x2, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x6);
    vm_write8(ctx->gpr[10] + 0x1, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945990;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_00945990;
loc_00945CD0:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    goto loc_009455B4;
loc_00945CE4:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x4);
    ctx->gpr[7] = vm_read8(ctx->gpr[11] + 0x3);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945EC0;
loc_00945CFC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[31] + 0x4, ctx->gpr[7]);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x2);
    vm_write8(ctx->gpr[31] + 0x5, ctx->gpr[0]);
    ctx->gpr[11] = vm_read8(ctx->gpr[9] + 0x1);
    vm_write8(ctx->gpr[31] + 0x6, ctx->gpr[11]);
loc_00945D1C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00945748;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[11] = vm_read8(ctx->gpr[31] + 0x7);
    ctx->gpr[9] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945840;
    goto loc_00945748;
loc_00945D44:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->lr = 0x00945D5C; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00945564;
loc_00945D64:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x193);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2C04);
    ctx->lr = 0x00945D74; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    goto loc_009456CC;
loc_00945D84:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x5);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[1] + 0x93);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009457B8;
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x6);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[1] + 0x93);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009457B8;
    goto loc_009457FC;
loc_00945DC8:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[29];
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_009459E8;
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[24];
    goto loc_00945990;
loc_00945DE0:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xC);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009455D4;
    goto loc_0094560C;
loc_00945E00:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1F4);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2C04);
    ctx->lr = 0x00945E10; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    goto loc_00945A3C;
loc_00945E1C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x208);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2C00);
    ctx->lr = 0x00945E2C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    goto loc_00945924;
loc_00945E3C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
    vm_write32(ctx->gpr[11] + 0x14, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->lr = 0x00945E5C; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_009458F0;
loc_00945E64:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2BFC);
    ctx->lr = 0x00945E74; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    goto loc_00945C68;
loc_00945E80:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2C2C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1CE);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2BFC);
    ctx->lr = 0x00945E90; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x44);
    goto loc_00945814;
loc_00945E9C:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x5);
    ctx->gpr[9] = vm_read8(ctx->gpr[11] + 0x1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945700;
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x2);
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x6);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945700;
    goto loc_00945720;
loc_00945EC0:
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x5);
    ctx->gpr[9] = vm_read8(ctx->gpr[11] + 0x2);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945CFC;
    ctx->gpr[0] = vm_read8(ctx->gpr[11] + 0x1);
    ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x6);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00945CFC;
    goto loc_00945D1C;
}


void june_evt_func_00943C78(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x100, ctx->gpr[1]); ctx->gpr[1] += -0x100;
    ctx->gpr[0] = ctx->lr;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x84);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x18);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->lr = 0x00943D00; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943D28; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943D4C; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00944208;
loc_00943D60:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x14);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943D94; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943DB8; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943DDC; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009441E0;
loc_00943DF0:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00943E30; june_evt_func_0086343C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C6C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C68); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C64); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00943E74; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943E98; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00943EBC; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009441B8;
loc_00943ED0:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943F1C;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
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
loc_00943F1C:
    ctx->lr = 0x00943F20; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[31] = ctx->fpr[30];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00943F30; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943F5C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2C60);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00943F5C:
    ctx->lr = 0x00943F60; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00943F6C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00943FB0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943FDC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2C5C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00943FDC:
    ctx->lr = 0x00943FE0; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00943FF0; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[9] = (int64_t)(int32_t)(-52);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[3]);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x82, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x83, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write8(ctx->gpr[1] + 0x81, ctx->gpr[9]);
    ctx->gpr[29] = ctx->gpr[30] + (int64_t)(0x10);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C58); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00944074; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[3] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[28]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[30]));
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x5);
    ctx->gpr[10] = vm_read8(ctx->gpr[31] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[11] = vm_read8(ctx->gpr[31] + 0x6);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2C4C);
    ctx->gpr[8] = ctx->gpr[28] | ctx->gpr[28];
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[11]);
    ctx->lr = 0x009440B8; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009440E0; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00944108; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C48); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00944138; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0094417C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
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
loc_009441B8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x009441D8; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00943ED0;
loc_009441E0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00944200; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00943DF0;
loc_00944208:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x00944228; june_evt_func_0085EF60(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00943D60;
}


void june_evt_func_00943C68(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x4C, ctx->gpr[4]);
    return;
}


void june_evt_func_00943C70(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4C);
    return;
}


void june_evt_func_0084DD2C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0084DD50; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x5D);
    ctx->gpr[3] = (int64_t)(int32_t)(0x103C);
    ctx->lr = 0x0084DD70; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DF14;
loc_0084DD8C:
    ctx->gpr[4] = (int64_t)(int32_t)(0x103C);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084DD98; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[9] = ctx->gpr[9] | 0x8;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084DEAC;
loc_0084DDC4:
    ctx->lr = 0x0084DDC8; june_evt_func_0088A110(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write32(ctx->gpr[31] + 0x102C, ctx->gpr[3]);
    ctx->lr = 0x0084DDD8; june_evt_func_0089ACC4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = vm_read32(ctx->gpr[31] + 0x102C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6870); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[3]);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    ctx->lr = 0x0084DDFC; june_evt_func_008F6ECC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[4] = (int64_t)(int32_t)(0x18);
    vm_write32(ctx->gpr[31] + 0x1030, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x4, tmp); }
    ctx->lr = 0x0084DE34; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)(0x82);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DE5C;
    ctx->gpr[0] = (int64_t)(int32_t)(0x9B);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
loc_0084DE5C:
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x686C);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6860);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6868);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6864);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x80, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0x84, ctx->gpr[10]);
    ctx->lr = 0x0084DE80; june_evt_func_00856B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    vm_write32(ctx->gpr[31] + 0x1014, ctx->gpr[3]);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_0084DEAC:
    ctx->lr = 0x0084DEB0; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6874); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(2);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x8C, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    vm_write32(ctx->gpr[1] + 0xA0, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0x94, ctx->gpr[11]);
    ctx->lr = 0x0084DEF0; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x101C, ctx->gpr[3]);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write16(ctx->gpr[1] + 0x8C, ctx->gpr[0]);
    ctx->lr = 0x0084DF08; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x1018, ctx->gpr[3]);
    goto loc_0084DDC4;
loc_0084DF14:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6878);
    ctx->lr = 0x0084DF1C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084DD8C;
}


void june_evt_func_0084E168(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1E0, ctx->gpr[1]); ctx->gpr[1] += -0x1E0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x1014);
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x178, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x188, ctx->gpr[30]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x1A0, tmp); }
    ctx->gpr[20] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[15] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x1C8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1D0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1D8, tmp); }
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[27]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    ctx->lr = 0x0084E214; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084E234;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_0084E234:
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[20] + 0x1014);
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0084E258; june_evt_func_00857B80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    { uint32_t a = (ctx->cr >> (31 - 24)) & 1; uint32_t b = (ctx->cr >> (31 - 26)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E888;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E888;
    { double a = ctx->fpr[26]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
    { uint32_t a = (ctx->cr >> (31 - 4)) & 1; uint32_t b = (ctx->cr >> (31 - 6)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E888;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    if ((((ctx->cr >> 7) & 1) == 1)) goto loc_0084E888;
    if ((((ctx->cr >> 27) & 1) == 1)) goto loc_0084E888;
    ctx->lr = 0x0084E2CC; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[14] = ctx->gpr[3] + (int64_t)(4);
    ctx->gpr[18] = ctx->gpr[1] + (int64_t)(0xB8);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->gpr[5] = ctx->gpr[18] | ctx->gpr[18];
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0084E2F4; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[20] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E390;
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x1020); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x1024); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084EB04;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x1018);
    vm_write8(ctx->gpr[1] + 0xE4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
loc_0084E354:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084EAD4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x101C);
    vm_write8(ctx->gpr[1] + 0xE5, ctx->gpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ;
loc_0084E390:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[17] = ctx->gpr[1] + (int64_t)(0xA8);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6858); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6854); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x0084E3C8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E904;
    ctx->gpr[5] = vm_read32(ctx->gpr[20] + 0x8);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x88);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E430;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[7] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0084E428; june_evt_func_0084C668(ctx); DRAIN_TRAMPOLINE(ctx);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ;
loc_0084E430:
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x102C);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->lr = 0x0084E448; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xE0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[23] = vm_read32(ctx->gpr[20] + 0x1030);
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[24] = vm_read32(ctx->gpr[1] + 0xE0);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084E46C;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_0084E46C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xE0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xE0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xE0);
    ctx->gpr[31] = (uint64_t)((int64_t)(int32_t)ctx->gpr[23] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 0x1F);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[11] >> 32));
    ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[9];
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 3);
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[10];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 4, 0, 27, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[11];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084E6D4;
    ctx->fpr[13] = ctx->fpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084E6D8;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[23], 2, 0, 29, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x689C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[30] = ctx->gpr[23] + ctx->gpr[31];
    ctx->gpr[22] = ctx->gpr[0] + ctx->gpr[23];
    ctx->gpr[29] = ctx->gpr[30] | ctx->gpr[30];
    goto loc_0084E580;
loc_0084E530:
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0084E544; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[5] = ctx->fpr[31];
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084E698;
loc_0084E55C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[24]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[30] = ctx->gpr[30] + ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084E6D4;
loc_0084E574:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084E6D8;
loc_0084E580:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    { int32_t _a=(int32_t)ctx->gpr[31], _b=(int32_t)ctx->gpr[22]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[0] * (int64_t)(int32_t)ctx->gpr[22]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084E610;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xC8);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0084E610;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x30);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    ctx->lr = 0x0084E5D4; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6850); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[3] = ctx->fpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[1]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[13]));
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x684C);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->lr = 0x0084E60C; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0084E610:
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[31] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0084E628; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[23] + ctx->gpr[29];
    ctx->fpr[5] = ctx->fpr[31];
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E530;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x34);
    ctx->lr = 0x0084E664; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0084E67C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[5] = ctx->fpr[31];
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E55C;
    ;
loc_0084E698:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x34);
    ctx->lr = 0x0084E6B8; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[24]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[30] + ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084E574;
loc_0084E6D4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0084E6D8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[18] | ctx->gpr[18];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->lr = 0x0084E704; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084E92C;
loc_0084E714:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[28]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[28]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x0084E744; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E7A8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[28]));
    ctx->gpr[5] = vm_read32(ctx->gpr[20] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E7A8;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x0084E7A4; june_evt_func_0084CAF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0084E7A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E910;
    ctx->gpr[9] = vm_read8(ctx->gpr[20] + 0x1038);
    ctx->gpr[11] = (int64_t)(int32_t)(-128);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084E7D0;
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(0x10);
loc_0084E7D0:
    vm_write8(ctx->gpr[20] + 0x1038, ctx->gpr[11]);
    ctx->gpr[0] = vm_read8(ctx->gpr[20] + 0x1038);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EA70;
loc_0084E7E0:
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x102C);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->lr = 0x0084E7F4; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xE0);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[23]));
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xE0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xE0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xE0);
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[23] * (int64_t)(int32_t)ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084E960;
loc_0084E82C:
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E888;
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xE5);
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EB30;
loc_0084E854:
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xE4);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E888;
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x1018);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x98);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x0084E884; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0084E888:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1F0);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A0); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A8); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1E0);
    return;
loc_0084E904:
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x88);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_0084E430;
loc_0084E910:
    ctx->gpr[9] = vm_read8(ctx->gpr[20] + 0x1038);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084E7D0;
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(-16);
    goto loc_0084E7D0;
loc_0084E92C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    vm_write32(ctx->gpr[1] + 0xF8, ctx->gpr[29]);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x34);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0084E958; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084E714;
loc_0084E960:
    ctx->fpr[0] = ctx->fpr[13];
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xE0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[23]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xE0);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 0x1F);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[11] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[11] >> 32));
    ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[9];
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 3);
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[10];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 4, 0, 27, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[11];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[29], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0084EA04; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084EB58;
loc_0084EA14:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0084EA30; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E82C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x38);
    ctx->lr = 0x0084EA68; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084E82C;
loc_0084EA70:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[28]));
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[14] + 0x0);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[5] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read8(ctx->gpr[20] + 0x1038);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write8(ctx->gpr[1] + 0x77, ctx->gpr[11]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x0084EACC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084E7E0;
loc_0084EAD4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 27, 27, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write8(ctx->gpr[1] + 0xE5, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E390;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    vm_write8(ctx->gpr[1] + 0xE5, ctx->gpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    goto loc_0084E390;
loc_0084EB04:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write8(ctx->gpr[1] + 0xE4, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084E354;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    vm_write8(ctx->gpr[1] + 0xE4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    goto loc_0084E354;
loc_0084EB30:
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x101C);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x98);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x0084EB50; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084E854;
loc_0084EB58:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x38);
    ctx->lr = 0x0084EB80; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084EA14;
}


void june_evt_func_0084C634(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[0] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1014);
    vm_write32(ctx->gpr[9] + 0x1028, ctx->gpr[0]);
    ctx->lr = 0x0084C654; june_evt_func_00856068(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
}


void june_evt_func_0084BDD8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1028);
    return;
}


void june_evt_func_0094218C(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[30]);
    ctx->lr = 0x009421C8; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[25] = vm_read32(ctx->gpr[2] + -0x2D3C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x28);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[6] = (int64_t)(int32_t)(0x36);
    ctx->lr = 0x009421EC; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009425AC;
loc_009421FC:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00942210; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[9] = vm_read16(ctx->gpr[26] + 0x0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    ctx->gpr[0] = vm_read16(ctx->gpr[26] + 0x2);
    vm_write16(ctx->gpr[31] + 0x2, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942350;
    ctx->gpr[11] = ctx->gpr[10] + (int64_t)(8);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
loc_00942240:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00942240;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[28], 2, 0, 29, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[28], 4, 0, 27, 0, 0);
    ctx->gpr[30] = ctx->gpr[0] - ctx->gpr[9];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094234C;
loc_0094226C:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[6] = (int64_t)(int32_t)(0x45);
    ctx->lr = 0x00942288; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942594;
loc_00942298:
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->lr = 0x009422A4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x2);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094236C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094247C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942490;
loc_009422C4:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_009422D8;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
loc_009422D8:
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[28]);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x18);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x1C);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[11]);
    vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00942310;
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D00);
    ctx->gpr[4] = (int64_t)(int32_t)(0x86);
    ctx->lr = 0x0094230C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00942310:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[24] = _cs_24;
    ctx->gpr[23] = _cs_23;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_0094234C:
    ctx->gpr[30] = (int64_t)(int32_t)(0);
loc_00942350:
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D14);
    ctx->gpr[4] = (int64_t)(int32_t)(0x42);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00942364; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0094226C;
loc_0094236C:
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->lr = 0x0094237C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x8C);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0x12);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x84, ctx->gpr[9]);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[10]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009422C4;
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D0C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_009423B8:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[27], 3, 0, 28, 0, 0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[29] = ctx->gpr[30] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    ctx->lr = 0x009423E4; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[3]);
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[11];
    ctx->gpr[4] = (int64_t)(int32_t)(0x56);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942468;
loc_0094240C:
    ctx->gpr[29] = ctx->gpr[30] + ctx->gpr[11];
    ctx->lr = 0x00942414; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x88);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00942428; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00942460;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
loc_00942460:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_009423B8;
    goto loc_009422C4;
loc_00942468:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D10);
    ctx->lr = 0x00942470; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0094240C;
loc_0094247C:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_009422C4;
loc_00942490:
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->lr = 0x009424A0; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009422C4;
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ;
loc_009424D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[27], 3, 0, 28, 0, 0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[29] = ctx->gpr[30] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x009424FC; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[3]);
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[11];
    ctx->gpr[4] = (int64_t)(int32_t)(0x6A);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942580;
loc_00942524:
    ctx->gpr[29] = ctx->gpr[30] + ctx->gpr[11];
    ctx->lr = 0x0094252C; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00942540; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00942578;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
loc_00942578:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_009424D0;
    goto loc_009422C4;
loc_00942580:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D08);
    ctx->lr = 0x00942588; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00942524;
loc_00942594:
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D1C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x46);
    ctx->lr = 0x009425A4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00942298;
loc_009425AC:
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D18);
    ctx->gpr[4] = (int64_t)(int32_t)(0x37);
    ctx->lr = 0x009425BC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_009421FC;
}


void june_evt_func_00942088(ppu_context* ctx) {
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
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094216C;
loc_009420BC:
    ctx->gpr[28] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094213C;
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942184;
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
loc_009420D8:
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[0];
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942100;
    ctx->lr = 0x009420F8; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
loc_00942100:
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[0];
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0xC);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942128;
    ctx->lr = 0x00942120; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
loc_00942128:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009420D8;
loc_00942130:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x00942138; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0094213C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->lr = 0x00942144; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
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
loc_0094216C:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D1C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x97);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2D3C);
    ctx->lr = 0x0094217C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_009420BC;
loc_00942184:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    goto loc_00942130;
}


void june_evt_func_00941AC0(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[29]);
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[30] = vm_read32(ctx->gpr[4] + 0xC);
    ctx->lr = 0x00941B2C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941B50;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D28);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941C64;
loc_00941B50:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941BC0;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941BA0;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941C58;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941D90;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941D9C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2D3C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x12A);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D38);
    ctx->lr = 0x00941B9C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00941BA0:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941BBC;
    ctx->lr = 0x00941BAC; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941C14;
loc_00941BBC:
    ctx->gpr[3] = (int64_t)(int32_t)(1);
loc_00941BC0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_00941C14:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x13);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00941C44; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00941C4C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    goto loc_00941BC0;
loc_00941C58:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[30]);
    goto loc_00941BA0;
loc_00941C64:
    ctx->lr = 0x00941C68; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941C98;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D24);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[29] = ppu_frsp(ctx->fpr[0]);
loc_00941C98:
    ctx->lr = 0x00941C9C; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00941CA8; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00941CE4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941D10;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D20);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[28] = ppu_frsp(ctx->fpr[0]);
loc_00941D10:
    ctx->lr = 0x00941D14; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00941D20; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x2);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[26] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[28]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[28] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941DA8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941E98;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941F7C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2D3C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x114);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D38);
    ctx->lr = 0x00941D88; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00941B50;
loc_00941D90:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[30]);
    goto loc_00941BA0;
loc_00941D9C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[30]);
    goto loc_00941BA0;
loc_00941DA8:
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941B50;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x70);
    goto loc_00941E04;
loc_00941DE0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ;
loc_00941DF8:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941B50;
loc_00941E04:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x00941E14; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941DF8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941E70;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941E70;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941E70;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942078;
    ;
loc_00941E70:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941DE0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    goto loc_00941DF8;
loc_00941E98:
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941B50;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x70);
    goto loc_00941EEC;
loc_00941ECC:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ;
loc_00941EE0:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941B50;
loc_00941EEC:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x00941EFC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941EE0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941F58;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941F58;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941F58;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942078;
    ;
loc_00941F58:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941ECC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    goto loc_00941EE0;
loc_00941F7C:
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941B50;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x70);
    goto loc_00941FD8;
loc_00941FB0:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ;
loc_00941FC8:
    ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941B50;
loc_00941FD8:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00941FFC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941FC8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00942058;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00942058;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00942058;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942080;
    ;
loc_00942058:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941FB0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    goto loc_00941FC8;
loc_00942078:
    ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
    goto loc_00941B50;
loc_00942080:
    ctx->gpr[30] = ctx->gpr[25] | ctx->gpr[25];
    goto loc_00941B50;
}


void june_evt_func_00941420(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x150, ctx->gpr[1]); ctx->gpr[1] += -0x150;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[26]);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0xA0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[28]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00941498; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x009414D0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941518;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x2);
    ctx->gpr[28] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[24] = vm_read32(ctx->gpr[31] + 0x8);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941578;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941640;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009417E0;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D38);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1F0);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2D3C);
    ctx->lr = 0x00941514; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00941518:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_00941578:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941518;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    goto loc_009415D8;
loc_009415B4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_009415C8:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941518;
loc_009415D8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[0] = ctx->gpr[29] ^ ctx->gpr[0];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[9] = ctx->gpr[11] ^ ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[11];
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x4);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[4] = ctx->gpr[11] | ctx->gpr[11];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
    vm_write32(ctx->gpr[11] + 0xC, ctx->gpr[9]);
    ctx->lr = 0x00941614; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009415B4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_009415C8;
loc_00941640:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->lr = 0x0094165C; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941954;
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941518;
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[21] = vm_read32(ctx->gpr[2] + -0x2D34);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[23] = ctx->gpr[24] + (int64_t)(-1);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = ctx->gpr[21] + (int64_t)(4);
    ctx->gpr[25] = ctx->gpr[27] | ctx->gpr[27];
    goto loc_00941784;
loc_009416A0:
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
loc_009416A8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009416D8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009416F4;
    ctx->gpr[0] = ctx->gpr[28] + (int64_t)(-1);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941720;
loc_009416F4:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0094171C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00941720:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00941758; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094179C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094179C;
loc_00941770:
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941518;
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
loc_00941784:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009416A0;
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    goto loc_009416A8;
loc_0094179C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x009417D8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00941770;
loc_009417E0:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941518;
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[18] = vm_read32(ctx->gpr[2] + -0x2D30);
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[23] = vm_read32(ctx->gpr[2] + -0x2D48);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D54); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[17] = ctx->gpr[18] + (int64_t)(0x10);
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x90);
    goto loc_00941874;
loc_00941850:
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_00941864:
    ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941518;
loc_00941874:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[7] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[9] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->lr = 0x0094189C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[7] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[9] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941948;
loc_009418CC:
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941928;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->lr = 0x00941924; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00941928:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941850;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_00941864;
loc_00941948:
    ctx->lr = 0x0094194C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_009418CC;
loc_00941954:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941518;
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[23] = vm_read32(ctx->gpr[2] + -0x2D34);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[25] = ctx->gpr[24] + (int64_t)(-1);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[21] = ctx->gpr[23] + (int64_t)(4);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[22], 0, 32);
    goto loc_00941A64;
loc_0094197C:
    ctx->gpr[11] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
loc_00941984:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094199C;
    ctx->gpr[0] = ctx->gpr[28] + (int64_t)(-1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009419D4;
loc_0094199C:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x009419D0; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_009419D4:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x009419FC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00941A38; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941A7C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941A7C;
loc_00941A50:
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941518;
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
loc_00941A64:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0094197C;
    ctx->gpr[11] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    goto loc_00941984;
loc_00941A7C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00941AB8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00941A50;
}


void june_evt_func_00941410(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x24, ctx->gpr[4]);
    return;
}


void june_evt_func_00941418(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x24);
    return;
}


void june_evt_func_0093FDFC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_24 = ctx->gpr[24];
    uint64_t _cs_25 = ctx->gpr[25];
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x210, ctx->gpr[1]); ctx->gpr[1] += -0x210;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x1D0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x1F8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x220, ctx->gpr[0]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x208, tmp); }
    vm_write64(ctx->gpr[1] + 0x1C8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x1D8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x1E8, ctx->gpr[28]);
    ctx->lr = 0x0093FE34; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2D88);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x54);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x52);
    ctx->gpr[30] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0093FE5C; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940118;
loc_0093FE6C:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x54);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->lr = 0x0093FE80; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[30] + (int64_t)(4);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0xC);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x18);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    ctx->lr = 0x0093FEA4; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[1] = ctx->fpr[0];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->fpr[2] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->lr = 0x0093FEE0; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0093FEF8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[11] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[26]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->lr = 0x0093FF20; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009400D0;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2D74);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->lr = 0x0093FF54; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[26]);
    ctx->lr = 0x0093FF6C; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[3]);
    ctx->lr = 0x0093FF78; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FF88; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->fpr[1] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
    ctx->lr = 0x0093FFB4; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0093FFC0; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[3]);
    ctx->lr = 0x0093FFCC; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FFDC; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
    ctx->lr = 0x00940004; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00940010; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x0);
    vm_write32(ctx->gpr[31] + 0x38, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->fpr[1] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1B0);
    ctx->lr = 0x00940044; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00940050; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    vm_write32(ctx->gpr[31] + 0x3C, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00940064; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(4);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    vm_write16(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    vm_write32(ctx->gpr[1] + 0x84, ctx->gpr[0]);
    ctx->lr = 0x009400A8; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    vm_write32(ctx->gpr[31] + 0x40, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->lr = 0x009400C8; june_evt_func_0085DB70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x44, ctx->gpr[3]);
loc_009400D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x1C);
    vm_write32(ctx->gpr[31] + 0x4C, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x220);
    vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[9]);
    ctx->gpr[24] = _cs_24;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[25] = _cs_25;
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x208); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x210);
    return;
loc_00940118:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D7C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x53);
    ctx->lr = 0x00940128; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093FE6C;
}


void june_evt_func_0093FD20(ppu_context* ctx) {
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
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FD54;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FD50; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093FD54:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FD6C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FD68; june_evt_func_0085DAE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093FD6C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FD84;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FD80; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093FD84:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FD9C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FD98; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093FD9C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FDB4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FDB0; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093FDB4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x3C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FDCC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0093FDC8; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093FDCC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->lr = 0x0093FDD4; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0093FDE0; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}


void june_evt_func_00940130(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x2A0, ctx->gpr[1]); ctx->gpr[1] += -0x2A0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x240, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x248, tmp); }
    vm_write64(ctx->gpr[1] + 0x220, ctx->gpr[27]);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x250, tmp); }
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x258, tmp); }
    vm_write64(ctx->gpr[1] + 0x210, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x238, ctx->gpr[30]);
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x260, tmp); }
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x268, tmp); }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x270, tmp); }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x278, tmp); }
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x280, tmp); }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x208, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x218, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x228, ctx->gpr[28]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x288, tmp); }
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x290, tmp); }
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x298, tmp); }
    ctx->gpr[26] = ctx->gpr[31] | ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x1F8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x230, ctx->gpr[29]);
    ctx->lr = 0x009401C8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940248;
loc_009401DC:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x2B0);
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
loc_00940248:
    ctx->lr = 0x0094024C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094027C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D70);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0094027C:
    ctx->lr = 0x00940280; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0094028C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x009402C8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009402F4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D6C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_009402F4:
    ctx->lr = 0x009402F8; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00940304; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940368;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940B30;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D84);
    ctx->gpr[4] = (int64_t)(int32_t)(0x282);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2D88);
    ctx->lr = 0x0094035C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    goto loc_009401DC;
loc_00940368:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940D44;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 27, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094040C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009412CC;
loc_009403A8:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941278;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_009403BC:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D78);
    ctx->fpr[1] = ctx->fpr[13];
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[2] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->lr = 0x009403F8; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00940408; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0094040C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x88);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[24], ctx->fpr[13], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x40);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[22]));
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x00940488; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094049C;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
loc_0094049C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940E3C;
loc_009404AC:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[22]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[24], ctx->fpr[13], 0, 0));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x009404E0; june_evt_func_0085E078(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009404F4;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
loc_009404F4:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940E74;
loc_00940504:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940D4C;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009410A8;
loc_00940540:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094118C;
loc_0094054C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009405D4;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[8] = ctx->fpr[31];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00940EF0;
loc_00940588:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[1]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[12], ctx->fpr[11], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[1]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[0], ctx->fpr[11], 0, 0));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[31]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00940FB8;
loc_009405C8:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940928;
loc_009405D4:
    ctx->lr = 0x009405D8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940928;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D68);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ctx->fpr[28];
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_009413D0;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[21] = ctx->fpr[31];
loc_0094061C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[24]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[22]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0094066C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009407A4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009407A4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009407A4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009407A4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009407A4;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0xD0);
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D74);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x009406EC; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->lr = 0x00940700; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ctx->fpr[29];
    ctx->fpr[13] = ctx->fpr[27];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->lr = 0x00940738; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    ctx->lr = 0x0094075C; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[6] = (int64_t)(int32_t)(0x18);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[8] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x00940784; june_evt_func_0083FE10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x00940794; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ctx->gpr[0] | 0x8;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
loc_009407A4:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[22]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[22] | ctx->gpr[22];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[21]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x009407E8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940928;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D74);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->fpr[1] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->lr = 0x00940870; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->lr = 0x00940884; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ctx->fpr[29];
    ctx->fpr[13] = ctx->fpr[27];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->lr = 0x009408BC; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    ctx->lr = 0x009408E0; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->gpr[6] = (int64_t)(int32_t)(0x18);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[8] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x00940908; june_evt_func_0083FE10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x00940918; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ctx->gpr[0] | 0x10;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
loc_00940928:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[21];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_009401DC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009409CC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[31]));
    ctx->lr = 0x0094095C; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ctx->fpr[1];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1D0);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[10], ctx->fpr[31]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x1D0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->lr = 0x0094099C; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1D0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x1D0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[10] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_009409CC:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00940EA8;
    ctx->fpr[10] = ctx->fpr[30];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->fpr[13] = ctx->fpr[11];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_009409EC:
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[10]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00941090;
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00940A00:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0094132C;
loc_00940A0C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xD0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x2D78);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[1] = ctx->fpr[0];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[4] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->fpr[2] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->lr = 0x00940A64; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00940A74; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940AAC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x34);
    ctx->lr = 0x00940A94; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4C);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
loc_00940AAC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940ACC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
loc_00940ACC:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009401DC;
    ctx->lr = 0x00940ADC; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009401DC;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x50);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[23] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0x14);
    ctx->lr = 0x00940B1C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00940B24; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    goto loc_009401DC;
loc_00940B30:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940B60;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940EF8;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940EF8;
loc_00940B60:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940D44;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940D44;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940D44;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940D44;
    ctx->lr = 0x00940BAC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940D44;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D68);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940D44;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[5] = (int64_t)(int32_t)(0x1C2);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[6] = (int64_t)(int32_t)(0x24);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->lr = 0x00940C1C; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0x1C2);
    ctx->gpr[6] = (int64_t)(int32_t)(0x24);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    ctx->lr = 0x00940C40; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x2080 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x24);
    ctx->gpr[5] = (int64_t)(int32_t)(0x1C2);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x00940C6C; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D58); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    vm_write32(ctx->gpr[1] + 0x8C, ctx->gpr[11]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[11]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x00940CD4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0xC);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x18);
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[0]);
    ctx->lr = 0x00940CF4; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = (int64_t)(int32_t)(0x14);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[26]);
    ctx->lr = 0x00940D18; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00940D28; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00940D38; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    goto loc_009401DC;
loc_00940D44:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    goto loc_009401DC;
loc_00940D4C:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->lr = 0x00940D54; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940ED0;
loc_00940D68:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940F54;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941344;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940928;
    ctx->lr = 0x00940DA4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009413AC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D5C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009413AC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940E2C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940E2C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00940E2C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { double a = ctx->fpr[9]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00940E14;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[10]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
loc_00940E14:
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00940E20;
    ctx->fpr[13] = ctx->fpr[0];
loc_00940E20:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
loc_00940E2C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D64);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_00940928;
loc_00940E3C:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009404AC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[1]));
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[1]));
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_009404AC;
loc_00940E74:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940504;
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[12]));
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00940504;
loc_00940EA8:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00941324;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_009409EC;
loc_00940ED0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D60);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    goto loc_00940D68;
loc_00940EF0:
    ctx->fpr[8] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    goto loc_00940588;
loc_00940EF8:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->fpr[1] = ctx->fpr[12];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0xD0);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1E8);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->fpr[2] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x1E8);
    ctx->lr = 0x00940F38; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00940F48; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    goto loc_009401DC;
loc_00940F54:
    ctx->lr = 0x00940F58; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941334;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D5C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941334;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00940FA8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[31]));
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00940FA8:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D64);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_00940928;
loc_00940FB8:
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[21]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->lr = 0x00940FEC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941044;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941044;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941044;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941044;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094104C;
loc_00941044:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_009405C8;
loc_0094104C:
    ctx->lr = 0x00941050; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941404;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D68);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941404;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ctx->gpr[11] | 0x4;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
loc_00941080:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D64);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_009405C8;
loc_00941090:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00940A00;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00940A0C;
loc_009410A8:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[21]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x009410E8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941140;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941140;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941140;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00941140;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941148;
loc_00941140:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_00940540;
loc_00941148:
    ctx->lr = 0x0094114C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009413D8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D68);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009413D8;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ctx->gpr[11] | 0x1;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
loc_0094117C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D64);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_00940540;
loc_0094118C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[21]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x009411D4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094122C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0094122C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0094122C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0094122C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941234;
loc_0094122C:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_0094054C;
loc_00941234:
    ctx->lr = 0x00941238; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009413E0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D68);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009413E0;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ctx->gpr[11] | 0x2;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
loc_00941268:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D64);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0094054C;
loc_00941278:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x3C);
    ctx->lr = 0x00941280; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00941294; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[13]));
    { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009413C0;
loc_009412B0:
    ctx->fpr[0] = ctx->fpr[1];
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 27, 1, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 5, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
    goto loc_009403BC;
loc_009412CC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->lr = 0x009412D4; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->lr = 0x009412F4; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[1]));
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009413C8;
loc_0094130C:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 28, 1, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 4, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
    goto loc_009403A8;
loc_00941324:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_009409EC;
loc_0094132C:
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00940A0C;
loc_00941334:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
    goto loc_00940928;
loc_00941344:
    ctx->lr = 0x00941348; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941398;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2D5C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00941398;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00940FA8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_009413E8;
loc_00941384:
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[10]));
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00940FA8;
loc_00941398:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[11]);
    goto loc_00940928;
loc_009413AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    goto loc_00940928;
loc_009413C0:
    ctx->fpr[1] = ctx->fpr[0];
    goto loc_009412B0;
loc_009413C8:
    ctx->fpr[0] = ctx->fpr[31];
    goto loc_0094130C;
loc_009413D0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    goto loc_0094061C;
loc_009413D8:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_0094117C;
loc_009413E0:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_00941268;
loc_009413E8:
    if ((((ctx->cr >> 6) & 1) == 1)) goto loc_00941384;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_00940FA8;
loc_00941404:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_00941080;
}


void june_evt_func_0093F590(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[19], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    ctx->gpr[23] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    ctx->lr = 0x0093F608; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093F614; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F6AC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093FAE8;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2D84);
    ctx->gpr[4] = (int64_t)(int32_t)(0x331);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2D88);
    ctx->lr = 0x0093F640; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093F644:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[19], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[22], &tmp, 8); }
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
loc_0093F6AC:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { double a = ctx->fpr[25]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[24] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093FAC8;
    { double a = ctx->fpr[27]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->fpr[19] = ctx->fpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093FAE0;
loc_0093F6F8:
    ctx->fpr[21] = ctx->fpr[19];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
loc_0093F700:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[29]));
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2DAC);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[24] = vm_read32(ctx->gpr[2] + -0x2DA8);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[19]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[19]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[24]));
    ctx->gpr[6] = ctx->gpr[24] | ctx->gpr[24];
    ctx->fpr[20] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[19]));
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[19]));
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093F764; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[19]));
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x10);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093F794; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x20);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093F7C0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x30);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093F7EC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[19]));
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[19]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x40);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093F828; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x50);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093F854; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[19]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x60);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093F88C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[24] + (int64_t)(0x70);
    ctx->lr = 0x0093F8B8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093F8F8; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093F92C; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[19]));
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093F964; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093F9AC; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x44);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093F9DC; june_evt_func_0085D9E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { double a = ctx->fpr[10]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093F644;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[9]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[23], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[23], ctx->fpr[13], ctx->fpr[12], 0, 0));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[19]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[19]));
    { double a = ctx->fpr[31]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093FC58;
loc_0093FA34:
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[19]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093FA68; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093FC2C;
loc_0093FA78:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0093FA8C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F644;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2D8C);
    ctx->lr = 0x0093FAC0; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093F644;
loc_0093FAC8:
    { double a = ctx->fpr[27]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[23] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[8]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[19] = ftmp; }
    ctx->fpr[22] = ppu_fp_single(ppu_fmul(ctx->fpr[25], ctx->fpr[0]));
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093F6F8;
loc_0093FAE0:
    ctx->fpr[21] = ppu_fp_single(ppu_fsub(ctx->fpr[27], ctx->fpr[8]));
    goto loc_0093F700;
loc_0093FAE8:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0093FB2C; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x80);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093FB50; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093FC90;
loc_0093FB60:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093FB80; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093FCF0;
loc_0093FB90:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0093FBBC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093FCC0;
loc_0093FBCC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093FBEC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F644;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093FC24; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093F644;
loc_0093FC2C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2D8C);
    ctx->lr = 0x0093FC50; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093FA78;
loc_0093FC58:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[31]));
    ctx->gpr[4] = ctx->gpr[24] + (int64_t)(0x80);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0093FC88; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093FA34;
loc_0093FC90:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093FCB8; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093FB60;
loc_0093FCC0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093FCE8; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093FBCC;
loc_0093FCF0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2D80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0093FD18; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093FB90;
}


void june_evt_func_0093F580(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x50, ctx->gpr[4]);
    return;
}


void june_evt_func_0093F588(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x50);
    return;
}

