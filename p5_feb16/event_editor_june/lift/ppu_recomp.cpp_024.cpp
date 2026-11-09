
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

void june_evt_func_0001A304(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    ctx->gpr[31] = ctx->gpr[4] | 0x0;
    ctx->gpr[30] = ctx->gpr[3] | 0x0;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A5E0;
    ctx->gpr[5] = vm_read64(ctx->gpr[31] + 0x10);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[5], 0, 16, 16, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001A5E0;
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[5], 0, 28, 28, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A36C;
    ctx->gpr[3] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    ctx->gpr[4] = vm_read64(ctx->gpr[31] + 0x0);
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(0x647C); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->lr = 0x0001A360; june_evt_func_00824CF8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    goto loc_0001A604;
loc_0001A36C:
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[5], 0, 30, 31, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001A394;
    ctx->gpr[3] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    ctx->gpr[4] = vm_read64(ctx->gpr[31] + 0x0);
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(0x64C4); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->lr = 0x0001A388; june_evt_func_00824CF8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    goto loc_0001A604;
loc_0001A394:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x0001A39C; june_evt_func_008C21F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read64(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x24);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 8, 56);
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A41C;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A3EC;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A3D8;
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A5E0;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A484;
    goto loc_0001A494;
loc_0001A3D8:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A4A4;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A4B4;
    goto loc_0001A4C4;
loc_0001A3EC:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A408;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A4D4;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xA; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A4D4;
    goto loc_0001A4E4;
loc_0001A408:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A588;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xD; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A598;
    goto loc_0001A5A8;
loc_0001A41C:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x14; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A454;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A440;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5AC;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5BC;
    goto loc_0001A5CC;
loc_0001A440:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5CC;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x13; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5D0;
    goto loc_0001A5E0;
loc_0001A454:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A470;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x15; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5E0;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5E0;
    goto loc_0001A5D0;
loc_0001A470:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x1A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5E0;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x1B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A5E0;
    goto loc_0001A5E0;
loc_0001A484:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A48C; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A494:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A49C; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A4A4:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A4AC; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A4B4:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x178);
    ctx->lr = 0x0001A4BC; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A4C4:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A4CC; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A4D4:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x180);
    ctx->lr = 0x0001A4DC; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A4E4:
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x604);
    ctx->gpr[3] = vm_read16(ctx->gpr[5] + 0x14);
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A570;
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(0x204); ctx->gpr[29] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[27] = ctx->gpr[28] | 0x0;
loc_0001A504:
    ctx->gpr[3] = vm_read32(ctx->gpr[5] + 0x0);
    { uint64_t result = ctx->gpr[3] + ctx->gpr[28]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[3]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[4] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(5); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[4] = vm_read8(ctx->gpr[4] + 0x0);
    ctx->gpr[3] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[3], 3);
    { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[3] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[3]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    ctx->gpr[4] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[4], 3);
    { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[4] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[4]) ? (1u << 29) : 0); ctx->gpr[4] = result; }
    ctx->gpr[4] = ppc_rldicr(ctx->gpr[4], 4, 59);
    { uint64_t result = ctx->gpr[4] + ctx->gpr[3]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[4]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    ctx->gpr[3] = ppc_rldicr(ctx->gpr[3], 2, 61);
    { uint64_t result = ctx->gpr[3] + ctx->gpr[29]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[3]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->lr = 0x0001A554; june_evt_func_0001A304(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x604);
    { uint64_t _a = ctx->gpr[27]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[27] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[28]; uint64_t _r = _a + (uint64_t)(int64_t)(0xC); ctx->gpr[28] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = vm_read16(ctx->gpr[5] + 0x14);
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0001A504;
loc_0001A570:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x3B44);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A5E8;
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->lr = 0x0001A584; june_evt_func_0001A304(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_0001A5E8;
loc_0001A588:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x218);
    ctx->lr = 0x0001A590; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A598:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x188);
    ctx->lr = 0x0001A5A0; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A5A8:
    goto loc_0001A5E0;
loc_0001A5AC:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A5B4; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A5BC:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A5C4; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A5CC:
    goto loc_0001A5E0;
loc_0001A5D0:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x174);
    ctx->lr = 0x0001A5D8; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001A5E8;
loc_0001A5E0:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    goto loc_0001A604;
loc_0001A5E8:
    ctx->gpr[3] = (int64_t)(int32_t)((uint32_t)0x0 << 16);
    ctx->gpr[5] = vm_read64(ctx->gpr[31] + 0x10);
    ctx->gpr[4] = ctx->gpr[3] | 0x8000;
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[30]);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[4];
    vm_write64(ctx->gpr[31] + 0x10, ctx->gpr[4]);
loc_0001A604:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_0001A628(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[25]);
    ctx->gpr[24] = ctx->gpr[4] | 0x0;
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[31]);
    ctx->gpr[25] = ctx->gpr[3] | 0x0;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[20]);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001AD4C;
    ctx->gpr[5] = vm_read64(ctx->gpr[24] + 0x10);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[5], 0, 17, 17, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001AD4C;
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[5], 0, 28, 28, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A6B0;
    ctx->gpr[3] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    ctx->gpr[4] = vm_read64(ctx->gpr[24] + 0x0);
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(0x647C); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->lr = 0x0001A6A4; june_evt_func_00824CF8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | 0x0;
    goto loc_0001AD68;
loc_0001A6B0:
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[5], 0, 30, 31, 0, 0);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001A6D8;
    ctx->gpr[3] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    ctx->gpr[4] = vm_read64(ctx->gpr[24] + 0x0);
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(0x64C4); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->lr = 0x0001A6CC; june_evt_func_00824CF8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | 0x0;
    goto loc_0001AD68;
loc_0001A6D8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->lr = 0x0001A6E0; june_evt_func_008C21F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read64(ctx->gpr[24] + 0x0);
    ctx->gpr[30] = ctx->gpr[3] | 0x0;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 8, 56);
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A760;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A730;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A71C;
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001AD4C;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A7C8;
    goto loc_0001A8D0;
loc_0001A71C:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A8E8;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A964;
    goto loc_0001A97C;
loc_0001A730:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A74C;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A9A4;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xA; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001A9A4;
    goto loc_0001A9BC;
loc_0001A74C:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AA74;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xD; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001ACC8;
    goto loc_0001ACFC;
loc_0001A760:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x14; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A798;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A784;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD00;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD18;
    goto loc_0001AD30;
loc_0001A784:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD30;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x13; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD34;
    goto loc_0001AD4C;
loc_0001A798:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0001A7B4;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x15; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD4C;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD4C;
    goto loc_0001AD34;
loc_0001A7B4:
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x1A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD4C;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x1B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD4C;
    goto loc_0001AD4C;
loc_0001A7C8:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0001A7D8; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0xFD);
    ctx->lr = 0x0001A7E8; june_evt_func_008BD444(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ctx->gpr[24] | 0x0;
    ctx->gpr[29] = ctx->gpr[31] | 0x0;
loc_0001A7F4:
    { uint64_t _a = ctx->gpr[30]; uint64_t _r = _a + (uint64_t)(int64_t)(0x18C); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A82C;
    ctx->lr = 0x0001A80C; june_evt_func_00016768(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[28] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->lr = 0x0001A818; june_evt_func_008D5034(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[28] | 0x0;
    ctx->lr = 0x0001A828; june_evt_func_008D5F90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0001A82C:
    { uint64_t _a = ctx->gpr[30]; uint64_t _r = _a + (uint64_t)(int64_t)(0x194); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A868;
    ctx->lr = 0x0001A844; june_evt_func_00016768(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->gpr[28] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[4] | 0x0;
    ctx->lr = 0x0001A854; june_evt_func_008D5034(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[28] | 0x0;
    ctx->lr = 0x0001A864; june_evt_func_008D5F90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0001A868:
    { uint64_t _a = ctx->gpr[29]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[29] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[30]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[30] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0001A7F4;
    ctx->gpr[30] = ctx->gpr[31] | 0x0;
    ctx->gpr[31] = ctx->gpr[24] | 0x0;
loc_0001A880:
    { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(0x19C); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A8BC;
    ctx->lr = 0x0001A898; june_evt_func_00016768(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->gpr[29] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[4] | 0x0;
    ctx->lr = 0x0001A8A8; june_evt_func_008D5034(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[29] | 0x0;
    ctx->lr = 0x0001A8B8; june_evt_func_008D5F90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0001A8BC:
    { uint64_t _a = ctx->gpr[30]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[30] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[31] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0001A880;
    goto loc_0001AD54;
loc_0001A8D0:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->lr = 0x0001A8E0; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001A8E8:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->lr = 0x0001A8F8; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0xFD);
    ctx->lr = 0x0001A908; june_evt_func_008BD444(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ctx->gpr[31] | 0x0;
    ctx->gpr[31] = ctx->gpr[24] | 0x0;
loc_0001A914:
    { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(0x180); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001A950;
    ctx->lr = 0x0001A92C; june_evt_func_00016768(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->gpr[29] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[4] | 0x0;
    ctx->lr = 0x0001A93C; june_evt_func_008D5034(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[3] | 0x0;
    ctx->gpr[3] = ctx->gpr[29] | 0x0;
    ctx->lr = 0x0001A94C; june_evt_func_008D5F90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0001A950:
    { uint64_t _a = ctx->gpr[30]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[30] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[31] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0001A914;
    goto loc_0001AD54;
loc_0001A964:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x174);
    ctx->lr = 0x0001A974; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001A97C:
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->lr = 0x0001A98C; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0xFD);
    ctx->lr = 0x0001A99C; june_evt_func_008BD444(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001A9A4:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x17C);
    ctx->lr = 0x0001A9B4; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001A9BC:
    ctx->gpr[5] = vm_read32(ctx->gpr[24] + 0x604);
    ctx->gpr[3] = vm_read16(ctx->gpr[5] + 0x14);
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AA44;
    { uint64_t _a = ctx->gpr[24]; uint64_t _r = _a + (uint64_t)(int64_t)(0x204); ctx->gpr[30] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[29] = ctx->gpr[31] | 0x0;
loc_0001A9D8:
    ctx->gpr[3] = vm_read32(ctx->gpr[5] + 0x0);
    { uint64_t result = ctx->gpr[3] + ctx->gpr[31]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[3]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[4] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[3]; uint64_t _r = _a + (uint64_t)(int64_t)(5); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read8(ctx->gpr[4] + 0x0);
    ctx->gpr[3] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[4] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[4], 3);
    { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[4] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[4]) ? (1u << 29) : 0); ctx->gpr[4] = result; }
    ctx->gpr[3] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[3], 3);
    ctx->gpr[4] = ppc_rldicr(ctx->gpr[4], 4, 59);
    { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[3] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[3]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    { uint64_t result = ctx->gpr[4] + ctx->gpr[3]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[4]) ? (1u << 29) : 0); ctx->gpr[3] = result; }
    ctx->gpr[4] = ppc_rldicr(ctx->gpr[3], 2, 61);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    { uint64_t result = ctx->gpr[4] + ctx->gpr[30]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[4]) ? (1u << 29) : 0); ctx->gpr[4] = result; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->lr = 0x0001AA28; june_evt_func_0001A628(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[5] = vm_read32(ctx->gpr[24] + 0x604);
    { uint64_t _a = ctx->gpr[29]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[29] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(0xC); ctx->gpr[31] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = vm_read16(ctx->gpr[5] + 0x14);
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0001A9D8;
loc_0001AA44:
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x3B44);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001AA58;
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    ctx->lr = 0x0001AA58; june_evt_func_0001A628(ctx); DRAIN_TRAMPOLINE(ctx);
loc_0001AA58:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0xABC);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001AD54;
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0001AA6C; june_evt_func_003CFED0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001AA74:
    ctx->gpr[3] = vm_read16(ctx->gpr[24] + 0x1B8);
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 30) & 1) == 0)) goto loc_0001AD54;
    ctx->gpr[22] = ctx->gpr[24] | 0x0;
    ctx->gpr[29] = ctx->gpr[31] | 0x0;
    ctx->gpr[21] = ctx->gpr[22] | 0x0;
loc_0001AA90:
    { uint64_t _a = ctx->gpr[21]; uint64_t _r = _a + (uint64_t)(int64_t)(0x1F8); ctx->gpr[28] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001AABC;
    ctx->gpr[4] = vm_read64(ctx->gpr[24] + 0x10);
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[25]);
    ctx->gpr[3] = ctx->gpr[31] | 0x0;
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0x1 << 16);
    vm_write64(ctx->gpr[24] + 0x10, ctx->gpr[4]);
    goto loc_0001AD68;
loc_0001AABC:
    { uint64_t _a = ctx->gpr[22]; uint64_t _r = _a + (uint64_t)(int64_t)(0x1C4); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[6] = vm_read64(ctx->gpr[24] + 0x0);
    { uint64_t _a = ctx->gpr[1]; uint64_t _r = _a + (uint64_t)(int64_t)(0x70); ctx->gpr[27] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[6], 48, 48);
    ctx->gpr[4] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[6], 32, 48);
    ctx->gpr[3] = ctx->gpr[27] | 0x0;
    ctx->gpr[8] = vm_read16(ctx->gpr[5] + 0x0);
    { uint64_t _a = ctx->gpr[4]; uint64_t _r = _a + (uint64_t)(int64_t)(0x6B1C); ctx->gpr[4] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[7];
    ctx->gpr[6] = ppc_rlw64(ctx->gpr[6], 0, 24, 31, 0, 0);
    ctx->gpr[7] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->lr = 0x0001AAF4; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[4] = ctx->gpr[27] | 0x0;
    ctx->lr = 0x0001AB04; june_evt_func_008BD624(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001AB24;
    { uint64_t _a = ctx->gpr[1]; uint64_t _r = _a + (uint64_t)(int64_t)(0x70); ctx->gpr[5] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0001AB20; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0001AB24:
    { uint64_t _a = ctx->gpr[21]; uint64_t _r = _a + (uint64_t)(int64_t)(0x29C); ctx->gpr[28] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[26] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[27] = (int64_t)(int32_t)((uint32_t)0xDF << 16);
    { uint64_t _a = ctx->gpr[26]; uint64_t _r = _a + (uint64_t)(int64_t)(0x6368); ctx->gpr[26] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[27]; uint64_t _r = _a + (uint64_t)(int64_t)(0x6318); ctx->gpr[27] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[23] = vm_read32(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001ABB0;
    ctx->gpr[20] = ctx->gpr[31] | 0x0;
loc_0001AB4C:
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 0)) goto loc_0001ABB0;
    ctx->gpr[5] = vm_read32(ctx->gpr[23] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001AB7C;
    ctx->gpr[5] = ctx->gpr[26] | 0x0;
    ctx->gpr[3] = ctx->gpr[27] | 0x0;
    ctx->gpr[4] = (int64_t)(int32_t)(0x45);
    ctx->lr = 0x0001AB74; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read32(ctx->gpr[23] + 0xC);
loc_0001AB7C:
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x4);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    ctx->gpr[4] = (uint64_t)((int64_t)(int32_t)ctx->gpr[20] * (int64_t)(int32_t)ctx->gpr[4]);
    { uint64_t result = ctx->gpr[5] + ctx->gpr[4]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[5]) ? (1u << 29) : 0); ctx->gpr[4] = result; }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->lr = 0x0001AB98; june_evt_func_0001A628(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    ctx->lr = 0x0001ABA4; june_evt_func_0001A304(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[23] = vm_read32(ctx->gpr[28] + 0x0);
    { uint64_t _a = ctx->gpr[20]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[20] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    goto loc_0001AB4C;
loc_0001ABB0:
    { uint64_t _a = ctx->gpr[21]; uint64_t _r = _a + (uint64_t)(int64_t)(0x2BC); ctx->gpr[28] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[23] = vm_read32(ctx->gpr[28] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001AC2C;
    ctx->gpr[20] = ctx->gpr[31] | 0x0;
loc_0001ABC8:
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 0)) goto loc_0001AC2C;
    ctx->gpr[5] = vm_read32(ctx->gpr[23] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001ABF8;
    ctx->gpr[5] = ctx->gpr[26] | 0x0;
    ctx->gpr[3] = ctx->gpr[27] | 0x0;
    ctx->gpr[4] = (int64_t)(int32_t)(0x45);
    ctx->lr = 0x0001ABF0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read32(ctx->gpr[23] + 0xC);
loc_0001ABF8:
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x4);
    ctx->gpr[4] = (uint64_t)((int64_t)(int32_t)ctx->gpr[20] * (int64_t)(int32_t)ctx->gpr[4]);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    { uint64_t result = ctx->gpr[5] + ctx->gpr[4]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[5]) ? (1u << 29) : 0); ctx->gpr[4] = result; }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->lr = 0x0001AC14; june_evt_func_0001A628(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    ctx->lr = 0x0001AC20; june_evt_func_0001A304(ctx); DRAIN_TRAMPOLINE(ctx);
    { uint64_t _a = ctx->gpr[20]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[20] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[23] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_0001ABC8;
loc_0001AC2C:
    { uint64_t _a = ctx->gpr[21]; uint64_t _r = _a + (uint64_t)(int64_t)(0x2DC); ctx->gpr[23] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[20] = vm_read32(ctx->gpr[23] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001ACA8;
    ctx->gpr[28] = ctx->gpr[31] | 0x0;
loc_0001AC44:
    ctx->gpr[3] = vm_read32(ctx->gpr[20] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 0)) goto loc_0001ACA8;
    ctx->gpr[5] = vm_read32(ctx->gpr[20] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0001AC74;
    ctx->gpr[5] = ctx->gpr[26] | 0x0;
    ctx->gpr[3] = ctx->gpr[27] | 0x0;
    ctx->gpr[4] = (int64_t)(int32_t)(0x45);
    ctx->lr = 0x0001AC6C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read32(ctx->gpr[20] + 0xC);
loc_0001AC74:
    ctx->gpr[4] = vm_read32(ctx->gpr[20] + 0x4);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    ctx->gpr[4] = (uint64_t)((int64_t)(int32_t)ctx->gpr[28] * (int64_t)(int32_t)ctx->gpr[4]);
    { uint64_t result = ctx->gpr[5] + ctx->gpr[4]; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[5]) ? (1u << 29) : 0); ctx->gpr[4] = result; }
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->lr = 0x0001AC90; june_evt_func_0001A628(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[3] = ctx->gpr[25] | 0x0;
    ctx->lr = 0x0001AC9C; june_evt_func_0001A304(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[20] = vm_read32(ctx->gpr[23] + 0x0);
    { uint64_t _a = ctx->gpr[28]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[28] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    goto loc_0001AC44;
loc_0001ACA8:
    ctx->gpr[3] = vm_read16(ctx->gpr[24] + 0x1B8);
    { uint64_t _a = ctx->gpr[29]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[29] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    { uint64_t _a = ctx->gpr[22]; uint64_t _r = _a + (uint64_t)(int64_t)(2); ctx->gpr[22] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { uint64_t _a = ctx->gpr[21]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[21] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0001AA90;
    goto loc_0001AD54;
loc_0001ACC8:
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x180);
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0001ACD8; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x184);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0001AD54;
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0001ACF4; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001ACFC:
    goto loc_0001AD4C;
loc_0001AD00:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->lr = 0x0001AD10; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001AD18:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->lr = 0x0001AD28; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001AD30:
    goto loc_0001AD4C;
loc_0001AD34:
    ctx->gpr[3] = ctx->gpr[30] | 0x0;
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x170);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0001AD44; june_evt_func_008BF338(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0001AD54;
loc_0001AD4C:
    ctx->gpr[3] = ctx->gpr[31] | 0x0;
    goto loc_0001AD68;
loc_0001AD54:
    ctx->gpr[4] = vm_read64(ctx->gpr[24] + 0x10);
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[25]);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->gpr[4] = ctx->gpr[4] | 0x4000;
    vm_write64(ctx->gpr[24] + 0x10, ctx->gpr[4]);
loc_0001AD68:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
}

void june_evt_func_008E049C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x008E04BC; june_evt_func_0087694C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[31] + (int64_t)(0xB8);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x22);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0548;
loc_008E04DC:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008E04E4; june_evt_func_00834F80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E052C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[29];
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_008E0520;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    { g_trampoline_fn = (void(*)(void*))june_evt_func_008E000C; return; }
loc_008E0520:
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x49E8);
    ctx->lr = 0x008E0528; june_evt_func_00876CB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E052C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_008E0548:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x49EC);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x49F0);
    ctx->lr = 0x008E0554; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008E04DC;
}

void june_evt_func_008BD624(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BD6D4;
loc_008BD64C:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x7C);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BD6B4;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008BD664; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008BD674; june_evt_func_00947720(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ;
loc_008BD680:
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(4);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008BD694; june_evt_func_00878E6C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BD6B4;
    ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BD680;
loc_008BD6B4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_008BD6D4:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x529C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1D7);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x52EC);
    ctx->lr = 0x008BD6E4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008BD64C;
}

void june_evt_func_008E000C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x10);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 20, 20, 0, 0);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E03C4;
loc_008E0044:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0070;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0058; june_evt_func_008E1AC0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0070;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->lr = 0x008E006C; june_evt_func_008E5258(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0070:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x008E0080; june_evt_func_008F38E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xA8);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E00F4;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E00E8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[28] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E00E8;
    ctx->gpr[29] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[31] = (int64_t)(int32_t)(0);
loc_008E00B8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(4);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x008E00D0; june_evt_func_008E9494(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E00B8;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xA8);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E00F4;
loc_008E00E8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E00F0; june_evt_func_008C4670(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E00F4:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xAC);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0xA8, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E019C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E013C;
loc_008E0118:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->lr = 0x008E012C; june_evt_func_008DF748(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E0118;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xAC);
loc_008E013C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0190;
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0184;
    ;
loc_008E0160:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->lr = 0x008E0170; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E0160;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xAC);
loc_008E0184:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0x8, ctx->gpr[0]);
loc_008E0190:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0198; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E019C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x3C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E01B4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E01B0; june_evt_func_0088B650(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E01B4:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E01CC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E01C8; june_evt_func_0088DE70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E01CC:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E01E4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E01E0; june_evt_func_008E29D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E01E4:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E01FC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E01F8; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E01FC:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0214;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0210; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0214:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E022C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0228; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E022C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0244;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0240; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0244:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E025C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0258; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E025C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0274;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0270; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0274:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E028C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0288; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E028C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E02A4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E02A0; june_evt_func_0094C3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E02A4:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E02BC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E02B8; june_evt_func_0089D250(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E02BC:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E02D4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E02D0; june_evt_func_0089D250(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E02D4:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E02EC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E02E8; june_evt_func_0089D250(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E02EC:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0304;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0300; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0304:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x78);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E031C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0318; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E031C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0334;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0330; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0334:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xA0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E034C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0348; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E034C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xA4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0364;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0360; june_evt_func_0097E008(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0364:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xB0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E037C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0378; june_evt_func_00875AEC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E037C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xB4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E0394;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008E0390; june_evt_func_00915460(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008E0394:
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x008E039C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
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
loc_008E03C4:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4890);
    ctx->gpr[4] = (int64_t)(int32_t)(0x54);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4AE0);
    ctx->lr = 0x008E03D4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008E0044;
}

void june_evt_func_00915460(ppu_context* ctx) {
    PPU_MM_SNAP
    return;
}

void june_evt_func_0094C3AC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0094C3C8; june_evt_func_0087694C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094C408;
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2AE8);
    ctx->lr = 0x0094C3EC; june_evt_func_00876CB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_0094C408:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->lr = 0x0094C410; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->lr = 0x0094C41C; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->lr = 0x0094C428; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0094C434; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0097E008(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[3] + (int64_t)(8);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x22);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0097E094;
loc_0097E040:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0097E048; june_evt_func_00834F80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0097E078;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->lr = 0x0097E05C; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->lr = 0x0097E068; june_evt_func_008772D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0097E074; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0097E078:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0097E094:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x1C9C);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x1CA0);
    ctx->lr = 0x0097E0A0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0097E040;
}
