
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

void june_evt_func_0087AC30(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x0087AC48; june_evt_func_0087A928(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write8(ctx->gpr[29] + 0x3, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    return;
}


void june_evt_func_0084C000(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[3] + 0x28);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[31] = vm_read32(ctx->gpr[4] + 0x20);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C14C;
loc_0084C044:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C114;
loc_0084C058:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084C058;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0084C074:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x34);
    vm_write32(ctx->gpr[28] + 0x34, ctx->gpr[31]);
    vm_write32(ctx->gpr[29] + 0x34, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C130;
loc_0084C088:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C124;
loc_0084C098:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084C098;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0084C0B0:
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x44);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[27] + 0x34, ctx->gpr[30]);
    vm_write32(ctx->gpr[26] + 0x34, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C0EC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0084C0E8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_0084C0EC:
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
loc_0084C114:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0084C074;
loc_0084C124:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0084C0B0;
loc_0084C130:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4DC);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68E8);
    ctx->lr = 0x0084C140; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[27] + 0x28);
    goto loc_0084C088;
loc_0084C14C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4CB);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68E8);
    ctx->lr = 0x0084C15C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[27] + 0x28);
    goto loc_0084C044;
}


void june_evt_func_0088A110(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x5C1C);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(0x3D);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x20);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x0088A140; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x3E);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A198;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x0088A168; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x1C);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0088A178; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0088A198:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C04);
    ctx->lr = 0x0088A1A0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x0088A1B0; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x1C);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0088A1C0; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_0089ACC4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x59D8);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(0x94);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x30);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x0089ACF4; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x95);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089AD54;
loc_0089AD10:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x30);
    ctx->lr = 0x0089AD1C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x24);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5984); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
    ctx->lr = 0x0089AD34; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0089AD54:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5988);
    ctx->lr = 0x0089AD5C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0089AD10;
}


void june_evt_func_0084C668(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[19] = ctx->gpr[8] | ctx->gpr[8];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[31]);
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[30] = ctx->gpr[8] | ctx->gpr[8];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[27]);
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[20] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[21] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[23] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[26] = ctx->gpr[6] | ctx->gpr[6];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084C764;
loc_0084C704:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_0084C764:
    ctx->lr = 0x0084C768; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084C7E8;
loc_0084C790:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C704;
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C704;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[19], 0, 32);
loc_0084C7B8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0084C7D8; june_evt_func_0084C668(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084C7B8;
    goto loc_0084C704;
loc_0084C7E8:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[17] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    ctx->gpr[18] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->lr = 0x0084C808; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[4] = ctx->gpr[25] + (int64_t)(4);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0084C874; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0084C89C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084CA68;
loc_0084C8AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x1C);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C8F4;
    ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0084C8E4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084CA94;
loc_0084C8F4:
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084CAD8;
loc_0084C900:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084CAC0;
loc_0084C910:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_0084C920:
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x14);
    ctx->gpr[6] = ppc_rlw64(ctx->gpr[28], 6, 0, 25, 0, 0);
    ctx->gpr[31] = ctx->gpr[28] | ctx->gpr[28];
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[28], 2, 0, 29, 0, 0);
    ctx->gpr[8] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[9] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084CA40;
loc_0084C954:
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x18);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[11];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[7] = vm_read32(ctx->gpr[10] + 0xC);
    ctx->gpr[7] = ctx->gpr[6] + ctx->gpr[7];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0084C998; june_evt_func_009425C8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(1);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0084C9D4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084C920;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[31] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->lr = 0x0084CA0C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[6] = ppc_rlw64(ctx->gpr[28], 6, 0, 25, 0, 0);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[28], 2, 0, 29, 0, 0);
    ctx->gpr[8] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[9] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084C954;
loc_0084CA40:
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x0);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[31], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
    goto loc_0084C790;
loc_0084CA68:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0084CA8C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084C8AC;
loc_0084CA94:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0084CAB8; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084C8F4;
loc_0084CAC0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0xA61);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68C0);
    ctx->lr = 0x0084CAD0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084C910;
loc_0084CAD8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0xA60);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68C4);
    ctx->lr = 0x0084CAE8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x8);
    goto loc_0084C900;
}


void june_evt_func_0084CAF4(ppu_context* ctx) {
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
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0x220, ctx->gpr[1]); ctx->gpr[1] += -0x220;
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x178, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x188, ctx->gpr[29]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[16], 8); vm_write64(ctx->gpr[1] + 0x1A0, tmp); }
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[17], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    ctx->gpr[16] = ctx->gpr[8] | ctx->gpr[8];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[18], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[27] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[19], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0x1C8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x1D0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x1D8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x1E0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x1F0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x1F8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x200, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x208, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x210, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x218, tmp); }
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x198, ctx->gpr[31]);
    ctx->gpr[0] = ctx->lr;
    vm_write32(ctx->gpr[1] + 0x228, ctx->gpr[12]);
    vm_write64(ctx->gpr[1] + 0x230, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[4]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[5]);
    vm_write32(ctx->gpr[1] + 0xB8, ctx->gpr[4]);
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[5]);
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[7]);
    vm_write32(ctx->gpr[1] + 0xDC, ctx->gpr[8]);
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084CC74;
loc_0084CBD4:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x230);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x228);
    ctx->gpr[14] = _cs_14;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[15] = _cs_15;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A0); memcpy(&ctx->fpr[16], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A8); memcpy(&ctx->fpr[17], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[18], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B8); memcpy(&ctx->fpr[19], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C0); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C8); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D0); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D8); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E0); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1E8); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1F0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1F8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x200); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x208); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x210); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x218); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x220);
    return;
loc_0084CC74:
    ctx->lr = 0x0084CC78; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DA10;
loc_0084CC8C:
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x102C);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x1030);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D060;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)(1);
    ctx->fpr[30] = ctx->fpr[20];
    ctx->gpr[28] = (int64_t)(int32_t)(1);
loc_0084CCB4:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[13] = ctx->fpr[0];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[28]));
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[0]);
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[23], ctx->fpr[1]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[18] = (uint64_t)((int64_t)(int32_t)ctx->gpr[10] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 0x1F);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0084CD44; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[30]));
    ctx->lr = 0x0084CD78; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[23], ctx->fpr[0]));
    ctx->lr = 0x0084CD90; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[28];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[30]));
    ctx->lr = 0x0084CDC8; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084CE54;
    { double a = ctx->fpr[0]; double b = ctx->fpr[25]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084D084;
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0xC8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D09C;
loc_0084CE24:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[11] + (int64_t)(4);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ctx->gpr[0] | ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[0]);
loc_0084CE40:
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0084CE4C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_0084CE54:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xF8);
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    ctx->lr = 0x0084CE84; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D91C;
loc_0084CE94:
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[3] = ctx->fpr[20];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->gpr[8] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68A4);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[20]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[30]));
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[22]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[20]));
    ctx->lr = 0x0084CEDC; june_evt_func_008683B0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0084CEFC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D8EC;
loc_0084CF0C:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0084CF24; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D8BC;
loc_0084CF34:
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[3] = ctx->fpr[20];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[13]));
    ctx->gpr[8] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68A0);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[20]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[30]));
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[22]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[20]));
    ctx->lr = 0x0084CF74; june_evt_func_008683B0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0084CF94; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D88C;
loc_0084CFA4:
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[20]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0084CFD8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D85C;
loc_0084CFE8:
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xD0);
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[16] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D0C8;
loc_0084D008:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084CBD4;
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[5] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084CBD4;
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0xB8);
    ctx->gpr[30] = vm_read32(ctx->gpr[1] + 0xD8);
    ctx->gpr[31] = vm_read32(ctx->gpr[1] + 0xDC);
loc_0084D030:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0084D050; june_evt_func_0084CAF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D030;
    goto loc_0084CBD4;
loc_0084D060:
    ctx->gpr[10] = ctx->gpr[0] | ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[28] = ctx->gpr[10] | ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    goto loc_0084CCB4;
loc_0084D084:
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0xC8);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084CE24;
loc_0084D09C:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084CE24;
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(8);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ctx->gpr[10] | ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[10]);
    goto loc_0084CE40;
loc_0084D0C8:
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x102C);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->lr = 0x0084D0EC; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xC0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x1030);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[10], (uint32_t)tmp); }
    ctx->gpr[29] = vm_read32(ctx->gpr[1] + 0xB0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D114;
    vm_write32(ctx->gpr[1] + 0xB4, ctx->gpr[0]);
loc_0084D114:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->fpr[11] = ctx->fpr[12];
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xD0);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[28]));
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0xB4);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[11], 0); memcpy(&ctx->fpr[11], &iv, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[30]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[11], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xB0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (uint64_t)((int64_t)(int32_t)ctx->gpr[10] * (int64_t)(int32_t)ctx->gpr[0]);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 0x1F);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[25]));
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint32_t)ctx->gpr[8]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[11] >> 32));
    ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 3);
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[10];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 4, 0, 27, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[11];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084D2C4;
    ctx->fpr[13] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xB4);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 2, 0, 29, 0, 0);
    ctx->gpr[30] = ctx->gpr[9] + ctx->gpr[8];
    ctx->gpr[28] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[31] = ctx->gpr[30] | ctx->gpr[30];
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084D224;
    goto loc_0084D2C4;
loc_0084D1F4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB4);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[30] = ctx->gpr[30] + ctx->gpr[0];
    ctx->gpr[31] = ctx->gpr[0] + ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084D2C0;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084D2C0;
loc_0084D224:
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[8]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    { uint32_t _b=(uint32_t)ctx->gpr[28]; ctx->gpr[0] = (uint32_t)(_b==0?0:(uint32_t)ctx->gpr[8]/_b); }
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[0] * (int64_t)(int32_t)ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D270;
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0084D270;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[10] + (int64_t)(0x14);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0084D268; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ;
loc_0084D270:
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = 0x0084D280; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D1F4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->lr = 0x0084D2B8; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084D1F4;
loc_0084D2C0:
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_0084D2C4:
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084DA28;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xB4);
    ctx->gpr[0] = ctx->gpr[0] | 0x2493;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x689C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 0x1F);
    ctx->fpr[28] = ppu_frsp(ctx->fpr[10]);
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[11] * (int64_t)(int32_t)ctx->gpr[0] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 3);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[9];
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[29] = ppu_frsp(ctx->fpr[0]);
loc_0084D340:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[28]));
    ctx->lr = 0x0084D36C; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[4] = ctx->gpr[31] + (int64_t)(8);
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF8);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[25], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0084D3C8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D340;
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_0084D3DC:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xB0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->fpr[28] = ctx->fpr[30];
    ctx->gpr[0] = ctx->gpr[0] | 0x2493;
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x102C);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0xC);
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
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[26] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0084D44C; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
    ctx->gpr[17] = vm_read32(ctx->gpr[3] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D488;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x955);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6898);
    ctx->lr = 0x0084D47C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
loc_0084D488:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D7D8;
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0xB4);
    ctx->gpr[14] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF8);
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0xA0);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[11]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6894); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[18] = ftmp; }
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x688C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[17] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[16] = ftmp; }
    ;
loc_0084D4B8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->lr = 0x0084D4C8; june_evt_func_008946F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D698;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE8);
    ctx->fpr[24] = ctx->fpr[18];
    ctx->fpr[29] = ctx->fpr[17];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->fpr[19] = ctx->fpr[16];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[19] = vm_read32(ctx->gpr[2] + -0x6880);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[9]);
    ctx->fpr[27] = ppu_frsp(ctx->fpr[0]);
    goto loc_0084D5C0;
loc_0084D510:
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xE0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 2, 0, 29, 0, 0);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ctx->gpr[10] + (int64_t)(0x18);
    ctx->gpr[5] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0084D540; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[29]));
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[22]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->lr = 0x0084D584; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D5A8;
    if ((((ctx->cr >> 14) & 1) == 1)) goto loc_0084D66C;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084D6F8;
    ;
loc_0084D5A8:
    ctx->gpr[9] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0084D698;
loc_0084D5BC:
    ctx->gpr[30] = ctx->gpr[30] + ctx->gpr[21];
loc_0084D5C0:
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[26] = ppc_rlw64(ctx->gpr[27], 2, 0, 29, 0, 0);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[31]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[25] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[26] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[23], ctx->fpr[1]));
    ctx->lr = 0x0084D5EC; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[27]));
    ctx->lr = 0x0084D620; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[24]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 14) & 1) == 0)) goto loc_0084D510;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x963);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6890);
    ctx->lr = 0x0084D664; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084D510;
loc_0084D66C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x97D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6888);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    ctx->lr = 0x0084D680; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0084D5BC;
    ;
loc_0084D698:
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[25]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[20]));
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xF8);
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0084D6CC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0084D79C;
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[14] = ctx->gpr[14] + (int64_t)(1);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[14]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0084D7D8;
loc_0084D6F0:
    ctx->gpr[17] = ctx->gpr[17] + (int64_t)(0x30);
    goto loc_0084D4B8;
loc_0084D6F8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[20]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[9] = ctx->gpr[26] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[20]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[23], ctx->fpr[1]));
    ctx->lr = 0x0084D734; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[27]));
    ctx->lr = 0x0084D768; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[19], ctx->fpr[30], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0084D794; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084D5A8;
loc_0084D79C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[14] = ctx->gpr[14] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x687C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->lr = 0x0084D7C4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[14]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084D6F0;
loc_0084D7D8:
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xC0);
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D830;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xB8);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = vm_read32(ctx->gpr[1] + 0xB4);
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x6884);
    ctx->gpr[31] = ctx->gpr[9] + (int64_t)(0x10);
    ;
loc_0084D808:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0xC8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084D94C;
loc_0084D81C:
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0084D808;
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x14);
loc_0084D830:
    ctx->gpr[11] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[25], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[16] + 0x0, tmp); }
    goto loc_0084D008;
loc_0084D85C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->lr = 0x0084D884; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084CFE8;
loc_0084D88C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->lr = 0x0084D8B4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084CFA4;
loc_0084D8BC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->lr = 0x0084D8E4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084CF34;
loc_0084D8EC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->lr = 0x0084D914; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084CF0C;
loc_0084D91C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x68C8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->lr = 0x0084D944; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084CE94;
loc_0084D94C:
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[23], ctx->fpr[1]));
    ctx->lr = 0x0084D964; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[18];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[0]));
    ctx->lr = 0x0084D9A8; june_evt_func_00AEC35C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[16] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[25], ctx->fpr[31], 0, 0));
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF8);
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[25], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0084DA00; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x1010);
    goto loc_0084D81C;
loc_0084DA10:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x8D9);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68B8);
    ctx->lr = 0x0084DA20; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0084CC8C;
loc_0084DA28:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x689C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    goto loc_0084D3DC;
}


void june_evt_func_00947BAC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[31]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x98, tmp); }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xA0, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_00947BE8:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[1] = ctx->fpr[31];
    ctx->lr = 0x00947BF8; june_evt_func_00947A08(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[29]));
    ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0x2C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(2);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[9] + 0x24, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947BE8;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
}


void june_evt_func_00947778(ppu_context* ctx) {
    PPU_MM_SNAP
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[1]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[13]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[12]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[11]));
    ctx->fpr[10] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[10]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[9]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[10], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[1], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x0, tmp); }
    return;
}


void june_evt_func_00947DF4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[31]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x98, tmp); }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xA0, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_00947E30:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[1] = ctx->fpr[31];
    ctx->lr = 0x00947E40; june_evt_func_00947C4C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[29]));
    ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0x2C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(2);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[9] + 0x34, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947E30;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
}


void june_evt_func_009478C0(ppu_context* ctx) {
    PPU_MM_SNAP
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[1]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[13]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[12]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[11]));
    ctx->fpr[10] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[10]));
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[9]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[10], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[1], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x0, tmp); }
    return;
}


void june_evt_func_0093B4FC(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->gpr[31] = vm_read32(ctx->gpr[3] + 0x88);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[25] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[23] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[26] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B67C;
loc_0093B554:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B5B0;
loc_0093B55C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->lr = 0x0093B578; june_evt_func_008714AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_0093B5B0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x34);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B5F8;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    ctx->gpr[31] = ctx->gpr[10] + (int64_t)(-1);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B5E0;
    goto loc_0093B654;
loc_0093B5DC:
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0093B654;
loc_0093B5E0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(-1);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B5DC;
loc_0093B5F8:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B6B0;
loc_0093B604:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B694;
loc_0093B618:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[27]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B640;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1BA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E6C);
    ctx->lr = 0x0093B638; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x48);
loc_0093B640:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093B64C; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093B55C;
loc_0093B654:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093B6CC;
loc_0093B65C:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B674;
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write32(ctx->gpr[29] + 0x2C, ctx->gpr[0]);
loc_0093B674:
    vm_write32(ctx->gpr[29] + 0x30, ctx->gpr[31]);
    goto loc_0093B5F8;
loc_0093B67C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0x19F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E7C);
    ctx->lr = 0x0093B68C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093B554;
loc_0093B694:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1B9);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E70);
    ctx->lr = 0x0093B6A4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x48);
    goto loc_0093B618;
loc_0093B6B0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1B8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E74);
    ctx->lr = 0x0093B6C0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x48);
    goto loc_0093B604;
loc_0093B6CC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1AF);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E78);
    ctx->lr = 0x0093B6DC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093B65C;
}


void june_evt_func_0086E3D4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6268);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[3] + (int64_t)(4);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[3], (uint32_t)tmp); }
    ctx->lr = 0x0086E40C; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[0];
    vm_write32(ctx->gpr[29] + 0x8, ctx->gpr[3]);
    ctx->lr = 0x0086E424; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[3]);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_00848F4C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00848F9C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->gpr[4] = (int64_t)(int32_t)(0x264);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6944);
    ctx->lr = 0x00848F98; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00848F9C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 4, 0, 27, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 6, 0, 25, 0, 0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[30] = ctx->gpr[29] + ctx->gpr[27];
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00848FD4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00848FCC; june_evt_func_00848E50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[0]);
loc_00848FD4:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00848FF4;
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00848FF4; june_evt_func_00849090(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00848FF4:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849064;
    ctx->gpr[9] = ctx->gpr[28] + (int64_t)(0x74);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00849024;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6938);
    ctx->gpr[4] = (int64_t)(int32_t)(0x273);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->lr = 0x00849020; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00849024:
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write16((ctx->gpr[29] + ctx->gpr[27]), ctx->gpr[0]);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[26]);
    ctx->lr = 0x00849038; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
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
loc_00849064:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    vm_write16((ctx->gpr[29] + ctx->gpr[27]), ctx->gpr[28]);
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}


void june_evt_func_00849090(ppu_context* ctx) {
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
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[23] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008490F0;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->gpr[4] = (int64_t)(int32_t)(0x22A);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6944);
    ctx->lr = 0x008490EC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008490F0:
    ctx->gpr[25] = ppc_rlw64(ctx->gpr[28], 4, 0, 27, 0, 0);
    ctx->gpr[26] = ppc_rlw64(ctx->gpr[28], 6, 0, 25, 0, 0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[11] = ctx->gpr[26] - ctx->gpr[25];
    ctx->gpr[0] = ctx->gpr[11] + (int64_t)(0x10);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[31] = ctx->gpr[30] + ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084916C;
    ctx->gpr[0] = ctx->gpr[11] + (int64_t)(0x20);
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[3] = vm_read64((ctx->gpr[30] + ctx->gpr[27]));
    ctx->lr = 0x00849128; june_evt_func_0086E280(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0084915C;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[11] + 0x48, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 25, 1, 31, 0, 0);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 7, 0, 31, 0, 0);
    vm_write32(ctx->gpr[11] + 0x14, ctx->gpr[0]);
    ctx->lr = 0x00849158; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0084915C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    vm_write64((ctx->gpr[30] + ctx->gpr[27]), ctx->gpr[9]);
loc_0084916C:
    ctx->gpr[31] = ctx->gpr[26] - ctx->gpr[25];
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[26] = ctx->gpr[30] + ctx->gpr[27];
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00849198;
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00849198; june_evt_func_00848F4C(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00849198:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008492A0;
    ctx->gpr[9] = ctx->gpr[24] + (int64_t)(0x98);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008491C8;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->gpr[4] = (int64_t)(int32_t)(0x23F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6934);
    ctx->lr = 0x008491C4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008491C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x3C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008492EC;
loc_008491D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008491F4;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->gpr[4] = (int64_t)(int32_t)(0x241);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x692C);
    ctx->lr = 0x008491F0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008491F4:
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[5], 0, 20, 20, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00849284;
loc_00849204:
    ctx->gpr[9] = ctx->gpr[31] + (int64_t)(0x20);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x3C);
    ctx->gpr[6] = vm_read64(ctx->gpr[29] + 0x90);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[5] = ctx->gpr[5] | 0x40;
    ctx->gpr[10] = ctx->gpr[30] + ctx->gpr[9];
    ctx->gpr[8] = vm_read32(ctx->gpr[11] + 0x1C);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[24]);
    vm_write16((ctx->gpr[30] + ctx->gpr[27]), ctx->gpr[0]);
    vm_write32(ctx->gpr[29] + 0x48, ctx->gpr[23]);
    ctx->gpr[7] = vm_read32(ctx->gpr[10] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[8] + 0x20);
    vm_write32(ctx->gpr[29] + 0x14, ctx->gpr[5]);
    vm_write64((ctx->gpr[30] + ctx->gpr[9]), ctx->gpr[6]);
    vm_write32(ctx->gpr[7] + 0x20, ctx->gpr[11]);
    ctx->lr = 0x0084924C; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00849284:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6928);
    ctx->gpr[4] = (int64_t)(int32_t)(0x242);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->lr = 0x00849294; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0x14);
    goto loc_00849204;
loc_008492A0:
    ctx->gpr[9] = ctx->gpr[31] + (int64_t)(0x20);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    vm_write16((ctx->gpr[30] + ctx->gpr[27]), ctx->gpr[24]);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[23] = _cs_23;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[9];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[31] = _cs_31;
    vm_write32(ctx->gpr[11] + 0x20, ctx->gpr[24]);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_008492EC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6948);
    ctx->gpr[4] = (int64_t)(int32_t)(0x240);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6930);
    ctx->lr = 0x008492FC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008491D4;
}


void june_evt_func_0086E098(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x40);
    return;
}


void june_evt_func_00879AF8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xB0, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[9] = vm_read8(ctx->gpr[3] + 0x1);
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00879C98;
    ctx->gpr[11] = vm_read8(ctx->gpr[3] + 0x2);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00879DA8;
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00879CB0;
loc_00879B50:
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00879DEC;
    ;
loc_00879B60:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x605C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[1]));
    ctx->lr = 0x00879B84; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[28] + 0x4, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00879DB8;
    ctx->gpr[9] = ctx->gpr[30] - ctx->gpr[31];
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00879CC8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6044); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[1]));
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[31]));
    ctx->lr = 0x00879BD8; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[28] + 0x2, ctx->gpr[0]);
    ctx->gpr[3] = vm_read8(ctx->gpr[29] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00879D1C;
loc_00879BFC:
    ctx->gpr[0] = vm_read8(ctx->gpr[29] + 0x1);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00879DF4;
    ctx->gpr[0] = ctx->gpr[3] - ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6038); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
loc_00879C18:
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[30]));
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[13]));
    ctx->lr = 0x00879C38; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read16(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00879D78;
loc_00879C60:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x167; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00879C70;
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-360);
    vm_write16(ctx->gpr[28] + 0x0, ctx->gpr[0]);
loc_00879C70:
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
loc_00879C98:
    ctx->gpr[11] = vm_read8(ctx->gpr[3] + 0x2);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00879CB0;
    ctx->gpr[30] = ctx->gpr[10] | ctx->gpr[10];
loc_00879CB0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 31, 0, 0);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[0] | ctx->gpr[0];
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00879B60;
    ctx->gpr[31] = ctx->gpr[9] | ctx->gpr[9];
    goto loc_00879B60;
loc_00879CC8:
    ctx->gpr[9] = (uint64_t)0 - ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6044); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x1FE);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[1]));
    ctx->lr = 0x00879CF8; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[28] + 0x2, ctx->gpr[0]);
    ctx->gpr[3] = vm_read8(ctx->gpr[29] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00879BFC;
loc_00879D1C:
    ctx->gpr[9] = vm_read8(ctx->gpr[29] + 0x2);
    ctx->gpr[0] = vm_read8(ctx->gpr[29] + 0x1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[1], ctx->fpr[30]));
    ctx->lr = 0x00879D4C; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write16(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read16(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00879C60;
    ;
loc_00879D78:
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(0x168);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    vm_write16(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00879DA8:
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[9] | ctx->gpr[9];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00879B50;
    goto loc_00879CB0;
loc_00879DB8:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    vm_write16(ctx->gpr[28] + 0x0, ctx->gpr[0]);
    vm_write16(ctx->gpr[28] + 0x2, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00879DEC:
    ctx->gpr[31] = ctx->gpr[10] | ctx->gpr[10];
    goto loc_00879B60;
loc_00879DF4:
    ctx->gpr[0] = vm_read8(ctx->gpr[29] + 0x2);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x603C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    goto loc_00879C18;
}


void june_evt_func_0087A928(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x98, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[3] + 0x2);
    ctx->gpr[9] = vm_read16(ctx->gpr[3] + 0x4);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6044); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[9] = (int64_t)(int16_t)ctx->gpr[9];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xA0, tmp); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[31]);
    ctx->gpr[3] = vm_read16(ctx->gpr[3] + 0x0);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[29]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x602C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[12];
    ctx->fpr[1] = ctx->fpr[12];
    ctx->fpr[30] = ctx->fpr[12];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087AA84;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x605C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0087AB0C;
    ctx->fpr[10] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[11], ctx->fpr[12], ctx->fpr[12], 0, 0));
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[12]));
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(0x78);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x167; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[10]));
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087AB30;
loc_0087A9C8:
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-240);
loc_0087A9CC:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x3B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087AB40;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
loc_0087A9FC:
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0x167; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-360);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087AA14;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087AC28;
loc_0087AA14:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x3B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087ABE0;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
loc_0087AA44:
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-120);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x167; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087ABD0;
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(-480);
loc_0087AA54:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x3B; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087AB88;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
loc_0087AA84:
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[29]));
    ctx->lr = 0x0087AA8C; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ctx->fpr[1];
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[29]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write8(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x0087AAB0; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ctx->fpr[1];
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[29]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write8(ctx->gpr[31] + 0x1, ctx->gpr[0]);
    ctx->lr = 0x0087AAD4; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    vm_write8(ctx->gpr[31] + 0x2, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x98); memcpy(&ctx->fpr[29], &tmp, 8); }
    ctx->lr = ctx->gpr[0];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_0087AB0C:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[12]));
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(0x78);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x167; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[12]));
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[10]));
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087A9C8;
loc_0087AB30:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0087A9CC;
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(0x1E0);
    goto loc_0087A9CC;
loc_0087AB40:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xB3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[1] = ctx->fpr[10];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087A9FC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xEF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[1] = ctx->fpr[11];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087A9FC;
    { uint64_t _a = ctx->gpr[0]; uint64_t _b = (uint64_t)(int64_t)(0xF0); ctx->gpr[0] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    goto loc_0087A9FC;
loc_0087AB88:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xB3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ctx->fpr[10];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087AA84;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xEF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ctx->fpr[11];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087AA84;
    { uint64_t _a = ctx->gpr[0]; uint64_t _b = (uint64_t)(int64_t)(0xF0); ctx->gpr[0] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    goto loc_0087AA84;
loc_0087ABD0:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0087AA54;
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(0xF0);
    goto loc_0087AA54;
loc_0087ABE0:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xB3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[10];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087AA44;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xEF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[11];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087AA44;
    { uint64_t _a = ctx->gpr[0]; uint64_t _b = (uint64_t)(int64_t)(0xF0); ctx->gpr[0] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6040); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[9], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[9], 8); ctx->fpr[9] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[9]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[12]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    goto loc_0087AA44;
loc_0087AC28:
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(0x168);
    goto loc_0087AA14;
}


void june_evt_func_009425C8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[30] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942790;
    { uint64_t a = (uint32_t)ctx->gpr[6]; uint64_t b = (uint64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00942670;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942AAC;
loc_00942620:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2CC8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2D4);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2CCC);
    ctx->lr = 0x00942630; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00942634:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_00942670:
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x13; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942950;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00942620;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->lr = 0x009426AC; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[8] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[9] = vm_read8(ctx->gpr[3] + 0x1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 24, 0, 7, 0, 0);
    ctx->fpr[3] = ctx->fpr[26];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 16, 0, 15, 0, 0);
    ctx->gpr[11] = vm_read8(ctx->gpr[3] + 0x3);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->gpr[7] = vm_read8(ctx->gpr[3] + 0x2);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CD0);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[29]));
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[7], 8, 0, 23, 0, 0);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[27]));
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[11];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[7] | ctx->gpr[0];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->lr = 0x00942708; june_evt_func_008683B0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[27]));
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2CD8);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(4);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00942750; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[3] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[27]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[26]));
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CD4);
    ctx->gpr[7] = ctx->gpr[7] | ((uint64_t)0xD0D0 << 16);
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[8] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[7] | 0xD0FF;
    ctx->lr = 0x00942788; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00942634;
loc_00942790:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->lr = 0x009427C0; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[8] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = vm_read8(ctx->gpr[3] + 0x1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 24, 0, 7, 0, 0);
    ctx->fpr[3] = ctx->fpr[27];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 16, 0, 15, 0, 0);
    ctx->gpr[11] = vm_read8(ctx->gpr[3] + 0x3);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->gpr[7] = vm_read8(ctx->gpr[3] + 0x2);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CF0);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[29]));
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[7], 8, 0, 23, 0, 0);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[26]));
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[11];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[7] | ctx->gpr[0];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->lr = 0x00942820; june_evt_func_008683B0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2CE4);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00942864; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[8] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[27]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[29]));
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[11], 0); memcpy(&ctx->fpr[11], &iv, 8); }
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[11], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x90);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->lr = 0x00942910; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_00942950:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->lr = 0x0094297C; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x0);
    ctx->gpr[8] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[9] = vm_read8(ctx->gpr[3] + 0x1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 24, 0, 7, 0, 0);
    ctx->fpr[3] = ctx->fpr[27];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 16, 0, 15, 0, 0);
    ctx->gpr[11] = vm_read8(ctx->gpr[3] + 0x3);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->gpr[7] = vm_read8(ctx->gpr[3] + 0x2);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CDC);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[29]));
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[7], 8, 0, 23, 0, 0);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[26]));
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[11];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[7] | ctx->gpr[0];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->lr = 0x009429D8; june_evt_func_008683B0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[26]));
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2CD8);
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00942A1C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[3] = ctx->fpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[26]));
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[27]));
    ctx->gpr[7] = ctx->gpr[7] | ((uint64_t)0xD0D0 << 16);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CD4);
    ctx->gpr[8] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[7] | 0xD0FF;
    { uint64_t iv = ppu_f2i64(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0xA0, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->lr = 0x00942A6C; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_00942AAC:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->lr = 0x00942ACC; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CF4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read8(ctx->gpr[6] + 0x0);
    ctx->gpr[11] = vm_read8(ctx->gpr[6] + 0x1);
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 24, 0, 7, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 16, 0, 15, 0, 0);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[1]));
    ctx->gpr[10] = vm_read8(ctx->gpr[6] + 0x3);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[11];
    ctx->gpr[7] = vm_read8(ctx->gpr[6] + 0x2);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[10];
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[7], 8, 0, 23, 0, 0);
    ctx->gpr[7] = ctx->gpr[7] | ctx->gpr[0];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->lr = 0x00942B20; june_evt_func_008683B0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
}


void june_evt_func_00947A08(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->lr;
    { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[30] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947B84;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947B84;
loc_00947A48:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[6], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[1]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00947B9C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[2] = ctx->fpr[10];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    goto loc_00947AE0;
loc_00947A8C:
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[9]));
loc_00947A90:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[3], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[10]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[6], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[0], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00947B30;
loc_00947AD0:
    ctx->fpr[9] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[2]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00947B30;
loc_00947AE0:
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00947A8C;
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00947A90;
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[3], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[10]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[6], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[0], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947AD0;
loc_00947B30:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947B50;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00947B64;
loc_00947B50:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2BBC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x47);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2BD4);
    ctx->lr = 0x00947B60; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00947B64:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[30], &tmp, 8); }
    ctx->lr = ctx->gpr[0];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00947B84:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2BD4);
    ctx->gpr[4] = (int64_t)(int32_t)(0x3F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2BD0);
    ctx->lr = 0x00947B94; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00947A48;
loc_00947B9C:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[1]));
    goto loc_00947B30;
}


void june_evt_func_00947C4C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->lr;
    { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[30] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947DCC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947DCC;
loc_00947C8C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[6], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[1]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00947DE4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[2] = ctx->fpr[10];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    goto loc_00947D24;
loc_00947CD0:
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[9]));
loc_00947CD4:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[3], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[10]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[6], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[0], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00947D78;
loc_00947D14:
    ctx->fpr[9] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[2]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00947D78;
loc_00947D24:
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00947CD0;
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00947CD4;
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[3], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[10]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[10], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[6], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[0], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947D14;
    ;
loc_00947D78:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00947D98;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2BEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00947DAC;
loc_00947D98:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2BBC);
    ctx->gpr[4] = (int64_t)(int32_t)(0xCD);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2BD4);
    ctx->lr = 0x00947DA8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00947DAC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[30], &tmp, 8); }
    ctx->lr = ctx->gpr[0];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00947DCC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2BD4);
    ctx->gpr[4] = (int64_t)(int32_t)(0xC5);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2BD0);
    ctx->lr = 0x00947DDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00947C8C;
loc_00947DE4:
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[4], ctx->fpr[10], ctx->fpr[0], 0, 0));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[1]));
    goto loc_00947D78;
}
