
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

void june_evt_func_0004F4AC(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x40, ctx->gpr[1]); ctx->gpr[1] += -0x40;
        ctx->gpr[7] = (int64_t)(int32_t)((uint32_t)0xFF7F << 16);
        { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        ctx->gpr[6] = (int64_t)(int32_t)((uint32_t)0x7F7F << 16);
        ctx->gpr[10] = (int64_t)(int32_t)(3);
        ctx->gpr[7] = ctx->gpr[7] | 0xFFFD;
        ctx->gpr[9] = (int64_t)(int32_t)(0);
        ctx->gpr[6] = ctx->gpr[6] | 0xFFFD;
        ctx->gpr[8] = ctx->gpr[3] | 0x0;
        ctx->ctr = ctx->gpr[10];
loc_0004F4D4:
        { uint64_t _a = ctx->gpr[8]; uint64_t _r = _a + (uint64_t)(int64_t)(0xC); ctx->gpr[10] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
        { uint64_t _a = ctx->gpr[9]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[9] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint64_t _a = ctx->gpr[8]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[8] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        vm_write32((ctx->gpr[3] + ctx->gpr[11]), ctx->gpr[7]);
        vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[6]);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_0004F4D4;
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        ctx->gpr[6] = (int64_t)(int32_t)(0x10);
        { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)0; ppu_vstu4(&ctx->vr[3],v); }
        ctx->gpr[7] = (int64_t)(int32_t)(0x20);
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        ctx->gpr[8] = (int64_t)(int32_t)(0x30);
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
        { uint64_t ea = ((ctx->gpr[6] + ctx->gpr[5])) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
        { uint64_t ea = ((ctx->gpr[7] + ctx->gpr[5])) & ~0xFULL; vm_raw_ld16(&ctx->vr[5], ea); }
        { uint64_t ea = ((ctx->gpr[8] + ctx->gpr[5])) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[5] = (int64_t)(int32_t)(0x30);
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[2], &ctx->vr[3], &ctx->vr[7], ctx->vscr);
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { float ftmp = (float)ctx->fpr[2]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[7], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[5]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F584;
        goto loc_0004F58C;
loc_0004F584:
        ctx->fpr[4] = ctx->fpr[5];
        { float ftmp = (float)ctx->fpr[5]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004F58C:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
        { double a = ctx->fpr[3]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F59C;
        goto loc_0004F5A4;
loc_0004F59C:
        ctx->fpr[3] = ctx->fpr[6];
        { float ftmp = (float)ctx->fpr[6]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004F5A4:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F5B4;
        goto loc_0004F5BC;
loc_0004F5B4:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004F5BC:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        { double a = ctx->fpr[1]; double b = ctx->fpr[5]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F5CC;
        goto loc_0004F5D4;
loc_0004F5CC:
        ctx->fpr[1] = ctx->fpr[5];
        { float ftmp = (float)ctx->fpr[5]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004F5D4:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { double a = ctx->fpr[5]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F5E4;
        goto loc_0004F5EC;
loc_0004F5E4:
        ctx->fpr[5] = ctx->fpr[6];
        { float ftmp = (float)ctx->fpr[6]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004F5EC:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F5FC;
        goto loc_0004F604;
loc_0004F5FC:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004F604:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[3], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F670;
        goto loc_0004F678;
loc_0004F670:
        ctx->fpr[4] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004F678:
        { double a = ctx->fpr[3]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F684;
        goto loc_0004F68C;
loc_0004F684:
        ctx->fpr[3] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004F68C:
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F698;
        goto loc_0004F6A0;
loc_0004F698:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004F6A0:
        { double a = ctx->fpr[1]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F6AC;
        goto loc_0004F6B4;
loc_0004F6AC:
        ctx->fpr[1] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004F6B4:
        { double a = ctx->fpr[5]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F6C0;
        goto loc_0004F6C8;
loc_0004F6C0:
        ctx->fpr[5] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004F6C8:
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F6D4;
        goto loc_0004F6DC;
loc_0004F6D4:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004F6DC:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[3], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F748;
        goto loc_0004F750;
loc_0004F748:
        ctx->fpr[4] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004F750:
        { double a = ctx->fpr[3]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F75C;
        goto loc_0004F764;
loc_0004F75C:
        ctx->fpr[3] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004F764:
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F770;
        goto loc_0004F778;
loc_0004F770:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004F778:
        { double a = ctx->fpr[1]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F784;
        goto loc_0004F78C;
loc_0004F784:
        ctx->fpr[1] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004F78C:
        { double a = ctx->fpr[5]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F798;
        goto loc_0004F7A0;
loc_0004F798:
        ctx->fpr[5] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004F7A0:
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F7AC;
        goto loc_0004F7B4;
loc_0004F7AC:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004F7B4:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[3], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F820;
        goto loc_0004F828;
loc_0004F820:
        ctx->fpr[4] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004F828:
        { double a = ctx->fpr[3]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F834;
        goto loc_0004F83C;
loc_0004F834:
        ctx->fpr[3] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004F83C:
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F848;
        goto loc_0004F850;
loc_0004F848:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004F850:
        { double a = ctx->fpr[1]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F85C;
        goto loc_0004F864;
loc_0004F85C:
        ctx->fpr[1] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004F864:
        { double a = ctx->fpr[5]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F870;
        goto loc_0004F878;
loc_0004F870:
        ctx->fpr[5] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004F878:
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F884;
        goto loc_0004F88C;
loc_0004F884:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004F88C:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[3], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F8F8;
        goto loc_0004F900;
loc_0004F8F8:
        ctx->fpr[4] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004F900:
        { double a = ctx->fpr[3]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F90C;
        goto loc_0004F914;
loc_0004F90C:
        ctx->fpr[3] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004F914:
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F920;
        goto loc_0004F928;
loc_0004F920:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004F928:
        { double a = ctx->fpr[1]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F934;
        goto loc_0004F93C;
loc_0004F934:
        ctx->fpr[1] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004F93C:
        { double a = ctx->fpr[5]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F948;
        goto loc_0004F950;
loc_0004F948:
        ctx->fpr[5] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004F950:
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004F95C;
        goto loc_0004F964;
loc_0004F95C:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004F964:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[3], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F9D0;
        goto loc_0004F9D8;
loc_0004F9D0:
        ctx->fpr[4] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004F9D8:
        { double a = ctx->fpr[3]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F9E4;
        goto loc_0004F9EC;
loc_0004F9E4:
        ctx->fpr[3] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004F9EC:
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004F9F8;
        goto loc_0004FA00;
loc_0004F9F8:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004FA00:
        { double a = ctx->fpr[1]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FA0C;
        goto loc_0004FA14;
loc_0004FA0C:
        ctx->fpr[1] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004FA14:
        { double a = ctx->fpr[5]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FA20;
        goto loc_0004FA28;
loc_0004FA20:
        ctx->fpr[5] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004FA28:
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FA34;
        goto loc_0004FA3C;
loc_0004FA34:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004FA3C:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[9], ea); }
        vmx_vmaddfp(&ctx->vr[8], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[9]; uint32_t val = ((uint32_t*)&ctx->vr[9])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[3], &ctx->vr[9], ctx->vscr);
        vmx_vaddfp(&ctx->vr[8], &ctx->vr[4], &ctx->vr[8], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[7], &ctx->vr[5], &ctx->vr[9], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[7], &ctx->vr[7], &ctx->vr[8], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[7]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004FAA8;
        goto loc_0004FAB0;
loc_0004FAA8:
        ctx->fpr[4] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004FAB0:
        { double a = ctx->fpr[3]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004FABC;
        goto loc_0004FAC4;
loc_0004FABC:
        ctx->fpr[3] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[3]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004FAC4:
        { double a = ctx->fpr[2]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004FAD0;
        goto loc_0004FAD8;
loc_0004FAD0:
        ctx->fpr[2] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[2]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004FAD8:
        { double a = ctx->fpr[1]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FAE4;
        goto loc_0004FAEC;
loc_0004FAE4:
        ctx->fpr[1] = ctx->fpr[9];
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004FAEC:
        { double a = ctx->fpr[5]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FAF8;
        goto loc_0004FB00;
loc_0004FAF8:
        ctx->fpr[5] = ctx->fpr[8];
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004FB00:
        { double a = ctx->fpr[6]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FB0C;
        goto loc_0004FB14;
loc_0004FB0C:
        ctx->fpr[6] = ctx->fpr[7];
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004FB14:
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        { uint32_t* d = (uint32_t*)&ctx->vr[7]; uint32_t val = ((uint32_t*)&ctx->vr[7])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[8], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(-16);
        { uint32_t* d = (uint32_t*)&ctx->vr[8]; uint32_t val = ((uint32_t*)&ctx->vr[8])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + -0x10, tmp); }
        vmx_vmaddfp(&ctx->vr[6], &ctx->vr[6], &ctx->vr[3], &ctx->vr[8], ctx->vscr);
        vmx_vaddfp(&ctx->vr[4], &ctx->vr[4], &ctx->vr[6], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[6], ea); }
        { uint32_t* d = (uint32_t*)&ctx->vr[6]; uint32_t val = ((uint32_t*)&ctx->vr[6])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[3], &ctx->vr[6], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[5], &ctx->vr[2], &ctx->vr[7], ctx->vscr);
        vmx_vaddfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[4], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004FB80;
        goto loc_0004FB84;
loc_0004FB80:
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
loc_0004FB84:
        { double a = ctx->fpr[3]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004FB90;
        goto loc_0004FB94;
loc_0004FB90:
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x10, tmp); }
loc_0004FB94:
        { double a = ctx->fpr[2]; double b = ctx->fpr[4]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_0004FBA0;
        goto loc_0004FBA4;
loc_0004FBA0:
        { float ftmp = (float)ctx->fpr[4]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x14, tmp); }
loc_0004FBA4:
        { double a = ctx->fpr[1]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FBB0;
        goto loc_0004FBB4;
loc_0004FBB0:
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
loc_0004FBB4:
        { double a = ctx->fpr[5]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FBC0;
        goto loc_0004FBC4;
loc_0004FBC0:
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
loc_0004FBC4:
        { double a = ctx->fpr[6]; double b = ctx->fpr[4]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0004FBD0;
        goto loc_0004FBD4;
loc_0004FBD0:
        { float ftmp = (float)ctx->fpr[4]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_0004FBD4:
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x40);
        return;
}

void june_evt_func_00051720(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x100, ctx->gpr[1]); ctx->gpr[1] += -0x100;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
        ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xA0);
        { uint64_t ea = (ctx->gpr[0]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[30]); }
        { uint64_t _a = ctx->gpr[0]; uint64_t _r = _a + (uint64_t)(int64_t)(0x10); ctx->gpr[0] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint64_t ea = (ctx->gpr[0]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[31]); }
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[30]);
        ctx->gpr[9] = (int64_t)(int32_t)((uint32_t)0x5 << 16);
        ctx->gpr[30] = ctx->gpr[5] | 0x0;
        { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)0; ppu_vstu4(&ctx->vr[31],v); }
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x1708);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
        ctx->gpr[5] = (int64_t)(int32_t)((uint32_t)0xFF7F << 16);
        ctx->gpr[29] = ctx->gpr[4] | 0x0;
        ctx->gpr[4] = (int64_t)(int32_t)(0x10);
        ctx->gpr[5] = ctx->gpr[5] | 0xFFFD;
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[5]);
        { uint64_t ea = ((ctx->gpr[4] + ctx->gpr[30])) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        { uint64_t ea = (ctx->gpr[30]) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
        vmx_vsubfp(&ctx->vr[30], &ctx->vr[2], &ctx->vr[3], ctx->vscr);
        { uint64_t* d = (uint64_t*)&ctx->vr[5]; uint64_t* a = (uint64_t*)&ctx->vr[31]; uint64_t* b = (uint64_t*)&ctx->vr[31]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)(8);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[28]);
        ctx->fpr[30] = ctx->fpr[1];
        ctx->gpr[28] = ctx->gpr[3] | 0x0;
        ctx->gpr[3] = (int64_t)(int32_t)(0x70);
        { uint64_t ea = (ctx->gpr[4] + ctx->gpr[9]); uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[6]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
        ctx->gpr[12] = (int64_t)(int32_t)(0);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[31]);
        { uint32_t* d = (uint32_t*)&ctx->vr[6]; uint32_t val = ((uint32_t*)&ctx->vr[6])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        ctx->gpr[31] = ctx->gpr[12] | 0x0;
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[30], &ctx->vr[31], &ctx->vr[30], ctx->vscr);
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[30], 16); memcpy(tmp+16, &ctx->vr[30], 16); memcpy(&ctx->vr[3], tmp + 4, 16); }
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[30], 16); memcpy(tmp+16, &ctx->vr[30], 16); memcpy(&ctx->vr[4], tmp + 8, 16); }
        ctx->gpr[7] = ctx->gpr[6] | 0x0;
        ctx->gpr[10] = ctx->gpr[31] | 0x0;
        { uint64_t _a = ctx->gpr[1]; uint64_t _r = _a + (uint64_t)(int64_t)(0x70); ctx->gpr[8] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[3], &ctx->vr[2], &ctx->vr[3], ctx->vscr);
        { uint64_t ea = (ctx->gpr[5] + ctx->gpr[9]); uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[3]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
        { uint32_t* d = (uint32_t*)&ctx->vr[3]; uint32_t val = ((uint32_t*)&ctx->vr[3])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[4], &ctx->vr[2], &ctx->vr[4], ctx->vscr);
        { uint32_t* d = (uint32_t*)&ctx->vr[4]; uint32_t val = ((uint32_t*)&ctx->vr[2])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vrsqrtefp(&ctx->vr[7], &ctx->vr[4], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[4], &ctx->vr[7], &ctx->vr[5], &ctx->vr[4], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[5], &ctx->vr[7], &ctx->vr[5], &ctx->vr[6], ctx->vscr);
        vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[7], &ctx->vr[3], &ctx->vr[4], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[7], &ctx->vr[5], ctx->vscr);
        (void)vmx_vcmpeqfp(&ctx->vr[4], &ctx->vr[3], &ctx->vr[3]);
        { uint64_t* d=(uint64_t*)&ctx->vr[3]; uint64_t* a=(uint64_t*)&ctx->vr[7]; uint64_t* b=(uint64_t*)&ctx->vr[3]; uint64_t* c=(uint64_t*)&ctx->vr[4]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
        vmx_vmaddfp(&ctx->vr[30], &ctx->vr[30], &ctx->vr[31], &ctx->vr[3], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[3] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[30]); }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
loc_00051810:
        ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[10];
        { uint32_t tmp = vm_read32((ctx->gpr[8] + ctx->gpr[11])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { uint32_t tmp = vm_read32((ctx->gpr[30] + ctx->gpr[11])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
        ctx->fpr[6] = fabs(ctx->fpr[5]);
        { double a = ctx->fpr[6]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { uint32_t a = (ctx->cr >> (31 - 0)) & 1; uint32_t b = (ctx->cr >> (31 - 2)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 6))) | ((a | b) << (31 - 6)); }
        if ((((ctx->cr >> 25) & 1) == 1)) goto loc_00051830;
        goto loc_00051854;
loc_00051830:
        { uint64_t _a = ctx->gpr[7]; uint64_t _r = _a + (uint64_t)(int64_t)(0xC); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[5]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_00051914;
        { uint32_t tmp = vm_read32((ctx->gpr[6] + ctx->gpr[11])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { double a = ctx->fpr[4]; double b = ctx->fpr[5]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_00051914;
        goto loc_000518C8;
loc_00051854:
        ctx->fpr[6] = ppu_fp_single(ppu_fdiv(ctx->fpr[2], ctx->fpr[5]));
        { uint64_t _a = ctx->gpr[7]; uint64_t _r = _a + (uint64_t)(int64_t)(0xC); ctx->gpr[3] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint32_t tmp = vm_read32((ctx->gpr[6] + ctx->gpr[11])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        ctx->fpr[7] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[4]));
        ctx->fpr[4] = ppu_fp_single(ppu_fsub(ctx->fpr[5], ctx->fpr[4]));
        ctx->fpr[5] = ppu_fp_single(ppu_fmul(ctx->fpr[7], ctx->fpr[6]));
        ctx->fpr[4] = ppu_fp_single(ppu_fmul(ctx->fpr[4], ctx->fpr[6]));
        { double a = ctx->fpr[5]; double b = ctx->fpr[4]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_00051884;
        goto loc_00051890;
loc_00051884:
        ctx->fpr[6] = ctx->fpr[5];
        ctx->fpr[5] = ctx->fpr[4];
        ctx->fpr[4] = ctx->fpr[6];
loc_00051890:
        ctx->fpr[6] = ctx->fpr[1];
        ctx->fpr[1] = ctx->fpr[30];
        { double a = ctx->fpr[5]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_000518A4;
        goto loc_000518B0;
loc_000518A4:
        ctx->fpr[30] = ctx->fpr[5];
        { float ftmp = (float)ctx->fpr[5]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
        ctx->fpr[1] = ctx->fpr[30];
loc_000518B0:
        { double a = ctx->fpr[4]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_000518BC;
        goto loc_000518C0;
loc_000518BC:
        ctx->fpr[31] = ctx->fpr[4];
loc_000518C0:
        { double a = ctx->fpr[1]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_00051914;
loc_000518C8:
        { uint64_t _a = ctx->gpr[12]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[12] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint64_t _a = ctx->gpr[7]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[7] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint64_t _a = ctx->gpr[10]; uint64_t _r = _a + (uint64_t)(int64_t)(4); ctx->gpr[10] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { int64_t a = (int32_t)ctx->gpr[12]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_00051810;
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
        { double a = ctx->fpr[1]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_000518EC;
        goto loc_000518F4;
loc_000518EC:
        { double a = ctx->fpr[31]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_00051914;
loc_000518F4:
        ctx->gpr[3] = (int64_t)(int32_t)(0x80);
        { uint64_t ea = ((ctx->gpr[3] + ctx->gpr[1])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        ctx->lr = 0x00051904; june_evt_func_00AED020(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { double a = ctx->fpr[30]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_00051914;
        goto loc_0005191C;
loc_00051914:
        ctx->gpr[3] = ctx->gpr[31] | 0x0;
        goto loc_00051958;
loc_0005191C:
        { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        { double a = ctx->fpr[1]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 31) & 1) == 1)) goto loc_0005192C;
        goto loc_00051934;
loc_0005192C:
        ctx->fpr[1] = ctx->fpr[31];
        { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
loc_00051934:
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
        { uint64_t ea = (ctx->gpr[30]) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        ctx->gpr[3] = (int64_t)(int32_t)(1);
        ctx->gpr[0] = (int64_t)(int32_t)(0x90);
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
        { uint32_t* d = (uint32_t*)&ctx->vr[3]; uint32_t val = ((uint32_t*)&ctx->vr[3])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[3], &ctx->vr[30], &ctx->vr[31], &ctx->vr[3], ctx->vscr);
        vmx_vaddfp(&ctx->vr[2], &ctx->vr[3], &ctx->vr[2], ctx->vscr);
        { uint64_t ea = (ctx->gpr[28]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
loc_00051958:
        ctx->gpr[0] = (int64_t)(int32_t)(0xA0);
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[30], ea); }
        { uint64_t _a = ctx->gpr[0]; uint64_t _r = _a + (uint64_t)(int64_t)(0x10); ctx->gpr[0] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint64_t ea = ((ctx->gpr[1] + ctx->gpr[0])) & ~0xFULL; vm_raw_ld16(&ctx->vr[31], ea); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x110);
        ctx->lr = ctx->gpr[0];
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[29], &tmp, 8); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[30], &tmp, 8); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
        return;
}

void june_evt_func_00141290(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[6] = ppc_rlw64(ctx->gpr[4], 8, 24, 31, 0, 0);
        ctx->gpr[7] = ppc_rlw64(ctx->gpr[4], 16, 24, 31, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[4], 24, 24, 31, 0, 0);
        ctx->gpr[8] = ppc_rlw64(ctx->gpr[4], 0, 24, 31, 0, 0);
        vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[6]);
        vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[7]);
        ctx->gpr[5] = ctx->gpr[3] | 0x0;
        vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[9]);
        ctx->gpr[3] = (int64_t)(int32_t)(0xC);
        { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[4]);
        if ((((ctx->cr >> 31) & 1) == 0)) goto loc_001412D0;
        ctx->gpr[3] = (int64_t)(int32_t)(0x11);
loc_001412D0:
        ctx->gpr[6] = (int64_t)(int32_t)((uint32_t)0x128 << 16);
        { uint64_t _a = ctx->gpr[1]; uint64_t _r = _a + (uint64_t)(int64_t)(0x70); ctx->gpr[4] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        { uint64_t _a = ctx->gpr[6]; uint64_t _r = _a + (uint64_t)(int64_t)(0x5E80); ctx->gpr[6] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
        ctx->lr = 0x001412E0; june_evt_func_008EF218(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
}

void june_evt_func_0021CD48(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[5] = (int64_t)(int32_t)(0x410);
        { uint64_t ea = (ctx->gpr[4]) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        ctx->gpr[7] = (int64_t)(int32_t)(0x10);
        ctx->gpr[6] = (int64_t)(int32_t)(0x420);
        ctx->gpr[9] = (int64_t)(int32_t)(0x20);
        ctx->gpr[8] = (int64_t)(int32_t)(0x430);
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[3])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        { uint64_t ea = ((ctx->gpr[7] + ctx->gpr[4])) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        ctx->gpr[5] = (int64_t)(int32_t)(0x30);
        { uint64_t ea = ((ctx->gpr[6] + ctx->gpr[3])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        ctx->gpr[7] = (int64_t)(int32_t)(0x440);
        { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[4])) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        { uint64_t ea = ((ctx->gpr[8] + ctx->gpr[3])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        { uint64_t ea = ((ctx->gpr[5] + ctx->gpr[4])) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        { uint64_t ea = ((ctx->gpr[7] + ctx->gpr[3])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        return;
}

void june_evt_func_002344D0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[5] = (int64_t)(int32_t)((uint32_t)0x23 << 16);
        { uint64_t ea = (ctx->gpr[4]) & ~0xFULL; vm_raw_ld16(&ctx->vr[2], ea); }
        ctx->gpr[6] = (int64_t)(int32_t)(0x10);
        { uint32_t v[4]; for(int i=0;i<4;i++) v[i]=(uint32_t)(int32_t)0; ppu_vstu4(&ctx->vr[8],v); }
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(0x4480);
        ctx->gpr[7] = (int64_t)(int32_t)(0x10);
        { uint64_t ea = ((ctx->gpr[6] + ctx->gpr[4])) & ~0xFULL; vm_raw_ld16(&ctx->vr[3], ea); }
        { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[4], ea); }
        { uint64_t ea = ((ctx->gpr[7] + ctx->gpr[5])) & ~0xFULL; vm_raw_ld16(&ctx->vr[5], ea); }
        { uint8_t* a = (uint8_t*)&ctx->vr[2]; uint8_t* b = (uint8_t*)&ctx->vr[2]; uint8_t* c = (uint8_t*)&ctx->vr[4]; uint8_t* d = (uint8_t*)&ctx->vr[6]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        { uint8_t* a = (uint8_t*)&ctx->vr[3]; uint8_t* b = (uint8_t*)&ctx->vr[3]; uint8_t* c = (uint8_t*)&ctx->vr[5]; uint8_t* d = (uint8_t*)&ctx->vr[7]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        ctx->gpr[7] = (int64_t)(int32_t)(0x20);
        { uint8_t* a = (uint8_t*)&ctx->vr[2]; uint8_t* b = (uint8_t*)&ctx->vr[2]; uint8_t* c = (uint8_t*)&ctx->vr[5]; uint8_t* d = (uint8_t*)&ctx->vr[2]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        { uint8_t* a = (uint8_t*)&ctx->vr[3]; uint8_t* b = (uint8_t*)&ctx->vr[3]; uint8_t* c = (uint8_t*)&ctx->vr[4]; uint8_t* d = (uint8_t*)&ctx->vr[3]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        vmx_vmaddfp(&ctx->vr[9], &ctx->vr[6], &ctx->vr[8], &ctx->vr[7], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[7] + ctx->gpr[4])) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[11], tmp + 4, 16); }
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[10], 16); memcpy(tmp+16, &ctx->vr[10], 16); memcpy(&ctx->vr[12], tmp + 8, 16); }
        { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[10]; uint8_t* c = (uint8_t*)&ctx->vr[5]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        { uint8_t* a = (uint8_t*)&ctx->vr[10]; uint8_t* b = (uint8_t*)&ctx->vr[10]; uint8_t* c = (uint8_t*)&ctx->vr[4]; uint8_t* d = (uint8_t*)&ctx->vr[4]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        ctx->gpr[4] = (int64_t)(int32_t)(0x20);
        ctx->gpr[8] = (int64_t)(int32_t)(0x40);
        vmx_vnmsubfp(&ctx->vr[9], &ctx->vr[2], &ctx->vr[9], &ctx->vr[3], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[8], &ctx->vr[5], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[4], &ctx->vr[8], &ctx->vr[2], ctx->vscr);
        { uint64_t ea = (ctx->gpr[4] + ctx->gpr[5]); uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[1]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
        ctx->gpr[4] = (int64_t)(int32_t)(0x30);
        { uint32_t* d = (uint32_t*)&ctx->vr[1]; uint32_t val = ((uint32_t*)&ctx->vr[1])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vmaddfp(&ctx->vr[10], &ctx->vr[9], &ctx->vr[8], &ctx->vr[10], ctx->vscr);
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[13], tmp + 4, 16); }
        { uint8_t tmp[32]; memcpy(tmp, &ctx->vr[9], 16); memcpy(tmp+16, &ctx->vr[9], 16); memcpy(&ctx->vr[0], tmp + 8, 16); }
        vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[7], &ctx->vr[3], &ctx->vr[4], ctx->vscr);
        vmx_vnmsubfp(&ctx->vr[2], &ctx->vr[5], &ctx->vr[2], &ctx->vr[6], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[4] + ctx->gpr[5])) & ~0xFULL; vm_raw_ld16(&ctx->vr[7], ea); }
        vmx_vmaddfp(&ctx->vr[10], &ctx->vr[13], &ctx->vr[10], &ctx->vr[11], ctx->vscr);
        { uint32_t* d=(uint32_t*)&ctx->vr[5]; uint32_t* a=(uint32_t*)&ctx->vr[3]; uint32_t* b=(uint32_t*)&ctx->vr[9]; uint32_t t[4]; t[0]=a[0]; t[1]=b[0]; t[2]=a[1]; t[3]=b[1]; memcpy(d, t, 16); }
        { uint32_t* d=(uint32_t*)&ctx->vr[3]; uint32_t* a=(uint32_t*)&ctx->vr[3]; uint32_t* b=(uint32_t*)&ctx->vr[9]; uint32_t t[4]; t[0]=a[2]; t[1]=b[2]; t[2]=a[3]; t[3]=b[3]; memcpy(d, t, 16); }
        { uint32_t* d=(uint32_t*)&ctx->vr[9]; uint32_t* a=(uint32_t*)&ctx->vr[5]; uint32_t* b=(uint32_t*)&ctx->vr[2]; uint32_t t[4]; t[0]=a[0]; t[1]=b[0]; t[2]=a[1]; t[3]=b[1]; memcpy(d, t, 16); }
        { uint8_t* a = (uint8_t*)&ctx->vr[5]; uint8_t* b = (uint8_t*)&ctx->vr[2]; uint8_t* c = (uint8_t*)&ctx->vr[7]; uint8_t* d = (uint8_t*)&ctx->vr[5]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        vmx_vmaddfp(&ctx->vr[4], &ctx->vr[0], &ctx->vr[10], &ctx->vr[12], ctx->vscr);
        { uint64_t ea = ((ctx->gpr[8] + ctx->gpr[5])) & ~0xFULL; vm_raw_ld16(&ctx->vr[10], ea); }
        { uint8_t* a = (uint8_t*)&ctx->vr[3]; uint8_t* b = (uint8_t*)&ctx->vr[2]; uint8_t* c = (uint8_t*)&ctx->vr[10]; uint8_t* d = (uint8_t*)&ctx->vr[2]; uint8_t tmp[16]; for (int i = 0; i < 16; i++) { uint8_t sel = c[i] & 0x1F; tmp[i] = (sel < 16) ? a[sel] : b[sel - 16]; } memcpy(d, tmp, 16); }
        { uint32_t* d = (uint32_t*)&ctx->vr[4]; uint32_t val = ((uint32_t*)&ctx->vr[4])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        vmx_vrefp(&ctx->vr[6], &ctx->vr[4], ctx->vscr);
        vmx_vnmsubfp(&ctx->vr[3], &ctx->vr[4], &ctx->vr[1], &ctx->vr[6], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[3], &ctx->vr[3], &ctx->vr[6], &ctx->vr[6], ctx->vscr);
        (void)vmx_vcmpeqfp(&ctx->vr[4], &ctx->vr[3], &ctx->vr[3]);
        { uint64_t* d=(uint64_t*)&ctx->vr[3]; uint64_t* a=(uint64_t*)&ctx->vr[6]; uint64_t* b=(uint64_t*)&ctx->vr[3]; uint64_t* c=(uint64_t*)&ctx->vr[4]; d[0]=(a[0]&~c[0])|(b[0]&c[0]); d[1]=(a[1]&~c[1])|(b[1]&c[1]); }
        vmx_vmaddfp(&ctx->vr[4], &ctx->vr[9], &ctx->vr[8], &ctx->vr[3], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[5], &ctx->vr[5], &ctx->vr[8], &ctx->vr[3], ctx->vscr);
        vmx_vmaddfp(&ctx->vr[2], &ctx->vr[2], &ctx->vr[8], &ctx->vr[3], ctx->vscr);
        { uint64_t ea = (ctx->gpr[3]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[4]); }
        { uint64_t ea = ((ctx->gpr[6] + ctx->gpr[3])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[5]); }
        { uint64_t ea = ((ctx->gpr[7] + ctx->gpr[3])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[2]); }
        return;
}





void june_evt_func_00829288(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0xE);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008292E0;
        ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x18);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008292C0;
        ctx->lr = 0x008292BC; june_evt_func_00815C68(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008292C0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x008292C8; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_008292E0:
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008292C0;
        ctx->lr = 0x008292F4; june_evt_func_008BC054(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x00829300; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
}

void june_evt_func_00876314(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x6168);
        ctx->gpr[4] = (int64_t)(int32_t)(2);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[3] = (int64_t)(int32_t)(0x14);
        ctx->gpr[6] = (int64_t)(int32_t)(0x19);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->lr = 0x00876350; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876468;
loc_00876360:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0x14);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x00876370; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[28], 0, 16, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008763D4;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876438;
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6124);
        ctx->gpr[4] = (int64_t)(int32_t)(0x80);
        ctx->lr = 0x00876398; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_0087639C:
        ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x10);
        vm_write16(ctx->gpr[31] + 0x2, ctx->gpr[28]);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008763AC; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_008763D4:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087639C;
        ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0xC);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008763F0; june_evt_func_008350D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0x10);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        vm_write16(ctx->gpr[31] + 0x2, ctx->gpr[28]);
        ctx->lr = 0x00876410; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_00876438:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0087639C;
        ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0xC);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00876454; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ctx->gpr[0] | 0x1;
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_0087639C;
loc_00876468:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x60DC);
        ctx->gpr[4] = (int64_t)(int32_t)(0x1A);
        ctx->lr = 0x00876478; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_00876360;
}

void june_evt_func_00876480(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
        ctx->lr = 0x008764C0; june_evt_func_00876314(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[23] = ctx->gpr[27] | ctx->gpr[27];
        ctx->lr = 0x008764D0; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008765EC;
        ctx->gpr[25] = (int64_t)(int32_t)(0);
        ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x6168);
loc_008764E8:
        ctx->gpr[4] = (int64_t)(int32_t)(2);
        ctx->gpr[3] = (int64_t)(int32_t)(0x30);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[6] = (int64_t)(int32_t)(0x13A);
        ctx->lr = 0x00876500; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876630;
loc_00876514:
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[31] = ctx->gpr[30] + (int64_t)(4);
        vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876624;
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x4);
        vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[29]);
loc_00876548:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[30]);
        ctx->lr = 0x00876554; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876568; june_evt_func_008792D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876574; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008765CC;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x60D8);
        ctx->gpr[9] = ppc_rldic(ctx->gpr[0], 2, 30);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x008765CCu: goto loc_008765CC;case 0x00876648u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876648; return; }case 0x00876678u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876678; return; }case 0x008766A8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766A8; return; }case 0x008766D8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766D8; return; }case 0x00876738u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876738; return; }case 0x00876788u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876788; return; }case 0x008767E8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008767E8; return; }case 0x00876838u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876838; return; }case 0x00876898u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876898; return; } default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
loc_008765CC:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6124);
        ctx->gpr[4] = (int64_t)(int32_t)(0x176);
        ctx->lr = 0x008765DC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008764E8;
loc_008765EC:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return;
loc_00876624:
        vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        goto loc_00876548;
loc_00876630:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x610C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x13B);
        ctx->lr = 0x00876640; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_00876514;
}

void june_evt_func_008764E8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_23 = vm_read64(ctx->gpr[1] + 0x78);
        uint64_t _cs_24 = vm_read64(ctx->gpr[1] + 0x80);
        uint64_t _cs_25 = vm_read64(ctx->gpr[1] + 0x88);
        uint64_t _cs_26 = vm_read64(ctx->gpr[1] + 0x90);
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0x98);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xA0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xB8);
loc_008764E8:
        ctx->gpr[4] = (int64_t)(int32_t)(2);
        ctx->gpr[3] = (int64_t)(int32_t)(0x30);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[6] = (int64_t)(int32_t)(0x13A);
        ctx->lr = 0x00876500; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876630;
loc_00876514:
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[31] = ctx->gpr[30] + (int64_t)(4);
        vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876624;
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x4);
        vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[29]);
loc_00876548:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[30]);
        ctx->lr = 0x00876554; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876568; june_evt_func_008792D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876574; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008765CC;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x60D8);
        ctx->gpr[9] = ppc_rldic(ctx->gpr[0], 2, 30);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x008765CCu: goto loc_008765CC;case 0x00876648u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876648; return; }case 0x00876678u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876678; return; }case 0x008766A8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766A8; return; }case 0x008766D8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766D8; return; }case 0x00876738u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876738; return; }case 0x00876788u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876788; return; }case 0x008767E8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008767E8; return; }case 0x00876838u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876838; return; }case 0x00876898u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876898; return; } default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
loc_008765CC:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6124);
        ctx->gpr[4] = (int64_t)(int32_t)(0x176);
        ctx->lr = 0x008765DC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008764E8;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return;
loc_00876624:
        vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        goto loc_00876548;
loc_00876630:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x610C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x13B);
        ctx->lr = 0x00876640; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_00876514;
}

void june_evt_func_00876514(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_23 = vm_read64(ctx->gpr[1] + 0x78);
        uint64_t _cs_24 = vm_read64(ctx->gpr[1] + 0x80);
        uint64_t _cs_25 = vm_read64(ctx->gpr[1] + 0x88);
        uint64_t _cs_26 = vm_read64(ctx->gpr[1] + 0x90);
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0x98);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xA0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xB8);
loc_00876514:
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[31] = ctx->gpr[30] + (int64_t)(4);
        vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876624;
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x4);
        vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[29]);
loc_00876548:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[30]);
        ctx->lr = 0x00876554; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876568; june_evt_func_008792D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876574; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008765CC;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x60D8);
        ctx->gpr[9] = ppc_rldic(ctx->gpr[0], 2, 30);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x008765CCu: goto loc_008765CC;case 0x00876648u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876648; return; }case 0x00876678u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876678; return; }case 0x008766A8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766A8; return; }case 0x008766D8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766D8; return; }case 0x00876738u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876738; return; }case 0x00876788u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876788; return; }case 0x008767E8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008767E8; return; }case 0x00876838u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876838; return; }case 0x00876898u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876898; return; } default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
loc_008765CC:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6124);
        ctx->gpr[4] = (int64_t)(int32_t)(0x176);
        ctx->lr = 0x008765DC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_008764E8; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return;
loc_00876624:
        vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        goto loc_00876548;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x610C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x13B);
        ctx->lr = 0x00876640; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_00876514;
}

void june_evt_func_00876548(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_23 = vm_read64(ctx->gpr[1] + 0x78);
        uint64_t _cs_24 = vm_read64(ctx->gpr[1] + 0x80);
        uint64_t _cs_25 = vm_read64(ctx->gpr[1] + 0x88);
        uint64_t _cs_26 = vm_read64(ctx->gpr[1] + 0x90);
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0x98);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xA0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xB8);
loc_00876548:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[30]);
        ctx->lr = 0x00876554; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876568; june_evt_func_008792D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00876574; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008765CC;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x60D8);
        ctx->gpr[9] = ppc_rldic(ctx->gpr[0], 2, 30);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x008765CCu: goto loc_008765CC;case 0x00876648u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876648; return; }case 0x00876678u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876678; return; }case 0x008766A8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766A8; return; }case 0x008766D8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008766D8; return; }case 0x00876738u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876738; return; }case 0x00876788u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876788; return; }case 0x008767E8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_008767E8; return; }case 0x00876838u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876838; return; }case 0x00876898u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00876898; return; } default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
loc_008765CC:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6124);
        ctx->gpr[4] = (int64_t)(int32_t)(0x176);
        ctx->lr = 0x008765DC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_008764E8; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return;
        vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        goto loc_00876548;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x610C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x13B);
        ctx->lr = 0x00876640; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00876514; return; }
}

void june_evt_func_008765E0(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_23 = vm_read64(ctx->gpr[1] + 0x78);
        uint64_t _cs_24 = vm_read64(ctx->gpr[1] + 0x80);
        uint64_t _cs_25 = vm_read64(ctx->gpr[1] + 0x88);
        uint64_t _cs_26 = vm_read64(ctx->gpr[1] + 0x90);
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0x98);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xA0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xB8);
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_008764E8; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return;
        vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00876548; return; }
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x610C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x13B);
        ctx->lr = 0x00876640; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00876514; return; }
}

void june_evt_func_00876648(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876664;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6104);
        ctx->gpr[4] = (int64_t)(int32_t)(0x147);
        ctx->lr = 0x00876660; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00876664:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087666C; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[3]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_00876678(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876694;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6100);
        ctx->gpr[4] = (int64_t)(int32_t)(0x14B);
        ctx->lr = 0x00876690; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00876694:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087669C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x14, tmp); }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_008766A8(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008766C4;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x60FC);
        ctx->gpr[4] = (int64_t)(int32_t)(0x14F);
        ctx->lr = 0x008766C0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008766C4:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008766CC; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x14, ctx->gpr[3]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_008766D8(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[4] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[6] = (int64_t)(int32_t)(0x153);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008766F0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[5] = vm_read32(ctx->gpr[30] + 0x24);
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-1);
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[0]);
        ctx->lr = 0x00876714; june_evt_func_00878340(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x24);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x14);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_00876738(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876754;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x60F0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x158);
        ctx->lr = 0x00876750; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00876754:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087675C; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x14, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087676C; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x15, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087677C; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x16, ctx->gpr[3]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_00876788(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008767A4;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x60EC);
        ctx->gpr[4] = (int64_t)(int32_t)(0x15E);
        ctx->lr = 0x008767A0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008767A4:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008767AC; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x14, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008767BC; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x15, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008767CC; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x16, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008767DC; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write8(ctx->gpr[30] + 0x17, ctx->gpr[3]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_008767E8(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876804;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x60E8);
        ctx->gpr[4] = (int64_t)(int32_t)(0x165);
        ctx->lr = 0x00876800; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00876804:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087680C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x14, tmp); }
        ctx->lr = 0x0087681C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x18, tmp); }
        ctx->lr = 0x0087682C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x1C, tmp); }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_00876838(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00876854;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x60E4);
        ctx->gpr[4] = (int64_t)(int32_t)(0x16B);
        ctx->lr = 0x00876850; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00876854:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x0087685C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x14, tmp); }
        ctx->lr = 0x0087686C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x18, tmp); }
        ctx->lr = 0x0087687C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x1C, tmp); }
        ctx->lr = 0x0087688C; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x20, tmp); }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_00876898(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[4] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[6] = (int64_t)(int32_t)(0x172);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008768B0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[5] = vm_read32(ctx->gpr[30] + 0x24);
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[0]);
        ctx->lr = 0x008768CC; june_evt_func_00878340(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008765E0; return; }
}

void june_evt_func_008784F4(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[5] = (int64_t)(int32_t)(1);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->lr = 0x0087850C; june_evt_func_00878340(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[4] = (int64_t)(int32_t)(0x395);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00878528;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6088);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x60A0);
        ctx->lr = 0x00878524; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00878528:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = vm_read8(ctx->gpr[1] + 0x70);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_008785D0(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[5] = (int64_t)(int32_t)(4);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->lr = 0x008785E8; june_evt_func_00878340(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[4] = (int64_t)(int32_t)(0x33B);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00878604;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6088);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x60A0);
        ctx->lr = 0x00878600; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00878604:
        ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_008861A0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x5DD8);
        ctx->gpr[11] = (int64_t)(int32_t)(1);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008861D8;
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5DD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008861D0;
        ctx->gpr[11] = (int64_t)(int32_t)(0);
loc_008861D0:
        ctx->gpr[0] = ctx->gpr[11] ^ 0x1;
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
loc_008861D8:
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        return;
}

void june_evt_func_0088A6E8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[9] = ctx->gpr[3] + (int64_t)(0x1C);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A7C8;
loc_0088A71C:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x0088A724; june_evt_func_00834F80(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A74C;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_0088A74C:
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
        ctx->lr = 0x0088A754; june_evt_func_0089D190(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x18);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A7A0;
        ;
loc_0088A768:
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
        ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x14);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A788;
        ctx->lr = 0x0088A784; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_0088A788:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x0088A790; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088A768;
loc_0088A7A0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->lr = 0x0088A7A8; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_0088A7C8:
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BE8);
        ctx->gpr[4] = (int64_t)(int32_t)(0x22);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5BEC);
        ctx->lr = 0x0088A7D8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0088A71C;
}

void june_evt_func_0088A7E0(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
        ctx->gpr[22] = vm_read32(ctx->gpr[2] + -0x5C1C);
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[20]);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
        ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[3] = (int64_t)(int32_t)(0x20);
        ctx->gpr[5] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[6] = (int64_t)(int32_t)(0x3D);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[17]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[18]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[19]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[21]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
        ctx->lr = 0x0088A850; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB94;
loc_0088A860:
        ctx->gpr[17] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0x20);
        ctx->gpr[3] = ctx->gpr[17] | ctx->gpr[17];
        ctx->lr = 0x0088A870; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[26] + (int64_t)(0x1C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x0088A880; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0xC);
        { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x24);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x4, tmp); }
        vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[9]);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x0088A8A0; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
        ctx->gpr[23] = vm_read32(ctx->gpr[30] + 0x8);
        ctx->gpr[25] = vm_read32(ctx->gpr[30] + 0xC);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[23]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[21] = vm_read32(ctx->gpr[31] + 0xC);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A8D0;
        ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BE4);
        ctx->gpr[4] = (int64_t)(int32_t)(0x81);
        ctx->lr = 0x0088A8CC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_0088A8D0:
        ctx->gpr[31] = vm_read32(ctx->gpr[20] + 0x18);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AAB0;
        ctx->gpr[24] = (int64_t)(int32_t)(0);
        ctx->gpr[18] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[19] = (int64_t)(int32_t)(0);
loc_0088A8E8:
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[3] = (int64_t)(int32_t)(0x1C);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[6] = (int64_t)(int32_t)(0x88);
        ctx->lr = 0x0088A900; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB68;
loc_0088A914:
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x8);
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x4, tmp); }
        vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[4]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB08;
        ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
        ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0xC);
        ctx->lr = 0x0088A944; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[5] = ctx->gpr[18] | ctx->gpr[18];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[6] = (int64_t)(int32_t)(0x95);
        ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x18);
        vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[27]);
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[31], 6, 0, 25, 0, 0);
        ctx->lr = 0x0088A96C; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[3]);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0088A9B0;
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[8] = vm_read32(ctx->gpr[29] + 0x1C);
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        ctx->ctr = ctx->gpr[31];
        ;
loc_0088A990:
        ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[8];
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x40);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(0x30);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[11] + 0x30, ctx->gpr[0]);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_0088A990;
loc_0088A9B0:
        ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0x10);
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB18;
loc_0088A9BC:
        { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int32_t)ctx->gpr[24]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0088AA20;
        ctx->gpr[7] = ppc_rlw64(ctx->gpr[24], 2, 0, 29, 0, 0);
        ctx->gpr[8] = ctx->gpr[24] | ctx->gpr[24];
        ctx->gpr[10] = ctx->gpr[7] + ctx->gpr[25];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB38;
        ctx->gpr[0] = ctx->gpr[23] - ctx->gpr[24];
        ctx->gpr[11] = ctx->gpr[7] + (int64_t)(4);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[9] = ctx->gpr[10] + (int64_t)(4);
        ctx->gpr[8] = ctx->gpr[24] | ctx->gpr[24];
        ctx->ctr = ctx->gpr[0];
        goto loc_0088AA08;
loc_0088A9FC:
        ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB38;
loc_0088AA08:
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[7] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(4);
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(4);
        ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_0088A9FC;
loc_0088AA20:
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0088AA70;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[24], 0, 32);
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[11] = (int64_t)(int32_t)(4);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088AA58;
        goto loc_0088AB80;
loc_0088AA4C:
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB50;
loc_0088AA58:
        ctx->gpr[8] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[25];
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(4);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_0088AA4C;
loc_0088AA70:
        ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BDC);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB5);
        ctx->lr = 0x0088AA80; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_0088AA84:
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[19]);
        ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AB2C;
loc_0088AA94:
        vm_write32(ctx->gpr[30] + 0x18, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x14);
        vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[30]);
loc_0088AAA0:
        vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[30]);
        ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0x14);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088A8E8;
loc_0088AAB0:
        ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x10);
        ctx->gpr[3] = ctx->gpr[17] | ctx->gpr[17];
        ctx->gpr[18] = _cs_18;
        vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
        ctx->gpr[17] = _cs_17;
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
        return;
loc_0088AB08:
        ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0x10);
        vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[19]);
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088A9BC;
loc_0088AB18:
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[19]);
        ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
        vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[6]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088AA94;
loc_0088AB2C:
        vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[30]);
        vm_write32(ctx->gpr[30] + 0x18, ctx->gpr[0]);
        goto loc_0088AAA0;
loc_0088AB38:
        ctx->gpr[9] = ctx->gpr[21] + ctx->gpr[7];
        ctx->gpr[24] = ctx->gpr[8] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[0]);
        goto loc_0088AA84;
loc_0088AB50:
        ctx->gpr[9] = ctx->gpr[21] + ctx->gpr[8];
        ctx->gpr[24] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[0]);
        goto loc_0088AA84;
loc_0088AB68:
        ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BE0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x89);
        ctx->lr = 0x0088AB78; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0088A914;
loc_0088AB80:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[24] = (int64_t)(int32_t)(1);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[0]);
        goto loc_0088AA84;
loc_0088AB94:
        ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C04);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3E);
        ctx->lr = 0x0088ABA4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0088A860;
}

void june_evt_func_0088ABAC(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[21]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
        ctx->gpr[21] = vm_read32(ctx->gpr[2] + -0x5C1C);
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[5] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[3] = (int64_t)(int32_t)(0x20);
        ctx->gpr[6] = (int64_t)(int32_t)(0x3D);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[17]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[18]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[19]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[20]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
        ctx->lr = 0x0088AC18; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AEC4;
loc_0088AC28:
        ctx->gpr[18] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0x20);
        ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
        ctx->gpr[26] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x0088AC3C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x1C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x0088AC4C; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x0088AC58; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[25] + 0x0, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x0088AC68; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x4, tmp); }
        ctx->lr = 0x0088AC78; june_evt_func_0089C390(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[19] = vm_read32(ctx->gpr[30] + 0xC);
        ctx->gpr[20] = vm_read32(ctx->gpr[30] + 0x8);
        vm_write32(ctx->gpr[25] + 0xC, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x0088AC90; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AE08;
        ctx->gpr[24] = (int64_t)(int32_t)(0);
        ctx->gpr[17] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        vm_write32(ctx->gpr[25] + 0x10, ctx->gpr[3]);
loc_0088ACB0:
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[3] = (int64_t)(int32_t)(0x1C);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[6] = (int64_t)(int32_t)(0x186);
        ctx->lr = 0x0088ACC8; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AEAC;
loc_0088ACDC:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x0088ACE4; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
        ctx->lr = 0x0088ACF4; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x4, tmp); }
        ctx->lr = 0x0088AD04; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[29] + 0x8, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AE58;
        ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0xC);
        ctx->lr = 0x0088AD20; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[28] = vm_read32(ctx->gpr[3] + 0x18);
        vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[31]);
        ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[5] = ctx->gpr[17] | ctx->gpr[17];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[6] = (int64_t)(int32_t)(0x193);
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[28], 6, 0, 25, 0, 0);
        ctx->lr = 0x0088AD48; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088ADA8;
        ctx->gpr[31] = (int64_t)(int32_t)(0);
        ctx->gpr[30] = (int64_t)(int32_t)(0);
loc_0088AD60:
        ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x1C);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[31], 6, 0, 25, 0, 0);
        ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0xC);
        ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[9];
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5BD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
        ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[0];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
        ctx->gpr[3] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0x30);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[11] + 0x30, ctx->gpr[0]);
        ctx->lr = 0x0088AD9C; june_evt_func_0089800C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088AD60;
loc_0088ADA8:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x0088ADB0; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0088AE74;
loc_0088ADC0:
        { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0088AE94;
loc_0088ADC8:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 2, 0, 29, 0, 0);
        vm_write32(ctx->gpr[29] + 0x14, ctx->gpr[23]);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[19];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x14);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088AE88;
loc_0088ADEC:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[0]);
        vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[29]);
loc_0088ADF8:
        ctx->gpr[24] = ctx->gpr[24] + (int64_t)(1);
        vm_write32(ctx->gpr[25] + 0x14, ctx->gpr[29]);
        { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[24]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088ACB0;
loc_0088AE08:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
        ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
        return;
loc_0088AE58:
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[23]);
        ctx->lr = 0x0088AE64; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0088ADC0;
loc_0088AE74:
        vm_write32(ctx->gpr[29] + 0x14, ctx->gpr[23]);
        ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x14);
        vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[23]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088ADEC;
loc_0088AE88:
        vm_write32(ctx->gpr[25] + 0x18, ctx->gpr[29]);
        vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[0]);
        goto loc_0088ADF8;
loc_0088AE94:
        ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BD8);
        ctx->gpr[4] = (int64_t)(int32_t)(0x1A1);
        ctx->lr = 0x0088AEA4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0088ADC8;
loc_0088AEAC:
        ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BF4);
        ctx->gpr[4] = (int64_t)(int32_t)(0x187);
        ctx->lr = 0x0088AEBC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0088ACDC;
loc_0088AEC4:
        ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C04);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3E);
        ctx->lr = 0x0088AED4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_0088AC28;
}

void june_evt_func_0089800C(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        ctx->fpr[31] = ctx->fpr[1];
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x30);
        ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x0);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[30]);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[29]);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089805C;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5AA8);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB60);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x59F0);
        ctx->lr = 0x00898054; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x0);
loc_0089805C:
        ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089807C;
        ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0xC);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00898120;
loc_0089807C:
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089823C;
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008980E4;
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0x13; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898274;
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0x16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008980EC;
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5AA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x2C, tmp); }
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
        ctx->gpr[9] = (int64_t)(int32_t)(3);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[11]);
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[10]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[11]);
        vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[10]);
        vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[11]);
        vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[11]);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
        goto loc_00898100;
loc_008980E4:
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898288;
loc_008980EC:
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5A8C);
        ctx->gpr[4] = (int64_t)(int32_t)(0xB90);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5AA8);
        ctx->lr = 0x008980FC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00898100:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_00898120:
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[11], 5, 0, 26, 0, 0);
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x5AAC);
        ctx->gpr[12] = ppc_rlw64(ctx->gpr[11], 2, 0, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[3] - ctx->gpr[12];
        ctx->gpr[5] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[29];
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[6] = vm_read32(ctx->gpr[9] + 0x8);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008981B0;
        ctx->gpr[11] = ctx->gpr[7] + (int64_t)(-1);
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        goto loc_00898170;
loc_00898164:
        ctx->gpr[10] = ctx->gpr[5] | ctx->gpr[5];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008981A4;
loc_00898170:
        ctx->gpr[0] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[0];
        ctx->gpr[5] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[5], 2, 0, 29, 0, 0);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[8];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00898164;
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(-1);
        ctx->gpr[11] = ctx->gpr[5] | ctx->gpr[5];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00898170;
loc_008981A4:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[5], 2, 0, 29, 0, 0);
        ctx->gpr[4] = (uint64_t)((int64_t)(int32_t)ctx->gpr[6] * (int64_t)(int32_t)ctx->gpr[5]);
        ctx->gpr[11] = ctx->gpr[0] | ctx->gpr[0];
loc_008981B0:
        ctx->gpr[9] = ctx->gpr[0] + ctx->gpr[8];
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x8);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[4] = ctx->gpr[4] + ctx->gpr[10];
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008982C4;
        ctx->gpr[0] = ctx->gpr[7] + (int64_t)(-1);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[5]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008982C4;
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[8];
        ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[13]));
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5AB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[0]));
        { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008982C4;
        ctx->gpr[0] = ctx->gpr[3] - ctx->gpr[12];
        ctx->gpr[5] = ctx->gpr[5] + (int64_t)(1);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[5] = (uint64_t)((int64_t)(int32_t)ctx->gpr[6] * (int64_t)(int32_t)ctx->gpr[5]);
        ctx->gpr[9] = vm_read32((ctx->gpr[29] + ctx->gpr[0]));
        ctx->gpr[5] = ctx->gpr[10] + ctx->gpr[5];
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00898234; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        goto loc_00898100;
loc_0089823C:
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5AA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x4, tmp); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x8, tmp); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_00898274:
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5AA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x4, tmp); }
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00898100;
loc_00898288:
        ctx->gpr[9] = (int64_t)(int32_t)(0);
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5AA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
        vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[9]);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
        goto loc_00898100;
loc_008982C4:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[6], 0, 32);
        ctx->lr = 0x008982D4; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x88); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
}

void june_evt_func_00899E54(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x8);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[4] = (int64_t)(int32_t)(0x171);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00899E90;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x59C8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x59D8);
        ctx->lr = 0x00899E8C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_00899E90:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[31], 5, 0, 26, 0, 0);
        ctx->gpr[30] = _cs_30;
        ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[0];
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_008BB4F0(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[21]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x4);
        ctx->gpr[21] = ctx->gpr[4] + (int64_t)(8);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[22]);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[24]);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-4);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
        ctx->gpr[25] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[29] = ctx->gpr[9] + ctx->gpr[21];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB724;
loc_008BB554:
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB70C;
loc_008BB55C:
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB628;
        ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x4);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB6D8;
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x7F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[27] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 2);
        { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[27] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[27]) ? (1u << 29) : 0); ctx->gpr[27] = result; }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008BB6BC;
loc_008BB598:
        ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0xC);
        ctx->gpr[30] = vm_read32(ctx->gpr[28] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB788;
loc_008BB5A8:
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB7A0;
loc_008BB5B0:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[23], 5, 0, 26, 0, 0);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[22], 0, 32);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[27];
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 29, 0, 0);
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x70);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        vm_write32(ctx->gpr[11] + 0x8, ctx->gpr[30]);
        ctx->gpr[9] = ctx->gpr[0] + ctx->gpr[9];
        vm_write32(ctx->gpr[10] + 0xC, ctx->gpr[31]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB73C;
loc_008BB5E8:
        ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB680;
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        ctx->gpr[31] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        vm_write32(ctx->gpr[25] + 0x4, ctx->gpr[0]);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB668;
loc_008BB618:
        ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[21];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-4);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[26]);
loc_008BB628:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
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
loc_008BB668:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5378);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x78);
        ctx->lr = 0x008BB678; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB618;
loc_008BB680:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5310);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x100);
        ctx->lr = 0x008BB690; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x4);
        ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        ctx->gpr[31] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        vm_write32(ctx->gpr[25] + 0x4, ctx->gpr[0]);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BB618;
        goto loc_008BB668;
loc_008BB6BC:
        ctx->gpr[9] = (uint32_t)ctx->gpr[11] ? __builtin_clz((uint32_t)ctx->gpr[11]) : 32;
        { uint64_t _a = ctx->gpr[9]; uint64_t _b = (uint64_t)(int64_t)(0x20); ctx->gpr[9] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
        ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-6);
        ctx->gpr[23] = ctx->gpr[9] + (int64_t)(-7);
        ctx->gpr[0] = (uint64_t)((ctx->gpr[0] & 0x20) ? 0u : ((uint32_t)ctx->gpr[11] >> (ctx->gpr[0] & 0x1F)));
        ctx->gpr[27] = ctx->gpr[0] ^ 0x20;
        goto loc_008BB598;
loc_008BB6D8:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5314);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0xE8);
        ctx->lr = 0x008BB6EC; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x7F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[27] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 2);
        { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[27] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[27]) ? (1u << 29) : 0); ctx->gpr[27] = result; }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008BB598;
        goto loc_008BB6BC;
loc_008BB70C:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5318);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0xE8);
        ctx->lr = 0x008BB71C; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB55C;
loc_008BB724:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5378);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x78);
        ctx->lr = 0x008BB734; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB554;
loc_008BB73C:
        { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[30]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BB5E8;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[23], 2, 0, 29, 0, 0);
        ctx->gpr[8] = (int64_t)(int32_t)(1);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[11] = (uint64_t)((ctx->gpr[27] & 0x20) ? 0u : ((uint32_t)ctx->gpr[8] << (ctx->gpr[27] & 0x1F)));
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[9];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->gpr[11] = ctx->gpr[0] & ~ctx->gpr[11];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[11]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BB5E8;
        ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x10);
        ctx->gpr[0] = (uint64_t)((ctx->gpr[23] & 0x20) ? 0u : ((uint32_t)ctx->gpr[8] << (ctx->gpr[23] & 0x1F)));
        ctx->gpr[0] = ctx->gpr[9] & ~ctx->gpr[0];
        vm_write32(ctx->gpr[24] + 0x10, ctx->gpr[0]);
        goto loc_008BB5E8;
loc_008BB788:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5384);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x60);
        ctx->lr = 0x008BB798; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB5A8;
loc_008BB7A0:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5380);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x60);
        ctx->lr = 0x008BB7B0; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB5B0;
}

void june_evt_func_008BB7B8(ppu_context* ctx) {
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
        { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB9F0;
        ctx->gpr[29] = ctx->gpr[4] + (int64_t)(-8);
        ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-4);
        ctx->gpr[31] = ctx->gpr[9] + ctx->gpr[4];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB954;
loc_008BB814:
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[24], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x4);
        vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[29]);
        ctx->gpr[9] = ctx->gpr[9] | 0x2;
        vm_write32(ctx->gpr[11] + 0x4, ctx->gpr[9]);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
        ctx->gpr[0] = ctx->gpr[0] | 0x1;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
        vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[0]);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB90C;
        ctx->gpr[26] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BBA70;
loc_008BB850:
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB96C;
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x7F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[25] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 2);
        { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[25] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[25]) ? (1u << 29) : 0); ctx->gpr[25] = result; }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008BB9A0;
loc_008BB87C:
        ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0xC);
        ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BBA88;
loc_008BB88C:
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BBAA0;
loc_008BB894:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[23], 5, 0, 26, 0, 0);
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[24], 0, 32);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[25];
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 29, 0, 0);
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x70);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        vm_write32(ctx->gpr[11] + 0x8, ctx->gpr[29]);
        ctx->gpr[9] = ctx->gpr[27] + ctx->gpr[9];
        vm_write32(ctx->gpr[10] + 0xC, ctx->gpr[31]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[26]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BBA24;
loc_008BB8CC:
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB9D4;
loc_008BB8DC:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        ctx->gpr[31] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
        vm_write32(ctx->gpr[28] + 0x4, ctx->gpr[0]);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008BB9BC;
loc_008BB8FC:
        ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[26];
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[26]);
loc_008BB90C:
        ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->lr = 0x008BB918; june_evt_func_008BB4F0(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[23] = _cs_23;
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[24] = _cs_24;
        ctx->gpr[25] = _cs_25;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008B9BBC; return; }
loc_008BB954:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5378);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x78);
        ctx->lr = 0x008BB964; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB814;
loc_008BB96C:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5308);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x110);
        ctx->lr = 0x008BB980; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 0, 29, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x7F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[25] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], 2);
        { uint64_t ca = (ctx->xer >> 29) & 1; uint64_t result = ctx->gpr[25] + ca; ctx->xer = (ctx->xer & ~(1u << 29)) | ((result < ctx->gpr[25]) ? (1u << 29) : 0); ctx->gpr[25] = result; }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008BB87C;
        ;
loc_008BB9A0:
        ctx->gpr[9] = (uint32_t)ctx->gpr[11] ? __builtin_clz((uint32_t)ctx->gpr[11]) : 32;
        { uint64_t _a = ctx->gpr[9]; uint64_t _b = (uint64_t)(int64_t)(0x20); ctx->gpr[9] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
        ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-6);
        ctx->gpr[23] = ctx->gpr[9] + (int64_t)(-7);
        ctx->gpr[0] = (uint64_t)((ctx->gpr[0] & 0x20) ? 0u : ((uint32_t)ctx->gpr[11] >> (ctx->gpr[0] & 0x1F)));
        ctx->gpr[25] = ctx->gpr[0] ^ 0x20;
        goto loc_008BB87C;
loc_008BB9BC:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5378);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x78);
        ctx->lr = 0x008BB9CC; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB8FC;
loc_008BB9D4:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5310);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x100);
        ctx->lr = 0x008BB9E4; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
        goto loc_008BB8DC;
loc_008BB9F0:
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
loc_008BBA24:
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[29]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BB8CC;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[23], 2, 0, 29, 0, 0);
        ctx->gpr[8] = (int64_t)(int32_t)(1);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[11] = (uint64_t)((ctx->gpr[25] & 0x20) ? 0u : ((uint32_t)ctx->gpr[8] << (ctx->gpr[25] & 0x1F)));
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[9];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->gpr[11] = ctx->gpr[0] & ~ctx->gpr[11];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[11]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BB8CC;
        ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x10);
        ctx->gpr[0] = (uint64_t)((ctx->gpr[23] & 0x20) ? 0u : ((uint32_t)ctx->gpr[8] << (ctx->gpr[23] & 0x1F)));
        ctx->gpr[0] = ctx->gpr[9] & ~ctx->gpr[0];
        vm_write32(ctx->gpr[24] + 0x10, ctx->gpr[0]);
        goto loc_008BB8CC;
loc_008BBA70:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x530C);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x110);
        ctx->lr = 0x008BBA80; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB850;
loc_008BBA88:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5384);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x60);
        ctx->lr = 0x008BBA98; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB88C;
loc_008BBAA0:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x53B8);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5380);
        ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x60);
        ctx->lr = 0x008BBAB0; june_evt_func_00D31FE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008BB894;
}

void june_evt_func_008BC054(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        ctx->lr = 0x008BC078; june_evt_func_0087694C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008BC13C;
        if ((((ctx->cr >> 5) & 1) == 1)) goto loc_008BC11C;
        ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0xC);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[0] = ctx->gpr[30] + (int64_t)(0xC);
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008BC0B0; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x10);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = (int64_t)(int32_t)(0x1F5);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008BC168;
loc_008BC0C4:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 2, 14, 29, 0, 0);
        ctx->gpr[7] = vm_read32(ctx->gpr[31] + 0x8);
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[7] = ctx->gpr[7] + (int64_t)(0x94);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[9] = ctx->gpr[8] + ctx->gpr[9];
        ctx->gpr[11] = vm_read32(ctx->gpr[8] + 0x18);
        ctx->gpr[10] = vm_read32(ctx->gpr[8] + 0x14);
        ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[7];
        ctx->gpr[3] = vm_read32(ctx->gpr[8] + 0x8);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(-1);
        vm_write32(ctx->gpr[8] + 0x18, ctx->gpr[11]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
        vm_write32(ctx->gpr[8] + 0x14, ctx->gpr[10]);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[7];
        vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
        ctx->lr = 0x008BC110; june_evt_func_008BB7B8(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008BC118; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008BC11C:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_008BC13C:
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x5300);
        ctx->lr = 0x008BC144; june_evt_func_00876CB0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_008BC168:
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5304);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x53A4);
        ctx->lr = 0x008BC174; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x10);
        goto loc_008BC0C4;
}

void june_evt_func_008C2300(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->lr = 0x008C2314; june_evt_func_00828220(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_008E28C4(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read16(ctx->gpr[3] + 0xCE);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E28FC;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[4];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E29BC;
loc_008E28FC:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E291C;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3DB);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E2918; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E291C:
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x47F8);
        ctx->lr = 0x008E2924; june_evt_func_00834F80(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2954;
        ;
loc_008E2938:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0xC8);
        ctx->lr = 0x008E294C; june_evt_func_008E28C4(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E2938;
loc_008E2954:
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E297C;
loc_008E2960:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->lr = 0x008E2970; june_evt_func_008F3EB4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E2960;
loc_008E297C:
        ctx->gpr[3] = ctx->gpr[30] + (int64_t)(0xD0);
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008E298C; june_evt_func_00878FA4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xEC);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E29A8;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->lr = 0x008E29A4; june_evt_func_00875AEC(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E29A8:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x47F4);
        ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x1254);
        ctx->lr = 0x008E29B8; june_evt_func_00884DB8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E29BC:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
}

void june_evt_func_008E29D8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3FE);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2A0C;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4808);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->lr = 0x008E2A08; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E2A0C:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[4] = (int64_t)(int32_t)(1);
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->lr = ctx->gpr[0];
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008E28C4; return; }
}

void june_evt_func_008E2D48(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_24 = ctx->gpr[24];
        uint64_t _cs_25 = ctx->gpr[25];
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
        ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2F48;
        ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0xC7);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x47B0);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x008E2DA0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E2DA4:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x47F4);
        ctx->gpr[4] = (int64_t)(int32_t)(3);
        ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[6] = (int64_t)(int32_t)(0xC9);
        ctx->gpr[30] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x1254);
        ctx->lr = 0x008E2DC4; june_evt_func_00884FE0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2F04;
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(3);
        vm_write32(ctx->gpr[30] + 0xE4, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0xE8, ctx->gpr[29]);
        ctx->lr = 0x008E2DF0; june_evt_func_008F38D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ctx->gpr[25] + (int64_t)(0xD0);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[4] = ctx->gpr[24] + (int64_t)(0xD0);
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[29]);
        vm_write32(ctx->gpr[9] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[29]);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[29]);
        ctx->lr = 0x008E2E1C; june_evt_func_0087948C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = (int64_t)(int32_t)(0xA0);
        ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0xE8);
        ctx->gpr[11] = (int64_t)(int32_t)(0x90);
        ctx->gpr[0] = vm_read16(ctx->gpr[28] + 0xCC);
        ctx->gpr[10] = (int64_t)(int32_t)(0xB0);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint64_t ea = ((ctx->gpr[27] + ctx->gpr[9])) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
        vm_write32(ctx->gpr[30] + 0xC0, ctx->gpr[25]);
        { uint64_t ea = ((ctx->gpr[27] + ctx->gpr[11])) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
        { uint64_t ea = ((ctx->gpr[27] + ctx->gpr[10])) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
        { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
        { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[11])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
        { uint64_t ea = ((ctx->gpr[30] + ctx->gpr[10])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
        vm_write16(ctx->gpr[30] + 0xCC, ctx->gpr[0]);
        vm_write16(ctx->gpr[30] + 0xCE, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0xC4, ctx->gpr[29]);
        vm_write32(ctx->gpr[30] + 0xC8, ctx->gpr[29]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E2E78;
        goto loc_008E2ED0;
loc_008E2E6C:
        ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2ED0;
loc_008E2E78:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0xE5);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2F38;
loc_008E2E90:
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x008E2E98; june_evt_func_008F3A78(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2E6C;
        ctx->lr = 0x008E2EB0; june_evt_func_008F3FD4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008E2EC0; june_evt_func_008F38E8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E2E78;
loc_008E2ED0:
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xEC);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E2F50;
        ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x10);
        vm_write32(ctx->gpr[30] + 0xEC, ctx->gpr[9]);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008E2EEC; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E2EF0:
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x47F8);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0xE0, tmp); }
        ctx->lr = 0x008E2F00; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E2F04:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[24] = _cs_24;
        ctx->gpr[25] = _cs_25;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
        return;
loc_008E2F38:
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x47AC);
        ctx->lr = 0x008E2F40; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008E2E90;
loc_008E2F48:
        ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x4830);
        goto loc_008E2DA4;
loc_008E2F50:
        vm_write32(ctx->gpr[30] + 0xEC, ctx->gpr[9]);
        goto loc_008E2EF0;
}

void june_evt_func_008E3EF8(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
        vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[14]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[15]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[16]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[17]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[18]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[19]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[20]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[21]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[29]);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
        vm_write32(ctx->gpr[1] + 0x128, ctx->gpr[12]);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E3F80;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E3F7C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E3F80:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E3F9C;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[19] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4548;
loc_008E3F9C:
        ctx->gpr[0] = ctx->gpr[27] ^ 0x3;
        ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0xE8);
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[24] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[19] = ctx->gpr[24] | ctx->gpr[24];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E3FFC;
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E45B4;
loc_008E3FCC:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[19] = ctx->gpr[19] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E3FCC;
loc_008E3FFC:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4548;
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->cr & 0x0000000Fu;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 28, 0, 3, 0, 0);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[9]);
        { uint32_t field = (ctx->cr >> 0) & 0xF; ctx->cr = (ctx->cr & ~(0xFu << 16)) | (field << 16); }
loc_008E401C:
        ctx->gpr[14] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[14] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4040;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E403C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E4040:
        ctx->gpr[0] = vm_read16(ctx->gpr[14] + 0xCE);
        ctx->gpr[15] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4538;
        ctx->gpr[11] = vm_read32(ctx->gpr[14] + 0xE8);
        ctx->gpr[15] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4098;
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E45D4;
loc_008E4068:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[15] = ctx->gpr[15] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4068;
loc_008E4098:
        ctx->gpr[0] = vm_read32(ctx->gpr[14] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4538;
loc_008E40A4:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E40CC;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E40C8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E40CC:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[16] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4524;
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xE8);
        ctx->gpr[16] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4124;
        if ((((ctx->cr >> 17) & 1) == 1)) goto loc_008E45F4;
loc_008E40F4:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[16] = ctx->gpr[16] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E40F4;
loc_008E4124:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4524;
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 20)) | (cr_val << 20); }
loc_008E4138:
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[10]);
        ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4160;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E415C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E4160:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[17] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4510;
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xE8);
        ctx->gpr[17] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E41B8;
        if ((((ctx->cr >> 17) & 1) == 1)) goto loc_008E4634;
loc_008E4188:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[17] = ctx->gpr[17] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4188;
loc_008E41B8:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4510;
loc_008E41C8:
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[10]);
        ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E41F0;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E41EC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E41F0:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[18] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E44FC;
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xE8);
        ctx->gpr[18] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4258;
        ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[10] = ppc_rlw64(ctx->gpr[10], 4, 0, 31, 0, 0);
        ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[10] & 0x0000000Fu);
        ctx->gpr[10] = ppc_rlw64(ctx->gpr[10], 28, 0, 31, 0, 0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4614;
loc_008E4228:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[18] = ctx->gpr[18] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4228;
loc_008E4258:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E44FC;
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
loc_008E426C:
        ctx->gpr[20] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4290;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E428C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E4290:
        ctx->gpr[0] = vm_read16(ctx->gpr[20] + 0xCE);
        ctx->gpr[21] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E44EC;
        ctx->gpr[11] = vm_read32(ctx->gpr[20] + 0xE8);
        ctx->gpr[21] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E42F4;
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
        ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[0] & 0x0000000Fu);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 0, 31, 0, 0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E46B4;
loc_008E42C4:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[21] = ctx->gpr[21] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E42C4;
loc_008E42F4:
        ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E44EC;
loc_008E4300:
        ctx->gpr[22] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4324;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E4320; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E4324:
        ctx->gpr[0] = vm_read16(ctx->gpr[22] + 0xCE);
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E44DC;
        ctx->gpr[11] = vm_read32(ctx->gpr[22] + 0xE8);
        ctx->gpr[23] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4378;
        if ((((ctx->cr >> 21) & 1) == 1)) goto loc_008E4694;
loc_008E4348:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[23] = ctx->gpr[23] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4348;
loc_008E4378:
        ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E44DC;
        ctx->gpr[26] = ppc_rldicl(ctx->gpr[27], 0, 32);
loc_008E4388:
        ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E43AC;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E43A8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E43AC:
        ctx->gpr[0] = vm_read16(ctx->gpr[25] + 0xCE);
        ctx->gpr[28] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E44CC;
        ctx->gpr[11] = vm_read32(ctx->gpr[25] + 0xE8);
        ctx->gpr[28] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4400;
        if ((((ctx->cr >> 21) & 1) == 1)) goto loc_008E4674;
loc_008E43D0:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[28] = ctx->gpr[28] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E43D0;
loc_008E4400:
        ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E44CC;
        ;
loc_008E4410:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4434;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E4430; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E4434:
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0xCE);
        ctx->gpr[31] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E44BC;
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xE8);
        ctx->gpr[31] = ctx->gpr[24] | ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4488;
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E4654;
loc_008E4458:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4458;
loc_008E4488:
        ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E44BC;
        ;
loc_008E4498:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008E44AC; june_evt_func_008E3EF8(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0xC8);
        ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4498;
loc_008E44BC:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC8);
        ctx->gpr[28] = ctx->gpr[28] + ctx->gpr[31];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4410;
loc_008E44CC:
        ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0xC8);
        ctx->gpr[23] = ctx->gpr[23] + ctx->gpr[28];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4388;
loc_008E44DC:
        ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0xC8);
        ctx->gpr[21] = ctx->gpr[21] + ctx->gpr[23];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4300;
loc_008E44EC:
        ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0xC8);
        ctx->gpr[18] = ctx->gpr[18] + ctx->gpr[21];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E426C;
loc_008E44FC:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[17] = ctx->gpr[17] + ctx->gpr[18];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E41C8;
loc_008E4510:
        ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[16] = ctx->gpr[16] + ctx->gpr[17];
        ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4138;
loc_008E4524:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[15] = ctx->gpr[15] + ctx->gpr[16];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E40A4;
loc_008E4538:
        ctx->gpr[0] = vm_read32(ctx->gpr[14] + 0xC8);
        ctx->gpr[19] = ctx->gpr[19] + ctx->gpr[15];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E401C;
loc_008E4548:
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x128);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[19];
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
        ctx->gpr[14] = _cs_14;
        ctx->cr = (ctx->cr & 0xFF0FFFFFu) | ((uint32_t)ctx->gpr[12] & 0x00F00000u);
        ctx->gpr[15] = _cs_15;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[16] = _cs_16;
        ctx->cr = (ctx->cr & 0xFFF0FFFFu) | ((uint32_t)ctx->gpr[12] & 0x000F0000u);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
        return;
loc_008E45B4:
        ctx->gpr[3] = (int64_t)(int32_t)(0);
loc_008E45B8:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E45B8;
        ctx->gpr[19] = ctx->gpr[24] + ctx->gpr[3];
        goto loc_008E3FFC;
loc_008E45D4:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E45D8:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E45D8;
        ctx->gpr[15] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E4098;
loc_008E45F4:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E45F8:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E45F8;
        ctx->gpr[16] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E4124;
loc_008E4614:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E4618:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4618;
        ctx->gpr[18] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E4258;
loc_008E4634:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E4638:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4638;
        ctx->gpr[17] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E41B8;
loc_008E4654:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E4658:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4658;
        ctx->gpr[31] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E4488;
loc_008E4674:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E4678:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4678;
        ctx->gpr[28] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E4400;
loc_008E4694:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E4698:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4698;
        ctx->gpr[23] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E4378;
loc_008E46B4:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E46B8:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E46B8;
        ctx->gpr[21] = ctx->gpr[24] + ctx->gpr[10];
        goto loc_008E42F4;
}

void june_evt_func_008E46D4(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4740;
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x584);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4808);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E4718; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4740;
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x563);
        ctx->lr = 0x008E4738; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ;
loc_008E4740:
        ctx->gpr[0] = ctx->gpr[30] ^ 0x3;
        ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0xE8);
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[31] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E47A0;
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E47F8;
        ;
loc_008E4770:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        ctx->gpr[0] = ctx->gpr[0] ^ ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4770;
loc_008E47A0:
        ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E47D4;
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
loc_008E47B0:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008E47C4; june_evt_func_008E3EF8(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0xC8);
        ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E47B0;
loc_008E47D4:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[31];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_008E47F8:
        ctx->gpr[10] = (int64_t)(int32_t)(0);
loc_008E47FC:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E47FC;
        ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[10];
        goto loc_008E47A0;
}

void june_evt_func_008E4818(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xCE);
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E485C;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E490C;
loc_008E485C:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008E4864; june_evt_func_008E2D48(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E490C;
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E498C;
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
        vm_write32(ctx->gpr[29] + 0xC0, ctx->gpr[30]);
loc_008E4884:
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E490C;
        ctx->gpr[27] = (int64_t)(int32_t)(0);
        goto loc_008E48A4;
loc_008E4898:
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E48E4;
loc_008E48A4:
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[5] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E4898;
        ctx->lr = 0x008E48C4; june_evt_func_008E4818(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4934;
        vm_write32(ctx->gpr[3] + 0xC8, ctx->gpr[27]);
        vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[28]);
        ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E48A4;
loc_008E48E4:
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E490C;
        ctx->gpr[0] = ctx->gpr[27] | ctx->gpr[27];
loc_008E48F0:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0xC4);
        vm_write32(ctx->gpr[29] + 0xC4, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC8);
        vm_write32(ctx->gpr[9] + 0xC8, ctx->gpr[11]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E48F0;
loc_008E490C:
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
loc_008E4934:
        ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E4954;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4808);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3FE);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->lr = 0x008E4950; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E4954:
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(1);
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008E4964; june_evt_func_008E28C4(ctx); DRAIN_TRAMPOLINE(ctx);
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
loc_008E498C:
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        goto loc_008E4884;
}

void june_evt_func_008E4994(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3C6);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E49C8;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4808);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->lr = 0x008E49C4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E49C8:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[31] = _cs_31;
        ctx->gpr[5] = (int64_t)(int32_t)(1);
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_008E4818; return; }
}

void june_evt_func_008E6790(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + -0x130, ctx->gpr[1]); ctx->gpr[1] += -0x130;
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[21]);
        vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[14]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[15]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[16]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[17]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[18]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[19]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[20]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[28]);
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[25] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[21] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        vm_write32(ctx->gpr[1] + 0x138, ctx->gpr[12]);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6820;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E681C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E6820:
        ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E683C;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[31];
        ctx->gpr[14] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6CC8;
loc_008E683C:
        { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E6E64;
        ctx->gpr[14] = (int64_t)(int32_t)(0);
loc_008E6848:
        ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E68C8;
        { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E689C;
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[21], 0, 32);
        goto loc_008E6870;
loc_008E6864:
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E68C8;
loc_008E6870:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6864;
        ctx->gpr[14] = ctx->gpr[14] + (int64_t)(1);
        ctx->lr = 0x008E6890; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E6864;
loc_008E689C:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E68A0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[14] = ctx->gpr[14] + (int64_t)(1);
        ctx->lr = 0x008E68B0; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E68A0;
loc_008E68C8:
        ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6CC8;
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[21], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 20)) | (cr_val << 20); }
        ctx->gpr[9] = ctx->cr & 0x00F00000u;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 8, 0, 3, 0, 0);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[9]);
loc_008E68EC:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[9]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6914;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E6910; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E6914:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[15] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0xCE);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6CB4;
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E6E80;
loc_008E692C:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6984;
        if ((((ctx->cr >> 21) & 1) == 0)) goto loc_008E6950;
        goto loc_008E6E9C;
loc_008E6944:
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6984;
loc_008E6950:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6944;
        ctx->lr = 0x008E696C; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[15] = ctx->gpr[15] + (int64_t)(1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6950;
loc_008E6984:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6CB4;
        ctx->gpr[17] = ppc_rldicl(ctx->gpr[21], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 16)) | (cr_val << 16); }
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[17]);
loc_008E69A0:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[9]);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E69CC;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E69C4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
loc_008E69CC:
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0xCE);
        ctx->gpr[16] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6CA0;
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E6D88;
loc_008E69E0:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6A38;
        if ((((ctx->cr >> 21) & 1) == 0)) goto loc_008E6A04;
        goto loc_008E6DD4;
loc_008E69F8:
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6A38;
loc_008E6A04:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E69F8;
        ctx->lr = 0x008E6A20; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[16] = ctx->gpr[16] + (int64_t)(1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6A04;
loc_008E6A38:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6CA0;
loc_008E6A48:
        ctx->gpr[18] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[18] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6A6C;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E6A68; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E6A6C:
        ctx->gpr[0] = vm_read16(ctx->gpr[18] + 0xCE);
        ctx->gpr[19] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6C90;
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E6D6C;
loc_008E6A80:
        ctx->gpr[31] = vm_read32(ctx->gpr[18] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6AE4;
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
        ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[0] & 0x0000000Fu);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 0, 31, 0, 0);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6AB0;
        goto loc_008E6DA4;
loc_008E6AA4:
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6AE4;
loc_008E6AB0:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = ctx->gpr[17] | ctx->gpr[17];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6AA4;
        ctx->lr = 0x008E6ACC; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[19] = ctx->gpr[19] + (int64_t)(1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6AB0;
loc_008E6AE4:
        ctx->gpr[31] = vm_read32(ctx->gpr[18] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6C90;
        ctx->gpr[20] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[24] = ppc_rldicl(ctx->gpr[25], 0, 32);
loc_008E6AF8:
        ctx->gpr[22] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6B1C;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E6B18; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E6B1C:
        ctx->gpr[0] = vm_read16(ctx->gpr[22] + 0xCE);
        ctx->gpr[23] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6C80;
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E6D50;
loc_008E6B30:
        ctx->gpr[31] = vm_read32(ctx->gpr[22] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6B94;
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x90);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
        ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[0] & 0x0000000Fu);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 0, 31, 0, 0);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6B60;
        goto loc_008E6E34;
loc_008E6B54:
        ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6B94;
loc_008E6B60:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6B54;
        ctx->lr = 0x008E6B7C; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[23] = ctx->gpr[23] + (int64_t)(1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6B60;
loc_008E6B94:
        ctx->gpr[31] = vm_read32(ctx->gpr[22] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6C80;
        ctx->gpr[26] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ;
loc_008E6BA8:
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6BCC;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->lr = 0x008E6BC8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E6BCC:
        ctx->gpr[0] = vm_read16(ctx->gpr[27] + 0xCE);
        ctx->gpr[28] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6C70;
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008E6D34;
loc_008E6BE0:
        ctx->gpr[29] = vm_read32(ctx->gpr[27] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6C38;
        if ((((ctx->cr >> 17) & 1) == 0)) goto loc_008E6C00;
        goto loc_008E6E04;
loc_008E6BF4:
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6C38;
loc_008E6C00:
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[25]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6BF4;
        ctx->lr = 0x008E6C1C; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[29]);
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6C00;
        ;
loc_008E6C38:
        ctx->gpr[29] = vm_read32(ctx->gpr[27] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E6C70;
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E6C48:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[4] = ctx->gpr[24] | ctx->gpr[24];
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[6] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008E6C60; june_evt_func_008E6790(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0xC8);
        ctx->gpr[28] = ctx->gpr[28] + ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6C48;
loc_008E6C70:
        ctx->gpr[31] = vm_read32(ctx->gpr[27] + 0xC8);
        ctx->gpr[23] = ctx->gpr[23] + ctx->gpr[28];
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6BA8;
loc_008E6C80:
        ctx->gpr[31] = vm_read32(ctx->gpr[22] + 0xC8);
        ctx->gpr[19] = ctx->gpr[19] + ctx->gpr[23];
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6AF8;
loc_008E6C90:
        ctx->gpr[31] = vm_read32(ctx->gpr[18] + 0xC8);
        ctx->gpr[16] = ctx->gpr[16] + ctx->gpr[19];
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6A48;
loc_008E6CA0:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[15] = ctx->gpr[15] + ctx->gpr[16];
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E69A0;
loc_008E6CB4:
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[14] = ctx->gpr[14] + ctx->gpr[15];
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0xC8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E68EC;
loc_008E6CC8:
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x138);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[14];
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
        ctx->gpr[14] = _cs_14;
        ctx->cr = (ctx->cr & 0xFF0FFFFFu) | ((uint32_t)ctx->gpr[12] & 0x00F00000u);
        ctx->gpr[15] = _cs_15;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[16] = _cs_16;
        ctx->cr = (ctx->cr & 0xFFF0FFFFu) | ((uint32_t)ctx->gpr[12] & 0x000F0000u);
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x130);
        return;
loc_008E6D34:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E6D44; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E6BE0;
loc_008E6D50:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[23] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E6D60; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E6B30;
loc_008E6D6C:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[19] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E6D7C; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E6A80;
loc_008E6D88:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[16] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E6D98; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E69E0;
loc_008E6DA4:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E6DA8:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[19] = ctx->gpr[19] + (int64_t)(1);
        ctx->lr = 0x008E6DB8; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6DA8;
        goto loc_008E6AE4;
loc_008E6DD4:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E6DD8:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[16] = ctx->gpr[16] + (int64_t)(1);
        ctx->lr = 0x008E6DE8; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6DD8;
        goto loc_008E6A38;
loc_008E6E04:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E6E08:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
        ctx->lr = 0x008E6E18; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[29]);
        ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6E08;
        goto loc_008E6C38;
loc_008E6E34:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E6E38:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[23] = ctx->gpr[23] + (int64_t)(1);
        ctx->lr = 0x008E6E48; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6E38;
        goto loc_008E6B94;
loc_008E6E64:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[14] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E6E74; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[30]);
        goto loc_008E6848;
loc_008E6E80:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[15] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E6E90; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E692C;
loc_008E6E9C:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008E6EA0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[15] = ctx->gpr[15] + (int64_t)(1);
        ctx->lr = 0x008E6EB0; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E6EA0;
        goto loc_008E6984;
}

void june_evt_func_008E718C(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[25] = ctx->gpr[5] | ctx->gpr[5];
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
        ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E7208;
        ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x4830);
        ctx->gpr[4] = (int64_t)(int32_t)(0x5BE);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4808);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x008E71E4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E7208;
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x482C);
        ctx->gpr[4] = (int64_t)(int32_t)(0x599);
        ctx->lr = 0x008E7204; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E7208:
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[28] = (int64_t)(int32_t)(0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E72E0;
loc_008E7214:
        ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0xE8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E7270;
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E72FC;
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
        goto loc_008E723C;
loc_008E7230:
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E7270;
loc_008E723C:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[27]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7230;
        ctx->lr = 0x008E7258; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E723C;
loc_008E7270:
        ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0xC4);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E72B0;
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ;
loc_008E7288:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[6] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008E72A0; june_evt_func_008E6790(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0xC8);
        ctx->gpr[28] = ctx->gpr[28] + ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7288;
loc_008E72B0:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[28];
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
loc_008E72E0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = (int64_t)(int32_t)(1);
        ctx->lr = 0x008E72F0; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        goto loc_008E7214;
loc_008E72FC:
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
loc_008E7300:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
        ctx->lr = 0x008E7310; june_evt_func_00877804(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[9] + 0x8);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7300;
        goto loc_008E7270;
}

void june_evt_func_008E7998(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x47F4);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
        ctx->gpr[4] = (int64_t)(int32_t)(3);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4830);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[26]);
        ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
        ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x1254);
        ctx->gpr[6] = (int64_t)(int32_t)(0x79);
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xB0, tmp); }
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[31]);
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xB8, tmp); }
        ctx->gpr[31] = ctx->gpr[26] | ctx->gpr[26];
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
        ctx->lr = 0x008E79EC; june_evt_func_00884FE0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = (int64_t)(int32_t)(0);
        ctx->gpr[27] = (int64_t)(int32_t)(0xD0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008E7AF8;
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x4828); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[4] = (int64_t)(int32_t)(3);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        vm_write32(ctx->gpr[28] + 0xC0, ctx->gpr[0]);
        ctx->gpr[0] = (int64_t)(int32_t)(-1);
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0xE0, tmp); }
        vm_write32(ctx->gpr[28] + 0xC4, ctx->gpr[29]);
        vm_write16(ctx->gpr[28] + 0xCC, ctx->gpr[0]);
        vm_write32(ctx->gpr[28] + 0xC8, ctx->gpr[29]);
        vm_write16(ctx->gpr[28] + 0xCE, ctx->gpr[29]);
        vm_write32(ctx->gpr[28] + 0xE4, ctx->gpr[29]);
        vm_write32(ctx->gpr[28] + 0xE8, ctx->gpr[29]);
        ctx->lr = 0x008E7A48; june_evt_func_008F38D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x46C0);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x46BC);
        ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x46B8);
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(0x50);
        ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x46B4);
        ctx->gpr[7] = (int64_t)(int32_t)(0x10);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[6] = (int64_t)(int32_t)(0x20);
        { uint64_t ea = (ctx->gpr[3]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        { uint64_t ea = (ctx->gpr[5]) & ~0xFULL; vm_raw_ld16(&ctx->vr[13], ea); }
        { uint64_t ea = (ctx->gpr[4]) & ~0xFULL; vm_raw_ld16(&ctx->vr[11], ea); }
        ctx->gpr[11] = ctx->gpr[30] + (int64_t)(0x10);
        { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
        ctx->gpr[10] = ctx->gpr[2] + (int64_t)(-18472);
        { uint64_t ea = (ctx->gpr[9]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[7])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
        ctx->gpr[27] = ctx->gpr[30] + (int64_t)(0xD0);
        { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
        { uint64_t ea = ((ctx->gpr[9] + ctx->gpr[6])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[11]); }
        ctx->gpr[9] = (int64_t)(int32_t)(0xA0);
        { uint64_t ea = ctx->gpr[10]; uint32_t sh = (uint32_t)(ea & 0xF); uint8_t* m = vm_base + (uint32_t)(ea & ~0xFULL); uint8_t* d = (uint8_t*)&ctx->vr[12]; for (int i = 0; i < 16; i++) d[i] = (sh + (uint32_t)i < 16) ? m[sh + i] : 0; }
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[27], 0, 32);
        { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[0])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
        { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
        { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[7])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[13]); }
        { uint64_t ea = ((ctx->gpr[11] + ctx->gpr[6])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[11]); }
        ctx->gpr[11] = (int64_t)(int32_t)(0xB0);
        { uint64_t ea = ((ctx->gpr[28] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
        ctx->gpr[9] = (int64_t)(int32_t)(0x90);
        { uint32_t* d = (uint32_t*)&ctx->vr[12]; uint32_t val = ((uint32_t*)&ctx->vr[12])[0]; d[0] = d[1] = d[2] = d[3] = val; }
        { uint64_t* d = (uint64_t*)&ctx->vr[0]; uint64_t* a = (uint64_t*)&ctx->vr[0]; uint64_t* b = (uint64_t*)&ctx->vr[0]; d[0] = a[0] ^ b[0]; d[1] = a[1] ^ b[1]; }
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x47F8);
        vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[29]);
        vm_write32(ctx->gpr[28] + 0xEC, ctx->gpr[29]);
        { uint64_t ea = ((ctx->gpr[28] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
        { uint64_t ea = ((ctx->gpr[28] + ctx->gpr[11])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[12]); }
        vm_write32(ctx->gpr[10] + 0x4, ctx->gpr[29]);
        vm_write32(ctx->gpr[10] + 0x8, ctx->gpr[29]);
        vm_write32(ctx->gpr[10] + 0xC, ctx->gpr[29]);
        ctx->lr = 0x008E7AF4; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008E7AF8:
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7B08; june_evt_func_008792D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7B14; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[30] = ctx->fpr[1];
        ctx->lr = 0x008E7B24; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[31] = ctx->fpr[1];
        ctx->lr = 0x008E7B34; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(0x90);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
        { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
        ctx->lr = 0x008E7B54; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[29] = ctx->fpr[1];
        ctx->lr = 0x008E7B64; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[30] = ctx->fpr[1];
        ctx->lr = 0x008E7B74; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[31] = ctx->fpr[1];
        ctx->lr = 0x008E7B84; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(0xA0);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
        { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
        { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
        ctx->lr = 0x008E7BA8; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[30] = ctx->fpr[1];
        ctx->lr = 0x008E7BB8; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->fpr[31] = ctx->fpr[1];
        ctx->lr = 0x008E7BC8; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(0xB0);
        ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x4);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x109 << 16);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
        { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
        { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7CC0;
loc_008E7BF0:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7BF8; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7C38;
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        ;
loc_008E7C10:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
        ctx->lr = 0x008E7C1C; june_evt_func_008F43E0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008E7C2C; june_evt_func_008F38E8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7C10;
loc_008E7C38:
        ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x106 << 16);
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008E7CA0;
loc_008E7C48:
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x110 << 16);
        ctx->gpr[0] = ctx->gpr[0] | 0x4230;
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7C68;
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7C60; june_evt_func_00878664(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0xE0, tmp); }
loc_008E7C68:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[29], &tmp, 8); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
        return;
loc_008E7CA0:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7CA8; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 0, 24, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7CD0;
        ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
        goto loc_008E7C48;
loc_008E7CC0:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7CC8; june_evt_func_008784F4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        goto loc_008E7BF0;
loc_008E7CD0:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x008E7CD8; june_evt_func_00876480(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
        vm_write32(ctx->gpr[28] + 0xEC, ctx->gpr[3]);
        goto loc_008E7C48;
}

void june_evt_func_008E7CE8(ppu_context* ctx) {
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
        vm_write64(ctx->gpr[1] + -0x130, ctx->gpr[1]); ctx->gpr[1] += -0x130;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[0]);
        ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[14]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[15]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[16]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[17]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[18]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[19]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[20]);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[21]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[30]);
        ctx->lr = 0x008E7D44; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[0]);
        ctx->lr = 0x008E7D58; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[3]);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7FBC;
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = (int64_t)(int32_t)(0);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
loc_008E7D7C:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7D84; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->lr = 0x008E7D98; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[3]);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F94;
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[14] = (int64_t)(int32_t)(0);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[31]);
loc_008E7DB8:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7DC0; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        ctx->lr = 0x008E7DD4; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[15] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F78;
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[16] = (int64_t)(int32_t)(0);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
loc_008E7DF0:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7DF8; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->lr = 0x008E7E0C; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F60;
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[18] = (int64_t)(int32_t)(0);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[31]);
loc_008E7E28:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7E30; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[21] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x008E7E40; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[19] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F48;
        ctx->gpr[20] = (int64_t)(int32_t)(0);
loc_008E7E54:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7E5C; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[23] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x008E7E6C; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F30;
        ctx->gpr[24] = (int64_t)(int32_t)(0);
loc_008E7E80:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7E88; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[26] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x008E7E98; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F18;
        ctx->gpr[27] = (int64_t)(int32_t)(0);
        ;
loc_008E7EB0:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x008E7EB8; june_evt_func_008E7998(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->lr = 0x008E7EC8; june_evt_func_008785D0(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008E7F00;
        ctx->gpr[31] = (int64_t)(int32_t)(0);
        ;
loc_008E7EE0:
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
        ctx->lr = 0x008E7EEC; june_evt_func_008E7CE8(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x008E7EF8; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7EE0;
loc_008E7F00:
        ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        ctx->lr = 0x008E7F10; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int32_t)ctx->gpr[27]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7EB0;
loc_008E7F18:
        ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[24] = ctx->gpr[24] + (int64_t)(1);
        ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
        ctx->lr = 0x008E7F28; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[24]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7E80;
loc_008E7F30:
        ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
        ctx->gpr[20] = ctx->gpr[20] + (int64_t)(1);
        ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
        ctx->lr = 0x008E7F40; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int32_t)ctx->gpr[20]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7E54;
loc_008E7F48:
        ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[18] = ctx->gpr[18] + (int64_t)(1);
        ctx->lr = 0x008E7F58; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[17]; int64_t b = (int32_t)ctx->gpr[18]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7E28;
loc_008E7F60:
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[16] = ctx->gpr[16] + (int64_t)(1);
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x88);
        ctx->lr = 0x008E7F70; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[15]; int64_t b = (int32_t)ctx->gpr[16]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7DF0;
loc_008E7F78:
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x88);
        ctx->gpr[14] = ctx->gpr[14] + (int64_t)(1);
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->lr = 0x008E7F88; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[14]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7DB8;
loc_008E7F94:
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x74);
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x98);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
        ctx->lr = 0x008E7FAC; june_evt_func_008E7484(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x78);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x74);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008E7D7C;
loc_008E7FBC:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
        ctx->gpr[3] = vm_read64(ctx->gpr[1] + 0x98);
        ctx->gpr[14] = _cs_14;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[15] = _cs_15;
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
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x130);
        return;
}

void june_evt_func_008F3A78(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x1DB);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008F3B10;
loc_008F3AA4:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 6, 0, 25, 0, 0);
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4410);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
        ctx->gpr[10] = (int64_t)(int32_t)(1);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x30);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008F3AF4;
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x008F3AEC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        ctx->gpr[10] = (int64_t)(int8_t)ctx->gpr[3];
loc_008F3AF4:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[10];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_008F3B10:
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4418);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x440C);
        ctx->lr = 0x008F3B1C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
        goto loc_008F3AA4;
}

void june_evt_func_008F3FD4(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x110);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008F4014;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4418);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4404);
        ctx->lr = 0x008F400C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
loc_008F4014:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 6, 0, 25, 0, 0);
        ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4410);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[10];
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[11] + 0x4);
        ctx->lr = 0x008F4048; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
}

void june_evt_func_008F43E0(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->lr = 0x008F43FC; june_evt_func_0087861C(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { uint64_t a = (uint32_t)ctx->gpr[3]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[4] = (int64_t)(int32_t)(0x196);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008F4420;
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4418);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x43E4);
        ctx->lr = 0x008F441C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
loc_008F4420:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 3, 0, 28, 0, 0);
        ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x4410);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 6, 0, 25, 0, 0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[10];
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xC);
        ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[11] + 0x4);
        ctx->lr = 0x008F4458; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[30] = _cs_30;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
}

void june_evt_func_00AED5D8(ppu_context* ctx) {
    PPU_MM_SNAP
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9E0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00AED5FC;
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9DC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->gpr[4] = ctx->gpr[4] | 0x1;
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9D8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
        ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[0], ctx->fpr[12], 1, 0));
        ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
loc_00AED5FC:
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x9D4);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 28, 28, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[4], 0, 30, 30, 0, 0);
        ctx->gpr[10] = ppc_rlw64(ctx->gpr[4], 2, 27, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->fpr[0] = -ctx->fpr[13];
        { double a = ctx->fpr[0]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00AED64C;
        { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00AED64C;
loc_00AED62C:
        if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00AED634;
        ctx->fpr[1] = -ctx->fpr[1];
loc_00AED634:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x9C4);
        { uint32_t tmp = vm_read32((ctx->gpr[9] + ctx->gpr[10])); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
        if ((((ctx->cr >> 25) & 1) == 1)) return;
        ctx->fpr[1] = -ctx->fpr[1];
        return;
loc_00AED64C:
        ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[1]));
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x9D8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[13]));
        ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[12], ctx->fpr[11], 0, 0));
        ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[9], ctx->fpr[1], 0, 0));
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[12], ctx->fpr[10], 0, 0));
        ctx->fpr[1] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
        goto loc_00AED62C;
}

void june_evt_func_00D16584(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(2);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        return;
}

void june_evt_func_00D16590(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(3);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        return;
}

void june_evt_func_00D29898(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[10] = vm_read16(ctx->gpr[3] + 0x0);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
        ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 60, 53);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0x7FF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D299A4;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D29980;
loc_00D298D4:
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0x7FE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D29914;
loc_00D298DC:
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[10];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D299D0;
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x50D8);
        { uint64_t tmp = vm_read64(ctx->gpr[9] + 0x0); memcpy(&ctx->fpr[0], &tmp, 8); }
loc_00D298F0:
        ctx->gpr[11] = (int64_t)(int32_t)(1);
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
loc_00D298F8:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[11];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_00D29914:
        ctx->gpr[3] = ctx->gpr[30] + ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0x7FE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D298DC;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D29A10;
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(-1);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)-52; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[7] = ppc_rlw64(ctx->gpr[10], 0, 0, 16, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] | 0x10;
        ctx->gpr[4] = ctx->gpr[7] | ctx->gpr[7];
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[10] = ctx->gpr[0] | ctx->gpr[0];
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D29A30;
        ctx->gpr[8] = (int64_t)(int32_t)(0);
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[7]);
        vm_write16(ctx->gpr[31] + 0x6, ctx->gpr[8]);
        vm_write16(ctx->gpr[31] + 0x2, ctx->gpr[8]);
        vm_write16(ctx->gpr[31] + 0x4, ctx->gpr[8]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        ctx->gpr[30] = _cs_30;
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[11];
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_00D29980:
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        ctx->lr = 0x00D29988; june_evt_func_00D38524(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D298F8;
        ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
        goto loc_00D298D4;
loc_00D299A4:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D299E0;
loc_00D299B0:
        ctx->gpr[11] = (int64_t)(int32_t)(2);
loc_00D299B4:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[11];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_00D299D0:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x50D8);
        { uint64_t tmp = vm_read64(ctx->gpr[9] + 0x0); memcpy(&ctx->fpr[0], &tmp, 8); }
        ctx->fpr[0] = -ctx->fpr[0];
        goto loc_00D298F0;
loc_00D299E0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x2);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D299B0;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D299B0;
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        ctx->gpr[11] = (int64_t)(int32_t)(1);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D298F8;
        ctx->gpr[11] = (int64_t)(int32_t)(2);
        goto loc_00D299B4;
loc_00D29A10:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 17, 11, 31, 0, 0);
        ctx->gpr[9] = (int64_t)(int16_t)ctx->gpr[3];
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 15, 0, 31, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 4, 0, 27, 0, 0);
        ctx->gpr[11] = (int64_t)(int32_t)(-1);
        ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D298F8;
loc_00D29A30:
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)-16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D29A74;
        ctx->gpr[8] = vm_read16(ctx->gpr[31] + 0x2);
        ctx->gpr[11] = vm_read16(ctx->gpr[31] + 0x4);
loc_00D29A44:
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x10);
        ctx->gpr[9] = ctx->gpr[11] | ctx->gpr[11];
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[3];
        ctx->gpr[11] = ctx->gpr[8] | ctx->gpr[8];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)-16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[8] = ctx->gpr[10] | ctx->gpr[10];
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        vm_write16(ctx->gpr[31] + 0x2, ctx->gpr[8]);
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[10]);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D29A44;
        vm_write16(ctx->gpr[31] + 0x4, ctx->gpr[11]);
        vm_write16(ctx->gpr[31] + 0x6, ctx->gpr[9]);
loc_00D29A74:
        ctx->gpr[0] = (uint64_t)0 - ctx->gpr[3];
        ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D29AD4;
        ctx->gpr[7] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x4);
        { uint64_t _a = ctx->gpr[3]; uint64_t _b = (uint64_t)(int64_t)(0x10); ctx->gpr[10] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x2);
        ctx->gpr[11] = vm_read16(ctx->gpr[31] + 0x6);
        ctx->gpr[6] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], (int)(ctx->gpr[3] & 0x3F));
        ctx->gpr[5] = (uint64_t)((ctx->gpr[10] & 0x20) ? 0u : ((uint32_t)ctx->gpr[7] << (ctx->gpr[10] & 0x1F)));
        ctx->gpr[8] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], (int)(ctx->gpr[3] & 0x3F));
        ctx->gpr[0] = (uint64_t)((ctx->gpr[10] & 0x20) ? 0u : ((uint32_t)ctx->gpr[0] << (ctx->gpr[10] & 0x1F)));
        ctx->gpr[9] = (uint64_t)((ctx->gpr[10] & 0x20) ? 0u : ((uint32_t)ctx->gpr[9] << (ctx->gpr[10] & 0x1F)));
        ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[11], (int)(ctx->gpr[3] & 0x3F));
        ctx->gpr[7] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[7], (int)(ctx->gpr[3] & 0x3F));
        ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
        ctx->gpr[6] = ctx->gpr[6] | ctx->gpr[9];
        ctx->gpr[8] = ctx->gpr[8] | ctx->gpr[5];
        ctx->gpr[10] = ctx->gpr[7] | ctx->gpr[7];
        vm_write16(ctx->gpr[31] + 0x6, ctx->gpr[11]);
        vm_write16(ctx->gpr[31] + 0x4, ctx->gpr[6]);
        vm_write16(ctx->gpr[31] + 0x2, ctx->gpr[8]);
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[7]);
loc_00D29AD4:
        ctx->gpr[0] = ctx->gpr[4] | ctx->gpr[10];
        ctx->gpr[11] = (int64_t)(int32_t)(-1);
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D298F8;
}

void june_evt_func_00D29AE4(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[11] = vm_read16(ctx->gpr[4] + 0x0);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ppc_rlw64(ctx->gpr[11], 28, 21, 31, 0, 0);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7FF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[4], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D29B9C;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D29B64;
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        ctx->lr = 0x00D29B28; june_evt_func_00D38524(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D29B60;
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        vm_write16(ctx->gpr[30] + 0x0, ctx->gpr[0]);
loc_00D29B44:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_00D29B60:
        ctx->gpr[11] = vm_read16(ctx->gpr[31] + 0x0);
loc_00D29B64:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 17, 11, 31, 0, 0);
        ctx->gpr[9] = ctx->gpr[3] + (int64_t)(-1022);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 15, 0, 31, 0, 0);
        ctx->gpr[3] = (int64_t)(int32_t)(-1);
        ctx->gpr[0] = ctx->gpr[0] | 0x3FE0;
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        vm_write16(ctx->gpr[30] + 0x0, ctx->gpr[9]);
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_00D29B9C:
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        vm_write16(ctx->gpr[30] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read16(ctx->gpr[4] + 0x0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 28, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D29BD4;
loc_00D29BB4:
        ctx->gpr[3] = (int64_t)(int32_t)(2);
loc_00D29BB8:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
loc_00D29BD4:
        ctx->gpr[0] = vm_read16(ctx->gpr[4] + 0x2);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D29BB4;
        ctx->gpr[0] = vm_read16(ctx->gpr[4] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D29BB4;
        ctx->gpr[0] = vm_read16(ctx->gpr[4] + 0x6);
        ctx->gpr[3] = (int64_t)(int32_t)(1);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D29B44;
        ctx->gpr[3] = (int64_t)(int32_t)(2);
        goto loc_00D29BB8;
}

void june_evt_func_00D2ED90(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[6] = vm_read32(ctx->gpr[6] + 0x0);
        ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[5] = ctx->gpr[3] | ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EF38;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x6);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EF74;
        ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[9];
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D2EF08;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 29, 0, 0);
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[0] = vm_read32((ctx->gpr[7] + ctx->gpr[9]));
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EF08;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[4], 1, 23, 30, 0, 0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[6] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[12] = (int64_t)(int32_t)(1);
        ctx->gpr[10] = vm_read16(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2EEC8;
        goto loc_00D2EF08;
loc_00D2EE10:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 19, 19, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EE2C;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 16, 31, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 24, 8, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 8, 0, 23, 0, 0);
        ctx->gpr[4] = ctx->gpr[0] | ctx->gpr[9];
loc_00D2EE2C:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 18, 18, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EE6C;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2EE48;
        ctx->gpr[10] = ctx->gpr[4] | ctx->gpr[4];
loc_00D2EE48:
        ctx->gpr[9] = ctx->gpr[5] + ctx->gpr[6];
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(1);
        ctx->gpr[12] = (int64_t)(int32_t)(0);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[10]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2EE6C;
        ctx->gpr[3] = (int64_t)(int32_t)(1);
loc_00D2EE6C:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 17, 17, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[3];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2EEE8;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[8];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D2EF08;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 2, 26, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[7] + ctx->gpr[9]));
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EF08;
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D2EF08;
        ctx->gpr[12] = ctx->gpr[12] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[12]; int64_t b = (int64_t)0xFEF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D2EF08;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[4], 1, 23, 30, 0, 0);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[10] = vm_read16(ctx->gpr[9] + 0x0);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 0, 16, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2EF08;
loc_00D2EEC8:
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[10];
        ctx->gpr[8] = ppc_rlw64(ctx->gpr[11], 24, 28, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D2EE10;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[4], 0, 0, 23, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 24, 31, 0, 0);
        ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[0];
        goto loc_00D2EE10;
loc_00D2EEE8:
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[8];
        vm_write16(ctx->gpr[31] + 0x6, ctx->gpr[0]);
loc_00D2EEF0:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[6];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->lr = ctx->gpr[0];
        return;
loc_00D2EF08:
        ctx->lr = 0x00D2EF0C; june_evt_func_00D23CA8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x8001 << 16);
        ctx->gpr[31] = _cs_31;
        ctx->gpr[0] = ctx->gpr[0] | 0x1D;
        ctx->gpr[6] = (int64_t)(int32_t)(-1);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[6];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->lr = ctx->gpr[0];
        return;
loc_00D2EF38:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2F040;
        ctx->gpr[10] = ppc_rlw64(ctx->gpr[4], 0, 16, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 16, 24, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2EFB4;
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(1);
        vm_write8(ctx->gpr[5] + 0x0, ctx->gpr[4]);
        ctx->gpr[6] = ctx->gpr[3] - ctx->gpr[5];
loc_00D2EF5C:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[6];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->lr = ctx->gpr[0];
        return;
loc_00D2EF74:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x51B0);
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[6], 0, 32);
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x6);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[10] = vm_read16(ctx->gpr[9] + 0x4);
        vm_write16(ctx->gpr[31] + 0x6, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[11]);
        vm_write16(ctx->gpr[31] + 0x4, ctx->gpr[10]);
        ctx->gpr[0] = vm_read16(ctx->gpr[8] + 0x0);
        ctx->gpr[31] = _cs_31;
        ctx->gpr[6] = ppc_rlw64(ctx->gpr[0], 0, 20, 23, 0, 0);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[6];
        ctx->lr = ctx->gpr[0];
        return;
loc_00D2EFB4:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 16, 20, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D2F024;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 26, 6, 31, 0, 0);
        ctx->gpr[0] = (int64_t)(int32_t)(-64);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        ctx->gpr[8] = (int64_t)(int32_t)(1);
        vm_write8(ctx->gpr[5] + 0x0, ctx->gpr[9]);
loc_00D2EFD8:
        ctx->gpr[0] = ctx->gpr[8] + (int64_t)(-1);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[8], 1, 0, 30, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[8], 3, 0, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[11];
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[8], 0, 32);
        ctx->gpr[11] = (int64_t)(int32_t)(-128);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-6);
        ctx->ctr = ctx->gpr[8];
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D2F060;
loc_00D2F000:
        ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[10], (int)(ctx->gpr[9] & 0x3F));
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-6);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 31, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[11];
        vm_write8(ctx->gpr[3] + 0x0, ctx->gpr[0]);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(1);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D2F000;
        ctx->gpr[6] = ctx->gpr[3] - ctx->gpr[5];
        goto loc_00D2EF5C;
loc_00D2F024:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 20, 12, 31, 0, 0);
        ctx->gpr[0] = (int64_t)(int32_t)(-32);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        ctx->gpr[8] = (int64_t)(int32_t)(2);
        vm_write8(ctx->gpr[5] + 0x0, ctx->gpr[9]);
        goto loc_00D2EFD8;
loc_00D2F040:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x51B0);
        ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x6);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
        ctx->gpr[10] = vm_read16(ctx->gpr[9] + 0x4);
        vm_write16(ctx->gpr[31] + 0x6, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[11]);
        vm_write16(ctx->gpr[31] + 0x4, ctx->gpr[10]);
        goto loc_00D2EEF0;
loc_00D2F060:
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        ctx->ctr = ctx->gpr[0];
        goto loc_00D2F000;
}

void june_evt_func_00D2F06C(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[9] = ctx->gpr[13] + (int64_t)(int32_t)((uint32_t)0x0 << 16);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[6] = ctx->gpr[9] + (int64_t)(-28364);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[9] = ctx->gpr[13] + (int64_t)(int32_t)((uint32_t)0x0 << 16);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 48);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D2F098;
        ctx->gpr[7] = ctx->gpr[9] + (int64_t)(-28200);
loc_00D2F098:
        ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D2ED90; return; }
}

void june_evt_func_00D38524(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[6] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[9] = vm_read16(ctx->gpr[6] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 31, 0, 0);
        ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
        vm_write16(ctx->gpr[6] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D385A8;
        ctx->gpr[7] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[11] = vm_read16(ctx->gpr[7] + 0x2); ctx->gpr[7] += 0x2;
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D386C4;
        ctx->gpr[8] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[10] = ctx->gpr[6] + (int64_t)(6);
        ctx->gpr[0] = vm_read16(ctx->gpr[8] + 0x4); ctx->gpr[8] += 0x4;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D386D0;
loc_00D38570:
        ctx->gpr[3] = (int64_t)(int32_t)(1);
        goto loc_00D3857C;
loc_00D38578:
        ctx->gpr[11] = vm_read16(ctx->gpr[7] + 0x0);
loc_00D3857C:
        vm_write16(ctx->gpr[6] + 0x0, ctx->gpr[11]);
        ctx->gpr[0] = vm_read16(ctx->gpr[8] + 0x0);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(-16);
        vm_write16(ctx->gpr[7] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read16(ctx->gpr[10] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        vm_write16(ctx->gpr[8] + 0x0, ctx->gpr[9]);
        vm_write16(ctx->gpr[10] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read16(ctx->gpr[6] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38578;
loc_00D385A8:
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 16, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D38624;
        ctx->gpr[7] = ctx->gpr[5] + (int64_t)(2);
        ctx->gpr[8] = ctx->gpr[5] + (int64_t)(4);
        ctx->gpr[10] = ctx->gpr[5] + (int64_t)(6);
loc_00D385C0:
        ctx->gpr[0] = vm_read16(ctx->gpr[7] + 0x0);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(-1);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 17, 15, 31, 0, 0);
        ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
        vm_write16(ctx->gpr[6] + 0x0, ctx->gpr[11]);
        ctx->gpr[9] = vm_read16(ctx->gpr[7] + 0x0);
        ctx->gpr[0] = vm_read16(ctx->gpr[8] + 0x0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 0, 30, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 17, 15, 31, 0, 0);
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        vm_write16(ctx->gpr[7] + 0x0, ctx->gpr[9]);
        ctx->gpr[11] = vm_read16(ctx->gpr[8] + 0x0);
        ctx->gpr[0] = vm_read16(ctx->gpr[10] + 0x0);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 1, 0, 30, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 17, 15, 31, 0, 0);
        ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
        vm_write16(ctx->gpr[8] + 0x0, ctx->gpr[11]);
        ctx->gpr[0] = vm_read16(ctx->gpr[10] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
        vm_write16(ctx->gpr[10] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read16(ctx->gpr[6] + 0x0);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 16, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0xF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D385C0;
loc_00D38624:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 16, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D386A0;
        ctx->gpr[8] = ctx->gpr[5] + (int64_t)(2);
        ctx->gpr[7] = ctx->gpr[5] + (int64_t)(6);
        ctx->gpr[10] = ctx->gpr[5] + (int64_t)(4);
loc_00D3863C:
        ctx->gpr[0] = vm_read16(ctx->gpr[7] + 0x0);
        ctx->gpr[3] = ctx->gpr[3] + (int64_t)(1);
        ctx->gpr[9] = vm_read16(ctx->gpr[10] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 1, 31, 0, 0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 15, 0, 16, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
        vm_write16(ctx->gpr[7] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = vm_read16(ctx->gpr[10] + 0x0);
        ctx->gpr[0] = vm_read16(ctx->gpr[8] + 0x0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 31, 1, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 15, 0, 16, 0, 0);
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        vm_write16(ctx->gpr[10] + 0x0, ctx->gpr[9]);
        ctx->gpr[11] = vm_read16(ctx->gpr[8] + 0x0);
        ctx->gpr[0] = vm_read16(ctx->gpr[6] + 0x0);
        ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 31, 1, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 15, 0, 16, 0, 0);
        ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
        vm_write16(ctx->gpr[8] + 0x0, ctx->gpr[11]);
        ctx->gpr[0] = vm_read16(ctx->gpr[6] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 1, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write16(ctx->gpr[6] + 0x0, ctx->gpr[0]);
        ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3863C;
loc_00D386A0:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 31, 0, 0);
        ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
        ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
        vm_write16(ctx->gpr[6] + 0x0, ctx->gpr[0]);
loc_00D386B0:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 0, 16, 0, 0);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
        vm_write16(ctx->gpr[6] + 0x0, ctx->gpr[0]);
        return;
loc_00D386C4:
        ctx->gpr[8] = ctx->gpr[6] + (int64_t)(4);
        ctx->gpr[10] = ctx->gpr[6] + (int64_t)(6);
        goto loc_00D38570;
loc_00D386D0:
        ctx->gpr[10] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[3] = (int64_t)(int32_t)(1);
        ctx->gpr[0] = vm_read16(ctx->gpr[10] + 0x6); ctx->gpr[10] += 0x6;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D38570;
        goto loc_00D386B0;
}

void june_evt_func_00D389F0(ppu_context* ctx) {
    PPU_MM_SNAP
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write64(ctx->gpr[1] + -0x10, tmp); }
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write64(ctx->gpr[1] + -0x8, tmp); }
        ctx->gpr[10] = (int64_t)(int32_t)(-1);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x8);
        ctx->gpr[9] = vm_read64(ctx->gpr[1] + -0x10);
        ctx->gpr[10] = ppc_rldicr(ctx->gpr[10], 0, 0);
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[0], 12, 53);
        ctx->gpr[0] = ppc_rldicr(ctx->gpr[9], 1, 62);
        vm_write64(ctx->gpr[8] + 0x8, ctx->gpr[10]);
        ctx->gpr[0] = ppc_rldicr(ctx->gpr[0], 63, 11);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D38A88;
        ctx->gpr[0] = (int64_t)(int32_t)(0x34);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 12);
        ctx->gpr[9] = (int64_t)(int32_t)(0xC);
        ctx->ctr = ctx->gpr[0];
        goto loc_00D38A40;
loc_00D38A38:
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) == 0)) goto loc_00D38A80;
loc_00D38A40:
        ctx->gpr[0] = (ctx->gpr[9] & 64) ? 0 : (ctx->gpr[10] >> (ctx->gpr[9] & 63));
        ctx->gpr[0] = ctx->gpr[0] & ctx->gpr[11];
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38A38;
        ctx->gpr[7] = ctx->gpr[9] + (int64_t)(-12);
loc_00D38A54:
        ctx->gpr[0] = (int64_t)(int32_t)(-1);
        ctx->gpr[9] = (ctx->gpr[9] & 64) ? 0 : (ctx->gpr[11] << (ctx->gpr[9] & 63));
        ctx->gpr[0] = ppc_rldicr(ctx->gpr[0], 0, 0);
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        vm_write64(ctx->gpr[8] + 0x8, ctx->gpr[9]);
loc_00D38A68:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[6], 0, 16, 31, 0, 0);
        ctx->gpr[3] = ctx->gpr[8] | ctx->gpr[8];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x3C00);
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[7];
        vm_write32(ctx->gpr[8] + 0x0, ctx->gpr[9]);
        return;
loc_00D38A80:
        ctx->gpr[7] = (int64_t)(int32_t)(0x34);
        goto loc_00D38A54;
loc_00D38A88:
        ctx->gpr[0] = ppc_rldic(ctx->gpr[9], 11, 1);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[10];
        vm_write64(ctx->gpr[8] + 0x8, ctx->gpr[0]);
        goto loc_00D38A68;
}

void june_evt_func_00D38A9C(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + 0x40, ctx->gpr[5]);
        vm_write64(ctx->gpr[1] + 0x50, ctx->gpr[7]);
        ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0x54);
        ctx->gpr[12] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0x50);
        ctx->gpr[8] = vm_read32(ctx->gpr[1] + 0x40);
        ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0x44);
        vm_write64(ctx->gpr[1] + 0x38, ctx->gpr[4]);
        ctx->gpr[0] = ctx->gpr[11] * ctx->gpr[10];
        ctx->gpr[11] = ctx->gpr[11] * ctx->gpr[7];
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 32, 32);
        vm_write64(ctx->gpr[1] + 0x48, ctx->gpr[6]);
        vm_write64(ctx->gpr[1] + -0x8, ctx->gpr[31]);
        ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x48);
        ctx->gpr[6] = (int64_t)(int32_t)(0);
        ctx->gpr[31] = vm_read32(ctx->gpr[1] + 0x38);
        ctx->gpr[4] = (int64_t)(int32_t)((uint32_t)0x8000 << 16);
        ctx->gpr[9] = ctx->gpr[8] * ctx->gpr[7];
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 32, 32);
        ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[5];
        ctx->gpr[8] = ctx->gpr[8] * ctx->gpr[10];
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 32, 32);
        ctx->gpr[7] = ctx->gpr[11] + ctx->gpr[10];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[7], 32, 32);
        ctx->gpr[8] = ctx->gpr[8] + ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[8];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[9], 32, 32);
        vm_write32(ctx->gpr[12] + 0xC, ctx->gpr[9]);
        ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
        vm_write32(ctx->gpr[12] + 0x8, ctx->gpr[0]);
        goto loc_00D38B34;
loc_00D38B2C:
        ctx->gpr[6] = ctx->gpr[6] + (int64_t)(1);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) == 0)) goto loc_00D38B44;
loc_00D38B34:
        ctx->gpr[0] = (uint64_t)((ctx->gpr[6] & 0x20) ? 0u : ((uint32_t)ctx->gpr[4] >> (ctx->gpr[6] & 0x1F)));
        ctx->gpr[11] = ctx->gpr[0] & ctx->gpr[9];
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00D38B2C;
loc_00D38B44:
        ctx->gpr[10] = vm_read32(ctx->gpr[12] + 0xC);
        { uint64_t _a = ctx->gpr[6]; uint64_t _b = (uint64_t)(int64_t)(0x20); ctx->gpr[8] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
        ctx->gpr[9] = (uint64_t)((ctx->gpr[6] & 0x20) ? 0u : ((uint32_t)ctx->gpr[9] << (ctx->gpr[6] & 0x1F)));
        ctx->gpr[0] = (uint64_t)((ctx->gpr[8] & 0x20) ? 0u : ((uint32_t)ctx->gpr[10] >> (ctx->gpr[8] & 0x1F)));
        ctx->gpr[10] = (uint64_t)((ctx->gpr[6] & 0x20) ? 0u : ((uint32_t)ctx->gpr[10] << (ctx->gpr[6] & 0x1F)));
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[7], 0, 32);
        { uint64_t _a = ctx->gpr[6]; uint64_t _b = (uint64_t)(int64_t)(0x1F); ctx->gpr[0] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
        vm_write32(ctx->gpr[12] + 0xC, ctx->gpr[10]);
        vm_write32(ctx->gpr[12] + 0x8, ctx->gpr[9]);
        ctx->gpr[9] = vm_read64(ctx->gpr[12] + 0x8);
        ctx->gpr[11] = (ctx->gpr[8] & 64) ? 0 : (ctx->gpr[11] >> (ctx->gpr[8] & 63));
        ctx->gpr[0] = (ctx->gpr[0] & 64) ? 0 : (ctx->gpr[7] >> (ctx->gpr[0] & 63));
        ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[9];
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 63);
        ctx->gpr[10] = ctx->gpr[31] + ctx->gpr[3];
        ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
        ctx->gpr[10] = ctx->gpr[10] - ctx->gpr[6];
        { uint64_t a = (uint64_t)ctx->gpr[11]; uint64_t b = (uint64_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[12] + 0x8, ctx->gpr[0]);
        ctx->gpr[9] = (int64_t)(int32_t)(1);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D38BA0;
        ctx->gpr[9] = (int64_t)(int32_t)(0);
loc_00D38BA0:
        ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[9];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[3] = ctx->gpr[12] | ctx->gpr[12];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-16382);
        vm_write32(ctx->gpr[12] + 0x0, ctx->gpr[9]);
        return;
}

void june_evt_func_00D38BB8(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + 0x50, ctx->gpr[7]);
        vm_write64(ctx->gpr[1] + 0x40, ctx->gpr[5]);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x44);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x54);
        ctx->gpr[8] = vm_read32(ctx->gpr[1] + 0x40);
        ctx->gpr[11] = ctx->gpr[9] - ctx->gpr[0];
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x50);
        vm_write64(ctx->gpr[1] + 0x38, ctx->gpr[4]);
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0x38);
        vm_write64(ctx->gpr[1] + 0x48, ctx->gpr[6]);
        ctx->gpr[10] = ctx->gpr[8] - ctx->gpr[0];
        vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[11]);
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D38CD8;
        { uint64_t a = (uint32_t)ctx->gpr[8]; uint64_t b = (uint32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[11] = (int64_t)(int32_t)(1);
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D38CEC;
loc_00D38C00:
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(-1);
        ctx->gpr[0] = ~(ctx->gpr[10] | ctx->gpr[10]);
        ctx->gpr[6] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[6] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[6];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 1, 31, 31, 0, 0);
        ctx->gpr[0] = ctx->gpr[11] | ctx->gpr[9];
loc_00D38C20:
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[3] + 0x8, ctx->gpr[10]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38CD0;
        ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0xC);
        ctx->gpr[0] = ~(ctx->gpr[10] | ctx->gpr[10]);
        ctx->gpr[11] = (uint64_t)0 - ctx->gpr[11];
        vm_write32(ctx->gpr[3] + 0x8, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[11]);
loc_00D38C40:
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x8);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x8000 << 16);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D38C90;
        ctx->gpr[10] = (int64_t)(int32_t)((uint32_t)0x8000 << 16);
loc_00D38C60:
        ctx->gpr[0] = (uint64_t)((ctx->gpr[9] & 0x20) ? 0u : ((uint32_t)ctx->gpr[10] >> (ctx->gpr[9] & 0x1F)));
        ctx->gpr[6] = ctx->gpr[0] & ctx->gpr[11];
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00D38C78;
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D38C60;
loc_00D38C78:
        ctx->gpr[9] = ctx->gpr[7] - ctx->gpr[9];
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = ctx->gpr[9] + (int64_t)(-32);
        vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[11]);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
        return;
loc_00D38C90:
        ctx->gpr[0] = (uint64_t)((ctx->gpr[10] & 0x20) ? 0u : ((uint32_t)ctx->gpr[8] >> (ctx->gpr[10] & 0x1F)));
        ctx->gpr[6] = ctx->gpr[0] & ctx->gpr[9];
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00D38CA8;
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D38C90;
loc_00D38CA8:
        { uint64_t _a = ctx->gpr[10]; uint64_t _b = (uint64_t)(int64_t)(0x20); ctx->gpr[0] = _b - _a; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_b >= _a) ? (1u << 29) : 0u); }
        ctx->gpr[9] = (uint64_t)((ctx->gpr[10] & 0x20) ? 0u : ((uint32_t)ctx->gpr[9] << (ctx->gpr[10] & 0x1F)));
        ctx->gpr[0] = (uint64_t)((ctx->gpr[0] & 0x20) ? 0u : ((uint32_t)ctx->gpr[11] >> (ctx->gpr[0] & 0x1F)));
        ctx->gpr[11] = (uint64_t)((ctx->gpr[10] & 0x20) ? 0u : ((uint32_t)ctx->gpr[11] << (ctx->gpr[10] & 0x1F)));
        ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[7] - ctx->gpr[10];
        vm_write32(ctx->gpr[3] + 0x8, ctx->gpr[9]);
        vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[11]);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
        return;
loc_00D38CD0:
        ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0xC);
        goto loc_00D38C40;
loc_00D38CD8:
        { uint64_t a = (uint32_t)ctx->gpr[8]; uint64_t b = (uint32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D38C20;
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        goto loc_00D38C20;
loc_00D38CEC:
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        goto loc_00D38C00;
}

void june_evt_func_00D38D90(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[9] = (int64_t)(int32_t)(-1);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
        ctx->gpr[9] = ppc_rldicr(ctx->gpr[9], 0, 0);
        ctx->gpr[0] = (int64_t)(int32_t)(0x3FFF);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
        ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
        vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[4]);
        vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[5]);
        vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[9]);
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D38EDC;
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38E68;
        ctx->gpr[30] = (int64_t)(int32_t)(0);
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + 0x52E8);
        ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x70);
        goto loc_00D38E0C;
loc_00D38DF4:
        ctx->gpr[31] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[31], 1);
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
        if ((((ctx->cr >> 30) & 1) == 0)) goto loc_00D38E50;
loc_00D38E08:
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38E94;
loc_00D38E0C:
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 0, 31, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[30], 4, 0, 27, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[9] = ctx->gpr[0] + ctx->gpr[29];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38DF4;
        ctx->gpr[7] = vm_read64(ctx->gpr[9] + 0xA8);
        ctx->gpr[31] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[31], 1);
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[6] = vm_read64(ctx->gpr[9] + 0xA0);
        ctx->lr = 0x00D38E40; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
        if ((((ctx->cr >> 30) & 1) == 1)) goto loc_00D38E08;
loc_00D38E50:
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0xE8);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xF0);
        ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->lr = 0x00D38E68; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D38E68:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
        return;
loc_00D38E94:
        ctx->gpr[30] = vm_read32(ctx->gpr[2] + 0x52E8);
        ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
loc_00D38E9C:
        ctx->gpr[31] = ctx->gpr[31] + (int64_t)(-1);
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[6] = vm_read64(ctx->gpr[30] + 0x130);
        ctx->gpr[7] = vm_read64(ctx->gpr[30] + 0x138);
        ctx->lr = 0x00D38EB8; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D38E9C;
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0xE8);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xF0);
        ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->lr = 0x00D38ED8; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D38E68;
loc_00D38EDC:
        ctx->gpr[31] = (uint64_t)0 - ctx->gpr[6];
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38E50;
        ctx->gpr[29] = (int64_t)(int32_t)(0);
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + 0x52E8);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 0, 31, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 4, 0, 27, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[31] = ppc_rlw64(ctx->gpr[31], 31, 1, 31, 0, 0);
        ctx->gpr[11] = ctx->gpr[0] + ctx->gpr[28];
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D38F48;
loc_00D38F14:
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38E50;
        if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00D38F60;
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 0, 31, 31, 0, 0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 4, 0, 27, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[31] = ppc_rlw64(ctx->gpr[31], 31, 1, 31, 0, 0);
        ctx->gpr[11] = ctx->gpr[0] + ctx->gpr[28];
        ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D38F14;
loc_00D38F48:
        ctx->gpr[7] = vm_read64(ctx->gpr[11] + 0x8);
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[6] = vm_read64((ctx->gpr[28] + ctx->gpr[0]));
        ctx->lr = 0x00D38F5C; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D38F14;
loc_00D38F60:
        ctx->gpr[30] = vm_read32(ctx->gpr[2] + 0x52E8);
        ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
loc_00D38F68:
        ctx->gpr[31] = ctx->gpr[31] + (int64_t)(-1);
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->gpr[6] = vm_read64(ctx->gpr[30] + 0x90);
        ctx->gpr[7] = vm_read64(ctx->gpr[30] + 0x98);
        ctx->lr = 0x00D38F84; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D38F68;
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[4] = vm_read64(ctx->gpr[1] + 0xE8);
        ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xF0);
        ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0x70);
        ctx->gpr[7] = vm_read64(ctx->gpr[1] + 0x78);
        ctx->lr = 0x00D38FA4; june_evt_func_00D38A9C(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D38E68;
}

void june_evt_func_00D38FA8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xD0);
        ctx->gpr[11] = ppc_rldicr(ctx->gpr[4], 1, 62);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-15360);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
        ctx->gpr[0] = ~(ctx->gpr[9] | ctx->gpr[9]);
        { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x7FE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
        ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[31] = vm_read64(ctx->gpr[2] + 0x52F0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 21, 31, 0, 0);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[4]);
        ctx->gpr[11] = ppc_rldicr(ctx->gpr[11], 63, 53);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3904C;
        ctx->gpr[31] = ppc_rldimi(ctx->gpr[31], ctx->gpr[0], 52, 1);
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[31] = ppc_rldimi(ctx->gpr[31], ctx->gpr[0], 0, 12);
        ctx->lr = 0x00D39004; june_evt_func_00D23CA8(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x8001 << 16);
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->gpr[0] | 0x1C;
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D39028;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        ctx->gpr[0] = ctx->gpr[0] | 0x1;
        vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
loc_00D39028:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[31]);
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[1], &tmp, 8); }
        ctx->lr = ctx->gpr[0];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
        return;
loc_00D3904C:
        ctx->gpr[31] = ppc_rldimi(ctx->gpr[31], ctx->gpr[9], 52, 1);
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 53, 11);
        ctx->gpr[0] = ppc_rldicr(ctx->gpr[31], 1, 62);
        ctx->gpr[9] = (int64_t)(int32_t)(0xA);
        ctx->gpr[0] = ppc_rldicr(ctx->gpr[0], 63, 11);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D390C8;
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 52, 12);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xDC);
        ctx->gpr[9] = (int64_t)(int32_t)(0xB);
loc_00D39074:
        ctx->gpr[0] = (uint64_t)((ctx->gpr[9] & 0x20) ? 0u : ((uint32_t)ctx->gpr[0] >> (ctx->gpr[9] & 0x1F)));
        ctx->gpr[11] = (int64_t)(int32_t)(1);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 63);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[10];
        ctx->gpr[31] = ppc_rldimi(ctx->gpr[31], ctx->gpr[0], 0, 12);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 12);
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[31], 12, 53);
        { uint64_t a = (uint64_t)ctx->gpr[9]; uint64_t b = (uint64_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3909C;
        ctx->gpr[11] = (int64_t)(int32_t)(0);
loc_00D3909C:
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = ppc_rldimi(ctx->gpr[31], ctx->gpr[0], 52, 1);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[31]);
        ctx->lr = ctx->gpr[0];
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x70); memcpy(&ctx->fpr[1], &tmp, 8); }
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
        return;
loc_00D390C8:
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xDC);
        goto loc_00D39074;
}

void june_evt_func_00D3C948(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->lr = 0x00D3C958; june_evt_func_00D29898(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        return;
}

void june_evt_func_00D3D3DC(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->lr = 0x00D3D3EC; june_evt_func_00D29AE4(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[3] = (int64_t)(int16_t)ctx->gpr[3];
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        return;
}

void june_evt_func_00D3DCB0(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        vm_write64(ctx->gpr[1] + -0x18, ctx->gpr[29]);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(-48);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x30);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        vm_write8(ctx->gpr[29] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_00D3DCEC(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        vm_write64(ctx->gpr[1] + -0x18, ctx->gpr[29]);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(-48);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x30);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_00D3E2C8(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { int32_t _a=(int32_t)ctx->gpr[4], _b=(int32_t)ctx->gpr[5]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        ctx->gpr[5] = (uint64_t)((int64_t)(int32_t)ctx->gpr[5] * (int64_t)(int32_t)ctx->gpr[0]);
        ctx->gpr[4] = ctx->gpr[4] - ctx->gpr[5];
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[4]);
        return;
}

void june_evt_func_00D3E35C(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_24 = ctx->gpr[24];
        uint64_t _cs_25 = ctx->gpr[25];
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + 0x5398);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[24]);
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x2C);
        ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[25] = vm_read8(ctx->gpr[11] + 0x0);
        ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[29] = ctx->gpr[7] | ctx->gpr[7];
        ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E7B8;
loc_00D3E3B0:
        ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x66; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E5FC;
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x46; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E5FC;
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x67; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E5A0;
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x47; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[29];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E5A0;
loc_00D3E3D8:
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x61; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E8E0;
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x41; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E828;
loc_00D3E3E8:
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[10] = vm_read8(ctx->gpr[26] + 0x0);
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[10]);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x44);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E8E8;
loc_00D3E414:
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[25]);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x44);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E480;
        ctx->gpr[5] = ctx->gpr[27] + (int64_t)(-1);
        ctx->gpr[28] = (int64_t)(int16_t)ctx->gpr[5];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3E7C4;
loc_00D3E448:
        ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[4] = ctx->gpr[26] + (int64_t)(1);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[3] = ctx->gpr[11] + ctx->gpr[3];
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00D3E464; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x44);
        ctx->gpr[11] = ctx->gpr[28] + ctx->gpr[11];
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[28];
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        vm_write32(ctx->gpr[30] + 0x38, ctx->gpr[0]);
loc_00D3E480:
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[3] = ctx->gpr[9] + (int64_t)(1);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[24]);
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3E924;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[25] = ctx->gpr[3] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_00D3E4AC:
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[28] = (int64_t)(int32_t)(0);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E8FC;
        ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x78);
loc_00D3E4C0:
        ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[31];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[5] = (int64_t)(int32_t)(0xA);
        ctx->lr = 0x00D3E4D0; june_evt_func_00D3F960(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = ctx->gpr[28] + ctx->gpr[27];
        ctx->gpr[29] = vm_read16(ctx->gpr[1] + 0x72);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x74);
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[29];
        ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3E4C0;
        { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E8FC;
loc_00D3E500:
        { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E6FC;
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E548;
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[7] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[8] = ctx->gpr[25] | ctx->gpr[25];
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
loc_00D3E520:
        ctx->gpr[7] = ctx->gpr[7] + (int64_t)(-1);
        ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
        ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[7];
        ctx->gpr[8] = ctx->gpr[8] + (int64_t)(1);
        ctx->gpr[11] = ctx->gpr[26] + ctx->gpr[11];
        ctx->gpr[9] = vm_read8(ctx->gpr[11] + 0x8);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x30);
        vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[9]);
        if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00D3E520;
        ctx->gpr[25] = ctx->gpr[25] + ctx->gpr[28];
loc_00D3E548:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[8] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        ctx->gpr[0] = ctx->gpr[25] - ctx->gpr[0];
        vm_write32(ctx->gpr[30] + 0x3C, ctx->gpr[0]);
loc_00D3E560:
        ctx->gpr[9] = (int64_t)(int32_t)(0x14);
        ctx->gpr[0] = ctx->gpr[8] & ctx->gpr[9];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E694;
loc_00D3E570:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[24] = _cs_24;
        ctx->gpr[25] = _cs_25;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
loc_00D3E5A0:
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[29];
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)-4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E7CC;
loc_00D3E5AC:
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x67; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E5BC;
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0x47; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3E3D8;
loc_00D3E5BC:
        ctx->gpr[11] = (int64_t)(int16_t)ctx->gpr[27];
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3E940;
loc_00D3E5CC:
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(-1);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[10]);
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3E99C;
loc_00D3E5DC:
        ctx->gpr[0] = ctx->gpr[9] ^ 0x67;
        ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
        ctx->gpr[9] = ctx->gpr[11] ^ ctx->gpr[0];
        ctx->gpr[9] = ctx->gpr[11] - ctx->gpr[9];
        ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 0x1F);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 0, 26, 0, 0);
        ctx->gpr[24] = ctx->gpr[9] + (int64_t)(0x65);
        goto loc_00D3E3E8;
loc_00D3E5FC:
        ctx->gpr[28] = ctx->gpr[29] + (int64_t)(1);
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[27];
        ctx->gpr[29] = (int64_t)(int16_t)ctx->gpr[28];
        ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
loc_00D3E60C:
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E830;
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E710;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[0];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00D3E638; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x34);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->gpr[29] - ctx->gpr[31];
        ctx->gpr[11] = ctx->gpr[9] + ctx->gpr[31];
        vm_write32(ctx->gpr[30] + 0x38, ctx->gpr[0]);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E970;
loc_00D3E65C:
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[25]);
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x3C);
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[8] = vm_read16(ctx->gpr[30] + 0x50);
        vm_write32(ctx->gpr[30] + 0x40, ctx->gpr[10]);
        vm_write32(ctx->gpr[30] + 0x3C, ctx->gpr[11]);
loc_00D3E684:
        ctx->gpr[9] = (int64_t)(int32_t)(0x14);
        ctx->gpr[0] = ctx->gpr[8] & ctx->gpr[9];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x10; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3E570;
loc_00D3E694:
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x2C);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x38);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x3C);
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x40);
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[10];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x4C);
        ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E570;
        ctx->gpr[0] = ctx->gpr[11] - ctx->gpr[0];
        ctx->gpr[24] = _cs_24;
        ctx->gpr[25] = _cs_25;
        vm_write32(ctx->gpr[30] + 0x30, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
loc_00D3E6FC:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        goto loc_00D3E548;
loc_00D3E710:
        ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[0];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00D3E72C; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x34);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[11];
        ctx->gpr[5] = ctx->gpr[27] - ctx->gpr[28];
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E95C;
loc_00D3E74C:
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[25]);
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
loc_00D3E768:
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[5];
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E778;
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[10];
loc_00D3E778:
        ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[4] = ctx->gpr[26] + ctx->gpr[29];
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[11];
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00D3E794; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x44);
        ctx->gpr[11] = ctx->gpr[9] + ctx->gpr[31];
        ctx->gpr[8] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[31];
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        vm_write32(ctx->gpr[30] + 0x38, ctx->gpr[0]);
        goto loc_00D3E560;
loc_00D3E7B8:
        ctx->gpr[26] = vm_read32(ctx->gpr[2] + 0x539C);
        ctx->gpr[27] = (int64_t)(int32_t)(1);
        goto loc_00D3E3B0;
loc_00D3E7C4:
        ctx->gpr[28] = (int64_t)(int16_t)ctx->gpr[0];
        goto loc_00D3E448;
loc_00D3E7CC:
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E5AC;
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[28] = ctx->gpr[29] + (int64_t)(1);
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[27];
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3E804;
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[27];
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E804;
        ctx->gpr[10] = ctx->gpr[31] | ctx->gpr[31];
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[31]);
loc_00D3E804:
        ctx->gpr[29] = (int64_t)(int16_t)ctx->gpr[28];
        ctx->gpr[10] = ctx->gpr[10] - ctx->gpr[29];
        ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[10]);
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E60C;
        ctx->gpr[10] = (int64_t)(int32_t)(0);
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[10]);
        goto loc_00D3E60C;
loc_00D3E828:
        ctx->gpr[24] = (int64_t)(int32_t)(0x50);
        goto loc_00D3E3E8;
loc_00D3E830:
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3E988;
loc_00D3E85C:
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[25]);
        ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x44);
        vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[11]);
loc_00D3E878:
        ctx->gpr[0] = (uint64_t)0 - ctx->gpr[29];
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E890;
        ctx->gpr[0] = (uint64_t)0 - ctx->gpr[10];
        ctx->gpr[8] = (int64_t)(int16_t)ctx->gpr[0];
        ctx->gpr[0] = (uint64_t)0 - ctx->gpr[8];
loc_00D3E890:
        ctx->gpr[10] = ctx->gpr[8] + ctx->gpr[10];
        vm_write32(ctx->gpr[30] + 0x38, ctx->gpr[0]);
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[10]);
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3E8A8;
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[10];
loc_00D3E8A8:
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[30] + 0x3C, ctx->gpr[31]);
        ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[0];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00D3E8C8; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x44);
        ctx->gpr[8] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[31];
        vm_write32(ctx->gpr[30] + 0x40, ctx->gpr[0]);
        goto loc_00D3E560;
loc_00D3E8E0:
        ctx->gpr[24] = (int64_t)(int32_t)(0x70);
        goto loc_00D3E3E8;
loc_00D3E8E8:
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E480;
        goto loc_00D3E414;
loc_00D3E8FC:
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[24];
        { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0x65; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E910;
        { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0x45; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3E500;
loc_00D3E910:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        goto loc_00D3E500;
loc_00D3E924:
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[29] = (uint64_t)0 - ctx->gpr[29];
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[25] = ctx->gpr[3] + (int64_t)(1);
        ctx->gpr[31] = (int64_t)(int16_t)ctx->gpr[29];
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        goto loc_00D3E4AC;
loc_00D3E940:
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3E5CC;
        ctx->gpr[10] = ctx->gpr[11] | ctx->gpr[11];
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[11]);
        goto loc_00D3E5CC;
loc_00D3E95C:
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E768;
        goto loc_00D3E74C;
loc_00D3E970:
        ctx->gpr[8] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3E65C;
        vm_write32(ctx->gpr[30] + 0x40, ctx->gpr[10]);
        goto loc_00D3E684;
loc_00D3E988:
        ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3E878;
        goto loc_00D3E85C;
loc_00D3E99C:
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        vm_write32(ctx->gpr[30] + 0x44, ctx->gpr[0]);
        goto loc_00D3E5DC;
}

void june_evt_func_00D3E9A8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_25 = ctx->gpr[25];
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0x30, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + -0x20, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        vm_write64(ctx->gpr[1] + -0x38, ctx->gpr[25]);
        ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + -0x28, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + -0x18, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + -0x10, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + -0x8, ctx->gpr[31]);
        ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EA38;
        ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[6], 0, 32);
        ctx->gpr[25] = ctx->gpr[1] + (int64_t)(-80);
        ctx->gpr[30] = (int64_t)(int32_t)(0);
        goto loc_00D3E9FC;
loc_00D3E9F8:
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EA38;
loc_00D3E9FC:
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
        ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        ctx->gpr[29] = ctx->gpr[30] + ctx->gpr[26];
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x50);
        { uint64_t _a = ctx->gpr[31]; uint64_t _r = _a + (uint64_t)(int64_t)(-1); ctx->gpr[31] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); } { int64_t _r = (int64_t)ctx->gpr[31]; uint32_t _c = (_r < 0) ? 8u : (_r > 0) ? 4u : 2u; _c |= (uint32_t)((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & 0x0FFFFFFFu) | (_c << 28); }
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write8(ctx->gpr[29] + 0x0, ctx->gpr[0]);
        if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00D3E9F8;
loc_00D3EA38:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[25] = _cs_25;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        return;
}

void june_evt_func_00D3EA64(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_26 = ctx->gpr[26];
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4C);
        ctx->gpr[5] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0x30);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[5];
        ctx->gpr[10] = vm_read32(ctx->gpr[3] + 0x34);
        ctx->gpr[9] = vm_read16(ctx->gpr[3] + 0x50);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[11];
        ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0x38);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
        ctx->gpr[10] = vm_read32(ctx->gpr[3] + 0x3C);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x40);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[11];
        vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[27]);
        ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
        vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[30]);
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[26] = ctx->gpr[0] - ctx->gpr[9];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EADC;
        { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3ECAC;
loc_00D3EADC:
        { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3EEC4;
loc_00D3EAE4:
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3EC18;
loc_00D3EAF0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EB78;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x34);
        { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3EF3C;
loc_00D3EB0C:
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x38);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3EE58;
loc_00D3EB18:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x3C);
        { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3EF84;
loc_00D3EB24:
        ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x40);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3EDEC;
loc_00D3EB30:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EB48;
        { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3ED18;
loc_00D3EB48:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[26] = _cs_26;
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[27] = _cs_27;
        ctx->gpr[28] = _cs_28;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
        return;
loc_00D3EB78:
        ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
        ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x34);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EF0C;
        { uint64_t a = (uint32_t)ctx->gpr[30]; uint64_t b = (uint64_t)0x80; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3ED80;
        ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[27], 0, 32);
        goto loc_00D3EBE8;
loc_00D3EBA8:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EBC4; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { uint64_t a = (uint32_t)ctx->gpr[30]; uint64_t b = (uint64_t)0x80; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x80);
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[9]);
        if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D3ED84;
loc_00D3EBE8:
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x54);
        ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[6] = (int64_t)(int32_t)(0x80);
        ctx->gpr[30] = ctx->gpr[30] + (int64_t)(-128);
        ctx->lr = 0x00D3EC00; june_evt_func_00D3E9A8(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x80);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
        ctx->gpr[5] = (int64_t)(int32_t)(0x80);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EBA8;
        goto loc_00D3EB48;
loc_00D3EC18:
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + 0x53A8);
        goto loc_00D3EC30;
loc_00D3EC20:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D3EAF0;
loc_00D3EC30:
        { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x20; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3EC44;
        ctx->gpr[30] = (int64_t)(int32_t)(0x20);
loc_00D3EC44:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[29] = ctx->gpr[29] - ctx->gpr[30];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EC68; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EC20;
loc_00D3EC7C:
        ctx->gpr[3] = (int64_t)(int32_t)(-1);
loc_00D3EC80:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->gpr[26] = _cs_26;
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
        return;
loc_00D3ECAC:
        ctx->gpr[29] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + 0x53A0);
loc_00D3ECB4:
        { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x20; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3ECC8;
        ctx->gpr[30] = (int64_t)(int32_t)(0x20);
loc_00D3ECC8:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[29] = ctx->gpr[29] - ctx->gpr[30];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3ECEC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        if ((((ctx->cr >> 6) & 1) == 1)) goto loc_00D3ECB4;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x2C);
        goto loc_00D3EADC;
loc_00D3ED18:
        ctx->gpr[29] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + 0x53A0);
loc_00D3ED20:
        { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x20; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3ED34;
        ctx->gpr[30] = (int64_t)(int32_t)(0x20);
loc_00D3ED34:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[29] = ctx->gpr[29] - ctx->gpr[30];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3ED58; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        if ((((ctx->cr >> 6) & 1) == 1)) goto loc_00D3ED20;
        goto loc_00D3EB48;
loc_00D3ED80:
        ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x70);
loc_00D3ED84:
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x54);
        ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->lr = 0x00D3ED98; june_evt_func_00D3E9A8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EB48;
loc_00D3EDA0:
        { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3EB24;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EDCC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        goto loc_00D3EB24;
loc_00D3EDEC:
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + 0x53A8);
        goto loc_00D3EE04;
loc_00D3EDF4:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D3EB30;
loc_00D3EE04:
        { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x20; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3EE18;
        ctx->gpr[30] = (int64_t)(int32_t)(0x20);
loc_00D3EE18:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[29] = ctx->gpr[29] - ctx->gpr[30];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EE3C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EDF4;
        ctx->gpr[3] = (int64_t)(int32_t)(-1);
        goto loc_00D3EC80;
loc_00D3EE58:
        ctx->gpr[28] = vm_read32(ctx->gpr[2] + 0x53A8);
        goto loc_00D3EE70;
loc_00D3EE60:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        if ((((ctx->cr >> 6) & 1) == 0)) goto loc_00D3EB18;
loc_00D3EE70:
        { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x20; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
        ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D3EE84;
        ctx->gpr[30] = (int64_t)(int32_t)(0x20);
loc_00D3EE84:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[29] = ctx->gpr[29] - ctx->gpr[30];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EEA8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3EE60;
        ctx->gpr[3] = (int64_t)(int32_t)(-1);
        goto loc_00D3EC80;
loc_00D3EEC4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EEE8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        goto loc_00D3EAE4;
loc_00D3EF0C:
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + 0x53A4);
        ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[5] = (int64_t)(int32_t)(7);
        ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
        ctx->lr = 0x00D3EF24; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x00D3EF30; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        goto loc_00D3EDA0;
loc_00D3EF3C:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EF60; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        goto loc_00D3EB0C;
loc_00D3EF84:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
        ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
        ctx->gpr[4] = ctx->gpr[4] + ctx->gpr[0];
        ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
        ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->lr = 0x00D3EFB4; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3EC7C;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x3C);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
        goto loc_00D3EB24;
}

void june_evt_func_00D3EFD8(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        vm_write64(ctx->gpr[1] + -0x18, ctx->gpr[29]);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(-48);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x30);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_00D3F014(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        vm_write64(ctx->gpr[1] + -0x18, ctx->gpr[29]);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(-48);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + -0x30);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_00D3F050(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        lv2_syscall(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_00D3F074(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
        lv2_syscall(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_00D3F098(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = ctx->gpr[27];
        uint64_t _cs_28 = ctx->gpr[28];
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[12] = ctx->cr;
        vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
        ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
        ctx->gpr[9] = ctx->gpr[30] + (int64_t)(-37);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
        ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[29] = ctx->gpr[7] | ctx->gpr[7];
        ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
        vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[12]);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x53; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F250;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53AC);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F250u: goto loc_00D3F250;case 0x00D3F2ACu: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F2AC; return; }case 0x00D3F2B4u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F2B4; return; }case 0x00D3F394u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F394; return; }case 0x00D3F43Cu: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F43C; return; }case 0x00D3F4C0u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4C0; return; }case 0x00D3F4F8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4F8; return; }case 0x00D3F538u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F538; return; }case 0x00D3F5E8u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F5E8; return; }case 0x00D3F6B0u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F6B0; return; } default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
loc_00D3F250:
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x2C);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F2A4;
loc_00D3F260:
        ctx->gpr[9] = ctx->gpr[11] + ctx->gpr[29];
        ctx->gpr[0] = ctx->gpr[11] + (int64_t)(1);
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[0]);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xD8);
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
loc_00D3F2A4:
        ctx->gpr[30] = (int64_t)(int32_t)(0x25);
        goto loc_00D3F260;
}

void june_evt_func_00D3F274(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xB8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xC0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xC8);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xD8);
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[28] = _cs_28;
        ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
        ctx->gpr[29] = _cs_29;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
}

void june_evt_func_00D3F2AC(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xB8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xC0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xC8);
        ctx->gpr[0] = (int64_t)(int32_t)(0x6C);
        vm_write8(ctx->gpr[3] + 0x52, ctx->gpr[0]);
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        vm_write32(ctx->gpr[31] + 0x54, ctx->gpr[4]);
        ctx->gpr[0] = ctx->gpr[0] | 0x20;
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
        vm_write16(ctx->gpr[31] + 0x50, ctx->gpr[0]);
        ctx->lr = 0x00D3F2D8; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x74);
        ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x80);
        vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[3]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        goto loc_00D3F300;
loc_00D3F2F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[9]);
loc_00D3F300:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3F318;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3F358;
loc_00D3F318:
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[0] = ctx->gpr[4] + (int64_t)(1);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write8(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3F2F4;
loc_00D3F358:
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xD8);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[9]);
        ctx->gpr[29] = _cs_29;
        ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
}

void june_evt_func_00D3F2B4(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xB8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xC0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xC8);
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        vm_write32(ctx->gpr[31] + 0x54, ctx->gpr[4]);
        ctx->gpr[0] = ctx->gpr[0] | 0x20;
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
        vm_write16(ctx->gpr[31] + 0x50, ctx->gpr[0]);
        ctx->lr = 0x00D3F2D8; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x74);
        ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x80);
        vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[3]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        goto loc_00D3F300;
loc_00D3F2F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[9]);
loc_00D3F300:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x44);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3F318;
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D3F358;
loc_00D3F318:
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[0] = ctx->gpr[4] + (int64_t)(1);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write8(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D3F2F4;
loc_00D3F358:
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xD8);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[9]);
        ctx->gpr[29] = _cs_29;
        ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
}

void june_evt_func_00D3F358(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_27 = vm_read64(ctx->gpr[1] + 0xA8);
        uint64_t _cs_28 = vm_read64(ctx->gpr[1] + 0xB0);
        uint64_t _cs_29 = vm_read64(ctx->gpr[1] + 0xB8);
        uint64_t _cs_30 = vm_read64(ctx->gpr[1] + 0xC0);
        uint64_t _cs_31 = vm_read64(ctx->gpr[1] + 0xC8);
        ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xD8);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[27] = _cs_27;
        ctx->lr = ctx->gpr[0];
        vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[9]);
        ctx->gpr[29] = _cs_29;
        ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
        ctx->gpr[28] = _cs_28;
        ctx->gpr[30] = _cs_30;
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
        return;
}

void june_evt_func_00D3F394(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F6FC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F808;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F808;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F3C4; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
loc_00D3F3D0:
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
loc_00D3F3D4:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F890;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F914;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8E4;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8E4;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
loc_00D3F404:
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3F8F4;
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F864;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F6D0;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F470:
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 16, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F834;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_00D3F4A0:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: goto loc_00D3F748;case 0x00D3F760u: goto loc_00D3F760;case 0x00D3F780u: goto loc_00D3F780;case 0x00D3F7A0u: goto loc_00D3F7A0;case 0x00D3F7C0u: goto loc_00D3F7C0; default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F6D0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F470;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F6FC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F3D0;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F748:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
loc_00D3F760:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F780:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7A0:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7C0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F808:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F3D4;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F834:
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F864:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
loc_00D3F890:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F4A0;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F8D8:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8E4:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
loc_00D3F914:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
}

void june_evt_func_00D3F3D0(ppu_context* ctx) {
    PPU_MM_SNAP
loc_00D3F3D0:
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
loc_00D3F3D4:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F890;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F914;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8E4;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8E4;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
loc_00D3F404:
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3F8F4;
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F864;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F6D0;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F470:
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 16, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F834;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_00D3F4A0:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: goto loc_00D3F748;case 0x00D3F760u: goto loc_00D3F760;case 0x00D3F780u: goto loc_00D3F780;case 0x00D3F7A0u: goto loc_00D3F7A0;case 0x00D3F7C0u: goto loc_00D3F7C0; default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F6D0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F470;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F3D0;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F748:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
loc_00D3F760:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F780:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7A0:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7C0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F3D4;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F834:
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F864:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
loc_00D3F890:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F4A0;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F8D8:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8E4:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
loc_00D3F914:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
}

void june_evt_func_00D3F3D4(ppu_context* ctx) {
    PPU_MM_SNAP
loc_00D3F3D4:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F890;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F914;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8E4;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8E4;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
loc_00D3F404:
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3F8F4;
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F864;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F6D0;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F470:
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 16, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F834;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_00D3F4A0:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: goto loc_00D3F748;case 0x00D3F760u: goto loc_00D3F760;case 0x00D3F780u: goto loc_00D3F780;case 0x00D3F7A0u: goto loc_00D3F7A0;case 0x00D3F7C0u: goto loc_00D3F7C0; default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F6D0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F470;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F748:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
loc_00D3F760:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F780:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7A0:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7C0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F3D4;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F834:
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F864:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
loc_00D3F890:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F4A0;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F8D8:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8E4:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
loc_00D3F914:
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
}

void june_evt_func_00D3F404(ppu_context* ctx) {
    PPU_MM_SNAP
loc_00D3F404:
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00D3F8F4;
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F864;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F6D0;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F470:
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 16, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F834;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_00D3F4A0:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: goto loc_00D3F748;case 0x00D3F760u: goto loc_00D3F760;case 0x00D3F780u: goto loc_00D3F780;case 0x00D3F7A0u: goto loc_00D3F7A0;case 0x00D3F7C0u: goto loc_00D3F7C0; default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F6D0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F470;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F748:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
loc_00D3F760:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F780:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7A0:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7C0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F834:
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F864:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F4A0;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
loc_00D3F8D8:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F404;
loc_00D3F8F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
}

void june_evt_func_00D3F43C(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x4C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F6D0;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x90);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F470:
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 16, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F834;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_00D3F4A0:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: goto loc_00D3F748;case 0x00D3F760u: goto loc_00D3F760;case 0x00D3F780u: goto loc_00D3F780;case 0x00D3F7A0u: goto loc_00D3F7A0;case 0x00D3F7C0u: goto loc_00D3F7C0; default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8D8; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8D8; return; }
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F6D0:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F470;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F748:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
loc_00D3F760:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F780:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7A0:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7C0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F834:
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F4A0;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
}

void june_evt_func_00D3F470(ppu_context* ctx) {
    PPU_MM_SNAP
loc_00D3F470:
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[31] + 0x0, tmp); }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 16, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F834;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_00D3F4A0:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: goto loc_00D3F748;case 0x00D3F760u: goto loc_00D3F760;case 0x00D3F780u: goto loc_00D3F780;case 0x00D3F7A0u: goto loc_00D3F7A0;case 0x00D3F7C0u: goto loc_00D3F7C0; default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8D8; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8D8; return; }
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F470;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F748:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
loc_00D3F760:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F780:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7A0:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7C0:
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F834:
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F4A0;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F4A0;
}

void june_evt_func_00D3F4A0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F4B8; june_evt_func_00D3CA28(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
}

void june_evt_func_00D3F4C0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F4D0; june_evt_func_00D3EFD8(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
}

void june_evt_func_00D3F4F8(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x88);
        ctx->lr = 0x00D3F508; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
        ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
        ctx->gpr[4] = (int64_t)(int32_t)(0x78);
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[9]);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F530; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
}

void june_evt_func_00D3F538(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->lr = 0x00D3F54C; june_evt_func_00D3F014(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-98);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
        { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x18; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00D3F728;
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + 0x53B0);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 22, 29, 0, 0);
        ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
        ctx->ctr = ctx->gpr[0];
        switch ((uint32_t)ctx->ctr) { case 0x00D3F728u: goto loc_00D3F728;case 0x00D3F748u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F748; return; }case 0x00D3F760u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F760; return; }case 0x00D3F780u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F780; return; }case 0x00D3F7A0u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F7A0; return; }case 0x00D3F7C0u: { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F7C0; return; } default: ps3_indirect_call(ctx); return; } return;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F7DC; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F7DC; return; }
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8A0; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F924; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8D8; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F8D8; return; }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F930; return; }
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F470; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
loc_00D3F728:
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
}

void june_evt_func_00D3F5E8(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read8(ctx->gpr[3] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6C; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F608;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x71; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x6A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F7DC;
loc_00D3F608:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 0)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 0, 32);
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F924;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F930;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F930;
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F470; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
loc_00D3F7DC:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
loc_00D3F8D8:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
loc_00D3F924:
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F930:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        goto loc_00D3F690;
}

void june_evt_func_00D3F634(ppu_context* ctx) {
    PPU_MM_SNAP
loc_00D3F634:
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x52);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x68; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8A0;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x62; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F924;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x74; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x7A; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8D8;
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F930;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F930;
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F470; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00D3F634;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
loc_00D3F8A0:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
loc_00D3F8D8:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
loc_00D3F924:
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F930:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        goto loc_00D3F690;
}

void june_evt_func_00D3F660(ppu_context* ctx) {
    PPU_MM_SNAP
loc_00D3F660:
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x78; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F930;
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x58; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F930;
loc_00D3F690:
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x78); memcpy(&ctx->fpr[0], &tmp, 8); }
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x88, tmp); }
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F470; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[5] + 0x0);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D0; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F73C; june_evt_func_00D3F050(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00D3F73C:
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        goto loc_00D3F73C;
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F634; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F690;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        goto loc_00D3F690;
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_00D3F660;
loc_00D3F930:
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        goto loc_00D3F690;
}

void june_evt_func_00D3F690(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[4] = (int64_t)(int8_t)ctx->gpr[30];
        ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[0];
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->lr = 0x00D3F6A8; june_evt_func_00D3D404(ctx); DRAIN_TRAMPOLINE(ctx);
        ;
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
}

void june_evt_func_00D3F6B0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x25);
        ctx->gpr[11] = ctx->gpr[7] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
}

void june_evt_func_00D3F73C(ppu_context* ctx) {
    PPU_MM_SNAP
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F358; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
}

void june_evt_func_00D3F748(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->lr = 0x00D3F75C; june_evt_func_00D3F074(ctx); DRAIN_TRAMPOLINE(ctx);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F73C; return; }
}

void june_evt_func_00D3F760(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F73C; return; }
}

void june_evt_func_00D3F780(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(2);
        ctx->gpr[5] = (int64_t)(int16_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F73C; return; }
}

void june_evt_func_00D3F7A0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[5] = (int64_t)(int8_t)ctx->gpr[5];
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F73C; return; }
}

void june_evt_func_00D3F7C0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
        ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x48);
        ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x70);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB5);
        lv2_syscall(ctx);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F73C; return; }
}

void june_evt_func_00D3F7DC(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F634; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[4], 0, 32);
        ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
        ctx->gpr[6] = (int64_t)(int32_t)(8);
        ctx->gpr[11] = (int64_t)(int32_t)(0xB6);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[0]);
        if ((((ctx->cr >> 13) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F3D4; return; }
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F274; return; }
        ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x50);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D3F8AC;
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2B);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
loc_00D3F8AC:
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
}

void june_evt_func_00D3F8A0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x6);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x20);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F4A0; return; }
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
}

void june_evt_func_00D3F8D8(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x2D);
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
        ctx->gpr[0] = vm_read64(ctx->gpr[31] + 0x0);
        ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F404; return; }
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
}

void june_evt_func_00D3F924(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x7);
        vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F660; return; }
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
}

void june_evt_func_00D3F930(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x2C);
        ctx->gpr[0] = (int64_t)(int32_t)(0x30);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(1);
        ctx->gpr[9] = ctx->gpr[29] + ctx->gpr[9];
        ctx->gpr[11] = ctx->gpr[29] + ctx->gpr[10];
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[10] = ctx->gpr[10] + (int64_t)(1);
        vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        vm_write8(ctx->gpr[11] + 0x0, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
        { g_trampoline_fn = (void(*)(void*))june_evt_func_00D3F690; return; }
}

void june_evt_func_00D3F960(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
        { int32_t _a=(int32_t)ctx->gpr[4], _b=(int32_t)ctx->gpr[5]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
        ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        ctx->gpr[5] = (uint64_t)((int64_t)(int32_t)ctx->gpr[5] * (int64_t)(int32_t)ctx->gpr[0]);
        ctx->gpr[4] = ctx->gpr[4] - ctx->gpr[5];
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[4]);
        return;
}


void june_evt_func_00D2901C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[31] = vm_read32(ctx->gpr[3] + 0x4);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    { uint64_t a = (uint32_t)ctx->gpr[31]; uint64_t b = (uint32_t)ctx->gpr[5]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D29050;
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
loc_00D29050:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D29080;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->lr = 0x00D29064; june_evt_func_00D25328(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[31];
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[9]);
loc_00D29080:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_00D32078(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    ctx->gpr[11] = (int64_t)(int32_t)(0x193);
    lv2_syscall(ctx);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D320C8;
loc_00D320AC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00D320C8:
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00D320D0; june_evt_func_00D23CA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[31]);
    goto loc_00D320AC;
}

void june_evt_func_00D44BD4(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + 0x55F8);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(8);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x00D44BEC; june_evt_func_00D40A38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_00D40A38(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + 0x55D0);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x00D40A4C; june_evt_func_00DF4A68(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D40A68;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
loc_00D40A68:
    ctx->lr = 0x00D40A6C; june_evt_func_00D23638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
june_evt_func_00D40A70(ctx); DRAIN_TRAMPOLINE(ctx);
}

void june_evt_func_00D40A70(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + 0x55D4);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + 0x55D0);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->lr = 0x00D40A98; june_evt_func_00DF4A68(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D40AD8;
    ctx->gpr[0] = vm_read64(ctx->gpr[30] + 0x0);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read64(ctx->gpr[30] + 0x8);
    vm_write64(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[30] + 0x10);
    vm_write64(ctx->gpr[31] + 0x8, ctx->gpr[9]);
    vm_write64(ctx->gpr[31] + 0x10, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
loc_00D40AD8:
    ctx->lr = 0x00D40ADC; june_evt_func_00D23638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
june_evt_func_00D40AE0(ctx); DRAIN_TRAMPOLINE(ctx);
}

void june_evt_func_00D40AE0(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    { uint64_t a = (uint32_t)ctx->gpr[29]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00D40B30;
loc_00D40B0C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00D40B30:
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + 0x55D4);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ctx->gpr[30] + (int64_t)(0x18);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00D40B48; june_evt_func_00DF49A8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D40B8C;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[29], 3, 0, 28, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[30];
    vm_write8(ctx->gpr[9] + 0x30, ctx->gpr[3]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x50);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00D40B74;
    vm_write32(ctx->gpr[30] + 0x50, ctx->gpr[29]);
loc_00D40B74:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00D40B7C; june_evt_func_00DF49C8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00D40B0C;
loc_00D40B8C:
    ctx->lr = 0x00D40B90; june_evt_func_00D23638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
june_evt_func_00D40B94(ctx); DRAIN_TRAMPOLINE(ctx);
}

void june_evt_func_00D40B94(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + 0x55D4);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x58);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x00D40BAC; june_evt_func_00DF4C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00D40BC8;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
loc_00D40BC8:
    ctx->lr = 0x00D40BCC; june_evt_func_00D23638(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
june_evt_func_00D40BD0(ctx); DRAIN_TRAMPOLINE(ctx);
}
