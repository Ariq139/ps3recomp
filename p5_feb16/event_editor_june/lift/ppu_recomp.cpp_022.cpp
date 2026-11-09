
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

void june_evt_func_00856FE8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x170, ctx->gpr[1]); ctx->gpr[1] += -0x170;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->fpr[28] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->fpr[29] = ctx->fpr[2];
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x160, tmp); }
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    ctx->gpr[26] = ctx->gpr[10] | ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[16] = vm_read32(ctx->gpr[1] + 0x1E4);
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[17] = vm_read32(ctx->gpr[1] + 0x1EC);
    ctx->gpr[19] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[20] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[18] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00857104;
loc_0085709C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x180);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x160); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x170);
    return;
loc_00857104:
    ctx->lr = 0x00857108; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857138;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00857138:
    ctx->lr = 0x0085713C; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00857148; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00857184; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008571B0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_008571B0:
    ctx->lr = 0x008571B4; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008571C4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[28]));
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[13]);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[12]));
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[11]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857244;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
loc_00857244:
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085725C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857280;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857568;
    ;
loc_00857280:
    ctx->gpr[23] = (int64_t)(int32_t)(0);
loc_00857284:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085761C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085729C; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[1]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00857560;
loc_008572B0:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857328;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6718); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6714); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->lr = 0x00857314; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008574DC;
    ;
loc_00857328:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    ctx->lr = 0x0085734C; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857360;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_00857360:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857624;
loc_0085736C:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857428;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857428;
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_008573B8;
loc_00857390:
    ctx->lr = 0x00857394; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008573A8;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_008573A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857428;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x2C);
loc_008573B8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x008573DC; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00857418;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
loc_00857418:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857390;
loc_00857428:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857458;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[26] + 0x0, tmp); }
loc_00857458:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008574D4;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008574D4;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ;
loc_00857488:
    ctx->fpr[1] = ctx->fpr[28];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[10] = ctx->gpr[28] | ctx->gpr[28];
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[30]);
    ctx->lr = 0x008574B8; june_evt_func_00856FE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008574C8;
    ctx->gpr[23] = (int64_t)(int32_t)(1);
loc_008574C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857488;
loc_008574D4:
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[23];
    goto loc_0085709C;
loc_008574DC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857328;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857328;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857328;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857328;
    ctx->lr = 0x00857528; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857328;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66DC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857328;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[23] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = ctx->gpr[0] ^ 0x2;
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[11]);
    goto loc_00857328;
loc_00857560:
    ctx->fpr[11] = ctx->fpr[0];
    goto loc_008572B0;
loc_00857568:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857280;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857280;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857280;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[17]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[30]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008575D4;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x8, tmp); }
loc_008575D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857280;
    ctx->lr = 0x008575E4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857280;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857280;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[23] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = ctx->gpr[0] ^ 0x2;
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[11]);
    goto loc_00857284;
loc_0085761C:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_008572B0;
loc_00857624:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x670C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857708;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6708); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_00857654:
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6704); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x00857678; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085736C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085736C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085736C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085736C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085736C;
    ctx->lr = 0x008576D4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085736C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66DC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085736C;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[11] = ctx->gpr[0] ^ 0x4;
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[11]);
    goto loc_0085736C;
loc_00857708:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x672C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00857654;
}

void june_evt_func_00857710(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x30);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[21]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    vm_write32(ctx->gpr[3] + 0x2C, ctx->gpr[4]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857884;
loc_0085775C:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x30);
    vm_write32(ctx->gpr[24] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857878;
loc_00857770:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x30);
    vm_write32(ctx->gpr[21] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085786C;
loc_00857784:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x30);
    vm_write32(ctx->gpr[22] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857860;
loc_00857798:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x30);
    vm_write32(ctx->gpr[23] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857854;
loc_008577AC:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x30);
    vm_write32(ctx->gpr[25] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857848;
loc_008577C0:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x30);
    vm_write32(ctx->gpr[26] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085783C;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[28], 0, 32);
loc_008577D8:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    vm_write32(ctx->gpr[27] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857830;
    ;
loc_008577F0:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857824;
    ;
loc_00857808:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00857818; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857808;
loc_00857824:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008577F0;
loc_00857830:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008577D8;
loc_0085783C:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008577C0;
loc_00857848:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008577AC;
loc_00857854:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857798;
loc_00857860:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857784;
loc_0085786C:
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857770;
loc_00857878:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085775C;
loc_00857884:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
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
}

void june_evt_func_00858E50(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x130, ctx->gpr[1]); ctx->gpr[1] += -0x130;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[28]));
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[25]);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[27]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[22] = ctx->gpr[5] | ctx->gpr[5];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[13]));
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[21] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[20]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[31]);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00858EF0;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[6] + 0x0, tmp); }
loc_00858EF0:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858F1C;
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_00858F1C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859304;
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859304;
    ctx->fpr[23] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[28]));
loc_00858F38:
    ctx->gpr[17] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[17] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[23], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00858F58;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_00858F58:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[17] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858F84;
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_00858F84:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592F8;
    ctx->gpr[0] = vm_read32(ctx->gpr[17] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592F8;
    ctx->fpr[24] = ppu_fp_single(ppu_fadd(ctx->fpr[23], ctx->fpr[28]));
loc_00858FA0:
    ctx->gpr[18] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[18] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00858FC0;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_00858FC0:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[18] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858FEC;
    { uint32_t tmp = vm_read32(ctx->gpr[18] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_00858FEC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592EC;
    ctx->gpr[0] = vm_read32(ctx->gpr[18] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592EC;
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[28]));
loc_00859008:
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[19] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00859028;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_00859028:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[19] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859054;
    { uint32_t tmp = vm_read32(ctx->gpr[19] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_00859054:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592E0;
    ctx->gpr[0] = vm_read32(ctx->gpr[19] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592E0;
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[28]));
loc_00859070:
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00859090;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_00859090:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008590BC;
    { uint32_t tmp = vm_read32(ctx->gpr[20] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_008590BC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592D4;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592D4;
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[28]));
loc_008590D8:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008590F8;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_008590F8:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859124;
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_00859124:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592C8;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592C8;
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[28]));
loc_00859140:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00859160;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_00859160:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085918C;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_0085918C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592BC;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592BC;
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[28]));
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[21], 0, 32);
loc_008591B0:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008591D0;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_008591D0:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008591FC;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_008591FC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592B0;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592B0;
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[28]));
loc_00859218:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00859238;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0x0, tmp); }
loc_00859238:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00859264;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
loc_00859264:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592A4;
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008592A4;
    ;
loc_00859280:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00859298; june_evt_func_00858E50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859280;
loc_008592A4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859218;
loc_008592B0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008591B0;
loc_008592BC:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859140;
loc_008592C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008590D8;
loc_008592D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859070;
loc_008592E0:
    ctx->gpr[0] = vm_read32(ctx->gpr[19] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00859008;
loc_008592EC:
    ctx->gpr[0] = vm_read32(ctx->gpr[18] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00858FA0;
loc_008592F8:
    ctx->gpr[3] = vm_read32(ctx->gpr[17] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00858F38;
loc_00859304:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
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
}

void june_evt_func_0085BA44(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x10);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[4], 0, 32);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BB2C;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BB48;
loc_0085BA80:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BB60;
loc_0085BA98:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x38, ctx->gpr[11]);
loc_0085BAA0:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BAB8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085BAB4; june_evt_func_008B15CC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085BAB8:
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085BAC4; june_evt_func_00878FA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BAE0;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085BADC; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085BAE0:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BAF8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085BAF4; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085BAF8:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->lr = 0x0085BB00; june_evt_func_008632F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085BB0C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0085BB2C:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BB68;
    vm_write32(ctx->gpr[3] + 0x10, ctx->gpr[0]);
loc_0085BB3C:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085BA80;
loc_0085BB48:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085BA98;
loc_0085BB60:
    vm_write32(ctx->gpr[3] + 0x8, ctx->gpr[11]);
    goto loc_0085BAA0;
loc_0085BB68:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x38);
    vm_write32(ctx->gpr[3] + 0x10, ctx->gpr[0]);
    goto loc_0085BB3C;
}

void june_evt_func_0086C384(ppu_context* ctx) {
    PPU_MM_SNAP
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0086C3D0;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62E4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ;
loc_0086C398:
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086C3B0;
    ctx->fpr[12] = ctx->fpr[0];
loc_0086C3B0:
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086C398;
    ctx->fpr[0] = ctx->fpr[12];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(-16);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + -0x10);
loc_0086C3D0:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
    return;
}

void june_evt_func_0086C3D8(ppu_context* ctx) {
    PPU_MM_SNAP
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0086C420;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62E4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0086C3E8:
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086C400;
    ctx->fpr[12] = ctx->fpr[0];
loc_0086C400:
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086C3E8;
    ctx->fpr[0] = ctx->fpr[12];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(-16);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + -0x10);
loc_0086C420:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
    return;
}







void june_evt_func_0085D840(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x190, ctx->gpr[1]); ctx->gpr[1] += -0x190;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x188, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x1A0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x2C);
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D898;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D8E8;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085D8B0;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D970;
    ;
loc_0085D888:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0085D894; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085D898:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1A0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x190);
    return;
loc_0085D8B0:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D920;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6638);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085D8DC; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    goto loc_0085D888;
loc_0085D8E8:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D948;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6630);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085D914; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    goto loc_0085D888;
loc_0085D920:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x663C);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085D93C; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    goto loc_0085D888;
loc_0085D948:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6634);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085D964; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    goto loc_0085D888;
loc_0085D970:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D9B0;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6628);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085D9A4; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    goto loc_0085D888;
loc_0085D9B0:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x662C);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085D9D4; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    goto loc_0085D888;
}

void june_evt_func_00834970(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x8, ctx->gpr[31]);
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x70A8);
    vm_write64(ctx->gpr[1] + 0x10, ctx->gpr[0]);
    ctx->gpr[0] = vm_read8(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00834A0C;
    ctx->gpr[11] = (int64_t)(int32_t)(0x93);
    lv2_syscall(ctx);
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x70A4);
    ctx->gpr[7] = ctx->gpr[8] | ctx->gpr[8];
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_00834A80;
    ctx->gpr[9] = (uint64_t)ppc_srad(&ctx->xer, (int64_t)ctx->gpr[3], 0x35);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 53);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x7FF);
    { uint64_t a = (uint64_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[3];
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[11], 0, 52);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00834A70;
loc_008349C4:
    vm_write64(ctx->gpr[1] + -0x20, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x20); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_008349D4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x70A0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x0, tmp); }
loc_008349E0:
    ctx->gpr[9] = ppu_timebase_now();
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008349E0;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x709C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    vm_write64(ctx->gpr[8] + 0x8, ctx->gpr[9]);
    vm_write8(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    return;
loc_00834A0C:
    ctx->gpr[0] = ppu_timebase_now();
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00834A0C;
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x70A4);
    ctx->gpr[9] = vm_read64(ctx->gpr[8] + 0x8);
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9]; { int64_t _r = (int64_t)ctx->gpr[0]; uint32_t _c = (_r < 0) ? 8u : (_r > 0) ? 4u : 2u; _c |= (uint32_t)((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & 0x0FFFFFFFu) | (_c << 28); }
    if ((((ctx->cr >> 31) & 1) == 1)) goto loc_00834AC8;
    ctx->gpr[9] = (uint64_t)ppc_srad(&ctx->xer, (int64_t)ctx->gpr[0], 0x35);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[0], 0, 53);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x7FF);
    { uint64_t a = (uint64_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[11], 0, 52);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00834A78;
loc_00834A48:
    vm_write64(ctx->gpr[1] + -0x20, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x20); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
loc_00834A58:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[1], ctx->fpr[0]));
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    return;
loc_00834A70:
    ctx->gpr[11] = ctx->gpr[3] | ctx->gpr[3];
    goto loc_008349C4;
loc_00834A78:
    ctx->gpr[11] = ctx->gpr[0] | ctx->gpr[0];
    goto loc_00834A48;
loc_00834A80:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[3], 0, 63);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[3], 63, 1);
    ctx->gpr[10] = ctx->gpr[10] | ctx->gpr[0];
    ctx->gpr[9] = (uint64_t)ppc_srad(&ctx->xer, (int64_t)ctx->gpr[10], 0x35);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 53);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x7FF);
    { uint64_t a = (uint64_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[10];
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[11], 0, 52);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00834AB0;
    ctx->gpr[11] = ctx->gpr[10] | ctx->gpr[10];
loc_00834AB0:
    vm_write64(ctx->gpr[1] + -0x20, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x20); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[13]));
    goto loc_008349D4;
loc_00834AC8:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 63);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 63, 1);
    ctx->gpr[10] = ctx->gpr[10] | ctx->gpr[0];
    ctx->gpr[9] = (uint64_t)ppc_srad(&ctx->xer, (int64_t)ctx->gpr[10], 0x35);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[10], 0, 53);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x7FF);
    { uint64_t a = (uint64_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[10];
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[11], 0, 52);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00834AF8;
    ctx->gpr[11] = ctx->gpr[10] | ctx->gpr[10];
loc_00834AF8:
    vm_write64(ctx->gpr[1] + -0x20, ctx->gpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + -0x20); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[1]));
    goto loc_00834A58;
}

void june_evt_func_0086DDD0(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x7C);
    return;
}

void june_evt_func_0086E0AC(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6270);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x8);
    return;
}

void june_evt_func_0086F880(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0x7C);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086F93C;
loc_0086F8AC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086F930;
    ;
loc_0086F8C0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086F8C0;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x84);
    vm_write32(ctx->gpr[9] + 0x84, ctx->gpr[0]);
loc_0086F8D8:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x80);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x78, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x84, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x7C, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086F914;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
loc_0086F8F8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0086F908; june_evt_func_0086F6D0(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086F8F8;
loc_0086F914:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0086F930:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x84);
    vm_write32(ctx->gpr[9] + 0x80, ctx->gpr[0]);
    goto loc_0086F8D8;
loc_0086F93C:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6214);
    ctx->gpr[4] = (int64_t)(int32_t)(0x23B);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->lr = 0x0086F94C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086F8AC;
}

void june_evt_func_00855600(ppu_context* ctx) {
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
    ctx->gpr[4] = (int64_t)(int32_t)(0x162);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085563C;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6774);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6778);
    ctx->lr = 0x00855638; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085563C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 4, 0, 27, 0, 0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x0);
    return;
}

void june_evt_func_00882C2C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x00882C44; june_evt_func_00876934(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00882ABC; return; }
}

void june_evt_func_0083F098(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6D60);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x0083F0AC; june_evt_func_00887394(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[3], 0x1F);
    ctx->gpr[3] = ctx->gpr[0] ^ ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[0] - ctx->gpr[3];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 1, 31, 31, 0, 0);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    return;
}

void june_evt_func_00942D78(ppu_context* ctx) {
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
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[18] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[27] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[26] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[24] = ctx->gpr[10] | ctx->gpr[10];
    ctx->gpr[19] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094310C;
    { uint64_t a = (uint32_t)ctx->gpr[6]; uint64_t b = (uint64_t)0x12; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00942E68;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x11; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009436A8;
loc_00942E00:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2CC8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2A7);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2CCC);
    ctx->lr = 0x00942E10; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00942E14:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_00942E68:
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x13; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009433F8;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0x16; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00942E00;
    ctx->lr = 0x00942E7C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943A10;
loc_00942E90:
    ctx->lr = 0x00942E94; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00942EA0; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00942EAC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009439F0;
loc_00942EC0:
    ctx->lr = 0x00942EC4; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00942ED4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943AF0;
loc_00942EEC:
    ctx->lr = 0x00942EF0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->lr = 0x00942F24; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[23];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[22];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[20];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[21];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[9]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[13]));
    ctx->gpr[5] = (int64_t)(int32_t)(0x118);
    ctx->gpr[6] = (int64_t)(int32_t)(0xC8);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[3], (uint32_t)tmp); }
    ctx->lr = 0x0094300C; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x2080 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2C98);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[5] = (int64_t)(int32_t)(0x118);
    ctx->gpr[6] = (int64_t)(int32_t)(0xC8);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x00943038; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[29] + 0x88, ctx->gpr[19]);
    ctx->lr = 0x0094304C; june_evt_func_0086E170(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x3C);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2C94);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = vm_read32(ctx->gpr[29] + 0x20);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->lr = 0x00943088; june_evt_func_00869310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00943098; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = vm_read64(ctx->gpr[29] + 0x90);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x8;
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->lr = 0x009430BC; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(0x16);
loc_009430E8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x009430FC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00943104; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00942E14;
loc_0094310C:
    ctx->lr = 0x00943110; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_009433A8;
    ctx->lr = 0x00943128; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00943134; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00943140; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943388;
    ;
loc_00943158:
    ctx->lr = 0x0094315C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0094316C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943954;
loc_00943184:
    ctx->lr = 0x00943188; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->lr = 0x009431BC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[22];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[21];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[20];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[10]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[13]));
    ctx->gpr[5] = (int64_t)(int32_t)(0x110);
    ctx->gpr[6] = (int64_t)(int32_t)(0x3C);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[3], (uint32_t)tmp); }
    ctx->lr = 0x009432A4; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x2080 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2C98);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[5] = (int64_t)(int32_t)(0x110);
    ctx->gpr[6] = (int64_t)(int32_t)(0x3C);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x009432D0; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[29] + 0x88, ctx->gpr[19]);
    ctx->lr = 0x009432E4; june_evt_func_0086E170(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x3C);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2CD8);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = vm_read32(ctx->gpr[29] + 0x20);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(8);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x98);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->lr = 0x00943324; june_evt_func_00869310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00943334; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = vm_read64(ctx->gpr[29] + 0x90);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x8;
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->lr = 0x00943358; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(0x12);
    goto loc_009430E8;
loc_00943388:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    goto loc_00943158;
loc_009433A8:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CBC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x009433C8; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x009433D4; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x009433E0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943158;
    goto loc_00943388;
loc_009433F8:
    ctx->lr = 0x009433FC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094342C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CBC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0094342C:
    ctx->lr = 0x00943430; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0094343C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00943448; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943478;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00943478:
    ctx->lr = 0x0094347C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0094348C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943A30;
loc_009434A4:
    ctx->lr = 0x009434A8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->lr = 0x009434DC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[22];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[21];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[20];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[9]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CFC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2C9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[13]));
    ctx->gpr[5] = (int64_t)(int32_t)(0xE0);
    ctx->gpr[6] = (int64_t)(int32_t)(0x16);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[3], (uint32_t)tmp); }
    ctx->lr = 0x009435C4; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x2080 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2C98);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[5] = (int64_t)(int32_t)(0xE0);
    ctx->gpr[6] = (int64_t)(int32_t)(0x16);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x009435F0; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[29] + 0x88, ctx->gpr[19]);
    ctx->lr = 0x00943604; june_evt_func_0086E170(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x3C);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x2CD8);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = vm_read32(ctx->gpr[29] + 0x20);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x78);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00943644; june_evt_func_00869310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00943654; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = vm_read64(ctx->gpr[29] + 0x90);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x8;
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->lr = 0x00943678; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(0x13);
    goto loc_009430E8;
loc_009436A8:
    ctx->lr = 0x009436AC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009436DC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CBC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_009436DC:
    ctx->lr = 0x009436E0; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x009436EC; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x009436F8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2CC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943728;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00943728:
    ctx->lr = 0x0094372C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0094373C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943A90;
loc_00943754:
    ctx->lr = 0x00943758; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->lr = 0x0094378C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[22];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[21];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[20];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00942E14;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943C2C;
loc_00943840:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943C10;
loc_00943850:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094387C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2CCC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2CA8);
    ctx->lr = 0x00943874; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x8);
loc_0094387C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[30] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009438B8;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x102C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943C48;
loc_0094389C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->lr = 0x009438A8; june_evt_func_00889FA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xC0);
    ctx->lr = 0x009438B4; june_evt_func_008E8F38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_009438B8:
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2CA0);
    ctx->gpr[4] = ctx->gpr[18] + (int64_t)(4);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x88);
    vm_write32(ctx->gpr[1] + 0x8C, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[19]);
    vm_write32(ctx->gpr[1] + 0x94, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[25]);
    ctx->lr = 0x009438E8; june_evt_func_00843138(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00943900; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = vm_read64(ctx->gpr[29] + 0x90);
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x8;
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    ctx->lr = 0x00943924; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942E14;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(4);
    ctx->gpr[6] = (int64_t)(int32_t)(0x11);
    goto loc_009430E8;
loc_00943954:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00943960; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0094397C;
    ctx->gpr[0] = vm_read64(ctx->gpr[3] + 0x90);
    ctx->gpr[9] = vm_read64(ctx->gpr[31] + 0x20);
    { int64_t a = (int64_t)ctx->gpr[9]; int64_t b = (int64_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943184;
loc_0094397C:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009439D4;
    ctx->lr = 0x00943988; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009439D4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x009439C8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x009439D0; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_009439D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 1, 31, 0, 0);
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00943184;
loc_009439F0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CB8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00942EC0;
loc_00943A10:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2CBC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    goto loc_00942E90;
loc_00943A30:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00943A3C; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943A58;
    ctx->gpr[0] = vm_read64(ctx->gpr[3] + 0x90);
    ctx->gpr[9] = vm_read64(ctx->gpr[31] + 0x20);
    { int64_t a = (int64_t)ctx->gpr[9]; int64_t b = (int64_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_009434A4;
loc_00943A58:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943A74;
    ctx->lr = 0x00943A64; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943B90;
loc_00943A74:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 1, 31, 0, 0);
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_009434A4;
loc_00943A90:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00943A9C; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943AB8;
    ctx->gpr[0] = vm_read64(ctx->gpr[3] + 0x90);
    ctx->gpr[9] = vm_read64(ctx->gpr[31] + 0x20);
    { int64_t a = (int64_t)ctx->gpr[9]; int64_t b = (int64_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943754;
loc_00943AB8:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943AD4;
    ctx->lr = 0x00943AC4; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943B50;
loc_00943AD4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 1, 31, 0, 0);
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00943754;
loc_00943AF0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00943AFC; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943B18;
    ctx->gpr[0] = vm_read64(ctx->gpr[3] + 0x90);
    ctx->gpr[9] = vm_read64(ctx->gpr[31] + 0x20);
    { int64_t a = (int64_t)ctx->gpr[9]; int64_t b = (int64_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00942EEC;
loc_00943B18:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00943B34;
    ctx->lr = 0x00943B24; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00943BD0;
loc_00943B34:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 1, 31, 0, 0);
    vm_write64(ctx->gpr[31] + 0x20, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00942EEC;
loc_00943B50:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00943B80; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00943B88; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00943AD4;
loc_00943B90:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00943BC0; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00943BC8; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00943A74;
loc_00943BD0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00943C00; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00943C08; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00943B34;
loc_00943C10:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2CCC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2CAC);
    ctx->lr = 0x00943C20; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_00943850;
loc_00943C2C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2CCC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2CB0);
    ctx->lr = 0x00943C3C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x8);
    goto loc_00943840;
loc_00943C48:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2CCC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x33);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2CA4);
    ctx->lr = 0x00943C58; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x102C);
    goto loc_0094389C;
}

void june_evt_func_008946F8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    { uint64_t a = (uint32_t)ctx->gpr[3]; uint64_t b = (uint64_t)0x1E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x4C3);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00894728;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5AA8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5AA4);
    ctx->lr = 0x00894724; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00894728:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 2, 0, 29, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x5AAC);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 5, 0, 26, 0, 0);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x8);
    return;
}

void june_evt_func_008588D4(ppu_context* ctx) {
    PPU_MM_SNAP
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) return;
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008589B8;
    return;
loc_0085894C:
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
loc_008589B8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) return;
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x28);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[9]);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085894C;
    return;
}

void june_evt_func_00858A98(ppu_context* ctx) {
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
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[18] = ctx->gpr[5] | ctx->gpr[5];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[5] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E48;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E48;
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858B28:
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[19] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[19] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D84;
    ctx->gpr[0] = vm_read32(ctx->gpr[19] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D84;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858B60:
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DA0;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DA0;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858B98:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[21] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DD8;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DD8;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858BD0:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[22] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DF4;
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DF4;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858C08:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E10;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E10;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858C40:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E2C;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E2C;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858C78:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DBC;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DBC;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
loc_00858CB0:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[27] + 0x0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D68;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D68;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[18], 0, 32);
    goto loc_00858CF8;
loc_00858CEC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D68;
loc_00858CF8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00858D0C; june_evt_func_00858A98(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858CEC;
loc_00858D18:
    ctx->gpr[3] = (int64_t)(int32_t)(1);
loc_00858D1C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_00858D68:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DBC;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858CB0;
loc_00858D84:
    ctx->gpr[0] = vm_read32(ctx->gpr[19] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E48;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858B28;
loc_00858DA0:
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D84;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858B60;
loc_00858DBC:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E2C;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858C78;
loc_00858DD8:
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DA0;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858B98;
loc_00858DF4:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DD8;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858BD0;
loc_00858E10:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858DF4;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858C08;
loc_00858E2C:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858E10;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858D18;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00858C40;
loc_00858E48:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    goto loc_00858D1C;
}

void june_evt_func_008982F8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x150, ctx->gpr[1]); ctx->gpr[1] += -0x150;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    ctx->fpr[31] = ctx->fpr[1];
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x30);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[27]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089838C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5AA8);
    ctx->gpr[4] = (int64_t)(int32_t)(0xBB6);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x59F0);
    ctx->lr = 0x00898384; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
loc_0089838C:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write32(ctx->gpr[1] + 0xA0, ctx->gpr[9]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[18] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008983AC; june_evt_func_0089800C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x5AAC);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 5, 0, 26, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 29, 0, 0);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[26] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->lr = 0x008983E4; june_evt_func_00D335C8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008984C0;
    ctx->gpr[25] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0089851C;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898694;
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5AA8);
    ctx->gpr[4] = (int64_t)(int32_t)(0xBC9);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x59EC);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0089841C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00898420:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = (int64_t)(int32_t)(4);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = (int64_t)(int32_t)(0xBCA);
    ctx->lr = 0x00898438; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898460;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x59E8);
    ctx->gpr[4] = (int64_t)(int32_t)(0xBCC);
    ctx->lr = 0x0089845C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00898460:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[6] = (int64_t)(int32_t)(0xBCD);
    ctx->lr = 0x00898478; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[3]);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[9]);
loc_00898498:
    ctx->gpr[9] = ctx->gpr[10] + ctx->gpr[11];
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[0] + ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    ctx->lr = 0x008984BC; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008984C0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_0089851C:
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[20] = (int64_t)(int32_t)(0);
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008985A8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = (int64_t)(int32_t)(4);
    ctx->ctr = ctx->gpr[9];
    goto loc_00898574;
loc_00898564:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898670;
loc_00898574:
    ctx->gpr[31] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[27] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(4);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00898564;
    ctx->gpr[21] = ppc_rlw64(ctx->gpr[28], 2, 0, 29, 0, 0);
    ctx->gpr[23] = (uint64_t)((int64_t)(int32_t)ctx->gpr[26] * (int64_t)(int32_t)ctx->gpr[28]);
    ctx->gpr[27] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[22] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[31] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[20] = ctx->gpr[23] | ctx->gpr[23];
loc_008985A8:
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[0] = ctx->gpr[23] | ctx->gpr[23];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898498;
loc_008985B8:
    ctx->gpr[29] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[14] = vm_read32(ctx->gpr[2] + -0x5AA8);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[29], 2, 0, 29, 0, 0);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[6] = (int64_t)(int32_t)(0xBDE);
    ctx->gpr[29] = (uint64_t)((int64_t)(int32_t)ctx->gpr[26] * (int64_t)(int32_t)ctx->gpr[29]);
    ctx->lr = 0x008985DC; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[19] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0xBDF);
    ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00898600; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[15] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[16] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0089869C;
loc_00898614:
    { uint64_t a = (uint32_t)ctx->gpr[25]; uint64_t b = (uint32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008986C8;
loc_0089861C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089873C;
loc_00898628:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00898630; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898720;
loc_00898640:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->lr = 0x00898648; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[10] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[0] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[11] = ctx->gpr[19] | ctx->gpr[19];
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[19]);
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[15]);
    vm_write32(ctx->gpr[30] + 0x4, ctx->gpr[9]);
    goto loc_00898498;
loc_00898670:
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[21] = ppc_rlw64(ctx->gpr[28], 2, 0, 29, 0, 0);
    ctx->gpr[23] = (uint64_t)((int64_t)(int32_t)ctx->gpr[26] * (int64_t)(int32_t)ctx->gpr[28]);
    ctx->gpr[10] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[20] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[22] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[0] = ctx->gpr[23] | ctx->gpr[23];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00898498;
    goto loc_008985B8;
loc_00898694:
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5AA8);
    goto loc_00898420;
loc_0089869C:
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x008986AC; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->lr = 0x008986C0; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00898614;
loc_008986C8:
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[3] = ctx->gpr[17] + ctx->gpr[31];
    ctx->gpr[29] = ctx->gpr[25] - ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[31] + ctx->gpr[4];
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(4);
    ctx->gpr[5] = ppc_rlw64(ctx->gpr[29], 2, 0, 29, 0, 0);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[29] = (uint64_t)((int64_t)(int32_t)ctx->gpr[26] * (int64_t)(int32_t)ctx->gpr[29]);
    ctx->lr = 0x008986F0; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[3] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[4] = ctx->gpr[20] + ctx->gpr[4];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = (uint64_t)((int64_t)(int32_t)ctx->gpr[26] * (int64_t)(int32_t)ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[16] + ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00898718; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0089861C;
loc_00898720:
    ctx->gpr[3] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5A94);
    ctx->gpr[4] = (int64_t)(int32_t)(0xBF0);
    ctx->lr = 0x00898730; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    goto loc_00898640;
loc_0089873C:
    ctx->gpr[3] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5A98);
    ctx->gpr[4] = (int64_t)(int32_t)(0xBED);
    ctx->lr = 0x0089874C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    goto loc_00898628;
}

void june_evt_func_008F3E04(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x9; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[4] = (int64_t)(int32_t)(0x15A);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008F3E9C;
loc_008F3E38:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 6, 0, 25, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x4410);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 28, 0, 0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[0] = vm_read32((ctx->gpr[11] + ctx->gpr[9]));
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008F3E80;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x008F3E7C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_008F3E80:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_008F3E9C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x4418);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x4414);
    ctx->lr = 0x008F3EA8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    goto loc_008F3E38;
}

void june_evt_func_00852A50(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[4]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852E3C;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852E04;
loc_00852AAC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852BD0;
loc_00852AB8:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[24] + 0x2C, ctx->gpr[28]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852DDC;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852E88;
loc_00852AD0:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852BC4;
loc_00852ADC:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[23] + 0x2C, ctx->gpr[28]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852EE4;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852DA4;
loc_00852AF4:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852BB8;
loc_00852B00:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[25] + 0x2C, ctx->gpr[28]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852D1C;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852D44;
loc_00852B18:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852BAC;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[28], 0, 32);
loc_00852B28:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[27] + 0x2C, ctx->gpr[28]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852C94;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852CBC;
loc_00852B40:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852BA0;
    ;
loc_00852B50:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[28]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852C0C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852C34;
loc_00852B68:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852B94;
    ;
loc_00852B78:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00852B88; june_evt_func_00852A50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852B78;
loc_00852B94:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852B50;
loc_00852BA0:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852B28;
loc_00852BAC:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852B00;
loc_00852BB8:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852ADC;
loc_00852BC4:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852AB8;
loc_00852BD0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xC8);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00852C0C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852C2C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->lr = 0x00852C24; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[28]);
loc_00852C2C:
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[28]);
    goto loc_00852B68;
loc_00852C34:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852C64;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x00852C4C; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852C6C;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
loc_00852C64:
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[3]);
    goto loc_00852B68;
loc_00852C6C:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[31] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->lr = 0x00852C84; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[3]);
    goto loc_00852B68;
loc_00852C94:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852CB4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->lr = 0x00852CAC; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[28]);
loc_00852CB4:
    vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[28]);
    goto loc_00852B40;
loc_00852CBC:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852CEC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0xC);
    ctx->lr = 0x00852CD4; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[27] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[27] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852CF4;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
loc_00852CEC:
    vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[3]);
    goto loc_00852B40;
loc_00852CF4:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[27] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->lr = 0x00852D0C; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[3]);
    goto loc_00852B40;
loc_00852D1C:
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852D3C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->lr = 0x00852D34; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[25] + 0x8, ctx->gpr[28]);
loc_00852D3C:
    vm_write32(ctx->gpr[25] + 0x4, ctx->gpr[28]);
    goto loc_00852B18;
loc_00852D44:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852D74;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0xC);
    ctx->lr = 0x00852D5C; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[25] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[25] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852D7C;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
loc_00852D74:
    vm_write32(ctx->gpr[25] + 0x4, ctx->gpr[3]);
    goto loc_00852B18;
loc_00852D7C:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[25] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[25] + 0x18);
    ctx->lr = 0x00852D94; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    vm_write32(ctx->gpr[25] + 0x4, ctx->gpr[3]);
    goto loc_00852B18;
loc_00852DA4:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852DD4;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0xC);
    ctx->lr = 0x00852DBC; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[23] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[23] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852F0C;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
loc_00852DD4:
    vm_write32(ctx->gpr[23] + 0x4, ctx->gpr[3]);
    goto loc_00852AF4;
loc_00852DDC:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852DFC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x8);
    ctx->lr = 0x00852DF4; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[24] + 0x8, ctx->gpr[28]);
loc_00852DFC:
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[28]);
    goto loc_00852AD0;
loc_00852E04:
    ctx->gpr[3] = vm_read32(ctx->gpr[4] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852E34;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x00852E1C; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852E64;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
loc_00852E34:
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[3]);
    goto loc_00852AAC;
loc_00852E3C:
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852E5C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->lr = 0x00852E54; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[26]);
loc_00852E5C:
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[26]);
    goto loc_00852AAC;
loc_00852E64:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[31] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->lr = 0x00852E7C; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    goto loc_00852E34;
loc_00852E88:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852EB8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->lr = 0x00852EA0; june_evt_func_0088A44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = vm_read32(ctx->gpr[24] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[24] + 0x8, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852EC0;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
loc_00852EB8:
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[3]);
    goto loc_00852AD0;
loc_00852EC0:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[24] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[24] + 0x18);
    ctx->lr = 0x00852ED8; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    goto loc_00852EB8;
loc_00852EE4:
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852F04;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x8);
    ctx->lr = 0x00852EFC; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[23] + 0x8, ctx->gpr[28]);
loc_00852F04:
    vm_write32(ctx->gpr[23] + 0x4, ctx->gpr[28]);
    goto loc_00852AF4;
loc_00852F0C:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read16(ctx->gpr[23] + 0x10);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[23] + 0x18);
    ctx->lr = 0x00852F24; june_evt_func_00889C28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    goto loc_00852DD4;
}

void june_evt_func_008578C0(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0x28);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085797C;
loc_008578EC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857970;
    ;
loc_00857900:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857900;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_00857918:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857954;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
loc_00857938:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00857948; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857938;
loc_00857954:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00857970:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_00857918;
loc_0085797C:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66D8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x335);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->lr = 0x0085798C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008578EC;
}

void june_evt_func_00889C28(ppu_context* ctx) {
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
    ctx->gpr[12] = ctx->cr;
    { uint64_t a = (uint32_t)ctx->gpr[5]; uint64_t b = (uint64_t)0x5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[23] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[26] = ctx->gpr[7] | ctx->gpr[7];
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00889E7C;
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[25] = vm_read32(ctx->gpr[2] + -0x5C1C);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00889E9C;
loc_00889C94:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889CB4;
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C10);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1FE);
    ctx->lr = 0x00889CB0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00889CB4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[22] = ppc_rlw64(ctx->gpr[26], 6, 0, 25, 0, 0);
    ctx->gpr[29] = ppc_rlw64(ctx->gpr[26], 4, 0, 27, 0, 0);
    ctx->lr = 0x00889CC4; june_evt_func_0089B3B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[29] = ctx->gpr[22] - ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[0] | ctx->gpr[0];
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[0]);
    ctx->lr = 0x00889CE0; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x208);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write16(ctx->gpr[28] + 0x0, ctx->gpr[27]);
    vm_write32(ctx->gpr[28] + 0x4, ctx->gpr[30]);
    ctx->lr = 0x00889D0C; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00889D24; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[21]);
    vm_write32(ctx->gpr[28] + 0x1C, ctx->gpr[27]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889D50;
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C0C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20E);
    ctx->lr = 0x00889D4C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00889D50:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[6] = (int64_t)(int32_t)(0x211);
    ctx->lr = 0x00889D6C; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[0]);
    ctx->lr = 0x00889D84; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00889E38;
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    goto loc_00889DD0;
loc_00889D9C:
    ctx->gpr[11] = ctx->gpr[27] + ctx->gpr[24];
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[28], 6, 0, 25, 0, 0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[10]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x30);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00889E38;
loc_00889DD0:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[23];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x216);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(4);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint64_t)0x1E; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00889D9C;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C08);
    ctx->lr = 0x00889DF8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = ctx->gpr[27] + ctx->gpr[24];
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[28], 6, 0, 25, 0, 0);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[10]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x30);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00889DD0;
    ;
loc_00889E38:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xD8);
    ctx->gpr[21] = _cs_21;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[22] = _cs_22;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
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
loc_00889E7C:
    ctx->gpr[25] = vm_read32(ctx->gpr[2] + -0x5C1C);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[4] = (int64_t)(int32_t)(0x1FC);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C18);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x00889E94; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_00889C94;
loc_00889E9C:
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C14);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1FD);
    ctx->lr = 0x00889EAC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00889C94;
}

void june_evt_func_0088A44C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x12D);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0088A564;
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5C1C);
loc_0088A480:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x12F);
    ctx->gpr[3] = (int64_t)(int32_t)(0x1C);
    ctx->lr = 0x0088A498; june_evt_func_00884198(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x130);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A57C;
loc_0088A4B4:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x5C20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x4, tmp); }
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[9]);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x14);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[28]);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A528;
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[28] = _cs_28;
    vm_write32(ctx->gpr[10] + 0x14, ctx->gpr[30]);
    ctx->lr = ctx->gpr[0];
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[11]);
    vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[9]);
    vm_write32(ctx->gpr[29] + 0x14, ctx->gpr[31]);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0088A528:
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(1);
    ctx->gpr[28] = _cs_28;
    vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[30]);
    ctx->lr = ctx->gpr[0];
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[11]);
    vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[9]);
    vm_write32(ctx->gpr[29] + 0x14, ctx->gpr[31]);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0088A564:
    ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5C1C);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C04);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0088A574; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0088A480;
loc_0088A57C:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BF4);
    ctx->lr = 0x0088A584; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0088A4B4;
}

void june_evt_func_0088A58C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0088A6D0;
loc_0088A5BC:
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A6B0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A6C8;
loc_0088A5E0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[11]);
loc_0088A5E8:
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)-1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A648;
    ctx->gpr[31] = vm_read32(ctx->gpr[29] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A638;
loc_0088A600:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = (int64_t)(int32_t)(0x151);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A69C;
loc_0088A60C:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0088A628;
    vm_write32(ctx->gpr[11] + 0x8, ctx->gpr[9]);
loc_0088A628:
    ctx->gpr[31] = vm_read32(ctx->gpr[11] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088A600;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
loc_0088A638:
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->lr = 0x0088A644; june_evt_func_0089C9A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0088A648:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088A670;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0088A66C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0088A670:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0088A678; june_evt_func_00884168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0088A69C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5C1C);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5BF0);
    ctx->lr = 0x0088A6A8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0088A60C;
loc_0088A6B0:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x14);
    vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088A5E0;
loc_0088A6C8:
    vm_write32(ctx->gpr[29] + 0x14, ctx->gpr[11]);
    goto loc_0088A5E8;
loc_0088A6D0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5C1C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x148);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5C04);
    ctx->lr = 0x0088A6E0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0088A5BC;
}

void june_evt_func_0085A110(ppu_context* ctx) {
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
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x28);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A1B4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A244;
loc_0085A160:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A160;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0085A178:
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x30);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A1D4;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[28], 0, 32);
loc_0085A198:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085A1A8; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A198;
loc_0085A1B4:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A1D4;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66CC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x31E);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->lr = 0x0085A1D0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085A1D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x30);
    vm_write32(ctx->gpr[27] + 0x30, ctx->gpr[28]);
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x30);
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[26]);
    vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A21C;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ;
loc_0085A200:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085A210; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A200;
loc_0085A21C:
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
loc_0085A244:
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0085A178;
}

void june_evt_func_00853138(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1C);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->lr = 0x00853160; june_evt_func_0085A250(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085317C;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0085317C; june_evt_func_00852F30(ctx); DRAIN_TRAMPOLINE(ctx);
loc_0085317C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008531A4;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ;
loc_00853190:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[11]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00853190;
loc_008531A4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008531C4;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6814);
    ctx->gpr[4] = (int64_t)(int32_t)(0x541);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->lr = 0x008531C0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008531C4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008531DC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008531D8; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008531DC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853208;
loc_008531E8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->lr = 0x008531F8; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008531E8;
loc_00853208:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[31] + 0x38, ctx->gpr[0]);
    ctx->lr = 0x00853218; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_00853438(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
    ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x30);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[18]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008537D8;
loc_00853494:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x30);
    ctx->gpr[18] = vm_read32(ctx->gpr[22] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085376C;
loc_008534A8:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x30);
    ctx->gpr[21] = vm_read32(ctx->gpr[24] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008536FC;
loc_008534BC:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0x30);
    ctx->gpr[20] = vm_read32(ctx->gpr[25] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853668;
loc_008534D0:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x30);
    ctx->gpr[19] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008535F4;
loc_008534E4:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x30);
    ctx->gpr[27] = vm_read32(ctx->gpr[28] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853580;
loc_008534F8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085352C;
    ;
loc_00853510:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x34);
    ctx->lr = 0x00853520; june_evt_func_00853438(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00853510;
loc_0085352C:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[29]);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[29]);
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[29]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853568;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008535D4;
loc_00853554:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00853560; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[29]);
loc_00853568:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00853570; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853580;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    goto loc_008534F8;
loc_00853580:
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[28] + 0x28, ctx->gpr[31]);
    vm_write32(ctx->gpr[28] + 0x30, ctx->gpr[31]);
    vm_write32(ctx->gpr[28] + 0x34, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008535BC;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853648;
loc_008535A8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x008535B4; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[28] + 0x8, ctx->gpr[31]);
loc_008535BC:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x008535C4; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008535F4;
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    goto loc_008534E4;
loc_008535D4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x008535E4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00853554;
loc_008535F4:
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[26] + 0x28, ctx->gpr[31]);
    vm_write32(ctx->gpr[26] + 0x30, ctx->gpr[31]);
    vm_write32(ctx->gpr[26] + 0x34, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853630;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008536BC;
loc_0085361C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00853628; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[31]);
loc_00853630:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x00853638; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853668;
    ctx->gpr[3] = ctx->gpr[19] | ctx->gpr[19];
    goto loc_008534D0;
loc_00853648:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x00853658; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x4);
    goto loc_008535A8;
loc_00853668:
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[25] + 0x28, ctx->gpr[31]);
    vm_write32(ctx->gpr[25] + 0x30, ctx->gpr[31]);
    vm_write32(ctx->gpr[25] + 0x34, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008536A4;
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008536DC;
loc_00853690:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0085369C; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[25] + 0x8, ctx->gpr[31]);
loc_008536A4:
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x008536AC; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008536FC;
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    goto loc_008534BC;
loc_008536BC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x008536CC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x4);
    goto loc_0085361C;
loc_008536DC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x008536EC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0x4);
    goto loc_00853690;
loc_008536FC:
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x8);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[24] + 0x28, ctx->gpr[31]);
    vm_write32(ctx->gpr[24] + 0x30, ctx->gpr[31]);
    vm_write32(ctx->gpr[24] + 0x34, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853754;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00853740;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x00853734; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[24] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x4);
loc_00853740:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0085374C; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[24] + 0x8, ctx->gpr[31]);
loc_00853754:
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->lr = 0x0085375C; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085376C;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    goto loc_008534A8;
loc_0085376C:
    ctx->gpr[4] = vm_read32(ctx->gpr[22] + 0x8);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[22] + 0x28, ctx->gpr[31]);
    vm_write32(ctx->gpr[22] + 0x30, ctx->gpr[31]);
    vm_write32(ctx->gpr[22] + 0x34, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008537C4;
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008537B0;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x008537A4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[22] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x4);
loc_008537B0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x008537BC; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[22] + 0x8, ctx->gpr[31]);
loc_008537C4:
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->lr = 0x008537CC; june_evt_func_00853138(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[18] | ctx->gpr[18];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00853494;
loc_008537D8:
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x8);
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[23] + 0x28, ctx->gpr[31]);
    vm_write32(ctx->gpr[23] + 0x30, ctx->gpr[31]);
    vm_write32(ctx->gpr[23] + 0x34, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853830;
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085381C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5E7);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x680C);
    ctx->lr = 0x00853810; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[23] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x4);
loc_0085381C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x00853828; june_evt_func_0088A58C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[23] + 0x8, ctx->gpr[31]);
loc_00853830:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[18] = _cs_18;
    ctx->gpr[17] = _cs_17;
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
    { g_trampoline_fn = (void(*)(void*))june_evt_func_00853138; return; }
}

void june_evt_func_0083F0D4(ppu_context* ctx) {
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
    ctx->gpr[12] = ctx->cr;
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + -0x1E0, ctx->gpr[1]); ctx->gpr[1] += -0x1E0;
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x178, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x188, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x198, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x1A0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x1A8, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x1C8, tmp); }
    ctx->gpr[29] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1D0, tmp); }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1D8, tmp); }
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[24]);
    ctx->gpr[20] = ctx->gpr[8] | ctx->gpr[8];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[18] = ctx->gpr[7] | ctx->gpr[7];
    vm_write32(ctx->gpr[1] + 0x1E8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0083FDDC;
    ctx->gpr[19] = vm_read32(ctx->gpr[2] + -0x6D5C);
loc_0083F174:
    ctx->lr = 0x0083F178; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x48);
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = (int64_t)(int32_t)(0x73);
    ctx->lr = 0x0083F198; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0083FDF8;
loc_0083F1A8:
    ctx->gpr[15] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x48);
    ctx->gpr[3] = ctx->gpr[15] | ctx->gpr[15];
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->lr = 0x0083F1BC; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x78);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[29]);
    ctx->lr = 0x0083F1E0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[24] + 0x10, ctx->gpr[28]);
    vm_write32(ctx->gpr[24] + 0xC, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0083F1F4; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[3]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0083F210;
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(-1);
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[0]);
loc_0083F210:
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0083F220; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0083FAA8;
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 16)) | (cr_val << 16); }
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_0083FC0C;
    ctx->gpr[3] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D40);
    ctx->gpr[4] = (int64_t)(int32_t)(0x8D);
    ctx->lr = 0x0083F240; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0083F244:
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x100);
    ctx->fpr[13] = ctx->fpr[30];
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[20], 0, 31, 31, 0, 0);
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x6040 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6D3C);
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(-1);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = ctx->gpr[8] & ctx->gpr[0];
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[26];
    ctx->gpr[8] = ctx->gpr[8] + (int64_t)(int32_t)((uint32_t)0x2080 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 0, 0, 9, 0, 0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x88);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x8C);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x82);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0x100);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[6] = vm_read32(ctx->gpr[1] + 0x100);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->lr = 0x0083F2BC; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6D38);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->fpr[27] = ctx->fpr[11];
    vm_write32(ctx->gpr[24] + 0x0, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0xBC, ctx->gpr[0]);
    vm_write16(ctx->gpr[24] + 0x38, ctx->gpr[20]);
    vm_write32(ctx->gpr[31] + 0x88, ctx->gpr[24]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    vm_write8(ctx->gpr[1] + 0x82, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x83, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x84, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x85, ctx->gpr[21]);
    ctx->lr = 0x0083F320; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = (int64_t)(int32_t)(0xC);
    ctx->lr = 0x0083F338; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F34C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F35C; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x200);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->lr = 0x0083F370; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F3BC; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[17] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA4);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->lr = 0x0083F3EC; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F404; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F414; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x201);
    ctx->lr = 0x0083F424; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[24] | ctx->gpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    vm_write32(ctx->gpr[24] + 0x8, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x24, tmp); } ctx->gpr[9] += 0x24;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    vm_write8(ctx->gpr[1] + 0x82, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x83, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x84, ctx->gpr[21]);
    vm_write8(ctx->gpr[1] + 0x85, ctx->gpr[21]);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D0C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[28], ctx->fpr[28], 0, 0));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F4A8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = (int64_t)(int32_t)(0xC);
    ctx->lr = 0x0083F4C0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F4D8; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F4E8; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x200);
    ctx->lr = 0x0083F4F8; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[28]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0083F91C;
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_0083FC18;
    ctx->gpr[3] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D40);
    ctx->gpr[4] = (int64_t)(int32_t)(0x153);
    ctx->lr = 0x0083F518; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0083FD64;
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_0083FDD0;
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
loc_0083F52C:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CE4);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[29] = ctx->fpr[10];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x0083F568; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F57C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F58C; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0083F59C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CE0);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F5CC; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F5E0; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F5F0; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x7F);
    ctx->lr = 0x0083F600; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CD8);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F644; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F658; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F668; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->lr = 0x0083F678; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CD4);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F6B8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F6CC; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F6DC; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0083F6EC; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CD0);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F71C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F730; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0083F740; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->lr = 0x0083F750; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CCC);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F790; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F7A4; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F7B4; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->lr = 0x0083F7C4; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0083FD70;
loc_0083F7E0:
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CC4);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[10], ctx->fpr[31], ctx->fpr[12], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[10]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083F828; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F83C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F84C; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0xD);
    ctx->lr = 0x0083F85C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_0083FAB8;
loc_0083F864:
    ctx->gpr[5] = (int64_t)(int32_t)(-1);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6CAC);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6D60);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 36);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[15]);
    ctx->lr = 0x0083F898; june_evt_func_00887480(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x1E8);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1F0);
    vm_write32(ctx->gpr[24] + 0x34, ctx->gpr[3]);
    ctx->gpr[15] = _cs_15;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[16] = _cs_16;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[17] = _cs_17;
    ctx->cr = (ctx->cr & 0xFFF0FFFFu) | ((uint32_t)ctx->gpr[12] & 0x000F0000u);
    ctx->gpr[18] = _cs_18;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1D8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1E0);
    return;
loc_0083F91C:
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    vm_write8(ctx->gpr[1] + 0x85, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x82, ctx->gpr[21]);
    vm_write8(ctx->gpr[1] + 0x83, ctx->gpr[21]);
    vm_write8(ctx->gpr[1] + 0x84, ctx->gpr[18]);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x0083F954; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0x12);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D00); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[31], 0, 0));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[31], 0, 0));
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[11] = vm_read32(ctx->gpr[17] + 0x0);
    ctx->gpr[20] = vm_read32(ctx->gpr[2] + -0x6CFC);
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write16(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0xC0, ctx->gpr[18]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    vm_write8(ctx->gpr[1] + 0x81, ctx->gpr[21]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
loc_0083F9AC:
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ctx->gpr[19] * (uint64_t)(int64_t)(0x33);
    ctx->gpr[27] = ctx->gpr[0] + ctx->gpr[20];
    goto loc_0083FA2C;
loc_0083F9C4:
    vm_write8(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[30]);
    ctx->lr = 0x0083F9D8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F9EC; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083F9FC; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 56);
    ctx->lr = 0x0083FA0C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0x32; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ctx->gpr[26] + (int64_t)(1);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0083FA84;
loc_0083FA2C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    ctx->gpr[9] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[28] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0xD; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0083F9C4;
    ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0x32; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[25], 2, 0, 29, 0, 0);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[26] = ctx->gpr[26] + (int64_t)(1);
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0083FA2C;
loc_0083FA84:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = ctx->gpr[19] + (int64_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { int64_t a = (int64_t)ctx->gpr[19]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0083F9AC;
loc_0083FA9C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CF0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    goto loc_0083F52C;
loc_0083FAA8:
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 16)) | (cr_val << 16); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_0083F244;
loc_0083FAB8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CBC);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083FAF0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FB04; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FB14; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->lr = 0x0083FB24; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CF4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CB4);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->lr = 0x0083FB68; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FB7C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FB8C; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0xC);
    ctx->lr = 0x0083FB9C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CB0);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x0083FBD0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FBE4; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FBF4; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0xE);
    ctx->lr = 0x0083FC04; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0083F864;
loc_0083FC0C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D48); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_0083F244;
loc_0083FC18:
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    vm_write8(ctx->gpr[1] + 0x85, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x84, ctx->gpr[30]);
    vm_write8(ctx->gpr[1] + 0x82, ctx->gpr[21]);
    vm_write8(ctx->gpr[1] + 0x83, ctx->gpr[21]);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->lr = 0x0083FC48; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[20] = vm_read32(ctx->gpr[2] + -0x6CFC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC8);
    ctx->gpr[11] = vm_read32(ctx->gpr[17] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(0x12);
    ctx->gpr[25] = vm_read32(ctx->gpr[2] + -0x6CF8);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[30] = ctx->gpr[20] + (int64_t)(0x78);
    ctx->gpr[27] = ctx->gpr[20] + (int64_t)(0x68);
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write16(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[1] + 0x81, ctx->gpr[21]);
    vm_write16(ctx->gpr[1] + 0xC0, ctx->gpr[9]);
    goto loc_0083FD04;
loc_0083FC8C:
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0083FD54;
    vm_write8(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[26]);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
loc_0083FC9C:
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    ctx->lr = 0x0083FCB4; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FCC8; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FCD8; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 56);
    ctx->lr = 0x0083FCE8; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0083FD3C;
loc_0083FD04:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read8(ctx->gpr[9] + 0x0);
    ctx->gpr[28] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0xD; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0xA; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0083FC8C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0083FD04;
loc_0083FD3C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
loc_0083FD48:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CEC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    goto loc_0083F52C;
loc_0083FD54:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6CF4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[25]);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    goto loc_0083FC9C;
loc_0083FD64:
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    goto loc_0083FA9C;
loc_0083FD70:
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6CC8);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->lr = 0x0083FD8C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FDA0; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0083FDB0; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x0083FDC0; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_0083F7E0;
loc_0083FDD0:
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6D08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    goto loc_0083FD48;
loc_0083FDDC:
    ctx->gpr[19] = vm_read32(ctx->gpr[2] + -0x6D5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x6F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D58);
    ctx->gpr[3] = ctx->gpr[19] | ctx->gpr[19];
    ctx->lr = 0x0083FDF0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0083F174;
loc_0083FDF8:
    ctx->gpr[3] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6D54);
    ctx->gpr[4] = (int64_t)(int32_t)(0x74);
    ctx->lr = 0x0083FE08; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0083F1A8;
}

void june_evt_func_0085A250(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x28);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[10] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A3A4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A3AC;
    ;
loc_0085A290:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A290;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0085A2A8:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[10]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[10], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A30C;
loc_0085A2C8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085A2D8; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A2C8;
loc_0085A2E4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A30C;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ;
loc_0085A2F8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[11]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A2F8;
loc_0085A30C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A324;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085A320; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085A324:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A350;
loc_0085A330:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->lr = 0x0085A340; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A330;
loc_0085A350:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A37C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085A378; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_0085A37C:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0085A384; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_0085A3A4:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    goto loc_0085A2E4;
loc_0085A3AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0085A2A8;
}

void june_evt_func_00843138(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x160, ctx->gpr[1]); ctx->gpr[1] += -0x160;
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[30]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[31]);
    ctx->gpr[20] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[26]);
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[19] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[27] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[1] + 0x168, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->lr = 0x008431A0; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x6BE0);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0xC0);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = (int64_t)(int32_t)(0xA7);
    ctx->lr = 0x008431C4; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00843678;
loc_008431D4:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0xC0);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x008431E4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[30]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008435C0;
    ctx->gpr[8] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[10] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[11] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; } ctx->gpr[8] += 0x90;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x50, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x54, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x58, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; } ctx->gpr[10] += 0xA0;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x5C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x60, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x64, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x68, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; } ctx->gpr[11] += 0xB0;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x6C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x70, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x74, tmp); }
    { uint64_t ea = (ctx->gpr[8]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = ((ctx->gpr[23] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x20);
    { uint64_t ea = (ctx->gpr[10]) & ~0xFULL; vm_raw_ld16(&ctx->vr[1], ea); }
    { uint64_t ea = ((ctx->gpr[23] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[1]); }
    ctx->gpr[9] = (int64_t)(int32_t)(0x30);
    { uint64_t ea = (ctx->gpr[11]) & ~0xFULL; vm_raw_ld16(&ctx->vr[0], ea); }
    { uint64_t ea = ((ctx->gpr[23] + ctx->gpr[9])) & ~0xFULL; vm_raw_st16(ea, &ctx->vr[0]); }
loc_00843274:
    ctx->gpr[29] = ppc_rlw64(ctx->gpr[27], 0, 16, 31, 0, 0);
    ctx->gpr[0] = (int64_t)(int32_t)(0x6C);
    ctx->gpr[28] = ppc_rlw64(ctx->gpr[29], 0, 31, 31, 0, 0);
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = (uint64_t)0 - ctx->gpr[28];
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ctx->gpr[9] & ctx->gpr[0];
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0xE8);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6BB0);
    ctx->gpr[5] = (int64_t)(int32_t)(0xF2);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x80 << 16);
    vm_write8(ctx->gpr[31] + 0x78, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x7C, ctx->gpr[27]);
    ctx->lr = 0x008432C0; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008432E0;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6BAC);
    vm_write32(ctx->gpr[3] + 0xBC, ctx->gpr[0]);
loc_008432E0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 0, 29, 29, 0, 0);
    vm_write32(ctx->gpr[30] + 0x88, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[21], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008433F4;
    vm_write8(ctx->gpr[31] + 0x4C, ctx->gpr[27]);
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00843468;
loc_008432FC:
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x40, ctx->gpr[21]);
    vm_write32(ctx->gpr[31] + 0x80, ctx->gpr[24]);
    vm_write32(ctx->gpr[31] + 0x84, ctx->gpr[19]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00843328;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(0xB0);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x00843324; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00843328:
    ctx->gpr[29] = ctx->gpr[31] + (int64_t)(0x8C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = (int64_t)(int32_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x6B78);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 36);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6B84);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6B80);
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x6B7C);
    ctx->gpr[6] = ctx->gpr[5] | ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x14, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x1C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[28]);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[28]);
    vm_write32(ctx->gpr[29] + 0x20, ctx->gpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
    ctx->lr = 0x00843398; june_evt_func_00887480(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x168);
    vm_write32(ctx->gpr[31] + 0x88, ctx->gpr[3]);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_008433F4:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x3C);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x00843428; june_evt_func_0086E170(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008435B8;
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6BA0);
loc_00843438:
    ctx->gpr[7] = ctx->gpr[31] + (int64_t)(0x50);
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->lr = 0x00843454; june_evt_func_00869310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[31] + 0x4C, ctx->gpr[0]);
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_008432FC;
    ;
loc_00843468:
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[26]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x94);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    vm_write8(ctx->gpr[1] + 0x83, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x81, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x82, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6BA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x008434D4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6B90);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[11]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[10]);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[1] + 0x94, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x98, ctx->gpr[9]);
    ctx->lr = 0x00843514; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0084352C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->lr = 0x0084353C; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x1C);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6B88);
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6B8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write32(ctx->gpr[31] + 0x44, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00843580; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00843598; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008435A8; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
    goto loc_008432FC;
loc_008435B8:
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6B9C);
    goto loc_00843438;
loc_008435C0:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00843690;
loc_008435C8:
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x50, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x54, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x58, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x5C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x60, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x64, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x68, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x6C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x70, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x74, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); } ctx->gpr[9] += 0x10;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x20, tmp); } ctx->gpr[9] += 0x20;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x30, tmp); } ctx->gpr[9] += 0x30;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    goto loc_00843274;
loc_00843678:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BDC);
    ctx->gpr[4] = (int64_t)(int32_t)(0xA8);
    ctx->lr = 0x00843688; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008431D4;
loc_00843690:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6BB4);
    ctx->gpr[4] = (int64_t)(int32_t)(0xDC);
    ctx->lr = 0x008436A0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008435C8;
}

void june_evt_func_00869310(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x5D0, ctx->gpr[1]); ctx->gpr[1] += -0x5D0;
    vm_write64(ctx->gpr[1] + 0x528, ctx->gpr[17]);
    ctx->gpr[17] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x5B0, tmp); }
    vm_write64(ctx->gpr[1] + 0x510, ctx->gpr[14]);
    ctx->gpr[14] = ctx->gpr[1] + (int64_t)(0x78);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x5B8, tmp); }
    vm_write64(ctx->gpr[1] + 0x520, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x530, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x540, ctx->gpr[20]);
    ctx->gpr[18] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x5A0, tmp); }
    vm_write64(ctx->gpr[1] + 0x550, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x588, ctx->gpr[29]);
    ctx->gpr[22] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x5A8, tmp); }
    vm_write64(ctx->gpr[1] + 0x590, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x598, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x5C0, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x5C8, tmp); }
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x518, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x538, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x548, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x558, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x560, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x568, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x570, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x578, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x580, ctx->gpr[28]);
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x8C);
    ctx->gpr[30] = ctx->gpr[31] | ctx->gpr[31];
    vm_write32(ctx->gpr[1] + 0x5D8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x5E0, ctx->gpr[0]);
    ctx->lr = 0x008693A8; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    vm_write64(ctx->gpr[1] + 0x4E8, ctx->gpr[3]);
    ctx->gpr[4] = (int64_t)(int32_t)(8);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->gpr[3] = ctx->gpr[14] | ctx->gpr[14];
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x008693E4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x008693F4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x138);
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    ctx->lr = 0x00869404; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x114);
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    ctx->lr = 0x00869414; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x15C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x28);
    ctx->lr = 0x00869424; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x00869434; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->lr = 0x00869444; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    ctx->lr = 0x00869454; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x21C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x58);
    ctx->lr = 0x00869464; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x274);
    ctx->gpr[4] = (int64_t)(int32_t)(0x68);
    ctx->lr = 0x00869474; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xBC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x14);
    ctx->lr = 0x00869484; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = (int64_t)(int32_t)(0xC);
    ctx->lr = 0x00869494; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xD0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x008694A4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x1DC);
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->lr = 0x008694B4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x4E8);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x1B0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2C);
    ctx->gpr[29] = ctx->gpr[0] | ctx->gpr[0];
    ctx->lr = 0x008694CC; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x184);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2C);
    ctx->lr = 0x008694DC; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x94);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x144);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x1E8);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x18C, tmp); }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x190, tmp); }
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x194, tmp); }
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[10]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1C0);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x198, tmp); }
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[11] = (int64_t)(int32_t)(0x10);
    vm_write32(ctx->gpr[8] + 0x0, ctx->gpr[10]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write8(ctx->gpr[1] + 0x19E, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x19F, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x19C, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x19D, ctx->gpr[11]);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(7);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write16(ctx->gpr[1] + 0x184, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x188, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869624;
    ctx->fpr[30] = ctx->fpr[0];
    ctx->gpr[24] = ctx->gpr[31] + (int64_t)(0x38);
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = ctx->gpr[24] | ctx->gpr[24];
    goto loc_008695E0;
loc_00869580:
    ctx->gpr[0] = (ctx->gpr[11] & 64) ? 0 : (ctx->gpr[28] << (ctx->gpr[11] & 63));
    ctx->gpr[9] = (int64_t)(int32_t)(0x5138);
    ctx->gpr[0] = ctx->gpr[0] & ctx->gpr[9];
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008695E8;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008695C0;
    ctx->lr = 0x008695A8; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008695B8; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ;
loc_008695C0:
    { double a = ctx->fpr[1]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008695CC;
    ctx->fpr[30] = ctx->fpr[1];
loc_008695CC:
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[31] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x38);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008696A0;
loc_008695E0:
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0xE; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00869580;
loc_008695E8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    goto loc_008695C0;
loc_008695F0:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086960C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x50D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62F8);
    ctx->lr = 0x00869608; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086960C:
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_00869624;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62F4);
    ctx->gpr[4] = (int64_t)(int32_t)(0x50E);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->lr = 0x00869620; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869624:
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x5D8);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x5E0);
    ctx->gpr[14] = _cs_14;
    ctx->cr = (ctx->cr & 0xFFF0FFFFu) | ((uint32_t)ctx->gpr[12] & 0x000F0000u);
    ctx->gpr[15] = _cs_15;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[16] = _cs_16;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x5A0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x5A8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x5B0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x5B8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x5C0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x5C8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x5D0);
    return;
loc_008696A0:
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x0);
    { uint32_t field = (ctx->cr >> 0) & 0xF; ctx->cr = (ctx->cr & ~(0xFu << 16)) | (field << 16); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ctx->gpr[30] + (int64_t)(0x28);
    ctx->gpr[26] = ctx->gpr[30] + (int64_t)(0x18);
    ctx->gpr[25] = ctx->gpr[30] + (int64_t)(0x30);
    vm_write32(ctx->gpr[1] + 0x4E4, ctx->gpr[0]);
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->gpr[15] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869624;
    ;
loc_008696D0:
    { uint64_t a = (uint32_t)ctx->gpr[10]; uint64_t b = (uint64_t)0x20; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00869778;
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x63B4);
    ctx->gpr[9] = ppc_rldic(ctx->gpr[10], 2, 30);
    ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->ctr = ctx->gpr[0];
    switch ((uint32_t)ctx->ctr) { case 0x00869778u: goto loc_00869778;case 0x008697ACu: goto loc_008697AC;case 0x00869870u: goto loc_00869870;case 0x008698ECu: goto loc_008698EC;case 0x00869AA8u: goto loc_00869AA8;case 0x00869BD0u: goto loc_00869BD0;case 0x00869CF0u: goto loc_00869CF0;case 0x00869DE8u: goto loc_00869DE8;case 0x00869E2Cu: goto loc_00869E2C;case 0x00869FA0u: goto loc_00869FA0;case 0x00869FB8u: goto loc_00869FB8;case 0x0086A10Cu: goto loc_0086A10C;case 0x0086A2FCu: goto loc_0086A2FC;case 0x0086A55Cu: goto loc_0086A55C;case 0x0086A5BCu: goto loc_0086A5BC;case 0x0086A748u: goto loc_0086A748;case 0x0086A884u: goto loc_0086A884;case 0x0086A920u: goto loc_0086A920;case 0x0086A92Cu: goto loc_0086A92C;case 0x0086AA84u: goto loc_0086AA84;case 0x0086ABCCu: goto loc_0086ABCC;case 0x0086ABE8u: goto loc_0086ABE8;case 0x0086AC8Cu: goto loc_0086AC8C; default: ps3_indirect_call(ctx); return; } return;
loc_00869778:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_00869780:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[13]));
    ctx->gpr[10] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(0x38);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ctx->gpr[26] + (int64_t)(0x38);
    ctx->gpr[25] = ctx->gpr[25] + (int64_t)(0x38);
    ctx->gpr[30] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008695F0;
    ctx->gpr[24] = ctx->gpr[24] + (int64_t)(0x38);
    goto loc_008696D0;
loc_008697AC:
    ctx->lr = 0x008697B0; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x008697C0; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[1]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x140, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write16(ctx->gpr[1] + 0x138, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B040;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[22];
    vm_write32(ctx->gpr[1] + 0x154, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x150, ctx->gpr[0]);
loc_00869800:
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x138);
    ctx->gpr[3] = (int64_t)(int32_t)(7);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->lr = 0x00869818; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869834; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x00869844; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086AC68;
loc_0086984C:
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x00869858; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086985C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_00869860:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_00869870:
    ctx->gpr[0] = ctx->gpr[10] + (int64_t)(-12);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x14; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0086AE00;
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6348);
    ctx->gpr[9] = ppc_rldic(ctx->gpr[0], 2, 30);
    ctx->gpr[0] = vm_read32((ctx->gpr[9] + ctx->gpr[11]));
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->ctr = ctx->gpr[0];
    ps3_indirect_call(ctx); return;
loc_008698EC:
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x7FFF << 16);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x63A4);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = ctx->gpr[11] | 0xFFFF;
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x1B4, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x1CC, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x1C8, ctx->gpr[11]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x2E0);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[5] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->lr = 0x0086992C; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B880;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(4);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0x1D8, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x1D4, ctx->gpr[0]);
loc_0086994C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x1B0);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[10]);
    vm_write16(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[11]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008699F0;
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[10] | ctx->gpr[10];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    ctx->lr = 0x008699AC; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    ctx->lr = 0x008699D0; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B65C;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x008699E4; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x1B0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_008699F0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[0] | 0x4;
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[0] = (int64_t)(int32_t)(0x18);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x1B0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    vm_write32(ctx->gpr[1] + 0x1D0, ctx->gpr[0]);
    ctx->gpr[3] = (int64_t)(int32_t)(0x11);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[1] + 0x1B0, ctx->gpr[9]);
    vm_write16(ctx->gpr[1] + 0x1B8, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x1BC, ctx->gpr[23]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x00869A3C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869A58; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x00869A68; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B094;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x00869A7C; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869A80:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x1B0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write16(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    goto loc_00869780;
loc_00869AA8:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B878;
loc_00869AB4:
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->lr = 0x00869ABC; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x90);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00869ACC; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869AEC;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x8C);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write16(ctx->gpr[1] + 0x8C, ctx->gpr[0]);
loc_00869AEC:
    ctx->gpr[27] = (int64_t)(int8_t)ctx->gpr[15];
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00869B10;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
loc_00869B10:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->lr = 0x00869B48; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00869B64; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B070;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00869B78; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869B7C:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x00869B88; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869BA8;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x8C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write16(ctx->gpr[1] + 0x8C, ctx->gpr[0]);
loc_00869BA8:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869BB4;
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
loc_00869BB4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086AF50;
    ctx->gpr[15] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6308); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_00869BD0:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B19C;
    ctx->lr = 0x00869BE0; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x00869BF8; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x00869C3C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869C58; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B688;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x00869C6C; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869C70:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_00869C7C:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x184);
    ctx->gpr[3] = (int64_t)(int32_t)(4);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0x188, ctx->gpr[0]);
    ctx->lr = 0x00869CA8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869CC4; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x00869CD4; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086ABA8;
loc_00869CDC:
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x00869CE8; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00869860;
loc_00869CF0:
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B20C;
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[1] + 0x1DC, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x1E0, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x1E4, ctx->gpr[11]);
    ctx->lr = 0x00869D14; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00869D24; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
loc_00869D30:
    ctx->gpr[27] = (int64_t)(int8_t)ctx->gpr[15];
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00869D54;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
loc_00869D54:
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x1DC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(2);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->lr = 0x00869D78; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869D94; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B1E8;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869DA8; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869DAC:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x00869DB8; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869DC8;
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
loc_00869DC8:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086AF7C;
    ctx->gpr[15] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_00869DE8:
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_0086BB88;
loc_00869DEC:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B0F8;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x4E4);
    ctx->lr = 0x00869E04; june_evt_func_00854B70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[19] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x638C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    goto loc_00869780;
loc_00869E2C:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6324);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[1] + 0xD4, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0xD2, ctx->gpr[9]);
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B868;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    vm_write32(ctx->gpr[1] + 0xEC, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xE8, ctx->gpr[9]);
loc_00869E6C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[9]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[10]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869F28;
    ctx->lr = 0x00869E94; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x00869EAC; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x00869EE8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00869F04; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B7B4;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x00869F18; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869F1C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_00869F28:
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6320); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(0x13);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->lr = 0x00869F4C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00869F68; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B13C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00869F7C; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00869F80:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x00869F8C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x631C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    goto loc_00869780;
loc_00869FA0:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_00869FAC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    goto loc_00869780;
loc_00869FB8:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xF4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x104, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x108, tmp); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B88C;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[1] + 0x10C, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xFC, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x100, tmp); }
loc_0086A008:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B8C0;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[9]);
    ctx->lr = 0x0086A050; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    ctx->lr = 0x0086A07C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B3CC;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0086A094; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6364); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(0x14);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x0086A0CC; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A0E0; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A0E4:
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A0F0; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A0F4:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086A100; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6360); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00869FAC;
loc_0086A10C:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B8A0;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086AFA8;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 8, 24, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 24, 8, 31, 0, 0);
    vm_write8(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xA1, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0xA2, ctx->gpr[11]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BB2C;
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write8(ctx->gpr[1] + 0xA3, ctx->gpr[0]);
loc_0086A160:
    ctx->gpr[0] = (int64_t)(int32_t)(4);
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[0]);
loc_0086A168:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0xA4, ctx->gpr[0]);
loc_0086A174:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086AF94;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x9C);
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[9]);
loc_0086A190:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B37C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
loc_0086A1A4:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B050;
    ctx->lr = 0x0086A1B4; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x0086A1CC; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086A210; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A22C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B790;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A240; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A244:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6374); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_0086A254:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6370); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)6; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B2F0;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6368); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0086A26C:
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[3] = (int64_t)(int32_t)(0xA);
    ctx->lr = 0x0086A284; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A2A0; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086A2B0; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B218;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A2C4; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A2C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869860;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6374); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    goto loc_00869780;
loc_0086A2FC:
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B2F8;
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B7D8;
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BAC0;
loc_0086A318:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086AF18;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086BB70;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(4);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[22];
    vm_write32(ctx->gpr[1] + 0x134, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x130, ctx->gpr[0]);
loc_0086A340:
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x134);
    vm_write32(ctx->gpr[1] + 0x180, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x130);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    vm_write32(ctx->gpr[1] + 0x17C, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B638;
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x114);
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x15C);
    ctx->gpr[9] = ctx->gpr[9] | 0x2;
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[1] + 0x114, ctx->gpr[9]);
    vm_write16(ctx->gpr[1] + 0x15C, ctx->gpr[0]);
loc_0086A378:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[9]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[10]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086A408;
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->lr = 0x0086A3C8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    ctx->lr = 0x0086A3EC; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B6AC;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A400; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0086A408:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086B260;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086B23C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6394); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
loc_0086A438:
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x114);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x638C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(6);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[9] = (int64_t)(int32_t)(0xE);
    ctx->gpr[3] = (int64_t)(int32_t)(9);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x128, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x12C, ctx->gpr[9]);
    ctx->lr = 0x0086A464; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A480; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B600;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A494; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A498:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086B248;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x15C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6384); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    vm_write16(ctx->gpr[1] + 0x15C, ctx->gpr[0]);
loc_0086A4DC:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x15C);
    ctx->gpr[3] = (int64_t)(int32_t)(0xB);
    vm_write32(ctx->gpr[1] + 0x178, ctx->gpr[0]);
    ctx->lr = 0x0086A4F8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A514; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086A524; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B5DC;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A538; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A53C:
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x12C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x4F8, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x4F8); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    goto loc_00869FAC;
loc_0086A55C:
    ctx->lr = 0x0086A560; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0086A570; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[1]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x140, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write16(ctx->gpr[1] + 0x138, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B040;
    ctx->gpr[9] = (int64_t)(int32_t)(2);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    vm_write32(ctx->gpr[1] + 0x154, ctx->gpr[9]);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[22];
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write32(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x158, ctx->gpr[9]);
    goto loc_00869800;
loc_0086A5BC:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B934;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x21C);
    ctx->gpr[5] = (int64_t)(int32_t)(0x54);
    ctx->gpr[4] = ctx->gpr[22] + ctx->gpr[4];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0086A5DC; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0x270, ctx->gpr[0]);
loc_0086A5EC:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B1BC;
    ctx->lr = 0x0086A5FC; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x0086A614; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086A658; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A674; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B76C;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A688; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A68C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_0086A698:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x635C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6358); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x21C);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = (int64_t)(int32_t)(0x15);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->lr = 0x0086A6CC; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0086A6E8; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086AA60;
loc_0086A6F0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0086A6FC; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A700:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086A70C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00869860;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6374); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    goto loc_00869780;
loc_0086A748:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B184;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->gpr[4] = ctx->gpr[22] + ctx->gpr[4];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0086A768; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
loc_0086A778:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B94C;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[9]);
    ctx->lr = 0x0086A7C0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    ctx->lr = 0x0086A7EC; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B388;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0086A804; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6364); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(0x18);
loc_0086A82C:
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x0086A83C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A850; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A854:
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A860; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A864:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086A870; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6360); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    goto loc_00869780;
loc_0086A884:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B1DC;
    ctx->lr = 0x0086A894; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->lr = 0x0086A8A8; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->fpr[26] = ctx->fpr[31];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086A8E0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A8FC; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B740;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A910; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_0086A920:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_0086A92C:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B91C;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x274);
    ctx->gpr[5] = (int64_t)(int32_t)(0x64);
    ctx->gpr[4] = ctx->gpr[22] + ctx->gpr[4];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0086A94C; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    vm_write32(ctx->gpr[1] + 0x2D8, ctx->gpr[0]);
loc_0086A95C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B0D8;
    ctx->lr = 0x0086A96C; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x0086A984; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086A9C8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086A9E4; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B71C;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086A9F8; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086A9FC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_0086AA08:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x635C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6354); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x274);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = (int64_t)(int32_t)(0x16);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->lr = 0x0086AA3C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0086AA58; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_0086A6F0;
loc_0086AA60:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BB20;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A700;
loc_0086AA84:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B0B8;
    ctx->lr = 0x0086AA94; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x0086AAAC; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086AAF0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086AB0C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B6F8;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086AB20; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086AB24:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_0086AB30:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(-128);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(3);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[4] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[1] + 0x7E, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x7F, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x7C, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x7D, ctx->gpr[11]);
    ctx->lr = 0x0086AB74; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086AB90; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086ABA0; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_00869CDC;
loc_0086ABA8:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BB14;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869860;
loc_0086ABCC:
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_0086B9EC;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 16)) | (cr_val << 16); }
    vm_write32(ctx->gpr[1] + 0x4E4, ctx->gpr[0]);
    goto loc_00869780;
loc_0086ABE8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x4E8);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x18);
    ctx->gpr[4] = ctx->gpr[14] | ctx->gpr[14];
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086AC34; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086AC50; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086AC60; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_0086984C;
loc_0086AC68:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B85C;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086985C;
loc_0086AC8C:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6318);
    ctx->gpr[9] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[1] + 0xD4, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0xD2, ctx->gpr[9]);
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B9DC;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    vm_write32(ctx->gpr[1] + 0xEC, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xE8, ctx->gpr[9]);
loc_0086ACCC:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[9]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[10]);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086AD88;
    ctx->lr = 0x0086ACF4; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    ctx->lr = 0x0086AD0C; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[31]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086AD48; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086AD64; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B6D4;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086AD78; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086AD7C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[26]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_0086AD88:
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xD0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6314); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(0x13);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->lr = 0x0086ADAC; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0086ADC8; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B160;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0086ADDC; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086ADE0:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086ADEC; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6310); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    goto loc_00869780;
loc_0086AE00:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x323);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6328);
    ctx->lr = 0x0086AE10; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086AF68;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xC0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
loc_0086AE30:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086B9CC;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    vm_write32(ctx->gpr[1] + 0xCC, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[9]);
loc_0086AE60:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B9A8;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[9]);
    ctx->lr = 0x0086AEA8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[0]);
    ctx->lr = 0x0086AED4; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B464;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0086AEEC; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(0x17);
    goto loc_0086A82C;
loc_0086AF18:
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086BB7C;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)(4);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[22];
    vm_write32(ctx->gpr[1] + 0x1D8, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x1D4, ctx->gpr[0]);
loc_0086AF30:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B624;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x1B0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write16(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    goto loc_0086A378;
loc_0086AF50:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[15] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    goto loc_00869780;
loc_0086AF68:
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xC0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    goto loc_0086AE30;
loc_0086AF7C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[15] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_0086AF94:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x9C);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 30, 1, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[9]);
    goto loc_0086A190;
loc_0086AFA8:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x4E0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x4E0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x4E0);
    vm_write8(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[10], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x4E0);
    vm_write8(ctx->gpr[1] + 0xA1, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[10], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x4E0);
    vm_write8(ctx->gpr[1] + 0xA2, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BB40;
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[9] = (int64_t)(int32_t)(0xC);
    vm_write8(ctx->gpr[1] + 0xA3, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[9]);
    goto loc_0086A168;
loc_0086B040:
    vm_write8(ctx->gpr[1] + 0x14C, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x150, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0x154, ctx->gpr[22]);
    goto loc_00869800;
loc_0086B050:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[9]);
    goto loc_0086A254;
loc_0086B070:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BBF8;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869B7C;
loc_0086B094:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BC04;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869A80;
loc_0086B0B8:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[9]);
    goto loc_0086AB30;
loc_0086B0D8:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[9]);
    goto loc_0086AA08;
loc_0086B0F8:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD60;
loc_0086B100:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[10] = ctx->fpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x638C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6388); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[11]));
    ctx->fpr[29] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[0]));
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    goto loc_00869780;
loc_0086B13C:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BBEC;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869F80;
loc_0086B160:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BBE0;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086ADE0;
loc_0086B184:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6350); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x634C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[22]);
    goto loc_0086A778;
loc_0086B19C:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[9]);
    goto loc_00869C7C;
loc_0086B1BC:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[18], 0, 32);
    vm_write64(ctx->gpr[1] + 0x500, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x508, ctx->gpr[9]);
    goto loc_0086A698;
loc_0086B1DC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63B0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00869780;
loc_0086B1E8:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BBD4;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869DAC;
loc_0086B20C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6398); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    vm_write16(ctx->gpr[1] + 0x1DC, ctx->gpr[11]);
    goto loc_00869D30;
loc_0086B218:
    vm_write32(ctx->gpr[29] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BBC8;
    vm_write32(ctx->gpr[29] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[29]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A2C8;
loc_0086B23C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6390); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0086A438;
loc_0086B248:
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x15C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6380); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    vm_write16(ctx->gpr[1] + 0x15C, ctx->gpr[0]);
    goto loc_0086A4DC;
loc_0086B260:
    ctx->gpr[0] = (int64_t)(int32_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x2E0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x1B0);
    vm_write32(ctx->gpr[1] + 0x1D0, ctx->gpr[0]);
    ctx->gpr[3] = (int64_t)(int32_t)(0x11);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write32(ctx->gpr[1] + 0x1BC, ctx->gpr[9]);
    vm_write16(ctx->gpr[1] + 0x1B8, ctx->gpr[0]);
    ctx->lr = 0x0086B29C; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086B2B8; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086B2C8; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_0086B4FC;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x0086B2DC; june_evt_func_008549D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086B2E0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    goto loc_00869FAC;
loc_0086B2F0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x636C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_0086A26C;
loc_0086B2F8:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086B520;
    vm_write32(ctx->gpr[1] + 0x118, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    vm_write32(ctx->gpr[1] + 0x120, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write32(ctx->gpr[1] + 0x11C, ctx->gpr[0]);
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086BD58;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086BBA0;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0x124, ctx->gpr[0]);
loc_0086B348:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
loc_0086B34C:
    vm_write32(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x124);
    vm_write32(ctx->gpr[1] + 0x168, ctx->gpr[9]);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write32(ctx->gpr[1] + 0x16C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x164, ctx->gpr[11]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x10);
    vm_write32(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x14);
    vm_write32(ctx->gpr[1] + 0x174, ctx->gpr[9]);
    goto loc_0086A318;
loc_0086B37C:
    ctx->gpr[9] = ctx->gpr[9] | 0x1;
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[9]);
    goto loc_0086A1A4;
loc_0086B388:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B574;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
loc_0086B3A4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[3] = (int64_t)(int32_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6364); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0086B4A4;
loc_0086B3CC:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B564;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
loc_0086B3E8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xF0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6364); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x0086B420; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086B438; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086B43C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BC5C;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[30]);
    goto loc_0086A0F4;
loc_0086B464:
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086B56C;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
loc_0086B480:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xBC);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[3] = (int64_t)(int32_t)(0x17);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0086B4A4:
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x0086B4B8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086B4D0; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086B4D4:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BB34;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[30]);
    goto loc_0086A864;
loc_0086B4FC:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD90;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086B2E0;
loc_0086B520:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x63A4);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x2E0);
    vm_write32(ctx->gpr[1] + 0x1B4, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    vm_write32(ctx->gpr[1] + 0x1CC, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write32(ctx->gpr[1] + 0x1C8, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[5] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->lr = 0x0086B558; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    goto loc_0086A318;
loc_0086B564:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    goto loc_0086B3E8;
loc_0086B56C:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    goto loc_0086B480;
loc_0086B574:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    goto loc_0086B3A4;
loc_0086B5DC:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BC84;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086A53C;
loc_0086B600:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BC90;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A498;
loc_0086B624:
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x1B0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write16(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    goto loc_0086A378;
loc_0086B638:
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x114);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x15C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[1] + 0x114, ctx->gpr[9]);
    vm_write16(ctx->gpr[1] + 0x15C, ctx->gpr[0]);
    goto loc_0086A378;
loc_0086B65C:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD1C;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
loc_0086B678:
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x1B0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_008699F0;
loc_0086B688:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD24;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_00869C70;
loc_0086B6AC:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD30;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
loc_0086B6C8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A408;
loc_0086B6D4:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BCC0;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086AD7C;
loc_0086B6F8:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BCB4;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086AB24;
loc_0086B71C:
    vm_write32(ctx->gpr[27] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BC9C;
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[27]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[23]);
    goto loc_0086A9FC;
loc_0086B740:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD38;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
loc_0086B75C:
    ctx->fpr[13] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869780;
loc_0086B76C:
    vm_write32(ctx->gpr[27] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD10;
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[27]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[23]);
    goto loc_0086A68C;
loc_0086B790:
    vm_write32(ctx->gpr[27] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BCA8;
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[27]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[23]);
    goto loc_0086A244;
loc_0086B7B4:
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[19]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD40;
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[30]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_00869F1C;
loc_0086B7D8:
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086BC10;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086BCCC;
loc_0086B7F4:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x118, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x11C, tmp); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086BD78;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x124, tmp); }
loc_0086B828:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x168, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x124); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x16C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x164, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x170, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x174, tmp); }
    goto loc_0086A318;
loc_0086B85C:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086985C;
loc_0086B868:
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0xE8, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0xEC, ctx->gpr[22]);
    goto loc_00869E6C;
loc_0086B878:
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x630C);
    goto loc_00869AB4;
loc_0086B880:
    vm_write32(ctx->gpr[1] + 0x1D4, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0x1D8, ctx->gpr[22]);
    goto loc_0086994C;
loc_0086B88C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x10C, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xFC, ctx->gpr[0]);
    goto loc_0086A008;
loc_0086B8A0:
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write32(ctx->gpr[1] + 0xA4, ctx->gpr[22]);
    vm_write8(ctx->gpr[1] + 0xA3, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xA1, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xA2, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[22]);
    goto loc_0086A174;
loc_0086B8C0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6364); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x0086B8F8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086B910; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_0086A0E4;
    goto loc_0086B43C;
loc_0086B91C:
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x274);
    ctx->gpr[4] = (int64_t)(int32_t)(0x64);
    ctx->lr = 0x0086B928; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[1] + 0x2D8, ctx->gpr[22]);
    goto loc_0086A95C;
loc_0086B934:
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x21C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x54);
    ctx->lr = 0x0086B940; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[1] + 0x270, ctx->gpr[22]);
    goto loc_0086A5EC;
loc_0086B94C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6364); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0086B96C:
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x0086B984; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086B99C; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    if ((((ctx->cr >> 13) & 1) == 0)) goto loc_0086A854;
    goto loc_0086B4D4;
loc_0086B9A8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xBC);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x637C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(0x17);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6378); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0086B96C;
loc_0086B9CC:
    vm_write32(ctx->gpr[1] + 0xC4, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0xC8, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0xCC, ctx->gpr[22]);
    goto loc_0086AE60;
loc_0086B9DC:
    vm_write32(ctx->gpr[1] + 0xD8, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0xE8, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0xEC, ctx->gpr[22]);
    goto loc_0086ACCC;
loc_0086B9EC:
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086BC68;
loc_0086B9FC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    ctx->gpr[0] = (int64_t)(int32_t)(8);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[3] = (int64_t)(int32_t)(0x10);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x0086BA38; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->lr = 0x0086BA60; june_evt_func_0086C428(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x4E4);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0086BA7C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x4E4);
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BD4C;
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x20);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[31]);
loc_0086BAA0:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x62F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x4E4, ctx->gpr[31]);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 16)) | (cr_val << 16); }
    goto loc_00869780;
loc_0086BAC0:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086B520;
    vm_write32(ctx->gpr[1] + 0x118, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    vm_write32(ctx->gpr[1] + 0x120, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write32(ctx->gpr[1] + 0x11C, ctx->gpr[0]);
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_0086BD84;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086BCE4;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0x124, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    goto loc_0086B34C;
loc_0086BB14:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869860;
loc_0086BB20:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A700;
loc_0086BB2C:
    vm_write8(ctx->gpr[1] + 0xA3, ctx->gpr[10]);
    goto loc_0086A160;
loc_0086BB34:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[30]);
    goto loc_0086A864;
loc_0086BB40:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(0x10);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[0]);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[10], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x4E0);
    vm_write8(ctx->gpr[1] + 0xA3, ctx->gpr[0]);
    goto loc_0086A168;
loc_0086BB70:
    vm_write32(ctx->gpr[1] + 0x130, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0x134, ctx->gpr[22]);
    goto loc_0086A340;
loc_0086BB7C:
    vm_write32(ctx->gpr[1] + 0x1D4, ctx->gpr[22]);
    vm_write32(ctx->gpr[1] + 0x1D8, ctx->gpr[22]);
    goto loc_0086AF30;
loc_0086BB88:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4EF);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6300);
    ctx->lr = 0x0086BB98; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00869DEC;
loc_0086BBA0:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x124);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    goto loc_0086B348;
loc_0086BBC8:
    vm_write32(ctx->gpr[29] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A2C8;
loc_0086BBD4:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869DAC;
loc_0086BBE0:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086ADE0;
loc_0086BBEC:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869F80;
loc_0086BBF8:
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869B7C;
loc_0086BC04:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_00869A80;
loc_0086BC10:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x1CC, tmp); }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x2E0);
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x1C8, tmp); }
    vm_write32(ctx->gpr[1] + 0x1B4, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x639C);
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x4F0, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x4F0);
    ctx->lr = 0x0086BC50; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    goto loc_0086A318;
loc_0086BC5C:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[30]);
    goto loc_0086A0F4;
loc_0086BC68:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4DC);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6304);
    ctx->lr = 0x0086BC78; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[17] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    goto loc_0086B9FC;
loc_0086BC84:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086A53C;
loc_0086BC90:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086A498;
loc_0086BC9C:
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[23]);
    goto loc_0086A9FC;
loc_0086BCA8:
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[23]);
    goto loc_0086A244;
loc_0086BCB4:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086AB24;
loc_0086BCC0:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_0086AD7C;
loc_0086BCCC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x119);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x63A0);
    ctx->lr = 0x0086BCDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086B7F4;
loc_0086BCE4:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x63A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[22] + ctx->gpr[9];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x124);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    goto loc_0086B34C;
loc_0086BD10:
    vm_write32(ctx->gpr[27] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[23]);
    goto loc_0086A68C;
loc_0086BD1C:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    goto loc_0086B678;
loc_0086BD24:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_00869C70;
loc_0086BD30:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    goto loc_0086B6C8;
loc_0086BD38:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    goto loc_0086B75C;
loc_0086BD40:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[27]);
    goto loc_00869F1C;
loc_0086BD4C:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x4E4);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[9]);
    goto loc_0086BAA0;
loc_0086BD58:
    vm_write32(ctx->gpr[1] + 0x124, ctx->gpr[22]);
    goto loc_0086B348;
loc_0086BD60:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x63D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4FA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x62FC);
    ctx->lr = 0x0086BD70; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0086B100;
loc_0086BD78:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x124, ctx->gpr[0]);
    goto loc_0086B828;
loc_0086BD84:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x124, ctx->gpr[22]);
    goto loc_0086B34C;
loc_0086BD90:
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[19]);
    vm_write32(ctx->gpr[20] + 0x20, ctx->gpr[29]);
    goto loc_0086B2E0;
}

void june_evt_func_0089B3B8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    { int64_t a = (int64_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0089B530;
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x59D8);
loc_0089B3E4:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089B49C;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 5, 0, 26, 0, 0);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x20);
    ctx->gpr[6] = (int64_t)(int32_t)(0x12C);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[3], 0, 0, 26, 0, 0);
    ctx->lr = 0x0089B414; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089B54C;
loc_0089B428:
    ctx->gpr[5] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[5] = ppc_rlw64(ctx->gpr[5], 5, 0, 26, 0, 0);
    ctx->lr = 0x0089B43C; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->lr = 0x0089B448; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 5, 0, 26, 0, 0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->gpr[31] = ctx->gpr[31] + ctx->gpr[0];
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[29]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x0089B468; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(1);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[31] = _cs_31;
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    return;
loc_0089B49C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0089B518;
loc_0089B4A8:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = (int64_t)(int32_t)(0x20);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = (int64_t)(int32_t)(0x126);
    ctx->lr = 0x0089B4C0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0089B564;
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[29]);
loc_0089B4D8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x0089B4E4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[0] = ctx->gpr[3] + (int64_t)(1);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[31] = _cs_31;
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    return;
loc_0089B518:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5964);
    ctx->gpr[4] = (int64_t)(int32_t)(0x124);
    ctx->lr = 0x0089B528; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0089B4A8;
loc_0089B530:
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x59D8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x120);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5988);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0089B544; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0089B3E4;
loc_0089B54C:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5960);
    ctx->gpr[4] = (int64_t)(int32_t)(0x12D);
    ctx->lr = 0x0089B55C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0089B428;
loc_0089B564:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5960);
    ctx->gpr[4] = (int64_t)(int32_t)(0x127);
    ctx->lr = 0x0089B574; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[30] + 0xC, ctx->gpr[29]);
    goto loc_0089B4D8;
}
