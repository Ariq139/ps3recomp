
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

void june_evt_func_008504F8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1F0, ctx->gpr[1]); ctx->gpr[1] += -0x1F0;
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x178, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x180, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x188, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x198, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x1A0, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x1C8, tmp); }
    ctx->gpr[25] = ctx->gpr[7] | ctx->gpr[7];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x1D0, tmp); }
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x1D8, tmp); }
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1E0, tmp); }
    ctx->gpr[24] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1E8, tmp); }
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[23]);
    vm_write32(ctx->gpr[1] + 0x1F8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[0]);
    ctx->gpr[8] = vm_read32(ctx->gpr[4] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00850EE0;
loc_008505A4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850AD0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850A70;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    ctx->lr = 0x008505D8; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00850A30;
loc_008505F4:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0xD8);
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x0085061C; june_evt_func_0085A7E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850A0C;
loc_0085062C:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008509AC;
loc_00850648:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008509D0;
loc_00850668:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008509F8;
loc_0085068C:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008506C0;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ;
loc_008506A0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008506B0; june_evt_func_0084FC44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008506A0;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x1014);
loc_008506C0:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850F44;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085141C;
loc_008506DC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850D9C;
loc_008506EC:
    ctx->lr = 0x008506F0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850720;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6848);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_00850720:
    ctx->lr = 0x00850724; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00850730; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085073C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085076C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6844);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[25] = ppu_frsp(ctx->fpr[0]);
loc_0085076C:
    ctx->lr = 0x00850770; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[16] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085077C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    { double a = ctx->fpr[10]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[13]));
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x102C);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0xC);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[1]));
    ctx->lr = 0x00850814; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[27] = vm_read32(ctx->gpr[28] + 0x1030);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[18] = vm_read32(ctx->gpr[1] + 0xF0);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008516E4;
loc_00850834:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850F4C;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6838); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6858); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { int32_t _a=(int32_t)ctx->gpr[18], _b=(int32_t)ctx->gpr[27]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008517DC;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6854); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008514B4;
loc_00850898:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008508D4;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008508D4;
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x84);
loc_008508B8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008508C8; june_evt_func_0084F79C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008508B8;
loc_008508D4:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x1020); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008514A4;
loc_008508E4:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x1020, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x1024, tmp); }
loc_008508F0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008516EC;
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008516EC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008514DC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[14] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x1018);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
loc_0085093C:
    { double a = ctx->fpr[12]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008516FC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x101C);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write8(ctx->gpr[1] + 0xF4, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
loc_00850970:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    vm_write32(ctx->gpr[11] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x14, tmp); }
    goto loc_00850F54;
loc_008509AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850668;
loc_008509D0:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 29, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085068C;
loc_008509F8:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085068C;
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[10]);
    goto loc_0085068C;
loc_00850A0C:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008509AC;
    goto loc_00850648;
loc_00850A30:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[19] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0xD8);
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x00850A5C; june_evt_func_0085A7E0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085062C;
    goto loc_00850A0C;
loc_00850A70:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->lr = 0x00850A80; june_evt_func_00889EB4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 2, 0, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[8]);
    ctx->lr = 0x00850AB0; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008505F4;
    goto loc_00850A30;
loc_00850AD0:
    ctx->lr = 0x00850AD4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6840);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->lr = 0x00850AFC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850B2C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6848);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_00850B2C:
    ctx->lr = 0x00850B30; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00850B3C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00850B48; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850B78;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6844);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00850B78:
    ctx->lr = 0x00850B7C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00850B88; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    ctx->lr = 0x00850B9C; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0xC8);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB8);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xA8);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write32(ctx->gpr[1] + 0xAC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    ctx->lr = 0x00850BF4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[26];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00850D10;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00850D10;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00850D10;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00850D10;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[11], 29, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 3, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[8]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->lr = 0x00850CC4; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D10;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00850D04; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00850D0C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00850D10:
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[24];
loc_00850D14:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x200);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x1F8);
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
loc_00850D9C:
    ctx->lr = 0x00850DA0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850EF0;
loc_00850DB4:
    ctx->lr = 0x00850DB8; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00850DC4; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00850E00; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00850F24;
loc_00850E10:
    ctx->lr = 0x00850E14; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00850E20; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x2C);
    ctx->lr = 0x00850E48; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00850F10;
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008506EC;
loc_00850E68:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008506EC;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[27], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[29]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008506EC;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[11]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008506EC;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x101C);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x34);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[10] + 0x18, tmp); }
    goto loc_008506EC;
loc_00850EE0:
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 0, 1, 31, 0, 0);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[8]);
    goto loc_008505A4;
loc_00850EF0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6848);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00850DB4;
loc_00850F10:
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ctx->fpr[1];
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008506EC;
    goto loc_00850E68;
loc_00850F24:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6844);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[27] = ppu_frsp(ctx->fpr[0]);
    goto loc_00850E10;
loc_00850F44:
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[0]);
    goto loc_008506DC;
loc_00850F4C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6858); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6854); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
loc_00850F54:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[15] = ctx->gpr[1] + (int64_t)(0xC8);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[28]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[26]));
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->gpr[5] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    ctx->lr = 0x00850F84; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850FF0;
    ctx->gpr[5] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850FF0;
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[8] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->lr = 0x00850FE4; june_evt_func_0084DA30(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008514D4;
loc_00850FF0:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[22];
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[30]));
    ctx->fpr[26] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008510CC;
    { uint64_t a = (uint32_t)ctx->gpr[11]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00851430;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write32(ctx->gpr[1] + 0xAC, ctx->gpr[31]);
    ctx->lr = 0x00851050; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851074;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6834);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085181C;
loc_00851074:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851494;
    ctx->lr = 0x00851080; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851494;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(2);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
loc_008510A0:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[23] + 0x4);
    ctx->lr = 0x008510BC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x008510C4; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00851494;
loc_008510CC:
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)5; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851760;
loc_008510D4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[28]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[28]));
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    ctx->gpr[4] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->lr = 0x00851110; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008513AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[28]));
    ctx->gpr[5] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[8] = (int64_t)(int32_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6820); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    vm_write32(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0x90, ctx->gpr[8]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008511AC;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x88);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x008511A8; june_evt_func_0084EB88(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
loc_008511AC:
    ctx->lr = 0x008511B0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008511E0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6848);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_008511E0:
    ctx->lr = 0x008511E4; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008511F0; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085122C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851258;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6844);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_00851258:
    ctx->lr = 0x0085125C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00851268; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008512E4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008512E4;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008512E4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008514FC;
loc_008512E4:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851600;
loc_008512F0:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[16];
    { double a = ctx->fpr[26]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[17];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[25]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00851608;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00851608;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00851608;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00851608;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008513AC;
    ctx->lr = 0x00851364; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851608;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6840);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851608;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851608;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(5);
    vm_write32(ctx->gpr[28] + 0x1034, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0x4, ctx->gpr[9]);
    goto loc_008513B0;
loc_008513AC:
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
loc_008513B0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D14;
loc_008513C0:
    ctx->gpr[9] = vm_read8(ctx->gpr[1] + 0xF4);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085171C;
loc_008513D8:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[14];
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[24];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D14;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->lr = 0x00851400; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[24];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00850D14;
loc_00851414:
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    goto loc_00850D14;
loc_0085141C:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00851428; june_evt_func_00859374(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008506DC;
loc_00851430:
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008510D4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x00851454; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851478;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6834);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851A18;
loc_00851478:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851494;
    ctx->lr = 0x00851484; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851A04;
loc_00851494:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    vm_write32(ctx->gpr[28] + 0x4, ctx->gpr[0]);
    goto loc_008513B0;
loc_008514A4:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x1024); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008508F0;
    goto loc_008508E4;
loc_008514B4:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x14);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[26], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_00850898;
loc_008514D4:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    goto loc_00850FF0;
loc_008514DC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851800;
    ctx->gpr[14] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x1018);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085093C;
loc_008514FC:
    ctx->lr = 0x00851500; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085151C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6840);
    ctx->gpr[11] = vm_read16(ctx->gpr[9] + 0x0);
loc_0085151C:
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x88);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851F78;
    ctx->gpr[31] = ppc_rlw64(ctx->gpr[11], 0, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 0, 29, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851F78;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00851F78;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851554;
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[9]);
loc_00851554:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x1C);
    ctx->lr = 0x00851568; june_evt_func_00859374(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851FF0;
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0x88);
    ctx->gpr[11] = ctx->gpr[26] + (int64_t)(0x10);
    ctx->gpr[6] = vm_read32(ctx->gpr[1] + 0x8C);
    ctx->gpr[8] = vm_read32(ctx->gpr[1] + 0x90);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->ctr = ctx->gpr[0];
    goto loc_008515A8;
loc_0085159C:
    ctx->gpr[7] = ctx->gpr[7] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0x10);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) == 0)) goto loc_00852000;
loc_008515A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085159C;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085159C;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[8]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085159C;
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085159C;
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852000;
loc_008515E0:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008515FC;
loc_008515E8:
    ctx->lr = 0x008515EC; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852504;
loc_008515FC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_00851600:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    goto loc_008512F0;
loc_00851608:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008513AC;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    vm_write32(ctx->gpr[1] + 0xAC, ctx->gpr[9]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851654;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
loc_00851654:
    ctx->gpr[3] = ctx->gpr[15] | ctx->gpr[15];
    ctx->gpr[5] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB8);
    ctx->lr = 0x00851664; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008513AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008513AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008513AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008513AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008513AC;
    ctx->lr = 0x008516B0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008513AC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6840);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008513AC;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    vm_write32(ctx->gpr[28] + 0x4, ctx->gpr[0]);
    goto loc_008513B0;
loc_008516E4:
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    goto loc_00850834;
loc_008516EC:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[14] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0xF4, ctx->gpr[0]);
    goto loc_00850F54;
loc_008516FC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008517E4;
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x101C);
    vm_write8(ctx->gpr[1] + 0xF4, ctx->gpr[9]);
    goto loc_00850970;
loc_0085171C:
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x101C);
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[7] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x00851734; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008513D8;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x101C);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x34);
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    goto loc_008513D8;
loc_00851760:
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write32(ctx->gpr[1] + 0xAC, ctx->gpr[31]);
    ctx->lr = 0x0085177C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008517AC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6834);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008517AC;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851B74;
loc_008517AC:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851494;
    ctx->lr = 0x008517B8; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851494;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    goto loc_008510A0;
loc_008517DC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6854); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    goto loc_008508D4;
loc_008517E4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    vm_write8(ctx->gpr[1] + 0xF4, ctx->gpr[11]);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x101C);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    goto loc_00850970;
loc_00851800:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[14] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x1018);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    goto loc_0085093C;
loc_0085181C:
    ctx->gpr[3] = ctx->gpr[15] | ctx->gpr[15];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB8);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->lr = 0x0085182C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851074;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851074;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x108, ctx->gpr[31]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x108); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008518D8;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[0] = -ctx->fpr[0];
    ctx->fpr[12] = ctx->fpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[26] = vm_read32(ctx->gpr[1] + 0xF0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xF0);
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
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_008518D8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[30] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = ctx->gpr[26] + ctx->gpr[0];
    ctx->gpr[31] = (uint64_t)((int64_t)(int32_t)ctx->gpr[27] * (int64_t)(int32_t)ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852438;
loc_00851914:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085220C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008520C4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851990;
loc_00851930:
    ctx->fpr[12] = ctx->fpr[26];
loc_00851934:
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851414;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085227C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[27];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
    goto loc_008513B0;
loc_00851990:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00852478;
    ctx->gpr[0] = ctx->gpr[18] + (int64_t)(-1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085245C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
loc_008519B8:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851930;
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(1);
    ctx->fpr[12] = ctx->fpr[26];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    goto loc_00851934;
loc_00851A04:
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    goto loc_008510A0;
loc_00851A18:
    ctx->gpr[3] = ctx->gpr[15] | ctx->gpr[15];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB8);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->lr = 0x00851A28; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851478;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[13] = ctx->fpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851AC4;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[0] = -ctx->fpr[0];
    ctx->fpr[12] = ctx->fpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0xF0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xF0);
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
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_00851AC4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[0];
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[27] * (int64_t)(int32_t)ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00851B18;
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00852320;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00851B18:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x102C);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[0]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008524E8;
loc_00851B30:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851414;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00852484;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[27];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    goto loc_008513B0;
loc_00851B74:
    ctx->gpr[3] = ctx->gpr[15] | ctx->gpr[15];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xB8);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xA8);
    ctx->lr = 0x00851B84; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008517AC;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    vm_write32(ctx->gpr[1] + 0x108, ctx->gpr[31]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x108); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851C24;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->gpr[11] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = ctx->gpr[11] | 0x2493;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[0] = -ctx->fpr[0];
    ctx->fpr[12] = ctx->fpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[8] = vm_read32(ctx->gpr[1] + 0xF0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xF0);
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
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_00851C24:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[0];
    ctx->gpr[31] = (uint64_t)((int64_t)(int32_t)ctx->gpr[27] * (int64_t)(int32_t)ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00852450;
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00852304;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_00851C74:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fdiv(ctx->fpr[30], ctx->fpr[28]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852348;
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x682C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[26] + (int64_t)(0x1C);
    ctx->ctr = ctx->gpr[10];
    ctx->gpr[11] = (int64_t)(int32_t)(0);
loc_00851CA4:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[0]));
    ctx->fpr[0] = fabs(ctx->fpr[0]);
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00851CC4;
    ctx->fpr[13] = ctx->fpr[0];
    ctx->gpr[30] = ctx->gpr[11] | ctx->gpr[11];
loc_00851CC4:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(1);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00851CA4;
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852344;
loc_00851CD4:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[30], 4, 0, 27, 0, 0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[28], ctx->fpr[1]));
    ctx->lr = 0x00851CF4; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[22] = ctx->gpr[31] - ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008513AC;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851F54;
    ctx->gpr[0] = ctx->gpr[18] + (int64_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    ctx->gpr[17] = (int64_t)(int32_t)ctx->gpr[27];
    ctx->gpr[20] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[31] = ctx->gpr[26] + (int64_t)(0x10);
    ctx->gpr[24] = (int64_t)(int32_t)(0);
loc_00851D38:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[3] = vm_read32(ctx->gpr[11] + 0xC);
    ctx->gpr[4] = vm_read32(ctx->gpr[10] + 0x8);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->lr = 0x00851D54; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 6, 0, 25, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[11] = vm_read32(ctx->gpr[3] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[29] = vm_read32(ctx->gpr[27] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00852554;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->ctr = ctx->gpr[0];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851DBC;
    goto loc_00851DE0;
loc_00851DB0:
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851DE0;
loc_00851DBC:
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(4);
    ctx->gpr[25] = ctx->gpr[25] + (int64_t)(1);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00851DB0;
loc_00851DCC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x275);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6824);
    ctx->lr = 0x00851DDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00851DE0:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[11] = ctx->fpr[30];
    ctx->fpr[13] = ctx->fpr[25];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851E38;
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[28], ctx->fpr[31]));
    ctx->lr = 0x00851DF8; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085204C;
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00851F60;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[28]));
loc_00851E38:
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00851E78;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->ctr = ctx->gpr[29];
    ctx->gpr[11] = ctx->gpr[0] | ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851E70;
    goto loc_00851ECC;
loc_00851E60:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851ECC;
loc_00851E70:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(4);
    if (((ctx->ctr = (uint64_t)(ctx->ctr - 1)) != 0)) goto loc_00851E60;
loc_00851E78:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[25], 2, 0, 29, 0, 0);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0xC);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[8], 0, 31, 31, 0, 0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851ECC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00852058;
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[17]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
loc_00851ECC:
    ctx->gpr[24] = ctx->gpr[24] + (int64_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint32_t)ctx->gpr[24]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00851D38;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851F54;
    ctx->gpr[31] = ctx->gpr[26] + (int64_t)(0x10);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
loc_00851EF0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[3] = vm_read32(ctx->gpr[10] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x8);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->lr = 0x00851F14; june_evt_func_00899E54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ppc_rlw64(ctx->gpr[0], 6, 0, 25, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 27, 0, 0);
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x1C);
    ctx->gpr[3] = ctx->gpr[3] + ctx->gpr[9];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00851F3C; june_evt_func_008933E4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[30] = ctx->gpr[9] | ctx->gpr[9];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00851EF0;
loc_00851F54:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_008513B0;
loc_00851F60:
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[20]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[28]));
    goto loc_00851E38;
loc_00851F78:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00851FDC;
loc_00851F88:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851FE8;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00851F9C;
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[9]);
loc_00851F9C:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x1014);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851FB4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x1C);
loc_00851FB4:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00851FC4; june_evt_func_00859374(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[28] + 0x1010, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008515E8;
    goto loc_008515FC;
loc_00851FDC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1010);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851F88;
loc_00851FE8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_008512E4;
loc_00851FF0:
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0x88);
    ctx->gpr[6] = vm_read32(ctx->gpr[1] + 0x8C);
    ctx->gpr[8] = vm_read32(ctx->gpr[1] + 0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_00852000:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00852020;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[31], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00852548;
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00852548;
loc_00852020:
    ctx->gpr[0] = ctx->gpr[9] + (int64_t)(1);
loc_00852024:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 4, 0, 27, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[11] = ctx->gpr[30] + ctx->gpr[9];
    vm_write32((ctx->gpr[30] + ctx->gpr[9]), ctx->gpr[10]);
    vm_write32(ctx->gpr[11] + 0x8, ctx->gpr[8]);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    vm_write32(ctx->gpr[11] + 0x4, ctx->gpr[6]);
    vm_write32(ctx->gpr[28] + 0x1010, ctx->gpr[0]);
    goto loc_008515E0;
loc_0085204C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[24], ctx->fpr[28]));
    goto loc_00851E38;
loc_00852058:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00851ECC;
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[17]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1018);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[10], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x18, tmp); }
    goto loc_00851ECC;
loc_008520C4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[26]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00852364;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[26]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x1034); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[23]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x1034, tmp); }
loc_008520E8:
    ctx->fpr[0] = ctx->fpr[12];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x9249 << 16);
    ctx->gpr[0] = ctx->gpr[0] | 0x2493;
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[11] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[9], 0x1F);
    ctx->gpr[0] = (int64_t)(int32_t)((int32_t)((int64_t)(int32_t)ctx->gpr[9] * (int64_t)(int32_t)ctx->gpr[0] >> 32));
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 3);
    ctx->gpr[11] = ctx->gpr[0] - ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00851930;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[11];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[30], 0, 32);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[30] = (uint64_t)((int64_t)(int32_t)ctx->gpr[27] * (int64_t)(int32_t)ctx->gpr[11]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[12], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x1034, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[1]));
    ctx->lr = 0x00852160; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    ctx->fpr[0] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[1]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->lr = 0x00852180; june_evt_func_00AECE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[1], 0); memcpy(&ctx->fpr[1], &iv, 8); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->fpr[12] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[10] = ctx->gpr[30] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[1], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[11] = ctx->gpr[9] - ctx->gpr[0];
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008521E4;
    ctx->gpr[11] = ctx->gpr[30] + ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00852390;
    ctx->gpr[9] = ctx->gpr[0] - ctx->gpr[9];
    ctx->gpr[11] = ctx->gpr[18] + (int64_t)(-1);
    ctx->gpr[9] = ctx->gpr[18] + ctx->gpr[9];
    ctx->gpr[31] = ctx->gpr[11] | ctx->gpr[11];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
loc_008521E4:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[11];
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[31]));
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x4, tmp); }
    goto loc_00851934;
loc_0085220C:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00852578;
    { int64_t a = (int32_t)ctx->gpr[18]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085255C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
loc_00852230:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x102C);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00851930;
    ctx->gpr[0] = ctx->gpr[31] + (int64_t)(-1);
    ctx->fpr[12] = ctx->fpr[26];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    goto loc_00851934;
loc_0085227C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085233C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[27];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
loc_008522EC:
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[13], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
    goto loc_008513B0;
loc_00852304:
    ctx->gpr[31] = ctx->gpr[18] + (int64_t)(-1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[11]);
    goto loc_00851C74;
loc_00852320:
    ctx->gpr[0] = ctx->gpr[18] + (int64_t)(-1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[11]);
    goto loc_00851B18;
loc_0085233C:
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    goto loc_008513C0;
loc_00852344:
    ctx->gpr[0] = ctx->gpr[30] | ctx->gpr[30];
loc_00852348:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x262);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6828);
    ctx->gpr[30] = ctx->gpr[0] | ctx->gpr[0];
    ctx->lr = 0x0085235C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00851CD4;
loc_00852364:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[26]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008523F4;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x1034); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[23]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x1034, tmp); }
    goto loc_008520E8;
loc_00852390:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = ctx->gpr[11] | ctx->gpr[11];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008523B8;
    ctx->gpr[31] = ctx->gpr[10] | ctx->gpr[10];
loc_008523B8:
    ctx->gpr[0] = ctx->gpr[31] - ctx->gpr[26];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[10];
    { int32_t _a=(int32_t)ctx->gpr[0], _b=(int32_t)ctx->gpr[27]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[23], ctx->fpr[0], 0, 0));
    goto loc_008521E4;
loc_008523F4:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x1034); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->lr = 0x008523FC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085242C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6830);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
loc_0085242C:
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x1034, tmp); }
    goto loc_008520E8;
loc_00852438:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x68F0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1C6);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x68B8);
    ctx->lr = 0x00852448; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00851914;
loc_00852450:
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_00851C74;
loc_0085245C:
    ctx->gpr[31] = ctx->gpr[18] + (int64_t)(-2);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    goto loc_008519B8;
loc_00852478:
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68BC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_008519B8;
loc_00852484:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085233C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[27];
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1018);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68B4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[31], ctx->fpr[13]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x18, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[23]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[9]);
    goto loc_008522EC;
loc_008524E8:
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    ctx->lr = 0x008524F8; june_evt_func_00889FA8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_00851B30;
loc_00852504:
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x1028);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[23] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0x12);
    ctx->gpr[24] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00852534; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0085253C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_008512F0;
loc_00852548:
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    goto loc_00852024;
loc_00852554:
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    goto loc_00851DCC;
loc_0085255C:
    ctx->gpr[31] = ctx->gpr[18] + (int64_t)(-1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    goto loc_00852230;
loc_00852578:
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x68CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_00852230;
}

void june_evt_func_008530D8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008530F8; june_evt_func_00852F30(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853124;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00853120; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_00853124:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_008533C8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[4] + 0x20);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    ctx->gpr[28] = vm_read32(ctx->gpr[3] + 0x20);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x008533F0; june_evt_func_00853238(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x3C);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853420;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085341C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_00853420:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_00853A40(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853A78;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[3] + 0xC, ctx->gpr[0]);
loc_00853A78:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1014);
    ctx->lr = 0x00853A80; june_evt_func_00859374(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00853A90; june_evt_func_00853880(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1018);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853AA8;
    ctx->lr = 0x00853AA4; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00853AA8:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x101C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853AC0;
    ctx->lr = 0x00853ABC; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00853AC0:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x102C);
    ctx->lr = 0x00853AC8; june_evt_func_0088A6E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x1014);
    ctx->lr = 0x00853AD4; june_evt_func_0085A690(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x00853AE0; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_00853B64(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xA8, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[31]);
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x18);
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[30]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xB0, tmp); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00853BD0; june_evt_func_0085D2F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6804); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00853CBC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67EC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
loc_00853C00:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67E8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x10);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x00853C48; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    ctx->gpr[4] = vm_read32(ctx->gpr[27] + 0x1C);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x00853C88; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00853CBC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6800); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67FC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67EC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_00853C00;
}

void june_evt_func_00853CD4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->lr = 0x00853CF0; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->lr = 0x00853CFC; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x18);
    ctx->lr = 0x00853D08; june_evt_func_0085D660(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->lr = 0x00853D14; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00853D20; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_00853F8C(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x330, ctx->gpr[1]); ctx->gpr[1] += -0x330;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x2C0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x2D8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x2F8, ctx->gpr[30]);
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x308, tmp); }
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x310, tmp); }
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x318, tmp); }
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x320, tmp); }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x88);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x328, tmp); }
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0x2C8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x2E8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x300, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x2D0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x2E0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x2F0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x340, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    ctx->lr = 0x00853FF4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854054;
loc_00854008:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x340);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[31];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x308); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x310); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x318); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x320); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x328); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x330);
    return;
loc_00854054:
    ctx->lr = 0x00854058; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854088;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x67D0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00854088:
    ctx->lr = 0x0085408C; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00854098; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[28] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x008540D4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854100;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x67CC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00854100:
    ctx->lr = 0x00854104; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00854110; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[3]);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085414C; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x00854160; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00854174; june_evt_func_00D2F758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[9] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[3], 0x1F);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[9] ^ ctx->gpr[3];
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[0];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 31, 31, 0, 0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085420C;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00854228;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008546B8;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008546A4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854720;
    ;
loc_008541C0:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->lr = 0x008541D0; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->lr = 0x008541E8; june_evt_func_00D2F758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854270;
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00854204; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00854270;
loc_0085420C:
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00854270;
    ;
loc_00854228:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854488;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085465C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085480C;
    ;
loc_00854248:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854264;
    ctx->lr = 0x00854254; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008547C4;
loc_00854264:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[25] = (int64_t)(int32_t)(1);
loc_00854270:
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x008542AC; june_evt_func_0085D5E4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008542C0;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
loc_008542C0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6804); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854534;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6800); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67FC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67EC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_008542E8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67E8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00854334; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854348;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
loc_00854348:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    ctx->gpr[7] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x1C);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085438C; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008543A0;
    ctx->gpr[31] = (int64_t)(int8_t)ctx->gpr[25];
loc_008543A0:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854550;
    ctx->lr = 0x008543B4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854008;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x67B4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854008;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854760;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
loc_00854430:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008548AC;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x2A0);
    ctx->fpr[12] = ctx->fpr[28];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = (int64_t)(int32_t)(0x136);
    ctx->gpr[6] = (int64_t)(int32_t)(0xAA);
    ctx->fpr[13] = ctx->fpr[0];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x2A0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-155);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->lr = 0x0085447C; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = (int64_t)(int32_t)(1);
    goto loc_0085461C;
loc_00854488:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0xA);
    ctx->lr = 0x00854498; june_evt_func_00D2D29C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = vm_read32(ctx->gpr[30] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008546F0;
loc_008544A8:
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x67C8);
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008544C4; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008544D4; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008544E4; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854248;
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854708;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00854914;
loc_00854510:
    vm_write64(ctx->gpr[1] + 0x2B0, ctx->gpr[27]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x2B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67B8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    goto loc_00854248;
loc_00854534:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67F0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x67EC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    goto loc_008542E8;
loc_00854550:
    ctx->lr = 0x00854554; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854008;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x67B4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854008;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085476C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854008;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085494C;
    ctx->fpr[0] = ctx->fpr[28];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x2A0);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = (int64_t)(int32_t)(0x136);
    ctx->gpr[6] = (int64_t)(int32_t)(0xAA);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x2A0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-155);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    ctx->lr = 0x00854614; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = (int64_t)(int32_t)(1);
loc_0085461C:
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 62);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[8] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[5] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x00854644; june_evt_func_0083FE10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x00854654; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00854008;
loc_0085465C:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00854668; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854248;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00854680; june_evt_func_00D28758(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00854248;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0085469C; june_evt_func_00D2F8AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00854248;
loc_008546A4:
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->lr = 0x008546B0; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008541C0;
loc_008546B8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x67C8);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->lr = 0x008546D8; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008546E8; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008541C0;
loc_008546F0:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008544A8;
    ctx->gpr[31] = ctx->gpr[0] | ctx->gpr[0];
    goto loc_008544A8;
loc_00854708:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008548F8;
loc_00854714:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[31]);
    goto loc_00854248;
loc_00854720:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x67C4);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[1] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x2B8, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x2B8);
    ctx->lr = 0x00854748; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00854758; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008541C0;
loc_00854760:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00854430;
loc_0085476C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085478C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x67E4);
    ctx->gpr[4] = (int64_t)(int32_t)(0x12C);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x67B0);
    ctx->lr = 0x00854788; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085478C:
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->lr = 0x00854794; june_evt_func_00881B7C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x67C8);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[3], 0, 33);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008547AC; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008547BC; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00854008;
loc_008547C4:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x2C);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0x11);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->lr = 0x008547FC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00854804; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00854270;
loc_0085480C:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085481C; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    { double a = ctx->fpr[1]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00854840;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[1]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00854840;
    ctx->fpr[31] = ctx->fpr[1];
loc_00854840:
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x67C4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x2B8, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x2B8);
    ctx->lr = 0x00854864; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00854874; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00854884; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854248;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x28);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00854930;
loc_008548A0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    goto loc_00854248;
loc_008548AC:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x2A0);
    ctx->fpr[12] = ctx->fpr[28];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    ctx->fpr[13] = ctx->fpr[0];
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x2A0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-77);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->lr = 0x008548EC; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    goto loc_0085461C;
loc_008548F8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x67E4);
    ctx->gpr[4] = (int64_t)(int32_t)(0xCA);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x67C0);
    ctx->lr = 0x00854908; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
    goto loc_00854714;
loc_00854914:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x67E4);
    ctx->gpr[4] = (int64_t)(int32_t)(0xCD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x67BC);
    ctx->lr = 0x00854924; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
    goto loc_00854510;
loc_00854930:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x67E4);
    ctx->gpr[4] = (int64_t)(int32_t)(0xDE);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x67BC);
    ctx->lr = 0x00854940; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
    goto loc_008548A0;
loc_0085494C:
    ctx->fpr[0] = ctx->fpr[28];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x2A0);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x2A0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-77);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    ctx->lr = 0x00854984; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    goto loc_0085461C;
}

void june_evt_func_00854990(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[4]);
    return;
}

void june_evt_func_00854998(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    return;
}

void june_evt_func_00854D64(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    ctx->gpr[26] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[22]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854E00;
loc_00854DC0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_00854E00:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00854E18; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854DC0;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854FA0;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00854FA8;
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_00854E70:
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x14);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854F50;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6798); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x70);
    ;
loc_00854EA0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00854ECC; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854F40;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854F0C;
    goto loc_00854F40;
loc_00854F08:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_00854F0C:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    ctx->lr = 0x00854F2C; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x10, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854F08;
loc_00854F40:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00854EA0;
    ctx->gpr[11] = vm_read32(ctx->gpr[27] + 0x0);
loc_00854F50:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854DC0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00854DC0;
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x00854F98; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00854DC0;
loc_00854FA0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_00854E70;
loc_00854FA8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6798); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x8);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_00854E70;
}

void june_evt_func_00854FD0(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[20]);
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0x80);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[31]);
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[25] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[30]);
    ctx->gpr[23] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    vm_write32(ctx->gpr[1] + 0x118, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    ctx->lr = 0x0085504C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855278;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085512C;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x678C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008550D4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6798); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_008550A8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[30]));
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008550C8;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
loc_008550C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008550A8;
loc_008550D4:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855340;
loc_008550E4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x678C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085512C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085534C;
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x8);
loc_00855104:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6794); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_0085512C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[0]);
    ctx->lr = 0x00855148; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855330;
loc_0085515C:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855264;
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6798); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[21], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
loc_00855180:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x008551B0; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008551C4;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
loc_008551C4:
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008551D8;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008552D0;
loc_008551D8:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855258;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855214;
    goto loc_00855258;
loc_00855210:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_00855214:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
    ctx->lr = 0x00855234; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855248;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
loc_00855248:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855210;
loc_00855258:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855180;
loc_00855264:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[26];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008552E0;
loc_00855278:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x118);
    ctx->gpr[19] = _cs_19;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_008552D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] ^ 0x1;
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
    goto loc_008551D8;
loc_008552E0:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[19];
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855278;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[6] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->lr = 0x00855314; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855278;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[26];
    goto loc_00855278;
loc_00855330:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6790);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    goto loc_0085515C;
loc_00855340:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[25] + 0xC, tmp); }
    goto loc_008550E4;
loc_0085534C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6798); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x8);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    goto loc_00855104;
}

void june_evt_func_00855378(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xB0, ctx->gpr[1]); ctx->gpr[1] += -0xB0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x67A8);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[30]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x22);
    ctx->gpr[3] = (int64_t)(int32_t)(0x18);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->lr = 0x008553B4; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x23);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855474;
loc_008553D0:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0x18);
    ctx->lr = 0x008553DC; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[9], 0, 28, 28, 0, 0);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855450;
    ctx->lr = 0x00855408; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6798); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6784); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(2);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6794); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[11]);
    ctx->lr = 0x00855448; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
loc_00855450:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xB0);
    return;
loc_00855474:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6788);
    ctx->lr = 0x0085547C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008553D0;
}

void june_evt_func_00855484(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x14);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085553C;
loc_008554B8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855588;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[30]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855598;
loc_008554E0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
loc_008554EC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855518;
loc_008554F8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->lr = 0x00855508; june_evt_func_0086C458(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008554F8;
loc_00855518:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->lr = 0x00855520; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0085552C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008554B8;
loc_0085553C:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855558;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00855554; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00855558:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00855560; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
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
loc_00855588:
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[30]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008554E0;
loc_00855598:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[0]);
    goto loc_008554EC;
}

void june_evt_func_008555A8(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x10, ctx->gpr[4]);
    return;
}

void june_evt_func_008555B0(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x10);
    return;
}

void june_evt_func_008556D0(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[4] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x677C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xC8, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[30]);
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[31] = vm_read32(ctx->gpr[4] + 0x4);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008557F0;
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { double a = ctx->fpr[31]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[27] = ctx->gpr[29] + (int64_t)(-1);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008557A8;
    goto loc_008557F0;
loc_00855754:
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->lr = 0x00855778; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008557F0;
loc_0085578C:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008557F0;
loc_008557A8:
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[27]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855754;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0x10);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x008557DC; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085578C;
loc_008557F0:
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
    return;
}

void june_evt_func_00855824(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0xC);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00855880;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
loc_00855858:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[31], 4, 0, 27, 0, 0);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(1);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00855874; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855858;
loc_00855880:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->lr = 0x0085588C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00855898; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_008558BC(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[31]);
    ctx->lr = 0x00855900; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[24] = vm_read32(ctx->gpr[2] + -0x6778);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x14);
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[6] = (int64_t)(int32_t)(0x2D);
    ctx->lr = 0x00855924; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855BFC;
loc_00855934:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x14);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00855948; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[25] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write32(ctx->gpr[27] + 0x0, ctx->gpr[0]);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int32_t)(0x10);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085599C;
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ;
loc_00855978:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[26] = ctx->gpr[26] + (int64_t)(1);
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855978;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[26], 4, 0, 27, 0, 0);
    ctx->gpr[21] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x10);
loc_0085599C:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[6] = (int64_t)(int32_t)(0x3A);
    ctx->lr = 0x008559B8; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[27] + 0x4, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855C14;
loc_008559C8:
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x008559D8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x008559E8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855A1C;
    ctx->gpr[0] = (int64_t)(int32_t)(6);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
loc_00855A1C:
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00855BF4;
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = vm_read32(ctx->gpr[25] + 0x4);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = vm_read32(ctx->gpr[27] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6760); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    goto loc_00855A68;
loc_00855A3C:
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0xC, tmp); }
    ctx->gpr[11] = ctx->gpr[31] + ctx->gpr[6];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[26]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855B38;
loc_00855A68:
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[5];
    ctx->gpr[31] = ppc_rlw64(ctx->gpr[28], 4, 0, 27, 0, 0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[29] = ctx->gpr[31] + ctx->gpr[6];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    ctx->lr = 0x00855A8C; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[27] + 0x4);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[3]);
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[6];
    ctx->gpr[4] = (int64_t)(int32_t)(0x4C);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855BE0;
loc_00855AB4:
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[6];
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x4D);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855BCC;
loc_00855AD4:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[6];
    ctx->gpr[5] = vm_read32(ctx->gpr[25] + 0x4);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[8] = ctx->gpr[30] + ctx->gpr[5];
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[7] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[10] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0x8, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00855A3C;
    ctx->gpr[9] = ctx->gpr[30] + ctx->gpr[5];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[7] + 0xC, tmp); }
    ctx->gpr[11] = ctx->gpr[31] + ctx->gpr[6];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[28] = ctx->gpr[28] + (int64_t)(1);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[26]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855A68;
loc_00855B38:
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x70);
    ctx->gpr[28] = ppc_rlw64(ctx->gpr[21], 4, 0, 27, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[28] + ctx->gpr[6];
    ctx->lr = 0x00855B60; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[21] = _cs_21;
    ctx->gpr[28] = ctx->gpr[28] + ctx->gpr[0];
    ctx->gpr[23] = _cs_23;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[25] = _cs_25;
    ctx->gpr[9] = ctx->gpr[26] + (int64_t)(1);
    ctx->gpr[30] = _cs_30;
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[3]);
    vm_write32(ctx->gpr[27] + 0xC, ctx->gpr[9]);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[26] = _cs_26;
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[22] = _cs_22;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_00855BCC:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6764);
    ctx->lr = 0x00855BD4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[27] + 0x4);
    goto loc_00855AD4;
loc_00855BE0:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6768);
    ctx->lr = 0x00855BE8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[27] + 0x4);
    goto loc_00855AB4;
loc_00855BF4:
    ctx->gpr[6] = vm_read32(ctx->gpr[27] + 0x4);
    goto loc_00855B38;
loc_00855BFC:
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6770);
    ctx->gpr[4] = (int64_t)(int32_t)(0x2E);
    ctx->lr = 0x00855C0C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00855934;
loc_00855C14:
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x676C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x3B);
    ctx->lr = 0x00855C24; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008559C8;
}

void june_evt_func_00855C2C(ppu_context* ctx) {
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
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x677C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[29]);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[30] = ctx->gpr[7] | ctx->gpr[7];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x00855CB8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6780); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855CE8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x675C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00855CE8:
    ctx->lr = 0x00855CEC; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00855CF8; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
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
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6780); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[29] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x00855D34; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855D60;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6758);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00855D60:
    ctx->lr = 0x00855D64; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00855D70; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[13]));
    ctx->fpr[28] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855F94;
loc_00855DB4:
    ctx->gpr[22] = (int64_t)(int32_t)(0);
loc_00855DB8:
    ctx->gpr[26] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00855F34;
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[17] = vm_read32(ctx->gpr[2] + -0x6740);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[19] = vm_read32(ctx->gpr[2] + -0x674C);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6748); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6744); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[20] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[18] = (int64_t)(int32_t)(2);
    goto loc_00855E48;
loc_00855DFC:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[3] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[7] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x00855E14; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855E28;
    ctx->gpr[22] = (int64_t)(int32_t)(1);
loc_00855E28:
    ctx->gpr[30] = ctx->gpr[30] + (int64_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[26]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855F34;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x10);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
loc_00855E48:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855DFC;
    ctx->lr = 0x00855E68; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855DFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[20], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00855EA0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855DFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855DFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855DFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855DFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00855DFC;
    ctx->lr = 0x00855EFC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855F2C;
    ctx->gpr[0] = vm_read16(ctx->gpr[17] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855F2C;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    vm_write32(ctx->gpr[28] + 0x8, ctx->gpr[29]);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[0]);
loc_00855F2C:
    vm_write16(ctx->gpr[19] + 0x0, ctx->gpr[18]);
    goto loc_00855DFC;
loc_00855F34:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[22], 0, 63);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_00855F94:
    ctx->lr = 0x00855F98; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00855DB4;
    ctx->lr = 0x00855FAC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856040;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6754);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856040;
    ctx->lr = 0x00855FD4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856060;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6750);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6780); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856060;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00856030;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
loc_00856030:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x674C);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_00855DB8;
loc_00856040:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x8, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[9]);
    goto loc_00855DB8;
loc_00856060:
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    goto loc_00856030;
}

void june_evt_func_00856068(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x44, ctx->gpr[4]);
    return;
}

void june_evt_func_00856070(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x44);
    return;
}

void june_evt_func_00856B24(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = (int64_t)(int32_t)(0x46);
    ctx->gpr[3] = (int64_t)(int32_t)(0x54);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->lr = 0x00856B64; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[30] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x47);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856CB0;
loc_00856B84:
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = (int64_t)(int32_t)(0x54);
    ctx->lr = 0x00856B90; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00856C74;
loc_00856BB0:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00856C34;
    ctx->lr = 0x00856BD4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x66F4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)(2);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[11]);
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[11]);
    ctx->lr = 0x00856C14; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[3]);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    ctx->lr = 0x00856C2C; june_evt_func_00860600(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[3]);
loc_00856C34:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x10);
    vm_write32(ctx->gpr[31] + 0x48, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x14);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    vm_write32(ctx->gpr[31] + 0x4C, ctx->gpr[11]);
    vm_write32(ctx->gpr[31] + 0x50, ctx->gpr[9]);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_00856C74:
    ctx->lr = 0x00856C78; june_evt_func_008558BC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[0]);
    ctx->lr = 0x00856C8C; june_evt_func_008555B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x4F);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00856BB0;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66F8);
    ctx->lr = 0x00856CA8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00856BB0;
loc_00856CB0:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66FC);
    ctx->lr = 0x00856CB8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00856B84;
}

void june_evt_func_00857B80(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x180, ctx->gpr[1]); ctx->gpr[1] += -0x180;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[27]);
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x160, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[0]);
    ctx->lr = 0x00857BEC; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[23] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00857DA8;
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00857DA8;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = ctx->gpr[3] + (int64_t)(4);
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xCC);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00857C64; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857E00;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857CFC;
loc_00857CA4:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    ctx->lr = 0x00857CD4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00858360;
loc_00857CE4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->lr = 0x00857CEC; june_evt_func_008555F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[31] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
loc_00857CFC:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->lr = 0x00857D2C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857FA0;
loc_00857D3C:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
loc_00857D40:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 25, 25, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857E7C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857DA8;
loc_00857D58:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[20];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00858338;
loc_00857D74:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[19];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857DA8;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x8C);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x00857DA0; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ;
loc_00857DA8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x190);
    ctx->gpr[19] = _cs_19;
    ctx->gpr[20] = _cs_20;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x160); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x170); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x180);
    return;
loc_00857E00:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008582C0;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008582DC;
loc_00857E2C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[12]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008582F4;
loc_00857E3C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085830C;
loc_00857E5C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[20] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0xBC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857CA4;
    goto loc_00857CFC;
loc_00857E7C:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ctx->gpr[21] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x7C);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->lr = 0x00857EEC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00857F20; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->lr = 0x00857F54; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x00857F88; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857D58;
    goto loc_00857DA8;
loc_00857FA0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(-128);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    ctx->gpr[9] = (int64_t)(int32_t)(0x10);
    ctx->gpr[11] = (int64_t)(int32_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    vm_write8(ctx->gpr[1] + 0x76, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x75, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    vm_write8(ctx->gpr[1] + 0x7A, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x7B, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x77, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x79, ctx->gpr[9]);
    ctx->lr = 0x00857FF4; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858070;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858070;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858070;
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858070;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008583E8;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
    goto loc_00858050;
loc_00858044:
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008583E8;
loc_00858050:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00858060; june_evt_func_00857994(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00858044;
    ;
loc_00858070:
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008580AC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->fpr[2] = ctx->fpr[1];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x008580AC; june_evt_func_00856178(ctx); DRAIN_TRAMPOLINE(ctx);
loc_008580AC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858158;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->lr = 0x008580DC; june_evt_func_008555B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00858158;
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[31]));
    ctx->gpr[30] = ctx->gpr[28] + (int64_t)(0x18);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[28] = ctx->gpr[3] + (int64_t)(-1);
    goto loc_00858114;
loc_008580FC:
    { double a = ctx->fpr[28]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00858158;
    ctx->lr = 0x00858108; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858158;
loc_00858114:
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->lr = 0x00858124; june_evt_func_00855668(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->fpr[2] = ctx->fpr[29];
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[4] = ctx->fpr[30];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[1] = ctx->fpr[31];
    ctx->fpr[3] = ctx->fpr[31];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008580FC;
    ;
loc_00858158:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857D40;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857D40;
    ctx->lr = 0x00858178; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008583AC;
loc_0085818C:
    ctx->lr = 0x00858190; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085819C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x008581D8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085838C;
loc_008581E8:
    ctx->lr = 0x008581EC; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008581F8; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857D3C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857D3C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857D3C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857D3C;
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[1]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[10] = (int64_t)(int32_t)(1);
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[2] = ctx->fpr[1];
    ctx->lr = 0x008582B8; june_evt_func_00856178(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
    goto loc_00857D40;
loc_008582C0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857E2C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008583CC;
loc_008582D8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
loc_008582DC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[12]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00857E3C;
loc_008582F4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00857E5C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00858514;
loc_0085830C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[20] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0xBC);
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00857CA4;
    goto loc_00857CFC;
loc_00858338:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x8C);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x00858358; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00857D74;
loc_00858360:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    vm_write32(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    ctx->lr = 0x00858384; june_evt_func_008556D0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00857CE4;
loc_0085838C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    goto loc_008581E8;
loc_008583AC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0085818C;
loc_008583CC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x14B);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66D4);
    ctx->lr = 0x008583DC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x0);
    goto loc_008582D8;
loc_008583E8:
    ctx->lr = 0x008583EC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085841C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085841C:
    ctx->lr = 0x00858420; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085842C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x671C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x8);
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_frsp(ctx->fpr[13]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[11], ctx->fpr[13], 0, 0));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858530;
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858530;
    ctx->gpr[9] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; } ctx->gpr[9] += 0x14;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x672C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->lr = 0x008584E8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858070;
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->lr = 0x0085850C; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00858070;
loc_00858514:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x15B);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66D0);
    ctx->lr = 0x00858524; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x98); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085830C;
loc_00858530:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[5] = ctx->gpr[23] + (int64_t)(0x14);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->lr = 0x0085854C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00858070;
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->lr = 0x00858570; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00858070;
}

void june_evt_func_0085A690(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085A6B8; june_evt_func_00859374(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A6D4;
    ctx->lr = 0x0085A6CC; june_evt_func_0085A3B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
loc_0085A6D4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A6EC;
    ctx->lr = 0x0085A6E8; june_evt_func_00855824(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085A6EC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A704;
    ctx->lr = 0x0085A700; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085A704:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A71C;
    ctx->lr = 0x0085A718; june_evt_func_00860578(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085A71C:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0085A724; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0085A7E0(ppu_context* ctx) {
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
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x158, tmp); }
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x2C);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x150, tmp); }
    ctx->fpr[30] = ctx->fpr[31];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[31]);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x8C);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A8D0;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x0085A8AC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AD60;
loc_0085A8BC:
    ctx->gpr[22] = (int64_t)(int32_t)(0);
loc_0085A8C0:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x2C);
    ctx->lr = 0x0085A8C8; june_evt_func_008555F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[30] = ctx->fpr[1];
loc_0085A8D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AAA8;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[10] = ctx->fpr[0];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A98C;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6730); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x8);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085A91C;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
loc_0085A91C:
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A944;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_0085A944:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A98C;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085A98C;
    ctx->gpr[30] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x88);
    ;
loc_0085A968:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x0085A980; june_evt_func_00858E50(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A968;
loc_0085A98C:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AD9C;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085A9A0; june_evt_func_008555C4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x2C);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0085A9B0; june_evt_func_008555F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[1]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_0085A9C0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AD3C;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AD3C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085AD54;
loc_0085A9E8:
    { double a = ctx->fpr[10]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085AD54;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085ADA4;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x30);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B094;
    ctx->gpr[18] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ;
loc_0085AA30:
    { double a = ctx->fpr[12]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085ADC4;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x34);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AA70;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    ;
loc_0085AA70:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6720); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    vm_write32(ctx->gpr[11] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x14, tmp); }
loc_0085AAA8:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    ctx->gpr[5] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = ctx->gpr[23] | ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->lr = 0x0085AAE4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AC34;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    ctx->gpr[5] = vm_read32(ctx->gpr[28] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    vm_write32(ctx->gpr[1] + 0x84, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AB6C;
    ctx->fpr[2] = ctx->fpr[1];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xBC);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x80);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    ctx->lr = 0x0085AB68; june_evt_func_00856FE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
loc_0085AB6C:
    ctx->lr = 0x0085AB70; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ACE8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[31] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0085ACE8;
loc_0085AB9C:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B0C4;
loc_0085ABA8:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[11] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ADDC;
loc_0085ABBC:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AC38;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 31, 31, 0, 0);
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[22];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AC48;
    ;
loc_0085ABE0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x170);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x150); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x158); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x160);
    return;
loc_0085AC34:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
loc_0085AC38:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 31, 31, 0, 0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABE0;
loc_0085AC48:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[19];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ACA8;
loc_0085AC64:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[18];
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABE0;
loc_0085AC74:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x30);
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->lr = 0x0085AC8C; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ABE0;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[22];
    goto loc_0085ABE0;
loc_0085ACA8:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x34);
    ctx->gpr[5] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->lr = 0x0085ACC0; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AC64;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[18];
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[22];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABE0;
    goto loc_0085AC74;
loc_0085ACE8:
    ctx->lr = 0x0085ACEC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AD10;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66DC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AB9C;
loc_0085AD10:
    ctx->lr = 0x0085AD14; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABA8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66DC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABA8;
    goto loc_0085AB9C;
loc_0085AD3C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { double a = ctx->fpr[9]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x38, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x3C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085A9E8;
loc_0085AD54:
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->gpr[18] = (int64_t)(int32_t)(0);
    goto loc_0085AAA8;
loc_0085AD60:
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x2C);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0085AD88; june_evt_func_00855C2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085A8C0;
    goto loc_0085A8BC;
loc_0085AD9C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_0085A9C0;
loc_0085ADA4:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B078;
    ctx->gpr[18] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x30);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085AA30;
loc_0085ADC4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B0AC;
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x34);
    goto loc_0085AA70;
loc_0085ADDC:
    ctx->lr = 0x0085ADE0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AE04;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66DC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B26C;
loc_0085AE04:
    ctx->lr = 0x0085AE08; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AFE0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66C8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AFE0;
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AFE0;
    ctx->lr = 0x0085AE3C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AE6C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085AE6C:
    ctx->lr = 0x0085AE70; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085AE7C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085AEB8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085AEE4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0085AEE4:
    ctx->lr = 0x0085AEE8; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085AEF4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B538;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B538;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[11] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[11]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B538;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B538;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B878;
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 3, 3, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B860;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[10] = ctx->gpr[10] | ((uint64_t)0x1000 << 16);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x24, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x28, tmp); }
loc_0085AFB4:
    ctx->gpr[9] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x84);
    ctx->gpr[11] = ctx->gpr[25] | ctx->gpr[25];
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); } ctx->gpr[9] += 0x14;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    goto loc_0085ABBC;
loc_0085AFE0:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[11] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 3, 3, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABBC;
    ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B068;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B058;
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int32_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B058;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B3B4;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[4], 0, 32);
    goto loc_0085B038;
loc_0085B02C:
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B3B0;
loc_0085B038:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085B048; june_evt_func_00857994(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B02C;
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
loc_0085B058:
    ctx->gpr[11] = ctx->gpr[25] | ctx->gpr[25];
loc_0085B05C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[28] + 0x40, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[0]);
loc_0085B068:
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[10], 3, 1, 31, 0, 0);
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[10], 29, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x0, ctx->gpr[10]);
    goto loc_0085ABBC;
loc_0085B078:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[18] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x30);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    goto loc_0085AA30;
loc_0085B094:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[18] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    goto loc_0085AA30;
loc_0085B0AC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6728); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x34);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0085AA70;
loc_0085B0C4:
    ctx->lr = 0x0085B0C8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B0F8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085B0F8:
    ctx->lr = 0x0085B0FC; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085B108; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085B144; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B170;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0085B170:
    ctx->lr = 0x0085B174; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085B180; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ABA8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ABA8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ABA8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085ABA8;
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x84);
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B224;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0085B224; june_evt_func_00859374(ctx); DRAIN_TRAMPOLINE(ctx);
loc_0085B224:
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABA8;
    ctx->lr = 0x0085B230; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ABA8;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0085B8C0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0085B888;
loc_0085B25C:
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0085B264; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085ABA8;
loc_0085B26C:
    ctx->lr = 0x0085B270; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B2A0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085B2A0:
    ctx->lr = 0x0085B2A4; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085B2B0; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085B2EC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B318;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0085B318:
    ctx->lr = 0x0085B31C; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085B328; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AE04;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AE04;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[31];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AE04;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085AE04;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write8(ctx->gpr[28] + 0x40, ctx->gpr[0]);
    goto loc_0085AE04;
loc_0085B3B0:
    ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0xC);
loc_0085B3B4:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4C);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B544;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085B3DC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_0085B3E0:
    ctx->lr = 0x0085B3E4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6700); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B414;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66E4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085B414:
    ctx->lr = 0x0085B418; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085B424; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[24] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->fpr[12] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_frsp(ctx->fpr[11]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x671C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[11], ctx->fpr[13], 0, 0));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B520;
    ctx->gpr[25] = vm_read32(ctx->gpr[26] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B520;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B6DC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085B4C4; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_0085B4C8:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x50);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B7BC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x10);
loc_0085B4DC:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0xC);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085B4F4; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_0085B4F8:
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B514;
    ctx->lr = 0x0085B504; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B694;
loc_0085B514:
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_0085B05C;
loc_0085B520:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B5B8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    goto loc_0085B4DC;
loc_0085B538:
    ctx->gpr[11] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_0085ABBC;
loc_0085B544:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B830;
loc_0085B554:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B814;
loc_0085B564:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B564;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0085B57C:
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x30);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x2C, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[30] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[30] + 0x28, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B3E0;
loc_0085B598:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085B5A8; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B598;
    goto loc_0085B3E0;
loc_0085B5B8:
    ctx->gpr[30] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B62C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B848;
loc_0085B5DC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B5DC;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0085B5F4:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B64C;
loc_0085B610:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085B620; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B610;
loc_0085B62C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B64C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x31E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66CC);
    ctx->lr = 0x0085B648; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085B64C:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x30);
    vm_write32(ctx->gpr[26] + 0x30, ctx->gpr[30]);
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[24]);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B4F8;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
loc_0085B674:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085B684; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B674;
    goto loc_0085B4F8;
loc_0085B694:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[21], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x44);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xE);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x0085B6C4; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0085B6CC; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    goto loc_0085B05C;
loc_0085B6DC:
    ctx->gpr[30] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B750;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B854;
loc_0085B700:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B700;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[0]);
loc_0085B718:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B770;
loc_0085B734:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085B744; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B734;
loc_0085B750:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B770;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x31E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66CC);
    ctx->lr = 0x0085B76C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085B770:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[25], 0, 32);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[25]);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x30);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[30]);
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[4]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B4C8;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[4], 0, 32);
loc_0085B79C:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085B7AC; june_evt_func_00857710(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B79C;
    goto loc_0085B4C8;
loc_0085B7BC:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B820;
loc_0085B7D4:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[11] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085B7D4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    vm_write32(ctx->gpr[11] + 0x34, ctx->gpr[0]);
loc_0085B7F8:
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x34);
    vm_write32(ctx->gpr[11] + 0x34, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xC);
    vm_write32(ctx->gpr[9] + 0x34, ctx->gpr[10]);
    goto loc_0085B4F8;
loc_0085B814:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0085B57C;
loc_0085B820:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x34);
    vm_write32(ctx->gpr[11] + 0x30, ctx->gpr[0]);
    goto loc_0085B7F8;
loc_0085B830:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x673C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x335);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x66D8);
    ctx->lr = 0x0085B840; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085B554;
loc_0085B848:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0085B5F4;
loc_0085B854:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    vm_write32(ctx->gpr[9] + 0x30, ctx->gpr[0]);
    goto loc_0085B718;
loc_0085B860:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085B8D4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085AFB4;
loc_0085B878:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085AFB4;
loc_0085B888:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
loc_0085B898:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x44);
    ctx->gpr[7] = (int64_t)(int32_t)(0xE);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0085B8B8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    goto loc_0085B25C;
loc_0085B8C0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    goto loc_0085B898;
loc_0085B8D4:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6718); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = fabs(ctx->fpr[13]);
    { double a = ctx->fpr[13]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085B918;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[13] = fabs(ctx->fpr[13]);
    { double a = ctx->fpr[13]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085AFB4;
loc_0085B918:
    vm_write32(ctx->gpr[28] + 0xC, ctx->gpr[9]);
    goto loc_0085AFB4;
}

void june_evt_func_0085B928(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    return;
}

void june_evt_func_0085BB74(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0xC);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BBB4;
loc_0085BB98:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x34);
    ctx->lr = 0x0085BBAC; june_evt_func_0085BA44(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085BB98;
loc_0085BBB4:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0085BBBC; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_0085BF10(ppu_context* ctx) {
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
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0xC);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BFB0;
    ctx->gpr[26] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[7];
loc_0085BF50:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BFA4;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[29] = vm_read32(ctx->gpr[9] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085BFA4;
    ;
loc_0085BF78:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0085BF94; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085BF78;
loc_0085BFA4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085BF50;
loc_0085BFB0:
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
}

void june_evt_func_0085BFD8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x130, ctx->gpr[1]); ctx->gpr[1] += -0x130;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[0]);
    ctx->gpr[9] = vm_read16(ctx->gpr[4] + 0x0);
    ctx->gpr[21] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[22]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[23]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[29]);
    ctx->gpr[23] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C068;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 29, 1, 31, 0, 0);
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write16(ctx->gpr[4] + 0x0, ctx->gpr[0]);
loc_0085C068:
    ctx->lr = 0x0085C06C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C09C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66A4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[27] = ppu_frsp(ctx->fpr[0]);
loc_0085C09C:
    ctx->lr = 0x0085C0A0; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085C0AC; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085C0B8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C0E8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x66A0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[26] = ppu_frsp(ctx->fpr[0]);
loc_0085C0E8:
    ctx->lr = 0x0085C0EC; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085C0F8; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->lr = 0x0085C114; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C138;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x669C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C30C;
loc_0085C138:
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
loc_0085C13C:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[25];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C2B0;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6698); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_0085C154:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6688); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6684); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[12]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085C1A4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C2B0;
    ctx->gpr[0] = vm_read16(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C4A8;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
loc_0085C1E0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[4] = vm_read32(ctx->gpr[11] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C274;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
loc_0085C1F8:
    ctx->gpr[0] = ctx->gpr[4] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x10, tmp); }
    ctx->lr = 0x0085C244; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C258;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
loc_0085C258:
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C1F8;
loc_0085C274:
    ctx->gpr[0] = vm_read16(ctx->gpr[28] + 0x0);
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[25];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C2B0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C5D8;
loc_0085C29C:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[20];
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C470;
    ;
loc_0085C2B0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x140);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[20] = _cs_20;
    ctx->gpr[21] = _cs_21;
    ctx->lr = ctx->gpr[0];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x130);
    return;
loc_0085C30C:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[27];
    ctx->fpr[6] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[29];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[27]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[7] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[7]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C138;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[6], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C138;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[24];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[29]));
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[26];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[26]));
    ctx->fpr[3] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[3]; double b = ctx->fpr[4]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C138;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[4], ctx->fpr[9]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C138;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C138;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6698); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6694); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6690); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x668C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
loc_0085C3D8:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C460;
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[30] = ctx->fpr[1];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[1]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[2]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085C404;
    ctx->fpr[13] = ctx->fpr[2];
loc_0085C404:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C45C;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[6], ctx->fpr[5]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[4], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { double a = ctx->fpr[7]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C45C;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C45C;
    { double a = ctx->fpr[3]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C45C;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[29]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[3]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C654;
loc_0085C45C:
    ctx->fpr[5] = ppu_fp_single(ppu_fadd(ctx->fpr[5], ctx->fpr[13]));
loc_0085C460:
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C3D8;
    goto loc_0085C13C;
loc_0085C470:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x24);
    ctx->lr = 0x0085C48C; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C2B0;
    ctx->gpr[9] = (int64_t)(int8_t)ctx->gpr[25];
    goto loc_0085C2B0;
loc_0085C4A8:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[3] = vm_read32(ctx->gpr[11] + 0x20);
    ctx->lr = 0x0085C4B8; june_evt_func_0086C384(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x14);
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x20);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0085C4E0; june_evt_func_0086C3D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { double a = ctx->fpr[13]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C5C8;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C5C8;
loc_0085C520:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085C60C;
    { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085C60C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085C618;
    ctx->gpr[20] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
loc_0085C548:
    { double a = ctx->fpr[10]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085C588;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x66A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[23] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_0085C588:
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x66AC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[10] + 0x24);
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    vm_write32(ctx->gpr[11] + 0xC, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x10, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x14, tmp); }
    goto loc_0085C1E0;
loc_0085C5C8:
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x30, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x2C, tmp); }
    goto loc_0085C520;
loc_0085C5D8:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x28);
    ctx->lr = 0x0085C5F4; june_evt_func_00860850(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C29C;
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    goto loc_0085C29C;
loc_0085C60C:
    ctx->gpr[20] = (int64_t)(int32_t)(0);
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    goto loc_0085C1E0;
loc_0085C618:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x66A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[8], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[20] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x24);
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    goto loc_0085C548;
loc_0085C654:
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[9]);
    goto loc_0085C154;
}

void june_evt_func_0085C660(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[24]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[14], 8); vm_write64(ctx->gpr[1] + 0x160, tmp); }
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[15], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    ctx->gpr[23] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[16], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    ctx->gpr[18] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[17], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    ctx->gpr[19] = ctx->gpr[5] | ctx->gpr[5];
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
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[0]);
    ctx->gpr[0] = vm_read16(ctx->gpr[4] + 0x0);
    ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[29]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C7C0;
loc_0085C720:
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
loc_0085C7C0:
    ctx->lr = 0x0085C7C4; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085C7D0; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[17] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[0] = vm_read8(ctx->gpr[29] + 0x1C);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D030;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[22] = ftmp; }
    ctx->fpr[14] = ctx->fpr[22];
    ctx->gpr[20] = vm_read32(ctx->gpr[2] + -0x6664);
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6660);
    ctx->gpr[15] = ctx->gpr[20] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6698); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[15] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[14] = ctx->gpr[20] + (int64_t)(0x20);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6694); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[18] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x84);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x666C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[17] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[0]);
    goto loc_0085CA6C;
loc_0085C860:
    ctx->gpr[21] = vm_read32(ctx->gpr[24] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6684); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[19] = ftmp; }
    ctx->fpr[23] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085CAB4;
loc_0085C878:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6690); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[30]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6674);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[22] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[26]));
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[14]));
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085C8E4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6670); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085C920; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085C960; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[14], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085C9A0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[24]));
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x665C);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6670); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[19], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085C9E4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[24]));
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6660);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CA20; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[17]));
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[17]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->lr = 0x0085CA38; june_evt_func_008B07F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC8);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0085CA4C; june_evt_func_008B080C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->lr = 0x0085CA5C; june_evt_func_008B0858(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085CA60:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D030;
loc_0085CA6C:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read16(ctx->gpr[26] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085CA60;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[24] = ctx->fpr[15];
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[15]));
    { double a = ctx->fpr[26]; double b = ctx->fpr[18]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085C860;
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[18], ctx->fpr[26]));
    ctx->gpr[21] = vm_read32(ctx->gpr[24] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x667C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[26] = ctx->fpr[18];
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6678); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->fpr[19] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085C878;
loc_0085CAB4:
    ctx->fpr[16] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[22]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6688); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->fpr[30] = ctx->fpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[20] = ctx->fpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[21]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[16]));
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[12]));
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[24]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { double a = ctx->fpr[27]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085D21C;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[22] = vm_read32(ctx->gpr[2] + -0x6674);
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->gpr[5] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->lr = 0x0085CB44; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
loc_0085CB4C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[16], 0, 32);
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->fpr[27] = ctx->fpr[30];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085CB90; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6684); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { double a = ctx->fpr[28]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085CBE4;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x665C);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085CBE0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085CBE4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[22]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[20]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085D254;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085CC2C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[22], ctx->fpr[27]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[15], 0, 32);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0085CC68; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085CC6C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ctx->fpr[25];
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[27] = ctx->fpr[24];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { double a = ctx->fpr[28]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085CCD0;
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085CCC8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
loc_0085CCD0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[0]));
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[6] = ctx->gpr[22] + (int64_t)(0x60);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[25] = ctx->gpr[22] | ctx->gpr[22];
    ctx->lr = 0x0085CD18; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[27]));
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { double a = ctx->fpr[27]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085CD6C;
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[6] = ctx->gpr[22] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085CD64; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
loc_0085CD6C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    ctx->gpr[6] = ctx->gpr[22] + (int64_t)(0x80);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085CDA4; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[21]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[25]));
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[24]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CE04; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->gpr[6] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CE4C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6668); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CE88; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085CEC8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[14], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085CF08; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[30];
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[24]));
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x665C);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6668); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[24]));
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CF48; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[24]));
    ctx->gpr[6] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->fpr[31] = ctx->fpr[9];
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CF88; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[22]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085D244;
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_0085CF9C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6660);
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->gpr[4] = ctx->gpr[16] | ctx->gpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->fpr[22] = ctx->fpr[16];
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085CFDC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[19], ctx->fpr[25]));
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x18);
    ctx->fpr[2] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[17]));
    ctx->lr = 0x0085CFF8; june_evt_func_008B07F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC8);
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x18);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->lr = 0x0085D00C; june_evt_func_008B080C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x18);
    ctx->lr = 0x0085D01C; june_evt_func_008B0858(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085CA6C;
    ;
loc_0085D030:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C720;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6688); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6698); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x94);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6684); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[0] | ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[9]));
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085D0A4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C720;
    ctx->gpr[10] = vm_read16(ctx->gpr[24] + 0x0);
    ctx->fpr[9] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D23C;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085D280;
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
loc_0085D0FC:
    { double a = ctx->fpr[28]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085D138;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x66A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[27], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x28);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
loc_0085D138:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085D158;
    goto loc_0085D1C4;
loc_0085D154:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
loc_0085D158:
    ctx->gpr[29] = ctx->gpr[9] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[28] + 0x10, tmp); }
    ctx->lr = 0x0085D1A0; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x28);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085D154;
    ctx->gpr[10] = vm_read16(ctx->gpr[24] + 0x0);
loc_0085D1C4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C720;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085D2B4;
loc_0085D1E4:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085C720;
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x24);
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x0085D214; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085C720;
loc_0085D21C:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[22] = vm_read32(ctx->gpr[2] + -0x6674);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->gpr[16] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[17], 0, 32);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    goto loc_0085CB4C;
loc_0085D23C:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x10);
    goto loc_0085D138;
loc_0085D244:
    ctx->fpr[27] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[24]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    goto loc_0085CF9C;
loc_0085D254:
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6680); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085D278; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085CC6C;
loc_0085D280:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x66A8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[28], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x24);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    goto loc_0085D0FC;
loc_0085D2B4:
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x10);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[18], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x28);
    ctx->gpr[4] = ctx->gpr[9] | ctx->gpr[9];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    ctx->lr = 0x0085D2D8; june_evt_func_00860090(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085D1E4;
}

void june_evt_func_0085D2E0(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x8, ctx->gpr[4]);
    return;
}

void june_evt_func_0085D2E8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x8);
    return;
}

void june_evt_func_0085D2F0(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[29]);
    ctx->gpr[28] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[0]);
    ctx->lr = 0x0085D348; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[25] + 0x0, ctx->gpr[9]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D5A4;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6654); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x74);
    { double a = ctx->fpr[30]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6658); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[27] = ctx->fpr[28];
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085D56C;
loc_0085D3A8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6650); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[31]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x664C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085D3EC; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[0]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085D45C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085D454; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
loc_0085D45C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[25], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085D4B4; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085D4F0; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0085D524; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x100);
    return;
loc_0085D56C:
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[29]));
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085D590; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    goto loc_0085D3A8;
loc_0085D5A4:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6658); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085D5D8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6648); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    goto loc_0085D45C;
}

void june_evt_func_0085D5E4(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x4);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D64C;
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0085D63C; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 56);
    ctx->gpr[3] = (uint64_t)0 - ctx->gpr[3];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[3], 1, 63);
loc_0085D64C:
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_0085D660(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D68C;
    ctx->lr = 0x0085D688; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085D68C:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0085D694; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0085D9E0(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[9] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x2C);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[5], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DA54;
    ctx->lr = 0x0085DA4C; june_evt_func_0086640C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0085DA54:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6620); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    { double a = ctx->fpr[12]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085DAC8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6624); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085DA9C; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0085DAC4; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085DAC8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_0085DAE8(ppu_context* ctx) {
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
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DB1C;
    ctx->lr = 0x0085DB18; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085DB1C:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DB34;
    ctx->lr = 0x0085DB30; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085DB34:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->lr = 0x0085DB3C; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->lr = 0x0085DB48; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0085DB54; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_0085E078(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x230, ctx->gpr[1]); ctx->gpr[1] += -0x230;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x1C8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x240, ctx->gpr[0]);
    ctx->gpr[11] = vm_read16(ctx->gpr[4] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x1D0, ctx->gpr[22]);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[11], 0, 28, 28, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x220, tmp); }
    vm_write64(ctx->gpr[1] + 0x1D8, ctx->gpr[23]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x228, tmp); }
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x1F8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x218, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x1E8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x208, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x210, ctx->gpr[30]);
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[22] = ctx->gpr[5] | ctx->gpr[5];
    vm_write32(ctx->gpr[4] + 0x4, ctx->gpr[0]);
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[21] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[26] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E0FC;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 28, 1, 31, 0, 0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write16(ctx->gpr[4] + 0x0, ctx->gpr[0]);
loc_0085E0FC:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x2C);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E144;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->lr = 0x0085E140; june_evt_func_00866694(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085E144:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6620); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085E250;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x78);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6624); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0085E1A8; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E3E4;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085E1E0; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E420;
loc_0085E1F0:
    ctx->lr = 0x0085E1F4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E438;
loc_0085E204:
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x6608);
    ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E438;
loc_0085E218:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E700;
loc_0085E22C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E758;
loc_0085E23C:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085E474;
    ;
loc_0085E250:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E388;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085E380;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0085E558;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    vm_write32(ctx->gpr[9] + 0x14, ctx->gpr[0]);
loc_0085E280:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[26];
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
loc_0085E28C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->lr = 0x0085E294; june_evt_func_0085D840(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E2D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E52C;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085E4E4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E624;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x226);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6614);
    ctx->lr = 0x0085E2CC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085E2D0:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E2EC;
    ctx->lr = 0x0085E2DC; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E658;
loc_0085E2EC:
    ctx->lr = 0x0085E2F0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E338;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65EC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E338;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E338;
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0085E32C; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E7B0;
loc_0085E338:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x240);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x220); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x228); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x230);
    return;
loc_0085E380:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E698;
loc_0085E388:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0085E280;
loc_0085E394:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E3D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E5D8;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085E6B4;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EB80;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x205);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6614);
    ctx->lr = 0x0085E3CC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085E3D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E2EC;
    goto loc_0085E28C;
loc_0085E3E4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0085E410; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E1F0;
loc_0085E420:
    ctx->lr = 0x0085E424; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E204;
loc_0085E438:
    ctx->lr = 0x0085E43C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E460;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x6608);
    ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E218;
loc_0085E460:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085E250;
loc_0085E474:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x30);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E250;
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0085E488; june_evt_func_0086E168(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E250;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->lr = 0x0085E49C; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->lr = 0x0085E4AC; june_evt_func_00866310(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EDF0;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085ED8C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ED8C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EE74;
loc_0085E4D4:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 17, 31, 0, 0);
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_0085E250;
loc_0085E4E4:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E5BC;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EB0C;
loc_0085E500:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65FC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    goto loc_0085E2D0;
loc_0085E52C:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E640;
loc_0085E53C:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x3E);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EAD8;
loc_0085E548:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0085E2D0;
loc_0085E558:
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[10] = vm_read32(ctx->gpr[8] + 0x14);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E394;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085E5A8;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    { int32_t _a=(int32_t)ctx->gpr[10], _b=(int32_t)ctx->gpr[11]; ctx->gpr[0] = (uint32_t)((_b==0||(_a==(int32_t)0x80000000&&_b==-1))?0:_a/_b); }
    ctx->gpr[0] = (uint64_t)((int64_t)(int32_t)ctx->gpr[11] * (int64_t)(int32_t)ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085E5A0;
    ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EAD0;
    ;
loc_0085E5A0:
    vm_write32(ctx->gpr[8] + 0x14, ctx->gpr[9]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
loc_0085E5A8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    goto loc_0085E280;
loc_0085E5BC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EAF0;
loc_0085E5C8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0085E2D0;
loc_0085E5D8:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085EBB4;
loc_0085E5E8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085ED54;
loc_0085E5F4:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x38);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    ctx->gpr[9] = vm_read16(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = (int64_t)(int16_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E3D0;
loc_0085E610:
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    goto loc_0085E28C;
loc_0085E624:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EBCC;
loc_0085E630:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    goto loc_0085E2D0;
loc_0085E640:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x21D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65F8);
    ctx->lr = 0x0085E650; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E53C;
loc_0085E658:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x40);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xB);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x0085E688; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0085E690; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E2EC;
loc_0085E698:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0085E9FC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x14, tmp); }
    goto loc_0085E280;
loc_0085E6B4:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EA84;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EBE8;
loc_0085E6D0:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65FC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1A0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x1A0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E610;
    goto loc_0085E3D0;
loc_0085E700:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EABC;
    ctx->lr = 0x0085E710; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EB28;
    ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EB28;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = ctx->gpr[10] - ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EB40;
loc_0085E748:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[10]);
    goto loc_0085E22C;
loc_0085E758:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EAA8;
    ctx->lr = 0x0085E768; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EB54;
    ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EB54;
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085EB6C;
loc_0085E7A0:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[10]);
    goto loc_0085E23C;
loc_0085E7B0:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65E8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x88);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x78);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x65E4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0085E7F0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E338;
    ctx->lr = 0x0085E804; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6620); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E834;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65E0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0085E834:
    ctx->lr = 0x0085E838; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085E844; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6620); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0085E880; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E8AC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x65DC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0085E8AC:
    ctx->lr = 0x0085E8B0; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085E8BC; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E338;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E338;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[30]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E338;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E338;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EE28;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085EE08;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EE4C;
loc_0085E958:
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->lr = 0x0085E964; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[12] = ctx->fpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1A0);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = (int64_t)(int32_t)(0x9B);
    ctx->gpr[6] = (int64_t)(int32_t)(0x80);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0x1A0);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-77);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[4], (uint32_t)tmp); }
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    ctx->lr = 0x0085E9AC; june_evt_func_00871840(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0x74);
    ctx->gpr[6] = (int64_t)(int32_t)(0xC);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x30);
    ctx->gpr[7] = (int64_t)(int32_t)(2);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[8] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->lr = 0x0085E9D4; june_evt_func_0083FE10(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0085E9E4; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(-32768);
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_0085E338;
loc_0085E9FC:
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[8] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E394;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6620); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EA70;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x1A0);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x1A0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x1B0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085EA68;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EA68;
    ctx->fpr[11] = ctx->fpr[0];
loc_0085EA68:
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[8] + 0x14, tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x34);
loc_0085EA70:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085E280;
loc_0085EA84:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085EC04;
loc_0085EA90:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E610;
    goto loc_0085E3D0;
loc_0085EAA8:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EC84;
loc_0085EAB0:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    goto loc_0085E23C;
loc_0085EABC:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EC20;
loc_0085EAC4:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    goto loc_0085E22C;
loc_0085EAD0:
    ctx->gpr[9] = ctx->gpr[0] | ctx->gpr[0];
    goto loc_0085E5A0;
loc_0085EAD8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x21E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65F0);
    ctx->lr = 0x0085EAE8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E548;
loc_0085EAF0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x215);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6604);
    ctx->lr = 0x0085EB00; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    goto loc_0085E5C8;
loc_0085EB0C:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x218);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6600);
    ctx->lr = 0x0085EB1C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    goto loc_0085E500;
loc_0085EB28:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = ctx->gpr[10] - ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085E748;
loc_0085EB40:
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    goto loc_0085E22C;
loc_0085EB54:
    ctx->gpr[10] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085E7A0;
loc_0085EB6C:
    ctx->gpr[0] = ctx->gpr[9] - ctx->gpr[11];
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    goto loc_0085E23C;
loc_0085EB80:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x3C);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085ED38;
loc_0085EB8C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085E3D0;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    goto loc_0085E28C;
loc_0085EBB4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1F2);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65F8);
    ctx->lr = 0x0085EBC4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E5E8;
loc_0085EBCC:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x222);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6600);
    ctx->lr = 0x0085EBDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    goto loc_0085E630;
loc_0085EBE8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1E8);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6600);
    ctx->lr = 0x0085EBF8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    goto loc_0085E6D0;
loc_0085EC04:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1E0);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6604);
    ctx->lr = 0x0085EC14; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    goto loc_0085EA90;
loc_0085EC20:
    ctx->lr = 0x0085EC24; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ECE8;
    ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ECE8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085ED70;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
loc_0085EC64:
    ctx->fpr[0] = fabs(ctx->fpr[0]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085EAC4;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    goto loc_0085E22C;
loc_0085EC84:
    ctx->lr = 0x0085EC88; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ED10;
    ctx->gpr[0] = vm_read16(ctx->gpr[29] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085ED10;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085ED80;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
loc_0085ECC8:
    ctx->fpr[0] = fabs(ctx->fpr[0]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085EAB0;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    goto loc_0085E23C;
loc_0085ECE8:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085ED6C;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085EC64;
loc_0085ED10:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085ED7C;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085ECC8;
loc_0085ED38:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1FC);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6600);
    ctx->lr = 0x0085ED48; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x38);
    goto loc_0085EB8C;
loc_0085ED54:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1F3);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65F4);
    ctx->lr = 0x0085ED64; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E5F4;
loc_0085ED6C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_0085ED70:
    ctx->fpr[0] = ctx->fpr[11];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085EC64;
loc_0085ED7C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_0085ED80:
    ctx->fpr[0] = ctx->fpr[11];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085ECC8;
loc_0085ED8C:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0xA);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085ED9C; june_evt_func_00D2D29C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = (int64_t)(int32_t)(0xA);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x0085EDB4; june_evt_func_00D2D29C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EE44;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EDE0;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[0] = ctx->gpr[28] | ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0085EDE0;
    ctx->gpr[0] = ctx->gpr[9] | ctx->gpr[9];
loc_0085EDE0:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    goto loc_0085E4D4;
loc_0085EDF0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x17E);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6614);
    ctx->lr = 0x0085EE00; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E4D4;
loc_0085EE08:
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x65D8);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085EE20; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E958;
loc_0085EE28:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6614);
    ctx->gpr[4] = (int64_t)(int32_t)(0x248);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->lr = 0x0085EE3C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E958;
loc_0085EE44:
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[3]);
    goto loc_0085E4D4;
loc_0085EE4C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->fpr[1] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1B8);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x65D4);
    ctx->lr = 0x0085EE6C; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085E958;
loc_0085EE74:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085EE84; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->fpr[31] = ppu_frsp(ctx->fpr[1]);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0085EE9C; june_evt_func_00D2BE80(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ppu_frsp(ctx->fpr[1]);
    { double a = ctx->fpr[31]; double b = ctx->fpr[1]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085EED8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085EEC8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085EEC8;
    ctx->fpr[0] = ctx->fpr[31];
loc_0085EEC8:
    ctx->gpr[26] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    ctx->gpr[30] = (int64_t)(int32_t)(1);
    goto loc_0085E4D4;
loc_0085EED8:
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    goto loc_0085E4D4;
}
