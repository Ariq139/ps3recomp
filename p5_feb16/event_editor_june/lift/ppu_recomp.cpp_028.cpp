
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

void june_evt_func_0093BB6C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093BB98; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x30);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(0x68);
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0093BBC0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x31);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093BC98;
loc_0093BBDC:
    ctx->gpr[29] = ctx->gpr[31] + (int64_t)(0xC);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x68);
    ctx->lr = 0x0093BBF0; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(0x54);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093BC04; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x54);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write32(ctx->gpr[31] + 0x60, ctx->gpr[0]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x0093BC20; june_evt_func_00947BAC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->lr = 0x0093BC30; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x2E54);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[10]);
    ctx->lr = 0x0093BC64; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_0093BC98:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E58);
    ctx->lr = 0x0093BCA0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BBDC;
}


void june_evt_func_0093BB2C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x8);
    ctx->lr = 0x0093BB48; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093BB54; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_0093DDE0(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[23]);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[29]);
    ctx->lr = 0x0093DE54; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[25] = ctx->gpr[31] | ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DE8C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DE8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0093DE8C:
    ctx->lr = 0x0093DE90; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093DE9C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
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
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[28] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0093DED8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DF04;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DE4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0093DF04:
    ctx->lr = 0x0093DF08; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093DF14; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0xC);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[7] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[8], ctx->fpr[10]));
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[7]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E288;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093E14C;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x60);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DFB0;
loc_0093DF9C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[11]);
loc_0093DFB0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[27] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[30]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E0F0;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E224;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E464;
loc_0093DFF0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E370;
loc_0093DFFC:
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
loc_0093E004:
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E00); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->gpr[27] = (int64_t)(int32_t)(1);
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x0093E060; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E074;
    ctx->gpr[27] = (int64_t)(int8_t)ctx->gpr[21];
loc_0093E074:
    ctx->lr = 0x0093E078; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E0B0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E0B0;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E174;
    ;
loc_0093E0B0:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0093E190;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x60);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E0F0;
    ctx->gpr[0] = ctx->gpr[25] + (int64_t)(0x20);
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x0093E0E4; june_evt_func_00834760(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E2B8;
loc_0093E0F0:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[27];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_0093E14C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E55C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E2D8;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x98);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->lr = 0x0093E16C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093DFB0;
loc_0093E174:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x2C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
    ;
loc_0093E190:
    ctx->gpr[0] = ctx->gpr[25] + (int64_t)(0xC);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093E1A8; june_evt_func_00947BAC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x60);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E1CC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0x54);
    ctx->lr = 0x0093E1C8; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093E1CC:
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E0F0;
    ctx->lr = 0x0093E1D8; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E0F0;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x64);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0x15);
    ctx->lr = 0x0093E214; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0093E21C; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093E0F0;
loc_0093E224:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E330;
loc_0093E238:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E004;
    ctx->lr = 0x0093E24C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E270;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E578;
loc_0093E270:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    goto loc_0093E004;
loc_0093E288:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x60);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DFB0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
    goto loc_0093DFB0;
loc_0093E2B8:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x60);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x14);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093E2D0; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093E0F0;
loc_0093E2D8:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x60);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DFB0;
    ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x10);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093E308; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x60);
    ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x18);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0xC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093E328; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093DFB0;
loc_0093E330:
    ctx->lr = 0x0093E334; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E358;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E670;
loc_0093E358:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[11]);
    goto loc_0093E238;
loc_0093E370:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[12], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[11], ctx->fpr[10], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[9], ctx->fpr[12], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[9], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093E3C4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DFFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093DFFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093DFFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093DFFC;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093DFFC;
    ctx->lr = 0x0093E420; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E454;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E454;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
    ctx->gpr[11] = ctx->gpr[11] | 0x2;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[11]);
loc_0093E454:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093DFFC;
loc_0093E464:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[9]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[12], ctx->fpr[11], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[0], ctx->fpr[10], 0, 0));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093E4B4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E554;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E554;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E554;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E554;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E554;
    ctx->lr = 0x0093E510; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E7F4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E7F4;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 0, 0, 29, 0, 0);
    ctx->gpr[11] = ctx->gpr[11] | 0x1;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[11]);
loc_0093E544:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093DFF0;
loc_0093E554:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0093DFF0;
loc_0093E55C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x60);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093DF9C;
    goto loc_0093DFB0;
loc_0093E578:
    ctx->lr = 0x0093E57C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E5AC;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DC8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
loc_0093E5AC:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E5D0;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
loc_0093E5D0:
    ctx->lr = 0x0093E5D4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E768;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0093E5E8:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E754;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x2C, tmp); }
loc_0093E60C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093E628;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093E630;
loc_0093E628:
    ctx->fpr[13] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
loc_0093E630:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093E648;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093E64C;
loc_0093E648:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x2C, tmp); }
loc_0093E64C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093E65C;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
loc_0093E65C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093E004;
loc_0093E670:
    ctx->lr = 0x0093E674; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E7C8;
    ctx->fpr[0] = ctx->fpr[29];
loc_0093E688:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E7B0;
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
loc_0093E6A0:
    ctx->lr = 0x0093E6A4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E78C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0093E6B8:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E740;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
loc_0093E6DC:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093E7E8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093E700;
    ctx->fpr[13] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
loc_0093E700:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093E718;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093E71C;
loc_0093E718:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
loc_0093E71C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093E72C;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
loc_0093E72C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093E238;
loc_0093E740:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0093E72C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_0093E6DC;
loc_0093E754:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0093E65C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_0093E60C;
loc_0093E768:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DCC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = -ctx->fpr[0];
    goto loc_0093E5E8;
loc_0093E78C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DCC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = -ctx->fpr[0];
    goto loc_0093E6B8;
loc_0093E7B0:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
    goto loc_0093E6A0;
loc_0093E7C8:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DC8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xA0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    goto loc_0093E688;
loc_0093E7E8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
    goto loc_0093E700;
loc_0093E7F4:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0093E544;
}


void june_evt_func_0093D068(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1C0, ctx->gpr[1]); ctx->gpr[1] += -0x1C0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[19]);
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x8C);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[31]);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[30]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x190, tmp); }
    ctx->gpr[25] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x1A0, tmp); }
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x1D0, ctx->gpr[0]);
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    ctx->lr = 0x0093D0F8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[20] = ctx->gpr[26] | ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D174;
loc_0093D10C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1D0);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x190); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1C0);
    return;
loc_0093D174:
    ctx->lr = 0x0093D178; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[13]));
    ctx->fpr[24] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[0] = vm_read16(ctx->gpr[26] + 0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093D758;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093D6CC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093D9E8;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DC0C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x1EC);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->lr = 0x0093D1EC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
loc_0093D1F8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[24]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093D10C;
    ctx->fpr[1] = ctx->fpr[29];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[23] = vm_read32(ctx->gpr[2] + -0x2E10);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[23] + (int64_t)(0xC);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[21] = ctx->gpr[23] | ctx->gpr[23];
    ctx->lr = 0x0093D244; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x9C);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xCC);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->lr = 0x0093D2AC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D970;
loc_0093D2BC:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xDC);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093D2D4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D940;
loc_0093D2E4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E0C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093D318; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D914;
loc_0093D328:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093D338; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D8E8;
loc_0093D348:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093D378; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D8BC;
loc_0093D388:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093D398; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D890;
loc_0093D3A8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093D3D8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D864;
loc_0093D3E8:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093D3F8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D838;
loc_0093D408:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093D438; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D808;
loc_0093D448:
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x0093D458; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D7D8;
loc_0093D468:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x7C);
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[21] | ctx->gpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093D4AC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093D5D8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[31] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[25] = ppu_fp_single(ppu_fdiv(ctx->fpr[23], ctx->fpr[0]));
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093D5E8;
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[29]));
    ctx->gpr[31] = ctx->gpr[20] + (int64_t)(0xC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[23] = vm_read32(ctx->gpr[2] + -0x2DEC);
loc_0093D520:
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[25]));
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->fpr[1] = ctx->fpr[26];
    ctx->lr = 0x0093D538; june_evt_func_00947778(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[30], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[29], ctx->fpr[28], 0, 0));
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[31]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { double a = ctx->fpr[27]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    { double a = ctx->fpr[11]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    { double a = ctx->fpr[9]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093D9A0;
loc_0093D5C8:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[12];
    ctx->fpr[27] = ctx->fpr[13];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D520;
loc_0093D5D8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
loc_0093D5E8:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[23];
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[2]));
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[13], ctx->fpr[30], 0, 0));
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(-1);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[2], ctx->fpr[29], ctx->fpr[28], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[24]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[24]));
    ctx->lr = 0x0093D628; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[2]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[3] = ctx->fpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[7] = (int64_t)(int32_t)(-1);
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[13], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[2], ctx->fpr[13], ctx->fpr[0], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[24]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[24]));
    ctx->lr = 0x0093D674; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E00); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[6] = ctx->gpr[21] | ctx->gpr[21];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093D6C4; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D10C;
loc_0093D6CC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ctx->fpr[26];
    ctx->fpr[2] = ctx->fpr[25];
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E48);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0093D700; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ctx->fpr[26];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[2]));
    ctx->lr = 0x0093D734; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093D738:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0093D1F8;
loc_0093D758:
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[2] = ctx->fpr[25];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E34);
    ctx->fpr[4] = ctx->fpr[0];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0093D794; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[31];
    ctx->fpr[4] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[2]));
    ctx->lr = 0x0093D7D0; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D738;
loc_0093D7D8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] + (int64_t)(0x10);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093D800; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D468;
loc_0093D808:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093D830; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D448;
loc_0093D838:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[21] + (int64_t)(0x10);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0093D85C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D408;
loc_0093D864:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[4] = ctx->gpr[23] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0093D888; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D3E8;
loc_0093D890:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[23] + (int64_t)(0x10);
    ctx->lr = 0x0093D8B4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D3A8;
loc_0093D8BC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ctx->gpr[23] + (int64_t)(0x10);
    ctx->lr = 0x0093D8E0; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D388;
loc_0093D8E8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[23] + (int64_t)(0x10);
    ctx->lr = 0x0093D90C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D348;
loc_0093D914:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ctx->gpr[23] + (int64_t)(0x10);
    ctx->lr = 0x0093D938; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D328;
loc_0093D940:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] + (int64_t)(0x10);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093D968; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D2E4;
loc_0093D970:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093D998; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D2BC;
loc_0093D9A0:
    { double a = ctx->fpr[13]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->fpr[2] = ctx->fpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->fpr[3] = ctx->fpr[12];
    ctx->fpr[4] = ctx->fpr[13];
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093D5C8;
    ctx->lr = 0x0093D9D8; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_0093D5C8;
loc_0093D9E8:
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2E24);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[21], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0093DA2C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DDA0;
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[1] + 0xBD, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC4);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[7]);
    vm_write8(ctx->gpr[1] + 0xBE, ctx->gpr[8]);
    vm_write8(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0xC8);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[4] = ctx->gpr[11] | ctx->gpr[11];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[7] + 0x0, ctx->gpr[10]);
    vm_write8(ctx->gpr[1] + 0xC7, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[0]);
    ctx->lr = 0x0093DAF4; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093DAF8:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[24]));
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093DB30; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[23]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[30]));
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093DD60;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    vm_write8(ctx->gpr[1] + 0xBD, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC4);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[7]);
    vm_write8(ctx->gpr[1] + 0xBE, ctx->gpr[10]);
    vm_write8(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xC8);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0xC7, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[0]);
    ctx->lr = 0x0093DBEC; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093DBF0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0093D1F8;
loc_0093DC0C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[1] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[2] = ctx->fpr[25];
    ctx->fpr[4] = ctx->fpr[0];
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E34);
    ctx->fpr[3] = ctx->fpr[29];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0093DC48; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[30]));
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[28]));
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x2E18);
    ctx->fpr[3] = ctx->fpr[29];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ctx->fpr[31];
    ctx->fpr[1] = ctx->fpr[27];
    ctx->lr = 0x0093DC80; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[4] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->lr = 0x0093DCBC; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[26];
    ctx->fpr[4] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[3] = ctx->fpr[29];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[28]));
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[30]));
    ctx->lr = 0x0093DCF8; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[27];
    ctx->fpr[3] = ctx->fpr[29];
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ctx->fpr[31];
    ctx->lr = 0x0093DD1C; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[4] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->lr = 0x0093DD58; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093D738;
loc_0093DD60:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[7]);
    ctx->lr = 0x0093DD98; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093DBF0;
loc_0093DDA0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[7]);
    ctx->lr = 0x0093DDD8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093DAF8;
}


void june_evt_func_0093BAC0(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x64, ctx->gpr[4]);
    return;
}


void june_evt_func_0093BAC8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x64);
    return;
}


void june_evt_func_0093CF14(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xE0, ctx->gpr[1]); ctx->gpr[1] += -0xE0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093CF40; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[6] = (int64_t)(int32_t)(0x297);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(0x7C);
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0093CF68; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x298);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093D058;
loc_0093CF84:
    ctx->gpr[29] = ctx->gpr[31] + (int64_t)(0x10);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x7C);
    ctx->lr = 0x0093CF98; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = (int64_t)(int32_t)(0x64);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093CFAC; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x64);
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write32(ctx->gpr[31] + 0x74, ctx->gpr[0]);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x0093CFC8; june_evt_func_00947DF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093CFD8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x2DF0);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    vm_write16(ctx->gpr[1] + 0x74, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[10]);
    ctx->lr = 0x0093D00C; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2E54);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    ctx->lr = 0x0093D024; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xF0);
    ctx->gpr[26] = _cs_26;
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[3]);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xE0);
    return;
loc_0093D058:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2DF4);
    ctx->lr = 0x0093D060; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093CF84;
}


void june_evt_func_0093BAE0(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x8);
    ctx->lr = 0x0093BAFC; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->lr = 0x0093BB08; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093BB14; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_0093E7FC(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x150, ctx->gpr[1]); ctx->gpr[1] += -0x150;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x148, tmp); }
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[23]);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[31]);
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    ctx->gpr[27] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x140, tmp); }
    ctx->gpr[23] = ctx->gpr[7] | ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[29]);
    ctx->lr = 0x0093E870; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[25] = ctx->gpr[31] | ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[26] = ctx->gpr[30] | ctx->gpr[30];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E8A8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DE8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0093E8A8:
    ctx->lr = 0x0093E8AC; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093E8B8; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[6]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0093E8F4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E920;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DE4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0093E920:
    ctx->lr = 0x0093E924; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093E930; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x10);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[8], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    ctx->fpr[7] = ppu_frsp(ctx->fpr[6]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[7]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093ED20;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093EBC4;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E9DC;
loc_0093E9B8:
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x4);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0xC);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x14);
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[11] + 0x1C);
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[9]);
loc_0093E9DC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[31]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EB68;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093ECBC;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EF6C;
loc_0093EA1C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EE78;
loc_0093EA28:
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
loc_0093EA30:
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x90);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[24], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E00); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x0093EA90; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EAA4;
    ctx->gpr[21] = (int64_t)(int32_t)(1);
loc_0093EAA4:
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[27] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[7] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0093EAC8; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093EADC;
    ctx->gpr[28] = (int64_t)(int8_t)ctx->gpr[21];
loc_0093EADC:
    ctx->lr = 0x0093EAE0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EB28;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EB28;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EE18;
loc_0093EB14:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EBEC;
    ;
loc_0093EB28:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_0093EC28;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EB68;
    ctx->gpr[0] = ctx->gpr[25] + (int64_t)(0x34);
    ctx->gpr[3] = ctx->gpr[3] + (int64_t)(0x24);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->lr = 0x0093EB5C; june_evt_func_00834760(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093ED60;
loc_0093EB68:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x160);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[28];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x120); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x128); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x140); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x148); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x150);
    return;
loc_0093EBC4:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F064;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093ED80;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x310);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->lr = 0x0093EBE4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093E9DC;
loc_0093EBEC:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F248;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093F270;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F378;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F528;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x3B6);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->lr = 0x0093EC20; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ;
loc_0093EC28:
    ctx->gpr[0] = ctx->gpr[25] + (int64_t)(0x10);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093EC40; june_evt_func_00947DF4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EC64;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0x64);
    ctx->lr = 0x0093EC60; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093EC64:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EB68;
    ctx->lr = 0x0093EC70; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EB68;
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x78);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0x15);
    ctx->lr = 0x0093ECAC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0093ECB4; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093EB68;
loc_0093ECBC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 31, 31, 0, 0);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093EE38;
loc_0093ECD0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 30, 30, 0, 0);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EA30;
    ctx->lr = 0x0093ECE4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093ED08;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F150;
loc_0093ED08:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[26], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    goto loc_0093EA30;
loc_0093ED20:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E9DC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x2C, tmp); }
    goto loc_0093E9DC;
loc_0093ED60:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x74);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = (int64_t)(int32_t)(0x10);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x24);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093ED78; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093EB68;
loc_0093ED80:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093E9DC;
    ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x14);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093EDB0; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x74);
    ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x1C);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0xC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093EDD0; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x74);
    ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x24);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x14);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093EDF0; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x74);
    ctx->gpr[3] = ctx->gpr[25] + (int64_t)(0x2C);
    ctx->gpr[4] = ctx->gpr[4] + (int64_t)(0x1C);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093EE10; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093E9DC;
loc_0093EE18:
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x38, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x40, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x34, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x3C, tmp); }
    goto loc_0093EB14;
loc_0093EE38:
    ctx->lr = 0x0093EE3C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EE60;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F080;
loc_0093EE60:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[11]);
    goto loc_0093ECD0;
loc_0093EE78:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[12], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[11], ctx->fpr[10], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[9], ctx->fpr[12], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[9], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093EECC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EA28;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093EA28;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093EA28;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093EA28;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093EA28;
    ctx->lr = 0x0093EF28; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EF5C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093EF5C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 0, 0, 29, 0, 0);
    ctx->gpr[11] = ctx->gpr[11] | 0x2;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[11]);
loc_0093EF5C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093EA28;
loc_0093EF6C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[10]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[13] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[9]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[31], ctx->fpr[12], ctx->fpr[11], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[25], ctx->fpr[0], ctx->fpr[10], 0, 0));
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x0093EFBC; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F05C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[28]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F05C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F05C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[27]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F05C;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F05C;
    ctx->lr = 0x0093F018; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F578;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DD8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F578;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 0, 0, 29, 0, 0);
    ctx->gpr[11] = ctx->gpr[11] | 0x1;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[11]);
loc_0093F04C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093EA1C;
loc_0093F05C:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0093EA1C;
loc_0093F064:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x74);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093E9B8;
    goto loc_0093E9DC;
loc_0093F080:
    ctx->lr = 0x0093F084; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F2E8;
    ctx->fpr[0] = ctx->fpr[29];
loc_0093F098:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F2D0;
    ctx->gpr[21] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = (int64_t)(int32_t)(0);
loc_0093F0B0:
    ctx->lr = 0x0093F0B4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F354;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0093F0C8:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F340;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x38, tmp); }
loc_0093F0EC:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093F56C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093F110;
    ctx->fpr[13] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x34, tmp); }
loc_0093F110:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093F128;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093F12C;
loc_0093F128:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x38, tmp); }
loc_0093F12C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093F13C;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x3C, tmp); }
loc_0093F13C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093ECD0;
loc_0093F150:
    ctx->lr = 0x0093F154; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F184;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DC8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[6]);
loc_0093F184:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F1A8;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x3C, tmp); }
loc_0093F1A8:
    ctx->lr = 0x0093F1AC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093F31C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0093F1C0:
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093F308;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x40, tmp); }
loc_0093F1E4:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093F200;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093F208;
loc_0093F200:
    ctx->fpr[13] = ctx->fpr[12];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x3C, tmp); }
loc_0093F208:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093F220;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093F224;
loc_0093F220:
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x40, tmp); }
loc_0093F224:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093F234;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x34, tmp); }
loc_0093F234:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DDC);
    ctx->gpr[0] = (int64_t)(int32_t)(3);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[26], 0, 32);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093EA30;
loc_0093F248:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[12], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[12], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    goto loc_0093EC28;
loc_0093F270:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[6] = ctx->gpr[31] + (int64_t)(0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[8] = ctx->gpr[31] + (int64_t)(0x24);
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[11]);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[6]);
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[11], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[11], 0, 0));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    goto loc_0093EC28;
loc_0093F2D0:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[21] = (int64_t)(int32_t)(1);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[22] = (int64_t)(int32_t)(1);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x34, tmp); }
    goto loc_0093F0B0;
loc_0093F2E8:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DC8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    goto loc_0093F098;
loc_0093F308:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0093F234;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_0093F1E4;
loc_0093F31C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DCC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[6]);
    ctx->fpr[0] = -ctx->fpr[0];
    goto loc_0093F1C0;
loc_0093F340:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[22];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0093F13C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_0093F0EC;
loc_0093F354:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2DCC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[6]);
    ctx->fpr[0] = -ctx->fpr[0];
    goto loc_0093F0C8;
loc_0093F378:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[9], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[0], 8, 24, 31, 0, 0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[9], 48, 56);
    ctx->gpr[11] = ctx->gpr[11] - ctx->gpr[8];
    ctx->gpr[8] = (int64_t)(int32_t)ctx->gpr[8];
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[8]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[0], 48, 56);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    ctx->gpr[10] = ctx->gpr[10] - ctx->gpr[7];
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    ctx->gpr[7] = (int64_t)(int32_t)ctx->gpr[7];
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[7]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    ctx->gpr[5] = ppc_rlw64(ctx->gpr[9], 0, 24, 31, 0, 0);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[10]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->gpr[6] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 56, 56);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 56, 56);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->gpr[6] = ctx->gpr[6] - ctx->gpr[5];
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[9];
    ctx->fpr[8] = ppu_frsp(ctx->fpr[6]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->fpr[5] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[8], ctx->fpr[12], ctx->fpr[13], 0, 0));
    ctx->fpr[7] = ppu_frsp(ctx->fpr[6]);
    ctx->fpr[8] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[8], ctx->fpr[11], ctx->fpr[13], 0, 0));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[5], 0); memcpy(&ctx->fpr[5], &iv, 8); }
    ctx->fpr[4] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[12], ctx->fpr[0], 0, 0));
    ctx->fpr[7] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[11], ctx->fpr[0], 0, 0));
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[5]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    ctx->fpr[9] = ppu_frsp(ctx->fpr[13]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[6]);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[8], 0); memcpy(&ctx->fpr[8], &iv, 8); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA8);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[4], 0); memcpy(&ctx->fpr[4], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[7], 0); memcpy(&ctx->fpr[7], &iv, 8); }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[10] = ppu_frsp(ctx->fpr[6]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[6], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[6], 8); ctx->fpr[6] = (double)iv; }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[5], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[4], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[7] = vm_read32(ctx->gpr[1] + 0xA8);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[8], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0xA4);
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xAC);
    ctx->gpr[7] = ppc_rlw64(ctx->gpr[7], 24, 0, 7, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[7], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[6]);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 16, 0, 15, 0, 0);
    ctx->fpr[6] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[12], ctx->fpr[9], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[11], ctx->fpr[9], 0, 0));
    ctx->gpr[7] = ctx->gpr[7] | ctx->gpr[9];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[12], ctx->fpr[10], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[11], ctx->fpr[10], 0, 0));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[6], 0); memcpy(&ctx->fpr[6], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[13], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[12], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[6], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[11] = vm_read32(ctx->gpr[1] + 0xA8);
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xAC);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 24, 0, 7, 0, 0);
    ctx->gpr[10] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 16, 0, 15, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[8] = vm_read32(ctx->gpr[1] + 0xA4);
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[10], 8, 0, 23, 0, 0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[6], (uint32_t)tmp); }
    ctx->gpr[11] = ctx->gpr[11] | ctx->gpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xA4);
    ctx->gpr[8] = ctx->gpr[8] | ctx->gpr[7];
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[9] = ctx->gpr[9] | ctx->gpr[11];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 8, 0, 23, 0, 0);
    ctx->gpr[10] = ctx->gpr[10] | ctx->gpr[8];
    ctx->gpr[0] = ctx->gpr[0] | ctx->gpr[9];
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[10]);
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[0]);
    goto loc_0093EC28;
loc_0093F528:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[8] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[0], ctx->fpr[10], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[11], 0, 0));
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[9], ctx->fpr[10], 0, 0));
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[9], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x24, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x28, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
    goto loc_0093EC28;
loc_0093F56C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x34, tmp); }
    goto loc_0093F110;
loc_0093F578:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0093F04C;
}


void june_evt_func_0093BCA8(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x1D0, ctx->gpr[1]); ctx->gpr[1] += -0x1D0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[19]);
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x8C);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    vm_write64(ctx->gpr[1] + 0x178, ctx->gpr[31]);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[19], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[30]);
    ctx->gpr[23] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x190, tmp); }
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x198, tmp); }
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x1A0, tmp); }
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x1A8, tmp); }
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x1B0, tmp); }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x1B8, tmp); }
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[24]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x1C8, tmp); }
    ctx->gpr[24] = ctx->gpr[6] | ctx->gpr[6];
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[0]);
    ctx->lr = 0x0093BD3C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[20] = ctx->gpr[30] | ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093BDBC;
loc_0093BD50:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1E0);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x190); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x198); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1A8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1B8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x1C8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x1D0);
    return;
loc_0093BDBC:
    ctx->lr = 0x0093BDC0; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[21] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[11]));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E50); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[24] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[13]));
    ctx->fpr[22] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->fpr[26] = ctx->fpr[24];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093C434;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093C340;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093C75C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093CB80;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E5C);
    ctx->gpr[4] = (int64_t)(int32_t)(0x536);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->lr = 0x0093BE38; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
loc_0093BE48:
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[24]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[24]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093BD50;
    ctx->fpr[1] = ctx->fpr[30];
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x2E10);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xAC);
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[22] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0093BE90; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x9C);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xCC);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->lr = 0x0093BEF8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C6E4;
loc_0093BF08:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0xDC);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093BF20; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C6B4;
loc_0093BF30:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E0C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093BF64; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C688;
loc_0093BF74:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093BF84; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C65C;
loc_0093BF94:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093BFC4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C630;
loc_0093BFD4:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093BFE4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C604;
loc_0093BFF4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093C024; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C5D8;
loc_0093C034:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093C044; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C5AC;
loc_0093C054:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[31], ctx->fpr[0], 0, 0));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x0093C084; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C57C;
loc_0093C094:
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x0093C0A4; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C54C;
loc_0093C0B4:
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x7C);
    { uint32_t tmp = vm_read32(ctx->gpr[23] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093C0F8; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093C228;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E08); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xF0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    ctx->fpr[31] = ctx->fpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0xF0);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[25] = ppu_fp_single(ppu_fdiv(ctx->fpr[23], ctx->fpr[0]));
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093C238;
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[28]));
    ctx->gpr[31] = ctx->gpr[20] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[29] = (int64_t)(int32_t)(0);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[22] = vm_read32(ctx->gpr[2] + -0x2DFC);
    ;
loc_0093C170:
    ctx->fpr[29] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[25]));
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(1);
    ctx->fpr[1] = ctx->fpr[29];
    ctx->lr = 0x0093C188; june_evt_func_009478C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[12], ctx->fpr[13], ctx->fpr[27], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[28], ctx->fpr[26], 0, 0));
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
    { double a = ctx->fpr[31]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { double a = ctx->fpr[30]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[0]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    { double a = ctx->fpr[11]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    { double a = ctx->fpr[9]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093C714;
loc_0093C218:
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ctx->fpr[12];
    ctx->fpr[30] = ctx->fpr[13];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C170;
loc_0093C228:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
loc_0093C238:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[23];
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[2]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x34); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[13], ctx->fpr[27], 0, 0));
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(-1);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[2], ctx->fpr[28], ctx->fpr[26], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[24]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[24]));
    ctx->lr = 0x0093C27C; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[2]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[3] = ctx->fpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[7] = (int64_t)(int32_t)(-1);
    ctx->fpr[1] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[1], ctx->fpr[13], ctx->fpr[0], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[8] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[9] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[2], ctx->fpr[13], ctx->fpr[0], 0, 0));
    ctx->fpr[1] = ppu_fp_single(ppu_fsub(ctx->fpr[1], ctx->fpr[24]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[24]));
    ctx->lr = 0x0093C2C8; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[11]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E00); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E04); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093C318; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0093C338; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BD50;
loc_0093C340:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->fpr[2] = ctx->fpr[24];
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E48);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x2C);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0093C374; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[22], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[2], ctx->fpr[24], 0, 0));
    ctx->lr = 0x0093C3AC; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[30], ctx->fpr[2], ctx->fpr[24], 0, 0));
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0093C3DC; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[22]));
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[29]));
    ctx->lr = 0x0093C40C; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093C410:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0093BE48;
loc_0093C434:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[2] = ctx->fpr[24];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->fpr[4] = ctx->fpr[0];
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E34);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0093C470; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[22], ctx->fpr[2], ctx->fpr[26], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    ctx->fpr[4] = ctx->fpr[0];
    ctx->fpr[1] = ctx->fpr[25];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[24]));
    ctx->lr = 0x0093C4B0; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[22], ctx->fpr[2], ctx->fpr[26], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[4] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[24]));
    ctx->lr = 0x0093C4EC; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[26], ctx->fpr[22]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[4] = ctx->fpr[0];
    ctx->fpr[3] = ctx->fpr[31];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[2]));
    ctx->lr = 0x0093C528; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0093BE48;
loc_0093C54C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[21] + (int64_t)(0x10);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093C574; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C0B4;
loc_0093C57C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[4] = ctx->gpr[21] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093C5A4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C094;
loc_0093C5AC:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[22] + (int64_t)(4);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0093C5D0; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C054;
loc_0093C5D8:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[4] = ctx->gpr[27] + (int64_t)(4);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->lr = 0x0093C5FC; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C034;
loc_0093C604:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[27] + (int64_t)(4);
    ctx->lr = 0x0093C628; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BFF4;
loc_0093C630:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ctx->gpr[27] + (int64_t)(4);
    ctx->lr = 0x0093C654; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BFD4;
loc_0093C65C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[27] + (int64_t)(4);
    ctx->lr = 0x0093C680; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BF94;
loc_0093C688:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[4] = ctx->gpr[27] + (int64_t)(4);
    ctx->lr = 0x0093C6AC; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BF74;
loc_0093C6B4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[4] = ctx->gpr[21] + (int64_t)(0x10);
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093C6DC; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BF30;
loc_0093C6E4:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[4] = ctx->gpr[21] + (int64_t)(0x10);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->fpr[4] = ctx->fpr[2];
    ctx->fpr[3] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[3]));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x0093C70C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093BF08;
loc_0093C714:
    { double a = ctx->fpr[13]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->fpr[2] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
    ctx->fpr[3] = ctx->fpr[12];
    ctx->fpr[4] = ctx->fpr[13];
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093C218;
    ctx->lr = 0x0093C74C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    goto loc_0093C218;
loc_0093C75C:
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x7C);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2E24);
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0093C7A0; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x2C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093CED4;
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[1] + 0xBD, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC4);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[7]);
    vm_write8(ctx->gpr[1] + 0xBE, ctx->gpr[8]);
    vm_write8(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0xC8);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[4] = ctx->gpr[11] | ctx->gpr[11];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[7] + 0x0, ctx->gpr[10]);
    vm_write8(ctx->gpr[1] + 0xC7, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[0]);
    ctx->lr = 0x0093C868; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093C86C:
    ctx->fpr[29] = ppu_fp_single(ppu_fsub(ctx->fpr[22], ctx->fpr[28]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[29], ctx->fpr[0], ctx->fpr[24], 0, 0));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093C8A8; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093CE94;
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[1] + 0xBD, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC4);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[7]);
    vm_write8(ctx->gpr[1] + 0xBE, ctx->gpr[8]);
    vm_write8(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0xC8);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[4] = ctx->gpr[11] | ctx->gpr[11];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[7] + 0x0, ctx->gpr[10]);
    vm_write8(ctx->gpr[1] + 0xC7, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[0]);
    ctx->lr = 0x0093C968; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093C96C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[29], ctx->fpr[0], ctx->fpr[24], 0, 0));
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093C9A4; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093CE54;
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    vm_write8(ctx->gpr[1] + 0xBD, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC4);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[7]);
    vm_write8(ctx->gpr[1] + 0xBE, ctx->gpr[8]);
    vm_write8(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = ctx->gpr[1] + (int64_t)(0xC8);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[7], 0, 32);
    ctx->gpr[4] = ctx->gpr[11] | ctx->gpr[11];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[10] = vm_read32(ctx->gpr[11] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[7] + 0x0, ctx->gpr[10]);
    vm_write8(ctx->gpr[1] + 0xC7, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[0]);
    ctx->lr = 0x0093CA64; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093CA68:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[22]));
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093CAA0; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[23]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[23]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xFF; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093CE14;
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0xBC);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    vm_write8(ctx->gpr[1] + 0xBD, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xC4);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[7]);
    vm_write8(ctx->gpr[1] + 0xBE, ctx->gpr[10]);
    vm_write8(ctx->gpr[1] + 0xBC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0xC8);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0xC7, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0xBF, ctx->gpr[0]);
    ctx->lr = 0x0093CB5C; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093CB60:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[27], ctx->fpr[30]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[30]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[24] = ftmp; }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    goto loc_0093BE48;
loc_0093CB80:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E4C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[2] = ctx->fpr[24];
    ctx->fpr[4] = ctx->fpr[0];
    ctx->gpr[28] = vm_read32(ctx->gpr[2] + -0x2E34);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[22], 0, 32);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->lr = 0x0093CBBC; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E44); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[29]));
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[26]));
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x2E18);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[23] | ctx->gpr[23];
    ctx->fpr[28] = ppu_fp_single(ppu_fsub(ctx->fpr[22], ctx->fpr[26]));
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->fpr[1] = ctx->fpr[27];
    ctx->lr = 0x0093CBF8; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x30); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x2C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[4] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->lr = 0x0093CC34; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E40); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[28], ctx->fpr[2], ctx->fpr[24], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[4] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[29]));
    ctx->lr = 0x0093CC70; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[27];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->lr = 0x0093CC94; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x28); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x24); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[4] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->lr = 0x0093CCD0; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[28], ctx->fpr[0], ctx->fpr[24], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[4] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[29]));
    ctx->fpr[2] = ctx->fpr[28];
    ctx->lr = 0x0093CD10; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[27];
    ctx->fpr[2] = ctx->fpr[30];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->lr = 0x0093CD34; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x20); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[4] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->lr = 0x0093CD70; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[22]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[1] = ctx->fpr[25];
    ctx->fpr[4] = ctx->fpr[0];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[2], ctx->fpr[26]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[29]));
    ctx->lr = 0x0093CDAC; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[27];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[9] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[2] = ctx->fpr[30];
    ctx->lr = 0x0093CDD0; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[2] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[29]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->fpr[1] = ctx->fpr[25];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[3] = ctx->fpr[31];
    ctx->gpr[8] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[7] = (int64_t)(int32_t)(0xFF);
    ctx->fpr[4] = ctx->fpr[13];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[10] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->lr = 0x0093CE0C; june_evt_func_00868D40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C410;
loc_0093CE14:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[7]);
    ctx->lr = 0x0093CE4C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093CB60;
loc_0093CE54:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[7]);
    ctx->lr = 0x0093CE8C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093CA68;
loc_0093CE94:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[7]);
    ctx->lr = 0x0093CECC; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C96C;
loc_0093CED4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[7], 8, 24, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2DF8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[7], 16, 16, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[7], 24, 8, 31, 0, 0);
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[6] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[7]);
    ctx->lr = 0x0093CF0C; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093C86C;
}


void june_evt_func_0093BAD0(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x78, ctx->gpr[4]);
    return;
}


void june_evt_func_0093BAD8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x78);
    return;
}


void june_evt_func_0093B6E4(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0xC);
    ctx->lr = 0x0093B700; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->lr = 0x0093B70C; june_evt_func_00866928(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x14);
    ctx->lr = 0x0093B718; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093B724; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}


void june_evt_func_0093AD80(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x2C);
    ctx->gpr[21] = ctx->gpr[4] | ctx->gpr[4];
    ctx->gpr[10] = vm_read32(ctx->gpr[4] + 0x30);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[22]);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[29]);
    ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[27] = ctx->gpr[7] | ctx->gpr[7];
    ctx->gpr[24] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[23] = ctx->gpr[6] | ctx->gpr[6];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AEE0;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 4, 0, 27, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x14);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[10], 2, 0, 29, 0, 0);
    ctx->gpr[3] = vm_read32(ctx->gpr[4] + 0x8);
    vm_write32(ctx->gpr[4] + 0x2C, ctx->gpr[10]);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x0093AE20; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AE60;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B478;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B4AC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B44C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0xAD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E94);
    ctx->lr = 0x0093AE5C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093AE60:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AE7C;
loc_0093AE68:
    ctx->lr = 0x0093AE6C; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B31C;
loc_0093AE7C:
    ctx->gpr[25] = (int64_t)(int32_t)(1);
loc_0093AE80:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AF30;
loc_0093AE90:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[25];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_0093AEE0:
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B470;
    ctx->gpr[11] = vm_read32(ctx->gpr[4] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B490;
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B42C;
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B45C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x2E98);
    ctx->gpr[4] = (int64_t)(int32_t)(0xDB);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2E94);
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093AF1C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AE90;
loc_0093AF30:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0093AF44; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AE90;
    ctx->lr = 0x0093AF58; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B4DC;
loc_0093AF6C:
    ctx->lr = 0x0093AF70; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[20] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093AF7C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093AF88; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E9C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B4BC;
loc_0093AF9C:
    ctx->lr = 0x0093AFA0; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093AFB0; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[10]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[13]));
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[7] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[6] = ctx->gpr[19] | ctx->gpr[19];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[10]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093B010; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B108;
    ctx->lr = 0x0093B028; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AE90;
    ctx->lr = 0x0093B03C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AE90;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2E88);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093AE90;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[20];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[28];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AE90;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x88); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AE90;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[24];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AE90;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AE90;
loc_0093B108:
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = 0x0093B110; june_evt_func_0086E3D4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[26] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x38); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0093B150;
    ctx->fpr[12] = ctx->fpr[10];
loc_0093B150:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x3C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2E84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0093B168; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    ctx->lr = 0x0093B184; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093B1B0;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
loc_0093B1B0:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[12]));
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093B204;
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x94);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093B204;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_0093B204:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xA0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[12], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[11], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x2080 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2E80);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[3] = vm_read32(ctx->gpr[1] + 0xA0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[12], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[4] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[11], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[4] = (int64_t)(int32_t)ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[6] = vm_read32(ctx->gpr[1] + 0xA0);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->lr = 0x0093B268; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x2E80);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    ctx->gpr[4] = ctx->gpr[21] + (int64_t)(0x14);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = ctx->gpr[19] | ctx->gpr[19];
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[29] + 0x20);
    ctx->gpr[3] = (int64_t)(int32_t)(5);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-24);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[11]);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write32(ctx->gpr[29] + 0x88, ctx->gpr[21]);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[11]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0093B2E8; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->lr = 0x0093B2FC; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[0]);
    ctx->lr = 0x0093B314; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093AE90;
loc_0093B31C:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x48);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x17);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[25] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0093B354; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[5], 4, 0, 27, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[5], 2, 0, 29, 0, 0);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(0x17);
    ctx->gpr[8] = vm_read32(ctx->gpr[9] + 0xC);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->lr = 0x0093B3A0; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[7] = (int64_t)(int32_t)(0x17);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x48);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->lr = 0x0093B3D0; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x2C);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[5], 4, 0, 27, 0, 0);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[5], 2, 0, 29, 0, 0);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[11];
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = (int64_t)(int32_t)(0x17);
    ctx->gpr[8] = vm_read32(ctx->gpr[9] + 0xC);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[2] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->lr = 0x0093B41C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0093B424; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093AE80;
loc_0093B42C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    ctx->gpr[9] = vm_read8(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093B470;
loc_0093B440:
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[9]);
    goto loc_0093AE80;
loc_0093B44C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093AE60;
loc_0093B45C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B440;
    ;
loc_0093B470:
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    goto loc_0093AE80;
loc_0093B478:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093AE68;
    goto loc_0093AE7C;
loc_0093B490:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 16, 31, 0, 0);
    ctx->gpr[9] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093B440;
    ctx->gpr[25] = (int64_t)(int32_t)(0);
    goto loc_0093AE80;
loc_0093B4AC:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x2C);
    vm_write8(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093AE60;
loc_0093B4BC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2E8C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
    goto loc_0093AF9C;
loc_0093B4DC:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2E90);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xB0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0093AF6C;
}


void june_evt_func_00939C34(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2EF0);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[28]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x45);
    ctx->gpr[3] = (int64_t)(int32_t)(0x18);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x00939C70; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x46);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00939CE4;
loc_00939C90:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x18);
    ctx->lr = 0x00939C9C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] + (int64_t)(8);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x00939CB4; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[30] = _cs_30;
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00939CE4:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2EEC);
    ctx->lr = 0x00939CEC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00939C90;
}


void june_evt_func_00939C10(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x00939C20; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
}


void june_evt_func_0093A2CC(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[25]);
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x70);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[29]);
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[4] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[23] = ctx->gpr[7] | ctx->gpr[7];
    ctx->lr = 0x0093A334; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[24] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A394;
loc_0093A34C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[11];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_0093A394:
    ctx->lr = 0x0093A398; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A3C8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EC8);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_0093A3C8:
    ctx->lr = 0x0093A3CC; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093A3D8; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093A3E4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A414;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EC4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_0093A414:
    ctx->lr = 0x0093A418; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0093A424; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 30, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A5B0;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[26];
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    ctx->fpr[10] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[29]));
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 31, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[9] = ppu_frsp(ctx->fpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[13]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->fpr[9] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[30]));
    ctx->fpr[8] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[11] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[31]));
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[9], ctx->fpr[8]));
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[11], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A744;
loc_0093A4C4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 30, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A4D8;
loc_0093A4D0:
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    goto loc_0093A34C;
loc_0093A4D8:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[10], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093A510; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A4D0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A4D0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A4D0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A4D0;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A4D0;
    ctx->lr = 0x0093A56C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A59C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EBC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A59C;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[10] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[10]);
loc_0093A59C:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EB8);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093A34C;
loc_0093A5B0:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->lr = 0x0093A5B8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A5E8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EB4);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_0093A5E8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[31] = ppu_fp_single(ppu_fdiv(ctx->fpr[13], ctx->fpr[0]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A830;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A4D0;
    ctx->lr = 0x0093A614; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A8A8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EB0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A8A8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A670;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093A670;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
loc_0093A670:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EB8);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[11];
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_0093A34C;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ED0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093A8C0;
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
loc_0093A6A8:
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093A820;
loc_0093A6B4:
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
loc_0093A6B8:
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A6DC;
    ctx->gpr[4] = ctx->gpr[24] + (int64_t)(8);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[5] = (int64_t)(int32_t)(8);
    ctx->lr = 0x0093A6D8; june_evt_func_008347AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093A6DC:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[31];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A34C;
    ctx->lr = 0x0093A6EC; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[31];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A34C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[23], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[29] + 0x14);
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0x18);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x0093A730; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x0093A738; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = (int64_t)(int8_t)ctx->gpr[31];
    goto loc_0093A34C;
loc_0093A744:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[10], ctx->fpr[13], 0, 0));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[11]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0093A77C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A818;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A818;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A818;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[30]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A818;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A818;
    ctx->lr = 0x0093A7D8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A8D4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EBC);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A8D4;
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[10] = ctx->gpr[10] | 0x1;
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[10]);
loc_0093A808:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EB8);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    goto loc_0093A4C4;
loc_0093A818:
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x4);
    goto loc_0093A4C4;
loc_0093A820:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093A6B8;
    goto loc_0093A6B4;
loc_0093A830:
    ctx->lr = 0x0093A834; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A894;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x2EB0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A894;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = (int64_t)(int32_t)(0);
    { double a = ctx->fpr[31]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A670;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[11] = (int64_t)(int32_t)(1);
    { double a = ctx->fpr[13]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093A670;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0xC, tmp); }
    goto loc_0093A670;
loc_0093A894:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[10] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[10]);
    goto loc_0093A34C;
loc_0093A8A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[11] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[0]);
    goto loc_0093A34C;
loc_0093A8C0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093A6A8;
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[29] + 0x8, tmp); }
    goto loc_0093A6A8;
loc_0093A8D4:
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x4);
    goto loc_0093A808;
}


void june_evt_func_00939CF4(ppu_context* ctx) {
    PPU_MM_SNAP
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
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[25]);
    ctx->gpr[25] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x128, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[31]);
    ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x120, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    ctx->lr = 0x00939D60; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { double a = ctx->fpr[29]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[24] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[25] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093A23C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->fpr[21] = ctx->fpr[23];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
loc_00939DA0:
    { double a = ctx->fpr[31]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0093A234;
    ctx->fpr[22] = ctx->fpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
loc_00939DB0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ECC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[26] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[25]));
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x2EDC);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x2ED8);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[4], 0, 32);
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[21]));
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[27] = ctx->fpr[21];
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[21]));
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939E14; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x10);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939E40; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[24]));
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x20);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[21]));
    ctx->fpr[29] = ctx->fpr[30];
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[24], ctx->fpr[21]));
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x00939E7C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x30);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939EA8; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x40);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939EE0; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x50);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939F0C; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x60);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939F44; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->gpr[9] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[26] + (int64_t)(0x70);
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00939F70; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[23], ctx->fpr[0], ctx->fpr[21], 0, 0));
    { uint32_t tmp = vm_read32(ctx->gpr[24] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[23], ctx->fpr[13], ctx->fpr[21], 0, 0));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { double a = ctx->fpr[31]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ctx->fpr[31];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093A00C;
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x2ED4);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x94);
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x2EDC);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[8] + 0x0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[8] + 0x0);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[11]);
    ctx->lr = 0x0093A000; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_0093A00C:
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[31]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093A04C;
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x0093A040; june_evt_func_00861B54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0093A04C:
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[23], ctx->fpr[29]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { double a = ctx->fpr[0]; double b = ctx->fpr[26]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0093A0C8;
    ctx->gpr[10] = vm_read32(ctx->gpr[2] + -0x2EDC);
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0x94);
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x2ED4);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->fpr[1] = ctx->fpr[26];
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    vm_write32(ctx->gpr[4] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x0);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x9C);
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[8] + 0x0);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[6] = ctx->gpr[31] | ctx->gpr[31];
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[11] = vm_read32(ctx->gpr[8] + 0x0);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[11]);
    ctx->lr = 0x0093A0C4; june_evt_func_00861970(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093A0C8:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[13]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2ED0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[27]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093A110; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A2A0;
loc_0093A120:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093A13C; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A274;
loc_0093A14C:
    { uint32_t tmp = vm_read32(ctx->gpr[25] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[27]));
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093A174; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0093A248;
loc_0093A184:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[21]));
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->lr = 0x0093A1A0; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0093A1D8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0x80);
    ctx->lr = 0x0093A1D4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0093A1D8:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[24] = _cs_24;
    ctx->gpr[25] = _cs_25;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
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
loc_0093A234:
    ctx->fpr[22] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[10]));
    goto loc_00939DB0;
loc_0093A23C:
    ctx->fpr[23] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x2EE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    goto loc_00939DA0;
loc_0093A248:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0x80);
    ctx->lr = 0x0093A26C; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093A184;
loc_0093A274:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0x80);
    ctx->lr = 0x0093A298; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093A14C;
loc_0093A2A0:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x80); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[5] = ctx->fpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x84); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->fpr[3] = ctx->fpr[1];
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[4] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[2], ctx->fpr[4]));
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0x80);
    ctx->lr = 0x0093A2C4; june_evt_func_008EFA0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0093A120;
}


void june_evt_func_00939C00(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write32(ctx->gpr[3] + 0x14, ctx->gpr[4]);
    return;
}


void june_evt_func_00939C08(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x14);
    return;
}


void june_evt_func_00939A68(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x2EF8);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[6] = (int64_t)(int32_t)(0x42);
    ctx->gpr[3] = (int64_t)(int32_t)(0x18);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x00939AA0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x43);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00939B08;
loc_00939ABC:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = (int64_t)(int32_t)(0x18);
    ctx->lr = 0x00939AC8; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[29] = _cs_29;
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[9]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
loc_00939B08:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x2EF4);
    ctx->lr = 0x00939B10; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00939ABC;
}


void june_evt_func_00939A44(ppu_context* ctx) {
    PPU_MM_SNAP
    vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x00939A54; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = ctx->gpr[0];
    return;
}

