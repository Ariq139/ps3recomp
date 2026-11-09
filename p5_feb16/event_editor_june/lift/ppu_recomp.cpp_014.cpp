
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

void june_evt_func_0085D6AC(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x6644);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[30]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x22);
    ctx->gpr[3] = (int64_t)(int32_t)(0xC);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->lr = 0x0085D6E8; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(0x23);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D7AC;
loc_0085D704:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0xC);
    ctx->lr = 0x0085D710; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085D788;
    ctx->lr = 0x0085D734; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[9] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write8(ctx->gpr[1] + 0x7A, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x7B, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[11]);
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x78, ctx->gpr[9]);
    vm_write8(ctx->gpr[1] + 0x79, ctx->gpr[9]);
    ctx->lr = 0x0085D764; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[3]);
    ctx->lr = 0x0085D770; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085D780; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x0, tmp); }
loc_0085D788:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[28] = _cs_28;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
loc_0085D7AC:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6640);
    ctx->lr = 0x0085D7B4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085D704;
}

void june_evt_func_0085D7C0(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x40, ctx->gpr[4]);
        return;
}

void june_evt_func_0085D7C8(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x40);
        return;
}

void june_evt_func_0085DB70(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x200, ctx->gpr[1]); ctx->gpr[1] += -0x200;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x1D0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x1E8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x1F0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x1F8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x210, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x1D8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x1E0, ctx->gpr[28]);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0085DB9C; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x661C);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x44);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = (int64_t)(int32_t)(0x40);
    ctx->lr = 0x0085DBC0; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DFD4;
loc_0085DBD0:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x44);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0085DBE0; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write16(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[9]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085DC1C;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DED4;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6614);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5A);
    ctx->lr = 0x0085DC14; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DC44;
loc_0085DC1C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[11]);
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[10]);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[8]);
loc_0085DC44:
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0085DC58; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(9);
    vm_write16(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    ctx->lr = 0x0085DC68; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x6610);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write16(ctx->gpr[1] + 0x94, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    ctx->lr = 0x0085DCA4; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[3]);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    ctx->lr = 0x0085DCD4; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read16(ctx->gpr[31] + 0x0);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[3]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085DE3C;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DD4C;
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085DE80;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DE80;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085DD78;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E024;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x662C);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1C0);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DD44; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DD78;
loc_0085DD4C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085DFB0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6634);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DD74; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0085DD78:
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085DD8C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[28]);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[11]);
    ctx->lr = 0x0085DDB4; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x2C, ctx->gpr[3]);
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085DF00;
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[11] = (int64_t)(int32_t)(0xD);
    ctx->gpr[0] = (ctx->gpr[9] & 64) ? 0 : (ctx->gpr[0] << (ctx->gpr[9] & 63));
    ctx->gpr[0] = ctx->gpr[0] & ctx->gpr[11];
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DF00;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085DDE8; june_evt_func_00866964(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[3]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DF10;
    ;
loc_0085DE00:
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0085DF40;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DF40;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DFEC;
loc_0085DE18:
    ctx->lr = 0x0085DE1C; june_evt_func_0086E4A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x0085DE2C; june_evt_func_008B2B08(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x660C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x20, tmp); }
loc_0085DE3C:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x20);
    vm_write32(ctx->gpr[31] + 0x3C, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x210);
    vm_write32(ctx->gpr[31] + 0x34, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x38, ctx->gpr[11]);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x200);
    return;
loc_0085DE80:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085DEB0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6638);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DEA8; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DD78;
loc_0085DEB0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x663C);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DECC; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DD78;
loc_0085DED4:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x1C, tmp); }
    goto loc_0085DC44;
loc_0085DF00:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[31] + 0x30, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085DE00;
loc_0085DF10:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085DF90;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6634);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DF38; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DE18;
loc_0085DF40:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085DF70;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x663C);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DF68; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DE18;
loc_0085DF70:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6638);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DF88; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DE18;
loc_0085DF90:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6630);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DFA8; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DE18;
loc_0085DFB0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6630);
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[5] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085DFCC; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DD78;
loc_0085DFD4:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6618);
    ctx->gpr[4] = (int64_t)(int32_t)(0x41);
    ctx->lr = 0x0085DFE4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DBD0;
loc_0085DFEC:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0085E050;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x662C);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1C0);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085E01C; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DE18;
loc_0085E024:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6628);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1C0);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085E048; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DD78;
loc_0085E050:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[1] = ctx->fpr[0];
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6628);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write64(ctx->gpr[1] + 0x1C0, tmp); }
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0x1C0);
    ctx->gpr[4] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->lr = 0x0085E070; june_evt_func_00D3033C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085DE18;
}

void june_evt_func_0085F4A8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xD0, ctx->gpr[1]); ctx->gpr[1] += -0xD0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[29]);
    ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x65B8);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[30]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)(0x38);
    ctx->gpr[6] = (int64_t)(int32_t)(0x46);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[0]);
    ctx->lr = 0x0085F4E4; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F620;
loc_0085F4F4:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x38);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x0085F504; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F560;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0085F604;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65B0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5A);
    ctx->lr = 0x0085F538; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085F584;
loc_0085F548:
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x18);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0085F598;
loc_0085F558:
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[0]);
    goto loc_0085F5A0;
loc_0085F560:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0xC);
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[11]);
loc_0085F578:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x14);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0x7; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085F548;
loc_0085F584:
    ctx->gpr[0] = (int64_t)(int32_t)(8);
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x18);
    { uint64_t a = (uint32_t)ctx->gpr[0]; uint64_t b = (uint64_t)0xB; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0085F558;
loc_0085F598:
    ctx->gpr[0] = (int64_t)(int32_t)(0xC);
    vm_write32(ctx->gpr[31] + 0x1C, ctx->gpr[0]);
loc_0085F5A0:
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[4] = (int64_t)(int32_t)(0x40);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0085F5B4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write16(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    ctx->lr = 0x0085F5C8; june_evt_func_0086D76C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x1C);
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xE0);
    vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[3]);
    vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[9]);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[29] = _cs_29;
    ctx->gpr[28] = _cs_28;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xD0);
    return;
loc_0085F604:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x14, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x10, tmp); }
    goto loc_0085F578;
loc_0085F620:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x65B4);
    ctx->gpr[4] = (int64_t)(int32_t)(0x47);
    ctx->lr = 0x0085F630; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0085F4F4;
}

void june_evt_func_00860090(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[30]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[31]);
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xC8, tmp); }
    ctx->gpr[31] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0xD0, tmp); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[26]);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->lr = 0x008600F0; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[6] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[6], ctx->fpr[7]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[29] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[9]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    { double a = ctx->fpr[10]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086014C;
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[6]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x658C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[7] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x18, tmp); }
loc_0086014C:
    { double a = ctx->fpr[7]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00860158;
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[30] + 0x18, tmp); }
loc_00860158:
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860348;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[30]));
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6584);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6588); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x008601AC; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6580); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { double a = ctx->fpr[31]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086024C;
    { double a = ctx->fpr[31]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008604B4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_008601E0:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x008601FC; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[31]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086026C;
    { double a = ctx->fpr[31]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008604E4;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_0086021C:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00860238; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[31]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860294;
    goto loc_0086030C;
loc_0086024C:
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00860268; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086026C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00860290; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00860294:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086030C;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0086030C;
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[9]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { double a = ctx->fpr[12]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008604C0;
loc_008602F4:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x00860308; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0086030C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_00860348:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[31]));
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x6584);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[26], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6588); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00860390; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6580); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { double a = ctx->fpr[30]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086054C;
    { double a = ctx->fpr[30]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008604F0;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x008603E0; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[30]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860520;
loc_008603F0:
    { double a = ctx->fpr[30]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0086056C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
loc_00860400:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0086041C; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[30]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086030C;
    ;
loc_00860430:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0086030C;
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { double a = ctx->fpr[12]; double b = ctx->fpr[29]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0086030C;
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[9]));
    ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[12] = ppu_fp_single(ppu_fdiv(ctx->fpr[12], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { double a = ctx->fpr[12]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008602F4;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    goto loc_008602F4;
loc_008604B4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    goto loc_008601E0;
loc_008604C0:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[10], ctx->fpr[12]));
    { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[9]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[29]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[8]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    goto loc_008602F4;
loc_008604E4:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[31], ctx->fpr[0]));
    goto loc_0086021C;
loc_008604F0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00860510; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[30]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008603F0;
loc_00860520:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00860544; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00860430;
loc_0086054C:
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x24);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x00860564; june_evt_func_0086D2A0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00860520;
loc_0086056C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    goto loc_00860400;
}

void june_evt_func_00860578(ppu_context* ctx) {
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
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008605AC;
    ctx->lr = 0x008605A8; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008605AC:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x24);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008605C4;
    ctx->lr = 0x008605C0; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008605C4:
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008605DC;
    ctx->lr = 0x008605D8; june_evt_func_0086D6FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008605DC:
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->lr = 0x008605E4; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}

void june_evt_func_00860850(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0x120, ctx->gpr[1]); ctx->gpr[1] += -0x120;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[26], 8); vm_write64(ctx->gpr[1] + 0xF0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x118, tmp); }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[20], 8); vm_write64(ctx->gpr[1] + 0xC0, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[21], 8); vm_write64(ctx->gpr[1] + 0xC8, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[22], 8); vm_write64(ctx->gpr[1] + 0xD0, tmp); }
    ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[23], 8); vm_write64(ctx->gpr[1] + 0xD8, tmp); }
    ctx->gpr[27] = ctx->gpr[5] | ctx->gpr[5];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[24], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[25], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0xF8, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x110, tmp); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    vm_write32(ctx->gpr[1] + 0x128, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[0]);
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[26] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { double a = ctx->fpr[26]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860B8C;
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[28] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    ctx->fpr[22] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[6] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[25] = ftmp; }
    ctx->lr = 0x00860900; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860C10;
loc_00860910:
    ctx->lr = 0x00860914; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00860920; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[30] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    ctx->lr = 0x0086095C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860BF0;
loc_0086096C:
    ctx->lr = 0x00860970; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0086097C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x4);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[3]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    ctx->fpr[24] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860D50;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6580); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[13]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860C30;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6588); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00860A10; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 56);
    ctx->gpr[3] = (uint64_t)0 - ctx->gpr[3];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 1, 63);
loc_00860A24:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
loc_00860A34:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00860A4C; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860A60;
    ctx->gpr[27] = (int64_t)(int32_t)(1);
loc_00860A60:
    ctx->fpr[9] = ppu_fp_single(ppu_fsub(ctx->fpr[29], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[7] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[25]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[10]));
    ctx->fpr[10] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[26]));
    { double a = ctx->fpr[10]; double b = ctx->fpr[20]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00860D34;
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[9], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[23]));
loc_00860A8C:
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->fpr[12] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[23]));
    ctx->fpr[11] = ctx->fpr[23];
    ctx->fpr[29] = ctx->fpr[9];
    ctx->fpr[28] = ctx->fpr[30];
    ctx->fpr[13] = ctx->fpr[7];
    ctx->fpr[8] = ctx->fpr[23];
    ctx->fpr[31] = ctx->fpr[27];
loc_00860AB0:
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860AD0;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[10]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860E58;
    ;
loc_00860AD0:
    { double a = ctx->fpr[30]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860AF8;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860FA8;
loc_00860AF8:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { double a = ctx->fpr[21]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_00860CB4;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860B24;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6570);
    ctx->gpr[4] = (int64_t)(int32_t)(0x118);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6548);
    ctx->lr = 0x00860B20; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00860B24:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860F78;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    if ((((ctx->cr >> 14) & 1) == 0)) goto loc_0086117C;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = -ctx->fpr[0];
loc_00860B44:
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    ;
loc_00860B60:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[11]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[8]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860B80;
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x658C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00860B80:
    { double a = ctx->fpr[11]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00860B8C;
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00860B8C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x130);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x128);
    ctx->gpr[26] = _cs_26;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[27] = _cs_27;
    ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[20], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC8); memcpy(&ctx->fpr[21], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[22], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD8); memcpy(&ctx->fpr[23], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[24], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[25], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[26], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF8); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x110); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x118); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x120);
    return;
loc_00860BF0:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6558);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0086096C;
loc_00860C10:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x655C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_00860910;
loc_00860C30:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6588); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    { double a = ctx->fpr[29]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00860FFC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[29], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
loc_00860C48:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00860C74; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[29]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 56);
    ctx->gpr[3] = (uint64_t)0 - ctx->gpr[3];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 1, 63);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860A24;
    { double a = ctx->fpr[29]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ctx->fpr[20];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00860CA8;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[29], ctx->fpr[0]));
loc_00860CA8:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    goto loc_00860A34;
loc_00860CB4:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860EE0;
    ctx->lr = 0x00860CC8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008610A8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x653C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008610A8;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861184;
    { double a = ctx->fpr[27]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00861204;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00861260;
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[10]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_00860D34:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[20], ctx->fpr[10]));
    ctx->fpr[10] = ctx->fpr[20];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[23]));
    goto loc_00860A8C;
loc_00860D50:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861008;
loc_00860D60:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[29]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
loc_00860D68:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6580); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    { double a = ctx->fpr[25]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6588); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[23] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008610D8;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { float ftmp = (float)ctx->fpr[23]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00860DBC; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 56);
    ctx->gpr[3] = (uint64_t)0 - ctx->gpr[3];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 1, 63);
loc_00860DD0:
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[25], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[23]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
loc_00860DE0:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00860DF8; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860E0C;
    ctx->gpr[27] = (int64_t)(int32_t)(1);
loc_00860E0C:
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[27]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[10]));
    ctx->fpr[8] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[26]));
    { double a = ctx->fpr[8]; double b = ctx->fpr[20]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00861150;
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[11], ctx->fpr[13]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[23]));
loc_00860E34:
    ctx->fpr[27] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[0]));
    ctx->fpr[7] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[23]));
    ctx->fpr[9] = ctx->fpr[23];
    ctx->fpr[25] = ctx->fpr[11];
    ctx->fpr[28] = ctx->fpr[24];
    ctx->fpr[12] = ctx->fpr[31];
    ctx->fpr[10] = ctx->fpr[23];
    ctx->fpr[13] = ctx->fpr[27];
    goto loc_00860AB0;
loc_00860E58:
    { double a = ctx->fpr[13]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860AD0;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[8]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860AD0;
    ctx->lr = 0x00860E78; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860E9C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6550);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860EA8;
loc_00860E9C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
loc_00860EA8:
    ctx->lr = 0x00860EAC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860AF8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x654C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860AF8;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00860AF8;
loc_00860EE0:
    ctx->lr = 0x00860EE4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861078;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6550);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861078;
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008611BC;
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->lr = 0x00860F1C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ctx->fpr[31];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860F4C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6540);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_00860F4C:
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[26], ctx->fpr[29]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[0], ctx->fpr[13], ctx->fpr[30], 0, 0));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
loc_00860F58:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write32(ctx->gpr[9] + 0xC, ctx->gpr[0]);
    goto loc_00860B60;
loc_00860F78:
    ctx->fpr[0] = -ctx->fpr[21];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[26], ctx->fpr[25]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[12]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[11], 0, 0));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_00860FA8:
    { double a = ctx->fpr[24]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860AF8;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[7], ctx->fpr[11]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860AF8;
    ctx->lr = 0x00860FC8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860AF8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x654C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860AF8;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00860AF8;
loc_00860FFC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->fpr[0] = ctx->fpr[20];
    goto loc_00860C48;
loc_00861008:
    { double a = ctx->fpr[28]; double b = ctx->fpr[30]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00860D60;
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[29]));
    { double a = ctx->fpr[30]; double b = ctx->fpr[31]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008610D0;
    { double a = ctx->fpr[22]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008610D0;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[22], ctx->fpr[25]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[24]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008610D0;
    ctx->lr = 0x00861044; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860D68;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6554);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[21] = ppu_frsp(ctx->fpr[0]);
    goto loc_00860D68;
loc_00861078:
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x20);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 0, 30, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[9]);
    vm_write32(ctx->gpr[11] + 0xC, ctx->gpr[0]);
    goto loc_00860B60;
loc_008610A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
    goto loc_00860B60;
loc_008610D0:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6574); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    goto loc_00860D68;
loc_008610D8:
    { double a = ctx->fpr[25]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0086116C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->fpr[0] = ctx->fpr[20];
loc_008610E8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[28], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[26] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[6] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00861110; june_evt_func_0086D860(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { double a = ctx->fpr[25]; double b = ctx->fpr[27]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 56);
    ctx->gpr[3] = (uint64_t)0 - ctx->gpr[3];
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 1, 63);
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00860DD0;
    { double a = ctx->fpr[25]; double b = ctx->fpr[23]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[0] = ctx->fpr[20];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00861144;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[25], ctx->fpr[0]));
loc_00861144:
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    goto loc_00860DE0;
loc_00861150:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[20], ctx->fpr[8]));
    ctx->fpr[8] = ctx->fpr[20];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[13], ctx->fpr[0]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[26]));
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[23]));
    goto loc_00860E34;
loc_0086116C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6578); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[25], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x657C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    goto loc_008610E8;
loc_0086117C:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x1C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    goto loc_00860B44;
loc_00861184:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x24);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861230;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861288;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[27];
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    goto loc_00860B60;
loc_008611BC:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->lr = 0x008611C4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[13] = ctx->fpr[31];
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008611F4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6544);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
loc_008611F4:
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[26], ctx->fpr[25]));
    ctx->fpr[11] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[13], ctx->fpr[0], ctx->fpr[30], 0, 0));
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860F58;
loc_00861204:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[10]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00861274;
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[10]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_00861230:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008612B8;
    ctx->fpr[11] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_00861260:
    ctx->fpr[11] = ctx->fpr[9];
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_00861274:
    ctx->fpr[11] = ctx->fpr[8];
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_00861288:
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x10); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[8], ctx->fpr[12]));
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008612D0;
    ctx->fpr[11] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_008612B8:
    ctx->fpr[11] = ctx->fpr[9];
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
loc_008612D0:
    ctx->fpr[11] = ctx->fpr[8];
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
    { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x18, tmp); }
    goto loc_00860B60;
}

void june_evt_func_008612E8(ppu_context* ctx) {
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
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        ctx->fpr[10] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
        ctx->fpr[7] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
        { double a = ctx->fpr[7]; double b = ctx->fpr[10]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008613DC;
        ctx->fpr[6] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[9]));
        { double a = ctx->fpr[6]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008613DC;
        { uint32_t tmp = vm_read32(ctx->gpr[5] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[9]));
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { double a = ctx->fpr[11]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008613DC;
        { uint32_t tmp = vm_read32(ctx->gpr[5] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[8]));
        { uint32_t tmp = vm_read32(ctx->gpr[4] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { double a = ctx->fpr[12]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008613DC;
        { double a = ctx->fpr[10]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00861414;
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008613E8;
loc_0086137C:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6538); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008613DC;
loc_00861390:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008613DC;
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[12]));
        { double a = ctx->fpr[0]; double b = ctx->fpr[7]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008613B8;
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[7], ctx->fpr[12]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
loc_008613B8:
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        ctx->gpr[0] = (int64_t)(int32_t)(1);
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[13]));
        { double a = ctx->fpr[0]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008613E0;
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[12]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
        return;
loc_008613DC:
        ctx->gpr[0] = (int64_t)(int32_t)(0);
loc_008613E0:
        ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[0];
        return;
loc_008613E8:
        ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[12], ctx->fpr[0]));
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x4, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6538); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0xC, tmp); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00861390;
        goto loc_008613DC;
loc_00861414:
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[10]));
        { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x0, tmp); }
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[0]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[3] + 0x8, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[3] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { double a = ctx->fpr[0]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0086137C;
        goto loc_008613E8;
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ;
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->lr = ctx->gpr[0];
        return;
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ;
        ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        return;
        { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
        { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
        ctx->fpr[31] = ctx->fpr[1];
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
        ctx->gpr[31] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[30] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[28] = ctx->gpr[7] | ctx->gpr[7];
        ctx->gpr[24] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[23] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[22] = ctx->gpr[9] | ctx->gpr[9];
        ctx->gpr[26] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[25] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[27] = ctx->gpr[7] | ctx->gpr[7];
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086169C;
loc_008614E8:
        { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008616D4;
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[9] = ftmp; }
        ctx->fpr[6] = ppu_fp_single(ppu_fadd(ctx->fpr[9], ctx->fpr[0]));
        { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[31] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[8] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[30] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[5] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[10] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[31] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[30] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[7] = ftmp; }
        { float ftmp = (float)ctx->fpr[10]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
        { float ftmp = (float)ctx->fpr[8]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
        { double a = ctx->fpr[13]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { float ftmp = (float)ctx->fpr[2]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
        { float ftmp = (float)ctx->fpr[5]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
        { float ftmp = (float)ctx->fpr[3]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
        { float ftmp = (float)ctx->fpr[7]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
        { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861660;
        ctx->fpr[4] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[0]));
        { double a = ctx->fpr[10]; double b = ctx->fpr[4]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861660;
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
        { double a = ctx->fpr[0]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861660;
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[10], ctx->fpr[8]));
        { double a = ctx->fpr[0]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        { uint32_t a = (ctx->cr >> (31 - 28)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00861660;
        { double a = ctx->fpr[13]; double b = ctx->fpr[9]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008615B8;
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[9], ctx->fpr[13]));
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6534); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { float ftmp = (float)ctx->fpr[9]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
        ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
        ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
        ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[12], ctx->fpr[0]));
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[7], ctx->fpr[0]));
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[7], ctx->fpr[13], ctx->fpr[3], 0, 0));
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_008615B8:
        { double a = ctx->fpr[10]; double b = ctx->fpr[11]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008615F0;
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[11], ctx->fpr[10]));
        { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6534); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
        ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[8]));
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
        ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[13], ctx->fpr[0]));
        ctx->fpr[12] = ppu_fp_single(ppu_fmul(ctx->fpr[8], ctx->fpr[0]));
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[0]));
        ctx->fpr[13] = ppu_fp_single(ppu_fmadd_core(ctx->fpr[5], ctx->fpr[13], ctx->fpr[2], 0, 0));
        { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
        { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
loc_008615F0:
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
        { double a = ctx->fpr[0]; double b = ctx->fpr[6]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00861618;
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[6], ctx->fpr[13]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
        ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[7], ctx->fpr[0]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
loc_00861618:
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
        { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
        ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[12]));
        { double a = ctx->fpr[0]; double b = ctx->fpr[4]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00861640;
        ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[4], ctx->fpr[13]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
        ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[12]));
        ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[5], ctx->fpr[0]));
        { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
loc_00861640:
        ctx->fpr[1] = ctx->fpr[31];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[23], 0, 32);
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[22], 0, 32);
        ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x80);
        ;
        ;
loc_00861660:
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
        { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
        return;
loc_0086169C:
        ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[4] = (int64_t)(int32_t)(9);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[5] = (int64_t)(int32_t)(1);
        ;
        ;
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[4] = (int64_t)(int32_t)(4);
        ctx->gpr[5] = (int64_t)(int32_t)(5);
        ctx->gpr[6] = (int64_t)(int32_t)(4);
        ctx->gpr[7] = (int64_t)(int32_t)(5);
        ;
        ;
        goto loc_008614E8;
loc_008616D4:
        ctx->fpr[1] = ctx->fpr[31];
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[24], 0, 32);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[23], 0, 32);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[26], 0, 32);
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[25], 0, 32);
        ctx->gpr[8] = ppc_rldicl(ctx->gpr[22], 0, 32);
        ;
        ;
        goto loc_00861660;
}





void june_evt_func_00861D40(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write32(ctx->gpr[3] + 0x28, ctx->gpr[4]);
        return;
}

void june_evt_func_00862B7C(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x190, ctx->gpr[1]); ctx->gpr[1] += -0x190;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x1A0, ctx->gpr[0]);
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[27], 8); vm_write64(ctx->gpr[1] + 0x168, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[28], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[29], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
    ctx->gpr[29] = ctx->gpr[28] | ctx->gpr[28];
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x180, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x188, tmp); }
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[25]);
    ctx->lr = 0x00862BD8; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = vm_read32(ctx->gpr[2] + -0x6528);
    ctx->gpr[4] = (int64_t)(int32_t)(0xB);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[3] = (int64_t)(int32_t)(0x30);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = (int64_t)(int32_t)(0x3B);
    ctx->gpr[27] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00862C00; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863284;
loc_00862C10:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x30);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
    ctx->gpr[25] = ctx->gpr[1] + (int64_t)(0x78);
    ctx->lr = 0x00862C24; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = (int64_t)(int32_t)(0x10);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0xB0);
    vm_write32(ctx->gpr[30] + 0x24, ctx->gpr[0]);
    ctx->lr = 0x00862C40; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[21] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[3] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[4] = (int64_t)(int32_t)(0x24);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00862C58; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[4] = (int64_t)(int32_t)(8);
    ctx->lr = 0x00862C68; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[10] = ctx->gpr[1] + (int64_t)(0x80);
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xBC);
    ctx->gpr[10] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[10] + 0x0, ctx->gpr[8]);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x0);
    vm_write32(ctx->gpr[11] + 0x0, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862F44;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[9] = vm_read16(ctx->gpr[28] + 0xC);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[13], &tmp, 8); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[7]);
    vm_write32(ctx->gpr[1] + 0x8C, ctx->gpr[0]);
    vm_write16(ctx->gpr[1] + 0x84, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6518); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862F20;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x78);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6508); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ctx->gpr[0] | 0x8;
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    vm_write16(ctx->gpr[1] + 0x78, ctx->gpr[0]);
loc_00862CFC:
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x88);
    ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    ctx->lr = 0x00862D14; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00862D2C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[29] + 0x24, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086327C;
    vm_write32(ctx->gpr[29] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x0);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[29]);
loc_00862D50:
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x98);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[4] = (int64_t)(int32_t)(0x18);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[31]);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862D6C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[9]);
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
    ctx->lr = 0x00862D88; june_evt_func_00861E0C(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[1] + 0x98);
    ctx->fpr[28] = ctx->fpr[1];
    ctx->gpr[6] = (int64_t)(int32_t)(8);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862E0C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6530); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x652C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_00862DB0:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862F3C;
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = (int64_t)(int32_t)(0x262);
    if ((((ctx->cr >> 6) & 1) == 1)) goto loc_00862F14;
    if ((((ctx->cr >> 27) & 1) == 0)) goto loc_00862F18;
loc_00862DD4:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6524);
    ctx->lr = 0x00862DDC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862DE0:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x14);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862DB0;
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xE0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[1] + 0xE0);
    ctx->gpr[6] = ctx->gpr[9] + (int64_t)(8);
loc_00862E0C:
    ctx->fpr[0] = ctx->fpr[28];
    ctx->gpr[11] = ctx->gpr[1] + (int64_t)(0xE0);
    ctx->gpr[8] = (int64_t)(int32_t)((uint32_t)0x80 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = ctx->gpr[8] | 0x1;
    ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[11], (uint32_t)tmp); }
    ctx->gpr[5] = vm_read32(ctx->gpr[1] + 0xE0);
    ctx->gpr[5] = ctx->gpr[5] + (int64_t)(8);
    ctx->gpr[5] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->lr = 0x00862E48; june_evt_func_0086EA3C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[5] = ctx->gpr[24] | ctx->gpr[24];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[30] + 0x8, ctx->gpr[3]);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = (int64_t)(int32_t)(5);
    vm_write32(ctx->gpr[1] + 0x8C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[0]);
    ctx->lr = 0x00862E78; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862E8C; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00862E9C; june_evt_func_0086E50C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00862EAC; june_evt_func_0086DE40(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[0] = vm_read64(ctx->gpr[9] + 0x90);
    vm_write64(ctx->gpr[30] + 0x10, ctx->gpr[0]);
loc_00862EBC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x1A0);
    ctx->gpr[3] = ctx->gpr[23] | ctx->gpr[23];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x168); memcpy(&ctx->fpr[27], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x170); memcpy(&ctx->fpr[28], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x178); memcpy(&ctx->fpr[29], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x180); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x188); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x190);
    return;
loc_00862F14:
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862DD4;
loc_00862F18:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[29]));
    goto loc_00862DE0;
loc_00862F20:
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x78);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x651C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 1, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write16(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    goto loc_00862CFC;
loc_00862F3C:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    goto loc_00862DE0;
loc_00862F44:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_008632B8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
loc_00862F64:
    ctx->gpr[27] = vm_read32(ctx->gpr[28] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    { float ftmp = (float)ctx->fpr[1]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00862EBC;
    ctx->fpr[30] = ctx->fpr[0];
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x98);
    ctx->gpr[24] = ctx->gpr[1] + (int64_t)(0x88);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6508); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6530); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6514); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x64E4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[27] = ftmp; }
    goto loc_0086300C;
loc_00862FA8:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008631C4;
loc_00862FB0:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6524);
    ctx->gpr[4] = (int64_t)(int32_t)(0x8B);
    ctx->lr = 0x00862FC0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00862FC4:
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->lr = 0x00862FD0; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[29] + 0x24, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008630DC;
loc_00862FE8:
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x14);
    vm_write32(ctx->gpr[29] + 0x28, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[29]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[29]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008630F8;
loc_0086300C:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863194;
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00862FA8;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863200;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862FB0;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008632C4;
loc_00863038:
    ctx->gpr[3] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[4] = (int64_t)(int32_t)(0x18);
    ctx->lr = 0x00863044; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x90); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = (int64_t)(int32_t)(5);
    vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[9]);
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[4] = ctx->gpr[20] | ctx->gpr[20];
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    vm_write32(ctx->gpr[1] + 0x9C, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xA8, ctx->gpr[11]);
    vm_write16(ctx->gpr[1] + 0xA4, ctx->gpr[10]);
    vm_write32(ctx->gpr[1] + 0xAC, ctx->gpr[8]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ;
loc_00863098:
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->lr = 0x008630A0; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x94); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[13]));
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x008630C4; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[29] + 0x24, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862FE8;
loc_008630DC:
    ctx->gpr[27] = ctx->gpr[27] + (int64_t)(0x14);
    vm_write32(ctx->gpr[29] + 0x28, ctx->gpr[0]);
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[27], 0, 32);
    vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[29]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086300C;
loc_008630F8:
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[28] = ftmp; }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086325C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6530); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[29] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x652C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ;
loc_00863120:
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x3; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863274;
    { uint64_t a = (uint32_t)ctx->gpr[9]; uint64_t b = (uint64_t)0x1; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 24)) | (cr_val << 24); }
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = (int64_t)(int32_t)(0x262);
    if ((((ctx->cr >> 6) & 1) == 1)) goto loc_00863250;
    if ((((ctx->cr >> 27) & 1) == 0)) goto loc_00863254;
loc_00863144:
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6524);
    ctx->lr = 0x0086314C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00863150:
    ctx->gpr[29] = ctx->gpr[29] + (int64_t)(0x14);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863120;
    { double a = ctx->fpr[31]; double b = ctx->fpr[28]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086326C;
loc_00863170:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00862EBC;
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64D0);
    ctx->gpr[4] = (int64_t)(int32_t)(0x94);
    ctx->lr = 0x0086318C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00862EBC;
loc_00863194:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)(7);
    vm_write8(ctx->gpr[1] + 0xC4, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(7);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    vm_write16(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0xB8, ctx->gpr[9]);
    goto loc_00863098;
loc_008631C4:
    ctx->gpr[0] = vm_read16(ctx->gpr[31] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[9]);
    vm_write16(ctx->gpr[1] + 0x84, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086329C;
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x78);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = ctx->gpr[0] | 0x8;
    vm_write16(ctx->gpr[1] + 0x78, ctx->gpr[0]);
loc_008631EC:
    ctx->gpr[3] = (int64_t)(int32_t)(3);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    goto loc_00863098;
loc_00863200:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x70);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[21], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = ctx->gpr[0] | 0x10;
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[24], 0, 32);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x00863234; june_evt_func_0086C4C0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x8C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[28]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    goto loc_00862FC4;
loc_00863250:
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00863144;
loc_00863254:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[29]));
    goto loc_00863150;
loc_0086325C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6520); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    { double a = ctx->fpr[28]; double b = ctx->fpr[0]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00863170;
loc_0086326C:
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    goto loc_00863170;
loc_00863274:
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[30]));
    goto loc_00863150;
loc_0086327C:
    vm_write32(ctx->gpr[29] + 0x28, ctx->gpr[0]);
    goto loc_00862D50;
loc_00863284:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64D8);
    ctx->gpr[4] = (int64_t)(int32_t)(0x3C);
    ctx->lr = 0x00863294; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00862C10;
loc_0086329C:
    ctx->gpr[0] = vm_read16(ctx->gpr[1] + 0x78);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x651C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 28, 1, 31, 0, 0);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 4, 0, 31, 0, 0);
    vm_write16(ctx->gpr[1] + 0x78, ctx->gpr[0]);
    goto loc_008631EC;
loc_008632B8:
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008632C0; june_evt_func_00861E0C(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_00862F64;
loc_008632C4:
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x64D4);
    ctx->gpr[4] = (int64_t)(int32_t)(0x7C);
    ctx->lr = 0x008632D4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00863038;
}

void june_evt_func_00866308(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x24);
        return;
}

void june_evt_func_0086640C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x110, ctx->gpr[1]); ctx->gpr[1] += -0x110;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[0]);
    ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(-1);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[28]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x100, tmp); }
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[31]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x108, tmp); }
    ctx->gpr[31] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[27]);
    ctx->gpr[29] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[28] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866520;
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0x94);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[6], 0, 32);
    ctx->gpr[3] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x00866480; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866520;
    ctx->gpr[10] = vm_read16(ctx->gpr[31] + 0x0);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 31, 31, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[29] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[10], 0, 28, 28, 0, 0);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[11], ctx->fpr[13]));
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866594;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00866550;
loc_008664CC:
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_008665E4;
loc_008664D0:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
loc_008664D4:
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->fpr[2] = ctx->fpr[30];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6444); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->lr = 0x008664E8; june_evt_func_008B07F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = ctx->gpr[26] + (int64_t)(0x18);
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->lr = 0x008664FC; june_evt_func_008B0800(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x0086650C; june_evt_func_008B080C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->lr = 0x0086651C; june_evt_func_008B0858(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00866520:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x120);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[30] = _cs_30;
    ctx->gpr[31] = _cs_31;
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x100); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x108); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x110);
    return;
loc_00866550:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_008664D0;
    goto loc_008665E4;
loc_00866594:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    if ((((ctx->cr >> 5) & 1) == 0)) goto loc_00866654;
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
loc_008665D4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008664CC;
    goto loc_00866550;
loc_008665E4:
    ctx->gpr[29] = ctx->gpr[1] + (int64_t)(0x74);
    ctx->lr = 0x008665EC; june_evt_func_0086E350(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read16(ctx->gpr[31] + 0x20);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6448); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[30], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->lr = 0x0086661C; june_evt_func_00866AB8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6444); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[28], 0, 32);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x84);
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[10] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00866644; june_evt_func_008616F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x644C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    goto loc_008664D4;
loc_00866654:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x644C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    { uint32_t tmp = vm_read32(ctx->gpr[28] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0xB0);
    ctx->gpr[0] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 1);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xC0); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[0]));
    goto loc_008665D4;
}

extern "C" void june_evt_button_probe(ppu_context* ctx);

void june_evt_func_00866694(ppu_context* ctx) {
    june_evt_button_probe(ctx);
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
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    ctx->gpr[0] = ctx->lr;
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0xE8, tmp); }
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[30]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0xE0, tmp); }
    ctx->gpr[27] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    ctx->gpr[23] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[5] | ctx->gpr[5];
    ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
    ctx->gpr[24] = ctx->gpr[7] | ctx->gpr[7];
    ctx->lr = 0x008666E4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6444); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866714;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6440);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
loc_00866714:
    ctx->lr = 0x00866718; june_evt_func_0086E44C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00866724; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00866730; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6444); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866760;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x643C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[30] = ppu_frsp(ctx->fpr[0]);
loc_00866760:
    ctx->lr = 0x00866764; june_evt_func_0086E3AC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00866770; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->lr = 0x00866788; june_evt_func_008612E8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008667D8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[12]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x70); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086683C;
loc_008667D8:
    ctx->gpr[9] = vm_read16(ctx->gpr[27] + 0x0);
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008667FC;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 29, 1, 31, 0, 0);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 3, 0, 31, 0, 0);
    vm_write16(ctx->gpr[27] + 0x0, ctx->gpr[0]);
loc_008667FC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE0); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xE8); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_0086683C:
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x78); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008667D8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[26];
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[13], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[13], 8); ctx->fpr[13] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x80); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[13]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[12]);
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[30], ctx->fpr[0]));
    ctx->fpr[13] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { double a = ctx->fpr[13]; double b = ctx->fpr[12]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008667D8;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[12], ctx->fpr[0]));
    { double a = ctx->fpr[0]; double b = ctx->fpr[13]; uint32_t cr_val = (a != a || b != b) ? 1 : (a < b) ? 8 : (a > b) ? 4 : 2; ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { uint32_t a = (ctx->cr >> (31 - 29)) & 1; uint32_t b = (ctx->cr >> (31 - 30)) & 1; ctx->cr = (ctx->cr & ~(1u << (31 - 30))) | ((a | b) << (31 - 30)); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008667D8;
    ctx->lr = 0x008668A8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008667D8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6438);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008667D8;
    { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008667D8;
    ctx->lr = 0x008668D8; june_evt_func_0086DF2C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008667D8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[24], 0, 32);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[27] + 0x24);
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(3);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->ctr = ctx->gpr[0];
    ctx->lr = 0x00866918; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->lr = 0x00866920; june_evt_func_0086DF54(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008667D8;
}

void june_evt_func_00866928(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->lr = 0x00866944; june_evt_func_00866318(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x0086694C; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[29] = _cs_29;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    return;
}

void june_evt_func_00866A34(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_29 = ctx->gpr[29];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
    ctx->gpr[29] = (int64_t)(int8_t)ctx->gpr[29];
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00866A58; june_evt_func_0086E368(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00866A84;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x8);
    ctx->gpr[29] = _cs_29;
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    return;
loc_00866A84:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[29] = _cs_29;
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
    ctx->lr = ctx->gpr[0];
    return;
}

void june_evt_func_00866AB0(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x40);
        return;
}

void june_evt_func_00866AB8(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_30 = ctx->gpr[30];
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[30]);
    ctx->gpr[30] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    { uint64_t a = (uint32_t)ctx->gpr[30]; uint64_t b = (uint64_t)0x22; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    ctx->gpr[4] = (int64_t)(int32_t)(0x29F);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00866AF0;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6430);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x642C);
    ctx->lr = 0x00866AEC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00866AF0:
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[30], 4, 0, 27, 0, 0);
    ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x6428);
    ctx->gpr[30] = _cs_30;
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->lr = ctx->gpr[0];
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[31] + 0x8, tmp); }
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}



void june_evt_func_0086BE34(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[6] = ctx->gpr[4] + (int64_t)(0xC);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[4] + 0x4);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
        ctx->gpr[6] = ppc_rldicl(ctx->gpr[6], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086BE94;
        ctx->gpr[9] = vm_read32(ctx->gpr[4] + 0x0);
        ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x62EC);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 5, 0, 26, 0, 0);
        ctx->gpr[4] = vm_read32(ctx->gpr[4] + 0x1C);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0xC);
        ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[11] + 0x4);
        ctx->lr = 0x0086BE90; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
loc_0086BE94:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_0086BEF4(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0x70, ctx->gpr[1]); ctx->gpr[1] += -0x70;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x0);
        ctx->gpr[0] = vm_read32(ctx->gpr[2] + -0x62EC);
        ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 5, 0, 26, 0, 0);
        ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x1C);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0x10);
        ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
        ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
        ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
        ctx->gpr[0] = vm_read32(ctx->gpr[11] + 0x0);
        vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
        ctx->ctr = ctx->gpr[0];
        ctx->gpr[2] = vm_read32(ctx->gpr[11] + 0x4);
        ctx->lr = 0x0086BF34; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[2] = vm_read64(ctx->gpr[1] + 0x28);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x80);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x70);
        return;
}

void june_evt_func_0086D298(ppu_context* ctx) {
    PPU_MM_SNAP
        ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x10);
        return;
}
