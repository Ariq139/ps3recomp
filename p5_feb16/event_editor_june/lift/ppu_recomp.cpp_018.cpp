
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

void june_evt_func_0087199C(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[24]);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[16], 8); vm_write64(ctx->gpr[1] + 0x170, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[17], 8); vm_write64(ctx->gpr[1] + 0x178, tmp); }
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
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[14]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[15]);
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[16]);
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[30]);
    ctx->gpr[24] = vm_read32(ctx->gpr[2] + -0x626C);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[31]);
    vm_write64(ctx->gpr[1] + 0x200, ctx->gpr[0]);
    vm_write32(ctx->gpr[1] + 0x1F8, ctx->gpr[12]);
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = vm_read8(ctx->gpr[24] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872074;
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x6270);
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871ABC;
    ctx->gpr[10] = vm_read32(ctx->gpr[30] + 0x8);
    goto loc_00871AA0;
loc_00871A64:
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[8] = (int64_t)(int32_t)(0x24);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 25, 25, 0, 0);
    ctx->gpr[8] = ctx->gpr[11] & ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871A94;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00871B5C;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871B5C;
loc_00871A94:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871ABC;
loc_00871AA0:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871A64;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871AA0;
loc_00871ABC:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x200);
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
loc_00871B5C:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(9);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00871B6C; june_evt_func_008C2024(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00871B7C; june_evt_func_008C2024(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->lr = 0x00871B8C; june_evt_func_008C2024(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(9);
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00871BA0; june_evt_func_008C20D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00871BB4; june_evt_func_008C20D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00871BC8; june_evt_func_008C20D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[5] = (int64_t)(int32_t)(5);
    ctx->gpr[6] = (int64_t)(int32_t)(4);
    ctx->gpr[7] = (int64_t)(int32_t)(5);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00871BE4; june_evt_func_008C2768(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(0x24);
    ctx->lr = 0x00871BF0; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00871C00; june_evt_func_0082699C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(9);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00871C14; june_evt_func_008C20D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872468;
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871DA4;
    ctx->gpr[27] = ctx->gpr[1] + (int64_t)(0xA4);
loc_00871C40:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871D90;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871D90;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x60);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x64);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x68);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x6C);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6210);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->lr = 0x00871CE4; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871D90;
    ;
loc_00871D08:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00871D80;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871D80;
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x60);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00871D80;
    ctx->gpr[8] = vm_read32(ctx->gpr[29] + 0x4);
    ctx->gpr[5] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->gpr[7] = vm_read32(ctx->gpr[28] + 0x64);
    ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[5];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00871D80;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x68);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00871D80;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x6C);
    ctx->gpr[9] = ctx->gpr[10] | ((uint64_t)0x8000 << 16);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[0];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[8]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0087244C;
loc_00871D80:
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871D08;
    ;
loc_00871D90:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871C40;
    ctx->gpr[0] = ctx->gpr[1] + (int64_t)(0x74);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[0]);
loc_00871DA4:
    ctx->gpr[4] = (int64_t)(int32_t)(9);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    ctx->lr = 0x00871DB4; june_evt_func_008C20D8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871F1C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x61BC);
    ctx->gpr[23] = ctx->gpr[1] + (int64_t)(0x70);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xC8);
    ctx->gpr[22] = ctx->gpr[1] + (int64_t)(0x84);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6238); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[18] = ftmp; }
    ctx->gpr[19] = ctx->gpr[9] + (int64_t)(0x108);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6244); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[16] = ftmp; }
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6240); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[21] = ftmp; }
    ctx->fpr[22] = ctx->fpr[16];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x623C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[20] = ftmp; }
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[23], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6230); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[17] = ftmp; }
    ctx->fpr[19] = ctx->fpr[18];
    ctx->gpr[21] = ctx->gpr[9] + (int64_t)(0xE8);
    ctx->gpr[20] = ctx->gpr[9] + (int64_t)(0xF8);
    ctx->gpr[18] = ctx->gpr[9] + (int64_t)(0x118);
    ctx->gpr[17] = ctx->gpr[9] + (int64_t)(0x128);
    ctx->gpr[16] = ctx->gpr[9] + (int64_t)(0x138);
    ctx->gpr[15] = ctx->gpr[9] + (int64_t)(0x148);
    ctx->gpr[14] = ctx->gpr[9] + (int64_t)(0x158);
loc_00871E18:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[8] = (int64_t)(int32_t)(0x64);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[8] = ctx->gpr[0] & ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871F10;
    ctx->gpr[10] = vm_read8(ctx->gpr[28] + 0x4F);
    ctx->gpr[8] = vm_read8(ctx->gpr[24] + 0x1C);
    ctx->gpr[0] = vm_read8(ctx->gpr[28] + 0x4C);
    ctx->gpr[10] = (uint64_t)((int64_t)(int32_t)ctx->gpr[10] * (int64_t)(int32_t)ctx->gpr[8]);
    ctx->gpr[9] = vm_read8(ctx->gpr[28] + 0x4D);
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    vm_write8(ctx->gpr[1] + 0x70, ctx->gpr[0]);
    vm_write8(ctx->gpr[1] + 0x71, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rldicr(ctx->gpr[10], 7, 56);
    ctx->gpr[11] = vm_read8(ctx->gpr[28] + 0x4E);
    ctx->gpr[9] = ppc_rldicr(ctx->gpr[10], 15, 48);
    vm_write8(ctx->gpr[1] + 0x72, ctx->gpr[11]);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
    ctx->gpr[11] = ppc_rldicr(ctx->gpr[0], 16, 47);
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[11];
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[10];
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[0], 32, 32);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 25, 7, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 0, 24, 31, 0, 0);
    vm_write8(ctx->gpr[1] + 0x73, ctx->gpr[0]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871F10;
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[27] = vm_read32(ctx->gpr[28] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[7] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0xC);
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00872100;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xB4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871EDC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00871ED8; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_00871EDC:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xB8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871F10;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)(0);
loc_00871EF8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00871F0C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_00871F10:
    ctx->gpr[4] = vm_read32(ctx->gpr[28] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00871E18;
loc_00871F1C:
    ctx->gpr[4] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->lr = 0x00871F2C; june_evt_func_00826908(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00871F3C; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x00871F5C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00871F84; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00871F90; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->lr = 0x00871FC8; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00871FD4; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->lr = 0x00872010; june_evt_func_008268F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xC8);
    ctx->gpr[8] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x624C);
    ctx->gpr[6] = ctx->gpr[1] + (int64_t)(0x94);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00872030; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[30] + (int64_t)(0x24);
    ctx->lr = 0x0087203C; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->lr = 0x0087204C; june_evt_func_008C1F70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->lr = 0x0087205C; june_evt_func_008C1F70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = (int64_t)(int32_t)(9);
    ctx->lr = 0x0087206C; june_evt_func_008C1F70(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00871ABC;
loc_00872074:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6268);
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0xC0);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[29] = vm_read32(ctx->gpr[1] + 0xC0);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0xC0);
    ctx->lr = 0x008720A0; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->lr = 0x008720A8; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x61D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6224); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[2] = ftmp; }
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[28];
    ctx->gpr[7] = vm_read32(ctx->gpr[2] + -0x61CC);
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[3] = ftmp; }
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[3]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[31], 0, 32);
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[1] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[1] = ppu_fp_single(ppu_fadd(ctx->fpr[1], ctx->fpr[13]));
    ctx->fpr[2] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[2]));
    ctx->lr = 0x008720F8; june_evt_func_008F6918(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00871ABC;
loc_00872100:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 9, 9, 0, 0);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[9]);
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[0], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[8] = ppc_rlw64(ctx->gpr[8], 0, 1, 31, 0, 0);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[11], &tmp, 8); }
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[10]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[12], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[12], 8); ctx->fpr[12] = (double)iv; }
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[8]);
    ctx->fpr[29] = ppu_frsp(ctx->fpr[0]);
    ctx->fpr[28] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[24] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[25] = ppu_frsp(ctx->fpr[12]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087252C;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->fpr[23] = ctx->fpr[18];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[18]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_008726E4;
    { float ftmp = (float)ctx->fpr[20]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_00872180:
    ctx->fpr[26] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[24]));
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[23]));
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[21], 0, 32);
    ctx->fpr[27] = ctx->fpr[17];
    ctx->lr = 0x008721AC; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->fpr[30] = ctx->fpr[1];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[23]));
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[20], 0, 32);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x008721D8; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[31], ctx->fpr[17]));
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[17]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->lr = 0x00872204; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[27], 0, 8, 9, 0, 0);
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0xC0 << 16);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008726A0;
loc_0087221C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x61C0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[0]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->fpr[26] = ppu_fp_single(ppu_fsub(ctx->fpr[26], ctx->fpr[16]));
    ctx->fpr[31] = ctx->fpr[16];
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[30]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x00872254; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x7C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[13], ctx->fpr[0]));
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[17], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { float ftmp = (float)ctx->fpr[27]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00872288; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[1] + 0x74); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[0], ctx->fpr[27]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[16], 0, 32);
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x008722B8; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[25]));
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[16]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { float ftmp = (float)ctx->fpr[16]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[16]));
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x008722F0; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->lr = 0x00872314; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[23]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[16]));
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61B8);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[16]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00872348; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_008726CC;
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[16]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[20]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[20]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
loc_00872368:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[23], 0, 32);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61B4);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x0087238C; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61B0);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { float ftmp = (float)ctx->fpr[26]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x008723B0; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008723B4:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xB4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008723E8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x008723E4; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
loc_008723E8:
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x38);
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    { float ftmp = (float)ctx->fpr[24]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    { float ftmp = (float)ctx->fpr[25]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087242C;
    ;
loc_00872408:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[22], 0, 32);
    ctx->lr = 0x0087241C; june_evt_func_0086BE34(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872408;
loc_0087242C:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0xB8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00871F10;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[5] = (int64_t)(int32_t)(1);
    goto loc_00871EF8;
loc_0087244C:
    vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[9]);
    vm_write32(ctx->gpr[29] + 0x60, ctx->gpr[11]);
    vm_write32(ctx->gpr[29] + 0x64, ctx->gpr[8]);
    vm_write32(ctx->gpr[29] + 0x68, ctx->gpr[6]);
    vm_write32(ctx->gpr[29] + 0x6C, ctx->gpr[5]);
    ctx->lr = 0x00872464; june_evt_func_00870214(ctx); DRAIN_TRAMPOLINE(ctx);
    goto loc_00871D80;
loc_00872468:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0087247C; june_evt_func_008268F8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x28);
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x74);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[8]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x008724A8; june_evt_func_00826900(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[1] = ctx->fpr[31];
    ctx->gpr[4] = vm_read32(ctx->gpr[2] + -0x6210);
    ctx->gpr[0] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[5] = vm_read64(ctx->gpr[1] + 0xC8);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xD0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x008724D8; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872518;
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x8);
loc_008724EC:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087250C;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] | ((uint64_t)0x8000 << 16);
    vm_write32(ctx->gpr[9] + 0x10, ctx->gpr[0]);
loc_0087250C:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008724EC;
loc_00872518:
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[30] + 0x1C, ctx->gpr[0]);
    goto loc_00871DA4;
loc_0087252C:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 8, 8, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008723B4;
    ctx->fpr[31] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[25]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[24]));
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61AC);
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x0087256C; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->fpr[31] = ppu_fp_single(ppu_fsub(ctx->fpr[31], ctx->fpr[22]));
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->fpr[30] = ppu_fp_single(ppu_fsub(ctx->fpr[30], ctx->fpr[22]));
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[15], 0, 32);
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x00872598; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[6] = ppc_rldicl(ctx->gpr[14], 0, 32);
    ctx->lr = 0x008725BC; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61A8);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    ctx->lr = 0x008725E0; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[29], ctx->fpr[22]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[24], ctx->fpr[19]));
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61A4);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[28]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x00872618; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61B8);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[31]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->lr = 0x0087263C; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->fpr[0] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[22]));
    ctx->fpr[13] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[19]));
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61B4);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[29]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[22]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->lr = 0x00872674; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x18);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = vm_read32(ctx->gpr[2] + -0x61B0);
    ctx->gpr[4] = ctx->gpr[25] | ctx->gpr[25];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[5] = ctx->gpr[26] | ctx->gpr[26];
    { float ftmp = (float)ctx->fpr[30]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    ctx->lr = 0x00872698; june_evt_func_008EFFA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_008723B4;
loc_008726A0:
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_008726F4;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 15, 15, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008726EC;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x61C4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_008726B4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 14, 14, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087221C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6218); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    goto loc_0087221C;
loc_008726CC:
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[25], ctx->fpr[16]));
    ctx->fpr[13] = ppu_fp_single(ppu_fadd(ctx->fpr[28], ctx->fpr[21]));
    ctx->fpr[0] = ppu_fp_single(ppu_fsub(ctx->fpr[0], ctx->fpr[21]));
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    goto loc_00872368;
loc_008726E4:
    { float ftmp = (float)ctx->fpr[21]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    goto loc_00872180;
loc_008726EC:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6220); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_008726B4;
loc_008726F4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 15, 15, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087271C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x61C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
loc_00872704:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[27], 0, 14, 14, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087221C;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x621C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    ctx->fpr[30] = ppu_fp_single(ppu_fadd(ctx->fpr[30], ctx->fpr[0]));
    goto loc_0087221C;
loc_0087271C:
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x6240); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    goto loc_00872704;
}

void june_evt_func_00872724(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x140, ctx->gpr[1]); ctx->gpr[1] += -0x140;
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[24]);
    ctx->gpr[24] = vm_read32(ctx->gpr[2] + -0x6270);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[30], 8); vm_write64(ctx->gpr[1] + 0x130, tmp); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[31], 8); vm_write64(ctx->gpr[1] + 0x138, tmp); }
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
    vm_write64(ctx->gpr[1] + 0xF8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[31]);
    vm_write32(ctx->gpr[1] + 0x148, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873118;
    ctx->lr = 0x0087279C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873450;
loc_008727B0:
    ctx->lr = 0x008727B4; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008727C0; june_evt_func_00827264(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x80);
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[22] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->lr = 0x0087280C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873430;
loc_0087281C:
    ctx->lr = 0x00872820; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x0087282C; june_evt_func_00827274(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = (int64_t)(int32_t)ctx->gpr[29];
    ctx->gpr[3] = (int64_t)(int32_t)ctx->gpr[3];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[29]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x80);
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[3]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    ctx->fpr[0] = ppu_fp_single(ppu_fmul(ctx->fpr[0], ctx->fpr[31]));
    ctx->fpr[0] = ppu_fp_single(ppu_fdiv(ctx->fpr[0], ctx->fpr[13]));
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[21] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->lr = 0x00872874; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873310;
loc_00872884:
    ctx->gpr[16] = vm_read32(ctx->gpr[2] + -0x626C);
    ctx->gpr[0] = vm_read8(ctx->gpr[16] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087374C;
loc_00872894:
    ctx->gpr[0] = vm_read8(ctx->gpr[24] + 0x10);
    ctx->gpr[11] = ctx->gpr[24] + (int64_t)(0x24);
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[0];
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x6190);
    ctx->gpr[10] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[9] = ctx->gpr[10] ^ ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[10] - ctx->gpr[9];
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 3, 29, 29, 0, 0);
    vm_write16(ctx->gpr[11] + 0x0, ctx->gpr[9]);
    ctx->lr = 0x008728C4; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872A40;
    ctx->gpr[31] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[30] = vm_read32(ctx->gpr[2] + -0x6194);
    goto loc_008728EC;
loc_008728E0:
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872968;
loc_008728EC:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 2, 2, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087290C;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008736A8;
loc_0087290C:
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008728E0;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x1C);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 0, 1, 31, 0, 0);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[0] = ctx->gpr[0] | ((uint64_t)0x4000 << 16);
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[0]);
    vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[9]);
    ctx->lr = 0x00872934; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00872940; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00872948; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872A40;
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008728EC;
loc_00872968:
    ctx->gpr[27] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872A40;
    ctx->gpr[23] = (int64_t)(int32_t)(0);
    ctx->gpr[19] = (int64_t)(int32_t)(0);
    ctx->gpr[20] = (int64_t)(int32_t)(0);
loc_00872980:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0x74);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872A00;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 25, 25, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873180;
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 20, 20, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008729C8;
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 20, 1, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] | 0x10;
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 12, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[11]);
loc_008729C8:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008729F4;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008735CC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008735CC;
loc_008729F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873614;
loc_00872A00:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[27] = ctx->gpr[31] | ctx->gpr[31];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872980;
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872A40;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873E64;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008742B8;
loc_00872A2C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[23], 0, 32);
    vm_write32(ctx->gpr[9] + 0x74, ctx->gpr[23]);
    vm_write32(ctx->gpr[11] + 0x70, ctx->gpr[0]);
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[19]);
loc_00872A40:
    ctx->gpr[3] = ctx->gpr[24] + (int64_t)(0x24);
    ctx->lr = 0x00872A48; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[25] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873118;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[25], 0, 32);
    ctx->gpr[8] = (int64_t)(int32_t)(0x24);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[8] = ctx->gpr[0] & ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873118;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[20] = vm_read32(ctx->gpr[26] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[23] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[17] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[18] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[19] = vm_read32(ctx->gpr[26] + 0x1C);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872ED8;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872AEC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872AEC;
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[31] = vm_read64(ctx->gpr[26] + 0x90);
    ctx->gpr[0] = vm_read64(ctx->gpr[30] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[31]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872AEC;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872AD4;
    goto loc_008741B4;
loc_00872AC8:
    ctx->gpr[3] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008741B4;
loc_00872AD4:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00872AE4; june_evt_func_0086FB48(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872AC8;
loc_00872AEC:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xAC);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873C0C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[28] = (int64_t)(int32_t)(1);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00872B18; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873C0C;
loc_00872B28:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[20];
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[17];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[18];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872BE4;
    ctx->gpr[30] = (int64_t)(int32_t)(0);
    ctx->gpr[31] = ctx->gpr[1] + (int64_t)(0x70);
    ;
loc_00872BA0:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[4], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->lr = 0x00872BB4; june_evt_func_0086BDA0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872BC8;
    ctx->gpr[30] = (int64_t)(int32_t)(1);
loc_00872BC8:
    ctx->gpr[4] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[4]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872BA0;
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00872BE4;
    ctx->gpr[28] = (int64_t)(int32_t)(1);
loc_00872BE4:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xB0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872C1C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[3] = ctx->gpr[26] | ctx->gpr[26];
    ctx->gpr[8] = vm_read32(ctx->gpr[9] + 0x0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[8];
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00872C0C; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872C28;
loc_00872C1C:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 1)) goto loc_00872C60;
loc_00872C28:
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872C60;
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xC000 << 16);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[10]);
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[8]);
loc_00872C60:
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[20]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873118;
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[23]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873118;
    ctx->gpr[0] = ctx->gpr[20] + ctx->gpr[17];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873118;
    ctx->gpr[0] = ctx->gpr[23] + ctx->gpr[18];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873118;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872ED8;
    ctx->lr = 0x00872C98; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872ED8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872ED8;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874BA4;
loc_00872CC8:
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x40);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872DE4;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872DCC;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872D30;
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x80);
    ctx->gpr[9] = ctx->gpr[9] | 0x4;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x200;
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872DC4;
loc_00872D14:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00872D24; june_evt_func_0086FE44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872D14;
loc_00872D30:
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872DC4;
    ctx->gpr[27] = (int64_t)(int32_t)(0);
loc_00872D40:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x7C);
    ctx->gpr[28] = vm_read32(ctx->gpr[31] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874A68;
loc_00872D54:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008746EC;
loc_00872D64:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872D64;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x84);
    vm_write32(ctx->gpr[9] + 0x84, ctx->gpr[0]);
loc_00872D7C:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x80);
    vm_write32(ctx->gpr[31] + 0x78, ctx->gpr[30]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[31] + 0x7C, ctx->gpr[27]);
    vm_write32(ctx->gpr[31] + 0x84, ctx->gpr[27]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872DB0;
loc_00872D94:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00872DA4; june_evt_func_0086F6D0(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872D94;
loc_00872DB0:
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00872DBC; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872D40;
loc_00872DC4:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->lr = 0x00872DCC; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00872DCC:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x40);
    ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x98);
    vm_write32(ctx->gpr[24] + 0xC, ctx->gpr[9]);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->lr = 0x00872DE0; june_evt_func_00834F5C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00872DE4:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6268);
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0x80);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[31] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->lr = 0x00872E10; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00872E1C; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x40);
    ctx->gpr[29] = ctx->gpr[29] - ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0x8);
    ctx->gpr[0] = ctx->gpr[22] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00872E40;
    ctx->gpr[5] = ctx->gpr[22] - ctx->gpr[11];
loc_00872E40:
    ctx->gpr[11] = vm_read32(ctx->gpr[10] + 0xC);
    ctx->gpr[9] = ctx->gpr[3] - ctx->gpr[31];
    ctx->gpr[7] = ctx->gpr[21] | ctx->gpr[21];
    ctx->gpr[0] = ctx->gpr[21] + ctx->gpr[11];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00872E5C;
    ctx->gpr[7] = ctx->gpr[21] - ctx->gpr[11];
loc_00872E5C:
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872E78;
    ctx->gpr[0] = vm_read32(ctx->gpr[10] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[10] + 0x18, ctx->gpr[0]);
loc_00872E78:
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x40);
    ctx->gpr[0] = vm_read32(ctx->gpr[8] + 0x18);
    ctx->gpr[6] = vm_read32(ctx->gpr[8] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 22, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[5]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 10, 0, 31, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] | 0x400;
    vm_write32(ctx->gpr[8] + 0x18, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874AFC;
loc_00872E9C:
    ctx->gpr[10] = vm_read32(ctx->gpr[8] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872ED0;
    ctx->gpr[10] = ctx->gpr[10] | ((uint64_t)0xC000 << 16);
    ctx->gpr[0] = vm_read32(ctx->gpr[8] + 0x4);
    ctx->gpr[9] = vm_read32(ctx->gpr[8] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[8] + 0xC);
    vm_write32(ctx->gpr[8] + 0x60, ctx->gpr[6]);
    vm_write32(ctx->gpr[8] + 0x64, ctx->gpr[0]);
    vm_write32(ctx->gpr[8] + 0x68, ctx->gpr[9]);
    vm_write32(ctx->gpr[8] + 0x6C, ctx->gpr[11]);
    vm_write32(ctx->gpr[8] + 0x10, ctx->gpr[10]);
loc_00872ED0:
    vm_write32(ctx->gpr[8] + 0x4, ctx->gpr[7]);
    vm_write32(ctx->gpr[8] + 0x0, ctx->gpr[5]);
loc_00872ED8:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 26, 26, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008737EC;
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872F20;
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xC000 << 16);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[10]);
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[8]);
loc_00872F20:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008742EC;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x50, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x54, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x58, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x5C, ctx->gpr[10]);
loc_00872F50:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6268);
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x80);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[13], &iv, 8); }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[13], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[28] = vm_read32(ctx->gpr[1] + 0x80);
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[29] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->lr = 0x00872F7C; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[28], 1, 0, 30, 0, 0);
    ctx->gpr[27] = ctx->gpr[3] - ctx->gpr[0];
    ctx->lr = 0x00872F8C; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[0] - ctx->gpr[28];
    ctx->gpr[5] = ctx->gpr[11] - ctx->gpr[29];
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[29], 1, 0, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[4] + ctx->gpr[9];
    ctx->gpr[11] = ctx->gpr[5] + ctx->gpr[11];
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[3]);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[28]);
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[29]);
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[27]);
    ctx->gpr[6] = ctx->gpr[27] - ctx->gpr[9];
    ctx->gpr[3] = ctx->gpr[3] - ctx->gpr[11];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873018;
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[31] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[3];
loc_00872FEC:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[10], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[6] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[7] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00873008; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[10]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872FEC;
loc_00873018:
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x00873028; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008730AC;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087305C;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x559);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x00873058; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0087305C:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008746D0;
loc_0087306C:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    ctx->lr = 0x008730A8; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008730AC:
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x008730BC; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008730FC;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008730EC;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6178);
    ctx->gpr[4] = (int64_t)(int32_t)(0x565);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->lr = 0x008730E8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008730EC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x10;
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_008730FC:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 26, 1, 31, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 6, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[9]);
loc_00873118:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x150);
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x148);
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
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x130); memcpy(&ctx->fpr[30], &tmp, 8); }
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x138); memcpy(&ctx->fpr[31], &tmp, 8); }
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x140);
    return;
loc_00873180:
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[30] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 22, 22, 0, 0);
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[26] = vm_read32(ctx->gpr[28] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[25] = vm_read32(ctx->gpr[28] + 0xC);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008732E0;
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008734A0;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 22, 1, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] | 0x4;
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 10, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[11]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008734D8;
loc_008731D0:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873200;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x44);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873570;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873570;
loc_00873200:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 21, 21, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872A00;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[3]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008732CC;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008732CC;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087324C;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873B78;
loc_0087324C:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008740D4;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x70);
    vm_write32(ctx->gpr[9] + 0x70, ctx->gpr[0]);
loc_00873264:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008740C8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x74);
    vm_write32(ctx->gpr[9] + 0x74, ctx->gpr[0]);
loc_0087327C:
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x4);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[28] + 0x70, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008740BC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    vm_write32(ctx->gpr[28] + 0x74, ctx->gpr[11]);
    vm_write32(ctx->gpr[9] + 0x70, ctx->gpr[27]);
loc_0087329C:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008732BC;
    ctx->gpr[0] = vm_read8(ctx->gpr[24] + 0x0);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873BC8;
loc_008732BC:
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008737B0;
loc_008732CC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 21, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 11, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[0]);
    goto loc_00872A00;
loc_008732E0:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 23, 23, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873470;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 20, 20, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873C5C;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 20, 1, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] | 0x10;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 12, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[0]);
    goto loc_008731D0;
loc_00873310:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6198);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872884;
    ctx->gpr[16] = vm_read32(ctx->gpr[2] + -0x626C);
    ctx->gpr[0] = vm_read8(ctx->gpr[16] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008733B4;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008733B4;
    ctx->gpr[7] = vm_read32(ctx->gpr[24] + 0x8);
    ;
loc_00873348:
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873C14;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008733A8;
    ctx->gpr[0] = vm_read8(ctx->gpr[9] + 0x4F);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008733A8;
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[8] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008733A8;
    ctx->gpr[11] = ctx->gpr[11] + ctx->gpr[10];
    if ((((ctx->cr >> 7) & 1) == 1)) goto loc_008733A8;
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008740EC;
    ;
loc_008733A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873348;
loc_008733B4:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873420;
    ctx->gpr[10] = vm_read32(ctx->gpr[24] + 0x8);
    goto loc_00873404;
loc_008733C8:
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[8] = (int64_t)(int32_t)(0x24);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[11], 0, 25, 25, 0, 0);
    ctx->gpr[8] = ctx->gpr[11] & ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 4)) | (cr_val << 4); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008733F8;
    if ((((ctx->cr >> 5) & 1) == 1)) goto loc_00873C3C;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873C3C;
loc_008733F8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873420;
loc_00873404:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008733C8;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873404;
loc_00873420:
    ctx->gpr[0] = vm_read8(ctx->gpr[16] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872894;
    goto loc_00873118;
loc_00873430:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x619C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_0087281C;
loc_00873450:
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x61A0);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0x90); memcpy(&ctx->fpr[0], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[0], 8); ctx->fpr[0] = (double)iv; }
    ctx->fpr[31] = ppu_frsp(ctx->fpr[0]);
    goto loc_008727B0;
loc_00873470:
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873778;
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 23, 1, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 29, 1, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 9, 0, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 3, 0, 31, 0, 0);
    ctx->gpr[11] = ctx->gpr[11] | 0x400;
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[11]);
    goto loc_008731D0;
loc_008734A0:
    ctx->gpr[0] = ctx->gpr[10] | ((uint64_t)0xC000 << 16);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 22, 1, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    ctx->gpr[9] = ctx->gpr[9] | 0x4;
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 10, 0, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[28] + 0x60, ctx->gpr[30]);
    vm_write32(ctx->gpr[28] + 0x64, ctx->gpr[29]);
    vm_write32(ctx->gpr[28] + 0x68, ctx->gpr[26]);
    vm_write32(ctx->gpr[28] + 0x6C, ctx->gpr[25]);
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[11]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008731D0;
loc_008734D8:
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874414;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x70);
    vm_write32(ctx->gpr[9] + 0x70, ctx->gpr[0]);
loc_008734EC:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874408;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x74);
    vm_write32(ctx->gpr[9] + 0x74, ctx->gpr[0]);
loc_00873504:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[28] + 0x70, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008743FC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[23], 0, 32);
    vm_write32(ctx->gpr[28] + 0x74, ctx->gpr[23]);
    vm_write32(ctx->gpr[9] + 0x70, ctx->gpr[27]);
loc_00873520:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087354C;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0087354C;
    ctx->gpr[0] = vm_read8(ctx->gpr[24] + 0x0);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087411C;
loc_0087354C:
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x4);
    ctx->gpr[23] = ctx->gpr[27] | ctx->gpr[27];
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008731D0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x400;
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    goto loc_008731D0;
loc_00873570:
    ctx->gpr[29] = ctx->gpr[24] + (int64_t)(0x24);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00873580; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00873590; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00873598; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 27, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 5, 0, 31, 0, 0);
    ctx->gpr[9] = ctx->gpr[9] | 0x400;
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873200;
    ctx->gpr[20] = (int64_t)(int32_t)(1);
    goto loc_00873200;
loc_008735CC:
    ctx->gpr[29] = ctx->gpr[24] + (int64_t)(0x24);
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008735DC; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->lr = 0x008735EC; june_evt_func_008710A4(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008735F4; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 27, 1, 31, 0, 0);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 5, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872A00;
loc_00873614:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008740E0;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x70);
    vm_write32(ctx->gpr[9] + 0x70, ctx->gpr[0]);
loc_0087362C:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087417C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x74);
    vm_write32(ctx->gpr[9] + 0x74, ctx->gpr[0]);
loc_00873644:
    { int64_t a = (int32_t)ctx->gpr[23]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[28] + 0x70, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874188;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[23], 0, 32);
    vm_write32(ctx->gpr[28] + 0x74, ctx->gpr[23]);
    vm_write32(ctx->gpr[9] + 0x70, ctx->gpr[27]);
loc_00873660:
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0087367C;
    goto loc_008743E4;
loc_00873670:
    ctx->gpr[29] = vm_read32(ctx->gpr[3] + 0x48);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873C28;
loc_0087367C:
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x40D);
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 25, 25, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873670;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
    ctx->gpr[23] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[0] = ctx->gpr[0] | 0x400;
    vm_write32(ctx->gpr[3] + 0x18, ctx->gpr[0]);
    goto loc_00872A00;
loc_008736A8:
    ctx->lr = 0x008736AC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008736CC;
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873714;
loc_008736CC:
    ctx->lr = 0x008736D0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008736F0;
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873714;
loc_008736F0:
    ctx->lr = 0x008736F4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087290C;
    ctx->gpr[0] = vm_read16(ctx->gpr[30] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087290C;
loc_00873714:
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087373C;
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087373C;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x8);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008740F4;
loc_0087373C:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | ((uint64_t)0x8000 << 16);
    vm_write32(ctx->gpr[29] + 0x18, ctx->gpr[0]);
    goto loc_0087290C;
loc_0087374C:
    ctx->lr = 0x00873750; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873420;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873420;
    goto loc_008733B4;
loc_00873778:
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 23, 1, 31, 0, 0);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 29, 1, 31, 0, 0);
    ctx->gpr[11] = ppc_rlw64(ctx->gpr[11], 9, 0, 31, 0, 0);
    ctx->gpr[0] = ctx->gpr[10] | ((uint64_t)0xC000 << 16);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 3, 0, 31, 0, 0);
    ctx->gpr[11] = ctx->gpr[11] | 0x400;
    vm_write32(ctx->gpr[28] + 0x60, ctx->gpr[30]);
    vm_write32(ctx->gpr[28] + 0x64, ctx->gpr[29]);
    vm_write32(ctx->gpr[28] + 0x68, ctx->gpr[26]);
    vm_write32(ctx->gpr[28] + 0x6C, ctx->gpr[25]);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[11]);
    goto loc_008731D0;
loc_008737B0:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xC000 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x4);
    vm_write32(ctx->gpr[28] + 0x60, ctx->gpr[0]);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 21, 1, 31, 0, 0);
    ctx->gpr[10] = vm_read32(ctx->gpr[28] + 0xC);
    vm_write32(ctx->gpr[28] + 0x64, ctx->gpr[9]);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 11, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x68, ctx->gpr[11]);
    vm_write32(ctx->gpr[28] + 0x6C, ctx->gpr[10]);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[0]);
    goto loc_00872A00;
loc_008737EC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 24, 24, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873E90;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874108;
    ctx->gpr[0] = ctx->gpr[9] | ((uint64_t)0xC000 << 16);
    ctx->gpr[5] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[5]);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[6]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[10]);
loc_00873830:
    ctx->gpr[4] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[11] = ctx->gpr[5] + ctx->gpr[9];
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x50);
    ctx->gpr[10] = ctx->gpr[6] + ctx->gpr[10];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 31, 31, 0, 0);
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x54);
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x58);
    ctx->gpr[28] = ctx->gpr[5] - ctx->gpr[7];
    ctx->gpr[30] = vm_read32(ctx->gpr[26] + 0x5C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[31];
    ctx->gpr[9] = ctx->gpr[8] + ctx->gpr[30];
    ctx->gpr[15] = ctx->gpr[0] - ctx->gpr[11];
    ctx->gpr[14] = ctx->gpr[9] - ctx->gpr[10];
    ctx->gpr[25] = ctx->gpr[6] - ctx->gpr[8];
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874420;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[4], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008744DC;
loc_0087387C:
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x0087388C; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008738D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008738BC;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5AD);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6178);
    ctx->lr = 0x008738B8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008738BC:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 27, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 5, 0, 31, 0, 0);
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_008738D0:
    ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0x38);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873918;
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[28];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[15];
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[25];
    ctx->gpr[31] = (int64_t)(int32_t)ctx->gpr[14];
loc_008738EC:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00873908; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008738EC;
loc_00873918:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 24, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 8, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[0]);
loc_00873928:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[19], 0, 13, 13, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008739B4;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008739B4;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008746F8;
    ctx->gpr[8] = (int64_t)(int32_t)(0x301E);
    ctx->gpr[8] = ctx->gpr[0] & ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008739B4;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 18, 19, 0, 0);
    ctx->gpr[0] = (uint64_t)0 - ctx->gpr[0];
    ctx->gpr[31] = ppc_rlw64(ctx->gpr[0], 1, 31, 31, 0, 0);
    ctx->lr = 0x00873970; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873998;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6174);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[31];
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874898;
loc_00873998:
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[31];
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00874898;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 27, 4, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 5, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[0]);
loc_008739B4:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[19], 0, 9, 9, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873A48;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873A48;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008745BC;
    ctx->gpr[0] = ~(ctx->gpr[19] | ctx->gpr[19]);
    ctx->gpr[31] = ppc_sraw(&ctx->xer, (int32_t)ctx->gpr[0], 0x1F);
    ctx->lr = 0x008739E8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[31] = ppc_rlw64(ctx->gpr[31], 0, 29, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[31] + (int64_t)(0xC);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873A48;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873A48;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[20]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873A48;
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[23]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873A48;
    ctx->gpr[0] = ctx->gpr[20] + ctx->gpr[17];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873A48;
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[23];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873A48;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[0]);
loc_00873A48:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[19], 0, 3, 4, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873118;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6268);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF18); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[31] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xF14); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[30] = ftmp; }
    ctx->lr = 0x00873A64; june_evt_func_0082AF28(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00873A70; june_evt_func_0082AF38(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[6] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[19], 0, 3, 3, 0, 0);
    ctx->gpr[28] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[8] = ctx->gpr[6] | ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[30] = ctx->gpr[28] | ctx->gpr[28];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873AB8;
    ctx->fpr[0] = ctx->fpr[31];
    ctx->gpr[8] = ctx->gpr[1] + (int64_t)(0x80);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[8], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    ctx->gpr[8] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[30] = ctx->gpr[3] - ctx->gpr[9];
loc_00873AB8:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[19], 0, 4, 4, 0, 0);
    ctx->gpr[11] = ctx->gpr[7] | ctx->gpr[7];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[10] = ctx->gpr[31] | ctx->gpr[31];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008743C8;
    ctx->fpr[0] = ctx->fpr[30];
    ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x80);
    { int64_t iv = (int32_t)ppu_f2i32(ctx->fpr[0], 0); memcpy(&ctx->fpr[0], &iv, 8); }
    { uint64_t tmp; memcpy(&tmp, &ctx->fpr[0], 8); vm_write32(ctx->gpr[9], (uint32_t)tmp); }
    ctx->gpr[0] = vm_read32(ctx->gpr[1] + 0x80);
    ctx->gpr[11] = ctx->gpr[0] | ctx->gpr[0];
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 0, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[3] = ctx->gpr[29] - ctx->gpr[0];
    ctx->gpr[10] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008743C4;
loc_00873AF8:
    ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[7] - ctx->gpr[11];
    ctx->gpr[5] = ctx->gpr[6] - ctx->gpr[8];
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[4];
    ctx->gpr[9] = ctx->gpr[28] + ctx->gpr[5];
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[8]);
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[10]);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[30]);
    ctx->gpr[0] = ctx->gpr[10] - ctx->gpr[0];
    ctx->gpr[7] = ctx->gpr[30] - ctx->gpr[9];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873B68;
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[31] = (int64_t)(int32_t)ctx->gpr[7];
loc_00873B3C:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00873B58; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873B3C;
loc_00873B68:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | ((uint64_t)0x4000 << 16);
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[0]);
    goto loc_00873118;
loc_00873B78:
    ctx->gpr[0] = vm_read8(ctx->gpr[24] + 0x0);
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0087324C;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[4] = (int64_t)(int32_t)(5);
    vm_write8(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->lr = 0x00873BBC; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    vm_write8(ctx->gpr[24] + 0x0, ctx->gpr[30]);
    goto loc_0087324C;
loc_00873BC8:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write8(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00873C00; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    vm_write8(ctx->gpr[24] + 0x0, ctx->gpr[30]);
    goto loc_008732BC;
loc_00873C0C:
    ctx->gpr[28] = (int64_t)(int32_t)(0);
    goto loc_00872B28;
loc_00873C14:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873348;
    goto loc_008733B4;
loc_00873C28:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x618C);
    ctx->lr = 0x00873C34; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0087367C;
loc_00873C3C:
    ctx->gpr[9] = vm_read8(ctx->gpr[16] + 0x20);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x1C);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
    ctx->gpr[0] = ctx->gpr[0] | ((uint64_t)0x4000 << 16);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 1, 63);
    vm_write32(ctx->gpr[24] + 0x1C, ctx->gpr[0]);
    vm_write8(ctx->gpr[16] + 0x20, ctx->gpr[9]);
    goto loc_00873118;
loc_00873C5C:
    ctx->lr = 0x00873C60; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873D34;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873D34;
loc_00873C84:
    ctx->gpr[0] = (int64_t)(int8_t)ctx->gpr[20];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_00873D98;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873D98;
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873D98;
    ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873D98;
    ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873D98;
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873D98;
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873D98;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x7C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873D18;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873D08;
    goto loc_00873D98;
loc_00873CF8:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873D98;
loc_00873D08:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x7C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873CF8;
loc_00873D18:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ctx->gpr[9] | 0x400;
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874D40;
    goto loc_008731D0;
loc_00873D34:
    ctx->lr = 0x00873D38; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873D5C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873C84;
loc_00873D5C:
    ctx->lr = 0x00873D60; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873D84;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873C84;
loc_00873D84:
    ctx->gpr[0] = (int64_t)(int32_t)((uint32_t)0x80 << 16);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873C84;
loc_00873D98:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x1C);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008731D0;
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[30]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874194;
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874194;
    ctx->gpr[0] = ctx->gpr[30] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874194;
    ctx->gpr[0] = ctx->gpr[29] + ctx->gpr[25];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874194;
    ctx->gpr[8] = (int64_t)(int8_t)ctx->gpr[20];
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 28)) | (cr_val << 28); }
    if ((((ctx->cr >> 29) & 1) == 0)) goto loc_008731D0;
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008731D0;
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008731D0;
    ctx->gpr[7] = (int64_t)(int32_t)((uint32_t)0x80 << 16);
    ctx->gpr[8] = vm_read32(ctx->gpr[24] + 0x8);
    ctx->gpr[7] = ctx->gpr[7] | 0x1;
    goto loc_00873E50;
loc_00873E00:
    ctx->gpr[6] = vm_read32(ctx->gpr[9] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873E44;
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873E44;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873E44;
    ctx->gpr[10] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00873E44;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874D24;
loc_00873E44:
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008731D0;
loc_00873E50:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873E00;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
    goto loc_00873E44;
loc_00873E64:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873E84;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x498);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6188);
    ctx->lr = 0x00873E80; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00873E84:
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[23]);
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[19]);
    goto loc_00872A40;
loc_00873E90:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 25, 25, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873928;
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[8], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873ED4;
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xC000 << 16);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[10]);
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[8]);
loc_00873ED4:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874600;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x50, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x54, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x58, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x5C, ctx->gpr[10]);
loc_00873F08:
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_0087485C;
    ctx->gpr[0] = (int64_t)(int32_t)(0x15);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[0]);
loc_00873F14:
    ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[10] = ctx->gpr[20] + ctx->gpr[17];
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[11] = ctx->gpr[23] + ctx->gpr[18];
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[5] = ctx->gpr[23] - ctx->gpr[7];
    ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[7] + ctx->gpr[9];
    ctx->gpr[6] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[11];
    ctx->gpr[0] = ctx->gpr[20] - ctx->gpr[8];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873F8C;
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[6];
    ctx->gpr[31] = (int64_t)(int32_t)ctx->gpr[9];
loc_00873F60:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[27] | ctx->gpr[27];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[6] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[7] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00873F7C; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873F60;
loc_00873F8C:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x00873F9C; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873FB4;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[0]);
loc_00873FB4:
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x00873FC4; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087404C;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873FF8;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5EE);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x00873FF4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00873FF8:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874160;
loc_00874008:
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00874868;
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1D4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1C8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1CC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1D0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
loc_00874038:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x00874048; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0087404C:
    ctx->gpr[4] = (int64_t)(int32_t)(6);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x0087405C; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087409C;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)4; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087408C;
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6178);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5FB);
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->lr = 0x00874088; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_0087408C:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ctx->gpr[0] | 0x10;
    vm_write16(ctx->gpr[9] + 0x0, ctx->gpr[0]);
loc_0087409C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 25, 1, 31, 0, 0);
    ctx->gpr[0] = ctx->gpr[0] | 0x2;
    ctx->gpr[9] = ppc_rlw64(ctx->gpr[9], 7, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[9]);
    goto loc_00873118;
loc_008740BC:
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[27]);
    vm_write32(ctx->gpr[28] + 0x74, ctx->gpr[11]);
    goto loc_0087329C;
loc_008740C8:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x74);
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[0]);
    goto loc_0087327C;
loc_008740D4:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x70);
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[0]);
    goto loc_00873264;
loc_008740E0:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x70);
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[0]);
    goto loc_0087362C;
loc_008740EC:
    if ((((ctx->cr >> 7) & 1) == 1)) goto loc_008733A8;
    goto loc_00872894;
loc_008740F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0xC);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_0087290C;
    goto loc_0087373C;
loc_00874108:
    ctx->gpr[5] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0xC);
    goto loc_00873830;
loc_0087411C:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    vm_write8(ctx->gpr[24] + 0x0, ctx->gpr[0]);
    ctx->gpr[4] = (int64_t)(int32_t)(5);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x0);
    ctx->gpr[5] = (int64_t)(int32_t)(0);
    vm_write64(ctx->gpr[1] + 0x28, ctx->gpr[2]);
    ctx->ctr = ctx->gpr[0];
    ctx->gpr[6] = (int64_t)(int32_t)(0);
    ctx->gpr[7] = (int64_t)(int32_t)(0);
    ctx->gpr[2] = vm_read32(ctx->gpr[9] + 0x4);
    ctx->gpr[8] = (int64_t)(int32_t)(0);
    ctx->lr = 0x00874154; ps3_indirect_call(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[2] = june_evt_overlay_toc(); 
    vm_write8(ctx->gpr[24] + 0x0, ctx->gpr[30]);
    goto loc_0087354C;
loc_00874160:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5F0);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x617C);
    ctx->lr = 0x00874170; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00874008;
loc_0087417C:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x74);
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[0]);
    goto loc_00873644;
loc_00874188:
    ctx->gpr[19] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[28] + 0x74, ctx->gpr[23]);
    goto loc_00873660;
loc_00874194:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008731D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x200;
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[0]);
    goto loc_008731D0;
loc_008741B4:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874200;
    ctx->gpr[29] = vm_read32(ctx->gpr[30] + 0x80);
    ctx->gpr[9] = ctx->gpr[9] | 0x4;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x200;
    vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[9]);
    vm_write32(ctx->gpr[30] + 0x18, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874298;
loc_008741E4:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x008741F4; june_evt_func_0086FE44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008741E4;
loc_00874200:
    ctx->gpr[27] = vm_read32(ctx->gpr[30] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874298;
    ctx->gpr[30] = (int64_t)(int32_t)(0);
loc_00874210:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x7C);
    ctx->gpr[31] = vm_read32(ctx->gpr[28] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008742D4;
loc_00874224:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008742AC;
    ;
loc_00874238:
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874238;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x84);
    vm_write32(ctx->gpr[9] + 0x84, ctx->gpr[0]);
loc_00874250:
    ctx->gpr[29] = vm_read32(ctx->gpr[28] + 0x80);
    vm_write32(ctx->gpr[28] + 0x78, ctx->gpr[27]);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[28] + 0x7C, ctx->gpr[30]);
    vm_write32(ctx->gpr[28] + 0x84, ctx->gpr[30]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874284;
loc_00874268:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00874278; june_evt_func_0086F6D0(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874268;
loc_00874284:
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[27] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x00874290; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874210;
loc_00874298:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->lr = 0x008742A0; june_evt_func_00870D78(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    vm_write32(ctx->gpr[24] + 0xC, ctx->gpr[0]);
    goto loc_00872AEC;
loc_008742AC:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x84);
    vm_write32(ctx->gpr[9] + 0x80, ctx->gpr[0]);
    goto loc_00874250;
loc_008742B8:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x49C);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6184);
    ctx->lr = 0x008742C8; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x20);
    goto loc_00872A2C;
loc_008742D4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x23B);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6214);
    ctx->lr = 0x008742E4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00874224;
loc_008742EC:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x008742FC; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874314;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[0]);
loc_00874314:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x00874324; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872F50;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874358;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x53B);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x00874354; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00874358:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874AB0;
loc_00874368:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087482C;
    ctx->gpr[11] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
loc_0087439C:
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x008743AC; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    goto loc_00872F50;
loc_008743C4:
    ctx->gpr[11] = ctx->gpr[7] | ctx->gpr[7];
loc_008743C8:
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873AF8;
    { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873AF8;
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int32_t)ctx->gpr[28]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00873AF8;
    goto loc_00873118;
loc_008743E4:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x409);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x618C);
    ctx->lr = 0x008743F4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_0087367C;
loc_008743FC:
    ctx->gpr[19] = ctx->gpr[27] | ctx->gpr[27];
    vm_write32(ctx->gpr[28] + 0x74, ctx->gpr[23]);
    goto loc_00873520;
loc_00874408:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x74);
    vm_write32(ctx->gpr[24] + 0x4, ctx->gpr[0]);
    goto loc_00873504;
loc_00874414:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x70);
    vm_write32(ctx->gpr[24] + 0x20, ctx->gpr[0]);
    goto loc_008734EC;
loc_00874420:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[31]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[30]);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[7]);
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[8]);
    ctx->lr = 0x00874440; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008744CC;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874474;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x583);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x00874470; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00874474:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874C20;
loc_00874484:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00874A80;
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
loc_008744B8:
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x008744C8; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008744CC:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    goto loc_0087387C;
loc_008744DC:
    ctx->gpr[4] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[31]);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[30]);
    ctx->lr = 0x008744F4; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087450C;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
    ctx->gpr[0] = ctx->gpr[0] | 0x1;
    vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[0]);
loc_0087450C:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x0087451C; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008745A8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874550;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x59D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x0087454C; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00874550:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874B88;
loc_00874560:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00874ACC;
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xE4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xD8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xDC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xE0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
loc_00874594:
    ctx->gpr[4] = (int64_t)(int32_t)(3);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x008745A4; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_008745A8:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 30, 1, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 2, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    goto loc_0087387C;
loc_008745BC:
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873A48;
    ctx->lr = 0x008745CC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008745F0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6174);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874C3C;
loc_008745F0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[0]);
    goto loc_00873A48;
loc_00874600:
    ctx->gpr[4] = (int64_t)(int32_t)(4);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x50);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x54);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x58);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x5C);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[10]);
    ctx->lr = 0x00874634; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873F08;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874668;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5CC);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x00874664; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00874668:
    ctx->gpr[31] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874B58;
loc_00874678:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00874B28;
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->gpr[11] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xB8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xBC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
loc_008746AC:
    ctx->gpr[4] = (int64_t)(int32_t)(2);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[30], 0, 32);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x008746BC; june_evt_func_0086BEA4(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 0, 30, 0, 0);
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    goto loc_00873F08;
loc_008746D0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x55C);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x617C);
    ctx->lr = 0x008746E0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_0087306C;
loc_008746EC:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x84);
    vm_write32(ctx->gpr[9] + 0x80, ctx->gpr[0]);
    goto loc_00872D7C;
loc_008746F8:
    ctx->gpr[4] = (int64_t)(int32_t)(1);
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[4] = ctx->gpr[4] | ((uint64_t)0xFF00 << 16);
    ctx->lr = 0x00874708; june_evt_func_0086C634(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[31] = ctx->gpr[3] | ctx->gpr[3];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874748;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)2; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874738;
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x60D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6180);
    ctx->lr = 0x00874734; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
loc_00874738:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x1C);
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008739B4;
loc_00874748:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874A38;
    ctx->gpr[0] = ctx->gpr[20] + (int64_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874788;
    ctx->gpr[0] = ctx->gpr[23] + (int64_t)(0xC);
    ctx->gpr[31] = (int64_t)(int32_t)(6);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874A58;
    ctx->gpr[9] = ctx->gpr[23] + ctx->gpr[18];
    ctx->gpr[31] = (int64_t)(int32_t)(0x12);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-12);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874A58;
loc_00874788:
    ctx->gpr[11] = ctx->gpr[20] + ctx->gpr[17];
    ctx->gpr[0] = ctx->gpr[11] + (int64_t)(-12);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008747BC;
    ctx->gpr[0] = ctx->gpr[23] + (int64_t)(0xC);
    ctx->gpr[31] = (int64_t)(int32_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874A58;
    ctx->gpr[9] = ctx->gpr[23] + ctx->gpr[18];
    ctx->gpr[31] = (int64_t)(int32_t)(0x18);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-12);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874A58;
loc_008747BC:
    ctx->gpr[0] = ctx->gpr[20] + (int64_t)(4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874B20;
    ctx->gpr[0] = ctx->gpr[11] + (int64_t)(-4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874A54;
    ctx->gpr[0] = ctx->gpr[23] + (int64_t)(4);
    ctx->gpr[31] = (int64_t)(int32_t)(4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874A58;
    ctx->gpr[9] = ctx->gpr[23] + ctx->gpr[18];
    ctx->gpr[31] = (int64_t)(int32_t)(0x10);
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874A58;
loc_008747F8:
    ctx->gpr[31] = (int64_t)(int32_t)(0);
loc_008747FC:
    ctx->lr = 0x00874800; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008739B4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6194);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008739B4;
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[31]);
    goto loc_008739B4;
loc_0087482C:
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->gpr[11] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    goto loc_0087439C;
loc_0087485C:
    ctx->gpr[0] = (int64_t)(int32_t)(0xE);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[0]);
    goto loc_00873F14;
loc_00874868:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1E4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1D8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1DC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0x1E0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_00874038;
loc_00874898:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[9], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874B74;
    ctx->gpr[0] = ctx->gpr[9] | ((uint64_t)0xC000 << 16);
    ctx->gpr[16] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[28] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[27] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[25] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[16]);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[28]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[27]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[25]);
loc_008748D0:
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874904;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[9] = (int64_t)(int32_t)(1);
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x6190);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 30, 30, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write16(ctx->gpr[8] + 0x0, ctx->gpr[9]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874DB4;
loc_008748F4:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 28, 28, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874D48;
loc_00874904:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x9C);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_0087491C;
    ctx->gpr[0] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[9]);
loc_0087491C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xA4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874930;
    ctx->gpr[0] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[9]);
loc_00874930:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x5F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00874940;
    ctx->gpr[0] = (int64_t)(int32_t)(0x60);
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[0]);
loc_00874940:
    { int64_t a = (int32_t)ctx->gpr[30]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0087496C;
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 29, 29, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874F00;
loc_0087495C:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 27, 27, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874E94;
loc_0087496C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xA0);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0xC);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874984;
    ctx->gpr[0] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[9]);
loc_00874984:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xA8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874998;
    ctx->gpr[0] = ctx->gpr[9] | ctx->gpr[9];
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[9]);
loc_00874998:
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00874B0C;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x14; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008749AC;
    ctx->gpr[0] = (int64_t)(int32_t)(0x15);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[0]);
loc_008749AC:
    ctx->gpr[29] = vm_read32(ctx->gpr[26] + 0x38);
    ctx->gpr[10] = ctx->gpr[16] + ctx->gpr[27];
    ctx->gpr[8] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[11] = ctx->gpr[28] + ctx->gpr[25];
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[4] = ctx->gpr[16] - ctx->gpr[8];
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[5] = ctx->gpr[28] - ctx->gpr[7];
    ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[7] + ctx->gpr[9];
    ctx->gpr[0] = ctx->gpr[0] - ctx->gpr[10];
    ctx->gpr[9] = ctx->gpr[9] - ctx->gpr[11];
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874A24;
    ctx->gpr[31] = (int64_t)(int32_t)ctx->gpr[4];
    ctx->gpr[30] = (int64_t)(int32_t)ctx->gpr[5];
    ctx->gpr[28] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[27] = (int64_t)(int32_t)ctx->gpr[9];
loc_008749F8:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[31] | ctx->gpr[31];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[5] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[6] = ctx->gpr[28] | ctx->gpr[28];
    ctx->gpr[7] = ctx->gpr[27] | ctx->gpr[27];
    ctx->lr = 0x00874A14; june_evt_func_0086BFB0(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x28);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008749F8;
loc_00874A24:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 18, 2, 31, 0, 0);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 14, 0, 31, 0, 0);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[0]);
    goto loc_008739B4;
loc_00874A38:
    ctx->gpr[0] = ctx->gpr[20] + (int64_t)(4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00874B20;
    ctx->gpr[9] = ctx->gpr[20] + ctx->gpr[17];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-4);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008747F8;
loc_00874A54:
    ctx->gpr[31] = (int64_t)(int32_t)(8);
loc_00874A58:
    ctx->gpr[8] = vm_read32(ctx->gpr[2] + -0x6190);
    ctx->gpr[0] = (int64_t)(int32_t)(1);
    vm_write16(ctx->gpr[8] + 0x0, ctx->gpr[0]);
    goto loc_008747FC;
loc_00874A68:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x23B);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x6214);
    ctx->lr = 0x00874A78; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    goto loc_00872D54;
loc_00874A80:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_008744B8;
loc_00874AB0:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x53D);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x617C);
    ctx->lr = 0x00874AC0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00874368;
loc_00874ACC:
    ctx->gpr[11] = ctx->gpr[11] + (int64_t)(0xC);
    ctx->gpr[9] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[11] = ppc_rldicl(ctx->gpr[11], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xD4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xC8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xCC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[9] + 0xD0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[11] + 0x8, tmp); }
    goto loc_00874594;
loc_00874AFC:
    ctx->gpr[0] = vm_read32(ctx->gpr[8] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00872E9C;
    goto loc_00872ED8;
loc_00874B0C:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_008749AC;
    ctx->gpr[0] = (int64_t)(int32_t)(0xD);
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[0]);
    goto loc_008749AC;
loc_00874B20:
    ctx->gpr[31] = (int64_t)(int32_t)(2);
    goto loc_00874A58;
loc_00874B28:
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(0xC);
    ctx->gpr[11] = ctx->gpr[16] + (int64_t)(int32_t)((uint32_t)0x1 << 16);
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xB4); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[0] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xA8); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[13] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xAC); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[12] = ftmp; }
    { uint32_t tmp = vm_read32(ctx->gpr[11] + 0xB0); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[11] = ftmp; }
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0xC, tmp); }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x0, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x4, tmp); }
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[9] + 0x8, tmp); }
    goto loc_008746AC;
loc_00874B58:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x5CE);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x617C);
    ctx->lr = 0x00874B68; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00874678;
loc_00874B74:
    ctx->gpr[16] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[28] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[27] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[25] = vm_read32(ctx->gpr[26] + 0xC);
    goto loc_008748D0;
loc_00874B88:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x59F);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x617C);
    ctx->lr = 0x00874B98; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00874560;
loc_00874BA4:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872CC8;
    ctx->gpr[10] = (int64_t)(int32_t)((uint32_t)0x80 << 16);
    ctx->gpr[10] = ctx->gpr[10] | 0x1;
    goto loc_00874C0C;
loc_00874BBC:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x1C);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874C00;
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[22]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874C00;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x8);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[22]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874C00;
    ctx->gpr[11] = vm_read32(ctx->gpr[9] + 0x4);
    { int64_t a = (int32_t)ctx->gpr[21]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00874C00;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0xC);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[21]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00872ED8;
loc_00874C00:
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x74);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00872CC8;
loc_00874C0C:
    { int64_t a = (int32_t)ctx->gpr[25]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874BBC;
    ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
    goto loc_00874C00;
loc_00874C20:
    ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x6264);
    ctx->gpr[4] = (int64_t)(int32_t)(0x585);
    ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x617C);
    ctx->lr = 0x00874C30; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x4);
    goto loc_00874484;
loc_00874C3C:
    ctx->lr = 0x00874C40; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874C60;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874C84;
loc_00874C60:
    ctx->lr = 0x00874C64; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873A48;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00873A48;
loc_00874C84:
    ctx->gpr[10] = vm_read32(ctx->gpr[26] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[10], 0, 1, 1, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00874D1C;
    ctx->gpr[10] = ctx->gpr[10] | ((uint64_t)0xC000 << 16);
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0xC);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[10]);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[31]);
loc_00874CBC:
    ctx->lr = 0x00874CC0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874CE0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874CE0:
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[31];
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x4);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    ctx->lr = 0x00874CF0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874D10;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874D10:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[0];
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[0]);
    goto loc_00873A48;
loc_00874D1C:
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x0);
    goto loc_00874CBC;
loc_00874D24:
    { int64_t a = (int32_t)ctx->gpr[27]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008731D0;
    ctx->gpr[0] = vm_read32(ctx->gpr[9] + 0x18);
    { int64_t a = (int32_t)ctx->gpr[6]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ctx->gpr[0] | 0x500;
    vm_write32(ctx->gpr[9] + 0x18, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008731D0;
loc_00874D40:
    ctx->gpr[20] = (int64_t)(int32_t)(1);
    goto loc_008731D0;
loc_00874D48:
    ctx->lr = 0x00874D4C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874D6C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874D6C:
    ctx->gpr[9] = ctx->gpr[20] + ctx->gpr[17];
    ctx->gpr[0] = ctx->gpr[22] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-12);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00874904;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->lr = 0x00874D88; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874DA8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874DA8:
    ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[31];
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[0]);
    goto loc_00874904;
loc_00874DB4:
    ctx->lr = 0x00874DB8; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874DD8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874DD8:
    ctx->gpr[0] = ctx->gpr[22] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[20] + (int64_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008748F4;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->lr = 0x00874DF0; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874E10;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874E10:
    ctx->gpr[0] = ctx->gpr[31] - ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x5F; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[26] + 0x8, ctx->gpr[0]);
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874E54;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->lr = 0x00874E28; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874E48;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874E48:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[0];
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[0]);
    goto loc_008748F4;
loc_00874E54:
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->lr = 0x00874E5C; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874E7C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x6170);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874E7C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-96);
    vm_write32(ctx->gpr[26] + 0x0, ctx->gpr[9]);
    goto loc_008748F4;
loc_00874E94:
    ctx->lr = 0x00874E98; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874EB8;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874EB8:
    ctx->gpr[9] = ctx->gpr[23] + ctx->gpr[18];
    ctx->gpr[0] = ctx->gpr[21] - ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-12);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_0087496C;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->lr = 0x00874ED4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874EF4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874EF4:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[0];
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[0]);
    goto loc_0087496C;
loc_00874F00:
    ctx->lr = 0x00874F04; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874F24;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874F24:
    ctx->gpr[9] = ctx->gpr[21] - ctx->gpr[0];
    ctx->gpr[0] = ctx->gpr[23] + (int64_t)(0xC);
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087495C;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->lr = 0x00874F40; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874F60;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874F60:
    { int64_t a = (int32_t)ctx->gpr[19]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
    ctx->gpr[0] = ctx->gpr[31] - ctx->gpr[0];
    vm_write32(ctx->gpr[26] + 0xC, ctx->gpr[0]);
    if ((((ctx->cr >> 15) & 1) == 1)) goto loc_00874FEC;
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0x14; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 0)) goto loc_00874FAC;
loc_00874F78:
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->lr = 0x00874F80; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874FA0;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874FA0:
    ctx->gpr[0] = ctx->gpr[31] + ctx->gpr[0];
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[0]);
    goto loc_0087495C;
loc_00874FAC:
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->lr = 0x00874FB4; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00874FD4;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_00874FD4:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-21);
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[9]);
    goto loc_0087495C;
loc_00874FEC:
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0xC; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 2) & 1) == 1)) goto loc_00874F78;
    ctx->gpr[31] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->lr = 0x00874FFC; june_evt_func_008861FC(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = (int64_t)(int8_t)ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087501C;
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x616C);
    ctx->gpr[0] = vm_read16(ctx->gpr[9] + 0x0);
    ctx->gpr[0] = (int64_t)(int16_t)ctx->gpr[0];
loc_0087501C:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[9] = ctx->gpr[31] + ctx->gpr[9];
    ctx->gpr[9] = ctx->gpr[9] + ctx->gpr[0];
    ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-13);
    vm_write32(ctx->gpr[26] + 0x4, ctx->gpr[9]);
    goto loc_0087495C;
}

void june_evt_func_008268F8(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read16(ctx->gpr[3] + 0x3C);
    return;
}

void june_evt_func_00826900(ppu_context* ctx) {
    PPU_MM_SNAP
    ctx->gpr[3] = vm_read16(ctx->gpr[3] + 0x3E);
    return;
}

void june_evt_func_00826908(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x00826930; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x00826940; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x00826954; june_evt_func_008ECFE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x7400);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[26]);
    vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[9]);
    ctx->lr = 0x00826978; june_evt_func_008EDDE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_0082699C(ppu_context* ctx) {
    PPU_MM_SNAP
    uint64_t _cs_26 = ctx->gpr[26];
    uint64_t _cs_27 = ctx->gpr[27];
    uint64_t _cs_28 = ctx->gpr[28];
    uint64_t _cs_29 = ctx->gpr[29];
    vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
    ctx->gpr[26] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[29]);
    ctx->gpr[27] = ctx->gpr[3] | ctx->gpr[3];
    ctx->lr = 0x008269C4; june_evt_func_00827B24(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = (int64_t)(int32_t)(0x20);
    ctx->lr = 0x008269D4; june_evt_func_008EED20(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
    ctx->lr = 0x008269E8; june_evt_func_008ECFE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = (int64_t)(int32_t)(0);
    ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x73FC);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[27], 0, 32);
    ctx->gpr[4] = ctx->gpr[28] | ctx->gpr[28];
    vm_write32(ctx->gpr[29] + 0x0, ctx->gpr[0]);
    vm_write32(ctx->gpr[29] + 0x4, ctx->gpr[26]);
    vm_write32(ctx->gpr[29] + 0xC, ctx->gpr[9]);
    ctx->lr = 0x00826A0C; june_evt_func_008EDDE8(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
    ctx->gpr[26] = _cs_26;
    ctx->gpr[27] = _cs_27;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[28] = _cs_28;
    ctx->gpr[29] = _cs_29;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
    return;
}

void june_evt_func_00870214(ppu_context* ctx) {
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
    vm_write64(ctx->gpr[1] + -0x180, ctx->gpr[1]); ctx->gpr[1] += -0x180;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x168, ctx->gpr[29]);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    vm_write64(ctx->gpr[1] + 0x190, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x130, ctx->gpr[22]);
    ctx->gpr[22] = vm_read32(ctx->gpr[2] + -0x6210);
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0x70);
    vm_write64(ctx->gpr[1] + 0x148, ctx->gpr[25]);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x60);
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x64);
    ctx->gpr[25] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[0];
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x68);
    ctx->gpr[10] = vm_read32(ctx->gpr[29] + 0x6C);
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[9];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[11];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[10];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0x158, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0x108, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x110, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x118, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0x120, ctx->gpr[20]);
    vm_write64(ctx->gpr[1] + 0x128, ctx->gpr[21]);
    vm_write64(ctx->gpr[1] + 0x138, ctx->gpr[23]);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    vm_write64(ctx->gpr[1] + 0x140, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x150, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x160, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0x170, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0x178, ctx->gpr[31]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x70, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x74, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x78, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x7C, tmp); }
    ctx->lr = 0x008702E8; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x6270);
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x10);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write32(ctx->gpr[29] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087038C;
    ;
loc_00870310:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[31] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870380;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870380;
    ctx->gpr[7] = vm_read32(ctx->gpr[31] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[31] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x60);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870380;
    ctx->gpr[12] = vm_read32(ctx->gpr[31] + 0x4);
    ctx->gpr[30] = vm_read32(ctx->gpr[31] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[29] + 0x64);
    ctx->gpr[0] = ctx->gpr[12] + ctx->gpr[30];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870380;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x68);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870380;
    ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x6C);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[12]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008703D8;
    ;
loc_00870380:
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00870310;
loc_0087038C:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x190);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x180);
    return;
loc_008703D8:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[12];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[6];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[30];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xD0);
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD0, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[31] + 0x60, ctx->gpr[7]);
    vm_write32(ctx->gpr[31] + 0x64, ctx->gpr[12]);
    vm_write32(ctx->gpr[31] + 0x68, ctx->gpr[6]);
    vm_write32(ctx->gpr[31] + 0x6C, ctx->gpr[30]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xD8, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xDC, tmp); }
    ctx->lr = 0x00870468; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870380;
loc_00870488:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[30] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008704E8;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008704E8;
    ctx->gpr[7] = vm_read32(ctx->gpr[30] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[30] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x60);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008704E8;
    ctx->gpr[12] = vm_read32(ctx->gpr[30] + 0x4);
    ctx->gpr[28] = vm_read32(ctx->gpr[30] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[31] + 0x64);
    ctx->gpr[0] = ctx->gpr[12] + ctx->gpr[28];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008704E8;
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x68);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_008704F8;
    ;
loc_008704E8:
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00870488;
    goto loc_00870380;
loc_008704F8:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x6C);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[12]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008704E8;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[12];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[6];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[28];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xC0);
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC0, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[30] + 0x60, ctx->gpr[7]);
    vm_write32(ctx->gpr[30] + 0x64, ctx->gpr[12]);
    vm_write32(ctx->gpr[30] + 0x68, ctx->gpr[6]);
    vm_write32(ctx->gpr[30] + 0x6C, ctx->gpr[28]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xC8, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xCC, tmp); }
    ctx->lr = 0x00870598; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[30] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008704E8;
loc_008705B8:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[28] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870624;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870624;
    ctx->gpr[7] = vm_read32(ctx->gpr[28] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[28] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x60);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870624;
    ctx->gpr[12] = vm_read32(ctx->gpr[28] + 0x4);
    ctx->gpr[26] = vm_read32(ctx->gpr[28] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x64);
    ctx->gpr[0] = ctx->gpr[12] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870624;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x68);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870624;
    ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x6C);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[12]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 0)) goto loc_00870634;
loc_00870624:
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008705B8;
    goto loc_008704E8;
loc_00870634:
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[12];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[6];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[26];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xB0);
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB0, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[28] + 0x60, ctx->gpr[7]);
    vm_write32(ctx->gpr[28] + 0x64, ctx->gpr[12]);
    vm_write32(ctx->gpr[28] + 0x68, ctx->gpr[6]);
    vm_write32(ctx->gpr[28] + 0x6C, ctx->gpr[26]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xB8, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xBC, tmp); }
    ctx->lr = 0x008706C4; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[28] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870624;
    ctx->gpr[17] = ctx->gpr[1] + (int64_t)(0x90);
    goto loc_008706F8;
loc_008706EC:
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870624;
loc_008706F8:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[24] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[7] = vm_read32(ctx->gpr[24] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[24] + 0x8);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x60);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[26] = vm_read32(ctx->gpr[24] + 0x4);
    ctx->gpr[23] = vm_read32(ctx->gpr[24] + 0xC);
    ctx->gpr[11] = vm_read32(ctx->gpr[28] + 0x64);
    ctx->gpr[0] = ctx->gpr[26] + ctx->gpr[23];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x68);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x6C);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[26]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[7];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[26];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[6];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[5] = ctx->gpr[1] + (int64_t)(0xA0);
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[5], 0, 32);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA0, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[24] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[24] + 0x60, ctx->gpr[7]);
    vm_write32(ctx->gpr[24] + 0x64, ctx->gpr[26]);
    vm_write32(ctx->gpr[24] + 0x68, ctx->gpr[6]);
    vm_write32(ctx->gpr[24] + 0x6C, ctx->gpr[23]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xA8, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xAC, tmp); }
    ctx->lr = 0x008707F4; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[24] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008706EC;
    ctx->gpr[18] = ctx->gpr[1] + (int64_t)(0x80);
    goto loc_00870828;
loc_0087081C:
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008706EC;
loc_00870828:
    ctx->gpr[20] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[20] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[6] = vm_read32(ctx->gpr[20] + 0x0);
    ctx->gpr[26] = vm_read32(ctx->gpr[20] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[24] + 0x60);
    ctx->gpr[0] = ctx->gpr[6] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[7] = vm_read32(ctx->gpr[20] + 0x4);
    ctx->gpr[23] = vm_read32(ctx->gpr[20] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x64);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[23];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x68);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x6C);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[26];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[17], 0, 32);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x90, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[20] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[20] + 0x60, ctx->gpr[6]);
    vm_write32(ctx->gpr[20] + 0x64, ctx->gpr[7]);
    vm_write32(ctx->gpr[20] + 0x68, ctx->gpr[26]);
    vm_write32(ctx->gpr[20] + 0x6C, ctx->gpr[23]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x94, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x98, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x9C, tmp); }
    ctx->lr = 0x00870920; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[20] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087081C;
    ctx->gpr[19] = ctx->gpr[1] + (int64_t)(0xE0);
    goto loc_00870954;
loc_00870948:
    ctx->gpr[9] = vm_read32(ctx->gpr[21] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087081C;
loc_00870954:
    ctx->gpr[21] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[21] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870948;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870948;
    ctx->gpr[6] = vm_read32(ctx->gpr[21] + 0x0);
    ctx->gpr[26] = vm_read32(ctx->gpr[21] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[20] + 0x60);
    ctx->gpr[0] = ctx->gpr[6] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870948;
    ctx->gpr[7] = vm_read32(ctx->gpr[21] + 0x4);
    ctx->gpr[23] = vm_read32(ctx->gpr[21] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[20] + 0x64);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[23];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870948;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x68);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870948;
    ctx->gpr[0] = vm_read32(ctx->gpr[20] + 0x6C);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870948;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[26];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[23];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[18], 0, 32);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x80, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[21] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[21] + 0x60, ctx->gpr[6]);
    vm_write32(ctx->gpr[21] + 0x64, ctx->gpr[7]);
    vm_write32(ctx->gpr[21] + 0x68, ctx->gpr[26]);
    vm_write32(ctx->gpr[21] + 0x6C, ctx->gpr[23]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x84, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x88, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0x8C, tmp); }
    ctx->lr = 0x00870A4C; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[21] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00870A7C;
    goto loc_00870948;
loc_00870A70:
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870948;
loc_00870A7C:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[8] = vm_read32(ctx->gpr[23] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[8]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870A70;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870A70;
    ctx->gpr[6] = vm_read32(ctx->gpr[23] + 0x0);
    ctx->gpr[26] = vm_read32(ctx->gpr[23] + 0x8);
    ctx->gpr[11] = vm_read32(ctx->gpr[21] + 0x60);
    ctx->gpr[0] = ctx->gpr[6] + ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870A70;
    ctx->gpr[7] = vm_read32(ctx->gpr[23] + 0x4);
    ctx->gpr[12] = vm_read32(ctx->gpr[23] + 0xC);
    ctx->gpr[9] = vm_read32(ctx->gpr[21] + 0x64);
    ctx->gpr[0] = ctx->gpr[7] + ctx->gpr[12];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870A70;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x68);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[6]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870A70;
    ctx->gpr[0] = vm_read32(ctx->gpr[21] + 0x6C);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[7]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870A70;
    ctx->gpr[0] = (int64_t)(int32_t)ctx->gpr[6];
    { uint32_t tmp = vm_read32(ctx->gpr[2] + -0x620C); float ftmp; memcpy(&ftmp, &tmp, 4); ctx->fpr[1] = ftmp; }
    ctx->gpr[9] = (int64_t)(int32_t)ctx->gpr[7];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[0]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    ctx->gpr[11] = (int64_t)(int32_t)ctx->gpr[26];
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[9]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[11], &tmp, 8); }
    ctx->gpr[10] = (int64_t)(int32_t)ctx->gpr[12];
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[11]);
    { int64_t iv; memcpy(&iv, &ctx->fpr[11], 8); ctx->fpr[11] = (double)iv; }
    ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0x8000 << 16);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[4] = ctx->gpr[22] | ctx->gpr[22];
    ctx->gpr[5] = ppc_rldicl(ctx->gpr[19], 0, 32);
    ctx->fpr[0] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write64(ctx->gpr[1] + 0xF0, ctx->gpr[10]);
    ctx->fpr[13] = ppu_frsp(ctx->fpr[11]);
    { float ftmp = (float)ctx->fpr[0]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE0, tmp); }
    ctx->fpr[12] = ppu_frsp(ctx->fpr[10]);
    { uint64_t tmp = vm_read64(ctx->gpr[1] + 0xF0); memcpy(&ctx->fpr[10], &tmp, 8); }
    { int64_t iv; memcpy(&iv, &ctx->fpr[10], 8); ctx->fpr[10] = (double)iv; }
    vm_write32(ctx->gpr[23] + 0x10, ctx->gpr[8]);
    vm_write32(ctx->gpr[23] + 0x60, ctx->gpr[6]);
    vm_write32(ctx->gpr[23] + 0x64, ctx->gpr[7]);
    vm_write32(ctx->gpr[23] + 0x68, ctx->gpr[26]);
    vm_write32(ctx->gpr[23] + 0x6C, ctx->gpr[12]);
    { float ftmp = (float)ctx->fpr[13]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE4, tmp); }
    { float ftmp = (float)ctx->fpr[12]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xE8, tmp); }
    ctx->fpr[11] = ppu_frsp(ctx->fpr[10]);
    { float ftmp = (float)ctx->fpr[11]; uint32_t tmp; memcpy(&tmp, &ftmp, 4); vm_write32(ctx->gpr[1] + 0xEC, tmp); }
    ctx->lr = 0x00870B74; june_evt_func_008EFC90(ctx); DRAIN_TRAMPOLINE(ctx);
    ;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x10);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x20);
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 1, 1, 31, 0, 0);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 31, 0, 31, 0, 0);
    vm_write32(ctx->gpr[23] + 0x10, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870A70;
loc_00870B94:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[9], 0, 32);
    ctx->gpr[7] = vm_read32(ctx->gpr[26] + 0x10);
    { int64_t a = (int32_t)ctx->gpr[7]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870C24;
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x8);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int32_t)ctx->gpr[0]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870C24;
    ctx->gpr[11] = vm_read32(ctx->gpr[26] + 0x0);
    ctx->gpr[6] = vm_read32(ctx->gpr[26] + 0x8);
    ctx->gpr[10] = vm_read32(ctx->gpr[23] + 0x60);
    ctx->gpr[0] = ctx->gpr[11] + ctx->gpr[6];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[10]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870C24;
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x4);
    ctx->gpr[5] = vm_read32(ctx->gpr[26] + 0xC);
    ctx->gpr[8] = vm_read32(ctx->gpr[23] + 0x64);
    ctx->gpr[0] = ctx->gpr[9] + ctx->gpr[5];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[8]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870C24;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x68);
    ctx->gpr[0] = ctx->gpr[10] + ctx->gpr[0];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[11]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870C24;
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x6C);
    ctx->gpr[10] = ctx->gpr[7] | ((uint64_t)0x8000 << 16);
    ctx->gpr[3] = ctx->gpr[25] | ctx->gpr[25];
    ctx->gpr[0] = ctx->gpr[8] + ctx->gpr[0];
    ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int32_t)ctx->gpr[9]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 3) & 1) == 1)) goto loc_00870C24;
    vm_write32(ctx->gpr[26] + 0x10, ctx->gpr[10]);
    vm_write32(ctx->gpr[26] + 0x60, ctx->gpr[11]);
    vm_write32(ctx->gpr[26] + 0x64, ctx->gpr[9]);
    vm_write32(ctx->gpr[26] + 0x68, ctx->gpr[6]);
    vm_write32(ctx->gpr[26] + 0x6C, ctx->gpr[5]);
    ctx->lr = 0x00870C24; june_evt_func_00870214(ctx); DRAIN_TRAMPOLINE(ctx);
loc_00870C24:
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x70);
    { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00870B94;
    goto loc_00870A70;
}





void june_evt_func_0086FB48(ppu_context* ctx) {
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
    ctx->gpr[9] = ctx->gpr[3] | ctx->gpr[3];
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[22]);
    ctx->gpr[0] = vm_read64(ctx->gpr[9] + 0x90);
    ctx->gpr[29] = ctx->gpr[4] | ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[4]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FBCC;
loc_0086FB90:
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
    ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xC0);
    return;
loc_0086FBCC:
    ctx->gpr[3] = vm_read32(ctx->gpr[9] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FD3C;
loc_0086FBD8:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[22] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FD30;
loc_0086FBF4:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[23] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FD24;
loc_0086FC10:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[24] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FD18;
loc_0086FC2C:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[25] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FD0C;
loc_0086FC48:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[26] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FD00;
loc_0086FC64:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[27] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FCF4;
loc_0086FC80:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[28] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FCE8;
loc_0086FC9C:
    ctx->gpr[30] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read64(ctx->gpr[30] + 0x90);
    { int64_t a = (int64_t)ctx->gpr[0]; int64_t b = (int64_t)ctx->gpr[29]; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0086FCDC;
loc_0086FCB8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[4] = ctx->gpr[29] | ctx->gpr[29];
    ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
    ctx->lr = 0x0086FCC8; june_evt_func_0086FB48(ctx); DRAIN_TRAMPOLINE(ctx);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FB90;
    ctx->gpr[3] = vm_read32(ctx->gpr[31] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FCB8;
loc_0086FCDC:
    ctx->gpr[3] = vm_read32(ctx->gpr[30] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FC9C;
loc_0086FCE8:
    ctx->gpr[3] = vm_read32(ctx->gpr[28] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FC80;
loc_0086FCF4:
    ctx->gpr[3] = vm_read32(ctx->gpr[27] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FC64;
loc_0086FD00:
    ctx->gpr[3] = vm_read32(ctx->gpr[26] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FC48;
loc_0086FD0C:
    ctx->gpr[3] = vm_read32(ctx->gpr[25] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FC2C;
loc_0086FD18:
    ctx->gpr[3] = vm_read32(ctx->gpr[24] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FC10;
loc_0086FD24:
    ctx->gpr[3] = vm_read32(ctx->gpr[23] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FBF4;
loc_0086FD30:
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FBD8;
loc_0086FD3C:
    ctx->gpr[3] = (int64_t)(int32_t)(0);
    goto loc_0086FB90;
}

void june_evt_func_0086FE44(ppu_context* ctx) {
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
    ctx->gpr[12] = ctx->cr;
    vm_write64(ctx->gpr[1] + -0xF0, ctx->gpr[1]); ctx->gpr[1] += -0xF0;
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[20]);
    ctx->gpr[20] = (int64_t)(int8_t)ctx->gpr[4];
    vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[21]);
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[17]);
    vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[18]);
    vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[19]);
    vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[22]);
    vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[23]);
    vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[24]);
    vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[25]);
    vm_write64(ctx->gpr[1] + 0xC0, ctx->gpr[26]);
    vm_write64(ctx->gpr[1] + 0xC8, ctx->gpr[27]);
    vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[28]);
    vm_write64(ctx->gpr[1] + 0xD8, ctx->gpr[29]);
    vm_write64(ctx->gpr[1] + 0xE0, ctx->gpr[30]);
    vm_write64(ctx->gpr[1] + 0xE8, ctx->gpr[31]);
    ctx->gpr[21] = ctx->gpr[4] | ctx->gpr[4];
    vm_write32(ctx->gpr[1] + 0xF8, ctx->gpr[12]);
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x100, ctx->gpr[0]);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008701F8;
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[3] + 0x18, ctx->gpr[0]);
loc_0086FEB0:
    ctx->gpr[3] = vm_read32(ctx->gpr[3] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087009C;
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[17] = ctx->cr & 0x0000000Fu;
    ctx->gpr[17] = ppc_rlw64(ctx->gpr[17], 28, 0, 3, 0, 0);
    { uint32_t field = (ctx->cr >> 0) & 0xF; ctx->cr = (ctx->cr & ~(0xFu << 20)) | (field << 20); }
loc_0086FECC:
    if ((((ctx->cr >> 21) & 1) == 1)) goto loc_008701D8;
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[22] + 0x18, ctx->gpr[0]);
loc_0086FEE0:
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870090;
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[18] = ctx->cr & 0x0000000Fu;
    ctx->gpr[18] = ppc_rlw64(ctx->gpr[18], 28, 0, 3, 0, 0);
loc_0086FEF8:
    if ((((ctx->cr >> 21) & 1) == 1)) goto loc_008701B8;
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[23] + 0x18, ctx->gpr[0]);
loc_0086FF0C:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870084;
loc_0086FF18:
    ctx->gpr[17] = ppc_rlw64(ctx->gpr[17], 4, 0, 31, 0, 0);
    ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[17] & 0x0000000Fu);
    ctx->gpr[17] = ppc_rlw64(ctx->gpr[17], 28, 0, 31, 0, 0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870198;
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[0]);
loc_0086FF38:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870078;
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    ctx->gpr[19] = ctx->cr & 0x0000000Fu;
    ctx->gpr[19] = ppc_rlw64(ctx->gpr[19], 28, 0, 3, 0, 0);
loc_0086FF50:
    ctx->gpr[17] = ppc_rlw64(ctx->gpr[17], 4, 0, 31, 0, 0);
    ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[17] & 0x0000000Fu);
    ctx->gpr[17] = ppc_rlw64(ctx->gpr[17], 28, 0, 31, 0, 0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870178;
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[25] + 0x18, ctx->gpr[0]);
loc_0086FF70:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087006C;
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 16)) | (cr_val << 16); }
loc_0086FF80:
    ctx->gpr[18] = ppc_rlw64(ctx->gpr[18], 4, 0, 31, 0, 0);
    ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[18] & 0x0000000Fu);
    ctx->gpr[18] = ppc_rlw64(ctx->gpr[18], 28, 0, 31, 0, 0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870158;
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[0]);
loc_0086FFA0:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870060;
    { int64_t a = (int32_t)ctx->gpr[20]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
loc_0086FFB0:
    ctx->gpr[19] = ppc_rlw64(ctx->gpr[19], 4, 0, 31, 0, 0);
    ctx->cr = (ctx->cr & 0xFFFFFFF0u) | ((uint32_t)ctx->gpr[19] & 0x0000000Fu);
    ctx->gpr[19] = ppc_rlw64(ctx->gpr[19], 28, 0, 31, 0, 0);
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870138;
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[27] + 0x18, ctx->gpr[0]);
loc_0086FFD0:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870054;
    ctx->gpr[30] = (int64_t)(int8_t)ctx->gpr[21];
loc_0086FFE0:
    if ((((ctx->cr >> 17) & 1) == 1)) goto loc_00870118;
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[0]);
loc_0086FFF4:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00870048;
loc_00870000:
    if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008700F8;
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x100;
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
loc_00870014:
    ctx->gpr[29] = vm_read32(ctx->gpr[31] + 0x80);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0087003C;
loc_00870020:
    ctx->gpr[29] = ppc_rldicl(ctx->gpr[29], 0, 32);
    ctx->gpr[4] = ctx->gpr[30] | ctx->gpr[30];
    ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
    ctx->lr = 0x00870030; june_evt_func_0086FE44(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[29] = vm_read32(ctx->gpr[29] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00870020;
loc_0087003C:
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00870000;
loc_00870048:
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FFE0;
loc_00870054:
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FFB0;
loc_00870060:
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FF80;
loc_0087006C:
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FF50;
loc_00870078:
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FF18;
loc_00870084:
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FEF8;
loc_00870090:
    ctx->gpr[3] = vm_read32(ctx->gpr[22] + 0x84);
    { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
    if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0086FECC;
loc_0087009C:
    ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0xF8);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x100);
    ctx->gpr[17] = _cs_17;
    ctx->cr = (ctx->cr & 0xFF0FFFFFu) | ((uint32_t)ctx->gpr[12] & 0x00F00000u);
    ctx->gpr[18] = _cs_18;
    ctx->lr = ctx->gpr[0];
    ctx->gpr[19] = _cs_19;
    ctx->cr = (ctx->cr & 0xFFF0FFFFu) | ((uint32_t)ctx->gpr[12] & 0x000F0000u);
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
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xF0);
    return;
loc_008700F8:
    ctx->gpr[31] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[9]);
    goto loc_00870014;
loc_00870118:
    ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[28] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[28] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[28] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[28] + 0x18, ctx->gpr[9]);
    goto loc_0086FFF4;
loc_00870138:
    ctx->gpr[27] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[27] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[27] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[27] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[27] + 0x18, ctx->gpr[9]);
    goto loc_0086FFD0;
loc_00870158:
    ctx->gpr[26] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[26] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[26] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[26] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[26] + 0x18, ctx->gpr[9]);
    goto loc_0086FFA0;
loc_00870178:
    ctx->gpr[25] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[25] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[25] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[25] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[25] + 0x18, ctx->gpr[9]);
    goto loc_0086FF70;
loc_00870198:
    ctx->gpr[24] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[24] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[24] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[24] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[24] + 0x18, ctx->gpr[9]);
    goto loc_0086FF38;
loc_008701B8:
    ctx->gpr[23] = ppc_rldicl(ctx->gpr[0], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[23] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[23] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[23] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[23] + 0x18, ctx->gpr[9]);
    goto loc_0086FF0C;
loc_008701D8:
    ctx->gpr[22] = ppc_rldicl(ctx->gpr[3], 0, 32);
    ctx->gpr[0] = vm_read32(ctx->gpr[22] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[22] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[22] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[22] + 0x18, ctx->gpr[9]);
    goto loc_0086FEE0;
loc_008701F8:
    ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x14);
    ctx->gpr[9] = vm_read32(ctx->gpr[3] + 0x18);
    ctx->gpr[0] = ctx->gpr[0] | 0x4;
    ctx->gpr[9] = ctx->gpr[9] | 0x200;
    vm_write32(ctx->gpr[3] + 0x14, ctx->gpr[0]);
    vm_write32(ctx->gpr[3] + 0x18, ctx->gpr[9]);
    goto loc_0086FEB0;
}
