#include "ppu_recomp.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

void june_evt_func_008F705C(ppu_context* ctx);
void june_evt_func_00884080(ppu_context* ctx);
void june_evt_func_00885420(ppu_context* ctx);
void june_evt_func_008852D4(ppu_context* ctx);
void june_evt_func_00834E18(ppu_context* ctx);
void june_evt_func_008871E0(ppu_context* ctx);
void june_evt_func_00825AF0(ppu_context* ctx);

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

void june_evt_func_008F705C(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x4258);
        ctx->gpr[4] = (int64_t)(int32_t)(0x12A0);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x008F707C; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[9] = (int64_t)(int32_t)(0x100);
        ctx->gpr[0] = (int64_t)(int32_t)(0x80);
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x94);
        vm_write32(ctx->gpr[29] + 0x1258, ctx->gpr[0]);
        vm_write32(ctx->gpr[29] + 0x1268, ctx->gpr[9]);
        vm_write32(ctx->gpr[29] + 0x1260, ctx->gpr[9]);
        ctx->lr = 0x008F709C; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x74);
        ctx->lr = 0x008F70A8; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x7C);
        ctx->lr = 0x008F70B4; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x44);
        ctx->lr = 0x008F70C0; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x9A0);
        ctx->lr = 0x008F70CC; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0xB10);
        ctx->lr = 0x008F70D8; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x1244);
        ctx->lr = 0x008F70E4; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x124C);
        ctx->lr = 0x008F70F0; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x1274);
        ctx->lr = 0x008F70FC; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x128C);
        ctx->lr = 0x008F7108; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x1148);
        ctx->lr = 0x008F7114; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x6C);
        ctx->lr = 0x008F7120; june_evt_func_00834E18(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->lr = 0x008F7128; june_evt_func_00884080(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008F7134; june_evt_func_00825AF0(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->lr = 0x008F713C; june_evt_func_008871E0(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
}

void june_evt_func_00884080(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_31 = ctx->gpr[31];
        vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5EA0);
        ctx->gpr[4] = (int64_t)(int32_t)(0x11F);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x1250);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008840B4;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5E7C);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5E98);
        ctx->lr = 0x008840B0; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
loc_008840B4:
        ctx->gpr[8] = (int64_t)(int32_t)(2);
        ctx->gpr[4] = (int64_t)(int32_t)(0x80);
        ctx->gpr[5] = (int64_t)(int32_t)(0x10);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[3] = (int64_t)(int32_t)(0x200);
        ctx->lr = 0x008840D4; june_evt_func_00885420(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[8] = (int64_t)(int32_t)(3);
        ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x1258);
        vm_write32(ctx->gpr[31] + 0x1250, ctx->gpr[3]);
        ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = (int64_t)(int32_t)(0x10);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = (int64_t)(int32_t)(0xF0);
        ctx->lr = 0x008840FC; june_evt_func_00885420(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[8] = (int64_t)(int32_t)(2);
        ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x1260);
        vm_write32(ctx->gpr[31] + 0x1254, ctx->gpr[3]);
        ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[5] = (int64_t)(int32_t)(0x10);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = (int64_t)(int32_t)(0x70);
        ctx->lr = 0x00884124; june_evt_func_00885420(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[8] = (int64_t)(int32_t)(2);
        ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x1268);
        vm_write32(ctx->gpr[31] + 0x125C, ctx->gpr[3]);
        ctx->gpr[8] = ctx->gpr[8] | ((uint64_t)0xFF00 << 16);
        ctx->gpr[3] = (int64_t)(int32_t)(0x18);
        ctx->gpr[5] = (int64_t)(int32_t)(0x10);
        ctx->gpr[6] = (int64_t)(int32_t)(1);
        ctx->gpr[7] = (int64_t)(int32_t)(0);
        ctx->lr = 0x0088414C; june_evt_func_00885420(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
        vm_write32(ctx->gpr[31] + 0x1264, ctx->gpr[3]);
        ctx->lr = ctx->gpr[0];
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
        return;
}

void june_evt_func_00885420(ppu_context* ctx) {
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
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + -0xC0, ctx->gpr[1]); ctx->gpr[1] += -0xC0;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[24]);
        vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[26]);
        vm_write64(ctx->gpr[1] + 0x98, ctx->gpr[27]);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[28]);
        vm_write64(ctx->gpr[1] + 0xA8, ctx->gpr[29]);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[30]);
        vm_write64(ctx->gpr[1] + 0xB8, ctx->gpr[31]);
        vm_write64(ctx->gpr[1] + 0x70, ctx->gpr[22]);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[23]);
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[25]);
        vm_write64(ctx->gpr[1] + 0xD0, ctx->gpr[0]);
        ctx->gpr[26] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[30] = ctx->gpr[7] | ctx->gpr[7];
        ctx->gpr[24] = ctx->gpr[4] | ctx->gpr[4];
        ctx->gpr[28] = ctx->gpr[5] | ctx->gpr[5];
        ctx->gpr[29] = ctx->gpr[6] | ctx->gpr[6];
        ctx->gpr[27] = ctx->gpr[8] | ctx->gpr[8];
        ctx->gpr[31] = ctx->gpr[7] | ctx->gpr[7];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008856BC;
loc_00885478:
        { int64_t a = (int32_t)ctx->gpr[24]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008856A4;
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[24], 0, 29, 31, 0, 0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00885644;
loc_0088548C:
        { int64_t a = (int32_t)ctx->gpr[28]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00885498;
        ctx->gpr[28] = (int64_t)(int32_t)(0x10);
loc_00885498:
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008856DC;
        ctx->gpr[0] = (int64_t)(int32_t)(3);
        ctx->gpr[22] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[31], 0, 32);
        vm_write32(ctx->gpr[30] + 0x0, ctx->gpr[0]);
loc_008854B0:
        ctx->gpr[9] = ctx->gpr[28] + ctx->gpr[26];
        ctx->gpr[0] = (uint64_t)0 - ctx->gpr[28];
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(-1);
        ctx->gpr[11] = ctx->gpr[24] + (int64_t)(7);
        ctx->gpr[3] = ctx->gpr[22] + (int64_t)(0x1C);
        ctx->gpr[23] = ctx->gpr[9] & ctx->gpr[0];
        ctx->gpr[30] = ppc_rlw64(ctx->gpr[11], 29, 3, 31, 0, 0);
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x8, ctx->gpr[23]);
        vm_write32(ctx->gpr[31] + 0xC, ctx->gpr[24]);
        vm_write32(ctx->gpr[31] + 0x10, ctx->gpr[28]);
        vm_write32(ctx->gpr[31] + 0x4, ctx->gpr[30]);
        vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[0]);
        vm_write32(ctx->gpr[31] + 0x20, ctx->gpr[27]);
        ctx->lr = 0x008854FC; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008855C0;
        ctx->gpr[0] = ctx->gpr[28] + ctx->gpr[30];
        ctx->gpr[26] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[27] = vm_read32(ctx->gpr[2] + -0x5E64);
        ctx->gpr[25] = (int64_t)(int32_t)(0);
        ctx->gpr[9] = (uint64_t)((int64_t)(int32_t)ctx->gpr[24] * (int64_t)(int32_t)ctx->gpr[23]);
        ctx->gpr[9] = ctx->gpr[9] + (int64_t)(7);
        ctx->gpr[0] = ctx->gpr[0] + ctx->gpr[9];
        ctx->gpr[28] = ppc_rldicl(ctx->gpr[0], 0, 32);
        goto loc_0088555C;
loc_0088552C:
        vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[31] + 0x14);
        ctx->gpr[29] = ctx->gpr[29] + (int64_t)(-1);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[3]);
        ctx->gpr[3] = ctx->gpr[11] + (int64_t)(8);
        vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[11]);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x00885550; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008855C0;
loc_0088555C:
        ctx->gpr[4] = ctx->gpr[26] | ctx->gpr[26];
        ctx->gpr[5] = ctx->gpr[27] | ctx->gpr[27];
        ctx->gpr[6] = (int64_t)(int32_t)(0x73);
        ctx->gpr[3] = ctx->gpr[28] | ctx->gpr[28];
        ctx->lr = 0x00885570; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[11] = ctx->gpr[3] | ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_0088565C;
        vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[25]);
        vm_write32(ctx->gpr[3] + 0x0, ctx->gpr[25]);
        ctx->gpr[0] = vm_read32(ctx->gpr[31] + 0x14);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088552C;
        vm_write32(ctx->gpr[31] + 0x18, ctx->gpr[3]);
        vm_write32(ctx->gpr[3] + 0x4, ctx->gpr[0]);
        ctx->gpr[3] = ctx->gpr[11] + (int64_t)(8);
        ctx->gpr[29] = ctx->gpr[29] + (int64_t)(-1);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[30], 0, 32);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        vm_write32(ctx->gpr[31] + 0x14, ctx->gpr[11]);
        ctx->lr = 0x008855B4; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        { int64_t a = (int32_t)ctx->gpr[29]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088555C;
loc_008855C0:
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x5E50);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->gpr[29] = ctx->gpr[29] + (int64_t)(-4684);
        ctx->lr = 0x008855D0; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        vm_write32(ctx->gpr[31] + 0x24, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[29] + 0x1248);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008856D4;
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        ctx->gpr[9] = vm_read32(ctx->gpr[29] + 0x1248);
        vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[22]);
loc_008855F4:
        ctx->gpr[9] = vm_read32(ctx->gpr[2] + -0x5E40);
        ctx->gpr[3] = ctx->gpr[9] + (int64_t)(0x124C);
        vm_write32(ctx->gpr[9] + 0x1248, ctx->gpr[22]);
        ctx->lr = 0x00885604; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
loc_00885608:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
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
loc_00885644:
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5E64);
        ctx->gpr[4] = (int64_t)(int32_t)(0x40);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5E44);
        ctx->lr = 0x00885654; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        goto loc_0088548C;
loc_0088565C:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->gpr[31] = (int64_t)(int32_t)(0);
        ctx->lr = 0x00885668; june_evt_func_008852D4(ctx); DRAIN_TRAMPOLINE(ctx);
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xD0);
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
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
loc_008856A4:
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5E64);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3F);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5E48);
        ctx->lr = 0x008856B4; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        goto loc_0088548C;
loc_008856BC:
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x5E64);
        ctx->gpr[4] = (int64_t)(int32_t)(0x3E);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5E4C);
        ctx->lr = 0x008856CC; june_evt_func_00824F18(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        goto loc_00885478;
loc_008856D4:
        vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[0]);
        goto loc_008855F4;
loc_008856DC:
        ctx->gpr[3] = (int64_t)(int32_t)(0x2C);
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x5E64);
        ctx->gpr[4] = ppc_rldicl(ctx->gpr[27], 0, 32);
        ctx->gpr[6] = (int64_t)(int32_t)(0x49);
        ctx->gpr[31] = (int64_t)(int32_t)(0);
        ctx->lr = 0x008856F4; june_evt_func_00884414(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00885608;
        ctx->gpr[31] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->gpr[0] = (int64_t)(int32_t)(2);
        ctx->gpr[22] = ctx->gpr[3] | ctx->gpr[3];
        vm_write32(ctx->gpr[31] + 0x0, ctx->gpr[0]);
        goto loc_008854B0;
}

void june_evt_func_008852D4(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        uint64_t _cs_30 = ctx->gpr[30];
        uint64_t _cs_31 = ctx->gpr[31];
        ctx->gpr[12] = ctx->cr;
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[30]);
        ctx->gpr[30] = ctx->gpr[3] | ctx->gpr[3];
        vm_write64(ctx->gpr[1] + 0x88, ctx->gpr[31]);
        ctx->gpr[31] = vm_read32(ctx->gpr[2] + -0x5E50);
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        ctx->gpr[29] = ctx->gpr[3] | ctx->gpr[3];
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        vm_write32(ctx->gpr[1] + 0x98, ctx->gpr[12]);
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->lr = 0x00885308; june_evt_func_008350F0(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[11] = vm_read32(ctx->gpr[30] + 0x28);
        { int64_t a = (int32_t)ctx->gpr[11]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00885324;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[11], 0, 32);
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x24);
        vm_write32(ctx->gpr[9] + 0x24, ctx->gpr[0]);
loc_00885324:
        ctx->gpr[9] = vm_read32(ctx->gpr[30] + 0x24);
        { int64_t a = (int32_t)ctx->gpr[9]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00885418;
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[9], 0, 32);
        vm_write32(ctx->gpr[9] + 0x28, ctx->gpr[11]);
loc_00885338:
        ctx->gpr[3] = ctx->gpr[31] | ctx->gpr[31];
        ctx->lr = 0x00885340; june_evt_func_0083513C(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x14);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_00885384;
        goto loc_008853C8;
loc_00885354:
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        vm_write32(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_008853B0;
loc_0088536C:
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
        vm_write32(ctx->gpr[9] + 0x4, ctx->gpr[0]);
        ctx->lr = 0x00885378; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = ctx->gpr[31] | ctx->gpr[31];
        if ((((ctx->cr >> 13) & 1) == 1)) goto loc_008853C8;
loc_00885384:
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[0], 0, 32);
        ctx->gpr[31] = vm_read32(ctx->gpr[3] + 0x4);
        { int64_t a = (int32_t)ctx->gpr[31]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 12)) | (cr_val << 12); }
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[31], 0, 32);
        if ((((ctx->cr >> 13) & 1) == 0)) goto loc_00885354;
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        vm_write32(ctx->gpr[30] + 0x18, ctx->gpr[0]);
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x0);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[9] = ppc_rldicl(ctx->gpr[0], 0, 32);
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_0088536C;
loc_008853B0:
        ctx->gpr[0] = vm_read32(ctx->gpr[3] + 0x4);
        vm_write32(ctx->gpr[30] + 0x14, ctx->gpr[0]);
        ctx->lr = 0x008853BC; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = ctx->gpr[31] | ctx->gpr[31];
        if ((((ctx->cr >> 13) & 1) == 0)) goto loc_00885384;
loc_008853C8:
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x1C);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        ctx->lr = 0x008853D4; june_evt_func_008350D0(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = vm_read32(ctx->gpr[30] + 0x0);
        ctx->gpr[0] = ppc_rlw64(ctx->gpr[0], 0, 31, 31, 0, 0);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[29], 0, 32);
        { int64_t a = (int32_t)ctx->gpr[0]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        if ((((ctx->cr >> 1) & 1) == 0)) goto loc_008853F4;
        ctx->lr = 0x008853F0; june_evt_func_00884214(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
loc_008853F4:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[12] = vm_read32(ctx->gpr[1] + 0x98);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[30] = _cs_30;
        ctx->cr = (ctx->cr & 0xFFFF0FFFu) | ((uint32_t)ctx->gpr[12] & 0x0000F000u);
        ctx->gpr[31] = _cs_31;
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
loc_00885418:
        vm_write32(ctx->gpr[31] + -0x4, ctx->gpr[11]);
        goto loc_00885338;
}

void june_evt_func_00834E18(ppu_context* ctx) {
    PPU_MM_SNAP
        vm_write64(ctx->gpr[1] + -0xA0, ctx->gpr[1]); ctx->gpr[1] += -0xA0;
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[9] = (int64_t)(int32_t)(0x20);
        vm_write64(ctx->gpr[1] + 0xB0, ctx->gpr[0]);
        ctx->gpr[0] = (int64_t)(int32_t)(2);
        ctx->gpr[11] = (int64_t)(int32_t)(0);
        vm_write32(ctx->gpr[1] + 0x70, ctx->gpr[0]);
        ctx->gpr[0] = (int64_t)(int32_t)(0x200);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[9]);
        vm_write32(ctx->gpr[1] + 0x78, ctx->gpr[0]);
        ctx->gpr[9] = (int64_t)(int32_t)(0x2000);
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        vm_write32(ctx->gpr[1] + 0x7C, ctx->gpr[9]);
        vm_write32(ctx->gpr[1] + 0x88, ctx->gpr[11]);
        vm_write64(ctx->gpr[1] + 0x80, ctx->gpr[11]);
        vm_write8(ctx->gpr[1] + 0x90, ctx->gpr[0]);
        ctx->gpr[11] = vm_read32(ctx->gpr[2] + -0x7084);
        ctx->gpr[9] = ctx->gpr[1] + (int64_t)(0x90);
        ctx->gpr[0] = vm_read64(ctx->gpr[11] + 0x0);
        vm_write64(ctx->gpr[9] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = (int64_t)(int32_t)(0);
        ctx->gpr[10] = ctx->gpr[9] + (int64_t)(7);
        vm_write8(ctx->gpr[10] + 0x0, ctx->gpr[0]);
        ctx->gpr[0] = (int64_t)(int32_t)(0x10);
        ctx->gpr[3] = ppc_rldicl(ctx->gpr[3], 0, 32);
        vm_write32(ctx->gpr[1] + 0x74, ctx->gpr[0]);
        ctx->gpr[4] = ctx->gpr[1] + (int64_t)(0x70);
        ctx->gpr[11] = (int64_t)(int32_t)(0x64);
        lv2_syscall(ctx);
        { int64_t a = (int32_t)ctx->gpr[3]; int64_t b = (int64_t)0; uint32_t cr_val = ((a < b) ? 8 : (a > b) ? 4 : 2) | ((ctx->xer >> 31) & 1u); ctx->cr = (ctx->cr & ~(0xFu << 0)) | (cr_val << 0); }
        ctx->gpr[4] = (int64_t)(int32_t)(0x125);
        ctx->gpr[6] = (int64_t)(int32_t)ctx->gpr[3];
        if ((((ctx->cr >> 1) & 1) == 1)) goto loc_00834EAC;
        ctx->gpr[5] = vm_read32(ctx->gpr[2] + -0x7080);
        ctx->gpr[3] = vm_read32(ctx->gpr[2] + -0x7098);
        ctx->lr = 0x00834EA8; june_evt_func_00824E08(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
loc_00834EAC:
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xB0);
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0xA0);
        ctx->lr = ctx->gpr[0];
        return;
}

void june_evt_func_008871E0(ppu_context* ctx) {
    PPU_MM_SNAP
        uint64_t _cs_29 = ctx->gpr[29];
        vm_write64(ctx->gpr[1] + -0x90, ctx->gpr[1]); ctx->gpr[1] += -0x90;
        ctx->gpr[0] = ctx->lr;
        vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[29]);
        ctx->gpr[29] = vm_read32(ctx->gpr[2] + -0x5D14);
        ctx->gpr[4] = (int64_t)(int32_t)(0x50);
        vm_write64(ctx->gpr[1] + 0xA0, ctx->gpr[0]);
        ctx->gpr[3] = ctx->gpr[29] | ctx->gpr[29];
        ctx->lr = 0x00887200; june_evt_func_00834944(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[3] = ctx->gpr[29] + (int64_t)(0x40);
        ctx->lr = 0x0088720C; june_evt_func_00835034(ctx); DRAIN_TRAMPOLINE(ctx);
         ;
        ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0xA0);
        ctx->gpr[29] = _cs_29;
        ctx->lr = ctx->gpr[0];
        ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x90);
        return;
}

void june_evt_func_00825AF0(ppu_context* ctx) {
    PPU_MM_SNAP
        return;
}
