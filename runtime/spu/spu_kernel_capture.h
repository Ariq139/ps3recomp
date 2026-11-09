/* Private drain instrumentation for one pure lifted function. No hook is
 * installed unless SPU_KERNEL_CAPTURE_PREFIX, _IMAGE and _PC are all set.
 * Included by spu_drain.c so standalone runtime fixtures need no new unit.
 * The snapshot format contains no host pointers; see tools/spu_kernel_replay.py.
 */
#ifndef PS3RECOMP_SPU_KERNEL_CAPTURE_H
#define PS3RECOMP_SPU_KERNEL_CAPTURE_H
#include "spu_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <fenv.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
#define SPU_KERNEL_CAPTURE_X86 1
#else
#define SPU_KERNEL_CAPTURE_X86 0
#endif

typedef void (*spu_kernel_fn)(spu_context*);
#define SPU_KERNEL_SNAPSHOT_WORDS 13u
#define SPU_KERNEL_SNAPSHOT_BYTES (SPU_KERNEL_SNAPSHOT_WORDS * 4u + 128u * 16u + SPU_LS_SIZE)

static int spu_kernel_put32(FILE* f, uint32_t v)
{
    const unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
        (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    return fwrite(b, 1, 4, f) == 4;
}
static int spu_kernel_get32(FILE* f, uint32_t* v)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (uint32_t)b[0] | (uint32_t)b[1] << 8 |
         (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 1;
}
/* Serialize only the state a pure LS/GPR kernel can observe or change.
 * Channels, DMA engines, interrupt frames and host pointers are intentionally
 * outside this format: a kernel touching those is not supported by replay. */
static int spu_kernel_snapshot_write(FILE* f, const spu_context* c)
{
    const uint32_t words[SPU_KERNEL_SNAPSHOT_WORDS] = {
        c->pc, c->status, c->stop_code, (uint32_t)c->image_id, c->spu_id,
        c->int_enable, c->event_mask, spu_event_load(c), c->srr0,
        c->host_depth, c->drain_ret_pc, (uint32_t)c->policy_mode,
        g_spu_trampoline_fn != NULL
    };
    for (unsigned i = 0; i < SPU_KERNEL_SNAPSHOT_WORDS; ++i)
        if (!spu_kernel_put32(f, words[i])) return 0;
    for (unsigned r = 0; r < 128; ++r)
        for (unsigned w = 0; w < 4; ++w)
            if (!spu_kernel_put32(f, c->gpr[r]._u32[w])) return 0;
    return fwrite(c->ls, 1, SPU_LS_SIZE, f) == SPU_LS_SIZE;
}
static int spu_kernel_snapshot_read(FILE* f, spu_context* c, uint32_t* has_tramp)
{
    uint32_t w[SPU_KERNEL_SNAPSHOT_WORDS];
    for (unsigned i = 0; i < SPU_KERNEL_SNAPSHOT_WORDS; ++i)
        if (!spu_kernel_get32(f, &w[i])) return 0;
    c->pc = w[0]; c->status = w[1]; c->stop_code = w[2]; c->image_id = (int)w[3];
    c->spu_id = w[4]; c->int_enable = w[5]; c->event_mask = w[6];
    SPU_EVENT_CLEAR(c, UINT32_MAX); SPU_EVENT_SET(c, w[7]); c->srr0 = w[8];
    c->host_depth = w[9]; c->drain_ret_pc = w[10]; c->policy_mode = (int)w[11];
    *has_tramp = w[12];
    for (unsigned r = 0; r < 128; ++r)
        for (unsigned lane = 0; lane < 4; ++lane)
            if (!spu_kernel_get32(f, &c->gpr[r]._u32[lane])) return 0;
    return fread(c->ls, 1, SPU_LS_SIZE, f) == SPU_LS_SIZE;
}
static unsigned spu_kernel_rounding_code(void)
{
    const int r = fegetround();
    return r == FE_DOWNWARD ? 1u : r == FE_UPWARD ? 2u : r == FE_TOWARDZERO ? 3u :
           r == FE_TONEAREST ? 0u : UINT32_MAX;
}
static int spu_kernel_header_write(FILE* f, unsigned image, unsigned pc)
{
    const uint32_t words[8] = {1u, SPU_KERNEL_SNAPSHOT_BYTES, image, pc,
        SPU_KERNEL_CAPTURE_X86,
#if SPU_KERNEL_CAPTURE_X86
        _mm_getcsr(),
#else
        0u,
#endif
        spu_kernel_rounding_code(), 0u};
    if (fwrite("SPUKRN01", 1, 8, f) != 8) return 0;
    for (unsigned i = 0; i < 8; ++i)
        if (!spu_kernel_put32(f, words[i])) return 0;
    return 1;
}

static struct {
    atomic_int configured, selected;
    int enabled, image, outlined_only;
    uint32_t pc;
    char* path;
    spu_kernel_fn original;
    atomic_int claimed;
} s_spu_kernel_capture;

static void spu_kernel_capture_wrapper(spu_context* ctx)
{
    int expected = 0;
    if (!atomic_compare_exchange_strong(&s_spu_kernel_capture.claimed, &expected, 1)) {
        s_spu_kernel_capture.original(ctx);
        return;
    }
    FILE* f = fopen(s_spu_kernel_capture.path, "wb");
    int ok = f && spu_kernel_header_write(f, (unsigned)s_spu_kernel_capture.image,
                                         s_spu_kernel_capture.pc) &&
                  spu_kernel_snapshot_write(f, ctx);
    /* Capture failure must never suppress, retry or follow a guest call. */
    s_spu_kernel_capture.original(ctx);
    if (ok) ok = spu_kernel_snapshot_write(f, ctx);
    if (f && fclose(f) != 0) ok = 0;
    fprintf(stderr, "[SPU-KERNEL-CAPTURE] %s image=%d pc=0x%05X file=%s\n",
            ok ? "saved" : "FAILED (guest execution continued)",
            s_spu_kernel_capture.image, s_spu_kernel_capture.pc, s_spu_kernel_capture.path);
}

static spu_kernel_fn spu_kernel_capture_select_scoped(int image, uint32_t pc,
                                                     spu_kernel_fn fn, int outlined)
{
    int expected = 0;
    if (atomic_compare_exchange_strong(&s_spu_kernel_capture.configured, &expected, 1)) {
        const char* prefix = getenv("SPU_KERNEL_CAPTURE_PREFIX");
        const char* img = getenv("SPU_KERNEL_CAPTURE_IMAGE");
        const char* addr = getenv("SPU_KERNEL_CAPTURE_PC");
        const char* scope = getenv("SPU_KERNEL_CAPTURE_SCOPE");
        s_spu_kernel_capture.outlined_only = scope && !strcmp(scope, "outlined");
        if (prefix && *prefix && img && *img && addr && *addr) {
            char *a, *b;
            unsigned long im = strtoul(img, &a, 0), off = strtoul(addr, &b, 0);
            if (!*a && !*b && im <= INT32_MAX && off < SPU_LS_SIZE && !(off & 3u)) {
                const size_t n = strlen(prefix);
                s_spu_kernel_capture.path = (char*)malloc(n + 5);
                if (s_spu_kernel_capture.path) {
                    memcpy(s_spu_kernel_capture.path, prefix, n);
                    memcpy(s_spu_kernel_capture.path + n, ".bin", 5);
                    s_spu_kernel_capture.image = (int)im;
                    s_spu_kernel_capture.pc = (uint32_t)off;
                    s_spu_kernel_capture.enabled = 1;
                }
            }
        }
        atomic_store(&s_spu_kernel_capture.configured, 2);
    } else {
        while (atomic_load(&s_spu_kernel_capture.configured) != 2) {}
    }
    if (s_spu_kernel_capture.outlined_only && !outlined) return fn;
    if (s_spu_kernel_capture.enabled && image == s_spu_kernel_capture.image &&
        pc == s_spu_kernel_capture.pc) {
        expected = 0;
        if (atomic_compare_exchange_strong(&s_spu_kernel_capture.selected, &expected, 1)) {
            s_spu_kernel_capture.original = fn;
            atomic_store(&s_spu_kernel_capture.selected, 2);
            fprintf(stderr, "[SPU-KERNEL-CAPTURE] armed image=%d pc=0x%05X\n", image, pc);
        } else {
            while (atomic_load(&s_spu_kernel_capture.selected) != 2) {}
        }
        if (s_spu_kernel_capture.original == fn)
            return spu_kernel_capture_wrapper;
    }
    return fn;
}
static spu_kernel_fn spu_kernel_capture_select(int image, uint32_t pc, spu_kernel_fn fn)
{
    return spu_kernel_capture_select_scoped(image, pc, fn, 0);
}
#endif
