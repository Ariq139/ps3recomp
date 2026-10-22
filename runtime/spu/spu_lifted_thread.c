/* spu_lifted_thread.c -- see spu_lifted_thread.h for what this is for. */

#include "spu_lifted_thread.h"
#include "spu_coherency.h"
#include "spu_context.h"
#include "spu_workload.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern uint8_t* vm_base;
extern uint32_t ps3_spu_image_source_ea(uint32_t img_ea);

/* The lifted-code registry, the dispatcher and the per-context MFC engine
 * table, all in spu_channels.c. */
int  spu_have_function(uint32_t addr);
int  spu_image_of_function(uint32_t addr);
int  spu_have_function_in_image(uint32_t addr, int image_id);
void spu_indirect_branch(struct spu_context* ctx);
int  spu_run_with_halt(void (*entry)(struct spu_context*), struct spu_context* ctx);

static int spu_guest_image_id(uint32_t img_ea, int* image_id_out,
                              uint32_t* source_out, size_t* size_out,
                              uint64_t* fingerprint_out)
{
    if (image_id_out) *image_id_out = -1;
    if (source_out) *source_out = 0;
    if (size_out) *size_out = 0;
    if (fingerprint_out) *fingerprint_out = 0;
    if (!img_ea || !vm_base) return 0;

    const uint32_t src = ps3_spu_image_source_ea(img_ea);
    if (!src) return 0;
    const size_t size = spu_elf_image_size(vm_base + src, 1u << 20);
    if (!size) return 0;

    const uint64_t fingerprint = spu_workload_fingerprint(vm_base + src, size);
    int image_id = -1;
    if (!spu_workload_find_img(fingerprint, &image_id)) return 0;
    if (image_id_out) *image_id_out = image_id;
    if (source_out) *source_out = src;
    if (size_out) *size_out = size;
    if (fingerprint_out) *fingerprint_out = fingerprint;
    return 1;
}

/* Guest memory is big-endian and this file reads descriptors out of it. lv2 has
 * its own copy of this helper; duplicating four lines is cheaper than exporting
 * one from a translation unit full of syscall handlers. */
static uint32_t rd_be32(uint32_t ea)
{
    if (!vm_base) return 0;
    const uint8_t* p = vm_base + ea;
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

/* Lay a sys_spu_image's segments into a local store.
 *
 *   sys_spu_image   { u32 type; u32 entry_point; sys_spu_segment* segs; s32 nsegs; }
 *   sys_spu_segment { s32 type; u32 ls_start; s32 size; u64 src; }   (0x18 bytes)
 *
 * COPY (type 1) reads from the segment's source EA; FILL (type 2) writes the
 * value in the same field, which the SDK counts as a segment of its own for an
 * ELF's bss tail. INFO (type 4) is metadata and is not loaded.
 *
 * _sys_spu_image_import writes the source EA at both +0x10 and +0x14 (the
 * comment there explains why: readers disagree about whether the field is a u32
 * or the low half of a big-endian u64). Reading +0x10 first and falling back
 * keeps this working for images built by either convention. */
static void spu_deploy_image(spu_context* ctx, uint32_t img_ea)
{
    if (!img_ea || !vm_base) return;
    uint32_t segs_ea = rd_be32(img_ea + 0x08);
    int32_t  nsegs   = (int32_t)rd_be32(img_ea + 0x0C);
    if (!segs_ea) return;

    for (int32_t i = 0; i < nsegs; i++) {
        uint32_t s    = segs_ea + (uint32_t)i * 0x18u;
        uint32_t type = rd_be32(s + 0x00);
        uint32_t ls   = rd_be32(s + 0x04) & SPU_LS_MASK;
        uint32_t size = rd_be32(s + 0x08);
        uint32_t src  = rd_be32(s + 0x10);
        if (!src) src = rd_be32(s + 0x14);
        if (size > SPU_LS_SIZE - ls) size = SPU_LS_SIZE - ls;
        if (type == 1) {
            if (src) memcpy(&ctx->ls[ls], vm_base + src, size);
        } else if (type == 2) {
            memset(&ctx->ls[ls], (int)(src & 0xFFu), size);
        }
    }
}

int spu_lifted_thread_available(uint32_t entry)
{
    return spu_have_function(entry & SPU_LS_MASK);
}

int spu_lifted_thread_available_for_image(uint32_t entry, uint32_t img_ea)
{
    int image_id = -1;
    uint32_t src = 0;
    size_t size = 0;
    uint64_t fingerprint = 0;
    const int registered = spu_guest_image_id(img_ea, &image_id, &src, &size,
                                               &fingerprint);
    const int exact = registered &&
        spu_have_function_in_image(entry & SPU_LS_MASK, image_id);
    if (getenv("SPU_IMAGE_MATCH_TRACE")) {
        fprintf(stderr,
                "[SPU] image-match img=0x%08X src=0x%08X size=%zu "
                "fp=0x%016llX entry=0x%05X image_id=%d lifted=%d\n",
                img_ea, src, size, (unsigned long long)fingerprint,
                entry & SPU_LS_MASK, registered ? image_id : -1, exact);
    }
    return exact;
}

void spu_lifted_thread_setup(spu_context* ctx, const spu_lifted_thread_desc* d)
{
    if (!ctx || !d) return;

    spu_context_init(ctx, d->tid);
    ctx->spu_id       = d->tid;
    ctx->spu_group_id = d->group_id;
    ctx->pc           = d->entry & SPU_LS_MASK;

    /* Start the thread in the image its entry point belongs to, rather than in
     * image 0. Zero is the registry's wildcard, so a context that keeps it will
     * happily resolve a later branch into whatever other image registered that
     * LS address -- SPURS loads its kernel, policy modules and job binaries at
     * overlapping addresses, so that is not a rare collision but the normal
     * case. Asking the registry which image owns the entry point gets the same
     * answer the title's own registration order implies, without the lv2 layer
     * having to know any image ids. */
    { int img = d->image_id > 0 ? d->image_id : -1;
      if (img < 0 && d->img_ea &&
          spu_guest_image_id(d->img_ea, &img, NULL, NULL, NULL) &&
          spu_have_function_in_image(ctx->pc, img)) {
          ctx->image_id = img;
      } else if (img >= 0) {
          ctx->image_id = img;
      } else {
          img = spu_image_of_function(ctx->pc);
          ctx->image_id = (img >= 0) ? img : 0;
      }
    }

    if (d->img_ea) spu_deploy_image(ctx, d->img_ea);

    /* RPCS3_REFERENCE_REVIEWED: the guest ABI lane placement below follows the
     * cited sys_spu contract and is expressed in this runtime's own types.
     * See docs/RPCS3_LICENSE_AUDIT.md; this is not license clearance.
     * sys_spu_thread_argument is four u64s and lv2 puts each in the preferred
     * doubleword of r3..r6 (RPCS3 sys_spu.cpp: gpr[3+i] = from64(0, arg[i])).
     * Lane 0 is the preferred word here, so the u64 occupies lanes 0 and 1 and
     * the rest of the register is zero, which memset already did. */
    for (int a = 0; a < 4; a++) {
        ctx->gpr[3 + a]._u32[0] = (uint32_t)(d->args[a] >> 32);
        ctx->gpr[3 + a]._u32[1] = (uint32_t)d->args[a];
    }

    /* r1 is deliberately left at 0: an SPU image's own crt sets its stack up,
     * and lv2 does not. The job helpers in spu_lifted_job.h do seed it, because
     * they enter a lifted function directly rather than starting a thread. */
}

void spu_lifted_thread_run(spu_context* ctx, spu_lifted_thread_result* out)
{
    spu_lifted_thread_result r;
    memset(&r, 0, sizeof(r));
    if (!ctx) { if (out) *out = r; return; }

    fprintf(stderr, "[SPU] tid=0x%X RUNNING lifted image entry=0x%05X (image %d)\n",
            ctx->spu_id, ctx->pc, ctx->image_id);
    fflush(stderr);

    ctx->status = SPU_STATUS_RUNNING;
    /* The dispatcher resolves ctx->pc in ctx->image_id and runs it; the halt
     * landing pad catches an SPU that stops from inside a nested call, and
     * drains the tail-transfer chain when it returns normally. */
    spu_run_with_halt(spu_indirect_branch, ctx);

    /* Optional postmortem: preserve the actual local store and registers that
     * led a guest SPU to stop, including dynamically loaded policy code. */
    {
        const char* dir = getenv("SPU_STOP_DUMP");
        if (dir && *dir) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/spu-stop-%X.ls", dir, ctx->spu_id);
            FILE* dump = fopen(path, "wb");
            if (dump) { fwrite(ctx->ls, 1, SPU_LS_SIZE, dump); fclose(dump); }
            snprintf(path, sizeof(path), "%s/spu-stop-%X.txt", dir, ctx->spu_id);
            dump = fopen(path, "w");
            if (dump) {
                fprintf(dump, "image=%d pc=%05X status=%X stop=%X\n",
                        ctx->image_id, ctx->pc, ctx->status, ctx->stop_code);
                for (unsigned i = 0; i < 128; ++i) {
                    unsigned char bytes[16];
                    memcpy(bytes, &ctx->gpr[i], sizeof(bytes));
                    fprintf(dump, "r%u=", i);
                    for (unsigned j = 0; j < 16; ++j) fprintf(dump, "%02X", bytes[j]);
                    fputc('\n', dump);
                }
                fclose(dump);
            }
        }
    }

    /* What the SPU meant by stopping. The SPU-side sys_spu_thread_exit ABI
     * writes the status to SPU_WrOutMbox and then executes stop 0x102; lv2 pops
     * that mailbox value as the thread's exit status. The 14-bit stop code is
     * the selector, not the status (CBEA p97, SPU_Status.StopCode): 0x102
     * THREAD_EXIT carries this thread's status, 0x101 GROUP_EXIT the group's. */
    extern int spu_group_was_stopped(spu_context* ctx);
    if (spu_group_was_stopped(ctx)) {
        /* A peer requested GROUP_EXIT. This cooperative halt is not a guest
         * fault, and must let the PPU join finish before the next group run. */
        r.exit_status = 0;
    } else if (ctx->status == SPU_STATUS_STOPPED_BY_STOP &&
        ctx->stop_code == 0x102u && ctx->ch_out_mbox.count) {
        r.exit_status = (int32_t)ctx->ch_out_mbox.value;
        ctx->ch_out_mbox.count = 0;
    } else if (ctx->status == SPU_STATUS_STOPPED_BY_STOP &&
               ctx->stop_code == 0x101u && ctx->ch_out_mbox.count) {
        r.group_exit   = 1;
        r.group_status = (int32_t)ctx->ch_out_mbox.value;
        ctx->ch_out_mbox.count = 0;
    } else {
        /* Anything else -- a halt from the dispatcher's unresolved-branch
         * unwind, or a stop code outside the exit protocol -- is a fault. The
         * thread still exits with 0, as it does on the fallback path, but the
         * caller is told so it can report or raise an exception event. */
        r.faulted = 1;
    }

    fprintf(stderr, "[SPU] tid=0x%X stopped (status=0x%X pc=0x%05X code=0x%X) "
            "-> status=%d%s%s\n",
            ctx->spu_id, ctx->status, ctx->pc, ctx->stop_code, r.exit_status,
            r.group_exit ? " group-exit" : "", r.faulted ? " FAULT" : "");
    fflush(stderr);

    /* This thread run has ended; remove observer references and release its
     * private engine before restart can reset the context or destroy frees it. */
    spu_coh_unregister(ctx);
    spu_mfc_release(ctx);

    if (out) *out = r;
}
