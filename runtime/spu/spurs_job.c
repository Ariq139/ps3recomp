/* spurs_job.c — stage and enter a SPURS jobchain job the way Sony's jm2 does.
 *
 * A job binary is not an SPU ELF and not a policy module: it is a raw image
 * built by the SDK's job_elf-to-bin, linked against Sony's job_start_w_crt.o.
 * That CRT is what every job entry actually runs first — it saves r3/r4, runs
 * the ctors, and calls
 *
 *     void cellSpursJobMain2(CellSpursJobContext2 *jobContext,   // r3
 *                            CellSpursJob256      *job);         // r4
 *
 * (target/spu/include/cell/spurs/job_chain.h). Both are LOCAL STORE pointers,
 * so the manager must have laid the whole working set out in LS before entry.
 * Entering a job without that context is what left LBP's job parked with zero
 * DMA traffic: it reads its ioBuffer pointer out of a zeroed context and stalls.
 *
 * The layout is pinned by _cellSpursCheckJob() in
 * target/common/include/cell/spurs/job_descriptor.h, which sums the exact LS a
 * job consumes. For a jobchain it runs with align = 1024, isScratchStackUnified
 * = 1, maxUsableMemorySize = CELL_SPURS_MAX_SIZE_JOB_MEMORY (234 KB):
 *
 *     align1024(sizeBinary<<4)                                   binary + bss
 *   + align1024(sizeInOrInOut)                                   ioBuffer
 *   + align1024(sizeOut)                                         oBuffer
 *   + align1024((sizeScratch<<4) + (sizeStack ? sizeStack<<4 : 8192))
 *                                                                scratch + stack
 *   + per cache entry: align1024(size)                           cacheBuffer[0..3]
 *
 * We reproduce that arithmetic rather than invent a layout, so a job that
 * range-checks its own buffers sees what it would on hardware.
 *
 * Load address: the job binary is position-independent (job_start_w_crt derives
 * its load delta with a brsl-to-next / subtract-link-time-address idiom), so jm2
 * may place it anywhere. We place it at LS 0 — the lifted C is keyed to
 * link-time addresses, which makes the delta zero and the lifted branch targets
 * exact.
 */

#include "spu_coherency.h"
#include "spu_workload.h"
#include "spu_job_profile.h"
#include "spu_context.h"
#include "spu_job_context_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include "ps3emu/log_first.h"
#include "ps3emu/ps3_env_once.h"   /* ps3_env_once: each switch looked up once */

extern uint8_t* vm_base;
extern int spu_run_with_halt(void (*)(spu_context*), spu_context*);

/* ---- guest reads (big-endian) ------------------------------------------ */
static uint32_t g32(uint32_t ea)
{
    const uint8_t* p = vm_base + ea;
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}
static uint16_t g16(uint32_t ea)
{
    const uint8_t* p = vm_base + ea;
    return (uint16_t)(((uint32_t)p[0] << 8) | (uint32_t)p[1]);
}
static uint64_t g64(uint32_t ea)
{
    return ((uint64_t)g32(ea) << 32) | (uint64_t)g32(ea + 4);
}

/* ---- local-store writes (big-endian) ----------------------------------- */
static void ls32(uint8_t* ls, uint32_t o, uint32_t v)
{
    ls[o] = (uint8_t)(v >> 24); ls[o+1] = (uint8_t)(v >> 16);
    ls[o+2] = (uint8_t)(v >> 8); ls[o+3] = (uint8_t)v;
}
static void ls16(uint8_t* ls, uint32_t o, uint16_t v)
{
    ls[o] = (uint8_t)(v >> 8); ls[o+1] = (uint8_t)v;
}

/* CellSpursJobHeader, 48 B — job_descriptor.h */
enum {
    JH_EA_BINARY           = 0x00,   /* be u64; low bit = CACHE_INHIBIT flag */
    JH_SIZE_BINARY         = 0x08,   /* be u16; bytes >> 4                   */
    JH_SIZE_DMA_LIST       = 0x0A,   /* be u16; input list size in BYTES     */
    JH_EA_HIGH_INPUT       = 0x0C,
    JH_USE_INOUT_BUFFER    = 0x10,
    JH_SIZE_IN_OR_INOUT    = 0x14,
    JH_SIZE_OUT            = 0x18,
    JH_SIZE_STACK          = 0x1C,   /* be u16; quadwords */
    JH_SIZE_SCRATCH        = 0x1E,   /* be u16; quadwords */
    JH_EA_HIGH_CACHE       = 0x20,
    JH_SIZE_CACHE_DMA_LIST = 0x24,   /* be u32; bytes */
    JH_JOB_TYPE            = 0x2C,
    JH_SIZE                = 0x30,
};

/* CellSpursJobContext2, 48 B — job_context_types.h. SPU pointers are 32-bit LS
 * offsets and the struct lives in local store, so every field is big-endian. */
enum {
    JC_IO_BUFFER          = 0x00,
    JC_CACHE_BUFFER       = 0x04,   /* void *cacheBuffer[4] */
    JC_SIZE_JOB_DESC      = 0x14,   /* sizeJobDescriptor:4 + manager table:28 */
    JC_NUM_IO_BUFFER      = 0x18,
    JC_NUM_CACHE_BUFFER   = 0x1A,
    JC_O_BUFFER           = 0x1C,
    JC_S_BUFFER           = 0x20,
    JC_DMA_TAG            = 0x24,
    JC_EA_JOB_DESCRIPTOR  = 0x28,   /* be u64 */
    JC_SIZE               = 0x30,
};

#define ALIGN1024(x)  (((x) + 1023u) & ~1023u)
#define JOB_STACK_DEFAULT 8192u    /* _cellSpursCheckJob's sizeStack==0 default */

/* Local store of the most recent job, kept alive past the run.
 *
 * A title can poll the SPU that runs a job chain with sys_spu_thread_read_LS,
 * identifying it by the CHAIN HANDLE rather than an lv2 thread id -- Tokyo
 * Jungle reads its bus counts that way, 45k polls of offset 0 in a 45 s run.
 * The job context is a stack local, so by the time the PPU looks there is
 * nothing to read and every poll fails. Retain the store and let the syscall
 * resolve a chain handle to it. */
static uint8_t  s_job_ls[SPU_LS_SIZE];
static int      s_job_ls_valid;
uint32_t        g_spurs_job_ls_handle;   /* set by the chain walker */

const uint8_t* spurs_job_ls_for_handle(uint32_t handle)
{
    if (!s_job_ls_valid || !handle || handle != g_spurs_job_ls_handle) return 0;
    return s_job_ls;
}

/* Last outbound mailbox posted by a finished job (see below). */
/* THREAD-LOCAL. A job's answer is read back by the completion event that
 * follows it, and jc_execute signals immediately after each job -- but the walk
 * runs on whichever guest thread drives the chain, and more than one can be in
 * flight. As plain globals these were clobbered between a job finishing and its
 * own completion being pushed, so a query could be answered with an unrelated
 * job's mailbox. Per-thread keeps each job paired with its own answer. */
/* __declspec(thread) for MSVC only: MinGW GCC ignores it, which would make
 * these process-wide. spurs_job.c and cellSpurs.c must agree. */
#if defined(_MSC_VER)
#  define SPURS_JOB_TLS __declspec(thread)
#else
#  define SPURS_JOB_TLS __thread
#endif
SPURS_JOB_TLS uint32_t g_spurs_job_mbox, g_spurs_job_mbox_intr;
/* The command this job ran, read from its own command block. Only a query
 * expects an answer back through the completion event. */
SPURS_JOB_TLS uint32_t g_spurs_job_cmd;
SPURS_JOB_TLS int g_spurs_job_mbox_valid;
/* The SPU number a job sees (cellSpursGetCurrentSpuId, kernel LS 0x1C8). A host
 * thread that runs jobs concurrently with others must claim its own: titles
 * key per-SPU resources on it (Edge output rings, one RSX label per SPU), and
 * two jobs that both think they are SPU 0 corrupt each other's ring. */
SPURS_JOB_TLS uint32_t g_spurs_job_spu_num;
void spurs_job_set_spu_num(uint32_t n) { g_spurs_job_spu_num = n; }

/* Immutable copies supplied by the asynchronous JQ. Addresses in DMA entries
 * remain guest EAs; only the descriptor itself and initial binary are snapped. */
static uint32_t job_g32(uint32_t job_ea, const uint8_t* snapshot,
                        uint32_t snapshot_size, uint32_t ea)
{
    if (snapshot && ea >= job_ea && ea - job_ea <= snapshot_size &&
        snapshot_size - (ea - job_ea) >= 4u) {
        const uint8_t* p = snapshot + (ea - job_ea);
        return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
    return g32(ea);
}
static uint16_t job_g16(uint32_t job_ea, const uint8_t* snapshot,
                        uint32_t snapshot_size, uint32_t ea)
{
    if (snapshot && ea >= job_ea && ea - job_ea <= snapshot_size &&
        snapshot_size - (ea - job_ea) >= 2u) {
        const uint8_t* p = snapshot + (ea - job_ea);
        return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
    }
    return g16(ea);
}
static uint64_t job_g64(uint32_t job_ea, const uint8_t* snapshot,
                        uint32_t snapshot_size, uint32_t ea)
{
    return ((uint64_t)job_g32(job_ea, snapshot, snapshot_size, ea) << 32) |
           job_g32(job_ea, snapshot, snapshot_size, ea + 4u);
}

static int spu_run_spurs_job_internal(spu_lifted_entry_fn entry, int image_id,
                                     uint32_t job_ea, uint32_t job_desc_size,
                                     const spu_job_abi* job_abi_in,
                                     const uint8_t* image_copy,
                                     const uint8_t* desc_copy,
                                     spu_job_phase_times* phases)
{
    if (phases) memset(phases, 0, sizeof(*phases));
    if (!entry || !job_ea) return -1;
    double phase_start = phases ? spu_job_profile_now() : 0.0;

    /* Diagnostic options are launch settings. Cache even absent options:
     * getenv takes the CRT environment lock on Windows, and this path runs
     * for every submitted job on several host threads. */
    static SPURS_JOB_TLS int trace_initialized;
    static SPURS_JOB_TLS int trace_desc, trace_desc_level, trace_dump, trace_manager;
    if (!trace_initialized) {
        const char* desc = getenv("SPURS_JOB_DESCDUMP");
        trace_desc = desc != NULL;
        trace_desc_level = desc ? atoi(desc) : 0;
        trace_dump = getenv("SPURS_JOB_DUMP") != NULL;
        trace_manager = getenv("SPURS_JOB_MANAGER_TRACE") != NULL ||
                        getenv("GFD_QUEUE_TRACE") != NULL;
        trace_initialized = 1;
    }

    /* ---- descriptor ---------------------------------------------------- */
    uint32_t ea_bin   = (uint32_t)(job_g64(job_ea, desc_copy, job_desc_size, job_ea + JH_EA_BINARY) & ~1ull);
    uint32_t size_bin = (uint32_t)job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_BINARY) << 4;
    uint32_t n_dma    = (uint32_t)job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_DMA_LIST) / 8u;
    uint32_t use_io   = job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_USE_INOUT_BUFFER);
    uint32_t size_io  = job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_IN_OR_INOUT);
    uint32_t size_out = job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_OUT);
    uint32_t size_stk = (uint32_t)job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_STACK) << 4;
    uint32_t size_scr = (uint32_t)job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_SCRATCH) << 4;
    uint32_t n_cache  = job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_CACHE_DMA_LIST) / 8u;

    if (n_cache > 4) n_cache = 4;                    /* jm2 caps at 4 */
    if (!size_stk) size_stk = JOB_STACK_DEFAULT;
    if (!size_bin || size_bin > SPU_LS_SIZE) {
        fprintf(stderr, "[spurs-job] job 0x%08X: bad sizeBinary=%u -- not run\n",
                job_ea, size_bin);
        if (phases) phases->reset = spu_job_profile_now() - phase_start;
        return -1;
    }

    /* Stage directly into the execution context. The old path zeroed a
     * separate 256 KB buffer, zeroed the context's own local store, then
     * copied the full buffer into it for every synchronous job. */
    spu_context* ctx = (spu_context*)job_context_take(sizeof(*ctx));
    if (!ctx) {
        if (phases) phases->reset = spu_job_profile_now() - phase_start;
        return -1;
    }
    spu_context_init(ctx, 0);
    uint8_t* ls = ctx->ls;
    if (phases) {
        double now = spu_job_profile_now();
        phases->reset = now - phase_start;
        phase_start = now;
    }

    /* ---- binary at LS 0 (PIC, so delta 0 keeps the lifted addresses) ---- */
    memcpy(ls, image_copy ? image_copy : vm_base + ea_bin, size_bin);

    /* ---- regions, following _cellSpursCheckJob's own arithmetic --------- */
    uint32_t p         = ALIGN1024(size_bin);
    /* Same rule as the scratch pointer: these are REGION BASES and stay valid
     * when the region is empty. NULL invites the job to dereference 0 -- and
     * since the job binary loads at LS 0, reading through a null base returns
     * the job's OWN INSTRUCTION WORDS. That is exactly what Tokyo Jungle's
     * wedged jobs were doing: each spun DMAing to the word at offset 0x64 of
     * its own image, which is why the addresses were misaligned and looked
     * like floats -- they were opcodes. */
    uint32_t io_ls     = p;                p += ALIGN1024(size_io);
    uint32_t out_ls    = p;                p += ALIGN1024(size_out);
    if (use_io && !out_ls) out_ls = io_ls;    /* in-out: output shares ioBuffer */
    uint32_t ss_size   = ALIGN1024(size_scr + size_stk);
    /* The scratch pointer is the BASE OF THE SCRATCH+STACK BLOCK, and it is
     * valid even when sizeScratch is 0 -- a zero-length buffer still has an
     * address. Handing the job a NULL here meant it dereferenced 0 and read
     * ITS OWN CODE as data: every wedged job in Tokyo Jungle was loading a
     * word from LS 0x64 and using that SPU instruction word as a DMA address,
     * which is why the addresses looked like misaligned floats. */
    uint32_t s_ls      = p;
    uint32_t stack_top = p + ss_size;
    p += ss_size;

    /* ---- cache-hinted read-only inputs ---------------------------------- */
    uint32_t dma_base   = job_ea + JH_SIZE;          /* input list ... */
    uint32_t cache_base = dma_base + n_dma * 8u;     /* ... then cache list */
    uint32_t cache_ls[4] = { 0, 0, 0, 0 };
    for (uint32_t i = 0; i < n_cache; i++) {
        uint64_t e   = job_g64(job_ea, desc_copy, job_desc_size, cache_base + i * 8u);
        uint32_t sz  = (uint32_t)(e >> 32);
        uint32_t eal = (uint32_t)e;
        if (!sz || !eal) continue;
        if (p + sz > SPU_LS_SIZE) break;
        cache_ls[i] = p;
        memcpy(ls + p, vm_base + eal, sz);
        p += ALIGN1024(sz);
    }

    /* ---- input DMA list -> ioBuffer, each entry 16-byte aligned ---------- */
    uint32_t io_cur = io_ls, io_used = 0;
    for (uint32_t i = 0; i < n_dma; i++) {
        uint64_t e   = job_g64(job_ea, desc_copy, job_desc_size, dma_base + i * 8u);
        uint32_t sz  = (uint32_t)((e >> 32) & 0x7FFFu);
        uint32_t eal = (uint32_t)e;
        if (!sz || !eal) continue;
        if (!io_ls || io_cur + sz > SPU_LS_SIZE) break;
        memcpy(ls + io_cur, vm_base + eal, sz);
        io_cur  += (sz + 15u) & ~15u;
        io_used += (sz + 15u) & ~15u;
    }

    /* ---- context + descriptor, parked above the job's memory region ------
     * jm2 keeps these in its own bss (outside job memory); we load the job at
     * LS 0, so the top of LS is the equivalent free space. */
    uint32_t desc_ls = SPU_LS_SIZE - 0x400;     /* descriptors are <= 1024 B */
    uint32_t ctx_ls  = desc_ls - 0x40;          /* CellSpursJobContext2, 48 B */
    uint32_t dsz     = job_desc_size ? job_desc_size : 256u;
    if (dsz > 0x400) dsz = 0x400;
    if (p > ctx_ls)
        fprintf(stderr, "[spurs-job] job 0x%08X: WARNING regions reach 0x%X, "
                        "past the context at 0x%X\n", job_ea, p, ctx_ls);
    memcpy(ls + desc_ls, desc_copy ? desc_copy : vm_base + job_ea, dsz);

    ls32(ls, ctx_ls + JC_IO_BUFFER, io_ls);
    for (int i = 0; i < 4; i++)
        ls32(ls, ctx_ls + JC_CACHE_BUFFER + (uint32_t)i * 4u, cache_ls[i]);
    /* sizeJobDescriptor:4 is the first bitfield of a big-endian u32, so it
     * occupies the TOP 4 bits. 0 means 64 bytes, otherwise n*128. */
    /* The low 28 bits carry the job manager interface for Context2 CRTs.
     * Exact-image ABI metadata supplies validated per-binary exceptions. */
    const int jobcrt_ver13 = memcmp(ls + 0x30, "JOBCRT Ver13", 12) == 0;
    spu_job_abi job_abi = { 0u, 20u, 0u, 0u, 0u };
    if (job_abi_in) job_abi = *job_abi_in;
    if (jobcrt_ver13) {
        job_abi.flags |= SPU_JOB_ABI_CONTEXT2 | SPU_JOB_ABI_KERNEL_CONTEXT;
        if (!job_abi.manager_entry_ls) job_abi.manager_entry_ls = 0x3FF10u;
    }
    const int use_context2 =
        (job_abi.flags & SPU_JOB_ABI_CONTEXT2) != 0;
    const int overlay_kernel_context =
        (job_abi.flags & SPU_JOB_ABI_KERNEL_CONTEXT) != 0;
    const int fixed_manager_interface = job_abi.manager_interface_ls != 0;
    uint32_t jm_iface = use_context2
        ? (fixed_manager_interface ? job_abi.manager_interface_ls : ctx_ls + 0x30u)
        : 0u;
    if (jm_iface && job_abi.manager_entry_ls) {
        const uint32_t entry_slot = job_abi.manager_entry_slot_ls
            ? job_abi.manager_entry_slot_ls : jm_iface + 4u;
        ls32(ls, entry_slot, job_abi.manager_entry_ls);
    }
    /* The same crt also reads the SPURS kernel context with ABSOLUTE loads
     * (lqa): spuNum at 0x1C8 (cellSpursGetCurrentSpuId), the trace buffer at
     * 0x210, a trace-enable byte at 0x175. On hardware the kernel owns
     * 0x100-0x2FF; here the job is loaded at 0, so those reads returned the
     * job's own instruction words and it indexed per-SPU tables with an opcode
     * and halted. The lifted code never reads those bytes as data, so overlay
     * the kernel's view: SPU 0, our DMA tag, tracing off. */
    if (overlay_kernel_context) {
        memset(ls + 0x170, 0, 0x10);
        memset(ls + 0x1C0, 0, 0x10);
        ls32(ls, 0x1C8, g_spurs_job_spu_num);
        ls32(ls, 0x1CC, job_abi.context_tag);
        memset(ls + 0x210, 0, 0x10);
    }
    ls32(ls, ctx_ls + JC_SIZE_JOB_DESC,
         ((dsz == 64 ? 0u : (dsz / 128u) & 0xFu) << 28) | jm_iface);
    ls16(ls, ctx_ls + JC_NUM_IO_BUFFER,    (uint16_t)n_dma);
    ls16(ls, ctx_ls + JC_NUM_CACHE_BUFFER, (uint16_t)n_cache);
    ls32(ls, ctx_ls + JC_O_BUFFER, out_ls);
    ls32(ls, ctx_ls + JC_S_BUFFER, s_ls);
    /* The SDK default is DMA tag 20. A verified per-image ABI can use the
     * Context2 word as a worker-slot selector instead. */
    ls32(ls, ctx_ls + JC_DMA_TAG, job_abi.context_tag);
    ls32(ls, ctx_ls + JC_EA_JOB_DESCRIPTOR,     0);
    ls32(ls, ctx_ls + JC_EA_JOB_DESCRIPTOR + 4, job_ea);

    { static int _n = 0;
      if (PS3_LOG_FIRST(_n, 4))
          fprintf(stderr,
              "[spurs-job] job 0x%08X: bin 0x%08X+%u -> LS 0 | io=0x%05X(%u, %u dma, %u used)"
              " out=0x%05X(%u) scratch=0x%05X(%u) stack_top=0x%05X(%u) cache=%u"
              " ctx=0x%05X desc=0x%05X(%u)\n",
              job_ea, ea_bin, size_bin, io_ls, size_io, n_dma, io_used,
              out_ls, size_out, s_ls, size_scr, stack_top, size_stk, n_cache,
              ctx_ls, desc_ls, dsz); }

    /* SPURS_JOB_DESCDUMP=1: hexdump the guest job descriptor. The SPU derives
     * its DMA and atomic EAs from these words, so when a job spins on a
     * lock-line address that is not backed, this is what to read first. */
    { static int _d = 0;
      if (trace_desc && PS3_LOG_FIRST(_d, 4)) {
          fprintf(stderr, "[spurs-job] desc @0x%08X (%u bytes):", job_ea, dsz);
          for (uint32_t o = 0; o < dsz && o < 128; o += 4) {
              if ((o & 31) == 0) fprintf(stderr, "\n    +%02X:", o);
              fprintf(stderr, " %08X", job_g32(job_ea, desc_copy, job_desc_size, job_ea + o));
          }
          fprintf(stderr, "\n");
          /* SPURS_JOB_DESCDUMP=2: follow plausible guest EAs in the job user
           * data (past JH_SIZE) and dump what they point at. The job reads its
           * parameters from there, so a wild DMA/atomic address usually
           * originates in one of these blocks. */
          if (trace_desc_level >= 2) {
              for (uint32_t o = JH_SIZE; o < dsz && o < 128; o += 4) {
                  uint32_t v = job_g32(job_ea, desc_copy, job_desc_size, job_ea + o);
                  if (v < 0x10000u || v >= 0xD0000000u) continue;
                  fprintf(stderr, "[spurs-job]   user+0x%02X -> 0x%08X:", o, v);
                  for (uint32_t q = 0; q < 8; q++)
                      fprintf(stderr, " %08X", g32(v + q * 4));
                  fprintf(stderr, "\n");
              }
          }
      } }

    /* LBP job protocol probe (SPURS_JOB_DUMP): the game's one shared job binary
     * ("JOBCRT Ver13" crt + main @0x1570) reads a be u64 EA out of the
     * descriptor's USER DATA at +0x30, GETs a 128-byte command block from it,
     * then jump-tables on the command TYPE (word1 of the block, cases 0..5).
     * The header's input-DMA-list/io fields are legitimately zero for it. Dump
     * the user data + that command block so the bail path is attributable. */
    uint32_t dump_cmd = 0;
    if (trace_dump) {
        uint64_t ud  = job_g64(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE);        /* desc+0x30 */
        uint32_t cmd = (uint32_t)ud;                 /* low word = EA */
        fprintf(stderr, "[spurs-job] job 0x%08X userdata: %016llX %016llX %016llX %016llX\n",
                job_ea, (unsigned long long)ud,
                (unsigned long long)job_g64(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE + 8),
                (unsigned long long)job_g64(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE + 16),
                (unsigned long long)job_g64(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE + 24));
        if (cmd && cmd < 0x50000000u) {
            dump_cmd = cmd;
            fprintf(stderr, "[spurs-job]   cmdblock @0x%08X:", cmd);
            for (int o = 0; o < 64; o += 4) {
                if ((o & 15) == 0) fprintf(stderr, "\n      +%02X:", o);
                fprintf(stderr, " %08X", g32(cmd + o));
            }
            fprintf(stderr, "\n");
            /* cmdblock+0x1C points at a 2048-byte WORK-ITEM table the type-0/1
             * handler GETs (pc=0x1A78) before entering its per-item worker with
             * count = cmdblock+0x24. Dump its head: all-zero means the game
             * never filled the items (upstream/ordering); real entries mean the
             * SPU worker mishandles them. */
            uint32_t tbl = g32(cmd + 0x1C);
            if (tbl && tbl < 0x50000000u) {
                fprintf(stderr, "[spurs-job]   worktable @0x%08X (n=%u):", tbl, g32(cmd + 0x24));
                for (int o = 0; o < 96; o += 4) {
                    if ((o & 15) == 0) fprintf(stderr, "\n      +%02X:", o);
                    fprintf(stderr, " %08X", g32(tbl + o));
                }
                fprintf(stderr, "\n");
            }
        }
        fflush(stderr);
    }

    /* ---- enter the job -------------------------------------------------- */
    ctx->image_id = image_id;
    ctx->spurs_job_services = 1;
    /* SPU stack grows down. A function saves its link register in its
     * caller's frame, at 16($sp), so the entry's first store lands 16 bytes
     * above $sp. The cache buffers start right at stack_top: with $sp at
     * stack_top - 16 that save overwrote the first quadword of
     * cacheBuffer[0] (Twisted Metal's Edge jobs read their viewport there and
     * got width = height = 0). Leave the caller's 32-byte frame inside the
     * stack. */
    ctx->gpr[1]._u32[0] = (stack_top - 32u) & ~15u;
    /* Link register: where the job returns when it is done. The job manager
     * would pass an address inside itself; 0 makes the job re-enter its own
     * entry at LS 0 and run a bogus second lap. */
    ctx->gpr[0]._u32[0] = 0x3FF00u;                   /* SPU_JOB_RETURN_LS */
    ctx->gpr[3]._u32[0] = ctx_ls;                     /* CellSpursJobContext2* */
    ctx->gpr[4]._u32[0] = desc_ls;                    /* CellSpursJob256*      */

    { static int _n = 0;
      if (PS3_LOG_FIRST(_n, 4))
          fprintf(stderr, "[spurs-job] ENTER r3=%08X r4=%08X r1=%08X entry-fn image=%d\n",
                  ctx->gpr[3]._u32[0], ctx->gpr[4]._u32[0], ctx->gpr[1]._u32[0],
                  image_id); }
    /* NOTE: after its work the job's runtime jumps to LS 0 ("return to the
     * jm2 kernel" -- real layout has the kernel at 0, jobs above; we load the
     * job at 0). That second lap re-enters the job's crt with dead registers
     * and trips its parameter guard, which HALTS -- ending the run. So the
     * HALT-ASSERT heqi 0x1650 seen once per jm2 job is NORMAL COMPLETION
     * noise, not a failure: the job's real work finished before the jump. */
    if (phases) {
        double now = spu_job_profile_now();
        phases->stage = now - phase_start;
        phase_start = now;
    }
    const int halted = spu_run_with_halt(entry, ctx);
    if (phases) {
        double now = spu_job_profile_now();
        phases->execute = now - phase_start;
        phase_start = now;
    }
    const uint32_t exit_pc = ctx->pc & SPU_LS_MASK;
    /* The manager return trampolines terminate via spu_halt without being an
     * SPU halt instruction. A STOP or halt at any other PC is not completion. */
    const int manager_return = halted &&
        (exit_pc == 0x3FF00u || exit_pc == 0u) &&
        ctx->status != SPU_STATUS_STOPPED_BY_STOP &&
        ctx->status != SPU_STATUS_STOPPED_BY_HALT;
    const int completed = manager_return || (!halted &&
        ctx->status != SPU_STATUS_STOPPED_BY_STOP &&
        ctx->status != SPU_STATUS_STOPPED_BY_HALT);
    if (!completed) {
        static atomic_uint failures;
        if (atomic_fetch_add_explicit(&failures, 1, memory_order_relaxed) < 32)
            fprintf(stderr, "[spurs-job] execution failed image=%d job=%08X "
                    "pc=%05X status=%X stop=%X halted=%d\n", image_id,
                    job_ea, exit_pc, ctx->status, ctx->stop_code, halted);
    }
    static atomic_int manager_trace_count;
    if (trace_manager && fixed_manager_interface &&
        atomic_fetch_add_explicit(&manager_trace_count, 1, memory_order_relaxed) < 16)
    {
        fprintf(stderr, "[spu-job-manager] job=%08X pc=%05X status=%X stop=%X r3=%08X\n",
                job_ea, ctx->pc, ctx->status, ctx->stop_code, ctx->gpr[3]._u32[0]);
    }
    if (trace_manager && fixed_manager_interface &&
        ctx->status == SPU_STATUS_STOPPED_BY_HALT) {
        static atomic_uint manager_halt_trace_count;
        if (atomic_fetch_add_explicit(&manager_halt_trace_count, 1, memory_order_relaxed) < 4) {
            fprintf(stderr,
                    "[spu-job-manager-guard] r1=%08X r6=%08X r90=%08X/%08X/%08X/%08X "
                    "r101=%08X/%08X/%08X/%08X r61=%08X/%08X/%08X/%08X "
                    "r66=%08X r69=%08X r75=%08X r96=%08X\n",
                    ctx->gpr[1]._u32[0], ctx->gpr[6]._u32[0],
                    ctx->gpr[90]._u32[0], ctx->gpr[90]._u32[1],
                    ctx->gpr[90]._u32[2], ctx->gpr[90]._u32[3],
                    ctx->gpr[101]._u32[0], ctx->gpr[101]._u32[1],
                    ctx->gpr[101]._u32[2], ctx->gpr[101]._u32[3],
                    ctx->gpr[61]._u32[0], ctx->gpr[61]._u32[1],
                    ctx->gpr[61]._u32[2], ctx->gpr[61]._u32[3],
                    ctx->gpr[66]._u32[0], ctx->gpr[69]._u32[0],
                    ctx->gpr[75]._u32[0], ctx->gpr[96]._u32[0]);
            const uint32_t p = ctx->gpr[101]._u32[0] + 0x10u;
            if (p <= SPU_LS_SIZE - 16u) {
                fprintf(stderr, "[spu-job-manager-guard] LS[%05X..%05X]:",
                        p, p + 15u);
                for (uint32_t i = 0; i < 16; ++i)
                    fprintf(stderr, " %02X", ctx->ls[p + i]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr,
                    "[spu-job-manager-desc] ea=%08X binary=%08X size16=%04X dmaBytes=%04X "
                    "io=%08X in=%08X out=%08X stack/scratch=%04X/%04X\n",
                    job_ea, (uint32_t)job_g64(job_ea, desc_copy, job_desc_size, job_ea + JH_EA_BINARY),
                    job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_BINARY), job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_DMA_LIST),
                    job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_USE_INOUT_BUFFER),
                    job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_IN_OR_INOUT), job_g32(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_OUT),
                    job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_STACK), job_g16(job_ea, desc_copy, job_desc_size, job_ea + JH_SIZE_SCRATCH));
            fflush(stderr);
        }
    }

    /* Capture the job's outbound mailbox. These are LS POINTERS TO STRINGS for
     * the SPU printf service, NOT query results -- value 0x83C0 from the sound
     * job points at " compressor, 1/n..." inside its own image. An earlier
     * comment here claimed they carried the title's bus counts; they do not,
     * and feeding them into the completion event turned 0x40000000 into a 1 GB
     * allocation request. Kept because the printf service needs them. */
    /* Only legacy job chains publish a handle-addressable LS snapshot.
     * Concurrent JQ jobs must not overwrite a job-chain's retained LS. */
    if (!desc_copy) memcpy(s_job_ls, ctx->ls, SPU_LS_SIZE);
    /* TEMP SPURS_JOB_LSWORD=<img>:<lsa>: after a job of that image, print the
     * LS word at lsa and every other LS offset holding the same value. */
    { static int img = -2; static uint32_t lsa; static int n = 0;
      if (img == -2) { const char* e = ps3_env_once("SPURS_JOB_LSWORD"); img = -1;
          if (e) { img = atoi(e); const char* c = strchr(e, ':'); lsa = c ? (uint32_t)strtoul(c + 1, 0, 16) : 0; } }
      if (img == image_id && n++ < 3) {
          const uint8_t* q = ctx->ls + (lsa & 0x3FFFC);
          uint32_t v = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) | ((uint32_t)q[2] << 8) | q[3];
          fprintf(stderr, "[lsword] img=%d job=0x%08X LS[0x%05X]=0x%08X io=0x%05X also at:", image_id, job_ea, lsa, v, io_ls);
          for (uint32_t o = 0; o + 4 <= SPU_LS_SIZE; o += 4) {
              const uint8_t* p = ctx->ls + o;
              if (o != (lsa & 0x3FFFC) && ((uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]) == v)
                  fprintf(stderr, " 0x%05X", o);
          }
          fprintf(stderr, "\n");
      } }
    if (!desc_copy) s_job_ls_valid = 1;

    g_spurs_job_mbox      = spu_channel_has_data(&ctx->ch_out_mbox)
                          ? ctx->ch_out_mbox.value : 0;
    g_spurs_job_mbox_intr = spu_channel_has_data(&ctx->ch_out_intr_mbox)
                          ? ctx->ch_out_intr_mbox.value : 0;
    g_spurs_job_mbox_valid = g_spurs_job_mbox || g_spurs_job_mbox_intr;
    { uint32_t _cb = job_g32(job_ea, desc_copy, job_desc_size, job_ea + 0x4C);
      g_spurs_job_cmd = _cb ? (g32(_cb) >> 16) : 0; }
    { static int s_t = -1; if (s_t < 0) s_t = ps3_env_once("SPURS_JOB_MBOX") ? 1 : 0;
      static int n = 0;
      if (s_t && PS3_LOG_FIRST(n, 16) && g_spurs_job_mbox_valid)
          fprintf(stderr, "[spurs-job] job 0x%08X posted mbox=0x%08X intr=0x%08X\n",
                  job_ea, g_spurs_job_mbox, g_spurs_job_mbox_intr); }

    if (trace_dump) {
        fprintf(stderr, "[spurs-job] job 0x%08X exit: status=0x%X stop=0x%X pc=0x%05X\n",
                job_ea, ctx->status, ctx->stop_code, ctx->pc);
        /* Post-run view of the same command block: the game polls result/status
         * fields the job writes back -- a before/after diff shows whether the
         * handler delivered its completion or the PPU waits on stale state. */
        if (dump_cmd) {
            fprintf(stderr, "[spurs-job]   cmdblock @0x%08X AFTER:", dump_cmd);
            for (int o = 0; o < 64; o += 4) {
                if ((o & 15) == 0) fprintf(stderr, "\n      +%02X:", o);
                fprintf(stderr, " %08X", g32(dump_cmd + o));
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }

    spu_coh_unregister(ctx);   /* remove callbacks before releasing owned state */
    spu_mfc_release(ctx);
    job_context_put(ctx);
    if (phases) phases->teardown = spu_job_profile_now() - phase_start;
    return completed ? 0 : -1;
}

int spu_run_spurs_job_with_abi(spu_lifted_entry_fn entry, int image_id,
                               uint32_t job_ea, uint32_t job_desc_size,
                               const spu_job_abi* job_abi)
{
    return spu_run_spurs_job_internal(entry, image_id, job_ea, job_desc_size,
                                      job_abi, NULL, NULL, NULL);
}

int spu_run_spurs_job_with_abi_snapshot(spu_lifted_entry_fn entry, int image_id,
                                        uint32_t job_ea, uint32_t job_desc_size,
                                        const spu_job_abi* job_abi,
                                        const uint8_t* image_copy,
                                        const uint8_t* desc_copy)
{
    if (!image_copy || !desc_copy) return -1;
    return spu_run_spurs_job_internal(entry, image_id, job_ea, job_desc_size,
                                      job_abi, image_copy, desc_copy, NULL);
}

int spu_run_spurs_job(spu_lifted_entry_fn entry, int image_id,
                      uint32_t job_ea, uint32_t job_desc_size)
{
    spu_job_abi default_abi = { 0u, 20u, 0u, 0u, 0u };
    (void)spu_workload_get_job_abi(image_id, &default_abi);
    return spu_run_spurs_job_with_abi(entry, image_id, job_ea,
                                      job_desc_size, &default_abi);
}

int spu_run_spurs_job_profiled(spu_lifted_entry_fn entry, int image_id,
                               uint32_t job_ea, uint32_t job_desc_size,
                               const spu_job_abi* job_abi,
                               const uint8_t* image_copy, const uint8_t* desc_copy,
                               spu_job_phase_times* phases)
{
    return spu_run_spurs_job_internal(entry, image_id, job_ea, job_desc_size,
                                      job_abi, image_copy, desc_copy, phases);
}
