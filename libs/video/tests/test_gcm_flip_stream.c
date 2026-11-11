/* A flip queued in the command stream (cellGcmSys.c).
 *
 * _cellGcmSetFlipCommandWithWaitLabel(ctx, buffer, label, value): libgcm puts
 * NV406E SEMAPHORE_OFFSET and SEMAPHORE_ACQUIRE for the label into the context,
 * then the flip. The import used to write the label itself from the PPU and flip
 * at call time.
 *
 * And the flip completes when the drain reaches it: the 60 Hz tick used to mark
 * it done on the next beat wherever the walker was, so a title pacing itself on
 * the flip status rebuilt command segments the walker had not executed yet.
 *
 *   1. The context holds OFFSET, ACQUIRE, then the flip marker, current has
 *      moved past them, and the label in memory is untouched.
 *   2. A tick before the drain is a vblank but no flip.
 *   3. The drain executes up to the marker; the next tick completes the flip.
 *   4. A flip written but not flushed does not complete; an executed one does,
 *      even with a newer flip written behind it and not yet flushed.
 *   5. A flip the walker abandons when it resyncs out of unmapped IO does not
 *      hold back the flip status.
 *   6. With every flip completed, a beat completes nothing: not after the
 *      status was reset, and not for a flip queued after that beat.
 *   7. A flip requested outside the stream completes on the next beat, once.
 *   8. A flip in the stream puts ZSTENCIL_CLEAR_VALUE back to its reset value
 *      for every consumer of the methods, after the methods ahead of it.
 *
 * gcc -std=gnu17 -I include -I libs/video libs/video/tests/test_gcm_flip_stream.c
 *   -ffunction-sections -fdata-sections -Wl,--gc-sections -pthread
 *   -o /tmp/test_gcm_flip_stream        (clang on macOS: -Wl,-dead_strip)
 */
#include "../cellGcmSys.c"
#include "../rsx_tiling.c"
#include <stdio.h>
#include <stdlib.h>
/* GCM_REPORT_SPU_HOLD_MS asks the SPU task layer whether a task round is
 * running; there are no SPU tasks here. */
int spu_taskset_round_busy(uint32_t* ts, uint32_t* tid) { (void)ts; (void)tid; return 0; }
uint8_t* vm_base;
uint32_t ppu_vm_size;
int g_resv_store_active;
uint32_t g_ww_lo, g_ww_hi;
void ppu_resv_break_store(uint64_t ea) { (void)ea; }
void ps3_ww_report_inline(uint32_t a, uint64_t v, int w) { (void)a; (void)v; (void)w; }
int spu_coh_is_reserved(uint32_t a) { (void)a; return 0; }
void spu_coh_notify_write(uint32_t a) { (void)a; }
void spu_lockline_lock(void) {}
void spu_lockline_unlock(void) {}
ps3_guest_caller_fn g_ps3_guest_caller;
/* The rest of the walker's world: no graphics backend, no lifted program. */
PPU_THREAD_LOCAL ppu_context* g_active_ctx;
u32 g_rsx_last_reference;
void (*g_spu_put_hook)(uint32_t ea, uint32_t size);
void rsx_raise_user_cmd(uint32_t arg) { (void)arg; }
/* gcm_fifo_next_segment wakes the drain thread; this fixture has none, and
 * runtime/platform/win32_compat.c is not linked. */
#ifndef _WIN32
BOOL SetEvent(HANDLE h) { (void)h; return 1; }
#endif
void ppu_dump_bctrl_ring(uint32_t n, const char* why) { (void)n; (void)why; }
void ppu_dump_guest_stack(ppu_context* c, const char* why) { (void)c; (void)why; }
void rsx_state_init(rsx_state* state) { (void)state; }
/* ZSTENCIL_CLEAR_VALUE writes as each consumer of the methods saw them. */
static u32 g_zs_state[4], g_zs_engine[4];
static int g_n_zs_state, g_n_zs_engine, g_engine_on;
int  rsx_process_method(rsx_state* state, u32 method, u32 data)
{
    (void)state;
    if (method == 0x1D8Cu && g_n_zs_state < 4) g_zs_state[g_n_zs_state++] = data;
    return 0;
}
int  rsx_draw_engine_enabled(void) { return g_engine_on; }
void rsx_draw_engine_method(u32 method, u32 arg)
{
    if ((method & 0x1FFCu) == 0x1D8Cu && g_n_zs_engine < 4) g_zs_engine[g_n_zs_engine++] = arg;
}

/* No microcode or constant runs here: the walker hands every word to
 * rsx_process_method instead (#554). */
int  rsx_process_vp_run(rsx_state* state, u32 method, const u8* src, u32 count)
{ (void)state; (void)method; (void)src; (void)count; return 0; }
/* cellGcmSys asks whether a vtable backend is registered (#466). */
rsx_backend* rsx_get_backend(void) { return NULL; }
void rsx_draw_engine_set_display_buffer(u32 id, u32 location, u32 offset,
                                        u32 pitch, u32 width, u32 height)
{ (void)id; (void)location; (void)offset; (void)pitch; (void)width; (void)height; }
int rsx_replay_skipped_sync_method(u32 subchannel, u32 method, u32 data)
{ (void)subchannel; (void)method; (void)data; return 0; }
#ifdef _WIN32
void rsx_d3d12_composite_from_offscreen(uint32_t offset) { (void)offset; }
#endif
void rsx_draw_engine_note_flip(u32 buffer_id) { (void)buffer_id; }
void rsx_draw_engine_guest_read(u32 l, u32 o, u32 p, u32 w, u32 h) { (void)l; (void)o; (void)p; (void)w; (void)h; }
void rsx_draw_engine_guest_wrote(u32 l, u32 o, u32 p, u32 w, u32 h) { (void)l; (void)o; (void)p; (void)w; (void)h; }
int  rsx_live_draw_enabled(void) { return 0; }
void rsx_live_draw_method(u32 method, u32 arg) { (void)method; (void)arg; }
int  rsx_live_draw_blit(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 w, u32 h)
{ (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)w; (void)h; return 0; }
void rsx_live_draw_note_resolve(u32 a, u32 b, u32 c, u32 d) { (void)a; (void)b; (void)c; (void)d; }
int  rsx_live_draw_resolve_blit(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h,
                                u32 i, u32 j, u32 k, u32 l)
{ (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; (void)j; (void)k; (void)l; return 0; }

#define IO_ADDR   0x00100000u
#define IO_SIZE   0x00400000u
#define CMD_SIZE  0x00080000u
#define TITLE_CB  0x012D6A78u      /* a title function: its own buffer-full callback */


#define IO_ADDR   0x00100000u
#define IO_SIZE   0x00400000u
#define CMD_SIZE  0x00080000u
#define TITLE_CB  0x012D6A78u      /* a title function: its own buffer-full callback */
#define LABEL     0x42u
#define VALUE     7u

static uint32_t g_heap = 0x00800000u;
static u32 host_alloc(u32 size, u32 align)
{
    if (!align) align = 4;
    g_heap = (g_heap + align - 1u) & ~(align - 1u);
    u32 r = g_heap;
    g_heap += size;
    return r;
}
static void host_w32(u32 ea, u32 v) { vm_write32(ea, v); }

static int fails;
static void check(int ok, const char* what)
{
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    fails += !ok;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const uint32_t vm_size = ppu_hle_inject_base + 0x10000u;
    vm_base = (uint8_t*)calloc(1, vm_size);
    if (!vm_base) { puts("FAIL: guest VM alloc"); return 1; }
    ppu_vm_size = vm_size;

    u32 ctx_out = host_alloc(4, 4);
    (void)cellGcmSetupContext(ctx_out, CMD_SIZE, IO_SIZE, IO_ADDR, host_alloc, host_w32);
    s_display_buffer_set[1] = 1;

    /* A title context inside the IO window, with its own callback (so the
     * flip never wraps it), and the walker idle at its start. */
    const u32 ctx = host_alloc(0x18, 16), buf = IO_ADDR + 0x200000u;
    vm_write32(ctx + 0x0, buf);
    vm_write32(ctx + 0x4, buf + 0x30000u);
    vm_write32(ctx + 0x8, buf);
    vm_write32(ctx + 0xC, ctx + 0x10);
    vm_write32(ctx + 0x10, TITLE_CB);
    vm_write32(ctx + 0x14, 0);
    const u32 io = gcm_ea2io(buf);
    s_fifo_getoff = io;
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, io);    /* put == get: nothing queued */
    const u32 label_ea = GCM_LABEL_GUEST_BASE + LABEL * GCM_LABEL_STRIDE;
    vm_write32(label_ea, 0);
    (void)GCM_PENDING_TAKE();

    /* 1 */
    const s32 rc = _cellGcmSetFlipCommandWithWaitLabel((void*)(uintptr_t)ctx, 1, LABEL, VALUE);
    check(rc == CELL_OK, "WithWaitLabel returns CELL_OK");
    check(vm_read32(buf + 0) == 0x00040064u && vm_read32(buf + 4) == LABEL * GCM_LABEL_STRIDE,
          "SEMAPHORE_OFFSET for the label is in the context");
    check(vm_read32(buf + 8) == 0x00040068u && vm_read32(buf + 12) == VALUE,
          "SEMAPHORE_ACQUIRE for the value follows it");
    check(vm_read32(buf + 16) == (GCM_FLIP_MARKER | 1u), "then the flip");
    check(vm_read32(ctx + 8) == buf + 20, "current is past the flip");
    check(vm_read32(label_ea) == 0, "the label itself is not written by the PPU");

    /* 2 */
    cellGcm_request_tick();
    long p = GCM_PENDING_TAKE();
    check((p & 1) && !(p & 2), "a tick before the drain is a vblank, not a flip");

    /* 3: the producer satisfies the acquire before the walker reaches it. */
    vm_write32(label_ea, VALUE);
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, io + 20);
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == io + 20, "the drain consumed the wait and the flip");
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check((p & 1) && (p & 2), "the next tick completes the flip");

    /* 4: a second flip is written but not flushed (put stays). The first one
     * has already completed, so the tick has nothing new to report... */
    const u32 cur2 = vm_read32(ctx + 8);
    (void)_cellGcmSetFlipCommandWithWaitLabel((void*)(uintptr_t)ctx, 1, LABEL, VALUE);
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check((p & 1) && !(p & 2), "a written, unflushed flip does not complete");
    /* ...and once it is flushed and executed, it completes even though a third
     * one has been written behind it and not flushed: the RSX reports the flip
     * it did, not the ones still ahead of it. */
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, gcm_ea2io(cur2) + 20);
    (void)_cellGcmSetFlipCommandWithWaitLabel((void*)(uintptr_t)ctx, 1, LABEL, VALUE);
    cellGcm_rsx_process_fifo();
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check((p & 1) && (p & 2), "an executed flip completes with a newer one still unflushed");

    /* 5: the walker is lost in unmapped IO and resyncs to put, abandoning the
     * flip it had not reached. That flip will never be executed, so the next
     * flip status must not wait for it. */
    const u32 cur3 = vm_read32(ctx + 8);
    (void)_cellGcmSetFlipCommandWithWaitLabel((void*)(uintptr_t)ctx, 1, LABEL, VALUE);
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check(!(p & 2), "a written, unflushed flip does not complete (again)");
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, gcm_ea2io(cur3) + 20);
    s_fifo_getoff = 0x0FD00000u;              /* no mapping there */
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == gcm_ea2io(cur3) + 20, "the walker resynced to put");
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check((p & 1) && (p & 2), "a flip abandoned by a resync does not hold back the status");

    /* 6: everything queued has completed. The title resets the status and
     * builds its next frame; beats go by before it queues the flip. */
    cellGcmResetFlipStatus();
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check((p & 1) && !(p & 2), "with no flip outstanding, a beat completes nothing");
    const u32 cur4 = vm_read32(ctx + 8);
    (void)_cellGcmSetFlipCommandWithWaitLabel((void*)(uintptr_t)ctx, 1, LABEL, VALUE);
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check(!(p & 2), "and the flip queued after it waits for the drain");
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, gcm_ea2io(cur4) + 20);
    cellGcm_rsx_process_fifo();
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check(p & 2, "which completes it");

    /* 7 */
    check(cellGcmSetFlipCommand(1) == CELL_OK, "a flip outside the stream is accepted");
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check(p & 2, "it completes on the next beat");
    cellGcm_request_tick();
    p = GCM_PENDING_TAKE();
    check(!(p & 2), "once");

    /* 8: the title sets the clear value to 0 for one clear and flips. (The
     * flips above wrote the reset too.) */
    g_engine_on = 1;
    g_n_zs_state = g_n_zs_engine = 0;
    const u32 cur5 = vm_read32(ctx + 8);
    vm_write32(cur5 + 0, 0x00041D8Cu);                 /* ZSTENCIL_CLEAR_VALUE */
    vm_write32(cur5 + 4, 0);
    vm_write32(cur5 + 8, GCM_FLIP_MARKER | 1u);
    vm_write32(ctx + 8, cur5 + 12);
    vm_write32(GCM_CONTROL_GUEST_ADDR + 0, gcm_ea2io(cur5) + 12);
    cellGcm_rsx_process_fifo();
    /* With the draw engine up and no vtable backend, #466 hands rsx_state
     * only methods with outside effects, so the stream's 0 may not reach it;
     * the flip's reset always does, and is the last word. */
    check(g_n_zs_state >= 1 && g_zs_state[g_n_zs_state - 1] == 0xFFFFFF00u,
          "the method state ends on the flip's 0xFFFFFF00");
    check(g_n_zs_engine >= 1 && g_zs_engine[g_n_zs_engine - 1] == 0xFFFFFF00u,
          "the draw engine gets the flip's 0xFFFFFF00 too");
    g_engine_on = 0;

    /* Retire the previous frame's label reset before flip completion can
     * let the vblank handler grant the next frame. */
    const u32 cur6 = vm_read32(ctx + 8);
    const u32 io6 = gcm_ea2io(cur6);
    vm_write32(cur6, GCM_FLIP_MARKER | 1u);
    vm_write32(cur6 + 4, 0x00040064u);
    vm_write32(cur6 + 8, LABEL * GCM_LABEL_STRIDE);
    vm_write32(cur6 + 12, 0x0004006Cu);
    vm_write32(cur6 + 16, 0);
    vm_write32(cur6 + 20, 0x00040068u);
    vm_write32(cur6 + 24, VALUE);
    vm_write32(ctx + 8, cur6 + 28);
    vm_write32(GCM_CONTROL_GUEST_ADDR, io6 + 28);
    vm_write32(label_ea, VALUE);
    (void)GCM_PENDING_TAKE();
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == io6 + 20, "flip retires only its OFFSET/RELEASE trailer and stops before ACQUIRE");
    check(vm_read32(label_ea) == 0, "previous-frame label reset precedes flip completion");
    check(cellGcm_take_flip_pending_synced(), "one FIFO flip supplies one present token");
    check(!cellGcm_take_flip_pending_synced(), "the present token cannot be consumed twice");
    s_flip_status = CELL_GCM_FLIP_STATUS_DONE;
    cellGcmTickFlip();
    p = GCM_PENDING_TAKE();
    check(p & 2, "an executed FIFO flip completes even when an older callback left status DONE");
    cellGcmTickFlip();
    check(!(GCM_PENDING_TAKE() & 2), "the executed FIFO completion is delivered only once");
    s_flip_status = CELL_GCM_FLIP_STATUS_WAITING;
    cellGcmTickFlip();
    check(!(GCM_PENDING_TAKE() & 2), "WAITING alone does not complete a flip the FIFO has not executed");
    vm_write32(label_ea, VALUE); /* guest completion/vblank grants the next frame */
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == io6 + 28 && vm_read32(label_ea) == VALUE,
          "the next drain preserves the new vblank grant and passes ACQUIRE");

    vm_write32(label_ea, VALUE);
    const u32 cur7 = vm_read32(ctx + 8);
    vm_write32(cur7, GCM_FLIP_MARKER | 1u);
    vm_write32(cur7 + 4, 0x00040064u);
    vm_write32(cur7 + 8, LABEL * GCM_LABEL_STRIDE);
    vm_write32(cur7 + 12, 0x0004006Cu);
    vm_write32(cur7 + 16, 0);
    s_fifo_getoff = gcm_ea2io(cur7);
    vm_write32(GCM_CONTROL_GUEST_ADDR, gcm_ea2io(cur7) + 16); /* release value not flushed */
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == gcm_ea2io(cur7) + 4 && vm_read32(label_ea) == VALUE,
          "an incomplete trailer does not read an unflushed release argument");

    const u32 cur8 = cur7 + 24;
    vm_write32(cur8, GCM_FLIP_MARKER | 1u);
    vm_write32(cur8 + 4, 0x00041D8Cu);
    vm_write32(cur8 + 8, 0);
    s_fifo_getoff = gcm_ea2io(cur8);
    vm_write32(GCM_CONTROL_GUEST_ADDR, s_fifo_getoff + 12);
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == gcm_ea2io(cur8) + 4,
          "graphics methods after a flip remain in the next drain");

    const u32 boundary = IO_ADDR + 0x002FFFFCu;
    const u32 boundary_io = gcm_ea2io(boundary);
    vm_write32(ctx + 4, IO_ADDR + IO_SIZE - 4);
    vm_write32(ctx + 8, boundary + 20);
    vm_write32(boundary, GCM_FLIP_MARKER | 1u);
    vm_write32(boundary + 4, 0x00040064u);
    vm_write32(boundary + 8, LABEL * GCM_LABEL_STRIDE);
    vm_write32(boundary + 12, 0x0004006Cu);
    vm_write32(boundary + 16, 0);
    s_fifo_getoff = boundary_io;
    vm_write32(label_ea, VALUE);
    vm_write32(GCM_CONTROL_GUEST_ADDR, boundary_io + 20);
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == boundary_io + 20 && vm_read32(label_ea) == 0,
          "a fully mapped trailer can cross an IO page boundary");

    const u32 trailer_page = (boundary_io + 4) >> 20;
    const u16 saved_page = s_ea_address_table[trailer_page];
    s_ea_address_table[trailer_page] = 0xFFFFu;
    s_fifo_getoff = boundary_io;
    vm_write32(label_ea, VALUE);
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == boundary_io + 4 && vm_read32(label_ea) == VALUE,
          "an unmapped trailer is left untouched at the flip boundary");
    s_ea_address_table[trailer_page] = saved_page;

    const u32 raw_flip = boundary + 64;
    vm_write32(raw_flip, 0xFEAD0001u);
    vm_write32(raw_flip + 4, 0x00040064u);
    vm_write32(raw_flip + 8, LABEL * GCM_LABEL_STRIDE);
    vm_write32(raw_flip + 12, 0x0004006Cu);
    vm_write32(raw_flip + 16, 0);
    vm_write32(ctx + 8, raw_flip + 20);
    s_fifo_getoff = gcm_ea2io(raw_flip);
    vm_write32(GCM_CONTROL_GUEST_ADDR, s_fifo_getoff + 20);
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == gcm_ea2io(raw_flip) + 20 && vm_read32(label_ea) == 0,
          "the raw FEAD flip form also retires its label-reset trailer");

    const u32 invalid = raw_flip + 32;
    vm_write32(invalid, GCM_FLIP_MARKER | 1u);
    vm_write32(invalid + 4, 0x00040064u);
    vm_write32(invalid + 8, 0x1000u);
    vm_write32(invalid + 12, 0x0004006Cu);
    vm_write32(invalid + 16, 0);
    vm_write32(ctx + 8, invalid + 20);
    s_fifo_getoff = gcm_ea2io(invalid);
    vm_write32(label_ea, VALUE);
    vm_write32(GCM_CONTROL_GUEST_ADDR, s_fifo_getoff + 20);
    cellGcm_rsx_process_fifo();
    check(s_fifo_getoff == gcm_ea2io(invalid) + 4 && vm_read32(label_ea) == VALUE,
          "a release outside the label window is not retired early");

    printf("[gcm_flip_stream] %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    free(vm_base);
    return fails ? 1 : 0;
}
