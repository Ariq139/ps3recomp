/* Exercise the public outlined-block API against the real drain TU. */
#define SPU_CAPTURE_USE_DRAIN
#define main kernel_capture_fixture_main
#include "test_kernel_capture.c"
#undef main

void synthetic_leaf(spu_context* ctx)
{
    ctx->gpr[3]._u32[0] += 9;
    spu_ls_write32(ctx, 128, spu_ls_read32(ctx, 128) ^
                   ((ctx->gpr[3]._u32[0] & 255u) << 24));
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    env_set("SPU_KERNEL_CAPTURE_PREFIX", argv[1]);
    env_set("SPU_KERNEL_CAPTURE_IMAGE", "20489");
    env_set("SPU_KERNEL_CAPTURE_PC", "0x658");
    env_set("SPU_KERNEL_CAPTURE_SCOPE", "outlined");
    spu_context* ctx = (spu_context*)calloc(1, sizeof(*ctx)); assert(ctx);
    spu_context_init(ctx, 0); ctx->image_id = 20488; ctx->pc = 0x40;
    ctx->gpr[3]._u32[0] = 5;
    g_spu_trampoline_fn = synthetic_continuation;
    /* A concrete lifted entry at the same image/PC must not steal the
     * outlined capture; the actual drain's selector must leave it alone. */
    ctx->image_id = 20489; ctx->pc = 0x658;
    assert(spu_drain_hop_select(ctx, synthetic_kernel) == synthetic_kernel);
    assert(!atomic_load(&s_spu_kernel_capture.selected));
    assert(!atomic_load(&s_spu_kernel_capture.claimed));
    ctx->image_id = 20488; ctx->pc = 0x40;
    spu_kernel_capture_invoke(ctx, 0x658, synthetic_leaf); /* wrong image */
    assert(ctx->gpr[3]._u32[0] == 14);
    ctx->image_id = 20489;
    spu_kernel_capture_invoke(ctx, 0x65C, synthetic_leaf); /* wrong logical ID */
    assert(ctx->gpr[3]._u32[0] == 23);
    assert(!atomic_load(&s_spu_kernel_capture.claimed));
    ctx->gpr[3]._u32[0] = 5; ctx->ls[128] = 7;
    spu_kernel_capture_invoke(ctx, 0x658, synthetic_leaf);
    assert(ctx->gpr[3]._u32[0] == 14 && ctx->ls[128] == (7 ^ 14));
    assert(ctx->pc == 0x40 && ctx->gpr[4]._u32[0] == 0);
    assert(g_spu_trampoline_fn == synthetic_continuation);
    spu_kernel_capture_invoke(ctx, 0x658, synthetic_leaf);
    assert(ctx->gpr[3]._u32[0] == 23 && ctx->ls[128] == (7 ^ 14 ^ 23));
    assert(ctx->pc == 0x40 && g_spu_trampoline_fn == synthetic_continuation);
    free(ctx);
    puts("leaf capture: logical0x658 preserves actualPC0x40 and TLS; one call per invocation");
    return 0;
}
