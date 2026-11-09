#include "../spu_kernel_capture.h"
#include "../../../tests/spu_ls_watch_fixture.h"
#include <assert.h>
#ifdef SPU_CAPTURE_USE_DRAIN
#include "../spu_drain.c"
/* The selected test path uses no interrupts, lockstep, stack reset or
 * resolver. Unexpected entry into those services is a fixture failure. */
unsigned g_spu_stack_reset_count;
volatile int g_yz_lockstep_on = 0;
void spu_check_stack_reset(spu_context* c, void (*fn)(spu_context*))
{ (void)c; (void)fn; abort(); }
void spu_indirect_branch(spu_context* c) { (void)c; abort(); }
void yz_lockstep_tick(spu_context* c) { (void)c; abort(); }
void spu_restart_dispatch(spu_context* c) { (void)c; abort(); }
void spu_halt(spu_context* c) { (void)c; abort(); }
void spu_jmp_no_unwind(void* p) { (void)p; abort(); }
spu_drain_fn spu_lookup(uint32_t pc, int image)
{ (void)pc; (void)image; abort(); }
#else
SPU_THREAD_LOCAL void (*g_spu_trampoline_fn)(spu_context*);
#endif
#ifdef SPU_CAPTURE_USE_DRAIN
void synthetic_continuation(spu_context* ctx)
{
    ctx->gpr[4]._u32[0]++;
    ctx->pc = 0x300;
#ifdef SPU_CAPTURE_USE_MACRO
    g_spu_trampoline_fn = NULL;
#else
    g_spu_trampoline_fn = spu_indirect_branch;
#endif
}
#else
void synthetic_continuation(spu_context* ctx) { (void)ctx; abort(); }
#endif
void synthetic_kernel(spu_context* ctx)
{
    ctx->gpr[3]._u32[0] += 9;
    spu_ls_write32(ctx, 128, spu_ls_read32(ctx, 128) ^
                   ((ctx->gpr[3]._u32[0] & 255u) << 24));
    if (ctx->gpr[3]._u32[0] & 1u) {
        ctx->pc = 0x210;
        g_spu_trampoline_fn = synthetic_continuation;
        return;
    }
    ctx->pc = 0x200;
    g_spu_trampoline_fn = synthetic_continuation;
}
static void env_set(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
int main(int argc, char** argv)
{
    assert(argc == 2 || argc == 3);
    const uint32_t seed = argc == 3 ? (uint32_t)strtoul(argv[2], NULL, 0) : 5u;
    env_set("SPU_KERNEL_CAPTURE_PREFIX", argv[1]);
    env_set("SPU_KERNEL_CAPTURE_IMAGE", "20489");
    env_set("SPU_KERNEL_CAPTURE_PC", "0x16a0");
    env_set("SPU_KERNEL_CAPTURE_SCOPE", "");
    assert(spu_kernel_capture_select(20488, 0x16A0, synthetic_kernel) == synthetic_kernel);
    assert(spu_kernel_capture_select(20489, 0x16A4, synthetic_kernel) == synthetic_kernel);
    spu_kernel_fn wrapped = spu_kernel_capture_select(20489, 0x16A0, synthetic_kernel);
    assert(wrapped != synthetic_kernel);
    assert(spu_kernel_capture_select(20489, 0x16A0, synthetic_kernel) == wrapped);
    spu_context* ctx = (spu_context*)calloc(1, sizeof(*ctx)); assert(ctx);
    spu_context_init(ctx, 0); ctx->image_id = 20489; ctx->pc = 0x16A0;
    ctx->gpr[3]._u32[0] = seed; ctx->ls[128] = 7;
    /* This assignment is how generated direct trampolines bypass a registry
     * wrapper. The real drain must capture the concrete target itself. */
#ifdef SPU_CAPTURE_USE_DRAIN
    g_spu_trampoline_fn = synthetic_kernel;
#ifdef SPU_CAPTURE_USE_MACRO
    SPU_DRAIN(ctx);
#else
    spu_drain_call(ctx, 0x300);
#endif
#else
    wrapped(ctx);
#endif
    assert(ctx->gpr[3]._u32[0] == seed+9 && ctx->ls[128] == (7 ^ (uint8_t)(seed+9)));
#ifdef SPU_CAPTURE_USE_DRAIN
    assert(ctx->pc == 0x300 && ctx->gpr[4]._u32[0] == 1);
#else
    assert(ctx->pc == (((seed+9)&1u) ? 0x210u : 0x200u) &&
           g_spu_trampoline_fn == synthetic_continuation);
#endif
    ctx->pc = 0x16A0;
#ifdef SPU_CAPTURE_USE_DRAIN
    g_spu_trampoline_fn = synthetic_kernel;
#ifdef SPU_CAPTURE_USE_MACRO
    SPU_DRAIN(ctx);
#else
    spu_drain_call(ctx, 0x300);
#endif
#else
    wrapped(ctx);
#endif
    assert(ctx->gpr[3]._u32[0] == seed+18 &&
           ctx->ls[128] == (7 ^ (uint8_t)(seed+9) ^ (uint8_t)(seed+18)));
    free(ctx);
    puts("kernel capture: exact image/PC; one-shot; original runs exactly once per call");
    return 0;
}
