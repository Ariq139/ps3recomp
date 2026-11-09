/* Optional instrumentation for an outlined pure LS/GPR block. The logical
 * identifier selects the block but never changes the architectural PC.
 * Set SPU_KERNEL_CAPTURE_SCOPE=outlined to exclude drain entries with the
 * same numeric PC from claiming this capture. */
#ifndef PS3RECOMP_SPU_KERNEL_CAPTURE_API_H
#define PS3RECOMP_SPU_KERNEL_CAPTURE_API_H
#include "spu_context.h"
#ifdef __cplusplus
extern "C" {
#endif
void spu_kernel_capture_invoke(spu_context* ctx, uint32_t logical_pc,
                               void (*fn)(spu_context*));
#ifdef __cplusplus
}
#endif
#endif
