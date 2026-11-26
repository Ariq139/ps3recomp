#ifndef PS3EMU_AIO_SERVICE_H
#define PS3EMU_AIO_SERVICE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* A native HLE service runs in a real registered PPU thread slot. The opaque
 * pointer is that slot's context, not a caller's transient scratch context. */
typedef void (*ps3_ppu_service_fn)(void*);
int32_t ppu_thread_start_service(ps3_ppu_service_fn entry, ps3_ppu_service_fn cleanup);
typedef struct ps3_aio_runtime_hooks {
    int32_t (*start)(ps3_ppu_service_fn, ps3_ppu_service_fn);
    int (*prepare)(void*);
    int (*resolve)(uint32_t opd, uint32_t* code, uint32_t* toc);
    int (*deliver)(void*, uint64_t sp, uint64_t tp, uint32_t code, uint32_t toc,
                   uint32_t aio, int32_t error, int32_t id, uint64_t size);
    void (*install_process_stop)(void (*stop)(void));
} ps3_aio_runtime_hooks;
const ps3_aio_runtime_hooks* ppu_aio_runtime_hooks(void);
/* Published before guest dispatch. Process termination must never join a
 * guest callback, which could depend on a lock owned by the exiting thread. */
extern void (*g_ps3_aio_process_stop)(void);
#ifdef __cplusplus
}
#endif
#endif
