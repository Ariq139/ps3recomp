#ifndef PS3EMU_LWSYNC_BRIDGE_H
#define PS3EMU_LWSYNC_BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Direct C sysPrxForUser guest-EA calls can share the context-HLE registry
 * when the runtime explicitly installs this bridge before guest dispatch. */
enum {
    PS3_LWSYNC_MUTEX_CREATE, PS3_LWSYNC_MUTEX_DESTROY, PS3_LWSYNC_MUTEX_LOCK,
    PS3_LWSYNC_MUTEX_TRYLOCK, PS3_LWSYNC_MUTEX_UNLOCK,
    PS3_LWSYNC_COND_CREATE, PS3_LWSYNC_COND_DESTROY, PS3_LWSYNC_COND_WAIT,
    PS3_LWSYNC_COND_SIGNAL, PS3_LWSYNC_COND_SIGNAL_ALL
};
typedef int32_t (*ps3_lwsync_bridge_fn)(unsigned op, uint32_t ea, uint64_t arg);
extern ps3_lwsync_bridge_fn g_ps3_lwsync_bridge;
#ifdef __cplusplus
}
#endif
#endif
