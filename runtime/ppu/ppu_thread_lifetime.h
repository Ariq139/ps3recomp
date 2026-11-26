#ifndef PS3_PPU_THREAD_LIFETIME_H
#define PS3_PPU_THREAD_LIFETIME_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    void (*retire)(void*);
    void (*zero_guest)(uint32_t,uint32_t);
} ps3_ppu_thread_lifetime_hooks;
#if defined(_MSC_VER) || defined(__MINGW32__)
__declspec(selectany) ps3_ppu_thread_lifetime_hooks ps3_ppu_thread_lifetime = {0};
#else
__attribute__((weak)) ps3_ppu_thread_lifetime_hooks ps3_ppu_thread_lifetime = {0};
#endif
#ifdef __cplusplus
}
#endif
#endif
