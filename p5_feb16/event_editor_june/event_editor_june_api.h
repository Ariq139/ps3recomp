#pragma once

#include <stdint.h>
#include "lift/ppu_recomp.h"

typedef struct {
    uint32_t source_ea;
    void (*func)(ppu_context*);
} JuneEventDispatchEntry;

#ifdef __cplusplus
extern "C" {
#endif
extern const JuneEventDispatchEntry june_evt_dispatch_table[];
extern const uint64_t june_evt_dispatch_table_count;
int june_evt_context_enter(ppu_context* ctx, uint32_t source_ea);
void june_evt_context_leave(ppu_context* ctx);
int event_editor_june_init(void);
void june_evt_window_update(ppu_context* ctx);
void june_evt_window_draw(ppu_context* ctx);
#ifdef __cplusplus
}
#endif
