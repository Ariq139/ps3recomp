#ifndef RSX_CONSTANT_SNAPSHOT_CACHE_H
#define RSX_CONSTANT_SNAPSHOT_CACHE_H
#include <stdint.h>
#include <string.h>
#define RSX_CONSTANT_SNAPSHOT_BYTES 8448u
typedef struct {int valid;uint32_t slot;unsigned char bytes[RSX_CONSTANT_SNAPSHOT_BYTES];} rsx_constant_snapshot_cache;
static uint32_t rsx_constant_snapshot_record(rsx_constant_snapshot_cache* c,const void* bytes,uint32_t slot,void* upload) {
 if(c->valid && !memcmp(c->bytes,bytes,RSX_CONSTANT_SNAPSHOT_BYTES)) return c->slot;
 memcpy((unsigned char*)upload+(uint64_t)slot*RSX_CONSTANT_SNAPSHOT_BYTES,bytes,RSX_CONSTANT_SNAPSHOT_BYTES);
 memcpy(c->bytes,bytes,RSX_CONSTANT_SNAPSHOT_BYTES);c->valid=1;c->slot=slot;return slot;
}
#endif
