#ifndef PS3RECOMP_RSX_FP_METADATA_CACHE_H
#define PS3RECOMP_RSX_FP_METADATA_CACHE_H
#include "rsx_fp_decompiler.h"
#include <stdint.h>
#include <string.h>
typedef struct {
 const u8* source;
 u32 bytes, hash, units;
 u8 snapshot[4096];
 rsx_fp_constant_block constants;
} rsx_fp_metadata;
typedef struct { rsx_fp_metadata entries[64]; } rsx_fp_metadata_cache;
static const rsx_fp_metadata* rsx_fp_metadata_get(rsx_fp_metadata_cache* cache, const u8* code)
{
 if (!code) return NULL;
 rsx_fp_metadata* e=&cache->entries[((uintptr_t)code>>4)&63u];
 if (e->bytes && e->source==code && !memcmp(e->snapshot,code,e->bytes)) return e;
 u32 bytes=0,units=0;
 const u32 hash=rsx_fp_code_hash_ex(code,4096,&bytes,&units);
 if (!bytes || bytes>4096) return NULL;
 rsx_fp_constant_block constants;
 if (rsx_fp_collect_constants(code,bytes,&constants)<0) return NULL;
 e->source=code;e->bytes=bytes;e->hash=hash;e->units=units;e->constants=constants;
 memcpy(e->snapshot,code,bytes);
 return e;
}
#endif
