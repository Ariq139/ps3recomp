#ifndef PS3RECOMP_RSX_FP_METADATA_CACHE_H
#define PS3RECOMP_RSX_FP_METADATA_CACHE_H
#include "rsx_fp_decompiler.h"
#include <stdint.h>
#include <string.h>

#define RSX_FP_METADATA_BYTES 4096u
/* Every constant follows a 16-byte instruction, so at most 128 payloads
 * fit in the bounded metadata scan. */
#define RSX_FP_METADATA_OFFSETS (RSX_FP_METADATA_BYTES / 32u)
typedef struct {
    const u8* source;
    u32 bytes, hash, units;
    u8 snapshot[RSX_FP_METADATA_BYTES];
    u32 offsets[RSX_FP_METADATA_OFFSETS];
    rsx_fp_constant_block constants;
} rsx_fp_metadata;
typedef struct { rsx_fp_metadata entries[64]; } rsx_fp_metadata_cache;

static const rsx_fp_metadata* rsx_fp_metadata_get(rsx_fp_metadata_cache* cache, const u8* code)
{
    if (!code) return NULL;
    rsx_fp_metadata* e = &cache->entries[((uintptr_t)code >> 4) & 63u];
    if (e->bytes && e->source == code) {
        if (!memcmp(e->snapshot, code, e->bytes)) return e;

        /* Validate ALL instruction bytes, including END and CONST operand
         * encoding, using offsets from the same decoder as a cold miss.
         * Constant payload bytes alone may differ. */
        /* Validate and refresh from the same captured bytes, rather than
         * repeating a live instruction scan and risking inconsistent data. */
        u8 captured[RSX_FP_METADATA_BYTES];
        memcpy(captured, code, e->bytes);
        u32 start = 0, i;
        for (i = 0; i < e->constants.count; ++i) {
            const u32 off = e->offsets[i];
            if (memcmp(e->snapshot + start, captured + start, off - start)) break;
            start = off + 16u;
        }
        if (i == e->constants.count &&
            !memcmp(e->snapshot + start, captured + start, e->bytes - start)) {
            for (i = 0; i < e->constants.count; ++i)
                for (u32 component = 0; component < 4; ++component)
                    e->constants.values[i][component] = rsx_fp_sanitize_constant_word(
                        rsx_fp_read_word(captured + e->offsets[i] + component * 4u));
            memcpy(e->snapshot, captured, e->bytes);
            return e;
        }
    }
    const u32 bytes = rsx_fp_program_size(code, RSX_FP_METADATA_BYTES);
    u32 units = 0;
    if (!bytes || bytes > RSX_FP_METADATA_BYTES) return NULL;
    /* Capture only the validated extent; never blindly read 4096 guest bytes.
     * Derive the key, layout and constants from this same immutable copy. */
    u8 captured[RSX_FP_METADATA_BYTES];
    memcpy(captured, code, bytes);
    u32 captured_bytes = 0;
    const u32 hash = rsx_fp_code_hash_ex(captured, bytes, &captured_bytes, &units);
    if (captured_bytes != bytes) return NULL;
    u32 offsets[RSX_FP_METADATA_OFFSETS];
    rsx_fp_constant_block constants;
    if (rsx_fp_collect_constants_layout(captured, bytes, &constants,
            offsets, RSX_FP_METADATA_OFFSETS) < 0) return NULL;
    e->source = code; e->bytes = bytes; e->hash = hash; e->units = units;
    e->constants.count = constants.count;
    memcpy(e->constants.values, constants.values, constants.count * 16u);
    memcpy(e->offsets, offsets, constants.count * sizeof(offsets[0]));
    memcpy(e->snapshot, captured, bytes);
    return e;
}
#endif
