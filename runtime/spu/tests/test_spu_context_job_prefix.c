#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../spu_context.h"

static spu_context full_ctx;
static spu_context prefix_ctx;
static uint8_t image[SPU_LS_SIZE];

static int check_size(size_t image_size, int use_prefix)
{
    memset(&full_ctx, 0xA5, sizeof(full_ctx));
    memset(&prefix_ctx, 0x5A, sizeof(prefix_ctx));
    for (size_t i = 0; i < sizeof(image); ++i)
        image[i] = (uint8_t)(i * 37u + 11u);

    spu_context_init(&full_ctx, 3);
    if (use_prefix) {
        spu_context_init_job_prefix(&prefix_ctx, 3, image_size);
        memcpy(prefix_ctx.ls, image, image_size);
        memcpy(full_ctx.ls, image, image_size);
    } else {
        spu_context_init_job_prefix(&prefix_ctx, 3, image_size);
    }

    if (full_ctx.ls != full_ctx.ls_store ||
        prefix_ctx.ls != prefix_ctx.ls_store)
        return 0;

    /* The two contexts intentionally have different self-pointers. Compare
     * all remaining state after normalizing that per-object pointer field. */
    prefix_ctx.ls = full_ctx.ls;
    return memcmp(&full_ctx, &prefix_ctx, sizeof(full_ctx)) == 0;
}

int main(void)
{
    static const size_t sizes[] = {
        0, 1, 15, 16, 17, 0xA00, 0x10000,
        SPU_LS_SIZE - 1u, SPU_LS_SIZE, SPU_LS_SIZE + 1u, SIZE_MAX
    };

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        size_t size = sizes[i];
        int valid = size != 0 && size <= SPU_LS_SIZE;
        if (!check_size(size, valid)) {
            fprintf(stderr, "context prefix mismatch at size %zu\n", size);
            return 1;
        }
    }

    puts("SPU job-prefix initialization matches full initialization: PASS");
    return 0;
}
