/* Runtime-captured SPU jobs registered by their full image fingerprints. */
#include "spu_workload.h"

extern void spu_begin_image(int image_id);
extern void p5_job_0485F313_spu_func_00000000(spu_context*);
extern void p5_job_0485F313_spu_recomp_register(void);
extern void p5_job_E643787D_spu_func_00000000(spu_context*);
extern void p5_job_E643787D_spu_recomp_register(void);
extern void p5_job_11C79FF8_spu_func_00000000(spu_context*);
extern void p5_job_11C79FF8_spu_recomp_register(void);
extern void p5_job_6AED6CC8_spu_func_00000000(spu_context*);
extern void p5_job_6AED6CC8_spu_recomp_register(void);
extern void p5_job_ED0D6BA5_spu_func_00000000(spu_context*);
extern void p5_job_ED0D6BA5_spu_recomp_register(void);
extern void p5_job_C2810FE1_spu_func_00000000(spu_context*);
extern void p5_job_C2810FE1_spu_recomp_register(void);
extern void p5_job_6E7DE3E6_spu_func_00000000(spu_context*);
extern void p5_job_6E7DE3E6_spu_recomp_register(void);

extern void p5_job_3459B086_spu_func_00000000(spu_context*);
extern void p5_job_3459B086_spu_recomp_register(void);
extern void p5_job_4ACD4F02_spu_func_00000000(spu_context*);
extern void p5_job_4ACD4F02_spu_recomp_register(void);
extern void p5_job_5779CE03_spu_func_00000000(spu_context*);
extern void p5_job_5779CE03_spu_recomp_register(void);
extern void p5_job_7A9C4F10_spu_func_00000000(spu_context*);
extern void p5_job_7A9C4F10_spu_recomp_register(void);
extern void p5_job_96B93CE5_spu_func_00000000(spu_context*);
extern void p5_job_96B93CE5_spu_recomp_register(void);
extern void p5_job_A96F7BFC_spu_func_00000000(spu_context*);
extern void p5_job_A96F7BFC_spu_recomp_register(void);
extern void p5_job_BA45BD5F_spu_func_00000000(spu_context*);
extern void p5_job_BA45BD5F_spu_recomp_register(void);
extern void p5_job_F39DC9F0_spu_func_00000000(spu_context*);
extern void p5_job_F39DC9F0_spu_recomp_register(void);
extern void p5_job_D07F4671_spu_func_00000000(spu_context*);
extern void p5_job_D07F4671_spu_recomp_register(void);

extern void p5_job_B60721C9_spu_func_00000000(spu_context*);
extern void p5_job_B60721C9_spu_recomp_register(void);

extern void p5_job_1CFDDEE5_spu_func_00000000(spu_context*);
extern void p5_job_1CFDDEE5_spu_recomp_register(void);

extern void p5_job_1E90A78B_spu_func_00000000(spu_context*);
extern void p5_job_1E90A78B_spu_recomp_register(void);
extern void p5_job_E19DFC7B_spu_func_00000000(spu_context*);
extern void p5_job_E19DFC7B_spu_recomp_register(void);
extern void p5_job_B391A122_spu_func_00000000(spu_context*);
extern void p5_job_B391A122_spu_recomp_register(void);

extern void p5_job_BBB7C0C1_spu_func_00000000(spu_context*);
extern void p5_job_BBB7C0C1_spu_recomp_register(void);

extern void p5_job_AF5FF4A5_spu_func_00000000(spu_context*);
extern void p5_job_AF5FF4A5_spu_recomp_register(void);

static void p5_feb16_register_jobs(void)
{
    /* Keep these ids outside the normal image (1..25), PM (100), and overlay
     * ranges used by this title. */
    spu_begin_image(101);
    p5_job_0485F313_spu_recomp_register();
    spu_workload_register_img(0x0485F313033E5FCFULL,
                              p5_job_0485F313_spu_func_00000000,
                              101, "CRI AsrVoice PS3SPU 1.00.00");

    spu_begin_image(102);
    p5_job_E643787D_spu_recomp_register();
    spu_workload_register_img(0xE643787D2822CB4DULL,
                              p5_job_E643787D_spu_func_00000000,
                              102, "CRI DspPack2 PS3SPU 1.00.00");

    /* Captured from the game's job binary after the runtime reported this exact
     * fingerprint as unregistered. It is lifted and executed; no completion is
     * fabricated for this workload. */
    spu_begin_image(103);
    p5_job_11C79FF8_spu_recomp_register();
    spu_workload_register_img(0x11C79FF8C1948A26ULL,
                              p5_job_11C79FF8_spu_func_00000000,
                              103, "runtime-captured SPURS job 11C79FF8");
    /* This fingerprint is the same deferred graphics worker registered as
     * image 152 in the Release profile. Keep its verified ABI keyed to the
     * binary fingerprint so this debug image gets the same manager service
     * table and worker-slot convention. */
    static const spu_job_abi graphics_worker_abi = {
        SPU_JOB_ABI_CONTEXT2,
        0u, 0x3FB80u, 0x3FB84u, 0x3FF10u
    };
    spu_workload_set_job_abi(0x11C79FF8C1948A26ULL, &graphics_worker_abi);

    /* These four full-image fingerprints are the exact missing job binaries
     * in the user's BATTLE -> encounter -> X capture. Their raw images were
     * saved by SPU_DUMP_MISS on a prior run and rechecked against the captured
     * FNV fingerprints and lengths before lifting. Keep their indirect-branch
     * ids in a separate range: the optional lifted libsre firmware already
     * owns ids 104 and 105 for its SPURS kernel and policy. */
    spu_begin_image(0x5000);
    p5_job_6AED6CC8_spu_recomp_register();
    spu_workload_register_img(0x6AED6CC8E6AE9733ULL,
                              p5_job_6AED6CC8_spu_func_00000000,
                              0x5000, "captured SPURS job 6AED6CC8");

    spu_begin_image(0x5001);
    p5_job_ED0D6BA5_spu_recomp_register();
    spu_workload_register_img(0xED0D6BA5A8868092ULL,
                              p5_job_ED0D6BA5_spu_func_00000000,
                              0x5001, "captured SPURS job ED0D6BA5");

    spu_begin_image(0x5002);
    p5_job_C2810FE1_spu_recomp_register();
    spu_workload_register_img(0xC2810FE1F32B0D47ULL,
                              p5_job_C2810FE1_spu_func_00000000,
                              0x5002, "captured SPURS job C2810FE1");

    spu_begin_image(0x5003);
    p5_job_6E7DE3E6_spu_recomp_register();
    spu_workload_register_img(0x6E7DE3E6FC4F280FULL,
                              p5_job_6E7DE3E6_spu_func_00000000,
                              0x5003, "captured SPURS job 6E7DE3E6");
    /* Captured SPURS job image 3459B0866D6A73E2; fingerprint and byte length verified. */
    spu_begin_image(0x5004);
    p5_job_3459B086_spu_recomp_register();
    spu_workload_register_img(0x3459B0866D6A73E2ULL,
                              p5_job_3459B086_spu_func_00000000,
                              0x5004, "captured SPURS job 3459B086");

    /* Captured SPURS job image 4ACD4F02ED2ADACC; fingerprint and byte length verified. */
    spu_begin_image(0x5005);
    p5_job_4ACD4F02_spu_recomp_register();
    spu_workload_register_img(0x4ACD4F02ED2ADACCULL,
                              p5_job_4ACD4F02_spu_func_00000000,
                              0x5005, "captured SPURS job 4ACD4F02");

    /* Captured SPURS job image 5779CE036C942995; fingerprint and byte length verified. */
    spu_begin_image(0x5006);
    p5_job_5779CE03_spu_recomp_register();
    spu_workload_register_img(0x5779CE036C942995ULL,
                              p5_job_5779CE03_spu_func_00000000,
                              0x5006, "captured SPURS job 5779CE03");

    /* Captured SPURS job image 7A9C4F10411D5CB9; fingerprint and byte length verified. */
    spu_begin_image(0x5007);
    p5_job_7A9C4F10_spu_recomp_register();
    spu_workload_register_img(0x7A9C4F10411D5CB9ULL,
                              p5_job_7A9C4F10_spu_func_00000000,
                              0x5007, "captured SPURS job 7A9C4F10");

    /* Captured SPURS job image 96B93CE571AC8650; fingerprint and byte length verified. */
    spu_begin_image(0x5008);
    p5_job_96B93CE5_spu_recomp_register();
    spu_workload_register_img(0x96B93CE571AC8650ULL,
                              p5_job_96B93CE5_spu_func_00000000,
                              0x5008, "captured SPURS job 96B93CE5");

    /* Captured SPURS job image A96F7BFCEB376D57; fingerprint and byte length verified. */
    spu_begin_image(0x5009);
    p5_job_A96F7BFC_spu_recomp_register();
    spu_workload_register_img(0xA96F7BFCEB376D57ULL,
                              p5_job_A96F7BFC_spu_func_00000000,
                              0x5009, "captured SPURS job A96F7BFC");

    /* Captured SPURS job image BA45BD5F75AED649; fingerprint and byte length verified. */
    spu_begin_image(0x500A);
    p5_job_BA45BD5F_spu_recomp_register();
    spu_workload_register_img(0xBA45BD5F75AED649ULL,
                              p5_job_BA45BD5F_spu_func_00000000,
                              0x500A, "captured SPURS job BA45BD5F");

    /* Captured SPURS job image F39DC9F0CB5CF41B; fingerprint and byte length verified. */
    spu_begin_image(0x500B);
    p5_job_F39DC9F0_spu_recomp_register();
    spu_workload_register_img(0xF39DC9F0CB5CF41BULL,
                              p5_job_F39DC9F0_spu_func_00000000,
                              0x500B, "captured SPURS job F39DC9F0");

    /* Recovered from the exact dispatch miss observed when battle rendering
     * stalled. The 20,400-byte image fingerprint matches the runtime capture. */
    spu_begin_image(0x500C);
    p5_job_D07F4671_spu_recomp_register();
    spu_workload_register_img(0xD07F46713A54187AULL,
                              p5_job_D07F4671_spu_func_00000000,
                              0x500C, "captured SPURS job D07F4671");

    /* Battle graphics dependency job captured from the Feb16 build. */
    spu_begin_image(0x500D);
    p5_job_B60721C9_spu_recomp_register();
    spu_workload_register_img(0xB60721C9C1AA64F2ULL,
                              p5_job_B60721C9_spu_func_00000000,
                              0x500D, "captured SPURS battle job B60721C9");

    spu_begin_image(0x500E);
    p5_job_1CFDDEE5_spu_recomp_register();
    spu_workload_register_img(0x1CFDDEE5FF398310ULL,
                              p5_job_1CFDDEE5_spu_func_00000000,
                              0x500E, "captured SPURS battle job 1CFDDEE5");

    spu_begin_image(0x500F);
    p5_job_1E90A78B_spu_recomp_register();
    spu_workload_register_img(0x1E90A78B7059F448ULL,
                              p5_job_1E90A78B_spu_func_00000000,
                              0x500F, "captured SPURS battle job 1E90A78B");

    /* Exact runtime captures, also byte-identical to the embedded boot image. */
    spu_begin_image(0x5010);
    p5_job_E19DFC7B_spu_recomp_register();
    spu_workload_register_img(0xE19DFC7B38EE621DULL,
                              p5_job_E19DFC7B_spu_func_00000000,
                              0x5010, "captured SPURS battle job E19DFC7B");
    spu_begin_image(0x5011);
    p5_job_B391A122_spu_recomp_register();
    spu_workload_register_img(0xB391A122C5C08F5DULL,
                              p5_job_B391A122_spu_func_00000000,
                              0x5011, "captured SPURS battle job B391A122");

    /* Exact 10,848-byte encounter job; entry branches to the Context2 CRT. */
    spu_begin_image(0x5012);
    p5_job_BBB7C0C1_spu_recomp_register();
    spu_workload_register_img(0xBBB7C0C1505EDC27ULL,
                              p5_job_BBB7C0C1_spu_func_00000000,
                              0x5012, "captured SPURS encounter job BBB7C0C1");

    /* Exact 20,704-byte encounter job; Context2 CRT and worker-slot table. */
    spu_begin_image(0x5013);
    p5_job_AF5FF4A5_spu_recomp_register();
    spu_workload_register_img(0xAF5FF4A5692E0412ULL,
                              p5_job_AF5FF4A5_spu_func_00000000,
                              0x5013, "captured SPURS encounter job AF5FF4A5");

    /* The runtime job runner is deliberately generic: it gets the ABI from
     * the registered image fingerprint, not from P5-specific image-id ranges.
     * These captured P5 jobs all use the Context2 CRT and SPURS kernel view.
     * Without this registration, their CRT sees a zero manager-interface
     * pointer and exits through STOPD; the queue then reports NOENT to the PPU.
     * Keep the two CRI audio jobs and all captured battle jobs on the same
     * verified ABI as the earlier image-id based setup. */
    static const spu_job_abi p5_context2_job_abi = {
        SPU_JOB_ABI_CONTEXT2 | SPU_JOB_ABI_KERNEL_CONTEXT,
        20u, 0u, 0u, 0x3FF10u
    };
    static const uint64_t p5_context2_job_fingerprints[] = {
        0x0485F313033E5FCFULL, 0xE643787D2822CB4DULL,
        0x6AED6CC8E6AE9733ULL, 0xED0D6BA5A8868092ULL,
        0xC2810FE1F32B0D47ULL, 0x6E7DE3E6FC4F280FULL,
        0x3459B0866D6A73E2ULL, 0x4ACD4F02ED2ADACCULL,
        0x5779CE036C942995ULL, 0x7A9C4F10411D5CB9ULL,
        0x96B93CE571AC8650ULL, 0xA96F7BFCEB376D57ULL,
        0xBA45BD5F75AED649ULL, 0xF39DC9F0CB5CF41BULL,
        0xD07F46713A54187AULL, 0xB60721C9C1AA64F2ULL,
        0x1CFDDEE5FF398310ULL, 0x1E90A78B7059F448ULL,
        0xE19DFC7B38EE621DULL, 0xB391A122C5C08F5DULL,
        0xBBB7C0C1505EDC27ULL, 0xAF5FF4A5692E0412ULL
    };
    for (size_t i = 0; i < sizeof(p5_context2_job_fingerprints) /
                            sizeof(p5_context2_job_fingerprints[0]); ++i)
        spu_workload_set_job_abi(p5_context2_job_fingerprints[i],
                                 &p5_context2_job_abi);

    /* These GFD jobs interpret context+0x24 as a worker slot. Each reads it,
     * multiplies by eight and indexes the verified tag-range table
     * {(3,9), (12,9), (21,10)}. This serial host runner uses slot zero.
     * Passing the SDK's ordinary DMA tag 20 indexes beyond that table:
     * ED0D6BA5 reads 0x55000000 at LS 0x3010 and halts at PC 0xFA4. */
    spu_workload_set_job_context_tag(0xED0D6BA5A8868092ULL, 0u);
    spu_workload_set_job_context_tag(0x6AED6CC8E6AE9733ULL, 0u);
    spu_workload_set_job_context_tag(0xC2810FE1F32B0D47ULL, 0u);
    spu_workload_set_job_context_tag(0x6E7DE3E6FC4F280FULL, 0u);
    spu_workload_set_job_context_tag(0x3459B0866D6A73E2ULL, 0u);
    spu_workload_set_job_context_tag(0x4ACD4F02ED2ADACCULL, 0u);
    spu_workload_set_job_context_tag(0x7A9C4F10411D5CB9ULL, 0u);
    spu_workload_set_job_context_tag(0xA96F7BFCEB376D57ULL, 0u);
    spu_workload_set_job_context_tag(0x1CFDDEE5FF398310ULL, 0u);
    spu_workload_set_job_context_tag(0x5779CE036C942995ULL, 0u);
    spu_workload_set_job_context_tag(0xB60721C9C1AA64F2ULL, 0u);
    spu_workload_set_job_context_tag(0xBA45BD5F75AED649ULL, 0u);
    spu_workload_set_job_context_tag(0xF39DC9F0CB5CF41BULL, 0u);
    spu_workload_set_job_context_tag(0x1E90A78B7059F448ULL, 0u);
    spu_workload_set_job_context_tag(0xE19DFC7B38EE621DULL, 0u);
    spu_workload_set_job_context_tag(0xB391A122C5C08F5DULL, 0u);
    spu_workload_set_job_context_tag(0xBBB7C0C1505EDC27ULL, 0u);
    spu_workload_set_job_context_tag(0xAF5FF4A5692E0412ULL, 0u);

    spu_begin_image(0);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void p5_feb16_cri_jobs_ctor(void)
{
    p5_feb16_register_jobs();
}
#endif
