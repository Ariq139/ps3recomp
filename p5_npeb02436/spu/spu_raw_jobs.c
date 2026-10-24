/* Release-build raw SPU jobs captured from this title and lifted separately
 * from its embedded SPU ELF. Keep registrations here instead of editing the
 * generated spu_workloads.c, which is owned by tools/build_spu_workloads.py. */
#include "spu_workload.h"

extern void spu_begin_image(int image_id);
extern void p5_rel_job_647773A4_spu_func_00000000(spu_context*);
extern void p5_rel_job_647773A4_spu_recomp_register(void);
extern void p5_rel_job_0567DCFF_spu_func_00000000(spu_context*);
extern void p5_rel_job_0567DCFF_spu_recomp_register(void);
extern void p5_rel_job_11C79FF8_spu_func_00000000(spu_context*);
extern void p5_rel_job_11C79FF8_spu_recomp_register(void);

void p5_npeb02436_spu_register_raw_jobs(void)
{
    /* These captured CRT requirements are attached to binary fingerprints,
     * so the generic job runner does not depend on generated image IDs. */
    static const spu_job_abi context2_job_abi = {
        SPU_JOB_ABI_CONTEXT2 | SPU_JOB_ABI_KERNEL_CONTEXT,
        20u, 0u, 0u, 0x3FF10u
    };
    static const spu_job_abi graphics_worker_abi = {
        SPU_JOB_ABI_CONTEXT2,
        0u, 0x3FB80u, 0x3FB84u, 0x3FF10u
    };

    /* The captured Port2 binary hashes with the runtime's legacy FNV-1a-style
     * seed to the fingerprint reported by the release boot's dispatch-miss
     * trace. Image id 150 avoids the base
     * image, SPURS policy, and job-code overlay id ranges used by this port. */
    spu_begin_image(150);
    p5_rel_job_647773A4_spu_recomp_register();
    spu_workload_register_img(0x647773A476FDD907ULL,
        p5_rel_job_647773A4_spu_func_00000000, 150,
        "spu_job_647773A476FDD907");
    spu_workload_set_job_abi(0x647773A476FDD907ULL, &context2_job_abi);
    /* Captured from the release's descriptor 0x30E00180, eaBinary=0xC01380,
     * sizeBinary=31232. Exact fingerprint; execute the lifted job body. */
    spu_begin_image(151);
    p5_rel_job_0567DCFF_spu_recomp_register();
    spu_workload_register_img(0x0567DCFFF6EE92D6ULL,
        p5_rel_job_0567DCFF_spu_func_00000000, 151,
        "spu_job_0567DCFFF6EE92D6");
    spu_workload_set_job_abi(0x0567DCFFF6EE92D6ULL, &context2_job_abi);
    /* Exact runtime capture from the deferred graphics worker's job. */
    spu_begin_image(152);
    p5_rel_job_11C79FF8_spu_recomp_register();
    spu_workload_register_img(0x11C79FF8C1948A26ULL,
        p5_rel_job_11C79FF8_spu_func_00000000, 152,
        "spu_job_11C79FF8C1948A26");
    spu_workload_set_job_abi(0x11C79FF8C1948A26ULL, &graphics_worker_abi);
    spu_begin_image(0);
}

__attribute__((constructor)) static void p5_npeb02436_spu_register_raw_jobs_ctor(void)
{
    p5_npeb02436_spu_register_raw_jobs();
}
