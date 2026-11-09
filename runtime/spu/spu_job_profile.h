#ifndef PS3RECOMP_SPU_JOB_PROFILE_H
#define PS3RECOMP_SPU_JOB_PROFILE_H
#include "spu_workload.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Optional per-dispatch wall time, accumulated with SPU_JOBPROF. A NULL sink
 * keeps the ordinary execution path free of phase clock reads. */
typedef struct spu_job_phase_times {
    double lookup, reset, stage, execute, teardown;
} spu_job_phase_times;
double spu_job_profile_now(void);
int spu_run_spurs_job_profiled(spu_lifted_entry_fn entry, int image_id,
                               uint32_t job_ea, uint32_t job_desc_size,
                               const spu_job_abi* job_abi,
                               const uint8_t* image_copy, const uint8_t* desc_copy,
                               spu_job_phase_times* phases);

#ifdef __cplusplus
}
#endif
#endif
