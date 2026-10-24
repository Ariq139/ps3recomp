# Runtime-captured SPU job

- Runtime fingerprint: `0x11C79FF8C1948A26` (55,824 bytes)
- Capture source: this game's `SPU_DUMP_MISS` diagnostic; not RPCS3 code.
- The raw image and `functions.json` are kept here to reproduce the lift.
- `find_spu_functions.py --raw` found 854 boundaries and classified the remaining
  8.7% of bytes as embedded data/padding. The lifter emitted no unsupported-op
  TODOs; runtime execution is still the check for semantic correctness.

Regenerate with:

```powershell
python tools/find_spu_functions.py p5_feb16/spu/spu_job_11C79FF8C1948A26/spujob_11C79FF8C1948A26_55824.bin --raw --out p5_feb16/spu/spu_job_11C79FF8C1948A26/functions.json
python tools/spu_lifter.py p5_feb16/spu/spu_job_11C79FF8C1948A26/spujob_11C79FF8C1948A26_55824.bin --base 0 --functions p5_feb16/spu/spu_job_11C79FF8C1948A26/functions.json --output p5_feb16/spu/spu_job_11C79FF8C1948A26 --symbol-prefix p5_job_11C79FF8_
((Get-Content p5_feb16/spu/spu_job_11C79FF8C1948A26/spu_recomp.c -Raw).Replace('../../runtime/spu/spu_helpers.h', 'spu_helpers.h')) | Set-Content p5_feb16/spu/spu_job_11C79FF8C1948A26/spu_recomp.c -NoNewline
((Get-Content p5_feb16/spu/spu_job_11C79FF8C1948A26/spu_recomp.h -Raw).Replace('../../runtime/spu/spu_context.h', 'spu_context.h')) | Set-Content p5_feb16/spu/spu_job_11C79FF8C1948A26/spu_recomp.h -NoNewline
```
