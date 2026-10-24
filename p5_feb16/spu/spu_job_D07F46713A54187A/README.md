# Captured SPURS job D07F46713A54187A

- Image size: 20,400 bytes.
- Fingerprint: `D07F46713A54187A` (matches the runtime's `dispatch MISS`).
- Source: exact job image recovered from the game's guest memory after the
  runtime reported the fingerprint; the saved blob is
  `spujob_D07F46713A54187A_20400.bin`.
- Lift: 191 detected functions, covering 17,516 bytes (85.9%); all 10
  indirect targets found in the function-pointer table resolve to detected
  function entries. The generated C has no `UNIMPLEMENTED` or `TODO` markers.
- Runtime registration: `spu/jobmods_register.c`, image id `0x500C`, entry
  offset `0`.

Reproduce the analysis from the saved raw image:

```powershell
python tools/find_spu_functions.py `
  p5_feb16/spu/spu_job_D07F46713A54187A/spujob_D07F46713A54187A_20400.bin `
  --raw --base 0 `
  --out p5_feb16/spu/spu_job_D07F46713A54187A/functions.json

python tools/spu_lifter.py `
  p5_feb16/spu/spu_job_D07F46713A54187A/spujob_D07F46713A54187A_20400.bin `
  --base 0 `
  --functions p5_feb16/spu/spu_job_D07F46713A54187A/functions.json `
  --output p5_feb16/spu/spu_job_D07F46713A54187A `
  --symbol-prefix p5_job_D07F4671_
```

The fingerprint and byte length are verified. Function semantics and successful
runtime completion still require confirmation from a rebuilt game run.
