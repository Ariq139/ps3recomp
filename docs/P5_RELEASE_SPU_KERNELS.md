# Release P5 SPU loop optimization

The release runner now exposes the same pure-loop outliner used by the February
build, but does not select a loop by default. The beta image is not an identity
anchor for a retail image. No release capture or runtime hotspot is shipped here.

1. Profile a representative release scene with `SPU_JOBPROF=5` and
   `SPU_JOBPHASE=1`; identify the actual expensive executed image/function using
   samples or a native profiler. Inspect the generated function and select a
   canonical pure loop accepted by `tools/outline_spu_loop.py`.
2. Record the exact lifted source SHA256. Configure the release target with
   `P5_RELEASE_SPU_LOOP_MODE=CAPTURE`, `P5_RELEASE_SPU_LOOP_SOURCE`,
   `P5_RELEASE_SPU_LOOP_SHA256`, `P5_RELEASE_SPU_LOOP_FUNCTION`, and the
   START, ENTRY and EXIT PC variables with the same prefix.
3. Run the title with `SPU_KERNEL_CAPTURE_SCOPE=outlined`,
   `SPU_KERNEL_CAPTURE_IMAGE=<runtime image ID>`,
   `SPU_KERNEL_CAPTURE_PC=<entry PC>`, and
   `SPU_KERNEL_CAPTURE_PREFIX=<new path prefix>`. Keep the real capture, scene,
   build SHA and exact compiler flags. Collect independent representative
   inputs; the first invocation capture is not universal coverage.
4. Configure NATIVE with the same source identity, function and PCs, the
   runtime `P5_RELEASE_SPU_LOOP_IMAGE`, and semicolon-separated
   `P5_RELEASE_SPU_LOOP_CAPTURES`. An optional
   `P5_RELEASE_SPU_REPLAY_COMPILER` selects a clang/cc executable. AVX2 replay
   is enabled when P5_SPU_AVX2 is enabled. The build checks the source SHA,
   verifies capture layout/image/logical PC and replays reference and exact
   guarded native source against every supplied capture before emitting the
   transformed build-cache file. Any failure stops the generation command.
5. Compare complete game behavior and frame-time distributions against OFF.
   The capture gate proves equivalence only for those captured inputs, not
   general correctness or performance. Keep the original fallback and default
   OFF; do not claim a release speedup until measured.

Use the existing replay tool's paired benchmark mode on a verified loop to
measure isolated kernel speed. The existing beta configuration is unchanged.
