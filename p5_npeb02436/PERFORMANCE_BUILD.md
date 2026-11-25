# Persona 5: opt-in host compiler optimization

The default Release configuration is unchanged. These options are independent
of P5_SPU_AVX2 and the generated-source directory. Record the exact compiler,
linker, source/lift hashes and configuration with benchmark results.

## IPO

Configure the usual P5 build with -DP5_ENABLE_IPO=ON. CMake checks support
for C and C++ before enabling Release IPO on the game and runtime targets.
Large generated units can increase link time and memory substantially.

## PGO

Use a dedicated Release build directory and keep generated sources and all
other flags unchanged between the GENERATE and USE configurations.
Training is not a performance benchmark. Cover representative gameplay,
menus, battles and streaming, and terminate normally to flush profiles.

### GNU-driver Clang

Add -DP5_PGO_MODE=GENERATE to the configure command, build Release, then set
LLVM_PROFILE_FILE to an absolute path such as P5-%p.profraw before running.
Merge the collected profiles with matching llvm-profdata, for example:

    llvm-profdata merge -output=P5.profdata P5-*.profraw

Reconfigure the same build with -DP5_PGO_MODE=USE
-DP5_PGO_DATA=/absolute/path/P5.profdata, then rebuild.

### Native MSVC

Add -DP5_PGO_MODE=GENERATE -DP5_PGO_DATA=C:/profiles/P5.pgd, build Release,
and train. Reconfigure with -DP5_PGO_MODE=USE and the same PGD. Preserve the
training .pgc files beside the database. The selected linker must support
/GENPROFILE and /USEPROFILE. The title's existing compiler requirements
(such as Clang musttail SPU sources) still apply; this is not a promise that
an otherwise unsupported native MSVC configuration will build.

### clang-cl and GCC

Explicit PGO setup for these driver/linker combinations is not provided.
Do not silently apply GNU-driver flags to link.exe. IPO may still be
available independently through CMake's capability check.

## Verify

Inspect actual compiler and linker commands; use matching profile-tool
versions and investigate stale/missing-profile warnings. Compare clean
unprofiled Release and profile-use builds with identical assets, scene,
runtime switches and frame pacing. Do not change guest FP semantics.
