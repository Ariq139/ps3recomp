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

### clang-cl (recommended Windows toolchain)

Use the existing clang-cl build and its MSVC-compatible linker. Add
-DP5_PGO_MODE=GENERATE at configure time, build Release, and collect raw
profiles (PowerShell):

    $env:LLVM_PROFILE_FILE = "C:/profiles/P5-%p.profraw"

Run the instrumented executable on representative gameplay. Use a normal
exit so the profile runtime can flush its counters. Merge profiles with
llvm-profdata from the same LLVM installation, for example:

    llvm-profdata merge -output=C:/profiles/P5.profdata C:/profiles

Reconfigure the same build with -DP5_PGO_MODE=USE and
-DP5_PGO_DATA=C:/profiles/P5.profdata, then rebuild. Use a directory containing
only the raw profiles from this training build. Do not mix training runs
from different revisions or LLVM versions.

CMake queries the compiler resource directory and adds its lib/windows
directory to the link search path for the training executable. Clang adds
the appropriate compiler-rt profile default-library directive to its COFF
objects; do not pass GNU driver-only flags directly to link.exe or lld-link.
If the matching profile runtime is installed elsewhere, explicitly set
-DP5_CLANG_PROFILE_LIB_DIR=C:/path/to/lib/windows. The runtime library must
match the compiler version and target architecture.

Reference: https://clang.llvm.org/docs/UsersManual.html#finding-clang-runtime-libraries

### GNU-driver Clang

Add -DP5_PGO_MODE=GENERATE to the configure command, build Release, then set
LLVM_PROFILE_FILE to an absolute path such as P5-%p.profraw before running.
Merge the collected profiles with matching llvm-profdata, for example:

    llvm-profdata merge -output=P5.profdata P5-*.profraw

Reconfigure the same build with -DP5_PGO_MODE=USE
-DP5_PGO_DATA=/absolute/path/P5.profdata, then rebuild.

### Native MSVC and GCC

Native cl.exe is not currently supported by the runtime/lifted code.
GCC PGO is not configured. IPO may be available independently through
CMake's capability check.

## Verify

Inspect actual compiler and linker commands; use matching profile-tool
versions and investigate stale/missing-profile warnings. Compare clean
unprofiled Release and profile-use builds with identical assets, scene,
runtime switches and frame pacing. Do not change guest FP semantics.
