cmake_minimum_required(VERSION 3.20)
project(P5Feb16Recomp
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "Persona 5 (Feb 26, 2016 prototype; XRD664 label, TEST00000 title ID)"
)

if(WIN32)
    enable_language(RC)
endif()

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer Ninja if available
if(NOT CMAKE_GENERATOR MATCHES "Ninja" AND NOT DEFINED ENV{CMAKE_GENERATOR})
    message(STATUS "Tip: use -G Ninja for faster builds")
endif()

# ---------------------------------------------------------------------------
# The two directories this project is built from
# ---------------------------------------------------------------------------
# PS3RECOMP_DIR is the toolkit checkout: include/, runtime/, libs/, tools/.
# RECOMP_DIR is where the lifter's output for THIS game lives. They are
# separate on purpose -- a port keeps its generated code in its own tree and
# points at whatever toolkit checkout it is building against:
#
#   cmake -B build -G Ninja \
#         -DPS3RECOMP_DIR=/path/to/ps3recomp \
#         -DRECOMP_DIR=/path/to/my_game/recompiled
#
if(NOT DEFINED PS3RECOMP_DIR)
    # Default: this project is still sitting inside ps3recomp/templates/project/
    get_filename_component(PS3RECOMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
endif()
get_filename_component(PS3RECOMP_DIR "${PS3RECOMP_DIR}" ABSOLUTE)

set(RECOMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/recompiled" CACHE PATH
    "Directory holding the lifter's output: ppu_recomp.h and the lifted sources")
get_filename_component(RECOMP_DIR "${RECOMP_DIR}" ABSOLUTE)

message(STATUS "ps3recomp SDK:   ${PS3RECOMP_DIR}")
message(STATUS "Lifted output:   ${RECOMP_DIR}")

if(NOT EXISTS "${RECOMP_DIR}/ppu_recomp.h")
    message(FATAL_ERROR
        "No ppu_recomp.h in ${RECOMP_DIR}.\n"
        "That header and the lifted sources beside it are what "
        "tools/ppu_lifter.py writes. Run the lifter, drop its output there, "
        "or point -DRECOMP_DIR= at wherever it already is.")
endif()

# The lifted translation units. A game's lift is hundreds of generated files,
# so they are globbed rather than listed; re-run CMake after a fresh lift.
file(GLOB RECOMP_SOURCES CONFIGURE_DEPENDS
    "${RECOMP_DIR}/*.c"
    "${RECOMP_DIR}/*.cpp"
)
if(NOT RECOMP_SOURCES)
    message(FATAL_ERROR
        "${RECOMP_DIR} has a ppu_recomp.h but no .c or .cpp beside it. The "
        "lifted functions and the function table live in those; without them "
        "there is nothing to run.")
endif()
list(LENGTH RECOMP_SOURCES RECOMP_SOURCE_COUNT)
message(STATUS "Lifted sources:  ${RECOMP_SOURCE_COUNT} file(s)")

# ---------------------------------------------------------------------------
# ps3recomp runtime library
# ---------------------------------------------------------------------------
# Built from source as a subdirectory rather than imported as a prebuilt
# archive. A static library carries none of its own link requirements, so an
# imported one arrives without SDL2, pthreads, the Metal frameworks or the
# Winsock/XInput imports -- the link then fails on every platform for a
# different reason. add_subdirectory hands all of that over with the target.
add_subdirectory("${PS3RECOMP_DIR}" "${CMAKE_BINARY_DIR}/ps3recomp_sdk"
                 EXCLUDE_FROM_ALL)

# This title's memory-map checks treat 0x30000000 as the main-memory base.
# Reserve the eight raw-SPU local-store windows (0x30000000-0x301FFFFF) and
# allocate sys_memory blocks immediately after them, within the title's range.
target_compile_definitions(ps3recomp_runtime PRIVATE
    PS3RECOMP_SYS_MEM_ALLOC_BASE=0x30200000u
    PS3RECOMP_SYS_MEM_ALLOC_END=0x40000000u
    PS3RECOMP_SYS_MEM_USER_TOTAL=0x0FE00000u
)

# ---------------------------------------------------------------------------
# The HLE NID table
# ---------------------------------------------------------------------------
# Maps each firmware import NID onto the handler in the runtime that answers
# it. A port may keep a hand-curated one beside its lift; otherwise generate
# the canonical every-module table, which is what the toolkit's own boot
# harness uses.
if(EXISTS "${RECOMP_DIR}/ppu_hle_nids.cpp")
    set(HLE_NID_TABLE "${RECOMP_DIR}/ppu_hle_nids.cpp")
    message(STATUS "HLE NID table:   ${HLE_NID_TABLE}")
else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(HLE_NID_TABLE "${CMAKE_CURRENT_BINARY_DIR}/ppu_hle_nids.cpp")
    # The generator scans library sources to discover exports and compute
    # their NIDs. Rebuild its output when an HLE source changes so adding a
    # handler cannot silently leave an older registration table in the game.
    file(GLOB_RECURSE PS3RECOMP_HLE_SOURCES CONFIGURE_DEPENDS
        "${PS3RECOMP_DIR}/libs/*.c")
    add_custom_command(
        OUTPUT "${HLE_NID_TABLE}"
        COMMAND "${Python3_EXECUTABLE}"
                "${PS3RECOMP_DIR}/tools/gen_hle_nids.py"
                --out "${HLE_NID_TABLE}" --all
        DEPENDS "${PS3RECOMP_DIR}/tools/gen_hle_nids.py"
                "${PS3RECOMP_DIR}/tools/nid_database.py"
                ${PS3RECOMP_HLE_SOURCES}
        COMMENT "Generating the HLE NID -> handler table")
    message(STATUS "HLE NID table:   generated (every module)")
endif()

# ---------------------------------------------------------------------------
# The game executable
# ---------------------------------------------------------------------------
# The PPU boot scaffold is compiled here rather than taken from the runtime
# library: every one of these files #includes the lifter's per-game
# ppu_recomp.h, so they can only be built against a particular lift.
set(PS3RECOMP_SCAFFOLD
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_loader.cpp"
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_hle.cpp"
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_fs.cpp"
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_sysprx.cpp"
)

# ---------------------------------------------------------------------------
# Lifted SPU images (optional)
# ---------------------------------------------------------------------------
# A title that runs SPU work gets its images lifted into <SPU_DIR>/<image>/ by
# tools/spu_lifter.py, with a registration TU generated by
# tools/build_spu_workloads.py. cellSpurs dispatches them by content
# fingerprint, so the generated table is keyed to the images THIS build ships:
# a table generated from another release registers signatures that can never
# match, which is worse than shipping none. Both the table and the job-module
# registry self-register from constructors and are therefore optional -- a tree
# that has not lifted its SPU images yet configures and links without them.
set(SPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spu" CACHE PATH
    "Directory holding the lifted SPU images (<SPU_DIR>/<image>/spu_recomp.c)")
option(PORT_ENABLE_SPU "Compile and register the title's lifted SPU images" ON)

set(SPU_SOURCES "")
if(PORT_ENABLE_SPU)
    file(GLOB SPU_LIFTED CONFIGURE_DEPENDS "${SPU_DIR}/*/spu_recomp.c")
    if(SPU_LIFTED)
        list(LENGTH SPU_LIFTED SPU_IMAGE_COUNT)
        message(STATUS "Lifted SPU:      ${SPU_IMAGE_COUNT} image(s)")
        # The musttail indirect-branch fast path. Excluded from the runtime
        # library on purpose (see the SDK root CMakeLists): it must be built
        # here, by clang, or the tail call degrades to a plain call and a guest
        # loop that branches indirectly walks the thread stack off its guard
        # page a few thousand iterations in.
        list(APPEND SPU_SOURCES ${SPU_LIFTED}
             "${PS3RECOMP_DIR}/runtime/spu/spu_dispatch_mt.c")
        # Generated beside the lift; each is present only once its generator
        # has been run, so each is included only if it is actually here.
        foreach(gen IN ITEMS spu_workloads.c spu_raw_jobs.c jobmods_register.c taskset_policy_bytes.c)
            if(EXISTS "${SPU_DIR}/${gen}")
                list(APPEND SPU_SOURCES "${SPU_DIR}/${gen}")
            else()
                message(STATUS "  (no ${gen} -- not registered)")
            endif()
        endforeach()
    else()
        message(STATUS "Lifted SPU:      none found in ${SPU_DIR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Lifted libsre PRX (optional)
# ---------------------------------------------------------------------------
# GFD's SPURS backend lives in the system libsre PRX. When the title has been
# lifted with tools/lift_prx.py + tools/ppu_lifter.py, compile that namespaced
# lift and its export table into the runner. The default location is beside the
# active build tree so the generated PRX artifacts stay out of source control.
option(PORT_ENABLE_LIBSRE_PRX "Load the lifted libsre SPURS firmware PRX" ON)
set(LIBSRE_PRX_DIR "${CMAKE_CURRENT_BINARY_DIR}/libsre-prx-lift" CACHE PATH
    "Directory containing the lifted libsre PRX artifacts")

set(LIBSRE_PRX_SOURCES "")
if(PORT_ENABLE_LIBSRE_PRX)
    file(GLOB LIBSRE_LIFTED CONFIGURE_DEPENDS
        "${LIBSRE_PRX_DIR}/ppu_recomp_*.cpp"
        "${LIBSRE_PRX_DIR}/ppu_recomp_*.c")
    if(LIBSRE_LIFTED AND EXISTS "${LIBSRE_PRX_DIR}/libsre_recomp.h"
       AND EXISTS "${LIBSRE_PRX_DIR}/libsre_exports.c"
       AND EXISTS "${LIBSRE_PRX_DIR}/libsre_image.bin")
        list(APPEND LIBSRE_PRX_SOURCES ${LIBSRE_LIFTED}
             "${LIBSRE_PRX_DIR}/libsre_exports.c")
        file(TO_CMAKE_PATH "${LIBSRE_PRX_DIR}/libsre_image.bin"
             P5_LIBSRE_IMAGE_PATH)
        message(STATUS "Lifted libsre:   ${LIBSRE_PRX_DIR}")
    else()
        message(STATUS "Lifted libsre:   not found; SPURS uses HLE")
        set(PORT_ENABLE_LIBSRE_PRX OFF)
    endif()
endif()

# SPU binaries embedded in libsre.prx are separate images from the PPU PRX.
# Compile their local lifts and fingerprint registry so group startup selects
# the correct kernel/policy code when LS entry addresses overlap other images.
set(LIBSRE_SPU_LIFT_DIR "${CMAKE_CURRENT_BINARY_DIR}/libsre-spu-lift" CACHE PATH
    "Directory containing lifted SPU images extracted from libsre.prx")
set(LIBSRE_SPU_SOURCES "")
if(PORT_ENABLE_LIBSRE_PRX AND EXISTS "${LIBSRE_SPU_LIFT_DIR}/kernel/spu_recomp.c"
   AND EXISTS "${LIBSRE_SPU_LIFT_DIR}/policy/spu_recomp.c"
   AND EXISTS "${LIBSRE_SPU_LIFT_DIR}/libsre_spu_register.c")
    list(APPEND LIBSRE_SPU_SOURCES
        "${LIBSRE_SPU_LIFT_DIR}/kernel/spu_recomp.c"
        "${LIBSRE_SPU_LIFT_DIR}/policy/spu_recomp.c"
        "${LIBSRE_SPU_LIFT_DIR}/libsre_spu_register.c")
    message(STATUS "libsre SPU lifts: ${LIBSRE_SPU_LIFT_DIR}")
elseif(PORT_ENABLE_LIBSRE_PRX)
    message(STATUS "libsre SPU lifts: not found; exact image startup is unavailable")
endif()

set(JUNE_EVENT_EDITOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/event_editor_june")
set(JUNE_EVENT_EDITOR_IMAGE "${JUNE_EVENT_EDITOR_DIR}/june_image.bin")
if(NOT EXISTS "${JUNE_EVENT_EDITOR_IMAGE}" OR
   NOT EXISTS "${JUNE_EVENT_EDITOR_DIR}/generated/june_dispatch.cpp" OR
   NOT EXISTS "${JUNE_EVENT_EDITOR_DIR}/generated/june_image_info.h")
    message(FATAL_ERROR "The June event-editor image and generated dispatch files are required")
endif()
file(GLOB JUNE_EVENT_EDITOR_LIFTED CONFIGURE_DEPENDS
    "${JUNE_EVENT_EDITOR_DIR}/lift/ppu_recomp.cpp_*.cpp")
set(JUNE_EVENT_EDITOR_SOURCES
    "${JUNE_EVENT_EDITOR_DIR}/event_editor_june.cpp"
    "${JUNE_EVENT_EDITOR_DIR}/generated/june_dispatch.cpp"
    ${JUNE_EVENT_EDITOR_LIFTED})
if(WIN32)
    file(TO_NATIVE_PATH "${JUNE_EVENT_EDITOR_IMAGE}" JUNE_EVENT_EDITOR_IMAGE_NATIVE)
    string(REPLACE "\\" "\\\\" JUNE_EVENT_EDITOR_IMAGE_RC_PATH
        "${JUNE_EVENT_EDITOR_IMAGE_NATIVE}")
    set(JUNE_EVENT_EDITOR_RC "${CMAKE_CURRENT_BINARY_DIR}/event_editor_june.rc")
    configure_file("${JUNE_EVENT_EDITOR_DIR}/event_editor_june.rc.in"
        "${JUNE_EVENT_EDITOR_RC}" @ONLY)
    set_source_files_properties("${JUNE_EVENT_EDITOR_RC}" PROPERTIES
        OBJECT_DEPENDS "${JUNE_EVENT_EDITOR_IMAGE}")
    list(APPEND JUNE_EVENT_EDITOR_SOURCES "${JUNE_EVENT_EDITOR_RC}")
else()
    file(TO_CMAKE_PATH "${JUNE_EVENT_EDITOR_IMAGE}" JUNE_EVENT_EDITOR_IMAGE_PATH)
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    stubs.cpp
    ${PS3RECOMP_SCAFFOLD}
    ${HLE_NID_TABLE}
    ${RECOMP_SOURCES}
    ${SPU_SOURCES}
    ${LIBSRE_PRX_SOURCES}
    ${LIBSRE_SPU_SOURCES}
    ${JUNE_EVENT_EDITOR_SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "xrd664")

if(WIN32)
    get_property(_sdl2_runtime GLOBAL PROPERTY PS3RECOMP_SDL2_RUNTIME)
    if(_sdl2_runtime AND EXISTS "${_sdl2_runtime}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_sdl2_runtime}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endif()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/xrd664.ico"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/ps3recomp.ico"
        COMMENT "Staging XRD664 window icon")
endif()

include("${PS3RECOMP_DIR}/cmake/PS3RecompBootImage.cmake")
ps3recomp_configure_boot_image(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}"
    CANDIDATES
        "${CMAKE_CURRENT_SOURCE_DIR}/../EBOOT_feb16.elf"
        "${CMAKE_CURRENT_SOURCE_DIR}/game/EBOOT.ELF")

# The lifted SPU sources use bare #include "spu_context.h". Scope that include
# directory (and their flags) to those TUs alone: adding it to the whole target
# changes the compile command of every lifted PPU TU, and those are large enough
# that the needless rebuild is measured in tens of minutes.
#   SPU_USE_DISPATCH_MT -- route SPU_IB_DISPATCH through the musttail path.
#   -O2                 -- the target-wide -O1 exists for the huge PPU TUs; the
#                          SPU sources are small and decode-hot, so they get
#                          full optimization. The last -O on the line wins.
if(SPU_SOURCES)
    set_source_files_properties(${SPU_SOURCES} PROPERTIES
        INCLUDE_DIRECTORIES "${PS3RECOMP_DIR}/runtime/spu"
        COMPILE_DEFINITIONS SPU_USE_DISPATCH_MT)
    if(NOT MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(${SPU_SOURCES} PROPERTIES COMPILE_OPTIONS "-O2")
    endif()
endif()

if(PORT_ENABLE_LIBSRE_PRX)
    if(MSVC)
        set_source_files_properties(${LIBSRE_PRX_SOURCES} PROPERTIES
            COMPILE_OPTIONS "/w;/bigobj")
    else()
        set_source_files_properties(${LIBSRE_PRX_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-w;-ffp-contract=off")
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        P5_HAS_LIBSRE_PRX=1
        "P5_LIBSRE_IMAGE_PATH=\"${P5_LIBSRE_IMAGE_PATH}\"")
endif()

if(LIBSRE_SPU_SOURCES)
    set_source_files_properties(${LIBSRE_SPU_SOURCES} PROPERTIES
        INCLUDE_DIRECTORIES "${PS3RECOMP_DIR}/runtime/spu"
        COMPILE_DEFINITIONS SPU_USE_DISPATCH_MT)
    if(MSVC)
        set_source_files_properties(${LIBSRE_SPU_SOURCES} PROPERTIES
            COMPILE_OPTIONS "/w;/bigobj")
    else()
        set_source_files_properties(${LIBSRE_SPU_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-O2;-w")
    endif()
endif()

# The helper header already has exact AVX2 arithmetic/shuffle paths, but the
# lifted files do not inherit the SDK runtime's private SIMD flags. Keep the
# portable default and allow a local AVX2 build without rebuilding PPU units.
option(P5_SPU_AVX2 "Use AVX2 helpers in lifted SPU code (requires an AVX2 host)" OFF)
if(P5_SPU_AVX2)
    if(MSVC)
        set(P5_SPU_VECTOR_FLAGS /arch:AVX2)
    else()
        set(P5_SPU_VECTOR_FLAGS -mavx2)
    endif()
    foreach(P5_VECTOR_SOURCE IN LISTS SPU_SOURCES LIBSRE_SPU_SOURCES)
        set_property(SOURCE ${P5_VECTOR_SOURCE} APPEND PROPERTY
            COMPILE_OPTIONS ${P5_SPU_VECTOR_FLAGS})
    endforeach()
endif()

# Runtime opt-in caches are enabled for this P5 runner by default. Users can
# override any setting by exporting its environment variable, including "0".
option(P5_ENABLE_HOST_OPTIMIZATION_DEFAULTS
    "Enable reviewed SPURS/RSX cache paths by default in the P5 runner" ON)
if(P5_ENABLE_HOST_OPTIMIZATION_DEFAULTS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        PS3RECOMP_P5_HOST_OPTIMIZATION_DEFAULTS=1)
endif()

# This image/function has been replayed against a real input/output capture.
# Only the build-cache copy is transformed; OFF compiles the original lift.
option(P5_SPU_NATIVE_CULL "Use the verified guarded SPU culling kernel" ON)
if(PORT_ENABLE_SPU AND P5_SPU_NATIVE_CULL)
    set(P5_NATIVE_CULL_SOURCE "${SPU_DIR}/spu_job_A96F7BFCEB376D57/spu_recomp.c")
    if(EXISTS "${P5_NATIVE_CULL_SOURCE}")
        include("${PS3RECOMP_DIR}/cmake/PS3RecompSPUKernel.cmake")
        ps3recomp_promote_spu_kernel(${PROJECT_NAME} "${P5_NATIVE_CULL_SOURCE}"
            p5_job_A96F7BFC_spu_func_000016A0 P5_NATIVE_CULL_TRANSFORMED)
    endif()
endif()

# The first two pure geometry loops in the main SPU function can be tested
# independently. CAPTURE preserves reference execution; NATIVE uses the guarded
# register-promoted helper after its real captured state has passed replay.
set(P5_SPU_MAIN40_LOOP_MODE "OFF" CACHE STRING "Main SPU geometry loop: OFF, CAPTURE, NATIVE")
set_property(CACHE P5_SPU_MAIN40_LOOP_MODE PROPERTY STRINGS OFF CAPTURE NATIVE)
set(P5_SPU_MAIN40_LOOP2_MODE "OFF" CACHE STRING "Second main SPU geometry loop: OFF, CAPTURE, NATIVE")
set_property(CACHE P5_SPU_MAIN40_LOOP2_MODE PROPERTY STRINGS OFF CAPTURE NATIVE)
if(PORT_ENABLE_SPU AND (NOT P5_SPU_MAIN40_LOOP_MODE STREQUAL "OFF" OR
                        NOT P5_SPU_MAIN40_LOOP2_MODE STREQUAL "OFF"))
    set(P5_MAIN40_INPUT "${SPU_DIR}/spu_job_A96F7BFCEB376D57/spu_recomp.c")
    if(P5_NATIVE_CULL_TRANSFORMED)
        set(P5_MAIN40_INPUT "${P5_NATIVE_CULL_TRANSFORMED}")
    endif()
    include("${PS3RECOMP_DIR}/cmake/PS3RecompSPUKernel.cmake")
    if(NOT P5_SPU_MAIN40_LOOP_MODE STREQUAL "OFF")
        ps3recomp_outline_spu_loop(${PROJECT_NAME} "${P5_MAIN40_INPUT}"
            p5_job_A96F7BFC_spu_func_00000040 0x654 0x658 0x1324
            "${P5_SPU_MAIN40_LOOP_MODE}" P5_MAIN40_LOOP1_OUTLINED)
        set(P5_MAIN40_INPUT "${P5_MAIN40_LOOP1_OUTLINED}")
    endif()
    if(NOT P5_SPU_MAIN40_LOOP2_MODE STREQUAL "OFF")
        ps3recomp_outline_spu_loop(${PROJECT_NAME} "${P5_MAIN40_INPUT}"
            p5_job_A96F7BFC_spu_func_00000040 0x2DFC 0x2E00 0x3AD0
            "${P5_SPU_MAIN40_LOOP2_MODE}")
    endif()
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "${RECOMP_DIR}"                            # the generated ppu_recomp.h
    "${JUNE_EVENT_EDITOR_DIR}"
    "${PS3RECOMP_DIR}/include"
    "${PS3RECOMP_DIR}/runtime/ppu"
    "${PS3RECOMP_DIR}/runtime/spu"
    "${PS3RECOMP_DIR}/runtime/memory"
    "${PS3RECOMP_DIR}/runtime/prx"
    "${PS3RECOMP_DIR}/runtime/syscalls"
    "${PS3RECOMP_DIR}/runtime/platform"        # win32_compat.h and friends
    "${PS3RECOMP_DIR}/libs/video"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NOT WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "PS3RECOMP_JUNE_EVENT_EDITOR_IMAGE_PATH=\"${JUNE_EVENT_EDITOR_IMAGE_PATH}\"")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ps3recomp_runtime
)

# The runtime is C++, and on Apple also Objective-C. Letting CMake infer the
# link language from the generated C would leave the C++ standard library out.
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Windows system import libraries. dbghelp symbolizes crash-handler addresses
# against the PDB, user32 and gdi32 are what the backend's window and its GDI
# present are made of, and winmm is timeBeginPeriod. None of the four names
# anything a POSIX linker can find; the D3D12 backend's own imports come in
# through the #pragma comment(lib) in libs/video/rsx_d3d12_backend.c.
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        dbghelp
        user32
        gdi32
        winmm
    )
endif()

# ---------------------------------------------------------------------------
# Compiler-specific flags
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wno-unused-parameter
        -fno-strict-aliasing
    )
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W3 /wd4100 /wd4244 /wd4267
        /Zc:__cplusplus         # report correct __cplusplus value
        # ppu_recomp.h defines PPU_FENCE, the lowering of the guest's
        # sync/lwsync/eieio/isync, and in C that is <stdatomic.h>. The lifter
        # writes C++ by default, but --single-file writes ppu_recomp.c, and a
        # port that took it would not compile without this. The runtime's own
        # build already carries the same flag for the same reason.
        /experimental:c11atomics
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# The lifted translation units are machine-generated and can run to hundreds of
# thousands of lines each, so they get their own flags:
#
#   - warnings off. They are about generated code and nobody will act on them,
#     and at this volume they bury the diagnostics that matter.
#   - no floating-point contraction. arm64 fuses a multiply and an add by
#     default; a PPC fmadd is spelt as one where the lifter means one.
#   - /bigobj on MSVC. A single lifted file routinely passes the 65,536-section
#     limit of the default object format.
if(MSVC)
    set_source_files_properties(${RECOMP_SOURCES} PROPERTIES
        COMPILE_OPTIONS "/w;/bigobj")
    set_source_files_properties(${JUNE_EVENT_EDITOR_LIFTED} PROPERTIES
        COMPILE_OPTIONS "/w;/bigobj")
else()
    set_source_files_properties(${RECOMP_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-w;-ffp-contract=off")
    set_source_files_properties(${JUNE_EVENT_EDITOR_LIFTED} PROPERTIES
        COMPILE_OPTIONS "-w;-ffp-contract=off")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
