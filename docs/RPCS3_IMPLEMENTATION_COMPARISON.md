# Local RPCS3 implementation comparison

Checked 2026-10-15 against the local RPCS3 checkout at
`D:\Other\GitRepo\rpcs3`, commit `2ba2f75bc`. The comparison uses local source
as a behavior reference. It does not establish code provenance; see
[`RPCS3_LICENSE_AUDIT.md`](RPCS3_LICENSE_AUDIT.md) for the separate license
review.

Graphics behavior is expressed through ps3recomp's RSX layer and Windows
D3D12 backend; audio behavior uses its existing audio interfaces. RPCS3's host
graphics/audio backends are references, not runtime dependencies.

## Subsystem findings

| Area | Local RPCS3 behavior | ps3recomp state and remaining gap |
|---|---|---|
| PPU | `PPUInterpreter.cpp` contains interpreter semantics for the VMX stores `STVLX`, `STVRX`, and their cache-hint forms. RPCS3 also has PPU translation/JIT paths. | `tools/ppu_lifter.py` lowers `stvlx/stvlxl` and `stvrx/stvrxl` at lines 2275–2289. Their byte ranges match the interpreter formulas. A past `UNIMPLEMENTED` line can therefore come from an older executable or generated project; this turn did not regenerate or run the target. Remaining `.word` output needs instruction/data and reachability analysis before being called a missing instruction. |
| SPU | RPCS3 has an SPU interpreter, LLVM/ASMJIT recompiler paths, SPU thread/group execution, and SPURS code. The local `cellSpursJq.cpp` Port2 exports are explicitly unimplemented and return `CELL_OK`; that behavior must not be mistaken for general SPU job execution. | ps3recomp has an SPU interpreter and static lifts, but only registered image/job fingerprints dispatch to lifted code. The release run's saved miss dump is 32,064 bytes; the runtime's legacy FNV-1a-style hash (seed `1469598103934665603`) is `0x647773A476FDD907`. The corresponding lift's entry and registration function exist. Added a separate `spu_raw_jobs.c` registration for that fingerprint and included it in the project CMake source list. Added RPCS3-compatible taskset-ID and owning-SPURS getters over the local big-endian taskset header. The saved miss predates this source change; execution of the lift and its completion behavior remain unverified. |
| RSX | RPCS3's `rsx_methods.cpp` and `RSXThread.cpp` maintain a much broader RSX method/state and draw pipeline. `VertexProgramDecompiler.cpp` implements NV40 boolean branch/call flow using the branch selector and polarity fields in D3. `NV0039_BUFFER_NOTIFY` performs stride-aware rectangular memory copies; `NV3089_IMAGE_IN` triggers the scaled-image blit. | ps3recomp implements VP `BRB`/`CLB` flow and uploads `NV4097_SET_TRANSFORM_BRANCH_BITS` in the shared D3D12, Vulkan, Metal, live-draw, and engine VP constant-buffer paths. `NV4097_SET_RENDER_ENABLE` tracks conditional draws, and D3D12 maps `NV4097_SET_ZMIN_MAX_CONTROL` into depth clipping/clamping. Added stride-aware NV0039 copies with overlap snapshots and guest-memory bounds checks; existing numeric NV3089 dispatch is now named in the method source. NV0039 binding currently recognizes RPCS3's standard default SET_OBJECT handle; arbitrary guest object handles remain unclassified. ZPASS report production remains approximate because this backend has no occlusion-query producer; other method and draw gaps remain. Unknown methods still require command-stream evidence. |
| LV2 / events | RPCS3 binds a broad syscall table and implements event queue, port, event-flag cancellation, and SPU group/thread behavior in its LV2 code. | ps3recomp now cancels event-flag waiters with `CELL_ECANCELED`, reports their count, and preserves the flag pattern. The syscall inventory finds 284 unregistered named slots. PPU raw thread create/start (52/53) remain absent; the higher-level `sysPrxForUser` HLE create export has a different ABI. See [`UNIMPLEMENTED_AUDIT.md`](UNIMPLEMENTED_AUDIT.md). |
| HLE / NIDs / PRX | RPCS3 provides many module handlers, while individual modules such as `cellSpursJq` and `cellSail` also contain explicit success-returning placeholders. Its `sys_prx_register_library` associates matching 0x1c-byte export descriptors with an initialized PRX. | The two P5 manifests have 339 debug and 340 release imports. The remaining SAIL/GCM compatibility handlers follow RPCS3's own TODO/success behavior; local PRX loading now retains library descriptor associations for loaded `prx_module` instances. It does not yet model arbitrary dynamically loaded PRXs or full unload/relink semantics. SAIL media playback is still absent. Eight `cellSysmodule` internal/extended management exports were added; firmware image fetch and arbitrary module-file loading remain absent. See [`UNIMPLEMENTED_AUDIT.md`](UNIMPLEMENTED_AUDIT.md). |
| Additional HLE exports | RPCS3 `cellVoice.cpp` registers 35 public functions covering init/end, event queues, ports, service state, attributes, bit rate, mute, volume, and port I/O. | `libs/audio/cellVoice.c` now exports that 35-function surface. State, endian-safe ABI access, port validation, and service notification events are implemented. Device capture/codec/transport are unavailable; see the module reference. This expands the exported surface beyond either P5 import manifest. |
| `cellAudio` | RPCS3 has stereo/six-channel input adapters, typed PPU/SPU notification queues, flagged queue subscriptions, and a 13 ms port-level transition. | Added the 2/6-channel adapters to the existing guest ring, typed Ex queue creation, flagged Ex queue subscriptions/removal, decimation/no-mix event handling, source IDs and ack timestamps, port initialization level, and smoothed `cellAudioSetPortLevel`; corrected CellAudio error values. Accessory volume follows RPCS3's initialized no-op. Personal-device set/unset now validate initialization and stream/device ranges as RPCS3 does before its routing TODO; actual device routing remains absent in both. Float arguments use context bridges for PPU FPR1. `BEFOREMIX` notifications carry the ack source, but the local mixer does not implement RPCS3's timing/ack wait. No RPCS3 function body was copied. |
| `cellNetCtl` game/update exports | RPCS3's eight game-internal NetCtl functions and update init/term/abort are TODO-success handlers. Update-check start/finish functions queue `NO_UPDATE` or `FINISHED` callbacks through the sysutil pump; Ex callbacks receive a temporary 16-byte result structure. | Added all 17 names. The 11 TODO-only calls preserve success/no-output behavior. The six callback APIs queue the matching result on the next sysutil callback poll; Ex result memory is allocated in guest memory for the callback duration and freed after return. This uses ps3recomp's local queue/allocator; no RPCS3 function body was copied. |
| `cellBGDLUtility` | RPCS3 registers four `cellBGDL*` functions; all currently return `CELL_OK` without touching arguments or outputs. | Added those exact exports and ABI types, preserving untouched-output behavior. The pre-existing lowercase local download helpers remain for compatibility, but are not RPCS3 exports and do not model the `cellBGDLUtility` ABI. |
| `cellResc` | RPCS3 registers RESC helpers for color-buffer counts, surface conversion, resolution IDs, register count, wait-flip, and interlace-table validation. Some are TODO-success handlers; several existing RPCS3 functions also have ABI signatures that differed from the local declarations. | Added the missing exports, corrected the affected ABIs, copied RPCS3's documented validation/return behavior, and corrected RESC error values. The local RESC flip path and flip-time counter remain because they provide behavior beyond RPCS3's current TODOs. No RPCS3 function body was copied. |
| `cellPamf` | RPCS3 validates packed PAMF headers, initializes its 128-byte guest reader structure, decodes stream metadata, and exposes 24 header, stream, and EP APIs. | Added an independent packed-header/EP parser, guest-memory reader ABI, stream metadata and iterator APIs, and all 24 currently registered RPCS3 names. Initialization now sets the current grouping-period index and bounds stream descriptors by the group size. No RPCS3 function body was copied; behavior still needs runtime validation with PAMF fixtures. |
| `cellRtc` | RPCS3 registers tick/date conversion, clock and timezone APIs, validation, RFC formatting/parsing, DOS/FILETIME/system-time conversions, and tick arithmetic. `cellRtcSetCurrentSecureTick` delegates to a success-only secure-RTC handler. | Added the 20 missing registered APIs, corrected `GetTime_t`/`SetTime_t` to take `CellRtcDateTime`, and aligned existing errors, date conversion, and format output. Clock setting uses a process-local wall-clock offset; timezone configuration is RTC-local. Gregorian conversion supports the CellRtc year-1 epoch without host `time_t` limits. No RPCS3 function body was copied. Export scan now finds zero `cellRtc` name gaps; runtime date/parser behavior has not been exercised. |
| USB HLE exports | RPCS3 registers 28 `cellUsbd` exports; 21 are explicit `UNIMPLEMENTED_FUNC` bodies returning `CELL_OK`. | Added those 21 names to the NID surface with the same success-only placeholder behavior, and corrected local public error values for no-device/pipe-not-found paths. USB device enumeration, pipe operations, and memory services remain unavailable. |
| Additional one-call APIs | RPCS3 implements `sceNpManagerIsSP` as a true-valued mode query and queues video-upload lifecycle callbacks. Its `cellCrossControllerInitialize` validates bounded guest strings, opens a sysutil dialog, queues INITIALIZED, and queues FINALIZED/CANCEL when canceled. The PS Vita connection loop remains a TODO. | Added the NP query and video-upload callbacks using local helpers. Added the cross-controller ABI, error/status values, bounded guest-string validation, deferred callbacks, and a host callback bridge through the local message-dialog HLE. The headless message is an English title-based approximation; no network connection is claimed because RPCS3 has none either. No RPCS3 function body was copied. |
| `cellL10n` converters | RPCS3 registers many code-page conversions, but a large portion are TODO-only handlers. Existing implementations use Win32 or iconv through ps3recomp's guest-memory conversion layer. | Preserved existing conversion implementations and added 16 signed-length wrappers for JIS, HZ, EUC-KR/UHC, EUC-JP, Big5, and Shift-JIS plus two single-character conversions using that local converter. No RPCS3 function body was copied. A further 105 registered names remain absent; several RPCS3 handlers have no usable parameter signature. |
| Filesystem positional I/O | RPCS3 implements `cellFsReadWithOffset` and `cellFsWriteWithOffset` through `sys_fs_fcntl` operations `0x8000000a` and `0x8000000b`; the calls leave the descriptor's ordinary file position unchanged and optionally return the byte count. | Added both exports to `runtime/ppu/ppu_fs.cpp` using the local FILE-backed descriptor table, guest-range checks, and saved/restored stream positions. They are registered by computed NID. Runtime I/O behavior has not yet been exercised. |

## Remaining work supported by current evidence

- A name-level scan of RPCS3's `REG_FUNC` declarations found 3,706 unique Cell
  exports and 2,620 names absent from shared `libs/*.c` definitions after
  accounting for 60 unprefixed `cellL10n` exports and three unprefixed
  `sysPrxForUser` console exports. This is a
  candidate gap inventory, not a behavior-completeness result; aliases and
  wrappers need manual mapping. Regenerate it with
  `tools/compare_rpcs3_exports.py` and inspect the JSON for per-module names.

- The LV2 syscall-slot inventory (`tools/compare_rpcs3_syscalls.py`) found 620
  RPCS3 named entries with explicit numeric slots. Of those, 284 `BIND_SYSC`
  slots are not registered locally; another 71 are served at that number by a
  differently named local handler. This is a slot/name inventory, not an ABI
  equivalence proof. RPCS3 also marks 173 entries as explicit `NULL_FUNC`.
  Direct syscall 52 (`_sys_ppu_thread_create`) and 53
  (`sys_ppu_thread_start`) remain unimplemented locally; the higher-level
  `sysPrxForUser` HLE create export is not ABI-equivalent. The same audit found
  and corrected the syscall 41 PPU thread exit mapping and registered the
  existing try-write rwlock implementation at both RPCS3 slots 126 and 148.
  Regenerate the inventory with `tools/compare_rpcs3_syscalls.py`.

- `tools/compare_rpcs3_calls.py` adds the internal-call view: 292 unique
  `sys_*` call targets occur in Cell HLE sources, and 209 are absent by exact
  name from local `libs/` and `runtime/` function definitions. It finds
  dependencies that are not `REG_FUNC` exports; review each candidate's ABI
  and local equivalent before porting it.
- The current RSX method scan finds 101 explicitly bound RPCS3 method names and
  no names absent from local video identifiers. The two former candidates were
  checked against numeric dispatch; NV0039 now has a stride-aware copy handler.
  This remains a name inventory, not proof that every method's full hardware
  behavior is covered.

- Run the release project with the new raw-job registration and capture whether
  the fingerprint changes from `dispatch MISS` to `dispatch HIT`, whether the
  job returns, and whether the title advances. This has not been verified.
- If the lifted job fails, identify the executed SPU PC and instruction before
  adding semantics. Static `.word` sites include embedded data and are not, by
  themselves, proof of reachable unsupported instructions.
- Compare each new RSX warning with RPCS3's register/method behavior and the
  guest command stream before adding a handler. Preserve unknown register data
  where safe; avoid fabricating draw semantics.
- Audit and implement the remaining recognized RSX methods and vertex/fragment
  ISA gaps against RPCS3 contracts. RPCS3 itself leaves some VP instructions
  such as `PSH`/`POP` unsupported; those need independent hardware/trace evidence.
- Resolve the remaining P5 Feb 16 import and syscall paths from the audit using
  call-site evidence. Some RPCS3 counterparts are themselves placeholders, so
  their return value alone does not prove the title's intended behavior.
- Continue the all-module export and LV2 scan as individual ABI-backed ports;
  the 2,620 export-name and 284 syscall-slot candidates are not safe targets for
  blanket success stubs.

## Change and verification status

- Refreshed the export, syscall, call-target, and RSX-name inventories on
  2026-10-15 after completing the 17 RPCS3-registered `cellNetCtl` exports and
  two `cellSpurs` taskset getters. Current counts are 2,620 export names absent
  by name, 284
  unregistered syscall slots, 209 internal call names absent by exact name,
  and zero RPCS3 RSX method identifiers absent from local video source. The RSX
  count is name-only; matching identifiers do not prove complete semantics.
- Verified the `p5_feb16` build in the VS 2022 Developer PowerShell
  environment (Ninja reported no work), then launched the current executable
  with the matching `EBOOT_feb16.elf`. The
  fresh run log reports `[VP] pipeline ready` and contains no `X4576` or
  `[VP] PS compile FAIL`. The exact current `vpps` source also compiles via
  Windows `D3DCompile`; the repeated user report was not reproduced by this
  executable.

- Expanded `cellVoice` from 15 local entry points to the 35 registered by
  RPCS3, corrected `CreatePort` argument order and guest layouts, and added
  queue notifications and stateful port controls. Debug and release projects
  compiled this change; the release link required the Windows SDK resource
  compiler directory on `PATH` in this shell. Voice-device data paths remain
  unavailable and were not runtime exercised.

This turn adds the VP `BRB`/`CLB` implementation, corrects the local SAIL player
ABI/state initialization, and keeps generated raw SPU job registrations
available in both recompiler project CMake files. Both projects use the same
runtime/decompiler source. Debug and release Ninja builds both linked after
these changes. No game run is used to claim that the new paths are reached or
that the title advances.

The docs review covered the 29 files inventoried before this report was added
under `docs/`. It corrected
stale compiler, NID, syscall, SPU/RSX, and stub guidance in the relevant guides;
historical Uncharted/flOw and RSX MVP measurements are now identified as
snapshots rather than current compatibility claims. Generated diagnostics and
external RSX reference artifacts remain source/reference material.

## 2026-10-19 follow-up against the current local RPCS3 checkout

This review uses RPCS3 commit `a99afd4cf` (the earlier tables above used
`2ba2f75bc`). The two implementations cannot be substituted as complete source
files: RPCS3's GCM/RSX/SPU code depends on its VM, thread, renderer, and logging
APIs. Ports therefore keep ps3recomp's C runtime and D3D12 backend interfaces.

- SPU group termination was updated in
  `runtime/syscalls/lv2_register.c`: it now requests cooperative stop, wakes
  paused workers, waits for active worker completion, and only then marks the
  group stopped. This follows RPCS3 commit `b1f8b02909` using ps3recomp's group
  stop hook and finish events. The release executable builds after the change.
- GCM now implements `cellGcmInitSystemMode`: mode 1 expands the RSX I/O mapping
  ceiling from 256 MiB to 512 MiB, with the default unchanged. The import is
  registered through the existing PPU context-aware HLE bridge. This follows
  RPCS3's stored system-mode selection and `_cellGcmInitBody` size choice;
  runtime use by the P5 release has not been confirmed.
- RPCS3 commit `9c8e4349b7` changes the order of conditional SPU branch targets
  returned by its analyzer. ps3recomp's function finder collects targets into
  a set and does not depend on traversal order, so no port was needed.
- RPCS3 commits `2e6604483c` and `3beb6504de` change cached FIFO read boundaries
  and rollback positions. ps3recomp's GCM walker reads commands directly and
  has no corresponding cache or rollback path; those internals are not
  applicable as a source transplant. Its separate command semantics still need
  comparison when a capture shows a missed method or branch.
- The current name-level scan finds 101 RPCS3-bound RSX methods and no names
  absent from local video source. This says nothing about full behavior parity.
- The SPU decoder declares 199 real operation names, and the interpreter has
  199 corresponding case labels. This is opcode coverage only; it does not
  prove every edge case or SPURS scheduling behavior matches RPCS3.
- The all-module export scan finds 30 `cellGcmSys` names absent by exact name
  from local library definitions. The list includes versioned/private entries
  and requires ABI-level review; a name match alone is not a reason to add a
  stub. Many corresponding RPCS3 handlers are TODOs.

### P5 release paths confirmed by the 2026-10-19 run

- `p5_npeb02436/run-depth-coherence-ffmpeg-20261019/console-attached.log`
  records repeated `NV4097_GET_REPORT` requests for report types 2 and 3.
  Keep the report implementation in scope; the trace does not show which DMA
  destination the game selected.
- The same run creates `gfdSpursCellSpursKernelGroup` with six SPU threads and
  repeatedly dispatches a 32,064-byte job at LS entry `0x150`. The matching
  release workload is registered in `p5_npeb02436/spu/spu_raw_jobs.c`. This is
  a game-reached SPU path, unlike speculative exports that have no call trace.
- The same profile reports roughly 9,000 PSO lookups and about 0.27 seconds in
  the PSO lookup/hash section per 20-frame sample. Frames with no scene draws
  exceed 400 FPS, while scene-render samples are around 10-12 FPS.
  `rsx_d3d12_backend.c` now hashes the
  existing complete PSO key into buckets and uses a ring for bounded eviction,
  retaining full equality checks for collision safety. This follows the
  hash-keyed cache pattern in RPCS3's `VKShaderInterpreter`, adapted to the
  existing D3D12 backend. Build success is confirmed; the effect on gameplay
  frame time requires a fresh run.

The modified release build completed successfully in VS 2022 Developer
PowerShell with Ninja. No runtime test was performed for the group-termination
change.
