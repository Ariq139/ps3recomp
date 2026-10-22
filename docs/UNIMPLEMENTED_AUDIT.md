# Implementation gaps in the two P5 recompiles

Evidence snapshot: 2026-10-15. The debug project is `p5_feb16`; the release
project is `p5_npeb02436`. Both compile the same shared runtime, lifters, and
RSX translation code. Behavior comparisons use local RPCS3 at commit
`2ba2f75bc`; see [`RPCS3_IMPLEMENTATION_COMPARISON.md`](RPCS3_IMPLEMENTATION_COMPARISON.md)
and [`RPCS3_LICENSE_AUDIT.md`](RPCS3_LICENSE_AUDIT.md).

## What the current generated code proves

| Project | EBOOT imports | PPU generated TODO sites | SPU generated TODO sites |
|---|---:|---:|---:|
| `p5_feb16` | 339 | 2,673,282, all `.word` | 183, all `.word` |
| `p5_npeb02436` | 340 | 32,881, all `.word` | 248, all `.word` |

These `.word` entries are raw words, including embedded data. The generated
outputs contain no TODO site for a decoded PPU or SPU mnemonic. Treating every
raw word as an instruction would miscompile embedded tables and strings. A
runtime trace showing a program counter at one of those words is the evidence
needed to revisit it.

## Remaining imported compatibility handlers

Each project registers 16 game-import compatibility handlers from its
`stubs.cpp`. The PRX handler now records descriptor associations for loaded
`prx_module` exports; the other 15 retain the available RPCS3 parity behavior
because their counterparts are success-only TODOs or provide no contract:

- `_cellGcmFunc15` (`0x3A33C1FD`): RPCS3 declares a `void` function and its
  implementation is TODO. Local code preserves the return register; side
  effects remain unknown.
- Thirteen `cellSail` imports: `GraphicsAdapterGetFrame`,
  `DescriptorSetEs`, graphics/sound `PtsToTimePosition` and `UpdateAvSync`,
  `PlayerReplaceEventHandler`, `SourceNotifyInputEos`,
  `SourceNotifyCallCompleted`, `SourceNotifyStreamOut`, `SourceInitialize`,
  `DescriptorOpen`, `SourceFinalize`, and `SoundAdapterUpdateAvSync`.
  RPCS3's matching exported handlers are TODO/success paths. The local media
  player has lifecycle state but no SAIL demux/decode/render pipeline, so a
  `CELL_OK` return does not mean media playback is implemented.
- `cellSailSoundAdapterGetFrame` (`0x7EB8D6B5`): local code mirrors RPCS3's
  initialization/registration checks and 2048-sample limit; it does not
  produce audio frames. RPCS3 also leaves frame production TODO.
- `sys_prx_register_library` (`0x42B23552`): local code validates the 0x1c-byte
  descriptor, matches it against loaded PRX export-library descriptors, and
  retains an external-registration flag on the matching PRX library. Like
  RPCS3, it returns `CELL_OK` for a valid descriptor even when no loaded PRX
  matches. The local loader currently models this for PRXs represented by its
  `prx_module` metadata; it does not load arbitrary runtime PRXs or reproduce
  RPCS3's full PRX unload/relink behavior.

The fallback list covers these two EBOOT manifests; it is not a complete list
of unimplemented APIs in the shared runtime.

## Additional shared-runtime gaps found by source scan

### RPCS3 export-name inventory

`tools/compare_rpcs3_exports.py` compares RPCS3 `REG_FUNC` names with
export-looking definitions in shared `libs/*.c`. On 2026-10-15, against
RPCS3 commit `2ba2f75bc`, it found 3,706 unique RPCS3 Cell export names and
2,620 names absent from the shared C library sources after adding a narrow
manifest for unprefixed exports. The inventory now recognizes 63 locally
defined exports that the old prefix-only scan missed: 60 `cellL10n` names and
three `sysPrxForUser` console functions. `cellL10n` still has 105 RPCS3
registered names without matching local definitions. The complete module
grouping can be regenerated with:

```powershell
py -3 tools/compare_rpcs3_exports.py --local . --rpcs3 D:\Other\GitRepo\rpcs3 `
  --json-out $env:TEMP\ps3recomp-rpcs3-export-gaps.json
```

This is a name-level inventory, not a finding that every listed API lacks
behavior. Aliases, wrappers, non-C implementations, and per-game handlers need
individual review. It does establish that the shared library surface is much
narrower than RPCS3's registered export catalog. No generated API is marked
implemented solely because its name appears in a module.

The `cellNetCtl` source now defines all 17 names registered by RPCS3.
Eight game-internal NetCtl exports plus update init/term/abort preserve the
eleven RPCS3 TODO-success behaviors. The six callback-bearing update-check
exports now queue the RPCS3 `NO_UPDATE` or `FINISHED` callback result through
the local sysutil pump. The `Ex` forms provide a guest-endian 16-byte result
structure in temporary guest memory, released after the guest callback
returns. The export scan confirms the module name gap is closed; callback
execution has not been exercised in a running game.

The `cellRtc` module name gap is closed: all registered RPCS3 `cellRtc` export
names now have local definitions, including validation, comparison, parsing,
system-time and DOS/FILETIME conversion, timezone formatting, and tick
arithmetic. The source-level implementation uses a process-local clock offset
for `cellRtcSetCurrentTick`; it does not change the host clock. The export scan
does not establish runtime date/parser correctness.

The `cellPamf` name gap has been closed: the local generated registry now
contains all 24 exports registered by the current RPCS3 source. The local
implementation parses the packed PAMF header, stream descriptors, per-stream
EP tables, and 24-byte `CellPamfEp` outputs. Runtime behavior against real PAMF
fixtures has not been verified, so this records export and implementation
coverage rather than claiming media-level validation.

The `cellResc` review added the missing wait-flip, surface-to-source,
resolution-ID, and register-count exports, and corrected ABI signatures for
`cellRescGetNumColorBuffers`, `cellRescSetConvertAndFlip`,
  `cellRescGetLastFlipTime`, and `cellRescCreateInterlaceTable`. RPCS3 currently
leaves several of these as TODO-success handlers; those paths preserve their
validation and output behavior without claiming a RESC conversion pipeline.

The `cellGcmSys` export inventory now includes `cellGcmMapEaIoAddressWithFlags`.
It accepts RPCS3's `CELL_GCM_IOMAP_FLAG_STRICT_ORDERING` value and reuses the
local EA/IO mapping implementation; invalid flag combinations return
`CELL_GCM_ERROR_INVALID_VALUE`. The shared guest-memory/FIFO model has no
separate cache-ordering mode, so this is API and mapping parity rather than a
distinct host cache-control operation.

`cellGcmSetDefaultCommandBufferAndSegmentWordSize` now validates the idle
control words and size relationship, stores the word counts, and feeds the
existing default-size getters. RPCS3 uses put/get `0x1000` as its initial FIFO
position; this runtime starts at zero, so the handler accepts either matching
initial pair.

The `cellSpursGetTasksetId` and `cellSpursTasksetGetSpursAddress` exports now
read the workload ID and owning SPURS effective address from the guest taskset
layout, with RPCS3-compatible null, alignment, and workload-range checks. The
handlers use local big-endian guest-memory helpers; the comparison verifies
their behavior against RPCS3 `cellSpurs.cpp`, not copied implementation code.

### RPCS3 HLE call-target inventory

`tools/compare_rpcs3_calls.py` scans RPCS3 Cell HLE sources for called and
address-passed `sys_*` functions, then compares them with same-named function
definitions in local `libs/` and `runtime/`. The 2026-10-15 scan found 292
unique targets and 209 names without a same-named local definition. Frequent
candidates include PRX load/start/stop/unload calls, `sys_fs_fcntl`, and shared
memory mapper operations. Regenerate the complete list with:

```powershell
py -3 tools/compare_rpcs3_calls.py --local . --rpcs3 D:\Other\GitRepo\rpcs3 `
  --json-out $env:TEMP\ps3recomp-rpcs3-call-gaps.json
```

This is a call-name inventory, not proof that every name lacks equivalent
behavior: RPCS3 helpers can lower to a different local API, and source scans do
not establish that a call is reached by either title. Review the guest ABI and
the concrete local implementation before adding a handler.

### RSX method inventory

`tools/compare_rpcs3_rsx_methods.py` compares the explicitly bound method
names in RPCS3 `rsx_methods.cpp` with identifiers across local video C/C++/ObjC
sources and headers. It also scans the FIFO walker in `cellGcmSys.c`; the
earlier `rsx_*.c`-only scan missed handlers implemented there and in headers.
The scan currently finds 101 RPCS3 method names and no names absent from local
video identifiers. The two previous candidates were reviewed: NV3089_IMAGE_IN
already had numeric dispatch and is now named at its trigger method; NV0039
BUFFER_NOTIFY now has a stride-aware transfer handler. This is still a name
inventory, not proof that every supported method has complete hardware behavior.
Review method values and the command path before concluding the method is
unimplemented. Regenerate it with:

```powershell
py -3 tools/compare_rpcs3_rsx_methods.py --local . --rpcs3 D:\Other\GitRepo\rpcs3
```

`NV4097_ARRAY_ELEMENT16/32` are now implemented in the shared dispatcher and
the dispatcher-backed draw engine, including the live D3D12 renderer: values
are staged during `BEGIN_END`, then fetched and drawn as an immediate u32
index stream at `END`. The default D3D12 vtable path also handles the stream.
Other direct vtable backends still lack the optional immediate-index callback;
the legacy vtable path also still returns an unimplemented-method result for
`NV4097_INLINE_ARRAY`.

### LV2 syscall inventory

`tools/compare_rpcs3_syscalls.py` compares numbered RPCS3 `BIND_SYSC` entries
with local syscall registrations. The 2026-10-15 scan found 620 named RPCS3
slots: 284 have no local handler, 71 are occupied by a local handler with a
different symbol, and 173 RPCS3 slots are explicit `NULL_FUNC` entries. The
tool reports candidates; local aliases and ABI differences still require
source review. Direct slots 52 (`_sys_ppu_thread_create`) and 53
(`sys_ppu_thread_start`) remain absent in ps3recomp. The public
`sys_ppu_thread_create` NID is a higher-level HLE path and cannot stand in for
those raw syscall ABIs. The audit also corrected raw slot 41 to thread exit
and registers the existing rwlock try-write handler at slots 126 and 148.

This inventory includes calls and kernel entry points that have no matching
local method; it is intentionally broader than the two P5 import manifests.
It does not justify success stubs for the remaining slots.

These are real unsupported areas beyond the two manifests. Their existence
does not prove either game imports or calls them. Implementing them requires
their guest ABI and the corresponding subsystem, rather than another blanket
success stub:

- RSX VP `PSH`/`POP` are named but unimplemented by both this decompiler and
  RPCS3's `VertexProgramDecompiler.cpp`; the local RPCS3 checkout therefore
  cannot supply their missing semantics.
- RSX FP `CAL` is rejected by RPCS3's `FPToCFG.cpp`; the local decompiler keeps
  it explicit as unsupported. `BEMLUM` and `TIMESWTEX` are named by the RPCS3
  ISA table but have no fragment-decompiler handler there. FP `RET` now emits
  a conditional early shader return in ps3recomp. The remaining instruction
  semantics need an authoritative ISA description or hardware evidence.
- The D3D12 VP path now binds 1D, 2D, and cube vertex `TXL` textures. 3D vertex
  textures still return the defined-zero fallback: their depth-aware upload,
  resource view, and three-coordinate sample path are not implemented. Vertex
  texture filter and wrap fields also still use the backend's default linear,
  wrap sampler.
- Device APIs report absent devices or provide no capture: `cellGem`,
  `cellCamera`, `cellUsbd`, and `cellMic`. `cellVoice` now has the RPCS3
  exported API surface and port/event state, but still has no microphone
  capture, voice codec, or voice-data transport.
- Media APIs include non-decoding or placeholder implementations in `cellAdec`,
  `cellVdec`, `cellVpost`, `cellDmux`, `cellSail`, and several audio/video
  codec modules. RPCS3 uses additional emulator and host multimedia machinery;
  matching those APIs requires their complete structures, callback ordering,
  buffering, and host decoder integration.
- `cellCrossControllerInitialize` now implements RPCS3's current validation,
  dialog, and deferred callback behavior. The PS Vita connection loop is a TODO
  in RPCS3; the local implementation does not invent a network service.
- Some RSX features remain backend-specific or partial. The implementation
  comparison lists known method/state and shader coverage boundaries. Unknown
  methods and ISA encodings need source-register or runtime evidence before a
  handler can be considered correct.

Graphics ports must use ps3recomp's RSX translation and D3D12 backend on
Windows; audio ports must use its existing audio interfaces. RPCS3's host
graphics and audio backends are behavior references, not replacement backends.

## Work completed and verification

- Extended the runtime SPU interpreter with double-precision arithmetic and
  comparison dispatch, event-gated `bisled`, and `dftsv`. The `dftsv` decoder
  now extracts its 7-bit selector from the instruction's RB-position field;
  the helper tests NaN, ±infinity, ±zero, and ±denormal in both doubleword
  lanes. Double operations use existing local helpers. This does not claim full
  SPU ISA coverage.
- Matched interpreted `rchcnt` results to the lifted SPU ABI: count in the
  preferred word and zeros in the other words. A static inventory check found
  every mnemonic in the current decoder table has both interpreter dispatch
  and lifter handling; this covers the implemented decoder table, not every
  instruction in the full SPU ISA.
- Matched interpreted `mfspr` with the existing lifted helper and RPCS3's
  current `MFSPR` implementation: modeled SPR reads clear the destination to
  zero, while `mtspr` writes remain ignored. This mirrors RPCS3's provisional
  behavior; it does not establish hardware SPR semantics.
- Matched interpreted `fscrrd` with RPCS3's fast SPU interpreter: all 128
  destination bits are cleared; `fscrwr` remains ignored. This reflects
  RPCS3's current fast-interpreter behavior, not a full FPSCR model.
- Added decoder and interpreter paths for `orbi`, `xorhi`, and `xorbi`. Their
  byte/halfword helpers and lifted translations already existed but were
  unreachable because the decoder table omitted their opcodes. The resulting
  mnemonic inventory now matches all 199 instruction names registered in the
  local RPCS3 checkout; this is name coverage, not validation of each
  instruction's complete edge-case semantics.
- D3D12 guest-fragment PSOs now use the RSX depth-test enable, write mask, and
  comparison function captured at each draw, and include those values in the
  PSO cache key. This fixes the prior hard-coded `LESS_EQUAL`/always-write
  state for the guest-FP path; fallback PSOs remain separate.
- Decoded NV4097 front/back stencil masks, functions, references, and
  operations. D3D12 guest-fragment PSOs now enable stencil and apply per-face
  compare/operation state, with front stencil reference set per draw. D3D12
  exposes one stencil read mask, write mask, and reference for both faces, so
  differing back-face masks/references cannot be represented exactly there;
  fallback PSOs remain separate. Metal now consumes the separate front/back
  masks and compare/operation state.
- Implemented NV40 VP boolean branches (`BRB`) and calls (`CLB`) using D3's
  selector/polarity bits and the `NV4097_SET_TRANSFORM_BRANCH_BITS` register.
  The value is now carried by the VP constant buffer in the D3D12, Vulkan,
  Metal, live-draw, and draw-engine shader paths. The branch contract was
  checked against RPCS3; local code was written independently.
- Corrected `cellSailPlayerInitialize2` to write the documented guest player
  layout, copy allocator/attribute/resource data, use RPCS3's public player
  state values, set the initialized/paused fields, and issue the initialization
  state callback. Sound/graphics adapter attachment now validates stream
  indices and stores adapter pointers, indices, and registration state. This
  improves lifecycle/ABI handling; it does not add a media decoder or renderer.
- Added optional `spu_raw_jobs.c` inclusion to the debug CMake project so raw
  registrations are built when present, matching the release project.
- Added PRX export-library descriptor metadata to `tools/prx_exports.py` and
  `runtime/prx/prx_loader.c`. `sys_prx_register_library` now associates a
  matching descriptor with a loaded PRX library, and each registered export
  retains its library association for later PRX operations.
- Added `sys_event_flag_cancel` with waiter counting, cancellation wakeups,
  `CELL_ECANCELED` returns, and pattern preservation. Corrected raw PPU thread
  exit to syscall 41 and registered the try-write rwlock handler at both LV2
  slots 126 and 148.
- Added `cellSysmodule` internal/extended load, unload, and loaded-state calls,
  reference counts, memory-container validation, and debug/internal mode entry
  points. The runtime remains static-HLE: internal PRX dependencies and
  dynamic module-file loading are still not implemented. `GetImagesize`,
  `FetchImage`, `LoadModuleFile`, and `UnloadModuleFile` remain absent because
  they require firmware-image VFS access or arbitrary runtime PRX loading.
- Replaced the mismatched `cellRemotePlay` lifecycle/status exports with the
  eight names registered by RPCS3 and ported its current TODO-success behavior;
  `GetComparativeVolume` writes the RPCS3 default `1.0f`. This does not create a
  Remote Play connection or host streaming path.
- Corrected `cellL10n` code-page IDs and generic converter ABIs, added
  guest-memory-safe platform-backed conversion, and implemented 23 additional
  Unicode/code-page exports. Sixteen signed-length JIS, HZ, EUC-KR/UHC,
  EUC-JP, Big5, and Shift-JIS conversion exports now use the same local checked
  converter, along with the single-character SJIS/UCS-2 exports; existing
  ps3recomp conversion implementations were kept. Added 60 locally defined
  RPCS3-registered names to the generated NID table. Many
  RPCS3 TODO-only converters remain absent because their ABI or contract is
  not established.
- Added `cellAudioAdd2chData`, `cellAudioAdd6chData`,
  `cellAudioCreateNotifyEventQueueEx`, `cellAudioSetNotifyEventQueueEx`,
  `cellAudioRemoveNotifyEventQueueEx`, `cellAudioSendAck`,
  `cellAudioMiscSetAccessoryVolume`, and `cellAudioSetPortLevel` using the
  existing mixer/event-queue architecture. Stereo/surround routing, PPU/SPU
  queue types, flagged subscription/removal, decimation, source IDs, ack
  timestamps, and the 13 ms level ramp were checked against RPCS3. Accessory
  volume follows RPCS3's initialized no-op. All four previously missing
  `cellAudio` export names now resolve in the export comparison. The local
  mixer does not reproduce RPCS3's before-mix timing or ack wait.
- `cellAudioSetPersonalDevice` and `cellAudioUnsetPersonalDevice` now return
  RPCS3-compatible not-initialized/parameter errors before their success path.
  RPCS3 also leaves personal-device routing as TODO, so no routing effect is
  claimed here.
- Added the four `cellBGDLUtility` exports registered by RPCS3. Their current
  handlers return `CELL_OK` without touching inputs or outputs; the older
  lowercase local download helpers remain available as compatibility APIs.
- Rebuilt the debug and release Ninja projects with Visual Studio 2022
  Developer PowerShell 17.14.41. Both regenerated `ppu_hle_nids.cpp` with the
  four `cellBGDL*` handlers and linked successfully; no game run or test suite
  was performed.
- Rebuilt both Ninja configurations successfully in Visual Studio 2022
  Developer PowerShell 17.14.41 on 2026-10-15 after implementing immediate
  array-element indices in the dispatcher and D3D12 vtable paths:
  `P5Feb16Recomp.exe` and `P5Npeb02436Recomp.exe` linked. The build emitted
  compiler warnings; no game run was performed for this change.
- Rebuilt `p5_feb16/build-clang-vs` in Visual Studio 2022 Developer PowerShell
  after adding the two `cellSpurs` taskset getters. The generated HLE registry
  contains both NIDs and `P5Feb16Recomp.exe` linked successfully. No game run
  or test suite was performed.

The audit is not complete for every runtime export, LV2 syscall, RSX method,
SPU interpreter opcode, platform backend, or optional library. No claim is made
that all missing functionality has been implemented. Continue only from a
specific behavior contract and validate the resulting code in the relevant
build/runtime path.
