# RPCS3 provenance and license audit

Audit date: 2026-10-13  
Scope: first-party source under `libs/`, `runtime/`, and `tools/`; repository
history and explicit RPCS3 provenance comments; project license and contributor
guidance. Compared relevant areas with the local RPCS3 checkout at
`D:\Other\GitRepo\rpcs3`.

## Result

The tracked repository was searched for RPCS3 references; implementation hits
and relevant commit history were reviewed, with targeted source comparison for
the flagged areas.

The repository declares MIT in `LICENSE`; `CONTRIBUTING.md` says not to port
code from GPL projects. RPCS3 declares GNU GPL version 2. The evidence below
means the repository is **not cleared as MIT-only**: the history explicitly
records an RPCS3 algorithm port, and two SPU lookup tables match RPCS3 exactly.
Other implementation areas are marked for provenance review where their own
comments describe close structural alignment or use RPCS3 implementation
semantics. A marker is a review flag, not a license grant or legal conclusion.

The confirmed material needs one of these resolutions before claiming the
whole tree is MIT-only: replace it with independently sourced material, obtain
and document permission/relicensing from the relevant rights holders, or
distribute the affected combined work under terms that satisfy the applicable
source license. This audit does not decide which path is legally available.

## Confirmed RPCS3-derived material

| File and marked area | Evidence | Finding |
|---|---|---|
| `runtime/ppu/sdata_decrypt.h` | Commit `c832ed97` says the algorithm and keys were ported from `RPCS3 Crypto/unedat.cpp`. | Confirmed implementation port. Marked `RPCS3_PROVENANCE_REVIEW_REQUIRED`. |
| `tools/spu_disasm.py` | Commit `7a02d919` says opcode tables were rebuilt from RPCS3; the source calls the `SPU_RR` table verbatim. | Confirmed copied/reconstructed table material. Opcode facts may be uncopyrightable, but the table's expression/organization has not been legally assessed. Marked for review. |
| `runtime/spu/spu_helpers.h` | `spu_frest_fraction_lut` and `spu_frsqest_fraction_lut` were compared with RPCS3 `SPUThread.cpp`; both arrays match exactly. The file also identifies interpreter-derived instruction semantics. | Confirmed copied lookup-table data; helper implementation provenance needs review. Marked for review. |
| `tools/spu_disasm_audit_operands.py` | Its oracle register/channel name tables are explicitly reproduced from RPCS3 `SPUDisAsm.h`. | Confirmed copied table data in an audit tool, not runtime implementation code. Marked for review. |

## Implementation areas flagged for provenance review

| File | Evidence | Assessment |
|---|---|---|
| `libs/filesystem/edat.c` | Header says its implementation structure follows RPCS3 `unedat.cpp` closely so the two can be diffed. | Strong structural-provenance concern; the source alone does not prove copied code. Marked for review. |
| `libs/video/rsx_fp_decompiler.c` | Comments map condition handling, flow control, derivatives, texture gradients, bump-environment transforms, and instruction formulas to RPCS3's GPL-2.0 fragment decompiler. | Behavior-source use is explicit; the new HLSL expressions were independently written, and no RPCS3 function body was copied. Marked for review. |
| `runtime/spu/spu_channels.c` | `WAIT_SIGNAL` is implemented using behavior documented by RPCS3's `spursTasksetProcessSyscall` and guest traces. | Behavior reference is explicit; no copied source was established. Marked at the implementation block. |
| `runtime/spu/spu_dma.h` | List-stall status behavior is explicitly compared with RPCS3; the implementation models per-tag guest state. | Behavior reference is explicit; direct copying is not established. Marked at the implementation block. |
| `runtime/syscalls/sys_timer.c` | The Windows sub-millisecond wait strategy is described as mirroring RPCS3's busy-tail. | Implementation approach is referenced; the local QPC/SwitchToThread code is visibly host-specific. Marked for provenance review. |
| `libs/spurs/spurs_pm.c` | Declares itself a clean-room reimplementation and identifies RPCS3 task-selection behavior/layout as a reference. | The clean-room claim is recorded; no copied source was established by the cited comments. Marked as reference-reviewed. |
| `libs/video/cellGcmSys.c` | A section is labeled “RPCS3 parity”; commit `64a1fbfc` says the functions were added after an RPCS3 audit. Sampled diff shows project-specific state handling, not copied RPCS3 bodies. | Retained as a provenance note; the examined history does not establish a code port. |
| `libs/audio/cellVoice.c` | Port and event semantics were compared with RPCS3 `cellVoice.cpp` at commit `2ba2f75bc`; local guest-memory, state, and event-queue code uses ps3recomp helpers. | Behavior-reference use is explicit; no RPCS3 function body was copied. Marked in the implementation comment. |
| `libs/audio/cellAudio.c`, `libs/audio/cellAudio.h`, and `runtime/syscalls/sys_event.c` | Stereo/six-channel sample routing, the 13 ms port-level ramp, Ex queue type selection, CellAudio errors, and personal-device initialization/argument checks were checked against RPCS3 at commit `2ba2f75bc`. The guest ring, event queue storage, and FPR bridges are local; no RPCS3 function body was copied. |
| `libs/video/cellResc.c` and `libs/video/cellResc.h` | RESC error values, argument validation, color-buffer counts, and TODO-success export behavior were checked against RPCS3 `cellResc.cpp`/headers at commit `2ba2f75bc`. The local HLE uses ps3recomp guest-memory and flip state; no RPCS3 function body was copied. Marked as behavior-reference use, not a code-copy finding. |
| `libs/misc/cellBGDL.c` and `libs/misc/cellBGDL.h` | The four `cellBGDLUtility` export names, guest ABI layout, and current TODO-success behavior were checked against RPCS3 `cellBgdl.cpp`/`.h` at commit `2ba2f75bc`. The compatibility handlers are local no-op functions; the pre-existing lowercase download helpers are local-only extensions. No RPCS3 function body was copied. |
| `libs/misc/cellRemotePlay.c` and `.h` | Export names, status/constants, successful TODO return behavior, and comparative-volume default were checked against RPCS3 `cellRemotePlay.cpp`/`.h` at commit `2ba2f75bc`. | API/behavior reference only; guest output uses ps3recomp memory helpers, with no RPCS3 body or streaming backend copied. |
| `libs/hardware/cellUsbd.c` | The added export names and `CELL_OK` placeholder results match RPCS3 `cellUsbd.cpp` TODO entries at commit `2ba2f75bc`; device errors use public ABI values. | Behavior-reference use is explicit; no RPCS3 function body was copied. Placeholder status is documented in source and module docs. |
| `libs/misc/cellVideoUpload.c` and `libs/network/sceNp.c` | Callback lifecycle and true-valued mode query were checked against RPCS3 `cellVideoUpload.cpp` and `sceNpPlus.cpp` at commit `2ba2f75bc`; local callback delivery uses `cellSysutilQueueGuestCallbackArgs`. | Behavior-reference use is explicit; local code is expressed through ps3recomp helpers, with no RPCS3 function body copied. |

## Additional behavior and ABI reference marks

These source blocks now carry `RPCS3_REFERENCE_REVIEWED` comments. They use
RPCS3 behavior or ABI contracts as a reference, but the targeted review did not
establish copied function bodies. The marks record provenance and do not grant
license clearance.

| File | Referenced behavior |
|---|---|
| `libs/spurs/cellSpurs.c` | Event-flag wait-slot protocol, single-lane job-manager behavior, and SPURS taskset getters. The new taskset-ID and owning-SPURS getters were checked against `cellSpurs.cpp` at RPCS3 commit `2ba2f75bc` and use this runtime's big-endian taskset fields; no RPCS3 function body was copied. |
| `runtime/spu/spu_workload.c` | SPURS `WAIT_SIGNAL` guest-state and host wait/wake contract. |
| `runtime/syscalls/lv2_register.c` | `sys_process_is_spu_lock_line_reservation_address` contract mapped onto this runtime's memory regions. |
| `runtime/syscalls/lv2_register.c` | SPU group RUN/EXCEPTION/SYSTEM_MODULE event types, event keys, start-time delivery, duplicate-connect behavior, and SYSTEM_MODULE group-type validation follow RPCS3's LV2 ABI behavior; the local implementation uses the runtime's own event queues. No RPCS3 function body copied. |
| `runtime/syscalls/lv2_register.c` | `sys_deci3_open` (880) returns `CELL_OK` with outputs untouched to match RPCS3's `NULL_FUNC`; behavior reference only, no RPCS3 body copied. |
| `runtime/syscalls/lv2_register.c` | The numeric LV2 `NULL_FUNC` dispatch set was transcribed from RPCS3's syscall table to distinguish explicit success stubs from unused `ENOSYS` slots. SPU syscall behavior was independently expressed using this runtime's group/thread/event state; no RPCS3 function body was copied. |
| `p5_feb16/stubs.cpp`, `runtime/prx/prx_loader.c`, and `tools/prx_exports.py` | SAIL frame checks and `sys_prx_register_library` descriptor validation/association were checked against RPCS3. The local loader records per-library descriptor metadata and external-registration state for statically loaded PRXs. The implementation is local; no RPCS3 function body copied. |
| `libs/codec/cellSail.h` and `p5_feb16/stubs.cpp` | SAIL error values and `cellSailSoundAdapterGetFrame` error ordering/sample limit are aligned with RPCS3's public header and handler. This is ABI/behavior reference material; no function body copied. |
| `runtime/syscalls/lv2_register.c` | SPU event-queue binding capacity (32 keys per thread) matches the RPCS3 SPU thread structure; local queue ownership and dispatch remain project-specific. No function body copied. |
| `runtime/syscalls/sys_event.c` | Unnamed `sys_event_port_send` sources use `(pid << 32) | port_id`, matching RPCS3's current PID 1 behavior; semantics only, no function body copied. |
| `runtime/syscalls/sys_event.c` | `sys_event_flag_cancel` waiter count, cancellation wakeup, return code, and pattern preservation were behavior-checked against RPCS3; implementation uses this runtime's condition variables and waiter state, with no RPCS3 function body copied. |
| `tools/compare_rpcs3_syscalls.py` | Parses RPCS3 syscall symbols and numeric comments to build a gap inventory; no RPCS3 implementation code is copied or executed. |
| `runtime/syscalls/sys_event.c` | `sys_event_flag_cancel` waiter count, cancellation wakeup, return code, and pattern preservation were behavior-checked against RPCS3; implementation uses this runtime's condition variables and waiter state, with no RPCS3 function body copied. |
| `tools/compare_rpcs3_syscalls.py` | Parses RPCS3 syscall symbols and numeric comments to build a gap inventory; no RPCS3 implementation code is copied or executed. |
| `runtime/spu/spu_lifted_thread.c` | SPU thread argument lane placement from the guest ABI. |
| `runtime/spu/spu_lifted_job.h` | SPURS task argument setup from the guest ABI. |
| `libs/video/rsx_live_draw.c` | RSX `SET_REFERENCE` synchronization behavior. |
| `libs/video/rsx_d3d12_backend.c` | RSX swizzled-texture address mapping compared with an RPCS3 helper. |
| `libs/system/cellSaveData.c` | Older save-data exports sharing behavior with RPCS3's newer NIDs. |
| `libs/system/cellGame.c` and `libs/system/cellMsgDialog.c` | Older API wrappers following RPCS3's forwarding behavior. |
| `libs/hardware/cellCamera.c` | Optional-camera behavior compared against RPCS3 runs. |
| `libs/network/cellNetCtl.c` | Offline-network behavior compared against RPCS3 runs. All 17 registered update/internal exports were checked against RPCS3 `cellNetCtl.cpp` at commit `2ba2f75bc`: eleven are success-only TODOs; six queue the same deferred callback results. Local guest-memory and callback code is independent; no RPCS3 function body is copied. |
| `libs/system/cellSysutil.c` | Callback timing was checked against RPCS3 `sysutil_register_cb` behavior in `cellNetCtl.cpp`; the local queue dispatches on the guest's next sysutil poll and owns/frees its guest payload locally. No RPCS3 function body is copied. |
| `libs/font/cellFontFT.c` | FreeType library output-pointer contract reviewed; rasterization remains local fallback behavior. |
| `libs/font/cellL10n.c` and `.h` | Code-page IDs, conversion result values, packed converter IDs, and string-length ABI checked against RPCS3 `cellL10n.h`/`.cpp` at commit `2ba2f75bc`. Sixteen signed-length wrappers and two single-character converters were added for RPCS3 TODO-only exports using ps3recomp's existing Win32/iconv converter and guest-memory helpers. No RPCS3 function body was copied. |
| `libs/input/cellPad.c`, `cellKb.c`, and `cellMouse.c` | CellPad/CellKb/CellMouse ABI and event behavior checked against RPCS3 at commit `2ba2f75bc`. The SCP extension struct layouts in `cellPad.c` are transcribed from RPCS3 `xinput_pad_handler.h` and marked `RPCS3_PROVENANCE_REVIEW_REQUIRED`; assess these definitions under RPCS3's GPL-2.0 terms before redistribution. The host polling, guest-API glue, keyboard fallback, and Win32/SDL event routing are local; no RPCS3 function body was copied. |
| `tools/hle_special_exports.json`, `tools/gen_hle_nids.py`, and `tools/compare_rpcs3_exports.py` | A narrow manifest records unprefixed local functions confirmed by RPCS3 `REG_FUNC` declarations (`cellL10n` conversions and `sysPrxForUser` console functions). It contains export names and source provenance only, not RPCS3 implementation code. |
| `libs/network/sceNp.c` | NP manager and NPDRM entrypoint/error contracts reviewed; local VFS and EDAT routines are used, with no RPCS3 function body included. |
| `libs/network/sceNpTus.c` | TUS guest structure sizes and argument layouts reviewed; calls use the independent in-memory backend. |
| `libs/network/sceNpMatching2.c` | Room-chat request layout/event ID reviewed; transport remains the local PSNR implementation. |
| `libs/codec/cellSail.c` | Player field offsets/state values, initialization callback, adapter association, and format offsets checked against RPCS3's `cellSail.h`/`cellSail.cpp`. Guest-memory handling is local; no RPCS3 function body was copied. Media decoding/rendering remains absent. |
| `libs/codec/cellPamf.c` | PAMF layouts, validation rules, stream coding/channel mapping, metadata fields, and EP iterator behavior checked against RPCS3 `rpcs3/Emu/Cell/Modules/cellPamf.cpp` and `cellPamf.h` (GPL-2.0). The parser and guest-memory operations are independently written; no RPCS3 function body was copied. |
| `libs/system/cellRtc.c` and `.h` | CellRtc errors, date/tick conversion contracts, time epochs, DOS/FILETIME conversions, formatting/parsing, and timezone behavior checked against RPCS3 `cellRtc.cpp`/`.h` at commit `2ba2f75bc` (GPL-2.0). Gregorian conversion, parsing, guest-memory access, and process-local clock state are independently expressed through ps3recomp helpers; no RPCS3 function body was copied. `cellRtcSetCurrentSecureTick` preserves RPCS3's success-only secure-RTC behavior. |
| `libs/misc/cellCrossController.c` and `.h`, `libs/system/cellMsgDialog.c` and `.h` | The CrossController error/status values, string-length checks, validation order, deferred INITIALIZED callback, and cancel-to-FINALIZED behavior were checked against RPCS3 `cellCrossController.cpp`/`.h` at commit `2ba2f75bc` (GPL-2.0). The local implementation uses its own guest-memory checks, sysutil queue, and host-dialog callback hook. RPCS3's PS Vita connection loop is a TODO; no network path was added. No RPCS3 function body was copied. |
| `libs/spurs/cellSpursJq.c` | Job descriptor binary pointer and size layout reviewed; descriptors dispatch through this runtime's lifted SPU registry. |
| `runtime/spu/spu_raw.c` | SPU image type values reviewed; close invalidates the local guest descriptor. |
| `tools/spu_lifter.py` | SPU preferred-word lane semantics cross-checked against RPCS3. |
| `runtime/syscalls/lv2_register.c` | SPU group-event types, RUN event key/payload, and start-time delivery reviewed against RPCS3 `sys_spu.h` and `sys_spu.cpp`; the local event queue/state helpers are used, with no RPCS3 function body copied. |
| `libs/video/rsx_commands.c`, `libs/video/rsx_d3d12_backend.c`, and `libs/video/rsx_metal_backend.m` | NV4097 clip range, fog/WPOS, polygon offset, window offset, context DMA, wait-idle, restart-index, and front/back stencil register behavior cross-checked against RPCS3 RSX enums and GL/Vulkan draw setup. The local state handling and D3D12/Metal pipelines were written for this runtime; no RPCS3 function body was copied. D3D12 has one stencil read mask, write mask, and reference for both faces, so asymmetric back-face masks/references remain constrained; ZPASS query values remain approximate because this backend has no occlusion-query producer. |
| `libs/video/rsx_vp_decompiler.c`, `rsx_vp_decompiler.h`, `rsx_live_draw.c`, `rsx_d3d12_backend.c`, `rsx_vulkan_backend.c`, `rsx_metal_backend.m`, and `rsx_draw_engine.c` | NV40 `BRB`/`CLB` selector/polarity fields and `NV4097_SET_TRANSFORM_BRANCH_BITS` upload were checked against RPCS3 `VertexProgramDecompiler.cpp` and `RSXVertexProgram.h`. Local HLSL/control-flow and constant-buffer code was written independently; no RPCS3 function body was copied. |
| `runtime/ppu/ppu_fs.cpp` | `cellFsReadWithOffset` and `cellFsWriteWithOffset` argument order, descriptor validation, optional byte-count output, and position-preserving I/O behavior were checked against RPCS3 `cellFs.cpp` at commit `2ba2f75bc`. The local handlers use this runtime's FILE-backed VFS and save/restore the host stream position; no RPCS3 function body was copied. RPCS3 is GPL-2.0; this is behavior-reference provenance, not license clearance. |
| `libs/video/rsx_fp_decompiler.c` | LIT/STR/SFL, conditional RET, DDX/DDY, TXD source-operand roles, DP2A, REFL, BEM/TEXBEM/TXPBEM, packed integer/half-float conversions, PKG/UPG sRGB conversion, and OPDEST `exp_tex`/precision range modifiers (FP16 ±65504 and FX12 ±2; RPCS3 commit `9e7a42d057`) checked against RPCS3 `FragmentProgramDecompiler.cpp`, `FPToCFG.cpp`, `GLSLCommon.cpp`, `RSXProgramCommon.glsl`, `RSXFragmentProgram.h`, and `RSXFragmentPrologue.glsl`. HLSL expressions and texture sampling are independently written for ps3recomp; no RPCS3 function body was copied. RPCS3 is GPL-2.0; this records behavior-reference provenance and is not license clearance. |
| `libs/video/rsx_dispatch.c`, `rsx_commands.c`, `rsx_d3d12_backend.c`, `rsx_draw_engine.c`, `rsx_draw_engine.h`, and `rsx_live_draw.c` | `NV4097_SET_SHADE_MODE` values and flat shading of diffuse/specular varyings were checked against RPCS3 `gcm_enums.h`, `nv4097.cpp`, and `RSXThread.cpp`. Local state decoding, HLSL qualifiers, and last-vertex rotation were independently written for both D3D12 draw paths; no RPCS3 function body was copied. |
| `libs/video/rsx_vp_decompiler.c`, `rsx_fp_decompiler.c`, `rsx_commands.c`, `rsx_d3d12_backend.c`, `rsx_dispatch.c`, `rsx_draw_engine.c`, and `rsx_live_draw.c` | Two-sided lighting enable and front/back diffuse/specular output selection were checked against RPCS3 `rsx_decode.h`, `RSXThread.cpp`, `GLFragmentProgram.cpp`, and `VKFragmentProgram.cpp`. Local varying declarations, face selection, and state handling are independently written for the D3D12 and live paths; no RPCS3 function body was copied. |
| `libs/video/rsx_vp_decompiler.c`, `rsx_vp_decompiler.h`, and `rsx_d3d12_backend.c` | Vertex `TXL` unit selection, 1D/2D/cube coordinate shape, and texture-unit state were checked against RPCS3 `VertexProgramDecompiler.cpp`, `RSXVertexProgram.h`, `RSXTexture.cpp`, and the RSX texture-cache upload path. The D3D12 path binds the decoded DXGI format for layout-supported packed, HILO, depth, float, and BC textures. Local HLSL and D3D12 descriptors/uploads are independently written; no RPCS3 function body was copied. 3D vertex textures and guest vertex sampler filter/wrap settings remain unsupported. |
| `libs/video/rsx_d3d12_backend.c` | The Z24-to-color byte order was checked against RPCS3 `RSXFragmentTextureDepthConversion.glsl::decode_depth24`; the color/depth last-write selection follows the local shared draw-engine serial model. The HLSL helper and D3D12 surface tracking are locally implemented; no RPCS3 source code was copied. RPCS3 is GPL-2.0; this records behavior-reference provenance, not license clearance. |
| `libs/codec/vdec_ffmpeg.c` | FFmpeg decoder setup was compared with RPCS3 `cellVdec.cpp`; this change leaves libavcodec's `thread_count` at its automatic default instead of forcing one thread. RPCS3's queued decoder worker was not copied. FFmpeg is loaded as an external runtime library and is separately licensed; no FFmpeg implementation source was copied into ps3recomp. |

## Other references reviewed

- `tools/compare_rpcs3_calls.py` scans RPCS3 HLE call names against local
  definitions as inventory data; it copies no implementation.
- `libs/system/cellSysmodule.c` uses RPCS3's internal module-ID ranges and
  loadable-ID sets as behavior references. The local state model uses
  ps3recomp counters and memory-container records; no RPCS3 function body was
  copied.

- `tools/generate_stubs.py` generates placeholder stubs from the local NID
  database. The old `docs/TOOLS.md` incorrectly said it parsed RPCS3 sources;
  that documentation has been corrected. Stub bodies are excluded from this
  audit as requested.
- `tools/hle_abi_audit.py` documents that it reads RPCS3 signatures only.
- `tools/spu_disasm_audit.py`, `runtime/syscalls/sys_mutex.c`, and
  `libs/spurs/spurs_taskset.h` explicitly describe RPCS3 as a reference/oracle
  and state that no source code was copied.
- `tools/ppu_lifter.py` implements VMX `stvlx`/`stvrx` and cache-hint XL forms
  after checking their byte-selection behavior against RPCS3's GPL-2.0
  `PPUInterpreter.cpp`; the local emitter is an independent expression of the
  instruction behavior, not a copied function body.
- `libs/video/rsx_dispatch.c` and `libs/video/rsx_vp_decompiler.c` identify
  RPCS3 as a read-only semantics oracle; their source comments do not claim
  copied code.
- Other matches describe ABI layouts, opcode/NID facts, captured runtime state,
  or behavior comparisons. A reference to RPCS3 alone is not evidence that an
  implementation body was copied.

## 2026-10-14 RSX and gamma behavior references

The following additions were written locally after reading the GPL-2.0 RPCS3
checkout as a behavior reference; no implementation body was copied:

- `libs/video/rsx_commands.h`, `rsx_commands.c`, and `rsx_dispatch.c`: texture
  coordinate control, vertex cache invalidation, anti-aliasing control, and
  transform timeout register handling. References: `gcm_enums.h`,
  `rsx_methods.cpp`, `rsx_methods.h`, and `rsx_decode.h`.
- `libs/video/rsx_fp_decompiler.c`: 2D texture-coordinate overrides and a
  single-sample stochastic alpha-coverage fallback, evaluated before
  alpha-to-one. References: `FragmentProgramDecompiler.cpp` and
  `RSXFragmentPrologue.glsl`; the pixel hash is independently chosen.
- `libs/video/rsx_live_draw.c` and `rsx_d3d12_backend.c`: locally written
  per-draw state capture, shader cache identity, and D3D12 sample masking.
- `libs/misc/cellAvconfExt.c`: gamma parameter validation matches
  `cellAvconfExt.cpp`. `runtime/ppu/ppu_sysprx.cpp` reads the PPU floating-point
  argument directly from f1 using this project's existing context adapter.

These changes do not resolve the confirmed copied material listed above.
The D3D12 paths continue to use one-sample render targets; this addition does
not implement multisample allocation or resolve. The vertex and fragment
shader interfaces were extended to carry TC8 and TC9 alongside TC0 through TC7.

## 2026-10-15 D3D12 depth and SPU interpreter references

- `libs/video/rsx_d3d12_backend.c`: guest depth-test enable, write mask, and
  NV4097/GL comparison values are captured per draw, included in the guest-FP
  PSO cache key, and translated to D3D12 comparison values. The register
  contract was checked against the local Vulkan/Metal mappings and RPCS3 RSX
  method definitions. The D3D12 state mapping is local; no RPCS3 function body
  was copied.
- `runtime/spu/spu_interp.c`: double-precision arithmetic/comparison opcodes
  dispatch to existing ps3recomp helpers. `bisled` link and event-gated branch
  behavior was checked against RPCS3 `SPUInterpreter.cpp::BISLED`; the local
  implementation uses ps3recomp's register, event, and PC state. No function
  body was copied. `dftsv` now uses bitwise IEEE-754 classification from the
  IBM SPU ISA v1.2 definition (p.230); the seven selector bits and lane masks
  were implemented locally, not taken from RPCS3.
- `runtime/spu/spu_interp.c`: `rchcnt` now produces the same preferred-word
  scalar result as the lifted emitter. This matches RPCS3's
  `SPUInterpreter.cpp::RCHCNT` (`v128::from32r`); only behavior was referenced,
  and no code was copied.
- `tools/spu_disasm.py`: added `orbi`, `xorhi`, and `xorbi` opcode entries
  present in RPCS3 `SPUOpcodes.h` but missing from the local RI10 decode table.
  The opcode values were checked against that registry; the local decoder uses
  its existing structure, and execution uses existing local helpers. No RPCS3
  function or table body was copied.

## 2026-10-15 RSX conditional-render reference

- `libs/video/rsx_dispatch.c`, `rsx_commands.c`, and `cellGcmSys.c`: added
  `NV4097_SET_RENDER_ENABLE` mode 1/2 tracking and report-value gating for
  array, indexed, and inline draws. The behavior was checked against RPCS3
  `NV47/HW/nv4097.cpp::set_render_mode` and `RSXZCULL.cpp`'s zero-value
  predicate rule; implementation uses ps3recomp's local report array and
  dispatch state. No RPCS3 function body was copied. Query production remains
  approximate because ps3recomp does not currently measure GPU occlusion.

## 2026-10-15 RSX immediate-index reference

- `libs/video/rsx_dispatch.c`, `rsx_dispatch.h`, `rsx_draw_engine.c`, and
  `rsx_live_draw.c`: `NV4097_ARRAY_ELEMENT16/32` writes are collected inside
  `BEGIN_END` and submitted as an immediate indexed draw at `END` when no
  explicit draw method selected another source. The default D3D12 vtable path
  now uses `rsx_commands.c`, `rsx_commands.h`, and `rsx_d3d12_backend.c` for
  the same behavior; other direct vtable backends have no immediate-index
  callback yet. Method ordering and immediate u32 index behavior were checked
  against RPCS3 `NV47/HW/nv4097.cpp::set_array_element16/32` and
  `Core/RSXDrawCommands.cpp::append_array_element` plus the `END` handler. The
  implementation uses ps3recomp's own dispatcher, vtable, vertex fetch, and
  topology conversion. The live renderer and D3D12 vtable submit through their
  existing D3D12 paths; the platform-neutral draw engine uses its registered
  host backend. No RPCS3 function body was copied.
  The dispatcher bounds one immediate-index stream to 64 KiB and drops the
  whole stream on overflow. The direct vtable path uses the same 64 KiB cap.

## 2026-10-15 CellAudio notification reference

- `libs/audio/cellAudio.c`, `cellAudio.h`, and `runtime/ppu/ppu_sysprx.cpp`:
  implemented `cellAudioSetNotifyEventQueueEx`, `cellAudioRemoveNotifyEventQueueEx`,
  `cellAudioSendAck`, and the FPR-aware `cellAudioMiscSetAccessoryVolume`
  bridge. Queue validation, duplicate association errors, event flag filtering,
  decimation, source IDs, and ack timestamp updates were compared with RPCS3
  `cellAudio.cpp` and `cellAudio.h` at commit `2ba2f75bc`. The accessory-volume
  implementation follows RPCS3's initialized no-op TODO. The mixer and event
  queue storage remain ps3recomp code; no RPCS3 function body was copied.
  `BEFOREMIX` carries the matching source ID, but the local mixer does not wait
  for acknowledgments or reproduce RPCS3's timing path.

## 2026-10-15 SPU special-register read reference

- `runtime/spu/spu_interp.c` and `runtime/spu/spu_helpers.h`: interpreted
  `mfspr` now clears its destination to zero, matching the local lifted helper
  and RPCS3 `SPUInterpreter.cpp::MFSPR` at commit `2ba2f75bc`. RPCS3 labels
  this as a provisional all-zero SPR model; this does not claim hardware
  completeness. `mtspr` remains ignored, matching RPCS3's current interpreter.
  The implementation uses local vector/register state; no RPCS3 code was
  copied.
- `runtime/spu/spu_interp.c`: interpreted `fscrrd` now clears all destination
  lanes, matching RPCS3's fast interpreter handler; `fscrwr` remains a no-op.
  The behavior was checked against `SPUInterpreter.cpp` at commit `2ba2f75bc`;
  local code was written independently.

## 2026-10-15 RSX depth clip reference

- `libs/video/rsx_commands.c`, `rsx_commands.h`, and `rsx_d3d12_backend.c`:
  stored `NV4097_SET_ZMIN_MAX_CONTROL` and mapped its depth-clip/clamp bits to
  D3D12 rasterizer state and the PSO cache key. The bit layout and backend
  behavior were checked against RPCS3 `rsx_decode.h`, `rsx_methods.h`, and
  `VKGSRender.cpp` at commit `2ba2f75bc`. Local state and D3D12 code were used;
  no RPCS3 function body was copied. The ignore-W bit has no D3D12 mapping.

## 2026-10-15 RSX current vertex attribute references

- `libs/video/rsx_commands.c` now decodes NV4097 current-attribute method files
  for 1F/2F/3F/4F, 2S/4S, scaled4S, and normalized 4UB values into the shared
  float4 register state. Method addresses, packing, component counts, and
  signed/normalized types were checked against RPCS3 `gcm_enums.h`,
  `NV47/HW/nv4097.h`, and `NV47/HW/common.cpp` at commit `2ba2f75bc`. The local
  decoder was written independently; no RPCS3 function body was copied.

## 2026-10-15 cellGcmSys mapping reference

- `libs/video/cellGcmSys.c` and `cellGcmSys.h` add
  `cellGcmMapEaIoAddressWithFlags` with NID `0x626E8518`. The accepted strict
  ordering flag, 1 MiB alignment rules, and EA/IO mapping behavior were checked
  against RPCS3 `cellGcmSys.cpp` and `gcm_enums.h` at commit `2ba2f75bc`. The
  local handler reuses its existing mapping path; strict cache ordering is
  implicit in the synchronous shared guest-memory/FIFO model. No RPCS3 source
  was copied.
- The same files add `cellGcmSetDefaultCommandBufferAndSegmentWordSize` with
  NID `0x172C3197`, and make the default-size getters return its configured
  values. Validation and state updates were checked against RPCS3
  `cellGcmSys.cpp` and `GCM.h` at commit `2ba2f75bc`; local code was written
  independently. The local FIFO's initial control pair is zero rather than
  RPCS3's `0x1000`, so both matching initial pairs are accepted.

## 2026-10-15 RSX notify reference

- `libs/video/rsx_commands.c`, `rsx_commands.h`, and `cellGcmSys.c` implement
  `NV4097_NOTIFY`: select one of the eight MAIN notify DMA slots, wait for the
  D3D12 backend to finish prior queued work, and write the timestamp/zero
  record into the mapped guest notify page at the RPCS3 0x40-byte stride.
  The register values, DMA handle decoding, guest address formula, and ordering
  were checked against RPCS3 `gcm_enums.h`, `NV47/HW/nv4097.cpp::set_notify`,
  `cellGcmSys.cpp`, and `Emu/Cell/lv2/sys_rsx.h` at commit `2ba2f75bc`.
  Guest memory translation and the monotonic-nanosecond timestamp are local
  adaptations; the local method/HLE code was written independently and no
  RPCS3 function body was copied. RPCS3 is GPL-2.0; this provenance note is not
  license clearance.

## 2026-10-15 NV0039 memory-transfer reference

- `libs/video/cellGcmSys.c` implements the `NV0039_BUFFER_NOTIFY` rectangle
  copy with independent input/output byte strides, row pitches, guest-range
  validation, overlap handling, and write-through guest-memory helpers. Register
  fields and transfer behavior were checked against RPCS3 `nv0039.cpp`,
  `rsx_decode.h`, and `gcm_enums.h` at commit `2ba2f75bc`. The standard
  `SET_OBJECT` handle values were checked against RPCS3's default RSX register
  initialization; arbitrary guest object handles are not yet mapped. The local
  code was written for ps3recomp's FIFO/VFS memory model; no RPCS3 function body
  was copied. RPCS3 is GPL-2.0; this behavior-reference note is not license
  clearance.

## 2026-10-16 RSX vertex address base references

- `libs/video/rsx_commands.c`, `rsx_commands.h`, and
  `rsx_d3d12_backend.c`: the direct indexed path now records
  `NV4097_SET_VERTEX_DATA_BASE_INDEX`, adds it to fetched indices in the RSX
  20-bit domain, and checks primitive restart against the raw element. The
  shared vertex fetch also records `NV4097_SET_VERTEX_DATA_BASE_OFFSET` and
  adds it to attribute offsets with the RSX 28-bit wrap before resolving the
  DMA address. Register decoding and address behavior were checked against
  RPCS3 `rsx_decode.h`, `rsx_methods.h`, `rsx_utils.h`, `RSXDrawCommands.cpp`,
  and `VKVertexBuffers.cpp` at commit `2ba2f75bc`. The local implementation was
  written independently; no RPCS3 function body was copied. RPCS3 is GPL-2.0;
  this behavior-reference note is not license clearance. Runtime use by the
  affected P5R draws has not yet been verified.

## 2026-10-19 RSX vertex-fetch format references

- `libs/video/rsx_vertex_fetch.c`: CMP X11/Y11/Z10 decoding, SNORM16 scaling,
  enabled-array component defaults, and zero-stride array behavior were checked
  against the local RPCS3 `Program/GLSLSnippets/RSXProg/RSXVertexFetch.glsl` and
  `Core/RSXDrawCommands.cpp`. RPCS3 is GPL-2.0. The scalar C implementation was
  written independently from the format rules; no RPCS3 function body was
  copied. The code carries `RPCS3_REFERENCE_REVIEW`; this records provenance
  and does not clear the combined project's existing licensing findings.

## 2026-10-19 SPU termination behavior

- `runtime/syscalls/lv2_register.c::sys_spu_thread_group_terminate_handler`:
  group termination now sets a cooperative stop request, wakes paused SPU
  workers, waits for active host workers to finish, and only then marks the
  group stopped. The behavior was checked against RPCS3
  `Emu/Cell/lv2/sys_spu.cpp` at commit `b1f8b02909` (GPL-2.0). The local code
  adapts the contract to ps3recomp's interpreter stop hook and thread-finish
  events; no RPCS3 function body was copied. The source block is marked
  `RPCS3_REFERENCE_REVIEWED`.

- RPCS3's 2026-10-19 RSX FIFO changes in `RSXFIFO.cpp` add cache trimming and
  saved positions around its asynchronous FIFO cache. ps3recomp's
  `cellGcmSys.c` reads one guest command at a time and does not use that cache
  or sync-point rollback path, so those changes have no direct equivalent to
  transplant. Its method decoding and jump/call handling remain separate
  implementation paths and need capture evidence before further changes.

- `libs/video/cellGcmSys.c::cellGcmInitSystemMode` applies RPCS3's mode-1
  512 MiB RSX I/O-space selection (default remains 256 MiB), and
  `runtime/ppu/ppu_sysprx.cpp` registers the context-aware import. The behavior
  was checked against RPCS3 `cellGcmInitSystemMode` and `_cellGcmInitBody` at
  commit `a99afd4cf` (GPL-2.0). The local implementation was written against
  ps3recomp's mapping tables; no RPCS3 function body was copied.

- `libs/video/cellGcmSys.c` handles the report DMA context and `NV4097_GET_REPORT`
  writes using ps3recomp's guest-memory mapping. P5's
  `run-depth-coherence-ffmpeg-20261019/console-attached.log` shows repeated
  report types 2 and 3, so this path is used by the release. The behavior was
  checked against RPCS3 `Emu/RSX/NV47/HW/nv4097.cpp` and `common.cpp` at
  `a99afd4cf` (GPL-2.0). No RPCS3 function body was copied; the trace does not
  identify the selected report DMA destination.

- `libs/video/rsx_d3d12_backend.c` now indexes P5's active fragment-program
  PSOs by a hash of the complete existing cache key, with equality checks and
  bounded ring eviction. The cache shape follows the hash-keyed pipeline cache
  pattern in RPCS3 `Emu/RSX/VK/VKShaderInterpreter.h` at `a99afd4cf` (GPL-2.0),
  adapted to ps3recomp's D3D12 API. No RPCS3 source body was copied. The release
  build passes; runtime performance has not yet been remeasured.

## Limits

This is a provenance audit from tracked repository references, commit history,
and targeted source comparison. It is not legal advice, a whole-tree
code-similarity scan, or proof about undocumented authoring history. It cannot
certify that every unmarked implementation was independently written. The
confirmed ports and review flags above should remain visible until their
provenance and licensing are resolved.
