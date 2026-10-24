# Programs collected before guest execution

Both P5 runners scan their executable (including the embedded PPU image), any
explicit boot ELF, and regular files under the game VFS root before starting
the guest. Linked/reparse child directory trees and the output cache are
excluded to avoid cycles. No Python installation or PS3 SDK is required at launch.

Outputs use `%LOCALAPPDATA%\ps3recomp\cache` on Windows, or the same root selected
by `PS3RECOMP_CACHE_DIR` for the D3D12 and SPU runtime caches:

- `shaders/rsx`: validated Cg revision 6/7 and CGB RSX vertex and fragment binaries,
  plus their original microcode. These are raw programs, not D3D12 pipelines.
- `spu_jobs/boot`: embedded SPU ELF files and explicitly labeled raw SPURS
  regions. JOBCRT-derived lengths remain candidates, not verified job sizes.
- `boot/<game-path-hash>`: per-source JSONL inventories, scan stamps,
  `errors.jsonl`, and `summary.json`.

The scan examines uncompressed archive bytes and recursively decodes CRILAYLA
members, with checked offsets/backreferences, a 256 MiB decoded-member limit
and a four-level nesting limit. Unsupported, malformed, and oversized
CRILAYLA members are recorded in the inventory. Other compressed or encrypted
formats and programs assembled dynamically are not discoverable by this scan.
Consequently the report never claims complete game coverage.
Rejected compression signatures can also be literal magic strings in code or
data; that count does not establish that actual archive members were omitted.

Unchanged source files reuse their inventories at later boots. A missing or
wrong-size extracted file invalidates its source stamp. Delete the game's
`boot/<game-path-hash>` directory to force a complete rescan. The first scan can
take time for large archives; later boots still enumerate the source files but
avoid rereading unchanged contents. Source assets are never modified.

`PS3RECOMP_BOOT_DUMP=0` disables the boot scan. `SPU_DUMP_JOB_IMAGES=0` disables
both boot SPU discovery and runtime job-image collection.
`RSX_DUMP_SHADER_IMAGES=0` disables boot shader discovery. Runtime SPU captures
are enabled by default and continue collecting exact dispatched images under
`spu_jobs`, covering programs the boot scan cannot discover. HLSL/DXBC caching
and D3D12 pipeline-library reuse continue independently; rendering state is
still needed to specialize shader variants and create pipelines.

An extracted raw region does not become executable merely by being cached.
Native SPU lifting still requires verified job size, entry, load address, ABI
and dependencies. Disk discovery does not execute jobs or register guesses.
