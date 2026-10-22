# Getting Started with ps3recomp

Step-by-step guide to recompiling your first PS3 game.

## Prerequisites

### Required Software

- **CMake** 3.20 or later: [cmake.org](https://cmake.org/download/)
- **C/C++ compiler** with C17 and C++20 support:
  - Windows: Visual Studio 2022 with the **Desktop development with C++** workload. MSVC works; `clang-cl` is the more commonly used compiler for game projects.
  - Linux: GCC 12+ or Clang 14+
  - macOS: Xcode 14+ (Apple Clang)
- **SDL2 development files** on Linux/macOS: required for the runtime's
  controller and audio backends (`sudo apt install libsdl2-dev` on Debian or
  Ubuntu; `brew install sdl2` on macOS).
- **Python** 3.9+: Required for the recompiler tools
- **Ninja** (recommended): Faster than Make/MSBuild -- `pip install ninja`

On Windows, run CMake from **Developer PowerShell for VS 2022** (or an x64
Native Tools Command Prompt). This sets up the compiler, Windows SDK, and
resource tools. To use `clang-cl`, install the **C++ Clang tools for Windows**
component in Visual Studio Installer, and check that `Get-Command clang-cl`
finds it. Ninja is a build generator; it does not install or locate a C/C++
compiler for you.

### Required Game Files

- A **decrypted PS3 EBOOT.BIN** (ELF format). You need:
  - A legally obtained copy of the game
  - The decryption keys (from your own console or PS3 firmware)
  - RPCS3's decrypted-ELF dump, or `tools/unfself.py` for an unencrypted
    debug/prototype fSELF (see Step 1)

### Python Dependencies

```bash
pip install -r tools/requirements.txt
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sp00nznet/ps3recomp.git
cd ps3recomp
```

### 2. Optional: Build the Runtime Library Separately

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

On Windows, run those commands in Developer PowerShell for VS 2022. To select
`clang-cl` explicitly, use a new build directory:

```powershell
cmake -S . -B build-clang -G Ninja `
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build-clang
```

For MSVC (`cl`), omit the compiler options and use a separate `build-msvc`
directory. CMake caches the selected compiler, so do not reuse a build folder
configured with a different compiler. You can skip this standalone runtime
build when using the project template below; its CMake project builds the
runtime from source automatically.

This builds `libps3recomp_runtime.a` (or `ps3recomp_runtime.lib` on Windows), which contains the memory manager, thread manager, and all HLE library stubs.

### 3. Runtime Library Output

The runtime library is under the build directory (`libps3recomp_runtime.a` on
Linux/macOS, `ps3recomp_runtime.lib` on Windows).

The game runner shows warning/error-class diagnostics by default and keeps the
guest TTY stream in `ps3_tty.log` in the launch directory. Japanese TTY text is
decoded from CP932 to UTF-8. Set `PS3_TTY_ENCODING=utf8` if the title writes
UTF-8, or `PS3_TTY_ENCODING=raw` to preserve the original bytes. Set
`PS3_TTY_LOG=off` to discard TTY output, or set it to a path to change the log
file. For more host output, set `PS3_LOG_LEVEL=info`, `debug`, or `trace`; the
default is `warning`. `PS3_VERBOSE=1` enables high-volume trace sites and is
most useful with `PS3_LOG_LEVEL=debug`. Warning/error recognition follows
standard severity labels and failure terms; untagged stdout is treated as info
and untagged stderr as debug. Set `PS3_TTY_STDOUT=1` to echo decoded TTY text
back to the launch console.

## First Recompilation Walkthrough

The whole pipeline is four commands. `tests/run_torture.py` runs exactly this
chain end to end on every change, so it is the in-tree reference if a command
here ever stops matching the tools:

```bash
python tools/ppu_loader.py  game/EBOOT.ELF -o out/          # 1. image, OPD functions, imports
python tools/ppu_lifter.py  game/EBOOT.ELF --functions out/EBOOT.functions.json \
                            --hle-stubs out/EBOOT.imports.json -o recompiled/   # 2. lift
python tools/gen_hle_nids.py --all --out recompiled/ppu_hle_nids.cpp   # 3. HLE handler table
cmake -B build -G Ninja && cmake --build build                      # 4. build
```

Steps 2 and 3 are the two halves of firmware imports, and leaving out either
one produces the *same* startup failure. They are described in full below.

### Step 1: Get a decrypted ELF

PS3 executables ship as SELF. The recompiler needs the plain ELF inside:

- **Retail `EBOOT.BIN`** — encrypted. Dump it with RPCS3 (it writes the
  decrypted ELF to its cache when it loads a game), or decrypt it yourself with
  keys from your own console. This repository ships no keys and does no
  decryption.
- **Debug / prototype fSELF** — unencrypted, and often carries symbols. Rebuild
  the original ELF with no keys at all:

```bash
cmake -E make_directory game
python tools/unfself.py EBOOT.BIN --output game/EBOOT.ELF
python tools/unfself.py EBOOT.BIN --info       # describe it, write nothing
```

See [PROTO_BUILDS.md](PROTO_BUILDS.md) for why a debug build is worth hunting
for: symbols make every later step checkable.

### Step 2: Create a project

From the repository root, copy the starter project. `cmake -E` works in
PowerShell, Command Prompt, and POSIX shells:

```bash
cmake -E copy_directory templates/project my_game
```

Edit `my_game/CMakeLists.txt` to set the project name. `config.toml` is a
reference for port settings; the loader and lifter commands below use their
explicit command-line paths.

### Step 3: Load the ELF — functions, imports, TOC

```bash
python tools/ppu_loader.py game/EBOOT.ELF -o my_game/out/
```

This is stage 1 of the pipeline and it writes everything the lifter needs:

| File | Contents |
|------|----------|
| `EBOOT.functions.json` | `[{start,end,toc,opd}]` — one entry per OPD descriptor |
| `EBOOT.imports.json` | `[{library,nid,stub}]` — every firmware import and the address of its trampoline |
| `EBOOT.image.json` | PT_LOAD segment manifest (vaddr/filesz/memsz/flags) |
| `EBOOT.loader.json` | entry OPD → (code, TOC), module TOC, OPD extent, counts |

It prints a summary as it goes — entry point, module TOC (`r2`), segment layout,
function count and imports per library. Those numbers are the first sanity check
on a title: no OPD found, or a suspiciously small function count, means the
input is not what you think it is.

Function boundaries come from the `.opd` table, which lists a descriptor for
every address-taken function — far better than scanning for prologues.
`tools/find_functions.py` is the heuristic scanner for binaries where that is
not enough (it does prologue detection, leaf detection and branch-target
detection, and takes `--seed-json` to merge in what the loader found).

`tools/gen_imports.py` produces the same import list with the NIDs resolved to
names where the database knows them, which is useful to read but not required
by the lifter:

```bash
python tools/gen_imports.py game/EBOOT.ELF -o my_game/out/imports_named.json
# 1 libraries, 12 imports, 12 named (100%) -> out/imports_named.json
```

### Step 4: Lift PPU code to C++

```bash
python tools/ppu_lifter.py game/EBOOT.ELF \
    --functions my_game/out/EBOOT.functions.json \
    --hle-stubs my_game/out/EBOOT.imports.json \
    --output my_game/recompiled/
```

This writes `ppu_recomp.h` plus `ppu_recomp_NNN.cpp` chunks (which can total
100 MB+ for a large title) and the `ppu_recomp_register()` the template's
`main.cpp` calls. No post-processing step is needed; the lifter emits C++
directly.

**`--hle-stubs` is not optional for a real title.** Without it the lifter
translates each firmware import trampoline literally — `li r12,0; oris r12,r12,hi;
lwz r12,lo(r12); lwz r0,0(r12); mtctr r0; bctr` — against an import table that,
in a file lifted offline, no host loader ever patched. The guest then branches to
whatever word it read, which is the next trampoline's own first instruction:

```
[ppu] unresolved indirect call -> 0x39800000 (tid=1 lr=0x000103C0)
```

`0x39800000` is not an address, it is the PowerPC instruction `li r12,0`. With
`--hle-stubs` each stub is instead emitted as its own function whose body is
`ps3_hle_call(<nid>, ctx)`, so both direct `bl` and indirect calls to it reach
the HLE handler.

### Step 5: Generate the HLE handler table

Lifting produces the game's own code. It does **not** produce the bridge from
firmware NIDs to this runtime's HLE implementations — that is a separate
generated file:

For a custom build system, generate the table into the lifted output directory
so it can be compiled with the lifted code:

```bash
python tools/gen_hle_nids.py --all --out my_game/recompiled/ppu_hle_nids.cpp
```

Compile the result into the port along with the lifted sources. It defines
`ppu_hle_register_all()`, which `ppu_hle_init()` calls at startup;
`runtime/ppu/ppu_hle.cpp` carries a weak do-nothing version so a build without
it still links, which is why the omission is not a build error. The runtime
warns at startup when it ends up with no handlers registered.

**The project template does this for you.** Its `CMakeLists.txt` generates the
every-module table during the build unless
`my_game/recompiled/ppu_hle_nids.cpp` already exists. You do not need to run
the generator manually when using the template.

Regenerate it whenever you update ps3recomp: the table is generated from the
toolkit's registered modules, so one built against a different revision can
reference handlers this one does not have.

**The two failures look identical, and both print `0x39800000`.** No handler
table means every import returns 0; no `--hle-stubs` means the call never
reaches the table in the first place. Do both.

### Step 6: Build the Recompiled Game

On Linux/macOS, configure from the repository root with:

```bash
cmake -S my_game -B my_game/build -G Ninja \
    -DPS3RECOMP_DIR="$PWD" \
    -DRECOMP_DIR="$PWD/my_game/recompiled"
cmake --build my_game/build
```

On Windows, use Developer PowerShell for VS 2022. For `clang-cl`, configure
with a fresh build folder and explicit compiler names:

```powershell
$repoRoot = (Get-Location).Path
cmake -S my_game -B my_game/build-clang -G Ninja `
  "-DPS3RECOMP_DIR=$repoRoot" `
  "-DRECOMP_DIR=$repoRoot/my_game/recompiled" `
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build my_game/build-clang
```

MSVC's `cl.exe` is not supported yet: the runtime and lifted code use compiler
builtins that it does not provide. Use `clang-cl` from the Visual Studio Clang
Tools component; it uses the same Windows SDK and Visual Studio libraries. The
template adds `/bigobj` for large generated files when using this compiler.

### Step 7: Run

On Windows, the project CMake embeds the configured decrypted ELF into the
executable as a resource and loads its original segments into guest memory at
startup. Replace `MyGameRecomp` below with the executable name derived from the
project name in `my_game/CMakeLists.txt` (and use the matching build directory
if you chose MSVC instead of `clang-cl`):

```bash
./my_game/build/MyGameRecomp game/EBOOT.ELF
```

Most ports also need to know where the title's files live, so that guest mount
points such as `/dev_hdd0` resolve to a real directory. The in-tree ports derive
that from the ELF path and accept `PS3_VFS_ROOT` as an override:

`PS3_VFS_ROOT` must point to the directory that contains `PS3_GAME`. In
PowerShell, set it and launch the program like this:

```powershell
$env:PS3_VFS_ROOT = 'D:\path\to\game-root'
& '.\my_game\build-clang\MyGameRecomp.exe'
```

On Linux/macOS:

```bash
PS3_VFS_ROOT=/path/to/game-root ./my_game/build/MyGameRecomp game/EBOOT.ELF
```

The explicit ELF argument remains necessary on non-Windows hosts unless the
port has configured a default `PORT_PPU_ELF` path.

You'll see:
- `[init]` messages as the runtime initializes (VM, syscalls, HLE modules)
- `[HLE]` messages as PS3 API functions are called
- `[LV2]` messages for unimplemented system calls
- A window opens for RSX output -- Direct3D 12 on Windows, Metal on macOS. See
  [Graphics backends](BUILDING.md#graphics-backends) for what each platform gets.

The template chooses the renderer at compile time from the host platform;
`config.toml`'s `graphics.backend` setting does not switch it. Windows uses
D3D12. The Vulkan backend is currently an early, opt-in Linux runtime path and
is not connected to the template game's renderer selection.

### Controller input

The template's `config.toml` lists `cellPad = "hle"`; the host input backend is
selected by platform: XInput on Windows and SDL2 GameController on Linux/macOS.
There is no per-title controller-mapping setting in the template. The default
Xbox-to-PS3 button mapping is documented in [Game Porting Guide: Input](GAME_PORTING_GUIDE.md#input),
and platform details are in [Platform Abstraction: Gamepad Backends](PLATFORM_ABSTRACTION.md#gamepad-backends).

On Windows, if XInput port 0 has no controller, `cellPad` also accepts this
built-in keyboard layout: **arrow keys** for the D-pad, **Z/X/A/S** for
Cross/Circle/Square/Triangle, **Q/W** for L1/R1, **1/2** for L2/R2, **Enter**
for Start, and **Tab** for Select. The left stick mirrors the arrow keys. This
feeds `cellPad` directly; it does not create a system-wide virtual XInput
device. Set `PAD_NO_KEYBOARD=1` before launching to disable the fallback. See
[Keyboard fallback](PLATFORM_ABSTRACTION.md#keyboard-fallback-windows) for details.

**Expected first-run behavior:** CRT startup executes, the game calls `cellSysmoduleLoadModule` to load libraries, then begins initialization. Missing HLE functions show as `[HLE] UNIMPLEMENTED` — add bridges in your game's `hle_modules.cpp`.

## Project Structure

After setup, your project should look like:

```
my_game/
  CMakeLists.txt          # Build configuration
  config.toml             # Port settings reference
  main.cpp                # Entry point (from template)
  stubs.cpp               # Game-specific overrides
  out/                    # ppu_loader.py output (my_game/out/)
    EBOOT.functions.json
    EBOOT.imports.json
    EBOOT.image.json
    EBOOT.loader.json
  recompiled/              # ppu_lifter.py output (my_game/recompiled/)
    ppu_recomp.h
    ppu_recomp_000.cpp
    ppu_recomp_001.cpp
    ...
  build/                   # CMake build directory; includes generated HLE table
```

## Troubleshooting

### `[ppu] unresolved indirect call -> 0x39800000`

The single most common first-port failure, and it has two independent causes —
check both. `0x39800000` is the instruction `li r12,0`, the first word of a
firmware import trampoline, so the guest is calling an import and landing in
raw stub bytes:

1. The lift was run without `--hle-stubs` (Step 4), so import trampolines were
   translated as literal code instead of `ps3_hle_call`.
2. `ppu_hle_nids.cpp` was never generated or never compiled in (Step 5), so no
   handler is registered for any NID.

The runtime names whichever of the two it can see at the point of failure.

### `[HOTREAD] spinning on 0xADDR (=0xVALUE)`

The guest is polling one word of its own memory and the value is not changing —
a mutex, a flag an SPU job was supposed to set, an RSX `get` pointer. It is a
report, not a crash, and some titles legitimately spin during startup; it only
matters if nothing else advances afterwards. The address is the thing to chase:
find who is supposed to write it.

### "Unimplemented NID 0xXXXXXXXX in module YYY"

The game is calling a PS3 API function that does not have an HLE handler. First
establish its ABI and caller-visible effects, then implement the behavior. Use
a documented placeholder only when the call is proven irrelevant to this path
or its behavior remains unknown.
1. Look up the NID: `python tools/nid_database.py --lookup 0xXXXXXXXX`.
2. Check the SDK-facing declaration and local callers to establish its ABI and
   required outputs and side effects.
3. Check RPCS3's source as a behavior reference, then implement the observable
   contract. Add a documented placeholder only when evidence shows the call is
   irrelevant to this path or its behavior is still unknown.

### `/* sync: cache/sync — no-op */` in the lifted output

Expected. `sync`, `lwsync`, `isync`, `eieio` and the `dcb*` cache hints have no
host equivalent worth emitting: the lifted code runs on a host with a stronger
memory model, and the ordering that actually matters — `lwarx`/`stwcx.` — is
lifted to real host atomics rather than to fences. A spin loop around
`lwarx/stwcx.` that never exits is not this comment's fault; it means the value
the loop waits on is never written (see `[HOTREAD]` above).

### Crash on startup / segfault

- Enable `break_on_unimplemented = true` in `config.toml` and run under a debugger.
- Check that the ELF was properly decrypted (wrong keys produce garbage code).
- Verify memory layout: some games require specific address ranges.

### Graphics not rendering

- The RSX command processor tracks GPU state but needs a rendering backend.
- Start with the **null backend** (`rsx_null_backend_init()`) to verify command flow.
- Switch to the **D3D12 backend** (`rsx_d3d12_backend_init()`) for real GPU rendering.
- See [RSX_GRAPHICS.md](RSX_GRAPHICS.md) for the full graphics architecture.

### CMake cannot find a Windows compiler

- Open **Developer PowerShell for VS 2022** before configuring; a regular
  PowerShell window does not have the Visual Studio compiler environment.
- Check `Get-Command cl, clang-cl, ninja`. If `cl` is missing, install the
  Visual Studio **Desktop development with C++** workload. If `clang-cl` is
  missing, install the Visual Studio Clang tools component or add LLVM's `bin`
  directory to `PATH`.
- If using MSVC, remove the `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`
  overrides. Configure into a fresh build directory after changing compilers;
  CMake caches its compiler selection.

### Audio crackling or silence

- Check that `cellAudio` is set to `"hle"` in modules config.
- Verify the audio backend matches your OS (`wasapi` for Windows, `pulseaudio` for Linux).
- Increase `buffer_size` to reduce crackling (at the cost of latency).

### Recompiler fails on indirect branches

- Some indirect branches (function pointers, virtual calls, switch tables) cannot be resolved statically.
- The recompiler emits a runtime dispatch table for these cases.
- If a jump target is missing, add it manually to the function table.

### Build errors in generated code

- Ensure your compiler supports C17 and C++20.
- The template adds `/bigobj` for the large lifted translation units.

---

## Next Steps

Once you have a basic build working:

- **[Game Porting Guide](GAME_PORTING_GUIDE.md)** — Full 12-phase walkthrough with [flOw case study](GAME_PORTING_GUIDE.md#case-study-flow)
- **[Custom Modules](CUSTOM_MODULES.md)** — How to write new HLE modules
- **[RSX Graphics](RSX_GRAPHICS.md)** — RSX command processor, D3D12/Vulkan backends, shader translation
- **[FAQ & Troubleshooting](FAQ.md)** — Common issues and solutions
- **[Contributing](../CONTRIBUTING.md)** — How to contribute code and docs

Reference docs:
- **[Runtime](RUNTIME.md)** — VM, PPU/SPU contexts, types, syscalls
- **[Syscalls](SYSCALLS.md)** — LV2 kernel syscall implementations
- **[NID System](NID_SYSTEM.md)** — PS3 function linking and NID resolution
- **[Module Reference](MODULES_REFERENCE.md)** — All 95 HLE modules documented
- **[Module Status](MODULE_STATUS.md)** — Quick coverage check
- **[Tools](TOOLS.md)** — Recompiler pipeline tools reference
- **[Building](BUILDING.md)** — Build system and compiler notes
- **[Architecture](ARCHITECTURE.md)** — Cell processor and pipeline overview
- **[Platform Abstraction](PLATFORM_ABSTRACTION.md)** — Win32/POSIX cross-platform details
