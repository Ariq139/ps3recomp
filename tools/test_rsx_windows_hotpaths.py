"""Native Windows regressions for RSX FIFO and D3D12 CPU hot paths."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", help="Path to clang.exe")
args = parser.parse_args()
if os.name != "nt":
    raise SystemExit("This runner requires Windows and the Windows SDK.")
compiler = args.compiler or shutil.which("clang")
if not compiler:
    cache = root / "p5_feb16/build-clang-vs/CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_C_COMPILER:"):
                compiler = str(Path(line.split("=", 1)[1]).with_name("clang.exe"))
                break
if not compiler or not Path(compiler).exists():
    raise SystemExit("Pass --compiler with the path to clang.exe.")

def run(command, env=None):
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90, env=env)
    if result.returncode:
        print(result.stdout[-4000:])
        print(result.stderr[-6000:])
        raise SystemExit(result.returncode)
    return result.stdout.strip()

with tempfile.TemporaryDirectory(prefix="ps3recomp-rsx-tests-") as folder:
    folder = Path(folder)
    report_stub = folder / "reports.c"
    report_stub.write_text("#include <stdint.h>\n"
        "int cellGcmGetReportValueAtOffset(uint32_t o,uint32_t* v)"
        "{(void)o;if(v)*v=0;return 0;}\n")
    fifo_stubs = folder / "fifo_stubs.c"
    fifo_stubs.write_text("#include <stdint.h>\n"
        "void rsx_d3d12_composite_from_offscreen(uint32_t o){(void)o;}\n"
        "void rsx_draw_engine_set_display_buffer(uint32_t b,uint32_t l,uint32_t o,"
        "uint32_t p,uint32_t w,uint32_t h){(void)b;(void)l;(void)o;(void)p;(void)w;(void)h;}\n"
        "void rsx_live_draw_set_display_buffer(uint32_t b,uint32_t l,uint32_t o,"
        "uint32_t p,uint32_t w,uint32_t h){(void)b;(void)l;(void)o;(void)p;(void)w;(void)h;}\n"
        "int rsx_replay_skipped_sync_method(uint32_t s,uint32_t m,uint32_t d)"
        "{(void)s;(void)m;(void)d;return 0;}\n")
    cases = [
        ("vertex_decode_cache", [], []),
        ("constant_snapshot_cache", [], []),
        ("raw_upload_cache", [], []),
        ("vertex_raw_gpu", ["libs/video/rsx_vertex_fetch.c",
            "libs/video/rsx_vp_decompiler.c"], ["-ld3d12", "-ldxgi", "-ld3dcompiler"]),
        ("vertex_fetch_prepared", ["libs/video/rsx_vertex_fetch.c"], []),
        ("d3d12_srv_cache", [], []),
        ("d3d12_shadow_border", [], []),
        ("d3d12_srv_cache_gpu", [], ["-ld3d12", "-ldxgi"]),
        ("d3d12_retire_gpu", [], ["-ld3d12", "-ldxgi"]),
        ("fp_runtime_analysis", ["libs/video/rsx_fp_decompiler.c"], []),
        ("gcm_nop_runs", [], ["-luser32"]),
        ("gcm_ordering", [], ["-luser32"]),
        ("gcm_vblank_clock", [], ["-luser32"]),
        ("vertex_fetch_plan", ["libs/video/rsx_vertex_compact.c",
            "libs/video/rsx_dispatch.c", str(report_stub)], []),
        ("gcm_fifo_user_command", [str(fifo_stubs)], ["-luser32"]),
    ]
    for name, sources, libraries in cases:
        binary = folder / (name + ".exe")
        run([compiler, "-std=gnu17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
            "-ffunction-sections", "-fdata-sections", "-fuse-ld=lld", "-Wl,/opt:ref",
            "-Iinclude", "-Ilibs/video", f"libs/video/tests/test_{name}.c",
            *sources, "-o", str(binary), *libraries])
        output = run([str(binary)])
        print(f"{name}: {output.splitlines()[-1]}")
        if name == "vertex_raw_gpu":
            output = run([str(binary), "--indexed"])
            print(f"{name} indexed: {output.splitlines()[-1]}")
        if name in ("gcm_nop_runs", "gcm_ordering"):
            output = run([str(binary)], env={**os.environ, "GCM_WORD_STATS": "60"})
            print(f"{name} diagnostic path: {output.splitlines()[-1]}")
        if name == "gcm_ordering":
            output = run([str(binary), "--bypass"])
            print(f"{name} bypass: {output.splitlines()[-1]}")
    print(f"{len(cases) + 4} RSX Windows regressions passed")
