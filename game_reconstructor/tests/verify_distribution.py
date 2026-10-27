"""Build and verify an isolated distribution using installed local tools only."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def verify(native=False):
    from setuptools import build_meta
    from reconstructor.config import Config
    work_root = ROOT / "build_checks"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="distribution-", dir=work_root)).resolve()
    assert work.is_relative_to(work_root.resolve())
    env = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    checks = []
    def command(name, args, cwd, environment=None):
        log = work / (name + ".log")
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(args, cwd=cwd, env=environment or env, text=True,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {log}\n{log.read_text(encoding='utf-8')[-2500:]}")
        checks.append({"name": name, "passed": True, "log": str(log)})
        return log

    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with (work / "build.log").open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            wheel_name = build_meta.build_wheel("dist")
            source_name = build_meta.build_sdist("dist")
    finally:
        os.chdir(previous)
    wheel, source = ROOT / "dist" / wheel_name, ROOT / "dist" / source_name
    checks.append({"name": "wheel_and_source_build", "passed": True, "log": str(work / "build.log")})
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {"reconstructor/__main__.py", "reconstructor/controller.py",
                    "reconstructor/analysis_engine/p5_analysis/ghidra.py", "reconstructor/profiles/p5_npeb02436.json"}
        if not required <= names:
            raise RuntimeError("Wheel omitted required runtime files")
        if any("workspaces" in Path(name).parts or "build_checks" in Path(name).parts for name in names):
            raise RuntimeError("Wheel contains generated analysis data")
    checks.append({"name": "wheel_runtime_contents", "passed": True})
    extracted = work / "source"
    extracted.mkdir()
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        for member in members:
            target = (extracted / member.name).resolve()
            if not target.is_relative_to(extracted.resolve()) or not (member.isfile() or member.isdir()):
                raise RuntimeError("Unsafe source archive member")
        archive.extractall(extracted, filter="data")
    source_root = next(extracted.iterdir())
    for name in ("Start.cmd", "Start.pyw", "reconstruct.py"):
        if not (source_root / name).is_file():
            raise RuntimeError("Source archive omitted its launcher")
    rebuilt = work / "rebuilt"
    rebuilt.mkdir()
    command("source_archive_rebuild", [sys.executable, "-c", "from setuptools import build_meta; build_meta.build_wheel(" + repr(str(rebuilt)) + ")"], source_root)
    installed = work / "installed"
    command("offline_install", [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(installed), str(wheel)], work)
    isolated_env = {**env, "PYTHONPATH": str(installed)}
    command("installed_help", [sys.executable, "-m", "reconstructor", "--help"], installed, isolated_env)
    command("installed_gui", [sys.executable, "-m", "reconstructor", "gui", "--check"], installed, isolated_env)
    command("installed_setup", [sys.executable, "-m", "reconstructor", "doctor"], installed, isolated_env)
    check_code = """
import importlib.metadata as metadata
from pathlib import Path
from reconstructor.config import ROOT
assert ROOT == Path.cwd(), (ROOT,Path.cwd())
entries = {entry.name:entry.value for entry in metadata.distribution('offline-game-reconstructor').entry_points}
assert entries['game-reconstructor'] == 'reconstructor.cli:main'
assert entries['offline-game-reconstructor'] == 'reconstructor.cli:gui_main'
print('Installed package and both entry points verified:',ROOT)
"""
    command("installed_entry_points", [sys.executable, "-c", check_code], installed, isolated_env)
    if native:
        profile = Config.load(ROOT / "reconstructor/profiles/p5_npeb02436.json")
        workspace = work / "native"
        workspace.mkdir()
        native_code = """
import json,sys
from pathlib import Path
from reconstructor.config import Config,sha256
from reconstructor.controller import Controller
from reconstructor.offline import enforce_offline
enforce_offline()
config=Config(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
config.validate()
binary=config.project/'game/EBOOT.ELF'
controller=Controller(config,emit=lambda event:None)
output,result=controller.worker({'action':'export','binary':str(binary),'expected_sha256':sha256(binary),
                                 'functions':['0x001F3CE0'],'profile':'installed-package-build-check'})
rows=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
functions=[row for row in rows if row.get('kind')=='function']
assert len(functions)==1
assert functions[0]['coverage']['complete']
assert functions[0]['decompilation_status']=='completed'
assert rows[-1]['kind']=='complete'
print(json.dumps({'installed_worker':True,'native_function':'0x001F3CE0','complete_instruction_coverage':True,
                  'decompilation':'passed','game_build_run':False},indent=2))
"""
        command("installed_native_worker", [sys.executable, "-c", native_code, str(profile.project), str(profile.ghidra), str(workspace)], installed, isolated_env)
    report = {"passed": True, "wheel": str(wheel), "source_archive": str(source),
              "checks": checks, "game_build_run": False, "standalone_executable_built": False}
    destination = work / "verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "checks": len(checks), "report": str(destination)}, indent=2))
    return report


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Also decompile one real function through the installed worker")
    verify(parser.parse_args().native)
