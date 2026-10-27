"""Owned local Ghidra worker; no game executable is run."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from .config import sha256
from .engine import bind_engine


def inventory_reference(request):
    import pyghidra
    pyghidra.start(install_dir=request["ghidra"])
    from java.io import File
    from java.lang import Object
    from ghidra.framework.store.db import PackedDatabase
    from ghidra.framework.data import OpenMode
    from ghidra.program.database import ProgramDB
    from ghidra.util.task import TaskMonitor
    from p5_analysis.ghidra import java_items
    binary = Path(request["binary"])
    if binary.suffix.lower() != ".gzf":
        raise ValueError("Reference inventory currently requires a local Ghidra .gzf archive")
    cache = Path(request["workspace"]) / "reference_cache" / request["expected_sha256"]
    cache.mkdir(parents=True, exist_ok=True)
    private = cache / "reference.gzf"
    if not private.exists():
        shutil.copyfile(binary, private)
    if sha256(private) != request["expected_sha256"]:
        raise ValueError("Reference cache identity changed")
    packed = PackedDatabase.getPackedDatabase(File(str(private)), TaskMonitor.DUMMY)
    program, consumer = None, Object()
    try:
        program = ProgramDB(packed.openForUpdate(TaskMonitor.DUMMY), OpenMode.UPGRADE, TaskMonitor.DUMMY, consumer)
        rows = [{"address": int(fn.getEntryPoint().getOffset()), "name": str(fn.getName()),
                 "size": int(fn.getBody().getNumAddresses())}
                for fn in java_items(program.getFunctionManager().getFunctions(True)) if not fn.isExternal()]
        return {"functions": rows, "program_sha256": str(program.getExecutableSHA256()), "language": str(program.getLanguageID())}
    finally:
        if program is not None:
            program.release(consumer)
        packed.dispose()


def execute(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    workspace = Path(request["workspace"]).resolve()
    output = Path(request["output"]).resolve()
    if not output.is_relative_to(workspace) or not request_path.is_relative_to(workspace):
        raise ValueError("Worker outputs must stay inside the selected workspace")
    binary = Path(request["binary"]).resolve()
    if sha256(binary) != request["expected_sha256"]:
        raise ValueError("Worker input changed before analysis")
    bind_engine(request["project"])
    if request["action"] == "inventory":
        result = inventory_reference(request)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif request["action"] == "export":
        from p5_analysis.ghidra import standalone_export
        result = standalone_export(SimpleNamespace(
            project=Path(request["project"]), ghidra=Path(request["ghidra"]), binary=binary,
            output=output, language=request["language"], functions=request["functions"],
            all=False, no_decompile=False))
    else:
        raise ValueError("Unknown worker action")
    if sha256(binary) != request["expected_sha256"]:
        raise ValueError("Original input changed during local analysis")
    result["output_sha256"] = sha256(output)
    output.with_suffix(output.suffix + ".result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
