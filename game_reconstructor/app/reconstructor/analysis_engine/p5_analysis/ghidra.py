"""Export observed Ghidra facts without changing the supplied executable."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from .common import address, digest, file_hash, json_text


def _bytes(values):
    return bytes(int(v) & 255 for v in values)


def java_items(iterator):
    while iterator.hasNext():
        yield iterator.next()


def type_records(roots, limit=1024):
    from ghidra.program.model.data import Array, Composite, Enum, Pointer, TypeDef
    queue, seen, records = list(roots), set(), []
    while queue and len(records) < limit:
        dtype = queue.pop()
        if dtype is None:
            continue
        path = str(dtype.getPathName())
        if path in seen:
            continue
        seen.add(path)
        archive = dtype.getSourceArchive()
        record = {"kind": "data_type", "path": path, "name": str(dtype.getName()),
                  "length": int(dtype.getLength()), "storage_kind": str(dtype.getClass().getSimpleName()),
                  "source_archive": str(archive.getName()) if archive is not None else None,
                  "status": "ghidra_observed_layout"}
        if isinstance(dtype, Pointer):
            base = dtype.getDataType()
            record["points_to"] = str(base.getPathName()) if base is not None else None
            queue.append(base)
        elif isinstance(dtype, Array):
            base = dtype.getDataType()
            record.update(element_type=str(base.getPathName()), count=int(dtype.getNumElements()), stride=int(dtype.getElementLength()))
            queue.append(base)
        elif isinstance(dtype, TypeDef):
            base = dtype.getDataType()
            record["base_type"] = str(base.getPathName())
            queue.append(base)
        elif isinstance(dtype, Composite):
            components = dtype.getDefinedComponents() if hasattr(dtype, "getDefinedComponents") else dtype.getComponents()
            record["components"] = []
            for component in components:
                base = component.getDataType()
                field = component.getFieldName()
                record["components"].append({"offset": int(component.getOffset()), "length": int(component.getLength()),
                                              "field": str(field if field is not None else component.getDefaultFieldName()),
                                              "name_status": "stored_name" if field is not None else "generated_default",
                                              "type": str(base.getPathName())})
                queue.append(base)
        elif isinstance(dtype, Enum):
            record["values"] = {str(name): int(dtype.getValue(name)) for name in dtype.getNames()}
        records.append(record)
    return records, bool(queue)


def export_program(program, output, selected, monitor, decompile=True, origin=None):
    from ghidra.app.decompiler import DecompInterface
    from ghidra.program.model.block import BasicBlockModel

    listing = program.getListing()
    manager = program.getFunctionManager()
    memory = program.getMemory()
    language = str(program.getLanguageID())
    recorded_hash = str(program.getExecutableSHA256() or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
        raise ValueError("The Ghidra program has no executable SHA-256 identity; export refused")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    decompiler = DecompInterface()
    if decompile:
        decompiler.openProgram(program)
    count, failed, used_types = 0, 0, []

    def node_expression(node, seen=None, depth=0):
        if node is None:
            return {"kind": "missing"}
        if node.isConstant():
            return {"kind": "constant", "value": int(node.getOffset()) & ((1 << (8 * node.getSize())) - 1), "size": int(node.getSize())}
        key = str(node)
        seen = set() if seen is None else seen
        if key in seen or depth >= 20:
            return {"kind": "unresolved", "reason": "cycle_or_depth_limit", "node": key}
        high = node.getHigh()
        if high is not None and high.getClass().getSimpleName() == "HighParam":
            return {"kind": "parameter", "slot": int(high.getSlot()), "name": str(high.getName()),
                    "type": str(high.getDataType().getPathName()), "type_status": "ghidra_recovered"}
        definition = node.getDef()
        if definition is None:
            return {"kind": "input", "location": str(node.getAddress()), "size": int(node.getSize())}
        seen = seen | {key}
        expression = {"kind": "expression", "operation": str(definition.getMnemonic()), "size": int(node.getSize()),
                      "site": int(definition.getSeqnum().getTarget().getOffset()),
                      "inputs": [node_expression(v, seen, depth + 1) for v in definition.getInputs()]}
        if expression["operation"] in ("LOAD", "STORE"):
            space_id = int(definition.getInput(0).getOffset())
            if space_id >= 1 << 31:
                space_id -= 1 << 32
            space = program.getAddressFactory().getAddressSpace(space_id)
            expression["is_default_memory_space"] = space == program.getAddressFactory().getDefaultAddressSpace()
        return expression

    try:
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json_text({"kind": "metadata", "schema": 1,
                                   "artifact_sha256": recorded_hash, "language": language,
                                   "compiler": str(program.getCompilerSpec().getCompilerSpecID()),
                                   "program_name": str(program.getName()), "identity": "ghidra_recorded_executable_sha256",
                                   "export_scope": "selected_functions" if selected is not None else "all_known_functions",
                                   "source_container": str(origin) if origin is not None else None,
                                   "source_container_sha256": file_hash(origin) if origin is not None else None,
                                   "limitations": ["Static observations do not prove runtime execution",
                                                    "Recovered types and expressions are analysis results, not original source"]}) + "\n")
            for function in java_items(manager.getFunctions(True)):
                monitor.checkCancelled()
                if function.isExternal():
                    continue
                entry = int(function.getEntryPoint().getOffset())
                if selected is not None and entry not in selected:
                    continue
                used_types.append(function.getReturnType())
                used_types.extend(parameter.getDataType() for parameter in function.getParameters())
                body = function.getBody()
                ranges, content = [], bytearray()
                for span in java_items(body.getAddressRanges()):
                    start, end = span.getMinAddress(), span.getMaxAddress()
                    length = int(end.subtract(start)) + 1
                    import jpype
                    buf = jpype.JArray(jpype.JByte)(length)
                    read = memory.getBytes(start, buf)
                    if read != length:
                        raise ValueError(f"Incomplete bytes at {start}")
                    content.extend(_bytes(buf))
                    ranges.append([int(start.getOffset()), int(end.getOffset())])
                instructions, calls, references, control, shape = [], [], [], [], []
                for inst in java_items(listing.getInstructions(body, True)):
                    pc = int(inst.getAddress().getOffset())
                    raw = _bytes(inst.getBytes())
                    instructions.append({"address": pc, "bytes": raw.hex(), "text": str(inst)})
                    shape.append([str(inst.getMnemonicString()), [str(op.getMnemonic()) for op in inst.getPcode()]])
                    flow = inst.getFlowType()
                    destinations = [int(a.getOffset()) for a in inst.getFlows()]
                    if flow.isCall() or flow.isJump():
                        kind = "call" if flow.isCall() else "branch"
                        control.append({"site": pc, "kind": kind, "computed": bool(flow.isComputed()),
                                        "targets": destinations, "outside": [a for a in destinations if not body.contains(inst.getAddress().getNewAddress(a))]})
                        if flow.isCall():
                            for target in destinations or [None]:
                                calls.append({"site": pc, "target": target,
                                              "kind": "indirect_call" if flow.isComputed() else "direct_call",
                                              "status": "resolved_static" if target is not None else "unresolved"})
                    for ref in inst.getReferencesFrom():
                        target = ref.getToAddress()
                        if not target.isMemoryAddress():
                            continue
                        typ = ref.getReferenceType()
                        if typ.isData():
                            references.append({"site": pc, "target": int(target.getOffset()),
                                               "access": "write" if typ.isWrite() else "read" if typ.isRead() else "data",
                                               "reference_type": str(typ), "source": str(ref.getSource()),
                                               "primary": bool(ref.isPrimary())})
                blocks = []
                for block in java_items(BasicBlockModel(program).getCodeBlocksContaining(body, monitor)):
                    blocks.append({"start": int(block.getFirstStartAddress().getOffset()),
                                   "targets": [int(ref.getDestinationAddress().getOffset()) for ref in java_items(block.getDestinations(monitor))]})
                record = {"kind": "function", "address": entry, "name": str(function.getName()),
                          "qualified_name": str(function.getName(True)), "name_source": str(function.getSymbol().getSource()),
                          "signature": str(function.getSignature()), "ranges": ranges, "body_sha256": digest(content),
                          "instructions": instructions, "shape_sha256": digest(json_text(shape).encode()),
                          "calls": calls, "references": references, "control_flow": control, "blocks": blocks,
                          "dependencies": [], "decompilation_status": "not_requested"}
                decoded_bytes = sum(len(bytes.fromhex(i["bytes"])) for i in instructions)
                record["coverage"] = {"body_bytes": len(content), "decoded_bytes": decoded_bytes,
                                      "complete": bool(instructions) and decoded_bytes == len(content)}
                if decompile:
                    result = decompiler.decompileFunction(function, 45, monitor)
                    if result.decompileCompleted() and instructions:
                        record["decompilation_status"] = "completed"
                        record["pseudocode"] = str(result.getDecompiledFunction().getC())
                        high = result.getHighFunction()
                        if high is not None:
                            for op in java_items(high.getPcodeOps()):
                                if str(op.getMnemonic()) in ("CALL", "CALLIND", "LOAD", "STORE", "RETURN"):
                                    record["dependencies"].append({"site": int(op.getSeqnum().getTarget().getOffset()),
                                                                   "operation": str(op.getMnemonic()),
                                                                   "inputs": [node_expression(v) for v in op.getInputs()]})
                    else:
                        record["decompilation_status"] = "failed"
                        record["decompilation_error"] = "No decoded listing instructions" if not instructions else str(result.getErrorMessage())
                        failed += 1
                stream.write(json_text(record) + "\n")
                stream.flush()
                count += 1
            types, types_truncated = type_records(used_types)
            for record in types:
                stream.write(json_text(record) + "\n")
            stream.write(json_text({"kind": "complete", "functions": count,
                                   "types": len(types), "type_traversal_truncated": types_truncated}) + "\n")
    finally:
        decompiler.dispose()
    missing = sorted((selected or set()) - {int(f.getEntryPoint().getOffset()) for f in java_items(manager.getFunctions(True))})
    return {"output": str(output), "functions_exported": count, "decompilation_failures": failed,
            "types_exported": len(types), "type_traversal_truncated": types_truncated,
            "missing_requested_entries": missing, "artifact_sha256": recorded_hash, "language": language}


def standalone_export(args):
    import pyghidra
    pyghidra.start(install_dir=args.ghidra)
    from java.math import BigInteger
    from ghidra.util.task import TaskMonitor
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.base.project import GhidraProject
    from ghidra.app.cmd.function import CreateFunctionCmd

    selected = {address(v) for v in args.functions} if args.functions else None
    if not args.all and selected is None:
        raise ValueError("Select --functions or --all explicitly")
    binary = args.binary.resolve()
    cache = args.project / "symbols/analysis/ghidra_cache" / file_hash(binary)
    cache.mkdir(parents=True, exist_ok=True)
    if binary.suffix.lower() == ".gzf":
        import shutil
        from ghidra.framework.store.db import PackedDatabase
        from ghidra.framework.data import OpenMode
        from ghidra.program.database import ProgramDB
        from java.io import File
        from java.lang import Object
        consumer = Object()
        private_copy = cache / "reference.gzf"
        if not private_copy.exists():
            shutil.copyfile(binary, private_copy)
        if file_hash(private_copy) != cache.name:
            raise ValueError("Private packed-program copy does not match the source archive")
        packed = PackedDatabase.getPackedDatabase(File(str(private_copy)), TaskMonitor.DUMMY)
        program = None
        try:
            program = ProgramDB(packed.openForUpdate(TaskMonitor.DUMMY), OpenMode.UPGRADE, TaskMonitor.DUMMY, consumer)
            result = export_program(program, args.output, selected, TaskMonitor.DUMMY, not args.no_decompile, binary)
        finally:
            if program is not None:
                program.release(consumer)
            packed.dispose()
        if file_hash(binary) != cache.name:
            raise ValueError("Original packed program changed during export")
        return result
    language_key = digest(args.language.encode())[:12]
    with pyghidra.open_program(binary, project_location=cache, project_name="release_" + language_key,
                              analyze=False, language=args.language) as flat:
        program = flat.getCurrentProgram()
        monitor = TaskMonitor.DUMMY
        entries = selected if selected is not None else {int(f.getEntryPoint().getOffset()) for f in java_items(program.getFunctionManager().getFunctions(True))}
        if entries:
            transaction = program.startTransaction("Prepare selected analysis entries")
            try:
                toc = None
                loader = args.project / "out/EBOOT.loader.json"
                if loader.is_file() and binary == (args.project / "game/EBOOT.ELF").resolve():
                    toc = address(json.loads(loader.read_text())["module_toc"])
                for ea in sorted(entries, reverse=True):
                    target = flat.toAddr(ea)
                    if toc is not None:
                        program.getProgramContext().setValue(program.getRegister("r2"), target, target, BigInteger(str(toc)))
                    if program.getListing().getInstructionAt(target) is None:
                        if not flat.disassemble(target):
                            raise ValueError(f"Ghidra could not decode selected entry {ea:#x}")
                    function = program.getFunctionManager().getFunctionAt(target)
                    if function is None:
                        flat.createFunction(target, None)
                    else:
                        CreateFunctionCmd.fixupFunctionBody(program, function, monitor)
            finally:
                program.endTransaction(transaction, True)
        return export_program(program, args.output, selected, monitor, not args.no_decompile, binary)


def import_export(path, store, local_elf=None):
    with Path(path).open(encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        if metadata.get("kind") != "metadata" or metadata.get("schema") != 1:
            raise ValueError("Unknown Ghidra export schema")
        artifact = metadata["artifact_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", artifact):
            raise ValueError("Invalid artifact identity")
        if local_elf is not None and artifact != local_elf.sha256:
            raise ValueError("Ghidra export belongs to a different executable")
        source = str(Path(path).resolve())
        sha = file_hash(path)
        store.clear_source(artifact, source)
        if not store.db.execute("SELECT 1 FROM artifacts WHERE id=?", (artifact,)).fetchone():
            store.record("artifacts", id=artifact, path=metadata.get("program_name", ""),
                         kind="ghidra_export", metadata=json_text(metadata))
        count, complete, seen, type_count = 0, False, set(), 0
        for line in stream:
            record = json.loads(line)
            if complete:
                raise ValueError("Unexpected records after export completion")
            if record.get("kind") == "complete":
                if record["functions"] != count:
                    raise ValueError("Export completion count mismatch")
                if record.get("types", 0) != type_count:
                    raise ValueError("Export type count mismatch")
                store.fact(artifact, None, "ghidra_type_coverage", source,
                           "partial" if record.get("type_traversal_truncated") else "export_observation",
                           {"types": type_count, "truncated": bool(record.get("type_traversal_truncated")), "export_sha256": sha})
                complete = True
                continue
            if record.get("kind") == "data_type":
                store.fact(artifact, None, "ghidra_data_type", source, "export_observation",
                           {**record, "export_sha256": sha, "language": metadata["language"],
                            "compiler": metadata["compiler"],
                            "limitation": "A source-program layout is not assigned to release pointers without identity and data-flow evidence"})
                type_count += 1
                continue
            if record.get("kind") != "function":
                continue
            ea = int(record["address"])
            if ea in seen:
                raise ValueError("Duplicate exported function")
            seen.add(ea)
            if local_elf:
                content = bytearray()
                for start, end in record["ranges"]:
                    chunk = local_elf.read(start, end - start + 1)
                    if chunk is None:
                        raise ValueError(f"Unmapped exported body at {ea:#x}")
                    content.extend(chunk)
                if digest(content) != record["body_sha256"]:
                    raise ValueError(f"Exported bytes mismatch release at {ea:#x}")
                for instruction in record["instructions"]:
                    raw = bytes.fromhex(instruction["bytes"])
                    if local_elf.read(instruction["address"], len(raw)) != raw:
                        raise ValueError(f"Exported instruction mismatch at {instruction['address']:#x}")
            store.fact(artifact, ea, "ghidra_function", source, "bytes_verified" if local_elf else "export_observation",
                       {**record, "export_sha256": sha, "language": metadata["language"],
                        "compiler": metadata["compiler"], "local_bytes_verified": local_elf is not None,
                        "export_scope": metadata.get("export_scope", "unknown"),
                        "source_container": metadata.get("source_container"),
                        "source_container_sha256": metadata.get("source_container_sha256")})
            if not store.db.execute("SELECT 1 FROM functions WHERE artifact=? AND address=?", (artifact, ea)).fetchone():
                store.record("functions", artifact=artifact, address=ea, name=record["name"],
                             source_hash=record["body_sha256"], metadata=json_text({"source": source}))
            for call in record["calls"]:
                from .common import direct_branch
                if local_elf and call["kind"] == "direct_call" and call["target"] is not None:
                    if direct_branch(local_elf.u32(call["site"]), call["site"]) != ("direct_call", call["target"]):
                        raise ValueError("Exported direct-call target does not match the native instruction")
                store.edge(artifact, ea, call["target"], call["site"], call["kind"], source,
                           "instruction_verified" if local_elf and call["kind"] == "direct_call" and call["target"] is not None else call["status"],
                           {**call, "function_ownership": "ghidra_body", "export_sha256": sha})
            for ref in record["references"]:
                store.fact(artifact, ea, "global_access", source, "static_reference", {**ref, "export_sha256": sha})
            for dep in record["dependencies"]:
                store.fact(artifact, ea, "value_dependency", source, "ghidra_recovered", {**dep, "export_sha256": sha})
            count += 1
        if not complete:
            raise ValueError("Ghidra export is incomplete; no facts were committed")
        store.record("inputs", artifact=artifact, path=source, sha256=sha)
    store.db.commit()
    return {"artifact_sha256": artifact, "functions_imported": count, "types_imported": type_count}
