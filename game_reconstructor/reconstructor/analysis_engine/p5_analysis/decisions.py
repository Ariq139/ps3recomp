import csv
import io
import json
import os
import uuid
from pathlib import Path

from .accepted import load_saved
from .common import FUNCTION, IDENTIFIER, digest, file_hash, function_bodies, inside, json_text, read_csv, rewrite_tokens, write_json


def _read_sources(project):
    lift = project / "src/ps3/ppu"
    paths = sorted((lift / "split").rglob("*.cpp")) + [lift / "ppu_recomp.h", lift / "ppu_lift_support.inc"]
    return {p.relative_to(project).as_posix(): p.read_bytes() for p in paths}


def _verify_inventory(files, expected):
    functions = {}
    for path, data in files.items():
        if not path.startswith("src/ps3/ppu/split/"):
            continue
        for name, body in function_bodies(data).items():
            if name in functions:
                raise ValueError(f"Duplicate generated function: {name}")
            functions[name] = body
    if set(functions) != set(expected):
        raise ValueError(f"Generated function inventory changed: {len(set(expected)-functions.keys())} missing, {len(functions.keys()-set(expected))} extra")
    return functions


def prepare(project, store, elf, candidate_ids=()):
    rows = store.rows("SELECT * FROM candidates WHERE artifact=? AND status='verified' ORDER BY function,kind", (elf.sha256,))
    if candidate_ids:
        wanted = set(candidate_ids)
        rows = [r for r in rows if r["id"] in wanted]
        if {r["id"] for r in rows} != wanted:
            raise ValueError("Only existing verified candidates can be applied")
    inventory = read_csv(project / "symbols/EBOOT_GAME_FUNCTION_MODULES.csv")
    functions = {int(r["address"], 16): r for r in inventory}
    original_symbols = {row["compiled_symbol"]: f"func_{ea:08X}" for ea, row in functions.items()}
    saved = load_saved(project)
    files = _read_sources(project)
    old_files = dict(files)
    old_bodies = _verify_inventory(files, [r["compiled_symbol"] for r in inventory])
    replacements, body_changes, folder_changes = {}, {}, {}
    decisions = []
    for row in rows:
        ea, proof = row["function"], json.loads(row["evidence"])
        current = functions.get(ea)
        if current is None or current["compiled_symbol"] != proof["expected_name"] or current["body_sha256"] != proof["expected_source_hash"]:
            raise ValueError(f"Stale candidate {row['id']}")
        key = f"{ea:08X}"
        common = {"candidate_id": row["id"], "expected_source_hash": current["body_sha256"], "evidence": proof}
        if row["kind"] == "rename":
            for field in ("source_evidence_id", "target_evidence_id"):
                if field not in proof or not store.db.execute("SELECT 1 FROM facts WHERE id=? AND kind='ghidra_function'", (proof[field],)).fetchone():
                    raise ValueError("Source-name observation is missing or stale; rematch before applying")
            name = row["proposed"]
            if not IDENTIFIER.fullmatch(name):
                raise ValueError("Invalid proposed function name")
            replacements[current["compiled_symbol"]] = name
            saved["names"][key] = {**common, "name": name}
            current["compiled_symbol"] = name
            current["inferred_name_basis"] = "analysis verified source label: " + row["id"]
        elif row["kind"] == "body":
            from .readability import prove_opcode_rule, canonical
            locs = store.rows("SELECT * FROM facts WHERE artifact=? AND function=? AND kind='original_lift_body'", (elf.sha256, ea))
            if len(locs) != 1:
                raise ValueError("Missing unique original body location")
            location = json.loads(locs[0]["evidence"])
            with inside(project, locs[0]["source"]).open("rb") as source:
                source.seek(location["byte_offset"])
                body = source.read(location["byte_length"])
            verified = prove_opcode_rule(body, ea, proof["expected_name"], elf)
            if verified is None or any(verified[k] != proof[k] for k in ("body", "body_sha256", "rule_hash", "raw_body_hash")):
                raise ValueError("Readable rewrite proof no longer holds")
            active_body = rewrite_tokens(old_bodies[proof["expected_name"]], original_symbols)
            if canonical(active_body) != canonical(body):
                raise ValueError("Active function has edits beyond the checked original template")
            if file_hash(inside(project, proof["helper"])) != proof["helper_sha256"]:
                raise ValueError("Readable rewrite helper changed after proposing")
            body_changes[proof["expected_name"]] = proof["body"]
            saved["bodies"][key] = {**common, "body": proof["body"], "body_sha256": proof["body_sha256"]}
            current["readable_body_sha256"] = digest(proof["body"].replace("\n", "\r\n").encode())
            current["unreachable_tail_bytes_removed"] = "0"
        elif row["kind"] == "folder":
            directory = row["proposed"].strip("/")
            inside(project / "src/ps3/ppu/split", directory)
            folder_changes[ea] = directory
            saved["paths"][key] = {**common, "directory": directory}
        else:
            raise ValueError("Unsupported verified decision kind")
        decisions.append({"id": row["id"], "address": current["address"], "kind": row["kind"], "proposed": row["proposed"]})
    final_names = [r["compiled_symbol"] for r in inventory]
    if len(set(final_names)) != len(final_names):
        raise ValueError("Proposed function names collide")
    symbol_map = {f"func_{ea:08X}": row["compiled_symbol"] for ea, row in functions.items()}
    expected_new_bodies = {}
    for path, original in old_files.items():
        data = original
        if body_changes and path.startswith("src/ps3/ppu/split/"):
            def rewrite_body(match):
                name = match[1].decode()
                if name not in body_changes:
                    return match[0]
                replacement = body_changes[name].replace("\n", "\r\n").encode()
                replacement = rewrite_tokens(replacement, symbol_map)
                expected_new_bodies[replacements.get(name, name)] = replacement
                return replacement
            data = FUNCTION.sub(rewrite_body, data)
        if replacements:
            data = rewrite_tokens(data, replacements)
        files[path] = data
    if body_changes:
        support = "src/ps3/ppu/ppu_lift_support.inc"
        include = b'#include "include/p5/scripts/binary_opcode.h"\r\n'
        if include.strip() not in files[support]:
            files[support] = files[support].rstrip(b"\r\n") + b"\r\n" + include
    for ea, directory in folder_changes.items():
        row = functions[ea]
        old_path = "src/ps3/ppu/" + row["split_source"]
        data = files[old_path]
        matches = list(FUNCTION.finditer(data))
        selected = [m for m in matches if m[1].decode() == row["compiled_symbol"]]
        if len(selected) != 1:
            raise ValueError("Cannot locate function for incremental extraction")
        match = selected[0]
        if any(n.upper().startswith("SCRIPT_") for n in row["callable_names"].split(";")):
            directory += "/callables"
        relative = f"split/{directory}/{row['compiled_symbol']}.cpp"
        new_path = "src/ps3/ppu/" + relative
        if new_path != old_path:
            if new_path in files or inside(project, new_path).exists():
                raise ValueError("Incremental destination already exists")
            files[new_path] = data[:matches[0].start()] + match[0]
            files[old_path] = data[:match.start()] + data[match.end():]
            row["split_source"] = relative
    new_bodies = _verify_inventory(files, final_names)
    reverse = {new: old for old, new in replacements.items()}
    for old_name, body in old_bodies.items():
        name = replacements.get(old_name, old_name)
        expected = expected_new_bodies.get(name)
        if expected is not None:
            if new_bodies[name] != expected:
                raise ValueError("Applied readable body differs from its checked rule")
        elif rewrite_tokens(new_bodies[name], reverse) != body:
            raise ValueError(f"Non-rewrite function changed: {old_name}")
    for path in ("src/ps3/ppu/ppu_recomp.h", "src/ps3/ppu/split/registry/function_table.cpp"):
        if rewrite_tokens(files[path], reverse) != old_files[path]:
            raise ValueError("Declaration or registry contents changed beyond identifiers")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(inventory[0]), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(inventory)
    files["symbols/EBOOT_GAME_FUNCTION_MODULES.csv"] = output.getvalue().encode()
    for kind in ("names", "paths", "bodies"):
        if saved[kind] or (project / f"symbols/analysis/accepted_{kind}.json").is_file():
            value = {"schema": 1, "artifact_sha256": elf.sha256, "records": saved[kind]}
            files[f"symbols/analysis/accepted_{kind}.json"] = (json.dumps(value, indent=2) + "\n").encode()
    changes = {}
    for path, data in files.items():
        previous = old_files.get(path)
        if previous is None and inside(project, path).is_file():
            previous = inside(project, path).read_bytes()
        if data != previous:
            changes[path] = data
    inputs = {path: digest(data) for path, data in old_files.items()}
    inputs["symbols/EBOOT_GAME_FUNCTION_MODULES.csv"] = file_hash(project / "symbols/EBOOT_GAME_FUNCTION_MODULES.csv")
    for row in store.rows("SELECT path,sha256 FROM inputs"):
        inputs[row["path"]] = row["sha256"]
    inputs["include/p5/scripts/binary_opcode.h"] = file_hash(project / "include/p5/scripts/binary_opcode.h")
    return changes, inputs, decisions, len(new_bodies)


def make_plan(project, store, elf, candidate_ids, output):
    changes, inputs, decisions, count = prepare(project, store, elf, candidate_ids)
    plan = {"schema": 1, "artifact_sha256": elf.sha256, "inputs": inputs, "decisions": decisions,
            "function_count": count, "changes": []}
    plan_id = digest(json_text({"decisions": decisions, "inputs": inputs}).encode())[:24]
    blobs = project / "symbols/analysis/plans" / plan_id
    for index, (relative, data) in enumerate(sorted(changes.items())):
        blob = blobs / f"{index:04d}.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        path = inside(project, relative)
        plan["changes"].append({"path": relative, "before": file_hash(path) if path.exists() else None,
                                 "after": digest(data), "blob": blob.relative_to(project).as_posix()})
    plan["plan_id"] = plan_id
    plan["plan_sha256"] = digest(json_text(plan).encode())
    write_json(output, plan)
    return {"plan": str(output), "verified_decisions": len(decisions), "files_changed": len(changes),
            "function_count_verified": count, "plan_sha256": plan["plan_sha256"]}


def apply_plan(project, store, elf, plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    expected_hash = plan.pop("plan_sha256")
    if digest(json_text(plan).encode()) != expected_hash or plan["artifact_sha256"] != elf.sha256:
        raise ValueError("Plan contents or executable identity changed")
    for relative, expected in plan["inputs"].items():
        path = Path(relative) if Path(relative).is_absolute() else inside(project, relative)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Plan input changed; reindex and regenerate: {relative}")
    current_cpp = {p.relative_to(project).as_posix() for p in (project / "src/ps3/ppu/split").rglob("*.cpp")}
    expected_cpp = {p for p in plan["inputs"] if p.startswith("src/ps3/ppu/split/") and p.endswith(".cpp")}
    if current_cpp != expected_cpp:
        raise ValueError("Generated file inventory changed after planning")
    changes, _, decisions, count = prepare(project, store, elf, [r["id"] for r in plan["decisions"]])
    if decisions != plan["decisions"] or count != plan["function_count"]:
        raise ValueError("Candidate evidence changed after planning")
    planned = {c["path"]: c for c in plan["changes"]}
    if set(planned) != set(changes):
        raise ValueError("Plan no longer matches generated changes")
    for relative, data in changes.items():
        c = planned[relative]
        path = inside(project, relative)
        if (file_hash(path) if path.exists() else None) != c["before"] or digest(data) != c["after"]:
            raise ValueError("Changed output or destination since planning")
        if inside(project, c["blob"]).read_bytes() != data:
            raise ValueError("Plan preview bytes changed")
    backup = project / "symbols/analysis/backups" / uuid.uuid4().hex
    journal = {"status": "prepared", "plan_sha256": expected_hash, "changes": plan["changes"]}
    for relative, c in planned.items():
        if c["before"] is not None:
            target = inside(backup, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(inside(project, relative).read_bytes())
    write_json(backup / "journal.json", journal)
    installed = []
    try:
        for relative, data in changes.items():
            destination = inside(project, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (file_hash(destination) if destination.exists() else None) != planned[relative]["before"]:
                raise ValueError("Destination changed during publication")
            temporary = destination.with_name(destination.name + ".analysis-" + uuid.uuid4().hex)
            try:
                temporary.write_bytes(data)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            installed.append(relative)
        journal["status"] = "applied"
        write_json(backup / "journal.json", journal)
    except Exception:
        for relative in reversed(installed):
            target = inside(project, relative)
            if planned[relative]["before"] is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(inside(backup, relative).read_bytes())
        journal["status"] = "rolled_back"
        write_json(backup / "journal.json", journal)
        raise
    return {"decisions_applied": len(decisions), "files_changed": len(changes),
            "functions_preserved": count, "backup": str(backup), "build_run": False}
