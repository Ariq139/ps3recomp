import csv
import json
from collections import Counter

from .common import FUNCTION, inside, write_json


def csv_output(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(project, store, artifact):
    root = project / "symbols/analysis/reports"
    root.mkdir(parents=True, exist_ok=True)
    functions = {r["address"]: r for r in store.rows("SELECT * FROM functions WHERE artifact=?", (artifact,))}
    edges = store.rows("SELECT * FROM edges WHERE artifact=? ORDER BY caller,site,target", (artifact,))
    for row in edges:
        row["caller_name"] = functions.get(row["caller"], {}).get("name", "")
        row["target_name"] = functions.get(row["target"], {}).get("name", "")
        for field in ("caller", "target", "site"):
            row[field] = f"0x{row[field]:08X}" if row[field] is not None else ""
    fields = ["caller", "caller_name", "site", "target", "target_name", "kind", "status", "source", "evidence"]
    csv_output(root / "connections.csv", edges, fields)
    csv_output(root / "unresolved_connections.csv", [r for r in edges if r["status"] != "instruction_verified"], fields)
    candidates = store.rows("SELECT * FROM candidates WHERE artifact=? ORDER BY function,kind,proposed", (artifact,))
    for row in candidates:
        row["current_name"] = functions.get(row["function"], {}).get("name", "")
        row["function"] = f"0x{row['function']:08X}"
    csv_output(root / "candidates.csv", candidates,
               ["id", "function", "current_name", "kind", "proposed", "status", "evidence"])

    from ppu_readable_overrides import READABLE_BODY_OVERRIDES
    from .accepted import load_saved
    accepted_bodies = load_saved(project)["bodies"]
    byname = {r["name"]: r for r in functions.values()}
    worklist = []
    for relative in sorted({json.loads(r["metadata"]).get("split_source", "") for r in functions.values()} - {""}):
        path = inside(project / "src/ps3/ppu", relative)
        data = path.read_bytes()
        for match in FUNCTION.finditer(data):
            name = match[1].decode()
            fn = byname.get(name)
            if not fn:
                continue
            body = match[0]
            ea = fn["address"]
            meta = json.loads(fn["metadata"])
            worklist.append({"address": f"0x{ea:08X}", "name": name, "path": relative,
                             "stage": "structured_handler" if f"{ea:08X}" in accepted_bodies else
                                      "readable_override" if f"{ea:08X}" in READABLE_BODY_OVERRIDES else "lifted",
                             "named": not name.startswith(("func_", "FUN_")), "lines": body.count(b"\n") + 1,
                             "register_accesses": body.count(b"ctx->gpr["), "gotos": body.count(b"goto "),
                             "module": meta.get("module", ""),
                             "structure_evidence": meta.get("normalized_struct_accesses", ""),
                             "limitation": "Readability metrics do not establish semantic equivalence"})
    worklist.sort(key=lambda r: (not r["named"], r["stage"] != "lifted", r["lines"], r["address"]))
    csv_output(root / "readability_worklist.csv", worklist,
               ["address", "name", "path", "stage", "named", "lines", "register_accesses", "gotos", "module", "structure_evidence", "limitation"])
    shared = {}
    for row in store.rows("SELECT function,evidence FROM facts WHERE artifact=? AND kind='global_access'", (artifact,)):
        value = json.loads(row["evidence"])
        shared.setdefault(value["target"], set()).add(row["function"])
    csv_output(root / "shared_globals.csv",
               [{"global_address": f"0x{ea:08X}", "function_count": len(users),
                 "functions": ";".join(f"0x{u:08X}:{functions.get(u, {}).get('name', '')}" for u in sorted(users))}
                for ea, users in sorted(shared.items()) if len(users) > 1],
               ["global_address", "function_count", "functions"])
    write_json(root / "source_types.json",
               [{"artifact_sha256": row["artifact"], "layout": json.loads(row["evidence"])}
                for row in store.rows("SELECT artifact,evidence FROM facts WHERE kind='ghidra_data_type' ORDER BY artifact,id")])
    summary = {**store.summary(artifact), "end_goal": "Readable, typed game implementations with structured control flow",
               "human_readability_status": "in_progress",
               "readability": dict(Counter(r["stage"] for r in worklist)),
               "reports": str(root), "candidate_status": dict(Counter(r["status"] for r in candidates)),
               "limitations": ["Raw lift call ownership may include unreachable tails; retain each edge's ownership evidence",
                                "Register offsets alone do not establish a structure type",
                                "Candidate labels are never fed back as independent facts",
                                "Readable pseudocode is not automatically accepted as a replacement implementation"]}
    write_json(root / "summary.json", summary)
    return summary


def inspect_function(store, artifact, selector):
    try:
        ea = int(selector, 16)
        rows = store.rows("SELECT * FROM functions WHERE artifact=? AND address=?", (artifact, ea))
    except ValueError:
        rows = store.rows("SELECT * FROM functions WHERE artifact=? AND name=?", (artifact, selector))
    if len(rows) != 1:
        raise ValueError(f"Expected one matching function, found {len(rows)}")
    fn = rows[0]
    ea = fn["address"]
    fn["metadata"] = json.loads(fn["metadata"])
    fn["outgoing"] = store.rows("SELECT * FROM edges WHERE artifact=? AND caller=?", (artifact, ea))
    fn["incoming"] = store.rows("SELECT * FROM edges WHERE artifact=? AND target=?", (artifact, ea))
    fn["facts"] = store.rows("SELECT * FROM facts WHERE artifact=? AND function=?", (artifact, ea))
    for row in fn["outgoing"] + fn["incoming"] + fn["facts"]:
        row["evidence"] = json.loads(row["evidence"])
    return fn
