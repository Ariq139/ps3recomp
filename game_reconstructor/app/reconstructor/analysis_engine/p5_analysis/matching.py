import json
import re
from collections import defaultdict
from pathlib import Path

from .common import IDENTIFIER


def observations(store, artifact):
    result = {}
    for row in store.rows("SELECT * FROM facts WHERE artifact=? AND kind='ghidra_function'", (artifact,)):
        value = json.loads(row["evidence"])
        ea = row["function"]
        previous = result.get(ea)
        if previous and any(previous[k] != value[k] for k in ("body_sha256", "ranges", "language", "compiler")):
            raise ValueError(f"Conflicting Ghidra function boundaries or bytes at {ea:#x}; select consistent exports")
        result[ea] = {**value, "evidence_id": row["id"]}
    return result


def match_exports(store, source_artifact, target_artifact):
    for row in store.rows("SELECT id,evidence FROM candidates WHERE artifact=? AND kind='rename'", (target_artifact,)):
        if json.loads(row["evidence"]).get("source_artifact") == source_artifact:
            store.db.execute("DELETE FROM candidates WHERE id=?", (row["id"],))
    source = observations(store, source_artifact)
    target = observations(store, target_artifact)
    if not source or not target:
        raise ValueError("Import source and release Ghidra exports before matching")
    target_inventory = {r["address"]: r for r in store.rows("SELECT * FROM functions WHERE artifact=?", (target_artifact,))}
    complete_inventory = (set(target) == set(target_inventory) and
                          all(r.get("export_scope") == "all_known_functions" for r in source.values()))
    source_hashes, target_hashes, shapes = defaultdict(list), defaultdict(list), defaultdict(list)
    for ea, fn in source.items():
        source_hashes[fn["body_sha256"]].append(ea)
    for ea, fn in target.items():
        target_hashes[fn["body_sha256"]].append(ea)
        shapes[(fn["shape_sha256"], len(fn["instructions"]), len(fn["blocks"]))].append(ea)

    exact = {}
    for sha, entries in source_hashes.items():
        if len(entries) == 1 and len(target_hashes.get(sha, ())) == 1:
            old, new = entries[0], target_hashes[sha][0]
            old_ranges = [b - a for a, b in source[old]["ranges"]]
            new_ranges = [b - a for a, b in target[new]["ranges"]]
            if old_ranges == new_ranges:
                exact[old] = new

    verified = {}
    if source_artifact == target_artifact:
        verified = {ea: ea for ea in source if ea in target and target[ea]["local_bytes_verified"]}
    else:
        # Only previously applied source mappings may seed dependency validation.
        for row in store.rows("SELECT record FROM catalog WHERE artifact=? AND kind='priority_source_match'", (target_artifact,)):
            record = json.loads(row["record"])
            if not record.get("decision", "").startswith("applied_"):
                continue
            old, new = int(record["source_entry"], 16), int(record["target_address"], 16)
            if old in source and new in target and source[old]["name"] == record["source_name"]:
                origin = source[old].get("source_container")
                if not origin or Path(origin).name != record.get("source_artifact"):
                    continue
                # Source name and entry alone are insufficient across different archives.
                if source[old]["body_sha256"] == target[new]["body_sha256"]:
                    verified[old] = new

    reasons = {}
    progress = True
    while progress:
        progress = False
        for old, new in exact.items():
            if old in verified:
                continue
            src, dst = source[old], target[new]
            problems = []
            if not complete_inventory:
                problems.append("partial_inventory_cannot_establish_unique_identity")
            if src["language"] != dst["language"] or src["compiler"] != dst["compiler"]:
                problems.append("language_or_compiler_mismatch")
            if not dst["local_bytes_verified"]:
                problems.append("release_bytes_not_independently_verified")
            if not src.get("coverage", {}).get("complete") or not dst.get("coverage", {}).get("complete"):
                problems.append("incomplete_instruction_coverage")
            if src["references"] or dst["references"]:
                problems.append("external_data_dependencies_require_mapping")
            if any(re.search(r"\b(r2|r13)\b", i["text"]) for i in src["instructions"]):
                problems.append("TOC_or_small_data_base_requires_mapping")
            src_flow = src["control_flow"]
            dst_flow = dst["control_flow"]
            if len(src_flow) != len(dst_flow):
                problems.append("control_flow_count_mismatch")
            else:
                for a, b in zip(src_flow, dst_flow):
                    if a["computed"] or b["computed"]:
                        problems.append("computed_control_flow_requires_resolution")
                        continue
                    if a["site"] - old != b["site"] - new or a["kind"] != b["kind"]:
                        problems.append("control_flow_position_mismatch")
                    if [t - old for t in a["targets"] if t not in a["outside"]] != [t - new for t in b["targets"] if t not in b["outside"]]:
                        problems.append("internal_control_flow_mismatch")
                    mapped = [new if t == old else verified.get(t) for t in a["outside"]]
                    if None in mapped or mapped != b["outside"]:
                        problems.append("external_control_target_not_verified")
            reasons[old] = sorted(set(problems))
            if not problems:
                verified[old] = new
                progress = True

    created = 0
    for old, src in source.items():
        label = src["name"]
        if src["name_source"] not in ("USER_DEFINED", "IMPORTED") or re.match(r"^(?:_opd_)?(?:FUN_|func_)(?:0x)?[0-9a-f]+$", label, re.I):
            continue
        if label.lower() in ("something_bgm", "kasumipersona"):
            continue
        targets = [exact[old]] if old in exact else shapes.get((src["shape_sha256"], len(src["instructions"]), len(src["blocks"])), [])
        for new in targets:
            current = target_inventory.get(new)
            if current is None or current["name"] == label:
                continue
            status = "verified" if verified.get(old) == new else "candidate"
            blockers = reasons.get(old, ["structural_similarity_only"])
            meta = json.loads(current["metadata"])
            if not IDENTIFIER.fullmatch(label):
                status, blockers = "blocked", ["source_name_is_not_a_C_identifier"]
            elif meta.get("inferred_name_basis", "").startswith("priority source label:"):
                status, blockers = "blocked", ["conflicts_with_existing_priority_source_label"]
            collision = store.db.execute("SELECT address FROM functions WHERE artifact=? AND name=? AND address<>?", (target_artifact, label, new)).fetchone()
            if collision:
                status, blockers = "blocked", ["name_collision"]
            store.candidate(target_artifact, new, "rename", label, status,
                            {"source_artifact": source_artifact, "source_address": old,
                             "source_evidence_id": src["evidence_id"], "target_evidence_id": target[new]["evidence_id"],
                             "expected_name": current["name"], "expected_source_hash": current["source_hash"],
                             "match": "exact_bytes_and_boundaries" if exact.get(old) == new else "instruction_shape",
                             "blockers": [] if status == "verified" else blockers})
            created += 1
    store.db.commit()
    return {"source_functions": len(source), "target_functions": len(target),
            "unique_exact_matches": len(exact), "dependency_verified_matches": len(verified), "rename_candidates": created}
