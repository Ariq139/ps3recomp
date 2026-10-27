"""Adapter between the offline application and the bundled analysis engine."""
import csv
import json
import re
import sys
import uuid
from pathlib import Path

from .config import ROOT
from .routing import source_folder


def bind_engine(project):
    engine = str(ROOT / "reconstructor/analysis_engine")
    rules = str(Path(project) / "tools")
    for path in (rules, engine):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


class Engine:
    def __init__(self, config):
        bind_engine(config.project)
        from p5_analysis.common import Elf
        from p5_analysis.store import Store
        self.config = config
        self.store = Store(config.project / "symbols/analysis/functions.sqlite")
        self.elf = Elf(config.project / "game/EBOOT.ELF")

    def close(self):
        self.store.close()

    def index(self):
        from p5_analysis.indexer import index_project
        return index_project(self.config.project, self.store)

    def inventory(self):
        return self.store.rows("SELECT * FROM functions WHERE artifact=? ORDER BY address", (self.elf.sha256,))

    def import_export(self, path, role):
        from p5_analysis.ghidra import import_export
        return import_export(path, self.store, self.elf if role == "release" else None)

    def refine(self, reference_artifacts):
        from p5_analysis.flow import build_value_connections, resolve_indirects
        from p5_analysis.indexer import grouping_candidates
        from p5_analysis.matching import match_exports
        from p5_analysis.readability import rewrite_candidates
        result = {**build_value_connections(self.store, self.elf), **resolve_indirects(self.store, self.elf),
                  "folder_suggestions": grouping_candidates(self.store, self.elf.sha256),
                  **rewrite_candidates(self.config.project, self.store, self.elf)}
        matches = []
        for artifact in sorted(reference_artifacts):
            if not self.store.db.execute("SELECT 1 FROM facts WHERE artifact=? AND kind='ghidra_function' LIMIT 1", (artifact,)).fetchone():
                continue
            if not self.store.db.execute("SELECT 1 FROM facts WHERE artifact=? AND kind='ghidra_function' LIMIT 1", (self.elf.sha256,)).fetchone():
                continue
            try:
                matches.append(match_exports(self.store, artifact, self.elf.sha256))
            except ValueError as error:
                matches.append({"source_artifact": artifact, "blocked": str(error)})
        result["source_matches"] = matches
        from ppu_function_path_overrides import FUNCTION_PATH_OVERRIDES
        routed = 0
        for function in self.inventory():
            decision = source_folder(function, FUNCTION_PATH_OVERRIDES)
            if decision is None:
                continue
            folder, reason = decision
            current = json.loads(function["metadata"])
            if current.get("split_source", "").startswith("split/" + folder + "/"):
                continue
            self.store.candidate(self.elf.sha256, function["address"], "folder", folder, "verified",
                                 {"expected_name": function["name"], "expected_source_hash": function["source_hash"],
                                  "rule": "offline-source-folder-policy-v1", "reason": reason})
            routed += 1
        self.store.db.commit()
        result["verified_folder_decisions"] = routed
        result["verified_decisions"] = self.store.db.execute(
            "SELECT COUNT(*) FROM candidates WHERE artifact=? AND status='verified'", (self.elf.sha256,)).fetchone()[0]
        from p5_analysis.common import digest, json_text
        identifiers = [row["id"] for row in self.store.rows("SELECT id FROM candidates WHERE artifact=? AND status='verified' ORDER BY id", (self.elf.sha256,))]
        result["verified_signature"] = digest(json_text(identifiers).encode())
        return result

    def apply_verified(self, workspace):
        from p5_analysis.decisions import apply_plan, make_plan
        from p5_analysis.readability import audit_saved_bodies
        plan = workspace / "last_change_plan.json"
        proposed = make_plan(self.config.project, self.store, self.elf, [], plan)
        if not proposed["files_changed"]:
            return {"files_changed": 0, "decisions_applied": 0}
        result = apply_plan(self.config.project, self.store, self.elf, plan)
        self.index()
        result["audit"] = audit_saved_bodies(self.config.project, self.store, self.elf)
        return result

    def report(self):
        from p5_analysis.reports import report
        return report(self.config.project, self.store, self.elf.sha256)

    def accepted_count(self):
        from p5_analysis.accepted import load_saved
        saved = load_saved(self.config.project)
        return len(saved["bodies"])

    def export_drafts(self, export, workspace):
        from p5_analysis.common import rewrite_tokens, write_json, digest, inside
        inventory = {row["address"]: row for row in self.inventory()}
        names = {}
        for ea, row in inventory.items():
            for name in (f"FUN_{ea:08x}", f"FUN_{ea:08X}", f"_opd_FUN_{ea:08x}", f"_opd_FUN_{ea:08X}"):
                names[name] = row["name"]
        saved = []
        with Path(export).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("kind") != "function" or record.get("decompilation_status") != "completed":
                    continue
                row = inventory.get(record["address"])
                if not row or not record.get("coverage", {}).get("complete"):
                    continue
                metadata = json.loads(row["metadata"])
                relative = Path(metadata.get("split_source", "split/unclassified/unknown.cpp"))
                directory = relative.parent.relative_to("split")
                safe_name = re.sub(r"[^A-Za-z0-9_]", "_", row["name"])[:64]
                output = inside(workspace / "readable_drafts", directory / f"{safe_name}_{row['address']:08X}.c")
                output.parent.mkdir(parents=True, exist_ok=True)
                code = rewrite_tokens(record["pseudocode"].encode("utf-8"), names)
                local_edits = False
                if output.exists():
                    sidecar = output.with_suffix(".json")
                    try:
                        previous = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
                    except (ValueError, OSError):
                        previous = {}
                    if not isinstance(previous, dict):
                        previous = {}
                    local_edits = digest(output.read_bytes()) != previous.get("draft_sha256")
                    if local_edits:
                        output = inside(workspace / "readable_drafts/incoming", directory / f"{safe_name}_{digest(code)[:10]}_{row['address']:08X}.c")
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if output.exists() and digest(output.read_bytes()) != digest(code):
                            output = output.with_name(f"{safe_name}_{digest(code)[:10]}_{uuid.uuid4().hex[:8]}_{row['address']:08X}.c")
                output.write_bytes(code)
                provenance = {"address": row["address"], "name": row["name"], "stage": "decompiler_draft",
                              "compilation_target": False, "native_body_sha256": record["body_sha256"],
                              "draft_sha256": digest(code), "evidence": str(export),
                              "local_edits_preserved": local_edits,
                              "limitation": "Decompiler output requires validation before replacing a compiled implementation"}
                write_json(output.with_suffix(".json"), provenance)
                saved.append(str(output))
        return saved
