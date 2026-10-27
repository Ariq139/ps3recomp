import argparse
import json
from pathlib import Path

from .common import Elf, write_json
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent function evidence, readable rewrites, and incremental application; never builds the game")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--database", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Index originals and existing accepted catalogs; reuse unchanged inputs")
    commands.add_parser("report", help="Export connections, unresolved edges, naming candidates, and readability worklist")
    commands.add_parser("propose", help="Generate folder suggestions and proven body-rewrite candidates")
    commands.add_parser("audit", help="Recheck accepted readable bodies against native bytes, originals, and active files")
    inspect = commands.add_parser("inspect", help="Inspect a function by address or exact current name")
    inspect.add_argument("function")
    inspect.add_argument("--output", type=Path)
    imported = commands.add_parser("import-ghidra", help="Import an export; release observations must match ELF bytes")
    imported.add_argument("export", type=Path)
    imported.add_argument("--source", action="store_true", help="Import a reference build instead of the release")
    match = commands.add_parser("match", help="Compare reference and release Ghidra observations")
    match.add_argument("source_artifact", help="Source executable SHA-256 from import-ghidra")
    plan = commands.add_parser("plan", help="Prepare a reviewable, hash-bound incremental change plan")
    plan.add_argument("--output", type=Path)
    plan.add_argument("--candidate", action="append", default=[])
    apply = commands.add_parser("apply", help="Apply a verified plan and save reversible backups")
    apply.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    store = Store(args.database or project / "symbols/analysis/functions.sqlite")
    try:
        elf = Elf(project / "game/EBOOT.ELF")
        if args.command == "index":
            from .indexer import index_project
            result = index_project(project, store)
        elif args.command == "report":
            from .reports import report
            result = report(project, store, elf.sha256)
        elif args.command == "propose":
            from .indexer import grouping_candidates
            from .readability import rewrite_candidates
            from .flow import resolve_indirects, build_value_connections
            result = {"folder_candidates": grouping_candidates(store, elf.sha256),
                      **build_value_connections(store, elf), **resolve_indirects(store, elf),
                      **rewrite_candidates(project, store, elf)}
        elif args.command == "audit":
            from .readability import audit_saved_bodies
            result = audit_saved_bodies(project, store, elf)
        elif args.command == "inspect":
            from .reports import inspect_function
            result = inspect_function(store, elf.sha256, args.function)
            if args.output:
                write_json(args.output, result)
                result = {"output": str(args.output), "address": result["address"], "name": result["name"]}
        elif args.command == "import-ghidra":
            from .ghidra import import_export
            result = import_export(args.export, store, None if args.source else elf)
        elif args.command == "match":
            from .matching import match_exports
            result = match_exports(store, args.source_artifact, elf.sha256)
        elif args.command == "plan":
            from .decisions import make_plan
            result = make_plan(project, store, elf, args.candidate,
                               args.output or project / "symbols/analysis/change_plan.json")
        else:
            from .decisions import apply_plan
            result = apply_plan(project, store, elf, args.plan)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        store.db.rollback()
        parser.exit(2, f"Analysis refused: {exc}\n")
    finally:
        store.close()
