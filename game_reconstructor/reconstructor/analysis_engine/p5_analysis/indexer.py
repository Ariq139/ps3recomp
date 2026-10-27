import json
import re
from collections import Counter

from .common import Elf, FUNCTION, address, digest, direct_branch, file_hash, json_text, read_csv, write_json


CALL = re.compile(rb"ctx->lr\s*=\s*0x([0-9A-Fa-f]+);\s*func_([0-9A-Fa-f]{8})\(ctx\)")
REFERENCE = re.compile(rb"\bfunc_([0-9A-Fa-f]{8})\(ctx\)")
MEMORY = re.compile(rb"vm_(read|write)(8|16|32|64)\(ctx->gpr\[(\d+)\]\s*\+\s*(-?0x[0-9A-Fa-f]+|-?\d+)")


def index_project(project, store):
    elf = Elf(project / "game/EBOOT.ELF")
    artifact = elf.sha256
    baseline_path = project / "symbols/analysis/baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
    if baseline and baseline["artifact_sha256"] != artifact:
        raise ValueError("The executable differs from this project's analysis baseline")
    store.record("artifacts", id=artifact, path=str(elf.path), kind="release_elf",
                 metadata=json_text({"machine": "EM_PPC64", "endian": "big", "bits": 64,
                                     "language_status": "Ghidra language recorded per export"}))
    manifest_path = project / "symbols/EBOOT_GAME_FUNCTION_MODULES.csv"
    manifest_hash = file_hash(manifest_path)
    manifest = read_csv(manifest_path)
    byaddr = {address(row["address"]): row for row in manifest}
    source_identity = digest(json_text(sorted((ea, row["body_sha256"]) for ea, row in byaddr.items())).encode())
    if baseline and baseline["source_identity"] != source_identity:
        raise ValueError("The original function inventory differs from the analysis baseline")
    if len(byaddr) != len(manifest):
        raise ValueError("Duplicate manifest function addresses")
    for ea, row in byaddr.items():
        if not elf.executable(ea):
            raise ValueError(f"Manifest entry outside file-backed executable memory: {ea:#x}")
        store.record("functions", artifact=artifact, address=ea, name=row["compiled_symbol"],
                     source_hash=row["body_sha256"], metadata=json_text(row))
    existing = {r["address"] for r in store.rows("SELECT address FROM functions WHERE artifact=?", (artifact,))}
    for removed in existing - byaddr.keys():
        store.db.execute("DELETE FROM functions WHERE artifact=? AND address=?", (artifact, removed))
    store.record("inputs", artifact=artifact, path="symbols/EBOOT_GAME_FUNCTION_MODULES.csv", sha256=manifest_hash)

    changed, reused = 0, 0
    raw = project / "src/ps3/ppu/raw"
    if not raw.is_dir():
        raise ValueError("Original raw lift directory is missing")
    shard_paths = sorted(raw.glob("ppu_recomp_*.cpp"))
    if not shard_paths:
        raise ValueError("Original lift shards are missing")
    if {p.name for p in shard_paths} != {row["input_file"] for row in byaddr.values()}:
        raise ValueError("Original shard inventory does not match the manifest")
    for path in shard_paths:
        source = path.relative_to(project).as_posix()
        data = path.read_bytes()
        sha = digest(data)
        previous = store.db.execute("SELECT sha256 FROM inputs WHERE artifact=? AND path=?", (artifact, source)).fetchone()
        if previous and previous[0] == sha:
            reused += 1
            continue
        store.clear_source(artifact, source)
        found = set()
        for match in FUNCTION.finditer(data):
            symbol = match[1].decode("ascii")
            if not symbol.startswith("func_"):
                raise ValueError(f"Unexpected original function name: {symbol}")
            ea = int(symbol[5:], 16)
            row = byaddr.get(ea)
            if row is None or digest(match[0]) != row["body_sha256"]:
                raise ValueError(f"Original body does not match manifest at {ea:#x}")
            found.add(ea)
            body = match[0]
            store.fact(artifact, ea, "original_lift_body", source, "hash_verified",
                       {"byte_offset": match.start(), "byte_length": len(body), "sha256": digest(body)})
            observed_targets = set()
            verified_size = int(row.get("verified_source_size_bytes") or 0)
            for call in CALL.finditer(body):
                site, target = int(call[1], 16) - 4, int(call[2], 16)
                decoded = direct_branch(elf.u32(site), site)
                verified = decoded == ("direct_call", target)
                ownership = "verified_source_extent" if verified_size and ea <= site < ea + verified_size else "raw_lift_assignment"
                if verified_size and not ea <= site < ea + verified_size:
                    status = "outside_verified_body"
                else:
                    status = "instruction_verified" if verified else "unverified"
                store.edge(artifact, ea, target, site,
                           "direct_call" if verified else "lift_call", source, status,
                           {"return_pc": site + 4, "instruction": elf.u32(site),
                            "function_ownership": ownership, "lift_byte_offset": match.start() + call.start()})
                observed_targets.add(target)
            for target in {int(m[1], 16) for m in REFERENCE.finditer(body)} - observed_targets:
                store.edge(artifact, ea, target, None, "lift_reference", source, "unverified",
                           {"reason": "No verified original call-site instruction"})
            offsets = Counter((m[1].decode(), int(m[2]), int(m[3]), int(m[4], 0)) for m in MEMORY.finditer(body))
            for (mode, width, register, offset), count in offsets.items():
                if register == 1:
                    continue
                store.fact(artifact, ea, "register_memory_access", source, "candidate",
                           {"access": mode, "width_bits": width, "base_register": register,
                            "offset": offset, "occurrences": count,
                            "limitation": "Register identity and pointer provenance are not established by offset matching"})
        expected = {ea for ea, row in byaddr.items() if row["input_file"] == path.name}
        if found != expected:
            raise ValueError(f"Shard inventory mismatch: {path.name}: {len(expected-found)} missing, {len(found-expected)} extra")
        store.record("inputs", artifact=artifact, path=source, sha256=sha)
        changed += 1

    catalog_sources = {
        "EBOOT_GAME_CALLABLES.csv": "callable_registration",
        "EBOOT_PRIORITY_SYMBOL_LABELS.csv": "priority_source_match",
        "EBOOT_GHIDRA_US_FUNCTION_LABELS.csv": "accepted_us_function_label",
        "EBOOT_GHIDRA_US_GLOBAL_DATA_LABELS.csv": "accepted_global_label",
        "CKIT_variable_reference.csv": "ckit_declaration_reference",
        "EBOOT.strings.csv": "elf_string_inventory",
    }
    for filename, kind in catalog_sources.items():
        path = project / "symbols" / filename
        if not path.is_file():
            continue
        source = path.relative_to(project).as_posix()
        sha = file_hash(path)
        previous = store.db.execute("SELECT sha256 FROM inputs WHERE artifact=? AND path=?", (artifact, source)).fetchone()
        if previous and previous[0] == sha:
            continue
        store.clear_source(artifact, source)
        for row in read_csv(path):
            if row.get("source_name", "").lower() in ("something_bgm", "kasumipersona"):
                continue
            record = json_text(row)
            store.record("catalog", id=digest((artifact + source + record).encode()), artifact=artifact,
                         kind=kind, source=source, record=record)
            if kind == "accepted_global_label" and "accepted" in row.get("status", ""):
                store.fact(artifact, None, "global_label", source, "accepted_label_only",
                           {"address": address(row["source_address"]), "name": row["source_name"],
                            "limitation": "Name acceptance does not establish type or extent"})
        store.record("inputs", artifact=artifact, path=source, sha256=sha)

    loader = project / "out/EBOOT.loader.json"
    if loader.is_file():
        source = loader.relative_to(project).as_posix()
        metadata = json.loads(loader.read_text(encoding="utf-8-sig"))
        base, size = address(metadata["opd_base"]), address(metadata["opd_size"])
        store.clear_source(artifact, source)
        if size % 8:
            raise ValueError("PS3 descriptor table size is not a multiple of eight")
        for descriptor in range(base, base + size, 8):
            entry, toc = elf.u32(descriptor), elf.u32(descriptor + 4)
            if entry is None or toc is None or not elf.executable(entry):
                continue
            store.fact(artifact, entry, "ps3_function_descriptor", source, "bytes_verified",
                       {"descriptor": descriptor, "entry": entry, "toc": toc,
                        "raw_hex": elf.read(descriptor, 8).hex()})
        store.record("inputs", artifact=artifact, path=source, sha256=file_hash(loader))
    store.db.commit()
    if baseline is None:
        write_json(baseline_path, {"schema": 1, "artifact_sha256": artifact, "source_identity": source_identity,
                                   "identity_basis": "existing project ELF and hash-verified original lift inventory"})
    return {**store.summary(artifact), "shards_indexed": changed, "shards_reused": reused,
            "manifest_sha256": manifest_hash}


def grouping_candidates(store, artifact):
    store.db.execute("DELETE FROM candidates WHERE artifact=? AND kind='folder'", (artifact,))
    functions = {r["address"]: r for r in store.rows("SELECT * FROM functions WHERE artifact=?", (artifact,))}
    roots = {}
    for ea, fn in functions.items():
        meta = json.loads(fn["metadata"])
        if meta.get("classification") == "named_callable" and meta.get("confidence") == "high":
            module = meta.get("module")
            if module not in ("shared", "unclassified", "script_core", "personas"):
                roots[ea] = module
    neighbors = {}
    for edge in store.rows("SELECT * FROM edges WHERE artifact=? AND kind='direct_call' AND status='instruction_verified'", (artifact,)):
        if edge["target"] not in functions:
            continue
        neighbors.setdefault(edge["caller"], set()).add(edge["target"])
        neighbors.setdefault(edge["target"], set()).add(edge["caller"])
    count = 0
    for ea, fn in functions.items():
        meta = json.loads(fn["metadata"])
        if not fn["name"].startswith(("func_", "FUN_")) or meta.get("module") not in ("shared", "unclassified"):
            continue
        supporting = sorted(n for n in neighbors.get(ea, ()) if n in roots)
        modules = {roots[n] for n in supporting}
        if len(modules) == 1:
            store.candidate(artifact, ea, "folder", modules.pop(), "candidate",
                            {"named_neighbors": [{"address": n, "name": functions[n]["name"]} for n in supporting],
                             "limitation": "Call adjacency supports a subsystem suggestion, not function purpose or ownership"})
            count += 1
    store.db.commit()
    return count
