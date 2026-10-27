"""Load accepted decisions into the splitter without mixing names and bodies."""
import json

from .common import IDENTIFIER, digest, file_hash, inside, read_csv


def load_saved(project):
    root = project / "symbols/analysis"
    result = {}
    artifact = file_hash(project / "game/EBOOT.ELF")
    inventory = None
    for kind in ("names", "paths", "bodies"):
        path = root / f"accepted_{kind}.json"
        if not path.is_file():
            result[kind] = {}
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != 1 or value.get("artifact_sha256") != artifact:
            raise ValueError(f"Saved {kind} decisions belong to another executable or schema")
        if inventory is None:
            inventory = {r["address"].removeprefix("0x").upper(): r for r in read_csv(project / "symbols/EBOOT_GAME_FUNCTION_MODULES.csv")}
        for ea, record in value["records"].items():
            if ea not in inventory or inventory[ea]["body_sha256"] != record["expected_source_hash"]:
                raise ValueError(f"Stale saved decision at 0x{ea}")
            if kind == "names" and not IDENTIFIER.fullmatch(record["name"]):
                raise ValueError("Saved name is not a C identifier")
            if kind == "paths":
                inside(project / "src/ps3/ppu/split", record["directory"])
            if kind == "bodies" and digest(record["body"].encode()) != record["body_sha256"]:
                raise ValueError("Saved readable-body hash mismatch")
            if kind == "bodies":
                from .readability import RULE_HASH, RULE_ID
                proof = record["evidence"]
                if proof["rule"] != RULE_ID or proof["rule_hash"] != RULE_HASH:
                    raise ValueError("Saved readable rewrite uses an unknown or changed rule")
                if file_hash(inside(project, proof["helper"])) != proof["helper_sha256"]:
                    raise ValueError("Saved readable rewrite helper has changed")
        result[kind] = value["records"]
    return result


def apply_saved(functions, rename_map, saved, path_overrides):
    for ea, record in saved["bodies"].items():
        fn = functions.get(ea)
        if fn is None or fn.body_hash != record["expected_source_hash"]:
            raise ValueError(f"Saved readable body does not match original source 0x{ea}")
    owners = {fn.compiled_symbol or fn.symbol: ea for ea, fn in functions.items()}
    for ea, record in saved["names"].items():
        fn = functions.get(ea)
        if fn is None or fn.body_hash != record["expected_source_hash"]:
            raise ValueError(f"Saved source-name decision does not match original body 0x{ea}")
        label = record["name"]
        if label in owners and owners[label] != ea:
            raise ValueError(f"Saved name collision: {label}")
        owners.pop(fn.compiled_symbol or fn.symbol, None)
        owners[label] = ea
        fn.compiled_symbol = label
        fn.inferred_name_basis = "analysis verified source label: " + record["candidate_id"]
        rename_map[ea] = label
    for ea, record in saved["paths"].items():
        fn = functions.get(ea)
        if fn is None or fn.body_hash != record["expected_source_hash"]:
            raise ValueError(f"Saved folder decision does not match original body 0x{ea}")
        path_overrides[ea] = record["directory"]
    return rename_map
