"""Strict rewrite rules and the evidence required to apply them."""
import json
import re

from .common import digest, direct_branch, inside, file_hash, read_csv, rewrite_tokens, function_bodies, write_json


OPERATIONS = {
    "CodeFunc_Add": (0, "Add"), "CodeFunc_Sub": (1, "Subtract"),
    "CodeFunc_Mul": (2, "Multiply"), "CodeFunc_Div": (3, "Divide"),
    "CodeFunc_Or": (4, "LogicalOr"), "CodeFunc_And": (5, "LogicalAnd"),
    "CodeFunc_EQ": (6, "Equal"), "CodeFunc_NEQ": (7, "NotEqual"),
    "CodeFunc_S": (8, "LessThan"), "CodeFunc_L": (9, "GreaterThan"),
    "CodeFunc_SE": (10, "LessOrEqual"), "CodeFunc_LE": (11, "GreaterOrEqual"),
}

TEMPLATE = '''void func_ADDRESS(ppu_context* ctx) {
    uint64_t _cs_31 = ctx->gpr[31];
    vm_write64(ctx->gpr[1] + -0x80, ctx->gpr[1]); ctx->gpr[1] += -0x80;
    ctx->gpr[0] = ctx->lr;
    vm_write64(ctx->gpr[1] + 0x90, ctx->gpr[0]);
    vm_write64(ctx->gpr[1] + 0x78, ctx->gpr[31]);
    ctx->gpr[31] = ctx->gpr[3];
    ctx->gpr[4] = (int64_t)(int32_t)(OPERATION);
    ctx->lr = 0xRETURNPC; func_001F35B8(ctx); DRAIN_TRAMPOLINE(ctx);
    ctx->gpr[4] = vm_read32(ctx->gpr[31] + 0x28);
    ctx->gpr[3] = (int64_t)(int32_t)(1);
    { uint64_t _a = ctx->gpr[4]; uint64_t _r = _a + (uint64_t)(int64_t)(1); ctx->gpr[4] = _r; ctx->xer = (ctx->xer & ~(1u << 29)) | ((_r < _a) ? (1u << 29) : 0u); }
    vm_write32(ctx->gpr[31] + 0x28, ctx->gpr[4]);
    ctx->gpr[0] = vm_read64(ctx->gpr[1] + 0x90);
    ctx->lr = ctx->gpr[0];
    ctx->gpr[31] = _cs_31;
    ctx->gpr[1] = ctx->gpr[1] + (int64_t)(0x80);
    return;
}'''

RULE_ID = "flowscript-binary-opcode-v1"
RULE_HASH = digest(TEMPLATE.encode())
HELPER_HASH = "7c615949fc25b49fc135f8c99fba3b0d1dbed196f3e480e8600c907be0f1463b"


def canonical(data):
    from split_lifted_ppu import simplify_body, strip_c_comments
    return re.sub(rb"\s+", b"", strip_c_comments(simplify_body(data)))


def prove_opcode_rule(body, ea, name, elf):
    if name not in OPERATIONS:
        return None
    operation, enum_name = OPERATIONS[name]
    literals = re.findall(rb"ctx->gpr\[4\] = \(int64_t\)\(int32_t\)\((0x[0-9A-Fa-f]+|\d+)\);", body)
    if not literals or int(literals[0], 0) != operation:
        return None
    calls = re.findall(rb"ctx->lr = 0x([0-9A-Fa-f]+); func_001F35B8\(ctx\);", body)
    if len(calls) != 1:
        return None
    return_pc = int(calls[0], 16)
    expected = TEMPLATE.replace("ADDRESS", f"{ea:08X}").replace("OPERATION", literals[0].decode()).replace("RETURNPC", f"{return_pc:08X}").encode()
    if canonical(body) != canonical(expected):
        return None
    if direct_branch(elf.u32(return_pc - 4), return_pc - 4) != ("direct_call", 0x001F35B8):
        return None
    if elf.u32(return_pc - 8) != 0x38800000 | operation:
        return None
    native_words = [0xF821FF81, 0x7C0802A6, 0xF8010090, 0xFBE10078, 0x607F0000,
                    0x38800000 | operation,
                    0x48000001 | ((0x001F35B8 - (ea + 24)) & 0x03FFFFFC),
                    0x809F0028, 0x38600001, 0x30840001, 0x909F0028, 0xE8010090,
                    0x7C0803A6, 0xEBE10078, 0x38210080, 0x4E800020]
    if return_pc != ea + 28 or any(elf.u32(ea + i * 4) != word for i, word in enumerate(native_words)):
        return None
    generated = (f"void func_{ea:08X}(ppu_context* ctx) {{\n"
                 f"    p5::scripts::execute_binary_opcode(ctx,\n"
                 f"        p5::scripts::BinaryOperation::{enum_name}, func_001F35B8, 0x{return_pc:08X});\n"
                 "}\n")
    return {"rule": RULE_ID, "rule_hash": RULE_HASH, "raw_body_hash": digest(body),
            "native_body_sha256": digest(b"".join(word.to_bytes(4, "big") for word in native_words)),
            "body": generated, "body_sha256": digest(generated.encode()),
            "operation": operation, "return_pc": return_pc,
            "preserved_effects": ["guest stack stores and restore reads", "all GPR writes", "LR and trampoline sequence", "XER carry bit", "32-bit instruction-index store"],
            "source_evidence": "SW.c CodeFunc_2Operation cases 0 through 11; existing accepted CodeFunc source labels",
            "helper": "include/p5/scripts/binary_opcode.h"}


def rewrite_candidates(project, store, elf):
    from .accepted import load_saved
    accepted = load_saved(project)["bodies"]
    artifact = elf.sha256
    store.db.execute("DELETE FROM candidates WHERE artifact=? AND kind='body'", (artifact,))
    count = 0
    if file_hash(project / "include/p5/scripts/binary_opcode.h") != HELPER_HASH:
        raise ValueError("The checked binary-opcode helper has changed")
    for row in store.rows("SELECT * FROM functions WHERE artifact=?", (artifact,)):
        if row["name"] not in OPERATIONS:
            continue
        facts = store.rows("SELECT * FROM facts WHERE artifact=? AND function=? AND kind='original_lift_body'", (artifact, row["address"]))
        if len(facts) != 1:
            raise ValueError("Expected exactly one original body observation")
        fact = facts[0]
        loc = json.loads(fact["evidence"])
        with inside(project, fact["source"]).open("rb") as source:
            source.seek(loc["byte_offset"])
            body = source.read(loc["byte_length"])
        if digest(body) != row["source_hash"]:
            raise ValueError("Original body changed; reindex before proposing rewrites")
        proof = prove_opcode_rule(body, row["address"], row["name"], elf)
        if proof:
            previous = accepted.get(f"{row['address']:08X}")
            if previous and previous["body_sha256"] == proof["body_sha256"]:
                continue
            proof["helper_sha256"] = file_hash(project / proof["helper"])
            store.candidate(artifact, row["address"], "body", RULE_ID, "verified",
                            {**proof, "expected_name": row["name"], "expected_source_hash": row["source_hash"]})
            count += 1
    store.db.commit()
    return {"verified_rewrite_candidates": count, "rule": RULE_ID,
            "limitation": "Only the complete matched handler template is rewritten; no arbitrary pseudocode substitution"}


def audit_saved_bodies(project, store, elf):
    from .accepted import load_saved
    saved = load_saved(project)["bodies"]
    inventory = {int(r["address"], 16): r for r in read_csv(project / "symbols/EBOOT_GAME_FUNCTION_MODULES.csv")}
    symbols = {f"func_{ea:08X}": row["compiled_symbol"] for ea, row in inventory.items()}
    count, details = 0, []
    for key, record in saved.items():
        ea = int(key, 16)
        facts = store.rows("SELECT * FROM facts WHERE artifact=? AND function=? AND kind='original_lift_body'", (elf.sha256, ea))
        if len(facts) != 1:
            raise ValueError("An accepted rewrite lacks its unique original-body observation")
        location = json.loads(facts[0]["evidence"])
        with inside(project, facts[0]["source"]).open("rb") as source:
            source.seek(location["byte_offset"])
            body = source.read(location["byte_length"])
        if digest(body) != record["expected_source_hash"]:
            raise ValueError("An accepted rewrite's original body has changed")
        proof = prove_opcode_rule(body, ea, record["evidence"]["expected_name"], elf)
        if proof is None or proof["body"] != record["body"]:
            raise ValueError("An accepted rewrite no longer satisfies its native/template rule")
        row = inventory[ea]
        path = inside(project / "src/ps3/ppu", row["split_source"])
        current = function_bodies(path.read_bytes()).get(row["compiled_symbol"])
        expected = rewrite_tokens(record["body"].replace("\n", "\r\n").encode(), symbols)
        if current != expected:
            raise ValueError(f"Accepted implementation has local edits: {row['compiled_symbol']}")
        details.append({"address": row["address"], "name": row["compiled_symbol"],
                        "native_body_sha256": proof["native_body_sha256"], "original_lift_body_sha256": digest(body),
                        "active_body_sha256": digest(current), "helper_sha256": record["evidence"]["helper_sha256"],
                        "rule": proof["rule"]})
        count += 1
    result = {"accepted_bodies_reproved": count, "active_bodies_verified": count,
              "artifact_sha256": elf.sha256, "build_run": False}
    output = project / "symbols/analysis/reports/readable_audit.json"
    write_json(output, {**result, "implementations": details,
                        "limitation": "Checks matched rewrite rules and source identity; this is not a game build or runtime test"})
    return {**result, "audit": str(output)}
