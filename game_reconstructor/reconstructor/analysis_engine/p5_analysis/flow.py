"""Resolve static indirect-call expressions while retaining ambiguity."""
import json


def parameter_roots(expression):
    if expression.get("kind") == "parameter":
        return {expression["slot"]}
    return set().union(*(parameter_roots(child) for child in expression.get("inputs", [])))


def parameter_offset(expression):
    if expression.get("kind") == "parameter":
        return expression["slot"], 0
    if expression.get("kind") != "expression":
        return None
    operation, inputs = expression["operation"], expression["inputs"]
    if operation in ("COPY", "CAST", "INT_ZEXT") and len(inputs) == 1:
        return parameter_offset(inputs[0])
    if operation in ("INT_ADD", "PTRSUB", "INT_SUB") and len(inputs) == 2:
        base = parameter_offset(inputs[0])
        if base is not None and inputs[1].get("kind") == "constant":
            displacement = inputs[1]["value"]
            size = inputs[1].get("size", 0)
            if size and displacement & (1 << (size * 8 - 1)):
                displacement -= 1 << (size * 8)
            return base[0], base[1] + (-displacement if operation == "INT_SUB" else displacement)
    return None


def build_value_connections(store, elf):
    source = "analysis:value-connections"
    store.clear_source(elf.sha256, source)
    count = 0
    for row in store.rows("SELECT * FROM facts WHERE artifact=? AND kind='value_dependency'", (elf.sha256,)):
        value = json.loads(row["evidence"])
        op, inputs = value["operation"], value["inputs"]
        common = {"dependency_id": row["id"], "site": value["site"]}
        if op in ("CALL", "CALLIND"):
            target = inputs[0].get("value") if inputs and inputs[0].get("kind") == "constant" else None
            for slot, expression in enumerate(inputs[1:]):
                store.fact(elf.sha256, row["function"], "argument_flow", source, "ghidra_recovered",
                           {**common, "callee": target, "callee_parameter_slot": slot,
                            "caller_parameter_roots": sorted(parameter_roots(expression)), "expression": expression})
                count += 1
        elif op == "RETURN":
            for expression in inputs[1:]:
                store.fact(elf.sha256, row["function"], "return_flow", source, "ghidra_recovered",
                           {**common, "caller_parameter_roots": sorted(parameter_roots(expression)), "expression": expression})
                count += 1
        elif op in ("LOAD", "STORE") and len(inputs) > 1:
            expression = inputs[1]
            base = parameter_offset(expression)
            if base is not None:
                store.fact(elf.sha256, row["function"], "parameter_field_access", source, "ghidra_recovered",
                           {**common, "parameter_slot": base[0], "offset": base[1],
                            "access": "read" if op == "LOAD" else "write", "expression": expression,
                            "limitation": "An offset and parameter relation does not establish a structure type"})
                count += 1
            values, caveats = constant_values(expression, elf)
            for target in values:
                if elf.mapped(target):
                    store.fact(elf.sha256, row["function"], "global_access", source, "ghidra_recovered",
                               {**common, "target": target, "access": "read" if op == "LOAD" else "write",
                                "limitations": caveats, "expression": expression})
                    count += 1
    store.db.commit()
    return {"value_connections": count}


def constant_values(expression, elf, limit=32):
    kind = expression.get("kind")
    if kind == "constant":
        return {expression["value"]}, []
    if kind != "expression":
        return set(), ["nonconstant_input"]
    operation = expression["operation"]
    width = expression.get("size", 0)
    if width not in (1, 2, 4, 8):
        return set(), ["unknown_expression_width"]
    mask = (1 << (8 * width)) - 1
    parts = [constant_values(child, elf, limit) for child in expression["inputs"]]
    if any(not values for values, _ in parts):
        return set(), sorted({reason for _, reasons in parts for reason in reasons})
    sets = [values for values, _ in parts]
    caveats = [reason for _, reasons in parts for reason in reasons]
    if operation in ("COPY", "CAST", "INT_ZEXT") and len(sets) == 1:
        return {v & mask for v in sets[0]}, caveats
    if operation == "INT_SEXT" and len(sets) == 1:
        size = expression["inputs"][0].get("size", 0)
        if size not in (1, 2, 4, 8):
            return set(), ["unknown_sign_extension_width"]
        bits = size * 8
        return {((v & ((1 << bits) - 1)) - (1 << bits) if v & (1 << (bits - 1)) else v) & mask for v in sets[0]}, caveats
    if operation == "MULTIEQUAL":
        values = set().union(*sets)
        return (values, caveats + ["multiple_control_flow_inputs"]) if len(values) <= limit else (set(), ["candidate_limit"])
    if operation == "LOAD" and len(sets) == 2:
        if not expression.get("is_default_memory_space"):
            return set(), ["unverified_load_address_space"]
        values = set()
        for location in sets[1]:
            data = elf.read(location, width)
            if data is not None:
                values.add(int.from_bytes(data, "big"))
        return values, caveats + ["initial_image_pointer_may_change_at_runtime"]
    if len(sets) == 2 and operation in ("INT_ADD", "PTRSUB", "INT_SUB"):
        values = {a - b if operation == "INT_SUB" else a + b for a in sets[0] for b in sets[1]}
    elif len(sets) == 3 and operation == "PTRADD":
        values = {base + index * scale for base in sets[0] for index in sets[1] for scale in sets[2]}
    else:
        return set(), ["unsupported_expression:" + operation]
    return ({v & mask for v in values}, caveats) if len(values) <= limit else (set(), ["candidate_limit"])


def resolve_indirects(store, elf):
    source = "analysis:static-indirect-resolution"
    store.clear_source(elf.sha256, source)
    functions = {r["address"] for r in store.rows("SELECT address FROM functions WHERE artifact=?", (elf.sha256,))}
    descriptors = {}
    for row in store.rows("SELECT evidence FROM facts WHERE artifact=? AND kind='ps3_function_descriptor'", (elf.sha256,)):
        value = json.loads(row["evidence"])
        descriptors[value["descriptor"]] = value["entry"]
    resolved, unresolved = 0, 0
    for row in store.rows("SELECT * FROM facts WHERE artifact=? AND kind='value_dependency'", (elf.sha256,)):
        value = json.loads(row["evidence"])
        if value["operation"] != "CALLIND":
            continue
        possibilities, caveats = constant_values(value["inputs"][0], elf)
        targets = {p if p in functions else descriptors.get(p) for p in possibilities}
        targets.discard(None)
        if not targets:
            store.edge(elf.sha256, row["function"], None, value["site"], "indirect_call", source, "unresolved",
                       {"dependency_id": row["id"], "reasons": caveats, "expression": value["inputs"][0]})
            unresolved += 1
        else:
            status = "resolved_static" if len(targets) == 1 and not caveats else "candidate"
            for target in sorted(targets):
                store.edge(elf.sha256, row["function"], target, value["site"], "indirect_call", source, status,
                           {"dependency_id": row["id"], "pointer_values": sorted(possibilities), "limitations": caveats})
            resolved += 1
    store.db.commit()
    return {"indirect_sites_with_candidates": resolved, "unresolved_indirect_sites": unresolved}
