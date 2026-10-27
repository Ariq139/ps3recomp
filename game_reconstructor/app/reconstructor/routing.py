"""Offline folder policies for source-labeled functions."""
import json
import re


SOFDEC_HELPERS = {"setLoopSofdec", "setABLoopSofdec", "getStatusSofdec", "chkReadySofdec"}


def source_folder(function, explicit_paths):
    address = f"{function['address']:08X}"
    if address in explicit_paths:
        return explicit_paths[address], "explicit accepted function path"
    metadata = json.loads(function["metadata"])
    name = function["name"]
    if any(alias.upper().startswith("SCRIPT_") for alias in metadata.get("callable_names", "").split(";")):
        return None
    if re.match(r"^_*(?:cri|Cri)[A-Z_]", name) or name in SOFDEC_HELPERS:
        return "criware", "CRIWARE source symbol family"
    if re.match(r"^_*cell[A-Z_]", name):
        return "cell", "CELL source symbol family"
    return None
