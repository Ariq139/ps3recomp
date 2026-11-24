# ExportSpuImageNames.py - Run INSIDE Ghidra's Script Manager on an SPU image.
#@category PS3Recomp
#
# This is a producer of the EXISTING ps3recomp address -> {"label": ...} map.
# It is not a duplicate host-side CSV/JSON importer.
# Import/open the *SPU image itself*, not the PPU GZF, before running.
# The image SHA-256 must be computed from the EXACT image supplied to the
# lifter (and the same load/relocation version), then pasted when prompted.
# Refuse guesses or address-only transfers between unrelated images.

import json
import re
from ghidra.program.model.lang import Processor

if currentProgram is None:
    raise RuntimeError("Open the exact SPU program in Ghidra first")

processor = currentProgram.getLanguage().getProcessor()
spu_processor = Processor.findOrPossiblyCreateProcessor("SPU")
if not processor.equals(spu_processor):
    raise RuntimeError(
        "This is not an SPU program. Open one of the extracted raw SPU images; "
        "PPU EBOOT GZF projects must not be used for SPU labels.")

try:
    image_hash = str(currentProgram.getExecutableSHA256()).lower()
except Exception:
    image_hash = ""
if not re.match(r"^[0-9a-f]{64}$", image_hash):
    image_hash = askString(
        "SPU image provenance", "Exact input SPU SHA-256 from the image manifest")
if not re.match(r"^[0-9a-fA-F]{64}$", image_hash):
    raise RuntimeError("Expected the SHA-256 of the precise lifted SPU input")

output = askFile("Export SPU function names as ps3recomp JSON", "Save")
name_map = {
    "_meta": {
        "schema": "ps3recomp address-label JSON v1",
        "source": "Ghidra FunctionManager, SPU program",
        "image_sha256": image_hash.lower(),
        "program_name": str(currentProgram.getName()),
        "address_kind": "SPU local-store function entry",
        "note": "Do not reuse by LS address alone on another image"
    }
}
try:
    name_map["_meta"]["ghidra_executable_md5"] = str(currentProgram.getExecutableMD5())
except Exception:
    pass

count = 0
seen = {}
for fun in currentProgram.getFunctionManager().getFunctions(True):
    if monitor.isCancelled():
        raise RuntimeError("Symbol export cancelled")
    sym = fun.getSymbol()
    if sym is None or str(sym.getSource()) == "DEFAULT":
        continue
    addr = fun.getEntryPoint()
    if addr is None or not addr.isMemoryAddress():
        continue
    off = int(addr.getOffset())
    if off < 0 or off >= 0x40000:  # 256 KiB SPU local store.
        continue
    try:
        label = str(fun.getName(True))
    except Exception:
        label = str(fun.getName())
    if not label or '\n' in label or '\r' in label:
        continue
    key = "0x%08X" % off
    if key in seen and seen[key] != label:
        raise RuntimeError("Conflicting function labels at " + key)
    seen[key] = label
    name_map[key] = {"label": label}
    count += 1

with open(output.getAbsolutePath(), 'w') as fp:
    json.dump(name_map, fp, indent=2, sort_keys=True)
println("Exported %d SPU function symbols to %s" % (count, output.getAbsolutePath()))
