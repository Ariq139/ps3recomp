"""Conservative, opt-in straight-line GPR promotion for generated C.

Only exact whole-register assignments with proven context-free integer RHSs
are accepted. All other lines (labels, branches, calls, memory, tracing,
architectural flags, preprocessor directives) are materialization barriers.
This is not a full CFG/liveness pass. Keep runs short to bound live registers.
"""
import re

_REG = re.compile(r"ctx->gpr\[(\d+)\]")
_ASSIGN = re.compile(r"\s*ctx->gpr\[(\d+)\] = ([^\n;]+);\s*")
_CAST = re.compile(r"\b(?:u?int(?:8|16|32|64)_t)\b")
_INT = re.compile(r"-?(?:0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*")
_SPU_CALL = re.compile(r"(spu_[a-z0-9]+)\(([^()]*)\)")
# By-value helpers in runtime/spu/spu_helpers.h; no guest context,
# memory, channel, floating-point or halt effects.
_SPU_PURE = frozenset("spu_" + n for n in (
    "a", "ah", "ai", "ahi", "sf", "sfh", "sfi", "sfhi",
    "and", "or", "xor", "nand", "nor", "andc", "orc", "eqv",
    "il", "ila", "ilh", "ilhu", "iohl", "andi", "ori", "xori",
    "ceq", "ceqh", "ceqb", "cgt", "cgth", "cgtb", "clgt",
    "clgth", "clgtb", "selb",
))


# Instruction-effect declarations: only non-record, non-OE PPU integer
# operations and by-value SPU integer helpers can establish eligibility.
# The generated C shape is a second independent check, not the source of
# permission. No instruction that can access memory, trap, update another
# architectural register, halt, or invoke context-aware code belongs here.
PPU_PURE_GPR = frozenset((
    "li", "lis", "addi", "addis", "add", "subf", "neg",
    "mr", "ori", "oris", "xori", "xoris", "and", "or", "xor",
    "nand", "nor", "eqv", "andc", "orc", "extsb", "extsh", "extsw",
))
SPU_PURE_GPR = frozenset(name[4:] for name in _SPU_PURE)


def _candidate(line, arch):
    m = _ASSIGN.fullmatch(line)
    if not m:
        return None
    dest, rhs = int(m[1]), m[2]
    limit = 32 if arch == "ppu" else 128
    if dest >= limit or any(int(r) >= limit for r in _REG.findall(rhs)):
        return None
    if arch == "ppu":
        # Only emitted integer expressions composed of registers, integral
        # casts, numeric literals and non-mutating arithmetic/bitwise tokens.
        # No helper calls, dereferences, division, state/flag access or comma.
        simple = _CAST.sub("T", _REG.sub("R", rhs))
        simple = re.sub(r"(?<![A-Za-z0-9_])(?:0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*(?![A-Za-z0-9_])",
                        "N", simple)
        if not re.fullmatch(r"[TRN\s()+*~&|^<>-]+", simple):
            return None
        if any(x in simple for x in ("++", "--", "&&", "||")):
            return None
        # Restrict multiplication to a binary operator, never pointer
        # dereference. The input is already well-formed lifter output.
        if re.search(r"(?:^|[(+\-~&|^<>])\s*\*", simple):
            return None
    else:
        call = _SPU_CALL.fullmatch(rhs)
        if not call or call[1] not in _SPU_PURE:
            return None
        args = [a.strip() for a in call[2].split(",")]
        if not all(_REG.fullmatch(a) or _INT.fullmatch(a) for a in args):
            return None
    return dest, rhs


def promote_register_regions(lines, arch, eligible_lines):
    """Promote runs of 2–8 assignments; always commit every written GPR.

    eligible_lines is produced only by lowering whitelisted decoded
    instructions. Unknown or synthesized statements are materialization
    barriers even if their textual shape resembles an eligible assignment.
    Reads before a first write remain reads of ctx. Once a local is assigned,
    later reads of that register use the local. Braces keep goto targets
    outside the lifetime of declarations. Do not change input statements
    beyond register substitution. Single statements retain original form.
    """
    if arch not in ("ppu", "spu"):
        raise ValueError(arch)
    typ = "uint64_t" if arch == "ppu" else "u128"
    out, run = [], []

    def flush():
        if len(run) < 2:
            out.extend(line for line, _, _ in run)
        else:
            out.append("    { /* straight-line GPR region */")
            defined = set()
            for _, dest, rhs in run:
                rhs = _REG.sub(lambda m: ("_rr" + m[1]
                               if int(m[1]) in defined else m[0]), rhs)
                decl = typ + " " if dest not in defined else ""
                out.append(f"        {decl}_rr{dest} = {rhs};")
                defined.add(dest)
            for dest in sorted(defined):
                out.append(f"        ctx->gpr[{dest}] = _rr{dest};")
            out.append("    }")
        run.clear()

    for line in lines:
        candidate = _candidate(line, arch) if line in eligible_lines else None
        if candidate is None:
            flush()
            out.append(line)
        else:
            run.append((line, *candidate))
            if len(run) == 8:
                flush()
    flush()
    return out
