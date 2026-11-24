"""Exact, isolated integer idioms for structured PPU output.

Each rule replaces one complete canonical assignment. There is no buffered
register state, cross-instruction propagation, memory rewriting, or FP
reassociation. The original statement APIs remain unchanged.
"""

import re

from ppu_readable_format import tokenize_cpp


_GPR = r"ctx->gpr\[(?:[0-9]|[12][0-9]|3[01])\]"
_LITERAL = r"-?(?:0x[0-9A-Fa-f]+|[0-9]+)"
_ASSIGN = re.compile(rf"(?P<dst>{_GPR}) = (?P<expr>.*);")
_LI = re.compile(rf"\(int64_t\)\(int32_t\)\((?P<imm>{_LITERAL})\)")
_SHIFTED = re.compile(rf"\(int64_t\)\(int32_t\)\(\(uint32_t\)(?P<imm>{_LITERAL}) << 16\)")
_ADDI = re.compile(rf"(?P<src>{_GPR}) \+ \(int64_t\)\((?P<imm>{_LITERAL})\)")
_ADDIS = re.compile(rf"(?P<src>{_GPR}) \+ (?P<shifted>\(int64_t\)\(int32_t\)\(\(uint32_t\)(?P<imm>{_LITERAL}) << 16\))")
_BINARY = re.compile(rf"(?P<a>{_GPR}) (?P<op>[+*|&^\-]) (?P<b>{_GPR})")


def _integer(text):
    """Read a C integer token without eval; do not reinterpret octal as decimal."""
    negative = text.startswith("-")
    value = text[1:] if negative else text
    radix = 16 if value.startswith("0x") else (8 if len(value) > 1 and value.startswith("0") else 10)
    try:
        number = int(value, radix)
    except ValueError:
        return None
    return -number if negative else number


def _small_immediate(match, shifted=False):
    immediate = _integer(match["imm"])
    # The disassembler prints lis's encoded upper half as an unsigned 16-bit
    # literal; addis can use a signed spelling. Both feed the same uint32_t
    # shift and subsequent signed-word conversion in the canonical lowering.
    upper = 65535 if shifted else 32767
    return immediate if immediate is not None and -32768 <= immediate <= upper else None


def _shifted_value(immediate):
    word = ((immediate & 0xFFFFFFFF) << 16) & 0xFFFFFFFF
    return word - (1 << 32) if word & 0x80000000 else word


def _u64_literal(value):
    value &= (1 << 64) - 1
    return str(value) if value < 10 else f"0x{value:X}ULL"


def _add(dst, src, value):
    if value == 0:
        return f"{dst} = {src};"
    op = "+" if value > 0 else "-"
    literal = _u64_literal(abs(value))
    if dst == src:
        return f"{dst} {op}= {literal};"
    return f"{dst} = {src} {op} {literal};"


def simplify_body(lines: list[str]) -> list[str]:
    """Simplify known full-register assignments, preserving each write site."""
    result = []
    for line in lines:
        tokens = tokenize_cpp(line)
        # Only trailing comments are accepted. Code-looking strings, directives,
        # inline comments inside operands, and compound statements stay opaque.
        significant = [t for t in tokens if t.kind != "space"]
        if not significant or any(t.kind in {"unknown", "directive", "string", "char"}
                                  for t in significant):
            result.append(line)
            continue
        last_code = next((i for i in range(len(significant) - 1, -1, -1)
                          if significant[i].kind != "comment"), -1)
        if last_code < 0 or any(t.kind == "comment" for t in significant[:last_code]):
            result.append(line)
            continue
        end = significant[last_code].end
        code = line[:end].strip()
        suffix = line[end:]
        assignment = _ASSIGN.fullmatch(code)
        if assignment is None:
            result.append(line)
            continue
        dst, expr = assignment["dst"], assignment["expr"]
        replacement = None
        match = _LI.fullmatch(expr)
        if match and (value := _small_immediate(match)) is not None:
            replacement = f"{dst} = {_u64_literal(value)};"
        elif (match := _SHIFTED.fullmatch(expr)) and (value := _small_immediate(match, shifted=True)) is not None:
            replacement = f"{dst} = {_u64_literal(_shifted_value(value))};"
        elif (match := _ADDI.fullmatch(expr)) and (value := _small_immediate(match)) is not None:
            replacement = _add(dst, match["src"], value)
        elif (match := _ADDIS.fullmatch(expr)) and (value := _small_immediate(match, shifted=True)) is not None:
            replacement = _add(dst, match["src"], _shifted_value(value))
        elif (match := _BINARY.fullmatch(expr)):
            a, b, op = match["a"], match["b"], match["op"]
            if a == b and op in {"|", "&"}:
                replacement = f"{dst} = {a};"
            elif a == b and op == "^":
                replacement = f"{dst} = 0;"
            elif a == dst:
                replacement = f"{dst} {op}= {b};"
        if replacement is None:
            result.append(line)
        else:
            indent = line[:len(line) - len(line.lstrip())]
            result.append(indent + replacement + suffix)
    return result
