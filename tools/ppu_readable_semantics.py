"""Exact, named integer-comparison and CR-predicate presentation.

The input is the PPU lifter's canonical body after its existing CR fusion.
Recognizers accept only the integer compare blocks and boolean branch grammar
the lifter emits. They preserve register writes, XER.SO, operand widths, and
CTR short-circuit effects. Unknown expressions and FP comparisons stay raw.
"""

import re

from ppu_readable_format import tokenize_cpp


SEMANTIC_PREAMBLE = r"""/* Exact integer compares and boolean CR tests for structured PPU output. */
namespace ppu_readable {
static inline void set_comparison(uint32_t& cr, uint32_t xer, unsigned field,
                                  uint32_t relation)
{
    const unsigned shift = 28u - 4u * field;
    const uint32_t value = relation | ((xer >> 31) & 1u);
    cr = (cr & ~(0xFu << shift)) | (value << shift);
}
static inline void compare_s32(uint32_t& cr, uint32_t xer, unsigned field,
                              uint64_t left, uint64_t right)
{
    const int32_t a = (int32_t)left, b = (int32_t)right;
    set_comparison(cr, xer, field, a < b ? 8u : a > b ? 4u : 2u);
}
static inline void compare_s64(uint32_t& cr, uint32_t xer, unsigned field,
                              uint64_t left, uint64_t right)
{
    const int64_t a = (int64_t)left, b = (int64_t)right;
    set_comparison(cr, xer, field, a < b ? 8u : a > b ? 4u : 2u);
}
static inline void compare_u32(uint32_t& cr, uint32_t xer, unsigned field,
                              uint64_t left, uint64_t right)
{
    const uint32_t a = (uint32_t)left, b = (uint32_t)right;
    set_comparison(cr, xer, field, a < b ? 8u : a > b ? 4u : 2u);
}
static inline void compare_u64(uint32_t& cr, uint32_t xer, unsigned field,
                              uint64_t left, uint64_t right)
{
    set_comparison(cr, xer, field, left < right ? 8u : left > right ? 4u : 2u);
}
static inline bool cr_any(uint32_t cr, unsigned field, uint32_t mask)
{
    return ((cr >> (28u - 4u * field)) & mask) != 0;
}
static inline bool cr_less(uint32_t cr, unsigned field) { return cr_any(cr, field, 8u); }
static inline bool cr_greater(uint32_t cr, unsigned field) { return cr_any(cr, field, 4u); }
static inline bool cr_equal(uint32_t cr, unsigned field) { return cr_any(cr, field, 2u); }
/* The CR low bit is SO for integer compares and unordered for FP compares.
 * cr_so names/tests that architectural bit; it does not infer its producer. */
static inline bool cr_so(uint32_t cr, unsigned field) { return cr_any(cr, field, 1u); }
} // namespace ppu_readable
"""


_REG_INDEX = r"(?:[0-9]|[12][0-9]|3[01])"
_LITERAL = r"-?(?:0[xX][0-9A-Fa-f]+|0|[1-9][0-9]*)"
_REG_OPERAND = re.compile(
    r"\((?P<cast>int32_t|int64_t|uint32_t|uint64_t)\)"
    r"(?P<reg>ctx->gpr\[" + _REG_INDEX + r"\])"
)
_IMM_OPERAND = re.compile(r"\((?P<cast>int64_t|uint64_t)\)(?P<value>" + _LITERAL + r")")
_COMPARE = re.compile(
    r"\{(?P<type>int64_t|uint64_t)a=(?P<a>[^;]+);(?P=type)b=(?P<b>[^;]+);"
    r"uint32_tcr_val=\(\(a<b\)\?8:\(a>b\)\?4:2\)\|\(\(ctx->xer>>31\)&1u\);"
    r"ctx->cr=\(ctx->cr&~\(0xFu<<(?P<shift>[0-9]+)\)\)\|\(cr_val<<(?P=shift)\);\}"
)
_FLAG_NAMES = {8: "cr_less", 4: "cr_greater", 2: "cr_equal", 1: "cr_so"}
_CR_CONTEXT = re.compile(r"\bctx\s*->\s*cr\b")


def _code_tokens(text):
    tokens = [t for t in tokenize_cpp(text) if t.kind != "space"]
    if any(t.kind in {"directive", "unknown"} for t in tokens):
        return None
    return tokens


def _integer(text):
    if not re.fullmatch(_LITERAL, text):
        return None
    sign = -1 if text.startswith("-") else 1
    magnitude = text.lstrip("-")
    return sign * int(magnitude, 16 if magnitude.lower().startswith("0x") else 10)


def _named_compare(line):
    tokens = _code_tokens(line)
    if not tokens or any(t.kind in {"comment", "string", "char"} for t in tokens):
        return None
    # The compact pattern below deliberately ignores formatting, but must not
    # erase declaration boundaries. `int64_t a = ...` declares a temporary;
    # `int64_ta = ...` may instead assign a handwritten global. Likewise for
    # b and cr_val. Accept the actual three declaration token sequences only.
    words = [t.text for t in tokens]
    semicolons = [i for i, word in enumerate(words) if word == ";"]
    if (len(semicolons) != 4 or len(words) < 4 or words[0] != "{"
            or words[1] not in {"int64_t", "uint64_t"}
            or words[2:4] != ["a", "="]
            or words[semicolons[0] + 1:semicolons[0] + 4] != [words[1], "b", "="]
            or words[semicolons[1] + 1:semicolons[1] + 4] != ["uint32_t", "cr_val", "="]):
        return None
    match = _COMPARE.fullmatch("".join(t.text for t in tokens))
    if not match:
        return None
    shift = int(match["shift"])
    if shift not in range(0, 29, 4):
        return None
    left = _REG_OPERAND.fullmatch(match["a"])
    if not left:
        return None
    signed = match["type"] == "int64_t"
    casts = {"int32_t": "s32", "int64_t": "s64"} if signed else {
        "uint32_t": "u32", "uint64_t": "u64"}
    if left["cast"] not in casts:
        return None
    right = _REG_OPERAND.fullmatch(match["b"])
    if right:
        if right["cast"] != left["cast"]:
            return None
        right_value = right["reg"]
    else:
        immediate = _IMM_OPERAND.fullmatch(match["b"])
        if not immediate or immediate["cast"] != match["type"]:
            return None
        value = _integer(immediate["value"])
        # These are the complete immediate domains of cmpi/cmpli. Restricting
        # them also proves a 32-bit helper's right conversion equals the
        # original immediate's 64-bit capture (including negative cmpwi).
        if value is None or not (-32768 <= value <= 32767 if signed else 0 <= value <= 65535):
            return None
        right_value = immediate["value"]
    indent = line[:len(line) - len(line.lstrip())]
    helper = "compare_" + casts[left["cast"]]
    return (f"{indent}::ppu_readable::{helper}(ctx->cr, ctx->xer, {(28 - shift) // 4}, "
            f"{left['reg']}, {right_value});")


def _strip_parens(tokens):
    while len(tokens) >= 2 and tokens[0].text == "(" and tokens[-1].text == ")":
        depth = 0
        complete = True
        for i, token in enumerate(tokens):
            if token.text == "(":
                depth += 1
            elif token.text == ")":
                depth -= 1
            if depth == 0 and i != len(tokens) - 1:
                complete = False
                break
        if not complete or depth:
            break
        tokens = tokens[1:-1]
    return tokens


def _split_operator(tokens, operator):
    depth = 0
    pieces = []
    start = 0
    for i, token in enumerate(tokens):
        if token.text == "(":
            depth += 1
        elif token.text == ")":
            depth -= 1
        elif token.text == operator and depth == 0:
            pieces.append(tokens[start:i])
            start = i + 1
    if pieces:
        pieces.append(tokens[start:])
    return pieces


def _cr_predicate(tokens):
    parts = _split_operator(tokens, "&")
    if len(parts) != 2:
        return None
    left, right = (_strip_parens(part) for part in parts)
    if len(left) != 5 or [t.text for t in left[:4]] != ["ctx", "->", "cr", ">>"]:
        return None
    if len(right) != 1:
        return None
    shift, mask = _integer(left[4].text), _integer(right[0].text)
    if shift is None or mask is None or not 0 <= shift <= 31 or mask <= 0:
        return None
    if mask & (mask - 1) == 0:
        absolute_bit = shift + (mask.bit_length() - 1)
        if absolute_bit > 31:
            return None
        field = (31 - absolute_bit) // 4
        flag = 1 << (absolute_bit % 4)
        return f"::ppu_readable::{_FLAG_NAMES[flag]}(ctx->cr, {field})"
    if shift % 4 == 0 and shift <= 28 and mask <= 0xF:
        return f"::ppu_readable::cr_any(ctx->cr, {(28 - shift) // 4}, 0x{mask:X}u)"
    return None


def _boolean_condition(tokens, source=None):
    """Return (spelling, contains_CR_test) for exact CR predicates.

    Unknown logical atoms keep their original source span. This allows a CR
    test to be named beside a CTR decrement while preserving its short-circuit
    order and every side effect.
    """
    tokens = _strip_parens(tokens)
    if not tokens or any(t.kind in {"comment", "string", "char"} for t in tokens):
        return None
    for operator in ("||", "&&"):
        pieces = _split_operator(tokens, operator)
        if pieces:
            children = []
            for piece in pieces:
                child = _boolean_condition(piece, source)
                if child is None:
                    if source is None or not piece:
                        return None
                    raw = source[piece[0].start:piece[-1].end]
                    if not raw:
                        return None
                    child = (raw, False)
                children.append(child)
            return ("(" + f" {operator} ".join(child[0] for child in children) + ")",
                    any(child[1] for child in children))
    if tokens[0].text == "!":
        operand = tokens[1:]
        # In the generated grammar negation wraps the complete CR/BO atom.
        # Do not reinterpret arbitrary C++ such as `!(cr >> 28) & 2` as
        # `!((cr >> 28) & 2)`: unary ! binds more tightly than binary &.
        if (not operand or (_strip_parens(operand) is operand
                            and operand[0].text != "!"
                            and [t.text for t in operand] not in (["0"], ["1"]))):
            return None
        child = _boolean_condition(operand, source)
        return (f"!({child[0]})", child[1]) if child else None
    # The lifter spells a CR predicate as a one-bit integer equality, for
    # example `((ctx->cr >> 29) & 1) == 1`. Name only that exact boolean form;
    # never turn a wider numeric CR expression into a predicate.
    for operator in ("==", "!="):
        pieces = _split_operator(tokens, operator)
        if len(pieces) != 2:
            continue
        predicate = _cr_predicate(_strip_parens(pieces[0]))
        right = _strip_parens(pieces[1])
        if (predicate is None or len(right) != 1 or right[0].kind != "number"
                or right[0].text not in {"0", "1"}):
            continue
        positive = (operator == "==") == (right[0].text == "1")
        return (predicate if positive else f"!({predicate})"), True
    predicate = _cr_predicate(tokens)
    if predicate:
        return predicate, True
    words = [t.text for t in tokens]
    if (len(words) == 6 and words[:4] == ["--", "ctx", "->", "ctr"]
            and words[4] in {"==", "!="} and words[5] == "0"):
        return f"(--ctx->ctr {words[4]} 0)", False
    if words in (["0"], ["1"]):
        return words[0], False
    return None


def _named_conditions(line):
    tokens = _code_tokens(line)
    if not tokens:
        return line
    replacements = []
    i = 0
    while i + 1 < len(tokens):
        token = tokens[i]
        if token.kind != "identifier" or token.text != "if" or tokens[i + 1].text != "(":
            i += 1
            continue
        start = i + 1
        end = start + 1
        depth = 1
        while end < len(tokens) and depth:
            if tokens[end].text == "(":
                depth += 1
            elif tokens[end].text == ")":
                depth -= 1
            if depth:
                end += 1
        if depth:
            return line
        condition = _boolean_condition(tokens[start + 1:end], line)
        if condition and condition[1]:
            replacements.append((tokens[start].end, tokens[end].start, condition[0]))
        i = end + 1
    for start, end, replacement in reversed(replacements):
        line = line[:start] + replacement + line[end:]
    return line


def semantic_body(lines: list[str]) -> list[str]:
    """Give exact integer compares and CR branch tests descriptive names."""
    result = []
    for line in lines:
        if not _CR_CONTEXT.search(line):
            result.append(line)
            continue
        compare = _named_compare(line)
        result.append(compare if compare is not None else _named_conditions(line))
    return result
