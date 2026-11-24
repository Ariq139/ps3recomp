"""Readable presentation of the PPU lifter's already-lowered function bodies.

This is deliberately the last emission pass, not an instruction optimizer or a
C++ decompiler. The analyses in ppu_lifter consume the canonical ctx->... form.
Only the final source uses live references to the very same context members.
"""

from dataclasses import dataclass
import re

from ppu_readable_format import format_body, tokenize_cpp


READABLE_SOURCE_MARKER = "/* PPU_SOURCE_STYLE: readable */"
STRUCTURED_SOURCE_MARKER = "/* PPU_SOURCE_STYLE: structured */"


_SPECIAL_NAMES = {
    "cr": "cr",
    "lr": "lr",
    "ctr": "ctr",
    "xer": "xer",
    "fpscr": "fpscr",
    "vscr": "vscr",
    "cia": "pc",
    "vrsave": "vrsave",
}
_ACCESS = re.compile(
    r"ctx\s*->\s*(?:(?P<bank>gpr|fpr|vr)\s*\[\s*"
    r"(?P<index>[0-9]|[12][0-9]|3[01])\s*\]|"
    r"(?P<special>" + "|".join(_SPECIAL_NAMES) + r")\b)"
)
_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
                "<<=", ">>=", "++", "--"}
_UNEVALUATED_TYPES = {"decltype", "typeof", "__typeof", "__typeof__"}


@dataclass(frozen=True)
class _Register:
    bank: str
    index: int = 0

    @property
    def expression(self):
        if self.bank in _SPECIAL_NAMES:
            return f"ctx->{self.bank}"
        return f"ctx->{self.bank}[{self.index}]"

    @property
    def preferred_name(self):
        if self.bank in _SPECIAL_NAMES:
            return _SPECIAL_NAMES[self.bank]
        if self.bank == "gpr":
            return {1: "sp", 2: "toc"}.get(self.index, f"r{self.index}")
        return f"{'f' if self.bank == 'fpr' else 'v'}{self.index}"

    @property
    def order(self):
        banks = ["gpr", "fpr", "vr", *_SPECIAL_NAMES]
        return banks.index(self.bank), self.index

    @property
    def note(self):
        if self.bank == "gpr" and self.index == 1:
            return "  // r1: guest stack pointer"
        if self.bank == "gpr" and self.index == 2:
            return "  // r2: guest TOC pointer"
        if self.bank == "cia":
            return "  // current guest instruction address"
        return ""


def _can_alias(tokens):
    """Reject constructs outside the generated-body contract conservatively.

    Current generated functions never rebind ctx or use decltype. Downstream
    tools sometimes supply their own LiftedFunction.body_lines, though: a live
    reference would outlive a pointer rebind, and decltype would see its type.
    Such functions keep explicit context accesses instead of being guessed at.
    """
    if any(t.kind == "directive" or t.text == "#" for t in tokens):
        return False
    code = [t for t in tokens if t.kind not in {"space", "comment"}]
    for i, token in enumerate(code):
        if token.kind != "identifier":
            continue
        if token.text in _UNEVALUATED_TYPES:
            return False
        if token.text != "ctx":
            continue
        before = code[i - 1].text if i else ""
        after = code[i + 1].text if i + 1 < len(code) else ""
        if after in _ASSIGNMENTS or before in {"++", "--"} and after != "->":
            return False
        if before == "&" and after != "->":
            return False  # a callee could change ctx through &ctx
    return True


def name_registers(body_lines: list[str]) -> tuple[list[str], list[str]]:
    """Return (reference declarations, renamed body) without mutating input.

    Token boundaries protect comments and all literal forms. Only fixed,
    in-range context register accesses are candidates; dynamic indices and
    runtime metadata retain their explicit spelling. Reference names avoid
    every other identifier in this function, including instruction-patch
    guards and manually inserted locals.
    """
    text = "\n".join(body_lines)
    tokens = tokenize_cpp(text)
    if not _can_alias(tokens):
        return [], list(body_lines)

    accesses = []
    occupied = set()
    covered_until = -1
    previous = ""
    for token in tokens:
        if token.kind in {"space", "comment"}:
            continue
        if token.start < covered_until:
            continue
        match = None
        if (token.kind == "identifier" and token.text == "ctx"
                and previous not in {".", "->", "::"}):
            match = _ACCESS.match(text, token.start)
        if match:
            register = (_Register(match["bank"], int(match["index"]))
                        if match["bank"] else _Register(match["special"]))
            accesses.append((match.start(), match.end(), register))
            covered_until = match.end()
            previous = "]" if match["bank"] else match["special"]
        else:
            if token.kind == "identifier":
                occupied.add(token.text)
            previous = token.text

    if not accesses:
        return [], list(body_lines)

    names = {}
    declarations = ["    // Live register aliases; calls and callbacks share this state."]
    for register in sorted({a[2] for a in accesses}, key=lambda r: r.order):
        preferred = register.preferred_name
        name = preferred
        suffix = 1
        while name in occupied:
            name = f"ppu_{preferred}" if suffix == 1 else f"ppu_{preferred}_{suffix}"
            suffix += 1
        occupied.add(name)
        names[register] = name
        declarations.append(
            f"    auto& {name} = {register.expression};{register.note}")

    pieces = []
    cursor = 0
    for start, end, register in accesses:
        pieces.extend((text[cursor:start], names[register]))
        cursor = end
    pieces.append(text[cursor:])
    return declarations, "".join(pieces).split("\n")


def readable_body(body_lines: list[str]) -> list[str]:
    """Name live registers and lay out the final function body for reading."""
    declarations, renamed = name_registers(body_lines)
    body = format_body(renamed)
    return declarations + ([""] if declarations and body else []) + body


def structured_body(body_lines: list[str], helper_namespace: str = "ops") -> list[str]:
    """Recover verified source constructs, then name live registers.

    Every transformation is confined to the already-lowered generated body.
    Unknown control regions and expressions retain their original lowering.
    In particular, this does not buffer architectural register writes.
    """
    from ppu_readable_control import structure_body
    from ppu_readable_semantics import semantic_body
    from ppu_readable_simplify import simplify_body

    result = readable_body(structure_body(semantic_body(simplify_body(body_lines))))
    if helper_namespace == "ppu_readable":
        return result
    # The lifter chooses a TU-wide namespace which cannot conflict with an
    # instruction-patch guard. Rename only the qualified code identifier;
    # guard variables, comments, and string/character literals are untouched.
    text = "\n".join(result)
    tokens = [t for t in tokenize_cpp(text) if t.kind not in {"space", "comment"}]
    spans = [t for i, t in enumerate(tokens)
             if t.kind == "identifier" and t.text == "ppu_readable"
             and i > 0 and i + 1 < len(tokens)
             and tokens[i - 1].text == "::" and tokens[i + 1].text == "::"]
    for token in reversed(spans):
        text = text[:token.start] + helper_namespace + text[token.end:]
    return text.split("\n")
