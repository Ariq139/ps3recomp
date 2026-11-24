"""Whitespace-only layout for the PPU lifter's generated C++ statements.

This is deliberately not a C++ parser or a source-code optimizer.  It lays out
the bounded statement language emitted by ``ppu_lifter.py``; preprocessing,
line splicing and unfamiliar/malformed input keep their original layout.  The
tokenizer is also used by the final register-naming pass so comments and
literals never become candidate register accesses.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CppToken:
    kind: str
    text: str
    start: int
    end: int


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
# A preprocessing-number token.  Exponent signs and digit separators belong
# to the number, so 0x1p-2 / 1'000 cannot masquerade as operators or literals.
_NUMBER = re.compile(
    r"(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z_0-9.]|'[A-Za-z_0-9])*"
)
_RAW_PREFIX = re.compile(r'(?:u8|u|U|L)?R"')
_QUOTED_PREFIX = re.compile(r"(?:u8|u|U|L)?[\"']")
_PUNCTUATION = tuple(sorted((
    "->*", "<<=", ">>=", "...", "<=>", "::", "++", "--", "->",
    "&&", "||", "<=", ">=", "==", "!=", "*=", "/=", "%=", "+=",
    "-=", "&=", "^=", "|=", "<<", ">>", ".*", "##",
), key=len, reverse=True))
_SINGLE_PUNCTUATION = set("{}[]();:?,.~!+-*/%&|^=<>#")


def _line_end(text: str, start: int) -> int:
    """End of a logical line, excluding its final newline.

    C++ removes backslash-newline pairs before comments/directives are parsed.
    Keeping those pairs inside this opaque token prevents renaming a continued
    comment or macro.  The formatter subsequently declines such input.
    """
    end = start
    while True:
        newline = text.find("\n", end)
        if newline < 0:
            return len(text)
        previous = newline - 1
        if previous >= start and text[previous] == "\r":
            previous -= 1
        if previous < start or text[previous] != "\\":
            return newline
        end = newline + 1


def _literal_suffix_end(text: str, end: int) -> int:
    # User-defined literal suffixes are part of one literal token.  Joining or
    # separating them while laying out code would change its interpretation.
    suffix = _IDENTIFIER.match(text, end)
    return suffix.end() if suffix else end


def tokenize_cpp(text: str) -> list[CppToken]:
    """Return lossless tokens, including whitespace, with original spans.

    Kinds are ``identifier``, ``space``, ``comment``, ``string``, ``char``,
    ``number``, ``punct``, ``directive`` and ``unknown``.  This covers generated
    C++ plus ordinary/prefixed/raw literals; unsupported or unterminated input
    is an ``unknown`` token rather than guessed code.  Token concatenation is
    always exactly the input.  No preprocessing or escape decoding is done.
    """
    result: list[CppToken] = []
    i = 0
    n = len(text)
    while i < n:
        start = i
        kind = "unknown"
        if text[i].isspace():
            i += 1
            while i < n and text[i].isspace():
                i += 1
            kind = "space"
        elif text.startswith("//", i):
            i = _line_end(text, i)
            kind = "comment"
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            kind = "unknown" if end < 0 else "comment"
        elif text[i] == "#":
            # The known generated language has no non-directive '#' use.
            # Treat any such line opaquely, including continued directives.
            i = _line_end(text, i)
            kind = "directive"
        elif (match := _RAW_PREFIX.match(text, i)) is not None:
            delim_end = text.find("(", match.end(), match.end() + 17)
            if delim_end < 0:
                i = n
            else:
                delimiter = text[match.end():delim_end]
                valid = all(0x21 <= ord(c) <= 0x7e and c not in "()\\"
                            for c in delimiter)
                end_marker = ")" + delimiter + '"'
                end = text.find(end_marker, delim_end + 1) if valid else -1
                if end < 0:
                    i = n
                else:
                    i = _literal_suffix_end(text, end + len(end_marker))
                    kind = "string"
        elif (match := _QUOTED_PREFIX.match(text, i)) is not None:
            quote = text[match.end() - 1]
            i = match.end()
            while i < n:
                if text[i] == "\\":
                    # Preserve escapes, including physical line continuations.
                    i = min(n, i + 2)
                elif text[i] == quote:
                    i = _literal_suffix_end(text, i + 1)
                    kind = "string" if quote == '"' else "char"
                    break
                else:
                    i += 1
        elif (match := _NUMBER.match(text, i)) is not None:
            i = match.end()
            kind = "number"
        elif (match := _IDENTIFIER.match(text, i)) is not None:
            i = match.end()
            kind = "identifier"
        else:
            operator = next((p for p in _PUNCTUATION if text.startswith(p, i)), None)
            if operator:
                i += len(operator)
                kind = "punct"
            else:
                i += 1
                if text[start] in _SINGLE_PUNCTUATION:
                    kind = "punct"
        result.append(CppToken(kind, text[start:i], start, i))
    return result


def _significant(tokens: list[CppToken]) -> list[CppToken]:
    return [token for token in tokens if token.kind != "space"]


def format_body(lines: list[str]) -> list[str]:
    """Lay out generated statements as a function body, indented four spaces.

    Only whitespace between tokens changes.  Input with directives, line
    splices, multiline literals/comments, invalid delimiters, or unknown tokens
    is returned verbatim.  Ordinary labels are outdented one level.  References
    and other declarations must be inserted by the caller before these lines.
    This does not add braces, infer structured control flow, or rewrite tokens.
    """
    source = "\n".join(lines)
    tokens = _significant(tokenize_cpp(source))
    if not tokens:
        return list(lines)
    if any(token.kind in ("directive", "unknown")
           or "\n" in token.text or "\r" in token.text
           or token.kind == "identifier" and token.text in ("__LINE__", "__builtin_LINE")
           for token in tokens):
        return list(lines)

    # Exact delimiter matching also gives a cheap, conservative reject path
    # for incomplete fragments.  Angle brackets are operators in this language.
    groups: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for token in tokens:
        if token.kind != "punct":
            continue
        if token.text in ("(", "[", "{"):
            groups.append(token.text)
        elif token.text in pairs:
            if not groups or groups.pop() != pairs[token.text]:
                return list(lines)
    if groups:
        return list(lines)

    output: list[str] = []
    current: list[str] = []
    current_tokens: list[str] = []
    indent = 1
    current_indent = indent
    parens = 0
    brackets = 0
    break_after_comment = False

    def flush() -> None:
        nonlocal current, current_tokens, current_indent
        if current:
            output.append("    " * current_indent + "".join(current).strip())
        current = []
        current_tokens = []
        current_indent = indent

    def append(token: CppToken, previous: CppToken | None) -> None:
        nonlocal current_indent
        if not current:
            current_indent = indent
        elif previous is not None and (
                source[previous.end:token.start]
                or previous.text in (",", ";", "else")
                or token.text == "(" and previous.text in ("if", "for", "while", "switch", "catch")
                or token.text == "{" and previous.text not in ("(", "[", "{")
                or token.text in ("else", "while") and previous.text == "}"):
            # Collapse source whitespace, never remove a token-separating gap.
            current.append(" ")
        current.append(token.text)
        if token.kind != "comment":
            current_tokens.append(token.text)

    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index else None
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        t = token.text
        if token.kind == "comment":
            # A standalone source comment stays standalone.  Most lifter
            # annotations trail the instruction they describe.
            if current and previous and "\n" in source[previous.end:token.start]:
                flush()
            append(token, previous)
            if t.startswith("//") or break_after_comment:
                flush()
                break_after_comment = False
            continue

        if token.kind == "punct" and t == "}":
            flush()
            indent -= 1
            append(token, None)
            if following is None or following.text not in (
                    ";", ",", ")", "]", ".", "->", "else", "while"):
                flush()
            continue

        # Each switch arm begins on its own line.  The ':' handler below
        # keeps a ternary expression intact (including a case expression).
        if token.kind == "identifier" and t in ("case", "default") and not parens and not brackets:
            flush()

        append(token, previous)
        if token.kind != "punct":
            continue
        if t == "(":
            parens += 1
        elif t == ")":
            parens -= 1
        elif t == "[":
            brackets += 1
        elif t == "]":
            brackets -= 1
        elif t == "{":
            flush()
            indent += 1
        elif t == ";" and not parens and not brackets:
            if following and following.kind == "comment" and "\n" not in source[token.end:following.start]:
                break_after_comment = True
            elif following and following.text == "else":
                # Keep an unbraced if/else together.  Splitting only the
                # semicolon could visually attach a dangling else to the wrong
                # enclosing if, despite preserving the C++ token sequence.
                pass
            else:
                flush()
        elif t == ":" and not parens and not brackets:
            simple_label = len(current_tokens) == 2 and _IDENTIFIER.fullmatch(current_tokens[0])
            case_label = current_tokens[0] in ("case", "default")
            # Balance '?' / ':' pairs in the label expression before treating
            # this colon as the case separator.  The final case ':' exceeds
            # the ternary '?' count by one.
            unmatched_colon = current_tokens.count(":") > current_tokens.count("?")
            if simple_label or case_label and unmatched_colon:
                current_indent = max(0, indent - 1)
                flush()
    flush()

    # A defensive invariant, not a substitute parser: comments and literal
    # contents are tokens too.  It catches accidental token joining or a line
    # comment swallowing a subsequent statement before source reaches a build.
    before = [(token.kind, token.text) for token in tokens]
    after = [(token.kind, token.text) for token in _significant(tokenize_cpp("\n".join(output)))]
    return output if before == after else list(lines)
