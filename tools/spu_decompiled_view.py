#!/usr/bin/env python3
"""Conservative, C17-compatible presentation layer for ps3recomp SPU lifts.

It does *not* decompile guest control flow: this formatter preserves every
original C token after C preprocessing. Register aliases expand to the
original ctx->gpr[N] expression, not snapshots, and all added labels are
comments. This is intentionally independent of the unavailable live lifter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FUNC = re.compile(
    r"\bvoid\s+(?P<name>(?:[A-Za-z_]\w*_)?spu_func_(?P<addr>[0-9A-Fa-f]{8}))"
    r"\s*\(\s*spu_context\s*\*\s*ctx\s*\)\s*\{"
)
GPR = re.compile(r"\bctx\s*->\s*gpr\s*\[\s*(?P<reg>\d{1,3})\s*\]")
STACK = re.compile(
    r"\bspu_ls_(?P<op>read|write)(?P<width>8|16|32|64|128)\s*\(\s*ctx\s*,"
    r"\s*ctx\s*->\s*gpr\s*\[\s*1\s*\]\s*\.\s*_u32\s*\[\s*0\s*\]"
    r"\s*\+\s*(?P<offset>-?(?:0[xX][0-9a-fA-F]+|\d+))\s*,?"
)
TRANSFER = re.compile(
    r"\bctx\s*->\s*pc\s*=\s*(?P<pc>0[xX][0-9A-Fa-f]+)\s*;\s*"
    r"g_spu_trampoline_fn\s*=\s*(?P<target>[A-Za-z_]\w*spu_func_[0-9A-Fa-f]{8})\s*;"
)
SYMBOL_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,1024}$")
VIEW_MARKER = "/* ps3recomp SPU decompiled view v1: aliases expand to original registers. */"


def code_mask(src: str) -> str:
    """Return a same-length view containing spaces in comments and literals.

    This intentionally treats comments, strings, chars, and preprocessor lines
    as protected. The generated C has no trigraph processing; reject any input
    that could subvert this lexical view instead of silently rewriting it.
    """
    if "\\\n" in src or "\\\r\n" in src:
        raise ValueError("backslash-newline splice: unsafe for this view")
    if "??/" in src:
        raise ValueError("trigraph splice: unsafe for this view")
    chars = list(src)
    out = list(src)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch == '/' and i + 1 < len(chars) and chars[i + 1] in ('/', '*'):
            block = chars[i + 1] == '*'
            start = i
            if block:
                end = src.find('*/', i + 2)
                if end < 0:
                    raise ValueError("unterminated C block comment")
                i = end + 2
            else:
                end = src.find('\n', i + 2)
                i = len(src) if end < 0 else end
            for k in range(start, i):
                if out[k] not in ('\n', '\r'):
                    out[k] = ' '
            continue
        if ch in ('"', "'"):
            quote = ch
            start = i
            i += 1
            while i < len(src):
                if chars[i] == '\\':
                    i += 2
                    continue
                if chars[i] == quote:
                    i += 1
                    break
                i += 1
            else:
                raise ValueError("unterminated C string/character literal")
            for k in range(start, min(i, len(src))):
                if out[k] not in ('\n', '\r'):
                    out[k] = ' '
            continue
        i += 1
    # Mask preprocessor lines as indivisible content, including lines inside a function.
    pos = 0
    for line in ''.join(out).splitlines(keepends=True):
        if line.lstrip().startswith('#'):
            for k in range(pos, pos + len(line)):
                if out[k] not in ('\n', '\r'):
                    out[k] = ' '
        pos += len(line)
    return ''.join(out)


def _brace_end(mask: str, opening: int) -> int:
    depth = 0
    for i in range(opening, len(mask)):
        if mask[i] == '{':
            depth += 1
        elif mask[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                raise ValueError("unbalanced generated function")
    raise ValueError("unterminated generated function")


def _safe_symbol(name: object) -> str:
    if not isinstance(name, str):
        return ""
    # Output is a C comment, never a symbol identifier. Collapse all control
    # characters rather than exposing a line break or comment escape.
    clean = ''.join((c if (ord(c) >= 32 and ord(c) != 127) else ' ') for c in name)
    return clean.replace('*/', '* /').replace('/*', '/ *')[:160].strip()


def load_image_names(path: Path | None, image_sha256: str | None) -> dict[int, str]:
    if not path:
        return {}
    if not image_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", image_sha256):
        raise ValueError("SPU symbol annotations require --image-sha256 (exact image SHA-256)")
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError("name map must be an object")
    meta = data.get('_meta', {})
    if not isinstance(meta, dict) or str(meta.get('image_sha256', '')).lower() != image_sha256.lower():
        raise ValueError("symbol map image_sha256 differs from --image-sha256; refusing cross-image names")
    names: dict[int, str] = {}
    for key, val in data.items():
        if not isinstance(key, str) or not re.fullmatch(r"0[xX][0-9a-fA-F]{1,8}", key):
            continue
        try:
            label = _safe_symbol(val.get('label') if isinstance(val, dict) else val)
        except Exception:
            label = ''
        if label:
            names[int(key, 16)] = label
    return names


def _safe_alias(reg: int, used_ids: set[str]) -> str:
    candidate = f"r{reg}"
    if candidate not in used_ids:
        return candidate
    candidate = f"spu_reg_{reg}"
    k = 0
    while candidate in used_ids:
        k += 1
        candidate = f"spu_reg_{reg}_{k}"
    return candidate


def _annotation_lines(body: str, mask: str, aliases: dict[int, str]) -> tuple[str, int]:
    """Insert only comments before known single-line LS statements.

    No instruction is rewritten here; base and signed literal offsets are exact.
    """
    matches = []
    for match in STACK.finditer(mask):
        # Avoid repeated annotations at the same source line.
        line_beg = body.rfind('\n', 0, match.start()) + 1
        if body[line_beg:match.start()].strip():
            continue
        offset = int(match.group('offset'), 0)
        reg_name = aliases.get(1, 'r1')
        sign = '+' if offset >= 0 else '-'
        off_text = f"0x{abs(offset):X}"
        indent = re.match(r"[ \t]*", body[line_beg:match.start()]).group(0)
        note = (f"{indent}/* LS {match.group('op')} {match.group('width')}-bit; "
                f"address={reg_name}.word0 {sign} {off_text}; "
                f"no inferred object/type */\n")
        matches.append((line_beg, note))
    for pos, note in reversed(matches):
        body = body[:pos] + note + body[pos:]
    return body, len(matches)


def _format_transfer(body: str) -> tuple[str, int]:
    """Whitespace-only presentation of emitted handoff blocks.

    Only fixed emitted constructs are formatted; general if/goto/loops are
    deliberately not reconstructed without a validated CFG.
    """
    # Only matches one-line terminal transfers; no changed C tokens.
    pat = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<prefix>if\s*\([^\n{}]*\)\s*)?"
        r"\{\s*(?P<pc>ctx\s*->\s*pc\s*=\s*0[xX][0-9a-fA-F]+\s*;)\s*"
        r"(?P<dispatch>g_spu_trampoline_fn\s*=\s*[A-Za-z_]\w*\s*;)\s*"
        r"(?P<ret>return\s*;)\s*\}"
    )
    count = 0
    def rewrite(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        ind = m.group('indent')
        pre = m.group('prefix') or ''
        if pre:
            return f"{ind}{pre}{{\n{ind}    {m.group('pc')}\n{ind}    {m.group('dispatch')}\n{ind}    {m.group('ret')}\n{ind}}}"
        return f"{ind}{{\n{ind}    {m.group('pc')}\n{ind}    {m.group('dispatch')}\n{ind}    {m.group('ret')}\n{ind}}}"
    return pat.sub(rewrite, body), count



def _split_dense_statements(body: str) -> tuple[str, int]:
    """Split emitted same-line statements at semicolons outside parentheses.

    Character streams and comments are never inspected as C syntax; this
    changes whitespace only. The preprocessor-equivalence test is the gate.
    """
    mask = code_mask(body)
    paren = 0
    changes = []
    for i, ch in enumerate(mask):
        if ch == '(':
            paren += 1
        elif ch == ')':
            paren -= 1
        elif ch == ';' and paren == 0:
            nxt = i + 1
            while nxt < len(mask) and mask[nxt] in ' \t':
                nxt += 1
            if nxt < len(mask) and mask[nxt] not in ('\n', '\r'):
                line_start = body.rfind('\n', 0, i) + 1
                indent = re.match(r'[ \t]*', body[line_start:i]).group(0)
                changes.append((i + 1, nxt, '\n' + indent))
        if paren < 0:
            raise ValueError('unbalanced generated C parentheses')
    if paren != 0:
        raise ValueError('unbalanced generated C parentheses')
    for start, finish, text in reversed(changes):
        body = body[:start] + text + body[finish:]
    return body, len(changes)



def _annotate_named_targets(body: str, names: dict[int, str]) -> tuple[str, int]:
    """Only annotate a proven direct SPU trampoline or direct function call.

    A Ghidra name never becomes a callable C identifier or a replacement for
    the registered native symbol. LS/image provenance is validated upstream.
    """
    if not names:
        return body, 0
    mask = code_mask(body)
    assignment = re.compile(
        r"\bg_spu_trampoline_fn\s*=\s*(?:[A-Za-z_]\w*_)?spu_func_"
        r"(?P<pc>[0-9a-fA-F]{8})\s*;"
    )
    direct = re.compile(
        r"\b(?:[A-Za-z_]\w*_)?spu_func_(?P<pc>[0-9a-fA-F]{8})"
        r"\s*\(\s*ctx\s*\)\s*;"
    )
    changes = []
    for rx in (assignment, direct):
        for match in rx.finditer(mask):
            label = _safe_symbol(names.get(int(match.group('pc'), 16)))
            if label:
                changes.append((match.end(), f" /* Ghidra: {label} */"))
    for pos, note in sorted(changes, reverse=True):
        body = body[:pos] + note + body[pos:]
    return body, len(changes)


def generate_view(src: str, names: dict[int, str] | None = None) -> tuple[str, dict]:
    names = names or {}
    if VIEW_MARKER in src:
        raise ValueError("input is already a decompiled view; use the original lifted C")
    mask = code_mask(src)
    targets = []
    previous_end = 0
    for match in FUNC.finditer(mask):
        start = match.start()
        if start < previous_end:
            continue
        # Source header may contain named function pointers, but we require
        # the exact generated definition at file scope.
        if mask[start:match.end()].count('{') != 1:
            continue
        end = _brace_end(mask, match.end() - 1)
        targets.append((start, match, end))
        previous_end = end
    if not targets:
        raise ValueError("no recognizable ps3recomp SPU function definitions; refusing unrelated C")
    produced = []
    cursor = 0
    annotated_memory = 0
    formatted_transfer = 0
    formatted_statements = 0
    annotated_symbols = 0
    for start, match, end in targets:
        produced.append(src[cursor:start])
        entry = int(match.group('addr'), 16)
        head = src[start:match.end()]
        body = src[match.end():end - 1]
        body_mask = mask[match.end():end - 1]
        if re.search(r"(?m)^[ \t]*#", body):
            raise ValueError(f"preprocessor directives inside {match.group('name')}: cannot prove alias safety")
        # All named register replacements are confined to this function body.
        regs = sorted({int(m.group('reg')) for m in GPR.finditer(body_mask)})
        if any(r < 0 or r >= 128 for r in regs):
            raise ValueError(f"invalid SPU GPR index in {match.group('name')}")
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", body_mask))
        aliases = {reg: _safe_alias(reg, identifiers) for reg in regs}
        body, num_ann = _annotation_lines(body, body_mask, aliases)
        annotated_memory += num_ann
        # Remask after adding comments, then replace only live code spans.
        new_mask = code_mask(body)
        spans = []
        for m in GPR.finditer(new_mask):
            spans.append((m.start(), m.end(), aliases[int(m.group('reg'))]))
        for begin, finish, ident in reversed(spans):
            body = body[:begin] + ident + body[finish:]
        body, formatted = _format_transfer(body)
        formatted_transfer += formatted
        body, split_count = _split_dense_statements(body)
        formatted_statements += split_count
        body, symbol_count = _annotate_named_targets(body, names)
        annotated_symbols += symbol_count
        label = _safe_symbol(names.get(entry))
        details = f"Guest SPU LS entry 0x{entry:08X}; register aliases are live u128 lvalues."
        if label:
            details += f" Ghidra symbol (same image): {label}."
        produced.append(f"/* {details} */\n")
        for reg in regs:
            produced.append(f"#ifdef {aliases[reg]}\n#error {aliases[reg]} already defined before SPU register alias\n#endif\n")
            produced.append(f"#define {aliases[reg]} ctx->gpr[{reg}]\n")
        if regs:
            produced.append("/* Each alias expands to the ORIGINAL register expression (not a copy). */\n")
        produced.append(head)
        produced.append(body)
        produced.append('}')
        if regs:
            produced.append('\n')
            for reg in reversed(regs):
                produced.append(f"#undef {aliases[reg]}\n")
        cursor = end
    produced.append(src[cursor:])
    out = ''.join(produced)
    out = out.replace("/* Auto-generated by spu_lifter.py -- do not edit by hand. */",
                      "/* Auto-generated by spu_lifter.py -- do not edit by hand. */\n"+VIEW_MARKER, 1)
    return out, {"functions":len(targets), "annotated_ls_accesses": annotated_memory,
                 "formatted_trampolines": formatted_transfer,
                 "split_dense_statements": formatted_statements,
                 "annotated_symbol_references": annotated_symbols,
                 "input_sha256": hashlib.sha256(src.encode('utf-8')).hexdigest(),
                 "output_sha256": hashlib.sha256(out.encode('utf-8')).hexdigest(),
                 "method": "C17-preprocessor-exact-register-aliases-and-comments"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Existing generated spu_recomp.c')
    parser.add_argument('--output', type=Path, required=True, help='Write readable compiled C (different from input)')
    parser.add_argument('--names', type=Path, help='Existing address->label JSON, image-qualified')
    parser.add_argument('--image-sha256', help='Original SPU image SHA-256 for validating symbol names')
    parser.add_argument('--report', type=Path, help='Optional JSON report')
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error('--output must differ from input, to protect canonical generated source')
    try:
        data = args.input.read_text(encoding='utf-8')
        if not data.startswith('/* Auto-generated by spu_lifter.py'):
            raise ValueError('input does not have the expected SPU-lifter header')
        names = load_image_names(args.names, args.image_sha256)
        output, report = generate_view(data, names)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as err:
        print(f'spu_decompiled_view: {err}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
