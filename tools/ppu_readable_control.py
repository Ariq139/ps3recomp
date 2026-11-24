"""Conservative control-flow recovery for the PPU lifter's final bodies.

This recognizes explicit, flat, single-entry regions in the lifter's generated
statement language. It does not reconstruct arbitrary C++, split opaque
instructions, remove architectural writes, or change a branch condition.
Unknown shapes retain their labels and gotos. Run after existing lowering/CR
fusion and before register naming and whitespace formatting.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from ppu_readable_format import tokenize_cpp


_MAX_NODES = 4096
_MAX_REGION = 512
_MAX_PASSES = 8
_PAIRS = {")": "(", "]": "[", "}": "{"}
_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
                "<<=", ">>=", "++", "--"}
_KNOWN_CALL = re.compile(
    r"(?:[A-Za-z_]\w*)?func_[0-9A-Fa-f]{8}$|"
    r"(?:ps3_|ppu_|vm_)[A-Za-z_0-9]+$|"
    r"(?:DRAIN_TRAMPOLINE|PPU_FENCE|VM_RAW_ST|lv2_syscall|memcpy|memmove|memset)$"
)


@dataclass
class _Node:
    text: str
    kind: str
    refs: Counter
    label: str = ""
    condition: str = ""
    comments: tuple[str, ...] = ()
    movable: bool = False


def _code(text):
    return [token for token in tokenize_cpp(text)
            if token.kind not in {"space", "comment"}]


def _closing(tokens, start):
    """Index of the matching ')' for a known opening parenthesis."""
    depth = 0
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token.text == "(":
            depth += 1
        elif token.text == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _statement_segments(tokens):
    """Split complete top-level statements solely to reject escaped locals.

    Compound statements keep their existing scopes. This deliberately is not
    a general declaration parser: only known emitted expression/control shapes
    are movable; an unfamiliar statement makes the region ineligible.
    """
    start = 0
    stack = []
    for index, token in enumerate(tokens):
        t = token.text
        if t in {"(", "[", "{"}:
            stack.append(t)
        elif t in _PAIRS:
            if not stack or stack.pop() != _PAIRS[t]:
                return None
        if stack:
            continue
        following = tokens[index + 1].text if index + 1 < len(tokens) else ""
        boundary = t == ";" or (t == "}" and following not in {"else", "while", ";"})
        if boundary:
            yield tokens[start:index + 1]
            start = index + 1
    if start < len(tokens):
        yield tokens[start:]


def _scope_safe(tokens):
    # A new surrounding loop must never capture a break/continue from an
    # opaque statement. Existing nested loops could be analyzed, but refusal
    # is smaller and safer than guessing which construct owns one.
    if any(token.text in {"break", "continue", "case", "default"} for token in tokens):
        return False
    for segment in _statement_segments(tokens):
        if not segment:
            continue
        first = segment[0].text
        if first in {"{", "if", "while", "for", "do", "return", "goto", ";"}:
            continue
        if first == "ctx" and len(segment) > 1 and segment[1].text == "->":
            continue
        if (len(segment) > 4 and [t.text for t in segment[:3]] == ["::", "ppu_readable", "::"]
                and segment[3].kind == "identifier" and segment[4].text == "("):
            continue  # functions in the structured emitter's known helper namespace
        if first in {"++", "--"}:
            continue
        if segment[0].kind == "identifier" and len(segment) > 1:
            second = segment[1].text
            if second in _ASSIGNMENTS:
                continue
            if second == "(" and _KNOWN_CALL.fullmatch(first):
                continue
        # Includes all top-level type declarations and initialized snapshots.
        return False
    return True


def _node(text):
    tokens = tokenize_cpp(text)
    code = [token for token in tokens if token.kind not in {"space", "comment"}]
    comments = tuple(token.text for token in tokens if token.kind == "comment")
    refs = Counter()
    for index, token in enumerate(code):
        if token.text == "goto":
            if index + 1 >= len(code) or code[index + 1].kind != "identifier":
                return None
            refs[code[index + 1].text] += 1
    if not code:
        return _Node(text, "comment", refs, comments=comments, movable=True)
    words = [token.text for token in code]
    if (len(code) in (2, 3) and code[0].kind == "identifier"
            and words[1] == ":" and (len(code) == 2 or words[2] == ";")):
        return _Node(text, "label", refs, label=words[0], comments=comments)
    if len(code) == 3 and words[0] == "goto" and words[2] == ";":
        return _Node(text, "goto", refs, label=words[1], comments=comments, movable=True)
    if len(code) >= 7 and words[:2] == ["if", "("]:
        close = _closing(code, 1)
        if close is not None and words[close + 1:] == ["goto", words[-2], ";"]:
            if code[-2].kind != "identifier":
                return None
            begin, end = code[1].end, code[close].start
            # Moving a physical line comment next to the closing ')' can
            # swallow it; do not guess at that uncommon generated shape.
            inner = [token for token in tokens if begin <= token.start < end]
            # Canonical branch conditions contain no declarations or init
            # statements. Preserve handwritten if-initializers/assignments as
            # opaque statements instead of trying to negate a declaration.
            opaque_condition = any(token.kind == "punct" and token.text in {
                "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ";"
            } for token in inner)
            if (not opaque_condition and not any(
                    token.kind == "comment" and token.text.startswith("//") for token in inner)):
                outside = tuple(token.text for token in tokens
                                if token.kind == "comment" and not begin <= token.start < end)
                return _Node(text, "branch", refs, label=code[-2].text,
                             condition=text[begin:end], comments=outside, movable=True)
    return _Node(text, "statement", refs, comments=comments, movable=_scope_safe(code))


def _parse(lines):
    """Group balanced canonical units, including generated multiline blocks."""
    nodes = []
    pending = []
    stack = []
    last_word = ""
    for line in lines:
        tokens = tokenize_cpp(line)
        if any(token.kind in {"unknown", "directive"}
               or token.kind in {"string", "char", "comment"} and "\n" in token.text
               or token.kind == "identifier" and token.text in {
                   "__LINE__", "__builtin_LINE", "asm", "__asm", "__asm__"}
               for token in tokens):
            return None
        code = [token for token in tokens if token.kind not in {"space", "comment"}]
        for token in code:
            t = token.text
            if t in {"(", "[", "{"}:
                stack.append(t)
            elif t in _PAIRS:
                if not stack or stack.pop() != _PAIRS[t]:
                    return None
        pending.append(line)
        if code:
            last_word = code[-1].text
        if not stack and (not last_word or last_word in {";", "}", ":"}):
            node = _node("\n".join(pending))
            if node is None:
                return None
            nodes.append(node)
            pending = []
            last_word = ""
            if len(nodes) > _MAX_NODES:
                return None
    if pending or stack:
        return None
    # The only supported label definitions are standalone canonical labels.
    # Count references even when they occur inside opaque switch/patch nodes.
    labels = [node.label for node in nodes if node.kind == "label"]
    label_set = set(labels)
    if len(label_set) != len(labels):
        return None
    referenced = {label for node in nodes for label in node.refs}
    if not referenced <= label_set:
        return None
    named_targets = label_set | referenced
    for node in nodes:
        if node.kind == "label":
            continue
        code = _code(node.text)
        if any(code[i].kind == "identifier" and code[i].text in named_targets
               and code[i + 1].text == ":" for i in range(len(code) - 1)):
            return None
        # GCC label addresses / computed gotos are outside this proof model.
        if any(code[i].text == "&&" and code[i + 1].text in label_set
               for i in range(len(code) - 1)):
            return None
    return nodes


def _inventory(nodes):
    labels = {node.label: index for index, node in enumerate(nodes) if node.kind == "label"}

    incoming = defaultdict(list)
    for index, node in enumerate(nodes):
        for label, count in node.refs.items():
            incoming[label].append((index, count))
    return labels, incoming


def _region_safe(nodes, start, end):
    return end - start <= _MAX_REGION and all(node.movable for node in nodes[start:end])


def _comments(*nodes):
    return [comment for node in nodes for comment in node.comments]


def _strip_parens(expression):
    while True:
        code = _code(expression)
        if len(code) < 2 or code[0].text != "(" or _closing(code, 0) != len(code) - 1:
            return expression.strip()
        # Keep comments before/after the parentheses; they are not in _code
        # but can carry a recovered source annotation.
        expression = (expression[:code[0].start] + expression[code[0].end:code[-1].start]
                      + expression[code[-1].end:])


def _invert(condition):
    """Negate a scalar condition, cancelling only a proved outer logical !.

    In particular, do not invert relational operators: that changes floating
    comparisons involving NaNs. A `!` at the front of `!x && y` is not the
    outer operator, so it also must not be cancelled.
    """
    condition = _strip_parens(condition)
    code = _code(condition)
    if code and code[0].text == "!":
        operand = condition[code[0].end:]
        annotation = condition[:code[0].start]
        rest = _code(operand)
        if rest and rest[0].text == "(" and _closing(rest, 0) == len(rest) - 1:
            return (annotation + _strip_parens(operand)).strip()
        # An unparenthesized identifier/member/index/call expression may also
        # be the whole operand. No top-level arithmetic/logical operator is
        # allowed; operators inside its call/index arguments are irrelevant.
        depth = 0
        atom = bool(rest)
        for token in rest:
            if token.text in {"(", "["}:
                depth += 1
            elif token.text in {")", "]"}:
                depth -= 1
            elif not depth and token.kind != "identifier" and token.text not in {"::", "->", "."}:
                atom = False
        if atom:
            return (annotation + operand).strip()
    return f"!({condition})"


def _compound(lines):
    node = _node("\n".join(lines))
    # These nodes contain only scope-safe original statements and one of our
    # generated controls. Marking them movable allows a later outer region to
    # wrap one, while all remaining goto references are still inventoried.
    if node is not None:
        node.movable = True
    return node


def _loop(nodes, index, labels, incoming):
    head = nodes[index]
    if head.kind != "label":
        return None
    candidates = [source for source, count in incoming[head.label]
                  if index < source <= index + _MAX_REGION and count == 1]
    for back_index in candidates:
        back = nodes[back_index]
        if back.kind not in {"goto", "branch"} or back.label != head.label:
            continue
        # No other jump inside the proposed loop may re-enter its header.
        # External jumps to the header remain valid because its label is kept.
        if any(index < source < back_index for source, _ in incoming[head.label]):
            continue
        if back.kind == "branch" and _region_safe(nodes, index + 1, back_index):
            body = [node.text for node in nodes[index + 1:back_index]]
            formed = _compound(_comments(back) + ["do {"] + body
                               + [f"}} while ({back.condition});"])
            if formed is not None:
                return back_index + 1, [head, formed], {head.label}
        if back.kind == "goto" and index + 1 < back_index and _region_safe(nodes, index + 1, back_index):
            for guard_index in range(index + 1, back_index):
                guard = nodes[guard_index]
                exit_index = labels.get(guard.label) if guard.kind == "branch" else None
                if exit_index != back_index + 1:
                    continue
                body = [node.text for node in nodes[guard_index + 1:back_index]]
                if guard_index == index + 1:
                    formed = _compound(_comments(guard) + [f"while ({_invert(guard.condition)}) {{"]
                                       + body + _comments(back) + ["}"])
                else:
                    # Header compare/flag writes and calls must execute on
                    # every check, including the final check that exits. Keep
                    # those statements inside the loop, before its break.
                    header = [node.text for node in nodes[index + 1:guard_index]]
                    formed = _compound(["while (true) {"] + header + _comments(guard)
                                       + [f"if ({guard.condition}) break;"]
                                       + body + _comments(back) + ["}"])
                if formed is not None:
                    return back_index + 1, [head, formed], {head.label, guard.label}
    return None


def _conditional(nodes, index, labels, incoming):
    branch = nodes[index]
    if branch.kind != "branch":
        return None
    target_index = labels.get(branch.label, -1)
    if not index < target_index <= index + _MAX_REGION:
        return None
    # Diamond: branch to else; then arm skips to the common join.
    last_then = nodes[target_index - 1]
    end_index = labels.get(last_then.label, -1) if last_then.kind == "goto" else -1
    if (target_index < end_index <= index + _MAX_REGION
            and incoming[branch.label] == [(index, 1)]
            and _region_safe(nodes, index + 1, target_index - 1)
            and _region_safe(nodes, target_index + 1, end_index)):
        then_body = [node.text for node in nodes[index + 1:target_index - 1]]
        else_body = [node.text for node in nodes[target_index + 1:end_index]]
        formed = _compound(_comments(branch) + [f"if ({_invert(branch.condition)}) {{"]
                           + then_body + _comments(last_then) + ["} else {"]
                           + _comments(nodes[target_index]) + else_body + ["}"])
        if formed is not None:
            return end_index, [formed], {branch.label, last_then.label}
    # Forward branch skipping a single-entry sequence. The join label stays
    # outside the block, so any other incoming edge still lands correctly.
    if _region_safe(nodes, index + 1, target_index):
        body = [node.text for node in nodes[index + 1:target_index]]
        formed = _compound(_comments(branch) + [f"if ({_invert(branch.condition)}) {{"]
                           + body + ["}"])
        if formed is not None:
            return target_index, [formed], {branch.label}
    return None


def structure_body(lines: list[str]) -> list[str]:
    """Recover proved if/else and loop regions; leave unknown flow explicit.

    Conditions are copied exactly and evaluated once at the same point in
    each traversal. Calls, memory operations, flag writes, and all other
    statements are retained in order. Regions cannot contain an interior
    label or a declaration whose scope would change. Bounds keep this pass
    predictable on unusually large or badly delimited functions.
    """
    nodes = _parse(lines)
    if nodes is None:
        return list(lines)
    touched = set()
    changed_any = False
    for _ in range(_MAX_PASSES):
        labels, incoming = _inventory(nodes)
        out = []
        index = 0
        changed = False
        while index < len(nodes):
            replacement = (_loop(nodes, index, labels, incoming)
                           or _conditional(nodes, index, labels, incoming))
            if replacement:
                index, new, consumed = replacement
                out.extend(new)
                touched |= consumed
                changed = True
            else:
                out.append(nodes[index])
                index += 1
        nodes = out
        if not changed:
            break
        changed_any = True
        # Drop only labels touched by a proved rewrite and no longer used.
        # Comments on a consumed boundary remain standalone annotations.
        referenced = {label for node in nodes for label in node.refs}
        pruned = []
        for node in nodes:
            if node.kind == "label" and node.label in touched and node.label not in referenced:
                if node.comments:
                    pruned.append(_Node("\n".join(node.comments), "comment", Counter(),
                                        comments=node.comments, movable=True))
            else:
                pruned.append(node)
        nodes = pruned
    return [node.text for node in nodes] if changed_any else list(lines)

