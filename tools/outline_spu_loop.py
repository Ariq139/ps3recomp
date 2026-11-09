#!/usr/bin/env python3
"""Outline a validated pure generated-SPU loop into a build-cache source copy.

Only the generated one-line-per-instruction subset is supported. Entry must
skip to a label within the loop; the last instruction is its sole backedge.
No source input is overwritten, and no guest instruction is synthesized.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from spu_kernel_replay import extract, promote_registers


def mask(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]),
                  text, flags=re.S)


def outline(data, function, start, entry, exit_pc, mode):
    mode = mode.upper()
    if mode not in ('OFF', 'CAPTURE', 'NATIVE'):
        raise ValueError('unsupported outline mode')
    if not 0 <= start < entry < exit_pc <= 0x40000 or any(x & 3 for x in (start, entry, exit_pc)):
        raise ValueError('outline addresses must be ordered aligned local-store PCs')
    name = function + '_loop_%08X_%08X' % (start, exit_pc)
    info = dict(version=1, function=function, start=start, entry=entry, exit=exit_pc,
                mode=mode, helper=name, reference=name+'_reference',
                source_sha256=hashlib.sha256(data).hexdigest())
    if mode == 'OFF':
        return data, info
    text = data.decode('utf-8', 'surrogateescape')
    newline = '\r\n' if '\r\n' in text else '\n'
    normalized = text.replace('\r\n', '\n')
    body, _, span = extract(normalized, function, with_span=True, validate=False)
    plain = mask(body)
    start_label = 'loc_%08X' % start
    entry_label = 'loc_%08X' % entry
    labels = list(re.finditer(r'^\s*(loc_[0-9A-Fa-f]+):\s*$', plain, re.M))
    matches = [m for m in labels if m[1] == start_label]
    if len(matches) != 1:
        raise ValueError('loop start label must occur exactly once')
    begin = plain.rfind('\n', 0, matches[0].start()) + 1
    # Require one canonical conditional backedge to the start. Its following
    # instruction remains in the caller and is the loop's sole exit.
    backedges = list(re.finditer(r'\bgoto\s+'+start_label+r'\s*;', plain))
    if len(backedges) != 1:
        raise ValueError('loop requires one backedge to its start label')
    last = plain.rfind('\n', 0, backedges[0].start()) + 1
    finish = plain.find('\n', backedges[0].end())
    if finish < 0:
        raise ValueError('loop backedge must occupy its own generated line')
    if not re.fullmatch(r'\s*if\s*\([^;{}]+\)\s*goto\s+'+start_label+r'\s*;\s*', plain[last:finish]):
        raise ValueError('loop backedge must be a conditional generated instruction')
    finish += 1
    if finish <= begin:
        raise ValueError('loop backedge precedes its start')
    region = body[begin:finish]
    region_plain = plain[begin:finish]
    region_labels = re.findall(r'^\s*(loc_[0-9A-Fa-f]+):\s*$', region_plain, re.M)
    if len(region_labels) != len(set(region_labels)) or entry_label not in region_labels:
        raise ValueError('loop labels must be unique and contain the entry')
    internal = set(region_labels)
    gotos = set(re.findall(r'\bgoto\s+(\w+)\s*;', region_plain))
    if gotos - internal:
        raise ValueError('loop has an outgoing branch: '+str(sorted(gotos-internal)))
    outside = plain[:begin] + plain[finish:]
    if re.search(r'\bgoto\s*\*|&&\s*loc_', plain):
        raise ValueError('computed branches and label addresses are unsupported')
    incoming = set(re.findall(r'\bgoto\s+(\w+)\s*;', outside)) & internal
    if incoming != {entry_label}:
        raise ValueError('outside branches must enter only at '+entry_label)
    prefix_lines = [line.strip() for line in plain[:begin].splitlines() if line.strip()]
    if (not prefix_lines or
            not re.fullmatch(r'goto\s+'+entry_label+r'\s*;', prefix_lines[-1]) or
            (len(prefix_lines) > 1 and not prefix_lines[-2].endswith(';') and
             not (len(prefix_lines) == 2 and prefix_lines[-2].endswith('{'))) or
            plain[:begin].count('{') - plain[:begin].count('}') != 1):
        raise ValueError('prefix must end with a bare unconditional branch to the entry')
    instructions = []
    for line in region_plain.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r'loc_[0-9A-Fa-f]+:', line):
            continue
        # A canonical lifter statement is one instruction per physical line.
        # Masked nop/hint comments retain their semicolon and still count.
        if not line.endswith(';') or line.count(';') != 1:
            raise ValueError('loop requires one generated instruction per line')
        instructions.append(line)
    if start + len(instructions)*4 != exit_pc:
        raise ValueError('instruction count does not match the requested exit PC')
    info['instructions'] = len(instructions)
    # In this subset the guard's assumptions remain invariant: only GPRs,
    # local-store helpers and internal branches may occur in the loop.
    if set(re.findall(r'ctx\s*->\s*(\w+)', region_plain)) - {'gpr'}:
        raise ValueError('loop may not observe or mutate other context fields')
    if re.search(r'\b(?:return|g_spu_trampoline_fn)\b', region_plain):
        raise ValueError('loop may not return or replace the continuation')
    synthetic = 'void '+name+'(spu_context* ctx) {\n    goto '+entry_label+';\n'+region+'    return;\n}'
    # extract validates calls/context escapes; promote_registers enforces the
    # canonical alias-free expression/mutation subset even in CAPTURE mode.
    extract(synthetic, name, allow_leaf=True)
    promoted = promote_registers(synthetic, name)
    if re.search(r'\b'+re.escape(name)+r'(?:_reference|_registers_safe)?\b', normalized):
        raise ValueError('outline helper name collides with the source')
    if mode == 'NATIVE':
        helper = promoted
        invocation = '    '+name+'(ctx);\n'
    else:
        helper = synthetic.replace('void '+name+'(', 'void '+name+'_reference(', 1)
        invocation = '    spu_kernel_capture_invoke(ctx, 0x%Xu, %s_reference);\n' % (entry, name)
    updated_body = body[:begin]+entry_label+':\n'+invocation+body[finish:]
    output = normalized[:span[0]] + helper+'\n\n'+updated_body + normalized[span[1]:]
    if mode == 'CAPTURE':
        output = '#include "spu_kernel_capture_api.h"\n'+output
    return output.replace('\n', newline).encode('utf-8', 'surrogateescape'), info


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--function', required=True)
    for address in ('start', 'entry', 'exit'):
        parser.add_argument('--'+address, type=lambda s: int(s, 0), required=True)
    parser.add_argument('--mode', choices=('OFF', 'CAPTURE', 'NATIVE'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('output must differ from source')
    try:
        output, info = outline(args.source.read_bytes(), args.function, args.start,
                               args.entry, args.exit, args.mode)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=args.output.name+'.tmp-', delete=False) as f:
            temporary = Path(f.name)
            f.write(output)
        os.replace(temporary, args.output)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    info['output_sha256'] = hashlib.sha256(output).hexdigest()
    args.output.with_suffix(args.output.suffix+'.outline.json').write_text(json.dumps(info, indent=2)+'\n')
    print('outlined SPU loop: '+str(args.output)+' ('+info['reference']+')')


if __name__ == '__main__':
    main()
