#!/usr/bin/env python3
"""Emit a full lifted source copy with one guarded pure-kernel optimization.

The original source is never overwritten. Every byte outside the selected
function definition, including registration tables and public references,
remains unchanged. The transform is shared with spu_kernel_replay.py.
"""
import argparse
import os
from pathlib import Path
import re
import tempfile
from spu_kernel_replay import extract, promote_registers


def transform(data, function, *, allow_leaf=False):
    text = data.decode('utf-8', 'surrogateescape')
    for suffix in ('_reference', '_registers_safe'):
        if re.search(r'\b'+re.escape(function+suffix)+r'\s*\(', text):
            raise ValueError('generated helper name already exists: '+function+suffix)
    body, _, span = extract(text, function, with_span=True, allow_leaf=allow_leaf)
    newline = '\r\n' if '\r\n' in body else '\n'
    replacement = promote_registers(body.replace('\r\n', '\n'), function)
    replacement = replacement.replace('\n', newline)
    return (text[:span[0]]+replacement+text[span[1]:]).encode('utf-8', 'surrogateescape')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--function', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--leaf-block', action='store_true')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('output must differ from the original generated source')
    output = transform(args.source.read_bytes(), args.function, allow_leaf=args.leaf_block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=args.output.name+'.tmp-',
                                         delete=False) as f:
            temporary = Path(f.name)
            f.write(output)
        os.replace(temporary, args.output)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    print('guarded SPU kernel copy: '+str(args.output))


if __name__ == '__main__':
    main()
