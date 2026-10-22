#!/usr/bin/env python3
"""Inventory embedded SPU ELFs and raw SPURS GUIDs without running the game.

Scans files directly (including uncompressed archive contents), using mmap.
Compressed/encrypted contents are NOT decoded. Unknown raw extents are reported,
not guessed into runnable registrations. RPCS3 behavior reference:
Emu/Cell/PPUModule.cpp (roaming SPU search), SPUCommonRecompiler.cpp (precompile).
This is a local implementation; no RPCS3 source body is incorporated.
"""
import argparse
import hashlib
import json
import mmap
import re
import struct
from pathlib import Path

from extract_spu_images import parse_spu_elf_size
from build_spu_workloads import fnv1a64

# Four ILA r2 instructions encode the SPURS GUID. Instruction encoding, not
# arbitrary strings or filename extensions, identifies the raw candidates.
GUID = re.compile(rb'(?:[\x42\x43][\s\S]{2}[\x02\x82]){4}')
LS_SIZE = 0x40000


def known_images(paths):
    result = {}
    for root in paths:
        files = sorted(root.rglob('spujob_*.bin')) if root.is_dir() else [root]
        for path in files:
            data = path.read_bytes()
            if len(data) >= 16 and GUID.fullmatch(data[:16]):
                result.setdefault(data[:16], {})[hashlib.sha256(data).hexdigest()] = (data, str(path))
    return result


def scan(path, known, output):
    rows = []
    if not path.stat().st_size:
        return rows
    with path.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        offset = 0
        while True:
            offset = data.find(b'\x7fELF', offset)
            if offset < 0:
                break
            # The extractor's ELF64 defensive path needs more than a 52-byte
            # header; SPU ELF32 only is accepted by this inventory.
            if offset + 0x34 <= len(data) and data[offset + 4] == 1:
                try:
                    size = parse_spu_elf_size(data, offset)
                except (ValueError, struct.error, IndexError):
                    size = None
                if size and size <= 16 * 1024 * 1024 and offset + size <= len(data):
                    rows.append({'kind': 'spu-elf', 'offset': offset, 'size': size,
                                 'extent': 'elf-tables', 'ready_to_lift': True})
            offset += 4
        starts = [m.start() for m in GUID.finditer(data) if m.start() % 4 == 0]
        sections = []
        if data[:6] == b'\x7fELF\x02\x02' and len(data) >= 64:
            shoff = struct.unpack_from('>Q', data, 40)[0]
            stride, count = struct.unpack_from('>HH', data, 58)
            if stride >= 64 and shoff + stride * count <= len(data):
                for n in range(count):
                    _, typ, flags, _, off, size, *_ = struct.unpack_from('>IIQQQQIIQQ', data, shoff + stride*n)
                    if typ != 8 and flags & 2 and off + size <= len(data):
                        sections.append((off, off+size))
        for index, offset in enumerate(starts):
            if any(r['kind'] == 'spu-elf' and r['offset'] <= offset < r['offset'] + r['size'] for r in rows):
                continue  # Embedded ELF owns this code; do not reinterpret its LS base.
            row = {'kind': 'raw-spurs', 'offset': offset, 'guid': data[offset:offset+16].hex(),
                   'size': None, 'extent': 'unknown', 'ready_to_lift': False}
            bound = min(len(data), offset + LS_SIZE,
                        starts[index+1] if index+1 < len(starts) else len(data))
            for lo, hi in sections:
                if lo <= offset < hi:
                    bound = min(bound, hi)
            for blob, source in known.get(data[offset:offset+16], {}).values():
                if offset + len(blob) <= len(data) and data[offset:offset+len(blob)] == blob:
                    row.update(size=len(blob), extent='exact-runtime-capture',
                               capture=source, ready_to_lift=True)
                    break
            if row['size'] is None and data[offset+0x30:offset+0x3c] == b'JOBCRT Ver14':
                # Observed JOBCRT layout: initialized-end field + fixed 64B
                # tail. Verified against runtime sizes 32064 and 31232. Keep
                # the inferred extent labelled until a runtime match proves it.
                size = struct.unpack_from('>I', data, offset + 0x3c)[0] + 0x40
                if 0x80 <= size <= LS_SIZE and size % 16 == 0 and offset + size <= bound:
                    row.update(size=size, extent='jobcrt-header-candidate')
            row['bounded_available_bytes'] = bound - offset
            # Retain the enclosing bytes independently of inferred image size.
            # Runtime can verify an exact submitted subrange against this
            # immutable snapshot when its descriptor supplies the true size.
            region = bytes(data[offset:bound])
            region_sha = hashlib.sha256(region).hexdigest()
            region_path = output / ('region_' + region_sha[:20] + '.bin')
            if not region_path.exists():
                region_path.write_bytes(region)
            row['region'] = str(region_path.resolve())
            row['region_sha256'] = region_sha
            rows.append(row)
        for row in rows:
            row['source'] = str(path.resolve())
            if row['size']:
                blob = data[row['offset']:row['offset']+row['size']]
                row['fingerprint'] = f'{fnv1a64(blob):016X}'
                row['sha256'] = hashlib.sha256(blob).hexdigest()
                suffix = '.elf' if row['kind'] == 'spu-elf' else '.bin'
                dest = output / ('spu_' + row['sha256'][:20] + suffix)
                if not dest.exists():
                    dest.write_bytes(blob)
                row['extracted'] = str(dest.resolve())
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--known', action='append', default=[], type=Path,
                    help='Runtime dump file or directory (repeatable)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    known = known_images(args.known)
    rows, sources = [], []
    for path in args.inputs:
        print(f'Scanning {path} ({path.stat().st_size:,} bytes)', flush=True)
        found = scan(path, known, args.out)
        rows.extend(found)
        sources.append({'path': str(path.resolve()), 'bytes': path.stat().st_size,
                        'candidates': len(found), 'archive_decompression': False})
        manifest = {'sources': sources, 'candidates': rows,
                    'limits': ['Direct bytes only; compressed/encrypted/generated code may be absent.',
                               'Raw GUID candidates with unknown extents are not runnable images.',
                               'JOBCRT header extents require runtime confirmation before registration.']}
        (args.out / 'inventory.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'  {len(found)} candidates; {len(rows)} total', flush=True)
    print(f'Inventory: {args.out / "inventory.json"}', flush=True)


if __name__ == '__main__':
    main()
