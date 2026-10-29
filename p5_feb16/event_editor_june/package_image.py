import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elf', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    raw = args.elf.read_bytes()
    if len(raw) < 64 or raw[:6] != b'\x7fELF\x02\x02':
        raise SystemExit('expected a 64-bit big-endian ELF')
    phoff = struct.unpack_from('>Q', raw, 32)[0]
    phentsize = struct.unpack_from('>H', raw, 54)[0]
    phnum = struct.unpack_from('>H', raw, 56)[0]
    segments = []
    tls = None
    if phentsize < 56 or phoff + phentsize * phnum > len(raw):
        raise SystemExit('program header table is outside the ELF')
    for index in range(phnum):
        pos = phoff + index * phentsize
        p_type = struct.unpack_from('>I', raw, pos)[0]
        p_offset, p_vaddr = struct.unpack_from('>QQ', raw, pos + 8)
        p_filesz, p_memsz = struct.unpack_from('>QQ', raw, pos + 32)
        if p_type == 7:
            tls = {'vaddr': p_vaddr, 'filesz': p_filesz, 'memsz': p_memsz}
        if p_type != 1 or not p_memsz:
            continue
        if p_offset + p_filesz > len(raw) or p_vaddr + p_memsz > 0x100000000:
            raise SystemExit(f'invalid PT_LOAD segment {index}')
        segments.append({'index': index, 'offset': p_offset, 'vaddr': p_vaddr,
                         'filesz': p_filesz, 'memsz': p_memsz})
    if not segments:
        raise SystemExit('no nonempty PT_LOAD segments')
    shoff = struct.unpack_from('>Q', raw, 40)[0]
    shentsize = struct.unpack_from('>H', raw, 58)[0]
    shnum = struct.unpack_from('>H', raw, 60)[0]
    shstrndx = struct.unpack_from('>H', raw, 62)[0]
    if shentsize < 64 or shoff + shentsize * shnum > len(raw) or shstrndx >= shnum:
        raise SystemExit('section header table is outside the ELF')
    shstr_header = shoff + shstrndx * shentsize
    shstr_off = struct.unpack_from('>Q', raw, shstr_header + 24)[0]
    shstr_size = struct.unpack_from('>Q', raw, shstr_header + 32)[0]
    if shstr_off + shstr_size > len(raw):
        raise SystemExit('section-name table is outside the ELF')
    shstr = raw[shstr_off:shstr_off + shstr_size]
    opd = None
    toc_values = set()
    relocation_sections = []
    for index in range(shnum):
        pos = shoff + index * shentsize
        name_offset = struct.unpack_from('>I', raw, pos)[0]
        if name_offset >= len(shstr):
            continue
        name_end = shstr.find(b'\0', name_offset)
        name = shstr[name_offset:name_end if name_end >= 0 else len(shstr)].decode('ascii', errors='replace')
        section_type = struct.unpack_from('>I', raw, pos + 4)[0]
        section_flags = struct.unpack_from('>Q', raw, pos + 8)[0]
        section_vaddr = struct.unpack_from('>Q', raw, pos + 16)[0]
        section_offset = struct.unpack_from('>Q', raw, pos + 24)[0]
        section_size = struct.unpack_from('>Q', raw, pos + 32)[0]
        if section_type in (4, 9) and section_size and section_flags & 2:
            relocation_sections.append({'name': name, 'type': section_type,
                                        'vaddr': section_vaddr, 'size': section_size})
        if name == '.opd':
            if section_offset + section_size > len(raw) or section_size % 8:
                raise SystemExit('invalid .opd section')
            opd = {'vaddr': section_vaddr, 'size': section_size, 'entry_size': 8}
            for offset in range(section_offset, section_offset + section_size, 8):
                code, toc = struct.unpack_from('>II', raw, offset)
                if code:
                    toc_values.add(toc)
    if relocation_sections:
        raise SystemExit(f'allocated relocation sections are unsupported: {relocation_sections}')
    if opd is None or len(toc_values) != 1:
        raise SystemExit('expected one .opd section with a single TOC value')
    base = min(s['vaddr'] for s in segments)
    end = max(s['vaddr'] + s['memsz'] for s in segments)
    image = bytearray(end - base)
    for s in segments:
        start = s['vaddr'] - base
        image[start:start + s['filesz']] = raw[s['offset']:s['offset'] + s['filesz']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    manifest = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'image_base': base,
                'image_size': len(image), 'image_sha256': hashlib.sha256(image).hexdigest(),
                'segments': segments, 'tls': tls, 'opd': opd,
                'toc': next(iter(toc_values)), 'allocated_relocations': relocation_sections}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'base': hex(base), 'size': hex(len(image)),
                      'segments': len(segments), 'source_sha256': manifest['source_sha256'],
                      'image_sha256': manifest['image_sha256']}))

if __name__ == '__main__':
    main()
