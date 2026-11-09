import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lift-dir', type=Path, required=True)
    parser.add_argument('--image-manifest', type=Path, required=True)
    parser.add_argument('--data-refs', type=Path, required=True)
    parser.add_argument('--data-map', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    table_path = args.lift_dir / 'ppu_recomp.cpp_table.cpp'
    table_text = table_path.read_text(encoding='utf-8', errors='replace')
    rows = [
        (int(ea, 16), int(fn, 16))
        for ea, fn in re.findall(
            r'\{\s*0x([0-9A-Fa-f]+)ULL\s*,\s*june_evt_func_([0-9A-Fa-f]{8})\s*,',
            table_text,
        )
    ]
    if not rows or any(ea != fn for ea, fn in rows):
        raise SystemExit('June function table did not parse as expected')
    if len(rows) != len(set(ea for ea, _ in rows)):
        raise SystemExit('June function table contains duplicate addresses')
    rows.sort()
    declared_count = re.search(r'june_evt_function_table_count\s*=\s*(\d+)', table_text)
    if not declared_count or int(declared_count.group(1)) != len(rows):
        raise SystemExit('June function table count does not match parsed rows')

    manifest = json.loads(args.image_manifest.read_text(encoding='utf-8'))
    if manifest.get('allocated_relocations'):
        raise SystemExit('June image contains allocated relocations')
    if manifest.get('opd', {}).get('entry_size') != 8 or not manifest.get('toc'):
        raise SystemExit('June image metadata has no supported OPD layout')
    data_refs = json.loads(args.data_refs.read_text(encoding='utf-8'))
    if data_refs.get('source_elf_sha256') != manifest.get('source_sha256'):
        raise SystemExit('June data reference list is for a different ELF')
    # This count records the scan that produced the reference list. Later
    # callback recovery can extend the function table without changing it.
    scanned_functions = data_refs.get('function_count', 0)
    if not 0 < scanned_functions <= len(rows):
        raise SystemExit('June data reference scan is incompatible with the function table')
    data_addresses = [int(ea, 16) for ea in data_refs.get('addresses', [])]
    if len(data_addresses) != data_refs.get('address_count') or len(data_addresses) != len(set(data_addresses)):
        raise SystemExit('June data reference list has an invalid count or duplicate address')
    if data_addresses != sorted(data_addresses):
        raise SystemExit('June data reference list is not sorted')
    for ea in data_addresses:
        if not any(segment['vaddr'] <= ea < segment['vaddr'] + segment['memsz']
                   for segment in manifest['segments']):
            raise SystemExit(f'June data reference 0x{ea:08X} is outside PT_LOAD')

    shared_data = json.loads(args.data_map.read_text(encoding='utf-8'))
    if shared_data.get('source_elf_sha256') != manifest.get('source_sha256'):
        raise SystemExit('June shared-data map is for a different source ELF')
    if shared_data.get('mapping_count') != len(shared_data.get('mappings', [])):
        raise SystemExit('June shared-data map has an invalid count')
    data_pairs = sorted((int(x['source'], 16), int(x['target'], 16))
                        for x in shared_data['mappings'])
    if len({source for source, _ in data_pairs}) != len(data_pairs) or \
       len({target for _, target in data_pairs}) != len(data_pairs):
        raise SystemExit('June shared-data map is not one-to-one')
    if set(source for source, _ in data_pairs) & set(target for _, target in data_pairs):
        raise SystemExit('June shared-data source and target addresses overlap')
    if not set(source for source, _ in data_pairs).issubset(set(data_addresses)):
        raise SystemExit('June shared-data map includes an address outside the static reference list')

    image_path = args.image_manifest.with_suffix('.bin')
    image_bytes = image_path.read_bytes()
    if len(image_bytes) != manifest['image_size'] or hashlib.sha256(image_bytes).hexdigest() != manifest['image_sha256']:
        raise SystemExit('June image does not match its manifest')

    implementations = set()
    source_paths = list(args.lift_dir.glob('*.cpp'))
    source_paths.append(args.lift_dir.parent / 'event_editor_june.cpp')
    for source_path in source_paths:
        source = source_path.read_text(encoding='utf-8', errors='replace')
        for address in re.findall(
                r'\bvoid\s+june_evt_func_([0-9A-Fa-f]{8})\s*\([^)]*\)\s*\{', source):
            address = int(address, 16)
            if address in implementations:
                raise SystemExit(f'Duplicate June implementation: 0x{address:08X}')
            implementations.add(address)
    missing = {ea for ea, _ in rows} - implementations
    if missing:
        raise SystemExit(f'June function table has missing implementations: {sorted(missing)}')

    # Callbacks stored in the executable also need wrappers, even when they
    # were recovered separately from the original direct-call function table.
    callable_addresses = {ea for ea, _ in rows}
    opd = manifest['opd']
    for address in range(opd['vaddr'], opd['vaddr'] + opd['size'], 8):
        code, toc = struct.unpack_from('>II', image_bytes, address - manifest['image_base'])
        if toc == manifest['toc'] and code in implementations:
            callable_addresses.add(code)
    declarations = (args.lift_dir / 'ppu_recomp.h').read_text(encoding='utf-8', errors='replace')
    declared = {int(x, 16) for x in re.findall(r'\bvoid\s+june_evt_func_([0-9A-Fa-f]{8})\s*\(', declarations)}
    if callable_addresses - declared:
        raise SystemExit(f'June callbacks have missing declarations: {sorted(callable_addresses - declared)}')
    rows = [(ea, ea) for ea in sorted(callable_addresses)]
    fnv = 14695981039346656037
    for value in image_bytes:
        fnv = ((fnv ^ value) * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = [
        '#include "event_editor_june_api.h"',
        '',
    ]
    for ea, fn in rows:
        dispatch.extend([
            f'static void june_evt_dispatch_{ea:08X}(ppu_context* ctx)',
            '{',
            f'    if (!june_evt_context_enter(ctx, 0x{ea:08X}u)) return;',
            f'    june_evt_func_{fn:08X}(ctx);',
            '    june_evt_context_leave(ctx);',
            '}',
            '',
        ])
    dispatch.extend([
        'extern "C" const JuneEventDispatchEntry june_evt_dispatch_table[] = {',
    ])
    dispatch.extend(
        f'    {{0x{ea:08X}u, june_evt_dispatch_{ea:08X}}},' for ea, _ in rows
    )
    dispatch.extend([
        '};',
        f'extern "C" const uint64_t june_evt_dispatch_table_count = {len(rows)};',
        '',
    ])
    (args.output_dir / 'june_dispatch.cpp').write_text('\n'.join(dispatch), encoding='utf-8', newline='\n')

    data_include = [
        'static constexpr uint32_t june_static_data_refs[] = {',
    ]
    data_include.extend(f'    0x{ea:08X}u,' for ea in data_addresses)
    data_include.extend(['};', ''])
    (args.output_dir / 'june_static_data_refs.inc').write_text(
        '\n'.join(data_include), encoding='utf-8', newline='\n')

    shared_include = [
        'static constexpr JuneDataMapEntry june_data_map[] = {',
    ]
    shared_include.extend(
        f'    {{0x{source:08X}u, 0x{target:08X}u}},' for source, target in data_pairs)
    shared_include.extend(['};', ''])
    args.data_map.with_suffix('.inc').write_text(
        '\n'.join(shared_include), encoding='utf-8', newline='\n')

    image = manifest
    tls = image['tls']
    opd = image['opd']
    constants = [
        '#pragma once',
        f'#define JUNE_IMAGE_BASE 0x{image["image_base"]:08X}u',
        f'#define JUNE_IMAGE_SIZE 0x{image["image_size"]:08X}u',
        f'#define JUNE_TOC 0x{image["toc"]:08X}u',
        f'#define JUNE_OPD_BASE 0x{opd["vaddr"]:08X}u',
        f'#define JUNE_OPD_SIZE 0x{opd["size"]:08X}u',
        f'#define JUNE_TLS_VADDR 0x{tls["vaddr"]:08X}u',
        f'#define JUNE_TLS_FILE_SIZE 0x{tls["filesz"]:08X}u',
        f'#define JUNE_TLS_MEM_SIZE 0x{tls["memsz"]:08X}u',
        f'#define JUNE_IMAGE_FNV1A64 0x{fnv:016X}ull',
        f'#define JUNE_SOURCE_SHA256 "{manifest["source_sha256"]}"',
        f'#define JUNE_IMAGE_SHA256 "{manifest["image_sha256"]}"',
        f'#define JUNE_TARGET_ELF_SHA256 "{shared_data["target_elf_sha256"]}"',
        f'#define JUNE_LOAD_SEGMENT_COUNT {len(image["segments"])}u',
    ]
    for index, segment in enumerate(image['segments']):
        constants.append(f'#define JUNE_LOAD_{index}_BEGIN 0x{segment["vaddr"]:08X}u')
        constants.append(f'#define JUNE_LOAD_{index}_END 0x{segment["vaddr"] + segment["memsz"]:08X}u')
    constants.extend([
        '#define JUNE_EVENT_MENU_ROW 0x0111DE00u',
        '#define FEB16_EVENT_MENU_ROW 0x011F08ECu',
        '#define FEB16_TOC 0x011ECE08u',
        '',
    ])
    (args.output_dir / 'june_image_info.h').write_text('\n'.join(constants), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
