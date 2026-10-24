"""Run the native boot collector against bounded synthetic program archives."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def spu_elf():
    data = bytearray(112)
    data[:7] = b'\x7fELF\x01\x02\x01'
    struct.pack_into('>HHIIIIIHHHHHH', data, 16, 2, 23, 1, 0x1000,
                     52, 0, 0, 52, 32, 1, 40, 0, 0)
    struct.pack_into('>IIIIIIII', data, 52, 1, 96, 0x1000, 0x1000, 16, 16, 5, 16)
    data[96:] = bytes(range(16))
    return bytes(data)


def shader(profile):
    data = bytearray(80)
    struct.pack_into('>8I', data, 0, profile, 6, 80, 0, 0, 32, 16, 64)
    struct.pack_into('>6I', data, 32, 1, 0, 2, 1, 1, 0)
    data[64:] = bytes(range(16))
    return bytes(data)


def cgb(profile):
    data = bytearray(72)
    struct.pack_into('>4sHHHBB', data, 0, b'CGB\0', 0, 0, 16, profile, 3)
    data[32:48] = bytes(range(16))
    struct.pack_into('>HH', data, 48, 16, 0)
    struct.pack_into('>HHH', data, 64, 8, 0, 0)
    return bytes(data)


def compressed(payload, prefix=bytes(256)):
    bits = ''.join('0' + f'{b:08b}' for b in reversed(payload))
    return compressed_bits(bits, len(payload), prefix)


def compressed_bits(bits, size, prefix=bytes(256)):
    bits += '0' * (-len(bits) % 8)
    packed = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))[::-1]
    return b'CRILAYLA' + struct.pack('<II', size, len(packed)) + packed + prefix


class BootDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='boot-program-test-')
        cls.base = Path(cls.temp.name)
        source = (ROOT / 'runtime/cache/boot_program_dump.cpp').as_posix()
        harness = cls.base / 'main.cpp'
        harness.write_text(f'#include "{source}"\n#include <iostream>\n' + r'''
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--root") {
        std::cout << cache_root().string(); return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--unpack") {
        std::ifstream in(argv[2], std::ios::binary);
        std::vector<byte> data((std::istreambuf_iterator<char>(in)), {});
        try {
            auto out = unpack_crilayla(data.data(), data.size());
            std::ofstream file(argv[3], std::ios::binary);
            file.write(reinterpret_cast<const char*>(out.data()), out.size());
            return out.empty() ? 2 : 0;
        } catch (...) { return 3; }
    }
    if (argc != 2) return 1;
    ps3_boot_collect_programs(argv[1], nullptr, nullptr);
}
''', encoding='utf-8')
        if sys.platform == 'win32':
            compiler = shutil.which('clang-cl')
            if not compiler or not os.environ.get('INCLUDE'):
                raise unittest.SkipTest('requires VS developer environment and clang-cl')
            cls.exe = cls.base / 'collector.exe'
            cmd = [compiler, '/nologo', '/std:c++20', '/EHsc', '/O2',
                   str(harness), '/Fe:' + str(cls.exe)]
        else:
            compiler = shutil.which('clang++') or shutil.which('g++')
            if not compiler:
                raise unittest.SkipTest('requires C++20 compiler')
            cls.exe = cls.base / 'collector'
            cmd = [compiler, '-std=c++20', '-O2', str(harness), '-o', str(cls.exe)]
        result = subprocess.run(cmd, cwd=cls.base, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_boot_inventory_reuse_and_recovery(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            root = Path(temp); game = root / 'game'; game.mkdir()
            cache = root / 'cache'
            raw = bytearray(128)
            raw[:16] = b'\x42\x00\x00\x02' * 4
            raw[48:60] = b'JOBCRT Ver14'
            struct.pack_into('>I', raw, 60, 64)
            assets = shader(7003) + cgb(0) + spu_elf() + bytes(raw)
            (game / 'data.cpk').write_bytes(b'CPK ' + assets + compressed(shader(7004) + cgb(1)))
            # Corrupt program offsets and a truncated ELF must not be exported.
            bad = bytearray(shader(7003)); struct.pack_into('>I', bad, 28, 0xffffffff)
            bad_cgb = bytearray(cgb(0)); struct.pack_into('>H', bad_cgb, 48, 0xffff)
            (game / 'bad.bin').write_bytes(bad + bad_cgb + spu_elf()[:60])
            env = dict(os.environ, PS3RECOMP_CACHE_DIR=str(cache), PS3RECOMP_BOOT_DUMP='1',
                       SPU_DUMP_JOB_IMAGES='1', RSX_DUMP_SHADER_IMAGES='1')

            def run():
                result = subprocess.run([str(self.exe), str(game)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(next((cache / 'boot').rglob('summary.json')).read_text())

            first = run()
            self.assertEqual(first['errors'], 0)
            self.assertEqual(first['new_shader_instances'], 4)
            self.assertEqual(first['new_spu_elf_instances'], 1)
            self.assertEqual(first['new_raw_candidates'], 1)
            self.assertEqual(first['new_members_decompressed'], 1)
            self.assertFalse(first['complete_game_coverage'])
            self.assertEqual(len(list((cache / 'shaders/rsx').glob('*.cg'))), 2)
            self.assertEqual(len(list((cache / 'shaders/rsx').glob('*.cgb'))), 2)
            self.assertEqual(next((cache / 'spu_jobs/boot').glob('*.elf')).read_bytes(), spu_elf())
            reports = [json.loads(line) for p in (cache / 'boot').rglob('*.jsonl')
                       for line in p.read_text().splitlines()]
            self.assertTrue(any('!CRILAYLA@' in r['source'] for r in reports))
            self.assertTrue(any(r.get('extent') == 'jobcrt-header-candidate' for r in reports))
            second = run()
            self.assertEqual(second['files_scanned'], 0)
            self.assertGreaterEqual(second['files_reused'], 2)
            next((cache / 'shaders/rsx').glob('vp_*.ucode')).unlink()
            third = run()
            self.assertEqual(third['files_scanned'], 1)
            # A malformed stamp should force a fresh scan, not abort the source.
            stamp = next((cache / 'boot').rglob('*.stamp'))
            lines = stamp.read_text().splitlines(); stamp.write_text(lines[0] + '\nnot-a-size output\n')
            fourth = run()
            self.assertEqual(fourth['errors'], 0)
            self.assertEqual(fourth['files_scanned'], 1)

    def test_crilayla_literals_overlap_and_invalid_references(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            path = Path(temp); source = path / 'packed.bin'; dest = path / 'decoded.bin'
            prefix = bytes(range(256))
            payload = b'XYZXYZXYZ'
            literals = ''.join('0' + f'{b:08b}' for b in b'ZYX')
            # Distance 3, length 6: 3 + saturated 2-bit field + zero 3-bit field.
            bits = literals + '1' + '0' * 13 + '11' + '000'
            for blob in [compressed(payload, prefix), compressed_bits(bits, len(payload), prefix)]:
                source.write_bytes(blob)
                result = subprocess.run([str(self.exe), '--unpack', str(source), str(dest)])
                self.assertEqual(result.returncode, 0)
                self.assertEqual(dest.read_bytes(), prefix + payload)
            source.write_bytes(compressed_bits('1' + '0' * 13 + '00', 3))
            self.assertNotEqual(subprocess.run([str(self.exe), '--unpack', str(source), str(dest)]).returncode, 0)

    def test_disable(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            root = Path(temp); game = root / 'game'; game.mkdir()
            (game / 'program.bin').write_bytes(shader(7003))
            cache = root / 'cache'
            env = dict(os.environ, PS3RECOMP_CACHE_DIR=str(cache), PS3RECOMP_BOOT_DUMP='0')
            subprocess.run([str(self.exe), str(game)], env=env, check=True)
            self.assertFalse(cache.exists())

    @unittest.skipUnless(sys.platform == 'win32', 'Windows shared cache root')
    def test_default_root_matches_windows_environment(self):
        env = dict(os.environ)
        env.pop('PS3RECOMP_CACHE_DIR', None)
        env['LOCALAPPDATA'] = str(self.base / 'local')
        result = subprocess.run([str(self.exe), '--root'], env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(Path(result.stdout), self.base / 'local/ps3recomp/cache')


if __name__ == '__main__':
    unittest.main()
