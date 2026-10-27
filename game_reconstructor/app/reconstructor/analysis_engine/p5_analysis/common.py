import csv
import hashlib
import json
import re
import struct
from pathlib import Path


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
FUNCTION = re.compile(rb"(?ms)^void ([A-Za-z_][A-Za-z0-9_]*)\(ppu_context\* ctx\) \{.*?^\}(?:\r?\n|$)")
TOKEN = re.compile(
    rb'''(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n"]{0,16})\(.*?\)(?P=delimiter)"'''
    rb"""|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u8|u|U|L)?'(?:\\.|[^'\\])*'"""
    rb'''|//[^\r\n]*|/\*.*?\*/|(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z0-9_'.])*|[A-Za-z_][A-Za-z0-9_]*''',
    re.S)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def address(value):
    return int(value, 16) if isinstance(value, str) else int(value)


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes project: {relative}")
    return result


def rewrite_tokens(data, replacements):
    mapping = {k.encode("ascii"): v.encode("ascii") for k, v in replacements.items()}
    return TOKEN.sub(lambda m: mapping.get(m[0], m[0]), data)


def function_bodies(data):
    result = {}
    for match in FUNCTION.finditer(data):
        name = match[1].decode("ascii")
        if name in result:
            raise ValueError(f"Duplicate function definition: {name}")
        result[name] = match[0]
    return result


class Elf:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.sha256 = digest(self.data)
        if self.data[:6] != b"\x7fELF\x02\x02":
            raise ValueError("Expected a big-endian ELF64 image")
        self.machine = struct.unpack_from(">H", self.data, 18)[0]
        if self.machine != 21:
            raise ValueError("Expected EM_PPC64")
        self.entry = struct.unpack_from(">Q", self.data, 24)[0]
        phoff = struct.unpack_from(">Q", self.data, 32)[0]
        phsize, count = struct.unpack_from(">HH", self.data, 54)
        if phsize < 56 or phoff + phsize * count > len(self.data):
            raise ValueError("Invalid ELF program-header table")
        self.segments = []
        for i in range(count):
            typ, flags, offset, va, _, filesz, memsz, _ = struct.unpack_from(">IIQQQQQQ", self.data, phoff + i * phsize)
            if typ == 1:
                if offset + filesz > len(self.data) or filesz > memsz:
                    raise ValueError("Invalid ELF load segment")
                self.segments.append((va, offset, filesz, memsz, flags))

    def read(self, va, size):
        for base, offset, filesz, _, _ in self.segments:
            if base <= va and va + size <= base + filesz:
                return self.data[offset + va - base:offset + va - base + size]
        return None

    def mapped(self, va):
        return any(base <= va < base + memsz for base, _, _, memsz, _ in self.segments)

    def executable(self, va):
        return any(base <= va < base + filesz and flags & 1
                   for base, _, filesz, _, flags in self.segments)

    def u32(self, va):
        data = self.read(va, 4)
        return struct.unpack(">I", data)[0] if data is not None else None


def direct_branch(word, pc):
    if word is None:
        return None
    op = word >> 26
    if op not in (16, 18):
        return None
    bits, mask = (26, 0x03FFFFFC) if op == 18 else (16, 0xFFFC)
    displacement = word & mask
    if displacement & (1 << (bits - 1)):
        displacement -= 1 << bits
    target = displacement if word & 2 else pc + displacement
    return ("direct_call" if word & 1 else "direct_branch", target & 0xFFFFFFFFFFFFFFFF)
