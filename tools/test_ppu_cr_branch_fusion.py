#!/usr/bin/env python3
"""Differential test for integer compare/branch fusion with materialized CR."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ppu_disasm import decode
from ppu_lifter import PPULifter, LiftedFunction, _fuse_cr

ROOT = Path(__file__).resolve().parents[1]


def emit_pair(cmp_word, bo, bi, displacement=8, link=0):
    lifter = PPULifter()
    func = LiftedFunction(name="f", start_addr=0x1000, end_addr=0x1100)
    cmp_line = "    " + lifter._translate(decode(cmp_word, 0x1000), func)
    branch_word = (16 << 26) | (bo << 21) | (bi << 16) | (displacement & 0xfffc) | link
    branch_line = "    " + lifter._translate(decode(branch_word, 0x1004), func)
    return [cmp_line, branch_line]


def test_shape():
    # cmpwi cr4,r3,-1; branch on EQ; CR remains materialized.
    cmp_word = (11 << 26) | (4 << 23) | (3 << 16) | 0xffff
    pair = emit_pair(cmp_word, 12, 4 * 4 + 2)
    fused = _fuse_cr(pair)
    assert len(fused) == 1 and "if (a == b)" in fused[0], (pair, fused)
    assert "ctx->cr = (ctx->cr" in fused[0] and "ctx->xer" in fused[0]
    # A label permits independent entry at the branch: refuse.
    labelled = [pair[0], "loc_00001004:", pair[1]]
    assert _fuse_cr(labelled) == labelled
    # SO, a different CR field, CTR decrement, unconditional and linked.
    for bo, bi, link in [(12, 19, 0), (12, 2, 0), (8, 18, 0),
                         (20, 18, 0), (12, 18, 1)]:
        other = emit_pair(cmp_word, bo, bi, link=link)
        assert _fuse_cr(other) == other, other
    # Backward internal goto is safe because CR is still written.
    back = emit_pair(cmp_word, 4, 18, displacement=-4)
    assert len(_fuse_cr(back)) == 1 and "if (a != b)" in _fuse_cr(back)[0]


def test_execution():
    cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        if os.environ.get("CI"):
            raise SystemExit("FAIL: no C compiler")
        print("SKIP: no C compiler")
        return
    src = ["#include <stdint.h>", "#include <stdio.h>", "#include <assert.h>",
           "typedef struct { uint64_t gpr[32]; uint32_t cr,xer; } ctx_t;"]
    cases = []
    # Signed/unsigned, word/doubleword, immediate/register, all eight CR fields,
    # branch-if-set/clear for LT/GT/EQ. Independent entry labels are tested above.
    for signed in (False, True):
        for wide in (False, True):
            for imm in (False, True):
                for bf in range(8):
                    opcode = (11 if signed else 10) if imm else 31
                    cmp_word = (opcode << 26) | (bf << 23) | (int(wide) << 21) | (3 << 16)
                    cmp_word |= 0xffff if imm else ((4 << 11) | (0 if signed else (32 << 1)))
                    for bit in range(3):
                        for expected in (0, 1):
                            pair = emit_pair(cmp_word, 4 | (8 if expected else 0), bf * 4 + bit)
                            fused = _fuse_cr(pair)
                            assert len(fused) == 1, pair
                            i = len(cases)
                            for tag, body in (("ref", pair), ("opt", fused)):
                                src += [f"static int {tag}_{i}(ctx_t* ctx) {{", *body,
                                        "return 0;", "loc_0000100C: return 1;", "}"]
                            cases.append((i, signed, wide, imm, bf, bit, expected))
    src += ["typedef int (*test_fn)(ctx_t*);",
            "static test_fn const ref[] = {" + ",".join(f"ref_{i}" for i, *_ in cases) + "};",
            "static test_fn const opt[] = {" + ",".join(f"opt_{i}" for i, *_ in cases) + "};",
            "static const uint64_t edge[] = {0,1,0xffff,0x10000,0x7fffffff,0x80000000,0xffffffff,"
            "0x100000000ull,0x7fffffffffffffffull,0x8000000000000000ull,0xffffffffffffffffull};",
            "static uint64_t rng=0x6477abcdef123456ull;",
            "static uint64_t next(void) { rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; }",
            "int main(void) {",
            "for (unsigned i=0;i<sizeof(ref)/sizeof(ref[0]);++i) {",
            "for (unsigned k=0;k<400;++k) {",
            "ctx_t a={0}; a.gpr[3]=k<121?edge[k/11]:next();",
            "a.gpr[4]=k<121?edge[k%11]:next();",
            "a.cr=(uint32_t)next(); a.xer=(uint32_t)next(); ctx_t b=a;",
            "int ar=ref[i](&a), br=opt[i](&b);",
            "assert(ar==br && a.cr==b.cr && a.xer==b.xer);",
            "assert(a.gpr[3]==b.gpr[3] && a.gpr[4]==b.gpr[4]);",
            "}}",
            'puts("PASS: integer compare/branch decisions and materialized CR agree"); return 0; }']
    with tempfile.TemporaryDirectory(prefix="ppu-cr-fusion-") as td:
        cpath = Path(td) / "test.c"
        exe = Path(td) / ("test.exe" if os.name == "nt" else "test")
        cpath.write_text("\n".join(src))
        subprocess.run([cc, "-std=c11", "-O2", "-UNDEBUG", str(cpath), "-o", str(exe)],
                       check=True, timeout=180)
        subprocess.run([str(exe)], check=True, timeout=30)
    print(f"{len(cases)} compare/branch pairs x 400 inputs")


if __name__ == "__main__":
    test_shape()
    test_execution()
