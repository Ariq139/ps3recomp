#!/usr/bin/env python3
"""Compare the actual optional mixer helper to the original stereo loop."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(src, name):
    m = re.search(r"(?m)^static inline [^\n]*\b" + name + r"\s*\([^)]*\)\s*\{", src)
    assert m, name
    i, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[m.start():i]


HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#if !defined(_M_ARM64EC) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#include <emmintrin.h>
typedef uint32_t u32;
#define AUDIO_MIX_SSE2 1
HELPERS
static void original(float* dst, const float* src, u32 frames, float level)
{
    for (u32 s=0;s<frames;++s) {
        float g=level;
        if (!isfinite(g) || g < 0.0f) g=0.0f;
        if (g > 1.0f) g=1.0f;
        float left=ld_be_f32(&src[s*2])*g;
        float right=ld_be_f32(&src[s*2+1])*g;
        dst[s*2]+=left; dst[s*2+1]+=right;
    }
}
static void store_be(float* ptr,u32 u)
{
    unsigned char* b=(unsigned char*)ptr;
    b[0]=(unsigned char)(u>>24); b[1]=(unsigned char)(u>>16);
    b[2]=(unsigned char)(u>>8); b[3]=(unsigned char)u;
}
static u32 bits(float x) { u32 u; memcpy(&u,&x,4); return u; }
int main(void)
{
    const float levels[]={0.0f,-0.0f,0.125f,0.5f,1.0f,2.0f,-1.0f,NAN,INFINITY};
    const u32 samples[]={0,0x80000000u,0x3f000000u,0xbf000000u,0x3f800000u,
        0x00800000u,0x00000001u,0x7f7fffffu,0xff7fffffu,0x7fc00042u,0x7f800000u,0xff800000u};
    float src[512],want[512],got[512];
    for (u32 k=0;k<sizeof(levels)/sizeof(levels[0]);++k) {
        for (u32 frames=1;frames<=256;frames=frames<4?frames+1:frames*2) {
            for (u32 i=0;i<512;++i) {
                store_be(&src[i],samples[i%(sizeof(samples)/sizeof(samples[0]))]);
                want[i]=got[i]=(float)((int)(i%17)-8)/8.0f;
            }
            original(want,src,frames,levels[k]);
            audio_mix_stereo_constant(got,src,frames,levels[k]);
            for (u32 i=0;i<512;++i) {
                if (isnan(want[i])) assert(isnan(got[i]));
                else assert(bits(want[i])==bits(got[i]));
            }
        }
    }
    puts("PASS: native stereo helper vs original loop (non-NaNs bit exact, NaN classification)");
    return 0;
}
#else
int main(void) { puts("SKIP: SSE2 unavailable"); return 0; }
#endif
'''


def main():
    cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        if os.environ.get("CI"):
            raise SystemExit("FAIL: no C compiler")
        print("SKIP: no C compiler")
        return
    source = (ROOT / "libs/audio/cellAudio.c").read_text()
    helpers = "\n".join(extract(source, f) for f in ("ld_be_f32", "audio_mix_stereo_constant"))
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "stereo.c"
        exe = Path(tmp) / ("stereo.exe" if os.name == "nt" else "stereo")
        c.write_text(HARNESS.replace("HELPERS", helpers))
        subprocess.run([cc, "-std=c11", "-O2", "-UNDEBUG", "-ffp-contract=off",
                        str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == "__main__":
    main()
