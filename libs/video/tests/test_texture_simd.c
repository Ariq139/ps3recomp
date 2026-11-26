/* Standalone differential test for the exact production SSE2 helpers. */
#include "rsx_texture_simd.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#if RSX_TEXTURE_SSE2
static void check(int ok) { if (!ok) abort(); }
static void scalar(uint8_t out[4], unsigned v, unsigned fmt)
{
    if (fmt == 0x83u) {
        out[0]=((v>>8)&15)*17; out[1]=((v>>4)&15)*17;
        out[2]=(v&15)*17; out[3]=(v>>12)*17;
    } else if (fmt == 0x84u) {
        out[0]=(v>>11)*255/31; out[1]=((v>>5)&63)*255/63;
        out[2]=(v&31)*255/31; out[3]=255;
    } else if (fmt == 0x97u) {
        out[0]=(v>>11)*255/31; out[1]=((v>>6)&31)*255/31;
        out[2]=((v>>1)&31)*255/31; out[3]=(v&1)?255:0;
    } else {
        out[0]=((v>>10)&31)*255/31; out[1]=((v>>5)&31)*255/31;
        out[2]=(v&31)*255/31; out[3]=(fmt==0x9D || (v&0x8000))?255:0;
    }
}
int main(void)
{
    const unsigned formats[]={0x82,0x83,0x84,0x97,0x9D};
    uint8_t input[18], output[34], expected[4];
    for (unsigned f=0; f<sizeof(formats)/sizeof(formats[0]); ++f) {
        for (unsigned base=0; base<65536u; base+=8) {
            for (unsigned i=0;i<8;++i) {
                input[1+i*2]=(uint8_t)((base+i)>>8);
                input[2+i*2]=(uint8_t)(base+i);
            }
            memset(output,0xCD,sizeof(output));
            rsx_decode_packed16x8(output+1,input+1,formats[f]);
            for(unsigned i=0;i<8;++i) {
                scalar(expected,base+i,formats[f]);
                check(!memcmp(expected,output+1+i*4,4));
            }
            check(output[0]==0xCD && output[33]==0xCD);
        }
    }
    unsigned rng=0x12345678;
    for(unsigned trial=0;trial<65536;++trial) {
        for(unsigned i=0;i<16;++i) {
            rng=rng*1664525u+1013904223u; input[1+i]=(uint8_t)(rng>>24);
        }
        for(int rgba=0;rgba<2;++rgba) for(int opaque=0;opaque<2;++opaque) {
            memset(output,0xCD,sizeof(output));
            rsx_decode_argb_tile2x2(output+1,output+19,input+1,rgba,opaque);
            for(unsigned i=0;i<4;++i) {
                for(unsigned j=0;j<4;++j)
                    expected[j]=opaque && j==3?255:input[1+i*4+((j+!rgba)&3)];
                check(!memcmp(expected,output+(i<2?1:19)+(i&1)*4,4));
            }
            for(unsigned i=0;i<sizeof(output);++i)
                if(i==0 || (i>=9 && i<19) || i>=27) check(output[i]==0xCD);
        }
    }
    puts("PASS: 327680 packed texels and 262144 Morton tiles; exact scalar bytes and padding");
    return 0;
}
#else
int main(void) { puts("SKIP: SSE2 unavailable"); return 77; }
#endif
