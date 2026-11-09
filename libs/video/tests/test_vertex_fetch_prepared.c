#include "rsx_vertex_fetch.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
u8* vm_base;
static unsigned resolves;
static u32 main_shift;
u32 cellGcmResolveLocated(int local, u32 off) { ++resolves; return off + (local ? 0x100000u : main_shift + ((off >> 20) & 1u) * 0x100000u); }
u32 cellGcmResolveOffset(u32 off) { return off; }
int main(void) {
 vm_base = malloc(8u << 20); assert(vm_base);
 for (u32 i=0;i<(8u<<20);++i) vm_base[i]=(u8)(i*37u+(i>>8));
 rsx_state* state=calloc(1,sizeof(*state)); assert(state);
 for (unsigned draw=0;draw<32;++draw) {
  state->vertex_data_base_offset=draw*16;
  state->frequency_divider_op=draw*719u;
  main_shift=draw&1 ? 0x300000u : 0x400000u;
  for (unsigned a=0;a<16;++a) {
   state->vertex_attribs[a].enabled=(draw+a)%5 != 0;
   state->vertex_attribs[a].offset=(a&1 ? 0x80000000u:0) + 0xFFF00u + a*32;
   state->vertex_attribs[a].type=1+(draw+a)%7;
   state->vertex_attribs[a].size=1+(draw+a)%4;
   state->vertex_attribs[a].frequency=(draw+a)%5;
   state->vertex_attribs[a].stride=(a%4)*16;
   for(unsigned c=0;c<4;++c) state->vertex_data4f[a][c]=(float)(a+c);
  }
  rsx_vertex_fetch_prepared plan; rsx_vertex_fetch_prepare(&plan,state);
  for(unsigned v=0;v<128;++v) for(unsigned a=0;a<16;++a) {
   float expected[4], actual[4];
   rsx_fetch_attrib(state,a,v,expected);
   unsigned before=resolves;
   rsx_fetch_attrib_prepared(&plan,a,v,actual);
   assert(!memcmp(expected,actual,sizeof(actual)));
   if (!(a&1)) assert(resolves==before);
  }
 }
 free(state);free(vm_base);
 puts("prepared vertex fetch: formats, divisors, constants, page crossings and mapping changes passed");
}
