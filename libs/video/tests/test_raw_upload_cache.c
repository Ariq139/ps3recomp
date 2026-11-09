#include "rsx_raw_upload_cache.h"
#include <assert.h>
#include <stdio.h>
static u32 resolve(int local, u32 offset) { (void)local; return offset; }
int main(void)
{
    rsx_raw_upload_cache cache = {0};
    rsx_state state = {0};
    u8 guest[16384], output[16384], expected[16384];
    for (u32 i=0;i<sizeof(guest);++i) guest[i]=(u8)(i*37u);
    state.vertex_attribs[0].enabled=1;
    state.vertex_attribs[0].type=2; state.vertex_attribs[0].size=4;
    state.vertex_attribs[0].stride=16; state.vertex_attribs[0].offset=1024;
    u32 vertices[]={4,5,6,7}, indices[]={0,1,2,0,2,3}, offset=0;
    rsx_raw_vertex_plan plan;
    assert(rsx_raw_vertex_prepare(&state,1,vertices,4,sizeof(output)-24,resolve,&plan));
    assert(!rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,32,output,&offset));
    assert(offset==32);
    rsx_raw_vertex_write(&plan,vertices,guest,expected);
    memcpy(expected+plan.bytes,indices,sizeof(indices));
    assert(!memcmp(output,expected,plan.bytes+sizeof(indices)));
    memset(output,0xCA,sizeof(output));
    assert(rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,4096,output,&offset));
    assert(offset==32 && output[0]==0xCA);
    for (u32 byte=0;byte<plan.ranges[0].size;++byte) {
        guest[plan.ranges[0].source+byte]^=0xFF;
        assert(!rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,64,output,&offset));
        assert(offset==64);
        assert(rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,128,output,&offset));
    }
    vertices[1]=6;
    assert(!rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,128,output,&offset));
    indices[2]=3;
    assert(!rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,256,output,&offset));
    plan.attributes[3][8]^=0x100;
    assert(!rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,512,output,&offset));
    assert(rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,1024,output,&offset));
    rsx_raw_upload_reset(&cache);
    assert(!rsx_raw_upload_reuse(&cache,&plan,vertices,indices,6,guest,1024,output,&offset));
    assert(offset==1024);
    rsx_raw_upload_destroy(&cache);
    const u32 primitives[]={RSX_PRIMITIVE_TRIANGLES,RSX_PRIMITIVE_QUADS,
        RSX_PRIMITIVE_TRIANGLE_STRIP,RSX_PRIMITIVE_TRIANGLE_FAN};
    for(u32 p=0;p<4;++p) for(u32 flat=0;flat<2;++flat) {
        const u32 count=p==0?6:8;
        const u32 n=rsx_raw_vertex_count(primitives[p],count);
        u32 expanded[32],dense[32];
        u32 lo=1000,hi=0;
        for(u32 i=0;i<n;++i) {
            expanded[i]=17+rsx_raw_vertex_source(primitives[p],i,flat)%5;
            if(expanded[i]<lo)lo=expanded[i]; if(expanded[i]>hi)hi=expanded[i];
        }
        u32 original[32]; memcpy(original,expanded,n*4u);
        assert(rsx_raw_vertex_indexed_prepare(&state,1,expanded,n,dense,sizeof(output),resolve,&plan));
        assert(plan.header[1]==hi-lo+1u);
        for(u32 i=0;i<n;++i) assert(dense[expanded[i]]==original[i]);
        memcpy(expanded,original,n*4u);
        assert(!rsx_raw_vertex_indexed_prepare(&state,1,expanded,n,dense,32,resolve,&plan));
        assert(!memcmp(expanded,original,n*4u));
    }
    puts("raw upload reuse: byte mutations, IDs, indices, constants, batch reset and topology remaps passed");
}
