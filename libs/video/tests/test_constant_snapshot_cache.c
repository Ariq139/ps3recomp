#include "rsx_constant_snapshot_cache.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
int main(void) {
 rsx_constant_snapshot_cache c={0};unsigned char data[RSX_CONSTANT_SNAPSHOT_BYTES]={0};
 unsigned char* gpu=malloc(4*RSX_CONSTANT_SNAPSHOT_BYTES);assert(gpu);memset(gpu,0xA5,4*RSX_CONSTANT_SNAPSHOT_BYTES);
 assert(rsx_constant_snapshot_record(&c,data,0,gpu)==0);
 assert(rsx_constant_snapshot_record(&c,data,1,gpu)==0);
 for(unsigned i=0;i<RSX_CONSTANT_SNAPSHOT_BYTES;++i)assert(gpu[RSX_CONSTANT_SNAPSHOT_BYTES+i]==0xA5);
 data[8192]=1;assert(rsx_constant_snapshot_record(&c,data,2,gpu)==2);
 assert(gpu[8192]==0 && gpu[2*RSX_CONSTANT_SNAPSHOT_BYTES+8192]==1);
 data[8224]=2;assert(rsx_constant_snapshot_record(&c,data,3,gpu)==3);
 c.valid=0;assert(rsx_constant_snapshot_record(&c,data,0,gpu)==0);
 assert(!memcmp(gpu,data,sizeof(data)));free(gpu);
 puts("constant snapshots: reuse, no redundant upload, immutable previous draws, viewport, branch bits and batch reset passed");
}
