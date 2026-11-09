#ifndef RSX_D3D12_RETIRE_H
#define RSX_D3D12_RETIRE_H
#include <stdint.h>
#include <stdlib.h>
#include <d3d12.h>
typedef struct D3D12RetiredResource {struct D3D12RetiredResource* next; ID3D12Resource* resource; uint64_t fence;} D3D12RetiredResource;
static void rsx_d3d12_retire_collect(D3D12RetiredResource** list, uint64_t completed) {
 while(*list) {D3D12RetiredResource* n=*list;if(n->fence>completed){list=&n->next;continue;}*list=n->next;n->resource->lpVtbl->Release(n->resource);free(n);}
}
static int rsx_d3d12_retire_add(D3D12RetiredResource** list, ID3D12Resource** resource,uint64_t fence) {
 if(!*resource)return 1;D3D12RetiredResource* n=malloc(sizeof(*n));if(!n)return 0;
 n->resource=*resource;n->fence=fence;n->next=*list;*list=n;*resource=NULL;return 1;
}
#endif
