#ifndef RSX_D3D12_GPU_TIMER_H
#define RSX_D3D12_GPU_TIMER_H
#include <d3d12.h>
#include <stdint.h>
#include <string.h>
typedef struct {
 ID3D12QueryHeap* queries;
 ID3D12Resource* readback;
 uint64_t frequency, pending[2];
 int attempted, active;
 double total_ms, max_ms;
 unsigned samples;
} rsx_d3d12_gpu_timer;
static void rsx_gpu_timer_destroy(rsx_d3d12_gpu_timer* t) {
 if(t->queries)t->queries->lpVtbl->Release(t->queries);
 if(t->readback)t->readback->lpVtbl->Release(t->readback);
 memset(t,0,sizeof(*t));
}
static int rsx_gpu_timer_init(rsx_d3d12_gpu_timer* t, ID3D12Device* device, ID3D12CommandQueue* queue) {
 if(t->attempted)return t->queries && t->readback;
 t->attempted=1;
 if(FAILED(queue->lpVtbl->GetTimestampFrequency(queue,&t->frequency)) || !t->frequency)return 0;
 D3D12_QUERY_HEAP_DESC q={0};q.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;q.Count=4;
 if(FAILED(device->lpVtbl->CreateQueryHeap(device,&q,&IID_ID3D12QueryHeap,(void**)&t->queries)))return 0;
 D3D12_HEAP_PROPERTIES hp={0};hp.Type=D3D12_HEAP_TYPE_READBACK;
 D3D12_RESOURCE_DESC rd={0};rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;rd.Width=32;rd.Height=1;rd.DepthOrArraySize=1;rd.MipLevels=1;rd.SampleDesc.Count=1;rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
 if(FAILED(device->lpVtbl->CreateCommittedResource(device,&hp,0,&rd,D3D12_RESOURCE_STATE_COPY_DEST,NULL,&IID_ID3D12Resource,(void**)&t->readback)))return 0;
 return 1;
}
static int rsx_gpu_timer_begin(rsx_d3d12_gpu_timer* t,ID3D12Fence* fence, ID3D12GraphicsCommandList* list,unsigned slot) {
 t->active=0;
 if(slot>=2 || !t->queries || !t->readback)return 0;
 if(t->pending[slot]) {
  const uint64_t done=fence->lpVtbl->GetCompletedValue(fence);
  if(done==UINT64_MAX || done<t->pending[slot])return 0;
  D3D12_RANGE range={slot*16u,slot*16u+16u};uint64_t* data=NULL;
  if(SUCCEEDED(t->readback->lpVtbl->Map(t->readback,0,&range,(void**)&data))) {
   uint64_t first=data[slot*2],last=data[slot*2+1];
   if(last>=first) {double ms=(double)(last-first)*1000.0/(double)t->frequency;t->total_ms+=ms;if(ms>t->max_ms)t->max_ms=ms;++t->samples;}
   D3D12_RANGE written={0,0};t->readback->lpVtbl->Unmap(t->readback,0,&written);
  }
  t->pending[slot]=0;
 }
 list->lpVtbl->EndQuery(list,t->queries,D3D12_QUERY_TYPE_TIMESTAMP,slot*2);
 t->active=1;return 1;
}
static void rsx_gpu_timer_end(rsx_d3d12_gpu_timer* t,ID3D12GraphicsCommandList* list,unsigned slot) {
 if(!t->active)return;
 list->lpVtbl->EndQuery(list,t->queries,D3D12_QUERY_TYPE_TIMESTAMP,slot*2+1);
 list->lpVtbl->ResolveQueryData(list,t->queries,D3D12_QUERY_TYPE_TIMESTAMP,slot*2,2,t->readback,slot*16u);
}
#endif
