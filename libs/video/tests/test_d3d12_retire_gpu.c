#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "rsx_d3d12_retire.h"
#include "rsx_d3d12_gpu_timer.h"
static ID3D12Resource* buffer(ID3D12Device* d,D3D12_HEAP_TYPE type,D3D12_RESOURCE_STATES state) {
 D3D12_HEAP_PROPERTIES hp={0};hp.Type=type;
 D3D12_RESOURCE_DESC rd={0};rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;rd.Width=65536;rd.Height=1;rd.DepthOrArraySize=1;rd.MipLevels=1;rd.SampleDesc.Count=1;rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
 ID3D12Resource* r=NULL;assert(SUCCEEDED(d->lpVtbl->CreateCommittedResource(d,&hp,0,&rd,state,NULL,&IID_ID3D12Resource,(void**)&r)));return r;
}
int main(void) {
 ID3D12Debug* debug=NULL;assert(SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug,(void**)&debug)));debug->lpVtbl->EnableDebugLayer(debug);
 IDXGIFactory4* factory=NULL;IDXGIAdapter* adapter=NULL;ID3D12Device* device=NULL;
 assert(SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory4,(void**)&factory)));
 assert(SUCCEEDED(factory->lpVtbl->EnumWarpAdapter(factory,&IID_IDXGIAdapter,(void**)&adapter)));
 assert(SUCCEEDED(D3D12CreateDevice((IUnknown*)adapter,D3D_FEATURE_LEVEL_11_0,&IID_ID3D12Device,(void**)&device)));
 ID3D12InfoQueue* info=NULL;assert(SUCCEEDED(device->lpVtbl->QueryInterface(device,&IID_ID3D12InfoQueue,(void**)&info)));
 ID3D12CommandQueue* queue=NULL;D3D12_COMMAND_QUEUE_DESC qd={0};assert(SUCCEEDED(device->lpVtbl->CreateCommandQueue(device,&qd,&IID_ID3D12CommandQueue,(void**)&queue)));
 ID3D12CommandAllocator* allocator=NULL;assert(SUCCEEDED(device->lpVtbl->CreateCommandAllocator(device,0,&IID_ID3D12CommandAllocator,(void**)&allocator)));
 ID3D12GraphicsCommandList* list=NULL;assert(SUCCEEDED(device->lpVtbl->CreateCommandList(device,0,0,allocator,NULL,&IID_ID3D12GraphicsCommandList,(void**)&list)));
 ID3D12Fence* fence=NULL;assert(SUCCEEDED(device->lpVtbl->CreateFence(device,0,0,&IID_ID3D12Fence,(void**)&fence)));
 ID3D12Resource* old=buffer(device,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
 ID3D12Resource* later=buffer(device,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
 ID3D12Resource* readback=buffer(device,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
 void* p=NULL;assert(SUCCEEDED(old->lpVtbl->Map(old,0,NULL,&p)));memset(p,0xA7,65536);old->lpVtbl->Unmap(old,0,NULL);
 rsx_d3d12_gpu_timer timer={0};assert(rsx_gpu_timer_init(&timer,device,queue));
 assert(rsx_gpu_timer_begin(&timer,fence,list,0));
 list->lpVtbl->CopyBufferRegion(list,readback,0,old,0,65536);
 rsx_gpu_timer_end(&timer,list,0);timer.pending[0]=1;
 D3D12RetiredResource* retired=NULL;
 assert(rsx_d3d12_retire_add(&retired,&old,1) && !old);
 assert(rsx_d3d12_retire_add(&retired,&later,2) && !later);
 rsx_d3d12_retire_collect(&retired,0);assert(retired && retired->next);
 assert(SUCCEEDED(list->lpVtbl->Close(list)));ID3D12CommandList* lists[]={(ID3D12CommandList*)list};queue->lpVtbl->ExecuteCommandLists(queue,1,lists);
 HANDLE event=CreateEvent(NULL,FALSE,FALSE,NULL);assert(event);
 assert(SUCCEEDED(queue->lpVtbl->Signal(queue,fence,1)));assert(SUCCEEDED(fence->lpVtbl->SetEventOnCompletion(fence,1,event)));assert(WaitForSingleObject(event,10000)==WAIT_OBJECT_0);
 rsx_d3d12_retire_collect(&retired,1);assert(retired && retired->fence==2 && !retired->next);
 assert(SUCCEEDED(readback->lpVtbl->Map(readback,0,NULL,&p)));for(unsigned i=0;i<65536;++i)assert(((unsigned char*)p)[i]==0xA7);readback->lpVtbl->Unmap(readback,0,NULL);
 assert(SUCCEEDED(allocator->lpVtbl->Reset(allocator)));
 assert(SUCCEEDED(list->lpVtbl->Reset(list,allocator,NULL)));
 assert(rsx_gpu_timer_begin(&timer,fence,list,0));assert(timer.samples==1 && timer.total_ms>=0);
 rsx_gpu_timer_end(&timer,list,0);
 assert(SUCCEEDED(list->lpVtbl->Close(list)));queue->lpVtbl->ExecuteCommandLists(queue,1,lists);
 assert(SUCCEEDED(queue->lpVtbl->Signal(queue,fence,2)));assert(SUCCEEDED(fence->lpVtbl->SetEventOnCompletion(fence,2,event)));assert(WaitForSingleObject(event,10000)==WAIT_OBJECT_0);
 rsx_d3d12_retire_collect(&retired,2);assert(!retired);
 unsigned errors=0;for(UINT64 i=0;i<info->lpVtbl->GetNumStoredMessagesAllowedByRetrievalFilter(info);++i) {SIZE_T bytes=0;info->lpVtbl->GetMessage(info,i,NULL,&bytes);D3D12_MESSAGE* m=malloc(bytes);assert(SUCCEEDED(info->lpVtbl->GetMessage(info,i,m,&bytes)));if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)++errors;free(m);}assert(errors==0);
 rsx_gpu_timer_destroy(&timer);
 CloseHandle(event);readback->lpVtbl->Release(readback);fence->lpVtbl->Release(fence);list->lpVtbl->Release(list);allocator->lpVtbl->Release(allocator);queue->lpVtbl->Release(queue);info->lpVtbl->Release(info);device->lpVtbl->Release(device);adapter->lpVtbl->Release(adapter);factory->lpVtbl->Release(factory);debug->lpVtbl->Release(debug);
 puts("WARP retirement: open-list lifetime, fence ordering, copied bytes, GPU timestamps and zero debug-layer errors passed");
}
