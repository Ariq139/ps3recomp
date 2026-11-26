#!/usr/bin/env python3
"""Extract actual loader TLS allocator and check concurrent identity + churn reuse."""
import os, re, subprocess, tempfile
from pathlib import Path
if os.name == 'nt':
    print('SKIP: standalone POSIX compiler fixture')
    raise SystemExit(0)
root=Path(__file__).resolve().parents[1]
src=(root/'runtime/ppu/ppu_loader.cpp').read_text()
begin=src.index('static std::atomic<uint64_t> s_tls_slots')
end=src.index('static void ppu_thread_entry_trampoline',begin)
tls_img=re.search(r'^#define VM_PPU_TLS_BASE\s+(\S+)',(root/'runtime/memory/vm.h').read_text(),re.M).group(1)
pre=r'''
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cassert>
#define PPU_TLS_IMG %s
static uint8_t* vm_base;
static uint32_t ppu_vm_size=0x20000000u;
static uint32_t g_tls_vaddr=0x1000,g_tls_filesz=0x100,g_tls_memsz=0x240;
''' % tls_img
test=r'''
int main() {
    vm_base=(uint8_t*)calloc(ppu_vm_size,1); assert(vm_base);
    memset(vm_base+g_tls_vaddr,0x5a,g_tls_filesz);
    std::atomic<unsigned> ready{0}; std::atomic<bool> release{false};
    uint32_t blocks[100]={}; std::vector<std::thread> workers;
    for (unsigned i=0;i<100;i++) workers.emplace_back([&,i] {
        uint32_t tp=ppu_tls_new_block(); assert(tp);
        blocks[i]=tp-0x7000u; assert(ppu_tls_new_block()==tp);
        assert(!memcmp(vm_base+blocks[i],vm_base+g_tls_vaddr,g_tls_filesz));
        ready.fetch_add(1);
        while (!release.load()) std::this_thread::yield();
        memset(vm_base+blocks[i],0xa5,g_tls_memsz);
    });
    while (ready.load()!=100) std::this_thread::yield();
    for (unsigned i=0;i<100;i++) for (unsigned j=0;j<i;j++) assert(blocks[i]!=blocks[j]);
    release.store(true); for(auto& t:workers)t.join();
    for(unsigned i=0;i<600;i++) {
        std::thread t([] { uint32_t tp=ppu_tls_new_block(); assert(tp);
            uint32_t b=tp-0x7000u; assert(!memcmp(vm_base+b,vm_base+g_tls_vaddr,g_tls_filesz));
            for(unsigned n=g_tls_filesz;n<g_tls_memsz;n++)assert(vm_base[b+n]==0);
            memset(vm_base+b,0xbb,g_tls_memsz);
        }); t.join();
    }
    puts("PASS concurrent TLS isolation and reuse after 600 thread exits");
}
'''
with tempfile.TemporaryDirectory() as d:
    c=Path(d)/'tls.cpp'; exe=Path(d)/'tls'
    c.write_text(pre+src[begin:end]+test)
    subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O1','-pthread','-UNDEBUG',str(c),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=60)
