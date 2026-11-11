/*
 * ps3recomp - Memory management syscalls
 */

#ifndef SYS_MEMORY_H
#define SYS_MEMORY_H

#include "lv2_syscall_table.h"
#include "../ppu/ppu_context.h"
#include "../../include/ps3emu/ps3types.h"
#include "../../include/ps3emu/error_codes.h"
#include "../memory/vm.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Page sizes */
#define SYS_MEMORY_PAGE_SIZE_1M   0x400
#define SYS_MEMORY_PAGE_SIZE_64K  0x200

/* Maximum tracked allocations */
#define SYS_MEMORY_ALLOC_MAX      1024
#define SYS_MEMORY_CONTAINER_MAX  16
#define SYS_MMAPPER_SHARED_MAX    64

/* Guest EA ranges used by the flat-memory allocator. Projects may override
 * the primary window to match the address range their title validates. */
#ifndef SYS_MEM_ALLOC_BASE
#ifdef PS3RECOMP_SYS_MEM_ALLOC_BASE
#define SYS_MEM_ALLOC_BASE          PS3RECOMP_SYS_MEM_ALLOC_BASE
#else
#define SYS_MEM_ALLOC_BASE          0x40000000u
#endif
#endif
#ifndef SYS_MEM_ALLOC_END
#ifdef PS3RECOMP_SYS_MEM_ALLOC_END
#define SYS_MEM_ALLOC_END           PS3RECOMP_SYS_MEM_ALLOC_END
#else
#define SYS_MEM_ALLOC_END           0x50000000u
#endif
#endif
/* The spill arena stays after sys_vm and stops before RSX local memory. */
#define SYS_MEM_ALLOC_SPILL_BASE    0x70000000u
#define SYS_MEM_ALLOC_SPILL_END     0xC0000000u

#define SYS_MEM_ALLOC_KIND_NONE            0u
#define SYS_MEM_ALLOC_KIND_MEMORY          1u
#define SYS_MEM_ALLOC_KIND_MAPPER_ADDRESS  2u
#define SYS_MEM_ALLOC_KIND_MAPPER_MEMORY  3u

/* Allocation record */
typedef struct sys_mem_alloc_info {
    int      active;
    uint32_t addr;
    uint32_t size;
    int32_t  container_id;  /* 0 = main pool */
    uint32_t page_size;     /* 0x100000 (1M) or 0x10000 (64K) from the alloc flags */
    uint32_t kind;          /* SYS_MEM_ALLOC_KIND_* */
} sys_mem_alloc_info;

/* Container */
typedef struct sys_mem_container_info {
    int      active;
    uint32_t total_size;
    uint32_t used_size;
} sys_mem_container_info;

/* Shared memory object */
typedef struct sys_mmapper_shared_info {
    int      active;
    uint32_t size;
    uint32_t addr;   /* mapped address, 0 if not mapped */
    uint64_t key;
} sys_mmapper_shared_info;

extern sys_mem_alloc_info       g_sys_mem_allocs[SYS_MEMORY_ALLOC_MAX];
extern sys_mem_container_info   g_sys_mem_containers[SYS_MEMORY_CONTAINER_MAX];
extern sys_mmapper_shared_info  g_sys_mmapper_shared[SYS_MMAPPER_SHARED_MAX];
extern uint32_t                 g_sys_mem_bump_ptr;  /* high-water diagnostic */

/* True when the complete guest range belongs to a currently active allocation. */
int sys_memory_contains(uint32_t addr, uint32_t size);

/* Internal HLE scratch allocations in the guest address space. These use the
 * same tracked allocator as sys_memory_allocate but preserve the requested
 * small alignment/size, for callback-scoped guest payloads. */
int32_t ps3_guest_memory_alloc(uint32_t size, uint32_t alignment,
                               uint32_t* out_addr);
int32_t ps3_guest_memory_alloc_spill(uint32_t size, uint32_t alignment,
                                     uint32_t* out_addr);
int32_t ps3_guest_memory_free(uint32_t addr);

/* Syscall handlers */
int64_t sys_memory_allocate(ppu_context* ctx);
int64_t sys_memory_free(ppu_context* ctx);
int64_t sys_memory_get_user_memory_size(ppu_context* ctx);
int64_t sys_memory_get_page_attribute(ppu_context* ctx);
int64_t sys_memory_container_create(ppu_context* ctx);
int64_t sys_memory_container_destroy(ppu_context* ctx);
int64_t sys_memory_container_get_size(ppu_context* ctx);
int64_t sys_memory_allocate_from_container(ppu_context* ctx);
int64_t sys_mmapper_allocate_address(ppu_context* ctx);
int64_t sys_mmapper_search_map_address(uint32_t start, uint32_t size, uint32_t* out);
int64_t sys_mmapper_free_address(ppu_context* ctx);
int64_t sys_mmapper_allocate_shared_memory(ppu_context* ctx);
int64_t sys_mmapper_map_shared_memory(ppu_context* ctx);

/* Optional loader-owned image query: return addr unchanged when [addr, end)
 * is clear, otherwise the greatest exclusive end of an overlapping PT_LOAD.
 * Install before guest threads start. The callback and its ranges must outlive
 * allocations; sys_memory_init preserves it because ELF loading precedes LV2
 * initialization in the boot harness. NULL restores the no-loader behavior. */
void sys_memory_set_image_query(uint64_t (*query)(uint64_t addr, uint64_t end));

/* Registration */
void sys_memory_init(lv2_syscall_table* tbl);

#ifdef __cplusplus
}
#endif

#endif /* SYS_MEMORY_H */
