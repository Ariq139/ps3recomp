/*
 * ps3recomp - cellFsAioRead / cellFsAioWrite on a shared fd (ppu_fs.cpp).
 *
 * An AIO read names its own offset. It must not move the fd's position for
 * the plain cellFsRead that follows, and several guest threads (FIOS media
 * threads, audio streamers) issue them on one fd at once. The fixture writes
 * a file whose every 4-byte word holds its own offset, then checks both:
 * a cellFsRead after an AIO read continues from the fd's position, and
 * reads racing from four threads each get the words they asked for. First
 * of all, an AIO write lands at its own offset, and reads and writes draw
 * their ids from one counter: with one each, a read and a write shared one.
 *
 *   tools/test_ppu_fs_aio.py
 *   test_fs_aio [DIR]     (DIR: where the file goes; default the temp dir)
 */
#include "ppu_fs.cpp"
#include <thread>
#include <vector>
#include <chrono>
#ifndef _WIN32
#include <signal.h>
#include <sys/mman.h>
#endif

/* ---- link stubs: nothing on the paths this test drives calls them ---- */
extern "C" {
uint8_t* vm_base = nullptr;
uint32_t ppu_vm_size = 0;
void vm_write32(uint64_t a, uint32_t v) { uint32_t be = __builtin_bswap32(v); memcpy(vm_base + a, &be, 4); }
void vm_write64(uint64_t a, uint64_t v) { uint64_t be = __builtin_bswap64(v); memcpy(vm_base + a, &be, 8); }
void ppu_guest_caller(char* out, size_t n) { if (n) out[0] = 0; }
void ps3_vfs_ps3game_fallback(char* path, size_t cap) { (void)path; (void)cap; }
void ydkj_host_bt(const char* tag) { (void)tag; }
void ps3_hle_register_ctx(uint32_t, const char*, void (*)(ppu_context*)) {}
}
/* ps3_dev_flash_path (runtime/platform), which host_path calls for /dev_flash: the real one, it needs nothing. */
#include "../../platform/ps3_dev_flash.c"
ps3_guest_caller_fn g_ps3_guest_caller = nullptr;
const char* edat_resolve(const char* host_path, char* buf, size_t cap) { (void)buf; (void)cap; return host_path; }

static uint32_t rd32(uint32_t a) { uint32_t v; memcpy(&v, vm_base + a, 4); return __builtin_bswap32(v); }
static uint64_t rd64(uint32_t a) { uint64_t v; memcpy(&v, vm_base + a, 8); return __builtin_bswap64(v); }
static void wr32(uint32_t a, uint32_t v) { vm_write32(a, v); }
static void wr64(uint32_t a, uint64_t v) { vm_write64(a, v); }

static int fails = 0;
#define CHECK(c, ...) do { printf((c) ? "[PASS] " : "[FAIL] "); printf(__VA_ARGS__); printf("\n"); \
                           if (!(c)) fails++; } while (0)

static const uint32_t WORDS = 1u << 16;   /* 256 KiB file */

/* One AIO read (or write) of `len` bytes at `off` into (from) guest `buf`,
 * through a CellFsAio at guest `aio`: {fd u32, pad, offset u64, buf u32, pad,
 * size u64}. The request's id goes to guest `id_ptr` when it is not 0. */
static void aio_submit(void (*op)(ppu_context*), uint32_t aio, uint32_t fd, uint64_t off, uint32_t buf,
                       uint64_t len, uint32_t id_ptr)
{
    wr32(aio + 0x00, fd); wr64(aio + 0x08, off); wr32(aio + 0x10, buf); wr64(aio + 0x18, len);
    ppu_context ctx; memset(&ctx, 0, sizeof ctx);
    ctx.gpr[3] = aio; ctx.gpr[4] = id_ptr; ctx.gpr[5] = 0;
    op(&ctx);
}
static void aio_read(uint32_t aio, uint32_t fd, uint64_t off, uint32_t buf, uint64_t len, uint32_t id_ptr = 0)
{
    aio_submit(cellFsAioRead, aio, fd, off, buf, len, id_ptr);
}
static void aio_write(uint32_t aio, uint32_t fd, uint64_t off, uint32_t buf, uint64_t len, uint32_t id_ptr = 0)
{
    aio_submit(cellFsAioWrite, aio, fd, off, buf, len, id_ptr);
}

/* Every word at file offset o holds o (big-endian), so a buffer read from
 * offset `off` is right iff word i holds off + 4*i. */
static bool holds(uint32_t buf, uint64_t off, uint32_t len)
{
    for (uint32_t i = 0; i < len; i += 4) if (rd32(buf + i) != (uint32_t)(off + i)) return false;
    return true;
}

#ifndef _WIN32
/* Stand-in for the VEH that commits reserved guest pages on first touch: the
 * guest region is PROT_NONE, a user-mode fault commits the page, but a kernel
 * access (write(2) from a stdio flush) just fails with EFAULT. */
static uint8_t* g_res; static size_t g_res_len;
static void commit_page(int, siginfo_t* si, void*)
{
    /* The host's page size: Apple Silicon's is 16 KB, and mprotect refuses an
     * address that is only 4 KB aligned, so the fault would repeat forever. */
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)si->si_addr & ~(page - 1);
    if (a < (uintptr_t)g_res || a >= (uintptr_t)g_res + g_res_len) _exit(3);
    mprotect((void*)a, page, PROT_READ | PROT_WRITE);
}

/* cellFsAioWrite must pre-fault its source like cellFsWrite and cellFsAioRead. */
static void test_aio_write_prefault(const char* dir)
{
    const uint32_t SZ = 0x10000;
    g_res_len = 1 << 20;
    g_res = (uint8_t*)mmap(nullptr, g_res_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = commit_page; sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);    /* what macOS raises on a PROT_NONE page */
    uint8_t* saved = vm_base; uint32_t saved_sz = ppu_vm_size;
    vm_base = g_res; ppu_vm_size = (uint32_t)g_res_len;

    /* path + out pointers live in pages the test commits itself */
    mprotect(g_res, 0x1000, PROT_READ | PROT_WRITE);
    strcpy((char*)g_res + 0x100, "/dev_hdd1/ps3recomp_fs_aiow.bin");
    char host[1100]; snprintf(host, sizeof host, "%s/ps3recomp_fs_aiow.bin", dir);
    ppu_context c; memset(&c, 0, sizeof c);
    c.gpr[3] = 0x100; c.gpr[4] = 2 /* RDWR */ | 0x40 /* CREAT */ | 0x200 /* TRUNC */; c.gpr[5] = 0x200;
    cellFsOpen(&c);
    const uint32_t fd = rd32(0x200);
    CHECK((uint32_t)c.gpr[3] == 0, "cellFsOpen for the write test (fd %u)", fd);
    if ((uint32_t)c.gpr[3] == 0) {
        /* source buffer is 16 untouched pages */
        wr32(0x300, fd); wr64(0x308, 0); wr32(0x310, 0x20000); wr64(0x318, SZ);
        memset(&c, 0, sizeof c); c.gpr[3] = 0x300;
        cellFsAioWrite(&c);
        memset(&c, 0, sizeof c); c.gpr[3] = fd; cellFsClose(&c);
        struct stat st; bool ok = stat(host, &st) == 0 && (uint64_t)st.st_size == SZ;
        CHECK(ok, "cellFsAioWrite from untouched guest pages writes all %u bytes (file is %lld)",
              SZ, (long long)(stat(host, &st) == 0 ? st.st_size : -1));
    }
    remove(host);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    vm_base = saved; ppu_vm_size = saved_sz;
    munmap(g_res, g_res_len);
}
#endif

/* Exercise the actual async manager with a controlled service hook. This
 * tests the queue and FILE lifetime without requiring a lifted callback OPD.
 * Native PPU context/TLS integration is a separate runtime validation. */
static std::thread aio_fixture_thread;
static std::mutex aio_fixture_lock;
static std::condition_variable aio_fixture_cv;
static bool aio_fixture_entered=false, aio_fixture_continue=false;
struct AioFixtureCompletion { int32_t id,error; uint64_t size; };
static std::vector<AioFixtureCompletion> aio_fixture_done;
static int32_t aio_fixture_start(ps3_ppu_service_fn entry, ps3_ppu_service_fn cleanup)
{
    aio_fixture_thread = std::thread([=] {
        ppu_context c{}; c.thread_id=42; c.gpr[1]=0x80000; c.gpr[13]=0x90000;
        entry(&c); cleanup(&c);
    });
    return CELL_OK;
}
static int aio_fixture_prepare(void*) { return 1; }
static int aio_fixture_resolve(uint32_t, uint32_t* code, uint32_t* toc) { *code=0x10000; *toc=0; return 1; }
static void aio_fixture_install_stop(void (*)(void)) {}
static int aio_fixture_deliver(void* opaque,uint64_t,uint64_t,uint32_t,uint32_t,
                               uint32_t,int32_t error,int32_t id,uint64_t size)
{
    auto* c=static_cast<ppu_context*>(opaque);
    if (c->thread_id != 42) return 0;
    std::unique_lock<std::mutex> hold(aio_fixture_lock);
    if (!aio_fixture_entered) {
        aio_fixture_entered=true; aio_fixture_cv.notify_all();
        aio_fixture_cv.wait(hold,[]{return aio_fixture_continue;});
    }
    aio_fixture_done.push_back({id,error,size}); aio_fixture_cv.notify_all();
    return 1;
}
static const ps3_aio_runtime_hooks aio_fixture_hooks = {
    aio_fixture_start,aio_fixture_prepare,aio_fixture_resolve,aio_fixture_deliver,aio_fixture_install_stop
};
static int32_t aio_fixture_submit(uint32_t aio,int fd,uint64_t offset,uint32_t buf,
                                  uint64_t bytes,uint32_t id_ptr)
{
    wr32(aio,fd); wr64(aio+8,offset); wr32(aio+0x10,buf); wr64(aio+0x18,bytes);
    ppu_context c{}; c.gpr[3]=aio;c.gpr[4]=id_ptr;c.gpr[5]=0x900;
    fs_aio_async_submit(&c,false);
    return (int32_t)c.gpr[3];
}
static void test_async_queue(int fd)
{
    auto* manager=fs_aio_manager_create();
    manager->hooks=&aio_fixture_hooks;
    CHECK(fs_aio_async_init("/dev_hdd1")==CELL_OK,"async mount initialized");
    CHECK(aio_fixture_submit(0x800,fd,0x1000,0x6000,16,0x290)==CELL_OK,"first request queued");
    {
        std::unique_lock<std::mutex> hold(aio_fixture_lock);
        if (!aio_fixture_cv.wait_for(hold,std::chrono::seconds(5),[]{return aio_fixture_entered;})) {
            CHECK(false,"service reached callback"); std::abort();
        }
    }
    CHECK(holds(0x6000,0x1000,16),"guest bytes published before callback");
    CHECK(aio_fixture_submit(0x840,fd,0x2000,0x6100,16,0x294)==CELL_OK,"second request queued during callback");
    CHECK(aio_fixture_submit(0x880,fd,0x3000,0x6200,16,0x298)==CELL_OK,"third request queued during callback");
    CHECK(fs_aio_async_cancel((int32_t)rd32(0x298))==CELL_OK,"queued cancellation accepted");
    CHECK(fs_aio_async_finish("/dev_hdd1")==(int32_t)CELL_EBUSY,"finish never waits for active callback");
    ppu_context close_ctx{}; close_ctx.gpr[3]=fd; cellFsClose(&close_ctx);
    // Reuse the fd with different contents: the queued request must still read
    // the old stream. The new file is deliberately not a valid AIO mount.
    FILE* replacement=tmpfile();
    if (!replacement) std::abort();
    int newfd;
    {std::lock_guard<std::mutex> hold(g_fs_mutex); newfd=fd_alloc_file(replacement);}
    CHECK(newfd==fd,"numeric descriptor reused while old stream pinned");
    {
        std::lock_guard<std::mutex> hold(aio_fixture_lock);
        aio_fixture_continue=true; aio_fixture_cv.notify_all();
    }
    {
        std::unique_lock<std::mutex> hold(aio_fixture_lock);
        if (!aio_fixture_cv.wait_for(hold,std::chrono::seconds(5),[]{return aio_fixture_done.size()==3;})) {
            CHECK(false,"three completions delivered"); std::abort();
        }
        CHECK(aio_fixture_done[0].id==(int32_t)rd32(0x290) &&
              aio_fixture_done[1].id==(int32_t)rd32(0x294) &&
              aio_fixture_done[2].id==(int32_t)rd32(0x298),"completion FIFO and distinct IDs");
        CHECK(aio_fixture_done[1].error==CELL_OK && aio_fixture_done[1].size==16 &&
              holds(0x6100,0x2000,16),"pinned original file survived close/reuse");
        CHECK(aio_fixture_done[2].error==(int32_t)CELL_ECANCELED &&
              aio_fixture_done[2].size==0,"canceled request completes once with ECANCELED");
    }
    {std::unique_lock<std::mutex> hold(manager->mutex);
     CHECK(manager->cv.wait_for(hold,std::chrono::seconds(5),[&]{return manager->pending==0;}),"all callbacks retired");}
    CHECK(fs_aio_async_finish("/dev_hdd1")==CELL_OK,"idle mount finish");
    close_ctx.gpr[3]=newfd; cellFsClose(&close_ctx);
    fs_aio_process_stop(); aio_fixture_thread.join();
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    static uint8_t mem[1 << 20];
    vm_base = mem; ppu_vm_size = sizeof mem;

    const char* dir = argc > 1 ? argv[1] : getenv("TMPDIR");
    if (!dir) dir = "/tmp";
    char host[1100];
    snprintf(host, sizeof host, "%s/ps3recomp_fs_aio.bin", dir);
    {
        FILE* f = fopen(host, "wb");
        if (!f) { printf("FAIL: could not create %s\n", host); return 1; }
        for (uint32_t i = 0; i < WORDS; i++) { uint32_t be = __builtin_bswap32(i * 4); fwrite(&be, 4, 1, f); }
        fclose(f);
    }

    ppu_vfs_root = dir;
    ppu_context ctx;

    /* 1. An AIO write lands at its own offset and leaves the fd's position;
     *    reads and writes never share an id. First, so neither has drawn an
     *    id yet: two counters would both start at 1. */
    char whost[1100];
    snprintf(whost, sizeof whost, "%s/ps3recomp_fs_aio_w.bin", dir);
    strcpy((char*)mem + 0x100, "/dev_hdd1/ps3recomp_fs_aio_w.bin");
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = 0x100; ctx.gpr[4] = 0x242 /* RDWR|CREAT|TRUNC */; ctx.gpr[5] = 0x200;
    cellFsOpen(&ctx);
    const uint32_t wfd = rd32(0x200);
    CHECK((uint32_t)ctx.gpr[3] == 0, "cellFsOpen(O_RDWR|O_CREAT|O_TRUNC) returns CELL_OK (fd %u)", wfd);
    for (uint32_t i = 0; i < 0x100; i += 4) wr32(0x3000 + i, 0x1000 + i);
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = wfd; ctx.gpr[4] = 0x20; ctx.gpr[5] = 0 /* SET */; ctx.gpr[6] = 0x208;
    cellFsLseek(&ctx);
    std::vector<uint32_t> ids;
    for (int r = 0; r < 4; r++) {
        aio_write(0x300, wfd, 0x1000, 0x3000, 0x100, 0x280); ids.push_back(rd32(0x280));
        aio_read(0x340, wfd, 0x1000, 0x4000, 0x100, 0x284);  ids.push_back(rd32(0x284));
    }
    CHECK(holds(0x4000, 0x1000, 0x100), "an AIO read returns what the AIO write put at offset 0x1000");
    bool unique = true;
    for (size_t i = 0; i < ids.size(); i++)
        for (size_t j = i + 1; j < ids.size(); j++) if (ids[i] == ids[j]) unique = false;
    CHECK(unique, "interleaved AIO reads and writes get distinct ids (write %u, read %u, write %u, ...)",
          ids[0], ids[1], ids[2]);
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = wfd; ctx.gpr[4] = 0; ctx.gpr[5] = 1 /* CUR */; ctx.gpr[6] = 0x208;
    cellFsLseek(&ctx);
    CHECK(rd64(0x208) == 0x20, "the fd's position is still 0x20 after the AIO write (0x%llX)",
          (unsigned long long)rd64(0x208));
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = wfd;
    cellFsClose(&ctx);
    remove(whost);

    strcpy((char*)mem + 0x100, "/dev_hdd1/ps3recomp_fs_aio.bin");
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = 0x100; ctx.gpr[4] = 0 /* O_RDONLY */; ctx.gpr[5] = 0x200;
    cellFsOpen(&ctx);
    const uint32_t fd = rd32(0x200);
    CHECK((uint32_t)ctx.gpr[3] == 0, "cellFsOpen returns CELL_OK (fd %u)", fd);
    if ((uint32_t)ctx.gpr[3] != 0) { remove(host); return 1; }

    /* 2. The fd's position survives an AIO read. */
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = fd; ctx.gpr[4] = 0x40; ctx.gpr[5] = 0 /* SET */; ctx.gpr[6] = 0x208;
    cellFsLseek(&ctx);
    aio_read(0x300, fd, 0x8000, 0x1000, 0x100);
    CHECK(holds(0x1000, 0x8000, 0x100), "the AIO read returns the words at its own offset");
    memset(&ctx, 0, sizeof ctx); ctx.gpr[3] = fd; ctx.gpr[4] = 0x2000; ctx.gpr[5] = 16; ctx.gpr[6] = 0x210;
    cellFsRead(&ctx);
    CHECK(rd64(0x210) == 16 && holds(0x2000, 0x40, 16),
          "cellFsRead after it continues from the fd's position 0x40 (got word 0x%08X)", rd32(0x2000));

    /* 3. AIO reads racing on the same fd each get their own bytes. */
    const int THREADS = 4, ROUNDS = 2000;
    const uint32_t LEN = 0x400;
    std::vector<int> bad(THREADS, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < THREADS; t++)
        ts.emplace_back([&, t] {
            const uint32_t aio = 0x400 + 0x40 * t, buf = 0x10000 + 0x10000 * t;
            uint32_t seed = 0x9E3779B9u * (t + 1);
            for (int r = 0; r < ROUNDS; r++) {
                seed = seed * 1664525u + 1013904223u;
                const uint64_t off = (uint64_t)(seed % (WORDS * 4 - LEN)) & ~3ull;
                aio_read(aio, fd, off, buf, LEN);
                if (!holds(buf, off, LEN)) bad[t]++;
            }
        });
    for (auto& th : ts) th.join();
    int total = 0; for (int b : bad) total += b;
    CHECK(total == 0, "%d racing AIO reads over %d threads, %d returned the wrong bytes",
          THREADS * ROUNDS, THREADS, total);

    test_async_queue((int)fd); // also closes/reuses fd and joins test service
    remove(host);
#ifndef _WIN32
    test_aio_write_prefault(dir);
#endif
    printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
