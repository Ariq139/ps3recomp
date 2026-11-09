/* Reading the vblank counter must not advance the guest's clock. Reuse the
 * real GCM fixture; no renderer or guest program is needed for this check. */
#define main gcm_nop_fixture_main
#include "test_gcm_nop_runs.c"
#undef main
#include <assert.h>

#ifdef _WIN32
static DWORD WINAPI tick_worker(LPVOID unused)
{
    (void)unused;
    for (unsigned tick = 0; tick < 10000; ++tick)
        cellGcmTickVBlank();
    return 0;
}
#endif

int main(void)
{
    gcm_vblank_count_store(0);
    for (unsigned i = 0; i < 1000; ++i)
        assert(cellGcmGetVBlankCount() == 0);
    for (unsigned tick = 1; tick <= 60; ++tick) {
        cellGcmTickVBlank();
        for (unsigned poll = 0; poll < 100; ++poll)
            assert(cellGcmGetVBlankCount() == tick);
    }
    gcm_vblank_count_store(UINT32_MAX);
    assert(cellGcmGetVBlankCount() == UINT32_MAX);
    assert(cellGcmGetVBlankCount() == UINT32_MAX);
    cellGcmTickVBlank();
    assert(cellGcmGetVBlankCount() == 0);
#ifdef _WIN32
    HANDLE workers[2];
    for (unsigned i = 0; i < 2; ++i) {
        workers[i] = CreateThread(NULL, 0, tick_worker, NULL, 0, NULL);
        assert(workers[i]);
    }
    u32 previous = 0;
    while (WaitForMultipleObjects(2, workers, TRUE, 0) == WAIT_TIMEOUT) {
        const u32 current = cellGcmGetVBlankCount();
        assert(current >= previous && current <= 20000);
        previous = current;
    }
    assert(cellGcmGetVBlankCount() == 20000);
    for (unsigned i = 0; i < 2; ++i) CloseHandle(workers[i]);
#endif
    puts("vblank clock: ticks advance time; polling does not; wrap preserved");
    return 0;
}
