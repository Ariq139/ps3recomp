/* Windows waitable-timer smoke test; no GPU or game assets required. */
#include "../ps3_frame_pacing.h"
#include <assert.h>
int main(void)
{
#ifndef _WIN32
    puts("SKIP: native Windows waitable timer test");
#else
    ps3_frame_pacer p;
    ps3_frame_pacer_init(&p, 30);
    ps3_frame_pacer_wait(&p); /* first completed frame is immediately eligible */
    uint64_t first = ps3_frame_counter(), previous = first;
    for (unsigned i = 0; i < 8; ++i) {
        uint64_t expected = p.clock.next;
        ps3_frame_pacer_wait(&p);
        uint64_t now = ps3_frame_counter();
        assert(now >= expected); /* must not release before its deadline */
        printf("30 FPS interval %u: %.3f ms\n", i + 1,
               (double)(now - previous) * 1000.0 / p.clock.frequency);
        previous = now;
    }
    printf("30 FPS measured average: %.3f ms (no upper bound under host load)\n",
           (double)(previous - first) * 1000.0 / (p.clock.frequency * 8));
    Sleep(150); /* several missed deadlines must not create a release burst */
    ps3_frame_pacer_wait(&p);
    uint64_t after_stall = ps3_frame_counter(), expected = p.clock.next;
    ps3_frame_pacer_wait(&p);
    uint64_t now = ps3_frame_counter();
    assert(now >= expected);
    printf("post-stall following interval: %.3f ms\n",
           (double)(now - after_stall) * 1000.0 / p.clock.frequency);
    if (p.timer) CloseHandle(p.timer);
    puts("Windows guest frame pacer: PASS");
#endif
    return 0;
}
