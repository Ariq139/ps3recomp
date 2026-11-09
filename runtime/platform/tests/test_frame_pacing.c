#define _POSIX_C_SOURCE 200809L
#define PS3_FRAME_PACING_MATH_ONLY
#include "../ps3_frame_pacing.h"
#include <assert.h>
int main(void)
{
    const unsigned rates[] = {30, 60, 120};
    for (unsigned r = 0; r < 3; ++r) {
        ps3_frame_deadline c;
        const uint64_t hz = 10000000;
        ps3_frame_deadline_init(&c, hz, rates[r], 0);
        uint64_t now = 0;
        for (unsigned frame = 0; frame < rates[r]; ++frame) {
            ps3_frame_deadline_advance(&c, now);
            assert(c.next > now);
            now = c.next;
        }
        assert(now == hz); /* exact rational second at 30/60/120 */
        now += hz * 5; /* rendering/occlusion missed five seconds */
        ps3_frame_deadline_advance(&c, now);
        assert(c.next - now >= hz / rates[r]);
        assert(c.next - now <= hz / rates[r] + 1);
        /* Small scheduler jitter must not accumulate into clock drift. */
        ps3_frame_deadline_init(&c, hz, rates[r], 0);
        for (unsigned frame = 0; frame < rates[r]; ++frame)
            ps3_frame_deadline_advance(&c, c.next + hz / 10000);
        assert(c.next == hz);
    }
    ps3_frame_deadline c;
    ps3_frame_deadline_init(&c, 10000000, 0, 10);
    ps3_frame_deadline_advance(&c, 12345);
    assert(c.next == 12345);
#ifdef _WIN32
#define SET_LIMIT(v) _putenv_s("PS3_FPS_LIMIT", v)
#else
#define SET_LIMIT(v) setenv("PS3_FPS_LIMIT", v, 1)
#endif
    SET_LIMIT(""); assert(ps3_guest_frame_limit() == 30);
    SET_LIMIT("0"); assert(ps3_guest_frame_limit() == 0);
    SET_LIMIT("30"); assert(ps3_guest_frame_limit() == 30);
    SET_LIMIT("60"); assert(ps3_guest_frame_limit() == 60);
    SET_LIMIT("120"); assert(ps3_guest_frame_limit() == 120);
    SET_LIMIT("240"); assert(ps3_guest_frame_limit() == 240);
    SET_LIMIT("241"); assert(ps3_guest_frame_limit() == 30);
    SET_LIMIT("-1"); assert(ps3_guest_frame_limit() == 30);
    SET_LIMIT("30junk"); assert(ps3_guest_frame_limit() == 30);
    puts("frame deadline math: 30/60/120 Hz, jitter, stalls, limit parsing, uncapped PASS");
    return 0;
}
