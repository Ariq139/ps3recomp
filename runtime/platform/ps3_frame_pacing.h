/* Guest frame pacing, independent of monitor refresh and FIFO progress. */
#ifndef PS3_FRAME_PACING_H
#define PS3_FRAME_PACING_H
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct ps3_frame_deadline {
    uint64_t frequency, next, fraction;
    unsigned rate;
} ps3_frame_deadline;

static inline void ps3_frame_deadline_init(ps3_frame_deadline* c,
                                          uint64_t frequency, unsigned rate,
                                          uint64_t now)
{
    c->frequency = frequency; c->rate = rate;
    c->fraction = 0; c->next = now;
}

/* Keep ordinary timer jitter anchored to the previous scheduled deadline.
 * If a whole interval was missed, restart from now rather than issuing a
 * burst of catch-up frames. Carry fractional ticks instead of rounding every
 * interval to milliseconds. */
static inline void ps3_frame_deadline_advance(ps3_frame_deadline* c, uint64_t now)
{
    if (!c->rate) { c->next = now; return; }
    uint64_t step = c->frequency / c->rate;
    c->fraction += c->frequency % c->rate;
    if (c->fraction >= c->rate) { c->fraction -= c->rate; ++step; }
    uint64_t base = c->next;
    if (now >= base && now - base >= step) base = now;
    c->next = base + step;
}

static inline unsigned ps3_guest_frame_limit(void)
{
    const char* e = getenv("PS3_FPS_LIMIT");
    if (!e || !*e) return 30;
    char* end = NULL;
    unsigned long value = strtoul(e, &end, 10);
    if (end == e || *end || value > 240) {
        fprintf(stderr, "[frame-clock] invalid PS3_FPS_LIMIT; using 30 FPS\n");
        return 30;
    }
    return (unsigned)value;
}

#ifndef PS3_FRAME_PACING_MATH_ONLY
#include "win32_compat.h"
typedef struct ps3_frame_pacer {
    ps3_frame_deadline clock;
#ifdef _WIN32
    HANDLE timer;
#endif
} ps3_frame_pacer;

static inline uint64_t ps3_frame_counter(void)
{
    LARGE_INTEGER q; QueryPerformanceCounter(&q); return (uint64_t)q.QuadPart;
}
static inline void ps3_frame_pacer_init(ps3_frame_pacer* p, unsigned rate)
{
    LARGE_INTEGER q; QueryPerformanceFrequency(&q);
    ps3_frame_deadline_init(&p->clock, (uint64_t)q.QuadPart, rate, ps3_frame_counter());
#ifdef _WIN32
    /* Windows 10 1803+: high-resolution waitable timer. Earlier systems fall
     * back to the ordinary waitable timer without changing deadline math. */
    p->timer = CreateWaitableTimerExW(NULL, NULL, 0x2u, TIMER_ALL_ACCESS);
    if (!p->timer) p->timer = CreateWaitableTimerW(NULL, FALSE, NULL);
#endif
}
static inline void ps3_frame_pacer_wait(ps3_frame_pacer* p)
{
    if (!p->clock.rate) return;
    uint64_t now = ps3_frame_counter();
    while (now < p->clock.next) {
        uint64_t left = p->clock.next - now;
#ifdef _WIN32
        if (p->timer) {
            LARGE_INTEGER due;
            uint64_t units = (left * 10000000ull + p->clock.frequency - 1) / p->clock.frequency;
            due.QuadPart = -(LONGLONG)(units ? units : 1);
            if (SetWaitableTimer(p->timer, &due, 0, NULL, NULL, FALSE))
                WaitForSingleObject(p->timer, INFINITE);
            else Sleep(1);
        } else
#endif
        {
            DWORD ms = (DWORD)((left * 1000ull + p->clock.frequency - 1) / p->clock.frequency);
            Sleep(ms ? ms : 1);
        }
        now = ps3_frame_counter();
    }
    ps3_frame_deadline_advance(&p->clock, now);
}
#endif
#endif
