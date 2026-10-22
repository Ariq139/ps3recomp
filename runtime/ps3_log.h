/* ps3_log.h -- gate for high-volume diagnostic logging.
 * Debug/trace output is opt-in. The game runner defaults to warning-level
 * output, so redirecting stderr does not enable a high-volume trace by itself.
 */
#ifndef PS3_LOG_H
#define PS3_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
static inline int ps3_log_verbose(void)
#else
static __inline int ps3_log_verbose(void)
#endif
{
    /* PS3_VERBOSE explicitly enables or disables these high-volume trace sites.
     * Otherwise PS3_LOG_LEVEL=debug or trace enables them. */
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("PS3_VERBOSE");
        if (e) v = (e[0] && e[0] != '0') ? 1 : 0;
        else {
            const char* level = getenv("PS3_LOG_LEVEL");
            v = level && (strcmp(level, "debug") == 0 || strcmp(level, "trace") == 0);
        }
    }
    return v;
}

#endif /* PS3_LOG_H */
