#ifndef PS3_CONSOLE_H
#define PS3_CONSOLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Capture and pass through the process' standard streams unchanged.
 * Call once near the start of the game runner; shutdown drains captured output. */
void ps3_console_init(void);
void ps3_console_shutdown(void);

/* Save guest TTY bytes outside the host diagnostic stream. The default file is
 * ps3_tty.log, decoded from CP932 to UTF-8. */
void ps3_console_write_tty(uint32_t channel, const void* bytes, size_t length);

#ifdef __cplusplus
}
#endif

#endif
