# Lightweight condition event mode

The default context-HLE behavior and fast mutex registrations remain unchanged.
`PS3RECOMP_LWCOND_EVENTS=1` is the existing bounded notification/recheck
experiment. It does not implement exact guest timeouts.

`PS3RECOMP_LWCOND_EVENTS=hle-strict` selects exact condition waits for
**HLE-only** callers, including the direct C sysPrxForUser guest-EA bridge.
Finite waits return ETIMEDOUT if not notified; zero waits until notification.
Notifications are not retained for future waiters. Waiters reacquire the
paired mutex before returning. Targeted and broadcast notification are
supported. Single notification honors FIFO and the current priority for
SYS_SYNC_PRIORITY. The existing mutex acquisition queue remains FIFO.

In experimental modes the context mutex lock/try/unlock registrations now
pin their logical lifetime, with the old fast handlers disabled. Strict mode
requires a registered live mutex. The bounded compatibility mode still
permits pre-existing static mutexes. This is not a complete raw LV2
implementation: syscalls 95-99, 111-113, 115-117 use kernel queue IDs and
signaling modes, not guest object EAs. Titles with lifted code issuing those
syscalls or manipulating the mutex directly must not use hle-strict. The
legacy "strict" switch remains withdrawn. No default is changed.

Run `python tools/test_lwcond_lifetime.py` to exercise the registry and real
HLE condition handler bodies with a synthetic host mutex. Full runtime,
raw queue ABI, guest mutex scheduling/protocol correctness and mixed lifted
PRX testing remain required before enabling strict mode generally.
