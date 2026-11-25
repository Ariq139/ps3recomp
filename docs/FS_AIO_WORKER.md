# Experimental registered PPU AIO service

The default is still the existing synchronous transfer and inline completion.
`PS3RECOMP_FS_AIO_WORKER=1` enables an experimental asynchronous implementation
for the context-HLE FILE-backed cellFs descriptors. It requires the PPU loader
and native registered PPU service-thread hooks. It fails rather than silently
using a host-only callback thread when those hooks are unavailable.

The service uses the existing PPU thread table, guest stack, reservation
registration and a fresh TLS block. It executes a FIFO of staged transfers,
and dispatches each completion on its own registered PPU context, not the
submitting context or an unregistered host thread. The disk phase owns a
pinned FILE identity, uses positional I/O with save/restore, and accesses only
host staging memory. Close logically removes the fd immediately; a queued
transfer keeps the old FILE alive even if the numeric descriptor is reused.

Up to eight initialized mount roots, 64 pending requests and 128 MiB of staging
are supported. Requests use one positive, non-wrapping ID space. Write bytes
are snapshotted at submit; read bytes are copied before the callback. The
caller must keep its AIO structure and read buffer valid until completion.
The worker validates the VM base/size and guest ranges again before delivery;
it cannot detect arbitrary guest malloc/free/reuse. This is not a VM allocator
pin and does not support hot VM replacement or raw syscall-only descriptors.

Queued requests can be canceled; they complete in FIFO order with ECANCELED.
An already executing request returns EBUSY on cancellation. Finish returns
EBUSY while any request/callback for its mount is outstanding, and otherwise
ends that mount session. One dormant PPU service is retained across sessions.
These are conservative experimental policies, not verified exact PS3
cancellation/finish ABI. Do not use the mode for titles requiring another
policy without verification.

Process shutdown stops acceptance, drops undelivered queued requests, and
waits only for host I/O/startup, never for an active guest callback. The
existing standalone process termination model is required; this is not a
safe reusable-VM teardown protocol. Callback exceptions or nonlocal thread
exit quarantine the queue instead of retrying a possibly executed callback.

Regression: tools/test_ppu_fs_aio.py exercises the original synchronous
contract plus a synthetic registered-service hook fixture. Native guest
service thread integration, shutdown and gameplay still require testing.
