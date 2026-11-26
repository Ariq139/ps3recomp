# PPU callback lifetime and reservation contract
The two legacy callback entry forms use one checked execution mechanism. Scratch
frames use the same shared lifetime registry lock as registered PPU threads,
and are removed before slot/stack reclamation, including nonlocal thread exit.
Idle host threads release their callback stack slots after the outer invocation;
the guest thread identity and static TLS ownership remain host-thread-local.
Checked OPD and explicit-stack APIs distinguish refusal from completed zero.
Reservation address publication/reads/invalidation use atomic operations in
the loader, C helpers and newly lifted lwarx/ldarx/stwcx/stdcx. **Regenerate all
lifted PPU sources before exercising this change**; old pre-generated sources
still contain non-atomic reservation address accesses. This removes mixed
accesses on regenerated paths but does not establish exact Cell line-reservation
or same-value ABA behavior for every store path. The existing opt-in open-count
contract still requires compatible generation. Explicit non-arena stacks and
arbitrary guest pointer lifetimes remain caller responsibilities.
No build or test run was performed.
