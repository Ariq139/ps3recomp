# Async SPU task submission ownership

Distinguish STARTING, RUNNING, FAILED and UNAVAILABLE host states rather than
treating every reservation as an executing worker. Completion returns the host
entry to IDLE, and the guest taskset bitsets are retired. For the initial PPU
create path a failed host launch returns AGAIN and clears the reserved task
bitsets and output ID. For a scheduler scan, a failed launch restores the
original pending bit and remains eligible on a later scan. A worker accepted
by the host but unable to allocate/load LS retires with a nonzero Taskset2
exit result. Unregistered images preserve the prior track-only compatibility
policy and are explicitly unavailable.

This is not a full SPURS scheduler: the taskset tracker capacity and legacy
s_tasks per-slot bookkeeping are unchanged; an SPU-originated failed dispatch
requires a future scheduler scan to retry. Raw lifted policy-module task
creation remains outside the PPU transaction. No hardware equivalence or
host resource exhaustion test is claimed until the draft is validated.
