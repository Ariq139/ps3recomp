# Async SPU task ownership
The hash directory holds keyed taskset state and legacy join records without the
previous eight-taskset limit. A PPU creation reserves host state and guest
enabled together. Submission claims SUBMITTED exactly once. Initial host
launch failure rolls back the caller's reservation. An already-accepted
pending task is retired with ENOMEM on host launch failure without depending
on another scan. Worker exits, including the CRI diagnostic policy path and
LS initialization failure, retire the task. Persistent LS contents are cleared
before task ID reuse. Legacy v1 join preserves the immediate-completion
approximation with taskset/slot identity. Unlifted images retain track-only
compatibility. Directory nodes are retained for process lifetime; complete
taskset teardown/recreation and hardware-equivalent failure semantics remain
outside this change. Fixtures model host launch failure; not executed.
