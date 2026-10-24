# Encounter SPURS job AF5FF4A5692E0412

Exact runtime capture: 20,704 bytes, legacy fingerprint `AF5FF4A5692E0412`,
SHA-256 `f3973e9261d02de180c569230d0e0ef3e6c929db50fe69ef687673be96478636`. Its complete bytes match the prefix
of a 20,736-byte raw region in the boot corpus. The capture stays in the shared
cache; no external image is required at launch.

The raw-job runner loads at LS 0 and enters 0. Entry branches to the CRT at
0x4120, which calls main at 0x2630. The CRT reads context+0x14 at 0x41C0 and
masks its low 28 bits at 0x41FC. It reads the absolute kernel view at LS 0x170
at 0x4204, proving Context2/kernel-context requirements. Main reads the context
word at +0x24 at 0x277C, extracts it at 0x27BC, scales it by eight at 0x27D8,
and indexes the tag-pair table at 0x47A0. That table starts with (3,9), (12,9),
(21,10), so the serial runner uses worker slot zero.

Registration is in `jobmods_register.c`, image id 0x5013. Generation reports
no unsupported instructions. The same encounter passed the user retest on 2026-10-24 after both missing jobs were registered. Other encounters remain unverified.
