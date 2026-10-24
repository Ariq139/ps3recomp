# Encounter SPURS job BBB7C0C1505EDC27

Exact dispatched image: 10,848 bytes, legacy runtime fingerprint
`BBB7C0C1505EDC27`, SHA-256 `68b42391980e03b9c741753c12fa3857681c2b154c4a7877c47bd2cd9edeb773`.
The capture is stored in the shared cache's `spu_jobs` directory, not required
as an external launch file. Its complete bytes match the prefix of a 10,880-byte
region found by the boot scan in the original game assets.

The raw-job runner loads at LS 0 and enters LS 0. The entry's branch at 0x1C
reaches the CRT at 0x1AB0. That CRT reads context+0x14 and masks the low 28 bits
at 0x1B8C, then loads the manager entry through that pointer. It also reads the
kernel view at absolute LS 0x170. These instructions establish the Context2 and
kernel-context ABI flags. The body reads context+0x24 at 0x124 and scales the
value by eight at 0x16C before indexing a local tag table; the serial runner
uses worker slot zero, as for the other verified GFD workers.

Registration is in `jobmods_register.c`, image id 0x5012. The generated lift
has 161 entries and reports no unsupported instructions. Native compilation
checks instruction coverage; the same encounter passed the user retest on 2026-10-24 after both missing jobs were registered. Other encounters remain unverified.
