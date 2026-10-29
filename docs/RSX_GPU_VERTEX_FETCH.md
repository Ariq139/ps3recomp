# D3D12 GPU vertex fetch

The D3D12 backend can upload guest attribute bytes and decode them in the
translated vertex shader. This follows the architecture inspected in local
RPCS3 revision `385f86b85`: `Core/RSXDrawCommands.cpp`, `VK/VKVertexBuffers.cpp`
and `Program/GLSLSnippets/RSXProg/RSXVertexFetch.glsl`. The implementation is
independently written; it does not link RPCS3.

`RSX_GPU_VERTEX_FETCH=1` is the default. Set it to `0` before launch to compare
against CPU decoding. With `PERF=1`, `[PERF-GPUFETCH]` reports accepted draws,
raw upload bytes and bytes saved relative to converted vertex uploads.
Counters establish path usage; they do not establish an FPS improvement.

Each accepted draw snapshots its referenced guest ranges into the existing
submission buffer. Overlapping interleaved attributes share a copy. A small
index map preserves triangle, quad, strip, fan and flat-shading vertex order.
The HLSL fetch handles big-endian float32/float16, normalized and integer
16-bit values, normalized and integer bytes, packed 11/11/10 values, disabled
attribute constants, strides and frequency operations.

The CPU path handles unsupported programs and primitives, restart streams,
partial triangle batches, noncontiguous mappings, layouts requiring all 16
slots, insufficient buffer space and cases where raw uploads would cost more
bytes. Indexed draws combining a nonzero element base with frequency operations
also retain CPU decoding. Per-draw shader variants and PSO keys distinguish
the two layouts. Raw captures include `opNNN_vertices.raw` and `vs_NNN_raw.asm`.

Root SRV `t20` is vertex-visible and points to a `ByteAddressBuffer` in an upload
resource held in `GENERIC_READ`. This is permitted by Microsoft's documentation
for [root descriptors](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-descriptors-directly-in-the-root-signature)
and [upload heaps](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type).
FIFO order, GPU fences and buffer reuse continue through the backend's existing
submission path.

With `PS3RECOMP_BUILD_TESTS=ON`, build `test_vertex_raw_gpu` and run
`ctest -R '^vertex_raw_gpu$' --output-on-failure`. The executable also accepts
`--hardware` for the preferred high-performance adapter, and `--corpus DIR`
to decompile and compile each cached RPCS3 `.vp` through the raw fetch wrapper.
The GPU test checks decoded output against reference values, including special
half values, unaligned input, constants, frequency operations and packed formats.
It also checks range sharing, capacity rejection, noncontiguous mappings and
primitive remapping.
