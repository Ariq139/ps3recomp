/*
 * ps3recomp - RSX guest vertex fetch (see rsx_vertex_fetch.h)
 */
#include "rsx_vertex_fetch.h"
#include "rsx_vertex_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float rsx_rd_bef(const u8* p)
{
    u32 w; memcpy(&w, p, 4);
    w = ((w >> 24) & 0xFFu) | ((w >> 8) & 0xFF00u) |
        ((w << 8) & 0xFF0000u) | ((w << 24) & 0xFF000000u);
    float f; memcpy(&f, &w, 4); return f;
}

float rsx_rd_half_be(const u8* p)
{
    u16 h = (u16)((p[0] << 8) | p[1]);
    u32 sgn = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        f = sgn << 31;
        if (man) {
            int e = -14;
            while (!(man & 0x400u)) { man <<= 1; --e; }
            f |= ((u32)(e + 127) << 23) | ((man & 0x3FFu) << 13);
        }
    }
    else if (exp == 31)  f = (sgn << 31) | 0x7F800000u | (man << 13);
    else                 f = (sgn << 31) | ((exp - 15u + 127u) << 23) | (man << 13);
    float o; memcpy(&o, &f, 4); return o;
}

static void rsx_fetch_attrib_internal(const rsx_state* st, int idx, u32 vi, float out[4],
                                      const rsx_vertex_fetch_prepared* prepared)
{
    /* A disabled attribute array feeds the CONSTANT vertex attribute register
     * (NV4097_SET_VERTEX_DATA4F_M), not zero -- same rule as glColor4f with
     * the colour array off. Defaulting these to black multiplied Rubber
     * Ducky's duck texture away in the fragment program. Enabled arrays use
     * their own (0,0,0,1) defaults for absent components. */
    out[0] = st->vertex_data4f[idx][0];
    out[1] = st->vertex_data4f[idx][1];
    out[2] = st->vertex_data4f[idx][2];
    out[3] = st->vertex_data4f[idx][3];

    const rsx_vertex_attrib* a = &st->vertex_attribs[idx];
    if (!a->enabled || !vm_base) return;
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;

    /* Vertex frequency divisor (instancing). An ordinary array
     * (rsx_vertex_freq_plain: freq 0, or 1 with DIVIDE) is per-vertex.
     * Otherwise the element index is either vertex/freq (DIVIDE: per-instance
     * data advances once per freq verts) or vertex%freq (MODULO: a mesh
     * repeats every freq verts). NV4097_SET_FREQUENCY_DIVIDER_OPERATION's
     * per-attribute bit selects MODULO. DeferredShading's cube rings need
     * this: the shared cube mesh is MODULO, the per-instance transform in
     * attrib9 is DIVIDE -- without it attrib9 advanced every vertex and the
     * instanced cubes drew with garbage transforms (only the baseplate,
     * which isn't instanced, survived). */
    u32 ei = vi;
    if (!rsx_vertex_freq_plain(a->frequency,
                               (st->frequency_divider_op >> idx) & 1u)) {
        static int nofreq = -1;                 /* VP_NOFREQ: instancing kill-switch */
        if (nofreq < 0) nofreq = getenv("VP_NOFREQ") ? 1 : 0;
        if (!nofreq)
            ei = (st->frequency_divider_op & (1u << idx))
                     ? (vi % a->frequency)      /* MODULO: repeat mesh */
                     : (vi / a->frequency);     /* DIVIDE: per-instance */
    }

    /* Bit 31 of the vertex-array OFFSET selects the context DMA: 0 = LOCAL
     * (VRAM), 1 = MAIN (IO-mapped system memory). Resolve LOCAL as local --
     * routing it through cellGcmResolveOffset lets the IO table win for any
     * offset whose page the IO region also covers. One title's vertex arrays
     * sit at offsets like 0x4480, shadowed by its 1MB IO window at 0x11100000:
     * the array was uploaded to VRAM 0xC0004480 but resolved to main
     * 0x11104480, which is empty -- so every INDEXED mesh fetched zeros and
     * collapsed to the origin, while non-indexed geometry whose arrays lie
     * outside the shadowed pages drew fine. */
    /* NV4097_SET_VERTEX_DATA_BASE_OFFSET is added to the attribute's low
     * offset bits and wraps at 28 bits before the per-vertex stride is added.
     * Keep the DMA selector from the attribute offset unchanged. */
    u32 off = (st->vertex_data_base_offset + (a->offset & 0x7FFFFFFFu))
              & 0x0FFFFFFFu;
    off += ei * a->stride;
    const u8* p = prepared && prepared->local_base[idx] && (u64)off >= (u64)ei * a->stride &&
        (u64)(prepared->local_base[idx] - vm_base) + (u64)ei * a->stride <= 0xFFFFFFFFull
        ? prepared->local_base[idx] + (u32)(ei * a->stride)
        : vm_base + ((a->offset & 0x80000000u)
        ? cellGcmResolveLocated(0, off)   /* MAIN:  IO offset table          */
        : cellGcmResolveLocated(1, off)); /* LOCAL: VRAM, never the IO table */

    u32 n = a->size ? a->size : 4; if (n > 4) n = 4;
    switch (a->type) {
    case 2: /* CELL_GCM_VERTEX_F: float32 BE */
        for (u32 k = 0; k < n; k++) out[k] = rsx_rd_bef(p + k * 4);
        break;
    case 3: /* SF: half float BE */
        for (u32 k = 0; k < n; k++) out[k] = rsx_rd_half_be(p + k * 2);
        break;
    case 4: /* UB: u8 normalized [0,1] */
        for (u32 k = 0; k < n; k++) out[k] = p[k] / 255.0f;
        break;
    case 1: /* S1: s16 normalized [-1,1] */
        for (u32 k = 0; k < n; k++) {
            s16 s = (s16)((p[k*2] << 8) | p[k*2+1]);
            /* RSX maps the signed endpoints symmetrically around -0.5. */
            out[k] = ((float)s + 0.5f) / 32767.5f;
        }
        break;
    case 5: /* S32K: s16 integer */
        for (u32 k = 0; k < n; k++)
            out[k] = (float)(s16)((p[k*2] << 8) | p[k*2+1]);
        break;
    case 6: { /* CMP: signed X11/Y11/Z10, expanded to SNORM16, W=1. */
        /* RPCS3_REFERENCE_REVIEW: behavior checked against GPL-2.0 RPCS3's
         * RSXVertexFetch.glsl and Core/RSXDrawCommands.cpp. This is a scalar
         * implementation of the packed format; see docs/RPCS3_LICENSE_AUDIT.md. */
        u32 bits = ((u32)p[0] << 24) | ((u32)p[1] << 16) |
                   ((u32)p[2] << 8) | (u32)p[3];
        for (u32 k = 0; k < 3; ++k) {
            u32 width = k == 2 ? 10u : 11u;
            u32 shift = k == 2 ? 22u : k * 11u;
            u32 field = (bits >> shift) & ((1u << width) - 1u);
            s32 signed_value = (field & (1u << (width - 1u)))
                ? (s32)field - (s32)(1u << width) : (s32)field;
            out[k] = (float)(signed_value * (s32)(1u << (16u - width))) / 32767.0f;
        }
        break;
    }
    case 7: /* UB256: u8 unnormalized */
        for (u32 k = 0; k < n; k++) out[k] = (float)p[k];
        break;
    default:
        { static int t = -1; if (t < 0) t = getenv("VTX_TYPEDBG") ? 1 : 0;
          if (t) { static u32 seen = 0;
              if (!(seen & (1u << (a->type & 31)))) { seen |= 1u << (a->type & 31);
                  fprintf(stderr, "[VTXTYPE] UNHANDLED type=%u attr=%d size=%u stride=%u\n",
                          a->type, idx, a->size, a->stride); } } }
        for (u32 k = 0; k < n; k++) out[k] = rsx_rd_bef(p + k * 4);
        break;
    }

    /* When an attribute array is enabled with fewer than 4 components,
     * missing components expand according to standard graphics hardware rules:
     * missing y,z = 0.0, missing w = 1.0. A game may clear the constant
     * register vertex_data4f to all zeros; leaving w=0 on a 3-component position
     * array drops translation in the vertex shader and produces clip-space w=0,
     * which causes the hardware clipper to discard every vertex. */
    if (n < 4) out[3] = 1.0f;
    /* CMP stores one packed word, but that word expands to three components.
     * Its size describes the packed input, not the decoded vector width. */
    if (a->type != 6) {
        if (n < 3) out[2] = 0.0f;
        if (n < 2) out[1] = 0.0f;
    }
}

void rsx_vertex_fetch_prepare(rsx_vertex_fetch_prepared* plan, const rsx_state* state)
{
    memset(plan, 0, sizeof(*plan));
    plan->state = state;
    if (!state || !vm_base) return;
    for (u32 i = 0; i < 16; ++i) {
        const rsx_vertex_attrib* a = &state->vertex_attribs[i];
        if (!a->enabled || (a->offset & 0x80000000u)) continue;
        const u32 offset = (state->vertex_data_base_offset + (a->offset & 0x7FFFFFFFu)) & 0x0FFFFFFFu;
        plan->local_base[i] = vm_base + cellGcmResolveLocated(1, offset);
    }
}
void rsx_fetch_attrib_prepared(const rsx_vertex_fetch_prepared* plan, int idx, u32 vi, float out[4])
{
    rsx_fetch_attrib_internal(plan->state, idx, vi, out, plan);
}
void rsx_fetch_attrib(const rsx_state* st, int idx, u32 vi, float out[4])
{
    rsx_fetch_attrib_internal(st, idx, vi, out, NULL);
}
