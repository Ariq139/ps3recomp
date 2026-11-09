#ifndef PS3RECOMP_RSX_VERTEX_RAW_H
#define PS3RECOMP_RSX_VERTEX_RAW_H

#include "rsx_commands.h"
#include "rsx_primitives.h"
#include <stdio.h>
#include <string.h>

#define RSX_RAW_VERTEX_HEADER_BYTES (16u + 16u * 48u)

static u32 rsx_raw_vertex_count(u32 primitive, u32 count)
{
    if (primitive == RSX_PRIMITIVE_TRIANGLES) return count % 3 ? 0 : count;
    if (primitive == RSX_PRIMITIVE_QUADS)
        return count / 4 <= 0xFFFFFFFFu / 6 ? count / 4 * 6 : 0;
    if (primitive == RSX_PRIMITIVE_TRIANGLE_STRIP ||
        primitive == RSX_PRIMITIVE_TRIANGLE_FAN)
        return count >= 3 && count - 2 <= 0xFFFFFFFFu / 3 ? (count - 2) * 3 : 0;
    return 0;
}

static u32 rsx_raw_vertex_source(u32 primitive, u32 output, int flat)
{
    if (flat) output = output - output % 3 + (output % 3 + 2) % 3;
    if (primitive == RSX_PRIMITIVE_TRIANGLES) return output;
    if (primitive == RSX_PRIMITIVE_QUADS) {
        static const u32 corners[6] = {0,1,2,0,2,3};
        return (output / 6) * 4 + corners[output % 6];
    }
    const u32 triangle = output / 3, corner = output % 3;
    if (primitive == RSX_PRIMITIVE_TRIANGLE_FAN)
        return corner ? triangle + corner : 0;
    return triangle + ((triangle & 1) && corner ? 3 - corner : corner);
}

typedef struct {
    u32 source, size, destination;
} rsx_raw_vertex_range;

typedef struct {
    u32 header[4];
    u32 attributes[16][12];
    rsx_raw_vertex_range ranges[16];
    u32 range_count, bytes;
} rsx_raw_vertex_plan;

/* Snapshot the union of referenced guest ranges. Interleaved attributes share
 * one copy; descriptors retain their own stride, frequency and byte offset. */
static int rsx_raw_vertex_prepare(const rsx_state* state, u32 mask,
    const u32* vertices, u32 count, u32 capacity,
    u32 (*resolve)(int, u32), rsx_raw_vertex_plan* plan)
{
    if (!state || !vertices || !count || !resolve || !plan) return 0;
    u32 lo = vertices[0], hi = vertices[0], sources[16] = {0};
    for (u32 v = 1; v < count; ++v) {
        if (vertices[v] < lo) lo = vertices[v];
        if (vertices[v] > hi) hi = vertices[v];
    }
    memset(plan, 0, sizeof(*plan));
    plan->header[0] = RSX_RAW_VERTEX_HEADER_BYTES;
    plan->header[1] = count;
    for (u32 i = 0; i < 16; ++i) {
        if (!(mask & (1u << i))) continue;
        u32* d = plan->attributes[i];
        const rsx_vertex_attrib* a = &state->vertex_attribs[i];
        memcpy(d + 8, state->vertex_data4f[i], 16);
        if (!a->enabled) continue;
        if (a->type < 1 || a->type > 7) return 0;
        const u32 n = a->size ? (a->size > 4 ? 4 : a->size) : 4;
        const u32 component = (a->type == 2 || a->type == 6) ? 4
            : (a->type == 4 || a->type == 7) ? 1 : 2;
        const u32 element_bytes = a->type == 6 ? 4 : component * n;
        const u32 modulo = (state->frequency_divider_op >> i) & 1u;
        u32 first = lo, last = hi;
        if (a->frequency && (a->frequency != 1 || modulo)) {
            if (modulo) {
                first = 0;
                last = hi < a->frequency ? hi : a->frequency - 1;
            } else {
                first /= a->frequency;
                last /= a->frequency;
            }
        }
        const u32 base = (state->vertex_data_base_offset +
            (a->offset & 0x7FFFFFFFu)) & 0x0FFFFFFFu;
        const u64 begin = (u64)base + (u64)first * a->stride;
        const u64 size = (u64)(last - first) * a->stride + element_bytes;
        if (begin + size > 0x100000000ull || size > capacity) return 0;
        const int local = !(a->offset & 0x80000000u);
        const u32 source = resolve(local, (u32)begin);
        if (source == 0xFFFFFFFFu || (u64)source + size > 0x100000000ull)
            return 0;
        /* MAIN mappings can change at every 1 MiB boundary. */
        for (u64 p = (begin & ~0xFFFFFull) + 0x100000ull;
             p < begin + size; p += 0x100000ull)
            if (resolve(local, (u32)p) != (u64)source + p - begin) return 0;
        if (resolve(local, (u32)(begin + size - 1)) !=
            (u64)source + size - 1) return 0;
        sources[i] = source;
        d[1] = a->stride; d[2] = a->type; d[3] = n;
        d[4] = first; d[5] = a->frequency; d[6] = modulo; d[7] = 1;
        u32 slot = plan->range_count++;
        while (slot && plan->ranges[slot - 1].source > source) {
            plan->ranges[slot] = plan->ranges[slot - 1];
            --slot;
        }
        plan->ranges[slot] = (rsx_raw_vertex_range){source, (u32)size, 0};
    }
    u32 merged = 0;
    for (u32 i = 0; i < plan->range_count; ++i) {
        rsx_raw_vertex_range range = plan->ranges[i];
        if (merged && (u64)range.source <=
            (u64)plan->ranges[merged - 1].source + plan->ranges[merged - 1].size) {
            rsx_raw_vertex_range* prev = &plan->ranges[merged - 1];
            const u64 end = (u64)range.source + range.size;
            if (end > (u64)prev->source + prev->size)
                prev->size = (u32)(end - prev->source);
        } else {
            plan->ranges[merged++] = range;
        }
    }
    plan->range_count = merged;
    u64 bytes = RSX_RAW_VERTEX_HEADER_BYTES + (u64)count * 4;
    for (u32 i = 0; i < merged; ++i) {
        rsx_raw_vertex_range* range = &plan->ranges[i];
        if (((bytes + range->size + 3) & ~3ull) > capacity) return 0;
        range->destination = (u32)bytes;
        for (u32 a = 0; a < 16; ++a)
            if (plan->attributes[a][7] && sources[a] >= range->source &&
                (u64)sources[a] < (u64)range->source + range->size)
                plan->attributes[a][0] = (u32)bytes + sources[a] - range->source;
        bytes += ((u64)range->size + 3) & ~3ull;
    }
    bytes = (bytes + 15) & ~15ull;
    if (bytes > capacity) return 0;
    plan->bytes = (u32)bytes;
    return 1;
}

static int rsx_raw_vertex_indexed_prepare(const rsx_state* state, u32 mask,
    u32* indices, u32 count, u32* dense, u32 capacity,
    u32 (*resolve)(int, u32), rsx_raw_vertex_plan* plan)
{
    if (!count || count > capacity / 4u) return 0;
    u32 lo = indices[0], hi = lo;
    for (u32 i = 1; i < count; ++i) {
        if (indices[i] < lo) lo = indices[i];
        if (indices[i] > hi) hi = indices[i];
    }
    if ((u64)hi - lo + 1u >= count) return 0;
    const u32 dense_count = hi - lo + 1u;
    const u32 index_bytes = (count * 4u + 15u) & ~15u;
    if (index_bytes > capacity) return 0;
    for (u32 i = 0; i < dense_count; ++i) dense[i] = lo + i;
    if (!rsx_raw_vertex_prepare(state, mask, dense, dense_count,
            capacity - index_bytes, resolve, plan)) return 0;
    for (u32 i = 0; i < count; ++i) indices[i] -= lo;
    return 1;
}

static void rsx_raw_vertex_write(const rsx_raw_vertex_plan* plan,
    const u32* vertices, const u8* guest, void* destination)
{
    u8* out = (u8*)destination;
    memcpy(out, plan->header, 16);
    memcpy(out + 16, plan->attributes, sizeof(plan->attributes));
    memcpy(out + RSX_RAW_VERTEX_HEADER_BYTES, vertices,
           (size_t)plan->header[1] * 4);
    u32 end = RSX_RAW_VERTEX_HEADER_BYTES + plan->header[1] * 4;
    for (u32 i = 0; i < plan->range_count; ++i) {
        const rsx_raw_vertex_range* range = &plan->ranges[i];
        if (range->destination > end) memset(out + end, 0, range->destination - end);
        memcpy(out + range->destination, guest + range->source, range->size);
        end = range->destination + range->size;
    }
    if (plan->bytes > end) memset(out + end, 0, plan->bytes - end);
}

/* Independently emitted HLSL for the RSX attribute formats and frequency rules
 * also handled by RPCS3's RSXVertexFetch.glsl. Root SRV t20 contains raw bytes. */
static const char rsx_raw_vertex_hlsl[] =
    "ByteAddressBuffer rsx_raw_vertices : register(t20);\n"
    "uint rsx_raw_u8(uint p){return (rsx_raw_vertices.Load(p&~3u)>>((p&3u)*8u))&255u;}\n"
    "uint rsx_raw_be16(uint p){return (rsx_raw_u8(p)<<8)|rsx_raw_u8(p+1u);}\n"
    "uint rsx_raw_be32(uint p){uint s=(p&3u)*8u;uint b=p&~3u;"
    "uint x=rsx_raw_vertices.Load(b)>>s;"
    "if(s!=0u)x|=rsx_raw_vertices.Load(b+4u)<<(32u-s);"
    "return ((x&255u)<<24)|((x&65280u)<<8)|((x>>8)&65280u)|(x>>24);}\n"
    "float4 rsx_raw_attribute(uint a,uint vi){uint p=16u+a*48u;"
    "uint4 d=rsx_raw_vertices.Load4(p);uint4 e=rsx_raw_vertices.Load4(p+16u);"
    "if(e.w==0u)return asfloat(rsx_raw_vertices.Load4(p+32u));"
    "uint ei=vi;if(e.y!=0u&&(e.y!=1u||e.z!=0u))ei=e.z!=0u?vi%e.y:vi/e.y;"
    "uint b=d.x+(ei-e.x)*d.y;uint4 r=uint4(0,0,0,0);"
    "if(d.z==6u)r.x=rsx_raw_be32(b);else "
    "[unroll]for(uint k=0u;k<4u;++k){if(k<d.w){"
    "if(d.z==2u)r[k]=rsx_raw_be32(b+k*4u);"
    "else if(d.z==4u||d.z==7u)r[k]=rsx_raw_u8(b+k);"
    "else r[k]=rsx_raw_be16(b+k*2u);}}"
    "float4 v=float4(0,0,0,1);"
    "if(d.z==2u)v=asfloat(r);"
    "else if(d.z==3u)v=float4(f16tof32(r.x),f16tof32(r.y),f16tof32(r.z),f16tof32(r.w));"
    "else if(d.z==1u||d.z==5u){int4 s=int4(r<<16)>>16;v=float4(s);"
    "if(d.z==1u)v=(v+0.5)/32767.5;}"
    "else if(d.z==6u){int x=int(r.x<<21)>>21;int y=int(r.x<<10)>>21;"
    "int z=int(r.x)>>22;return float4(float3(x*32,y*32,z*64)/32767.0,1.0);}"
    "else {v=float4(r);if(d.z==4u)v/=255.0;}"
    "if(d.w<4u)v.w=1.0;return v;}\n";

static int rsx_raw_vertex_wrap_hlsl(const char* source, u32 mask,
    char* output, size_t capacity)
{
    const char* signature = "VSOutput main(VSInput input)";
    const char* entry = strstr(source, signature);
    if (!entry || strlen(source) + sizeof(rsx_raw_vertex_hlsl) + 2048 > capacity)
        return 0;
    const size_t prefix = (size_t)(entry - source);
    memcpy(output, source, prefix);
    snprintf(output + prefix, capacity - prefix,
        "VSOutput rsx_raw_guest_main(VSInput input)%s\n%s\n"
        "VSOutput main(uint vertex_id:SV_VertexID){VSInput input=(VSInput)0;"
        "uint vi=rsx_raw_vertices.Load(rsx_raw_vertices.Load(0u)+vertex_id*4u);\n",
        entry + strlen(signature), rsx_raw_vertex_hlsl);
    size_t used = strlen(output);
    for (u32 a = 0; a < 16; ++a) {
        char semantic[48];
        snprintf(semantic, sizeof(semantic), "a%u:ATTR%u", a, a);
        if (!(mask & (1u << a)) || !strstr(source, semantic)) continue;
        const int n = snprintf(output + used, capacity - used,
            "input.a%u=rsx_raw_attribute(%u,vi);\n", a, a);
        if (n < 0 || (size_t)n >= capacity - used) return 0;
        used += (size_t)n;
    }
    return snprintf(output + used, capacity - used,
        "return rsx_raw_guest_main(input);}\n") > 0;
}

#endif
