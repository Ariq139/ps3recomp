/*
 * ps3recomp - cellPamf HLE implementation.
 *
 * PAMF layouts and return behavior were checked against RPCS3's
 * rpcs3/Emu/Cell/Modules/cellPamf.h and cellPamf.cpp (GPL-2.0 reference).
 * This implementation reads guest bytes by their documented offsets and
 * writes the guest ABI explicitly; no RPCS3 function body is copied.
 */

#include "cellPamf.h"
#include "../../runtime/ppu/ppu_memory.h"
#include <stdint.h>
#include <string.h>

/* PAMF file structure offsets. All multibyte values in the file are BE. */
#define PAMF_SEQUENCE_OFFSET         0x50u
#define PAMF_SEQUENCE_FIXED_SIZE     0x20u
#define PAMF_GROUP_PERIOD_OFFSET     0x70u
#define PAMF_GROUP_OFFSET            0x82u
#define PAMF_STREAMS_OFFSET          0x88u
#define PAMF_STREAM_SIZE             48u
#define PAMF_EP_SIZE                 12u
#define PAMF_BLOCK_SIZE              0x800ull
#define PAMF_READER_GUEST_SIZE       128u

#define PAMF_H_MAGIC                 0x00u
#define PAMF_H_VERSION               0x04u
#define PAMF_H_HEADER_BLOCKS         0x08u
#define PAMF_H_DATA_BLOCKS           0x0Cu
#define PAMF_H_MARKS_OFFSET          0x10u
#define PAMF_H_MARKS_SIZE            0x14u
#define PAMF_H_UNK_OFFSET            0x18u
#define PAMF_H_UNK_SIZE              0x1Cu

#define PAMF_SEQ_SIZE                0x50u
#define PAMF_SEQ_START_HI            0x56u
#define PAMF_SEQ_START_LO            0x58u
#define PAMF_SEQ_END_HI              0x5Cu
#define PAMF_SEQ_END_LO              0x5Eu
#define PAMF_SEQ_MUX_RATE            0x62u
#define PAMF_SEQ_STREAM_COUNT        0x6Au
#define PAMF_SEQ_GROUP_PERIOD_COUNT  0x6Fu

#define PAMF_PERIOD_SIZE             0x70u
#define PAMF_PERIOD_START_HI         0x74u
#define PAMF_PERIOD_START_LO         0x76u
#define PAMF_PERIOD_END_LO           0x7Cu
#define PAMF_PERIOD_GROUP_COUNT      0x81u

#define PAMF_GROUP_SIZE              0x82u
#define PAMF_GROUP_STREAM_COUNT      0x87u
#define PAMF_STREAM_CODING_TYPE      0x00u
#define PAMF_STREAM_ID               0x04u
#define PAMF_STREAM_PRIVATE_ID       0x05u
#define PAMF_STREAM_EP_OFFSET        0x08u
#define PAMF_STREAM_EP_COUNT         0x0Cu
#define PAMF_STREAM_INFO             0x10u

#define PAMF_READER_ATTR             0x10u
#define PAMF_READER_IS_PSMF          0x14u
#define PAMF_READER_VERSION          0x16u
#define PAMF_READER_GROUP_INDEX      0x1Cu
#define PAMF_READER_STREAM_INDEX     0x20u
#define PAMF_READER_HEADER_PTR       0x24u
#define PAMF_READER_SEQUENCE_PTR     0x28u
#define PAMF_READER_PERIOD_PTR       0x2Cu
#define PAMF_READER_GROUP_PTR        0x30u
#define PAMF_READER_STREAM_PTR       0x34u

typedef struct PamfHeaderView {
    u32 ea;
    const u8* bytes;
    u64 file_size;
    u64 header_size;
    u64 data_size;
    u32 version;
} PamfHeaderView;

typedef struct PamfReaderView {
    u32 ea;
    u32 header_ea;
    u32 sequence_ea;
    u32 period_ea;
    u32 group_ea;
    u32 stream_ea;
    u32 attributes;
    u32 stream_index;
    u64 header_size;
    u64 data_size;
} PamfReaderView;

static u16 pamf_be16(const u8* p)
{
    return (u16)(((u16)p[0] << 8) | (u16)p[1]);
}

static u32 pamf_be32(const u8* p)
{
    return ((u32)p[0] << 24) | ((u32)p[1] << 16) |
           ((u32)p[2] << 8) | (u32)p[3];
}

static int pamf_range_ok(u64 limit, u64 offset, u64 size)
{
    return offset <= limit && size <= limit - offset;
}

static const u8* pamf_guest_bytes(u32 ea, u64 offset)
{
    if (!vm_base || offset > 0xFFFFFFFFull - (u64)ea)
        return NULL;
    return vm_base + ea + (u32)offset;
}

static u8 pamf_same_type(u8 actual, u8 requested)
{
    switch (requested) {
    case CELL_PAMF_STREAM_TYPE_VIDEO:
        return (u8)(actual == CELL_PAMF_STREAM_TYPE_AVC ||
                    actual == CELL_PAMF_STREAM_TYPE_M2V);
    case CELL_PAMF_STREAM_TYPE_AUDIO:
        return (u8)(actual == CELL_PAMF_STREAM_TYPE_ATRAC3PLUS ||
                    actual == CELL_PAMF_STREAM_TYPE_AC3 ||
                    actual == CELL_PAMF_STREAM_TYPE_PAMF_LPCM);
    case CELL_PAMF_STREAM_TYPE_UNK:
        return (u8)(actual == CELL_PAMF_STREAM_TYPE_PAMF_LPCM ||
                    actual == CELL_PAMF_STREAM_TYPE_PSMF_ATRAC3PLUS);
    default:
        return (u8)(actual == requested);
    }
}

static s32 pamf_decode_stream(u8 coding, u8 stream_id, u8 private_id,
                              u8* type, u8* channel)
{
    u8 decoded_type, decoded_channel;
    switch (coding) {
    case CELL_PAMF_CODEC_TYPE_AVC:
        decoded_type = CELL_PAMF_STREAM_TYPE_AVC;
        decoded_channel = stream_id & 0x0Fu;
        break;
    case CELL_PAMF_CODEC_TYPE_M2V:
        decoded_type = CELL_PAMF_STREAM_TYPE_M2V;
        decoded_channel = stream_id & 0x0Fu;
        break;
    case CELL_PAMF_CODEC_TYPE_ATRAC3PLUS:
        decoded_type = CELL_PAMF_STREAM_TYPE_ATRAC3PLUS;
        decoded_channel = private_id & 0x0Fu;
        break;
    case CELL_PAMF_CODEC_TYPE_PAMF_LPCM:
        decoded_type = CELL_PAMF_STREAM_TYPE_PAMF_LPCM;
        decoded_channel = private_id & 0x0Fu;
        break;
    case CELL_PAMF_CODEC_TYPE_AC3:
        decoded_type = CELL_PAMF_STREAM_TYPE_AC3;
        decoded_channel = private_id & 0x0Fu;
        break;
    case CELL_PAMF_CODEC_TYPE_USER_DATA:
        decoded_type = CELL_PAMF_STREAM_TYPE_USER_DATA;
        decoded_channel = private_id & 0x0Fu;
        break;
    case CELL_PAMF_CODEC_TYPE_PSMF:
        if ((stream_id & 0xF0u) == 0xE0u && private_id == 0) {
            decoded_type = CELL_PAMF_STREAM_TYPE_PSMF_AVC;
            decoded_channel = stream_id & 0x0Fu;
        } else if (stream_id == 0xBDu) {
            decoded_channel = private_id & 0x0Fu;
            switch (private_id & 0xF0u) {
            case 0x00u: decoded_type = CELL_PAMF_STREAM_TYPE_PSMF_ATRAC3PLUS; break;
            case 0x10u: decoded_type = CELL_PAMF_STREAM_TYPE_PSMF_LPCM; break;
            /* RPCS3 maps this reserved PSMF substream class to PSMF LPCM. */
            case 0x20u: decoded_type = CELL_PAMF_STREAM_TYPE_PSMF_LPCM; break;
            default: return (s32)CELL_PAMF_ERROR_STREAM_NOT_FOUND;
            }
        } else {
            return (s32)CELL_PAMF_ERROR_STREAM_NOT_FOUND;
        }
        break;
    default:
        return (s32)CELL_PAMF_ERROR_STREAM_NOT_FOUND;
    }

    if (type) *type = decoded_type;
    if (channel) *channel = decoded_channel;
    return CELL_OK;
}

static s32 pamf_read_header(u32 pamf_ea, u64 file_size,
                            PamfHeaderView* out)
{
    if (!pamf_ea || !out || !vm_base)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (file_size && file_size < PAMF_H_DATA_BLOCKS + 4u)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u8* base = pamf_guest_bytes(pamf_ea, 0);
    if (!base)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (pamf_be32(base + PAMF_H_MAGIC) != CELL_PAMF_MAGIC)
        return (s32)CELL_PAMF_ERROR_UNKNOWN_TYPE;

    const u32 version = pamf_be32(base + PAMF_H_VERSION);
    if (version != CELL_PAMF_VERSION_0040 && version != CELL_PAMF_VERSION_0041)
        return (s32)CELL_PAMF_ERROR_UNSUPPORTED_VERSION;

    const u64 header_size = (u64)pamf_be32(base + PAMF_H_HEADER_BLOCKS) * PAMF_BLOCK_SIZE;
    const u64 data_size = (u64)pamf_be32(base + PAMF_H_DATA_BLOCKS) * PAMF_BLOCK_SIZE;
    if (!header_size || (file_size && header_size + data_size != file_size))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    out->ea = pamf_ea;
    out->bytes = base;
    out->file_size = file_size;
    out->header_size = header_size;
    out->data_size = data_size;
    out->version = version == CELL_PAMF_VERSION_0040 ? 40u : 41u;
    return CELL_OK;
}

static s32 pamf_parse_ep(const PamfHeaderView* header, u32 ep_offset,
                         u32 ep_index, CellPamfEp* out)
{
    const u64 off = (u64)ep_offset + (u64)ep_index * PAMF_EP_SIZE;
    if (!header || !out || !ep_offset ||
        !pamf_range_ok(header->header_size, off, PAMF_EP_SIZE))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    if (header->file_size && !pamf_range_ok(header->file_size, off, PAMF_EP_SIZE))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u8* ep = header->bytes + off;
    const u16 value0 = pamf_be16(ep + 0);
    out->indexN = (u32)(value0 >> 14) + 1u;
    out->nThRefPictureOffset = ((u32)(value0 & 0x1FFFu) * 0x800u) + 0x800u;
    out->pts.upper = pamf_be16(ep + 2);
    out->pts.lower = pamf_be32(ep + 4);
    out->rpnOffset = (u64)pamf_be32(ep + 8) * PAMF_BLOCK_SIZE;
    return CELL_OK;
}

static void pamf_write_ep(u32 ep_ea, const CellPamfEp* ep)
{
    vm_write32(ep_ea + 0, ep->indexN);
    vm_write32(ep_ea + 4, ep->nThRefPictureOffset);
    vm_write32(ep_ea + 8, ep->pts.upper);
    vm_write32(ep_ea + 12, ep->pts.lower);
    vm_write64(ep_ea + 16, ep->rpnOffset);
}

static int pamf_reader_view(u32 reader_ea, PamfReaderView* out)
{
    if (!reader_ea || !out || !vm_base)
        return 0;
    out->ea = reader_ea;
    out->header_size = vm_read64(reader_ea + 0);
    out->data_size = vm_read64(reader_ea + 8);
    out->attributes = vm_read32(reader_ea + PAMF_READER_ATTR);
    if (vm_read16(reader_ea + PAMF_READER_IS_PSMF) != 0)
        return 0; /* this HLE accepts PAMF, not standalone PSMF files */
    out->stream_index = vm_read32(reader_ea + PAMF_READER_STREAM_INDEX);
    out->header_ea = vm_read32(reader_ea + PAMF_READER_HEADER_PTR);
    out->sequence_ea = vm_read32(reader_ea + PAMF_READER_SEQUENCE_PTR);
    out->period_ea = vm_read32(reader_ea + PAMF_READER_PERIOD_PTR);
    out->group_ea = vm_read32(reader_ea + PAMF_READER_GROUP_PTR);
    out->stream_ea = vm_read32(reader_ea + PAMF_READER_STREAM_PTR);
    return out->header_ea != 0;
}

static u32 pamf_stream_count(const PamfReaderView* reader)
{
    return reader && reader->group_ea
        ? vm_read8(reader->group_ea + 5u) : 0u;
}

static u32 pamf_stream_ea(const PamfReaderView* reader, u32 index)
{
    if (!reader || !reader->group_ea || index >= pamf_stream_count(reader))
        return 0;
    const u64 ea = (u64)reader->group_ea + 6u + (u64)index * PAMF_STREAM_SIZE;
    if (ea > 0xFFFFFFFFull || !pamf_guest_bytes((u32)ea, 0))
        return 0;
    return (u32)ea;
}

static s32 pamf_verify(const PamfHeaderView* h, u32 attribute)
{
    if (!h || !h->bytes)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u8* p = h->bytes;
    const u64 header_size = h->header_size;
    const u64 data_size = h->data_size;
    const u32 marks_off = pamf_be32(p + PAMF_H_MARKS_OFFSET);
    const u32 marks_size = pamf_be32(p + PAMF_H_MARKS_SIZE);
    const u32 unk_off = pamf_be32(p + PAMF_H_UNK_OFFSET);
    const u32 unk_size = pamf_be32(p + PAMF_H_UNK_SIZE);

    if ((marks_off == 0) != (marks_size == 0) ||
        (marks_off && !pamf_range_ok(header_size, marks_off, marks_size)) ||
        (unk_off == 0) != (unk_size == 0) ||
        (unk_off && !pamf_range_ok(header_size, unk_off, unk_size)) ||
        (u64)unk_off < (u64)marks_off + marks_size)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    if (!pamf_range_ok(header_size, PAMF_SEQUENCE_OFFSET,
                       PAMF_SEQUENCE_FIXED_SIZE + 4u))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    const u32 seq_size = pamf_be32(p + PAMF_SEQ_SIZE);
    if ((u64)PAMF_SEQUENCE_OFFSET + 4u + seq_size > header_size)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u64 start_pts = ((u64)pamf_be16(p + PAMF_SEQ_START_HI) << 32) |
                           pamf_be32(p + PAMF_SEQ_START_LO);
    const u64 end_pts = ((u64)pamf_be16(p + PAMF_SEQ_END_HI) << 32) |
                         pamf_be32(p + PAMF_SEQ_END_LO);
    if (end_pts > 0xFFFFFFFFull || start_pts >= end_pts ||
        p[PAMF_SEQ_GROUP_PERIOD_COUNT] != 1)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u32 period_size = pamf_be32(p + PAMF_PERIOD_SIZE);
    if (period_size < 0x12u || seq_size < PAMF_SEQUENCE_FIXED_SIZE ||
        !pamf_range_ok(header_size, PAMF_GROUP_PERIOD_OFFSET, 0x16u))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    if ((u64)PAMF_GROUP_PERIOD_OFFSET + 4u + period_size > header_size)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    const u64 period_start = ((u64)pamf_be16(p + PAMF_PERIOD_START_HI) << 32) |
                              pamf_be32(p + PAMF_PERIOD_START_LO);
    /* RPCS3 follows the SDK's high-word handling for this end timestamp. */
    const u64 period_end = ((u64)pamf_be16(p + PAMF_PERIOD_START_HI) << 32) |
                            pamf_be32(p + PAMF_PERIOD_END_LO);
    if (period_end > 0xFFFFFFFFull || period_start >= period_end ||
        period_start != start_pts || p[PAMF_PERIOD_GROUP_COUNT] != 1)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    if (!pamf_range_ok(header_size, PAMF_GROUP_OFFSET, 6u))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    const u32 group_size = pamf_be32(p + PAMF_GROUP_SIZE);
    const u8 stream_num = p[PAMF_GROUP_STREAM_COUNT];
    if (!stream_num ||
        (u64)PAMF_GROUP_OFFSET + 4u + group_size >
            (u64)PAMF_GROUP_PERIOD_OFFSET + 4u + period_size ||
        group_size < 2u + (u64)stream_num * PAMF_STREAM_SIZE ||
        !pamf_range_ok(header_size, PAMF_STREAMS_OFFSET,
                       (u64)stream_num * PAMF_STREAM_SIZE))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    u16 used_channels[10] = {0};
    u32 next_ep_table = 0;
    u32 first_data_tail = 0;
    for (u32 i = 0; i < stream_num; i++) {
        const u8* stream = p + PAMF_STREAMS_OFFSET + i * PAMF_STREAM_SIZE;
        u8 type = 0, channel = 0;
        if (pamf_decode_stream(stream[PAMF_STREAM_CODING_TYPE],
                               stream[PAMF_STREAM_ID],
                               stream[PAMF_STREAM_PRIVATE_ID],
                               &type, &channel) != CELL_OK)
            return (s32)CELL_PAMF_ERROR_UNKNOWN_STREAM;
        if (type >= 10u || (used_channels[type] & (u16)(1u << channel)))
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
        used_channels[type] |= (u16)(1u << channel);

        const u32 ep_off = pamf_be32(stream + PAMF_STREAM_EP_OFFSET);
        const u32 ep_count = pamf_be32(stream + PAMF_STREAM_EP_COUNT);
        const u64 ep_bytes = (u64)ep_count * PAMF_EP_SIZE;
        if ((!ep_count && ep_off) || (ep_count && !ep_off) ||
            (ep_count && !pamf_range_ok(header_size, ep_off, ep_bytes)))
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
        if (!ep_count || (attribute & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER))
            continue;

        if (!first_data_tail) {
            first_data_tail = ep_off;
            next_ep_table = ep_off;
        } else if (next_ep_table != ep_off) {
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
        }

        u32 previous_rpn = 0;
        for (u32 n = 0; n < ep_count; n++) {
            const u8* ep = p + ep_off + (u64)n * PAMF_EP_SIZE;
            const u64 pts = ((u64)pamf_be16(ep + 2) << 32) | pamf_be32(ep + 4);
            const u32 rpn = pamf_be32(ep + 8);
            if (pts > 0xFFFFFFFFull || (u64)rpn * PAMF_BLOCK_SIZE > data_size ||
                (n && rpn < previous_rpn))
                return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
            previous_rpn = rpn;
        }
        next_ep_table += (u32)ep_bytes;
    }

    if ((u64)group_size + 4u > (u64)period_size - 0x12u + 4u ||
        (u64)period_size + 4u > (u64)seq_size - PAMF_SEQUENCE_FIXED_SIZE + 4u ||
        pamf_be32(p + PAMF_SEQ_STREAM_COUNT) != stream_num)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    if (!first_data_tail) {
        if (marks_off) first_data_tail = marks_off;
        else if (unk_off) first_data_tail = unk_off;
    }
    const u32 grouping_end = PAMF_GROUP_PERIOD_OFFSET + 4u + period_size;
    if (first_data_tail && first_data_tail != grouping_end)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    if (next_ep_table && !marks_off) {
        if (unk_off && unk_off != next_ep_table)
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    } else if (next_ep_table && marks_off != next_ep_table) {
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    } else if (marks_off && !(attribute & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER)) {
        if (!pamf_range_ok(header_size, marks_off, marks_size) || marks_size < 8u)
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
        const u32 marks_payload = pamf_be32(p + marks_off);
        const u16 marks_count = pamf_be16(p + marks_off + 6u);
        if ((u64)marks_payload + 4u != marks_size ||
            2ull + (u64)marks_count * 0x28ull != marks_payload)
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    }
    return CELL_OK;
}

s32 cellPamfGetStreamOffsetAndSize(void* pamfAddr, u64 fileSize,
                                    u64* headerSize, u64* dataSize)
{
    PamfHeaderView h;
    s32 rc = pamf_read_header((u32)(uintptr_t)pamfAddr, fileSize, &h);
    if (rc != CELL_OK)
        return rc;
    if (headerSize)
        vm_write64((u32)(uintptr_t)headerSize, h.header_size);
    if (dataSize)
        vm_write64((u32)(uintptr_t)dataSize, h.data_size);
    return CELL_OK;
}

s32 cellPamfGetHeaderSize(void* pamfAddr, u64 fileSize, u64* headerSize)
{
    return cellPamfGetStreamOffsetAndSize(pamfAddr, fileSize, headerSize, NULL);
}

s32 cellPamfGetHeaderSize2(void* pamfAddr, u64 fileSize, u32 attribute,
                            u64* headerSize)
{
    PamfHeaderView h;
    s32 rc = pamf_read_header((u32)(uintptr_t)pamfAddr, fileSize, &h);
    if (rc != CELL_OK)
        return rc;
    u64 result = h.header_size;
    if (attribute & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER) {
        if (!pamf_range_ok(h.header_size, PAMF_SEQUENCE_OFFSET, 4u))
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
        const u32 seq_size = pamf_be32(h.bytes + PAMF_SEQ_SIZE);
        result = PAMF_SEQUENCE_OFFSET + 4ull + seq_size;
        if (result > h.header_size)
            return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    }
    if (headerSize)
        vm_write64((u32)(uintptr_t)headerSize, result);
    return CELL_OK;
}

s32 cellPamfVerify(void* pamfAddr, u64 fileSize)
{
    PamfHeaderView h;
    s32 rc = pamf_read_header((u32)(uintptr_t)pamfAddr, fileSize, &h);
    return rc == CELL_OK ? pamf_verify(&h, CELL_PAMF_ATTRIBUTE_VERIFY_ON) : rc;
}

static void pamf_clear_reader(u32 reader_ea)
{
    for (u32 i = 0; i < PAMF_READER_GUEST_SIZE; i += 4u)
        vm_write32(reader_ea + i, 0);
}

s32 cellPamfReaderInitialize(CellPamfReader* reader, void* pamfAddr,
                              u64 fileSize, u32 attribute)
{
    const u32 reader_ea = (u32)(uintptr_t)reader;
    const u32 pamf_ea = (u32)(uintptr_t)pamfAddr;
    if (!reader_ea || !pamf_ea || !vm_base)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;

    pamf_clear_reader(reader_ea);
    vm_write32(reader_ea + PAMF_READER_ATTR, attribute);

    PamfHeaderView h;
    s32 rc = pamf_read_header(pamf_ea, fileSize, &h);
    if (rc != CELL_OK)
        return rc;
    if (attribute & CELL_PAMF_ATTRIBUTE_VERIFY_ON) {
        rc = pamf_verify(&h, attribute);
        if (rc != CELL_OK)
            return rc;
    }

    if (!pamf_range_ok(h.header_size, PAMF_GROUP_PERIOD_OFFSET, 0x16u) ||
        !pamf_range_ok(h.header_size, PAMF_GROUP_OFFSET, 6u))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    if (h.bytes[PAMF_SEQ_GROUP_PERIOD_COUNT] == 0)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u32 period_size = pamf_be32(h.bytes + PAMF_PERIOD_SIZE);
    if ((u64)PAMF_GROUP_PERIOD_OFFSET + 4u + period_size > h.header_size)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    const u32 group_size = pamf_be32(h.bytes + PAMF_GROUP_SIZE);
    if ((u64)PAMF_GROUP_OFFSET + 4u + group_size >
        (u64)PAMF_GROUP_PERIOD_OFFSET + 4u + period_size)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    const u32 stream_count = h.bytes[PAMF_GROUP_STREAM_COUNT];
    if (group_size < 2u + (u64)stream_count * PAMF_STREAM_SIZE ||
        !pamf_range_ok(h.header_size, PAMF_STREAMS_OFFSET,
                       (u64)stream_count * PAMF_STREAM_SIZE))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u32 sequence_ea = pamf_ea + PAMF_SEQUENCE_OFFSET;
    const u32 period_ea = pamf_ea + PAMF_GROUP_PERIOD_OFFSET;
    const u32 group_ea = h.bytes[PAMF_PERIOD_GROUP_COUNT]
        ? pamf_ea + PAMF_GROUP_OFFSET : 0;
    vm_write64(reader_ea + 0, h.header_size);
    vm_write64(reader_ea + 8, h.data_size);
    vm_write16(reader_ea + PAMF_READER_IS_PSMF, 0);
    vm_write16(reader_ea + PAMF_READER_VERSION, (u16)h.version);
    vm_write32(reader_ea + 0x18u, 0u);
    vm_write32(reader_ea + PAMF_READER_GROUP_INDEX, group_ea ? 0u : 0xFFFFFFFFu);
    vm_write32(reader_ea + PAMF_READER_STREAM_INDEX,
               group_ea && stream_count ? 0u : 0xFFFFFFFFu);
    vm_write32(reader_ea + PAMF_READER_HEADER_PTR, pamf_ea);
    vm_write32(reader_ea + PAMF_READER_SEQUENCE_PTR, sequence_ea);
    vm_write32(reader_ea + PAMF_READER_PERIOD_PTR, period_ea);
    vm_write32(reader_ea + PAMF_READER_GROUP_PTR, group_ea);
    if (group_ea && stream_count)
        vm_write32(reader_ea + PAMF_READER_STREAM_PTR,
                   pamf_ea + PAMF_STREAMS_OFFSET);
    return CELL_OK;
}

static int pamf_load_reader(u32 reader_ea, PamfReaderView* reader)
{
    if (!pamf_reader_view(reader_ea, reader))
        return 0;
    return reader->sequence_ea != 0;
}

static u64 pamf_read_timestamp(u32 timestamp_ea)
{
    return ((u64)vm_read32(timestamp_ea) << 32) | vm_read32(timestamp_ea + 4u);
}

s32 cellPamfReaderGetPresentationStartTime(CellPamfReader* reader,
                                            CellCodecTimeStamp* startTime)
{
    PamfReaderView v;
    const u32 out = (u32)(uintptr_t)startTime;
    if (!out || !pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u8* seq = pamf_guest_bytes(v.sequence_ea, 0);
    if (!seq) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    vm_write32(out, pamf_be16(seq + PAMF_SEQ_START_HI));
    vm_write32(out + 4u, pamf_be32(seq + PAMF_SEQ_START_LO));
    return CELL_OK;
}

s32 cellPamfReaderGetPresentationEndTime(CellPamfReader* reader,
                                          CellCodecTimeStamp* endTime)
{
    PamfReaderView v;
    const u32 out = (u32)(uintptr_t)endTime;
    if (!out || !pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u8* seq = pamf_guest_bytes(v.sequence_ea, 0);
    if (!seq) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    vm_write32(out, pamf_be16(seq + PAMF_SEQ_END_HI));
    vm_write32(out + 4u, pamf_be32(seq + PAMF_SEQ_END_LO));
    return CELL_OK;
}

u32 cellPamfReaderGetMuxRateBound(CellPamfReader* reader)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v)) return 0;
    const u8* seq = pamf_guest_bytes(v.sequence_ea, 0);
    return seq ? pamf_be32(seq + PAMF_SEQ_MUX_RATE) & 0x003FFFFFu : 0;
}

u8 cellPamfReaderGetNumberOfStreams(CellPamfReader* reader)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v)) return 0;
    return (u8)pamf_stream_count(&v);
}

u8 cellPamfReaderGetNumberOfSpecificStreams(CellPamfReader* reader,
                                             u8 streamType)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v)) return 0;
    u8 count = 0;
    const u32 stream_count = pamf_stream_count(&v);
    for (u32 i = 0; i < stream_count; i++) {
        const u32 stream_ea = pamf_stream_ea(&v, i);
        const u8* stream = stream_ea ? pamf_guest_bytes(stream_ea, 0) : NULL;
        u8 type;
        if (stream && pamf_decode_stream(stream[PAMF_STREAM_CODING_TYPE],
                stream[PAMF_STREAM_ID], stream[PAMF_STREAM_PRIVATE_ID],
                &type, NULL) == CELL_OK && pamf_same_type(type, streamType))
            count++;
    }
    return count;
}

s32 cellPamfReaderSetStreamWithIndex(CellPamfReader* reader, u8 streamIndex)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u32 stream_count = pamf_stream_count(&v);
    if (streamIndex >= stream_count)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u32 stream_ea = pamf_stream_ea(&v, streamIndex);
    if (!stream_ea) return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    vm_write32(v.ea + PAMF_READER_STREAM_INDEX, streamIndex);
    vm_write32(v.ea + PAMF_READER_STREAM_PTR, stream_ea);
    return CELL_OK;
}

s32 cellPamfReaderSetStreamWithTypeAndChannel(CellPamfReader* reader,
                                               u8 streamType, u8 channel)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v) || channel >= 16u)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u32 stream_count = pamf_stream_count(&v);
    for (u32 i = 0; i < stream_count; i++) {
        const u32 stream_ea = pamf_stream_ea(&v, i);
        const u8* stream = stream_ea ? pamf_guest_bytes(stream_ea, 0) : NULL;
        u8 type, found_channel;
        if (stream && pamf_decode_stream(stream[PAMF_STREAM_CODING_TYPE],
                stream[PAMF_STREAM_ID], stream[PAMF_STREAM_PRIVATE_ID],
                &type, &found_channel) == CELL_OK && type == streamType &&
            found_channel == channel) {
            vm_write32(v.ea + PAMF_READER_STREAM_INDEX, i);
            vm_write32(v.ea + PAMF_READER_STREAM_PTR, stream_ea);
            return (s32)i;
        }
    }
    return (s32)CELL_PAMF_ERROR_STREAM_NOT_FOUND;
}

s32 cellPamfReaderSetStreamWithTypeAndIndex(CellPamfReader* reader,
                                             u8 streamType, u8 streamIndex)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (streamIndex >= pamf_stream_count(&v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    u32 found = 0;
    const u32 stream_count = pamf_stream_count(&v);
    for (u32 i = 0; i < stream_count; i++) {
        const u32 stream_ea = pamf_stream_ea(&v, i);
        const u8* stream = stream_ea ? pamf_guest_bytes(stream_ea, 0) : NULL;
        u8 type;
        if (!stream || pamf_decode_stream(stream[PAMF_STREAM_CODING_TYPE],
                stream[PAMF_STREAM_ID], stream[PAMF_STREAM_PRIVATE_ID],
                &type, NULL) != CELL_OK || !pamf_same_type(type, streamType))
            continue;
        if (found++ == streamIndex) {
            vm_write32(v.ea + PAMF_READER_STREAM_INDEX, i);
            vm_write32(v.ea + PAMF_READER_STREAM_PTR, stream_ea);
            return (s32)i;
        }
    }
    return (s32)CELL_PAMF_ERROR_STREAM_NOT_FOUND;
}

static s32 pamf_type_channel_to_ids(u8 type, u8 channel, u8* stream_id,
                                    u8* private_id, u32* supplemental)
{
    if (channel >= 16u) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    u8 sid = 0, pid = 0;
    u32 extra = 0;
    switch (type) {
    case CELL_PAMF_STREAM_TYPE_AVC:
    case CELL_PAMF_STREAM_TYPE_M2V:
        sid = (u8)(0xE0u | channel); extra = type == CELL_PAMF_STREAM_TYPE_AVC; break;
    case CELL_PAMF_STREAM_TYPE_ATRAC3PLUS: pid = channel; sid = 0xBD; break;
    case CELL_PAMF_STREAM_TYPE_PAMF_LPCM: pid = (u8)(0x40u | channel); sid = 0xBD; break;
    case CELL_PAMF_STREAM_TYPE_AC3: pid = (u8)(0x30u | channel); sid = 0xBD; break;
    case CELL_PAMF_STREAM_TYPE_USER_DATA: pid = (u8)(0x20u | channel); sid = 0xBD; break;
    case CELL_PAMF_STREAM_TYPE_PSMF_AVC: sid = (u8)(0xE0u | channel); break;
    case CELL_PAMF_STREAM_TYPE_PSMF_ATRAC3PLUS: pid = channel; sid = 0xBD; break;
    case CELL_PAMF_STREAM_TYPE_PSMF_LPCM: pid = (u8)(0x10u | channel); sid = 0xBD; break;
    case CELL_PAMF_STREAM_TYPE_PSMF_USER_DATA: pid = (u8)(0x20u | channel); sid = 0xBD; break;
    default: return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    }
    if (stream_id) *stream_id = sid;
    if (private_id) *private_id = pid;
    if (supplemental) *supplemental = extra;
    return CELL_OK;
}

s32 cellPamfStreamTypeToEsFilterId(u8 streamType, u8 channel,
                                    CellCodecEsFilterId* esFilterId)
{
    const u32 out = (u32)(uintptr_t)esFilterId;
    if (!out) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    u8 stream_id, private_id;
    u32 supplemental;
    s32 rc = pamf_type_channel_to_ids(streamType, channel, &stream_id,
                                      &private_id, &supplemental);
    if (rc != CELL_OK) return rc;
    vm_write32(out + 0u, stream_id);
    vm_write32(out + 4u, private_id);
    vm_write32(out + 8u, supplemental);
    vm_write32(out + 12u, 0);
    return CELL_OK;
}

s32 cellPamfReaderGetStreamIndex(CellPamfReader* reader)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    return (s32)v.stream_index;
}

s32 cellPamfReaderGetStreamTypeAndChannel(CellPamfReader* reader,
                                           u8* streamType, u8* channel)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v) || !v.stream_ea)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u8* stream = pamf_guest_bytes(v.stream_ea, 0);
    if (!stream) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    u8 type, ch;
    s32 rc = pamf_decode_stream(stream[PAMF_STREAM_CODING_TYPE],
        stream[PAMF_STREAM_ID], stream[PAMF_STREAM_PRIVATE_ID], &type, &ch);
    if (rc != CELL_OK) return rc;
    if (streamType) vm_write8((u32)(uintptr_t)streamType, type);
    if (channel) vm_write8((u32)(uintptr_t)channel, ch);
    return CELL_OK;
}

s32 cellPamfReaderGetEsFilterId(CellPamfReader* reader,
                                 CellCodecEsFilterId* esFilterId)
{
    PamfReaderView v;
    const u32 out = (u32)(uintptr_t)esFilterId;
    if (!out || !pamf_load_reader((u32)(uintptr_t)reader, &v) || !v.stream_ea)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u8* stream = pamf_guest_bytes(v.stream_ea, 0);
    if (!stream) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    vm_write32(out + 0u, stream[PAMF_STREAM_ID]);
    vm_write32(out + 4u, stream[PAMF_STREAM_PRIVATE_ID]);
    vm_write32(out + 8u,
        stream[PAMF_STREAM_CODING_TYPE] == CELL_PAMF_CODEC_TYPE_AVC ? 1u : 0u);
    vm_write32(out + 12u, 0);
    return CELL_OK;
}

static void pamf_zero_guest(u32 dst, u32 size)
{
    for (u32 i = 0; i < size; i += 4u)
        vm_write32(dst + i, 0);
}

s32 cellPamfReaderGetStreamInfo(CellPamfReader* reader, void* info,
                                 u32 infoSize)
{
    PamfReaderView v;
    const u32 out = (u32)(uintptr_t)info;
    if (!out || !pamf_load_reader((u32)(uintptr_t)reader, &v) || !v.stream_ea)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u8* stream = pamf_guest_bytes(v.stream_ea, 0);
    if (!stream) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    u8 type;
    if (pamf_decode_stream(stream[PAMF_STREAM_CODING_TYPE],
            stream[PAMF_STREAM_ID], stream[PAMF_STREAM_PRIVATE_ID],
            &type, NULL) != CELL_OK)
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;

    const u8* specific = stream + PAMF_STREAM_INFO;
    if (type == CELL_PAMF_STREAM_TYPE_AVC) {
        if (infoSize < 0x20u) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
        pamf_zero_guest(out, 0x20u);
        vm_write8(out + 0, specific[0]);
        vm_write8(out + 1, specific[1]);
        vm_write8(out + 2, (u8)((specific[2] >> 7) & 1u));
        vm_write8(out + 3, (u8)((specific[2] >> 6) & 1u));
        vm_write8(out + 4, (u8)((specific[2] & 0x0Fu) - 1u));
        vm_write8(out + 5, specific[3]);
        if (specific[3] == 0xFFu) {
            vm_write16(out + 6, pamf_be16(specific + 4));
            vm_write16(out + 8, pamf_be16(specific + 6));
        }
        vm_write16(out + 10, (u16)((u16)specific[9] * 16u));
        vm_write16(out + 12, (u16)((u16)specific[11] * 16u));
        vm_write16(out + 14, pamf_be16(specific + 12));
        vm_write16(out + 16, pamf_be16(specific + 14));
        vm_write16(out + 18, pamf_be16(specific + 16));
        vm_write16(out + 20, pamf_be16(specific + 18));
        if (specific[2] & 0x40u) {
            vm_write8(out + 22, (u8)(specific[20] >> 5));
            vm_write8(out + 23, (u8)((specific[20] >> 4) & 1u));
            vm_write8(out + 24, specific[21]);
            vm_write8(out + 25, specific[22]);
            vm_write8(out + 26, specific[23]);
        }
        vm_write8(out + 27, (u8)((specific[24] >> 7) & 1u));
        vm_write8(out + 28, (u8)((specific[24] >> 6) & 1u));
        vm_write8(out + 29, (u8)((specific[24] >> 4) & 3u));
        vm_write8(out + 30, (u8)(specific[24] & 3u));
        vm_write8(out + 31, specific[25]);
        return CELL_OK;
    }
    if (type == CELL_PAMF_STREAM_TYPE_M2V) {
        if (infoSize < 0x18u) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
        pamf_zero_guest(out, 0x18u);
        vm_write8(out + 0, specific[0] == 0x44u ? 3u :
                            specific[0] == 0x48u ? 1u : 0xFFu);
        vm_write8(out + 1, (u8)((specific[2] >> 7) & 1u));
        vm_write8(out + 2, (u8)((specific[2] >> 6) & 1u));
        vm_write8(out + 3, (u8)(specific[2] & 0x0Fu));
        vm_write8(out + 4, specific[3]);
        if (specific[3] == 0xFFu) {
            vm_write16(out + 6, pamf_be16(specific + 4));
            vm_write16(out + 8, pamf_be16(specific + 6));
        }
        vm_write16(out + 10, (u16)((u16)specific[9] * 16u));
        vm_write16(out + 12, (u16)((u16)specific[11] * 16u));
        vm_write16(out + 14, pamf_be16(specific + 12));
        vm_write16(out + 16, pamf_be16(specific + 14));
        if (specific[2] & 0x40u) {
            vm_write8(out + 18, (u8)(specific[20] >> 5));
            vm_write8(out + 19, (u8)((specific[20] >> 4) & 1u));
            vm_write8(out + 20, specific[21]);
            vm_write8(out + 21, specific[22]);
            vm_write8(out + 22, specific[23]);
        }
        return CELL_OK;
    }
    if (type == CELL_PAMF_STREAM_TYPE_ATRAC3PLUS ||
        type == CELL_PAMF_STREAM_TYPE_AC3) {
        if (infoSize < 8u) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
        pamf_zero_guest(out, 8u);
        vm_write32(out, specific[3] & 0x0Fu);
        vm_write8(out + 4u, specific[2] & 0x0Fu);
        return CELL_OK;
    }
    if (type == CELL_PAMF_STREAM_TYPE_PAMF_LPCM) {
        if (infoSize < 8u) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
        pamf_zero_guest(out, 8u);
        vm_write32(out, specific[3] & 0x0Fu);
        vm_write8(out + 4u, specific[2] & 0x0Fu);
        vm_write16(out + 6u, specific[4] >> 6);
        return CELL_OK;
    }
    return (s32)CELL_PAMF_ERROR_INVALID_ARG;
}

u32 cellPamfReaderGetNumberOfEp(CellPamfReader* reader)
{
    PamfReaderView v;
    if (!pamf_load_reader((u32)(uintptr_t)reader, &v) || !v.stream_ea)
        return 0;
    const u8* stream = pamf_guest_bytes(v.stream_ea, 0);
    return stream ? pamf_be32(stream + PAMF_STREAM_EP_COUNT) : 0;
}

static s32 pamf_current_ep_table(const PamfReaderView* reader,
                                 u32* table_off, u32* count)
{
    if (!reader || !reader->stream_ea ||
        (reader->attributes & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER))
        return (s32)CELL_PAMF_ERROR_NOT_AVAILABLE;
    const u8* stream = pamf_guest_bytes(reader->stream_ea, 0);
    if (!stream) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    *table_off = pamf_be32(stream + PAMF_STREAM_EP_OFFSET);
    *count = pamf_be32(stream + PAMF_STREAM_EP_COUNT);
    if (!*count || !*table_off) return (s32)CELL_PAMF_ERROR_EP_NOT_FOUND;
    if (!pamf_range_ok(reader->header_size, *table_off,
                       (u64)*count * PAMF_EP_SIZE))
        return (s32)CELL_PAMF_ERROR_INVALID_PAMF;
    return CELL_OK;
}

static void pamf_write_iterator(u32 iterator_ea, u32 index, u32 count,
                                u32 current_ep_ea)
{
    vm_write8(iterator_ea, 1u); /* isPamf */
    vm_write8(iterator_ea + 1u, 0);
    vm_write8(iterator_ea + 2u, 0);
    vm_write8(iterator_ea + 3u, 0);
    vm_write32(iterator_ea + 4u, index);
    vm_write32(iterator_ea + 8u, count);
    vm_write32(iterator_ea + 12u, current_ep_ea);
}

static s32 pamf_iterator_read(u32 iterator_ea, u32* index, u32* count,
                              u32* current_ep_ea)
{
    if (!iterator_ea || !vm_base || !vm_read8(iterator_ea))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    *index = vm_read32(iterator_ea + 4u);
    *count = vm_read32(iterator_ea + 8u);
    *current_ep_ea = vm_read32(iterator_ea + 12u);
    if (!*count || *index >= *count || !*current_ep_ea)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    return CELL_OK;
}

s32 cellPamfReaderGetEpIteratorWithIndex(CellPamfReader* reader,
                                          u32 epIndex,
                                          CellPamfEpIterator* iterator)
{
    PamfReaderView v;
    const u32 iterator_ea = (u32)(uintptr_t)iterator;
    if (!iterator_ea || !pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u32 count = cellPamfReaderGetNumberOfEp(reader);
    if (epIndex >= count) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (v.attributes & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER)
        return (s32)CELL_PAMF_ERROR_NOT_AVAILABLE;
    u32 table_off, checked_count;
    s32 rc = pamf_current_ep_table(&v, &table_off, &checked_count);
    if (rc != CELL_OK) return rc;
    pamf_write_iterator(iterator_ea, epIndex, checked_count,
                        v.header_ea + table_off + epIndex * PAMF_EP_SIZE);
    return CELL_OK;
}

s32 cellPamfReaderGetEpIteratorWithTimeStamp(
    CellPamfReader* reader, CellCodecTimeStamp* timeStamp,
    CellPamfEpIterator* iterator)
{
    PamfReaderView v;
    const u32 time_ea = (u32)(uintptr_t)timeStamp;
    const u32 iterator_ea = (u32)(uintptr_t)iterator;
    if (!time_ea || !iterator_ea ||
        !pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (v.attributes & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER)
        return (s32)CELL_PAMF_ERROR_NOT_AVAILABLE;
    u32 table_off, count;
    s32 rc = pamf_current_ep_table(&v, &table_off, &count);
    if (rc == (s32)CELL_PAMF_ERROR_EP_NOT_FOUND) return rc;
    if (rc != CELL_OK) return rc;

    const u64 requested = pamf_read_timestamp(time_ea);
    u32 index = count - 1u;
    for (;;) {
        const u8* ep = pamf_guest_bytes(v.header_ea,
            (u64)table_off + (u64)index * PAMF_EP_SIZE);
        const u64 timestamp = ((u64)pamf_be16(ep + 2) << 32) |
                               pamf_be32(ep + 4);
        if (timestamp <= requested || index == 0)
            break;
        index--;
    }
    pamf_write_iterator(iterator_ea, index, count,
                        v.header_ea + table_off + index * PAMF_EP_SIZE);
    return CELL_OK;
}

s32 cellPamfEpIteratorGetEp(CellPamfEpIterator* iterator, CellPamfEp* ep)
{
    const u32 ep_ea = (u32)(uintptr_t)ep;
    u32 index, count, current_ea;
    if (!ep_ea) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    s32 rc = pamf_iterator_read((u32)(uintptr_t)iterator, &index, &count,
                                &current_ea);
    if (rc != CELL_OK) return rc;
    (void)index; (void)count;
    /* Resolve a current entry by its guest address; its enclosing header is
     * already range-checked by the reader when the iterator was created. */
    const u8* packed = pamf_guest_bytes(current_ea, 0);
    if (!packed) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    CellPamfEp value;
    const u16 value0 = pamf_be16(packed);
    value.indexN = (u32)(value0 >> 14) + 1u;
    value.nThRefPictureOffset = ((u32)(value0 & 0x1FFFu) * 0x800u) + 0x800u;
    value.pts.upper = pamf_be16(packed + 2);
    value.pts.lower = pamf_be32(packed + 4);
    value.rpnOffset = (u64)pamf_be32(packed + 8) * PAMF_BLOCK_SIZE;
    pamf_write_ep(ep_ea, &value);
    return CELL_OK;
}

s32 cellPamfEpIteratorMove(CellPamfEpIterator* iterator, s32 steps,
                            CellPamfEp* ep)
{
    const u32 iterator_ea = (u32)(uintptr_t)iterator;
    u32 index, count, current_ea;
    s32 rc = pamf_iterator_read(iterator_ea, &index, &count, &current_ea);
    if (rc != CELL_OK) return rc;
    s64 target = (s64)index + (s64)steps;
    if (target < 0) target = 0;
    if (target >= count) target = (s64)count - 1;
    const s32 moved = (s32)(target - (s64)index);
    const s64 next_ea = (s64)current_ea + (s64)moved * PAMF_EP_SIZE;
    if (next_ea < 0 || next_ea > 0xFFFFFFFFll)
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    pamf_write_iterator(iterator_ea, (u32)target, count, (u32)next_ea);
    if (ep) {
        rc = cellPamfEpIteratorGetEp(iterator, ep);
        if (rc != CELL_OK) return rc;
    }
    return moved;
}

static s32 pamf_find_ep_at_time(const PamfReaderView* v, u64 requested,
                                u32* index, u32* table_off, u32* count)
{
    s32 rc = pamf_current_ep_table(v, table_off, count);
    if (rc != CELL_OK) return rc;
    u32 i = *count - 1u;
    for (;;) {
        const u8* ep = pamf_guest_bytes(v->header_ea,
            (u64)*table_off + (u64)i * PAMF_EP_SIZE);
        const u64 pts = ((u64)pamf_be16(ep + 2) << 32) | pamf_be32(ep + 4);
        if (pts <= requested || i == 0) break;
        i--;
    }
    *index = i;
    return CELL_OK;
}

s32 cellPamfReaderGetEpWithTimeStamp(CellPamfReader* reader,
                                      CellCodecTimeStamp* timeStamp,
                                      CellPamfEpUnk* ep, u32 unk)
{
    PamfReaderView v;
    const u32 time_ea = (u32)(uintptr_t)timeStamp;
    const u32 ep_ea = (u32)(uintptr_t)ep;
    if (!time_ea || !ep_ea ||
        !pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (v.attributes & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER)
        return (s32)CELL_PAMF_ERROR_NOT_AVAILABLE;
    u32 index, table_off, count;
    s32 rc = pamf_find_ep_at_time(&v, pamf_read_timestamp(time_ea),
                                  &index, &table_off, &count);
    if (rc != CELL_OK) return rc;

    /* RPCS3's current HLE does not populate the CellPamfEp portion for this
     * legacy call. Preserve that observable behavior; only its size-tagged
     * nextRpnOffset result is written. */
    if (unk == sizeof(CellPamfEpUnk)) {
        u64 next_rpn = v.data_size;
        if (index < count - 1u) {
            const u8* next = pamf_guest_bytes(v.header_ea,
                (u64)table_off + (u64)(index + 1u) * PAMF_EP_SIZE);
            next_rpn = pamf_be32(next + 8u); /* retained RPCS3's raw-unit result */
        }
        vm_write64(ep_ea + 24u, next_rpn);
    }
    return CELL_OK;
}

s32 cellPamfReaderGetEpWithIndex(CellPamfReader* reader, u32 epIndex,
                                  CellPamfEpUnk* ep, u32 unk)
{
    PamfReaderView v;
    const u32 ep_ea = (u32)(uintptr_t)ep;
    if (!ep_ea || !pamf_load_reader((u32)(uintptr_t)reader, &v))
        return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    const u32 count = cellPamfReaderGetNumberOfEp(reader);
    if (epIndex >= count) return (s32)CELL_PAMF_ERROR_INVALID_ARG;
    if (v.attributes & CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER)
        return (s32)CELL_PAMF_ERROR_NOT_AVAILABLE;
    u32 table_off, checked_count;
    s32 rc = pamf_current_ep_table(&v, &table_off, &checked_count);
    if (rc != CELL_OK) return rc;
    CellPamfEp value;
    rc = pamf_parse_ep(&(PamfHeaderView){v.header_ea,
            pamf_guest_bytes(v.header_ea, 0), 0, v.header_size,
            v.data_size, 0}, table_off, epIndex, &value);
    if (rc != CELL_OK) return rc;
    pamf_write_ep(ep_ea, &value);
    if (unk == sizeof(CellPamfEpUnk)) {
        u64 next_rpn = v.data_size;
        if (epIndex < checked_count - 1u) {
            const u8* next = pamf_guest_bytes(v.header_ea,
                (u64)table_off + (u64)(epIndex + 1u) * PAMF_EP_SIZE);
            next_rpn = (u64)pamf_be32(next + 8u) * PAMF_BLOCK_SIZE;
        }
        vm_write64(ep_ea + 24u, next_rpn);
    }
    return CELL_OK;
}
