/*
 * ps3recomp - cellL10n HLE implementation
 *
 * Unicode and code-page conversion using local codecs, Win32 conversion APIs,
 * or iconv. The PS3 code-page IDs and generic conversion entry points follow
 * RPCS3's cellL10n ABI; guest memory is accessed through ps3recomp helpers.
 */

#include "cellL10n.h"
#include "../../runtime/ppu/ppu_memory.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <iconv.h>
#endif

/* ---------------------------------------------------------------------------
 * Internal UTF helpers
 * -----------------------------------------------------------------------*/

/* Decode one UTF-8 codepoint. Returns bytes consumed, 0 on error. */
static int decode_utf8(const u8* s, u32 len, u32* cp)
{
    if (len == 0) return 0;

    u8 b = s[0];
    if (b < 0x80) {
        *cp = b;
        return 1;
    } else if ((b & 0xE0) == 0xC0) {
        if (len < 2 || (s[1] & 0xC0) != 0x80) return 0;
        *cp = ((u32)(b & 0x1F) << 6) | (s[1] & 0x3F);
        if (*cp < 0x80) return 0; /* overlong */
        return 2;
    } else if ((b & 0xF0) == 0xE0) {
        if (len < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
        *cp = ((u32)(b & 0x0F) << 12) | ((u32)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (*cp < 0x800) return 0;
        if (*cp >= 0xD800 && *cp <= 0xDFFF) return 0; /* surrogate */
        return 3;
    } else if ((b & 0xF8) == 0xF0) {
        if (len < 4 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80)
            return 0;
        *cp = ((u32)(b & 0x07) << 18) | ((u32)(s[1] & 0x3F) << 12)
            | ((u32)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (*cp < 0x10000 || *cp > 0x10FFFF) return 0;
        return 4;
    }
    return 0;
}

/* Encode one codepoint to UTF-8. Returns bytes written. */
static int encode_utf8(u32 cp, u8* out, u32 max)
{
    if (cp < 0x80) {
        if (max < 1) return 0;
        out[0] = (u8)cp;
        return 1;
    } else if (cp < 0x800) {
        if (max < 2) return 0;
        out[0] = 0xC0 | (u8)(cp >> 6);
        out[1] = 0x80 | (u8)(cp & 0x3F);
        return 2;
    } else if (cp < 0x10000) {
        if (max < 3) return 0;
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
        out[0] = 0xE0 | (u8)(cp >> 12);
        out[1] = 0x80 | (u8)((cp >> 6) & 0x3F);
        out[2] = 0x80 | (u8)(cp & 0x3F);
        return 3;
    } else if (cp <= 0x10FFFF) {
        if (max < 4) return 0;
        out[0] = 0xF0 | (u8)(cp >> 18);
        out[1] = 0x80 | (u8)((cp >> 12) & 0x3F);
        out[2] = 0x80 | (u8)((cp >> 6) & 0x3F);
        out[3] = 0x80 | (u8)(cp & 0x3F);
        return 4;
    }
    return 0;
}

/* Decode one UTF-16 codepoint (handles surrogates). Returns u16s consumed. */
static int decode_utf16(const u16* s, u32 len, u32* cp)
{
    if (len == 0) return 0;

    u16 w = s[0];
    if (w >= 0xD800 && w <= 0xDBFF) {
        /* High surrogate */
        if (len < 2) return 0;
        u16 lo = s[1];
        if (lo < 0xDC00 || lo > 0xDFFF) return 0;
        *cp = 0x10000 + ((u32)(w - 0xD800) << 10) + (lo - 0xDC00);
        return 2;
    } else if (w >= 0xDC00 && w <= 0xDFFF) {
        return 0; /* lone low surrogate */
    }
    *cp = w;
    return 1;
}

/* Encode one codepoint to UTF-16. Returns u16s written. */
static int encode_utf16(u32 cp, u16* out, u32 max)
{
    if (cp < 0x10000) {
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
        if (max < 1) return 0;
        out[0] = (u16)cp;
        return 1;
    } else if (cp <= 0x10FFFF) {
        if (max < 2) return 0;
        cp -= 0x10000;
        out[0] = 0xD800 + (u16)(cp >> 10);
        out[1] = 0xDC00 + (u16)(cp & 0x3FF);
        return 2;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * UTF-8 <-> UTF-16
 * -----------------------------------------------------------------------*/

s32 UTF8toUTF16(const u8* utf8, u32 utf8_len,
                u16* utf16, u32* utf16_len)
{
    if (!utf8 || !utf16 || !utf16_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 out_max = *utf16_len;
    u32 out_pos = 0;
    u32 in_pos = 0;

    while (in_pos < utf8_len) {
        u32 cp;
        int consumed = decode_utf8(utf8 + in_pos, utf8_len - in_pos, &cp);
        if (consumed == 0)
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;

        int written = encode_utf16(cp, utf16 + out_pos, out_max - out_pos);
        if (written == 0)
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;

        in_pos += consumed;
        out_pos += written;
    }

    *utf16_len = out_pos;
    return CELL_OK;
}

s32 UTF16toUTF8(const u16* utf16, u32 utf16_len,
                u8* utf8, u32* utf8_len)
{
    if (!utf16 || !utf8 || !utf8_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 out_max = *utf8_len;
    u32 out_pos = 0;
    u32 in_pos = 0;

    while (in_pos < utf16_len) {
        u32 cp;
        int consumed = decode_utf16(utf16 + in_pos, utf16_len - in_pos, &cp);
        if (consumed == 0)
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;

        int written = encode_utf8(cp, utf8 + out_pos, out_max - out_pos);
        if (written == 0)
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;

        in_pos += consumed;
        out_pos += written;
    }

    *utf8_len = out_pos;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * UTF-8 <-> UTF-32
 * -----------------------------------------------------------------------*/

s32 UTF8toUTF32(const u8* utf8, u32 utf8_len,
                u32* utf32, u32* utf32_len)
{
    if (!utf8 || !utf32 || !utf32_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 out_max = *utf32_len;
    u32 out_pos = 0;
    u32 in_pos = 0;

    while (in_pos < utf8_len) {
        u32 cp;
        int consumed = decode_utf8(utf8 + in_pos, utf8_len - in_pos, &cp);
        if (consumed == 0)
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;

        if (out_pos >= out_max)
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;

        utf32[out_pos++] = cp;
        in_pos += consumed;
    }

    *utf32_len = out_pos;
    return CELL_OK;
}

s32 UTF32toUTF8(const u32* utf32, u32 utf32_len,
                u8* utf8, u32* utf8_len)
{
    if (!utf32 || !utf8 || !utf8_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 out_max = *utf8_len;
    u32 out_pos = 0;

    for (u32 i = 0; i < utf32_len; i++) {
        int written = encode_utf8(utf32[i], utf8 + out_pos, out_max - out_pos);
        if (written == 0)
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;
        out_pos += written;
    }

    *utf8_len = out_pos;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * UTF-16 <-> UTF-32
 * -----------------------------------------------------------------------*/

s32 UTF16toUTF32(const u16* utf16, u32 utf16_len,
                 u32* utf32, u32* utf32_len)
{
    if (!utf16 || !utf32 || !utf32_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 out_max = *utf32_len;
    u32 out_pos = 0;
    u32 in_pos = 0;

    while (in_pos < utf16_len) {
        u32 cp;
        int consumed = decode_utf16(utf16 + in_pos, utf16_len - in_pos, &cp);
        if (consumed == 0)
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;

        if (out_pos >= out_max)
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;

        utf32[out_pos++] = cp;
        in_pos += consumed;
    }

    *utf32_len = out_pos;
    return CELL_OK;
}

s32 UTF32toUTF16(const u32* utf32, u32 utf32_len,
                 u16* utf16, u32* utf16_len)
{
    if (!utf32 || !utf16 || !utf16_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 out_max = *utf16_len;
    u32 out_pos = 0;

    for (u32 i = 0; i < utf32_len; i++) {
        int written = encode_utf16(utf32[i], utf16 + out_pos, out_max - out_pos);
        if (written == 0)
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;
        out_pos += written;
    }

    *utf16_len = out_pos;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * UCS-2 (BMP-only subset of UTF-16, no surrogates)
 * -----------------------------------------------------------------------*/

s32 UTF8toUCS2(const u8* utf8, u16* ucs2)
{
    const u32 src_ea = (u32)(uintptr_t)utf8;
    const u32 dst_ea = (u32)(uintptr_t)ucs2;
    if (!src_ea || !dst_ea)
        return 0;

    const u8* src = GUEST_PTR(utf8, const u8*);
    const u8 first = src[0];
    const u32 length = first < 0x80 ? 1u :
                       (first >= 0xC2 && first <= 0xDF ? 2u :
                        (first >= 0xE0 && first <= 0xEF ? 3u :
                         (first >= 0xF0 && first <= 0xF4 ? 4u : 1u)));
    u32 cp;
    const int consumed = decode_utf8(src, length, &cp);
    if (consumed == 0 || cp > 0xFFFFu)
        return 0;
    vm_write16(dst_ea, (u16)cp);
    return consumed;
}

s32 UCS2toUTF8(u16 ucs2, u8* utf8)
{
    const u32 dst_ea = (u32)(uintptr_t)utf8;
    if (!dst_ea || (ucs2 >= 0xD800u && ucs2 <= 0xDFFFu))
        return 0;
    u8 bytes[4];
    const int count = encode_utf8(ucs2, bytes, sizeof(bytes));
    if (count <= 0)
        return 0;
    for (int i = 0; i < count; i++)
        vm_write8(dst_ea + (u32)i, bytes[i]);
    return count;
}

s32 UCS2stoUTF8s(const u16* ucs2, u32* ucs2_len,
                 u8* utf8, u32* utf8_len)
{
    const u32 src_ea = (u32)(uintptr_t)ucs2;
    const u32 src_len_ea = (u32)(uintptr_t)ucs2_len;
    const u32 dst_ea = (u32)(uintptr_t)utf8;
    const u32 dst_len_ea = (u32)(uintptr_t)utf8_len;
    if (!src_len_ea || !dst_len_ea)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    const u32 source_units = vm_read32(src_len_ea);
    const u32 dst_capacity = vm_read32(dst_len_ea);
    if (source_units && !src_ea)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 written = 0;
    for (u32 i = 0; i < source_units; i++) {
        const u16 ch = vm_read16(src_ea + i * 2u);
        u8 encoded[4];
        const int n = (ch >= 0xD800u && ch <= 0xDFFFu)
                    ? 0 : encode_utf8(ch, encoded, sizeof(encoded));
        if (n == 0) {
            vm_write32(src_len_ea, source_units - i);
            vm_write32(dst_len_ea, written);
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;
        }
        if (dst_ea && (u64)written + (u32)n > dst_capacity) {
            vm_write32(src_len_ea, source_units - i);
            vm_write32(dst_len_ea, written);
            return (s32)CELL_L10N_ERROR_INSUFFICIENT;
        }
        if (dst_ea) {
            for (int j = 0; j < n; j++)
                vm_write8(dst_ea + written + (u32)j, encoded[j]);
        }
        written += (u32)n;
    }

    vm_write32(dst_len_ea, written);
    return CELL_OK;
}

static int sjis_to_unicode(const u8* src, u32 length, u16* out)
{
#ifdef _WIN32
    wchar_t wide[2] = {0, 0};
    const int count = MultiByteToWideChar(932, 0, (const char*)src,
                                           (int)length, wide, 2);
    if (count != 1 || ((u16)wide[0] >= 0xD800u && (u16)wide[0] <= 0xDFFFu) ||
        wide[0] == 0xFFFD)
        return 0;
    *out = (u16)wide[0];
    return count;
#else
    iconv_t cd = iconv_open("UTF-16BE", "CP932");
    if (cd == (iconv_t)-1)
        cd = iconv_open("UTF-16BE", "SHIFT_JIS");
    if (cd == (iconv_t)-1)
        return 0;
    char inbuf[2] = {(char)src[0], (char)(length > 1 ? src[1] : 0)};
    char outbuf[4] = {0, 0, 0, 0};
    char* inptr = inbuf;
    char* outptr = outbuf;
    size_t inleft = length;
    size_t outleft = sizeof(outbuf);
    const size_t rc = iconv(cd, &inptr, &inleft, &outptr, &outleft);
    iconv_close(cd);
    if (rc == (size_t)-1 || inleft != 0 || sizeof(outbuf) - outleft != 2)
        return 0;
    *out = (u16)(((u16)(u8)outbuf[0] << 8) | (u8)outbuf[1]);
    if (*out >= 0xD800u && *out <= 0xDFFFu)
        return 0;
    return 1;
#endif
}

s32 SJIStoUCS2(const u8* sjis, u16* ucs2)
{
    const u32 src_ea = (u32)(uintptr_t)sjis;
    const u32 dst_ea = (u32)(uintptr_t)ucs2;
    if (!src_ea || !dst_ea)
        return 0;
    const u8* src = GUEST_PTR(sjis, const u8*);
    const u8 first = src[0];
    u32 length = 1;
    if ((first >= 0x81 && first <= 0x9F) || (first >= 0xE0 && first <= 0xFC)) {
        const u8 trail = src[1];
        if (!((trail >= 0x40 && trail <= 0x7E) ||
              (trail >= 0x80 && trail <= 0xFC)))
            return 0;
        length = 2;
    } else if (!((first <= 0x7F) || (first >= 0xA1 && first <= 0xDF))) {
        return 0;
    }
    u16 decoded;
    if (!sjis_to_unicode(src, length, &decoded))
        return 0;
    vm_write16(dst_ea, decoded);
    return (s32)length;
}

s32 UTF8toSJIS(u8 utf8_byte, u8* sjis, u32* sjis_len)
{
    const u32 dst_ea = (u32)(uintptr_t)sjis;
    const u32 len_ea = (u32)(uintptr_t)sjis_len;
    if (!dst_ea || !len_ea)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;
    if (utf8_byte >= 0x80u) {
        vm_write32(len_ea, 0);
        return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;
    }
    vm_write8(dst_ea, utf8_byte);
    vm_write32(len_ea, 1);
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * String conversion (null-terminated)
 * -----------------------------------------------------------------------*/

s32 UTF8stoUTF16s(const u8* utf8, u32* utf8_len,
                  u16* utf16, u32* utf16_len)
{
    if (!utf8 || !utf8_len || !utf16 || !utf16_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    /* Find actual string length (up to *utf8_len) */
    u32 src_len = 0;
    while (src_len < *utf8_len && utf8[src_len] != 0)
        src_len++;

    u32 dst_max = *utf16_len;
    s32 rc = UTF8toUTF16(utf8, src_len, utf16, &dst_max);
    if (rc != CELL_OK) return rc;

    /* Null-terminate if space available */
    if (dst_max < *utf16_len)
        utf16[dst_max] = 0;

    *utf8_len = src_len;
    *utf16_len = dst_max;
    return CELL_OK;
}

s32 UTF16stoUTF8s(const u16* utf16, u32* utf16_len,
                  u8* utf8, u32* utf8_len)
{
    if (!utf16 || !utf16_len || !utf8 || !utf8_len)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 src_len = 0;
    while (src_len < *utf16_len && utf16[src_len] != 0)
        src_len++;

    u32 dst_max = *utf8_len;
    s32 rc = UTF16toUTF8(utf16, src_len, utf8, &dst_max);
    if (rc != CELL_OK) return rc;

    if (dst_max < *utf8_len)
        utf8[dst_max] = 0;

    *utf16_len = src_len;
    *utf8_len = dst_max;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Size queries
 * -----------------------------------------------------------------------*/

s32 UTF8toUTF16Size(const u8* utf8, u32 utf8_len, u32* out_size)
{
    if (!utf8 || !out_size)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 count = 0;
    u32 pos = 0;

    while (pos < utf8_len) {
        u32 cp;
        int consumed = decode_utf8(utf8 + pos, utf8_len - pos, &cp);
        if (consumed == 0)
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;
        pos += consumed;
        count += (cp >= 0x10000) ? 2 : 1;
    }

    *out_size = count;
    return CELL_OK;
}

s32 UTF16toUTF8Size(const u16* utf16, u32 utf16_len, u32* out_size)
{
    if (!utf16 || !out_size)
        return (s32)CELL_L10N_ERROR_INVALID_PARAM;

    u32 count = 0;
    u32 pos = 0;

    while (pos < utf16_len) {
        u32 cp;
        int consumed = decode_utf16(utf16 + pos, utf16_len - pos, &cp);
        if (consumed == 0)
            return (s32)CELL_L10N_ERROR_ILLEGAL_SEQUENCE;
        pos += consumed;

        if (cp < 0x80) count += 1;
        else if (cp < 0x800) count += 2;
        else if (cp < 0x10000) count += 3;
        else count += 4;
    }

    *out_size = count;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * Codepoint length queries
 * -----------------------------------------------------------------------*/

s32 UTF8Len(const u8* utf8)
{
    if (!utf8) return 0;

    u8 b = utf8[0];
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0; /* invalid */
}

s32 UTF16Len(const u16* utf16)
{
    if (!utf16) return 0;

    u16 w = utf16[0];
    if (w >= 0xD800 && w <= 0xDBFF) return 2; /* surrogate pair */
    return 1;
}

/* ---------------------------------------------------------------------------
 * Generic code-page conversion
 * -----------------------------------------------------------------------*/

/* RPCS3_REFERENCE_REVIEWED (RPCS3 commit 2ba2f75bc): PS3 code-page IDs,
 * conversion result values, packed converter IDs, and string-length ABI were
 * checked against cellL10n.h/cpp. The conversion itself uses Win32 or iconv
 * plus this runtime's guest-memory helpers; no RPCS3 function body is copied. */
static unsigned int l10n_win_codepage(s32 code)
{
    switch (code) {
    case L10N_UTF8: return CP_UTF8;
    case L10N_UTF16: return 1201;
    case L10N_UTF32: return 12001;
    case L10N_UCS2: return 1201;
    case L10N_UCS4: return 12001;
    case L10N_ISO_8859_1: return 28591;
    case L10N_ISO_8859_2: return 28592;
    case L10N_ISO_8859_3: return 28593;
    case L10N_ISO_8859_4: return 28594;
    case L10N_ISO_8859_5: return 28595;
    case L10N_ISO_8859_6: return 28596;
    case L10N_ISO_8859_7: return 28597;
    case L10N_ISO_8859_8: return 28598;
    case L10N_ISO_8859_9: return 28599;
    case L10N_ISO_8859_10: return 28600;
    case L10N_ISO_8859_11: return 28601;
    case L10N_ISO_8859_13: return 28603;
    case L10N_ISO_8859_14: return 28604;
    case L10N_ISO_8859_15: return 28605;
    case L10N_ISO_8859_16: return 28606;
    case L10N_CODEPAGE_437: return 437;
    case L10N_CODEPAGE_850: return 850;
    case L10N_CODEPAGE_863: return 863;
    case L10N_CODEPAGE_866: return 866;
    case L10N_CODEPAGE_932: return 932;
    case L10N_CODEPAGE_936: return 936;
    case L10N_CODEPAGE_949: return 949;
    case L10N_CODEPAGE_950: return 950;
    case L10N_CODEPAGE_1251: return 1251;
    case L10N_CODEPAGE_1252: return 1252;
    case L10N_EUC_CN: return 51936;
    case L10N_EUC_JP: return 51932;
    case L10N_EUC_KR: return 51949;
    case L10N_ISO_2022_JP: return 50222;
    case L10N_ARIB: return 20420;
    case L10N_HZ: return 52936;
    case L10N_GB18030: return 54936;
    case L10N_RIS_506: return 932;
    case L10N_CODEPAGE_852: return 852;
    case L10N_CODEPAGE_1250: return 1250;
    case L10N_CODEPAGE_737: return 737;
    case L10N_CODEPAGE_1253: return 1253;
    case L10N_CODEPAGE_857: return 857;
    case L10N_CODEPAGE_1254: return 1254;
    case L10N_CODEPAGE_775: return 775;
    case L10N_CODEPAGE_1257: return 1257;
    case L10N_CODEPAGE_855: return 855;
    case L10N_CODEPAGE_858: return 858;
    case L10N_CODEPAGE_860: return 860;
    case L10N_CODEPAGE_861: return 861;
    case L10N_CODEPAGE_865: return 865;
    case L10N_CODEPAGE_869: return 869;
    default: return 0;
    }
}

#ifndef _WIN32
static const char* l10n_iconv_encoding(s32 code)
{
    switch (code) {
    case L10N_UTF8: return "UTF-8";
    case L10N_UTF16: return "UTF-16BE";
    case L10N_UTF32: return "UTF-32BE";
    case L10N_UCS2: return "UCS-2BE";
    case L10N_UCS4: return "UCS-4BE";
    case L10N_ISO_8859_1: return "ISO-8859-1";
    case L10N_ISO_8859_2: return "ISO-8859-2";
    case L10N_ISO_8859_3: return "ISO-8859-3";
    case L10N_ISO_8859_4: return "ISO-8859-4";
    case L10N_ISO_8859_5: return "ISO-8859-5";
    case L10N_ISO_8859_6: return "ISO-8859-6";
    case L10N_ISO_8859_7: return "ISO-8859-7";
    case L10N_ISO_8859_8: return "ISO-8859-8";
    case L10N_ISO_8859_9: return "ISO-8859-9";
    case L10N_ISO_8859_10: return "ISO-8859-10";
    case L10N_ISO_8859_11: return "ISO-8859-11";
    case L10N_ISO_8859_13: return "ISO-8859-13";
    case L10N_ISO_8859_14: return "ISO-8859-14";
    case L10N_ISO_8859_15: return "ISO-8859-15";
    case L10N_ISO_8859_16: return "ISO-8859-16";
    case L10N_CODEPAGE_437: return "CP437";
    case L10N_CODEPAGE_850: return "CP850";
    case L10N_CODEPAGE_863: return "CP863";
    case L10N_CODEPAGE_866: return "CP866";
    case L10N_CODEPAGE_932: return "CP932";
    case L10N_CODEPAGE_936: return "CP936";
    case L10N_CODEPAGE_949: return "CP949";
    case L10N_CODEPAGE_950: return "CP950";
    case L10N_CODEPAGE_1251: return "CP1251";
    case L10N_CODEPAGE_1252: return "CP1252";
    case L10N_EUC_CN: return "EUC-CN";
    case L10N_EUC_JP: return "EUC-JP";
    case L10N_EUC_KR: return "EUC-KR";
    case L10N_ISO_2022_JP: return "ISO-2022-JP";
    case L10N_ARIB: return "ARABIC";
    case L10N_HZ: return "HZ";
    case L10N_GB18030: return "GB18030";
    case L10N_RIS_506: return "Shift_JIS";
    case L10N_CODEPAGE_852: return "CP852";
    case L10N_CODEPAGE_1250: return "CP1250";
    case L10N_CODEPAGE_737: return "CP737";
    case L10N_CODEPAGE_1253: return "CP1253";
    case L10N_CODEPAGE_857: return "CP857";
    case L10N_CODEPAGE_1254: return "CP1254";
    case L10N_CODEPAGE_775: return "CP775";
    case L10N_CODEPAGE_1257: return "CP1257";
    case L10N_CODEPAGE_855: return "CP855";
    case L10N_CODEPAGE_858: return "CP858";
    case L10N_CODEPAGE_860: return "CP860";
    case L10N_CODEPAGE_861: return "CP861";
    case L10N_CODEPAGE_865: return "CP865";
    case L10N_CODEPAGE_869: return "CP869";
    default: return NULL;
    }
}
#endif

static int guest_read_bytes(u32 ea, u8* bytes, u32 count)
{
    if (count && !ea)
        return 0;
    if (count && (u64)ea + count - 1u > UINT32_MAX)
        return 0;
    for (u32 i = 0; i < count; ++i)
        bytes[i] = vm_read8(ea + i);
    return 1;
}

static int guest_write_bytes(u32 ea, const u8* bytes, u32 count)
{
    if (count && !ea)
        return 0;
    if (count && (u64)ea + count - 1u > UINT32_MAX)
        return 0;
    for (u32 i = 0; i < count; ++i)
        vm_write8(ea + i, bytes[i]);
    return 1;
}

static s32 l10n_convert_guest_bytes(s32 src_code, u32 src_ea, u32 src_len,
                                   s32 dst_code, u32 dst_ea, u32 dst_capacity,
                                   u32* dst_len)
{
    if (!dst_len)
        return ConverterUnknown;
    *dst_len = 0;
    if (src_len == 0)
        return ConversionOK;
    if (!src_ea || (u64)src_ea + src_len - 1u > UINT32_MAX)
        return SRCIllegal;

    u8* source = (u8*)malloc((size_t)src_len);
    if (!source)
        return DSTExhausted;
    if (!guest_read_bytes(src_ea, source, src_len)) {
        free(source);
        return SRCIllegal;
    }

#ifdef _WIN32
    const UINT source_cp = l10n_win_codepage(src_code);
    const UINT target_cp = l10n_win_codepage(dst_code);
    if (!source_cp || !target_cp || src_len > INT_MAX) {
        free(source);
        return ConverterUnknown;
    }

    const DWORD source_flags = (source_cp == CP_UTF8 || source_cp == 54936)
        ? MB_ERR_INVALID_CHARS : 0;
    const int wide_count = MultiByteToWideChar(source_cp, source_flags,
        (const char*)source, (int)src_len, NULL, 0);
    if (wide_count <= 0) {
        free(source);
        return SRCIllegal;
    }

    wchar_t* wide = (wchar_t*)malloc((size_t)wide_count * sizeof(wchar_t));
    if (!wide) {
        free(source);
        return DSTExhausted;
    }
    if (MultiByteToWideChar(source_cp, source_flags, (const char*)source,
            (int)src_len, wide, wide_count) != wide_count) {
        free(wide);
        free(source);
        return SRCIllegal;
    }
    free(source);

    const int target_is_unicode = target_cp == CP_UTF8 || target_cp == 1200 ||
        target_cp == 1201 || target_cp == 12000 || target_cp == 12001;
    const DWORD target_flags = (target_cp == CP_UTF8 || target_cp == 54936)
        ? WC_ERR_INVALID_CHARS
        : ((target_cp == 1201 || target_cp == 12001) ? 0 : WC_NO_BEST_FIT_CHARS);
    BOOL used_default = FALSE;
    const int required = WideCharToMultiByte(target_cp, target_flags, wide,
        wide_count, NULL, 0, target_is_unicode ? NULL : "?",
        target_is_unicode ? NULL : &used_default);
    if (required <= 0) {
        free(wide);
        return ConverterUnknown;
    }
    if (dst_ea && dst_capacity < (u32)required) {
        free(wide);
        return DSTExhausted;
    }

    u8* output = (u8*)malloc((size_t)required);
    if (!output) {
        free(wide);
        return DSTExhausted;
    }
    used_default = FALSE;
    const int written = WideCharToMultiByte(target_cp, target_flags, wide,
        wide_count, (char*)output, required,
        target_is_unicode ? NULL : "?", target_is_unicode ? NULL : &used_default);
    free(wide);
    if (written != required || used_default) {
        free(output);
        return SRCIllegal;
    }
    *dst_len = (u32)written;
    const int copied = !dst_ea || guest_write_bytes(dst_ea, output, (u32)written);
    free(output);
    return copied ? ConversionOK : SRCIllegal;
#else
    const char* source_encoding = l10n_iconv_encoding(src_code);
    const char* target_encoding = l10n_iconv_encoding(dst_code);
    if (!source_encoding || !target_encoding) {
        free(source);
        return ConverterUnknown;
    }

    iconv_t converter = iconv_open(target_encoding, source_encoding);
    if (converter == (iconv_t)-1) {
        free(source);
        return ConverterUnknown;
    }

    const u64 sizing_capacity = dst_ea
        ? dst_capacity : (u64)src_len * 8u + 64u;
    if (sizing_capacity > UINT32_MAX) {
        iconv_close(converter);
        free(source);
        return DSTExhausted;
    }
    const u32 capacity = (u32)sizing_capacity;
    u8* output = (u8*)malloc(capacity ? capacity : 1u);
    if (!output) {
        iconv_close(converter);
        free(source);
        return DSTExhausted;
    }

    char* input_ptr = (char*)source;
    size_t input_left = (size_t)src_len;
    char* output_ptr = (char*)output;
    size_t output_left = (size_t)capacity;
    errno = 0;
    size_t result = iconv(converter, &input_ptr, &input_left,
        &output_ptr, &output_left);
    s32 status = ConversionOK;
    if (result == (size_t)-1)
        status = errno == E2BIG ? DSTExhausted : SRCIllegal;
    else if (input_left != 0)
        status = SRCIllegal;

    if (status == ConversionOK) {
        errno = 0;
        result = iconv(converter, NULL, NULL, &output_ptr, &output_left);
        if (result == (size_t)-1)
            status = errno == E2BIG ? DSTExhausted : SRCIllegal;
    }

    const u32 written = capacity - (u32)output_left;
    iconv_close(converter);
    free(source);
    *dst_len = written;
    const int copied = !dst_ea || guest_write_bytes(dst_ea, output, written);
    free(output);
    if (!copied)
        return SRCIllegal;
    return status;
#endif
}

static u32 l10n_character_length(s32 code, u32 src_ea)
{
    const u8 first = vm_read8(src_ea);
    switch (code) {
    case L10N_UTF8: {
        u32 cp = 0;
        const u32 count = first < 0x80 ? 1u :
            (first >= 0xC2 && first <= 0xDF ? 2u :
             (first >= 0xE0 && first <= 0xEF ? 3u :
              (first >= 0xF0 && first <= 0xF4 ? 4u : 0u)));
        if (!count)
            return 0;
        u8 bytes[4] = {0, 0, 0, 0};
        if (!guest_read_bytes(src_ea, bytes, count) ||
            decode_utf8(bytes, count, &cp) != (int)count)
            return 0;
        return count;
    }
    case L10N_UTF16:
        return (vm_read16(src_ea) >= 0xD800u && vm_read16(src_ea) <= 0xDBFFu) ? 4u : 2u;
    case L10N_UCS2:
        return 2;
    case L10N_UTF32:
    case L10N_UCS4:
        return 4;
    case L10N_CODEPAGE_932:
    case L10N_RIS_506:
        return ((first >= 0x81 && first <= 0x9F) ||
                (first >= 0xE0 && first <= 0xFC)) ? 2u : 1u;
    case L10N_EUC_JP:
        return first == 0x8Fu ? 3u :
            ((first == 0x8Eu || first >= 0xA1u) ? 2u : 1u);
    case L10N_EUC_CN:
    case L10N_EUC_KR:
        return first >= 0x80u ? 2u : 1u;
    case L10N_CODEPAGE_936:
    case L10N_CODEPAGE_949:
    case L10N_CODEPAGE_950:
        return first >= 0x81u ? 2u : 1u;
    case L10N_GB18030: {
        const u8 second = vm_read8(src_ea + 1u);
        if (first >= 0x81u && first <= 0xFEu && second >= 0x30u && second <= 0x39u)
            return 4;
        return first >= 0x81u ? 2u : 1u;
    }
    default:
        return 1;
    }
}

s32 l10n_get_converter(u32 src_code, u32 dst_code)
{
    return (s32)((src_code << 16) | (dst_code & 0xFFFFu));
}

s32 l10n_convert(s32 converter, const void* src, void* dst, u32* dst_len)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 dst_ea = (u32)(uintptr_t)dst;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!src_ea || !dst_len_ea)
        return ConverterUnknown;

    const u32 packed = (u32)converter;
    const s32 src_code = (s32)(packed >> 16);
    const s32 dst_code = (s32)(packed & 0xFFFFu);
    const u32 src_len = l10n_character_length(src_code, src_ea);
    if (!src_len)
        return SRCIllegal;

    u32 output_len = 0;
    const s32 status = l10n_convert_guest_bytes(src_code, src_ea, src_len,
        dst_code, dst_ea, vm_read32(dst_len_ea), &output_len);
    vm_write32(dst_len_ea, output_len);
    return status;
}

s32 l10n_convert_str(s32 converter, const void* src, const u32* src_len,
                     void* dst, u32* dst_len)
{
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!src_len_ea || !dst_len_ea)
        return ConverterUnknown;
    const u32 packed = (u32)converter;
    u32 output_len = 0;
    const s32 status = l10n_convert_guest_bytes((s32)(packed >> 16),
        (u32)(uintptr_t)src, vm_read32(src_len_ea), (s32)(packed & 0xFFFFu),
        (u32)(uintptr_t)dst, vm_read32(dst_len_ea), &output_len);
    vm_write32(dst_len_ea, output_len);
    return status;
}

s32 L10nConvertStr(s32 src_code, const void* src, const u32* src_len,
                   s32 dst_code, void* dst, u32* dst_len)
{
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!src_len_ea || !dst_len_ea)
        return ConverterUnknown;
    u32 output_len = 0;
    const s32 status = l10n_convert_guest_bytes(src_code,
        (u32)(uintptr_t)src, vm_read32(src_len_ea), dst_code,
        (u32)(uintptr_t)dst, vm_read32(dst_len_ea), &output_len);
    vm_write32(dst_len_ea, output_len);
    return status;
}

s32 SJISstoUTF8s(const void* src, const u32* src_len,
                 void* dst, u32* dst_len)
{
    return L10nConvertStr(L10N_CODEPAGE_932, src, src_len,
        L10N_UTF8, dst, dst_len);
}

s32 EUCJPstoUTF8s(const u8* src, const s32* src_len,
                  u8* dst, s32* dst_len)
{
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!src_len_ea || !dst_len_ea)
        return ConverterUnknown;
    const s32 input_len = (s32)vm_read32(src_len_ea);
    const s32 capacity = (s32)vm_read32(dst_len_ea);
    if (input_len < 0 || capacity < 0)
        return SRCIllegal;
    u32 output_len = 0;
    const s32 status = l10n_convert_guest_bytes(L10N_EUC_JP,
        (u32)(uintptr_t)src, (u32)input_len, L10N_UTF8,
        (u32)(uintptr_t)dst, (u32)capacity, &output_len);
    vm_write32(dst_len_ea, output_len);
    return status;
}

/* String export lengths are byte counts for byte encodings and code-unit
 * counts for UTF-16/UCS-2/UTF-32. The internal converter works in bytes. */
static u32 l10n_unit_width(s32 code)
{
    if (code == L10N_UTF16 || code == L10N_UCS2)
        return 2;
    if (code == L10N_UTF32 || code == L10N_UCS4)
        return 4;
    return 1;
}

static s32 l10n_convert_guest_span(s32 src_code, const void* src,
                                  const u32* src_len, s32 dst_code, void* dst,
                                  u32* dst_len)
{
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!src_len_ea || !dst_len_ea)
        return ConverterUnknown;

    const u32 src_units = vm_read32(src_len_ea);
    const u32 dst_capacity_units = vm_read32(dst_len_ea);
    const u32 src_width = l10n_unit_width(src_code);
    const u32 dst_width = l10n_unit_width(dst_code);
    const u64 src_bytes64 = (u64)src_units * src_width;
    const u64 dst_capacity64 = (u64)dst_capacity_units * dst_width;
    if (src_bytes64 > UINT32_MAX || dst_capacity64 > UINT32_MAX) {
        vm_write32(dst_len_ea, 0);
        return SRCIllegal;
    }

    u32 bytes_written = 0;
    const s32 status = l10n_convert_guest_bytes(src_code,
        (u32)(uintptr_t)src, (u32)src_bytes64, dst_code,
        (u32)(uintptr_t)dst, (u32)dst_capacity64, &bytes_written);
    if (bytes_written % dst_width != 0)
        return SRCIllegal;
    vm_write32(dst_len_ea, bytes_written / dst_width);
    return status;
}

/* RPCS3 declares these legacy code-page APIs with signed length pointers.
 * Keep their guest ABI, but share ps3recomp's checked converter and its
 * encoding-specific code-unit length rules. */
static s32 l10n_convert_guest_span_s32(s32 src_code, const void* src,
                                      const s32* src_len, s32 dst_code,
                                      void* dst, s32* dst_len)
{
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!src_len_ea || !dst_len_ea)
        return ConverterUnknown;

    const s32 src_units = (s32)vm_read32(src_len_ea);
    const s32 dst_capacity_units = (s32)vm_read32(dst_len_ea);
    if (src_units < 0 || dst_capacity_units < 0) {
        vm_write32(dst_len_ea, 0);
        return SRCIllegal;
    }

    const u32 src_width = l10n_unit_width(src_code);
    const u32 dst_width = l10n_unit_width(dst_code);
    const u64 src_bytes = (u64)(u32)src_units * src_width;
    const u64 dst_capacity = (u64)(u32)dst_capacity_units * dst_width;
    if (src_bytes > UINT32_MAX || dst_capacity > UINT32_MAX) {
        vm_write32(dst_len_ea, 0);
        return SRCIllegal;
    }

    u32 bytes_written = 0;
    const s32 status = l10n_convert_guest_bytes(src_code,
        (u32)(uintptr_t)src, (u32)src_bytes, dst_code,
        (u32)(uintptr_t)dst, (u32)dst_capacity, &bytes_written);
    if (bytes_written % dst_width != 0) {
        vm_write32(dst_len_ea, 0);
        return SRCIllegal;
    }
    vm_write32(dst_len_ea, bytes_written / dst_width);
    return status;
}

static int l10n_guest_u16_span_has_surrogate(u32 src_ea, u32 units)
{
    if (units && (!src_ea || (u64)src_ea + (u64)units * 2u - 1u > UINT32_MAX))
        return 1;
    for (u32 i = 0; i < units; ++i) {
        const u16 value = vm_read16(src_ea + i * 2u);
        if (value >= 0xD800u && value <= 0xDFFFu)
            return 1;
    }
    return 0;
}

static int l10n_guest_u32_span_not_ucs2(u32 src_ea, u32 units)
{
    if (units && (!src_ea || (u64)src_ea + (u64)units * 4u - 1u > UINT32_MAX))
        return 1;
    for (u32 i = 0; i < units; ++i) {
        const u32 value = vm_read32(src_ea + i * 4u);
        if (value > 0xFFFFu || (value >= 0xD800u && value <= 0xDFFFu))
            return 1;
    }
    return 0;
}

static int l10n_guest_utf8_span_not_ucs2(u32 src_ea, u32 bytes)
{
    if (bytes && (!src_ea || (u64)src_ea + bytes - 1u > UINT32_MAX))
        return 0; /* Let the checked converter report an invalid source span. */
    u32 pos = 0;
    while (pos < bytes) {
        const u32 length = l10n_character_length(L10N_UTF8, src_ea + pos);
        if (!length || length > bytes - pos)
            return 0; /* Let the converter report the malformed sequence. */
        u8 encoded[4];
        u32 cp = 0;
        if (!guest_read_bytes(src_ea + pos, encoded, length) ||
            decode_utf8(encoded, length, &cp) != (int)length)
            return 0;
        if (cp > 0xFFFFu)
            return 1;
        pos += length;
    }
    return 0;
}

/* Convert a short host buffer for the by-value SBCS APIs. The guest string
 * APIs above continue to go through the checked guest-memory converter. */
static s32 l10n_convert_host_bytes(s32 src_code, const u8* src, u32 src_len,
                                   s32 dst_code, u8* dst, u32 dst_capacity,
                                   u32* dst_len)
{
    if (!dst_len || (!src && src_len) || (!dst && dst_capacity))
        return ConverterUnknown;
    *dst_len = 0;
    if (!src_len)
        return ConversionOK;

#ifdef _WIN32
    const UINT source_cp = l10n_win_codepage(src_code);
    const UINT target_cp = l10n_win_codepage(dst_code);
    if (!source_cp || !target_cp || src_len > INT_MAX)
        return ConverterUnknown;
    const DWORD source_flags = (source_cp == CP_UTF8 || source_cp == 54936)
        ? MB_ERR_INVALID_CHARS : 0;
    const int wide_count = MultiByteToWideChar(source_cp, source_flags,
        (const char*)src, (int)src_len, NULL, 0);
    if (wide_count <= 0)
        return SRCIllegal;
    wchar_t* wide = (wchar_t*)malloc((size_t)wide_count * sizeof(wchar_t));
    if (!wide)
        return DSTExhausted;
    if (MultiByteToWideChar(source_cp, source_flags, (const char*)src,
            (int)src_len, wide, wide_count) != wide_count) {
        free(wide);
        return SRCIllegal;
    }
    const int target_unicode = target_cp == CP_UTF8 || target_cp == 1200 ||
        target_cp == 1201 || target_cp == 12000 || target_cp == 12001;
    const DWORD target_flags = (target_cp == CP_UTF8 || target_cp == 54936)
        ? WC_ERR_INVALID_CHARS
        : ((target_cp == 1201 || target_cp == 12001) ? 0 : WC_NO_BEST_FIT_CHARS);
    BOOL used_default = FALSE;
    const int required = WideCharToMultiByte(target_cp, target_flags, wide,
        wide_count, NULL, 0, target_unicode ? NULL : "?",
        target_unicode ? NULL : &used_default);
    if (required <= 0) {
        free(wide);
        return ConverterUnknown;
    }
    if ((u32)required > dst_capacity) {
        free(wide);
        return DSTExhausted;
    }
    used_default = FALSE;
    const int written = WideCharToMultiByte(target_cp, target_flags, wide,
        wide_count, (char*)dst, required,
        target_unicode ? NULL : "?", target_unicode ? NULL : &used_default);
    free(wide);
    if (written != required || used_default)
        return SRCIllegal;
    *dst_len = (u32)written;
    return ConversionOK;
#else
    const char* source_encoding = l10n_iconv_encoding(src_code);
    const char* target_encoding = l10n_iconv_encoding(dst_code);
    if (!source_encoding || !target_encoding)
        return ConverterUnknown;
    iconv_t converter = iconv_open(target_encoding, source_encoding);
    if (converter == (iconv_t)-1)
        return ConverterUnknown;
    char* input = (char*)src;
    size_t input_left = src_len;
    char* output = (char*)dst;
    size_t output_left = dst_capacity;
    errno = 0;
    size_t result = iconv(converter, &input, &input_left, &output, &output_left);
    s32 status = ConversionOK;
    if (result == (size_t)-1)
        status = errno == E2BIG ? DSTExhausted : SRCIllegal;
    else if (input_left)
        status = SRCIllegal;
    if (status == ConversionOK) {
        errno = 0;
        result = iconv(converter, NULL, NULL, &output, &output_left);
        if (result == (size_t)-1)
            status = errno == E2BIG ? DSTExhausted : SRCIllegal;
    }
    *dst_len = dst_capacity - (u32)output_left;
    iconv_close(converter);
    return status;
#endif
}

static int l10n_code_page_supported(s32 code)
{
#ifdef _WIN32
    return l10n_win_codepage(code) != 0;
#else
    return l10n_iconv_encoding(code) != NULL;
#endif
}

static s32 l10n_sbcs_character_from(s32 src_code, u8 src, s32 dst_code,
                                    u8* dst, u32 capacity, u32* written)
{
    return l10n_convert_host_bytes(src_code, &src, 1, dst_code,
        dst, capacity, written);
}

s32 EUCJPstoJISs(const u8* src, const u32* src_len, u8* dst, u32* dst_len)
{
    return l10n_convert_guest_span(L10N_EUC_JP, src, src_len,
        L10N_ISO_2022_JP, dst, dst_len);
}

s32 SBCSstoUCS2s(const u8* src, u32* src_len, u16* dst,
                 u32* dst_len, u32 code_page)
{
    return l10n_convert_guest_span((s32)code_page, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 SBCSstoUTF8s(const u8* src, u32* src_len, u8* dst,
                 u32* dst_len, u32 code_page)
{
    return l10n_convert_guest_span((s32)code_page, src, src_len,
        L10N_UTF8, dst, dst_len);
}

s32 SBCStoUCS2(u8 src, u16* dst, u32 code_page)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (!l10n_code_page_supported((s32)code_page))
        return -1;
    if (!dst_ea)
        return 0;
    if (src < 0x80u) {
        vm_write16(dst_ea, src);
        return 1;
    }
    u8 converted[4];
    u32 converted_len = 0;
    const s32 status = l10n_sbcs_character_from((s32)code_page, src,
        L10N_UCS2, converted, sizeof(converted), &converted_len);
    if (status != ConversionOK || converted_len != 2)
        return 0;
    vm_write16(dst_ea, (u16)(((u16)converted[0] << 8) | converted[1]));
    return 1;
}

s32 SBCStoUTF8(u8 src, u8* dst, u32 code_page)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (!l10n_code_page_supported((s32)code_page))
        return -1;
    if (!dst_ea)
        return 0;
    if (src < 0x80u) {
        vm_write8(dst_ea, src);
        return 1;
    }
    u8 converted[4];
    u32 converted_len = 0;
    const s32 status = l10n_sbcs_character_from((s32)code_page, src,
        L10N_UTF8, converted, sizeof(converted), &converted_len);
    if (status != ConversionOK || !converted_len || converted_len > sizeof(converted))
        return 0;
    return guest_write_bytes(dst_ea, converted, converted_len)
        ? (s32)converted_len : 0;
}

s32 SJISstoUCS2s(const u8* src, const u32* src_len,
                 u16* dst, u32* dst_len)
{
    return l10n_convert_guest_span(L10N_CODEPAGE_932, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UCS2stoSBCSs(const u16* src, u32* src_len, u8* dst,
                 u32* dst_len, u32 code_page)
{
    if (!l10n_code_page_supported((s32)code_page))
        return ConverterUnknown;
    return l10n_convert_guest_span(L10N_UCS2, src, src_len,
        (s32)code_page, dst, dst_len);
}

s32 UCS2stoUTF16s(const u16* src, u32* src_len, u16* dst, u32* dst_len)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    if (!src_len_ea)
        return ConverterUnknown;
    if (l10n_guest_u16_span_has_surrogate(src_ea, vm_read32(src_len_ea)))
        return SRCIllegal;
    return l10n_convert_guest_span(L10N_UCS2, src, src_len,
        L10N_UTF16, dst, dst_len);
}

s32 UCS2stoUTF32s(const u16* src, u32* src_len, u32* dst, u32* dst_len)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    if (!src_len_ea)
        return ConverterUnknown;
    if (l10n_guest_u16_span_has_surrogate(src_ea, vm_read32(src_len_ea)))
        return SRCIllegal;
    return l10n_convert_guest_span(L10N_UCS2, src, src_len,
        L10N_UTF32, dst, dst_len);
}

s32 UCS2toSBCS(u16 src, u8* dst, u32 code_page)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (!l10n_code_page_supported((s32)code_page))
        return -1;
    if (src >= 0xFFFEu || (src >= 0xD800u && src <= 0xDFFFu))
        return 0;
    if (!dst_ea)
        return 0;
    if (src < 0x80u) {
        vm_write8(dst_ea, (u8)src);
        return 1;
    }
    const u8 encoded_src[2] = {(u8)(src >> 8), (u8)src};
    u8 converted[8];
    u32 converted_len = 0;
    const s32 status = l10n_convert_host_bytes(L10N_UCS2, encoded_src, 2,
        (s32)code_page, converted, sizeof(converted), &converted_len);
    if (status != ConversionOK || converted_len != 1)
        return 0;
    vm_write8(dst_ea, converted[0]);
    return 1;
}

s32 UCS2toUTF16(u16 src, u16* dst)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (src >= 0xD800u && src <= 0xDFFFu)
        return 0;
    if (!dst_ea)
        return 0;
    vm_write16(dst_ea, src);
    return 1;
}

s32 UCS2toUTF32(u16 src, u32* dst)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (src >= 0xD800u && src <= 0xDFFFu)
        return 0;
    if (!dst_ea)
        return 0;
    vm_write32(dst_ea, src);
    return 1;
}

s32 UTF16stoUCS2s(const u16* src, u32* src_len, u16* dst, u32* dst_len)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    if (!src_len_ea)
        return ConverterUnknown;
    if (l10n_guest_u16_span_has_surrogate(src_ea, vm_read32(src_len_ea)))
        return SRCIllegal;
    return l10n_convert_guest_span(L10N_UTF16, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UTF16stoUTF32s(const u16* src, u32* src_len, u32* dst, u32* dst_len)
{
    return l10n_convert_guest_span(L10N_UTF16, src, src_len,
        L10N_UTF32, dst, dst_len);
}

s32 UTF16toUCS2(const u16* src, u16* dst)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (!src_ea)
        return 0;
    const u16 value = vm_read16(src_ea);
    if (value >= 0xD800u && value <= 0xDFFFu)
        return 0;
    if (!dst_ea)
        return 0;
    vm_write16(dst_ea, value);
    return 1;
}

s32 UTF32stoUCS2s(const u32* src, u32* src_len, u16* dst, u32* dst_len)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    if (!src_len_ea)
        return ConverterUnknown;
    if (l10n_guest_u32_span_not_ucs2(src_ea, vm_read32(src_len_ea)))
        return SRCIllegal;
    return l10n_convert_guest_span(L10N_UTF32, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UTF32stoUTF16s(const u32* src, u32* src_len, u16* dst, u32* dst_len)
{
    return l10n_convert_guest_span(L10N_UTF32, src, src_len,
        L10N_UTF16, dst, dst_len);
}

s32 UTF32stoUTF8s(const u32* src, u32* src_len, u8* dst, u32* dst_len)
{
    return l10n_convert_guest_span(L10N_UTF32, src, src_len,
        L10N_UTF8, dst, dst_len);
}

s32 UTF32toUCS2(u32 src, u16* dst)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (src > 0xFFFFu || (src >= 0xD800u && src <= 0xDFFFu))
        return 0;
    if (!dst_ea)
        return 0;
    vm_write16(dst_ea, (u16)src);
    return 1;
}

s32 UTF8stoSBCSs(const u8* src, u32* src_len, u8* dst,
                 u32* dst_len, u32 code_page)
{
    if (!l10n_code_page_supported((s32)code_page))
        return ConverterUnknown;
    return l10n_convert_guest_span(L10N_UTF8, src, src_len,
        (s32)code_page, dst, dst_len);
}

s32 UTF8stoUCS2s(const u8* src, u32* src_len, u16* dst, u32* dst_len)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 src_len_ea = (u32)(uintptr_t)src_len;
    if (!src_len_ea)
        return ConverterUnknown;
    const u32 bytes = vm_read32(src_len_ea);
    if (l10n_guest_utf8_span_not_ucs2(src_ea, bytes))
        return SRCIllegal;
    return l10n_convert_guest_span(L10N_UTF8, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UTF8stoUTF32s(const u8* src, u32* src_len, u32* dst, u32* dst_len)
{
    return l10n_convert_guest_span(L10N_UTF8, src, src_len,
        L10N_UTF32, dst, dst_len);
}

s32 UTF8toSBCS(const u8* src, u8* dst, u32 code_page)
{
    const u32 src_ea = (u32)(uintptr_t)src;
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (!l10n_code_page_supported((s32)code_page))
        return -1;
    if (!src_ea || !dst_ea)
        return 0;
    const u32 source_len = l10n_character_length(L10N_UTF8, src_ea);
    if (!source_len)
        return 0;
    u8 utf8[4];
    u32 cp = 0;
    if (!guest_read_bytes(src_ea, utf8, source_len) ||
        decode_utf8(utf8, source_len, &cp) != (int)source_len || cp > 0xFFFFu)
        return 0;
    const u16 ucs2 = (u16)cp;
    const u8 encoded_src[2] = {(u8)(ucs2 >> 8), (u8)ucs2};
    u8 converted[8];
    u32 converted_len = 0;
    const s32 status = l10n_convert_host_bytes(L10N_UCS2, encoded_src, 2,
        (s32)code_page, converted, sizeof(converted), &converted_len);
    if (status != ConversionOK || converted_len != 1)
        return 0;
    vm_write8(dst_ea, converted[0]);
    return (s32)source_len;
}

/* RPCS3_REFERENCE_REVIEWED (RPCS3 commit 2ba2f75bc): exported signatures and
 * encoding pairs were checked against cellL10n.cpp. RPCS3 currently leaves
 * these handlers as TODO success returns; these implementations use the
 * existing ps3recomp converter, guest-memory access, and status semantics. */
s32 JISstoUTF8s(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_ISO_2022_JP, src, src_len,
        L10N_UTF8, dst, dst_len);
}

s32 HZstoUTF8s(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_HZ, src, src_len,
        L10N_UTF8, dst, dst_len);
}

s32 SJISstoJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_CODEPAGE_932, src, src_len,
        L10N_ISO_2022_JP, dst, dst_len);
}

s32 HZstoUCS2s(const u8* src, const s32* src_len, u16* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_HZ, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UCS2stoHZs(const u16* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_UCS2, src, src_len,
        L10N_HZ, dst, dst_len);
}

s32 UCS2stoSJISs(const u16* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_UCS2, src, src_len,
        L10N_CODEPAGE_932, dst, dst_len);
}

s32 EUCKRstoUCS2s(const u8* src, const s32* src_len, u16* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_EUC_KR, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UHCstoEUCKRs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_CODEPAGE_949, src, src_len,
        L10N_EUC_KR, dst, dst_len);
}

s32 EUCJPstoSJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_EUC_JP, src, src_len,
        L10N_CODEPAGE_932, dst, dst_len);
}

s32 EUCKRstoUHCs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_EUC_KR, src, src_len,
        L10N_CODEPAGE_949, dst, dst_len);
}

s32 EUCJPstoUCS2s(const u8* src, const s32* src_len, u16* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_EUC_JP, src, src_len,
        L10N_UCS2, dst, dst_len);
}

s32 UTF8stoJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_UTF8, src, src_len,
        L10N_ISO_2022_JP, dst, dst_len);
}

s32 BIG5stoUTF8s(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_CODEPAGE_950, src, src_len,
        L10N_UTF8, dst, dst_len);
}

s32 UCS2stoEUCKRs(const u16* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_UCS2, src, src_len,
        L10N_EUC_KR, dst, dst_len);
}

s32 SJISstoEUCJPs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_CODEPAGE_932, src, src_len,
        L10N_EUC_JP, dst, dst_len);
}

s32 JISstoSJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len)
{
    return l10n_convert_guest_span_s32(L10N_ISO_2022_JP, src, src_len,
        L10N_CODEPAGE_932, dst, dst_len);
}

s32 SJIStoUTF8(u8 ch, void* dst, s32* dst_len)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    const u32 dst_len_ea = (u32)(uintptr_t)dst_len;
    if (!dst_len_ea)
        return ConverterUnknown;

    const s32 capacity = (s32)vm_read32(dst_len_ea);
    if (capacity < 0) {
        vm_write32(dst_len_ea, 0);
        return SRCIllegal;
    }

    u8 converted[4];
    u32 converted_len = 0;
    const s32 status = l10n_convert_host_bytes(L10N_CODEPAGE_932, &ch, 1,
        L10N_UTF8, converted, dst_ea ? (u32)capacity : sizeof(converted),
        &converted_len);
    if (status != ConversionOK) {
        vm_write32(dst_len_ea, 0);
        return status;
    }
    if (dst_ea && !guest_write_bytes(dst_ea, converted, converted_len)) {
        vm_write32(dst_len_ea, 0);
        return SRCIllegal;
    }
    vm_write32(dst_len_ea, converted_len);
    return ConversionOK;
}

s32 UCS2toSJIS(u16 ch, void* dst)
{
    const u32 dst_ea = (u32)(uintptr_t)dst;
    if (!dst_ea || (ch >= 0xD800u && ch <= 0xDFFFu))
        return 0;

    const u8 encoded_src[2] = {(u8)(ch >> 8), (u8)ch};
    u8 converted[2];
    u32 converted_len = 0;
    const s32 status = l10n_convert_host_bytes(L10N_UCS2, encoded_src,
        sizeof(encoded_src), L10N_CODEPAGE_932, converted,
        sizeof(converted), &converted_len);
    if (status != ConversionOK || converted_len == 0 || converted_len > 2)
        return 0;
    return guest_write_bytes(dst_ea, converted, converted_len)
        ? (s32)converted_len : 0;
}
