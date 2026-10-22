/*
 * ps3recomp - cellL10n HLE
 *
 * Character encoding conversion: UTF-8, UTF-16, UTF-32, Shift-JIS, EUC-JP,
 * ISO-8859-1, and other encodings used by PS3 titles.
 */

#ifndef PS3RECOMP_CELL_L10N_H
#define PS3RECOMP_CELL_L10N_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Error codes
 * -----------------------------------------------------------------------*/
#define CELL_L10N_ERROR_NOT_SUPPORTED      0x80610140
#define CELL_L10N_ERROR_ILLEGAL_SEQUENCE   0x80610141
#define CELL_L10N_ERROR_INSUFFICIENT       0x80610142
#define CELL_L10N_ERROR_INVALID_PARAM      0x80610143

/* ---------------------------------------------------------------------------
 * Encoding IDs
 * -----------------------------------------------------------------------*/
enum {
    L10N_UTF8 = 0,
    L10N_UTF16,
    L10N_UTF32,
    L10N_UCS2,
    L10N_UCS4,
    L10N_ISO_8859_1,
    L10N_ISO_8859_2,
    L10N_ISO_8859_3,
    L10N_ISO_8859_4,
    L10N_ISO_8859_5,
    L10N_ISO_8859_6,
    L10N_ISO_8859_7,
    L10N_ISO_8859_8,
    L10N_ISO_8859_9,
    L10N_ISO_8859_10,
    L10N_ISO_8859_11,
    L10N_ISO_8859_13,
    L10N_ISO_8859_14,
    L10N_ISO_8859_15,
    L10N_ISO_8859_16,
    L10N_CODEPAGE_437,
    L10N_CODEPAGE_850,
    L10N_CODEPAGE_863,
    L10N_CODEPAGE_866,
    L10N_CODEPAGE_932,
    L10N_SHIFT_JIS = L10N_CODEPAGE_932,
    L10N_CODEPAGE_936,
    L10N_GBK = L10N_CODEPAGE_936,
    L10N_CODEPAGE_949,
    L10N_UHC = L10N_CODEPAGE_949,
    L10N_CODEPAGE_950,
    L10N_BIG5 = L10N_CODEPAGE_950,
    L10N_CODEPAGE_1251,
    L10N_CODEPAGE_1252,
    L10N_EUC_CN,
    L10N_EUC_JP,
    L10N_EUC_KR,
    L10N_ISO_2022_JP,
    L10N_JIS = L10N_ISO_2022_JP,
    L10N_ARIB,
    L10N_HZ,
    L10N_GB18030,
    L10N_RIS_506,
    L10N_MUSIC_SHIFT_JIS = L10N_RIS_506,
    L10N_CODEPAGE_852,
    L10N_CODEPAGE_1250,
    L10N_CODEPAGE_737,
    L10N_CODEPAGE_1253,
    L10N_CODEPAGE_857,
    L10N_CODEPAGE_1254,
    L10N_CODEPAGE_775,
    L10N_CODEPAGE_1257,
    L10N_CODEPAGE_855,
    L10N_CODEPAGE_858,
    L10N_CODEPAGE_860,
    L10N_CODEPAGE_861,
    L10N_CODEPAGE_865,
    L10N_CODEPAGE_869,
    L10N_CODEPAGE_MAX
};

enum {
    ConversionOK = 0,
    SRCIllegal = 1,
    DSTExhausted = 2,
    ConverterUnknown = 3
};

/* ---------------------------------------------------------------------------
 * Unicode BOM
 * -----------------------------------------------------------------------*/
#define L10N_BOM_NONE           0
#define L10N_BOM_PREPEND        1

/* ---------------------------------------------------------------------------
 * Single-character conversion
 * -----------------------------------------------------------------------*/

/* UTF-8 <-> UTF-16 */
s32 UTF8toUTF16(const u8* utf8, u32 utf8_len,
                u16* utf16, u32* utf16_len);
s32 UTF16toUTF8(const u16* utf16, u32 utf16_len,
                u8* utf8, u32* utf8_len);

/* UTF-8 <-> UTF-32 */
s32 UTF8toUTF32(const u8* utf8, u32 utf8_len,
                u32* utf32, u32* utf32_len);
s32 UTF32toUTF8(const u32* utf32, u32 utf32_len,
                u8* utf8, u32* utf8_len);

/* UTF-16 <-> UTF-32 */
s32 UTF16toUTF32(const u16* utf16, u32 utf16_len,
                 u32* utf32, u32* utf32_len);
s32 UTF32toUTF16(const u32* utf32, u32 utf32_len,
                 u16* utf16, u32* utf16_len);

/* Single-character UCS-2 conversion. Return the number of source UTF-8 bytes
 * or destination UTF-8 bytes; zero means an invalid sequence. */
s32 UTF8toUCS2(const u8* utf8, u16* ucs2);
s32 UCS2toUTF8(u16 ucs2, u8* utf8);

/* Bounded UCS-2 string conversion: lengths are UCS-2 code units and UTF-8
 * bytes. A null destination performs a sizing pass. */
s32 UCS2stoUTF8s(const u16* ucs2, u32* ucs2_len,
                 u8* utf8, u32* utf8_len);

/* Single-character Shift-JIS conversions used by the PS3 cellL10n ABI. */
s32 SJIStoUCS2(const u8* sjis, u16* ucs2);
s32 UTF8toSJIS(u8 utf8_byte, u8* sjis, u32* sjis_len);

/* ---------------------------------------------------------------------------
 * String conversion (null-terminated)
 * -----------------------------------------------------------------------*/
s32 UTF8stoUTF16s(const u8* utf8, u32* utf8_len,
                  u16* utf16, u32* utf16_len);
s32 UTF16stoUTF8s(const u16* utf16, u32* utf16_len,
                  u8* utf8, u32* utf8_len);

/* ---------------------------------------------------------------------------
 * Utility
 * -----------------------------------------------------------------------*/

/* Count bytes needed for conversion */
s32 UTF8toUTF16Size(const u8* utf8, u32 utf8_len, u32* out_size);
s32 UTF16toUTF8Size(const u16* utf16, u32 utf16_len, u32* out_size);

/* Encoding queries */
s32 l10n_get_converter(u32 src_code, u32 dst_code);
s32 l10n_convert(s32 converter, const void* src, void* dst, u32* dst_len);
s32 l10n_convert_str(s32 converter, const void* src, const u32* src_len,
                     void* dst, u32* dst_len);
s32 L10nConvertStr(s32 src_code, const void* src, const u32* src_len,
                   s32 dst_code, void* dst, u32* dst_len);
s32 SJISstoUTF8s(const void* src, const u32* src_len,
                 void* dst, u32* dst_len);
s32 EUCJPstoUTF8s(const u8* src, const s32* src_len,
                  u8* dst, s32* dst_len);

/* Additional conversions supported by RPCS3's current cellL10n export ABI. */
s32 EUCJPstoJISs(const u8* src, const u32* src_len,
                 u8* dst, u32* dst_len);
s32 SBCSstoUCS2s(const u8* src, u32* src_len, u16* dst,
                 u32* dst_len, u32 code_page);
s32 SBCSstoUTF8s(const u8* src, u32* src_len, u8* dst,
                 u32* dst_len, u32 code_page);
s32 SBCStoUCS2(u8 src, u16* dst, u32 code_page);
s32 SBCStoUTF8(u8 src, u8* dst, u32 code_page);
s32 SJISstoUCS2s(const u8* src, const u32* src_len,
                 u16* dst, u32* dst_len);
s32 UCS2stoSBCSs(const u16* src, u32* src_len, u8* dst,
                 u32* dst_len, u32 code_page);
s32 UCS2stoUTF16s(const u16* src, u32* src_len, u16* dst, u32* dst_len);
s32 UCS2stoUTF32s(const u16* src, u32* src_len, u32* dst, u32* dst_len);
s32 UCS2toSBCS(u16 src, u8* dst, u32 code_page);
s32 UCS2toUTF16(u16 src, u16* dst);
s32 UCS2toUTF32(u16 src, u32* dst);
s32 UTF16stoUCS2s(const u16* src, u32* src_len, u16* dst, u32* dst_len);
s32 UTF16stoUTF32s(const u16* src, u32* src_len, u32* dst, u32* dst_len);
s32 UTF16toUCS2(const u16* src, u16* dst);
s32 UTF32stoUCS2s(const u32* src, u32* src_len, u16* dst, u32* dst_len);
s32 UTF32stoUTF16s(const u32* src, u32* src_len, u16* dst, u32* dst_len);
s32 UTF32stoUTF8s(const u32* src, u32* src_len, u8* dst, u32* dst_len);
s32 UTF32toUCS2(u32 src, u16* dst);
s32 UTF8stoSBCSs(const u8* src, u32* src_len, u8* dst,
                 u32* dst_len, u32 code_page);
s32 UTF8stoUCS2s(const u8* src, u32* src_len, u16* dst, u32* dst_len);
s32 UTF8stoUTF32s(const u8* src, u32* src_len, u32* dst, u32* dst_len);
s32 UTF8toSBCS(const u8* src, u8* dst, u32 code_page);

/* Legacy signed-length code-page conversions. Lengths count source and
 * destination code units (bytes for multibyte encodings, UCS-2 units for
 * UCS-2); destination length is both input capacity and output count. */
s32 JISstoUTF8s(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 HZstoUTF8s(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 SJISstoJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 HZstoUCS2s(const u8* src, const s32* src_len, u16* dst, s32* dst_len);
s32 UCS2stoHZs(const u16* src, const s32* src_len, u8* dst, s32* dst_len);
s32 UCS2stoSJISs(const u16* src, const s32* src_len, u8* dst, s32* dst_len);
s32 EUCKRstoUCS2s(const u8* src, const s32* src_len, u16* dst, s32* dst_len);
s32 UHCstoEUCKRs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 EUCJPstoSJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 EUCKRstoUHCs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 EUCJPstoUCS2s(const u8* src, const s32* src_len, u16* dst, s32* dst_len);
s32 UTF8stoJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 BIG5stoUTF8s(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 UCS2stoEUCKRs(const u16* src, const s32* src_len, u8* dst, s32* dst_len);
s32 SJISstoEUCJPs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 JISstoSJISs(const u8* src, const s32* src_len, u8* dst, s32* dst_len);
s32 SJIStoUTF8(u8 ch, void* dst, s32* dst_len);
s32 UCS2toSJIS(u16 ch, void* dst);

/* Single codepoint length in UTF-8 */
s32 UTF8Len(const u8* utf8);
/* Single codepoint length in UTF-16 */
s32 UTF16Len(const u16* utf16);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_L10N_H */
