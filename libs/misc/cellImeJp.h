/*
 * ps3recomp - cellImeJp HLE
 *
 * Japanese Input Method Editor. Handles kana-to-kanji conversion.
 * Stub — init/end lifecycle, input accepted but conversion is passthrough.
 */

#ifndef PS3RECOMP_CELL_IMEJP_H
#define PS3RECOMP_CELL_IMEJP_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define CELL_IMEJP_ERROR_NOT_INITIALIZED     0x80040601
#define CELL_IMEJP_ERROR_ALREADY_INITIALIZED 0x80040602
#define CELL_IMEJP_ERROR_INVALID_ARGUMENT    0x80040603
#define CELL_IMEJP_ERROR_OUT_OF_MEMORY       0x80040604
#define CELL_IMEJP_ERROR_NOT_SUPPORTED       0x80040605
#define CELL_IMEJP_ERROR_CONTEXT_DESTROYED   0x80040606

/* cellImeJpUtility error and state values. */
#define CELL_IMEJP_UTILITY_ERROR_ERR         ((s32)0x8002BF01u)
#define CELL_IMEJP_UTILITY_ERROR_CONTEXT    ((s32)0x8002BF11u)
#define CELL_IMEJP_UTILITY_ERROR_PARAM      ((s32)0x8002BF41u)
#define CELL_IMEJP_UTILITY_BEFORE_INPUT     0
#define CELL_IMEJP_UTILITY_BEFORE_CONVERT   1
#define CELL_IMEJP_UTILITY_CANDIDATES       7
#define CELL_IMEJP_UTILITY_RET_CONFIRMED   2
#define CELL_IMEJP_UTILITY_ROMAN_INPUT     0
#define CELL_IMEJP_UTILITY_KANA_INPUT      1
#define CELL_IMEJP_UTILITY_DSPCHAR_HIRA    1
#define CELL_IMEJP_UTILITY_DSPCHAR_FKANA   2
#define CELL_IMEJP_UTILITY_DSPCHAR_RAW     3
#define CELL_IMEJP_UTILITY_DSPCHAR_HKANA   4
#define CELL_IMEJP_UTILITY_DSPCHAR_HRAW    5
#define CELL_IMEJP_UTILITY_EXTENSION_MASK  0x001Du
#define CELL_IMEJP_UTILITY_STRING_MAXLENGTH 128

/* Input mode */
#define CELL_IMEJP_INPUT_MODE_HIRAGANA      0
#define CELL_IMEJP_INPUT_MODE_KATAKANA      1
#define CELL_IMEJP_INPUT_MODE_HALF_KATAKANA 2
#define CELL_IMEJP_INPUT_MODE_ALPHA         3

/* Max text lengths */
#define CELL_IMEJP_MAX_INPUT_LENGTH   256
#define CELL_IMEJP_MAX_RESULT_LENGTH  512

/* Handle */
typedef u32 CellImeJpHandle;

/* Config */
typedef struct CellImeJpConfig {
    void* dictionary;
    u32 dictionarySize;
    u32 inputMode;
    u32 flags;
    u32 reserved[4];
} CellImeJpConfig;

/* Functions */
s32 cellImeJpOpen(const CellImeJpConfig* config, CellImeJpHandle* handle);
s32 cellImeJpClose(CellImeJpHandle handle);
s32 cellImeJpReset(CellImeJpHandle handle);

s32 cellImeJpSetInputMode(CellImeJpHandle handle, u32 mode);
s32 cellImeJpGetInputMode(CellImeJpHandle handle, u32* mode);

s32 cellImeJpAddChar(CellImeJpHandle handle, u16 ch);
s32 cellImeJpBackspace(CellImeJpHandle handle);
s32 cellImeJpConfirm(CellImeJpHandle handle);

s32 cellImeJpGetInputString(CellImeJpHandle handle, u16* str, u32* len);
s32 cellImeJpGetConvertedString(CellImeJpHandle handle, u16* str, u32* len);

/* Utility APIs use an identity-conversion fallback when no dictionary is
 * available. Candidate mode then exposes the unchanged input as its single
 * candidate; dictionary-specific clause operations report NOT_SUPPORTED. */
s32 cellImeJpGetStatus(CellImeJpHandle handle, s16* status);
s32 cellImeJpEnterChar(CellImeJpHandle handle, u16 ch, s16* outputStatus);
s32 cellImeJpSetKanaInputMode(CellImeJpHandle handle, s16 mode);
s32 cellImeJpAllowExtensionCharacters(CellImeJpHandle handle, s16 extensions);
s32 cellImeJpSetInputCharType(CellImeJpHandle handle, s16 type);
s32 cellImeJpConvertForward(CellImeJpHandle handle);
s32 cellImeJpConvertBackward(CellImeJpHandle handle);
s32 cellImeJpCurrentPartConfirm(CellImeJpHandle handle, s16 listItem);
s32 cellImeJpAllConfirm(CellImeJpHandle handle);
s32 cellImeJpConvertCancel(CellImeJpHandle handle);
s32 cellImeJpExtendConvertArea(CellImeJpHandle handle);
s32 cellImeJpShortenConvertArea(CellImeJpHandle handle);
s32 cellImeJpTemporalConfirm(CellImeJpHandle handle, s16 selectIndex);
s32 cellImeJpPostConvert(CellImeJpHandle handle, s16 postType);
s32 cellImeJpMoveFocusClause(CellImeJpHandle handle, s16 moveType);
s32 cellImeJpGetFocusTop(CellImeJpHandle handle, s16* focusTop);
s32 cellImeJpGetFocusLength(CellImeJpHandle handle, s16* focusLength);
s32 cellImeJpGetConfirmString(CellImeJpHandle handle, u16* string);
s32 cellImeJpGetConvertString(CellImeJpHandle handle, u16* string);
s32 cellImeJpGetCandidateListSize(CellImeJpHandle handle, s16* listSize);
s32 cellImeJpGetCandidateList(CellImeJpHandle handle, s16* listNum,
                              u16* candidateString);
s32 cellImeJpGetCandidateSelect(CellImeJpHandle handle, s16* index);
s32 cellImeJpBackspaceWord(CellImeJpHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_IMEJP_H */
