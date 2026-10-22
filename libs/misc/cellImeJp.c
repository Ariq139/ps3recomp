/*
 * ps3recomp - cellImeJp HLE implementation
 *
 * Japanese Input Method Editor.
 * The local backend accepts and returns text without Japanese dictionary
 * conversion. Utility candidate mode exposes an identity candidate, while
 * operations that require dictionary clause segmentation report unsupported.
 */

#include "cellImeJp.h"
#include <stdio.h>
#include <string.h>
#include "../../runtime/ppu/ppu_memory.h"   /* vm_write*: guest EA -> host, byte-swapped */
#include "../guest_struct.h"   /* GUEST_EA, guest_struct_load/store */

/* Internal state */

#define IMEJP_MAX_HANDLES 4

typedef struct {
    int  in_use;
    u32  inputMode;
    u16  input[CELL_IMEJP_MAX_INPUT_LENGTH];
    u32  inputLen;
    u16  confirmed[CELL_IMEJP_MAX_RESULT_LENGTH];
    u32  confirmedLen;
    s16  utilityState;
    s16  kanaInputMode;
    s16  inputCharType;
    s16  allowedExtensions;
} ImeJpSlot;

static ImeJpSlot s_slots[IMEJP_MAX_HANDLES];

static int imejp_alloc(void)
{
    for (int i = 0; i < IMEJP_MAX_HANDLES; i++) {
        if (!s_slots[i].in_use) return i;
    }
    return -1;
}

static ImeJpSlot* imejp_slot(CellImeJpHandle handle)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return NULL;
    return &s_slots[handle];
}

static s32 imejp_utility_slot(CellImeJpHandle handle, ImeJpSlot** out)
{
    ImeJpSlot* slot = imejp_slot(handle);
    if (!slot) return CELL_IMEJP_UTILITY_ERROR_CONTEXT;
    if (out) *out = slot;
    return CELL_OK;
}

static void imejp_write_fixed_string(u32 ea, const u16* text, u32 length,
                                    u32 capacity)
{
    if (capacity == 0) return;
    const u32 max = capacity - 1;
    if (length > max) length = max;
    for (u32 i = 0; i < length; ++i)
        vm_write16(ea + i * 2u, text[i]);
    for (u32 i = length; i <= max; ++i)
        vm_write16(ea + i * 2u, 0);
}

/* API */

s32 cellImeJpOpen(const CellImeJpConfig* config, CellImeJpHandle* handle)
{
    printf("[cellImeJp] Open()\n");
    if (!handle) return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    int slot = imejp_alloc();
    if (slot < 0) return (s32)CELL_IMEJP_ERROR_OUT_OF_MEMORY;

    s_slots[slot].in_use = 1;
    s_slots[slot].inputMode = config ? vm_read32(GUEST_EA(config))   /* inputMode */
                                     : CELL_IMEJP_INPUT_MODE_HIRAGANA;
    s_slots[slot].inputLen = 0;
    s_slots[slot].confirmedLen = 0;
    s_slots[slot].utilityState = CELL_IMEJP_UTILITY_BEFORE_INPUT;
    s_slots[slot].kanaInputMode = CELL_IMEJP_UTILITY_ROMAN_INPUT;
    s_slots[slot].inputCharType = CELL_IMEJP_UTILITY_DSPCHAR_HIRA;
    s_slots[slot].allowedExtensions = CELL_IMEJP_UTILITY_EXTENSION_MASK;
    memset(s_slots[slot].input, 0, sizeof(s_slots[slot].input));
    memset(s_slots[slot].confirmed, 0, sizeof(s_slots[slot].confirmed));

    vm_write32((u32)(uintptr_t)handle, (CellImeJpHandle)slot);
    return CELL_OK;
}

s32 cellImeJpClose(CellImeJpHandle handle)
{
    printf("[cellImeJp] Close(%u)\n", handle);
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    memset(&s_slots[handle], 0, sizeof(s_slots[handle]));
    return CELL_OK;
}

s32 cellImeJpReset(CellImeJpHandle handle)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    s_slots[handle].inputLen = 0;
    s_slots[handle].confirmedLen = 0;
    s_slots[handle].utilityState = CELL_IMEJP_UTILITY_BEFORE_INPUT;
    memset(s_slots[handle].input, 0, sizeof(s_slots[handle].input));
    memset(s_slots[handle].confirmed, 0, sizeof(s_slots[handle].confirmed));
    return CELL_OK;
}

s32 cellImeJpSetInputMode(CellImeJpHandle handle, u32 mode)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    s_slots[handle].inputMode = mode;
    return CELL_OK;
}

s32 cellImeJpGetInputMode(CellImeJpHandle handle, u32* mode)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;
    if (!mode) return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    vm_write32((u32)(uintptr_t)mode, (u32)s_slots[handle].inputMode);
    return CELL_OK;
}

s32 cellImeJpAddChar(CellImeJpHandle handle, u16 ch)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    ImeJpSlot* s = &s_slots[handle];
    if (s->inputLen >= CELL_IMEJP_MAX_INPUT_LENGTH - 1)
        return (s32)CELL_IMEJP_ERROR_OUT_OF_MEMORY;

    s->input[s->inputLen++] = ch;
    s->utilityState = CELL_IMEJP_UTILITY_BEFORE_CONVERT;
    return CELL_OK;
}

s32 cellImeJpBackspace(CellImeJpHandle handle)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    ImeJpSlot* s = &s_slots[handle];
    if (s->inputLen > 0) {
        s->inputLen--;
        s->input[s->inputLen] = 0;
    }
    s->utilityState = s->inputLen ? CELL_IMEJP_UTILITY_BEFORE_CONVERT
                                  : CELL_IMEJP_UTILITY_BEFORE_INPUT;
    return CELL_OK;
}

s32 cellImeJpConfirm(CellImeJpHandle handle)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    ImeJpSlot* s = &s_slots[handle];
    s->confirmedLen = s->inputLen;
    if (s->confirmedLen > CELL_IMEJP_MAX_RESULT_LENGTH - 1)
        s->confirmedLen = CELL_IMEJP_MAX_RESULT_LENGTH - 1;
    memcpy(s->confirmed, s->input, s->confirmedLen * sizeof(u16));
    s->inputLen = 0;
    memset(s->input, 0, sizeof(s->input));
    s->utilityState = CELL_IMEJP_UTILITY_BEFORE_INPUT;
    return CELL_OK;
}

s32 cellImeJpGetInputString(CellImeJpHandle handle, u16* str, u32* len)
{
    if (handle >= IMEJP_MAX_HANDLES || !s_slots[handle].in_use)
        return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;
    if (!str || !len) return (s32)CELL_IMEJP_ERROR_INVALID_ARGUMENT;

    ImeJpSlot* s = &s_slots[handle];
    u32 copyLen = s->inputLen;
    u32 cap = vm_read32(GUEST_EA(len));
    if (copyLen > cap) copyLen = cap;

    /* UTF-16 in guest memory is big-endian, so it goes across a unit at a
     * time rather than as a block copy. */
    for (u32 i = 0; i < copyLen; i++)
        vm_write16(GUEST_EA(str) + i * 2, s->input[i]);
    vm_write32(GUEST_EA(len), copyLen);
    return CELL_OK;
}

s32 cellImeJpGetConvertedString(CellImeJpHandle handle, u16* str, u32* len)
{
    /* Passthrough: converted string = raw input (no kanji conversion) */
    return cellImeJpGetInputString(handle, str, len);
}

s32 cellImeJpGetStatus(CellImeJpHandle handle, s16* status)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!status) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    vm_write16((u32)(uintptr_t)status, (u16)slot->utilityState);
    return CELL_OK;
}

s32 cellImeJpEnterChar(CellImeJpHandle handle, u16 ch, s16* outputStatus)
{
    if (!outputStatus) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->inputLen >= CELL_IMEJP_UTILITY_STRING_MAXLENGTH - 1)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    rc = cellImeJpAddChar(handle, ch);
    if (rc != CELL_OK) return CELL_IMEJP_UTILITY_ERROR_ERR;
    vm_write16((u32)(uintptr_t)outputStatus,
               CELL_IMEJP_UTILITY_RET_CONFIRMED);
    return CELL_OK;
}

s32 cellImeJpSetKanaInputMode(CellImeJpHandle handle, s16 mode)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (mode != CELL_IMEJP_UTILITY_ROMAN_INPUT &&
        mode != CELL_IMEJP_UTILITY_KANA_INPUT)
        return CELL_IMEJP_UTILITY_ERROR_PARAM;
    if (slot->utilityState != CELL_IMEJP_UTILITY_BEFORE_INPUT)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    slot->kanaInputMode = mode;
    return CELL_OK;
}

s32 cellImeJpAllowExtensionCharacters(CellImeJpHandle handle, s16 extensions)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (((u16)extensions & ~CELL_IMEJP_UTILITY_EXTENSION_MASK) != 0)
        return CELL_IMEJP_UTILITY_ERROR_PARAM;
    if (slot->utilityState != CELL_IMEJP_UTILITY_BEFORE_INPUT)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    slot->allowedExtensions = extensions;
    return CELL_OK;
}

s32 cellImeJpSetInputCharType(CellImeJpHandle handle, s16 type)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (type < CELL_IMEJP_UTILITY_DSPCHAR_HIRA ||
        type > CELL_IMEJP_UTILITY_DSPCHAR_HRAW)
        return CELL_IMEJP_UTILITY_ERROR_PARAM;
    slot->inputCharType = type;
    return CELL_OK;
}

s32 cellImeJpConvertForward(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState == CELL_IMEJP_UTILITY_BEFORE_INPUT || !slot->inputLen)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    slot->utilityState = CELL_IMEJP_UTILITY_CANDIDATES;
    return CELL_OK;
}

s32 cellImeJpConvertBackward(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState == CELL_IMEJP_UTILITY_BEFORE_INPUT || !slot->inputLen)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    /* Only the identity candidate exists without a Japanese dictionary. */
    slot->utilityState = CELL_IMEJP_UTILITY_CANDIDATES;
    return CELL_OK;
}

s32 cellImeJpAllConfirm(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState == CELL_IMEJP_UTILITY_BEFORE_INPUT || !slot->inputLen)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    slot->confirmedLen = slot->inputLen;
    if (slot->confirmedLen > CELL_IMEJP_MAX_RESULT_LENGTH - 1)
        slot->confirmedLen = CELL_IMEJP_MAX_RESULT_LENGTH - 1;
    memcpy(slot->confirmed, slot->input, slot->confirmedLen * sizeof(u16));
    slot->inputLen = 0;
    memset(slot->input, 0, sizeof(slot->input));
    slot->utilityState = CELL_IMEJP_UTILITY_BEFORE_INPUT;
    return CELL_OK;
}

s32 cellImeJpCurrentPartConfirm(CellImeJpHandle handle, s16 listItem)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState == CELL_IMEJP_UTILITY_BEFORE_INPUT)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    if (listItem != 0) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    return cellImeJpAllConfirm(handle);
}

s32 cellImeJpConvertCancel(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState <= CELL_IMEJP_UTILITY_BEFORE_CONVERT)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    slot->utilityState = CELL_IMEJP_UTILITY_BEFORE_CONVERT;
    return CELL_OK;
}

s32 cellImeJpExtendConvertArea(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    (void)slot;
    return CELL_IMEJP_UTILITY_ERROR_ERR; /* one identity clause, no dictionary */
}

s32 cellImeJpShortenConvertArea(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    (void)slot;
    return CELL_IMEJP_UTILITY_ERROR_ERR; /* one identity clause, no dictionary */
}

s32 cellImeJpTemporalConfirm(CellImeJpHandle handle, s16 selectIndex)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState != CELL_IMEJP_UTILITY_CANDIDATES)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    if (selectIndex != 0) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    slot->utilityState = CELL_IMEJP_UTILITY_BEFORE_CONVERT;
    return CELL_OK;
}

s32 cellImeJpPostConvert(CellImeJpHandle handle, s16 postType)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    (void)slot; (void)postType;
    return CELL_IMEJP_UTILITY_ERROR_ERR; /* kana/half-width conversion unavailable */
}

s32 cellImeJpMoveFocusClause(CellImeJpHandle handle, s16 moveType)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    (void)slot; (void)moveType;
    return CELL_IMEJP_UTILITY_ERROR_ERR; /* clause boundaries require a dictionary */
}

s32 cellImeJpGetFocusTop(CellImeJpHandle handle, s16* focusTop)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!focusTop) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    vm_write16((u32)(uintptr_t)focusTop, 0);
    return CELL_OK;
}

s32 cellImeJpGetFocusLength(CellImeJpHandle handle, s16* focusLength)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!focusLength) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    u32 input_len = slot->inputLen;
    if (input_len >= CELL_IMEJP_UTILITY_STRING_MAXLENGTH)
        input_len = CELL_IMEJP_UTILITY_STRING_MAXLENGTH - 1;
    u32 length = input_len * 2u;
    if (length > 0x7FFFu) length = 0x7FFFu;
    vm_write16((u32)(uintptr_t)focusLength, (u16)length);
    return CELL_OK;
}

s32 cellImeJpGetConfirmString(CellImeJpHandle handle, u16* string)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!string) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    imejp_write_fixed_string((u32)(uintptr_t)string, slot->confirmed,
                             slot->confirmedLen,
                             CELL_IMEJP_UTILITY_STRING_MAXLENGTH);
    return CELL_OK;
}

s32 cellImeJpGetConvertString(CellImeJpHandle handle, u16* string)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!string) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    imejp_write_fixed_string((u32)(uintptr_t)string, slot->input,
                             slot->inputLen,
                             CELL_IMEJP_UTILITY_STRING_MAXLENGTH);
    return CELL_OK;
}

s32 cellImeJpGetCandidateListSize(CellImeJpHandle handle, s16* listSize)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!listSize) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    if (slot->utilityState != CELL_IMEJP_UTILITY_CANDIDATES)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    u32 length = slot->inputLen;
    if (length >= CELL_IMEJP_UTILITY_STRING_MAXLENGTH)
        length = CELL_IMEJP_UTILITY_STRING_MAXLENGTH - 1;
    const u32 size_bytes = (length * 2u + 3u) * 2u;
    vm_write16((u32)(uintptr_t)listSize, (u16)size_bytes);
    return CELL_OK;
}

s32 cellImeJpGetCandidateList(CellImeJpHandle handle, s16* listNum,
                              u16* candidateString)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!listNum || !candidateString) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    if (slot->utilityState != CELL_IMEJP_UTILITY_CANDIDATES)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    const u32 ea = (u32)(uintptr_t)candidateString;
    u32 length = slot->inputLen;
    if (length >= CELL_IMEJP_UTILITY_STRING_MAXLENGTH)
        length = CELL_IMEJP_UTILITY_STRING_MAXLENGTH - 1;
    u32 pos = 0;
    for (u32 i = 0; i < length; ++i)
        vm_write16(ea + pos++ * 2u, slot->input[i]);
    vm_write16(ea + pos++ * 2u, 0);
    for (u32 i = 0; i < length; ++i)
        vm_write16(ea + pos++ * 2u, slot->input[i]);
    vm_write16(ea + pos++ * 2u, 0);
    vm_write16(ea + pos * 2u, 0); /* identity candidate replacement offset */
    vm_write16((u32)(uintptr_t)listNum, 1);
    return CELL_OK;
}

s32 cellImeJpGetCandidateSelect(CellImeJpHandle handle, s16* index)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (!index) return CELL_IMEJP_UTILITY_ERROR_PARAM;
    if (slot->utilityState != CELL_IMEJP_UTILITY_CANDIDATES)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    vm_write16((u32)(uintptr_t)index, 0);
    return CELL_OK;
}

s32 cellImeJpBackspaceWord(CellImeJpHandle handle)
{
    ImeJpSlot* slot;
    s32 rc = imejp_utility_slot(handle, &slot);
    if (rc != CELL_OK) return rc;
    if (slot->utilityState != CELL_IMEJP_UTILITY_BEFORE_CONVERT || !slot->inputLen)
        return CELL_IMEJP_UTILITY_ERROR_ERR;
    /* Without lexical segmentation the whole unspaced run is one word. */
    u32 new_len = slot->inputLen;
    while (new_len && (slot->input[new_len - 1] == ' ' ||
                       slot->input[new_len - 1] == '\t' ||
                       slot->input[new_len - 1] == '\n'))
        --new_len;
    while (new_len && slot->input[new_len - 1] != ' ' &&
           slot->input[new_len - 1] != '\t' && slot->input[new_len - 1] != '\n')
        --new_len;
    while (new_len && (slot->input[new_len - 1] == ' ' ||
                       slot->input[new_len - 1] == '\t' ||
                       slot->input[new_len - 1] == '\n'))
        --new_len;
    memset(slot->input + new_len, 0,
           (slot->inputLen - new_len) * sizeof(slot->input[0]));
    slot->inputLen = new_len;
    slot->utilityState = new_len ? CELL_IMEJP_UTILITY_BEFORE_CONVERT
                                 : CELL_IMEJP_UTILITY_BEFORE_INPUT;
    return CELL_OK;
}
