/*
 * ps3recomp - cellRtc HLE
 *
 * Real-time clock: ticks, date/time conversion, formatting.
 */

#ifndef PS3RECOMP_CELL_RTC_H
#define PS3RECOMP_CELL_RTC_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Constants
 * -----------------------------------------------------------------------*/

/*
 * PS3 epoch: Jan 1, 0001 00:00:00.000000 UTC
 * Offset from Unix epoch (Jan 1, 1970) = 62135596800 seconds
 * Ticks are in microseconds.
 */
#define CELL_RTC_UNIX_EPOCH_OFFSET  62135596800ULL

/* Return codes from the public cellRtc ABI. */
#define CELL_RTC_ERROR_NOT_INITIALIZED      ((s32)0x80010601u)
#define CELL_RTC_ERROR_INVALID_POINTER      ((s32)0x80010602u)
#define CELL_RTC_ERROR_INVALID_VALUE        ((s32)0x80010603u)
#define CELL_RTC_ERROR_INVALID_ARG          ((s32)0x80010604u)
#define CELL_RTC_ERROR_NOT_SUPPORTED        ((s32)0x80010605u)
#define CELL_RTC_ERROR_NO_CLOCK             ((s32)0x80010606u)
#define CELL_RTC_ERROR_BAD_PARSE            ((s32)0x80010607u)
#define CELL_RTC_ERROR_INVALID_YEAR         ((s32)0x80010621u)
#define CELL_RTC_ERROR_INVALID_MONTH        ((s32)0x80010622u)
#define CELL_RTC_ERROR_INVALID_DAY          ((s32)0x80010623u)
#define CELL_RTC_ERROR_INVALID_HOUR         ((s32)0x80010624u)
#define CELL_RTC_ERROR_INVALID_MINUTE       ((s32)0x80010625u)
#define CELL_RTC_ERROR_INVALID_SECOND       ((s32)0x80010626u)
#define CELL_RTC_ERROR_INVALID_MICROSECOND  ((s32)0x80010627u)

/* ---------------------------------------------------------------------------
 * Structures
 * -----------------------------------------------------------------------*/

typedef struct CellRtcTick {
    u64 tick;  /* microseconds since PS3 epoch (Jan 1, 0001) */
} CellRtcTick;

typedef struct CellRtcDateTime {
    u16 year;
    u16 month;     /* 1-12 */
    u16 day;       /* 1-31 */
    u16 hour;      /* 0-23 */
    u16 minute;    /* 0-59 */
    u16 second;    /* 0-59 */
    u32 microsecond; /* 0-999999 */
} CellRtcDateTime;

/* Alias used by some SDK headers */
typedef CellRtcDateTime CellRtcTime;

/* ---------------------------------------------------------------------------
 * Functions
 * -----------------------------------------------------------------------*/

s32 cellRtcGetCurrentTick(CellRtcTick* pTick);

s32 cellRtcGetCurrentClock(CellRtcDateTime* pClock, s32 iTimeZone);

s32 cellRtcGetCurrentClockLocalTime(CellRtcDateTime* pClock);

s32 cellRtcGetCurrentClockUtc(CellRtcDateTime* pClock);

s32 cellRtcGetTime_t(const CellRtcDateTime* pDateTime, s64* pTime);

s32 cellRtcSetTime_t(CellRtcDateTime* pDateTime, u64 iTime);

s32 cellRtcGetSystemTime(const CellRtcDateTime* pDateTime, s64* pTimeStamp);
s32 cellRtcSetSystemTime(CellRtcDateTime* pDateTime, u64 iTime);
s32 cellRtcGetWin32FileTime(const CellRtcDateTime* pDateTime, u64* pFileTime);
s32 cellRtcSetWin32FileTime(CellRtcDateTime* pDateTime, u64 fileTime);
s32 cellRtcGetDosTime(const CellRtcDateTime* pDateTime, u32* pDosTime);
s32 cellRtcSetDosTime(CellRtcDateTime* pDateTime, u32 dosTime);
s32 cellRtcGetCurrentSecureTick(CellRtcTick* pTick);
s32 cellRtcSetCurrentSecureTick(const CellRtcTick* pTick);
s32 cellRtcSetCurrentTick(const CellRtcTick* pTick);
s32 cellRtcSetConf(s64 unk1, s64 unk2, u32 timezone, u32 summertime);
s32 cellRtcGetTickResolution(void);
s32 cellRtcIsLeapYear(s32 year);
s32 cellRtcCheckValid(const CellRtcDateTime* pDateTime);
s32 cellRtcCompareTick(const CellRtcTick* pTick0, const CellRtcTick* pTick1);
s32 cellRtcParseDateTime(CellRtcTick* pTick, const char* pszDateTime);
s32 cellRtcParseRfc3339(CellRtcTick* pTick, const char* pszDateTime);
s32 cellRtcFormatRfc2822LocalTime(char* pszDateTime, const CellRtcTick* pTick);
s32 cellRtcFormatRfc3339LocalTime(char* pszDateTime, const CellRtcTick* pTick);
s32 cellRtcTickAddTicks(CellRtcTick* pTick0, const CellRtcTick* pTick1, s64 iAdd);
s32 cellRtcTickAddWeeks(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd);

s32 cellRtcTickAddYears(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd);
s32 cellRtcTickAddMonths(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd);
s32 cellRtcTickAddDays(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd);
s32 cellRtcTickAddHours(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd);
s32 cellRtcTickAddMinutes(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd);
s32 cellRtcTickAddSeconds(CellRtcTick* pTick0, const CellRtcTick* pTick1, s64 iAdd);
s32 cellRtcTickAddMicroseconds(CellRtcTick* pTick0, const CellRtcTick* pTick1, s64 iAdd);

s32 cellRtcConvertLocalTimeToUtc(const CellRtcTick* pLocalTime, CellRtcTick* pUtc);
s32 cellRtcConvertUtcToLocalTime(const CellRtcTick* pUtc, CellRtcTick* pLocalTime);

s32 cellRtcFormatRfc2822(char* pszDateTime, const CellRtcTick* pTick, s32 iTimeZoneMinutes);
s32 cellRtcFormatRfc3339(char* pszDateTime, const CellRtcTick* pTick, s32 iTimeZoneMinutes);

s32 cellRtcGetDayOfWeek(s32 year, s32 month, s32 day);
s32 cellRtcGetDaysInMonth(s32 year, s32 month);

s32 cellRtcSetTick(CellRtcDateTime* pDateTime, const CellRtcTick* pTick);
s32 cellRtcGetTick(const CellRtcDateTime* pDateTime, CellRtcTick* pTick);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_RTC_H */
