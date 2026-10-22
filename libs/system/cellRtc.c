/*
 * ps3recomp - cellRtc HLE implementation
 *
 * Real-time clock using host system time.
 * PS3 epoch: Jan 1, 0001 00:00:00.000000 UTC
 * Tick unit: microseconds
 */

#include "cellRtc.h"
#include "../../runtime/ppu/ppu_memory.h"   /* vm_base, vm_read/write (guest mem) */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>   /* abs */
#include <ctype.h>
#include <stdatomic.h>
#include <time.h>

#ifdef _WIN32
#  include <windows.h>
#  include <sys/timeb.h>
#else
#  include <sys/time.h>
#endif

/* Seconds between PS3 epoch (0001-01-01) and Unix epoch (1970-01-01) */
#define PS3_UNIX_EPOCH_DIFF  62135596800ULL

/* Microseconds per second */
#define USEC_PER_SEC  1000000ULL
#define USEC_PER_DAY  (86400ULL * USEC_PER_SEC)
#define RTC_UNIX_EPOCH_TICK (PS3_UNIX_EPOCH_DIFF * USEC_PER_SEC)
#define RTC_FILETIME_EPOCH_TICK 50491123200000000ULL
#define RTC_SYSTEM_TIME_MIN 63082281600000000ULL
#define RTC_SYSTEM_TIME_MAX 66238041599999999ULL

/* cellRtcSetCurrentTick and cellRtcSetConf affect this emulated process only;
 * never alter the host machine clock or timezone. */
static _Atomic s64 s_rtc_wall_offset_usec = 0;
static _Atomic s32 s_rtc_timezone_minutes = 0;
static _Atomic s32 s_rtc_summertime_minutes = 0;
static _Atomic int s_rtc_timezone_override = 0;
static _Atomic u64 s_rtc_secure_origin_usec = 0;

/* ---------------------------------------------------------------------------
 * Internal helpers
 * -----------------------------------------------------------------------*/

/* Get current Unix time in microseconds */
static u64 get_raw_host_time_usec(void)
{
#ifdef _WIN32
    struct _timeb tb;
    _ftime_s(&tb);
    return (u64)tb.time * USEC_PER_SEC + (u64)tb.millitm * 1000ULL;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (u64)tv.tv_sec * USEC_PER_SEC + (u64)tv.tv_usec;
#endif
}

static u64 get_host_time_usec(void)
{
    const u64 host = get_raw_host_time_usec();
    {
        const s64 adjusted = (s64)host + atomic_load(&s_rtc_wall_offset_usec);
        return adjusted < 0 ? 0 : (u64)adjusted;
    }
}

static u64 get_host_monotonic_usec(void)
{
#ifdef _WIN32
    LARGE_INTEGER now, freq;
    if (!QueryPerformanceCounter(&now) || !QueryPerformanceFrequency(&freq) ||
        freq.QuadPart <= 0)
        return 0;
    return (u64)(now.QuadPart / freq.QuadPart) * USEC_PER_SEC +
        (u64)(now.QuadPart % freq.QuadPart) * USEC_PER_SEC /
            (u64)freq.QuadPart;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (u64)ts.tv_sec * USEC_PER_SEC + (u64)ts.tv_nsec / 1000ULL;
#endif
}

static s32 rtc_get_timezone_minutes(void)
{
    if (atomic_load(&s_rtc_timezone_override))
        return atomic_load(&s_rtc_timezone_minutes) +
               atomic_load(&s_rtc_summertime_minutes);
#ifdef _WIN32
    TIME_ZONE_INFORMATION tz;
    DWORD state = GetTimeZoneInformation(&tz);
    if (state == TIME_ZONE_ID_INVALID) return 0;
    if (state == TIME_ZONE_ID_DAYLIGHT)
        return -tz.Bias - tz.DaylightBias;
    if (state == TIME_ZONE_ID_STANDARD)
        return -tz.Bias - tz.StandardBias;
    return -tz.Bias;
#else
    {
        const time_t now = time(NULL);
        struct tm local_tm, utc_tm;
        if (now == (time_t)-1 || !localtime_r(&now, &local_tm) ||
            !gmtime_r(&now, &utc_tm))
            return 0;
        local_tm.tm_isdst = -1;
        utc_tm.tm_isdst = -1;
        {
            const time_t local_as_epoch = mktime(&local_tm);
            const time_t utc_as_local = mktime(&utc_tm);
            if (local_as_epoch == (time_t)-1 || utc_as_local == (time_t)-1)
                return 0;
            return (s32)((local_as_epoch - utc_as_local) / 60);
        }
    }
#endif
}

static int is_leap_year(int y)
{
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

static int rtc_is_leap_year_u64(u64 year)
{
    return (year % 4ULL == 0 && year % 100ULL != 0) || year % 400ULL == 0;
}

static const int s_days_in_month[12] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

static u64 rtc_days_before_year(u32 year)
{
    const u64 y = year ? (u64)year - 1ULL : 0ULL;
    return y * 365ULL + y / 4ULL - y / 100ULL + y / 400ULL;
}

/* Gregorian conversion follows the public CellRtc year-1 epoch. Keeping this
 * independent from host time_t also supports dates before 1970 on Windows. */
static u64 rtc_datetime_to_tick(const CellRtcDateTime* dt)
{
    u32 month = dt->month;
    if (month < 1) month = 1;
    if (month > 12) month = 12;
    u64 days = rtc_days_before_year(dt->year);
    for (u32 m = 1; m < month; m++) {
        days += (m == 2 && is_leap_year(dt->year))
            ? 29ULL : (u64)s_days_in_month[m - 1];
    }
    days += (u64)dt->day - 1ULL;
    return days * USEC_PER_DAY +
        (u64)dt->hour * 3600ULL * USEC_PER_SEC +
        (u64)dt->minute * 60ULL * USEC_PER_SEC +
        (u64)dt->second * USEC_PER_SEC + dt->microsecond;
}

static CellRtcDateTime rtc_tick_to_datetime(u64 tick)
{
    CellRtcDateTime dt;
    u64 days = tick / USEC_PER_DAY;
    const u64 days_per_400_years = 146097ULL;
    u64 year = (days / days_per_400_years) * 400ULL + 1ULL;
    days %= days_per_400_years;
    while (days >= (u64)(rtc_is_leap_year_u64(year) ? 366 : 365)) {
        days -= (u64)(rtc_is_leap_year_u64(year) ? 366 : 365);
        year++;
    }
    dt.year = (u16)year;
    dt.month = 1;
    while (dt.month < 12) {
        const u32 mdays = (dt.month == 2 && rtc_is_leap_year_u64(year))
            ? 29u : (u32)s_days_in_month[dt.month - 1];
        if (days < mdays) break;
        days -= mdays;
        dt.month++;
    }
    dt.day = (u16)(days + 1ULL);
    tick %= USEC_PER_DAY;
    dt.hour = (u16)(tick / (3600ULL * USEC_PER_SEC));
    tick %= 3600ULL * USEC_PER_SEC;
    dt.minute = (u16)(tick / (60ULL * USEC_PER_SEC));
    tick %= 60ULL * USEC_PER_SEC;
    dt.second = (u16)(tick / USEC_PER_SEC);
    dt.microsecond = (u32)(tick % USEC_PER_SEC);
    return dt;
}

/* Day of week names for RFC formatting */
static const char* s_dow_names[] = {
    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"
};
static const char* s_month_names[] = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
};

/* ---------------------------------------------------------------------------
 * Guest/host dual-mode accessors.
 *
 * Public entry points receive GUEST addresses (the generic HLE adapter passes
 * PPC registers raw), but internal helpers (TickAdd*) call the same functions
 * with HOST stack locals. Values in guest memory are BIG-ENDIAN, so a plain
 * pointer translation is not enough — each field store must byte-swap.
 * Discriminate by value: guest EAs are < 4GB, host pointers are not.
 * -----------------------------------------------------------------------*/
static int rtc_is_guest(const void* p)
{
    uintptr_t v = (uintptr_t)p;
    return v && v < 0x100000000ull;
}
static char* rtc_str(char* p)
{
    return rtc_is_guest(p) ? (char*)(vm_base + (uint32_t)(uintptr_t)p) : p;
}
static u64 rtc_tick_read(const CellRtcTick* p)
{
    return rtc_is_guest(p) ? vm_read64((uint32_t)(uintptr_t)p) : p->tick;
}
static void rtc_tick_write(CellRtcTick* p, u64 t)
{
    if (rtc_is_guest(p)) vm_write64((uint32_t)(uintptr_t)p, t);
    else                 p->tick = t;
}
static void rtc_dt_write(CellRtcDateTime* p, const CellRtcDateTime* v)
{
    if (rtc_is_guest(p)) {
        uint32_t ea = (uint32_t)(uintptr_t)p;
        vm_write16(ea + 0,  v->year);
        vm_write16(ea + 2,  v->month);
        vm_write16(ea + 4,  v->day);
        vm_write16(ea + 6,  v->hour);
        vm_write16(ea + 8,  v->minute);
        vm_write16(ea + 10, v->second);
        vm_write32(ea + 12, v->microsecond);
    } else {
        *p = *v;
    }
}
static void rtc_dt_read(const CellRtcDateTime* p, CellRtcDateTime* v)
{
    if (rtc_is_guest(p)) {
        uint32_t ea = (uint32_t)(uintptr_t)p;
        v->year        = vm_read16(ea + 0);
        v->month       = vm_read16(ea + 2);
        v->day         = vm_read16(ea + 4);
        v->hour        = vm_read16(ea + 6);
        v->minute      = vm_read16(ea + 8);
        v->second      = vm_read16(ea + 10);
        v->microsecond = vm_read32(ea + 12);
    } else {
        *v = *p;
    }
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 cellRtcGetCurrentTick(CellRtcTick* pTick)
{
    if (!pTick)
        return CELL_RTC_ERROR_INVALID_POINTER;

    const u64 unix_usec = get_host_time_usec();
    rtc_tick_write(pTick, unix_usec + RTC_UNIX_EPOCH_TICK);

    return CELL_OK;
}

/* cellRtcGetCurrentClock(CellRtcDateTime* pClock, s32 iTimeZone) — UTC shifted
 * by iTimeZone minutes. LBP's boot path imports this (was missing entirely). */
s32 cellRtcGetCurrentClock(CellRtcDateTime* pClock, s32 iTimeZone)
{
    if (!pClock)
        return CELL_RTC_ERROR_INVALID_POINTER;

    const s64 zone_usec = (s64)iTimeZone * 60LL * (s64)USEC_PER_SEC;
    const s64 tick = (s64)(get_host_time_usec() + RTC_UNIX_EPOCH_TICK) + zone_usec;
    const CellRtcDateTime dt = rtc_tick_to_datetime(tick < 0 ? 0 : (u64)tick);
    rtc_dt_write(pClock, &dt);
    return CELL_OK;
}

s32 cellRtcGetCurrentClockLocalTime(CellRtcDateTime* pClock)
{
    if (!pClock)
        return CELL_RTC_ERROR_INVALID_POINTER;

    const s64 zone_usec = (s64)rtc_get_timezone_minutes() *
                          60LL * (s64)USEC_PER_SEC;
    const s64 tick = (s64)(get_host_time_usec() + RTC_UNIX_EPOCH_TICK) + zone_usec;
    const CellRtcDateTime dt = rtc_tick_to_datetime(tick < 0 ? 0 : (u64)tick);
    rtc_dt_write(pClock, &dt);
    return CELL_OK;
}

s32 cellRtcGetCurrentClockUtc(CellRtcDateTime* pClock)
{
    if (!pClock)
        return CELL_RTC_ERROR_INVALID_POINTER;

    const CellRtcDateTime dt_u = rtc_tick_to_datetime(
        get_host_time_usec() + RTC_UNIX_EPOCH_TICK);
    rtc_dt_write(pClock, &dt_u);
    return CELL_OK;
}

s32 cellRtcGetTime_t(const CellRtcDateTime* pDateTime, s64* pTime)
{
    CellRtcDateTime dt;
    CellRtcTick tick;
    if (!pDateTime || !pTime) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_dt_read(pDateTime, &dt);
    if (dt.year == 0 || dt.year >= 10000) {
        if (rtc_is_guest(pTime)) vm_write64((u32)(uintptr_t)pTime, 0);
        else *pTime = 0;
        return CELL_RTC_ERROR_INVALID_VALUE;
    }
    tick.tick = rtc_datetime_to_tick(&dt);
    if (tick.tick < RTC_UNIX_EPOCH_TICK) {
        if (rtc_is_guest(pTime)) vm_write64((u32)(uintptr_t)pTime, 0);
        else *pTime = 0;
        return CELL_RTC_ERROR_INVALID_VALUE;
    }
    {
        const s64 seconds = (s64)((tick.tick - RTC_UNIX_EPOCH_TICK) / USEC_PER_SEC);
        if (rtc_is_guest(pTime)) vm_write64((u32)(uintptr_t)pTime, (u64)seconds);
        else *pTime = seconds;
    }
    return CELL_OK;
}

s32 cellRtcSetTime_t(CellRtcDateTime* pDateTime, u64 iTime)
{
    CellRtcTick tick = {iTime * USEC_PER_SEC + RTC_UNIX_EPOCH_TICK};
    if (!pDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    return cellRtcSetTick(pDateTime, &tick);
}

s32 cellRtcTickAddYears(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    /* Convert tick to datetime, add years, convert back */
    CellRtcDateTime dt;
    cellRtcSetTick(&dt, pTick1);
    {
        const s32 new_year = (s32)dt.year + iAdd;
        if (new_year < 1 || new_year > 9999)
            return CELL_RTC_ERROR_INVALID_VALUE;
        dt.year = (u16)new_year;
    }

    /* Clamp Feb 29 -> Feb 28 if not leap year */
    if (dt.month == 2 && dt.day == 29 && !is_leap_year(dt.year))
        dt.day = 28;

    return cellRtcGetTick(&dt, pTick0);
}

s32 cellRtcTickAddMonths(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    CellRtcDateTime dt;
    cellRtcSetTick(&dt, pTick1);

    s64 total_months = ((s64)dt.year - 1) * 12 + ((s64)dt.month - 1) + iAdd;
    if (total_months < 0 || total_months >= 9999LL * 12LL)
        return CELL_RTC_ERROR_INVALID_VALUE;

    dt.year  = (u16)(total_months / 12 + 1);
    dt.month = (u16)(total_months % 12 + 1);

    /* Clamp day */
    s32 max_day = cellRtcGetDaysInMonth((s32)dt.year, (s32)dt.month);
    if (dt.day > (u16)max_day)
        dt.day = (u16)max_day;

    return cellRtcGetTick(&dt, pTick0);
}

s32 cellRtcTickAddDays(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    rtc_tick_write(pTick0, rtc_tick_read(pTick1) + (u64)((s64)iAdd * 24LL * 3600LL * USEC_PER_SEC));
    return CELL_OK;
}

s32 cellRtcTickAddHours(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    rtc_tick_write(pTick0, rtc_tick_read(pTick1) + (u64)((s64)iAdd * 3600LL * USEC_PER_SEC));
    return CELL_OK;
}

s32 cellRtcTickAddMinutes(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    rtc_tick_write(pTick0, rtc_tick_read(pTick1) + (u64)((s64)iAdd * 60LL * USEC_PER_SEC));
    return CELL_OK;
}

s32 cellRtcTickAddSeconds(CellRtcTick* pTick0, const CellRtcTick* pTick1, s64 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    rtc_tick_write(pTick0, rtc_tick_read(pTick1) + (u64)(iAdd * (s64)USEC_PER_SEC));
    return CELL_OK;
}

s32 cellRtcTickAddMicroseconds(CellRtcTick* pTick0, const CellRtcTick* pTick1, s64 iAdd)
{
    if (!pTick0 || !pTick1)
        return CELL_RTC_ERROR_INVALID_POINTER;

    rtc_tick_write(pTick0, rtc_tick_read(pTick1) + (u64)iAdd);
    return CELL_OK;
}

s32 cellRtcConvertLocalTimeToUtc(const CellRtcTick* pLocalTime, CellRtcTick* pUtc)
{
    if (!pLocalTime || !pUtc)
        return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_tick_write(pUtc, rtc_tick_read(pLocalTime) -
        (s64)rtc_get_timezone_minutes() * 60LL * (s64)USEC_PER_SEC);
    return CELL_OK;
}

s32 cellRtcConvertUtcToLocalTime(const CellRtcTick* pUtc, CellRtcTick* pLocalTime)
{
    if (!pUtc || !pLocalTime)
        return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_tick_write(pLocalTime, rtc_tick_read(pUtc) +
        (s64)rtc_get_timezone_minutes() * 60LL * (s64)USEC_PER_SEC);
    return CELL_OK;
}

s32 cellRtcFormatRfc2822(char* pszDateTime, const CellRtcTick* pTick, s32 iTimeZoneMinutes)
{
    if (!pszDateTime || !pTick)
        return CELL_RTC_ERROR_INVALID_POINTER;
    {
        const s64 adjusted = (s64)rtc_tick_read(pTick) +
            (s64)iTimeZoneMinutes * 60LL * (s64)USEC_PER_SEC;
        const CellRtcDateTime dt = rtc_tick_to_datetime(adjusted < 0 ? 0 : (u64)adjusted);
        const s32 valid = cellRtcCheckValid(&dt);
        if (valid != CELL_OK) return valid;
        const s32 weekday = cellRtcGetDayOfWeek(dt.year, dt.month, dt.day);
        const s64 tz_abs = iTimeZoneMinutes < 0 ? -(s64)iTimeZoneMinutes : iTimeZoneMinutes;
        const s32 tz_h = (s32)((tz_abs / 60) % 100);
        const s32 tz_m = (s32)(tz_abs % 60);
        snprintf(rtc_str(pszDateTime), 32, "%s, %02u %s %04u %02u:%02u:%02u %c%02d%02d",
            s_dow_names[weekday], (unsigned)dt.day, s_month_names[dt.month - 1],
            (unsigned)dt.year, (unsigned)dt.hour, (unsigned)dt.minute,
            (unsigned)dt.second, iTimeZoneMinutes >= 0 ? '+' : '-', tz_h, tz_m);
    }

    return CELL_OK;
}

s32 cellRtcFormatRfc3339(char* pszDateTime, const CellRtcTick* pTick, s32 iTimeZoneMinutes)
{
    if (!pszDateTime || !pTick)
        return CELL_RTC_ERROR_INVALID_POINTER;
    {
        const s64 adjusted = (s64)rtc_tick_read(pTick) +
            (s64)iTimeZoneMinutes * 60LL * (s64)USEC_PER_SEC;
        const CellRtcDateTime dt = rtc_tick_to_datetime(adjusted < 0 ? 0 : (u64)adjusted);
        const s32 valid = cellRtcCheckValid(&dt);
        if (valid != CELL_OK) return valid;
        const s64 tz_abs = iTimeZoneMinutes < 0 ? -(s64)iTimeZoneMinutes : iTimeZoneMinutes;
        const s32 tz_h = (s32)((tz_abs / 60) % 100);
        const s32 tz_m = (s32)(tz_abs % 60);
        if (iTimeZoneMinutes == 0) {
            snprintf(rtc_str(pszDateTime), 30, "%04u-%02u-%02uT%02u:%02u:%02u.%02uZ",
                (unsigned)dt.year, (unsigned)dt.month, (unsigned)dt.day,
                (unsigned)dt.hour, (unsigned)dt.minute, (unsigned)dt.second,
                (unsigned)(dt.microsecond / 10000));
        } else {
            snprintf(rtc_str(pszDateTime), 32, "%04u-%02u-%02uT%02u:%02u:%02u.%02u%c%02d:%02d",
                (unsigned)dt.year, (unsigned)dt.month, (unsigned)dt.day,
                (unsigned)dt.hour, (unsigned)dt.minute, (unsigned)dt.second,
                (unsigned)(dt.microsecond / 10000),
                iTimeZoneMinutes >= 0 ? '+' : '-', tz_h, tz_m);
        }
    }

    return CELL_OK;
}

s32 cellRtcGetDayOfWeek(s32 year, s32 month, s32 day)
{
    if (month == 1 || month == 2) {
        year--;
        month += 12;
    }
    return ((month * 13 + 8) / 5 + year + year / 4 - year / 100 +
            year / 400 + day) % 7;
}

s32 cellRtcGetDaysInMonth(s32 year, s32 month)
{
    if (year <= 0 || month < 1 || month > 12)
        return CELL_RTC_ERROR_INVALID_ARG;

    if (month == 2 && is_leap_year(year))
        return 29;

    return s_days_in_month[month - 1];
}

s32 cellRtcSetTick(CellRtcDateTime* pDateTime, const CellRtcTick* pTick)
{
    if (!pDateTime || !pTick)
        return CELL_RTC_ERROR_INVALID_POINTER;
    {
        CellRtcDateTime dt = rtc_tick_to_datetime(rtc_tick_read(pTick));
        rtc_dt_write(pDateTime, &dt);
    }
    return CELL_OK;
}

s32 cellRtcGetTick(const CellRtcDateTime* pDateTime, CellRtcTick* pTick)
{
    if (!pDateTime || !pTick)
        return CELL_RTC_ERROR_INVALID_POINTER;
    {
        CellRtcDateTime dt;
        rtc_dt_read(pDateTime, &dt);
        if (dt.year == 0 || dt.year >= 10000)
            return CELL_RTC_ERROR_INVALID_VALUE;
        rtc_tick_write(pTick, rtc_datetime_to_tick(&dt));
    }
    return CELL_OK;
}

/* RPCS3_REFERENCE_REVIEWED: these CellRtc export names, return values, date
 * ranges, DOS/FILETIME epochs, and timezone semantics were checked against
 * RPCS3 cellRtc.cpp/cellRtc.h at commit 2ba2f75bc. Implementations here use
 * ps3recomp's guest-memory accessors and a process-local clock; no RPCS3
 * function body is copied. */
s32 cellRtcGetTickResolution(void)
{
    return (s32)USEC_PER_SEC;
}

s32 cellRtcIsLeapYear(s32 year)
{
    if (year < 1) return CELL_RTC_ERROR_INVALID_ARG;
    return is_leap_year(year) ? 1 : 0;
}

s32 cellRtcCheckValid(const CellRtcDateTime* pDateTime)
{
    CellRtcDateTime dt;
    if (!pDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_dt_read(pDateTime, &dt);
    if (dt.year == 0 || dt.year >= 10000) return CELL_RTC_ERROR_INVALID_YEAR;
    if (dt.month < 1 || dt.month > 12) return CELL_RTC_ERROR_INVALID_MONTH;
    if (dt.day == 0 || dt.day > (u16)cellRtcGetDaysInMonth(dt.year, dt.month))
        return CELL_RTC_ERROR_INVALID_DAY;
    if (dt.hour >= 24) return CELL_RTC_ERROR_INVALID_HOUR;
    if (dt.minute >= 60) return CELL_RTC_ERROR_INVALID_MINUTE;
    if (dt.second >= 60) return CELL_RTC_ERROR_INVALID_SECOND;
    if (dt.microsecond >= USEC_PER_SEC)
        return CELL_RTC_ERROR_INVALID_MICROSECOND;
    return CELL_OK;
}

s32 cellRtcCompareTick(const CellRtcTick* pTick0, const CellRtcTick* pTick1)
{
    u64 a, b;
    if (!pTick0 || !pTick1) return CELL_RTC_ERROR_INVALID_POINTER;
    a = rtc_tick_read(pTick0);
    b = rtc_tick_read(pTick1);
    return a < b ? -1 : (a > b ? 1 : 0);
}

s32 cellRtcTickAddTicks(CellRtcTick* pTick0, const CellRtcTick* pTick1, s64 iAdd)
{
    if (!pTick0 || !pTick1) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_tick_write(pTick0, rtc_tick_read(pTick1) + (u64)iAdd);
    return CELL_OK;
}

s32 cellRtcTickAddWeeks(CellRtcTick* pTick0, const CellRtcTick* pTick1, s32 iAdd)
{
    if (!pTick0 || !pTick1) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_tick_write(pTick0, rtc_tick_read(pTick1) +
        (u64)((s64)iAdd * 7LL * 24LL * 3600LL * (s64)USEC_PER_SEC));
    return CELL_OK;
}

s32 cellRtcGetDosTime(const CellRtcDateTime* pDateTime, u32* pDosTime)
{
    CellRtcDateTime dt;
    u32 value;
    if (!pDateTime || !pDosTime) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_dt_read(pDateTime, &dt);
    if (dt.year < 1980) value = 0;
    else if (dt.year >= 2108) value = 0xFF9FBF7Du;
    else {
        const u32 date = (((u32)(dt.year - 1980) & 0x7Fu) << 9) |
                         (((u32)dt.month & 0xFu) << 5) |
                         ((u32)dt.day & 0x1Fu);
        const u32 time = (((u32)dt.hour & 0x1Fu) << 11) |
                         (((u32)dt.minute & 0x3Fu) << 5) |
                         (((u32)dt.second >> 1) & 0x1Fu);
        value = (date << 16) | time;
    }
    if (rtc_is_guest(pDosTime)) vm_write32((u32)(uintptr_t)pDosTime, value);
    else *pDosTime = value;
    if (dt.year < 1980 || dt.year >= 2108) return -1;
    return CELL_OK;
}

s32 cellRtcSetDosTime(CellRtcDateTime* pDateTime, u32 dosTime)
{
    CellRtcDateTime dt;
    const u32 hms = dosTime & 0xFFFFu;
    const u32 ymd = dosTime >> 16;
    if (!pDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    dt.year = (u16)((ymd >> 9) + 1980u);
    dt.month = (u16)((ymd >> 5) & 0xFu);
    dt.day = (u16)(ymd & 0x1Fu);
    dt.hour = (u16)(hms >> 11);
    dt.minute = (u16)((hms >> 5) & 0x3Fu);
    dt.second = (u16)((hms << 1) & 0x3Eu);
    dt.microsecond = 0;
    rtc_dt_write(pDateTime, &dt);
    return CELL_OK;
}

s32 cellRtcGetSystemTime(const CellRtcDateTime* pDateTime, s64* pTimeStamp)
{
    CellRtcDateTime dt;
    u64 tick;
    s64 value;
    if (!pDateTime || !pTimeStamp) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_dt_read(pDateTime, &dt);
    if (dt.year == 0 || dt.year >= 10000) {
        value = 0;
        goto invalid;
    }
    tick = rtc_datetime_to_tick(&dt);
    if (tick < RTC_SYSTEM_TIME_MIN) {
        value = 0;
        goto invalid;
    }
    if (tick > RTC_SYSTEM_TIME_MAX) {
        value = (s64)((RTC_SYSTEM_TIME_MAX - RTC_SYSTEM_TIME_MIN) / USEC_PER_SEC);
        goto invalid;
    }
    value = (s64)((tick - RTC_SYSTEM_TIME_MIN) / USEC_PER_SEC);
    if (rtc_is_guest(pTimeStamp)) vm_write64((u32)(uintptr_t)pTimeStamp, (u64)value);
    else *pTimeStamp = value;
    return CELL_OK;
invalid:
    if (rtc_is_guest(pTimeStamp)) vm_write64((u32)(uintptr_t)pTimeStamp, (u64)value);
    else *pTimeStamp = value;
    return CELL_RTC_ERROR_INVALID_VALUE;
}

s32 cellRtcSetSystemTime(CellRtcDateTime* pDateTime, u64 iTime)
{
    CellRtcTick tick;
    if (!pDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    tick.tick = iTime * USEC_PER_SEC + RTC_SYSTEM_TIME_MIN;
    return cellRtcSetTick(pDateTime, &tick);
}

s32 cellRtcGetWin32FileTime(const CellRtcDateTime* pDateTime, u64* pFileTime)
{
    CellRtcDateTime dt;
    u64 tick, value;
    if (!pDateTime || !pFileTime) return CELL_RTC_ERROR_INVALID_POINTER;
    rtc_dt_read(pDateTime, &dt);
    if (dt.year == 0 || dt.year >= 10000) {
        value = 0;
        goto invalid;
    }
    tick = rtc_datetime_to_tick(&dt);
    if (tick < RTC_FILETIME_EPOCH_TICK) {
        value = 0;
        goto invalid;
    }
    value = (tick - RTC_FILETIME_EPOCH_TICK) * 10ULL;
    if (rtc_is_guest(pFileTime)) vm_write64((u32)(uintptr_t)pFileTime, value);
    else *pFileTime = value;
    return CELL_OK;
invalid:
    if (rtc_is_guest(pFileTime)) vm_write64((u32)(uintptr_t)pFileTime, value);
    else *pFileTime = value;
    return CELL_RTC_ERROR_INVALID_VALUE;
}

s32 cellRtcSetWin32FileTime(CellRtcDateTime* pDateTime, u64 fileTime)
{
    CellRtcTick tick;
    if (!pDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    tick.tick = fileTime / 10ULL + RTC_FILETIME_EPOCH_TICK;
    return cellRtcSetTick(pDateTime, &tick);
}

s32 cellRtcGetCurrentSecureTick(CellRtcTick* pTick)
{
    u64 now, origin;
    if (!pTick) return CELL_RTC_ERROR_INVALID_POINTER;
    now = get_host_monotonic_usec();
    origin = atomic_load(&s_rtc_secure_origin_usec);
    if (!origin) {
        u64 zero = 0;
        atomic_compare_exchange_strong(&s_rtc_secure_origin_usec, &zero, now);
        origin = atomic_load(&s_rtc_secure_origin_usec);
    }
    rtc_tick_write(pTick, RTC_SYSTEM_TIME_MIN + (now - origin));
    return CELL_OK;
}

s32 cellRtcSetCurrentSecureTick(const CellRtcTick* pTick)
{
    if (!pTick) return CELL_RTC_ERROR_INVALID_POINTER;
    if (rtc_tick_read(pTick) > RTC_SYSTEM_TIME_MAX)
        return CELL_RTC_ERROR_INVALID_VALUE;
    /* RPCS3's sys_ss_secure_rtc(0x3003) is a success-only TODO. Keep the same
     * externally visible result instead of inventing a persistent clock write. */
    return CELL_OK;
}

s32 cellRtcSetCurrentTick(const CellRtcTick* pTick)
{
    u64 tick, host;
    if (!pTick) return CELL_RTC_ERROR_INVALID_POINTER;
    tick = rtc_tick_read(pTick);
    if (tick < RTC_UNIX_EPOCH_TICK) return CELL_RTC_ERROR_INVALID_VALUE;
    host = get_raw_host_time_usec();
    atomic_store(&s_rtc_wall_offset_usec,
        (s64)(tick - RTC_UNIX_EPOCH_TICK) - (s64)host);
    return CELL_OK;
}

s32 cellRtcSetConf(s64 unk1, s64 unk2, u32 timezone, u32 summertime)
{
    (void)unk1;
    (void)unk2;
    atomic_store(&s_rtc_timezone_minutes, (s32)timezone);
    atomic_store(&s_rtc_summertime_minutes, (s32)summertime);
    atomic_store(&s_rtc_timezone_override, 1);
    return CELL_OK;
}

s32 cellRtcFormatRfc2822LocalTime(char* pszDateTime, const CellRtcTick* pTick)
{
    if (!pszDateTime || !pTick) return CELL_RTC_ERROR_INVALID_POINTER;
    return cellRtcFormatRfc2822(pszDateTime, pTick, rtc_get_timezone_minutes());
}

s32 cellRtcFormatRfc3339LocalTime(char* pszDateTime, const CellRtcTick* pTick)
{
    if (!pszDateTime || !pTick) return CELL_RTC_ERROR_INVALID_POINTER;
    return cellRtcFormatRfc3339(pszDateTime, pTick, rtc_get_timezone_minutes());
}

static void rtc_skip_space(const char** p)
{
    while (**p == ' ' || **p == '\t') (*p)++;
}

static int rtc_parse_number(const char** p, u32 min_digits, u32 max_digits, u32* value)
{
    u32 n = 0, count = 0;
    while (**p >= '0' && **p <= '9' && count < max_digits) {
        n = n * 10u + (u32)(**p - '0');
        (*p)++;
        count++;
    }
    if (count < min_digits) return 0;
    *value = n;
    return 1;
}

static int rtc_word_equal3(const char* word, const char* name)
{
    return word[0] && word[1] && word[2] &&
        tolower((unsigned char)word[0]) == tolower((unsigned char)name[0]) &&
        tolower((unsigned char)word[1]) == tolower((unsigned char)name[1]) &&
        tolower((unsigned char)word[2]) == tolower((unsigned char)name[2]);
}

static int rtc_ascii_casecmp(const char* a, const char* b)
{
    while (*a && *b) {
        const int ca = tolower((unsigned char)*a++);
        const int cb = tolower((unsigned char)*b++);
        if (ca != cb) return ca - cb;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

static int rtc_parse_name(const char** p, char* word, size_t capacity)
{
    size_t n = 0;
    while (((*p)[n] >= 'A' && (*p)[n] <= 'Z') ||
           ((*p)[n] >= 'a' && (*p)[n] <= 'z')) {
        if (n + 1 >= capacity) return 0;
        word[n] = (*p)[n];
        n++;
    }
    if (!n) return 0;
    word[n] = '\0';
    *p += n;
    return 1;
}

static int rtc_month_from_name(const char* word)
{
    static const char* const names[12] = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    };
    for (int i = 0; i < 12; i++)
        if (rtc_word_equal3(word, names[i])) return i + 1;
    return 0;
}

static int rtc_skip_weekday(const char** p)
{
    static const char* const names[7] = {
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
    };
    char word[16];
    if (!rtc_parse_name(p, word, sizeof(word))) return 0;
    for (int i = 0; i < 7; i++)
        if (rtc_word_equal3(word, names[i])) return 1;
    return 0;
}

static int rtc_parse_time(const char** p, CellRtcDateTime* dt, int seconds_optional)
{
    u32 hour, minute, second = 0;
    if (!rtc_parse_number(p, 1, 2, &hour) || **p != ':') return 0;
    (*p)++;
    if (!rtc_parse_number(p, 1, 2, &minute)) return 0;
    if (**p == ':') {
        (*p)++;
        if (!rtc_parse_number(p, 1, 2, &second)) return 0;
    } else if (!seconds_optional) {
        return 0;
    }
    dt->hour = (u16)hour;
    dt->minute = (u16)minute;
    dt->second = (u16)second;
    dt->microsecond = 0;
    return 1;
}

static int rtc_parse_rfc2822_zone(const char** p, s32* offset_minutes)
{
    static const struct { const char* name; s32 offset; } zones[] = {
        {"UT", 0}, {"UTC", 0}, {"GMT", 0},
        {"EST", -300}, {"EDT", -240}, {"CST", -360}, {"CDT", -300},
        {"MST", -420}, {"MDT", -360}, {"PST", -480}, {"PDT", -420},
        {"JST", 540}, {"KST", 540}, {"BST", 60}, {"CET", 60},
        {"CEST", 120}, {"EET", 120}, {"NZST", 720}, {"NZDT", 780}
    };
    u32 hour, minute;
    char word[16];
    if (**p == '+' || **p == '-') {
        const int sign = **p == '+' ? 1 : -1;
        (*p)++;
        if (!rtc_parse_number(p, 2, 2, &hour)) return 0;
        if (**p == ':') (*p)++;
        if (!rtc_parse_number(p, 2, 2, &minute) || hour > 23 || minute > 59)
            return 0;
        *offset_minutes = sign * (s32)(hour * 60u + minute);
        return 1;
    }
    if (!rtc_parse_name(p, word, sizeof(word))) return 0;
    for (size_t i = 0; i < sizeof(zones) / sizeof(zones[0]); i++) {
        if (rtc_ascii_casecmp(word, zones[i].name) == 0) {
            *offset_minutes = zones[i].offset;
            return 1;
        }
    }
    if (word[1] == '\0') {
        const char c = (char)toupper((unsigned char)word[0]);
        if (c >= 'A' && c <= 'I') *offset_minutes = (c - 'A' + 1) * 60;
        else if (c >= 'K' && c <= 'M') *offset_minutes = (c - 'A') * 60;
        else if (c >= 'N' && c <= 'Y') *offset_minutes = -((c - 'N' + 1) * 60);
        else if (c == 'Z') *offset_minutes = 0;
        else return 0;
        return 1;
    }
    return 0;
}

static s32 rtc_parse_rfc3339(const char* text, CellRtcDateTime* dt,
                             s32* offset_minutes)
{
    const char* p = text;
    u32 year, month, day, hour, minute, second;
    if (!rtc_parse_number(&p, 4, 4, &year) || *p++ != '-')
        return CELL_RTC_ERROR_INVALID_YEAR;
    if (!rtc_parse_number(&p, 2, 2, &month) || *p++ != '-')
        return CELL_RTC_ERROR_INVALID_MONTH;
    if (!rtc_parse_number(&p, 2, 2, &day) || (*p != 'T' && *p != 't'))
        return CELL_RTC_ERROR_INVALID_DAY;
    p++;
    if (!rtc_parse_number(&p, 2, 2, &hour) || *p++ != ':')
        return CELL_RTC_ERROR_INVALID_HOUR;
    if (!rtc_parse_number(&p, 2, 2, &minute) || *p++ != ':')
        return CELL_RTC_ERROR_INVALID_MINUTE;
    if (!rtc_parse_number(&p, 2, 2, &second))
        return CELL_RTC_ERROR_INVALID_SECOND;
    dt->year = (u16)year;
    dt->month = (u16)month;
    dt->day = (u16)day;
    dt->hour = (u16)hour;
    dt->minute = (u16)minute;
    dt->second = (u16)second;
    dt->microsecond = 0;
    if (*p == '.') {
        u32 digits = 0, place = 100000;
        p++;
        if (*p < '0' || *p > '9') return CELL_RTC_ERROR_BAD_PARSE;
        while (*p >= '0' && *p <= '9') {
            if (digits < 6) {
                dt->microsecond += (u32)(*p - '0') * place;
                place /= 10;
            }
            digits++;
            p++;
        }
    }
    if (*p == 'Z' || *p == 'z') {
        *offset_minutes = 0;
        p++;
    } else if (*p == '+' || *p == '-') {
        const int sign = *p == '+' ? 1 : -1;
        u32 zone_hour, zone_minute;
        p++;
        if (!rtc_parse_number(&p, 2, 2, &zone_hour) || *p++ != ':' ||
            !rtc_parse_number(&p, 2, 2, &zone_minute) ||
            zone_hour > 23 || zone_minute > 59)
            return CELL_RTC_ERROR_BAD_PARSE;
        *offset_minutes = sign * (s32)(zone_hour * 60u + zone_minute);
    } else {
        return CELL_RTC_ERROR_BAD_PARSE;
    }
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p) return CELL_RTC_ERROR_BAD_PARSE;
    return cellRtcCheckValid(dt);
}

s32 cellRtcParseRfc3339(CellRtcTick* pTick, const char* pszDateTime)
{
    CellRtcDateTime dt;
    CellRtcTick tick;
    s32 offset = 0, ret;
    const char* text;
    if (!pTick || !pszDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    text = rtc_str((char*)pszDateTime);
    ret = rtc_parse_rfc3339(text, &dt, &offset);
    if (ret != CELL_OK) return ret;
    tick.tick = rtc_datetime_to_tick(&dt) -
        (s64)offset * 60LL * (s64)USEC_PER_SEC;
    rtc_tick_write(pTick, tick.tick);
    return CELL_OK;
}

s32 cellRtcParseDateTime(CellRtcTick* pTick, const char* pszDateTime)
{
    char name[24];
    const char* p;
    CellRtcDateTime dt;
    CellRtcTick tick;
    s32 offset = 0, ret;
    u32 value, year;
    int month;
    if (!pTick || !pszDateTime) return CELL_RTC_ERROR_INVALID_POINTER;
    p = rtc_str((char*)pszDateTime);
    rtc_skip_space(&p);
    if (strlen(p) >= 4 && p[0] >= '0' && p[0] <= '9' && p[1] >= '0' && p[1] <= '9' &&
        p[2] >= '0' && p[2] <= '9' && p[3] >= '0' && p[3] <= '9')
        return cellRtcParseRfc3339(pTick, p);
    if (!rtc_skip_weekday(&p)) return CELL_RTC_ERROR_BAD_PARSE;
    if (*p == ',') p++;
    rtc_skip_space(&p);

    memset(&dt, 0, sizeof(dt));
    if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) {
        if (!rtc_parse_name(&p, name, sizeof(name))) return CELL_RTC_ERROR_BAD_PARSE;
        month = rtc_month_from_name(name);
    } else {
        month = 0;
    }
    if (month) {
        /* asctime: Wdy Mon dd HH:MM:SS YYYY */
        dt.month = (u16)month;
        rtc_skip_space(&p);
        if (!rtc_parse_number(&p, 1, 2, &value)) return CELL_RTC_ERROR_BAD_PARSE;
        dt.day = (u16)value;
        rtc_skip_space(&p);
        if (!rtc_parse_time(&p, &dt, 0)) return CELL_RTC_ERROR_BAD_PARSE;
        rtc_skip_space(&p);
        if (!rtc_parse_number(&p, 4, 4, &year)) return CELL_RTC_ERROR_BAD_PARSE;
        dt.year = (u16)year;
    } else {
        /* RFC 2822: Wdy, dd Mon yyyy HH:MM[:SS] zone */
        if (!rtc_parse_number(&p, 1, 2, &value)) return CELL_RTC_ERROR_BAD_PARSE;
        dt.day = (u16)value;
        if (*p != ' ' && *p != '-') return CELL_RTC_ERROR_BAD_PARSE;
        p++;
        if (!rtc_parse_name(&p, name, sizeof(name)) ||
            (month = rtc_month_from_name(name)) == 0)
            return CELL_RTC_ERROR_BAD_PARSE;
        dt.month = (u16)month;
        if (*p != ' ' && *p != '-') return CELL_RTC_ERROR_BAD_PARSE;
        p++;
        if (!rtc_parse_number(&p, 2, 4, &year)) return CELL_RTC_ERROR_BAD_PARSE;
        if (year < 100) year += year < 50 ? 2000 : 1900;
        dt.year = (u16)year;
        rtc_skip_space(&p);
        if (!rtc_parse_time(&p, &dt, 1)) return CELL_RTC_ERROR_BAD_PARSE;
        if (dt.hour > 25) return CELL_RTC_ERROR_BAD_PARSE;
        if (dt.hour >= 24) return CELL_RTC_ERROR_INVALID_HOUR;
        if (*p != ' ') return CELL_RTC_ERROR_BAD_PARSE;
        rtc_skip_space(&p);
        if (!rtc_parse_rfc2822_zone(&p, &offset)) return CELL_RTC_ERROR_BAD_PARSE;
    }
    rtc_skip_space(&p);
    if (*p) return CELL_RTC_ERROR_BAD_PARSE;
    ret = cellRtcCheckValid(&dt);
    if (ret != CELL_OK) return ret;
    tick.tick = rtc_datetime_to_tick(&dt) -
        (s64)offset * 60LL * (s64)USEC_PER_SEC;
    rtc_tick_write(pTick, tick.tick);
    return CELL_OK;
}
