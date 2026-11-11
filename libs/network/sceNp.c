/*
 * ps3recomp - sceNp HLE implementation
 *
 * Provides fake PSN identity so games can proceed through NP checks.
 * The username defaults to "PS3Player" but is configurable.
 */

#include "sceNp.h"
#include <stdio.h>
#include <string.h>

/* RPCS3 sceNpPlus reports true for this PS3 mode query (the ABI is inferred
 * there to be a one-byte boolean; the PPU return register carries 1). */
s32 sceNpManagerIsSP(void)
{
    return 1;
}
#include <stdint.h>
#include <time.h>
#include "../../runtime/ppu/ppu_memory.h"   /* vm_write*: guest EA -> host, byte-swapped */
#include "np_psnr.h"
#include "../system/cellSysutil.h"
#include "../system/cellRtc.h"
#include "../filesystem/cellFs.h"
#include "../filesystem/edat.h"
#include "../guest_struct.h"   /* GUEST_EA */
#include "../../runtime/memory/vm.h" /* VM_READY, ppu_vm_size */


/* ---------------------------------------------------------------------------
 * Internal state
 * -----------------------------------------------------------------------*/

static int  s_np_initialized = 0;
static int  s_score_initialized = 0;
static char s_fake_username[SCE_NP_ONLINEID_MAX_LENGTH + 1] = "PS3Player";
extern u32 ppu_vm_size;

static int np_guest_string(u32 ea, char* out, size_t cap)
{
    if (!VM_READY || !ea || !out || cap < 2)
        return 0;
    for (size_t i = 0; i < cap; ++i) {
        const u64 addr = (u64)ea + i;
        if (addr > UINT32_MAX || (ppu_vm_size && addr >= ppu_vm_size))
            return 0;
        out[i] = (char)vm_read8((u32)addr);
        if (!out[i])
            return 1;
    }
    out[cap - 1] = '\0';
    return 0;
}

/* RPCS3_REFERENCE_REVIEWED: the guest signature/error contract was checked;
 * the local VFS and EDAT decoder are used directly, with no RPCS3 body copied. */
s32 sceNpDrmIsAvailable2(const u8* klicensee, const char* drmPath)
{
    const u32 path_ea = GUEST_EA(drmPath);
    const u32 key_ea = GUEST_EA(klicensee);
    char guest_path[0x100];
    char host_path[CELL_FS_MAX_FS_PATH_LENGTH];
    char decrypted_path[CELL_FS_MAX_FS_PATH_LENGTH + 8];

    if (!path_ea || !np_guest_string(path_ea, guest_path, sizeof(guest_path)) ||
        (key_ea && (key_ea > UINT32_MAX - 16u ||
                    (ppu_vm_size && (u64)key_ea + 16u > ppu_vm_size))))
        return (s32)SCE_NP_DRM_ERROR_INVALID_PARAM;
    if (cellfs_translate_path(guest_path, host_path, sizeof(host_path)) != 0)
        return (s32)CELL_ENOENT;

    FILE* file = fopen(host_path, "rb");
    if (!file)
        return (s32)CELL_ENOENT;
    u8 magic[4] = {0};
    const size_t read = fread(magic, 1, sizeof(magic), file);
    fclose(file);
    if (read != sizeof(magic) || memcmp(magic, "NPD\0", 4) != 0) {
        /* RPCS3 treats non-NPD files as unencrypted. SELF verification is not
         * modeled here, so reject its distinct SCE header rather than claim
         * that an encrypted executable passed license verification. */
        if (read == sizeof(magic) && memcmp(magic, "SCE\0", 4) == 0)
            return (s32)SCE_NP_DRM_ERROR_NO_ENTITLEMENT;
        return CELL_OK;
    }

    const int n = snprintf(decrypted_path, sizeof(decrypted_path), "%s.dec", host_path);
    if (n < 0 || (size_t)n >= sizeof(decrypted_path))
        return (s32)SCE_NP_DRM_ERROR_INVALID_PARAM;
    const u8* guest_key = key_ea
        ? GUEST_PTR((void*)(uintptr_t)key_ea, const u8*) : NULL;
    const int rc = edat_decrypt_file_with_klicensee(host_path, decrypted_path, guest_key);
    if (rc == 0) {
        if (guest_key) edat_add_klicensee(guest_key);
        return CELL_OK;
    }
    if (rc == -1)
        return (s32)CELL_ENOENT;
    if (rc == -6)
        return (s32)SCE_NP_DRM_ERROR_LICENSE_NOT_FOUND;
    if (rc == -2 || rc == -3 || rc == -4)
        return (s32)SCE_NP_DRM_ERROR_BAD_FORMAT;
    return (s32)SCE_NP_DRM_ERROR_NO_ENTITLEMENT;
}

/* ---------------------------------------------------------------------------
 * Configuration
 * -----------------------------------------------------------------------*/

void sceNpSetFakeUsername(const char* username)
{
    if (username) {
        strncpy(s_fake_username, GUEST_PTR(username, const char*),
            SCE_NP_ONLINEID_MAX_LENGTH);
        s_fake_username[SCE_NP_ONLINEID_MAX_LENGTH] = '\0';
    }
}

const char* np_fake_username(void)
{
    return s_fake_username;
}

/* Build a fake NP ID from the current username (PS3_NP_ONLINE_ID wins, so
 * two instances on one machine can be two players). */
static void np_build_fake_id(SceNpId* npId)
{
    memset(npId, 0, sizeof(SceNpId));
    strncpy(npId->handle.data, np_psnr_online_id(), SCE_NP_ONLINEID_MAX_LENGTH);
    npId->handle.term = '\0';
}

/* ---------------------------------------------------------------------------
 * API implementations
 * -----------------------------------------------------------------------*/

s32 sceNpInit(u32 poolSize, void* poolPtr)
{
    (void)poolSize;
    (void)poolPtr;

    printf("[sceNp] Init(poolSize=%u, username=\"%s\")\n",
           poolSize, np_psnr_online_id());

    if (s_np_initialized)
        return SCE_NP_ERROR_ALREADY_INITIALIZED;

    s_np_initialized = 1;
    return CELL_OK;
}

/* Whether sceNpInit has run; sceNpBasic follows it (see sceNpBasic.c). */
int np_is_initialized(void) { return s_np_initialized; }

s32 sceNpTerm(void)
{
    printf("[sceNp] Term()\n");

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    s_np_initialized = 0;
    return CELL_OK;
}

s32 sceNpGetNpId(s32 userId, SceNpId* npId)
{
    (void)userId;

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!npId)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    /* A guest address, like every sibling getter's: writing through it
     * untranslated hit a host address and crashed Simpsons Arcade the first
     * time it asked for its NP ID online. */
    npId = GUEST_PTR(npId, SceNpId*);
    np_build_fake_id(npId);
    printf("[sceNp] GetNpId(user=%d) -> \"%s\"\n", userId, np_psnr_online_id());
    return CELL_OK;
}

s32 sceNpGetOnlineId(s32 userId, SceNpOnlineId* onlineId)
{
    onlineId = GUEST_PTR(onlineId, SceNpOnlineId*);
    (void)userId;

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!onlineId)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    memset(onlineId, 0, sizeof(SceNpOnlineId));
    strncpy(onlineId->data, np_psnr_online_id(), SCE_NP_ONLINEID_MAX_LENGTH);
    onlineId->term = '\0';

    printf("[sceNp] GetOnlineId(user=%d) -> \"%s\"\n", userId, np_psnr_online_id());
    return CELL_OK;
}

s32 sceNpGetOnlineName(s32 userId, SceNpOnlineName* onlineName)
{
    onlineName = GUEST_PTR(onlineName, SceNpOnlineName*);
    (void)userId;

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!onlineName)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    memset(onlineName, 0, sizeof(SceNpOnlineName));
    strncpy(onlineName->data, np_psnr_online_id(),
            SCE_NP_ONLINENAME_MAX_LENGTH - 1);

    printf("[sceNp] GetOnlineName(user=%d) -> \"%s\"\n",
           userId, np_psnr_online_id());
    return CELL_OK;
}

s32 sceNpGetUserProfile(s32 userId, SceNpUserInfo* userInfo)
{
    userInfo = GUEST_PTR(userInfo, SceNpUserInfo*);
    (void)userId;

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!userInfo)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    memset(userInfo, 0, sizeof(SceNpUserInfo));
    np_build_fake_id(&userInfo->npId);
    strncpy(userInfo->onlineName.data, np_psnr_online_id(),
            SCE_NP_ONLINENAME_MAX_LENGTH - 1);
    /* Leave avatar URL empty */

    printf("[sceNp] GetUserProfile(user=%d) -> \"%s\"\n",
           userId, np_psnr_online_id());
    return CELL_OK;
}

s32 sceNpGetAccountRegion(s32 userId, u32* region)
{
    (void)userId;

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!region)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    /* Region: US (SCEA) = 0x5553 ('US') */
    vm_write32((u32)(uintptr_t)region, (u32)0x5553);

    printf("[sceNp] GetAccountRegion(user=%d) -> US\n", userId);
    return CELL_OK;
}

s32 sceNpGetAccountAge(s32 userId, s32* age)
{
    (void)userId;

    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!age)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    vm_write32((u32)(uintptr_t)age, 25); /* default adult age */
    printf("[sceNp] GetAccountAge(user=%d) -> 25\n", userId);
    return CELL_OK;
}

s32 sceNpGetMyLanguages(SceNpMyLanguages* langs)
{
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;

    if (!langs)
        return SCE_NP_ERROR_INVALID_ARGUMENT;

    u32 langs_ea = (u32)(uintptr_t)langs;
    vm_write32(langs_ea + 0, SCE_NP_LANG_ENGLISH);
    vm_write32(langs_ea + 4, 0);
    vm_write32(langs_ea + 8, 0);

    printf("[sceNp] GetMyLanguages() -> English\n");
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * NP Manager (sign-in state)
 *
 * The toolkit runs with a fake local profile but no live PSN connection. So the
 * manager reports OFFLINE — games gate their online flows on GetStatus and skip
 * them cleanly — while the identity getters still hand back the fake account,
 * matching how NP behaves on a real signed-in-but-disconnected console.
 * Imported by most online-capable titles.
 * -----------------------------------------------------------------------*/

/* No purchased content is provisioned by the local NP implementation.
 * The API returns the number of IDs, so an empty list requires no guest write.
 * In particular, (NULL, 0) is a valid count query. */
static SceNpManagerCallback s_npmgr_cb = NULL;
static void*                s_npmgr_cb_arg = NULL;

s32 sceNpManagerGetStatus(s32* status)
{
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!status)
        return SCE_NP_ERROR_INVALID_ARGUMENT;
    /* `status` is a GUEST address -- the HLE ABI adapter passes pointer
     * parameters straight through as guest values, so dereferencing one
     * writes to whatever host address shares that number. Same trap as
     * cellGcmSys had. Tokyo Jungle calls this during its online init. */
    /* Signed in when there is a psnr server to be signed in to. */
    s32 st = np_psnr_enabled() ? SCE_NP_MANAGER_STATUS_ONLINE : SCE_NP_MANAGER_STATUS_OFFLINE;
    vm_write32((uint32_t)(uintptr_t)status, (uint32_t)st);
    { static s32 last = -2;
      if (st != last) printf("[sceNp] ManagerGetStatus() -> %s\n", st < 0 ? "OFFLINE" : "ONLINE");
      last = st; }
    return CELL_OK;
}

s32 sceNpManagerRegisterCallback(SceNpManagerCallback callback, void* arg)
{
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;
    s_npmgr_cb     = callback;
    s_npmgr_cb_arg = arg;
    printf("[sceNp] ManagerRegisterCallback()\n");
    /* Online: tell the title it is signed in, as the console does once the
     * manager reaches ONLINE. Delivered on its next cellSysutilCheckCallback. */
    if (np_psnr_enabled() && callback) {
        const u64 args[8] = { (u64)(u32)SCE_NP_MANAGER_STATUS_ONLINE, 0,
                              (u64)(u32)(uintptr_t)arg, 0, 0, 0, 0, 0 };
        cellSysutilQueueGuestCallbackArgs((u32)(uintptr_t)callback, args);
    }
    return CELL_OK;
}

s32 sceNpManagerUnregisterCallback(void)
{
    s_npmgr_cb     = NULL;
    s_npmgr_cb_arg = NULL;
    return CELL_OK;
}

s32 sceNpManagerGetEntitlementById(const char* entitlementId, void* entitlement)
{
    (void)entitlement;
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!entitlementId)
        return SCE_NP_ERROR_INVALID_ARGUMENT;
    /* This local runtime has no entitlement catalog. */
    return SCE_NP_ERROR_ID_NOT_FOUND;
}

s32 sceNpManagerGetEntitlementIdList(void* entitlementIds, u32 count)
{
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;
    if (count && !entitlementIds)
        return SCE_NP_ERROR_INVALID_ARGUMENT;
    if (count > (UINT32_MAX / SCE_NP_ENTITLEMENT_ID_SIZE))
        return SCE_NP_ERROR_INVALID_ARGUMENT;
    if (entitlementIds)
        memset(GUEST_PTR(entitlementIds, void*), 0,
               (size_t)count * SCE_NP_ENTITLEMENT_ID_SIZE);
    return CELL_OK; /* empty local entitlement list */
}

/* Identity getters: reuse the fake-profile implementations (offline-with-account). */
s32 sceNpManagerGetNpId(SceNpId* npId)               { return sceNpGetNpId(0, npId); }
s32 sceNpManagerGetOnlineId(SceNpOnlineId* onlineId) { return sceNpGetOnlineId(0, onlineId); }
s32 sceNpManagerGetOnlineName(SceNpOnlineName* name) { return sceNpGetOnlineName(0, name); }
s32 sceNpManagerGetAccountAge(s32* age)              { return sceNpGetAccountAge(0, age); }

/* Parental controls: an unrestricted adult account. Unimplemented, these
 * returned OK without writing, and the title read whatever was in its stack. */
s32 sceNpManagerGetContentRatingFlag(s32* isRestricted, s32* age)
{
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (isRestricted) vm_write32((u32)(uintptr_t)isRestricted, 0);
    if (age) vm_write32((u32)(uintptr_t)age, 25);
    return CELL_OK;
}

s32 sceNpManagerGetChatRestrictionFlag(s32* isRestricted)
{
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (isRestricted) vm_write32((u32)(uintptr_t)isRestricted, 0);
    return CELL_OK;
}

/* Tickets, the account region and the network time come from the PSN
 * servers. Unimplemented, these imports answered CELL_OK with nothing
 * written: a title that requested a ticket waited for a GOT_TICKET event
 * that never came, and one that read the region or the time used whatever
 * was in its buffer. Checks and offline answers as RPCS3's sceNp.cpp:
 * without a psnr server the manager is OFFLINE and so are these. Signed in
 * to psnr, which issues no tickets, a request gets a failed GOT_TICKET at
 * once and there is never a ticket to read; the region is the fake
 * account's (US, English) and the time is the host's. */
#define SCE_NP_COOKIE_MAX_SIZE 1024
#define SCE_NP_TICKET_PARAM_SERIAL_ID   0
#define SCE_NP_TICKET_PARAM_SUBJECT_DOB 11

s32 sceNpManagerRequestTicket2(u32 npId, u32 version, u32 serviceId, u32 cookie,
                               u32 cookieSize, u32 entitlementId, u32 consumedCount)
{
    (void)version; (void)cookie; (void)entitlementId; (void)consumedCount;
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!npId || !serviceId || cookieSize > SCE_NP_COOKIE_MAX_SIZE)
        return SCE_NP_AUTH_EINVALID_ARGUMENT;
    if (!np_psnr_enabled()) {
        printf("[sceNp] ManagerRequestTicket2() -> OFFLINE\n");
        return SCE_NP_ERROR_OFFLINE;
    }
    if (s_npmgr_cb) {
        const u64 args[8] = { SCE_NP_MANAGER_EVENT_GOT_TICKET, (u64)(u32)SCE_NP_ERROR_INVALID_STATE,
                              (u64)(u32)(uintptr_t)s_npmgr_cb_arg, 0, 0, 0, 0, 0 };
        cellSysutilQueueGuestCallbackArgs((u32)(uintptr_t)s_npmgr_cb, args);
    }
    printf("[sceNp] ManagerRequestTicket2() -> GOT_TICKET failed (psnr issues no tickets)\n");
    return CELL_OK;
}

s32 sceNpManagerGetTicket(u32 buffer, u32 bufferSize)
{
    (void)buffer;
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!bufferSize) return SCE_NP_ERROR_INVALID_ARGUMENT;
    vm_write32(bufferSize, 0);   /* no ticket: nothing to copy */
    return CELL_OK;
}

s32 sceNpManagerGetTicketParam(s32 paramId, u32 param)
{
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!param || paramId < SCE_NP_TICKET_PARAM_SERIAL_ID || paramId > SCE_NP_TICKET_PARAM_SUBJECT_DOB)
        return SCE_NP_ERROR_INVALID_ARGUMENT;
    return SCE_NP_ERROR_INVALID_STATE;   /* no ticket was ever issued */
}

s32 sceNpManagerGetAccountRegion(u32 countryCode, u32 language)
{
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!countryCode || !language) return SCE_NP_ERROR_INVALID_ARGUMENT;
    if (!np_psnr_enabled()) return SCE_NP_ERROR_OFFLINE;
    vm_write8(countryCode + 0, 'u'); vm_write8(countryCode + 1, 's');   /* SceNpCountryCode */
    vm_write8(countryCode + 2, 0);   vm_write8(countryCode + 3, 0);
    vm_write32(language, 1);         /* CELL_SYSUTIL_LANG_ENGLISH_US */
    return CELL_OK;
}

s32 sceNpManagerGetNetworkTime(u32 pTick)
{
    if (!s_np_initialized) return SCE_NP_ERROR_NOT_INITIALIZED;
    if (!pTick) return SCE_NP_ERROR_INVALID_ARGUMENT;
    if (!np_psnr_enabled()) return SCE_NP_ERROR_OFFLINE;
    /* CellRtcTick: microseconds since 0001-01-01 00:00 UTC. */
    const u64 unix_us = (u64)time(NULL) * 1000000ull;
    vm_write64(pTick, unix_us + 62135596800000000ull);
    return CELL_OK;
}

/* Score setup is local even while the NP manager is offline. Games may
 * initialize it before starting the worker that decides whether to use PSN. */
s32 sceNpScoreInit(void)
{
    if (s_score_initialized)
        return SCE_NP_COMMUNITY_ERROR_ALREADY_INITIALIZED;
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;
    s_score_initialized = 1;
    return CELL_OK;
}

s32 sceNpScoreTerm(void)
{
    if (!s_score_initialized)
        return SCE_NP_COMMUNITY_ERROR_NOT_INITIALIZED;
    if (!s_np_initialized)
        return SCE_NP_ERROR_NOT_INITIALIZED;
    s_score_initialized = 0;
    return CELL_OK;
}

/* ---------------------------------------------------------------------------
 * NPDRM
 *
 * On hardware sceNpDrmIsAvailable[2](klicensee, path) checks the licence for an
 * NPDRM file and leaves the klicensee with the kernel, so later cellFsOpen calls
 * on EDATs return plaintext. Free (license type 3) and disc-bound game data use a
 * title-specific klicensee that only the game knows; record it for the EDAT
 * reader (libs/filesystem/edat.c), which verifies it against each file's
 * dev_hash before using it. The two calls differ only in licence-check details
 * that do not apply here.
 * -----------------------------------------------------------------------*/

static s32 np_drm_is_available(const u8* k_licensee, const char* drm_path)
{
    const u8* k = GUEST_PTR(k_licensee, const u8*);
    const char* path = GUEST_PTR(drm_path, const char*);
    /* A NULL klicensee is legal for content that needs none; nothing to record. */
    if (k) edat_add_klicensee(k);
    fprintf(stderr, "[sceNp] DRM availability check: %s\n", path ? path : "(null)");
    return CELL_OK;
}

s32 sceNpDrmIsAvailable(const u8* k_licensee, const char* drm_path)
{
    return np_drm_is_available(k_licensee, drm_path);
}

