/*
 * ps3recomp - cellVideoUpload HLE
 *
 * Video upload to social networking services (YouTube, etc.).
 * Stub — init/term work, upload returns NOT_SUPPORTED.
 */

#ifndef PS3RECOMP_CELL_VIDEO_UPLOAD_H
#define PS3RECOMP_CELL_VIDEO_UPLOAD_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define CELL_VIDEO_UPLOAD_ERROR_CANCEL              0x8002D000u
#define CELL_VIDEO_UPLOAD_ERROR_NETWORK             0x8002D001u
#define CELL_VIDEO_UPLOAD_ERROR_SERVICE_STOP        0x8002D002u
#define CELL_VIDEO_UPLOAD_ERROR_SERVICE_BUSY        0x8002D003u
#define CELL_VIDEO_UPLOAD_ERROR_SERVICE_UNAVAILABLE 0x8002D004u
#define CELL_VIDEO_UPLOAD_ERROR_SERVICE_QUOTA       0x8002D005u
#define CELL_VIDEO_UPLOAD_ERROR_ACCOUNT_STOP         0x8002D006u
#define CELL_VIDEO_UPLOAD_ERROR_OUT_OF_MEMORY        0x8002D020u
#define CELL_VIDEO_UPLOAD_ERROR_FATAL                0x8002D021u
#define CELL_VIDEO_UPLOAD_ERROR_INVALID_VALUE        0x8002D022u
#define CELL_VIDEO_UPLOAD_ERROR_FILE_OPEN            0x8002D023u
#define CELL_VIDEO_UPLOAD_ERROR_INVALID_STATE        0x8002D024u

/* Compatibility aliases used by the older local API names. */
#define CELL_VIDEO_UPLOAD_ERROR_NOT_INITIALIZED     CELL_VIDEO_UPLOAD_ERROR_INVALID_STATE
#define CELL_VIDEO_UPLOAD_ERROR_ALREADY_INITIALIZED CELL_VIDEO_UPLOAD_ERROR_INVALID_STATE
#define CELL_VIDEO_UPLOAD_ERROR_INVALID_ARGUMENT    CELL_VIDEO_UPLOAD_ERROR_INVALID_VALUE
#define CELL_VIDEO_UPLOAD_ERROR_NOT_SUPPORTED       CELL_VIDEO_UPLOAD_ERROR_SERVICE_UNAVAILABLE

#define CELL_VIDEO_UPLOAD_STATUS_INITIALIZED 1
#define CELL_VIDEO_UPLOAD_STATUS_FINALIZED   2

/* Types */
typedef struct CellVideoUploadParam {
    const char* filePath;
    const char* title;
    const char* description;
    const char* tags;
    u32 reserved[4];
} CellVideoUploadParam;

typedef void (*CellVideoUploadCallback)(s32 status, s32 result,
                                        const char* resultURL, void* arg);

/* Functions */
s32 cellVideoUploadInit(void);
s32 cellVideoUploadTerm(void);

s32 cellVideoUploadInitialize(const CellVideoUploadParam* param,
                              CellVideoUploadCallback callback, void* userdata);

s32 cellVideoUploadStart(const CellVideoUploadParam* param,
                           CellVideoUploadCallback callback, void* arg);
s32 cellVideoUploadCancel(void);
s32 cellVideoUploadGetStatus(s32* status);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_VIDEO_UPLOAD_H */
