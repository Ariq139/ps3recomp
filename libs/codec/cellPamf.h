/*
 * ps3recomp - cellPamf HLE
 *
 * CellPamfReader is intentionally incomplete on the host: HLE arguments carry
 * its guest effective address, and cellPamf.c reads/writes the actual 128-byte
 * guest ABI structure through the VM accessors rather than dereferencing it.
 */

#ifndef PS3RECOMP_CELL_PAMF_H
#define PS3RECOMP_CELL_PAMF_H

#include "ps3emu/ps3types.h"
#include "ps3emu/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CELL_PAMF_ERROR_STREAM_NOT_FOUND    0x80610501u
#define CELL_PAMF_ERROR_INVALID_PAMF        0x80610502u
#define CELL_PAMF_ERROR_INVALID_HEADER      CELL_PAMF_ERROR_INVALID_PAMF
#define CELL_PAMF_ERROR_INVALID_ARG         0x80610503u
#define CELL_PAMF_ERROR_UNKNOWN_TYPE        0x80610504u
#define CELL_PAMF_ERROR_UNSUPPORTED_VERSION 0x80610505u
#define CELL_PAMF_ERROR_UNKNOWN_STREAM      0x80610506u
#define CELL_PAMF_ERROR_EP_NOT_FOUND        0x80610507u
#define CELL_PAMF_ERROR_NOT_AVAILABLE       0x80610508u

#define CELL_PAMF_ATTRIBUTE_VERIFY_ON       0x00000001u
#define CELL_PAMF_ATTRIBUTE_MINIMUM_HEADER  0x00000002u

#define CELL_PAMF_STREAM_TYPE_AVC             0u
#define CELL_PAMF_STREAM_TYPE_M2V             1u
#define CELL_PAMF_STREAM_TYPE_ATRAC3PLUS      2u
#define CELL_PAMF_STREAM_TYPE_PAMF_LPCM       3u
#define CELL_PAMF_STREAM_TYPE_AC3             4u
#define CELL_PAMF_STREAM_TYPE_USER_DATA       5u
#define CELL_PAMF_STREAM_TYPE_PSMF_AVC        6u
#define CELL_PAMF_STREAM_TYPE_PSMF_ATRAC3PLUS 7u
#define CELL_PAMF_STREAM_TYPE_PSMF_LPCM       8u
#define CELL_PAMF_STREAM_TYPE_PSMF_USER_DATA  9u
#define CELL_PAMF_STREAM_TYPE_VIDEO           20u
#define CELL_PAMF_STREAM_TYPE_AUDIO           21u
#define CELL_PAMF_STREAM_TYPE_UNK             22u

#define CELL_PAMF_CODEC_TYPE_M2V         0x02u
#define CELL_PAMF_CODEC_TYPE_AVC         0x1Bu
#define CELL_PAMF_CODEC_TYPE_PAMF_LPCM   0x80u
#define CELL_PAMF_CODEC_TYPE_AC3         0x81u
#define CELL_PAMF_CODEC_TYPE_ATRAC3PLUS  0xDCu
#define CELL_PAMF_CODEC_TYPE_USER_DATA   0xDDu
#define CELL_PAMF_CODEC_TYPE_PSMF        0xFFu

#define CELL_PAMF_MAGIC       0x50414D46u
#define CELL_PAMF_VERSION_0040 0x30303430u
#define CELL_PAMF_VERSION_0041 0x30303431u

typedef struct CellPamfReader CellPamfReader;

typedef struct CellCodecTimeStamp {
    u32 upper;
    u32 lower;
} CellCodecTimeStamp;

typedef struct CellPamfAvcInfo {
    u8 profileIdc, levelIdc, frameMbsOnlyFlag, videoSignalInfoFlag;
    u8 frameRateInfo, aspectRatioIdc;
    u16 sarWidth, sarHeight, horizontalSize, verticalSize;
    u16 frameCropLeftOffset, frameCropRightOffset;
    u16 frameCropTopOffset, frameCropBottomOffset;
    u8 videoFormat, videoFullRangeFlag, colourPrimaries;
    u8 transferCharacteristics, matrixCoefficients;
    u8 entropyCodingModeFlag, deblockingFilterFlag;
    u8 minNumSlicePerPictureIdc, nfwIdc, maxMeanBitrate;
} CellPamfAvcInfo;

typedef struct CellPamfM2vInfo {
    u8 profileAndLevelIndication, progressiveSequence;
    u8 videoSignalInfoFlag, frameRateInfo, aspectRatioIdc;
    u16 sarWidth, sarHeight, horizontalSize, verticalSize;
    u16 horizontalSizeValue, verticalSizeValue;
    u8 videoFormat, videoFullRangeFlag, colourPrimaries;
    u8 transferCharacteristics, matrixCoefficients;
} CellPamfM2vInfo;

typedef struct CellPamfAtrac3plusInfo {
    u32 samplingFrequency;
    u8 numberOfChannels;
} CellPamfAtrac3plusInfo;

typedef struct CellPamfLpcmInfo {
    u32 samplingFrequency;
    u8 numberOfChannels;
    u16 bitsPerSample;
} CellPamfLpcmInfo;

typedef struct CellPamfAc3Info {
    u32 samplingFrequency;
    u8 numberOfChannels;
} CellPamfAc3Info;

typedef struct CellPamfEp {
    u32 indexN;
    u32 nThRefPictureOffset;
    CellCodecTimeStamp pts;
    u64 rpnOffset;
} CellPamfEp;

typedef struct CellPamfEpUnk {
    CellPamfEp ep;
    u64 nextRpnOffset;
} CellPamfEpUnk;

typedef struct CellPamfEpIterator {
    u8 isPamf;
    u8 reserved[3];
    u32 index;
    u32 num;
    u32 pCur;
} CellPamfEpIterator;

typedef struct CellCodecEsFilterId {
    u32 filterIdMajor;
    u32 filterIdMinor;
    u32 supplementalInfo1;
    u32 supplementalInfo2;
} CellCodecEsFilterId;

/* Header inspection */
s32 cellPamfGetHeaderSize(void* pamfAddr, u64 fileSize, u64* headerSize);
s32 cellPamfGetHeaderSize2(void* pamfAddr, u64 fileSize, u32 attribute,
                            u64* headerSize);
s32 cellPamfGetStreamOffsetAndSize(void* pamfAddr, u64 fileSize,
                                    u64* headerSize, u64* dataSize);
s32 cellPamfVerify(void* pamfAddr, u64 fileSize);

/* Reader lifecycle and stream queries */
s32 cellPamfReaderInitialize(CellPamfReader* reader, void* pamfAddr,
                              u64 fileSize, u32 attribute);
s32 cellPamfReaderGetPresentationStartTime(CellPamfReader* reader,
                                            CellCodecTimeStamp* startTime);
s32 cellPamfReaderGetPresentationEndTime(CellPamfReader* reader,
                                          CellCodecTimeStamp* endTime);
u32 cellPamfReaderGetMuxRateBound(CellPamfReader* reader);
u8 cellPamfReaderGetNumberOfStreams(CellPamfReader* reader);
u8 cellPamfReaderGetNumberOfSpecificStreams(CellPamfReader* reader,
                                             u8 streamType);
s32 cellPamfReaderSetStreamWithIndex(CellPamfReader* reader, u8 streamIndex);
s32 cellPamfReaderSetStreamWithTypeAndChannel(CellPamfReader* reader,
                                               u8 streamType, u8 channel);
s32 cellPamfReaderSetStreamWithTypeAndIndex(CellPamfReader* reader,
                                             u8 streamType, u8 streamIndex);
s32 cellPamfStreamTypeToEsFilterId(u8 streamType, u8 channel,
                                    CellCodecEsFilterId* esFilterId);
s32 cellPamfReaderGetStreamIndex(CellPamfReader* reader);
s32 cellPamfReaderGetStreamTypeAndChannel(CellPamfReader* reader,
                                           u8* streamType, u8* channel);
s32 cellPamfReaderGetEsFilterId(CellPamfReader* reader,
                                 CellCodecEsFilterId* esFilterId);
s32 cellPamfReaderGetStreamInfo(CellPamfReader* reader, void* info,
                                 u32 infoSize);

/* Entry-point iteration and lookup */
u32 cellPamfReaderGetNumberOfEp(CellPamfReader* reader);
s32 cellPamfReaderGetEpIteratorWithIndex(CellPamfReader* reader,
                                          u32 epIndex,
                                          CellPamfEpIterator* iterator);
s32 cellPamfReaderGetEpIteratorWithTimeStamp(
    CellPamfReader* reader, CellCodecTimeStamp* timeStamp,
    CellPamfEpIterator* iterator);
s32 cellPamfEpIteratorGetEp(CellPamfEpIterator* iterator, CellPamfEp* ep);
s32 cellPamfEpIteratorMove(CellPamfEpIterator* iterator, s32 steps,
                            CellPamfEp* ep);
s32 cellPamfReaderGetEpWithTimeStamp(CellPamfReader* reader,
                                      CellCodecTimeStamp* timeStamp,
                                      CellPamfEpUnk* ep, u32 unk);
s32 cellPamfReaderGetEpWithIndex(CellPamfReader* reader, u32 epIndex,
                                  CellPamfEpUnk* ep, u32 unk);

#ifdef __cplusplus
}
#endif

#endif /* PS3RECOMP_CELL_PAMF_H */
