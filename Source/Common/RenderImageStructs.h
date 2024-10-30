#pragma once

#include "AnalyticStructs.h"
#include "TransientPool/TransientPool.h"
#include "Core/MemAlloc.h"

struct RenderBufferInfo
{
    // Buffer range
    const Byte*         data;
    size_t              totalSize;
    // Data types of the render buffer
    // actual underlying data type is always float
    MRayColorSpaceEnum  renderColorSpace;
    // Total size of the film
    Vector2ui           resolution;
    // Given render logic's may be morphed
    // according to the internals of the renderer
    // these indices should be set by the visior
    uint32_t            curRenderLogic0;
    uint32_t            curRenderLogic1;
};

struct RenderImageSection
{
    static constexpr size_t CHANNEL_START_ALIGNMENT = MemAlloc::DefaultSystemAlignment();

    // Logical layout of the data
    // Incoming data is between these pixel ranges
    // [min, max)
    Vector2ui   pixelMin;
    Vector2ui   pixelMax;
    //
    // In addition to the per pixel accumulation
    Float       globalWeight;
    // Semaphore wait number for Visor/Runner
    uint64_t    waitCounter;
    //
    size_t      pixelStartOffset;
    size_t      weightStartOffset;

    std::array<size_t, 3>
    PixelOffsetsRGB() const;
};

struct RenderImageSaveInfo
{
    std::string     prefix;
    Float           time;   // In seconds
    Float           sample; // Mostly integer,
    // but can be fractional
};

struct TracerOptions
{
    uint64_t    mainSeed            = 0;
    int32_t     maxTextureMipLevel  = 0;
};

struct RendererOutput
{
    Optional<RendererAnalyticData>  analytics;
    Optional<RenderImageSection>    imageOut;
    bool                            triggerSave = false;
};

inline std::array<size_t, 3>
RenderImageSection::PixelOffsetsRGB() const
{
    size_t pixCount = (pixelMax - pixelMin).Multiply();
    size_t channelByteCount = Math::NextMultiple(pixCount * sizeof(Float),
                                                 CHANNEL_START_ALIGNMENT);
    size_t rOffset = pixelStartOffset + 0 * channelByteCount;
    size_t gOffset = pixelStartOffset + 1 * channelByteCount;
    size_t bOffset = pixelStartOffset + 2 * channelByteCount;

    assert(weightStartOffset == pixelStartOffset + 3 * channelByteCount);
    return {rOffset, gOffset, bOffset};
}