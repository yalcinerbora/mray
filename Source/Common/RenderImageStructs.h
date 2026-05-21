#pragma once

#include "AnalyticStructs.h"

#include "Core/Optional.h"

struct RenderLogicChange
{
    uint32_t index;
    uint32_t newValue;
};

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
};

struct RenderImageSection
{
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
    std::array<size_t, 3>   pixStartOffsets;
    size_t                  weightStartOffset;
};

struct RenderImageSaveInfo
{
    std::string     prefix;
    double          time;         // In seconds
    double          workPerPixel;
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