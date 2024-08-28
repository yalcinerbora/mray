#pragma once

#include "AnalyticStructs.h"
#include "TransientPool/TransientPool.h"

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
    // Render output may be spectral data then this represents
    // amount of spectral samples (equally distributed)
    uint32_t            depth;
    // Given render logic's may be morphed
    // according to the internals of the renderer
    // these indices should be set by the visior
    uint32_t            curRenderLogic0;
    uint32_t            curRenderLogic1;
};

struct RenderImageSection
{
    // Logical layout of the data
    // Incoming data is between these pixel ranges
    Vector2ui   pixelMin;
    Vector2ui   pixelMax;
    // In addition to the per pixel accumulation
    float       globalWeight;
    //
    uint64_t    waitCounter;
    // Pixel data starts over this offset (this should be almost always zero)
    size_t      pixelStartOffset;
    // Pixel weights starts from this offset
    size_t      weightStartOffset;
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
};