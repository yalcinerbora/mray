#pragma once

#include "Core/Types.h"
#include "Common/RenderImageStructs.h"
#include "Device/GPUSystem.h"

struct RenderImageParams;

class RenderImage
{
    private:
    // Mem related
    // According to the profiling this staging style
    // transfer was the most performant
    DeviceMemory        deviceMemory;
    Span<Float>         dPixels;
    Span<Float>         dSamples;
    //
    HostLocalMemory     stagingMemory;
    Span<Float>         hPixels;
    Span<Float>         hSamples;
    //
    size_t              pixStartOffset      = 0;
    size_t              sampleStartOffset   = 0;

    GPUSemaphoreView    sem;
    //
    uint32_t            depth       = 0;
    MRayColorSpaceEnum  colorSpace  = MRayColorSpaceEnum::MR_RAW;
    Vector2ui           resolution  = Vector2ui::Zero();
    Vector2ui           pixelMin    = Vector2ui::Zero();
    Vector2ui           pixelMax    = Vector2ui::Zero();

    public:
    // Constructors & Destructor
                        RenderImage(const RenderImageParams&,
                                    uint32_t depth, MRayColorSpaceEnum,
                                    const GPUSystem& gpuSystem);
                        RenderImage(const RenderImage&) = delete;
    RenderImage&        operator=(const RenderImage&) = delete;
                        ~RenderImage() = default;

    // Members
    // Access
    Span<Float>         Pixels();
    Span<Float>         Samples();
    //
    Vector2ui           Resolution() const;
    //
    void                ClearImage(const GPUQueue& queue);
    void                AcquireImage(const GPUQueue& queue);
    RenderImageSection  ReleaseImage(const GPUQueue& queue);
    RenderBufferInfo    GetBufferInfo();
};

inline
Span<Float> RenderImage::Pixels()
{
    return dPixels;
}

inline
Span<Float> RenderImage::Samples()
{
    return dSamples;
}