#pragma once

#include "Core/Types.h"
#include "Common/RenderImageStructs.h"
#include "Device/GPUSystem.h"

struct RenderImageParams;

class RenderImage
{
    private:
    // Mem related
    Span<Float>     pixels;
    Span<Float>     samples;
    size_t          pixStartOffset      = 0;
    size_t          sampleStartOffset   = 0;
    HostLocalMemory memory;
    GPUSemaphore    sem;
    uint64_t        semCounter = 0;
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

    // Members;
    // Access
    Span<Float>         Pixels();
    Span<Float>         Samples();
    //
    void                ClearImage(const GPUQueue& queue);
    void                AcquireImage(const GPUQueue& queue);
    RenderImageSection  ReleaseImage(const GPUQueue& queue);
    RenderBufferInfo    GetBufferInfo();
};

inline
Span<Float> RenderImage::Pixels()
{
    return pixels;
}

inline
Span<Float> RenderImage::Samples()
{
    return samples;
}