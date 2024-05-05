#pragma once

#include "Core/Types.h"
#include "Device/GPUSystem.h"
#include "Common/RenderImageStructs.h"

#include <memory>

class RenderImageBuffer
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
                        RenderImageBuffer(SystemSemaphoreHandle sem,
                                          uint64_t initialCount,
                                          const GPUSystem& gpuSystem);
                        RenderImageBuffer(const RenderImageBuffer&) = delete;
    RenderImageBuffer&  operator=(const RenderImageBuffer&) = delete;
                        ~RenderImageBuffer() = default;

    // Members
    RenderBufferInfo    Resize(const Vector2ui& resolution,
                               const Vector2ui& pixelMin,
                               const Vector2ui& pixelMax,
                               uint32_t depth,
                               MRayColorSpaceEnum colorSpace);
    // Access
    Span<Float>         Pixels();
    Span<Float>         Samples();
    //
    void                ClearImage(const GPUQueue& queue);
    void                AcquireImage(const GPUQueue& queue);
    RenderImageSection  ReleaseImage(const GPUQueue& queue);
};

inline
Span<Float> RenderImageBuffer::Pixels()
{
    return pixels;
}

inline
Span<Float> RenderImageBuffer::Samples()
{
    return samples;
}