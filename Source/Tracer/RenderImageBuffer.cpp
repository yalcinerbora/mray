#include "RenderImageBuffer.h"
#include "Core/TracerI.h"

RenderImage::RenderImage(const RenderImageParams& p,
                         uint32_t depth, MRayColorSpaceEnum colorSpace,
                         const GPUSystem& gpuSystem)
    : memory(gpuSystem)
    , sem(p.semaphore)
    , semCounter(p.initialSemCounter)
    , depth(depth)
    , colorSpace(colorSpace)
    , resolution(p.resolution)
    , pixelMin(p.regionMin)
    , pixelMax(p.regionMax)
{
    uint32_t totalPixCount = (pixelMax - pixelMin).Multiply() * depth;
    static constexpr uint32_t channelCount = Vector3::Dims;

    MemAlloc::AllocateMultiData(std::tie(pixels, samples),
                                memory,
                                {totalPixCount * channelCount,
                                 totalPixCount});

    Byte* mem = static_cast<Byte*>(memory);
    pixStartOffset = static_cast<size_t>(std::distance(mem, reinterpret_cast<Byte*>(pixels.data())));
    sampleStartOffset = static_cast<size_t>(std::distance(mem, reinterpret_cast<Byte*>(samples.data())));
}

void RenderImage::AcquireImage(const GPUQueue& queue)
{
    queue.IssueSemaphoreWait(sem, semCounter);
}

RenderImageSection RenderImage::ReleaseImage(const GPUQueue& queue)
{
    semCounter++;
    queue.IssueSemaphoreSignal(sem, semCounter);

    return RenderImageSection
    {
        .pixelMin           = pixelMin,
        .pixelMax           = pixelMax,
        .globalWeight       = 0.0f,
        .waitCounter        = semCounter,
        .pixelStartOffset   = pixStartOffset,
        .sampleStartOffset  = sampleStartOffset
    };
}

void RenderImage::ClearImage(const GPUQueue& queue)
{
    queue.MemsetAsync(pixels, 0x00);
    queue.MemsetAsync(samples, 0x00);
}

RenderBufferInfo RenderImage::GetBufferInfo()
{
    return RenderBufferInfo
    {
        .data = static_cast<Byte*>(memory),
        .totalSize = memory.Size(),
        .renderColorSpace = colorSpace,
        .resolution = resolution,
        .depth = depth
    };
}