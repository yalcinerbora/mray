#include "RenderImageBuffer.h"

RenderImageBuffer::RenderImageBuffer(SystemSemaphoreHandle systemSem,
                                     uint64_t initialCount,
                                     const GPUSystem& gpuSystem)
    : memory(gpuSystem, true)
    , sem(systemSem)
    , semCounter(initialCount)
{}

RenderBufferInfo RenderImageBuffer::Resize(const Vector2ui& newResolution,
                                           const Vector2ui& newPixelMin,
                                           const Vector2ui& newPixelMax,
                                           uint32_t newDepth,
                                           MRayColorSpaceEnum newColorSpace)
{
    resolution  = newResolution;
    pixelMin    = newPixelMin;
    pixelMax    = newPixelMax;
    depth       = newDepth;
    colorSpace  = newColorSpace;

    uint32_t totalPixCount = (pixelMax - pixelMin).Multiply();
    Span<Float> newPixels;
    Span<Float> newSamples;
    MemAlloc::AllocateMultiData(std::tie(newPixels, newSamples),
                                memory,
                                {totalPixCount * 3, totalPixCount});
    pixels = newPixels;
    samples = newSamples;

    Byte* mem = static_cast<Byte*>(memory);
    pixStartOffset = std::distance(mem, reinterpret_cast<Byte*>(newPixels.data()));
    sampleStartOffset = std::distance(mem, reinterpret_cast<Byte*>(newSamples.data()));
    return RenderBufferInfo
    {
        .data = mem,
        .totalSize = memory.Size(),
        .renderColorSpace = colorSpace,
        .resolution = resolution,
        .depth = depth
    };
}

void RenderImageBuffer::AcquireImage(const GPUQueue& queue)
{
    queue.IssueSemaphoreWait(sem, semCounter);
}

RenderImageSection RenderImageBuffer::ReleaseImage(const GPUQueue& queue)
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

void RenderImageBuffer::ClearImage(const GPUQueue& queue)
{
    queue.MemsetAsync(pixels, 0x00);
    queue.MemsetAsync(samples, 0x00);
}