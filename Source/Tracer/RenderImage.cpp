#include "RenderImage.h"
#include "Core/TracerI.h"

RenderImage::RenderImage(const RenderImageParams& p,
                         uint32_t depth, MRayColorSpaceEnum colorSpace,
                         const GPUSystem& gpuSystem)
    : stagingMemory(gpuSystem)
    , deviceMemory(gpuSystem.AllGPUs(), 16_MiB, 128_MiB)
    , sem(p.semaphore, p.initialSemCounter)
    , processCompleteFence(gpuSystem.BestDevice().GetQueue(0))
    , depth(depth)
    , colorSpace(colorSpace)
    , resolution(p.resolution)
    , pixelMin(p.regionMin)
    , pixelMax(p.regionMax)
{
    uint32_t totalPixCount = (pixelMax - pixelMin).Multiply() * depth;
    static constexpr uint32_t channelCount = Vector3::Dims;

    MemAlloc::AllocateMultiData(std::tie(dPixels, dSamples),
                                deviceMemory,
                                {totalPixCount * channelCount,
                                 totalPixCount});
    MemAlloc::AllocateMultiData(std::tie(hPixels, hSamples),
                                stagingMemory,
                                {totalPixCount * channelCount,
                                 totalPixCount});

    // Calculate offsets, (common between buffers)
    Byte* mem = static_cast<Byte*>(deviceMemory);
    pixStartOffset = static_cast<size_t>(std::distance(mem, reinterpret_cast<Byte*>(dPixels.data())));
    sampleStartOffset = static_cast<size_t>(std::distance(mem, reinterpret_cast<Byte*>(dSamples.data())));
}

Optional<RenderImageSection> RenderImage::GetHostView(const GPUQueue& processQueue,
                                                      const GPUQueue& transferQueue)
{
    // Let's not wait on the driver here
    // So host does not runaway from CUDA
    // (I don't know if this is even an issue)
    // Host functions seems to have high variance
    // on execution so thats another reason.
    //
    // If we could not acquire the semaphore,
    // this means Visor is closing (either user dit it, or an error)
    // return nothing so that renderer do not send it etc.
    if(!sem.HostAcquire()) return std::nullopt;

    // Barrier the process queue
    processCompleteFence = processQueue.Barrier();
    // Wait the process queue to finish on the transfer queue
    transferQueue.IssueWait(processCompleteFence);
    // Copy to staging buffers when the data is ready
    transferQueue.MemcpyAsync(hPixels, ToConstSpan(dPixels));
    transferQueue.MemcpyAsync(hSamples, ToConstSpan(dSamples));
    // Here we can not wait on host here, (or we sync)
    // so we Issue the release of the semaphore as host launch
    transferQueue.IssueSemaphoreSignal(sem);
    // We should preset the next acquisition state he
    // and find the other threads wait value.
    uint64_t nextVal = sem.ChangeToNextState();

    return RenderImageSection
    {
        .pixelMin           = pixelMin,
        .pixelMax           = pixelMax,
        .globalWeight       = 0.0f,
        .waitCounter        = nextVal,
        .pixelStartOffset   = pixStartOffset,
        .sampleStartOffset  = sampleStartOffset
    };
}

Vector2ui RenderImage::Resolution() const
{
    return resolution;
}

void RenderImage::ClearImage(const GPUQueue& queue)
{
    queue.MemsetAsync(dPixels, 0x00);
    queue.MemsetAsync(dSamples, 0x00);
}

RenderBufferInfo RenderImage::GetBufferInfo()
{
    return RenderBufferInfo
    {
        .data = static_cast<Byte*>(stagingMemory),
        .totalSize = stagingMemory.Size(),
        .renderColorSpace = colorSpace,
        .resolution = resolution,
        .depth = depth
    };
}