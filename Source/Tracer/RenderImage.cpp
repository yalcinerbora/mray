#include "RenderImage.h"
#include "Core/TracerI.h"

RenderImage::RenderImage(TimelineSemaphore* semaphore,
                         uint32_t importAlignmentIn,
                         uint64_t initialSemCounter,
                         const GPUSystem& sys)
    : gpuSystem(sys)
    , importAlignment(importAlignmentIn)
    , deviceMemory(gpuSystem.AllGPUs(), 16_MiB, 256_MiB)
    , stagingMemory(gpuSystem, true)
    , sem(semaphore, initialSemCounter)
    , processCompleteFence(gpuSystem.BestDevice().GetComputeQueue(0))
{}

Optional<RenderImageSection> RenderImage::TransferToHost(const GPUQueue& processQueue,
                                                         const GPUQueue& copyQueue)
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
    copyQueue.IssueWait(processCompleteFence);
    // Copy to staging buffers when the data is ready
    copyQueue.MemcpyAsync(hPixels, ToConstSpan(dPixels));
    copyQueue.MemcpyAsync(hSamples, ToConstSpan(dSamples));
    // Here we can not wait on host here, (or we sync)
    // so we Issue the release of the semaphore as host launch
    copyQueue.IssueSemaphoreSignal(sem);
    // We should preset the next acquisition state he
    // and find the other threads wait value.
    uint64_t nextVal = sem.ChangeToNextState();

    return RenderImageSection
    {
        .pixelMin           = Vector2ui::Zero(),
        .pixelMax           = extent,
        .globalWeight       = 0.0f,
        .waitCounter        = nextVal,
        .pixelStartOffset   = pixStartOffset,
        .sampleStartOffset  = sampleStartOffset
    };
}

Vector2ui RenderImage::Extents() const
{
    return extent;
}

uint32_t RenderImage::Depth() const
{
    return depth;
}

uint32_t RenderImage::ChannelCount() const
{
    return channelCount;
}

void RenderImage::ClearImage(const GPUQueue& queue)
{
    queue.MemsetAsync(dPixels, 0x00);
    queue.MemsetAsync(dSamples, 0x00);
}

RenderBufferInfo RenderImage::GetBufferInfo(MRayColorSpaceEnum colorspace,
                                            const Vector2ui& resolution,
                                            uint32_t totalDepth)
{
    return RenderBufferInfo
    {
        .data = static_cast<Byte*>(stagingMemory),
        .totalSize = hostAllocTotalSize,
        .renderColorSpace = colorspace,
        .resolution = resolution,
        .depth = totalDepth
    };
}

bool RenderImage::Resize(const Vector2ui& extentIn,
                         uint32_t depthIn,
                         uint32_t channelCountIn)
{
    // Acquire the memory, we may delete it
    if(!sem.HostAcquire()) return false;
    extent = extentIn;
    depth = depthIn;
    channelCount = channelCountIn;

    uint32_t totalPixCount = extent.Multiply() * depth;
    MemAlloc::AllocateMultiData(std::tie(dPixels, dSamples),
                                deviceMemory,
                                {totalPixCount * channelCount,
                                 totalPixCount});

    // For host pixels allocate by hand first,
    // because vulkan etc needs exact multiple of the
    // import alignment
    size_t hSize = totalPixCount * channelCount * sizeof(Float);
    hSize = MathFunctions::NextMultiple(hSize, MemAlloc::DefaultSystemAlignment());
    hSize += totalPixCount * sizeof(Float);
    hSize = MathFunctions::NextMultiple<size_t>(hSize, importAlignment);
    hostAllocTotalSize = hSize;
    stagingMemory.ResizeBuffer(hostAllocTotalSize);
    // Since we construct host allocation as "never decrease"
    // this should not reallocate
    MemAlloc::AllocateMultiData(std::tie(hPixels, hSamples),
                                stagingMemory,
                                {totalPixCount * channelCount,
                                 totalPixCount});
    assert(hostAllocTotalSize >= stagingMemory.Size());

    // Calculate offsets
    Byte* mem = static_cast<Byte*>(deviceMemory);
    pixStartOffset = static_cast<size_t>(std::distance(mem, reinterpret_cast<Byte*>(dPixels.data())));
    sampleStartOffset = static_cast<size_t>(std::distance(mem, reinterpret_cast<Byte*>(dSamples.data())));

    sem.HostRelease();
    sem.SkipAState();
    return true;
}