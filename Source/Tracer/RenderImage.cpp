#include "RenderImage.h"
#include "Core/TracerI.h"

#ifdef MRAY_GPU_BACKEND_CPU
    #include "Device/GPUSystem.hpp"
#endif

Vector2ui ImageTiler::ResponsibleSize() const
{
    return range[1] - range[0];
}

Vector2ui ImageTiler::GlobalTileStart() const
{
    return range[0] + LocalTileStart();
}

Vector2ui ImageTiler::GlobalTileEnd() const
{

    return range[0] + LocalTileEnd();
}

Vector2ui ImageTiler::FindOptimumTileSize(const Vector2ui& fbSize,
                                          uint32_t parallelizationHint)
{
    using namespace Math;
    // Start with an ~ aspect ratio tile
    // and adjust it
    Float aspectRatio = Float(fbSize[0]) / Float(fbSize[1]);
    Float factor = std::sqrt(Float(parallelizationHint) / aspectRatio);
    Vector2ui tileHint(std::round(aspectRatio * factor), std::roundf(factor));

    // Find optimal tile size that evenly divides the image
    // This may not happen (i.e., width or height is prime)
    // then expand the tile size to pass the edge barely.
    auto Adjust = [&](uint32_t i)
    {
        // If w/h is small use the full fb w/h
        if(fbSize[i] < tileHint[i]) return fbSize[i];

        // Divide down to get an aggressive (lower) count,
        // but on second pass do a conservative divide
        Float tileCountF = Float(fbSize[i]) / Float(tileHint[i]);
        uint32_t tileCount = uint32_t(std::round(tileCountF));
        // Try to minimize residuals so that
        // GPU does consistent work
        uint32_t result = fbSize[i] / tileCount;
        uint32_t residual = fbSize[i] % tileCount;
        residual = DivideUp(residual, tileCount);
        result += residual;
        return result;
    };

    Vector2ui result = Vector2ui(Adjust(0), Adjust(1));
    return result;
}

ImageTiler::ImageTiler(RenderImage* rI,
                       const RenderImageParams& rIParams,
                       uint32_t parallelizationHint,
                       Vector2ui filterPadding)
    : renderBuffer(rI)
    , fullResolution(rIParams.resolution)
    , range{rIParams.regionMin, rIParams.regionMax}
{
    // Since we partition the image into tiles
    // some tiles can cover out of bound pixels
    // This value is the largest tile size.
    // In function call time this class will return
    // the actual tile size depending on current tile
    Vector2ui fbSize = ResponsibleSize();
    coveringTileSize = FindOptimumTileSize(fbSize,
                                       parallelizationHint);

    paddedTileSize = coveringTileSize + filterPadding * 2u;
    renderBuffer->Resize(paddedTileSize);
    tileCount = Math::DivideUp(fbSize, coveringTileSize);

    pixel1DRange = Vector2ui(0u, CurrentTileSize().Multiply());
}

Vector2ui ImageTiler::FullResolution() const
{
    return fullResolution;
}

Vector2ui ImageTiler::LocalTileStart() const
{
    Vector2ui tileIndex2D = CurrentTileIndex();
    Vector2ui start = tileIndex2D * coveringTileSize;
    return start;
}

Vector2ui ImageTiler::LocalTileEnd() const
{
    Vector2ui tileIndex2D = CurrentTileIndex();
    Vector2ui end = (tileIndex2D + 1u) * coveringTileSize;
    end = Vector2ui::Min(end, ResponsibleSize());
    return end;
}

Vector2ui ImageTiler::CurrentTileSize() const
{
    auto start = GlobalTileStart();
    auto end = GlobalTileEnd();
    return end - start;
}

Vector2ui ImageTiler::Tile1DRange() const
{
    return pixel1DRange;
}

Vector2ui ImageTiler::ConservativeTileSize() const
{
    return coveringTileSize;
}

Vector2ui ImageTiler::CurrentTileIndex() const
{
    return Vector2ui(currentTile % tileCount[0],
                     currentTile / tileCount[0]);
}

uint32_t ImageTiler::CurrentTileIndex1D() const
{
    return currentTile;
}

Vector2ui ImageTiler::TileCount() const
{
    return tileCount;
}

void ImageTiler::NextTile()
{
    using Math::Roll;
    currentTile = uint32_t(Roll(static_cast<int32_t>(currentTile + 1),
                                0, int32_t(tileCount.Multiply())));
    //
    pixel1DRange = Vector2ui(pixel1DRange[1],
                             pixel1DRange[1] + CurrentTileSize().Multiply());
    if(currentTile == 0)
    {
        pixel1DRange -= Vector2ui((range[1] - range[0]).Multiply());
        assert(pixel1DRange[0] == 0);
    }
}

Optional<RenderImageSection>
ImageTiler::TransferToHost(const GPUQueue& processQueue,
                           const GPUQueue& transferQueue)
{
    auto imageSection = renderBuffer->TransferToHost(processQueue,
                                                     transferQueue);
    if(!imageSection.has_value()) return imageSection;

    imageSection->pixelMin = GlobalTileStart();
    imageSection->pixelMax = GlobalTileEnd();
    return imageSection;
}

RenderImage::RenderImage(TimelineSemaphore* semaphore,
                         uint32_t importAlignmentIn,
                         uint64_t initialSemCounter,
                         const GPUSystem& sys)
    : gpuSystem(sys)
    , importAlignment(importAlignmentIn)
    , deviceMemory(gpuSystem.AllGPUs(), 16_MiB, 256_MiB)
    , stagingMemory(gpuSystem, importAlignment, true)
    , sem(semaphore, initialSemCounter)
    , processCompleteFence(gpuSystem.BestDevice().GetComputeQueue(0))
    , previousCopyCompleteFence(gpuSystem.BestDevice().GetComputeQueue(0))
{}

Optional<RenderImageSection> RenderImage::TransferToHost(const GPUQueue& processQueue,
                                                         const GPUQueue& copyQueue)
{
    using namespace std::string_view_literals;
    static const auto semWaitAnnotation = gpuSystem.CreateAnnotation("RB Semaphore Wait");

    // Let's not wait on the driver here
    // So host does not runaway from CUDA
    // (I don't know if this is even an issue)
    // Host functions seems to have high variance
    // on execution so thats another reason.
    //
    // If we could not acquire the semaphore,
    // this means Visor is closing (either user dit it, or an error)
    // return nothing so that renderer do not send it etc.
    {
        const auto _ = semWaitAnnotation.AnnotateScope();
        if(!sem.HostAcquire()) return std::nullopt;
    }
    // Barrier the process queue
    processCompleteFence = processQueue.Barrier();
    // Wait the process queue to finish on the transfer queue
    copyQueue.IssueWait(processCompleteFence);
    // Copy to staging buffers when the data is ready
    copyQueue.MemcpyAsync(hPixelsAll, ToConstSpan(dPixelsAll));
    copyQueue.MemcpyAsync(hWeights, ToConstSpan(dWeights));

    // Do not overwrite until memcpy finishes
    previousCopyCompleteFence = copyQueue.Barrier();
    // Here, we can not wait on host here, (or we sync)
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
        .pixStartOffsets    = pixStartOffsets,
        .weightStartOffset  = weightStartOffset
    };
}

Vector2ui RenderImage::Extents() const
{
    return extent;
}

uint32_t RenderImage::ChannelCount() const
{
    return 3;
}

void RenderImage::ClearImage(const GPUQueue& queue)
{
    queue.MemsetAsync(dPixelsAll, 0x00);
    queue.MemsetAsync(dWeights, 0x00);
}

Pair<const Byte*, size_t> RenderImage::SharedDataPtrAndSize() const
{
    auto constPtr = static_cast<const Byte*>(stagingMemory);
    return {constPtr, stagingMemory.AllocSize()};
}

bool RenderImage::Resize(const Vector2ui& extentIn)
{
    // Acquire the memory, we may delete it
    if(!sem.HostAcquire()) return false;
    extent = extentIn;

    uint32_t totalPixCount = extent.Multiply();

    MemAlloc::AllocateMultiData(std::tie(dPixelsR, dPixelsG,
                                         dPixelsB, dWeights),
                                deviceMemory,
                                {totalPixCount, totalPixCount,
                                 totalPixCount, totalPixCount});
    size_t allSizeDevice = size_t(std::distance(dPixelsR.data(), dWeights.data()));
    dPixelsAll = Span<Float>(dPixelsR.data(), allSizeDevice);

    // Reallocate host buffer
    MemAlloc::AllocateMultiData(std::tie(hPixelsR, hPixelsG,
                                         hPixelsB, hWeights),
                                stagingMemory,
                                {totalPixCount, totalPixCount,
                                 totalPixCount, totalPixCount});
    size_t allSizeHost = size_t(std::distance(hPixelsR.data(), hWeights.data()));
    hPixelsAll = Span<Float>(hPixelsR.data(), allSizeHost);

    // Calculate offsets
    Byte* mem = static_cast<Byte*>(stagingMemory);
    Float* memF = reinterpret_cast<Float*>(mem);
    std::array<ptrdiff_t, 3> pixOffsets =
    {
        std::distance(memF, hPixelsR.data()),
        std::distance(memF, hPixelsG.data()),
        std::distance(memF, hPixelsB.data())
    };
    assert(pixOffsets[0] >= 0 && pixOffsets[1] >= 0 && pixOffsets[2] >= 0);
    assert(pixOffsets[0] < std::ptrdiff_t(hPixelsAll.size()) &&
           pixOffsets[1] < std::ptrdiff_t(hPixelsAll.size()) &&
           pixOffsets[2] < std::ptrdiff_t(hPixelsAll.size()));
    pixStartOffsets =
    {
        size_t(pixOffsets[0]) * sizeof(Float),
        size_t(pixOffsets[1]) * sizeof(Float),
        size_t(pixOffsets[2]) * sizeof(Float)
    };
    weightStartOffset = static_cast<size_t>(std::distance(memF, hWeights.data()));
    weightStartOffset *= sizeof(Float);

    sem.HostRelease();
    sem.SkipAState();
    return true;
}