#include "PathTracerRendererBase.h"

#include "Tracer/RendererCommon.h"

#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgGeneric.h"
#include "Device/GPUSystem.hpp"

#include "Core/Timer.h"

#include <numeric>

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCWriteInvalidRaysIndirect(MRAY_GRID_CONSTANT const Span<PathDataPack> dPathDataPack,
                                MRAY_GRID_CONSTANT const Span<RayGMem> dRaysOut,
                                MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    static constexpr Vector3 LARGE_VEC = Vector3(std::numeric_limits<Float>::max());

    KernelCallParams kp;
    uint32_t rayCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        Ray ray = Ray(LARGE_VEC, LARGE_VEC);
        Vector2 tMM = Vector2(std::numeric_limits<Float>::max());

        dPathDataPack[index].status.Set(uint32_t(PathStatusEnum::INVALID));
        RayToGMem(dRaysOut, index, ray, tMM);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCInitPathStateIndirect(MRAY_GRID_CONSTANT const Span<Spectrum> dThroughputs,
                             MRAY_GRID_CONSTANT const Span<Spectrum> dPathRadiance,
                             MRAY_GRID_CONSTANT const Span<PathDataPack> dPathDataPack,
                             MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    KernelCallParams kp;
    uint32_t pathCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < pathCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        dPathRadiance[index] = Spectrum::Zero();
        // dImageCoordinates is set by cam ray gen
        // dFilmFilterWeights is set by cam ray gen
        dThroughputs[index] = Spectrum(1);
        dPathDataPack[index] = PathDataPack
        {
            .depth = 0,
            .status = PathStatus(uint8_t(0)),
            .type = RayType::CAMERA_RAY
        };
    }
}

uint64_t PathTracerRendererBase::TotalSampleLimit(uint32_t spp) const
{
    if(spp == std::numeric_limits<uint32_t>::max())
        return std::numeric_limits<uint64_t>::max();

    Vector2ul tileSize(imageTiler.CurrentTileSize());
    uint64_t result = uint64_t(spp) * tileSize.Multiply();
    return result;
}

typename PathTracerRendererBase::ReloadPathOutput
PathTracerRendererBase::ReloadPaths(Span<const RayIndex> dIndices,
                                   uint32_t sppLimit, const GPUQueue& processQueue)
{
    // RELOADING!!!
    // Find the dead rays
    auto [hDeadRayRanges, dDeadAliveRayIndices] = rayPartitioner.BinaryPartition
    (
        dIndices, processQueue,
        IsAliveFunctor(dPathDataPack, true)
    );
    processQueue.Barrier().Wait();

    // Generate RN for camera rays
    RNRequestList camSampleRNList = curCamWork->StochasticFilterSampleRayRNList();
    uint32_t camSamplePerRay = camSampleRNList.TotalRNCount();
    uint32_t deadRayCount = hDeadRayRanges[2] - hDeadRayRanges[1];
    Span<RayIndex> dDeadRayIndices = dDeadAliveRayIndices.subspan(hDeadRayRanges[1],
                                                                  deadRayCount);
    Span<RayIndex> dFilledRayIndices;
    uint32_t aliveRayCount = hDeadRayRanges[1] - hDeadRayRanges[0];
    if(!dDeadRayIndices.empty())
    {
        uint64_t& tilePixIndex = tilePathCounts[imageTiler.CurrentTileIndex1D()];
        uint64_t localPathLimit = TotalSampleLimit(sppLimit);
        uint64_t pixelIndexLimit = Math::Min(tilePixIndex + deadRayCount, localPathLimit);
        uint32_t fillRayCount = static_cast<uint32_t>(pixelIndexLimit - tilePixIndex);
        fillRayCount = Math::Min(deadRayCount, fillRayCount);
        aliveRayCount += fillRayCount;

        if(fillRayCount != 0)
        {
            dFilledRayIndices = dDeadRayIndices.subspan(0, fillRayCount);
            uint32_t camRayGenRNCount = fillRayCount * camSamplePerRay;
            auto dCamRayGenRNBuffer = dRandomNumBuffer.subspan(0, camRayGenRNCount);

            rnGenerator->IncrementSampleIdIndirect(ToConstSpan(dFilledRayIndices),
                                                   processQueue);
            rnGenerator->GenerateNumbersIndirect(dCamRayGenRNBuffer,
                                                 ToConstSpan(dFilledRayIndices),
                                                 0, camSampleRNList,
                                                 processQueue);

            const auto& cameraWork = *curCamWork;
            cameraWork.GenRaysStochasticFilter
            (
                dRayCones, dRays,
                dImageCoordinates,
                dFilmFilterWeights,
                ToConstSpan(dFilledRayIndices),
                ToConstSpan(dCamRayGenRNBuffer),
                dSubCameraBuffer, curCamTransformKey,
                tilePixIndex,
                imageTiler.CurrentTileSize(),
                tracerView.tracerParams.filmFilter,
                processQueue
            );
            tilePixIndex += fillRayCount;

            // Initialize the state of new rays
            processQueue.IssueWorkKernel<KCInitPathStateIndirect>
            (
                "KCInitPathStateIndirect",
                DeviceWorkIssueParams{.workCount = fillRayCount},
                dThroughputs,
                dPathRadiance,
                dPathDataPack,
                dFilledRayIndices
            );

            // Init path state set the throughput to one.
            // Now we can call spectral wavelength sample routine.
            // It stores the PDF values in the througput
            uint32_t usedRNDimensionCount = camSamplePerRay;
            if(spectrumContext != nullptr)
            {
                RNRequestList spectralRNList = spectrumContext->SampleSpectrumRNList();
                uint32_t rnPerSample = spectralRNList.TotalRNCount();
                uint32_t specSampleRNCount = fillRayCount * rnPerSample;
                // Offset the dimensions
                auto sampleDims = Vector2ui(camSamplePerRay);
                sampleDims[1] += rnPerSample;

                auto dSpecWaveGenRNBuffer = dRandomNumBuffer.subspan(0, specSampleRNCount);
                rnGenerator->GenerateNumbersIndirect(dSpecWaveGenRNBuffer,
                                                     ToConstSpan(dFilledRayIndices),
                                                     uint16_t(camSamplePerRay), spectralRNList,
                                                     processQueue);

                // Actual wavelength sampling
                spectrumContext->SampleSpectrumWavelengthsIndirect
                (
                    dPathWavelengths,
                    dSpectrumWavePDFs,
                    ToConstSpan(dSpecWaveGenRNBuffer),
                    ToConstSpan(dFilledRayIndices),
                    processQueue
                );

                usedRNDimensionCount += rnPerSample;
            }

            // Set the used dimensions of each path
            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, ToConstSpan(dFilledRayIndices), processQueue,
                SetFunctor_U16(uint16_t(usedRNDimensionCount))
            );
        }
        //
        if(fillRayCount != deadRayCount)
        {
            uint32_t unusedRays = deadRayCount - fillRayCount;
            auto dUnusedIndices = dDeadRayIndices.subspan(fillRayCount, unusedRays);
            // Fill the remaining values
            processQueue.IssueWorkKernel<KCWriteInvalidRaysIndirect>
            (
                "KCWriteInvalidRaysIndirect",
                DeviceWorkIssueParams{.workCount = unusedRays},
                //
                dPathDataPack,
                dRays,
                dUnusedIndices
            );
        }
    }
    // Index buffer may be invalidated (Binary partition should not
    // invalidate but lets return the new buffer)
    return ReloadPathOutput
    {
        .dAllIndices = dDeadAliveRayIndices,
        .dFilledRayIndices = dFilledRayIndices,
        .aliveRayCount = aliveRayCount
    };
}

void PathTracerRendererBase::ResetAllPaths(const GPUQueue& queue)
{
    // Clear states
    queue.MemsetAsync(dPathDataPack, 0x00);
    // Set all paths as dead (we just started)
    DeviceAlgorithms::InPlaceTransform(dPathDataPack, queue,
                                       SetPathStateFunctor(PathStatusEnum::INVALID, true));
    queue.MemsetAsync(dPathRNGDimensions, 0x00);
}

Optional<RenderImageSection>
PathTracerRendererBase::AddRadianceToRenderBufferThroughput(Span<const RayIndex> dDeadRayIndices,
                                                            const GPUQueue& processQueue,
                                                            const GPUQueue& transferQueue)
{
    if(dDeadRayIndices.empty()) return std::nullopt;

    Optional<RenderImageSection> renderOut;

    DeviceAlgorithms::InPlaceTransformIndirect
    (
        dPathDataPack, ToConstSpan(dDeadRayIndices), processQueue,
        SetPathStateFunctor(PathStatusEnum::INVALID)
    );

    if(spectrumContext != nullptr)
    {
        spectrumContext->ConvertSpectrumToRGBIndirect
        (
            dPathRadiance,
            //
            ToConstSpan(dPathWavelengths),
            ToConstSpan(dSpectrumWavePDFs),
            ToConstSpan(dDeadRayIndices),
            processQueue
        );
    }

    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    renderBuffer->ClearImage(processQueue);
    ImageSpan filmSpan = imageTiler.GetTileSpan();
    SetImagePixelsIndirectAtomic
    (
        filmSpan,
        dDeadRayIndices,
        ToConstSpan(dPathRadiance),
        ToConstSpan(dFilmFilterWeights),
        ToConstSpan(dImageCoordinates),
        Float(1), processQueue
    );
    // Issue a send of the FBO to Visor
    renderOut = imageTiler.TransferToHost(processQueue, transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value()) return std::nullopt;
    //
    renderOut->globalWeight = Float(1);
    return renderOut;
}

void
PathTracerRendererBase::AddRadianceToRenderBufferLatency(Span<const RayIndex> dDeadRayIndices,
                                                         const GPUQueue& processQueue) const
{
    if(dDeadRayIndices.empty()) return;

    DeviceAlgorithms::InPlaceTransformIndirect
    (
        dPathDataPack, ToConstSpan(dDeadRayIndices), processQueue,
        SetPathStateFunctor(PathStatusEnum::INVALID)
    );

    if(spectrumContext)
    {
        spectrumContext->ConvertSpectrumToRGBIndirect
        (
            dPathRadiance,
            //
            ToConstSpan(dPathWavelengths),
            ToConstSpan(dSpectrumWavePDFs),
            ToConstSpan(dDeadRayIndices),
            processQueue
        );
    }

    ImageSpan filmSpan = imageTiler.GetTileSpan();
    SetImagePixelsIndirect
    (
        filmSpan,
        ToConstSpan(dDeadRayIndices),
        ToConstSpan(dPathRadiance),
        ToConstSpan(dFilmFilterWeights),
        ToConstSpan(dImageCoordinates),
        Float(1), processQueue
    );
}

RendererAnalyticData
PathTracerRendererBase::CalculateAnalyticDataThroughput(size_t deadRayCount, uint32_t totalSPP,
                                                        const Timer& timer)
{
    totalDeadRayCount += deadRayCount;
    // Basic Idea is to track how fast we kill (complete) paths
    // per iteration
    double timeSec = timer.Elapsed<Second>();
    double pathPerSec = static_cast<double>(deadRayCount);
    pathPerSec /= timeSec;
    pathPerSec /= 1'000'000.0;
    uint64_t totalPixels = imageTiler.FullResolution().Multiply();
    double spp = double(totalDeadRayCount) / double(totalPixels);

    return RendererAnalyticData
    {
        pathPerSec,
        "M path/s",
        spp,
        double(totalSPP),
        "spp",
        float(timer.Elapsed<Millisecond>()),
        imageTiler.FullResolution(),
        tracerView.tracerParams.globalTextureColorSpace,
        GPUMemoryUsage(),
        0,
        0
    };
}

RendererAnalyticData
PathTracerRendererBase::CalculateAnalyticDataLatency(uint32_t passPathCount,
                                                     uint32_t totalSPP,
                                                     const Timer& timer) const
{
    uint32_t tileCount1D = imageTiler.TileCount().Multiply();

    double timeSec = timer.Elapsed<Second>();
    double pathPerSec = static_cast<double>(passPathCount);
    pathPerSec /= timeSec;
    pathPerSec /= 1'000'000;

    uint64_t sppSum = std::reduce(tileSPPs.cbegin(), tileSPPs.cend(), uint64_t(0));
    double spp = double(sppSum) / double(tileCount1D);

    return RendererAnalyticData
    {
        pathPerSec,
        "M path/s",
        spp,
        double(totalSPP),
        "spp",
        float(timer.Elapsed<Millisecond>()),
        imageTiler.FullResolution(),
        tracerView.tracerParams.globalTextureColorSpace,
        GPUMemoryUsage(),
        0,
        0
    };
}

typename PathTracerRendererBase::InitOutput
PathTracerRendererBase::InitializeForRender(CamSurfaceId camSurfId,
                                            uint32_t sppLimit,
                                            bool retainCameraTransform,
                                            const RenderImageParams& renderImgParams)
{
    // Create work classes specific for this renderer
    uint32_t totalWorkCount = GenerateWorks();

    // ========================= //
    //     Camera and Stuff      //
    // ========================= //
    assert(currentCameraWorks.size() != 0);
    // Find camera surface and get keys work instance for that
    // camera etc.
    auto surfLoc = std::find_if(tracerView.camSurfs.cbegin(),
                                tracerView.camSurfs.cend(),
                                [camSurfId](const auto& pair)
                                {
                                    return pair.first == camSurfId;
                                });
    if(surfLoc == tracerView.camSurfs.cend())
        throw MRayError("[{:s}]: Unknown camera surface id ({:d})",
                        rendererName, uint32_t(camSurfId));

    curCamSurfaceParams = surfLoc->second;
    // Find the transform/camera work for this specific surface
    curCamKey                 = std::bit_cast<CameraKey>(curCamSurfaceParams.cameraId);
    curCamTransformKey        = std::bit_cast<TransformKey>(curCamSurfaceParams.transformId);
    CameraGroupId camGroupId  = CameraGroupId(curCamKey.FetchBatchPortion());
    TransGroupId transGroupId = TransGroupId(curCamTransformKey.FetchBatchPortion());
    auto packLoc = std::find_if(currentCameraWorks.cbegin(),
                                currentCameraWorks.cend(),
                                [camGroupId, transGroupId](const auto& pack)
                                {
                                    return (pack.cgId == camGroupId &&
                                            pack.tgId == transGroupId);
                                });
    assert(packLoc != currentCameraWorks.cend());
    curCamWork = packLoc->workPtr.get();

    // ========================= //
    //           Filter          //
    // ========================= //
    auto FilterGen = tracerView.filterGenerators.at(tracerView.tracerParams.filmFilter.type);
    if(!FilterGen)
        throw MRayError("[{}]: Unknown film filter type {}.", rendererName,
                        uint32_t(tracerView.tracerParams.filmFilter.type));

    Float radius = tracerView.tracerParams.filmFilter.radius;
    filmFilter = FilterGen->get()(gpuSystem, Float(radius));

    // ========================= //
    //        Image Tiler        //
    // ========================= //
    // Image tiler, automatically tiles images with the given parallelization hint
    // This is desired since the renderer is wavefront so if we allocate path states
    // for each pixel on a 4k/8k image it will easily get couple of GiBs.
    imageTiler = ImageTiler(renderBuffer.get(), renderImgParams,
                            tracerView.tracerParams.parallelizationHint,
                            Vector2ui::Zero());
    // From the img tiles, we can get the maximum circulating ray/path count
    // that will be passed to the actual renderer at the end of this function.
    uint32_t maxRayCount = imageTiler.ConservativeTileSize().Multiply();

    // ========================= //
    //      Ray Paritioner       //
    // ========================= //
    // Initialize ray partitioner with worst case scenario,
    // All work types are used. (We do not use camera work
    // for this type of renderer)
    rayPartitioner = RayPartitioner(gpuSystem, maxRayCount, totalWorkCount);

    // ========================= //
    //  Base Accelerator Tracer  //
    // ========================= //
    // Also allocate for the partitioner inside the
    // base accelerator (This should not allocate for HW accelerators)
    tracerView.baseAccelerator.AllocateForTraversal(maxRayCount);

    // ========================= //
    //  Random Number Generator  //
    // ========================= //
    auto RngGen = tracerView.rngGenerators.at(tracerView.tracerParams.samplerType.e);
    if(!RngGen) throw MRayError("[{}]: Unknown random number generator type {}.",
                                rendererName,
                                uint32_t(tracerView.tracerParams.samplerType.e));
    //
    Vector2ui maxDeviceLocalRNGCount = imageTiler.ConservativeTileSize();
    uint64_t seed = tracerView.tracerParams.seed;
    rnGenerator = RngGen->get()(renderImgParams,
                                std::move(maxDeviceLocalRNGCount),
                                std::move(sppLimit), std::move(seed),
                                gpuSystem, globalThreadPool);

    // ========================= //
    //        SPP State          //
    // ========================= //
    totalIterationCount = 0;
    totalDeadRayCount = 0;
    // This vector is for tracking how many samples are passed to each tile
    tilePathCounts.resize(imageTiler.TileCount().Multiply(), 0u);
    tileSPPs.resize(imageTiler.TileCount().Multiply(), 0u);
    std::fill(tilePathCounts.begin(), tilePathCounts.end(), 0u);
    std::fill(tileSPPs.begin(), tileSPPs.end(), 0u);

    // Clear old transforms from the
    if(!retainCameraTransform)
    {
        cameraTransform = std::nullopt;
        curCamTransformOverride = std::nullopt;
    }
    return InitOutput
    {
        .maxRayCount = maxRayCount,
        .totalWorkCount = totalWorkCount
    };
}

PathTracerRendererBase::PathTracerRendererBase(const RenderImagePtr& rip,
                                               TracerView tracerView,
                                               ThreadPool& tp, const GPUSystem& gpuSys,
                                               const RenderWorkPack& wp,
                                               std::string_view rendererName)
    : RendererBase(rip, wp, tracerView, gpuSys, tp, rendererName)
    , rayPartitioner(gpuSys)
{}

RendererOutput
PathTracerRendererBase::DoRender()
{
    static const auto annotation = gpuSystem.CreateAnnotation("Render Frame");
    const auto _ = annotation.AnnotateScope();

    // TODO: Like many places of this codebase
    // we are using single queue (thus single GPU)
    // change this later
    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    // Change camera if requested
    if(cameraTransform.has_value())
    {
        totalIterationCount = 0;
        curCamTransformOverride = cameraTransform;
        cameraTransform = std::nullopt;
        std::fill(tilePathCounts.begin(), tilePathCounts.end(), 0u);
        std::fill(tileSPPs.begin(), tileSPPs.end(), 0u);
        totalDeadRayCount = 0;

        ResetAllPaths(processQueue);
    }

    // Do the rendering
    bool isSingleTile = imageTiler.TileCount().Multiply() == 1;
    if(renderMode == RenderMode::E::THROUGHPUT &&
       isSingleTile)
    {
        if(burstSize == 1)
            return DoThroughputSingleTileRender(device, processQueue);
        else
            return DoLatencyRender(burstSize, device, processQueue);
    }
    else if(renderMode == RenderMode::E::THROUGHPUT &&
            !isSingleTile)
    {
        return DoLatencyRender(burstSize, device, processQueue);
    }
    else
    {
        return DoLatencyRender(1u, device, processQueue);
    }
}

