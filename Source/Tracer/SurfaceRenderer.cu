
#include "SurfaceRenderer.h"

#include "Core/Timer.h"
#include "Core/Tuple.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"
#include "Device/GPUAlgBinaryPartition.h"

#include "RendererCommon.h"

struct IsValidRayFunctor
{
    private:
    MR_PF_DECL
    static bool AllNaN(const Vector3& v) noexcept
    {
        return (v[0] != v[0] &&
                v[1] != v[1] &&
                v[2] != v[2]);
    }

    Span<const RayGMem> dRays;

    public:
    MRAY_HOST inline
    IsValidRayFunctor(Span<const RayGMem> dRaysIn)
        : dRays(dRaysIn)
    {}

    MR_PF_DECL
    bool operator()(RayIndex i) const noexcept
    {
        RayGMem r = dRays[i];
        return !(AllNaN(r.dir) && AllNaN(r.pos) &&
                 r.tMin == std::numeric_limits<Float>::infinity() &&
                 r.tMax == std::numeric_limits<Float>::infinity());
    }
};

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCMemsetInvalidRays(MRAY_GRID_CONSTANT const Span<RayGMem> dRays)
{
    KernelCallParams kp;
    uint32_t rayCount = static_cast<uint32_t>(dRays.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayGMem r
        {
            .pos = Vector3(std::numeric_limits<Float>::quiet_NaN()),
            .tMin = std::numeric_limits<Float>::infinity(),
            .dir = Vector3(std::numeric_limits<Float>::quiet_NaN()),
            .tMax = std::numeric_limits<Float>::infinity()
        };
        dRays[i] = r;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCIsVisibleToSpectrum(MRAY_GRID_CONSTANT const Span<Spectrum> dOutputData,
                           MRAY_GRID_CONSTANT const Bitspan<const uint32_t> dIsVisibleBuffer,
                           MRAY_GRID_CONSTANT const Span<const uint32_t> dIndices)
{
    assert(dIsVisibleBuffer.Size() >= dIndices.size());

    KernelCallParams kp;
    uint32_t rayCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t index = dIndices[i];
        // Mask out the not visible rays
        if(!dIsVisibleBuffer[index])
            dOutputData[index] = Spectrum::Zero();
    }
}

SurfaceRenderer::SurfaceRenderer(const RenderImagePtr& rb,
                                 TracerView tv,
                                 ThreadPool& tp,
                                 const GPUSystem& s,
                                 const RenderWorkPack& wp)
    : RendererBase(rb, wp, tv, s, tp, TypeName())
    , rayPartitioner(s)
    , rendererGlobalMem(s.AllGPUs(), 32_MiB, 512_MiB)
    , saveImage(true)
{}

typename SurfaceRenderer::AttribInfoList
SurfaceRenderer::AttributeInfo() const
{
    return StaticAttributeInfo();
}

RendererOptionPack SurfaceRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));

    std::string_view curModeName = SurfRDetail::Mode::ToString(currentOptions.mode);
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{},
                                              curModeName.size()));
    auto svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == curModeName.size());
    std::copy(curModeName.cbegin(), curModeName.cend(), svRead.begin());

    result.attributes.push_back(TransientData(std::in_place_type_t<bool>{}, 1));
    result.attributes.back().Push(Span<const bool>(&currentOptions.doStochasticFilter, 1));

    result.attributes.push_back(TransientData(std::in_place_type_t<Float>{}, 1));
    result.attributes.back().Push(Span<const Float>(&currentOptions.tMaxAORatio, 1));

    if constexpr(MRAY_IS_DEBUG)
    {
        for([[maybe_unused]] const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void SurfaceRenderer::PushAttribute(uint32_t attributeIndex,
                                    TransientData data, const GPUQueue&)
{
    switch(attributeIndex)
    {
        case 0: newOptions.totalSPP = data.AccessAs<uint32_t>()[0]; break;
        case 1: newOptions.mode = SurfRDetail::Mode::FromString(std::as_const(data).AccessAsString()); break;
        case 2: newOptions.doStochasticFilter = data.AccessAs<bool>()[0]; break;
        case 3: newOptions.tMaxAORatio = data.AccessAs<Float>()[0]; break;
        default:
            throw MRayError("{} Unknown attribute index {}", TypeName(), attributeIndex);
    }
}

uint32_t SurfaceRenderer::FindMaxSamplePerIteration(uint32_t rayCount, SurfRDetail::Mode::E mode,
                                                    bool doStochasticFilter)
{
    using enum SurfRDetail::Mode::E;
    uint32_t camSample = (doStochasticFilter)
        ? curCamWork->StochasticFilterSampleRayRNList().TotalRNCount()
        : curCamWork->SampleRayRNList().TotalRNCount();

    uint32_t maxSample = camSample;
    if(mode == AO)
        maxSample = Math::Max(maxSample, 2u);
    else if(mode == FURNACE)
    {
        maxSample = std::transform_reduce
        (
            currentWorks.cbegin(), currentWorks.cend(), maxSample,
            [](uint32_t l, uint32_t r) -> uint32_t
            {
                return Math::Max(l, r);
            },
            [](const auto& renderWorkStruct) -> uint32_t
            {
                return renderWorkStruct.workPtr->SampleRNList(1).TotalRNCount();
            }
        );
    }
    return rayCount * maxSample;
}

RenderBufferInfo SurfaceRenderer::StartRender(const RenderImageParams& rIP,
                                              CamSurfaceId camSurfId,
                                              uint32_t customLogicIndex0,
                                              uint32_t)
{
    using namespace SurfRDetail;
    // TODO: These may be  common operations, every renderer
    // does this move to a templated intermediate class
    // on the inheritance chain
    cameraTransform = std::nullopt;
    curCamTransformOverride = std::nullopt;
    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    currentOptions = newOptions;
    anchorMode = currentOptions.mode;
    totalIterationCount = 0;
    std::fill(tilePathCounts.begin(),
              tilePathCounts.end(), 0u);

    // Generate the Filter
    auto FilterGen = tracerView.filterGenerators.at(tracerView.tracerParams.filmFilter.type);
    if(!FilterGen)
        throw MRayError("[{}]: Unknown film filter type {}.", TypeName(),
                        uint32_t(tracerView.tracerParams.filmFilter.type));
    Float radius = tracerView.tracerParams.filmFilter.radius;
    filmFilter = FilterGen->get()(gpuSystem, Float(radius));
    // Change the mode according to the render logic
    using Math::Roll;
    int32_t modeIndex = (int32_t(anchorMode) +
                         int32_t(customLogicIndex0));
    uint32_t sendMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                      int32_t(Mode::END)));
    uint32_t newMode = uint32_t(Roll(modeIndex, 0, int32_t(Mode::END)));
    currentOptions.mode = SurfRDetail::Mode::E(newMode);

    imageTiler = ImageTiler(renderBuffer.get(), rIP,
                            tracerView.tracerParams.parallelizationHint,
                            Vector2ui::Zero());
    tilePathCounts.resize(imageTiler.TileCount().Multiply(), 0u);

    // Generate Works to get the total work count
    // We will batch allocate
    uint32_t totalWorkCount = GenerateWorks();

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
                        TypeName(), uint32_t(camSurfId));
    curCamSurfaceParams = surfLoc->second;
    // Find the transform/camera work for this specific surface
    curCamKey = std::bit_cast<CameraKey>(curCamSurfaceParams.cameraId);
    curCamTransformKey = std::bit_cast<TransformKey>(curCamSurfaceParams.transformId);
    CameraGroupId camGroupId = CameraGroupId(curCamKey.FetchBatchPortion());
    TransGroupId transGroupId = TransGroupId(curCamTransformKey.FetchBatchPortion());
    auto packLoc = std::find_if(currentCameraWorks.cbegin(), currentCameraWorks.cend(),
    [camGroupId, transGroupId](const auto& pack)
    {
        return (pack.cgId == camGroupId && pack.tgId == transGroupId);
    });
    assert(packLoc != currentCameraWorks.cend());
    curCamWork = packLoc->workPtr.get();

    // Allocate the ray state buffers
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    // Find the ray count (1spp per tile)
    uint32_t maxRayCount = imageTiler.ConservativeTileSize().Multiply();
    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount, currentOptions.mode,
                                                        currentOptions.doStochasticFilter);
    if(currentOptions.mode == SurfRDetail::Mode::AO)
    {
        uint32_t isVisibleIntCount = Bitspan<uint32_t>::CountT(maxRayCount);
        MemAlloc::AllocateMultiData(Tie(dHits, dHitKeys,
                                        dRays, dRayCones,
                                        dRayStateAO.dVisibilityRays,
                                        dRayStateAO.dImageCoordinates,
                                        dRayStateAO.dOutputData,
                                        dRayStateAO.dFilmFilterWeights,
                                        dIsVisibleBuffer,
                                        dRandomNumBuffer,
                                        dWorkHashes, dWorkBatchIds,
                                        dSubCameraBuffer),
                                    rendererGlobalMem,
                                    {maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     isVisibleIntCount, maxSampleCount,
                                     totalWorkCount, totalWorkCount,
                                     SUB_CAMERA_BUFFER_SIZE});
        dRayStateCommon.dImageCoordinates = dRayStateAO.dImageCoordinates;
        dRayStateCommon.dOutputData = dRayStateAO.dOutputData;
        dRayStateCommon.dFilmFilterWeights = dRayStateAO.dFilmFilterWeights;
    }
    else
    {
        MemAlloc::AllocateMultiData(Tie(dHits, dHitKeys,
                                        dRays, dRayCones,
                                        dRayStateCommon.dImageCoordinates,
                                        dRayStateCommon.dOutputData,
                                        dRayStateCommon.dFilmFilterWeights,
                                        dRandomNumBuffer,
                                        dWorkHashes, dWorkBatchIds,
                                        dSubCameraBuffer),
                                    rendererGlobalMem,
                                    {maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxSampleCount,
                                     totalWorkCount, totalWorkCount,
                                     SUB_CAMERA_BUFFER_SIZE});

        dRayStateAO.dImageCoordinates = dRayStateCommon.dImageCoordinates;
        dRayStateAO.dOutputData = dRayStateCommon.dOutputData;
        dRayStateAO.dFilmFilterWeights = dRayStateCommon.dFilmFilterWeights;
    }

    // And initialize the hashes
    workHasher = InitializeHashes(dWorkHashes, dWorkBatchIds,
                                  maxRayCount, queue);

    // Initialize ray partitioner with worst case scenario,
    // All work types are used. (We do not use camera work
    // for this type of renderer)
    uint32_t maxWorkCount = uint32_t(currentWorks.size() +
                                     currentLightWorks.size());
    rayPartitioner = RayPartitioner(gpuSystem, maxRayCount,
                                    maxWorkCount);

    // Also allocate for the partitioner inside the
    // base accelerator (This should not allocate for HW accelerators)
    tracerView.baseAccelerator.AllocateForTraversal(maxRayCount);

    // Calculate tMax for ambient occlusion
    curTMaxAO = Math::Length(tracerView.baseAccelerator.SceneAABB().GeomSpan());
    curTMaxAO *= currentOptions.tMaxAORatio;

    // Finally generate RNG
    auto RngGen = tracerView.rngGenerators.at(tracerView.tracerParams.samplerType.e);
    if(!RngGen)
        throw MRayError("[{}]: Unknown random number generator type {}.", TypeName(),
                        uint32_t(tracerView.tracerParams.samplerType.e));

    Vector2ui maxDeviceLocalRNGCount = this->imageTiler.ConservativeTileSize();
    uint64_t seed = tracerView.tracerParams.seed;
    uint32_t spp = currentOptions.totalSPP;
    rnGenerator = RngGen->get()(rIP,
                                std::move(maxDeviceLocalRNGCount),
                                std::move(spp), std::move(seed),
                                gpuSystem, globalThreadPool);

    auto bufferPtrAndSize = renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = curColorSpace,
        .resolution = imageTiler.FullResolution(),
        .curRenderLogic0 = sendMode,
        .curRenderLogic1 = std::numeric_limits<uint32_t>::max()
    };
}

RendererOutput SurfaceRenderer::DoRender()
{
    static const auto annotation = gpuSystem.CreateAnnotation("Render Frame");
    const auto _ = annotation.AnnotateScope();

    // On each iteration do one tile fully,
    // so we can send it directly.
    // TODO: Like many places of this codebase
    // we are using single queue (thus single GPU)
    // change this later
    Timer timer; timer.Start();
    const auto& cameraWork = *curCamWork;
    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    if(cameraTransform.has_value())
    {
        totalIterationCount = 0;
        curCamTransformOverride = cameraTransform;
        cameraTransform = std::nullopt;
        std::fill(tilePathCounts.begin(),
                  tilePathCounts.end(), 0u);
        saveImage = false;
    }

    // Generate subcamera of this specific tile
    cameraWork.GenerateSubCamera
    (
        dSubCameraBuffer,
        curCamKey, curCamTransformOverride,
        imageTiler.CurrentTileIndex(),
        imageTiler.TileCount(),
        processQueue
    );

    // Find the ray count. Ray count is tile count
    // but tile can exceed film boundaries so clamp,
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();

    // Start the partitioner, again worst case work count
    // Get the K/V pair buffer
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto[dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                 processQueue, true);

    // Iota the indices
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);

    // Create RNG state for each ray
    // Generate rays
    rnGenerator->SetupRange(this->imageTiler.LocalTileStart(),
                            this->imageTiler.LocalTileEnd(),
                            processQueue);
    // Generate RN for camera rays
    RNRequestList camSamplePerRayList = currentOptions.doStochasticFilter
                        ? curCamWork->SampleRayRNList()
                        : curCamWork->StochasticFilterSampleRayRNList();
    uint32_t camRayGenRNCount = rayCount * camSamplePerRayList.TotalRNCount();
    auto dCamRayGenRNBuffer = dRandomNumBuffer.subspan(0, camRayGenRNCount);
    rnGenerator->IncrementSampleId(processQueue);
    rnGenerator->GenerateNumbers(dCamRayGenRNBuffer, 0,
                                 camSamplePerRayList, processQueue);

    uint64_t& tilePixIndex = tilePathCounts[imageTiler.CurrentTileIndex1D()];
    if(currentOptions.doStochasticFilter)
    {
        cameraWork.GenRaysStochasticFilter
        (
            dRayCones, dRays,
            dRayStateCommon.dImageCoordinates,
            dRayStateCommon.dFilmFilterWeights, dIndices,
            ToConstSpan(dCamRayGenRNBuffer),
            dSubCameraBuffer, curCamTransformKey,
            tilePixIndex, imageTiler.CurrentTileSize(),
            tracerView.tracerParams.filmFilter,
            processQueue
        );
    }
    else
    {
        cameraWork.GenerateRays
        (
            dRayCones, dRays,
            dRayStateCommon.dImageCoordinates,
            dRayStateCommon.dFilmFilterWeights, dIndices,
            ToConstSpan(dCamRayGenRNBuffer),
            dSubCameraBuffer, curCamTransformKey,
            tilePixIndex, imageTiler.CurrentTileSize(),
            processQueue
        );
    }
    tilePixIndex += rayCount;

    // Cast rays
    using namespace std::string_view_literals;
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    auto dHitKeysLocal = dHitKeys.subspan(0, rayCount);
    processQueue.IssueWorkKernel<KCSetBoundaryWorkKeys>
    (
        "KCSetBoundaryWorkKeys"sv,
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dHitKeysLocal.size())},
        dHitKeysLocal,
        boundaryLightKeyPack
    );

    // Ray Casting
    tracerView.baseAccelerator.CastRays
    (
        dHitKeysLocal, dHits, dBackupRNGStates,
        dRays, dIndices, processQueue
    );

    // Generate work keys from hit packs
    using namespace std::string_literals;
    static const std::string GenWorkKernelName = std::string(TypeName()) + "-KCGenerateWorkKeys"s;
    processQueue.IssueWorkKernel<KCGenerateWorkKeys>
    (
        GenWorkKernelName,
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dHitKeysLocal.size())},
        dKeys,
        ToConstSpan(dHitKeysLocal),
        workHasher
    );

    // Finally, partition using the generated keys.
    // Fully partition here using single sort
    auto
    [
        hPartitionCount,
        isHostVisible,
        hPartitionStartOffsets,
        hPartitionKeys,
        dPartitionIndices,
        dPartitionKeys
    ] = rayPartitioner.MultiPartition(dKeys, dIndices,
                                      workHasher.WorkBatchDataRange(),
                                      workHasher.WorkBatchBitRange(),
                                      processQueue, false);
    assert(isHostVisible);
    // Wait for results to be available in host buffers
    processQueue.Barrier().Wait();

    SurfRDetail::Mode::E curMode = currentOptions.mode;
    if(curMode == SurfRDetail::Mode::AO)
    {
        Span<RayGMem> dVisibilityRays = dRayStateAO.dVisibilityRays;
        processQueue.IssueWorkKernel<KCMemsetInvalidRays>
        (
            "KCSetInvalidRays",
            DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dVisibilityRays.size())},
            dVisibilityRays
        );
    }

    SurfRDetail::GlobalState globalState =
    {
        .mode = SurfRDetail::Mode(curMode),
        .tMaxAO = curTMaxAO
    };
    for(uint32_t i = 0; i < hPartitionCount[0]; i++)
    {
        uint32_t partitionStart = hPartitionStartOffsets[i];
        uint32_t partitionSize = (hPartitionStartOffsets[i + 1] -
                                  hPartitionStartOffsets[i]);
        auto dLocalIndices = dPartitionIndices.subspan(partitionStart,
                                                       partitionSize);

        // Find the work
        // TODO: Although work count should be small,
        // doing a linear search here may not be performant.
        CommonKey key = workHasher.BisectBatchPortion(hPartitionKeys[i]);
        auto wLoc = std::find_if(currentWorks.cbegin(), currentWorks.cend(),
        [key](const auto& workInfo)
        {
            return workInfo.workGroupId == key;
        });
        auto lightWLoc = std::find_if(currentLightWorks.cbegin(), currentLightWorks.cend(),
        [key](const auto& workInfo)
        {
            return workInfo.workGroupId == key;
        });
        if(wLoc != currentWorks.cend())
        {
            if(currentOptions.mode == SurfRDetail::Mode::AO ||
               currentOptions.mode == SurfRDetail::Mode::FURNACE)
            {
                using enum SurfRDetail::Mode::E;
                const auto& workI = UpcastRenderWork<SurfaceRenderer>(wLoc->workPtr);

                RNRequestList rnCountList = (currentOptions.mode == AO)
                                                ? GenRNRequestList<2>()
                                                : workI.SampleRNList(1);
                uint32_t rnCount = rnCountList.TotalRNCount();
                uint32_t rnStart = camSamplePerRayList.TotalRNCount();

                auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
                if(currentOptions.mode == SurfRDetail::Mode::AO ||
                   currentOptions.mode == SurfRDetail::Mode::FURNACE)
                {
                    rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                         uint16_t(rnStart), rnCountList,
                                                         processQueue);
                }

                workI.DoWork_1(dRayStateAO,
                               dLocalIndices,
                               dRandomNumBuffer,
                               dRayCones,
                               dRays,
                               dHits,
                               dHitKeys,
                               globalState,
                               processQueue);
            }
            else
            {
                const auto& workI = UpcastRenderWork<SurfaceRenderer>(wLoc->workPtr);
                workI.DoWork_0(dRayStateCommon,
                               dLocalIndices,
                               Span<const RandomNumber>{},
                               dRayCones,
                               dRays,
                               dHits,
                               dHitKeys,
                               globalState,
                               processQueue);
            }

        }
        else if(lightWLoc != currentLightWorks.cend())
        {
            const auto& workI = UpcastRenderLightWork<SurfaceRenderer>(lightWLoc->workPtr);
            workI.DoBoundaryWork_0(dRayStateCommon,
                                   dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dRayCones,
                                   dRays,
                                   dHits,
                                   dHitKeys,
                                   globalState,
                                   processQueue);
        }
        else throw MRayError("[{}]: Unknown work id is found ({}).",
                             TypeName(), key);

    }

    // Do shadow ray cast
    if(currentOptions.mode == SurfRDetail::Mode::AO)
    {
        Span<RayGMem> dVisibilityRays = dRayStateAO.dVisibilityRays;
        auto p = rayPartitioner.BinaryPartition(dPartitionIndices, processQueue,
                                                IsValidRayFunctor(dVisibilityRays));
        processQueue.Barrier().Wait();

        auto dValidIndices = p.dPartitionIndices.subspan(p.hPartitionStartOffsets[0],
                                                         p.hPartitionStartOffsets[1] - p.hPartitionStartOffsets[0]);

        if(!dValidIndices.empty())
        {
            // Ray Casting
            Bitspan<uint32_t> dIsVisibleBitSpan(dIsVisibleBuffer);
            tracerView.baseAccelerator.CastVisibilityRays
            (
                dIsVisibleBitSpan, dBackupRNGStates,
                dVisibilityRays, dValidIndices, processQueue
            );

            // Write either one or zero
            processQueue.IssueWorkKernel<KCIsVisibleToSpectrum>
            (
                "KCIsVisibleToSpectrum",
                DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dValidIndices.size())},
                dRayStateAO.dOutputData,
                ToConstSpan(dIsVisibleBitSpan),
                dValidIndices
            );
        }
    }
    // Filter the samples
    // Wait for previous copy to finish
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    renderBuffer->ClearImage(processQueue);
    ImageSpan filmSpan = imageTiler.GetTileSpan();
    if(currentOptions.doStochasticFilter)
    {
        SetImagePixels
        (
            filmSpan,
            ToConstSpan(dRayStateCommon.dOutputData),
            ToConstSpan(dRayStateCommon.dFilmFilterWeights),
            ToConstSpan(dRayStateCommon.dImageCoordinates),
            Float(1), processQueue
        );
    }
    else
    {
        // Old code used partition based filtering. It was slow,
        // invalidated ray partitioner (to save memory we shared
        // ray partitioner between the renderer and filter).
        //
        // Filter that uses atomic is faster, do not use any extra
        // memory. It can be used
        //
        //
        // Please note that ray partitioner will be invalidated here.
        // In this case, we do not use the partitioner anymore
        // so its fine.

        // Using atomic filter since the samples are uniformly distributed
        // And it is faster
        filmFilter->ReconstructionFilterAtomicRGB
        (
            filmSpan,
            ToConstSpan(dRayStateCommon.dOutputData),
            ToConstSpan(dRayStateCommon.dImageCoordinates),
            Float(1), processQueue
        );
    }
    // Issue a send of the FBO to Visor
    const GPUQueue& transferQueue = device.GetTransferQueue();
    Optional<RenderImageSection>
    renderOut = imageTiler.TransferToHost(processQueue,
                                          transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value())
        return RendererOutput{};
    // Actual global weight
    renderOut->globalWeight = Float(1);

    // We do not need to wait here, but we time
    // from CPU side so we need to wait
    // TODO: In future we should do OpenGL, Vulkan
    // style performance counters events etc. to
    // query the timing (may be couple of frame before even)
    // The timing is just a general performance indicator
    // It should not be super accurate.
    processQueue.Barrier().Wait();
    timer.Split();

    double timeSec = timer.Elapsed<Second>();
    double samplePerSec = static_cast<double>(rayCount) / timeSec;
    samplePerSec /= 1'000'000;
    double spp = double(1) / double(imageTiler.TileCount().Multiply());
    totalIterationCount++;
    spp *= static_cast<double>(totalIterationCount);
    // Roll to the next tile
    imageTiler.NextTile();

    bool triggerSave = saveImage &&
        (currentOptions.totalSPP * imageTiler.TileCount().Multiply() ==
         totalIterationCount);
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            samplePerSec,
            "M samples/s",
            spp,
            double(currentOptions.totalSPP),
            "spp",
            float(timer.Elapsed<Millisecond>()),
            imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            static_cast<uint32_t>(SurfRDetail::Mode::END),
            0
        },
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

void SurfaceRenderer::StopRender()
{
    ClearAllWorkMappings();
    filmFilter = {};
    rnGenerator = {};
    std::fill(tilePathCounts.begin(),
              tilePathCounts.end(), 0u);
}

std::string_view SurfaceRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "Surface"sv;
    return RendererTypeName<Name>;
}

typename SurfaceRenderer::AttribInfoList
SurfaceRenderer::StaticAttributeInfo()
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP",            MRayDataTypeRT(MR_UINT32),  IS_SCALAR, MR_MANDATORY},
        {"renderType",          MRayDataTypeRT(MR_STRING),  IS_SCALAR, MR_MANDATORY},
        {"doStochasticFilter",  MRayDataTypeRT(MR_BOOL),    IS_SCALAR, MR_MANDATORY},
        {"tMaxAORatio",         MRayDataTypeRT(MR_FLOAT),   IS_SCALAR, MR_MANDATORY}
    };
}

size_t SurfaceRenderer::GPUMemoryUsage() const
{
    return (rayPartitioner.GPUMemoryUsage() +
            rnGenerator->GPUMemoryUsage() +
            rendererGlobalMem.Size());
}