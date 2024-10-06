#pragma once

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCInitPathState(MRAY_GRID_CONSTANT const PathTraceRDetail::RayState dRayState,
                            MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    using namespace PathTraceRDetail;
    KernelCallParams kp;
    uint32_t pathCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < pathCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        dRayState.dPathRadiance[index] = Spectrum::Zero();
        // dImageCoordinates is set by cam ray gen
        // dFilmFilterWeights is set by cam ray gen
        dRayState.dThroughput[index] = Spectrum(1);
        dRayState.dPathDataPack[index] = PathDataPack
        {
            .depth = 0,
            .status = PathStatus(uint8_t(0)),
            .type = RayType::CAMERA_RAY
        };
        // dOutRays will be set when ray bounces
        // dOutRayDiffs will be set when ray bounces
        // dPrevMaterialPDF is set when MIS is enabled
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCCopyRays(MRAY_GRID_CONSTANT const Span<RayGMem> dRaysOut,
                       MRAY_GRID_CONSTANT const Span<RayDiff> dRayDiffOut,
                       MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                       MRAY_GRID_CONSTANT const Span<const RayGMem> dRaysIn,
                       MRAY_GRID_CONSTANT const Span<const RayDiff> dRayDiffIn)
{
    using namespace PathTraceRDetail;
    KernelCallParams kp;
    uint32_t pathCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < pathCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        dRaysOut[index] = dRaysIn[index];
        dRayDiffOut[index] = dRayDiffIn[index];
    }
}

struct SetPathToDeadFunctor
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    PathTraceRDetail::PathDataPack operator()(PathTraceRDetail::PathDataPack s) const
    {
        using namespace PathTraceRDetail;
        s.status.Set(uint32_t(PathStatusEnum::DEAD));
        return s;
    }
};

class IsDeadFunctor
{
    Span<PathTraceRDetail::PathDataPack> dPathDataPack;

    public:
    IsDeadFunctor(Span<PathTraceRDetail::PathDataPack> dPathDataPackIn)
        : dPathDataPack(dPathDataPackIn)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    bool operator()(RayIndex index)
    {
        using namespace PathTraceRDetail;
        const PathStatus state = dPathDataPack[index].status;
        return state[uint32_t(PathStatusEnum::DEAD)];
    }
};

template<class MLA>
PathTracerRenderer<MLA>::PathTracerRenderer(const RenderImagePtr& rb,
                                            TracerView tv,
                                            BS::thread_pool& tp,
                                            const GPUSystem& s,
                                            const RenderWorkPack& wp)
    : RendererT<PathTracerRenderer<MLA>>(rb, wp, tv, s, tp)
    , rayPartitioner(s)
    , redererGlobalMem(s.AllGPUs(), 128_MiB, 512_MiB)
    , metaLightArray(s)
{}

template<class MLA>
typename PathTracerRenderer<MLA>::AttribInfoList
PathTracerRenderer<MLA>::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP",        MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY},
        {"sampleMode",      MRayDataType<MR_STRING>{}, IS_SCALAR, MR_MANDATORY},
        {"rrRange",         MRayDataType<MR_VECTOR_2UI>{}, IS_SCALAR, MR_MANDATORY},
        {"neeSamplerType",  MRayDataType<MR_STRING>{}, IS_SCALAR, MR_MANDATORY}
    };
}

template<class MLA>
RendererOptionPack PathTracerRenderer<MLA>::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));

    std::string_view curModeName = currentOptions.sampleMode.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string>{},
                                              curModeName.size()));
    auto svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == curModeName.size());
    std::copy(curModeName.cbegin(), curModeName.cend(), svRead.begin());

    result.attributes.push_back(TransientData(std::in_place_type_t<Vector2>{}, 1));
    result.attributes.back().Push(Span<const Vector2ui>(&currentOptions.russianRouletteRange, 1));


    std::string_view lightSamplerName = currentOptions.lightSampler.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string>{}, lightSamplerName.size()));
    svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == lightSamplerName.size());
    std::copy(lightSamplerName.cbegin(), lightSamplerName.cend(), svRead.begin());

    if constexpr(MRAY_IS_DEBUG)
    {
        for(const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

template<class MLA>
void PathTracerRenderer<MLA>::PushAttribute(uint32_t attributeIndex,
                                            TransientData data, const GPUQueue&)
{    switch(attributeIndex)
    {
        case 0: newOptions.totalSPP = data.AccessAs<uint32_t>()[0]; break;
        case 1: newOptions.sampleMode = PathTraceRDetail::SampleMode(std::as_const(data).AccessAsString()); break;
        case 2: newOptions.russianRouletteRange = data.AccessAs<Vector2ui>()[0]; break;
        case 3: newOptions.lightSampler = PathTraceRDetail::LightSamplerType(std::as_const(data).AccessAsString()); break;
        default:
            throw MRayError("{} Unkown attribute index {}", TypeName(), attributeIndex);
    }
}

template<class MLA>
uint32_t PathTracerRenderer<MLA>::FindMaxSamplePerIteration(uint32_t rayCount,
                                                            PathTraceRDetail::SampleMode sampleMode)
{
    using enum PathTraceRDetail::SampleMode::E;
    uint32_t camSample = (*curCamWork)->StochasticFilterSampleRayRNCount();

    uint32_t maxSample = camSample;
    maxSample = std::transform_reduce
    (
        this->currentWorks.cbegin(), this->currentWorks.cend(), maxSample,
        [](uint32_t l, uint32_t r) -> uint32_t
        {
            return std::max(l, r);
        },
        [sampleMode](const auto& renderWorkStruct) -> uint32_t
        {
            uint32_t workIndex = (sampleMode == PURE) ? 0 : 1;
            return renderWorkStruct.workPtr->SampleRNCount(workIndex);
        }
    );
    return rayCount * maxSample;
}

template<class MLA>
Span<RayIndex> PathTracerRenderer<MLA>::ReloadPaths(Span<const RayIndex> dIndices,
                                                    const GPUQueue& processQueue)
{
    // RELOADING!!!
    // Find the dead rays
    auto [hDeadRayRanges, dDeadAliveRayIndices] = rayPartitioner.BinaryPartition
    (
        dIndices, processQueue,
        IsDeadFunctor(dRayState.dPathDataPack)
    );
    processQueue.Barrier().Wait();

    // Generate RN for camera rays
    uint32_t camSamplePerRay = (*curCamWork)->StochasticFilterSampleRayRNCount();
    uint32_t deadRayCount = hDeadRayRanges[1] - hDeadRayRanges[0];
    auto dDeadRayIndices = dDeadAliveRayIndices.subspan(hDeadRayRanges[0],
                                                        deadRayCount);
    uint32_t camRayGenRNCount = uint32_t(dDeadRayIndices.size()) * camSamplePerRay;
    auto dCamRayGenRNBuffer = dRandomNumBuffer.subspan(0, camRayGenRNCount);
    if(!dDeadRayIndices.empty())
    {
        rnGenerator->GenerateNumbersIndirect(dCamRayGenRNBuffer,
                                             ToConstSpan(dDeadRayIndices),
                                             Vector2ui(0, camSamplePerRay),
                                             processQueue);
        //
        const auto& cameraWork = (*curCamWork->get());
        cameraWork.GenRaysStochasticFilter
        (
            dRayDifferentials, dRays,
            dRayState.dImageCoordinates,
            dRayState.dFilmFilterWeights,
            ToConstSpan(dDeadRayIndices),
            ToConstSpan(dCamRayGenRNBuffer),
            dSubCameraBuffer, curCamTransformKey,
            globalPixelIndex,
            this->imageTiler.CurrentTileSize(),
            this->tracerView.tracerParams.filmFilter,
            processQueue
        );
        globalPixelIndex += dDeadRayIndices.size();

        // Initialize the state of new rays
        processQueue.IssueSaturatingKernel<KCInitPathState>
        (
            "KCInitPathState",
            KernelIssueParams{.workCount = deadRayCount},
            dRayState,
            dDeadRayIndices
        );
    }
    // Index buffer may be invalidated (Binary partition should not
    // invalidate but lets return the new buffer)
    return dDeadAliveRayIndices;
}

template<class MLA>
void PathTracerRenderer<MLA>::ResetAllPaths(const GPUQueue& queue)
{
    // Set all paths as dead (we just started)
    using namespace PathTraceRDetail;
    DeviceAlgorithms::InPlaceTransform(dRayState.dPathDataPack, queue,
                                       SetPathToDeadFunctor());
    queue.MemsetAsync(dPathRNGDimensions, 0x00);
}

template<class MLA>
RenderBufferInfo PathTracerRenderer<MLA>::StartRender(const RenderImageParams& rIP,
                                                      CamSurfaceId camSurfId,
                                                      uint32_t customLogicIndex0,
                                                      uint32_t)
{
    // TODO: These may be  common operations, every renderer
    // does this move to a templated intermediate class
    // on the inheritance chain
    this->cameraTransform = std::nullopt;
    curCamTransformOverride = std::nullopt;
    this->curColorSpace = this->tracerView.tracerParams.globalTextureColorSpace;
    currentOptions = newOptions;
    anchorSampleMode = currentOptions.sampleMode;
    this->totalIterationCount = 0;
    globalPixelIndex = 0;
    totalDeadRayCount = 0;

    // Generate the Filter
    auto FilterGen = this->tracerView.filterGenerators.at(this->tracerView.tracerParams.filmFilter.type);
    if(!FilterGen)
        throw MRayError("[{}]: Unkown film filter type {}.", TypeName(),
                        uint32_t(this->tracerView.tracerParams.filmFilter.type));
    Float radius = this->tracerView.tracerParams.filmFilter.radius;
    filmFilter = FilterGen->get()(this->gpuSystem, Float(radius));
    Vector2ui filterPadSize = filmFilter->FilterExtent();

    // Change the mode according to the render logic
    using Math::Roll;
    int32_t modeIndex = (int32_t(SampleMode::E(anchorSampleMode)) +
                         int32_t(customLogicIndex0));
    uint32_t sendMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                      int32_t(SampleMode::E::END)));
    uint32_t newMode = uint32_t(Roll(modeIndex, 0, int32_t(SampleMode::E::END)));
    currentOptions.sampleMode = SampleMode::E(newMode);

    this->imageTiler = ImageTiler(this->renderBuffer.get(), rIP,
                                  this->tracerView.tracerParams.parallelizationHint,
                                  Vector2ui::Zero(), 3, 1);

    // Generate Works to get the total work count
    // We will batch allocate
    uint32_t totalWorkCount = this->GenerateWorks();

    // Find camera surface and get keys work instance for that
    // camera etc.
    auto surfLoc = std::find_if
    (
        this->tracerView.camSurfs.cbegin(),
        this->tracerView.camSurfs.cend(),
        [camSurfId](const auto& pair)
        {
            return pair.first == camSurfId;
        }
    );
    if(surfLoc == this->tracerView.camSurfs.cend())
        throw MRayError("[{:s}]: Unkown camera surface id ({:d})",
                        TypeName(), uint32_t(camSurfId));
    curCamSurfaceParams = surfLoc->second;
    // Find the transform/camera work for this specific surface
    curCamKey = std::bit_cast<CameraKey>(curCamSurfaceParams.cameraId);
    curCamTransformKey = std::bit_cast<TransformKey>(curCamSurfaceParams.transformId);
    CameraGroupId camGroupId = CameraGroupId(curCamKey.FetchBatchPortion());
    TransGroupId transGroupId = TransGroupId(curCamTransformKey.FetchBatchPortion());
    auto packLoc = std::find_if
    (
        this->currentCameraWorks.cbegin(), this->currentCameraWorks.cend(),
        [camGroupId, transGroupId](const auto& pack)
        {
            return pack.idPack == Pair(camGroupId, transGroupId);
        }
    );
    curCamWork = &packLoc->workPtr;

    // Allocate the ray state buffers
    const GPUQueue& queue = this->gpuSystem.BestDevice().GetComputeQueue(0);
    // Find the ray count (1spp per tile)
    uint32_t maxRayCount = this->imageTiler.ConservativeTileSize().Multiply();
    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount, currentOptions.sampleMode);
    if(currentOptions.sampleMode == SampleMode::E::PURE)
    {
        MemAlloc::AllocateMultiData(std::tie(dHits, dHitKeys,
                                             dRays, dRayDifferentials,
                                             dRayState.dPathRadiance,
                                             dRayState.dImageCoordinates,
                                             dRayState.dFilmFilterWeights,
                                             dRayState.dThroughput,
                                             dRayState.dPathDataPack,
                                             dRayState.dOutRays,
                                             dRayState.dOutRayDiffs,
                                             dPathRNGDimensions,
                                             dRandomNumBuffer,
                                             dWorkHashes, dWorkBatchIds,
                                             dSubCameraBuffer),
                                    redererGlobalMem,
                                    {maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxSampleCount,
                                     totalWorkCount, totalWorkCount,
                                     this->SUB_CAMERA_BUFFER_SIZE});
    }
    else throw MRayError("Not yet implemented!");

    // And initialze the hashes
    workHasher = this->InitializeHashes(dWorkHashes, dWorkBatchIds,
                                        maxRayCount, queue);

    // Initialize ray partitioner with worst case scenario,
    // All work types are used. (We do not use camera work
    // for this type of renderer)
    uint32_t maxWorkCount = uint32_t(this->currentWorks.size() +
                                     this->currentLightWorks.size());
    rayPartitioner = RayPartitioner(this->gpuSystem, maxRayCount,
                                    maxWorkCount);

    // Also allocate for the partitioner inside the
    // base accelerator (This should not allocate for HW accelerators)
    this->tracerView.baseAccelerator.AllocateForTraversal(maxRayCount);

    // Finally generate RNG
    auto RngGen = this->tracerView.rngGenerators.at(this->tracerView.tracerParams.samplerType.type);
    if(!RngGen)
        throw MRayError("[{}]: Unkown random number generator type {}.", TypeName(),
                        uint32_t(this->tracerView.tracerParams.samplerType.type));
    uint32_t generatorCount = (rIP.regionMax - rIP.regionMin).Multiply();
    uint64_t seed = this->tracerView.tracerParams.seed;
    rnGenerator = RngGen->get()(std::move(generatorCount), std::move(seed),
                                this->gpuSystem, this->globalThreadPool);

    ResetAllPaths(queue);

    auto bufferPtrAndSize = this->renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = this->curColorSpace,
        .resolution = this->imageTiler.FullResolution(),
        .depth = this->renderBuffer->Depth(),
        .curRenderLogic0 = sendMode,
        .curRenderLogic1 = std::numeric_limits<uint32_t>::max()
    };
}

template<class MLA>
RendererOutput PathTracerRenderer<MLA>::DoRender()
{
    static const auto annotation = this->gpuSystem.CreateAnnotation("Render Frame");
    const auto _ = annotation.AnnotateScope();

    // On each iteration do one tile fully,
    // so we can send it directly.
    // TODO: Like many places of this codebase
    // we are using sinlge queue (thus single GPU)
    // change this later
    Timer timer; timer.Start();
    const auto& cameraWork = (*curCamWork->get());
    const GPUDevice& device = this->gpuSystem.BestDevice();
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    if(this->cameraTransform.has_value())
    {
        this->totalIterationCount = 0;
        curCamTransformOverride = this->cameraTransform;
        this->cameraTransform = std::nullopt;
        globalPixelIndex = 0;
        totalDeadRayCount = 0;

        ResetAllPaths(processQueue);
    }

    // Generate subcamera of this specific tile
    cameraWork.GenerateSubCamera
    (
        dSubCameraBuffer,
        curCamKey, curCamTransformOverride,
        this->imageTiler.CurrentTileIndex(),
        this->imageTiler.TileCount(),
        processQueue
    );

    // Find the ray count. Ray count is tile count
    // but tile can exceed film boundaries so clamp,
    uint32_t rayCount = this->imageTiler.CurrentTileSize().Multiply();
    // Start the partitioner, again worst case work count
    // Get the K/V pair buffer
    uint32_t maxWorkCount = uint32_t(this->currentWorks.size() + this->currentLightWorks.size());
    auto[dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount, true);

    // Iota the indices
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
    // Create RNG state for each ray
    rnGenerator->SetupRange(this->imageTiler.Tile1DRange());
    // Reload dead paths with new
    dIndices = ReloadPaths(dIndices, processQueue);

    {
        // Now binary partition the index buffer
        auto [hDeadAliveRanges, dDeadAliveIndices] = rayPartitioner.BinaryPartition
        (
            dIndices, processQueue,
            IsDeadFunctor(dRayState.dPathDataPack)
        );
        processQueue.Barrier().Wait();
        dIndices = dDeadAliveIndices;
    }

    // We refilled the path buffer,
    // and can roll back to the Iota
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);

    // Cast rays
    using namespace std::string_view_literals;
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    processQueue.IssueSaturatingKernel<KCSetBoundaryWorkKeys>
    (
        "KCSetBoundaryWorkKeys"sv,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dHitKeys.size())},
        dHitKeys,
        this->boundaryLightKeyPack
    );
    // Actual Ray Casting
    this->tracerView.baseAccelerator.CastRays
    (
        dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, processQueue
    );

    // Generate work keys from hit packs
    using namespace std::string_literals;
    static const std::string GenWorkKernelName = std::string(TypeName()) + "-KCGenerateWorkKeys"s;
    processQueue.IssueSaturatingKernel<KCGenerateWorkKeys>
    (
        GenWorkKernelName,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dHitKeys.size())},
        dKeys,
        ToConstSpan(dHitKeys),
        workHasher
    );

    // Finally, partition using the generated keys.
    // Fully partition here using single sort
    auto& rp = rayPartitioner;
    auto partitionOutput = rp.MultiPartition(dKeys, dIndices,
                                             workHasher.WorkBatchDataRange(),
                                             workHasher.WorkBatchBitRange(),
                                             processQueue, false);
    // Wait for results to be available in host buffers
    processQueue.Barrier().Wait();
    // Old Indices array (and the key) is invalidated
    // Change indices to the partitioned one
    dIndices = partitionOutput.dPartitionIndices;

    if(currentOptions.sampleMode == SampleMode::E::PURE)
    {
        using GlobalState = PathTraceRDetail::GlobalState<EmptyType>;
        GlobalState globalState
        {
            .russianRouletteRange = currentOptions.russianRouletteRange,
            .sampleMode = currentOptions.sampleMode
        };

        this->IssueWorkKernelsToPartitions(workHasher, partitionOutput,
        [&, this](const auto& workPtr, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t partitionSize)
        {
            uint32_t rnCount = workPtr.SampleRNCount(0);
            auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
            rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                 dPathRNGDimensions, rnCount,
                                                 processQueue);

            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, dLocalIndices, processQueue,
                ConstAddFunctor(rnCount)
            );

            workPtr.DoWork_0(dRayState, dLocalIndices,
                             dRandomNumBuffer, dRayDifferentials,
                             dRays, dHits, dHitKeys,
                             globalState, processQueue);
        },
        //
        [&, this](const auto& workPtr, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            workPtr.DoBoundaryWork_0(dRayState, dLocalIndices,
                                     Span<const RandomNumber>{},
                                     dRayDifferentials, dRays,
                                     dHits, dHitKeys,
                                     globalState, processQueue);
        });
    }


    // Find the dead paths again
    // Every path is processed, so we do not need to use the scambled
    // index buffer. Iota again
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
    // Now binary partition the index buffer
    auto[hDeadAliveRanges, dDeadAliveIndices] = rayPartitioner.BinaryPartition
    (
        dIndices, processQueue,
        IsDeadFunctor(dRayState.dPathDataPack)
    );
    processQueue.Barrier().Wait();

    // Find the alive ranges first, copy the generated next ray stuff
    uint32_t aliveRayCount = hDeadAliveRanges[2] - hDeadAliveRanges[1];
    Span<RayIndex> dAliveRayIndices = dDeadAliveIndices.subspan(hDeadAliveRanges[1], aliveRayCount);
    if(aliveRayCount != 0)
    {
        processQueue.IssueSaturatingKernel<KCCopyRays>
        (
            "KCCopyRays",
            KernelIssueParams{.workCount = aliveRayCount},
            dRays, dRayDifferentials,
            dAliveRayIndices,
            dRayState.dOutRays,
            dRayState.dOutRayDiffs
        );
    }

    // Now set the process queue
    uint32_t deadRayCount = hDeadAliveRanges[1] - hDeadAliveRanges[0];
    Span<RayIndex> dDeadRayIndices = dDeadAliveIndices.subspan(hDeadAliveRanges[0],
                                                               deadRayCount);
    Optional<RenderImageSection> renderOut;
    if(deadRayCount != 0)
    {
        processQueue.IssueWait(this->renderBuffer->PrevCopyCompleteFence());
        this->renderBuffer->ClearImage(processQueue);
        ImageSpan<3> filmSpan = this->imageTiler.template GetTileSpan<3>();

        SetImagePixelsIndirect
        (
            filmSpan,
            ToConstSpan(dDeadRayIndices),
            ToConstSpan(dRayState.dPathRadiance),
            ToConstSpan(dRayState.dFilmFilterWeights),
            ToConstSpan(dRayState.dImageCoordinates),
            Float(1), processQueue
        );
        // Issue a send of the FBO to Visor
        const GPUQueue& transferQueue = device.GetTransferQueue();
        renderOut = this->imageTiler.TransferToHost(processQueue,
                                                    transferQueue);
        // Semaphore is invalidated, visor is probably crashed
        if(!renderOut.has_value())
            return RendererOutput{};
        // Actual global weight
        renderOut->globalWeight = Float(1);
    }
    // We do not need to wait here, but we time
    // from CPU side so we need to wait
    // TODO: In future we should do OpenGL, Vulkan
    // style performance counters events etc. to
    // query the timing (may be couple of frame before even)
    // The timing is just a general performance indicator
    // It should not be super accurate.
    processQueue.Barrier().Wait();
    timer.Split();

    this->totalIterationCount++;
    totalDeadRayCount += deadRayCount;
    double timeSec = timer.Elapsed<Second>();
    double pathPerSec = static_cast<double>(deadRayCount) / timeSec;
    pathPerSec /= 1'000'000;

    uint64_t totalPixels = this->imageTiler.FullResolution().Multiply();
    double spp = double(totalDeadRayCount) / double(totalPixels);
    // Roll to the next tile
    this->imageTiler.NextTile();

    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            pathPerSec,
            "M path/s",
            spp,
            "spp",
            float(timer.Elapsed<Millisecond>()),
            this->imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            static_cast<uint32_t>(PathTraceRDetail::SampleMode::E::END),
            0
        },
        .imageOut = renderOut
    };
}

template<class MLA>
void PathTracerRenderer<MLA>::StopRender()
{
    this->ClearAllWorkMappings();
    filmFilter = {};
    rnGenerator = {};
    globalPixelIndex = 0;
}

template<class MLA>
std::string_view PathTracerRenderer<MLA>::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "PathTracerRGB"sv;
    return RendererTypeName<Name>;
}

template<class MLA>
size_t PathTracerRenderer<MLA>::GPUMemoryUsage() const
{
    return (rayPartitioner.UsedGPUMemory() +
            rnGenerator->UsedGPUMemory() +
            redererGlobalMem.Size());
}