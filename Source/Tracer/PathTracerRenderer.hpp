#pragma once

template<class MLA>
PathTracerRenderer<MLA>::PathTracerRenderer(const RenderImagePtr& rb,
                                            TracerView tv,
                                            BS::thread_pool& tp,
                                            const GPUSystem& s,
                                            const RenderWorkPack& wp)
    : RendererT<PathTracerRenderer<MLA>>(rb, wp, tv, s, tp)
    , rayPartitioner(s)
    , redererGlobalMem(s.AllGPUs(), 32_MiB, 512_MiB)
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
{
    switch(attributeIndex)
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
    curCamKey = CameraKey(static_cast<CommonKey>(curCamSurfaceParams.cameraId));
    curCamTransformKey = TransformKey(static_cast<CommonKey>(curCamSurfaceParams.transformId));
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
                                             dRays,
                                             dRayDifferentials,
                                             dRayState.dImageCoordinates,
                                             dRayState.dPathRadiance,
                                             dRayState.dFilmFilterWeights,
                                             dRayState.dDepth,
                                             dRayState.dOutRays,
                                             dRayState.dPathRadiance,
                                             dRayState.dThroughput,
                                             dRandomNumBuffer,
                                             dWorkHashes, dWorkBatchIds,
                                             dSubCameraBuffer),
                                    redererGlobalMem,
                                    {maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxRayCount,
                                     maxRayCount, maxSampleCount,
                                     totalWorkCount, totalWorkCount,
                                     this->SUB_CAMERA_BUFFER_SIZE});
    }
    else throw MRayError("Not yet implemented!");

    // And initialze the hashes
    workHasher = this->InitializeHashes(dWorkHashes, dWorkBatchIds, queue);

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
    rnGenerator = RngGen->get()(std::move(generatorCount),
                                std::move(seed),
                                this->gpuSystem, this->globalThreadPool);

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
    return RenderBufferInfo{};
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

    // Find the dead rays
    auto
    [
        hDeadRayRanges,
        dDeadAliveRayIndices
    ] = rayPartitioner.BinaryPartition
    (
        dIndices, processQueue,
        [this] MRAY_HYBRID(RayIndex index) -> bool
        {
            return dRayState.dDepth[index] == std::numeric_limits<uint8_t>::max();
        }
    );
    processQueue.Barrier().Wait();

    // Create RNG state for each ray
    // Generate rays
    rnGenerator->SetupRange(this->imageTiler.Tile1DRange());
    // Generate RN for camera rays
    uint32_t camSamplePerRay = (*curCamWork)->StochasticFilterSampleRayRNCount();
    auto dDeadRayIndices = dDeadAliveRayIndices.subspan(hDeadRayRanges[0],
                                                        hDeadRayRanges[1] - hDeadRayRanges[0]);
    uint32_t camRayGenRNCount = uint32_t(dDeadRayIndices.size()) * camSamplePerRay;
    auto dCamRayGenRNBuffer = dRandomNumBuffer.subspan(0, camRayGenRNCount);
    rnGenerator->GenerateNumbersIndirect(dCamRayGenRNBuffer,
                                         ToConstSpan(dDeadRayIndices),
                                         Vector2ui(0, camSamplePerRay),
                                         processQueue);

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

    // Refilled the path buffer
    // We can roll back to the Iota
    // to get the all rays.
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

    // Ray Casting
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
    auto partitionOutput = rayPartitioner.MultiPartition(dKeys, dIndices,
                                                         workHasher.WorkBatchDataRange(),
                                                         workHasher.WorkBatchBitRange(),
                                                         processQueue, false);
    assert(isHostVisible);
    // Wait for results to be available in host buffers
    processQueue.Barrier().Wait();

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

    // Copy the rays.


//    // Do shadow ray cast
//    if(currentOptions.mode == SurfRDetail::Mode::AO)
//    {
//        auto p = rayPartitioner.BinaryPartition(dPartitionIndices, processQueue,
//                                                IsValidRayFunctor(dRays[1]));
//        processQueue.Barrier().Wait();
//
//        auto dValidIndices = p.dPartitionIndices.subspan(p.hPartitionStartOffsets[0],
//                                                         p.hPartitionStartOffsets[1] - p.hPartitionStartOffsets[0]);
//
//        if(!dValidIndices.empty())
//        {
//            // Ray Casting
//            Bitspan<uint32_t> dIsVisibleBitSpan(dIsVisibleBuffer);
//            tracerView.baseAccelerator.CastVisibilityRays
//            (
//                dIsVisibleBitSpan, dBackupRNGStates,
//                dRays[1], dValidIndices, processQueue
//            );
//
//            // Write either one or zero
//            processQueue.IssueSaturatingKernel<KCIsVisibleToSpectrum>
//            (
//                "KCIsVisibleToSpectrum",
//                KernelIssueParams{.workCount = static_cast<uint32_t>(dValidIndices.size())},
//                dRayState.dOutputData,
//                ToConstSpan(dIsVisibleBitSpan),
//                dValidIndices
//            );
//        }
//    }
//    // Filter the samples
//    // Wait for previous copy to finish
//    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
//    renderBuffer->ClearImage(processQueue);
//    ImageSpan<3> filmSpan = imageTiler.GetTileSpan<3>();
//    if(currentOptions.doStochasticFilter)
//    {
//        SetImagePixels
//        (
//            filmSpan, ToConstSpan(dRayState.dOutputData),
//            ToConstSpan(dRayState.dFilmFilterWeights),
//            ToConstSpan(dRayState.dImageCoordinates),
//            Float(1), processQueue
//        );
//    }
//    else
//    {
//        // Using atomic filter since the samples are uniformly distributed
//        // And it is faster
//        filmFilter->ReconstructionFilterAtomicRGB
//        (
//            filmSpan,
//            ToConstSpan(dRayState.dOutputData),
//            ToConstSpan(dRayState.dImageCoordinates),
//            Float(1), processQueue
//        );
//    }
//    // Issue a send of the FBO to Visor
//    const GPUQueue& transferQueue = device.GetTransferQueue();
//    Optional<RenderImageSection>
//    renderOut = imageTiler.TransferToHost(processQueue,
//                                          transferQueue);
//    // Semaphore is invalidated, visor is probably crashed
//    if(!renderOut.has_value())
//        return RendererOutput{};
//    // Actual global weight
//    renderOut->globalWeight = Float(1);
//
//    // We do not need to wait here, but we time
//    // from CPU side so we need to wait
//    // TODO: In future we should do OpenGL, Vulkan
//    // style performance counters events etc. to
//    // query the timing (may be couple of frame before even)
//    // The timing is just a general performance indicator
//    // It should not be super accurate.
//    processQueue.Barrier().Wait();
//    timer.Split();
//
//    double timeSec = timer.Elapsed<Second>();
//    double samplePerSec = static_cast<double>(rayCount) / timeSec;
//    samplePerSec /= 1'000'000;
//    double spp = double(1) / double(imageTiler.TileCount().Multiply());
//    totalIterationCount++;
//    spp *= static_cast<double>(totalIterationCount);
//    // Roll to the next tile
//    imageTiler.NextTile();
//
//    return RendererOutput
//    {
//        .analytics = RendererAnalyticData
//        {
//            samplePerSec,
//            "M samples/s",
//            spp,
//            "spp",
//            float(timer.Elapsed<Millisecond>()),
//            imageTiler.FullResolution(),
//            MRayColorSpaceEnum::MR_ACES_CG,
//            GPUMemoryUsage(),
//            static_cast<uint32_t>(SurfRDetail::Mode::END),
//            0
//        },
//        .imageOut = renderOut
//    };
    return RendererOutput{};
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