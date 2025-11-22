#include "PathTracerRenderer.h"

#include "Core/MemAlloc.h"
#include "Core/Timer.h"
#include "Tracer/RendererCommon.h"

#include "Device/GPUAlgBinaryPartition.h"

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCAccumulateShadowRaysPT(MRAY_GRID_CONSTANT const Span<Spectrum> dRadianceOut,
                              MRAY_GRID_CONSTANT const Span<const Spectrum> dShadowRayRadiance,
                              MRAY_GRID_CONSTANT const Bitspan<const uint32_t> dIsVisibleBuffer,
                              MRAY_GRID_CONSTANT const Span<const PathDataPack> dPathDataPack,
                              MRAY_GRID_CONSTANT const Vector2ui rrRange)
{
    KernelCallParams kp;
    uint32_t shadowRayCount = static_cast<uint32_t>(dShadowRayRadiance.size());
    for(uint32_t i = kp.GlobalId(); i < shadowRayCount; i += kp.TotalSize())
    {
        PathDataPack dataPack = dPathDataPack[i];

        using enum RayType;
        bool isShadowRay = (dataPack.type == SHADOW_RAY);
        // +2 is correct here, we did not increment the depth yet
        bool inDepthLimit = ((dataPack.depth + 2u) <= rrRange[1]);
        if(inDepthLimit && isShadowRay && dIsVisibleBuffer[i])
            dRadianceOut[i] += dShadowRayRadiance[i];
    }
}

template<SpectrumContextC SC>
PathTracerRendererT<SC>::PathTracerRendererT(const RenderImagePtr& rb,
                                             TracerView tv,
                                             ThreadPool& tp,
                                             const GPUSystem& s,
                                             const RenderWorkPack& wp)
    : Base(rb, tv, tp, s, wp, TypeName())
    , metaLightArray(s)
    , rendererGlobalMem(s.AllGPUs(), 128_MiB, 512_MiB)
    , saveImage(true)
{}

template<SpectrumContextC SC>
typename PathTracerRendererT<SC>::AttribInfoList
PathTracerRendererT<SC>::AttributeInfo() const
{
    return StaticAttributeInfo();
}

template<SpectrumContextC SC>
RendererOptionPack
PathTracerRendererT<SC>::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.burstSize, 1));
    //
    std::string_view curRenderModeName = currentOptions.renderMode.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{},
                                              curRenderModeName.size()));
    auto svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == curRenderModeName.size());
    std::copy(curRenderModeName.cbegin(), curRenderModeName.cend(), svRead.begin());
    //
    std::string_view curModeName = currentOptions.sampleMode.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{},
                                              curModeName.size()));
    svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == curModeName.size());
    std::copy(curModeName.cbegin(), curModeName.cend(), svRead.begin());
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<Vector2>{}, 1));
    result.attributes.back().Push(Span<const Vector2ui>(&currentOptions.russianRouletteRange, 1));
    //
    std::string_view lightSamplerName = currentOptions.lightSampler.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{}, lightSamplerName.size()));
    svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == lightSamplerName.size());
    std::copy(lightSamplerName.cbegin(), lightSamplerName.cend(), svRead.begin());
    //
    if constexpr(MRAY_IS_DEBUG)
    {
        for([[maybe_unused]] const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

template<SpectrumContextC SC>
void PathTracerRendererT<SC>::PushAttribute(uint32_t attributeIndex,
                                            TransientData data, const GPUQueue&)
{
    switch(attributeIndex)
    {
        case 0: newOptions.totalSPP = data.AccessAs<uint32_t>()[0]; break;
        case 1: newOptions.burstSize = data.AccessAs<uint32_t>()[0]; break;
        case 2: newOptions.renderMode = RenderMode(std::as_const(data).AccessAsString()); break;
        case 3: newOptions.sampleMode = PathTraceRDetail::SampleMode(std::as_const(data).AccessAsString()); break;
        case 4: newOptions.russianRouletteRange = data.AccessAs<Vector2ui>()[0]; break;
        case 5: newOptions.lightSampler = LightSamplerType(std::as_const(data).AccessAsString()); break;
        default:
            throw MRayError("{} Unknown attribute index {}", TypeName(), attributeIndex);
    }
}

template<SpectrumContextC SC>
void PathTracerRendererT<SC>::CopyAliveRays(Span<const RayIndex> dAliveRayIndices,
                                            const GPUQueue& processQueue)
{
    if(dAliveRayIndices.empty()) return;

    uint32_t aliveRayCount = static_cast<uint32_t>(dAliveRayIndices.size());
    processQueue.IssueWorkKernel<KCCopyRaysIndirect>
    (
        "KCCopyRaysIndirect",
        DeviceWorkIssueParams{.workCount = aliveRayCount},
        //
        dRays, dRayCones,
        dAliveRayIndices,
        dOutRays,
        dOutRayCones
    );
}

template<SpectrumContextC SC>
uint32_t
PathTracerRendererT<SC>::FindMaxSamplePerIteration(uint32_t rayCount,
                                                   PathTraceRDetail::SampleMode sampleMode)
{
    uint32_t camSample = curCamWork->StochasticFilterSampleRayRNList().TotalRNCount();
    uint32_t spectrumSample = (spectrumContext)
                ? spectrumContext->SampleSpectrumRNList().TotalRNCount()
                : 0u;

    uint32_t maxSample = Math::Max(camSample, spectrumSample);
    maxSample = std::transform_reduce
    (
        currentWorks.cbegin(), currentWorks.cend(), maxSample,
        [](uint32_t l, uint32_t r) -> uint32_t
        {
            return Math::Max(l, r);
        },
        [sampleMode](const auto& renderWorkStruct) -> uint32_t
        {
            // TODO: Report bug (MSVC):
            // This:
            //
            // "using enum PathTraceRDetail::SampleMode::E;"
            //
            // does not work?
            //
            // But when PathTracerRendererT was not a template it did work.
            // Anyway report it...
            if(sampleMode == PathTraceRDetail::SampleMode::E::PURE)
                return renderWorkStruct.workPtr->SampleRNList(0).TotalRNCount();
            else
                return Math::Max(renderWorkStruct.workPtr->SampleRNList(0).TotalRNCount(),
                                 renderWorkStruct.workPtr->SampleRNList(1).TotalRNCount());
        }
    );
    return rayCount * maxSample;
}

template<SpectrumContextC SC>
Span<RayIndex>
PathTracerRendererT<SC>::DoRenderPass(uint32_t sppLimit,
                                      const GPUQueue& processQueue)
{
    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    RayState dRayState =
    {
        .dPathRadiance      = dPathRadiance,
        .dImageCoordinates  = dImageCoordinates,
        .dFilmFilterWeights = dFilmFilterWeights,
        .dThroughput        = dThroughputs,
        .dPathDataPack      = dPathDataPack,
        .dOutRays           = dOutRays,
        .dOutRayCones       = dOutRayCones,
        .dPrevMatPDF        = dPrevMatPDF,
        .dShadowRayRadiance = dShadowRayRadiance,
        .dPathWavelengths   = dPathWavelengths
    };

    assert(sppLimit != 0);
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
    rnGenerator->SetupRange(imageTiler.LocalTileStart(),
                            imageTiler.LocalTileEnd(),
                            processQueue);
    // Reload dead paths with new
    auto [dReloadIndices, _, aliveRayCount] = ReloadPaths(dIndices, sppLimit,
                                                          processQueue);
    dIndices = dReloadIndices.subspan(0, aliveRayCount);
    dKeys = dKeys.subspan(0, aliveRayCount);

    // Cast rays
    using namespace std::string_view_literals;
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    processQueue.IssueWorkKernel<KCSetBoundaryWorkKeysIndirect>
    (
        "KCSetBoundaryWorkKeys"sv,
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dHitKeys,
        ToConstSpan(dIndices),
        this->boundaryLightKeyPack
    );
    // Actual Ray Casting
    tracerView.baseAccelerator.CastRays
    (
        Span<VolumeIndex>(),
        dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, false, processQueue
    );

    if(currentOptions.sampleMedia)
    {
        // Generate work keys for media

        // N -way Partition wrt. medium/transform pair

        // Call Media Transmit

        // Binary Partition wrt. scatter/not scatter event

        // For scattered rays,
        //    set their hit pack to invalid.
        //
        //    Launch shadow rays and find visibility
        //      Shadow ray launch is recursive if media is on
        //    (if NEE is on)
        //    Resolve transmittence on the media
        //    Accumulate light if transmittance is ok and visibility
        //    is ok
    }


    // Generate work keys from hit packs
    using namespace std::string_literals;
    processQueue.IssueWorkKernel<KCGenerateSurfaceWorkKeysIndirect>
    (
        "KCGenerateSurfaceWorkKeysIndirect"sv,
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dKeys,
        ToConstSpan(dIndices),
        ToConstSpan(dHitKeys),
        surfaceWorkHasher
    );

    // Finally, partition using the generated keys.
    // Fully partitioning here by using a single sort
    auto& rp = rayPartitioner;
    auto partitionOutput = rp.MultiPartition(dKeys, dIndices,
                                             surfaceWorkHasher.WorkBatchDataRange(),
                                             surfaceWorkHasher.WorkBatchBitRange(),
                                             processQueue, false);
    // Wait for results to be available in host buffers
    // since we need partition ranges on the CPU to Issue kernels.
    processQueue.Barrier().Wait();
    // Old Indices array (and the key) is invalidated
    // Change indices to the partitioned one
    dIndices = partitionOutput.dPartitionIndices;

    if(currentOptions.sampleMode == SampleMode::E::PURE)
    {
        // =================== //
        //  Pure Path Tracing  //
        // =================== //
        // Work_0           = BxDF sample
        // BoundaryWork_0   = Accumulate light radiance value to the path
        using GlobalState = PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>;
        GlobalState globalState
        {
            .russianRouletteRange = currentOptions.russianRouletteRange,
            .sampleMode = currentOptions.sampleMode,
            .lightSampler = EmptyType{},
            .specContextData = typedSpectrumContext.GetData()
        };

        IssueSurfaceWorkKernelsToPartitions<This>(surfaceWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t partitionSize)
        {
            RNRequestList rnList = workI.SampleRNList(0);
            uint32_t rnCount = rnList.TotalRNCount();
            auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
            rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                 dPathRNGDimensions,
                                                 rnList,
                                                 processQueue);

            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, dLocalIndices, processQueue,
                ConstAddFunctor_U16(uint16_t(rnCount))
            );

            workI.DoWork_0(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRandomNumBuffer,
                           dHits, dHitKeys,
                           globalState, processQueue);
        },
        //
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            workI.DoBoundaryWork_0(dRayState,
                                   dRays, dRayCones,
                                   dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dHits, dHitKeys,
                                   globalState, processQueue);
        });
    }
    else
    {
        // =================================  //
        //  Path Tracing with NEE and/or MIS  //
        // ================================== //
        // Work_0           = BxDF sample
        // Work_1           = NEE sample only
        // BoundaryWork_1   = Same as light accumulation but with many states
        //                    regarding NEE and MIS
        UniformLightSampler lightSampler(metaLightArray.Array(),
                                         metaLightArray.IndexHashTable());

        using GlobalState = PathTraceRDetail::GlobalState<UniformLightSampler, SpectrumConverter>;
        GlobalState globalState
        {
            .russianRouletteRange = currentOptions.russianRouletteRange,
            .sampleMode = currentOptions.sampleMode,
            .lightSampler = lightSampler,
            .specContextData = typedSpectrumContext.GetData()
        };
        using GlobalStateE = PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>;
        GlobalStateE globalStateE
        {
            .russianRouletteRange = currentOptions.russianRouletteRange,
            .sampleMode = currentOptions.sampleMode,
            .lightSampler = EmptyType{},
            .specContextData = typedSpectrumContext.GetData()
        };

        // Clear the shadow ray radiance buffer
        processQueue.MemsetAsync(dShadowRayRadiance, 0x00);
        processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
        // CUDA Init check error, we access the rays even if it is not written
        processQueue.MemsetAsync(dOutRays, 0x00);
        // Do the NEE kernel + boundary work
        IssueSurfaceWorkKernelsToPartitions<This>(surfaceWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t partitionSize)
        {
            RNRequestList rnList = workI.SampleRNList(1);
            uint32_t rnCount = rnList.TotalRNCount();
            auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
            rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                 dPathRNGDimensions,
                                                 rnList,
                                                 processQueue);

            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, dLocalIndices, processQueue,
                ConstAddFunctor_U16(uint16_t(rnCount))
            );

            workI.DoWork_1(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRandomNumBuffer,  dHits,
                           dHitKeys, globalState,
                           processQueue);
        },
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            workI.DoBoundaryWork_1(dRayState,  dRays,
                                   dRayCones, dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dHits, dHitKeys,
                                   globalState, processQueue);
        });

        // After the kernel call(s), "dOutRays" holds the
        // shadow rays. Check for visibility.
        Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
        tracerView.baseAccelerator.CastVisibilityRays
        (
            dIsVisibleBitSpan, dBackupRNGStates,
            dOutRays, dIndices, processQueue
        );

        // Accumulate the pre-calculated radiance selectively
        processQueue.IssueWorkKernel<KCAccumulateShadowRaysPT>
        (
            "KCAccumulateShadowRays",
            DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dShadowRayRadiance.size())},
            //
            dPathRadiance,
            ToConstSpan(dShadowRayRadiance),
            ToConstSpan(dIsVisibleBitSpan),
            ToConstSpan(dPathDataPack),
            currentOptions.russianRouletteRange
        );

        // Do the actual kernel
        IssueSurfaceWorkKernelsToPartitions<This>(surfaceWorkHasher, partitionOutput,
        [&](const auto& workI, Span<uint32_t> dLocalIndices,
            uint32_t, uint32_t partitionSize)
        {
            RNRequestList rnList = workI.SampleRNList(0);
            uint32_t rnCount = rnList.TotalRNCount();
            auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
            rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                 dPathRNGDimensions,
                                                 rnList,
                                                 processQueue);

            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, dLocalIndices, processQueue,
                ConstAddFunctor_U16(uint16_t(rnCount))
            );

            workI.DoWork_0(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRandomNumBuffer,
                           dHits, dHitKeys,
                           globalStateE, processQueue);
        },
        // Empty Invocation for lights this pass
        [&](const auto&, Span<uint32_t>, uint32_t, uint32_t) {});
    }

    return dIndices;
}

template<SpectrumContextC SC>
RendererOutput
PathTracerRendererT<SC>::DoThroughputSingleTileRender(const GPUDevice& device,
                                                      const GPUQueue& processQueue)
{
    Timer timer; timer.Start();
    const auto& cameraWork = *curCamWork;
    // Generate subcamera of this specific tile
    if(totalIterationCount == 0)
    {
        cameraWork.GenerateSubCamera
        (
            dSubCameraBuffer,
            curCamKey, curCamTransformOverride,
            imageTiler.CurrentTileIndex(),
            imageTiler.TileCount(),
            processQueue
        );
    }

    // ====================== //
    //   Single Render Pass   //
    // ====================== //
    uint32_t sppLimit = (saveImage) ? currentOptions.totalSPP
                                    : std::numeric_limits<uint32_t>::max();
    Span<RayIndex> dIndices = DoRenderPass(sppLimit, processQueue);

    // Find the dead paths again
    // Do a 3-way partition here to catch potential invalid rays
    auto deadAlivePartitionOut = rayPartitioner.TernaryPartition
    (
        dIndices, processQueue,
        IsDeadAliveInvalidFunctor(ToConstSpan(dPathDataPack))
    );
    processQueue.Barrier().Wait();
    auto [dDeadRayIndices, dInvalidRayIndices, dAliveRayIndices] =
        deadAlivePartitionOut.Spanify();

    // Write radiance of dead rays to image buffer async
    // via transfer queue.
    Optional<RenderImageSection> renderOut;
    const GPUQueue& transferQueue = device.GetTransferQueue();
    renderOut = AddRadianceToRenderBufferThroughput(dDeadRayIndices,
                                                    processQueue,
                                                    transferQueue);

    // Copy continuing set of rays to actual ray buffer for next iteration
    CopyAliveRays(dAliveRayIndices, processQueue);

    // We do not need to wait here, but we time
    // from CPU side so we need to wait
    // TODO: In future we should do OpenGL, Vulkan
    // style performance counters events etc. to
    // query the timing (may be couple of frame before even)
    // The timing is just a general performance indicator
    // It should not be super accurate.
    processQueue.Barrier().Wait();
    timer.Split();

    // Report the results
    totalIterationCount++;
    RendererAnalyticData analyticData;
    analyticData = CalculateAnalyticDataThroughput(dDeadRayIndices.size(),
                                                   currentOptions.totalSPP,
                                                   timer);

    // We exhausted all alive rays while doing SPP limit.
    bool triggerSave = (saveImage && totalDeadRayCount == TotalSampleLimit(currentOptions.totalSPP));
    if(triggerSave) saveImage = false;

    analyticData.customLogicSize0 = uint32_t(PathTraceRDetail::SampleMode::E::END);
    return RendererOutput
    {
        .analytics = std::move(analyticData),
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

template<SpectrumContextC SC>
RendererOutput
PathTracerRendererT<SC>::DoLatencyRender(uint32_t passCount,
                                         const GPUDevice& device,
                                         const GPUQueue& processQueue)
{
    Vector2ui tileCount2D = imageTiler.TileCount();
    uint32_t tileCount1D = tileCount2D.Multiply();

    Timer timer; timer.Start();
    // Generate subcamera of this specific tile
    const auto& cameraWork = *curCamWork;
    cameraWork.GenerateSubCamera
    (
        dSubCameraBuffer,
        curCamKey, curCamTransformOverride,
        imageTiler.CurrentTileIndex(),
        tileCount2D,
        processQueue
    );

    // We are waiting too early here,
    // We should wait at least on the first render buffer write
    // but it was not working so I've put it here
    // TODO: Investigate
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    renderBuffer->ClearImage(processQueue);

    //
    uint32_t tileIndex = imageTiler.CurrentTileIndex1D();
    uint32_t tileSPP = static_cast<uint32_t>(tileSPPs[tileIndex]);
    uint32_t sppLimit = tileSPP + passCount;
    if(saveImage)
        sppLimit = Math::Min(sppLimit, currentOptions.totalSPP);

    tileSPPs[tileIndex] = sppLimit;
    uint32_t currentPassCount = sppLimit - tileSPP;
    uint32_t passPathCount = imageTiler.CurrentTileSize().Multiply() * currentPassCount;
    uint32_t invalidRayCount = 0;
    do
    {
        Span<RayIndex> dIndices = DoRenderPass(sppLimit, processQueue);

        // Find the dead paths again
        // Every path is processed, so we do not need to use the scrambled
        // index buffer. Iota again
        //DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
        // Do a 3-way partition,
        auto deadAlivePartitionOut = rayPartitioner.TernaryPartition
        (
            dIndices, processQueue,
            IsDeadAliveInvalidFunctor(ToConstSpan(dPathDataPack))
        );
        processQueue.Barrier().Wait();
        auto [dDeadRayIndices, dInvalidRayIndices, dAliveRayIndices] =
            deadAlivePartitionOut.Spanify();

        AddRadianceToRenderBufferLatency(dDeadRayIndices, processQueue);
        CopyAliveRays(dAliveRayIndices, processQueue);

        assert(dInvalidRayIndices.size() == 0);
        invalidRayCount += (static_cast<uint32_t>(dDeadRayIndices.size()));
    } while(invalidRayCount != passPathCount);

    // One spp of this tile should be done now.
    // Issue the transfer
    // Issue a send of the FBO to Visor
    const GPUQueue& transferQueue = device.GetTransferQueue();
    Optional<RenderImageSection> renderOut;
    renderOut = imageTiler.TransferToHost(processQueue, transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value()) return RendererOutput{};
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

    // Roll to the next tile
    imageTiler.NextTile();

    // Check save trigger
    // Notice, we've changed to the next tile above.
    uint64_t curSPPSum = std::reduce(tileSPPs.cbegin(), tileSPPs.cend(), uint64_t(0));
    uint64_t sppSumCheck = (uint64_t(currentOptions.totalSPP) *
                            uint64_t(tileCount1D));
    bool triggerSave = (saveImage && curSPPSum == sppSumCheck);
    if(triggerSave) saveImage = false;

    // Report the results
    totalIterationCount++;
    RendererAnalyticData analyticData;
    analyticData = CalculateAnalyticDataLatency(passPathCount, currentOptions.totalSPP,
                                                timer);
    analyticData.customLogicSize0 = uint32_t(PathTraceRDetail::SampleMode::E::END);

    return RendererOutput
    {
        .analytics   = std::move(analyticData),
        .imageOut    = renderOut,
        .triggerSave = triggerSave
    };
}

template<SpectrumContextC SC>
RenderBufferInfo
PathTracerRendererT<SC>::StartRender(const RenderImageParams& rIP,
                                     CamSurfaceId camSurfId,
                                     uint32_t customLogicIndex0,
                                     uint32_t)
{
    currentOptions = newOptions;
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);

    // Change the mode according to the render logic
    using Math::Roll;
    int32_t modeIndex = (int32_t(SampleMode::E(currentOptions.sampleMode)) +
                         int32_t(customLogicIndex0));
    uint32_t sendMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                      int32_t(SampleMode::E::END)));
    uint32_t newMode = uint32_t(Roll(modeIndex, 0, int32_t(SampleMode::E::END)));
    currentOptions.sampleMode = SampleMode::E(newMode);

    // ================================ //
    // Initialize common sub components //
    // ================================ //
    uint32_t sppLimit = currentOptions.totalSPP;
    auto [maxRayCount, totalWorkCount] = InitializeForRender(camSurfId, sppLimit,
                                                             false, rIP);
    renderMode = currentOptions.renderMode;
    burstSize = currentOptions.burstSize;

    // ========================= //
    //      Spectrum Context     //
    // ========================= //
    uint32_t wavelengthCount = (IsSpectral) ? maxRayCount : 0u;
    auto colorSpace = tracerView.tracerParams.globalTextureColorSpace;
    if constexpr(IsSpectral)
    {
        // Don't bother reloading context if colorspace is same
        if(!spectrumContext || colorSpace != spectrumContext->ColorSpace())
        {
            auto wlSampleMode = tracerView.tracerParams.wavelengthSampleMode;
            spectrumContext = std::make_unique<SC>(colorSpace, wlSampleMode,
                                                   gpuSystem);
        }
    }

    // ========================= //
    //   Path State Allocation   //
    // ========================= //
    // You can see why wavefront approach uses
    // quite a bit memory (and this is somewhat optimized).
    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount, currentOptions.sampleMode);
    if(currentOptions.sampleMode == SampleMode::E::PURE)
    {
        MemAlloc::AllocateMultiData
        (
            Tie
            (
                // Per path
                dHits, dHitKeys, dRays, dRayCones,
                dPathRadiance, dImageCoordinates,
                dFilmFilterWeights, dThroughputs,
                dPathDataPack, dOutRays,
                dOutRayCones, dPathRNGDimensions,
                // Per path (but can be zero if not spectral)
                dPathWavelengths,
                dSpectrumWavePDFs,
                // Per path bounce, max used random number of that bounce
                dRandomNumBuffer,
                // Per render work
                dSurfaceWorkHashes, dSurfaceWorkBatchIds,
                dMediumWorkHashes, dMediumWorkBatchIds,
                //
                dSubCameraBuffer
            ),
            rendererGlobalMem,
            {
                // Per path
                maxRayCount, maxRayCount, maxRayCount, maxRayCount,
                maxRayCount, maxRayCount, maxRayCount, maxRayCount,
                maxRayCount, maxRayCount, maxRayCount, maxRayCount,
                // Per path (but can be zero if not spectral)
                wavelengthCount, wavelengthCount,
                // Per path bounce, max used random number of that bounce
                maxSampleCount,
                // Per render work
                totalWorkCount, totalWorkCount,
                // TODO: This is a waste we should split the
                // total work count
                totalWorkCount, totalWorkCount,
                //
                RendererBase::SUB_CAMERA_BUFFER_SIZE});
    }
    else
    {
        uint32_t isVisibleIntCount = Bitspan<uint32_t>::CountT(maxRayCount);
        MemAlloc::AllocateMultiData
        (
            Tie
            (
                // Per path
                dHits, dHitKeys, dRays, dRayCones,
                dPathRadiance, dImageCoordinates,
                dFilmFilterWeights, dThroughputs,
                dPathDataPack, dOutRays, dOutRayCones,
                dPrevMatPDF, dShadowRayRadiance,
                dPathRNGDimensions,
                // Per path (but can be zero if not spectral)
                dPathWavelengths,
                dSpectrumWavePDFs,
                // Per path but a single bit is used
                dShadowRayVisibilities,
                // Per path bounce, max used random number of that bounce
                dRandomNumBuffer,
                // Per render work
                dSurfaceWorkHashes, dSurfaceWorkBatchIds,
                dMediumWorkHashes, dMediumWorkBatchIds,
                //
                dSubCameraBuffer
            ),
            rendererGlobalMem,
            {
                // Per path
                maxRayCount, maxRayCount, maxRayCount, maxRayCount,
                maxRayCount, maxRayCount, maxRayCount, maxRayCount,
                maxRayCount, maxRayCount, maxRayCount, maxRayCount,
                maxRayCount, maxRayCount,
                // Per path (but can be zero if not spectral)
                wavelengthCount, wavelengthCount,
                // Per path but a single bit is used
                isVisibleIntCount,
                // Per path bounce, max used random number of that bounce
                maxSampleCount,
                // Per render work
                totalWorkCount, totalWorkCount,
                // TODO: This is a waste but medium/transform
                // pairs should be small
                totalWorkCount, totalWorkCount,
                //
                RendererBase::SUB_CAMERA_BUFFER_SIZE
            }
        );
        // Since NEE is active generate meta light list
        MetaLightListConstructionParams mlParams =
        {
            .lightGroups = tracerView.lightGroups,
            .transformGroups = tracerView.transGroups,
            .lSurfList = Span<const Pair<LightSurfaceId, LightSurfaceParams>>(tracerView.lightSurfs)
        };
        metaLightArray.Construct(mlParams, tracerView.boundarySurface, queue);
    }

    // ===================== //
    // After Allocation Init //
    // ===================== //
    surfaceWorkHasher = InitializeSurfaceHashes(dSurfaceWorkHashes,
                                                dSurfaceWorkBatchIds,
                                                maxRayCount, queue);
    mediumWorkHasher = InitializeMediumHashes(dMediumWorkHashes,
                                              dMediumWorkBatchIds,
                                              maxRayCount, queue);

    // Reset hits and paths
    queue.MemsetAsync(dHits, 0x00);
    ResetAllPaths(queue);

    auto bufferPtrAndSize = renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = colorSpace,
        .resolution = imageTiler.FullResolution(),
        .curRenderLogic0 = sendMode,
        .curRenderLogic1 = std::numeric_limits<uint32_t>::max()
    };
}

template<SpectrumContextC SC>
void PathTracerRendererT<SC>::StopRender()
{
    ClearAllWorkMappings();
    filmFilter = {};
    rnGenerator = {};
    metaLightArray.Clear();
}

template<SpectrumContextC SC>
std::string_view PathTracerRendererT<SC>::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;

    if constexpr(IsSpectral)
    {
        static constexpr auto Name = "PathTracerSpectral"sv;
        return RendererTypeName<Name>;
    }
    else
    {
        static constexpr auto Name = "PathTracerRGB"sv;
        return RendererTypeName<Name>;
    }
}

template<SpectrumContextC SC>
typename PathTracerRendererT<SC>::AttribInfoList
PathTracerRendererT<SC>::StaticAttributeInfo()
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP",        MRayDataTypeRT(MR_UINT32),      IS_SCALAR, MR_MANDATORY},
        {"burstSize",       MRayDataTypeRT(MR_UINT32),      IS_SCALAR, MR_OPTIONAL},
        {"renderMode",      MRayDataTypeRT(MR_STRING),      IS_SCALAR, MR_MANDATORY},
        {"sampleMode",      MRayDataTypeRT(MR_STRING),      IS_SCALAR, MR_MANDATORY},
        {"rrRange",         MRayDataTypeRT(MR_VECTOR_2UI),  IS_SCALAR, MR_MANDATORY},
        {"neeSamplerType",  MRayDataTypeRT(MR_STRING),      IS_SCALAR, MR_MANDATORY}
    };
}

template<SpectrumContextC SC>
size_t PathTracerRendererT<SC>::GPUMemoryUsage() const
{
    size_t total = (rayPartitioner.GPUMemoryUsage() +
                    rnGenerator->GPUMemoryUsage() +
                    rendererGlobalMem.Size());
    if(spectrumContext)
        total += spectrumContext->GPUMemoryUsage();

    return total;
}

template class PathTracerRendererT<SpectrumContextIdentity>;
template class PathTracerRendererT<SpectrumContextJakob2019>;