#include "GuidedPTRenderer.h"

#include "Core/MemAlloc.h"
#include "Core/Timer.h"
#include "Tracer/RendererCommon.h"

static constexpr auto INVALID_MC_INDEX = std::numeric_limits<uint32_t>::max();

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCInitGuidedPathStatesIndirect(MRAY_GRID_CONSTANT const Span<uint32_t> dLiftedMarkovChainIndex,
                                    MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    KernelCallParams kp;
    uint32_t newPathCount = uint32_t(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < newPathCount; i += kp.TotalSize())
    {
        dLiftedMarkovChainIndex[dIndices[i]] = INVALID_MC_INDEX;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCDivideByPDFIndirect(// I-O
                           MRAY_GRID_CONSTANT const Span<Spectrum> dRadianceOut,
                           // Input
                           MRAY_GRID_CONSTANT const Span<const Float> dPDFChains,
                           MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    KernelCallParams kp;
    uint32_t deadRayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < deadRayCount; i += kp.TotalSize())
    {
        RayIndex index = dRayIndices[i];
        Spectrum radiance = dRadianceOut[index];
        radiance = Distribution::Common::DivideByPDF(radiance, dPDFChains[index]);
        dRadianceOut[index] = radiance;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCBackpropagateIrradiance(// I-O
                               MRAY_GRID_CONSTANT const Span<Float> dPrevPathReflectanceOrOutRadiance,
                               MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState globalState,
                               MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                               // Input
                               MRAY_GRID_CONSTANT const Span<const uint32_t> dLiftedMCIndices,
                               MRAY_GRID_CONSTANT const Span<const Float> dScoreSums,

                               // Determining if this is intermediate path (not hit light etc.)
                               MRAY_GRID_CONSTANT const Span<const HitKeyPack> dHitKeys,
                               MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                               MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    using namespace GuidedPTRDetail;

    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t rIndex = dRayIndices[i];
        uint32_t isLightFlag = dHitKeys[rIndex].lightOrMatKey.FetchFlagPortion();
        uint32_t prevMCIndex = dLiftedMCIndices[rIndex];
        //
        if(isLightFlag == IS_LIGHT_KEY_FLAG) continue;
        if(prevMCIndex == INVALID_MC_INDEX) continue;

        // ======================= //
        // Irradiance Cache Update //
        // ======================= //
        auto [ray, tMM] = RayFromGMem(dRays, rIndex);
        Vector3 position = ray.AdvancedPos(tMM[1]);
        Vector3 direction = Math::Normalize(-ray.dir);

        const HashGridView& hg = globalState.hashGrid;
        auto locationOpt = hg.Search(position, direction);
        //
        if(!locationOpt) continue;
        // Race condition here!
        // We read while somebody writes to this specific location
        // maybe. It is fine as long as it is properly updated by writers.
        // Reading a stale value just hinders backpropogation speed maybe.
        //
        // TODO: Check if proper syncronization is good / bad etc.
        auto& dIrradHashGrid = globalState.dMCIrradiances;
        Float irrad = dIrradHashGrid[*locationOpt].irrad;
        Float refl = dPrevPathReflectanceOrOutRadiance[rIndex];
        Float outRadiance = refl * irrad;
        MCIrradiance::AtomicEMA(dIrradHashGrid[prevMCIndex], outRadiance);

        // ========================= //
        //  Markov Chain Lock Update //
        // ========================= //
        Float weight = outRadiance;
        BackupRNG rng = BackupRNG(dRNGStates[rIndex]);
        Float scoreSum = dScoreSums[rIndex];
        // Not as good as previous mixture, skip
        if(rng.NextFloat() * scoreSum < weight * MC_LOBE_COUNT)
        {
            // This kernel will be used as deterministic lock mechanism
            // where largest rIndex will be the winner
            // At the next kernel, winners will use
            // "dPrevPathReflectanceOrOutRadiance" value
            // (which is outRadiance atm) to do the actual update.
            //
            // We lose some data here (only a single winner will update the
            // markov chain). But paper also does this via race condition.
            // But that is non-determistic.
            DeviceAtomic::AtomicMax(globalState.dMCLocks[i], rIndex);
        }
        dPrevPathReflectanceOrOutRadiance[rIndex] = outRadiance;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static
void KCUpdateMarkovChains(// I-O
                          MRAY_GRID_CONSTANT const GuidedPTRDetail::GlobalState globalState,
                          // Input
                          MRAY_GRID_CONSTANT const Span<const Float> dPrevPathReflectanceOrOutRadiance,
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dLiftedMCIndices,
                          // Determining if this is intermediate path (not hit light etc.)
                          MRAY_GRID_CONSTANT const Span<const HitKeyPack> dHitKeys,
                          MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                          MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    using namespace GuidedPTRDetail;

    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        uint32_t rIndex = dRayIndices[i];
        uint32_t isLightFlag = dHitKeys[rIndex].lightOrMatKey.FetchFlagPortion();
        uint32_t prevMCIndex = dLiftedMCIndices[rIndex];
        uint32_t mcLock = globalState.dMCLocks[rIndex];
        //
        if(isLightFlag == IS_LIGHT_KEY_FLAG) continue;
        if(prevMCIndex == INVALID_MC_INDEX) continue;
        if(mcLock != rIndex) continue;

        // ====================== //
        //   Markov Chain Update  //
        // ====================== //
        auto [ray, tMM] = RayFromGMem(dRays, rIndex);
        Vector3 pos = ray.AdvancedPos(tMM[1]);
        Vector3 dir = Math::Normalize(ray.dir);
        // If a thread is here we have the lock,
        // we can freely modify
        auto& dStates = globalState.dMCStates;
        auto& dCounts = globalState.dMCCounts;
        MCState s     = dStates[prevMCIndex];
        MCCount count = dCounts[prevMCIndex];
        // Update routine
        static constexpr Float MIN_EMA_RATIO_CM = Float(0.01);
        static constexpr uint16_t MAX_SAMPLE_MC = uint16_t(2048);
        Float weight = dPrevPathReflectanceOrOutRadiance[rIndex];
        count = Math::Min<uint16_t>(count + 1, MAX_SAMPLE_MC);

        Float tMC = Math::Max(Float(1) / Float(count), MIN_EMA_RATIO_CM);
        s.weight = Math::Lerp(s.weight, weight, tMC);
        s.target = Math::Lerp(s.target, pos * weight, tMC);
        //
        Vector3 stateDir = (s.weight < Float(0)) ? (s.target / s.weight) : s.target;
        Float newCos = Math::Max(Math::Dot(dir, stateDir), Float(0));
        s.cos = Math::Lerp(s.cos, weight * newCos, tMC);

        // Finally, write back
        dStates[prevMCIndex] = s;
        dCounts[prevMCIndex] = count;
    }
}

GuidedPTRenderer::GuidedPTRenderer(const RenderImagePtr& rb,
                                   TracerView tv,
                                   ThreadPool& tp,
                                   const GPUSystem& s,
                                   const RenderWorkPack& wp)
    : Base(rb, tv, tp, s, wp, TypeName())
    , metaLightArray(s)
    , hashGrid(s)
    , rendererGlobalMem(s.AllGPUs(), 128_MiB, 512_MiB)
    , saveImage(true)
{}

typename GuidedPTRenderer::AttribInfoList
GuidedPTRenderer::AttributeInfo() const
{
    return StaticAttributeInfo();
}

RendererOptionPack
GuidedPTRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    //result.paramTypes = AttributeInfo();
    ////
    //result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    //result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));
    ////
    //result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    //result.attributes.back().Push(Span<const uint32_t>(&currentOptions.burstSize, 1));
    ////
    //std::string_view curRenderModeName = currentOptions.renderMode.ToString();
    //result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{},
    //                                          curRenderModeName.size()));
    //auto svRead = result.attributes.back().AccessAsString();
    //assert(svRead.size() == curRenderModeName.size());
    //std::copy(curRenderModeName.cbegin(), curRenderModeName.cend(), svRead.begin());
    ////
    //std::string_view curModeName = currentOptions.sampleMode.ToString();
    //result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{},
    //                                          curModeName.size()));
    //svRead = result.attributes.back().AccessAsString();
    //assert(svRead.size() == curModeName.size());
    //std::copy(curModeName.cbegin(), curModeName.cend(), svRead.begin());
    ////
    //result.attributes.push_back(TransientData(std::in_place_type_t<Vector2>{}, 1));
    //result.attributes.back().Push(Span<const Vector2ui>(&currentOptions.russianRouletteRange, 1));
    ////
    //std::string_view lightSamplerName = currentOptions.lightSampler.ToString();
    //result.attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{}, lightSamplerName.size()));
    //svRead = result.attributes.back().AccessAsString();
    //assert(svRead.size() == lightSamplerName.size());
    //std::copy(lightSamplerName.cbegin(), lightSamplerName.cend(), svRead.begin());
    ////
    //if constexpr(MRAY_IS_DEBUG)
    //{
    //    for([[maybe_unused]] const auto& d: result.attributes)
    //        assert(d.IsFull());
    //}
    return result;
}

void GuidedPTRenderer::PushAttribute(uint32_t attributeIndex,
                                     TransientData data, const GPUQueue&)
{
    //switch(attributeIndex)
    //{
    //    case 0: newOptions.totalSPP = data.AccessAs<uint32_t>()[0]; break;
    //    case 1: newOptions.burstSize = data.AccessAs<uint32_t>()[0]; break;
    //    case 2: newOptions.renderMode = RenderMode(std::as_const(data).AccessAsString()); break;
    //    case 3: newOptions.sampleMode = PathTraceRDetail::SampleMode(std::as_const(data).AccessAsString()); break;
    //    case 4: newOptions.russianRouletteRange = data.AccessAs<Vector2ui>()[0]; break;
    //    case 5: newOptions.lightSampler = LightSamplerType(std::as_const(data).AccessAsString()); break;
    //    default:
    //        throw MRayError("{} Unknown attribute index {}", TypeName(), attributeIndex);
    //}
}

uint32_t
GuidedPTRenderer::FindMaxSamplePerIteration(uint32_t rayCount)
{
    return 0;
}

Span<RayIndex>
GuidedPTRenderer::DoRenderPass(uint32_t sppLimit,
                               const GPUQueue& processQueue)
{
    assert(sppLimit != 0);

    // Create RNG state for each ray
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    rnGenerator->SetupRange(imageTiler.LocalTileStart(),
                            imageTiler.LocalTileEnd(),
                            processQueue);

    RayState dRayState = RayState
    {
        .dPathRadiance          = dPathRadiance,
        .dImageCoordinates      = dImageCoordinates,
        .dFilmFilterWeights     = dFilmFilterWeights,
        .dPathWavelengths       = dPathWavelengths,
        //
        .dThroughputs           = dThroughputs,
        .dPathDataPack          = dPathDataPack,
        .dPrevPDF               = dPrevPDF,
        .dPrevPathReflectanceOrOutRadiance
                                = dPrevPathReflectanceOrOutRadiance,
        .dScoreSums             = dScoreSums,
        //
        .dBackupRNGStates       = dBackupRNGStates,
        //
        .dShadowRayRadiance     = dShadowRayRadiance,
        .dShadowRays            = dShadowRays,
        .dShadowRayCones        = dShadowRayCones,
        .dShadowPrevPathReflectance
                                = dShadowPrevPathReflectance,
        //
        .dLiftedMCIndices       = dLiftedMCIndices
    };
    //
    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    auto lightSampler = UniformLightSampler(metaLightArray.Array(),
                                            metaLightArray.IndexHashTable());
    GlobalState globalState = GlobalState
    {
        .russianRouletteRange = currentOptions.russianRouletteRange,
        .specContextData      = typedSpectrumContext.GetData(),
        .lightSampler         = lightSampler,
        .lobeProbability      = Float(0.5),
        // Hash Grid and Data
        .hashGrid             = hashGrid.View(),
        .dMCStates            = dMCStates,
        .dMCCounts            = dMCCounts,
        .dMCLocks             = dMCLocks,
        .dMCIrradiances       = dMCIrradiances
    };

    // Fill dead rays according to the render mode
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto [dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                  processQueue, true);
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);

    // Reload dead paths with new
    auto [dReloadIndices, dFilledIndices, aliveRayCount] = ReloadPaths(dIndices, sppLimit,
                                                                       processQueue);
    dIndices = dReloadIndices.subspan(0, aliveRayCount);
    dKeys = dKeys.subspan(0, aliveRayCount);

    // New rays do not have back path segment so set it
    processQueue.IssueWorkKernel<KCInitGuidedPathStatesIndirect>
    (
        "KCInitGuidedPathStatesIndirect",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dFilledIndices.size())},
        //
        dLiftedMCIndices,
        ToConstSpan(dFilledIndices)
    );

    // From this point on we have full buffers,
    // unless we are about to reach the spp limit
    // them some rays are marked invalid.
    // ============ //
    // Ray Casting  //
    // ============ //
    processQueue.IssueWorkKernel<KCSetBoundaryWorkKeysIndirect>
    (
        "KCSetBoundaryWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        dHitKeys,
        ToConstSpan(dIndices),
        this->boundaryLightKeyPack
    );
    // Actual Ray Casting
    tracerView.baseAccelerator.CastRays
    (
        dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, processQueue
    );
    // Generate work keys from hit packs
    // for partitioning
    using namespace std::string_literals;
    processQueue.IssueWorkKernel<KCGenerateWorkKeysIndirect>
    (
        "KCGenerateWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        dKeys,
        ToConstSpan(dIndices),
        ToConstSpan(dHitKeys),
        workHasher
    );
    // Finally, partition using the generated keys.
    // Fully partitioning here by using a single sort
    auto& rp = rayPartitioner;
    auto partitionOutput = rp.MultiPartition(dKeys, dIndices,
                                             workHasher.WorkBatchDataRange(),
                                             workHasher.WorkBatchBitRange(),
                                             processQueue, false);
    // Wait for results to be available in host buffers
    // since we need partition ranges on the CPU to Issue kernels.
    processQueue.Barrier().Wait();
    // Old Indices array (and the key) is invalidated
    // due to how partitioner works.
    // Change indices to the partitioned one
    dIndices = partitionOutput.dPartitionIndices;


    // Before calcuating new path, backpropogate the irradiance
    // for non-light hitting rays.
    // We could've does this as a "Shader" but we can get away with
    // tMM as position, and ray direction as normal
    //
    // Clear locks
    // TODO: Memset or indirect update via kernel? (which one as less elements?)
    processQueue.MemsetAsync(dMCLocks, 0x00);
    // Indirect irradiance backpropogation (as well as updater selection)
    processQueue.IssueWorkKernel<KCBackpropagateIrradiance>
    (
        "KCBackpropagateIrradiance",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        //
        dPrevPathReflectanceOrOutRadiance,
        globalState,
        dBackupRNGStates,
        dLiftedMCIndices,
        dScoreSums,
        dHitKeys,
        dRays,
        dIndices
    );
    processQueue.IssueWorkKernel<KCUpdateMarkovChains>
    (
        "KCUpdateMarkovChains",
        DeviceWorkIssueParams{.workCount = uint32_t(dIndices.size())},
        //
        globalState,
        dPrevPathReflectanceOrOutRadiance,
        dLiftedMCIndices,
        dHitKeys,
        dRays,
        dIndices
    );

    // TODO: Do we need this?
    // Clear the shadow ray stuff
    processQueue.MemsetAsync(dShadowRayRadiance, 0x00);
    processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
    processQueue.MemsetAsync(dShadowRays, 0x00);

    //
    IssueWorkKernelsToPartitions<This>
    (
        workHasher, partitionOutput,
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

            // 1. vMF Generation & Sampling / Shadow Ray Generation
            // 2. Virtually lifting a Markov Chain
            // 3. Path & Shadow ray single-channel irradiance
            //    throughput generation
            //
            workI.DoWork_0(dRayState, dRays, dRayCones,
                           dLocalIndices, dRandomNumBuffer,
                           dHits, dHitKeys,
                           globalState, processQueue);
        },
        // Light selection
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            // Light emission calculation / direct backpropogation
            // of irradiance
            workI.DoBoundaryWork_0(dRayState, dRays, dRayCones,
                                   dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dHits, dHitKeys,
                                   globalState, processQueue);
        }
    );

    // Check the shadow ray visibility
    Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
    tracerView.baseAccelerator.CastVisibilityRays
    (
        dIsVisibleBitSpan, dBackupRNGStates,
        dShadowRays, dIndices, processQueue
    );

    // Accumulate the pre-calculated radiance selectively
    processQueue.IssueWorkKernel<KCAccumulateShadowRays>
    (
        "KCAccumulateShadowRays",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        //
        dPathRadiance,
        ToConstSpan(dShadowRayRadiance),
        ToConstSpan(dIsVisibleBitSpan),
        ToConstSpan(dPathDataPack),
        currentOptions.russianRouletteRange
    );

    return dIndices;
}

RendererOutput
GuidedPTRenderer::DoThroughputSingleTileRender(const GPUDevice& device,
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
    //CopyAliveRays(dAliveRayIndices, processQueue);

    // We exhausted all alive rays while doing SPP limit.
    bool triggerSave = (saveImage && tilePathCounts[0] == TotalSampleLimit(currentOptions.totalSPP));
    if(triggerSave) saveImage = false;

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
    //analyticData.customLogicSize0 = uint32_t(PathTraceRDetail::SampleMode::E::END);
    return RendererOutput
    {
        .analytics = std::move(analyticData),
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}


RendererOutput
GuidedPTRenderer::DoLatencyRender(uint32_t passCount,
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
        //CopyAliveRays(dAliveRayIndices, processQueue);

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
    //analyticData.customLogicSize0 = uint32_t(PathTraceRDetail::SampleMode::E::END);

    return RendererOutput
    {
        .analytics   = std::move(analyticData),
        .imageOut    = renderOut,
        .triggerSave = triggerSave
    };
}

RenderBufferInfo
GuidedPTRenderer::StartRender(const RenderImageParams& rIP,
                              CamSurfaceId camSurfId,
                              uint32_t customLogicIndex0,
                              uint32_t)
{
    currentOptions = newOptions;
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);

    // Change the mode according to the render logic
    //using Math::Roll;
    //int32_t modeIndex = (int32_t(SampleMode::E(currentOptions.sampleMode)) +
    //                     int32_t(customLogicIndex0));
    //uint32_t sendMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
    //                                  int32_t(SampleMode::E::END)));
    //uint32_t newMode = uint32_t(Roll(modeIndex, 0, int32_t(SampleMode::E::END)));
    //currentOptions.sampleMode = SampleMode::E(newMode);

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
            using SC = SpectrumContext;
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
    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount);
    uint32_t hashGridEntryCount = hashGrid.EntryCapacity();
    uint32_t isVisibleIntCount = Bitspan<uint32_t>::CountT(maxRayCount);
    MemAlloc::AllocateMultiData
    (
        Tie
        (
            // Per path
            dHits, dHitKeys, dRays, dRayCones,
            dPathRadiance, dImageCoordinates,
            dFilmFilterWeights, dThroughputs,
            dPathDataPack, dPrevPDF,
            dShadowPrevPathReflectance,
            dPrevPathReflectanceOrOutRadiance,
            dScoreSums, dLiftedMCIndices,
            dShadowRays, dShadowRayCones,
            dShadowRayRadiance, dPathRNGDimensions,
            // Per path (but can be zero if not spectral)
            dPathWavelengths,
            dSpectrumWavePDFs,
            // Per path but a single bit is used
            dShadowRayVisibilities,
            // Per path bounce, max used random number of that bounce
            dRandomNumBuffer,
            // Per render work
            dWorkHashes, dWorkBatchIds,
            // Hash Grid
            dMCStates, dMCLocks,
            dMCCounts, dMCIrradiances,
            //
            dSubCameraBuffer
        ),
        rendererGlobalMem,
        {
            // Per path
            maxRayCount, maxRayCount, maxRayCount, maxRayCount,
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
            // Hash Grid
            hashGridEntryCount, hashGridEntryCount,
            hashGridEntryCount, hashGridEntryCount,
            //
            RendererBase::SUB_CAMERA_BUFFER_SIZE
        }
    );
    MetaLightListConstructionParams mlParams =
    {
        .lightGroups = tracerView.lightGroups,
        .transformGroups = tracerView.transGroups,
        .lSurfList = Span<const Pair<LightSurfaceId, LightSurfaceParams>>(tracerView.lightSurfs)
    };
    metaLightArray.Construct(mlParams, tracerView.boundarySurface, queue);

    // ===================== //
    // After Allocation Init //
    // ===================== //
    workHasher = InitializeHashes(dWorkHashes, dWorkBatchIds,
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
        .curRenderLogic0 = std::numeric_limits<uint32_t>::max(),//
        .curRenderLogic1 = std::numeric_limits<uint32_t>::max()
    };
}


void GuidedPTRenderer::StopRender()
{
    ClearAllWorkMappings();
    filmFilter = {};
    rnGenerator = {};
    metaLightArray.Clear();
}

std::string_view
GuidedPTRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "GuidedPathTracerSpectral"sv;
    return RendererTypeName<Name>;
}

typename GuidedPTRenderer::AttribInfoList
GuidedPTRenderer::StaticAttributeInfo()
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

size_t GuidedPTRenderer::GPUMemoryUsage() const
{
    size_t total = (rayPartitioner.GPUMemoryUsage() +
                    rnGenerator->GPUMemoryUsage() +
                    rendererGlobalMem.Size());
    if(spectrumContext)
        total += spectrumContext->GPUMemoryUsage();

    return total;
}