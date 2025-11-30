#include "PathTracerRenderer.h"

#include "Core/MemAlloc.h"
#include "Core/Timer.h"
#include "Tracer/RendererCommon.h"

#include "Device/GPUAlgBinaryPartition.h"

#include <numeric>

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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCAccumulateShadowRaysPTMedia(MRAY_GRID_CONSTANT const Span<Spectrum>,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum>,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum>,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum>,
                                   MRAY_GRID_CONSTANT const Bitspan<const uint32_t>,
                                   MRAY_GRID_CONSTANT const Span<const PathDataPack>,
                                   MRAY_GRID_CONSTANT const Vector2ui)
{
    // TODO:
    assert(false);
    //KernelCallParams kp;
    //uint32_t shadowRayCount = static_cast<uint32_t>(dShadowRayRadiance.size());
    //for(uint32_t i = kp.GlobalId(); i < shadowRayCount; i += kp.TotalSize())
    //{
    //    PathDataPack dataPack = dPathDataPack[i];

    //    using enum RayType;
    //    bool isShadowRay = (dataPack.type == SHADOW_RAY);
    //    // +2 is correct here, we did not increment the depth yet
    //    bool inDepthLimit = ((dataPack.depth + 2u) <= rrRange[1]);
    //    if(inDepthLimit && isShadowRay && dIsVisibleBuffer[i])
    //    {
    //        // Unlike simple PT, radiance does not hold the
    //        //
    //        dRadianceOut[i] += dShadowRayRadiance[i];
    //    }
    //}
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
PathTracerRendererT<SC>::DoRenderPassPure(Span<RayIndex> dIndices,
                                          Span<CommonKey> dKeys,
                                          const GPUQueue& processQueue)
{
    // Execution diagram (simplified).
    // This one is simple, just given for completeness.
    //
    // Rays  Ray Cast    Mat. Scatter
    //  |       |           |
    //  |       |           |
    //  |  -->  | -[Prt.]-> |      -------> NEXT
    //  |       |           |
    //  |       |           |
    //  |       |           |

    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    RayState dRayState =
    {
        .dPathRadiance      = dPathRadiance,
        .dImageCoordinates  = dImageCoordinates,
        .dFilmFilterWeights = dFilmFilterWeights,
        .dThroughput        = dThroughputs,
        .dPathDataPack      = dPathDataPack,
        .dPathWavelengths   = dPathWavelengths,
        .dShadowRays        = dShadowRays,
        .dShadowRayCones    = dShadowRayCones,
        .dShadowRayRadiance = dShadowRayRadiance,
        .dPrevMatPDF        = dPrevMatPDF,

    };

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
    // Repurpose random number buffer volume indices
    tracerView.baseAccelerator.CastRays
    (
        Span<VolumeIndex>(),
        dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, false, processQueue
    );

    // Generate work keys from hit packs
    processQueue.IssueWorkKernel<KCGenerateSurfaceWorkKeysIndirect>
    (
        "KCGenerateSurfaceWorkKeysIndirect",
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

    IssueSurfaceWorkKernelsToPartitions<This>
    (
        surfaceWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                                dLocalIndices, workI.SampleRNList(0),
                                rnGenerator, processQueue);
            workI.DoWork_0(dRayState, dRays,
                            dRayCones, dLocalIndices,
                            dRandomNumBuffer, dHits,
                            dHitKeys, globalState,
                            processQueue);
        },
        //
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            workI.DoBoundaryWork_0(dRayState,
                                    dRays, dRayCones,
                                    dLocalIndices,
                                    Span<const RandomNumber>{},
                                    dHits, dHitKeys,
                                    globalState, processQueue);
        }
    );
    return dIndices;
}

template<SpectrumContextC SC>
Span<RayIndex>
PathTracerRendererT<SC>::DoRenderPassNEE(Span<RayIndex> dIndices,
                                         Span<CommonKey> dKeys,
                                         const GPUQueue& processQueue)
{
    // Execution Diagram (simplified).
    // Rays  Ray Cast   Shadow R. Gen   Shadow R. Cast  Mat Scatter
    //  |       |             |                |             |
    //  |       |             |                |             |
    //  |  -->  | -[Prt.]->   |      -->       |     -->     |     -------> NEXT
    //  |       |             |                |             |
    //  |       |             |                |             |
    //  |       |             |                |             |

    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    RayState dRayState =
    {
        .dPathRadiance = dPathRadiance,
        .dImageCoordinates = dImageCoordinates,
        .dFilmFilterWeights = dFilmFilterWeights,
        .dThroughput = dThroughputs,
        .dPathDataPack = dPathDataPack,
        .dPathWavelengths = dPathWavelengths,
        .dShadowRays = dShadowRays,
        .dShadowRayCones = dShadowRayCones,
        .dShadowRayRadiance = dShadowRayRadiance,
        .dPrevMatPDF = dPrevMatPDF,

    };

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
    // Generate work keys from hit packs
    processQueue.IssueWorkKernel<KCGenerateSurfaceWorkKeysIndirect>
    (
        "KCGenerateSurfaceWorkKeysIndirect",
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

    // ================================== //
    //  Path Tracing with NEE and/or MIS  //
    // ================================== //
    // Work_0           = BxDF sample
    // Work_1           = NEE sample only
    // BoundaryWork_1   = Same as light accumulation but with many states
    //                    regarding NEE and MIS
    // ================================== //
    //  Sample Light and Gen. Shadow Ray  //
    // ================================== //
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
    // Clear the shadow ray radiance buffer
    processQueue.MemsetAsync(dShadowRayRadiance, 0x00);
    // CUDA Init check error, we access the rays even if it is not written
    processQueue.MemsetAsync(dShadowRays, 0x00);
    // Do the NEE kernel + boundary work
    IssueSurfaceWorkKernelsToPartitions<This>
    (
        surfaceWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                                dLocalIndices, workI.SampleRNList(1),
                                rnGenerator, processQueue);
            workI.DoWork_1(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRandomNumBuffer, dHits,
                           dHitKeys, globalState,
                           processQueue);
        },
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            workI.DoBoundaryWork_1(dRayState,  dRays,
                                    dRayCones, dLocalIndices,
                                    Span<const RandomNumber>{},
                                    dHits, dHitKeys,
                                    globalState, processQueue);
        }
    );

    // ================================== //
    //     Shadow Ray Visibility Check    //
    // ================================== //
    // If media is not tracked, do a simple visibility check.
    // If it is being tracked, oh boy...
    processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
    Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
    tracerView.baseAccelerator.CastVisibilityRays
    (
        dIsVisibleBitSpan, dBackupRNGStates,
        dShadowRays, dIndices, processQueue
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

    // ================================== //
    //     Scatter Rays via Material      //
    // ================================== //
    using GlobalStateE = PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>;
    GlobalStateE globalStateE
    {
        .russianRouletteRange = currentOptions.russianRouletteRange,
        .sampleMode = currentOptions.sampleMode,
        .lightSampler = EmptyType{},
        .specContextData = typedSpectrumContext.GetData()
    };
    // Do the actual kernel
    IssueSurfaceWorkKernelsToPartitions<This>
    (
        surfaceWorkHasher, partitionOutput,
        [&](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                                dLocalIndices, workI.SampleRNList(0),
                                rnGenerator, processQueue);
            workI.DoWork_0(dRayState, dRays,
                            dRayCones, dLocalIndices,
                            dRandomNumBuffer,
                            dHits, dHitKeys,
                            globalStateE, processQueue);
        },
        // Empty Invocation for lights this pass
        [&](const auto&, Span<uint32_t>, uint32_t) {}
    );
    return dIndices;
}

template<SpectrumContextC SC>
Span<RayIndex>
PathTracerRendererT<SC>::DoRenderPassWithMediaPure(Span<RayIndex> dIndices,
                                                   Span<CommonKey> dKeys,
                                                   const GPUQueue& processQueue)
{
    // Execution diagram. (simplified and hopefully it does clarify instead of
    // confuse)
    //
    // Rays     Media Resolve                   Mat. Scatter
    //  |            |                 |             |     |
    //  |            |  [Transmitted]  |  -[Prt]->   | --> |
    //  |  -[Prt]->  |_________________|_____________|     | ---->  NEXT
    //  |            |                                     |
    //  |            |       [Media Scattered]         --> |
    //  |            |                                     |
    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    RayState dRayState =
    {
        .dPathRadiance = dPathRadiance,
        .dImageCoordinates = dImageCoordinates,
        .dFilmFilterWeights = dFilmFilterWeights,
        .dThroughput = dThroughputs,
        .dPathDataPack = dPathDataPack,
        .dPathWavelengths = dPathWavelengths,
        .dShadowRays = dShadowRays,
        .dShadowRayCones = dShadowRayCones,
        .dShadowRayRadiance = dShadowRayRadiance,
        .dPrevMatPDF = dPrevMatPDF,

    };

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
    // Repurpose random number buffer volume indices
    Span<VolumeIndex> dVolumeIndices = MemAlloc::RepurposeAlloc<VolumeIndex>(dRandomNumBuffer);
    tracerView.baseAccelerator.CastRays
    (
        dVolumeIndices, dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, true,
        processQueue
    );
    mediaTracker->AddNewVolumeToRaysIndirect(dRayMediaListPacks,
                                             dVolumeIndices,
                                             dIndices,
                                             processQueue);
    // Generate work keys from hit packs
    processQueue.IssueWorkKernel<KCGenerateMediumWorkKeysIndirect>
    (
        "KCGenerateMediumWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dKeys,
        ToConstSpan(dIndices),
        ToConstSpan(dRayMediaListPacks),
        mediaTracker->View(),
        mediumWorkHasher
    );

    // N-way Partition wrt. medium/transform pair
    auto& rp = rayPartitioner;
    auto partitionOutput = rp.MultiPartition(dKeys, dIndices,
                                             mediumWorkHasher.WorkBatchDataRange(),
                                             mediumWorkHasher.WorkBatchBitRange(),
                                             processQueue, false);
    processQueue.Barrier().Wait();
    // Call Media Transmit
    // Repurpose shadow ray visiblity bit buffer
    // for media scatter events
    Bitspan<uint32_t> dIsScatteredBitSpan(dShadowRayVisibilities);
    IssueMediumWorkKernelsToPartitions<This>
    (
        mediumWorkHasher, partitionOutput,
        [&, this](const auto&, Span<uint32_t>, uint32_t)
        {
            assert(false);
            //FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
            //                 dLocalIndices, workI.SampleRNList(0),
            //                 rnGenerator, processQueue);
            //workI.DoWork_0(dRayState, dRays,
            //               dRayCones, dLocalIndices,
            //               dRandomNumBuffer, dHits,
            //               dHitKeys, globalState,
            //               processQueue);
        }
    );
    // Rename output buffers as input, since multi partition can change it
    dKeys = partitionOutput.dPartitionKeys;
    dIndices = partitionOutput.dPartitionIndices;
    // Binary Partition wrt. scatter/not scatter event
    auto bpOut = rp.BinaryPartition(dIndices,
                                    processQueue,
                                    IsBitSetFunctor(dIsScatteredBitSpan, true));

    auto dScatteredIndices = bpOut.Spanify()[1];
    // Scattered paths due to media interaction is handled here
    mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                  dScatteredIndices,
                                                  processQueue);

    // Generate work keys from hit packs
    auto dTransmittedIndices = bpOut.Spanify()[0];
    processQueue.IssueWorkKernel<KCGenerateSurfaceWorkKeysIndirect>
    (
        "KCGenerateSurfaceWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dKeys,
        ToConstSpan(dTransmittedIndices),
        ToConstSpan(dHitKeys),
        surfaceWorkHasher
    );

    // Finally, partition using the generated keys.
    // Fully partitioning here by using a single sort
    partitionOutput = rp.MultiPartition(dKeys, dTransmittedIndices,
                                        surfaceWorkHasher.WorkBatchDataRange(),
                                        surfaceWorkHasher.WorkBatchBitRange(),
                                        processQueue, false);
    // Wait for results to be available in host buffers
    // since we need partition ranges on the CPU to Issue kernels.
    processQueue.Barrier().Wait();
    // Old Indices array (and the key) is invalidated
    // Change indices to the partitioned one
    dTransmittedIndices = partitionOutput.dPartitionIndices;

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

    IssueSurfaceWorkKernelsToPartitions<This>
    (
        surfaceWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                                dLocalIndices, workI.SampleRNList(0),
                                rnGenerator, processQueue);
            workI.DoWork_0(dRayState, dRays,
                            dRayCones, dLocalIndices,
                            dRandomNumBuffer, dHits,
                            dHitKeys, globalState,
                            processQueue);
        },
        //
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            workI.DoBoundaryWork_0(dRayState,
                                    dRays, dRayCones,
                                    dLocalIndices,
                                    Span<const RandomNumber>{},
                                    dHits, dHitKeys,
                                    globalState, processQueue);
        }
    );

    // We scattered via material
    // resolve the next media of the path
    //
    // Material Kernels should've updated "dRayMediaListPacks"
    // accordingly (isPassedThrough bit should set or not etc.)
    mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                  dTransmittedIndices,
                                                  processQueue);
    return dIndices;
}

template<SpectrumContextC SC>
Span<RayIndex>
PathTracerRendererT<SC>::DoRenderPassWithMediaNEE(Span<RayIndex> dIndices,
                                                  Span<CommonKey> dKeys,
                                                  const GPUQueue& processQueue)
{
    // Execution diagram. (simplified and hopefully it does clarify instead of
    // confuse)
    //
    // Rays     Media Resolve                     SR Cast      Mat. Scatter
    //  |            |                           (Recursive)        |        |
    //  |            |                 |              |             |        |
    //  |            |  [Transmitted]  |   -[Prt]->   | ----------> | -----> |
    //  |  -[Prt]->  |_________________|______________|_____________|        | -->  NEXT
    //  |            |                                |                      |
    //  |            |     --[Media Scattered]-->     |         ----->       |
    //  |            |                                |                      |
    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    RayState dRayState =
    {
        .dPathRadiance = dPathRadiance,
        .dImageCoordinates = dImageCoordinates,
        .dFilmFilterWeights = dFilmFilterWeights,
        .dThroughput = dThroughputs,
        .dPathDataPack = dPathDataPack,
        .dPathWavelengths = dPathWavelengths,
        .dShadowRays = dShadowRays,
        .dShadowRayCones = dShadowRayCones,
        .dShadowRayRadiance = dShadowRayRadiance,
        .dPrevMatPDF = dPrevMatPDF,

    };

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
    // Repurpose random number buffer volume indices
    Span<VolumeIndex> dVolumeIndices = MemAlloc::RepurposeAlloc<VolumeIndex>(dRandomNumBuffer);
    tracerView.baseAccelerator.CastRays
    (
        dVolumeIndices,
        dHitKeys, dHits, dBackupRNGStates,
        dRays, dIndices, true,
        processQueue
    );

    mediaTracker->AddNewVolumeToRaysIndirect(dRayMediaListPacks,
                                             dVolumeIndices,
                                             dIndices,
                                             processQueue);
    // Generate work keys from hit packs
    processQueue.IssueWorkKernel<KCGenerateMediumWorkKeysIndirect>
    (
        "KCGenerateMediumWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dKeys,
        ToConstSpan(dIndices),
        ToConstSpan(dRayMediaListPacks),
        mediaTracker->View(),
        mediumWorkHasher
    );

    // N-way Partition wrt. medium/transform pair
    auto& rp = rayPartitioner;
    auto partitionOutput = rp.MultiPartition(dKeys, dIndices,
                                             mediumWorkHasher.WorkBatchDataRange(),
                                             mediumWorkHasher.WorkBatchBitRange(),
                                             processQueue, false);
    processQueue.Barrier().Wait();

    // Call Media Transmit
    // Repurpose shadow ray visiblity bit buffer
    // for media scatter events
    Bitspan<uint32_t> dIsScatteredBitSpan(dShadowRayVisibilities);
    IssueMediumWorkKernelsToPartitions<This>
    (
        mediumWorkHasher, partitionOutput,
        [&, this](const auto&, Span<uint32_t>, uint32_t)
        {
            assert(false);
        }
    );
    // Rename output buffers as input, since multi partition can change it
    dKeys = partitionOutput.dPartitionKeys;
    dIndices = partitionOutput.dPartitionIndices;
    // Binary Partition wrt. scatter/not scatter event
    auto bpOut = rp.BinaryPartition(partitionOutput.dPartitionIndices,
                                    processQueue,
                                    IsBitSetFunctor(dIsScatteredBitSpan, true));

    auto dScatteredIndices = bpOut.Spanify()[1];
    // Scattered paths due to media interaction is handled here
    mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                  dScatteredIndices,
                                                  processQueue);

    // Generate work keys from hit packs
    auto dTransmittedIndices = bpOut.Spanify()[0];
    processQueue.IssueWorkKernel<KCGenerateSurfaceWorkKeysIndirect>
    (
        "KCGenerateSurfaceWorkKeysIndirect",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dKeys,
        ToConstSpan(dTransmittedIndices),
        ToConstSpan(dHitKeys),
        surfaceWorkHasher
    );

    // Finally, partition using the generated keys.
    // Fully partitioning here by using a single sort
    partitionOutput = rp.MultiPartition(dKeys, dTransmittedIndices,
                                        surfaceWorkHasher.WorkBatchDataRange(),
                                        surfaceWorkHasher.WorkBatchBitRange(),
                                        processQueue, false);
    // Wait for results to be available in host buffers
    // since we need partition ranges on the CPU to Issue kernels.
    processQueue.Barrier().Wait();
    // Old Indices array (and the key) is invalidated
    // Change indices to the partitioned one
    dTransmittedIndices = partitionOutput.dPartitionIndices;

    // ================================== //
    //  Path Tracing with NEE and/or MIS  //
    // ================================== //
    // Work_0           = BxDF sample
    // Work_1           = NEE sample only
    // BoundaryWork_1   = Same as light accumulation but with many states
    //                    regarding NEE and MIS
    // ================================== //
    //  Sample Light and Gen. Shadow Ray  //
    // ================================== //
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
    // Clear the shadow ray radiance buffer
    processQueue.MemsetAsync(dShadowRayRadiance, 0x00);
    // CUDA Init check error, we access the rays even if it is not written
    processQueue.MemsetAsync(dShadowRays, 0x00);
    // Do the NEE kernel + boundary work
    IssueSurfaceWorkKernelsToPartitions<This>
    (
        surfaceWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                                dLocalIndices, workI.SampleRNList(1),
                                rnGenerator, processQueue);
            workI.DoWork_1(dRayState, dRays,
                            dRayCones, dLocalIndices,
                            dRandomNumBuffer,  dHits,
                            dHitKeys, globalState,
                            processQueue);
        },
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            workI.DoBoundaryWork_1(dRayState,  dRays,
                                    dRayCones, dLocalIndices,
                                    Span<const RandomNumber>{},
                                    dHits, dHitKeys,
                                    globalState, processQueue);
        }
    );

    // ================================== //
    //     Shadow Ray Visibility Check    //
    // ================================== //
    // If media is not tracked, do a simple visibility check.
    // If it is being tracked, oh boy...
    processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
    Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
    if(currentOptions.sampleMedia)
    {

        // We should be fine now.


        // Again rn buffer to the rescue. Use it as a temporary buffer
        // for shadow rays' media pack.
        //
        Span<RayMediaListPack> dShadowRayMediaListPack
            = MemAlloc::RepurposeAlloc<RayMediaListPack>(dRandomNumBuffer);

        // TODO: Copy all of the rays
        assert(false);


        //
        RecursiveShadowRayCast(dIsVisibleBitSpan, dBackupRNGStates,
                                dIndices, processQueue);

        // TODO: Accumulate shadow ray radiance
        assert(false);
    }

    // ================================== //
    //     Scatter Rays via Material      //
    // ================================== //
    using namespace PathTraceRDetail;
    using GlobalStateE = PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>;
    GlobalStateE globalStateE
    {
        .russianRouletteRange = currentOptions.russianRouletteRange,
        .sampleMode = currentOptions.sampleMode,
        .lightSampler = EmptyType{},
        .specContextData = typedSpectrumContext.GetData()
    };
    // Do the actual kernel
    IssueSurfaceWorkKernelsToPartitions<This>
    (
        surfaceWorkHasher, partitionOutput,
        [&](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                             dLocalIndices, workI.SampleRNList(0),
                             rnGenerator, processQueue);
            workI.DoWork_0(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRandomNumBuffer,
                           dHits, dHitKeys,
                           globalStateE, processQueue);
        },
        // Empty Invocation for lights this pass
        [&](const auto&, Span<uint32_t>, uint32_t) {}
    );

    mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                  dTransmittedIndices,
                                                  processQueue);

    // So we "restart" the partitioner and get fresh dIndices array.
    // We need to be careful since fresh array will have
    // invalid rays (due to we are about to reach spp limit and we did not
    // reload all paths that can fill the buffer).
    //
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto p = rayPartitioner.Start(rayCount, maxWorkCount,
                                  processQueue, true);
    DeviceAlgorithms::Iota(p.dIndices, RayIndex(0), processQueue);
    dIndices = p.dIndices;

    return dIndices;
}

template<SpectrumContextC SC>
Span<RayIndex>
PathTracerRendererT<SC>::DoRenderPass(uint32_t sppLimit, const GPUQueue& processQueue)
{
    assert(sppLimit != 0);
    // Find the ray count. Ray count is tile count
    // but tile can exceed film boundaries so clamp,
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();
    // Start the partitioner, again worst case work count
    // Get the K/V pair buffer
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto [dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
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

    // After 2 more modes (Media NEE, Media Pure),
    // I've split the function into multiple types.
    // It will be easier to understand I hope.
    if(currentOptions.sampleMode == SampleMode::E::PURE &&
       currentOptions.sampleMedia == false)
    {
        return DoRenderPassPure(dIndices, dKeys, processQueue);
    }
    if(currentOptions.sampleMode != SampleMode::E::PURE &&
       currentOptions.sampleMedia == false)
    {
        return DoRenderPassNEE(dIndices, dKeys, processQueue);
    }
    if(currentOptions.sampleMode == SampleMode::E::PURE &&
       currentOptions.sampleMedia == true)
    {
        return DoRenderPassWithMediaPure(dIndices, dKeys, processQueue);
    }
    if(currentOptions.sampleMode != SampleMode::E::PURE &&
       currentOptions.sampleMedia == true)
    {
        return DoRenderPassWithMediaNEE(dIndices, dKeys, processQueue);
    }
    return Span<RayIndex>();
}

template<SpectrumContextC SC>
void
PathTracerRendererT<SC>::RecursiveShadowRayCast(// Output
                                                Bitspan<uint32_t>,
                                                // I-O
                                                Span<BackupRNGState>,
                                                // Input
                                                Span<const RayIndex>,
                                                const GPUQueue&)
{
    //Span<RayGMem> dShadowRays,
    //Span<RayCone> dShadowRayCones,

    // We need to hold shadow ray's media indices
    // We also need storage for volume indices
    //
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
                dPathDataPack, dPathRNGDimensions,
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
                maxRayCount, maxRayCount,
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
                dPathDataPack, dShadowRays, dShadowRayCones,
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
