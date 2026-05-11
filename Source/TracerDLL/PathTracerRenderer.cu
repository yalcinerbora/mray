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
static
void KCAccumulateShadowRaysMediaPT(MRAY_GRID_CONSTANT const Span<Spectrum> dRadianceOut,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum> dShadowRayRadiance,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum> dRPathPDFShadow,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum> dRLightPDFShadow,
                                   MRAY_GRID_CONSTANT const Bitspan<const uint32_t> dIsVisibleBuffer,
                                   MRAY_GRID_CONSTANT const Span<const PathDataPack> dPathDataPack,
                                   MRAY_GRID_CONSTANT const Vector2ui rrRange,
                                   MRAY_GRID_CONSTANT const bool isRGB)
{
    const uint32_t channelCount = isRGB ? 3 : SpectraPerSpectrum;
    const Float invChannelCount = Float(1) / Float(channelCount);

    KernelCallParams kp;
    uint32_t shadowRayCount = static_cast<uint32_t>(dShadowRayRadiance.size());
    for(uint32_t i = kp.GlobalId(); i < shadowRayCount; i += kp.TotalSize())
    {
        PathDataPack dataPack = dPathDataPack[i];

        using enum RayType;
        bool isShadowRay = (dataPack.type == SHADOW_RAY);
        bool inDepthLimit = ((dataPack.depth + 1u) <= rrRange[1]);
        if(inDepthLimit && isShadowRay && dIsVisibleBuffer[i])
        {
            Float rPathMIS = Float(0);
            Float rLightMIS = Float(0);
            Spectrum rPath = dRPathPDFShadow[i];
            Spectrum rLight = dRPathPDFShadow[i];
            for(uint32_t c = 0; c < channelCount; c++)
            {
                rPathMIS += rPath[c];
                rLightMIS += rLight[c];
            }
            rPathMIS *= invChannelCount;
            rLightMIS *= invChannelCount;

            Float combinedMIS = rPathMIS + rLightMIS;

            using Distribution::Common::DivideByPDF;
            Spectrum radiance = dShadowRayRadiance[i];
            radiance = DivideByPDF(dShadowRayRadiance[i], combinedMIS);

            dRadianceOut[i] += radiance;
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCInitializePDFRatiosIndirect(MRAY_GRID_CONSTANT const Span<Spectrum> dRPath,
                                   MRAY_GRID_CONSTANT const Span<Spectrum> dRLight,
                                   MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    KernelCallParams kp;
    uint32_t filledRayCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < filledRayCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        dRPath[index] = Spectrum(1);
        dRLight[index] = Spectrum(1);
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
    result.attributes.push_back(TransientData(std::in_place_type_t<bool>{}, 1));
    result.attributes.back().Push(Span<const bool>(&currentOptions.sampleMedia, 1));
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
        case 6: newOptions.sampleMedia = data.AccessAs<bool>()[0]; break;
        default:
            throw MRayError("{} Unknown attribute index {}", TypeName(), attributeIndex);
    }
}

template<SpectrumContextC SC>
uint32_t
PathTracerRendererT<SC>::FindMaxWorkCount() const
{
    uint32_t matWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());

    if(currentOptions.sampleMedia)
    {
        uint32_t mediaWorkCount = uint32_t(currentMediumWorks.size());
        return std::max(matWorkCount, mediaWorkCount);
    }
    else return matWorkCount;
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
    //
    maxSample = std::transform_reduce
    (
        currentMediumWorks.cbegin(), currentMediumWorks.cend(), maxSample,
        [](uint32_t l, uint32_t r) -> uint32_t
        {
            return Math::Max(l, r);
        },
        [sampleMode](const auto& renderMediaWorkStruct) -> uint32_t
        {
            if(sampleMode == PathTraceRDetail::SampleMode::E::PURE)
                return renderMediaWorkStruct.workPtr->SampleRNList(0).TotalRNCount();
            else
                return Math::Max(renderMediaWorkStruct.workPtr->SampleRNList(0).TotalRNCount(),
                                 renderMediaWorkStruct.workPtr->SampleRNList(1).TotalRNCount());
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
        .dMediaListPack     = dRayMediaListPacks
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
        .specContextData = typedSpectrumContext.GetData(),
        .sampleMedia = currentOptions.sampleMedia
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
        .dMediaListPack     = dRayMediaListPacks
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
        .specContextData = typedSpectrumContext.GetData(),
        .sampleMedia = currentOptions.sampleMedia
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
        .russianRouletteRange   = currentOptions.russianRouletteRange,
        .sampleMode             = currentOptions.sampleMode,
        .lightSampler           = EmptyType{},
        .specContextData        = typedSpectrumContext.GetData(),
        .sampleMedia            = currentOptions.sampleMedia
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
        .dPathRadiance      = dPathRadiance,
        .dImageCoordinates  = dImageCoordinates,
        .dFilmFilterWeights = dFilmFilterWeights,
        .dThroughput        = dThroughputs,
        .dPathDataPack      = dPathDataPack,
        .dPathWavelengths   = dPathWavelengths,
        .dBackupRNGStates   = rnGenerator->GetBackupStates(),
        .dMediaListPack     = dRayMediaListPacks

    };
    using GlobalState = PathTraceRDetail::GlobalState<EmptyType, SpectrumConverter>;
    GlobalState globalState
    {
        .russianRouletteRange = currentOptions.russianRouletteRange,
        .sampleMode = currentOptions.sampleMode,
        .lightSampler = EmptyType{},
        .specContextData = typedSpectrumContext.GetData(),
        .sampleMedia = currentOptions.sampleMedia
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
    processQueue.MemsetAsync(dVolumeIndices, 0xFF);

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
    IssueMediumWorkKernelsToPartitions<This>
    (
        mediumWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            // We do not do any work but mark rays as transmitted when media is vacuum.
            if(workI.IsVacuumMedia())
            {
                MarkPathsTransmittedIndirect(dPathDataPack, dLocalIndices, processQueue);
                return;
            }

            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                             dLocalIndices, workI.SampleRNList(0),
                             rnGenerator, processQueue);
            workI.DoWork_0(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRayMediaListPacks,
                           dRandomNumBuffer,
                           mediaTracker->View(),
                           globalState,
                           processQueue);
        }
    );
    // Rename output buffers as input, since multi partition can change it
    dKeys = partitionOutput.dPartitionKeys;
    dIndices = partitionOutput.dPartitionIndices;
    // Binary Partition wrt. scatter/not scatter event
    auto bpOut = rp.BinaryPartition(dIndices, processQueue,
                                    IsMediumTransmittedFunctor(dPathDataPack));
    processQueue.Barrier().Wait();

    auto dScatteredIndices = bpOut.Spanify()[1];
    // Scattered paths due to media interaction is handled here
    if(!dScatteredIndices.empty())
    {
        mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                      dScatteredIndices,
                                                      processQueue);
    }

    // If all the rays are entered a media and did not get transmitted
    // We do not need to do material evaluation, directly skip the rest of
    // the function
    auto dTransmittedIndices = bpOut.Spanify()[0];
    if(dTransmittedIndices.empty())
        return dIndices;

    // Generate work keys from hit packs
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

    // Here we did process media on first N-way partition,
    // (indices buffer is changed). Then, we did another N-way partition
    // to a subportion of the first buffer (transmitted/scattered via media
    // event). Technically indices buffers are a mess we need to restart here
    // and return a proper indices buffer.
    //
    // Caller can do this but non-media path tracer do not need this iota
    // since all the indices are in a single buffer anyway.
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();
    uint32_t maxWorkCount = FindMaxWorkCount();
    auto [dIndicesOut, _] = rp.Start(rayCount, maxWorkCount, processQueue, true);
    DeviceAlgorithms::Iota(dIndicesOut, uint32_t(0), processQueue);

    return dIndicesOut;
}

template<SpectrumContextC SC>
Span<RayIndex>
PathTracerRendererT<SC>::DoRenderPassWithMediaNEE(Span<RayIndex> dIndices,
                                                  Span<CommonKey> dKeys,
                                                  Span<const RayIndex> dFilledRayIndices,
                                                  const GPUQueue& processQueue)
{
    throw MRayError("Media-Enabled Path Tracer with NEE is not yet implemented!");

    // Execution diagram. (simplified and hopefully it does clarify instead of
    // confuse)
    //
    // Med. = Media
    // Mat. = Material
    //
    // Rays   Media Resolve          Mat. Partition &             Recursive
    //  |          |                      Scatter              Shadow Ray Cast
    //  |          |                         |                       |
    //  |          | --[Med. Transmitted]--> | --[Mat. Scattered]--> |
    //  | -[Prt]-> |_________________________|_______________________|   ----->  NEXT
    //  |          |                                                 |
    //  |          |               --[Med. Scattered]-->             |
    //  |          |                                                 |
    //
    // All Media will cast shadow rays, but not all materials may cast shadow rays
    // such as perfectly specular materials (mirror) or "Passthrough" material.
    //
    const SpectrumContext& typedSpectrumContext = *static_cast<const SpectrumContext*>(spectrumContext.get());
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    RayState dRayState =
    {
        .dPathRadiance        = dPathRadiance,
        .dImageCoordinates    = dImageCoordinates,
        .dFilmFilterWeights   = dFilmFilterWeights,
        .dThroughput          = dThroughputs,
        .dPathDataPack        = dPathDataPack,
        .dPathWavelengths     = dPathWavelengths,
        .dShadowRays          = dShadowRays,
        .dShadowRayCones      = dShadowRayCones,
        .dShadowRayRadiance   = dShadowRayRadiance,
        .dRPathPDF            = dRPathPDF,
        .dRLightPDF           = dRLightPDF,
        .dRPathPDFShadow      = dRPathPDFShadow,
        .dRLightPDFShadow     = dRLightPDFShadow,
        .dBackupRNGStates     = dBackupRNGStates,
        .dMediaListPack       = dRayMediaListPacks,
        .dShadowMediaListPack = dShadowRayMediaListPacks
    };

    UniformLightSampler lightSampler(metaLightArray.Array(),
                                     metaLightArray.IndexHashTable());
    using GlobalState = PathTraceRDetail::GlobalState<UniformLightSampler, SpectrumConverter>;
    GlobalState globalState
    {
        .russianRouletteRange = currentOptions.russianRouletteRange,
        .sampleMode           = currentOptions.sampleMode,
        .lightSampler         = lightSampler,
        .specContextData      = typedSpectrumContext.GetData(),
        .sampleMedia          = currentOptions.sampleMedia
    };

    // Fill the PDF ratios for new rays/paths.
    using namespace std::string_view_literals;
    processQueue.IssueWorkKernel<KCInitializePDFRatiosIndirect>
    (
        "KCInitializePDFRatiosIndirect"sv,
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dFilledRayIndices.size())},
        dRPathPDF,
        dRLightPDF,
        ToConstSpan(dFilledRayIndices)
    );

    // Cast rays
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
    processQueue.MemsetAsync(dVolumeIndices, 0xFF);
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
    IssueMediumWorkKernelsToPartitions<This>
    (
        mediumWorkHasher, partitionOutput,
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            FillRandomBuffer(dRandomNumBuffer, dPathRNGDimensions,
                             dLocalIndices, workI.SampleRNList(1),
                             rnGenerator, processQueue);
            workI.DoWork_1(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRayMediaListPacks,
                           dRandomNumBuffer,
                           mediaTracker->View(),
                           globalState,
                           processQueue);
        }
    );
    // Rename output buffers as input, since multi partition can change it
    dKeys = partitionOutput.dPartitionKeys;
    dIndices = partitionOutput.dPartitionIndices;
    // Binary Partition wrt. scatter/not scatter event
    auto bpOut = rp.BinaryPartition(dIndices, processQueue,
                                    IsMediumTransmittedFunctor(dPathDataPack));
    processQueue.Barrier().Wait();

    auto dScatteredIndices = bpOut.Spanify()[1];
    // Scattered paths due to media interaction is handled here
    if(!dScatteredIndices.empty())
    {
        mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                      dScatteredIndices,
                                                      processQueue);
    }

    // If all the rays are entered a media and did not get transmitted
    // We do not need to do material evaluation, directly skip the rest of
    // the function
    auto dTransmittedIndices = bpOut.Spanify()[0];
    if(dTransmittedIndices.empty())
        return dIndices;

    // Generate work keys from hit packs
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

    // ========================================== //
    //  Path Tracing with NEE and MIS, with Media //
    // ========================================== //
    // Unlike other methods, we sample both BxDF and NEE in a single
    // kernel (Kernel #3).
    //
    // We disregard the mode "NEE", and it is calculated as if it is
    // "NEE_WITH_MIS" (States getting ridiculous, with spectral etc.).
    //
    // MediaWork_2      = Sample media scattering
    // MediaWork_3      = Calculate transmittance and track pdfs.
    //                    This will be called recursively while casting shadow rays
    //
    // Work_3           = BxDF/NEE sample on surfaces
    // BoundaryWork_3   = Accumulate light with MIS

    //
    // ===================================== //
    //  Sample Light and Gen. Shadow Ray Gen //
    // ===================================== //
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
            workI.DoWork_2(dRayState, dRays,
                           dRayCones, dLocalIndices,
                           dRandomNumBuffer,  dHits,
                           dHitKeys, globalState,
                           processQueue);
        },
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices, uint32_t)
        {
            workI.DoBoundaryWork_2(dRayState,  dRays,
                                   dRayCones, dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dHits, dHitKeys,
                                   globalState, processQueue);
        }
    );

    // Resolve the media after scatter/transmit event over the surface
    mediaTracker->ResolveCurVolumesOfRaysIndirect(dRayMediaListPacks,
                                                  dTransmittedIndices,
                                                  processQueue);

    // ================================== //
    //     Shadow Ray Visibility Check    //
    // ================================== //
    // Here we need all the rays that are media scattered and material
    // scattered. We partitition multiple times (one for media, and we sub partition
    // the scattered ones wrt. material), so index buffer is kinda mess. These index buffers
    // may reside on different buffers so we restart the indexing.
    //
    // Recursive ray cast will initially filter the shadow ray-requested paths etc.
    // Also it should filter invalid rays, these occur when we are about to reach
    // the spp limit.
    uint32_t maxRayCount = imageTiler.CurrentTileSize().Multiply();
    auto p = rayPartitioner.Start(maxRayCount, FindMaxWorkCount(),
                                  processQueue, true);
    DeviceAlgorithms::Iota(p.dIndices, RayIndex(0), processQueue);
    dIndices = p.dIndices;
    //
    Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
    RecursiveShadowRayCast(dIsVisibleBitSpan, dBackupRNGStates,
                           dIndices, processQueue);

    // Accumulate the pre-calculated radiance selectively
    processQueue.IssueWorkKernel<KCAccumulateShadowRaysMediaPT>
    (
        "KCAccumulateShadowRaysMedia",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dShadowRayRadiance.size())},
        //
        dPathRadiance,
        ToConstSpan(dShadowRayRadiance),
        ToConstSpan(dRPathPDFShadow),
        ToConstSpan(dRLightPDFShadow),
        ToConstSpan(dIsVisibleBitSpan),
        ToConstSpan(dPathDataPack),
        currentOptions.russianRouletteRange,
        std::is_same_v<SC, SpectrumContextIdentity>
    );

    // So we "restart" the partitioner and get fresh dIndices array.
    // We need to be careful since fresh array will have
    // invalid rays (due to "we are about to reach spp limit and we did not
    // reload all paths that can fill the buffer" case).
    //
    p = rayPartitioner.Start(maxRayCount, FindMaxWorkCount(),
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
    uint32_t maxWorkCount = FindMaxWorkCount();
    auto [dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                  processQueue, true);

     // Iota the indices
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
    // Create RNG state for each ray
    rnGenerator->SetupRange(imageTiler.LocalTileStart(),
                            imageTiler.LocalTileEnd(),
                            processQueue);
    // Reload dead paths with new
    auto
    [
        dReloadIndices,
        dFilledRayIndices,
        aliveRayCount
    ] = ReloadPaths(dIndices, sppLimit, processQueue);
    //
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
        return DoRenderPassWithMediaNEE(dIndices, dKeys,
                                        ToConstSpan(dFilledRayIndices),
                                        processQueue);
    }
    return Span<RayIndex>();
}

template<SpectrumContextC SC>
void
PathTracerRendererT<SC>::RecursiveShadowRayCast(// Output
                                                Bitspan<uint32_t> dIsVisibleBuffer,
                                                // I-O
                                                Span<BackupRNGState> dPackupRNGStates,
                                                // Input
                                                Span<const RayIndex> dTransmittedIndices,
                                                const GPUQueue& queue)
{
    assert(false);
    auto& rp = rayPartitioner;

    // TODO: How to check if done or not done?

    // Do:
    //    Binary partition alive shadow rays
    //    1. Do closest hit
    //    2. N-way partition wrt. current media
    //    3. Calculate next media
    //    While doing these, update isVisible bit set and decay shadow ray radiance
    //
    // While: All rays reached/occluded

    //Span<const RayIndex> dCurrentIndices = dTransmittedIndices;
    //do
    //{
    //   auto binPartitionOutput = rp.BinaryPartition
    //   (
    //       dCurrentIndices, queue, []()
    //        {
    //            ...
    //        }
    //   );

    //   dPartitionIndices = ;
    //}
    //while(....);


    //// TODO: Accumulate shadow ray radiance
    //// We will evaluate transmittance
    //// TODO: Copy all of the rays
    //
    //processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
    //Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
    //// Again rn buffer to the rescue. Use it as a temporary buffer
    //// for shadow rays' media pack.
    //using MemAlloc::RepurposeAlloc;
    //auto dShadowRayMediaListPack = RepurposeAlloc<RayMediaListPack>(dRandomNumBuffer);
    //
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
    if(!renderOut.HasValue()) return RendererOutput{};
    // Actual global weight
    renderOut.Value().globalWeight = Float(1);

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

    uint32_t perRayMediaListCount = 0;
    if(currentOptions.sampleMedia)
    {
        perRayMediaListCount = maxRayCount;
        mediaTracker = std::make_unique<MediaTracker>(tracerView.globalVolumeList,
                                                      tracerView.tracerParams.volumeTrackerEntryCount,
                                                      gpuSystem);

        std::vector<const SurfaceVolumeList*> sVolumes;
        sVolumes.reserve(tracerView.surfs.size());
        for(const auto& [_, p] : tracerView.surfs)
            sVolumes.push_back(&p.volumes);

        std::vector<const BoundaryVolumeList*> bVolumes;
        bVolumes.reserve(tracerView.lightSurfs.size() +
                         tracerView.camSurfs.size());
        for(const auto& [_, p] : tracerView.lightSurfs)
            bVolumes.push_back(&p.nestedVolumes);
        for(const auto& [_, p] : tracerView.camSurfs)
            bVolumes.push_back(&p.nestedVolumes);

        mediaTracker->PrimeHashTable(sVolumes, bVolumes,
                                     tracerView.boundaryVolume,
                                     queue);
        queue.Barrier().Wait();

        // Also warn user when the mode is NEE only,
        // since media sample always do MIS.
        // MIS option is for people to understand the usefullness of
        // MIS while sampling light so it is only for educational purposes.
        // This should be fine
        if(currentOptions.sampleMode == SampleMode::E::NEE)
            MRAY_WARNING_LOG("[%s]: While \"sampleMedia\" is true, \"sampleMode\" "
                             "can not be NEE. It is assumed as NEE_WITH_MIS.");
    }

    // ========================= //
    //   Path State Allocation   //
    // ========================= //
    // You can see why wavefront approach uses
    // quite a bit memory (and this is somewhat optimized).
    bool hasShadowRays = (currentOptions.sampleMode != SampleMode::E::PURE);
    bool doMediaSampleWithNEE = (hasShadowRays && currentOptions.sampleMedia == true);

    //
    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount, currentOptions.sampleMode);
    uint32_t mediaPDFRatioCount = (doMediaSampleWithNEE) ? maxRayCount : 0;
    uint32_t maxShadowRayCount = (hasShadowRays) ? maxRayCount : 0;
    uint32_t isVisibleIntCount = (hasShadowRays) ? Bitspan<uint32_t>::CountT(maxRayCount) : 0;
    uint32_t mediaListPackCount = (currentOptions.sampleMedia) ? maxRayCount : 0;
    uint32_t shadowMediaListPackCount = (doMediaSampleWithNEE) ? maxRayCount : 0;
    //
    bool hasPrevMatPDF = (currentOptions.sampleMode == SampleMode::E::NEE_WITH_MIS &&
                          currentOptions.sampleMedia == false);
    uint32_t prevMatPDFCount = (hasPrevMatPDF) ? maxRayCount : 0;

    MemAlloc::AllocateMultiData
    (
        Tie
        (
            // Per path
            dHits, dHitKeys, dRays, dRayCones,
            dPathRadiance, dImageCoordinates,
            dFilmFilterWeights, dThroughputs,
            dPathDataPack, dPathRNGDimensions,
            // Available when media sampling is on
            dRayMediaListPacks,
            dShadowRayMediaListPacks,
            // Per shadow ray
            dShadowRays, dShadowRayCones,
            dShadowRayRadiance,
            dShadowRayVisibilities,
            // MIS Related when media sampling is on/off. When media sampling
            // is on it must be pure rendering (without shadow rays).
            dPrevMatPDF,
            // MIS Related when media sampling is on with NEE
            dRPathPDF,
            dRLightPDF,
            dRPathPDFShadow,
            dRLightPDFShadow,
            dHitKeysShadow,
            // Per path available when spectral rendering is on
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
            // Available when media sampling is on
            mediaListPackCount,
            shadowMediaListPackCount,
            // Per shadow ray
            maxShadowRayCount, maxShadowRayCount, maxShadowRayCount,
            isVisibleIntCount,
            // MIS Related when media sampling is on/off. When media sampling
            // is on it must be pure rendering (without shadow rays).
            prevMatPDFCount,
            // MIS Related when media sampling is on with NEE
            mediaPDFRatioCount, mediaPDFRatioCount,
            mediaPDFRatioCount, mediaPDFRatioCount,
            mediaPDFRatioCount,
            // Per path available when spectral rendering is on
            wavelengthCount, wavelengthCount,
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

    if(currentOptions.sampleMode != SampleMode::E::PURE)
    {
        // When NEE is active, generate meta light list as well
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
    if(currentOptions.sampleMedia)
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
        {"neeSamplerType",  MRayDataTypeRT(MR_STRING),      IS_SCALAR, MR_MANDATORY},
        {"sampleMedia",     MRayDataTypeRT(MR_BOOL),      IS_SCALAR, MR_MANDATORY}
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
