#include "PathTracerRenderer.h"
#include "Core/MemAlloc.h"

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
                       MRAY_GRID_CONSTANT const Span<RayCone> dRayDiffOut,
                       MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                       MRAY_GRID_CONSTANT const Span<const RayGMem> dRaysIn,
                       MRAY_GRID_CONSTANT const Span<const RayCone> dRayDiffIn)
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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCWriteInvalidRays(MRAY_GRID_CONSTANT const Span<PathTraceRDetail::PathDataPack> dPathDataPack,
                               MRAY_GRID_CONSTANT const Span<RayGMem> dRaysOut,
                               MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices)
{
    static constexpr Vector3 LARGE_VEC = Vector3(std::numeric_limits<Float>::max());

    using namespace PathTraceRDetail;
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
static void KCAccumulateShadowRays(MRAY_GRID_CONSTANT const Span<Spectrum> dRadianceOut,
                                   MRAY_GRID_CONSTANT const Span<const Spectrum> dShadowRayRadiance,
                                   MRAY_GRID_CONSTANT const Bitspan<const uint32_t> dIsVisibleBuffer,
                                   MRAY_GRID_CONSTANT const Span<const PathTraceRDetail::PathDataPack> dPathDataPack,
                                   MRAY_GRID_CONSTANT const Vector2ui rrRange)
{
    using namespace PathTraceRDetail;
    KernelCallParams kp;
    uint32_t shadowRayCount = static_cast<uint32_t>(dShadowRayRadiance.size());
    for(uint32_t i = kp.GlobalId(); i < shadowRayCount; i += kp.TotalSize())
    {
        PathTraceRDetail::PathDataPack dataPack = dPathDataPack[i];

        using enum PathTraceRDetail::RayType;
        bool isShadowRay = (dataPack.type == SHADOW_RAY);
        // +2 is correct here, we did not increment the depth yet
        bool inDepthLimit = (dataPack.depth + 2 <= rrRange[1]);
        if(inDepthLimit && isShadowRay && dIsVisibleBuffer[i])
            dRadianceOut[i] += dShadowRayRadiance[i];
    }
}

struct SetPathStateFunctor
{
    PathTraceRDetail::PathStatusEnum e;
    bool firstInit;

    MRAY_HOST inline
    SetPathStateFunctor(PathTraceRDetail::PathStatusEnum eIn, bool firstInit = false)
        : e(eIn)
        , firstInit(firstInit)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    void operator()(PathTraceRDetail::PathDataPack& s) const
    {
        if(firstInit) s.status.Reset();

        using namespace PathTraceRDetail;
        if(e == PathStatusEnum::INVALID)
            s.status.Set(uint32_t(PathStatusEnum::DEAD), false);
        else if(e == PathStatusEnum::DEAD)
            s.status.Set(uint32_t(PathStatusEnum::INVALID), false);
        //
        s.status.Set(uint32_t(e));
    }
};

class IsDeadAliveInvalidFunctor
{
    Span<const PathTraceRDetail::PathDataPack> dPathDataPack;

    public:
    IsDeadAliveInvalidFunctor(Span<const PathTraceRDetail::PathDataPack> dPathDataPackIn)
        : dPathDataPack(dPathDataPackIn)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    uint32_t operator()(RayIndex index) const
    {
        using namespace PathTraceRDetail;
        const PathStatus state = dPathDataPack[index].status;

        bool isDead = state[uint32_t(PathStatusEnum::DEAD)];
        bool isInvalid = state[uint32_t(PathStatusEnum::INVALID)];
        assert(!isDead || !isInvalid);

        if(isDead)          return 0;
        else if(isInvalid)  return 1;
        else                return 2;
    }
};

class IsAliveFunctor
{
    Span<const PathTraceRDetail::PathDataPack> dPathDataPack;
    bool checkInvalidAsDead;

    public:
    IsAliveFunctor(Span<const PathTraceRDetail::PathDataPack> dPathDataPackIn,
                   bool checkInvalidAsDeadIn = false)
        : dPathDataPack(dPathDataPackIn)
        , checkInvalidAsDead(checkInvalidAsDeadIn)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    bool operator()(RayIndex index) const
    {
        using namespace PathTraceRDetail;
        const PathStatus state = dPathDataPack[index].status;
        bool result = state[uint32_t(PathStatusEnum::DEAD)];
        if(checkInvalidAsDead)
            result = result || state[uint32_t(PathStatusEnum::INVALID)];
        return !result;
    }
};

PathTracerRenderer::PathTracerRenderer(const RenderImagePtr& rb,
                                       TracerView tv,
                                       ThreadPool& tp,
                                       const GPUSystem& s,
                                       const RenderWorkPack& wp)
    : Base(rb, wp, tv, s, tp)
    , metaLightArray(s)
    , rayPartitioner(s)
    , redererGlobalMem(s.AllGPUs(), 128_MiB, 512_MiB)
    , saveImage(true)
{}

typename PathTracerRenderer::AttribInfoList
PathTracerRenderer::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP",        MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY},
        {"burstSize",       MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_OPTIONAL},
        {"renderMode",      MRayDataType<MR_STRING>{}, IS_SCALAR, MR_MANDATORY},
        {"sampleMode",      MRayDataType<MR_STRING>{}, IS_SCALAR, MR_MANDATORY},
        {"rrRange",         MRayDataType<MR_VECTOR_2UI>{}, IS_SCALAR, MR_MANDATORY},
        {"neeSamplerType",  MRayDataType<MR_STRING>{}, IS_SCALAR, MR_MANDATORY}
    };
}

RendererOptionPack PathTracerRenderer::CurrentAttributes() const
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
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string>{},
                                              curRenderModeName.size()));
    auto svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == curRenderModeName.size());
    std::copy(curRenderModeName.cbegin(), curRenderModeName.cend(), svRead.begin());
    //
    std::string_view curModeName = currentOptions.sampleMode.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string>{},
                                              curModeName.size()));
    svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == curModeName.size());
    std::copy(curModeName.cbegin(), curModeName.cend(), svRead.begin());
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<Vector2>{}, 1));
    result.attributes.back().Push(Span<const Vector2ui>(&currentOptions.russianRouletteRange, 1));
    //
    std::string_view lightSamplerName = currentOptions.lightSampler.ToString();
    result.attributes.push_back(TransientData(std::in_place_type_t<std::string>{}, lightSamplerName.size()));
    svRead = result.attributes.back().AccessAsString();
    assert(svRead.size() == lightSamplerName.size());
    std::copy(lightSamplerName.cbegin(), lightSamplerName.cend(), svRead.begin());
    //
    if constexpr(MRAY_IS_DEBUG)
    {
        for(const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void PathTracerRenderer::PushAttribute(uint32_t attributeIndex,
                                       TransientData data, const GPUQueue&)
{    switch(attributeIndex)
    {
        case 0: newOptions.totalSPP = data.AccessAs<uint32_t>()[0]; break;
        case 1: newOptions.burstSize = data.AccessAs<uint32_t>()[0]; break;
        case 2: newOptions.renderMode = PathTraceRDetail::RenderMode(std::as_const(data).AccessAsString()); break;
        case 3: newOptions.sampleMode = PathTraceRDetail::SampleMode(std::as_const(data).AccessAsString()); break;
        case 4: newOptions.russianRouletteRange = data.AccessAs<Vector2ui>()[0]; break;
        case 5: newOptions.lightSampler = PathTraceRDetail::LightSamplerType(std::as_const(data).AccessAsString()); break;
        default:
            throw MRayError("{} Unkown attribute index {}", TypeName(), attributeIndex);
    }
}

uint32_t PathTracerRenderer::FindMaxSamplePerIteration(uint32_t rayCount,
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
            if(sampleMode == PURE)
                return renderWorkStruct.workPtr->SampleRNCount(0);
            else
                return std::max(renderWorkStruct.workPtr->SampleRNCount(0),
                                renderWorkStruct.workPtr->SampleRNCount(1));
        }
    );
    return rayCount * maxSample;
}

uint64_t PathTracerRenderer::SPPLimit(uint32_t spp) const
{
    if(spp == std::numeric_limits<uint32_t>::max())
        return std::numeric_limits<uint64_t>::max();

    Vector2ul tileSizeLong(this->imageTiler.CurrentTileSize());
    uint64_t result = spp;
    result *= tileSizeLong.Multiply();
    return result;
}

std::pair<Span<RayIndex>, uint32_t>
PathTracerRenderer::ReloadPaths(Span<const RayIndex> dIndices,
                                uint32_t sppLimit,
                                const GPUQueue& processQueue)
{
    // RELOADING!!!
    // Find the dead rays
    auto [hDeadRayRanges, dDeadAliveRayIndices] = rayPartitioner.BinaryPartition
    (
        dIndices, processQueue,
        IsAliveFunctor(dRayState.dPathDataPack, true)
    );
    processQueue.Barrier().Wait();

    // Generate RN for camera rays
    uint32_t camSamplePerRay = (*curCamWork)->StochasticFilterSampleRayRNCount();
    uint32_t deadRayCount = hDeadRayRanges[2] - hDeadRayRanges[1];
    auto dDeadRayIndices = dDeadAliveRayIndices.subspan(hDeadRayRanges[1],
                                                        deadRayCount);

    uint32_t aliveRayCount = hDeadRayRanges[1] - hDeadRayRanges[0];
    if(!dDeadRayIndices.empty())
    {
        uint64_t& tilePixIndex = tilePathCounts[this->imageTiler.CurrentTileIndex1D()];
        uint64_t localPathLimit = SPPLimit(sppLimit);
        uint64_t pixelIndexLimit = std::min(tilePixIndex + deadRayCount, localPathLimit);
        uint32_t fillRayCount = static_cast<uint32_t>(pixelIndexLimit - tilePixIndex);
        fillRayCount = std::min(deadRayCount, fillRayCount);
        aliveRayCount += fillRayCount;

        if(fillRayCount != 0)
        {
            auto dFilledRayIndices = dDeadRayIndices.subspan(0, fillRayCount);
            uint32_t camRayGenRNCount = fillRayCount * camSamplePerRay;
            auto dCamRayGenRNBuffer = dRandomNumBuffer.subspan(0, camRayGenRNCount);
            rnGenerator->GenerateNumbersIndirect(dCamRayGenRNBuffer,
                                                 ToConstSpan(dFilledRayIndices),
                                                 Vector2ui(0, camSamplePerRay),
                                                 processQueue);
            //
            const auto& cameraWork = (*curCamWork->get());
            cameraWork.GenRaysStochasticFilter
            (
                dRayCones, dRays,
                dRayState.dImageCoordinates,
                dRayState.dFilmFilterWeights,
                ToConstSpan(dFilledRayIndices),
                ToConstSpan(dCamRayGenRNBuffer),
                dSubCameraBuffer, curCamTransformKey,
                tilePixIndex,
                this->imageTiler.CurrentTileSize(),
                this->tracerView.tracerParams.filmFilter,
                processQueue
            );
            tilePixIndex += fillRayCount;

            // Initialize the state of new rays
            processQueue.IssueSaturatingKernel<KCInitPathState>
            (
                "KCInitPathState",
                KernelIssueParams{.workCount = fillRayCount},
                dRayState,
                dFilledRayIndices
            );
        }
        //
        if(fillRayCount != deadRayCount)
        {
            uint32_t unusedRays = deadRayCount - fillRayCount;
            auto dUnusedIndices = dDeadRayIndices.subspan(fillRayCount, unusedRays);
            // Fill the remaining values
            processQueue.IssueSaturatingKernel<KCWriteInvalidRays>
            (
                "KCWriteInvalidRays",
                KernelIssueParams{.workCount = unusedRays},
                //
                dRayState.dPathDataPack,
                dRays,
                dUnusedIndices
            );
        }
    }
    // Index buffer may be invalidated (Binary partition should not
    // invalidate but lets return the new buffer)
    return std::pair(dDeadAliveRayIndices, aliveRayCount);
}

void PathTracerRenderer::ResetAllPaths(const GPUQueue& queue)
{
    // Clear states
    queue.MemsetAsync(dRayState.dPathDataPack, 0x00);

    // Set all paths as dead (we just started)
    using namespace PathTraceRDetail;
    DeviceAlgorithms::InPlaceTransform(dRayState.dPathDataPack, queue,
                                       SetPathStateFunctor(PathStatusEnum::INVALID, true));
    queue.MemsetAsync(dPathRNGDimensions, 0x00);
}

Span<RayIndex> PathTracerRenderer::DoRenderPass(uint32_t sppLimit,
                                                const GPUQueue& processQueue)
{
    assert(sppLimit != 0);
    // Find the ray count. Ray count is tile count
    // but tile can exceed film boundaries so clamp,
    uint32_t rayCount = this->imageTiler.CurrentTileSize().Multiply();
    // Start the partitioner, again worst case work count
    // Get the K/V pair buffer
    uint32_t maxWorkCount = uint32_t(this->currentWorks.size() +
                                     this->currentLightWorks.size());
    auto[dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount, true);

    // Iota the indices
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
    // Create RNG state for each ray
    rnGenerator->SetupRange(this->imageTiler.Tile1DRange());
    // Reload dead paths with new
    auto [dReloadIndices, aliveRayCount] = ReloadPaths(dIndices, sppLimit,
                                                       processQueue);
    dIndices = dReloadIndices.subspan(0, aliveRayCount);
    dKeys = dKeys.subspan(0, aliveRayCount);

    // Cast rays
    using namespace std::string_view_literals;
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    processQueue.IssueSaturatingKernel<KCSetBoundaryWorkKeysIndirect>
    (
        "KCSetBoundaryWorkKeys"sv,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dHitKeys,
        ToConstSpan(dIndices),
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
    processQueue.IssueSaturatingKernel<KCGenerateWorkKeysIndirect>
    (
        GenWorkKernelName,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        dKeys,
        ToConstSpan(dIndices),
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
                             dRandomNumBuffer, dRayCones,
                             dRays, dHits, dHitKeys,
                             globalState, processQueue);
        },
        //
        [&, this](const auto& workPtr, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            workPtr.DoBoundaryWork_0(dRayState, dLocalIndices,
                                     Span<const RandomNumber>{},
                                     dRayCones, dRays,
                                     dHits, dHitKeys,
                                     globalState, processQueue);
        });
    }
    else
    {
        UniformLightSampler lightSampler(metaLightArray.Array(),
                                         metaLightArray.IndexHashTable());

        using GlobalState = PathTraceRDetail::GlobalState<UniformLightSampler>;
        GlobalState globalState
        {
            .russianRouletteRange = currentOptions.russianRouletteRange,
            .sampleMode = currentOptions.sampleMode,
            .lightSampler = lightSampler
        };
        using GlobalStateE = PathTraceRDetail::GlobalState<EmptyType>;
        GlobalStateE globalStateE
        {
            .russianRouletteRange = currentOptions.russianRouletteRange,
            .sampleMode = currentOptions.sampleMode
        };

        // Clear the shadow ray radiance buffer
        processQueue.MemsetAsync(dRayState.dShadowRayRadiance, 0x00);
        processQueue.MemsetAsync(dShadowRayVisibilities, 0x00);
        // CUDA Init check error, we access the rays even if it is not written
        processQueue.MemsetAsync(dRayState.dOutRays, 0x00);


        // Do the NEE kernel + boundary work
        this->IssueWorkKernelsToPartitions(workHasher, partitionOutput,
        [&, this](const auto& workPtr, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t partitionSize)
        {
            uint32_t rnCount = workPtr.SampleRNCount(1);
            auto dLocalRNBuffer = dRandomNumBuffer.subspan(0, partitionSize * rnCount);
            rnGenerator->GenerateNumbersIndirect(dLocalRNBuffer, dLocalIndices,
                                                 dPathRNGDimensions, rnCount,
                                                 processQueue);

            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dPathRNGDimensions, dLocalIndices, processQueue,
                ConstAddFunctor(rnCount)
            );

            workPtr.DoWork_1(dRayState, dLocalIndices,
                             dRandomNumBuffer, dRayCones,
                             dRays, dHits, dHitKeys,
                             globalState, processQueue);
        },
        [&, this](const auto& workPtr, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            workPtr.DoBoundaryWork_1(dRayState, dLocalIndices,
                                     Span<const RandomNumber>{},
                                     dRayCones, dRays,
                                     dHits, dHitKeys,
                                     globalState, processQueue);
        });

        // After the kernel call(s), "dRayState.dOutRays" holds the shadow rays
        // check visibility.
        Bitspan<uint32_t> dIsVisibleBitSpan(dShadowRayVisibilities);
        this->tracerView.baseAccelerator.CastVisibilityRays
        (
            dIsVisibleBitSpan, dBackupRNGStates,
            dRayState.dOutRays, dIndices, processQueue
        );

        // Accumulate the pre-calculated radiance selectively
        processQueue.IssueSaturatingKernel<KCAccumulateShadowRays>
        (
            "KCAccumulateShadowRays",
            KernelIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
            //
            dRayState.dPathRadiance,
            ToConstSpan(dRayState.dShadowRayRadiance),
            ToConstSpan(dIsVisibleBitSpan),
            ToConstSpan(dRayState.dPathDataPack),
            currentOptions.russianRouletteRange
        );

        // Do the actual kernel
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
                             dRandomNumBuffer, dRayCones,
                             dRays, dHits, dHitKeys,
                             globalStateE, processQueue);
        },
        // Empty Kernel for light
        [&, this](const auto&, Span<uint32_t>, uint32_t, uint32_t) {});
    }

    return dIndices;
}

RendererOutput PathTracerRenderer::DoThroughputSingleTileRender(const GPUDevice& device,
                                                                const GPUQueue& processQueue)
{
    Timer timer; timer.Start();
    const auto& cameraWork = (*curCamWork->get());
    // Generate subcamera of this specific tile
    if(this->totalIterationCount == 0)
    {
        cameraWork.GenerateSubCamera
        (
            dSubCameraBuffer,
            curCamKey, curCamTransformOverride,
            this->imageTiler.CurrentTileIndex(),
            this->imageTiler.TileCount(),
            processQueue
        );
    }

    //
    uint32_t sppLimit = (saveImage) ? currentOptions.totalSPP
                                    : std::numeric_limits<uint32_t>::max();
    Span<RayIndex> dIndices = DoRenderPass(sppLimit, processQueue);

    // Find the dead paths again
    // Do a 3-way partition here to catch potential invalid rays
    auto deadAlivePartitionOut = rayPartitioner.TernaryPartition
    (
        dIndices, processQueue,
        IsDeadAliveInvalidFunctor(ToConstSpan(dRayState.dPathDataPack))
    );
    processQueue.Barrier().Wait();
    auto [dDeadRayIndices, dInvalidRayIndices, dAliveRayIndices] =
        deadAlivePartitionOut.Spanify();
    //
    Optional<RenderImageSection> renderOut;
    if(!dDeadRayIndices.empty())
    {
        DeviceAlgorithms::InPlaceTransformIndirect
        (
            dRayState.dPathDataPack, ToConstSpan(dDeadRayIndices), processQueue,
            SetPathStateFunctor(PathTraceRDetail::PathStatusEnum::INVALID)
        );

        processQueue.IssueWait(this->renderBuffer->PrevCopyCompleteFence());
        this->renderBuffer->ClearImage(processQueue);
        ImageSpan filmSpan = this->imageTiler.GetTileSpan();

        SetImagePixelsIndirectAtomic
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
    //
    bool triggerSave = false;
    if(!dAliveRayIndices.empty())
    {
        uint32_t aliveRayCount = static_cast<uint32_t>(dAliveRayIndices.size());
        processQueue.IssueSaturatingKernel<KCCopyRays>
        (
            "KCCopyRays",
            KernelIssueParams{.workCount = aliveRayCount},
            dRays, dRayCones,
            dAliveRayIndices,
            dRayState.dOutRays,
            dRayState.dOutRayCones
        );
    }
    else if(saveImage && tilePathCounts[0] == SPPLimit(currentOptions.totalSPP))
    {
        // We exausted all alive rays while doing SPP limit.
        // Trigger a save, and unleash the renderer.
        triggerSave = true;
        saveImage = false;
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
    auto deadRayCount = dDeadRayIndices.size();
    totalDeadRayCount += deadRayCount;
    double timeSec = timer.Elapsed<Second>();
    double pathPerSec = static_cast<double>(deadRayCount) / timeSec;
    pathPerSec /= 1'000'000;

    uint64_t totalPixels = this->imageTiler.FullResolution().Multiply();
    double spp = double(totalDeadRayCount) / double(totalPixels);

    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            pathPerSec,
            "M path/s",
            spp,
            double(currentOptions.totalSPP),
            "spp",
            float(timer.Elapsed<Millisecond>()),
            this->imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            static_cast<uint32_t>(PathTraceRDetail::SampleMode::E::END),
            0
        },
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

RendererOutput PathTracerRenderer::DoLatencyRender(uint32_t passCount,
                                                   const GPUDevice& device,
                                                   const GPUQueue& processQueue)
{
    Vector2ui tileCount2D = this->imageTiler.TileCount();
    uint32_t tileCount1D = tileCount2D.Multiply();

    Timer timer; timer.Start();
    // Generate subcamera of this specific tile
    const auto& cameraWork = (*curCamWork->get());
    cameraWork.GenerateSubCamera
    (
        dSubCameraBuffer,
        curCamKey, curCamTransformOverride,
        this->imageTiler.CurrentTileIndex(),
        tileCount2D,
        processQueue
    );

    // We are waiting too early here,
    // We should wait at least on the first render buffer write
    // but it was not working so I've put it here
    // TODO: Investigate
    processQueue.IssueWait(this->renderBuffer->PrevCopyCompleteFence());
    this->renderBuffer->ClearImage(processQueue);

    //
    uint32_t tileIndex = this->imageTiler.CurrentTileIndex1D();
    uint32_t tileSPP = static_cast<uint32_t>(tileSPPs[tileIndex]);
    uint32_t sppLimit = tileSPP + passCount;
    if(saveImage)
        sppLimit = std::min(sppLimit, currentOptions.totalSPP);

    tileSPPs[tileIndex] = sppLimit;
    uint32_t currentPassCount = sppLimit - tileSPP;
    uint32_t passPathCount = this->imageTiler.CurrentTileSize().Multiply() * currentPassCount;
    uint32_t invalidRayCount = 0;
    do
    {
        Span<RayIndex> dIndices = DoRenderPass(sppLimit, processQueue);

        // Find the dead paths again
        // Every path is processed, so we do not need to use the scambled
        // index buffer. Iota again
        //DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
        // Do a 3-way partition,
        auto deadAlivePartitionOut = rayPartitioner.TernaryPartition
        (
            dIndices, processQueue,
            IsDeadAliveInvalidFunctor(ToConstSpan(dRayState.dPathDataPack))
        );
        processQueue.Barrier().Wait();
        auto [dDeadRayIndices, dInvalidRayIndices, dAliveRayIndices] =
            deadAlivePartitionOut.Spanify();
        //
        if(!dDeadRayIndices.empty())
        {
            DeviceAlgorithms::InPlaceTransformIndirect
            (
                dRayState.dPathDataPack, ToConstSpan(dDeadRayIndices), processQueue,
                SetPathStateFunctor(PathTraceRDetail::PathStatusEnum::INVALID)
            );
            ImageSpan filmSpan = this->imageTiler.GetTileSpan();
            SetImagePixelsIndirect
            (
                filmSpan,
                ToConstSpan(dDeadRayIndices),
                ToConstSpan(dRayState.dPathRadiance),
                ToConstSpan(dRayState.dFilmFilterWeights),
                ToConstSpan(dRayState.dImageCoordinates),
                Float(1), processQueue
            );
        }
        //
        if(!dAliveRayIndices.empty())
        {
            uint32_t aliveRayCount = static_cast<uint32_t>(dAliveRayIndices.size());
            processQueue.IssueSaturatingKernel<KCCopyRays>
            (
                "KCCopyRays",
                KernelIssueParams{.workCount = aliveRayCount},
                dRays, dRayCones,
                dAliveRayIndices,
                dRayState.dOutRays,
                dRayState.dOutRayCones
            );
        }
        assert(dInvalidRayIndices.size() == 0);
        invalidRayCount += (static_cast<uint32_t>(dDeadRayIndices.size()));
    } while(invalidRayCount != passPathCount);

    // One spp of this tile should be done now.
    // Issue the transfer
    // Issue a send of the FBO to Visor
    const GPUQueue& transferQueue = device.GetTransferQueue();
    Optional<RenderImageSection> renderOut;
    renderOut = this->imageTiler.TransferToHost(processQueue,
                                                transferQueue);
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

    this->totalIterationCount++;
    double timeSec = timer.Elapsed<Second>();
    double pathPerSec = static_cast<double>(passPathCount) / timeSec;
    pathPerSec /= 1'000'000;

    uint64_t sppSum = std::reduce(tileSPPs.cbegin(), tileSPPs.cend(), uint64_t(0));
    double spp = double(sppSum) / double(tileCount1D);
    // Roll to the next tile
    this->imageTiler.NextTile();

    // Check save trigger
    // Notice, we've changed to the next tile above.
    uint64_t sppSumCheck = (uint64_t(currentOptions.totalSPP) *
                            uint64_t(tileCount1D));
    bool triggerSave = false;
    if(saveImage && sppSum == sppSumCheck)
    {
        triggerSave = true;
        saveImage = false;
    }
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            pathPerSec,
            "M path/s",
            spp,
            double(currentOptions.totalSPP),
            "spp",
            float(timer.Elapsed<Millisecond>()),
            this->imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            static_cast<uint32_t>(PathTraceRDetail::SampleMode::E::END),
            0
        },
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

RenderBufferInfo
PathTracerRenderer::StartRender(const RenderImageParams& rIP,
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
                                  Vector2ui::Zero());
    tilePathCounts.resize(this->imageTiler.TileCount().Multiply(), 0u);
    tileSPPs.resize(this->imageTiler.TileCount().Multiply(), 0u);
    std::fill(tilePathCounts.begin(), tilePathCounts.end(), 0u);
    std::fill(tileSPPs.begin(), tileSPPs.end(), 0u);

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
            return pack.idPack == std::pair(camGroupId, transGroupId);
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
                                             dRays, dRayCones,
                                             dRayState.dPathRadiance,
                                             dRayState.dImageCoordinates,
                                             dRayState.dFilmFilterWeights,
                                             dRayState.dThroughput,
                                             dRayState.dPathDataPack,
                                             dRayState.dOutRays,
                                             dRayState.dOutRayCones,
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
                                     Base::SUB_CAMERA_BUFFER_SIZE});
    }
    else
    {
        uint32_t isVisibleIntCount = Bitspan<uint32_t>::CountT(maxRayCount);
        MemAlloc::AllocateMultiData(std::tie(dHits, dHitKeys,
                                             dRays, dRayCones,
                                             dRayState.dPathRadiance,
                                             dRayState.dImageCoordinates,
                                             dRayState.dFilmFilterWeights,
                                             dRayState.dThroughput,
                                             dRayState.dPathDataPack,
                                             dRayState.dOutRays,
                                             dRayState.dOutRayCones,
                                             dRayState.dPrevMatPDF,
                                             dRayState.dShadowRayRadiance,
                                             dPathRNGDimensions,
                                             dShadowRayVisibilities,
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
                                     maxRayCount, maxRayCount,
                                     isVisibleIntCount, maxSampleCount,
                                     totalWorkCount, totalWorkCount,
                                     Base::SUB_CAMERA_BUFFER_SIZE});
        MetaLightListConstructionParams mlParams =
        {
            .lightGroups = this->tracerView.lightGroups,
            .transformGroups = this->tracerView.transGroups,
            .lSurfList = Span<const Pair<LightSurfaceId, LightSurfaceParams>>(this->tracerView.lightSurfs)
        };
        metaLightArray.Construct(mlParams, this->tracerView.boundarySurface,
                                 queue);
    }
    queue.MemsetAsync(dHits, 0x00);

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
        .curRenderLogic0 = sendMode,
        .curRenderLogic1 = std::numeric_limits<uint32_t>::max()
    };
}

RendererOutput PathTracerRenderer::DoRender()
{
    static const auto annotation = this->gpuSystem.CreateAnnotation("Render Frame");
    const auto _ = annotation.AnnotateScope();

    // TODO: Like many places of this codebase
    // we are using sinlge queue (thus single GPU)
    // change this later
    const GPUDevice& device = this->gpuSystem.BestDevice();
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    if(this->cameraTransform.has_value())
    {
        this->totalIterationCount = 0;
        curCamTransformOverride = this->cameraTransform;
        this->cameraTransform = std::nullopt;
        std::fill(tilePathCounts.begin(), tilePathCounts.end(), 0u);
        std::fill(tileSPPs.begin(), tileSPPs.end(), 0u);
        totalDeadRayCount = 0;

        ResetAllPaths(processQueue);
    }

    bool isSingleTile = this->imageTiler.TileCount().Multiply() == 1;
    using namespace PathTraceRDetail;
    if(currentOptions.renderMode == RenderMode::E::THROUGHPUT &&
       isSingleTile)
    {
        if(currentOptions.burstSize == 1)
            return DoThroughputSingleTileRender(device, processQueue);
        else
            return DoLatencyRender(currentOptions.burstSize, device, processQueue);
    }
    else if(currentOptions.renderMode == RenderMode::E::THROUGHPUT &&
            !isSingleTile)
    {
        return DoLatencyRender(currentOptions.burstSize, device, processQueue);
    }
    else
    {
        return DoLatencyRender(1u, device, processQueue);
    }
}

void PathTracerRenderer::StopRender()
{
    this->ClearAllWorkMappings();
    filmFilter = {};
    rnGenerator = {};
    metaLightArray.Clear();
}

std::string_view PathTracerRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "PathTracerRGB"sv;
    return RendererTypeName<Name>;
}

size_t PathTracerRenderer::GPUMemoryUsage() const
{
    return (rayPartitioner.GPUMemoryUsage() +
            rnGenerator->GPUMemoryUsage() +
            redererGlobalMem.Size());
}

static_assert(RendererC<PathTracerRenderer>,
              "\"PathTracerRenderer\" does not "
              "satisfy renderer concept.");