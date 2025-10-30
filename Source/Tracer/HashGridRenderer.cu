#include "HashGridRenderer.h"
#include "RendererCommon.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgGeneric.h"

#include "Core/Timer.h"
#include "Core/ColorFunctions.h"

class HashGridIsAliveFunctor
{
    Span<const HashGridRDetail::PathStatus> dPathDataPack;

    public:
    HashGridIsAliveFunctor(Span<const HashGridRDetail::PathStatus> dPathDataPackIn)
        : dPathDataPack(dPathDataPackIn)
    {}

    MR_HF_DECL
    bool operator()(RayIndex index) const
    {
        using namespace HashGridRDetail;
        const PathStatus state = dPathDataPack[index];
        bool result = state[uint32_t(PathStatusEnum::DEAD)];
        return !result;
    }
};

struct HashGridResetPaths
{
    MR_HF_DECL
    void operator()(HashGridRDetail::PathStatus& s) const
    {
        s.Reset();
    }
};

HashGridRenderer::HashGridRenderer(const RenderImagePtr& rb,
                                   TracerView tv,
                                   ThreadPool& tp,
                                   const GPUSystem& s,
                                   const RenderWorkPack& wp)
    : RendererBase(rb, wp, tv, s, tp, TypeName())
    , saveImage(true)
    , hashGrid(s)
    , rayPartitioner(gpuSystem)
    , rendererGlobalMem(s.AllGPUs(), 128_MiB, 512_MiB)
{}

typename HashGridRenderer::AttribInfoList
HashGridRenderer::AttributeInfo() const
{
    return StaticAttributeInfo();
}

RendererOptionPack HashGridRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.cacheEntryLimit, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.cachePosBits, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.cacheNormalBits, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.cacheLevelCount, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<Float>{}, 1));
    result.attributes.back().Push(Span<const Float>(&currentOptions.cacheConeAperture, 1));
    //
    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.pathTraceDepth, 1));

    if constexpr(MRAY_IS_DEBUG)
    {
        for([[maybe_unused]] const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void HashGridRenderer::PushAttribute(uint32_t attributeIndex,
                                     TransientData data, const GPUQueue&)
{
    switch(attributeIndex)
    {
        case 0: newOptions.cacheEntryLimit   = data.AccessAs<uint32_t>()[0]; break;
        case 1: newOptions.cachePosBits      = data.AccessAs<uint32_t>()[0]; break;
        case 2: newOptions.cacheNormalBits   = data.AccessAs<uint32_t>()[0]; break;
        case 3: newOptions.cacheLevelCount   = data.AccessAs<uint32_t>()[0]; break;
        case 4: newOptions.cacheConeAperture = data.AccessAs<Float>()[0]; break;
        case 5: newOptions.pathTraceDepth    = data.AccessAs<uint32_t>()[0]; break;
        //
        default: throw MRayError("{} Unknown attribute index {}",
                                 TypeName(), attributeIndex);
    }
}

RenderBufferInfo HashGridRenderer::StartRender(const RenderImageParams& rIP,
                                               CamSurfaceId camSurfId,
                                               uint32_t customLogicIndex0,
                                               uint32_t customLogicIndex1)
{
    totalIterationCount = 0;
    currentOptions = newOptions;
    if(currentOptions.cachePosBits > SpatioDirCode::MORTON_BITS_PER_DIM)
    {
        MRAY_WARNING_LOG("Max position bits for HashGrid does not "
                         "fit to the hash code! Clamping to {}.",
                         SpatioDirCode::MORTON_BITS_PER_DIM);
        currentOptions.cachePosBits = SpatioDirCode::MORTON_BITS_PER_DIM;
    }
    if(currentOptions.cacheNormalBits > SpatioDirCode::NORMAL_BITS_PER_DIM)
    {
        MRAY_WARNING_LOG("Max normal bits for HashGrid does not "
                         "fit to the hash code! Clamping to {}.",
                         SpatioDirCode::NORMAL_BITS_PER_DIM);
        currentOptions.cacheNormalBits = SpatioDirCode::NORMAL_BITS_PER_DIM;
    }
    if(currentOptions.cacheLevelCount > SpatioDirCode::MaxLevel())
    {
        MRAY_WARNING_LOG("Max level for HashGrid does not "
                         "fit to the hash code! Clamping to {}.",
                         SpatioDirCode::MaxLevel());
        currentOptions.cacheNormalBits = SpatioDirCode::MaxLevel();
    }

    //
    uint32_t totalWorkCount = GenerateWorks();

    // Get bit change from user..
    using Math::Roll;
    curPosBits = currentOptions.cachePosBits + customLogicIndex0;
    curPosBits = uint32_t(Roll(int32_t(curPosBits), 0,
                               int32_t(SpatioDirCode::MORTON_BITS_PER_DIM + 1)));
    //
    curNormalBits = currentOptions.cacheNormalBits + customLogicIndex1;
    curNormalBits = uint32_t(Roll(int32_t(curNormalBits), 0,
                                  int32_t(SpatioDirCode::NORMAL_BITS_PER_DIM + 1)));
    // Setup Image Tiler
    imageTiler = ImageTiler(renderBuffer.get(), rIP,
                            tracerView.tracerParams.parallelizationHint,
                            Vector2ui::Zero());

    // Setup ray tracing systems
    uint32_t maxRayCount = imageTiler.ConservativeTileSize().Multiply();
    uint32_t maxWorkCount = uint32_t(currentWorks.size() +
                                     currentLightWorks.size());
    rayPartitioner = RayPartitioner(gpuSystem, maxRayCount,
                                    maxWorkCount);
    // Find camera surface and get keys work instance for that
    // camera etc.
    auto surfLoc = std::find_if
    (
        tracerView.camSurfs.cbegin(),
        tracerView.camSurfs.cend(),
        [camSurfId](const auto& pair)
        {
            return pair.first == camSurfId;
        }
    );
    if(surfLoc == tracerView.camSurfs.cend())
        throw MRayError("[{:s}]: Unknown camera surface id ({:d})",
                        TypeName(), uint32_t(camSurfId));
    curCamSurfaceParams = surfLoc->second;
    // Find the transform/camera work for this specific surface
    curCamKey = std::bit_cast<CameraKey>(curCamSurfaceParams.cameraId);
    curCamTransformKey = std::bit_cast<TransformKey>(curCamSurfaceParams.transformId);
    CameraGroupId camGroupId = CameraGroupId(curCamKey.FetchBatchPortion());
    TransGroupId transGroupId = TransGroupId(curCamTransformKey.FetchBatchPortion());
    auto packLoc = std::find_if
    (
        currentCameraWorks.cbegin(), currentCameraWorks.cend(),
        [camGroupId, transGroupId](const auto& pack)
        {
            return (pack.cgId == camGroupId &&
                    pack.tgId == transGroupId);
        }
    );
    assert(packLoc != currentCameraWorks.cend());
    curCamWork = packLoc->workPtr.get();

    uint32_t maxSampleCount = FindMaxSamplePerIteration(maxRayCount);
    MemAlloc::AllocateMultiData(Tie(dHits, dHitKeys, dRays, dRayCones,
                                    dRayState.dImageCoordinates,
                                    dRayState.dFilmFilterWeights,
                                    dRayState.dPathColors,
                                    dRayState.dPathStatus,
                                    dRayState.dOutRays,
                                    dRayState.dOutRayCones,
                                    dPathRNGDimensions,
                                    //
                                    dRandomNumBuffer,
                                    dWorkHashes,
                                    dWorkBatchIds,
                                    dSubCameraBuffer,
                                    dCamPosBuffer),
                                rendererGlobalMem,
                                {maxRayCount, maxRayCount,
                                 maxRayCount, maxRayCount,
                                 maxRayCount, maxRayCount,
                                 maxRayCount, maxRayCount,
                                 maxRayCount, maxRayCount,
                                 maxRayCount,
                                 //
                                 maxSampleCount, totalWorkCount,
                                 totalWorkCount,
                                 RendererBase::SUB_CAMERA_BUFFER_SIZE, 1});
    //
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    workHasher = InitializeHashes(dWorkHashes, dWorkBatchIds,
                                  maxRayCount, queue);

    // Also allocate for the partitioner inside the
    // base accelerator (This should not allocate for HW accelerators)
    tracerView.baseAccelerator.AllocateForTraversal(maxRayCount);

    // Generate RNG
    // Finally generate RNG
    auto RngGen = tracerView.rngGenerators.at(tracerView.tracerParams.samplerType.e);
    if(!RngGen)
        throw MRayError("[{}]: Unknown random number generator type {}.", TypeName(),
                        uint32_t(tracerView.tracerParams.samplerType.e));

    Vector2ui maxDeviceLocalRNGCount = imageTiler.ConservativeTileSize();
    uint64_t seed = tracerView.tracerParams.seed;
    uint32_t spp = 512;
    rnGenerator = RngGen->get()(rIP, std::move(maxDeviceLocalRNGCount),
                                std::move(spp), std::move(seed),
                                gpuSystem, globalThreadPool);

    auto bufferPtrAndSize = renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = MRayColorSpaceEnum::MR_ACES_CG,
        .resolution = imageTiler.FullResolution(),
        .curRenderLogic0 = customLogicIndex0,
        .curRenderLogic1 = customLogicIndex1
    };
}

uint32_t HashGridRenderer::FindMaxSamplePerIteration(uint32_t rayCount)
{
    uint32_t camSample = curCamWork->StochasticFilterSampleRayRNList().TotalRNCount();

    uint32_t maxSample = camSample;
    maxSample = std::transform_reduce
    (
        currentWorks.cbegin(), currentWorks.cend(), maxSample,
        [](uint32_t l, uint32_t r) -> uint32_t
        {
            return Math::Max(l, r);
        },
        [](const auto& renderWorkStruct) -> uint32_t
        {
            return renderWorkStruct.workPtr->SampleRNList(0).TotalRNCount();
        }
    );
    return rayCount * maxSample;
}

bool HashGridRenderer::CopyAliveRays(uint32_t rayCount, uint32_t maxWorkCount,
                                     const GPUQueue& processQueue)
{
    auto [dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                  processQueue, true);
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
    // Partition the rays wrt. dead/alive
    auto [hDeadRayRanges, dDeadAliveRayIndices] = rayPartitioner.BinaryPartition
    (
        dIndices, processQueue,
        HashGridIsAliveFunctor(dRayState.dPathStatus)
    );
    processQueue.Barrier().Wait();

    dIndices = dDeadAliveRayIndices.subspan(0, hDeadRayRanges[1] - hDeadRayRanges[0]);
    if(dIndices.empty()) return false;

    uint32_t aliveRayCount = static_cast<uint32_t>(dIndices.size());
    processQueue.IssueWorkKernel<KCCopyRaysIndirect>
    (
        "KCCopyRaysIndirect",
        DeviceWorkIssueParams{.workCount = aliveRayCount},
        dRays, dRayCones,
        dIndices,
        dRayState.dOutRays,
        dRayState.dOutRayCones
    );
    return true;
}

void HashGridRenderer::PathTraceAndQuery()
{
    Vector2ui tileCount2D = imageTiler.TileCount();

    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    const auto& cameraWork = *curCamWork;
    cameraWork.GenerateSubCamera
    (
        dSubCameraBuffer,
        curCamKey, curCamTransformOverride,
        imageTiler.CurrentTileIndex(),
        tileCount2D,
        processQueue
    );
    // Launch rays
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();
    uint32_t maxWorkCount = uint32_t(currentWorks.size() +
                                     currentLightWorks.size());
    // RNG Stuff
    rnGenerator->SetupRange(imageTiler.LocalTileStart(),
                            imageTiler.LocalTileEnd(),
                            processQueue);
    RNRequestList camSampleRNList = curCamWork->StochasticFilterSampleRayRNList();
    uint32_t camSamplePerRay = camSampleRNList.TotalRNCount();
    uint32_t camRayGenRNCount = rayCount * camSamplePerRay;
    auto dCamRayGenRNBuffer = dRandomNumBuffer.subspan(0, camRayGenRNCount);
    rnGenerator->IncrementSampleId(processQueue);
    rnGenerator->GenerateNumbers(dCamRayGenRNBuffer,
                                 0, camSampleRNList,
                                 processQueue);

    // Just starting the partitioner for get a index buffer
    // "GenerateRays" routine does not have "non-indirect" variant
    auto [dCamIndices, _] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                  processQueue, true);
    DeviceAlgorithms::Iota(dCamIndices, RayIndex(0), processQueue);
    cameraWork.GenerateRays(dRayCones,
                            dRays, dRayState.dImageCoordinates,
                            dRayState.dFilmFilterWeights,
                            dCamIndices, dCamRayGenRNBuffer,
                            dSubCameraBuffer,
                            curCamTransformKey,
                            totalIterationCount,
                            imageTiler.ConservativeTileSize(),
                            processQueue);
    DeviceAlgorithms::InPlaceTransformIndirect(dRayState.dPathStatus,
                                               dCamIndices,
                                               processQueue,
                                               HashGridResetPaths());
    processQueue.MemsetAsync(dPathRNGDimensions, 0x00);
    processQueue.MemsetAsync(dRayState.dPathColors, 0x00);


    // We need camera location,
    if(totalIterationCount == 0)
    {
        Vector3 camPos;
        cameraWork.GenCameraPosition(Span<Vector3, 1>(dCamPosBuffer),
                                     dSubCameraBuffer,
                                     curCamTransformKey,
                                     processQueue);
        MRAY_LOG("Max PosBit {}, Max NormalBit {}",
                 curPosBits, curNormalBits);

        processQueue.MemcpyAsync(Span<Vector3>(&camPos, 1), ToConstSpan(dCamPosBuffer));
        processQueue.Barrier().Wait();
        hashGrid.Reset(tracerView.baseAccelerator.SceneAABB(), camPos,
                       curPosBits, curNormalBits, currentOptions.cacheLevelCount,
                       currentOptions.cacheConeAperture,
                       currentOptions.cacheEntryLimit, processQueue);
    }

    // Do full path trace
    for(uint32_t depth = 0; depth < currentOptions.pathTraceDepth; depth++)
    {
        auto [dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount,
                                                      processQueue, true);
        DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);
        // Partition the rays wrt. dead/alive
        auto [hDeadRayRanges, dDeadAliveRayIndices] = rayPartitioner.BinaryPartition
        (
            dIndices, processQueue,
            HashGridIsAliveFunctor(dRayState.dPathStatus)
        );
        processQueue.Barrier().Wait();

        dIndices = dDeadAliveRayIndices.subspan(0, hDeadRayRanges[1] - hDeadRayRanges[0]);
        dKeys = dKeys.subspan(0, hDeadRayRanges[1] - hDeadRayRanges[0]);

        //MRAY_LOG("Alive {}", dIndices.size());

        // Cast rays
        using namespace std::string_view_literals;
        Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
        processQueue.IssueWorkKernel<KCSetBoundaryWorkKeysIndirect>
        (
            "KCSetBoundaryWorkKeys"sv,
            DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
            dHitKeys,
            ToConstSpan(dIndices),
            boundaryLightKeyPack
        );
        // Actual Ray Casting
        tracerView.baseAccelerator.CastRays
        (
            dHitKeys, dHits, dBackupRNGStates,
            dRays, dIndices, processQueue
        );

        // Generate work keys from hit packs
        using namespace std::string_literals;
        static const std::string GenWorkKernelName = std::string(TypeName()) + "-KCGenerateWorkKeysIndirect"s;
        processQueue.IssueWorkKernel<KCGenerateWorkKeysIndirect>
        (
            GenWorkKernelName,
            DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
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
        // Change indices to the partitioned one
        dIndices = partitionOutput.dPartitionIndices;

        GlobalState globalState
        {
            .hashGrid = hashGrid.View(),
            .curDepth = depth + 1
        };

        // Issue the work kernels
        IssueWorkKernelsToPartitions<This>(workHasher, partitionOutput,
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
            workI.DoWork_0(dRayState, dLocalIndices,
                           dRandomNumBuffer, dRayCones,
                           dRays, dHits, dHitKeys,
                           globalState, processQueue);
        },
        //
        [&, this](const auto& workI, Span<uint32_t> dLocalIndices,
                  uint32_t, uint32_t)
        {
            workI.DoBoundaryWork_0(dRayState, dLocalIndices,
                                   Span<const RandomNumber>{},
                                   dRayCones, dRays,
                                   dHits, dHitKeys,
                                   globalState, processQueue);
        });

        // Repartition and copy rays back to input buffer
        if(!CopyAliveRays(rayCount, maxWorkCount, processQueue))
            break;
    }

    // After loop is done we filled the hash table
    // and generated some colors
}

RendererOutput HashGridRenderer::DoRender()
{
    static const auto annotation = gpuSystem.CreateAnnotation("Render Frame");
    const auto _ = annotation.AnnotateScope();

    // Use CPU timer here
    // TODO: Implement a GPU timer later
    Timer timer; timer.Start();
    const GPUDevice& device = gpuSystem.BestDevice();

    using namespace std::string_view_literals;
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    // Change camera and reset hash table
    if(cameraTransform.has_value())
    {
        totalIterationCount = 0;
        curCamTransformOverride = cameraTransform;
        cameraTransform = std::nullopt;
    }
    // Do the thing!
    PathTraceAndQuery();

    // Do not start writing to device side until copy is complete
    // (device buffer is read fully)
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    renderBuffer->ClearImage(processQueue);
    // Write to image
    ImageSpan filmSpan = imageTiler.GetTileSpan();
    //SetImagePixelsIndirect
    SetImagePixels
    (
        filmSpan,
        ToConstSpan(dRayState.dPathColors),
        ToConstSpan(dRayState.dFilmFilterWeights),
        ToConstSpan(dRayState.dImageCoordinates),
        Float(1), processQueue
    );

    // Send the resulting image to host (Acquire synchronization prims)
    const GPUQueue& transferQueue = device.GetTransferQueue();
    Optional<RenderImageSection>
    renderOut = imageTiler.TransferToHost(processQueue,
                                          transferQueue);
    // Semaphore is invalidated, visor is probably crashed
    if(!renderOut.has_value())
        return RendererOutput{};

    if(totalIterationCount % 50 == 0 ||
       totalIterationCount < 5)
    {
        uint32_t used = hashGrid.CalculateUsedGridCount(processQueue);
        uint32_t max = hashGrid.EntryCapacity();

        MRAY_LOG("HashGrid {} / {} | Mem: MiB: {} / {}", used, max,
                 double(used * sizeof(uint64_t)) / 1024. / 1204.,
                 double(max * sizeof(uint64_t)) / 1024. / 1204.);

    }
    // Actually set the section parameters
    renderOut->globalWeight = Float(1);
    // Now wait, and send the information about timing etc.
    processQueue.Barrier().Wait();
    timer.Split();

    // Calculate M sample per sec
    uint32_t curPixelCount = imageTiler.CurrentTileSize().Multiply();
    double timeSec = timer.Elapsed<Second>();
    double samplePerSec = static_cast<double>(curPixelCount) / timeSec;
    samplePerSec /= 1'000'000;
    double spp = double(1) / double(imageTiler.TileCount().Multiply());
    totalIterationCount++;
    spp *= static_cast<double>(totalIterationCount);

    bool triggerSave = false;
    // Roll to the next tile
    imageTiler.NextTile();
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            samplePerSec,
            "M paths/s",
            spp,
            std::numeric_limits<double>::infinity(),
            "pix",
            float(timer.Elapsed<Millisecond>()),
            imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            SpatioDirCode::MORTON_BITS_PER_DIM + 1,
            SpatioDirCode::NORMAL_BITS_PER_DIM + 1
        },
        .imageOut = renderOut,
        .triggerSave = triggerSave
    };
}

void HashGridRenderer::StopRender()
{
    this->ClearAllWorkMappings();
    rnGenerator = {};
}

std::string_view HashGridRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "HashGrid"sv;
    return RendererTypeName<Name>;
}

typename HashGridRenderer::AttribInfoList
HashGridRenderer::StaticAttributeInfo()
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"cacheEntryLimit", MRayDataTypeRT(MR_UINT32), IS_SCALAR, MR_OPTIONAL},
        {"cachePosBits", MRayDataTypeRT(MR_UINT32), IS_SCALAR, MR_OPTIONAL},
        {"cacheNormalBits", MRayDataTypeRT(MR_UINT32), IS_SCALAR, MR_OPTIONAL},
        {"cacheLevelCount", MRayDataTypeRT(MR_UINT32), IS_SCALAR, MR_OPTIONAL},
        {"cacheConeAperture", MRayDataTypeRT(MR_FLOAT), IS_SCALAR, MR_OPTIONAL},
        {"pathTraceDepth", MRayDataTypeRT(MR_UINT32), IS_SCALAR, MR_OPTIONAL}
    };
}

size_t HashGridRenderer::GPUMemoryUsage() const
{
    return (rayPartitioner.GPUMemoryUsage() +
            rnGenerator->GPUMemoryUsage() +
            hashGrid.GPUMemoryUsage() +
            rendererGlobalMem.Size());
}

