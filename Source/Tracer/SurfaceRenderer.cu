#include "SurfaceRenderer.h"
#include "RayGenKernels.h"

#include "Core/Error.hpp"
#include "Core/Timer.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"

// TODO: Remove later
#include "TypeFormat.h"
#include "Device/GPUDebug.h"

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateWorkKeys(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                        MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                        MRAY_GRID_CONSTANT const RenderWorkHasher workHasher)
{
    assert(dWorkKey.size() == dInputKeys.size());

    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dInputKeys.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        dWorkKey[i] = workHasher.GenerateWorkKeyGPU(dInputKeys[i]);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                                MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                                MRAY_GRID_CONSTANT const RenderWorkHasher workHasher)
{
    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        RayIndex keyIndex = dIndices[i];
        auto keyPack = dInputKeys[keyIndex];
        dWorkKey[keyIndex] = workHasher.GenerateWorkKeyGPU(keyPack);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetBoundaryWorkKeys(MRAY_GRID_CONSTANT const Span<HitKeyPack> dWorkKey,
                           MRAY_GRID_CONSTANT const HitKeyPack boundaryWorkKey)
{
    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dWorkKey.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        dWorkKey[i] = boundaryWorkKey;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetBoundaryWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<HitKeyPack> dWorkKey,
                                   MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                   MRAY_GRID_CONSTANT const HitKeyPack boundaryWorkKey)
{
    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        RayIndex keyIndex = dIndices[i];
        dWorkKey[keyIndex] = boundaryWorkKey;
    }
}

SurfaceRenderer::SurfaceRenderer(const RenderImagePtr& rb,
                                 TracerView tv,
                                 BS::thread_pool& tp,
                                 const GPUSystem& s,
                                 const RenderWorkPack& wp)
    : RendererT(rb, wp, tv, s, tp)
    , rayPartitioner(s)
    , redererGlobalMem(s.AllGPUs(), 32_MiB, 512_MiB)
{}

typename SurfaceRenderer::AttribInfoList
SurfaceRenderer::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP", MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY}
    };
}

RendererOptionPack SurfaceRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));

    if constexpr(MRAY_IS_DEBUG)
    {
        for(const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void SurfaceRenderer::PushAttribute(uint32_t attributeIndex,
                                    TransientData data, const GPUQueue&)
{
    if(attributeIndex != 0)
        throw MRayError("{} Unkown attribute index {}",
                        TypeName(), attributeIndex);
    newOptions.totalSPP = data.AccessAs<uint32_t>()[0];
}

RenderBufferInfo SurfaceRenderer::StartRender(const RenderImageParams& rIP,
                                              CamSurfaceId camSurfId,
                                              Optional<CameraTransform> optTransform,
                                              uint32_t customLogicIndex0,
                                              uint32_t)
{
    using namespace SurfRDetail;
    // TODO: These may be  common operations, every renderer
    // does this move to a templated intermediate class
    // on the inheritance chain
    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    currentOptions = newOptions;
    transOverride = optTransform;
    totalIterationCount = 0;
    globalPixelIndex = 0;

    // Generate the Filter
    auto FilterGen = tracerView.filterGenerators.at(tracerView.tracerParams.filmFilter.type);
    if(!FilterGen)
        throw MRayError("[{}]: Unkown film filter type {}.",
                        SurfaceRenderer::TypeName(),
                        uint32_t(tracerView.tracerParams.filmFilter.type));
    Float radius = tracerView.tracerParams.filmFilter.radius;
    filmFilter = FilterGen->get()(gpuSystem, Float(radius));
    Vector2ui filterPadSize = filmFilter->FilterExtent();

    // Change the mode according to the render logic
    using MathFunctions::Roll;
    uint32_t newMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                             int32_t(Mode::END)));
    currentOptions.mode = Mode(newMode);

    imageTiler = ImageTiler(renderBuffer.get(), rIP,
                            tracerView.tracerParams.parallelizationHint,
                            Vector2ui::Zero(), 3, 1);

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
        throw MRayError("[{:s}]: Unkown camera surface id ({:d})",
                        TypeName(), uint32_t(camSurfId));
    curCamSurfaceParams = surfLoc->second;
    // Find the transform/camera work for this specific surface
    curCamKey = CameraKey(static_cast<CommonKey>(curCamSurfaceParams.cameraId));
    curCamTransformKey = TransformKey(static_cast<CommonKey>(curCamSurfaceParams.transformId));
    CameraGroupId camGroupId = CameraGroupId(curCamKey.FetchBatchPortion());
    TransGroupId transGroupId = TransGroupId(curCamTransformKey.FetchBatchPortion());
    auto packLoc = std::find_if(currentCameraWorks.cbegin(), currentCameraWorks.cend(),
    [camGroupId, transGroupId](const auto& pack)
    {
        return pack.idPack == Pair(camGroupId, transGroupId);
    });
    curCamWork = &packLoc->workPtr;

    // Allocate the ray state buffers
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    // Find the ray count (1spp per tile)
    uint32_t maxRayCount = imageTiler.ConservativeTileSize().Multiply();
    MemAlloc::AllocateMultiData(std::tie(dHits, dHitKeys, dRays,
                                         dRayDifferentials,
                                         dRayState.dImageCoordinates,
                                         dRayState.dOutputData,
                                         dCamGenRandomNums,
                                         dWorkHashes, dWorkBatchIds,
                                         dSubCameraBuffer),
                                redererGlobalMem,
                                {maxRayCount, maxRayCount, maxRayCount,
                                 maxRayCount, maxRayCount, maxRayCount,
                                 maxRayCount* (*curCamWork)->SampleRayRNCount(),
                                 totalWorkCount, totalWorkCount,
                                 SUB_CAMERA_BUFFER_SIZE});
    // And initialze the hashes
    workHasher = InitializeHashes(dWorkHashes, dWorkBatchIds, queue);

    //DeviceDebug::DumpGPUMemToStdOut("WorkHashes", ToConstSpan(dWorkHashes), queue);
    //DeviceDebug::DumpGPUMemToStdOut("WorkBatchIds", ToConstSpan(dWorkBatchIds), queue);

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

    // Finally generate RNG
    auto RngGen = tracerView.rngGenerators.at(tracerView.tracerParams.samplerType.type);
    if(!RngGen)
        throw MRayError("[{}]: Unkown random number generator type {}.",
                        SurfaceRenderer::TypeName(),
                        uint32_t(tracerView.tracerParams.samplerType.type));
    Vector2ui generatorCount = rIP.regionMax - rIP.regionMin;
    uint64_t seed = tracerView.tracerParams.seed;
    rnGenerator = RngGen->get()(std::move(generatorCount),
                                std::move(seed),
                                gpuSystem, globalThreadPool);

    // TODO: Change this later
    gpuSystem.SyncAll();
    const GPUQueue& q = gpuSystem.BestDevice().GetComputeQueue(0);
    rnGenerator->SetupDeviceRange(imageTiler.LocalTileStart(),
                                  imageTiler.LocalTileEnd());
    rnGenerator->CopyStatesToGPUAsync(q);
    q.Barrier().Wait();


    auto bufferPtrAndSize = renderBuffer->SharedDataPtrAndSize();
    return RenderBufferInfo
    {
        .data = bufferPtrAndSize.first,
        .totalSize = bufferPtrAndSize.second,
        .renderColorSpace = curColorSpace,
        .resolution = imageTiler.FullResolution(),
        .depth = renderBuffer->Depth(),
        .curRenderLogic0 = newMode,
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
    // we are using sinlge queue (thus single GPU)
    // change this later
    Timer timer; timer.Start();
    const auto& cameraWork = (*curCamWork->get());
    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& processQueue = device.GetComputeQueue(0);

    // Generate subcamera of this specific tile
    cameraWork.GenerateSubCamera(dSubCameraBuffer,
                                 curCamKey, transOverride,
                                 imageTiler.CurrentTileIndex(),
                                 imageTiler.TileCount(),
                                 processQueue);

    // Find the ray count. Ray count is tile count
    // but tile can exceed film boundaries so clamp,
    uint32_t rayCount = imageTiler.CurrentTileSize().Multiply();

    // Start the partitioner, again worst case work count
    // Get the K/V pair buffer
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto[dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount, true);

    // Iota the indices
    DeviceAlgorithms::Iota(dIndices, RayIndex(0), processQueue);

    // Create RNG state for each ray
    // Generate rays
    rnGenerator->SetupDeviceRange(imageTiler.LocalTileStart(),
                                  imageTiler.LocalTileEnd());

    // Generate RN for camera rays
    rnGenerator->GenerateNumbers(dCamGenRandomNums,
                                 (*curCamWork)->SampleRayRNCount(),
                                 processQueue);

    cameraWork.GenerateRays(dRayDifferentials, dRays, EmptyType{},
                            dRayState, dIndices,
                            ToConstSpan(dCamGenRandomNums),
                            dSubCameraBuffer, curCamTransformKey,
                            globalPixelIndex, imageTiler.CurrentTileSize(),
                            processQueue);
    globalPixelIndex += rayCount;
    // Save the states back (we will issue next tile after on next iteration)
    //rnGenerator->CopyStatesFromGPUAsync(processQueue);

    // Cast rays
    using namespace std::string_view_literals;
    Span<BackupRNGState> dBackupRNGStates = rnGenerator->GetBackupStates();
    processQueue.IssueSaturatingKernel<KCSetBoundaryWorkKeys>
    (
        "KCSetBoundaryWorkKeys"sv,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dHitKeys.size())},
        dHitKeys,
        boundaryMatKeyPack
    );

    tracerView.baseAccelerator.CastRays(dHitKeys, dHits, dBackupRNGStates,
                                        dRays, dIndices, processQueue);

    //DeviceDebug::DumpGPUMemToFile("dHitKeys", ToConstSpan(dHitKeys), processQueue);

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

    //DeviceDebug::DumpGPUMemToFile("dHashes", ToConstSpan(dKeys), processQueue);
    //DeviceDebug::DumpGPUMemToFile("dIndicesIn2", ToConstSpan(dIndices), processQueue);

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
    //DeviceDebug::DumpGPUMemToFile("dPartitionIndicesFull", ToConstSpan(dPartitionIndices), processQueue);
    //DeviceDebug::DumpGPUMemToFile("dPartitionKeysFull", ToConstSpan(dPartitionKeys), processQueue);

    GlobalState globalState{currentOptions.mode};
    for(uint32_t i = 0; i < hPartitionCount[0]; i++)
    {
        uint32_t partitionStart = hPartitionStartOffsets[i];
        uint32_t partitionSize = (hPartitionStartOffsets[i + 1] -
                                  hPartitionStartOffsets[i]);

        auto dLocalIndices = dPartitionIndices.subspan(partitionStart,
                                                       partitionSize);

        //DeviceDebug::DumpGPUMemToFile(std::string("dPartitionIndicesLocal") + std::to_string(i),
        //                              ToConstSpan(dLocalIndices), processQueue);
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
            const auto& workPtr = *wLoc->workPtr.get();
            workPtr.DoWork_0(Span<RayDiff>{},
                             Span<RayGMem>{},
                             RayPayload{},
                             dRayState,
                             dLocalIndices,
                             Span<const RandomNumber>{},
                             dRayDifferentials,
                             dRays,
                             dHits,
                             dHitKeys,
                             RayPayload{},
                             globalState,
                             processQueue);
        }
        else if(lightWLoc != currentLightWorks.cend())
        {
            const auto& workPtr = *lightWLoc->workPtr.get();
            workPtr.DoBoundaryWork_0(dRayState,
                                     dLocalIndices,
                                     Span<const RandomNumber>{},
                                     dRayDifferentials,
                                     dRays,
                                     dHits,
                                     dHitKeys,
                                     RayPayload{},
                                     globalState,
                                     processQueue);
        }
        else throw MRayError("[{}]: Unkown work id is found ({}).",
                             TypeName(), key);

    }

    // Filter the samples
    // Wait for previous copy to finish
    processQueue.IssueWait(renderBuffer->PrevCopyCompleteFence());
    // Please note that ray partitioner will be invalidated here.
    // In this case, we do not use the partitioner anymore
    // so its fine.
    renderBuffer->ClearImage(processQueue);
    ImageSpan<3> filmSpan = imageTiler.GetTileSpan<3>();
    // Using atomic filter since the samples are uniformly distributed
    filmFilter->ReconstructionFilterAtomicRGB(filmSpan,
                                              ToConstSpan(dRayState.dOutputData),
                                              ToConstSpan(dRayState.dImageCoordinates),
                                              Float(1), processQueue);
    //filmFilter->ReconstructionFilterRGB(filmSpan, rayPartitioner,
    //                                    ToConstSpan(dRayState.dOutputData),
    //                                    ToConstSpan(dRayState.dImageCoordinates),
    //                                    tracerView.tracerParams.parallelizationHint,
    //                                    Float(1), processQueue);

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
    double spp = (double(imageTiler.CurrentTileIndex().Multiply() + 1) /
                  double(imageTiler.TileCount().Multiply()));
    spp *= static_cast<double>(totalIterationCount);
    totalIterationCount++;
    // Roll to the next tile
    imageTiler.NextTile();

    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            samplePerSec,
            "M samples/s",
            spp,
            "spp",
            float(timer.Elapsed<Millisecond>()),
            imageTiler.FullResolution(),
            MRayColorSpaceEnum::MR_ACES_CG,
            GPUMemoryUsage(),
            static_cast<uint32_t>(SurfRDetail::Mode::END),
            0
        },
        .imageOut = renderOut
    };
}

void SurfaceRenderer::StopRender()
{}