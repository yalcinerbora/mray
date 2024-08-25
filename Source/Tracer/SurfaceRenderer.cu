#include "SurfaceRenderer.h"
#include "RayGenKernels.h"

#include "Core/Error.hpp"

#include "Device/GPUSystem.hpp"

MRAY_KERNEL
void KCGenerateWorkKeys(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                        MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                        MRAY_GRID_CONSTANT const RenderWorkHash workHasher)
{
    assert(dWorkKey.size() == dInputKeys.size());
    uint32_t keyCount = static_cast<uint32_t>(dInputKeys.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        dWorkKey[i] = workHasher.GenerateWorkKeyGPU(dInputKeys[i]);
    }
}

SurfaceRenderer::SurfaceRenderer(const RenderImagePtr& rb,
                                 const RenderWorkPack& wp,
                                 TracerView tv, const GPUSystem& s)
    : RendererT(rb, wp, tv, s)
    , rayPartitioner(s)
    , rayStateMem(s.AllGPUs(), 32_MiB, 512_MiB)
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

RenderBufferInfo SurfaceRenderer::StartRender(const RenderImageParams& rip,
                                              CamSurfaceId camSurfId,
                                              Optional<CameraTransform> optTransform,
                                              uint32_t customLogicIndex0,
                                              uint32_t)
{
    using namespace SurfRDetail;

    // TODO: This is common assignment, every renderer
    // does this move to a templated intermediate class
    // on the inheritance chain
    currentOptions = newOptions;
    transOverride = optTransform;
    rIParams = rip;
    //
    // Find the texture index
    using MathFunctions::Roll;
    uint32_t newMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                             int32_t(Mode::END)));
    currentOptions.mode = Mode(newMode);


    // Calculate tile size according to the parallelization hint
    uint32_t parallelHint = tracerView.tracerParams.parallelizationHint;
    Vector2ui imgRegion = rIParams.regionMax - rIParams.regionMin;
    Vector2ui tileSize = FindOptimumTile(imgRegion, parallelHint);
    renderBuffer->Resize(tileSize, 1, 3);
    tileCount = MathFunctions::DivideUp(imgRegion, tileSize);
    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    curFramebufferSize = rIParams.resolution;
    curFBMin = rIParams.regionMin;
    curFBMax = rIParams.regionMax;
    RenderBufferInfo rbI = renderBuffer->GetBufferInfo(curColorSpace,
                                                       curFramebufferSize, 1);
    rbI.curRenderLogic0 = newMode;
    rbI.curRenderLogic1 = 0;
    curTileIndex = 0;

    // Generate works per
    // Material1/Primitive/Transform triplet,
    // Light/Transform pair,
    // Camera/Transform pair
    workCounter = 0;
    workCounter = GenerateWorkMappings(workCounter);
    workCounter = GenerateLightWorkMappings(workCounter);
    workCounter = GenerateCameraWorkMappings(workCounter);

    // Initialize ray partitioner with worst case scenario,
    // All work types are used. (We do not use camera work
    // for this type of renderer)
    uint32_t maxWorkCount = uint32_t(currentWorks.size() +
                                     currentLightWorks.size());
    rayPartitioner = RayPartitioner(gpuSystem, tileCount.Multiply(),
                                    maxWorkCount);

    // Get the surface params
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
    CameraKey curCamKey = CameraKey(static_cast<CommonKey>(curCamSurfaceParams.cameraId));
    curCamTransformKey = TransformKey(static_cast<CommonKey>(curCamSurfaceParams.transformId));
    CameraGroupId camGroupId = CameraGroupId(curCamKey.FetchBatchPortion());
    TransGroupId transGroupId = TransGroupId(curCamTransformKey.FetchBatchPortion());
    auto packLoc = std::find_if(currentCameraWorks.cbegin(), currentCameraWorks.cend(),
    [camGroupId, transGroupId](const auto& pack)
    {
        return pack.idPack == Pair(camGroupId, transGroupId);
    });
    curCamWork = &packLoc->workPtr;

    return rbI;

}

RendererOutput SurfaceRenderer::DoRender()
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    // Find the ray count (1spp)
    uint32_t rayCount = renderBuffer->Extents().Multiply();
    MemAlloc::AllocateMultiData(std::tie(dHits, dHitKeys, dRays,
                                         dRayDifferentials, dSubCameraBuffer),
                                rayStateMem,
                                {});

    // Each iteration do one tile fully,
    // so we can send it directly
    // Generate subcamera of this specific tile
    Vector2ui curTile2D = Vector2ui(curTileIndex % tileCount[0],
                                    curTileIndex / tileCount[0]);
    curTileIndex = MathFunctions::Roll(int32_t(curTileIndex) + 1, 0,
                                       int32_t(tileCount.Multiply()));
    //(*curCamWork)->GenerateSubCamera(dSubCameraBuffer,
    //                                 curCamKey, transOverride,
    //                                 curTile2D, tileCount,
    //                                 queue);

    // Allocate the ray state/payload etc.


    // Start partitioning, again worst case work count
    uint32_t maxWorkCount = uint32_t(currentWorks.size() + currentLightWorks.size());
    auto[dIndices, dKeys] = rayPartitioner.Start(rayCount, maxWorkCount, true);

    // Do Full photon mapping
    //workMap.(key);

    // Do path tracing

    // Combine?


    // Generate rays
    //(*curCamWork)->GenerateRays<>(dRayDiffOut, ....);
    // Cast rays
    Span<uint32_t> dBackupRNGStates;
    tracerView.baseAccelerator.CastRays(dHitKeys, dHits, dBackupRNGStates,
                                        dRays, dIndices);
    // Generate work keys
    using namespace std::string_literals;
    static const std::string GenWorkKernelName = std::string(TypeName()) + "-KCGenerateWorkKeys"s;
    queue.IssueSaturatingKernel<KCGenerateWorkKeys>
    (
        GenWorkKernelName,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dHitKeys.size())},
        dKeys,
        ToConstSpan(dHitKeys),
        workHasher
    );

    // Partition
    auto
    [
        hPartitionCount,
        isHostVisible,
        hPartitionStartOffsets,
        hPartitionKeys,
        dPartitionIndices,
        dPartitionKeys
    ] = rayPartitioner.MultiPartition(dKeys, dIndices,
                                      Vector2ui::Zero(),
                                      Vector2ui::Zero(),
                                      queue, false);
    assert(isHostVisible);
    // Wait for results to be available in host buffers
    queue.Barrier().Wait();

    // Find out maximum RN count for the given work
    //work->
    Span<RandomNumber> dRandomNumbers;


    for(uint32_t i = 0; i < hPartitionCount[0]; i++)
    {
        uint32_t partitionStart = hPartitionStartOffsets[i];
        uint32_t partitionSize = (hPartitionStartOffsets[i + 1] -
                                  hPartitionStartOffsets[i]);

        auto dLocalIndices = dPartitionIndices.subspan(partitionStart,
                                                       partitionSize);


        //SurfaceWorkKey surfKey = SurfaceWorkKey(hPartitionKeys[i]);
        //uint32_t bactchId = surfKey.FetchBatchPortion();
        //const auto* work = globalWorkMap.at(batchId);

        //// Another phase(photon phase)
        //const auto* work = globalWorkMap2.at(batchId);

        //// RTTI?

        //if()
        //{

        //}
        //else
        //{

        //}

        //// We do not bounce rays so output can be empty
        //work->DoWork(// Input
        //             Span<RayDiff>(),
        //             Span<RayGMem>(),
        //             RayPayload{},
        //             // I-O
        //             dRayState,
        //             // Input
        //             dLocalIndices,
        //             dRandomNumbers,
        //             // Index-relative
        //             dRayDifferentials,
        //             dRays,
        //             dHits,
        //             RayPayload{},
        //             // Constants
        //             GlobalState{},
        //             queue);
    }

    // Filter the samples
    // Send the FBO


    ////...


    return RendererOutput
    {
        //.analytics = RendererAnalyticData
        //{
        //    samplePerSec,
        //    "M samples/s",
        //    spp,
        //    "spp",
        //    float(timer.Elapsed<Millisecond>()),
        //    mipSize,
        //    MRayColorSpaceEnum::MR_ACES_CG,
        //    static_cast<uint32_t>(textures.size()),
        //    static_cast<uint32_t>(curTex->MipCount())
        //},
        //.imageOut = renderOut
    };
}

void SurfaceRenderer::StopRender()
{}