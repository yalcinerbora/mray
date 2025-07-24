#include "AcceleratorEmbree.h"
#include "TransformC.h"

#include <cassert>
#include <map>

#include "Core/System.h"
#include "Core/Expected.h"
#include "Core/Filesystem.h"

#include "Device/GPUAlgScan.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgGeneric.h"

#include <embree4/rtcore.h>

#ifdef MRAY_LINUX
    #include <pmmintrin.h>
    #include <xmmintrin.h>
#endif

MRayEmbreeContext::MRayEmbreeContext()
    : device(rtcNewDevice(""))
{
    // From the tutorials for best performance
    // we need to change these
    // First time setting a control register on x86 device :)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    ErrorCallback(nullptr, rtcGetDeviceError(device),
                  rtcGetDeviceLastErrorMessage(device));
    rtcSetDeviceErrorFunction(device, ErrorCallback, nullptr);

    rtcSetDeviceMemoryMonitorFunction(device, AllocationCallback,
                                      &size);
}

MRayEmbreeContext::~MRayEmbreeContext()
{
    rtcReleaseDevice(device);
    rtcReleaseScene(scene);
}

void MRayEmbreeContext::ErrorCallback(void*, const RTCError code, const char* str)
{
    if(code == RTC_ERROR_NONE) return;

    std::string_view mode;
    switch(code)
    {
        case RTC_ERROR_UNKNOWN:             mode = "UNKOWN";        break;
        case RTC_ERROR_INVALID_ARGUMENT:    mode = "INVALID_ARG";   break;
        case RTC_ERROR_INVALID_OPERATION:   mode = "INVALID_OP";    break;
        case RTC_ERROR_OUT_OF_MEMORY:       mode = "OOM";           break;
        case RTC_ERROR_UNSUPPORTED_CPU:     mode = "CPU";           break;
        default: break;
    }
    MRAY_ERROR_LOG("[Embree]: [{}] \"{}\"", mode, str);
    std::exit(1);
}

bool MRayEmbreeContext::AllocationCallback(void* userPtr, ssize_t bytes, bool)
{
    auto* atomicSize = reinterpret_cast<std::atomic_int64_t*>(userPtr);
    atomicSize->fetch_add(bytes);
    // We don't care about too much allocation
    return true;
}

std::string_view BaseAcceleratorEmbree::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Hardware"sv;
    return BaseAccelTypeName<Name>;
}

BaseAcceleratorEmbree::BaseAcceleratorEmbree(ThreadPool& tp, const GPUSystem& sys,
                                             const AccelGroupGenMap& genMap,
                                             const AccelWorkGenMap& workGenMap)
    : BaseAcceleratorT<BaseAcceleratorEmbree>(tp, sys, genMap, workGenMap)
    , allMem(sys.AllGPUs(), 2_MiB, 32_MiB, true)
{}

AABB3 BaseAcceleratorEmbree::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    size_t totalInstanceCount = instanceOffsets.back();
    std::vector<size_t> groupHitRecordOffsets;
    groupHitRecordOffsets.resize(this->generatedAccels.size() + 1, 0);
    std::transform_inclusive_scan
    (
        this->generatedAccels.cbegin(),
        this->generatedAccels.cend(),
        groupHitRecordOffsets.begin() + 1,
        std::plus{},
        [](const auto& pair)
        {
            const auto& [_, agBase] = pair;
            auto& ag = static_cast<AcceleratorGroupEmbreeI&>(*agBase);
            return ag.HitRecordCount();
        }
    );
    size_t totalRecordCount = groupHitRecordOffsets.back();

    MemAlloc::AllocateMultiData(std::tie(hInstanceBatchStartOffsets,
                                         hInstanceHRStartOffsets,
                                         hAllHitRecordPtrs,
                                         hGlobalInstanceInvTransforms,
                                         hGlobalSceneHandles),
                                allMem,
                                {instanceOffsets.size(), totalInstanceCount + 1,
                                totalRecordCount, totalInstanceCount,
                                totalInstanceCount});
    queue.MemsetAsync(hInstanceHRStartOffsets.subspan(0, 1), 0x00);
    queue.MemcpyAsync(hInstanceBatchStartOffsets,
                      Span<const size_t>(instanceOffsets));

    // Alias the memory here we will invert the matrices later
    Span<RTCScene> hSceneHandles = hGlobalSceneHandles;
    Span<Matrix4x4> hInstanceMatrices = hGlobalInstanceInvTransforms;
    Span<uint32_t> hInstanceHRCounts = hInstanceHRStartOffsets.subspan(1);
    //
    embreeContext.scene = rtcNewScene(embreeContext.device);
    assert(instanceOffsets.size() == this->generatedAccels.size() + 1);
    [[maybe_unused]] uint32_t instanceCounter = 0;
    uint32_t accelI = 0;
    for(const auto& [_, agBase] : this->generatedAccels)
    {
        auto& ag = static_cast<AcceleratorGroupEmbreeI&>(*agBase);

        size_t hrStart = groupHitRecordOffsets[accelI];
        size_t hrEnd = groupHitRecordOffsets[accelI + 1];
        size_t hrLocalSize = hrEnd - hrStart;
        auto localHRPointers = hAllHitRecordPtrs.subspan(hrStart, hrLocalSize);
        //
        size_t iStart = instanceOffsets[accelI];
        size_t iEnd = instanceOffsets[accelI + 1];
        size_t iLocalSize = iEnd - iStart;
        auto localHandles = hSceneHandles.subspan(iStart, iLocalSize);
        auto localMatrices = hInstanceMatrices.subspan(iStart, iLocalSize);
        auto localHRCounts = hInstanceHRCounts.subspan(iStart, iLocalSize);
        //
        ag.AcquireIASConstructionParams(localHandles, localMatrices,
                                        localHRCounts, localHRPointers,
                                        queue);
        ag.OffsetAccelKeyInRecords(uint32_t(instanceOffsets[accelI]));
        queue.Barrier().Wait();
        for(size_t i = 0; i < localHandles.size(); i++)
        {
            auto g = rtcNewGeometry(embreeContext.device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryInstancedScene(g, localHandles[i]);

            // Maybe there is some optimizations on embree
            // lets not give identity matrix to embree.
            if(localMatrices[i] != Matrix4x4::Identity())
                rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                                        &localMatrices[i]);
            [[maybe_unused]]
            uint32_t instanceId = rtcAttachGeometry(embreeContext.scene, g);
            assert(instanceId == instanceCounter++);
            // Remove our reference counts, so that
            // we can release all data via releasing base accelerator
            rtcCommitGeometry(g);
            rtcReleaseGeometry(g);
            rtcReleaseScene(localHandles[i]);
        }
        //
        accelI++;
    }
    rtcCommitScene(embreeContext.scene);
    baseTraversable = rtcGetSceneTraversable(embreeContext.scene);

    // Inverse transform the matrices we will need it
    DeviceAlgorithms::InPlaceTransform(hGlobalInstanceInvTransforms, queue,
                                       [](Matrix4x4& m)
    {
        m.InverseSelf();
    });

    // Do the scan on this thread
    queue.Barrier().Wait();
    std::inclusive_scan(hInstanceHRStartOffsets.cbegin(),
                        hInstanceHRStartOffsets.cend(),
                        hInstanceHRStartOffsets.begin());

    // Thankfully, embree has this function.
    RTCBounds bounds;
    rtcGetSceneBounds(embreeContext.scene, &bounds);
    sceneAABB = AABB3(Vector3(bounds.lower_x, bounds.lower_y, bounds.lower_z),
                      Vector3(bounds.upper_x, bounds.upper_y, bounds.upper_z));
    return sceneAABB;
}

void BaseAcceleratorEmbree::AllocateForTraversal(size_t)
{}

void BaseAcceleratorEmbree::CastRays(// Output
                                     Span<HitKeyPack> dHitIds,
                                     Span<MetaHit> dHitParams,
                                     // I-O
                                     Span<BackupRNGState> dRNGStates,
                                     Span<RayGMem> dRays,
                                     // Input
                                     Span<const RayIndex> dRayIndices,
                                     const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    uint32_t rayCount = uint32_t(dRayIndices.size());
    uint32_t blockCount = Math::DivideUp(rayCount, EMBREE_BATCH_SIZE);
    using namespace std::string_view_literals;
    queue.IssueBlockLambda
    (
        "KCEmbreeIntersect"sv,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = 1,
        },
        [=, this](KernelCallParams kp)
        {
            uint32_t rayStart = kp.blockId * EMBREE_BATCH_SIZE;
            uint32_t rayEnd = (kp.blockId + 1) * EMBREE_BATCH_SIZE;
            rayEnd = std::min(rayEnd, rayCount);
            uint32_t localRayCount = rayEnd - rayStart;

            EmbreeRayQueryContext rqContext;
            rtcInitRayQueryContext(&rqContext.baseContext);
            RTCIntersectArguments intersectArgs
            {
                .flags = RTC_RAY_QUERY_FLAG_INCOHERENT,
                .feature_mask = RTCFeatureFlags(RTC_FEATURE_FLAG_ALL),
                .context = reinterpret_cast<RTCRayQueryContext*>(&rqContext),
                .filter = nullptr,
                .intersect = nullptr,
            };

            // Load the data to stack
            RTCRayHit16 rh = {};
            std::array<int, EMBREE_BATCH_SIZE> validList = {};
            // Ray
            std::fill_n(validList.begin(), localRayCount, -1);
            std::fill_n(rh.ray.mask, localRayCount, EMBREE_ALL_VALID_MASK);
            std::iota(rh.ray.id, rh.ray.id + localRayCount, 0);
            // Hit
            std::fill_n(rh.hit.geomID, EMBREE_BATCH_SIZE,
                        RTC_INVALID_GEOMETRY_ID);
            // From GMem
            for(uint32_t i = 0; i < localRayCount; i++)
            {
                // Rays
                uint32_t rIndex = dRayIndices[rayStart + i];
                auto[ray, tMM] = RayFromGMem(dRays, rIndex);
                rh.ray.dir_x[i] = ray.Dir()[0];
                rh.ray.dir_y[i] = ray.Dir()[1];
                rh.ray.dir_z[i] = ray.Dir()[2];
                //
                rh.ray.org_x[i] = ray.Pos()[0];
                rh.ray.org_y[i] = ray.Pos()[1];
                rh.ray.org_z[i] = ray.Pos()[2];
                // Embree does not support negative
                // tnear/tfar so check it
                assert(tMM >= Vector2::Zero());
                rh.ray.tnear[i] = tMM[0];
                rh.ray.tfar[i] = tMM[1];
                // RNG
                rqContext.rng.emplace_back(dRNGStates[rIndex]);
            }

            // Launch!
            rtcTraversableIntersect16(validList.data(),
                                      baseTraversable,
                                      &rh, &intersectArgs);

            // No matter what, relaod the rng state back.
            // Even if there is not hit ray may used it
            // during traversal.
            //
            // Flush the array, RNG class automatically
            // writes back to the given global buffer
            rqContext.rng.clear();
            //
            for(uint32_t i = 0; i < localRayCount; i++)
            {
                uint32_t rIndex         = dRayIndices[rayStart + i];
                uint32_t primBatchIndex = rh.hit.geomID[i];
                uint32_t primIndex      = rh.hit.primID[i];
                uint32_t instanceIndex  = rh.hit.instID[0][i];

                // Skip if no hit has occured,
                assert(instanceIndex != RTC_INVALID_GEOMETRY_ID);
                if(primBatchIndex == RTC_INVALID_GEOMETRY_ID)
                    continue;

                uint32_t iOffset = hInstanceHRStartOffsets[instanceIndex];
                uint32_t globalRecordIndex = iOffset + primBatchIndex;
                const auto& record = *hAllHitRecordPtrs[globalRecordIndex];
                dHitIds[rIndex] = HitKeyPack
                {
                    .primKey = record.dPrimKeys[primIndex],
                    .lightOrMatKey = record.lmKey,
                    .transKey = record.transformKey,
                    .accelKey = record.acceleratorKey
                };

                // Embree-MRay barycentric coordinate mismatch
                Vector2 ab = Vector2(rh.hit.u[i], rh.hit.v[i]);
                if(record.isTriangle)
                    ab = EmbreeBaryToMRay(ab);

                dHitParams[rIndex] = MetaHit(ab);
                UpdateTMax(dRays, rIndex, rh.ray.tfar[i]);
            }
        }
    );
}

void BaseAcceleratorEmbree::CastVisibilityRays(Bitspan<uint32_t> dIsVisibleBuffer,
                                               // I-O
                                               Span<BackupRNGState> dRNGStates,
                                               // Input
                                               Span<const RayGMem> dRays,
                                               Span<const RayIndex> dRayIndices,
                                               const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Visibilty Casting"sv);
    const auto _ = annotation.AnnotateScope();

    uint32_t rayCount = uint32_t(dRayIndices.size());
    uint32_t blockCount = Math::DivideUp(rayCount, EMBREE_BATCH_SIZE);
    using namespace std::string_view_literals;
    queue.IssueBlockLambda
    (
        "KCEmbreeOccluded"sv,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = 1,
        },
        [=, this](KernelCallParams kp)
        {
            uint32_t rayStart = kp.blockId * EMBREE_BATCH_SIZE;
            uint32_t rayEnd = (kp.blockId + 1) * EMBREE_BATCH_SIZE;
            rayEnd = std::min(rayEnd, rayCount);
            uint32_t localRayCount = rayEnd - rayStart;

            EmbreeRayQueryContext rqContext;
            rtcInitRayQueryContext(&rqContext.baseContext);
            RTCOccludedArguments intersectArgs
            {
                .flags = RTC_RAY_QUERY_FLAG_INCOHERENT,
                .feature_mask = RTCFeatureFlags(RTC_FEATURE_FLAG_ALL),
                .context = reinterpret_cast<RTCRayQueryContext*>(&rqContext),
                .filter = nullptr,
                .occluded = nullptr
            };
            // Load the data to stack
            RTCRay16 r = {};
            std::array<int, EMBREE_BATCH_SIZE> validList = {};
            // Ray
            std::fill_n(validList.begin(), localRayCount, -1);
            std::fill_n(r.mask, localRayCount, EMBREE_ALL_VALID_MASK);
            std::iota(r.id, r.id + localRayCount, 0);
            // From GMem
            for(uint32_t i = 0; i < localRayCount; i++)
            {
                // Rays
                uint32_t rIndex = dRayIndices[rayStart + i];
                auto[ray, tMM] = RayFromGMem(dRays, rIndex);
                r.dir_x[i] = ray.Dir()[0];
                r.dir_y[i] = ray.Dir()[1];
                r.dir_z[i] = ray.Dir()[2];
                //
                r.org_x[i] = ray.Pos()[0];
                r.org_y[i] = ray.Pos()[1];
                r.org_z[i] = ray.Pos()[2];
                // Embree does not support negative
                // tnear/tfar so check it
                assert(tMM >= Vector2::Zero());
                r.tnear[i] = tMM[0];
                r.tfar[i] = tMM[1];
                // RNG
                rqContext.rng.emplace_back(dRNGStates[rIndex]);
            }

            // Launch!
            rtcTraversableOccluded16(validList.data(),
                                     baseTraversable, &r,
                                     &intersectArgs);

            // No matter what, relaod the rng state back.
            // Even if there is not hit ray may used it
            // during traversal.
            //
            // Flush the array, RNG class automatically
            // writes back to the given global buffer
            rqContext.rng.clear();
            //
            for(uint32_t i = 0; i < localRayCount; i++)
            {
                uint32_t rIndex = dRayIndices[rayStart + i];
                bool isVisible = !(r.tfar[i] == EMBREE_IS_OCCLUDED_RAY);
                dIsVisibleBuffer.SetBitParallel(rIndex, isVisible);
            }
        }
    );
}

void BaseAcceleratorEmbree::CastLocalRays(// Output
                                          Span<HitKeyPack> dHitIds,
                                          Span<MetaHit> dHitParams,
                                          // I-O
                                          Span<BackupRNGState> dRNGStates,
                                          Span<RayGMem> dRays,
                                          // Input
                                          Span<const RayIndex> dRayIndices,
                                          Span<const AcceleratorKey> dAccelKeys,
                                          CommonKey dAccelKeyBatchPortion,
                                          const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Local Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    size_t groupStart = hInstanceBatchStartOffsets[dAccelKeyBatchPortion];
    size_t groupEnd = hInstanceBatchStartOffsets[dAccelKeyBatchPortion + 1];
    Span<const Matrix4x4> hLocalInvTransforms = hGlobalInstanceInvTransforms.subspan(groupStart, groupEnd - groupStart);
    Span<const RTCScene> hLocalScenes =  hGlobalSceneHandles.subspan(groupStart, groupEnd - groupStart);

    uint32_t rayCount = uint32_t(dRayIndices.size());
    using namespace std::string_view_literals;
    queue.IssueWorkLambda
    (
        "Ray Casting"sv,
        DeviceWorkIssueParams{ .workCount = rayCount },
        [=, this](KernelCallParams kp)
        {
            // We can't guarantee that the adjacent
            // threads have the same accelerator key.
            // So we trace the rays one by one.
            // TODO: Technically dAccelerator keys
            // are sorted we may abuse that later maybe.
            // to increase batch size
            uint32_t i = kp.GlobalId();
            uint32_t rIndex = dRayIndices[i];
            CommonKey accIndex = dAccelKeys[rIndex].FetchIndexPortion();

            const Matrix4x4& transform = hLocalInvTransforms[accIndex];
            RTCScene t = hLocalScenes[accIndex];

            EmbreeRayQueryContext rqContext;
            rtcInitRayQueryContext(&rqContext.baseContext);
            RTCIntersectArguments intersectArgs
            {
                .flags = RTC_RAY_QUERY_FLAG_INCOHERENT,
                .feature_mask = RTCFeatureFlags(RTC_FEATURE_FLAG_ALL),
                .context = reinterpret_cast<RTCRayQueryContext*>(&rqContext),
                .filter = nullptr,
                .intersect = nullptr
            };
            // Load the data to stack
            auto [ray, tMM] = RayFromGMem(dRays, rIndex);
            // We need to manually transform the ray here
            Vector3 dir = transform * ray.Dir();
            Vector3 pos = Vector3(transform * Vector4(ray.Pos(), Float(1)));
            //
            RTCRayHit rh = RTCRayHit{};
            rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rh.ray.dir_x = dir[0];
            rh.ray.dir_y = dir[1];
            rh.ray.dir_z = dir[2];
            //
            rh.ray.org_x = pos[0];
            rh.ray.org_y = pos[1];
            rh.ray.org_z = pos[2];
            // Embree does not support negative
            // tnear/tfar so check it
            assert(tMM >= Vector2::Zero());
            rh.ray.tnear = tMM[0];
            rh.ray.tfar = tMM[1];
            // RNG
            rqContext.rng.emplace_back(dRNGStates[rIndex]);
            // Launch!
            rtcIntersect1(t, &rh, &intersectArgs);
            //
            uint32_t primBatchIndex = rh.hit.geomID;
            uint32_t primIndex      = rh.hit.primID;
            uint32_t instanceIndex  = rh.hit.instID[0];
            // No matter what, relaod the rng state back.
            // Even if there is not hit ray may used it
            // during traversal.
            //
            // Flush the array, RNG class automatically
            // writes back to the given global buffer
            rqContext.rng.clear();

            // Skip if no hit has occured,
            if(primBatchIndex == RTC_INVALID_GEOMETRY_ID)
                return;

            uint32_t iOffset = hInstanceHRStartOffsets[instanceIndex];
            uint32_t globalRecordIndex = iOffset + primBatchIndex;
            const auto& record = *hAllHitRecordPtrs[globalRecordIndex];
            dHitIds[rIndex] = HitKeyPack
            {
                .primKey        = record.dPrimKeys[primIndex],
                .lightOrMatKey  = record.lmKey,
                .transKey       = record.transformKey,
                .accelKey       = record.acceleratorKey
            };
            dHitParams[rIndex] = MetaHit(Vector2(rh.hit.u, rh.hit.v));
            UpdateTMax(dRays, rIndex, rh.ray.tfar);
        }
    );
}

size_t BaseAcceleratorEmbree::GPUMemoryUsage() const
{
    int64_t embreeAllocSize = embreeContext.size.load();
    assert(embreeAllocSize >= 0);
    size_t totalSize = size_t(embreeAllocSize);
    for(const auto& [_, accelGroup] : this->generatedAccels)
    {
        totalSize += accelGroup->GPUMemoryUsage();
    }
    return totalSize;
}

RTCDevice BaseAcceleratorEmbree::GetRTCDeviceHandle() const
{
    return embreeContext.device;
}
