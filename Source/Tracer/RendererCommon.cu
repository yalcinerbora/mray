#pragma once

#include "RendererC.h"
#include "MediumC.h"
#include "RendererCommon.h"

#include "Device/GPUSystem.hpp"

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateSurfaceWorkKeys(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                               MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                               MRAY_GRID_CONSTANT const RenderSurfaceWorkHasher workHasher)
{
    assert(dWorkKey.size() == dInputKeys.size());

    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dInputKeys.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        dWorkKey[i] = workHasher.GenerateWorkKeyGPU(dInputKeys[i], i);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateSurfaceWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                                       MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                       MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                                       MRAY_GRID_CONSTANT const RenderSurfaceWorkHasher workHasher)
{
    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        RayIndex keyIndex = dIndices[i];
        auto keyPack = dInputKeys[keyIndex];
        dWorkKey[i] = workHasher.GenerateWorkKeyGPU(keyPack, keyIndex);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateMediumWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                                      MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                      MRAY_GRID_CONSTANT const Span<const VolumeKeyPack> dInputKeys,
                                      MRAY_GRID_CONSTANT const RenderMediumWorkHasher workHasher)
{
    KernelCallParams kp;
    uint32_t keyCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        RayIndex keyIndex = dIndices[i];
        auto keyPack = dInputKeys[keyIndex];
        dWorkKey[i] = workHasher.GenerateWorkKeyGPU(keyPack, keyIndex);
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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCCopyRaysIndirect(MRAY_GRID_CONSTANT const Span<RayGMem> dRaysOut,
                        MRAY_GRID_CONSTANT const Span<RayCone> dRayDiffOut,
                        MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                        MRAY_GRID_CONSTANT const Span<const RayGMem> dRaysIn,
                        MRAY_GRID_CONSTANT const Span<const RayCone> dRayDiffIn)
{
    KernelCallParams kp;
    uint32_t pathCount = static_cast<uint32_t>(dIndices.size());
    for(uint32_t i = kp.GlobalId(); i < pathCount; i += kp.TotalSize())
    {
        RayIndex index = dIndices[i];
        dRaysOut[index] = dRaysIn[index];
        dRayDiffOut[index] = dRayDiffIn[index];
    }
}


MRAY_HOST
void
RenderSurfaceWorkHasher::PopulateHashesAndKeys(const TracerView& tracerView,
                                               const RenderWorkList& curWorks,
                                               const RenderLightWorkList& curLightWorks,
                                               const RenderCameraWorkList& curCamWorks,
                                               uint32_t maxRayCount,
                                               const GPUQueue& queue)
{
    size_t totalWorkBatchCount = (curWorks.size() +
                                  curLightWorks.size() +
                                  curCamWorks.size());
    std::vector<CommonKey> hHashes;
    std::vector<CommonKey> hBatchIds;
    hHashes.reserve(totalWorkBatchCount);
    hBatchIds.reserve(totalWorkBatchCount);
    uint32_t primMaxCount = 0;
    uint32_t lmMaxCount = 0;
    uint32_t transMaxCount = 0;

    for(const auto& work : curWorks)
    {
        MatGroupId matGroupId = work.mgId;
        PrimGroupId primGroupId = work.pgId;
        TransGroupId transGroupId = work.tgId;

        auto pK = PrimitiveKey::CombinedKey(std::bit_cast<CommonKey>(primGroupId), 0u);
        auto mK = LightOrMatKey::CombinedKey(IS_MAT_KEY_FLAG, std::bit_cast<CommonKey>(matGroupId), 0u);
        auto tK = TransformKey::CombinedKey(std::bit_cast<CommonKey>(transGroupId), 0u);
        HitKeyPack kp =
        {
            .primKey        = pK,
            .lightOrMatKey  = mK,
            .transKey       = tK,
            .accelKey       = AcceleratorKey::InvalidKey()
        };
        hHashes.emplace_back(HashWorkBatchPortion(kp));
        hBatchIds.push_back(work.workGroupId);

        // Might as well check the data amount here
        uint32_t primCount = uint32_t(tracerView.primGroups.at(primGroupId)->get()->TotalPrimCount());
        uint32_t matCount = uint32_t(tracerView.matGroups.at(matGroupId)->get()->TotalItemCount());
        uint32_t transformCount = uint32_t(tracerView.transGroups.at(transGroupId)->get()->TotalItemCount());
        primMaxCount = Math::Max(primMaxCount, primCount);
        lmMaxCount = Math::Max(lmMaxCount, matCount);
        transMaxCount = Math::Max(transMaxCount, transformCount);
    }
    // Push light hashes
    for(const auto& work : curLightWorks)
    {
        LightGroupId lightGroupId = work.lgId;
        TransGroupId transGroupId = work.tgId;
        const auto& lightGroup = tracerView.lightGroups.at(lightGroupId)->get();
        const auto& transformGroup = tracerView.transGroups.at(transGroupId)->get();
        CommonKey primGroupId = lightGroup->GenericPrimGroup().GroupId();

        auto lK = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG, static_cast<CommonKey>(lightGroupId), 0u);
        auto tK = TransformKey::CombinedKey(std::bit_cast<CommonKey>(transGroupId), 0u);
        auto pK = PrimitiveKey::CombinedKey(primGroupId, 0u);
        HitKeyPack kp =
        {
            .primKey        = pK,
            .lightOrMatKey  = lK,
            .transKey       = tK,
            .accelKey       = AcceleratorKey::InvalidKey()
        };
        hHashes.emplace_back(HashWorkBatchPortion(kp));
        hBatchIds.push_back(work.workGroupId);

        uint32_t lightCount = uint32_t(lightGroup->TotalItemCount());
        uint32_t transformCount = uint32_t(transformGroup->TotalItemCount());
        lmMaxCount = Math::Max(lmMaxCount, lightCount);
        transMaxCount = Math::Max(transMaxCount, transformCount);
    }
    // TODO: Add camera hashes as well, it may require some redesign
    // Currently we "FF..F" these since we do not support light tracing yet
    for(const auto& work : curCamWorks)
    {
        hHashes.emplace_back(std::numeric_limits<uint32_t>::max());
        hBatchIds.push_back(work.workGroupId);
    }

    // Find bit count
    maxMatOrLightIdBits = Bit::RequiredBitsToRepresent(lmMaxCount);
    maxPrimIdBits       = Bit::RequiredBitsToRepresent(primMaxCount);
    maxTransIdBits      = Bit::RequiredBitsToRepresent(transMaxCount);
    maxIndexBits        = Bit::RequiredBitsToRepresent(maxRayCount);

    queue.MemcpyAsync(dWorkBatchHashes, Span<const CommonKey>(hHashes));
    queue.MemcpyAsync(dWorkBatchIds, Span<const CommonKey>(hBatchIds));
    queue.Barrier().Wait();
}

MRAY_HOST
void
RenderMediumWorkHasher::PopulateHashesAndKeys(const TracerView& tracerView,
                                              const RenderMediumWorkList& curWorks,
                                              uint32_t maxRayCount,
                                              const GPUQueue& queue)
{
    size_t totalWorkBatchCount = (curWorks.size());
    std::vector<CommonKey> hHashes;
    std::vector<CommonKey> hBatchIds;
    hHashes.reserve(totalWorkBatchCount);
    hBatchIds.reserve(totalWorkBatchCount);

    uint32_t medMaxCount = 0;
    uint32_t transMaxCount = 0;
    for(const auto& work : curWorks)
    {
        MediumGroupId mediumGroupId = work.mgId;
        TransGroupId transGroupId = work.tgId;

        auto mK = MediumKey::CombinedKey(Bit::BitCast<CommonKey>(mediumGroupId), 0u);
        auto tK = TransformKey::CombinedKey(Bit::BitCast<CommonKey>(transGroupId), 0u);
        VolumeKeyPack kp =
        {
            .medKey   = mK,
            .transKey = tK
        };
        hHashes.emplace_back(HashWorkBatchPortion(kp));
        hBatchIds.push_back(work.workGroupId);

        // Might as well check the data amount here
        uint32_t mediaCount = uint32_t(tracerView.mediumGroups.at(mediumGroupId)->get()->TotalItemCount());
        uint32_t transformCount = uint32_t(tracerView.transGroups.at(transGroupId)->get()->TotalItemCount());
        medMaxCount = Math::Max(medMaxCount, mediaCount);
        transMaxCount = Math::Max(transMaxCount, transformCount);
    }

    // Find bit count
    maxMediumIdBits = Bit::RequiredBitsToRepresent(medMaxCount);
    maxTransIdBits  = Bit::RequiredBitsToRepresent(transMaxCount);
    maxIndexBits    = Bit::RequiredBitsToRepresent(maxRayCount);

    queue.MemcpyAsync(dWorkBatchHashes, Span<const CommonKey>(hHashes));
    queue.MemcpyAsync(dWorkBatchIds, Span<const CommonKey>(hBatchIds));
    queue.Barrier().Wait();
}

uint32_t RendererBase::GenerateWorkMappings(uint32_t workStart)
{
    using Algo::PartitionRange;
    const auto& flatSurfs = tracerView.flattenedSurfaces;
    assert(std::is_sorted(tracerView.flattenedSurfaces.cbegin(),
                          tracerView.flattenedSurfaces.cend()));
    auto partitions = Algo::PartitionRange(flatSurfs.cbegin(),
                                           flatSurfs.cend());
    for(const auto& p : partitions)
    {
        size_t i = p[0];
        MatGroupId mgId{std::bit_cast<MaterialKey>(flatSurfs[i].mId).FetchBatchPortion()};
        PrimGroupId pgId{std::bit_cast<PrimBatchKey>(flatSurfs[i].pId).FetchBatchPortion()};
        TransGroupId tgId{std::bit_cast<TransformKey>(flatSurfs[i].tId).FetchBatchPortion()};
        // These should be checked beforehand, while actually creating
        // the surface
        const MaterialGroupPtr& mg = tracerView.matGroups.at(mgId).value();
        const PrimGroupPtr& pg = tracerView.primGroups.at(pgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view mgName = mg->Name();
        std::string_view pgName = pg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderWorkType;
        std::string workName = CreateRenderWorkType(mgName, pgName, tgName);

        auto loc = workPack.workMap.at(workName);
        if(!loc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Mat/Prim/Transform "
                            "triplet of \"{}/{}/{}\"",
                            rendererName, mgName, pgName, tgName);
        }
        RenderWorkGenerator generator = loc->get();
        RenderWorkPtr ptr = generator(*mg.get(), *pg.get(), *tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentWorks.emplace_back
        (
            RenderWorkStruct
            {
                .mgId = mgId,
                .pgId = pgId,
                .tgId = tgId,
                .workGroupId = workStart++,
                .workPtr = std::move(ptr)
            }
        );
    }
    return workStart;
}

uint32_t RendererBase::GenerateLightWorkMappings(uint32_t workStart)
{
    using Algo::PartitionRange;
    const auto& lightSurfs = tracerView.lightSurfs;
    using LightSurfP = Pair<LightSurfaceId, LightSurfaceParams>;
    auto LightSurfIsLess = [](const LightSurfP& left, const LightSurfP& right)
    {
        auto GetLG = [](LightId id) -> CommonKey
        {
            return std::bit_cast<LightKey>(id).FetchBatchPortion();
        };
        auto GetTG = [](TransformId id) -> CommonKey
        {
            return std::bit_cast<TransformKey>(id).FetchBatchPortion();
        };
        return (Tuple(GetLG(left.second.lightId), GetTG(left.second.transformId)) <
                Tuple(GetLG(right.second.lightId), GetTG(right.second.transformId)));
    };
    assert(std::is_sorted(lightSurfs.cbegin(), lightSurfs.cend(),
                          LightSurfIsLess));

    auto partitions = Algo::PartitionRange(lightSurfs.cbegin(), lightSurfs.cend(),
                                           LightSurfIsLess);

    auto AddWork = [&, this](const LightSurfaceParams& lSurf,
                             bool isBoundaryLight)
    {
        LightGroupId lgId{std::bit_cast<LightKey>(lSurf.lightId).FetchBatchPortion()};
        TransGroupId tgId{std::bit_cast<TransformKey>(lSurf.transformId).FetchBatchPortion()};
        // These should be checked beforehand, while actually creating
        // the surface
        const LightGroupPtr& lg = tracerView.lightGroups.at(lgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view lgName = lg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderLightWorkType;
        std::string workName = CreateRenderLightWorkType(lgName, tgName);

        auto loc = workPack.lightWorkMap.at(workName);
        if(!loc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Light/Transform "
                            "pair of \"{}/{}\"",
                            rendererName, lgName, tgName);
        }
        if(isBoundaryLight)
        {
            bool isPrimBacked = lg.get()->IsPrimitiveBacked();
            if(isPrimBacked)
            {
                throw MRayError("[{}]: Primitive-backed light ({}) is requested "
                                "as a boundary material!",
                                rendererName, lgName);
            }
            // Boundary material is not primitive-backed by definition
            // we can set the index portion as zero
            auto lK = std::bit_cast<LightKey>(lSurf.lightId);
            auto lmK = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG,
                                                  lK.FetchBatchPortion(),
                                                  lK.FetchIndexPortion());
            auto tK = std::bit_cast<TransformKey>(lSurf.transformId);
            CommonKey primGroupId = lg->GenericPrimGroup().GroupId();
            auto pK = PrimitiveKey::CombinedKey(primGroupId, 0u);
            boundaryLightKeyPack = HitKeyPack
            {
                .primKey = pK,
                .lightOrMatKey = lmK,
                .transKey = tK,
                .accelKey = AcceleratorKey::InvalidKey()
            };
        }

        RenderLightWorkGenerator generator = loc->get();
        RenderLightWorkPtr ptr = generator(*lg.get(), *tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentLightWorks.emplace_back
        (
            RenderLightWorkStruct
            {
                .lgId = lgId,
                .tgId = tgId,
                .workGroupId = workStart++,
                .workPtr = std::move(ptr)
            }
        );
    };

    for(const Vector2ul& p : partitions)
    {
        const auto& lSurf = lightSurfs[p[0]].second;
        AddWork(lSurf, false);
    }
    AddWork(tracerView.boundarySurface, true);


    return workStart;
}

uint32_t RendererBase::GenerateCameraWorkMappings(uint32_t workStart)
{
    const auto& camSurfs = tracerView.camSurfs;
    using CamSurfP = Pair<CamSurfaceId, CameraSurfaceParams>;
    auto CamSurfIsLess = [](const CamSurfP& left, const CamSurfP& right)
    {
        auto GetCG = [](CameraId id) -> CommonKey
        {
            return std::bit_cast<CameraKey>(id).FetchBatchPortion();
        };
        auto GetTG = [](TransformId id) -> CommonKey
        {
            return std::bit_cast<TransformKey>(id).FetchBatchPortion();
        };
        return (Tuple(GetCG(left.second.cameraId), GetTG(left.second.transformId)) <
                Tuple(GetCG(right.second.cameraId), GetTG(right.second.transformId)));
    };
    assert(std::is_sorted(camSurfs.cbegin(), camSurfs.cend(),
                          CamSurfIsLess));

    auto partitions = Algo::PartitionRange(camSurfs.cbegin(), camSurfs.cend(),
                                           CamSurfIsLess);
    for(const auto& p : partitions)
    {
        size_t i = p[0];
        CameraGroupId cgId{std::bit_cast<CameraKey>(camSurfs[i].second.cameraId).FetchBatchPortion()};
        TransGroupId tgId{std::bit_cast<TransformKey>(camSurfs[i].second.transformId).FetchBatchPortion()};
        // These should be checked beforehand, while actually creating
        // the surface
        const CameraGroupPtr& cg = tracerView.camGroups.at(cgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view cgName = cg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderCameraWorkType;
        std::string workName = CreateRenderCameraWorkType(cgName, tgName);

        auto loc = workPack.camWorkMap.at(workName);
        if(!loc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Camera/Transform "
                            "pair of \"{}/{}\"",
                            rendererName, cgName, tgName);
        }
        RenderCameraWorkGenerator generator = loc->get();
        RenderCameraWorkPtr ptr = generator(*cg.get(), *tg.get(), gpuSystem);

        // Put this ptr somewhere... safe
        currentCameraWorks.emplace_back
        (
            RenderCameraWorkStruct
            {
                .cgId = cgId,
                .tgId = tgId,
                .workGroupId = workStart++,
                .workPtr = std::move(ptr)
            }
        );
    }
    return workStart;
}

uint32_t RendererBase::GenerateMediumWorkMappings(uint32_t workStart)
{
    auto GenMediumWork = [&](const VolumeKeyPack& kp)
    {
        // Skip identity interfaces
        if(kp.medKey == MediumKey::InvalidKey()) return;

        using Bit::BitCast;
        MediumGroupId mgId = BitCast<MediumGroupId>(kp.medKey.FetchBatchPortion());
        TransGroupId tgId = BitCast<TransGroupId>(kp.transKey.FetchBatchPortion());

        // Unlike other works we linear search here since it should not scale
        // (Like if you render 500 volumes on a scene, this should not be a bottleneck)
        auto loc = std::find_if(currentMediumWorks.cbegin(), currentMediumWorks.cend(),
                                [mgId, tgId](const RenderMediumWorkStruct& mw)
        {
            return (mw.mgId == mgId && mw.tgId == tgId);
        });
        if(loc != currentMediumWorks.cend()) return;

        // These should be checked beforehand, while actually creating
        // the surface
        const MediumGroupPtr& mg = tracerView.mediumGroups.at(mgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view mgName = mg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderMediumWorkType;
        std::string workName = CreateRenderMediumWorkType(mgName, tgName);
        auto workGenLoc = workPack.mediumWorkMap.at(workName);
        if(!workGenLoc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Medium/Transform "
                            "pair of \"{}/{}\"",
                            rendererName, mgName, tgName);
        }
        RenderMediumWorkGenerator generator = workGenLoc->get();
        RenderMediumWorkPtr ptr = generator(*mg.get(), *tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentMediumWorks.emplace_back
        (
            RenderMediumWorkStruct
            {
                .mgId = mgId,
                .tgId = tgId,
                .workGroupId = workStart++,
                .workPtr = std::move(ptr)
            }
        );
    };
    //
    for(const auto& [_, p] : tracerView.globalVolumeList)
        GenMediumWork(p);
    return workStart;
}