#pragma once

#include "AcceleratorWork.h"

template<AccelGroupC AG, TransformGroupC TG, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenPrimCenters(// Output
                      MRAY_GRID_CONSTANT const Span<Vector3> dAllPrimCenters,
                      // Inputs
                      MRAY_GRID_CONSTANT const Span<const uint32_t> dSegmentRanges,
                      MRAY_GRID_CONSTANT const Span<const TransformKey> dTransformKeys,
                      MRAY_GRID_CONSTANT const Span<const PrimitiveKey> dAllLeafs,
                      // Constants
                      MRAY_GRID_CONSTANT const uint32_t blockPerInstance,
                      MRAY_GRID_CONSTANT const uint32_t instanceCount,
                      MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                      MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA)
{
    using PG = typename AG::PrimitiveGroup;
    using TransContext = typename AccTransformContextType<AG, TG>::Result;
    using Prim = typename PG:: template Primitive<TransContext>;

    static constexpr auto TPB = StaticThreadPerBlock1D();
    // Block-stride loop
    KernelCallParams kp;
    uint32_t blockCount = instanceCount * blockPerInstance;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        // Current instance index of this iteration
        uint32_t instanceI = bI / blockPerInstance;
        uint32_t localBI = bI % blockPerInstance;

        if(instanceI >= instanceCount) continue;

        //
        uint32_t instanceLocalThreadId = localBI * TPB + kp.threadId;
        uint32_t primPerPass = TPB * blockPerInstance;
        //
        Vector2ui range = Vector2ui(dSegmentRanges[instanceI],
                                    dSegmentRanges[instanceI + 1]);
        auto dLocalLeafs = dAllLeafs.subspan(range[0],
                                             range[1] - range[0]);
        auto dLocalPrimCenters = dAllPrimCenters.subspan(range[0],
                                                         range[1] - range[0]);

        // Loop invariant data
        TransformKey tKey = TransformKey::InvalidKey();
        if constexpr(!std::is_same_v<TransformGroupIdentity, TG>)
            tKey = dTransformKeys[instanceI];

        // Finally multi-block primitive loop
        uint32_t totalPrims = static_cast<uint32_t>(dLocalPrimCenters.size());
        for(uint32_t i = instanceLocalThreadId; i < totalPrims;
            i += primPerPass)
        {
            PrimitiveKey pKey = dLocalLeafs[i];
            TransContext tContext = GenerateTransformContext(tSoA, pSoA, tKey, pKey);
            Prim prim(tContext, pSoA, pKey);

            dLocalPrimCenters[i] = prim.GetCenter();
        }
    }
}

template<AccelGroupC AG, TransformGroupC TG, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGeneratePrimAABBs(// Output
                         MRAY_GRID_CONSTANT const Span<AABB3> dAllPrimAABBs,
                         // Inputs
                         MRAY_GRID_CONSTANT const Span<const uint32_t> dSegmentRanges,
                         MRAY_GRID_CONSTANT const Span<const TransformKey> dTransformKeys,
                         MRAY_GRID_CONSTANT const Span<const PrimitiveKey> dAllLeafs,
                         // Constants
                         MRAY_GRID_CONSTANT const uint32_t blockPerInstance,
                         MRAY_GRID_CONSTANT const uint32_t instanceCount,
                         MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                         MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA)
{
    using PG = typename AG::PrimitiveGroup;
    using TransContext = typename AccTransformContextType<AG, TG>::Result;
    using Prim = typename PG:: template Primitive<TransContext>;

    static constexpr auto TPB = StaticThreadPerBlock1D();
    // Block-stride loop
    KernelCallParams kp;
    uint32_t blockCount = instanceCount * blockPerInstance;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        // Current instance index of this iteration
        uint32_t instanceI = bI / blockPerInstance;
        uint32_t localBI = bI % blockPerInstance;

        if(instanceI >= instanceCount) continue;

        //
        uint32_t instanceLocalThreadId = localBI * TPB + kp.threadId;
        uint32_t primPerPass = TPB * blockPerInstance;
        //
        Vector2ui range = Vector2ui(dSegmentRanges[instanceI],
                                    dSegmentRanges[instanceI + 1]);
        auto dLocalLeafs = dAllLeafs.subspan(range[0],
                                             range[1] - range[0]);
        auto dLocalPrimAABBs = dAllPrimAABBs.subspan(range[0],
                                                     range[1] - range[0]);

        // Loop invariant data
        TransformKey tKey = TransformKey::InvalidKey();
        if constexpr(!std::is_same_v<TransformGroupIdentity, TG>)
            tKey = dTransformKeys[instanceI];

        // Finally multi-block primitive loop
        uint32_t totalPrims = static_cast<uint32_t>(dLocalPrimAABBs.size());
        for(uint32_t i = instanceLocalThreadId; i < totalPrims;
            i += primPerPass)
        {
            PrimitiveKey pKey = dLocalLeafs[i];
            TransContext tContext = GenerateTransformContext(tSoA, pSoA, tKey, pKey);
            Prim prim(tContext, pSoA, pKey);

            dLocalPrimAABBs[i] = prim.GetAABB();
        }
    }
}

template<TransformGroupC TG>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGetCommonTransforms(// Output
                           MRAY_GRID_CONSTANT const Span<Matrix3x4> dTransforms,
                           // Inputs
                           MRAY_GRID_CONSTANT const Span<const TransformKey> dTransformKeys,
                           MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA)
{
    assert(dTransformKeys.size() == dTransforms.size());
    uint32_t tCount = static_cast<uint32_t>(dTransformKeys.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < tCount; i += kp.TotalSize())
    {
        Matrix3x4 transform = TG::AcquireCommonTransform(tSoA, dTransformKeys[i]);
        dTransforms[i] = transform;
    }
}

template<AccelGroupC AG, TransformGroupC TG, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCTransformLocallyConstantAABBs(// Output
                                     MRAY_GRID_CONSTANT const Span<AABB3> dInstanceAABBs,
                                     // Input
                                     MRAY_GRID_CONSTANT const Span<const AABB3> dConcreteAABBs,
                                     MRAY_GRID_CONSTANT const Span<const uint32_t> dConcreteIndicesOfInstances,
                                     MRAY_GRID_CONSTANT const Span<const TransformKey> dInstanceTransformKeys,
                                     // Constants
                                     MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                                     MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA)
{
    using PG = typename AG::PrimitiveGroup;
    static_assert(PG::TransformLogic == PrimTransformType::LOCALLY_CONSTANT_TRANSFORM,
                  "This kernel only works for LOCALLY_CONSTANT_TRANSFORM typed prims");

    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId();
        globalId < static_cast<uint32_t>(dConcreteIndicesOfInstances.size());
        globalId += kp.TotalSize())
    {
        TransformKey tKey = dInstanceTransformKeys[globalId];
        uint32_t index = dConcreteIndicesOfInstances[globalId];
        AABB3 cAABB = dConcreteAABBs[index];
        // Acquire Transform Context Generator & Type
        using TransContext = typename AccTransformContextType<AG, TG>::Result;
        TransContext tContext = GenerateTransformContext(tSoA, pSoA, tKey,
                                                         PrimitiveKey::InvalidKey());
        // Finally transform and write
        AABB3 worldAABB = tContext.Apply(cAABB);
        dInstanceAABBs[globalId] = worldAABB;
    }
}

template<AccelGroupC AG, TransformGroupC TG, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCLocalRayCast(// Output
                    MRAY_GRID_CONSTANT const Span<VolumeIndex> dVolumeIndices,
                    MRAY_GRID_CONSTANT const Span<HitKeyPack> dHitIds,
                    MRAY_GRID_CONSTANT const Span<MetaHit> dHitParams,
                    // I-O
                    MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                    MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                    // Input
                    MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                    MRAY_GRID_CONSTANT const Span<const CommonKey> dAcceleratorKeys,
                    // Constant
                    MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                    MRAY_GRID_CONSTANT const typename AG::DataSoA aSoA,
                    MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA,
                    MRAY_GRID_CONSTANT const bool resolveMedia)
{
    using PG = typename AG::PrimitiveGroup;
    using Accelerator = typename AG:: template Accelerator<TG>;
    KernelCallParams kp;

    uint32_t workCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < workCount; i += kp.TotalSize())
    {
        RayIndex index = dRayIndices[i];
        auto [ray, tMM] = RayFromGMem(dRays, index);

        BackupRNG rng(dRNGStates[index]);

        // Get ids
        AcceleratorKey aId(dAcceleratorKeys[i]);
        // Construct the accelerator view
        Accelerator acc(tSoA, pSoA, aSoA, aId);

        // Do work depending on the prim transform logic
        using enum PrimTransformType;
        if constexpr(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
        {
            // Transform is local
            // we can transform the ray and use it on iterations.
            // Since this prim "supports locally constant transforms"
            // Prim key does mean nothing, so set it to invalid and call the generator
            using TransContext = typename AccTransformContextType<AG, TG>::Result;
            TransContext tContext = GenerateTransformContext(tSoA, pSoA, acc.GetTransformKey(),
                                                             PrimitiveKey::InvalidKey());
            ray = tContext.InvApply(ray);
        }

        // Actual ray cast!
        OptionalHitR<PG> hitOpt = acc.ClosestHit(rng, ray, tMM);
        if(!hitOpt) continue;

        const auto& hit = hitOpt.value();
        dHitIds[index] = HitKeyPack
        {
            .primKey = hit.primitiveKey,
            .lightOrMatKey = hit.lmKey,
            .transKey = acc.GetTransformKey(),
            .accelKey = aId
        };
        UpdateTMax(dRays, index, hit.t);
        dHitParams[index] = hit.hit;

        if(resolveMedia)
            dVolumeIndices[index] = hit.volumeIndex;
    }
};

template<AccelGroupC AG, TransformGroupC TG, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCVisibilityRayCast(// Output
                         MRAY_GRID_CONSTANT const Bitspan<uint32_t> dIsVisibleBuffer,
                         // I-O
                         MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                         // Input
                         MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                         MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                         MRAY_GRID_CONSTANT const Span<const CommonKey> dAcceleratorKeys,
                         // Constant
                         MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                         MRAY_GRID_CONSTANT const typename AG::DataSoA aSoA,
                         MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA)
{
    using PG = typename AG::PrimitiveGroup;
    using Accelerator = typename AG:: template Accelerator<TG>;
    KernelCallParams kp;

    uint32_t workCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < workCount; i += kp.TotalSize())
    {
        RayIndex index = dRayIndices[i];
        auto [ray, tMM] = RayFromGMem(dRays, index);

        BackupRNG rng(dRNGStates[index]);

        // Get ids
        AcceleratorKey aId(dAcceleratorKeys[i]);
        // Construct the accelerator view
        Accelerator acc(tSoA, pSoA, aSoA, aId);

        // Do work depending on the prim transform logic
        using enum PrimTransformType;
        if constexpr(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
        {
            // Transform is local
            // we can transform the ray and use it on iterations.
            // Since this prim "supports locally constant transforms"
            // Prim key does mean nothing, so set it to invalid and call the generator
            using TransContext = typename AccTransformContextType<AG, TG>::Result;
            TransContext tContext = GenerateTransformContext(tSoA, pSoA, acc.GetTransformKey(),
                                                             PrimitiveKey::InvalidKey());
            ray = tContext.InvApply(ray);
        }

        // Actual ray cast!
        OptionalHitR<PG> hitOpt = acc.FirstHit(rng, ray, tMM);
        if(hitOpt) dIsVisibleBuffer.SetBitParallel(index, false);
    }
};