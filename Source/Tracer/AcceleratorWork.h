#pragma once

#include "AcceleratorC.h"
#include "TransformC.h"
#include "PrimitiveC.h"
#include "Random.h"
#include "TracerTypes.h"

template <class BaseAccel, class AccelGTypes, class AccelWorkTypes>
struct AccelTypePack
{
    using BaseType      = BaseAccel;
    using GroupTypes    = AccelGTypes;
    using WorkTypes     = AccelWorkTypes;
};

template<PrimitiveGroupC PG>
using OptionalHitR = Optional<HitResultT<typename PG::Hit>>;

template<AccelGroupC AG, TransformGroupC TG = TransformGroupIdentity,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
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

template<AccelGroupC AG, TransformGroupC TG = TransformGroupIdentity,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
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

template<TransformGroupC TG = TransformGroupIdentity>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGetCommonTransforms(// Output
                           MRAY_GRID_CONSTANT const Span<Matrix4x4> dTransforms,
                           // Inputs
                           MRAY_GRID_CONSTANT const Span<const TransformKey> dTransformKeys,
                           MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA)
{
    assert(dTransformKeys.size() == dTransforms.size());
    uint32_t tCount = static_cast<uint32_t>(dTransformKeys.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < tCount; i += kp.TotalSize())
    {
        Matrix4x4 transform = TG::AcquireCommonTransform(tSoA, dTransformKeys[i]);
        dTransforms[i] = transform;
    }
}

template<AccelGroupC AG, TransformGroupC TG,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCTransformLocallyConstantAABBs(// Output
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

template<AccelGroupC AG, TransformGroupC TG,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCLocalRayCast(// Output
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

        // Do work depending on the prim transorm logic
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
    }
};

template<AccelGroupC AG, TransformGroupC TG,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCVisibilityRayCast(// Output
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

        // Do work depending on the prim transorm logic
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

template<AccelGroupC AcceleratorGroupType,
         TransformGroupC TransformGroupType>
class AcceleratorWork : public AcceleratorWorkI
{
    public:
    using TransformGroup    = TransformGroupType;
    using AcceleratorGroup  = AcceleratorGroupType;
    using PrimitiveGroup    = typename AcceleratorGroup::PrimitiveGroup;

    static std::string_view TypeName();

    private:
    const PrimitiveGroup&   primGroup;
    const AcceleratorGroup& accelGroup;
    const TransformGroup&   transGroup;

    public:
    AcceleratorWork(const AcceleratorGroupI& ag,
                    const GenericGroupTransformT& tg);

    // Cast Local rays
    void CastLocalRays(// Output
                       Span<HitKeyPack> dHitKeys,
                       Span<MetaHit> dHitParams,
                       // I-O
                       Span<BackupRNGState> dRNGStates,
                       Span<RayGMem> dRays,
                       // Input
                       Span<const RayIndex> dRayIndices,
                       Span<const CommonKey> dAcceleratorKeys,
                       // Constants
                       const GPUQueue& queue) const override;

    void    CastVisibilityRays(// Output
                               Bitspan<uint32_t> dIsVisibleBuffer,
                               // I-O
                               Span<BackupRNGState> dRNGStates,
                               // Input
                               Span<const RayGMem> dRays,
                               Span<const RayIndex> dRayIndices,
                               Span<const CommonKey> dAcceleratorKeys,
                               // Constants
                               const GPUQueue& queue) const override;

    void GeneratePrimitiveCenters(Span<Vector3> dAllPrimCenters,
                                  Span<const uint32_t> dLeafSegmentRanges,
                                  Span<const PrimitiveKey> dAllLeafs,
                                  Span<const TransformKey> dTransformKeys,
                                  const GPUQueue& queue) const override;
    void GeneratePrimitiveAABBs(Span<AABB3> dAllLeafAABBs,
                                Span<const uint32_t> dLeafSegmentRanges,
                                Span<const PrimitiveKey> dAllLeafs,
                                Span<const TransformKey> dTransformKeys,
                                const GPUQueue& queue) const override;

    // Transformation Related
    void GetCommonTransforms(Span<Matrix4x4> dTransforms,
                             Span<const TransformKey> dTransformKeys,
                             const GPUQueue& queue) const override;
    void TransformLocallyConstantAABBs(// Output
                                       Span<AABB3> dInstanceAABBs,
                                       // Input
                                       Span<const AABB3> dConcreteAABBs,
                                       Span<const uint32_t> dConcreteIndicesOfInstances,
                                       Span<const TransformKey> dInstanceTransformKeys,
                                       // Constants
                                       const GPUQueue& queue) const override;

    size_t              TransformSoAByteSize() const override;
    void                CopyTransformSoA(Span<Byte>,
                                         const GPUQueue& queue) const override;
    std::string_view    TransformName() const override;
};

template<AccelGroupC AG, TransformGroupC TG>
AcceleratorWork<AG, TG>::AcceleratorWork(const AcceleratorGroupI& ag,
                                         const GenericGroupTransformT& tg)
    : accelGroup(static_cast<const AG&>(ag))
    , transGroup(static_cast<const TG&>(tg))
    , primGroup(static_cast<const PrimitiveGroup&>(ag.PrimGroup()))
{}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::CastLocalRays(// Output
                                            Span<HitKeyPack> dHitIds,
                                            Span<MetaHit> dHitParams,
                                            // I-O
                                            Span<BackupRNGState> dRNGStates,
                                            Span<RayGMem> dRays,
                                            // Input
                                            Span<const RayIndex> dRayIndices,
                                            Span<const CommonKey> dAcceleratorKeys,
                                            // Constants
                                            const GPUQueue& queue) const
{
    assert(dHitIds.size() == dHitParams.size());
    assert(dHitParams.size() == dRNGStates.size());
    assert(dRNGStates.size() == dRays.size());
    //
    assert(dRayIndices.size() == dAcceleratorKeys.size());

    using namespace std::string_literals;
    queue.IssueSaturatingKernel<KCLocalRayCast<AG, TG>>
    (
        "KCCastLocalRays-"s + std::string(TypeName()),
        KernelIssueParams{.workCount = static_cast<uint32_t>(dRayIndices.size())},
        //
        dHitIds,
        dHitParams,
        dRNGStates,
        dRays,
        dRayIndices,
        dAcceleratorKeys,
        transGroup.SoA(),
        accelGroup.SoA(),
        primGroup.SoA()
    );
}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::CastVisibilityRays(// Output
                                                 Bitspan<uint32_t> dIsVisibleBuffer,
                                                 // I-O
                                                 Span<BackupRNGState> dRNGStates,
                                                 // Input
                                                 Span<const RayGMem> dRays,
                                                 Span<const RayIndex> dRayIndices,
                                                 Span<const CommonKey> dAcceleratorKeys,
                                                 // Constants
                                                 const GPUQueue& queue) const
{
    assert(dIsVisibleBuffer.Size() == dRNGStates.size());
    assert(dRNGStates.size() == dRays.size());
    //
    assert(dRayIndices.size() == dAcceleratorKeys.size());

    using namespace std::string_literals;
    queue.IssueSaturatingKernel<KCVisibilityRayCast<AG, TG>>
    (
        "KCCastVisibilityRays-"s + std::string(TypeName()),
        KernelIssueParams{.workCount = static_cast<uint32_t>(dRayIndices.size())},
        //
        dIsVisibleBuffer,
        dRNGStates,
        dRays,
        dRayIndices,
        dAcceleratorKeys,
        transGroup.SoA(),
        accelGroup.SoA(),
        primGroup.SoA()
    );
}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::GeneratePrimitiveCenters(Span<Vector3> dAllPrimCenters,
                                                       Span<const uint32_t> dLeafSegmentRanges,
                                                       Span<const PrimitiveKey> dAllLeafs,
                                                       Span<const TransformKey> dTransformKeys,
                                                       const GPUQueue& queue) const
{
    static constexpr uint32_t TPB = StaticThreadPerBlock1D();
    static constexpr uint32_t BLOCK_PER_INSTANCE = 16;

    static constexpr auto KernelName = KCGenPrimCenters<AG, TG>;
    uint32_t processedAccelCount = static_cast<uint32_t>(dLeafSegmentRanges.size() - 1);
    uint32_t blockCount = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(&KernelName),
        TPB, 0
    );
    queue.IssueExactKernel<KernelName>
    (
        "KCGenPrimCenters",
        KernelExactIssueParams
        {
            .gridSize = blockCount,
            .blockSize = TPB
        },
        // Output
        dAllPrimCenters,
        // Inputs
        dLeafSegmentRanges,
        dTransformKeys,
        dAllLeafs,
        BLOCK_PER_INSTANCE,
        processedAccelCount,
        transGroup.SoA(),
        primGroup.SoA()
    );
}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::GeneratePrimitiveAABBs(Span<AABB3> dAllLeafAABBs,
                                                     Span<const uint32_t> dLeafSegmentRanges,
                                                     Span<const PrimitiveKey> dAllLeafs,
                                                     Span<const TransformKey> dTransformKeys,
                                                     const GPUQueue& queue) const
{
    static constexpr uint32_t TPB = StaticThreadPerBlock1D();
    static constexpr uint32_t BLOCK_PER_INSTANCE = 16;

    static constexpr auto KernelName = KCGeneratePrimAABBs<AG, TG>;
    uint32_t processedAccelCount = static_cast<uint32_t>(dLeafSegmentRanges.size() - 1);
    uint32_t blockCount = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(&KernelName),
        TPB, 0
    );
    queue.IssueExactKernel<KernelName>
    (
        "KCGeneratePrimAABBs",
        KernelExactIssueParams
        {
            .gridSize = blockCount,
            .blockSize = TPB
        },
        // Output
        dAllLeafAABBs,
        // Inputs
        dLeafSegmentRanges,
        dTransformKeys,
        dAllLeafs,
        // Constants
        BLOCK_PER_INSTANCE,
        processedAccelCount,
        transGroup.SoA(),
        primGroup.SoA()
    );
}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::GetCommonTransforms(Span<Matrix4x4> dTransforms,
                                                  Span<const TransformKey> dTransformKeys,
                                                  const GPUQueue& queue) const
{
    static const std::string KernelName = "KCGetCommonTransforms-" + std::string(TG::TypeName());
    uint32_t transformCount = static_cast<uint32_t>(dTransformKeys.size());
    queue.IssueSaturatingKernel<KCGetCommonTransforms<TG>>
    (
        KernelName,
        KernelIssueParams { .workCount = transformCount },
        // Output
        dTransforms,
        // Inputs
        dTransformKeys,
        transGroup.SoA()
    );
}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::TransformLocallyConstantAABBs(// Output
                                                            Span<AABB3> dInstanceAABBs,
                                                            // Input
                                                            Span<const AABB3> dConcreteAABBs,
                                                            Span<const uint32_t> dConcreteIndicesOfInstances,
                                                            Span<const TransformKey> dInstanceTransformKeys,
                                                            // Constants
                                                            const GPUQueue& queue) const
{
    using PG = typename AG::PrimitiveGroup;
    if constexpr(PG::TransformLogic ==
                 PrimTransformType::LOCALLY_CONSTANT_TRANSFORM)
    {
        assert(dConcreteIndicesOfInstances.size() == dInstanceTransformKeys.size());

        using namespace std::string_literals;
        static const auto KernelName = "KCTransformLocallyConstantAABBs-"s + std::string(TypeName());
        queue.IssueSaturatingKernel<KCTransformLocallyConstantAABBs<AG, TG>>
        (
            KernelName,
            KernelIssueParams{.workCount = static_cast<uint32_t>(dConcreteIndicesOfInstances.size())},
            //
            dInstanceAABBs,
            dConcreteAABBs,
            dConcreteIndicesOfInstances,
            dInstanceTransformKeys,
            transGroup.SoA(),
            primGroup.SoA()
        );
    }
    else
    {
        throw MRayError("{:s}: This primitive does not support \"LOCALLY_CONSTANT_TRANSFORM\" "
                        "but \"TransformLocallyConstantAABBs\" is called", AG::TypeName());
    }
}

template<AccelGroupC AG, TransformGroupC TG>
size_t AcceleratorWork<AG, TG>::TransformSoAByteSize() const
{
    return sizeof(typename TG::DataSoA);
}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::CopyTransformSoA(Span<Byte> dRegion, const GPUQueue& queue) const
{
    // TODO: Find a way to remove the barrier later
    typename TG::DataSoA tgSoA = transGroup.SoA();
    Span<const Byte> hSpan(reinterpret_cast<Byte*>(&tgSoA),
                           sizeof(typename TG::DataSoA));
    queue.MemcpyAsync(dRegion, hSpan);
    queue.Barrier().Wait();
}

template<AccelGroupC AG, TransformGroupC TG>
std::string_view AcceleratorWork<AG, TG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const std::string Name = AccelWorkTypeName(AG::TypeName(),
                                                      TG::TypeName());
    return Name;
}

template<AccelGroupC AG, TransformGroupC TG>
std::string_view AcceleratorWork<AG, TG>::TransformName() const
{
    return transGroup.Name();
}
