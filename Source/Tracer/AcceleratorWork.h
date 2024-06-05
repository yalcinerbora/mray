#pragma once

#include "AcceleratorC.h"
#include "TransformC.h"
#include "PrimitiveC.h"
#include "Random.h"
#include "TracerTypes.h"

template<PrimitiveGroupC PG>
using OptionalHitR = Optional<HitResultT<typename PG::Hit>>;


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
    // This kernel only works for LOCALLY_CONSTANT_TRANSFORM typed prims
    static_assert(PG::TransformLogic == PrimTransformType::LOCALLY_CONSTANT_TRANSFORM);

    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId();
        globalId < static_cast<uint32_t>(dConcreteIndicesOfInstances.size());
        globalId += kp.TotalSize())
    {
        TransformKey tKey = dInstanceTransformKeys[globalId];
        uint32_t index = dConcreteIndicesOfInstances[globalId];
        AABB3 cAABB = dConcreteAABBs[index];
        // Acquire Transform Context Generator & Type
        using TransContext = AccTransformContextType<AG, TG>;
        TransContext tContext = GenerateTransformContext(tSoA, pSoA, tKey,
                                                         PrimitiveKey::InvalidKey());
        // Finally transform and write
        AABB3 worldAABB = tContext.InvApply(cAABB);
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
                           MRAY_GRID_CONSTANT const Span<BackupRNGState> rngStates,
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

        BackupRNG rng(rngStates[index]);

        // Get ids
        AcceleratorKey aId(dAcceleratorKeys[index]);
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
            using TransContext = AccTransformContextType<AG, TG>;
            TransContext tContext = GenerateTransformContext(tSoA, pSoA, acc.TransformKey(),
                                                             PrimitiveKey::InvalidKey());
            ray = tContext.InvApply(ray);
        }

        // Actual ray cast!
        OptionalHitR<PG> hitOpt = acc.ClosestHit(rng, ray, tMM);
        if(!hitOpt) continue;

        const auto& hit = hitOpt.value();
        dHitIds[i] = HitKeyPack
        {
            .primKey = hit.primitiveKey,
            .lightOrMatKey = hit.lmKey,
            .transKey = acc.TransformKey(),
            .accelKey = aId
        };
        UpdateTMax(dRays, index, hit.t);
        dHitParams[i] = hit.hit;
    }
};

class AcceleratorWorkI
{
    public:
    virtual         ~AcceleratorWorkI() = default;

    virtual void    CastLocalRays(// Output
                                  Span<HitKeyPack> dHitKeys,
                                  Span<MetaHit> dHitParams,
                                  // I-O
                                  Span<BackupRNGState> rngStates,
                                  Span<RayGMem> dRays,
                                  // Input
                                  Span<const RayIndex> dRayIndices,
                                  Span<const CommonKey> dAccelIdPacks,
                                  // Constants
                                  const GPUQueue& queue) const = 0;
    virtual void    TransformLocallyConstantAABBs(// Output
                                                  Span<AABB3> dInstanceAABBs,
                                                  // Input
                                                  Span<const AABB3> dConcreteAABBs,
                                                  Span<const uint32_t> dConcreteIndicesOfInstances,
                                                  Span<const TransformKey> dInstanceTransformKeys,
                                                  // Constants
                                                  const GPUQueue& queue) const = 0;
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
                       Span<BackupRNGState> rngStates,
                       Span<RayGMem> dRays,
                       // Input
                       Span<const RayIndex> dRayIndices,
                       Span<const CommonKey> dAcceleratorKeys,
                       // Constants
                       const GPUQueue& queue) const override;

    // Transformation Related
    void TransformLocallyConstantAABBs(// Output
                                       Span<AABB3> dInstanceAABBs,
                                       // Input
                                       Span<const AABB3> dConcreteAABBs,
                                       Span<const uint32_t> dConcreteIndicesOfInstances,
                                       Span<const TransformKey> dInstanceTransformKeys,
                                       // Constants
                                       const GPUQueue& queue) const override;
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
                                            Span<BackupRNGState> rngStates,
                                            Span<RayGMem> dRays,
                                            // Input
                                            Span<const RayIndex> dRayIndices,
                                            Span<const CommonKey> dAcceleratorKeys,
                                            // Constants
                                            const GPUQueue& queue) const
{
    assert(dRays.size() == dRayIndices.size());
    assert(dRayIndices.size() == dAcceleratorKeys.size());
    assert(dAcceleratorKeys.size() == dHitIds.size());

    using namespace std::string_literals;
    queue.IssueSaturatingKernel<KCLocalRayCast<AG, TG>>
    (
        "KCCastLocalRays-"s + std::string(TypeName()),
        KernelIssueParams{.workCount = static_cast<uint32_t>(dRays.size())},
        //
        dHitIds,
        dHitParams,
        rngStates,
        dRays,
        dRayIndices,
        dAcceleratorKeys,
        transGroup.SoA(),
        accelGroup.SoA(),
        primGroup.SoA()
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
        queue.IssueSaturatingKernel<KCTransformLocallyConstantAABBs<AG, TG>>
        (
            "KCTransformLocallyConstantAABBs-"s + std::string(TypeName()),
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
std::string_view AcceleratorWork<AG, TG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const std::string Name = AccelWorkTypeName(AG::TypeName(),
                                                      TG::TypeName());
    return Name;
}
