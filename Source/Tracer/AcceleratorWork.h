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
                      MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA);

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
                         MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA);

template<TransformGroupC TG = TransformGroupIdentity>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGetCommonTransforms(// Output
                           MRAY_GRID_CONSTANT const Span<Matrix4x4> dTransforms,
                           // Inputs
                           MRAY_GRID_CONSTANT const Span<const TransformKey> dTransformKeys,
                           MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA);

template<AccelGroupC AG, TransformGroupC TG,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCTransformLocallyConstantAABBs(// Output
                                     MRAY_GRID_CONSTANT const Span<AABB3> dInstanceAABBs,
                                     // Input
                                     MRAY_GRID_CONSTANT const Span<const AABB3> dConcreteAABBs,
                                     MRAY_GRID_CONSTANT const Span<const uint32_t> dConcreteIndicesOfInstances,
                                     MRAY_GRID_CONSTANT const Span<const TransformKey> dInstanceTransformKeys,
                                     // Constants
                                     MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                                     MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA);

template<AccelGroupC AG, TransformGroupC TG,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCLocalRayCast(// Output
                    MRAY_GRID_CONSTANT const Span<InterfaceIndex> dInterfaceIndices,
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
                    MRAY_GRID_CONSTANT const bool writeInterfaceIndex);

template<AccelGroupC AG, TransformGroupC TG,
         auto GenerateTransformContext = MRAY_ACCEL_TGEN_FUNCTION(AG, TG)>
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
                         MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA);

template<class T>
concept AccelWorkBaseC = std::derived_from<T, AcceleratorWorkI>;

template<AccelGroupC AcceleratorGroupType, TransformGroupC TransformGroupType,
         AccelWorkBaseC BaseType = AcceleratorWorkI>
class AcceleratorWork : public BaseType
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
                       Span<InterfaceIndex> dInterfaceIndices,
                       Span<HitKeyPack> dHitKeys,
                       Span<MetaHit> dHitParams,
                       // I-O
                       Span<BackupRNGState> dRNGStates,
                       Span<RayGMem> dRays,
                       // Input
                       Span<const RayIndex> dRayIndices,
                       Span<const CommonKey> dAcceleratorKeys,
                       // Constants
                       bool writeInterfaceIndex,
                       const GPUQueue& queue) const override;

    void CastVisibilityRays(// Output
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

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
AcceleratorWork<AG, TG, BT>::AcceleratorWork(const AcceleratorGroupI& ag,
                                             const GenericGroupTransformT& tg)
    : primGroup(static_cast<const PrimitiveGroup&>(ag.PrimGroup()))
    , accelGroup(static_cast<const AG&>(ag))
    , transGroup(static_cast<const TG&>(tg))
{}

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::CastLocalRays(// Output
                                                Span<InterfaceIndex> dInterfaceIndices,
                                                Span<HitKeyPack> dHitIds,
                                                Span<MetaHit> dHitParams,
                                                // I-O
                                                Span<BackupRNGState> dRNGStates,
                                                Span<RayGMem> dRays,
                                                // Input
                                                Span<const RayIndex> dRayIndices,
                                                Span<const CommonKey> dAcceleratorKeys,
                                                // Constants
                                                bool writeInterfaceIndex,
                                                const GPUQueue& queue) const
{
    assert(dHitIds.size() == dHitParams.size());
    assert(dHitParams.size() == dRNGStates.size());
    assert(dRNGStates.size() == dRays.size());
    //
    assert(dRayIndices.size() == dAcceleratorKeys.size());

    using namespace std::string_literals;
    queue.IssueWorkKernel<KCLocalRayCast<AG, TG>>
    (
        "KCCastLocalRays-"s + std::string(TypeName()),
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dRayIndices.size())},
        //
        dInterfaceIndices,
        dHitIds,
        dHitParams,
        dRNGStates,
        dRays,
        dRayIndices,
        dAcceleratorKeys,
        transGroup.SoA(),
        accelGroup.SoA(),
        primGroup.SoA(),
        writeInterfaceIndex
    );
}

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::CastVisibilityRays(// Output
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
    queue.IssueWorkKernel<KCVisibilityRayCast<AG, TG>>
    (
        "KCCastVisibilityRays-"s + std::string(TypeName()),
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dRayIndices.size())},
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

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::GeneratePrimitiveCenters(Span<Vector3> dAllPrimCenters,
                                                           Span<const uint32_t> dLeafSegmentRanges,
                                                           Span<const PrimitiveKey> dAllLeafs,
                                                           Span<const TransformKey> dTransformKeys,
                                                           const GPUQueue& queue) const
{
    static constexpr uint32_t TPB = StaticThreadPerBlock1D();
    static constexpr uint32_t BLOCK_PER_INSTANCE = 16;
    uint32_t processedAccelCount = static_cast<uint32_t>(dLeafSegmentRanges.size() - 1);
    uint32_t blockCount = processedAccelCount * BLOCK_PER_INSTANCE;
    queue.IssueBlockKernel<KCGenPrimCenters<AG, TG>>
    (
        "KCGenPrimCenters",
        DeviceBlockIssueParams
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

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::GeneratePrimitiveAABBs(Span<AABB3> dAllLeafAABBs,
                                                         Span<const uint32_t> dLeafSegmentRanges,
                                                         Span<const PrimitiveKey> dAllLeafs,
                                                         Span<const TransformKey> dTransformKeys,
                                                         const GPUQueue& queue) const
{
    static constexpr uint32_t TPB = StaticThreadPerBlock1D();
    static constexpr uint32_t BLOCK_PER_INSTANCE = 16;
    uint32_t processedAccelCount = static_cast<uint32_t>(dLeafSegmentRanges.size() - 1);
    uint32_t blockCount = processedAccelCount * BLOCK_PER_INSTANCE;
    queue.IssueBlockKernel<KCGeneratePrimAABBs<AG, TG>>
    (
        "KCGeneratePrimAABBs",
        DeviceBlockIssueParams
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

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::GetCommonTransforms(Span<Matrix4x4> dTransforms,
                                                      Span<const TransformKey> dTransformKeys,
                                                      const GPUQueue& queue) const
{
    static const std::string KernelName = "KCGetCommonTransforms-" + std::string(TG::TypeName());
    uint32_t transformCount = static_cast<uint32_t>(dTransformKeys.size());
    queue.IssueWorkKernel<KCGetCommonTransforms<TG>>
    (
        KernelName,
        DeviceWorkIssueParams { .workCount = transformCount },
        // Output
        dTransforms,
        // Inputs
        dTransformKeys,
        transGroup.SoA()
    );
}

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::TransformLocallyConstantAABBs(// Output
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
        queue.IssueWorkKernel<KCTransformLocallyConstantAABBs<AG, TG>>
        (
            KernelName,
            DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dConcreteIndicesOfInstances.size())},
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

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
size_t AcceleratorWork<AG, TG, BT>::TransformSoAByteSize() const
{
    return sizeof(typename TG::DataSoA);
}

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
void AcceleratorWork<AG, TG, BT>::CopyTransformSoA(Span<Byte> dRegion, const GPUQueue& queue) const
{
    // TODO: Find a way to remove the barrier later
    typename TG::DataSoA tgSoA = transGroup.SoA();
    Span<const Byte> hSpan(reinterpret_cast<Byte*>(&tgSoA),
                           sizeof(typename TG::DataSoA));
    queue.MemcpyAsync(dRegion, hSpan);
    queue.Barrier().Wait();
}

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
std::string_view AcceleratorWork<AG, TG, BT>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const std::string Name = AccelWorkTypeName(AG::TypeName(),
                                                      TG::TypeName());
    return Name;
}

template<AccelGroupC AG, TransformGroupC TG, AccelWorkBaseC BT>
std::string_view AcceleratorWork<AG, TG, BT>::TransformName() const
{
    return transGroup.Name();
}
