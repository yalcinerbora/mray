#pragma once

#include "AcceleratorC.h"
#include "AcceleratorWork.h"    // IWYU pragma: keep

#include <numeric>

namespace HIPRTAccelDetail
{
    // Phony type to satisfy the concept
    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroup = TransformGroupIdentity>
    class AcceleratorHIPRT
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroup::DataSoA;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = EmptyType;

        public:
        MR_PF_DECL_V
        AcceleratorOptiX(TransDataSoA, PrimDataSoA,
                         DataSoA, AcceleratorKey) {}

        MR_PF_DECL
        Optional<HitResult>
        ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MR_PF_DECL
        Optional<HitResult>
        FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MR_PF_DECL
        TransformKey
        GetTransformKey() const { return TransformKey::InvalidKey(); }
    };
}

// Add Optix specific functions
class AcceleratorGroupHIPRTI
{
    public:
    virtual ~AcceleratorGroupHIPRTI() = default;
    //
};

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupHIPRT final
    : public AcceleratorGroupT<AcceleratorGroupHIPRT<PrimitiveGroupType>, AcceleratorGroupHIPRTI>
{
    using Base = AcceleratorGroupT<AcceleratorGroupHIPRT<PrimitiveGroupType>, AcceleratorGroupHIPRTI>;
    public:
    static std::string_view TypeName();
    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = EmptyType;
    using PGSoA             = typename PrimitiveGroup::DataSoA;

    template<class TG = TransformGroupIdentity>
    using Accelerator = OptiXAccelDetail::AcceleratorOptiX<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;
    static constexpr auto IsTriangle = TrianglePrimGroupC<PrimitiveGroup>;

    private:

    public:
    // Constructors & Destructor
    AcceleratorGroupHIPRT(uint32_t accelGroupId,
                          ThreadPool&,
                          const GPUSystem&,
                          const GenericGroupPrimitiveT& pg,
                          const AccelWorkGenMap&);
    //
    void    PreConstruct(const BaseAcceleratorI*) override;
    void    Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void    WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                      Span<AcceleratorKey> dKeyWriteRegion,
                                      const GPUQueue&) const override;
    void    AcquireIASConstructionParams(Span<OptixTraversableHandle> dTraversableHandles,
                                         Span<Matrix3x4> dInstanceMatrices,
                                         Span<uint32_t> dSBTCounts,
                                         Span<uint32_t> dFlags,
                                         const GPUQueue& queue) const override;
    // Functionality
    void    CastLocalRays(// Output
                          Span<VolumeIndex> dVolumeIndices,
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> dRNGStates,
                          Span<RayGMem> dRays,
                          // Input
                          Span<const RayIndex> dRayIndices,
                          Span<const CommonKey> dAccelKeys,
                          // Constants
                          CommonKey workId,
                          bool resolveMedia,
                          const GPUQueue& queue) override;

    void    CastVisibilityRays(// Output
                               Bitspan<uint32_t> dIsVisibleBuffer,
                               // I-O
                               Span<BackupRNGState> dRNGStates,
                               // Input
                               Span<const RayGMem> dRays,
                               Span<const RayIndex> dRayIndices,
                               Span<const CommonKey> dAccelKeys,
                               // Constants
                               CommonKey workId,
                               const GPUQueue& queue) override;

    DataSoA SoA() const;
    size_t  GPUMemoryUsage() const override;
};

class BaseAcceleratorHIPRT final : public BaseAcceleratorT<BaseAcceleratorHIPRT>
{
    public:
    static std::string_view TypeName();

    private:
    protected:
    AABB3           InternalConstruct(const std::vector<size_t>& instanceOffsets) override;
    void            GenerateShaders(EmptyHitRecord& rgCommonRecord, EmptyHitRecord& rgLocalRecord,
                                    EmptyHitRecord& missRecord, std::vector<GenericHitRecord<>>&,
                                    const ShaderNameMap&);
    public:
    // Constructors & Destructor
    BaseAcceleratorHIPRT(ThreadPool&, const GPUSystem&,
                         const AccelGroupGenMap&,
                         const AccelWorkGenMap&);

    //
    void    CastRays(// Output
                     Span<VolumeIndex> dVolumeIndices,
                     Span<HitKeyPack> dHitIds,
                     Span<MetaHit> dHitParams,
                     // I-O
                     Span<BackupRNGState> dRNGStates,
                     Span<RayGMem> dRays,
                     // Input
                     Span<const RayIndex> dRayIndices,
                     bool resolveMedia,
                     const GPUQueue& queue) override;

    void    CastVisibilityRays(// Output
                               Bitspan<uint32_t> dIsVisibleBuffer,
                               // I-O
                               Span<BackupRNGState> dRNGStates,
                               // Input
                               Span<const RayGMem> dRays,
                               Span<const RayIndex> dRayIndices,
                               const GPUQueue& queue) override;

    void    CastLocalRays(// Output
                          Span<VolumeIndex> dVolumeIndices,
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> dRNGStates,
                          Span<RayGMem> dRays,
                          // Input
                          Span<const RayIndex> dRayIndices,
                          Span<const AcceleratorKey> dAccelKeys,
                          CommonKey dAccelKeyBatchPortion,
                          bool resolveMedia,
                          const GPUQueue& queue) override;

    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;
};

#include "AcceleratorHIPRT.hpp"