#pragma once

#include "AcceleratorC.h"
#include "AcceleratorWork.h"

namespace OptiXAccelDetail
{
    // Phony type to satisfy the concept
    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroup = TransformGroupIdentity>
    class AcceleratorEmbree
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroup::DataSoA;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = EmptyType;

        public:
        MRAY_GPU MRAY_GPU_INLINE
        AcceleratorEmbree(TransDataSoA, PrimDataSoA,
                         DataSoA, AcceleratorKey) {}

        MRAY_GPU MRAY_GPU_INLINE
        Optional<HitResult>
        ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MRAY_GPU MRAY_GPU_INLINE
        Optional<HitResult>
        FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MRAY_GPU MRAY_GPU_INLINE
        TransformKey
        GetTransformKey() const { return TransformKey::InvalidKey(); }
    };
}

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupEmbree final
    : public AcceleratorGroupT<AcceleratorGroupEmbree<PrimitiveGroupType>, PrimitiveGroupType, AcceleratorGroupI>
{
    using Base = AcceleratorGroupT<AcceleratorGroupEmbree<PrimitiveGroupType>,
                                   PrimitiveGroupType, AcceleratorGroupI>;
    public:
    static std::string_view TypeName();
    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = EmptyType;
    using PGSoA             = typename PrimitiveGroup::DataSoA;

    template<class TG = TransformGroupIdentity>
    using Accelerator = OptiXAccelDetail::AcceleratorEmbree<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;
    static constexpr auto IsTriangle = TrianglePrimGroupC<PrimitiveGroup>;

    private:

    public:
    // Constructors & Destructor
    AcceleratorGroupEmbree(uint32_t accelGroupId,
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
    // Functionality
    void    CastLocalRays(// Output
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

class BaseAcceleratorEmbree final : public BaseAcceleratorT<BaseAcceleratorEmbree>
{
    public:
    static std::string_view TypeName();

    private:

    protected:
    AABB3   InternalConstruct(const std::vector<size_t>& instanceOffsets) override;
    public:
    // Constructors & Destructor
            BaseAcceleratorEmbree(ThreadPool&, const GPUSystem&,
                                  const AccelGroupGenMap&,
                                  const AccelWorkGenMap&);

    //
    void    CastRays(// Output
                     Span<HitKeyPack> dHitIds,
                     Span<MetaHit> dHitParams,
                     // I-O
                     Span<BackupRNGState> dRNGStates,
                     Span<RayGMem> dRays,
                     // Input
                     Span<const RayIndex> dRayIndices,
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
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> dRNGStates,
                          Span<RayGMem> dRays,
                          // Input
                          Span<const RayIndex> dRayIndices,
                          Span<const AcceleratorKey> dAccelKeys,
                          CommonKey dAccelKeyBatchPortion,
                          const GPUQueue& queue) override;

    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;
};

#include "AcceleratorEmbree.hpp"