#pragma once

#include "Core/Types.h"
#include "Core/MemAlloc.h"


#include "TracerTypes.h"
#include "AcceleratorC.h"
#include "PrimitiveC.h"
#include "TransformC.h"
#include "Random.h"
#include "RayPartitioner.h"

namespace LinearAccelDetail
{
    // SoA data of triangle group
    struct LinearAcceleratorSoA
    {
        // Per accelerator instance stuff
        Span<const VolumeIndexArray>         dVolumeIndices;
        Span<const CullFaceFlagArray>        dCullFace;
        Span<const AlphaMapArray>            dAlphaMaps;
        Span<const LightOrMatKeyArray>       dLightOrMatKeys;
        Span<const PrimRangeArray>           dPrimitiveRanges;
        Span<const TransformKey>             dInstanceTransforms;
        Span<const Span<const PrimitiveKey>> dLeafs;
    };

    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroup = TransformGroupIdentity>
    class AcceleratorLinear
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroup::DataSoA;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = LinearAcceleratorSoA;

        private:
        // Accel Related
        // TODO: Check if expansion reduces performance
        //
        // Specifically load the range array by value,
        // we will do linear search over it
        // Maybe compiler put this on register space
        PrimRangeArray              primRanges;
        // Rest is by reference (except the tKey & cullFace, these are a single word)
        CullFaceFlagArray           cullFaceFlags;
        const AlphaMapArray&        alphaMaps;
        const VolumeIndexArray&     volumeIndices;
        const LightOrMatKeyArray&   lmKeys;
        Span<const PrimitiveKey>    leafs;
        // Primitive Related
        TransformKey                transformKey;
        const TransDataSoA&         transformSoA;
        const PrimDataSoA&          primitiveSoA;

        template<auto = MRAY_PRIM_TGEN_FUNCTION(PrimGroup, TransGroup)>
        MR_GF_DECL
        Optional<HitResult>     IntersectionCheck(const Ray& ray,
                                                  const Vector2& tMinMax,
                                                  Float xi,
                                                  const PrimitiveKey& primKey) const;
        public:
        // Constructors & Destructor
        MR_GF_DECL              AcceleratorLinear(const TransDataSoA& tSoA,
                                                  const PrimDataSoA& pSoA,
                                                  const DataSoA& dataSoA,
                                                  AcceleratorKey aId);
        MR_GF_DECL
        TransformKey            GetTransformKey() const;
        MR_GF_DECL
        Optional<HitResult>     ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const;
        MR_GF_DECL
        Optional<HitResult>     FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const;
    };
}

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupLinear final
    : public AcceleratorGroupT<AcceleratorGroupLinear<PrimitiveGroupType>>
{
    using Base = AcceleratorGroupT<AcceleratorGroupLinear<PrimitiveGroupType>>;

    public:
    static std::string_view TypeName();

    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = LinearAccelDetail::LinearAcceleratorSoA;

    template<class TG = TransformGroupIdentity>
    using Accelerator = LinearAccelDetail::AcceleratorLinear<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;

    private:
    DeviceMemory                mem;
    DataSoA                     data;
    // Per-instance (All accelerators will have these)
    Span<VolumeIndexArray>      dInstanceVolumeIndices;
    Span<CullFaceFlagArray>     dCullFaceFlags;
    Span<AlphaMapArray>         dAlphaMaps;
    Span<LightOrMatKeyArray>    dLightOrMatKeys;
    Span<TransformKey>          dTransformKeys;
    // These are duplicated since we will have only 8 prim batch per instance
    Span<PrimRangeArray>        dPrimitiveRanges;
    // These are not-duplicated, Instances have copy of the spans.
    // spans may be the same
    Span<Span<const PrimitiveKey>>  dLeafs;
    // Global data, all accelerator leafs are in here
    Span<PrimitiveKey>              dAllLeafs;

    public:
    // Constructors & Destructor
            AcceleratorGroupLinear(uint32_t accelGroupId,
                                   ThreadPool&,
                                   const GPUSystem&,
                                   const GenericGroupPrimitiveT& pg,
                                   const AccelWorkGenMap&);
    //
    void    Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void    WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                      Span<AcceleratorKey> dKeyWriteRegion,
                                      const GPUQueue&) const override;

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

class BaseAcceleratorLinear final : public BaseAcceleratorT<BaseAcceleratorLinear>
{
    public:
    static std::string_view TypeName();

    private:
    DeviceMemory            accelMem;
    Span<AcceleratorKey>    dLeafs;
    Span<AABB3>             dAABBs;
    //
    DeviceMemory            stackMem;
    Span<uint32_t>          dTraversalStack;
    RayPartitioner          rayPartitioner;
    size_t                  maxPartitionCount;

    protected:
    AABB3 InternalConstruct(const std::vector<size_t>& instanceOffsets) override;

    public:
    // Constructors & Destructor
    BaseAcceleratorLinear(ThreadPool&, const GPUSystem&,
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
                     //
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
                          //
                          CommonKey dAccelKeyBatchPortion,
                          bool resolveMedia,
                          const GPUQueue& queue) override;

    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;
};

inline
BaseAcceleratorLinear::BaseAcceleratorLinear(ThreadPool& tp, const GPUSystem& sys,
                                             const AccelGroupGenMap& aGen,
                                             const AccelWorkGenMap& globalWorkMap)
    : BaseAcceleratorT<BaseAcceleratorLinear>(tp, sys, aGen, globalWorkMap)
    , accelMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , stackMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , rayPartitioner(sys)
    , maxPartitionCount(0)
{}

#include "AcceleratorLinear.hpp"