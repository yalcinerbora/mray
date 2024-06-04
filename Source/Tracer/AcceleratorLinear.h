#pragma once

#include <algorithm>
#include <bitset>

#include "Core/Types.h"
#include "Core/MemAlloc.h"
#include "Core/AABB.h"

#include "Device/GPUSystemForward.h"

#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "AcceleratorC.h"
#include "PrimitiveC.h"
#include "TransformC.h"
#include "Random.h"
#include "RayPartitioner.h"
#include "AcceleratorWork.h"

namespace LinearAccelDetail
{
    // SoA data of triangle group
    struct LinearAcceleratorSoA
    {
        // Per accelerator instance stuff
        Span<const CullFaceFlagArray>           dCullFace;
        Span<const AlphaMapArray>               dAlphaMaps;
        Span<const LightOrMatKeyArray>          dLightOrMatKeys;
        Span<const PrimRangeArray>              dPrimitiveRanges;
        Span<const TransformKey>                dInstanceTransforms;
        Span<const Span<const PrimitiveKey>>    dLeafs;
    };

    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroupType = TransformGroupIdentity>
    class AcceleratorLinear
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroupType::DataSoA;
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
        const LightOrMatKeyArray&   lmKeys;
        Span<const PrimitiveKey>    leafs;
        // Primitive Related
        TransformKey                transformKey;
        const TransDataSoA&         transformSoA;
        const PrimDataSoA&          primitiveSoA;

        MRAY_HYBRID
        Optional<HitResult>     IntersectionCheck(const Ray& ray,
                                                  const Vector2& tMinMax,
                                                  Float xi,
                                                  const PrimitiveKey& primKey) const;
        public:
        // Constructors & Destructor
        MRAY_HYBRID             AcceleratorLinear(const TransDataSoA& tSoA,
                                                  const PrimDataSoA& pSoA,
                                                  const DataSoA& dataSoA,
                                                  AcceleratorKey aId);
        MRAY_HYBRID
        TransformKey            TransformKey() const;
        MRAY_HYBRID
        Optional<HitResult>     ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const;
        MRAY_HYBRID
        Optional<HitResult>     FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const;
    };
}

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupLinear final : public AcceleratorGroupT<AcceleratorGroupLinear<PrimitiveGroupType>, PrimitiveGroupType>
{
    using Base = AcceleratorGroupT<AcceleratorGroupLinear<PrimitiveGroupType>, PrimitiveGroupType>;

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
    Span<CullFaceFlagArray>     dCullFaceFlags;
    Span<AlphaMapArray>         dAlphaMaps;
    Span<LightOrMatKeyArray>    dLightOrMatKeys;
    Span<TransformKey>          dTransformKeys;
    // These are duplicated since we will have only 8 prim batch per instance
    Span<PrimRangeArray>        dPrimitiveRanges;
    // These are not-duplicated, Instances have copy of the spans.
    // spans may be the same
    Span<Span<PrimitiveKey>>    dLeafs;
    // Global data, all accelerator leafs are in here
    Span<PrimitiveKey>          dAllLeafs;

    std::vector<Vector2ui>      hWorkInstanceRanges;

    public:
    // Constructors & Destructor
                AcceleratorGroupLinear(uint32_t accelGroupId,
                                       BS::thread_pool&, GPUSystem&,
                                       const GenericGroupPrimitiveT& pg,
                                       const AccelWorkGenMap&);
    //
    void        Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void        WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                          Span<AcceleratorKey> dKeyWriteRegion,
                                          const GPUQueue&) const override;

    // Functionality
    void        CastLocalRays(// Output
                              Span<HitKeyPack> dHitIds,
                              Span<MetaHit> dHitParams,
                              // I-O
                              Span<BackupRNGState> rngStates,
                              Span<RayGMem> dRays,
                              // Input
                              Span<const RayIndex> dRayIndices,
                              Span<const CommonKey> dAccelKeys,
                              // Constants
                              uint32_t workId,
                              const GPUQueue& queue) override;

    DataSoA     SoA() const;
    size_t      GPUMemoryUsage() const override;
};

class BaseAcceleratorLinear final : public BaseAcceleratorT<BaseAcceleratorLinear>
{
    public:
    static std::string_view     TypeName();

    private:
    DeviceMemory                accelMem;
    Span<AcceleratorKey>        dLeafs;
    Span<AABB3>                 dAABBs;
    //
    DeviceMemory                stackMem;
    Span<uint32_t>              dTraversalStack;
    RayPartitioner              rayPartitioner;
    size_t                      maxPartitionCount;

    protected:
    // Each leaf has a surface
    AABB3 InternalConstruct(const std::vector<size_t>& instanceOffsets) override;

    public:
    // Constructors & Destructor
    BaseAcceleratorLinear(BS::thread_pool&, GPUSystem&,
                          const AccelGroupGenMap&,
                          const AccelWorkGenMap&);

    //
    void    CastRays(// Output
                     Span<HitKeyPack> dHitIds,
                     Span<MetaHit> dHitParams,
                     // I-O
                     Span<BackupRNGState> rngStates,
                     Span<RayGMem> dRays,
                     // Input
                     Span<const RayIndex> dRayIndices) override;

    void    CastShadowRays(// Output
                           Bitspan<uint32_t> dIsVisibleBuffer,
                           Bitspan<uint32_t> dFoundMediumInterface,
                           // I-O
                           Span<BackupRNGState> rngStates,
                           // Input
                           Span<const RayIndex> dRayIndices,
                           Span<const RayGMem> dShadowRays) override;

    void    CastLocalRays(// Output
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> rngStates,
                          // Input
                          Span<const RayGMem> dRays,
                          Span<const RayIndex> dRayIndices,
                          Span<const AcceleratorKey> dAccelIdPacks) override;

    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;
};

inline
BaseAcceleratorLinear::BaseAcceleratorLinear(BS::thread_pool& tp, GPUSystem& sys,
                                             const AccelGroupGenMap& aGen,
                                             const AccelWorkGenMap& globalWorkMap)
    : BaseAcceleratorT<BaseAcceleratorLinear>(tp, sys, aGen, globalWorkMap)
    , accelMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , stackMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , rayPartitioner(sys)
    , maxPartitionCount(0)
{}

#include "AcceleratorLinear.hpp"