#pragma once

#include <algorithm>

#include "Core/Types.h"

#include "Device/GPUSystemForward.h"

#include "TracerTypes.h"
#include "ParamVaryingData.h"
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
        Bitspan<const uint32_t>             dCullFace;
        Span<Optional<AlphaMap>>            dAlphaMaps;
        Span<Span<const AcceleratorLeaf>>   dLeafs;
    };

    template<PrimitiveGroupC PrimGroup, TransformGroupC TransGroupType>
    class AcceleratorLinear
    {
        using PrimHit       = typename PrimGroup::Hit;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = LinearAcceleratorSoA;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroupType::DataSoA;

        private:
        // Accel Related
        bool                        cullFace;
        Span<const AcceleratorLeaf> leafs;
        const Optional<AlphaMap>&   alphaMap;
        // Primitive Related
        TransformKey                transformKey;
        const TransDataSoA&         transformSoA;
        const PrimDataSoA&          primitiveSoA;

        MRAY_HYBRID
        Optional<HitResult>     IntersectionCheck(const Ray& ray,
                                                  const Vector2& tMinMax,
                                                  Float xi,
                                                  const AcceleratorLeaf& l) const;
        public:
        // Constructors & Destructor
        MRAY_HYBRID             AcceleratorLinear(const TransDataSoA& tSoA,
                                                  const PrimDataSoA& pSoA,
                                                  const DataSoA& dataSoA,
                                                  AcceleratorKey aId,
                                                  TransformKey tId);

        MRAY_HYBRID
        Optional<HitResult>     ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const;
        MRAY_HYBRID
        Optional<HitResult>     FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const;
    };
}

using PrimRangeArray     = std::array<Vector2ui, TracerConstants::MaxPrimBatchPerSurface>;
using MaterialKeyArray   = std::array<MaterialKey, TracerConstants::MaxPrimBatchPerSurface>;
using CullFaceFlagArray  = std::array<bool, TracerConstants::MaxPrimBatchPerSurface>;
using AlphaMapArray      = std::array<Optional<AlphaMap>, TracerConstants::MaxPrimBatchPerSurface>;

MRAY_HYBRID MRAY_CGPU_INLINE
MaterialKey FindMaterialId(const PrimRangeArray& primRanges,
                           const MaterialKeyArray& matKeys,
                           PrimitiveKey k)
{
    static_assert(std::tuple_size_v<PrimRangeArray> ==
                  std::tuple_size_v<MaterialKeyArray>);
    static constexpr uint32_t N = static_cast<uint32_t>(std::tuple_size_v<MaterialKeyArray>);

    // Linear search over the index
    // List has few elements so linear search should suffice
    CommonKey primIndex = k.FetchIndexPortion();
    UNROLL_LOOP
    for(uint32_t i = 0; i < N; i++)
    {
        // Do not do early break here (not every accelerator will use all 8
        // slots, it may break unrolling. Unused element ranges should be int_max
        bool inRange = (primIndex >= primRanges[i][0] &&
                        primIndex < primRanges[i][1]);
        if(inRange) return matKeys[i];
    }
    return MaterialKey::InvalidKey();
}


template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupLinear final : public AcceleratorGroupI
{
    public:
    static std::string_view TypeName();

    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = LinearAccelDetail::LinearAcceleratorSoA;
    using PrimSoA           = PrimitiveGroupType::DataSoAConst;

    template<class TG>
    using Accelerator = LinearAccelDetail::AcceleratorLinear<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;

    private:
    const PrimitiveGroup&           pg;
    DeviceMemory                    mem;
    DataSoA                         data;

    //std::map<TransGroupId, auto>     ClosestHitKernels;
    //std::map<TransGroupId, auto>     AnyHitKernels;

    //std::map<>;

    // Per-instance (All accelerators will have these)
    Bitspan<uint32_t>           dCullFaceFlags;
    Span<AlphaMapArray>         dAlphaMaps;
    Span<MaterialKeyArray>      dMaterialKeys;
    Span<PrimRangeArray>        dPrimitiveRanges;
    Span<TransformKey>          dTransformKeys;
    Span<Span<PrimitiveKey>>    dLeafs;
    // Global data, all accelerator leafs in
    Span<PrimitiveKey>          dAllLeafs;

    // We do not have an accelerator structure
    // Internal concrete accel counter
    uint32_t accelGroupId       = 0;
    uint32_t concreteAccelCount = 0;
    uint32_t instanceCount      = 0;
    uint32_t instanceTypeCount  = 0;

    protected:
    void        DetermineConcereteAccelCount(const AccelGroupConstructParams&);

    public:
    // Constructors & Destructor
                AcceleratorGroupLinear(uint32_t accelGroupId,
                                       const GenericGroupPrimitiveT& pg);
    //
    void        Construct(AccelGroupConstructParams) override;

    size_t      InstanceCount() const override;
    uint32_t    InstanceTypeCount() const override;
    uint32_t    UsedIdBitsInKey() const override;
    void        WriteInstanceKeysAndAABBs(Span<AABB3> aabbWriteRegion,
                                          Span<AcceleratorKey> keyWriteRegion) const override;
    uint32_t    SetKeyOffset(uint32_t) override;

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
    Span<uint64_t>              dTraversalStack;
    RayPartitioner              rayPartitioner;
    size_t                      maxPartitionCount;

    protected:
    // Each leaf has a surface
    void InternalConstruct(const std::vector<size_t>& instanceOffsets) override;

    public:
    // Constructors & Destructor
    BaseAcceleratorLinear(BS::thread_pool&, GPUSystem&,
                          std::map<std::string_view, AccelGroupGenerator>&& aGen);

    //
    void    CastRays(// Output
                     Span<HitKeyPack> dHitIds,
                     Span<MetaHit> dHitParams,
                     Span<SurfaceWorkKey> dWorkKeys,
                     // I-O
                     Span<BackupRNGState> rngStates,
                     // Input
                     Span<const RayGMem> dRays,
                     Span<const RayIndex> dRayIndices,
                     // Constants
                     const GPUSystem& s) override;

    void    CastShadowRays(// Output
                           Bitspan<uint32_t> dIsVisibleBuffer,
                           Bitspan<uint32_t> dFoundMediumInterface,
                           // I-O
                           Span<BackupRNGState> rngStates,
                           // Input
                           Span<const RayIndex> dRayIndices,
                           Span<const RayGMem> dShadowRays,
                           // Constants
                           const GPUSystem& s) override;

    void    CastLocalRays(// Output
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> rngStates,
                          // Input
                          Span<const RayGMem> dRays,
                          Span<const RayIndex> dRayIndices,
                          Span<const AcceleratorKey> dAccelIdPacks,
                          // Constants
                          const GPUSystem& s) override;

    void    AllocateForTraversal(size_t maxRayCount) override;
};

inline
BaseAcceleratorLinear::BaseAcceleratorLinear(BS::thread_pool& tp, GPUSystem& sys,
                                             std::map<std::string_view, AccelGroupGenerator>&& aGen)
    : BaseAcceleratorT<BaseAcceleratorLinear>(tp, sys, std::move(aGen))
    , accelMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , stackMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , rayPartitioner(sys)
    , maxPartitionCount(0)
{}

#include "AcceleratorLinear.hpp"