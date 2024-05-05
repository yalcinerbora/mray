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
        // Constructor & Destructor
                                AcceleratorLinear(const TransDataSoA& tSoA,
                                                  const PrimDataSoA& pSoA,
                                                  const DataSoA& dataSoA,
                                                  TransformKey tId,
                                                  AcceleratorKey aId);

        MRAY_HYBRID             AcceleratorLinear(const TransDataSoA& tSoA,
                                                  const PrimDataSoA& pSoA,
                                                  const DataSoA& dataSoA,
                                                  AcceleratorId aId,
                                                  TransformKey tId);

        MRAY_HYBRID
        Optional<HitResult>     ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const;
        MRAY_HYBRID
        Optional<HitResult>     FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const;
    };
}


template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupLinear final : public AcceleratorGroupI
{
    public:
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

    Bitspan<uint32_t>               dCullFace;
    Span<Optional<AlphaMap>>        dAlphaMaps;
    Span<Span<AcceleratorLeaf>>     dLeafs;

    Span<Span<AABB3>>               dAABBs;
    Span<AcceleratorLeaf>           dXXXXX;
    Span<>;

    std::vector<std::vector>;

    // Internal concrete accel counter
    uint32_t accelGroupId;
    uint32_t concreteAccelCounter;
    uint32_t surfaceIdCounter;

    PrimBatchListToAccelMapping primBatchMapping;
    LocalSurfaceToAccelMapping  surfMapping;
    bool isInCommitState = false;

    public:
    AcceleratorGroupLinear(uint32_t accelGroupId,
                           const PrimitiveGroupI& pg)
        : accelGroupId(accelGroupId)
        , pg(static_cast<PrimitiveGroup&>(pg))
        , concreteAccelCounter(0)
        , surfaceIdCounter(0)
    {}

    //
    AcceleratorId ReserveSurface(const SurfPrimIdList& primIds,
                                 const SurfMatIdList& matIds) override
    {
        using enum PrimTransformType;

        // Generate a new accelerator per instance(surface)
        // only if the transform requirement is locally constant
        uint32_t concAccelIndex = concreteAccelCounter;
        if constexpr(TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
        {
            SurfPrimIdList sortedPrimIds = primIds;
            std::sort(sortedPrimIds);
            auto result = primBatchMapping.emplace(sortedPrimIds,
                                                   concreteAccelCounter);
            // Get ready for next
            if(result.second) concreteAccelCounter++;
            concAccelIndex = result.first->second;
        }
        else concreteAccelCounter++;

        surfaceIdCounter++;
        surfMapping.emplace(surfaceIdCounter, concAccelIndex);
    }

    // Commit the accelerators return world space AABBs
    Span<AABB3> CommitReservations(const std::vector<BaseAcceleratorLeaf>& baseLeafs,
                                   const AcceleratorWorkI& accWork) override;
    {
        accelAABBs.reserve(baseLeafs.size());
        perAccelLeafs.reserve(baseLeafs.size());

        if constexpr(TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
        {
            // Create Identity transform work
            auto tg = IdentityTransformGroup{};
            const auto& ag = decltype(this);
            AcceleratorWork<decltype(this), IdentityTransformGroup> work(ag, tg);

            std::unordered_map<uint32_t, Span<AcceleratorLeaf>> constructedAccels;
            assert(primBatchMapping.empty());
            std::array<PrimRange, MaxPrimBatchPerSurface> a;

            auto it = constructedAccels.find(primBatchMapping.at());
            for(const BaseAcceleratorLeaf& l : baseLeafs)
            {
                uint32_t accId = l.accelId.FetchIndexPortion();
                uint32_t instanceId = surfMapping.at(accId);
                auto r = constructedAccels.emplace(instanceId);
                if(r.second)
                {
                    MRAY_LOG("{:s}: Constructing Accelerator {}", TypeName(),
                             ...);
                    ConstructAccelerator(a, indentityWork);
                }
                else
                {
                    MRAY_LOG("{:s}: Reusing Accelerator for {}", TypeName(),
                             ...);
                    ReferAccelerator(a, r.first->second);
                }
                AABB3 aabb = GenerateAABB(r.first->second, l.transformKey, accWork);
                result.push_back(aabb);

            }
        }
        else
        {
            MRAY_LOG("{:s}: Constructing Accelerator {}", TypeName(),
                     ...);
            ConstructAccelerator(a, accWork);
            AABB3 aabb = GenerateAABB(r.first->second, l.transformKey, accWork);
            result.push_back(aabb);
            perAccelLeafs.push_back(leafs);
        }

        // Memcopy to a span

    }
};

class AcceleratorBaseLinear final : public BaseAcceleratorT<AcceleratorBaseLinear>
{
    private:
    std::vector<BaseAcceleratorLeaf>                    leafs;
    std::vector<std::unique_ptr<AcceleratorGroupI*>>    localAccelerators;


    public:
    AcceleratorBaseLinear(BS::thread_pool&, GPUSystem&);

    // Cast Rays and Find HitIds, and WorkKeys.
    // (the last one will be used to partition)
    void CastRays(// Output
                  Span<HitIdPack> dHitIds,
                  Span<MetaHit> dHitParams,
                  Span<SurfaceWorkKey> dWorkKeys,
                  // Input
                  Span<const RayGMem> dRays,
                  Span<const RayIndex> dRayIndices,
                  // Constants
                  const GPUSystem& s) override;

    void CastShadowRays(// Output
                        Bitspan<uint32_t> dIsVisibleBuffer,
                        Bitspan<uint32_t> dFoundMediumInterface,
                        // Input
                        Span<const RayIndex> dRayIndices,
                        Span<const RayGMem> dShadowRays,
                        // Constants
                        const GPUSystem& s) override;

    // Each leaf has a surface
    void Construct(BaseAccelConstructParams p) override;
};


inline
AcceleratorBaseLinear::AcceleratorBaseLinear(BS::thread_pool& tp, GPUSystem& sys)
    : BaseAcceleratorT<AcceleratorBaseLinear>(tp, sys)
{

}

template<class KeyType>
struct BatchFetcher
{
    typename KeyType::Type operator()(auto id)
    {
        uint32_t batchKeyRaw = static_cast<uint32_t>(id);
        return KeyType(batchKeyRaw).FetchBatchPortion();
    }
};
using PrimBatchFetcher = BatchFetcher<PrimBatchKey>;
using TransBatchFetcher = BatchFetcher<TransformKey>;

inline
void AcceleratorBaseLinear::Construct(BaseAccelConstructParams p)
{
    using SurfParam = Pair<SurfaceId, SurfaceParams>;
    using LightSurfParam = Pair<LightSurfaceId, LightSurfaceParams>;
    // Pack the surfaces via transform / primitive
    //
    // PG TG {S0,...,SN},  -> AccelGroup (One AccelInstance per S)
    // PG TG {S0,...,SN},  -> AccelGroup
    // ....
    // PG TG {S0,...,SN}   -> AccelGroup
    auto surfList = p.mSurfList;
    using SurfP = Pair<SurfaceId, SurfaceParams>;
    std::stable_sort(surfList.begin(), surfList.end(),
    [](const SurfP& left, const SurfP& right) -> bool
    {
        return (left.second.primBatches.front() <
                right.second.primBatches.front());
    });
    //
    std::stable_sort(surfList.begin(), surfList.end(),
    [](const SurfP& left, const SurfP& right) -> bool
    {
        return (left.second.transformId < right.second.transformId);
    });
    // TODO: One linear access to vector should be enough
    // to generate this after sort, but this is simpler to write
    // change this if this is a perf bottleneck.
    std::vector<AccelGroupConstructParams> partitions;
    auto start = surfList.begin();
    while(start != surfList.end())
    {
        auto pBatchId = start->second.primBatches.front();
        uint32_t pGroupId = PrimBatchFetcher()(pBatchId);
        auto end = std::upper_bound(start, surfList.end(), pGroupId,
        [](const auto& surf,const uint32_t& value)
        {
            uint32_t batchPortion = PrimBatchFetcher()(surf.second.primBatches.front());
            return batchPortion < value;
        });
        partitions.emplace_back(AccelGroupConstructParams{});
        partitions.back().primGroup = p.primGroups.at(PrimGroupId(pGroupId)).get();
        partitions.back().transformGroups = &p.transformGroups;
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            uint32_t tGroupId = TransBatchFetcher()(tId);
            auto innerEnd = std::upper_bound(start, surfList.end(), tGroupId,
            [](const auto& surf, const uint32_t& value)
            {
                auto tId = surf.second.transformId;
                return TransBatchFetcher()(tId) < value;
            });
            auto surfSpan = Span<SurfParam>(innerStart, innerEnd);
            partitions.back().tGroupSurfs.emplace_back(TransGroupId(tGroupId), surfSpan);
            innerStart = innerEnd;
        }
        start = end;
    }
    // Accumulate the light surfaces as well
    // ...

    // Generate Accelerators
    for(const auto& partition : partitions)
    {
        using namespace TypeNameGen::Runtime;
        using namespace std::string_view_literals;
        std::string accelTypeName = CreateAcceleratorType("Linear"sv,
                                                          partition.primGroup->Name());

        uint32_t aGroupId = idCounter++;
        auto accelPtr = accelGenerators.at(accelTypeName)(std::move(aGroupId),
                                                          threadPool, gpuSystem);
        auto loc = generatedAccels.emplace(aGroupId, std::move(accelPtr));
        AcceleratorGroupI* acc = loc.first->second.get();
        acc->Construct(std::move(partition));
    }
}

#include "AcceleratorLinear.hpp"