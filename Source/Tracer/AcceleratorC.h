#pragma once

#include <concepts>

#include "TracerTypes.h"

#include "Core/Types.h"
#include "Core/TypeGenFunction.h"
#include "Core/BitFunctions.h"

#include "Device/GPUSystemForward.h"

#include "Random.h"

#include "GenericGroup.h"
#include "PrimitiveC.h"
#include "LightC.h"

namespace BS { class thread_pool; }

class IdentityTransformContext;

template<class HitType>
struct HitResultT
{
    MaterialKey     materialKey;
    PrimitiveKey    primitiveKey;
    HitType         hit;
    Float           t;
};

// Main accelerator leaf, holds primitive id and
// material id. Most of the time this should be enough
struct AcceleratorLeaf
{
    PrimitiveKey primitiveKey;
    // TODO: Materials are comparably small wrt. primitives
    // so holding a 32-bit value for each triangle is a waste of space.
    // Reason and change this maybe?
    MaterialKey  materialKey;
};

template <class AccelType>
concept AccelC = requires(AccelType acc)
{
    typename AccelType::template PrimType<>;
    typename AccelType::PrimHit;
    typename AccelType::HitResult;
    typename AccelType::DataSoA;

    // Has this specific constructor
    requires requires(const IdentityTransformContext& tc,
                      const typename AccelType::DataSoA& soaData)
    {AccelType(tc, soaData, AcceleratorKey{});};

    // Closest hit and any hit
    {acc.ClosestHit(Ray{}, Vector2f{})
    } -> std::same_as<Optional<typename AccelType::HitResult>>;

    {acc.FirstHit(Ray{}, Vector2f{})
    }-> std::same_as<Optional<typename AccelType::HitResult>>;
};

template <class AccelGroupType>
concept AccelGroupC = requires()
{
    true;
};

template<class AccelInstance>
concept AccelInstanceGroupC = requires(AccelInstance ag,
                                       GPUSystem gpuSystem)
{
    {ag.CastLocalRays(// Output
                      Span<HitKeyPack>{},
                      Span<MetaHit>{},
                      // Input
                      Span<const RayGMem>{},
                      Span<const RayIndex>{},
                      Span<const AcceleratorKey>{},
                      // Constants
                      gpuSystem)} -> std::same_as<void>;
};

template <class BaseAccel>
concept BaseAccelC = requires(BaseAccel ac,
                              GPUSystem gpuSystem)
{
    {ac.CastRays(// Output
                 Span<HitKeyPack>{},
                 Span<MetaHit>{},
                 Span<SurfaceWorkKey>{},
                 // Input
                 Span<const RayGMem>{},
                 Span<const RayIndex>{},
                 // Constants
                 gpuSystem)} -> std::same_as<void>;

    {ac.CastShadowRays(// Output
                       Bitspan<uint32_t>{},
                       Bitspan<uint32_t>{},
                       // Input
                       Span<const RayIndex>{},
                       Span<const RayGMem>{},
                       // Constants
                       gpuSystem)} -> std::same_as<void>;
};

namespace TracerLimits
{
    static constexpr size_t MaxPrimBatchPerSurface = 8;
}

using AlphaMap = TextureView<2, Float>;

struct BaseAccelConstructParams
{
    using SurfPair = Pair<SurfaceId, SurfaceParams>;
    using LightSurfPair = Pair<LightSurfaceId, LightSurfaceParams>;

    const TextureViewMap&                               texViewMap;
    const std::map<PrimGroupId, PrimGroupPtr>&          primGroups;
    const std::map<LightGroupId, LightGroupPtr>&        lightGroups;
    const std::map<TransGroupId, TransformGroupPtr>&    transformGroups;
    Span<const SurfPair>                                mSurfList;
    Span<const LightSurfPair>                           lSurfList;
};

struct AccelGroupConstructParams
{
    using SurfPair              = typename BaseAccelConstructParams::SurfPair;
    using LightSurfPair         = typename BaseAccelConstructParams::LightSurfPair;
    using TGroupedSurfaces      = std::vector<Pair<TransGroupId, Span<const SurfPair>>>;
    using TGroupedLightSurfaces = std::vector<Pair<TransGroupId, Span<const LightSurfPair>>>;

    const std::map<TransGroupId, TransformGroupPtr>* transformGroups;
    const TextureViewMap*                            textureViews;
    const GenericGroupPrimitiveT*   primGroup;
    const GenericGroupLightT*       lightGroup;
    TGroupedSurfaces                tGroupSurfs;
    TGroupedLightSurfaces           tGroupLightSurfs;
};

class AcceleratorGroupI
{
    public:
    virtual         ~AcceleratorGroupI() = default;

    virtual void    CastLocalRays(// Output
                                  Span<HitKeyPack> dHitIds,
                                  Span<MetaHit> dHitParams,
                                  Span<SurfaceWorkKey> dWorkKeys,
                                  // I-O
                                  Span<BackupRNGState> rngStates,
                                  // Input
                                  Span<const RayGMem> dRays,
                                  Span<const RayIndex> dRayIndices,
                                  Span<const CommonKey> dAccelKeys,
                                  // Constants
                                  uint32_t instanceId,
                                  const GPUQueue& queue) = 0;

    virtual void        Construct(AccelGroupConstructParams) = 0;

    virtual size_t      InstanceCount() const = 0;
    virtual uint32_t    InstanceTypeCount() const = 0;
    virtual uint32_t    UsedIdBitsInKey() const = 0;
    virtual void        WriteInstanceKeysAndAABBs(Span<AABB3> aabbWriteRegion,
                                                  Span<AcceleratorKey> keyWriteRegion) const = 0;
    virtual uint32_t    SetKeyOffset(uint32_t) = 0;
};

using AccelGroupGenerator = GeneratorFuncType<AcceleratorGroupI, uint32_t,
                                              BS::thread_pool&, GPUSystem&>;

class BaseAcceleratorI
{
    public:
    virtual         ~BaseAcceleratorI() = default;

    // Interface
    // Fully cast rays to entire scene
    virtual void    CastRays(// Output
                             Span<HitKeyPack> dHitIds,
                             Span<MetaHit> dHitParams,
                             Span<SurfaceWorkKey> dWorkKeys,
                             // I-O
                             Span<BackupRNGState> rngStates,
                             // Input
                             Span<const RayGMem> dRays,
                             Span<const RayIndex> dRayIndices,
                             // Constants
                             const GPUSystem& s) = 0;
    // Fully cast rays to entire scene return true/false
    // If it hits to a surface
    virtual void    CastShadowRays(// Output
                                   Bitspan<uint32_t> dIsVisibleBuffer,
                                   Bitspan<uint32_t> dFoundMediumInterface,
                                   // I-O
                                   Span<BackupRNGState> rngStates,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   Span<const RayGMem> dShadowRays,
                                   // Constants
                                   const GPUSystem& s) = 0;
    // Locally cast rays to a accelerator instances
    // This is multi-ray multi-accelerator instance
    virtual void    CastLocalRays(// Output
                                  Span<HitKeyPack> dHitIds,
                                  Span<MetaHit> dHitParams,
                                  // I-O
                                  Span<BackupRNGState> rngStates,
                                  // Input
                                  Span<const RayGMem> dRays,
                                  Span<const RayIndex> dRayIndices,
                                  Span<const AcceleratorKey> dAccelIdPacks,
                                  // Constants
                                  const GPUSystem& s) = 0;

    // Construction
    virtual void    Construct(BaseAccelConstructParams) = 0;
    virtual void    AllocateForTraversal(size_t maxRayCount) = 0;
};

using AccelGroupPtr = std::unique_ptr<AcceleratorGroupI>;
using AcceleratorPtr = std::unique_ptr<BaseAcceleratorI>;

template <class Child>
class BaseAcceleratorT : public BaseAcceleratorI
{
    private:
    void PartitionSurfaces(std::vector<AccelGroupConstructParams>&,
                           Span<const typename BaseAccelConstructParams::SurfPair> surfList,
                           const std::map<PrimGroupId, PrimGroupPtr>& primGroups,
                           const std::map<TransGroupId, TransformGroupPtr>& transGroups,
                           const TextureViewMap& textureViews);
    void AddLightSurfacesToPartitions(std::vector<AccelGroupConstructParams>& partitions,
                                      Span<const typename BaseAccelConstructParams::LightSurfPair> surfList,
                                      const std::map<LightGroupId, LightGroupPtr>& lightGroups);

    protected:
    BS::thread_pool&    threadPool;
    GPUSystem&          gpuSystem;
    uint32_t            idCounter           = 0;
    Vector2ui           maxBitsUsedOnKey    = Vector2ui::Zero();
    std::map<std::string_view, AccelGroupGenerator> accelGenerators;
    std::map<uint32_t, AccelGroupPtr>               generatedAccels;
    std::map<uint32_t, AcceleratorGroupI*>          accelInstances;


    virtual void        InternalConstruct(const std::vector<size_t>& instanceOffsets) = 0;

    public:
    // Constructors & Destructor
                        BaseAcceleratorT(BS::thread_pool&, GPUSystem&,
                                         std::map<std::string_view, AccelGroupGenerator>&&);

    // ....
    void                Construct(BaseAccelConstructParams) override;
};

template<class KeyType>
struct GroupIdFetcher
{
    typename KeyType::Type operator()(auto id)
    {
        uint32_t batchKeyRaw = static_cast<uint32_t>(id);
        return KeyType(batchKeyRaw).FetchBatchPortion();
    }
};

using PrimGroupIdFetcher = GroupIdFetcher<PrimBatchKey>;
using TransGroupIdFetcher = GroupIdFetcher<TransformKey>;
using LightGroupIdFetcher = GroupIdFetcher<LightKey>;

template <class C>
BaseAcceleratorT<C>::BaseAcceleratorT(BS::thread_pool& tp, GPUSystem& system,
                                      std::map<std::string_view, AccelGroupGenerator>&& aGen)
    : threadPool(tp)
    , gpuSystem(system)
    , accelGenerators(aGen)
{}

template <class C>
void BaseAcceleratorT<C>::PartitionSurfaces(std::vector<AccelGroupConstructParams>& partitions,
                                            Span<const typename BaseAccelConstructParams::SurfPair> surfList,
                                            const std::map<PrimGroupId, PrimGroupPtr>& primGroups,
                                            const std::map<TransGroupId, TransformGroupPtr>& transGroups,
                                            const TextureViewMap& textureViews)
{
    using SurfParam = typename BaseAccelConstructParams::SurfPair;
    assert(std::is_sorted(surfList.begin(), surfList.end(),
    [](const SurfParam& left, const SurfParam& right) -> bool
    {
        return (left.second.primBatches.front() <
                right.second.primBatches.front());
    }));
    assert(std::is_sorted(surfList.begin(), surfList.end(),
    [](const SurfParam& left, const SurfParam& right) -> bool
    {
        return (left.second.transformId < right.second.transformId);
    }));

    // TODO: One linear access to vector should be enough
    // to generate this after sort, but this is simpler to write
    // change this if this is a perf bottleneck.
    auto start = surfList.begin();
    while(start != surfList.end())
    {
        auto pBatchId = start->second.primBatches.front();
        uint32_t pGroupId = PrimGroupIdFetcher()(pBatchId);
        auto end = std::upper_bound(start, surfList.end(), pGroupId,
        [](const uint32_t& value, const SurfParam& surf)
        {
            uint32_t batchPortion = PrimGroupIdFetcher()(surf.second.primBatches.front());
            return batchPortion < value;
        });

        partitions.emplace_back(AccelGroupConstructParams{});
        partitions.back().primGroup = primGroups.at(PrimGroupId(pGroupId)).get();
        partitions.back().textureViews = &textureViews;
        partitions.back().transformGroups = &transGroups;
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            uint32_t tGroupId = TransGroupIdFetcher()(tId);
            auto innerEnd = std::upper_bound(start, surfList.end(), tGroupId,
            [](uint32_t value, const SurfParam& surf)
            {
                auto tId = surf.second.transformId;
                return TransGroupIdFetcher()(tId) < value;
            });

            auto surfSpan = Span<const SurfParam>(innerStart, innerEnd);
            partitions.back().tGroupSurfs.emplace_back(TransGroupId(tGroupId), surfSpan);
            innerStart = innerEnd;
        }
        start = end;
    }
}

template <class C>
void BaseAcceleratorT<C>::AddLightSurfacesToPartitions(std::vector<AccelGroupConstructParams>& partitions,
                                                       Span<const typename BaseAccelConstructParams::LightSurfPair> lSurfList,
                                                       const std::map<LightGroupId, LightGroupPtr>& lightGroups)
{
    using LightSurfP = typename BaseAccelConstructParams::LightSurfPair;
    assert(std::is_sorted(lSurfList.begin(), lSurfList.end(),
    [](const LightSurfP& left, const LightSurfP& right)
    {
        return left.second.lightId < right.second.lightId;
    }));
    assert(std::is_sorted(lSurfList.begin(), lSurfList.end(),
    [](const LightSurfP& left, const LightSurfP& right)
    {
        return left.second.transformId < right.second.transformId;
    }));

    // Now partition
    auto start = lSurfList.begin();
    while(start != lSurfList.end())
    {
        uint32_t lGroupId = LightGroupIdFetcher()(start->second.lightId);
        auto end = std::upper_bound(start, lSurfList.end(), lGroupId,
        [](const uint32_t& value, const LightSurfP& surf)
        {
            uint32_t batchPortion = LightGroupIdFetcher()(surf.second.lightId);
            return batchPortion < value;
        });

        //
        auto groupId = LightGroupId(lGroupId);
        const GenericGroupLightT* lGroup = lightGroups.at(groupId).get();
        const GenericGroupPrimitiveT* pGroup = &lGroup->GenericPrimGroup();
        auto slot = std::find_if(partitions.begin(), partitions.end(),
        [pGroup](const auto& partition)
        {
            return (partition.primGroup == pGroup);
        });
        if(slot == partitions.end())
        {
            partitions.emplace_back(AccelGroupConstructParams
            {
                .transformGroups = partitions.front().transformGroups,
                .primGroup = pGroup,
                .lightGroup = lGroup,
                .tGroupSurfs = {},
                .tGroupLightSurfs = {}
            });
            slot = partitions.end() - 1;
        }
        else if(slot->lightGroup) slot->lightGroup = lGroup;

        // Sub-partition wrt. transform
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            uint32_t tGroupId = TransGroupIdFetcher()(tId);
            auto loc = std::upper_bound(innerStart, end, tGroupId,
            [](uint32_t value, const LightSurfP& surf) -> bool
            {
                auto tId = surf.second.transformId;
                return (TransGroupIdFetcher()(tId) < value);
            });
            size_t elemCount = std::distance(innerStart, loc);
            size_t startDistance = std::distance(lSurfList.begin(), innerStart);
            slot->tGroupLightSurfs.emplace_back(TransGroupId(tGroupId),
                                                lSurfList.subspan(startDistance, elemCount));
            innerStart = loc;
        }
    }
}

template <class C>
void BaseAcceleratorT<C>::Construct(BaseAccelConstructParams p)
{
    std::vector<AccelGroupConstructParams> partitions;
    PartitionSurfaces(partitions, p.mSurfList, p.primGroups,
                      p.transformGroups, p.texViewMap);
    // Add primitive-backed lights surfaces as well
    AddLightSurfacesToPartitions(partitions, p.lSurfList, p.lightGroups);

    // Generate the accelerators
    for(const auto& partition : partitions)
    {
        using namespace TypeNameGen::Runtime;
        using namespace std::string_view_literals;
        std::string accelTypeName = CreateAcceleratorType(C::TypeName(),
                                                          partition.primGroup->Name());
        uint32_t aGroupId = idCounter++;
        auto accelPtr = accelGenerators.at(accelTypeName)(std::move(aGroupId),
                                                          threadPool, gpuSystem);
        auto loc = generatedAccels.emplace(aGroupId, std::move(accelPtr));
        AcceleratorGroupI* acc = loc.first->second.get();
        acc->Construct(std::move(partition));
    }
    // Find the leaf count
    std::vector<size_t> instanceOffsets(generatedAccels.size() + 1, 0);
    std::transform_inclusive_scan(generatedAccels.cbegin(),
                                  generatedAccels.cend(),
                                  instanceOffsets.begin() + 1, std::plus{},
    [](const auto& pair) -> size_t
    {
        return pair.second->InstanceCount();
    });
    //
    std::vector<uint32_t> keyOffsets(generatedAccels.size() + 1, 0);
    std::transform_inclusive_scan(generatedAccels.cbegin(),
                                  generatedAccels.cend(),
                                  keyOffsets.begin() + 1, std::plus{},
    [](const auto& pair) -> uint32_t
    {
        return pair.second->InstanceTypeCount();
    });
    // Set the offsets
    uint32_t i = 0;
    for(auto& group : generatedAccels)
    {
        group.second->SetKeyOffset(keyOffsets[i]);
        i++;
    }

    // Find the maximum bits used on key
    uint32_t keyBatchPortionMax = keyOffsets.back();
    uint32_t keyIdPortionMax = std::transform_reduce(generatedAccels.cbegin(),
                                                     generatedAccels.cend(),
                                                     uint32_t(0),
    [](uint32_t rhs, uint32_t lhs)
    {
        return std::max(rhs, lhs);
    },
    [](const auto& pair)
    {
        return pair.second->UsedIdBitsInKey();
    });
    using namespace BitFunctions;
    maxBitsUsedOnKey = Vector2ui(RequiredBitsToRepresent(keyBatchPortionMax),
                                 RequiredBitsToRepresent(keyIdPortionMax));

    // Internal construction routine,
    // we can not fetch the leaf data here because some accelerators are
    // constructed on CPU (due to laziness)
    InternalConstruct(instanceOffsets);
}