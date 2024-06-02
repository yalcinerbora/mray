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

class AcceleratorWorkI;

template<class HitType>
struct HitResultT
{
    HitType         hit;
    Float           t;
    PrimitiveKey    primitiveKey;
    LightOrMatKey   lmKey;
};

template <class AccelType>
concept AccelC = requires(AccelType acc,
                          BackupRNG& rng)
{
    typename AccelType::HitResult;
    typename AccelType::DataSoA;
    typename AccelType::PrimDataSoA;
    typename AccelType::TransDataSoA;

    // Closest hit and any hit
    {acc.ClosestHit(rng, Ray{}, Vector2f{})
    } -> std::same_as<Optional<typename AccelType::HitResult>>;

    {acc.FirstHit(rng, Ray{}, Vector2f{})
    }-> std::same_as<Optional<typename AccelType::HitResult>>;
};

template <class AGType>
concept AccelGroupC = requires(AGType ag)
{
    // Mandatory Types
    // Accel type satisfies its concept (at least on default form)
    requires AccelC<typename AGType::template Accelerator<>>;

    // Should define the prim group type
    typename AGType::PrimitiveGroup;
    // And access it via function
    { ag.PrimGroup() } -> std::same_as<const GenericGroupPrimitiveT&>;

    // SoA fashion accelerator data. This will be used to access internal
    // of the accelerator with a given an index
    typename AGType::DataSoA;
    //std::is_same_v<typename AGType::DataSoA,
    //               typename AGType::template Accelerator<>::DataSoA>;
    // Acquire SoA struct of this material group
    { ag.SoA() } -> std::same_as<typename AGType::DataSoA>;

    // Must return its type name
    { AGType::TypeName() } -> std::same_as<std::string_view>;
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

    const TextureViewMap&                       texViewMap;
    const Map<PrimGroupId, PrimGroupPtr>&       primGroups;
    const Map<LightGroupId, LightGroupPtr>&     lightGroups;
    const Map<TransGroupId, TransformGroupPtr>& transformGroups;
    Span<const SurfPair>                        mSurfList;
    Span<const LightSurfPair>                   lSurfList;
};

struct AccelGroupConstructParams
{
    using SurfPair              = typename BaseAccelConstructParams::SurfPair;
    using LightSurfPair         = typename BaseAccelConstructParams::LightSurfPair;
    using TGroupedSurfaces      = std::vector<Pair<TransGroupId, Span<const SurfPair>>>;
    using TGroupedLightSurfaces = std::vector<Pair<TransGroupId, Span<const LightSurfPair>>>;

    const Map<TransGroupId, TransformGroupPtr>* transformGroups;
    const TextureViewMap*                       textureViews;
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

    virtual void        Construct(AccelGroupConstructParams,
                                  const GPUQueue&) = 0;

    virtual size_t      InstanceCount() const = 0;
    virtual uint32_t    InstanceTypeCount() const = 0;
    virtual uint32_t    UsedIdBitsInKey() const = 0;
    virtual void        WriteInstanceKeysAndAABBs(Span<AABB3> aabbWriteRegion,
                                                  Span<AcceleratorKey> keyWriteRegion) const = 0;
    virtual void        SetKeyOffset(uint32_t) = 0;
    virtual size_t      GPUMemoryUsage() const = 0;

    virtual const GenericGroupPrimitiveT& PrimGroup() const = 0;
};

using PrimRangeArray        = std::array<Vector2ui, TracerConstants::MaxPrimBatchPerSurface>;
using LightOrMatKeyArray    = std::array<LightOrMatKey, TracerConstants::MaxPrimBatchPerSurface>;
// TODO: std::bitset is CPU oriented class holds the data in (at least i checked MSVC std lib) 32/64-bit integer
// For CUDA, it can be 8/16-bit, properly packed data (since MaxPrimBatchPerSurface
// is 8 in the current impl.) This should not be a memory concern untill
// this codebase becomes production level renderer (doubt)
using CullFaceFlagArray     = Bitset<TracerConstants::MaxPrimBatchPerSurface>;
using AlphaMapArray         = std::array<Optional<AlphaMap>, TracerConstants::MaxPrimBatchPerSurface>;

struct AcceleratorLeafResult
{
    // Index to "perInstanceLeafRanges" for all unique(concerete) accels
    std::vector<uint32_t>   concereteAccelIndices;
    // Leaf [start,end) of each accelerator instance
    std::vector<Vector2ui>  perInstanceLeafRanges;
    // Total leaf count required for this accelerator group
    size_t                  totalLeafCount;
};

struct LinearizedSurfaceData
{
    std::vector<PrimRangeArray>     primRanges;
    std::vector<LightOrMatKeyArray> lightOrMatKeys;
    std::vector<AlphaMapArray>      alphaMaps;
    std::vector<CullFaceFlagArray>  cullFaceFlags;
    std::vector<TransformKey>       transformKeys;
    std::vector<SurfacePrimList>    instancePrimBatches;
};

using AccelWorkGenerator = GeneratorFuncType<AcceleratorWorkI, AcceleratorGroupI&,
                                             GenericGroupTransformT&>;
using AccelWorkGenMap = Map<std::string_view, AccelWorkGenerator>;

using AccelGroupGenerator = GeneratorFuncType<AcceleratorGroupI, uint32_t,
                                              BS::thread_pool&, GPUSystem&,
                                              const GenericGroupPrimitiveT&,
                                              const AccelWorkGenMap&>;

using AccelGroupGenMap = Map<std::string_view, AccelGroupGenerator>;


// Generic find routine in an accelerator instance
// An accelerator consists of multiple prim batches
// and each have some flags (cull-face etc.)
// we need to find the index of the corresponding batch
// to access the data.
// Since at most (currently 8) prim batch can be available on an accelerator instance,
// just linear searches over the ranges.
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t FindPrimBatchIndex(const PrimRangeArray& primRanges, PrimitiveKey k)
{
    static constexpr uint32_t N = static_cast<uint32_t>(std::tuple_size_v<LightOrMatKeyArray>);

    // Linear search over the index
    // List has few elements so linear search should suffice
    CommonKey primIndex = k.FetchIndexPortion();
    UNROLL_LOOP
    for(uint32_t i = 0; i < N; i++)
    {
        // Do not do early break here (not every accelerator will use all 8
        // slots, it may break unrolling. Unused element ranges should be int_max
        // thus, will fail
        bool inRange = (primIndex >= primRanges[i][0] &&
                        primIndex < primRanges[i][1]);
        if(inRange) return i;
    }
    // Return INT_MAX here to crash if something goes wrong
    return std::numeric_limits<uint32_t>::max();
}

template <PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupT : public AcceleratorGroupI
{
    using AccelWorkPtr = std::unique_ptr<AcceleratorWorkI>;

    protected:
    BS::thread_pool&            threadPool;
    GPUSystem&                  gpuSystem;
    const PrimitiveGroupType&   pg;
    uint32_t                    accelGroupId       = 0;
    uint32_t                    concreteAccelCount = 0;
    uint32_t                    instanceCount      = 0;
    uint32_t                    instanceTypeCount  = 0;

    // Type Related
    std::vector<uint32_t>           typeIds;
    const AccelWorkGenMap&          accelWorkGenerators;
    Map<uint32_t, AccelWorkPtr>     workInstances;

    // Common functionality for ineriting types
    static uint32_t                 DetermineInstanceCount(const AccelGroupConstructParams& p);
    static uint32_t                 DetermineInstanceTypeCount(const AccelGroupConstructParams& p);
    static AcceleratorLeafResult    DetermineConcereteAccelCount(std::vector<SurfacePrimList> instancePrimBatches,
                                                                 std::vector<PrimRangeArray> instancePrimRanges);
    template<class T>
    static std::vector<Span<T>>     CreateLeafSubspans(Span<T> fullRange,
                                                       const std::vector<Vector2ui>& leafRanges);
    static LinearizedSurfaceData    LinearizeSurfaceData(const AccelGroupConstructParams& p,
                                                         size_t totalSurfaceCount,
                                                         const PrimitiveGroupType& pg);

    public:
    // Constructors & Destructor
                AcceleratorGroupT(uint32_t accelGroupId,
                                  BS::thread_pool&, GPUSystem&,
                                  const GenericGroupPrimitiveT& pg,
                                  const AccelWorkGenMap&);

    size_t      InstanceCount() const override;
    uint32_t    InstanceTypeCount() const override;
    uint32_t    UsedIdBitsInKey() const override;
    void        SetKeyOffset(uint32_t) override;

    const GenericGroupPrimitiveT& PrimGroup() const override;
};

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
    virtual size_t  GPUMemoryUsage() const = 0;
    virtual AABB3   SceneAABB() const = 0;
};

using AccelGroupPtr = std::unique_ptr<AcceleratorGroupI>;
using AcceleratorPtr = std::unique_ptr<BaseAcceleratorI>;

template <class Child>
class BaseAcceleratorT : public BaseAcceleratorI
{
    private:
    void PartitionSurfaces(std::vector<AccelGroupConstructParams>&,
                           Span<const typename BaseAccelConstructParams::SurfPair> surfList,
                           const Map<PrimGroupId, PrimGroupPtr>& primGroups,
                           const Map<TransGroupId, TransformGroupPtr>& transGroups,
                           const TextureViewMap& textureViews);
    void AddLightSurfacesToPartitions(std::vector<AccelGroupConstructParams>& partitions,
                                      Span<const typename BaseAccelConstructParams::LightSurfPair> surfList,
                                      const Map<LightGroupId, LightGroupPtr>& lightGroups);

    protected:
    BS::thread_pool&    threadPool;
    GPUSystem&          gpuSystem;
    uint32_t            idCounter           = 0;
    Vector2ui           maxBitsUsedOnKey    = Vector2ui::Zero();
    AABB3               sceneAABB;
    //
    const AccelGroupGenMap&             accelGenerators;
    const AccelWorkGenMap&              workGenGlobalMap;
    Map<uint32_t, AccelGroupPtr>        generatedAccels;
    Map<uint32_t, AcceleratorGroupI*>   accelInstances;

    virtual AABB3       InternalConstruct(const std::vector<size_t>& instanceOffsets) = 0;

    public:
    // Constructors & Destructor
                        BaseAcceleratorT(BS::thread_pool&, GPUSystem&,
                                         const AccelGroupGenMap&,
                                         const AccelWorkGenMap&);

    // ....
    void                Construct(BaseAccelConstructParams) override;
    AABB3               SceneAABB() const override;
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
                                      const AccelGroupGenMap& aGen,
                                      const AccelWorkGenMap& globalWorkMap)
    : threadPool(tp)
    , gpuSystem(system)
    , accelGenerators(aGen)
    , workGenGlobalMap(globalWorkMap)
{}

template <class C>
void BaseAcceleratorT<C>::PartitionSurfaces(std::vector<AccelGroupConstructParams>& partitions,
                                            Span<const typename BaseAccelConstructParams::SurfPair> surfList,
                                            const Map<PrimGroupId, PrimGroupPtr>& primGroups,
                                            const Map<TransGroupId, TransformGroupPtr>& transGroups,
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
        auto pGroupOpt = primGroups.at(PrimGroupId(pGroupId));
        if(!pGroupOpt)
        {
            throw MRayError("{:s}: Unable to find primitive group()",
                            C::TypeName(), pGroupId);
        }
        partitions.back().primGroup = pGroupOpt.value().get().get();
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
                                                       const Map<LightGroupId, LightGroupPtr>& lightGroups)
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
        auto lGroupOpt = lightGroups.at(groupId);
        if(!lGroupOpt)
        {
            throw MRayError("{:s}: Unable to find light group()",
                            C::TypeName(), lGroupId);
        }
        const GenericGroupLightT* lGroup = lGroupOpt.value().get().get();
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
            size_t elemCount = static_cast<size_t>(std::distance(innerStart, loc));
            size_t startDistance = static_cast<size_t>(std::distance(lSurfList.begin(), innerStart));
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
    // TODO: do multi-gpu stuff here
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    for(const auto& partition : partitions)
    {
        using namespace TypeNameGen::Runtime;
        using namespace std::string_view_literals;
        std::string accelTypeName = std::string(C::TypeName()) + std::string(partition.primGroup->Name());
        uint32_t aGroupId = idCounter++;
        auto accelGenerator = accelGenerators.at(accelTypeName);
        if(!accelGenerator)
        {
            throw MRayError("{:s}: Unable to find generator for accelerator group \"{:s}\"",
                            C::TypeName(), accelTypeName);
        }
        auto accelPtr = accelGenerator.value()(std::move(aGroupId),
                                               threadPool, gpuSystem,
                                               *partition.primGroup,
                                               workGenGlobalMap);
        auto loc = generatedAccels.emplace(aGroupId, std::move(accelPtr));
        AcceleratorGroupI* acc = loc.first->second.get();
        acc->Construct(std::move(partition), queue);
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
    sceneAABB = InternalConstruct(instanceOffsets);
}

template <class C>
AABB3 BaseAcceleratorT<C>::SceneAABB() const
{
    return sceneAABB;
}

template <PrimitiveGroupC PG>
uint32_t AcceleratorGroupT<PG>::DetermineInstanceCount(const AccelGroupConstructParams& p)
{
    size_t surfCount = std::transform_reduce(p.tGroupSurfs.cbegin(),
                                             p.tGroupSurfs.cend(),
                                             size_t(0), std::plus{},
                                             [](const auto& groupedSurf)
    {
        return groupedSurf.second.size();
    });
    size_t lSurfCount = std::transform_reduce(p.tGroupLightSurfs.cbegin(),
                                              p.tGroupLightSurfs.cend(),
                                              size_t(0), std::plus{},
                                              [](const auto& groupedSurf)
    {
        return groupedSurf.second.size();
    });
    uint32_t totalSurfaceCount = static_cast<uint32_t>(surfCount + lSurfCount);
    return totalSurfaceCount;
}

template <PrimitiveGroupC PG>
uint32_t AcceleratorGroupT<PG>::DetermineInstanceTypeCount(const AccelGroupConstructParams& p)
{
    // Find the instance type count
    // Pair of [TransformType, PrimitiveType]
    // defines an instance, since we are on a prim typed class
    // we need to check transform groups, this is already partitioned by the
    // base accelerator, we just need to get the size
    //
    // However, light types also has its own partitioned array, so we need to
    // add only the non-duplicates there
    uint32_t instanceTypeCount = static_cast<uint32_t>(p.tGroupSurfs.size());
    size_t lightInstanceTypeCount = std::transform_reduce(p.tGroupLightSurfs.cbegin(),
                                                            p.tGroupLightSurfs.cend(),
                                                            size_t(0), std::plus{},
                                                            [&](const auto& groupedLightSurf)
    {
        TransGroupId tId = groupedLightSurf.first;
        auto loc = std::find_if(p.tGroupSurfs.cbegin(),
                                p.tGroupSurfs.cend(),
                                [tId](const auto groupedSurf)
        {
            return groupedSurf.first != tId;
        });
        return (loc == p.tGroupSurfs.cend()) ? 0 : 1;
    });
    instanceTypeCount += static_cast<uint32_t>(lightInstanceTypeCount);
    return instanceTypeCount;
};

template <PrimitiveGroupC PG>
AcceleratorLeafResult AcceleratorGroupT<PG>::DetermineConcereteAccelCount(std::vector<SurfacePrimList> instancePrimBatches,
                                                                          std::vector<PrimRangeArray> instancePrimRanges)
{
    assert(instancePrimBatches.size() == instancePrimRanges.size());

    std::vector<uint32_t> acceleratorIndices(instancePrimBatches.size());
    std::vector<uint32_t> uniqueIndices(instancePrimBatches.size());
    std::iota(acceleratorIndices.begin(), acceleratorIndices.end(), 0);
    using enum PrimTransformType;
    if constexpr(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
    {
        // This means we can fully utilize the primitive
        // Sort and find the unique primitive groups
        // only generate accelerators for these, refer with other instances
        std::vector<uint32_t>& nonUniqueIndices = acceleratorIndices;
        // Commonalize the innter lists (sort it)
        for(SurfacePrimList& lst : instancePrimBatches)
        {
            // Hopefully stl calls insertion sort here or something...
            std::sort(lst.begin(), lst.end(), [](PrimBatchId lhs, PrimBatchId rhs)
            {
                return (static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs));
            });
        }
        // Do an index/id sort here, c++ does not have it
        // so sort iota wrt. values and use it to find instances
        // Use stable sort here to keep the relative transform order of primitives
        std::stable_sort(nonUniqueIndices.begin(), nonUniqueIndices.end(),
        [&instancePrimBatches](uint32_t l, uint32_t r)
        {
            const SurfacePrimList& lhs = instancePrimBatches[l];
            const SurfacePrimList& rhs = instancePrimBatches[r];
            bool result = std::lexicographical_compare(lhs.cbegin(), lhs.cend(),
                                                       rhs.cbegin(), rhs.cend(),
            [](PrimBatchId lhs, PrimBatchId rhs)
            {
                return (static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs));
            });
            return result;
        });
        std::unique_copy(nonUniqueIndices.begin(), nonUniqueIndices.end(), uniqueIndices.begin(),
        [&instancePrimBatches](uint32_t l, uint32_t r)
        {
            const SurfacePrimList& lhs = instancePrimBatches[l];
            const SurfacePrimList& rhs = instancePrimBatches[r];
            auto result = std::lexicographical_compare_three_way(lhs.cbegin(), lhs.cend(),
                                                                 rhs.cbegin(), rhs.cend(),
            [](PrimBatchId lhs, PrimBatchId rhs)
            {
                return (static_cast<uint32_t>(lhs) <=> static_cast<uint32_t>(rhs));
            });
            return std::is_eq(result);
        });
        // Do an inverted search your id on unique indices
        std::for_each(nonUniqueIndices.begin(), nonUniqueIndices.end(),
        [&uniqueIndices](uint32_t& index)
        {
            auto loc = std::lower_bound(uniqueIndices.begin(), uniqueIndices.end(), index);
            index = static_cast<uint32_t>(std::distance(uniqueIndices.begin(), loc));
        });
    }
    else
    {
        // PG supports "PER_PRIMITIVE_TRANSFORM", we cannot refer to the same
        // accelerator, we need to construct an accelerator for each instance
        // Do not bother sorting etc here..
        // Copy here for continuation
        acceleratorIndices = uniqueIndices;
    }

    // Determine the leaf ranges
    // TODO: We are
    std::vector<uint32_t> uniquePrimCounts(uniqueIndices.size() + 1);
    for(uint32_t index : uniqueIndices)
    {
        uint32_t totalLocalPrimCount = 0;
        for(Vector2ui range : instancePrimRanges[index])
        {
            if(range == Vector2ui(std::numeric_limits<uint32_t>::max())) break;
            totalLocalPrimCount += (range[1] - range[0]);
        }
        uniquePrimCounts.push_back(totalLocalPrimCount);
    }
    std::exclusive_scan(uniquePrimCounts.cbegin(), uniquePrimCounts.cend(),
                        uniquePrimCounts.begin(), 0u);

    // Rename the variable for better maintenance
    std::vector<uint32_t>& uniquePrimOffsets = uniquePrimCounts;
    std::vector<Vector2ui> leafRangeList;
    leafRangeList.reserve(instancePrimBatches.size());
    for(uint32_t index : acceleratorIndices)
    {
        leafRangeList.push_back(Vector2ui(uniquePrimOffsets[index],
                                          uniquePrimOffsets[index + 1]));
    }

    // All done!
    return AcceleratorLeafResult
    {
        .concereteAccelIndices = std::move(uniqueIndices),
        .perInstanceLeafRanges = std::move(leafRangeList),
        .totalLeafCount = uniquePrimOffsets.back()
    };
}

template <PrimitiveGroupC PG>
template<class T>
std::vector<Span<T>>
AcceleratorGroupT<PG>::CreateLeafSubspans(Span<T> fullRange,
                                          const std::vector<Vector2ui>& leafRanges)
{
    std::vector<Span<T>> instanceLeafData(leafRanges.size());
    std::transform(leafRanges.cbegin(), leafRanges.cend(),
                   instanceLeafData.begin(),
                   [fullRange](const Vector2ui& offset)
    {
        uint32_t size = offset[1] - offset[0];
        return fullRange.subspan(offset[0], size);
    });
    return instanceLeafData;
}

template <PrimitiveGroupC PG>
LinearizedSurfaceData AcceleratorGroupT<PG>::LinearizeSurfaceData(const AccelGroupConstructParams& p,
                                                                  size_t totalSurfaceCount,
                                                                  const PG& pg)
{
    LinearizedSurfaceData result = {};
    result.primRanges.reserve(totalSurfaceCount);
    result.lightOrMatKeys.reserve(totalSurfaceCount);
    result.alphaMaps.reserve(totalSurfaceCount);
    result.cullFaceFlags.reserve(totalSurfaceCount);
    result.transformKeys.reserve(totalSurfaceCount);
    result.instancePrimBatches.reserve(totalSurfaceCount);

    const auto InitRest = [&](uint32_t restStart)
    {
        using namespace TracerConstants;
        for(uint32_t i = restStart; i < static_cast<uint32_t>(MaxPrimBatchPerSurface); i++)
        {
            result.alphaMaps.back()[i] = std::nullopt;
            result.cullFaceFlags.back()[i] = false;
            result.lightOrMatKeys.back()[i] = LightOrMatKey::InvalidKey();
            result.primRanges.back()[i] = Vector2ui(std::numeric_limits<uint32_t>::max());
        }
    };

    for(const auto& surfList : p.tGroupSurfs)
    for(const auto& [_, surf] : surfList.second)
    {
        result.instancePrimBatches.push_back(surf.primBatches);
        result.alphaMaps.emplace_back();
        result.cullFaceFlags.emplace_back();
        result.lightOrMatKeys.emplace_back();
        result.primRanges.emplace_back();
        result.transformKeys.emplace_back(TransformKey(static_cast<uint32_t>(surf.transformId)));;

        assert(surf.alphaMaps.size() == surf.cullFaceFlags.size());
        assert(surf.cullFaceFlags.size() == surf.materials.size());
        assert(surf.materials.size() == surf.primBatches.size());
        for(uint32_t i = 0; i < static_cast<uint32_t>(surf.alphaMaps.size()); i++)
        {
            if(surf.alphaMaps[i].has_value())
            {
                auto optView = p.textureViews->at(surf.alphaMaps[i].value());
                if(!optView)
                {
                    throw MRayError("Accelerator: Alpha map texture({:d}) is not found",
                                    static_cast<uint32_t>(surf.alphaMaps[i].value()));
                }
                const GenericTextureView& view = optView.value();
                assert(std::holds_alternative<AlphaMap>(view));
                result.alphaMaps.back()[i] = std::get<AlphaMap>(view);
            }
            else result.alphaMaps.back()[i] = std::nullopt;

            result.cullFaceFlags.back()[i] = surf.cullFaceFlags[i];
            result.primRanges.back()[i] = pg.BatchRange(surf.primBatches[i]);
            MaterialKey mKey(static_cast<CommonKey>(surf.materials[i]));
            result.lightOrMatKeys.back()[i] = LightOrMatKey::CombinedKey(IS_MAT_KEY_FLAG,
                                                                         mKey.FetchBatchPortion(),
                                                                         mKey.FetchIndexPortion());
        }
        InitRest(static_cast<uint32_t>(surf.alphaMaps.size()));
    }
    // TODO: Sharing std::array<..> for lights is waste of space
    // Maybe enable multi-prim lights later
    for(const auto& lSurfList : p.tGroupLightSurfs)
    for(const auto& [_, lSurf] : lSurfList.second)
    {
        result.alphaMaps.emplace_back();
        result.cullFaceFlags.emplace_back();
        result.lightOrMatKeys.emplace_back();
        result.primRanges.emplace_back();

        InitRest(0);
        PrimBatchId primBatchId = p.lightGroup->LightPrimBatch(lSurf.lightId);
        result.primRanges.back().front() = pg.BatchRange(primBatchId);
        result.transformKeys.back() = TransformKey(static_cast<uint32_t>(lSurf.transformId));
        result.instancePrimBatches.back().front() = primBatchId;

        LightKey lKey(static_cast<CommonKey>(lSurf.lightId));
        result.lightOrMatKeys.back().front() = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG,
                                                                          lKey.FetchBatchPortion(),
                                                                          lKey.FetchIndexPortion());
    }
    assert(result.alphaMaps.size()              == totalSurfaceCount);
    assert(result.cullFaceFlags.size()          == totalSurfaceCount);
    assert(result.lightOrMatKeys.size()         == totalSurfaceCount);
    assert(result.primRanges.size()             == totalSurfaceCount);
    assert(result.transformKeys.size()          == totalSurfaceCount);
    assert(result.instancePrimBatches.size()    == totalSurfaceCount);
    return result;
}

template <PrimitiveGroupC PG>
AcceleratorGroupT<PG>::AcceleratorGroupT(uint32_t groupId,
                                         BS::thread_pool& tp, GPUSystem& sys,
                                         const GenericGroupPrimitiveT& pgIn,
                                         const AccelWorkGenMap& workGenMap)
    : threadPool(tp)
    , gpuSystem(sys)
    , accelGroupId(groupId)
    , pg(static_cast<const PG&>(pgIn))
    , accelWorkGenerators(workGenMap)
{}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupT<PG>::InstanceCount() const
{
    return instanceCount;
}

template<PrimitiveGroupC PG>
uint32_t AcceleratorGroupT<PG>::InstanceTypeCount() const
{
    return instanceTypeCount;
}

template<PrimitiveGroupC PG>
uint32_t AcceleratorGroupT<PG>::UsedIdBitsInKey() const
{
    return BitFunctions::RequiredBitsToRepresent(instanceTypeCount);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupT<PG>::SetKeyOffset(uint32_t offset)
{
    std::for_each(typeIds.begin(), typeIds.end(),
    [offset](uint32_t& id)
    {
        id += offset;
    });
}

template<PrimitiveGroupC PG>
const GenericGroupPrimitiveT& AcceleratorGroupT<PG>::PrimGroup() const
{
    return pg;
}