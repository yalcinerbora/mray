#pragma once

#include <concepts>

#include "TracerTypes.h"

#include "Core/Types.h"
#include "Core/TypeGenFunction.h"
#include "Core/BitFunctions.h"

#include "Device/GPUSystemForward.h"
#include "Device/GPUAlgForward.h"

#include "Random.h"

#include "GenericGroup.h"
#include "PrimitiveC.h"
#include "LightC.h"
#include "TextureView.h"

namespace BS { class thread_pool; }

class AcceleratorWorkI;

using AccelWorkPtr = std::unique_ptr<AcceleratorWorkI>;

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

    {acc.GetTransformKey()
    }-> std::same_as<TransformKey>;


};

struct UnionAABB3Functor
{
    MRAY_GPU MRAY_GPU_INLINE
    AABB3 operator()(const AABB3& l, const AABB3& r) const
    {
        return l.Union(r);
    }
};

template<PrimitiveGroupC PG>
class AABBFetchFunctor
{
    typename PG::DataSoA pData;

    public:
    AABBFetchFunctor(typename PG::DataSoA pd)
        : pData(pd)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    AABB3 operator()(PrimitiveKey k) const
    {
        using Prim = typename PG:: template Primitive<>;
        Prim prim(TransformContextIdentity{}, pData, k);
        AABB3 aabb = prim.GetAABB();
        return aabb;
    }
};

class KeyGeneratorFunctor
{

    CommonKey accelBatchId;
    CommonKey accelIdStart;
    Span<AcceleratorKey> dLocalKeyWriteRegion;

    public:
    KeyGeneratorFunctor(CommonKey accelBatchIdIn,
                        CommonKey accelIdStartIn,
                        Span<AcceleratorKey> dWriteRegions)
        : accelBatchId(accelBatchIdIn)
        , accelIdStart(accelIdStartIn)
        , dLocalKeyWriteRegion(dWriteRegions)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    void operator()(KernelCallParams kp)
    {
        uint32_t keyCount = static_cast<uint32_t>(dLocalKeyWriteRegion.size());
        for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
            dLocalKeyWriteRegion[i] = AcceleratorKey::CombinedKey(accelBatchId, i + accelIdStart);
    }
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
concept BaseAccelC = true;
// TODO: Change these?
// Do we really need this?
//requires(BaseAccel ac, GPUSystem gpuSystem)
//{
//    {ac.CastRays(// Output
//                 Span<HitKeyPack>{},
//                 Span<MetaHit>{},
//                 Span<SurfaceWorkKey>{},
//                 // Input
//                 Span<const RayGMem>{},
//                 Span<const RayIndex>{},
//                 // Constants
//                 gpuSystem)} -> std::same_as<void>;
//
//    {ac.CastShadowRays(// Output
//                       Bitspan<uint32_t>{},
//                       Bitspan<uint32_t>{},
//                       // Input
//                       Span<const RayIndex>{},
//                       Span<const RayGMem>{},
//                       // Constants
//                       gpuSystem)} -> std::same_as<void>;
//};
class BaseAcceleratorI;

namespace TracerLimits
{
    static constexpr size_t MaxPrimBatchPerSurface = 8;
}

// Comparison Routines
inline bool SurfaceLessThan(const Pair<SurfaceId, SurfaceParams>& left,
                            const Pair<SurfaceId, SurfaceParams>& right)
{
    PrimBatchKey lpk = std::bit_cast<PrimBatchKey>(left.second.primBatches.front());
    TransformKey ltk = std::bit_cast<TransformKey>(left.second.transformId);
    //
    PrimBatchKey rpk = std::bit_cast<PrimBatchKey>(right.second.primBatches.front());
    TransformKey rtk = std::bit_cast<TransformKey>(right.second.transformId);
    //
    return (Tuple(lpk.FetchBatchPortion(), ltk.FetchBatchPortion()) <
            Tuple(rpk.FetchBatchPortion(), rtk.FetchBatchPortion()));
}

inline bool LightSurfaceLessThan(const Pair<LightSurfaceId, LightSurfaceParams>& left,
                                 const Pair<LightSurfaceId, LightSurfaceParams>& right)
{
    LightKey llk = std::bit_cast<LightKey>(left.second.lightId);
    TransformKey ltk = std::bit_cast<TransformKey>(left.second.transformId);
    //
    LightKey rlk = std::bit_cast<LightKey>(right.second.lightId);
    TransformKey rtk = std::bit_cast<TransformKey>(right.second.transformId);
    //
    return (Tuple(llk.FetchBatchPortion(), ltk.FetchBatchPortion()) <
            Tuple(rlk.FetchBatchPortion(), rtk.FetchBatchPortion()));
}

inline bool CamSurfaceLessThan(const Pair<CamSurfaceId, CameraSurfaceParams>& left,
                               const Pair<CamSurfaceId, CameraSurfaceParams>& right)
{
    CameraKey lck = std::bit_cast<CameraKey>(left.second.cameraId);
    TransformKey ltk = std::bit_cast<TransformKey>(left.second.transformId);
    //
    CameraKey rck = std::bit_cast<CameraKey>(right.second.cameraId);
    TransformKey rtk = std::bit_cast<TransformKey>(right.second.transformId);
    //
    return (Tuple(lck.FetchBatchPortion(), ltk.FetchBatchPortion()) <
            Tuple(rck.FetchBatchPortion(), rtk.FetchBatchPortion()));
}

// Alias some stuff to easily acquire the function and context type
// Using macro instead of "static constexpr auto" since it make
// GPU link errors
#define MRAY_ACCEL_TGEN_FUNCTION(AG, TG) \
    AcquireTransformContextGenerator<typename AG::PrimitiveGroup, TG>()

template<AccelGroupC AG, TransformGroupC TG>
using AccTransformContextType = PrimTransformContextType<typename AG::PrimitiveGroup, TG>;

using AlphaMap = TracerTexView<2, Float>;

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
                                  // I-O
                                  Span<BackupRNGState> dRNGStates,
                                  Span<RayGMem> dRays,
                                  // Input
                                  Span<const RayIndex> dRayIndices,
                                  Span<const CommonKey> dAccelKeys,
                                  // Constants
                                  uint32_t instanceId,
                                  const GPUQueue& queue) = 0;

    virtual void        PreConstruct(const BaseAcceleratorI*) = 0;
    virtual void        Construct(AccelGroupConstructParams,
                                  const GPUQueue&) = 0;

    virtual size_t      InstanceCount() const = 0;
    virtual uint32_t    InstanceTypeCount() const = 0;
    virtual uint32_t    AcceleratorCount() const = 0;
    virtual uint32_t    UsedIdBitsInKey() const = 0;
    virtual void        WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                                  Span<AcceleratorKey> dKeyWriteRegion,
                                                  const GPUQueue& queue) const = 0;
    virtual void                SetKeyOffset(uint32_t) = 0;
    virtual size_t              GPUMemoryUsage() const = 0;
    virtual std::string_view    Name() const = 0;

    virtual const GenericGroupPrimitiveT& PrimGroup() const = 0;
};

using AccelGroupPtr         = std::unique_ptr<AcceleratorGroupI>;
using PrimRangeArray        = std::array<Vector2ui, TracerConstants::MaxPrimBatchPerSurface>;
using LightOrMatKeyArray    = std::array<LightOrMatKey, TracerConstants::MaxPrimBatchPerSurface>;
// TODO: std::bitset is CPU oriented class holds the data in (at least i checked MSVC std lib) 32/64-bit integer
// For CUDA, it can be 8/16-bit, properly packed data (since MaxPrimBatchPerSurface
// is 8 in the current impl.) This should not be a memory concern untill
// this codebase becomes production level renderer (doubt)
using CullFaceFlagArray     = Bitset<TracerConstants::MaxPrimBatchPerSurface>;
using AlphaMapArray         = std::array<Optional<AlphaMap>, TracerConstants::MaxPrimBatchPerSurface>;

struct AccelLeafResult
{
    // Indices to the concrete accelerators (per instance)
    std::vector<uint32_t>   concreteIndicesOfInstances;
    // Leaf [start,end) of each accelerator instance
    std::vector<Vector2ui>  instanceLeafRanges;
    // Leaf [start,end) of only concrete accelerators
    std::vector<Vector2ui>  concreteLeafRanges;
    // PrimRange {[start,end)_0, ... [start,end)_8}
    // of only concrete accelerators
    std::vector<PrimRangeArray>  concretePrimRanges;
};

struct AccelPartitionResult
{
    struct IndexPack
    {
        TransGroupId    tId;
        uint32_t        tSurfIndex;
        uint32_t        tLightSurfIndex;
    };
    std::vector<IndexPack>  packedIndices;
    std::vector<uint32_t>   workPartitionOffsets;
    uint32_t                totalInstanceCount;
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

struct PreprocessResult
{
    LinearizedSurfaceData       surfData;
    std::vector<PrimRangeArray> concretePrimRanges;
};

using AccelWorkGenerator = GeneratorFuncType<AcceleratorWorkI,
                                             const AcceleratorGroupI&,
                                             const GenericGroupTransformT&>;
using AccelWorkGenMap = Map<std::string_view, AccelWorkGenerator>;

using AccelGroupGenerator = GeneratorFuncType<AcceleratorGroupI, uint32_t,
                                              BS::thread_pool&,
                                              const GPUSystem&,
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

template <class Child, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> Base = AcceleratorGroupI>
class AcceleratorGroupT : public Base
{
    using PrimitiveGroupType = PG;
    private:
    // Common functionality for ineriting types
    static AccelPartitionResult     PartitionParamsForWork(const AccelGroupConstructParams& p);
    static AccelLeafResult          DetermineConcreteAccelCount(std::vector<SurfacePrimList> instancePrimBatches,
                                                                const std::vector<PrimRangeArray>& instancePrimRanges);
    static LinearizedSurfaceData    LinearizeSurfaceData(const AccelGroupConstructParams& p,
                                                         const AccelPartitionResult& partitions,
                                                         const typename PrimitiveGroupType& pg);

    protected:
    BS::thread_pool&            threadPool;
    const GPUSystem&            gpuSystem;
    const PrimitiveGroupType&   pg;
    uint32_t                    accelGroupId = 0;

    // Each instance is backed with a concrete accelerator
    // This indirection represents it
    std::vector<uint32_t>       concreteIndicesOfInstances;
    // Offset list of instance/concrete leafs
    std::vector<Vector2ui>      concreteLeafRanges;
    std::vector<Vector2ui>      instanceLeafRanges;
    std::vector<uint32_t>       workInstanceOffsets;

    // Type Related
    uint32_t                    globalWorkIdToLocalOffset = std::numeric_limits<uint32_t>::max();
    const AccelWorkGenMap&      accelWorkGenerators;
    Map<uint32_t, AccelWorkPtr> workInstances;

    PreprocessResult            PreprocessConstructionParams(const AccelGroupConstructParams& p);
    template<class T>
    std::vector<Span<T>>        CreateInstanceSubspans(Span<T> fullRange,
                                                       const std::vector<Vector2ui>& rangeVector);
    void                        WriteInstanceKeysAndAABBsInternal(Span<AABB3> aabbWriteRegion,
                                                                  Span<AcceleratorKey> keyWriteRegion,
                                                                  // Input
                                                                  Span<const PrimitiveKey> dAllLeafs,
                                                                  Span<const TransformKey> dTransformKeys,
                                                                  // Constants
                                                                  const GPUQueue& queue) const;

    public:
    // Constructors & Destructor
                        AcceleratorGroupT(uint32_t accelGroupId,
                                          BS::thread_pool&,
                                          const GPUSystem&,
                                          const GenericGroupPrimitiveT& pg,
                                          const AccelWorkGenMap&);

    virtual void        PreConstruct(const BaseAcceleratorI*) override;
    size_t              InstanceCount() const override;
    uint32_t            InstanceTypeCount() const override;
    uint32_t            AcceleratorCount() const override;
    uint32_t            UsedIdBitsInKey() const override;
    void                SetKeyOffset(uint32_t) override;
    std::string_view    Name() const override;

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
                             // I-O
                             Span<BackupRNGState> dRNGStates,
                             Span<RayGMem> dRays,
                             // Input
                             Span<const RayIndex> dRayIndices,
                             const GPUQueue& queue) = 0;
    // Fully cast rays to entire scene return true/false
    // If it hits to a surface
    virtual void    CastShadowRays(// Output
                                   Bitspan<uint32_t> dIsVisibleBuffer,
                                   Bitspan<uint32_t> dFoundMediumInterface,
                                   // I-O
                                   Span<BackupRNGState> dRNGStates,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   Span<const RayGMem> dShadowRays,
                                   const GPUQueue& queue) = 0;
    // Locally cast rays to a accelerator instances
    // This is multi-ray multi-accelerator instance
    virtual void    CastLocalRays(// Output
                                  Span<HitKeyPack> dHitIds,
                                  Span<MetaHit> dHitParams,
                                  // I-O
                                  Span<BackupRNGState> dRNGStates,
                                  // Input
                                  Span<const RayGMem> dRays,
                                  Span<const RayIndex> dRayIndices,
                                  Span<const AcceleratorKey> dAccelIdPacks,
                                  const GPUQueue& queue) = 0;

    // Construction
    virtual void    Construct(BaseAccelConstructParams) = 0;
    virtual void    AllocateForTraversal(size_t maxRayCount) = 0;
    virtual size_t  GPUMemoryUsage() const = 0;
    virtual AABB3   SceneAABB() const = 0;
    virtual size_t  TotalAccelCount() const = 0;
    virtual size_t  TotalInstanceCount() const = 0;
};

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
                                      const Map<LightGroupId, LightGroupPtr>& lightGroups,
                                      const Map<TransGroupId, TransformGroupPtr>& transGroups);

    protected:
    BS::thread_pool&    threadPool;
    const GPUSystem&    gpuSystem;
    uint32_t            idCounter           = 0;
    Vector2ui           maxBitsUsedOnKey    = Vector2ui::Zero();
    AABB3               sceneAABB;
    //
    const AccelGroupGenMap&             accelGenerators;
    const AccelWorkGenMap&              workGenGlobalMap;
    Map<uint32_t, AccelGroupPtr>        generatedAccels;
    Map<CommonKey, AcceleratorGroupI*>  accelInstances;

    virtual AABB3       InternalConstruct(const std::vector<size_t>& instanceOffsets) = 0;
    public:
    // Constructors & Destructor
                        BaseAcceleratorT(BS::thread_pool&,
                                         const GPUSystem&,
                                         const AccelGroupGenMap&,
                                         const AccelWorkGenMap&);

    // ....
    void                Construct(BaseAccelConstructParams) override;
    AABB3               SceneAABB() const override;
    size_t              TotalAccelCount() const override;
    size_t              TotalInstanceCount() const override;
};

template<class KeyType>
struct GroupIdFetcher
{
    typename KeyType::Type operator()(auto id)
    {
        return std::bit_cast<KeyType>(id).FetchBatchPortion();
    }
};

using PrimGroupIdFetcher = GroupIdFetcher<PrimBatchKey>;
using TransGroupIdFetcher = GroupIdFetcher<TransformKey>;
using LightGroupIdFetcher = GroupIdFetcher<LightKey>;

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
AccelPartitionResult AcceleratorGroupT<C, PG, B>::PartitionParamsForWork(const AccelGroupConstructParams& p)
{
    using IndexPack = typename AccelPartitionResult::IndexPack;
    AccelPartitionResult result = {};
    result.workPartitionOffsets.reserve(p.tGroupSurfs.size() + p.tGroupLightSurfs.size() + 1);
    result.packedIndices.reserve(p.tGroupSurfs.size() + p.tGroupLightSurfs.size());

    // This is somewhat iota but we will add lights
    for(uint32_t tIndex = 0; tIndex < p.tGroupSurfs.size(); tIndex++)
    {
        const auto& groupedSurf = p.tGroupSurfs[tIndex];
        // Add this directly
        result.packedIndices.emplace_back(IndexPack
        {
            .tId = groupedSurf.first,
            .tSurfIndex = tIndex,
            .tLightSurfIndex = std::numeric_limits<uint32_t>::max()
        });
    }

    // Add the lights as well
    for(uint32_t ltIndex = 0; ltIndex < p.tGroupLightSurfs.size(); ltIndex++)
    {
        const auto& groupedLightSurf = p.tGroupLightSurfs[ltIndex];
        TransGroupId tId = groupedLightSurf.first;
        auto loc = std::find_if(p.tGroupSurfs.cbegin(),
                                p.tGroupSurfs.cend(),
                                [tId](const auto& groupedSurf)
        {
            return groupedSurf.first == tId;
        });

        if(loc != p.tGroupSurfs.cend())
        {
            size_t index = std::distance(loc, p.tGroupSurfs.begin());
            result.packedIndices[index].tLightSurfIndex = ltIndex;
        }
        else
        {
            result.packedIndices.emplace_back(IndexPack
            {
                .tId = groupedLightSurf.first,
                .tSurfIndex = std::numeric_limits<uint32_t>::max(),
                .tLightSurfIndex = ltIndex
            });
        }
    }

    // Find the partitioned offsets
    result.workPartitionOffsets.resize(result.packedIndices.size() + 1, 0);
    std::transform_inclusive_scan(result.packedIndices.cbegin(),
                                  result.packedIndices.cend(),
                                  result.workPartitionOffsets.begin() + 1,
                                  std::plus{},
    [&p](const IndexPack& indexPack) -> uint32_t
    {
        uint32_t result = 0;
        if(indexPack.tSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            size_t size = p.tGroupSurfs[indexPack.tSurfIndex].second.size();
            result += static_cast<uint32_t>(size);
        }
        if(indexPack.tLightSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            size_t size = p.tGroupLightSurfs[indexPack.tLightSurfIndex].second.size();
            result += static_cast<uint32_t>(size);
        };
        return result;
    });
    assert(result.packedIndices.size() + 1 == result.workPartitionOffsets.size());
    result.totalInstanceCount = result.workPartitionOffsets.back();

    return result;

};

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
AccelLeafResult AcceleratorGroupT<C, PG, B>::DetermineConcreteAccelCount(std::vector<SurfacePrimList> instancePrimBatches,
                                                                         const std::vector<PrimRangeArray>& instancePrimRanges)
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
        auto endLoc = std::unique_copy(nonUniqueIndices.begin(), nonUniqueIndices.end(), uniqueIndices.begin(),
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
        uniqueIndices.erase(endLoc, uniqueIndices.end());

        // Do an inverted search your id on unique indices
        std::iota(nonUniqueIndices.begin(), nonUniqueIndices.end(), 0);
        for(uint32_t& index : nonUniqueIndices)
        {
            auto loc = std::lower_bound(uniqueIndices.begin(), uniqueIndices.end(), index,
                                        [&instancePrimBatches](uint32_t value, uint32_t checked)
            {
                const SurfacePrimList& lhs = instancePrimBatches[value];
                const SurfacePrimList& rhs = instancePrimBatches[checked];
                auto result = std::lexicographical_compare_three_way(lhs.cbegin(), lhs.cend(),
                                                                     rhs.cbegin(), rhs.cend(),
                                                                     [](PrimBatchId lhs, PrimBatchId rhs)
                {
                    return (static_cast<uint32_t>(lhs) <=> static_cast<uint32_t>(rhs));
                });
                return std::is_lt(result);

            });
            index = static_cast<uint32_t>(std::distance(uniqueIndices.begin(), loc));
        }

    }
    else
    {
        // PG supports "PER_PRIMITIVE_TRANSFORM", we cannot refer to the same
        // accelerator, we need to construct an accelerator for each instance
        // Do not bother sorting etc here..
        // Copy here for continuation
        uniqueIndices = acceleratorIndices;
    }

    // Determine the leaf ranges
    // TODO: We are
    std::vector<uint32_t> uniquePrimCounts;
    uniquePrimCounts.reserve(uniqueIndices.size() + 1);
    uniquePrimCounts.push_back(0);
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
    std::inclusive_scan(uniquePrimCounts.cbegin() + 1, uniquePrimCounts.cend(),
                        uniquePrimCounts.begin() + 1);

    // Rename the variable for better maintenance
    std::vector<uint32_t>& uniquePrimOffsets = uniquePrimCounts;
    // Acquire instance leaf ranges
    std::vector<Vector2ui> instanceLeafRangeList;
    instanceLeafRangeList.reserve(instancePrimBatches.size());
    for(uint32_t index : acceleratorIndices)
    {
        instanceLeafRangeList.push_back(Vector2ui(uniquePrimOffsets[index],
                                                  uniquePrimOffsets[index + 1]));
    }

    // Find leaf of unique indices
    std::vector<Vector2ui> concreteLeafRangeList;
    std::vector<PrimRangeArray> concretePrimRangeList;
    concreteLeafRangeList.reserve(uniqueIndices.size());
    concretePrimRangeList.reserve(uniqueIndices.size());
    for(size_t index = 0; index < uniqueIndices.size(); index++)
    {
        concreteLeafRangeList.push_back(Vector2ui(uniquePrimOffsets[index],
                                                  uniquePrimOffsets[index + 1]));
        uint32_t lookupIndex = uniqueIndices[index];
        concretePrimRangeList.push_back(instancePrimRanges[lookupIndex]);
    }

    // All done!
    return AccelLeafResult
    {
        .concreteIndicesOfInstances = std::move(acceleratorIndices),
        .instanceLeafRanges = std::move(instanceLeafRangeList),
        .concreteLeafRanges = std::move(concreteLeafRangeList),
        .concretePrimRanges = std::move(concretePrimRangeList)
    };
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
template<class T>
std::vector<Span<T>>
AcceleratorGroupT<C, PG, B>::CreateInstanceSubspans(Span<T> fullRange,
                                                    const std::vector<Vector2ui>& rangeVector)
{
    std::vector<Span<const T>> instanceLeafData(rangeVector.size());
    std::transform(rangeVector.cbegin(), rangeVector.cend(),
                   instanceLeafData.begin(),
                   [fullRange](const Vector2ui& offset)
    {
        uint32_t size = offset[1] - offset[0];
        return ToConstSpan(fullRange.subspan(offset[0], size));
    });
    return instanceLeafData;
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
LinearizedSurfaceData AcceleratorGroupT<C, PG, B>::LinearizeSurfaceData(const AccelGroupConstructParams& p,
                                                                        const AccelPartitionResult& partitions,
                                                                        const typename PG& pg)
{
    LinearizedSurfaceData result = {};
    result.primRanges.reserve(partitions.totalInstanceCount);
    result.lightOrMatKeys.reserve(partitions.totalInstanceCount);
    result.alphaMaps.reserve(partitions.totalInstanceCount);
    result.cullFaceFlags.reserve(partitions.totalInstanceCount);
    result.transformKeys.reserve(partitions.totalInstanceCount);
    result.instancePrimBatches.reserve(partitions.totalInstanceCount);

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

    const auto LoadSurf = [&](const SurfaceParams& surf)
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
                    throw MRayError("{:s}: Alpha map texture({:d}) is not found",
                                    C::TypeName(),
                                    static_cast<uint32_t>(surf.alphaMaps[i].value()));
                }
                const GenericTextureView& view = optView.value();
                assert(std::holds_alternative<AlphaMap>(view));
                result.alphaMaps.back()[i] = std::get<AlphaMap>(view);
            }
            else result.alphaMaps.back()[i] = std::nullopt;

            result.cullFaceFlags.back()[i] = surf.cullFaceFlags[i];
            PrimBatchKey pBatchKey = PrimBatchKey(static_cast<uint32_t>(surf.primBatches[i]));
            result.primRanges.back()[i] = pg.BatchRange(pBatchKey);
            MaterialKey mKey(static_cast<CommonKey>(surf.materials[i]));
            result.lightOrMatKeys.back()[i] = LightOrMatKey::CombinedKey(IS_MAT_KEY_FLAG,
                                                                         mKey.FetchBatchPortion(),
                                                                         mKey.FetchIndexPortion());
        }
        InitRest(static_cast<uint32_t>(surf.alphaMaps.size()));
    };

    const auto LoadLightSurf = [&](const LightSurfaceParams& lSurf)
    {
        result.alphaMaps.emplace_back();
        result.cullFaceFlags.emplace_back();
        result.lightOrMatKeys.emplace_back();
        result.primRanges.emplace_back();
        result.instancePrimBatches.emplace_back();
        result.transformKeys.emplace_back();
        InitRest(0);

        LightKey lKey = LightKey(static_cast<uint32_t>(lSurf.lightId));
        PrimBatchKey primBatchKey = p.lightGroup->LightPrimBatch(lKey);
        result.primRanges.back().front() = pg.BatchRange(primBatchKey);
        result.lightOrMatKeys.back().front() = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG,
                                                                          lKey.FetchBatchPortion(),
                                                                          lKey.FetchIndexPortion());

        PrimBatchId primBatchId = PrimBatchId(static_cast<uint32_t>(primBatchKey));
        result.transformKeys.back() = TransformKey(static_cast<uint32_t>(lSurf.transformId));
        result.instancePrimBatches.back().push_back(primBatchId);
    };

    for(const auto& pIndices : partitions.packedIndices)
    {
        if(pIndices.tSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            for(const auto& [_, surf] : p.tGroupSurfs[pIndices.tSurfIndex].second)
            {
                LoadSurf(surf);
            }
        }
        //
        if(pIndices.tLightSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            for(const auto& [_, lSurf] : p.tGroupLightSurfs[pIndices.tLightSurfIndex].second)
            {
                LoadLightSurf(lSurf);
            }
        }
    }


    assert(result.alphaMaps.size() == partitions.totalInstanceCount);
    assert(result.cullFaceFlags.size() == partitions.totalInstanceCount);
    assert(result.lightOrMatKeys.size() == partitions.totalInstanceCount);
    assert(result.primRanges.size() == partitions.totalInstanceCount);
    assert(result.transformKeys.size() == partitions.totalInstanceCount);
    assert(result.instancePrimBatches.size() == partitions.totalInstanceCount);
    return result;
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
PreprocessResult AcceleratorGroupT<C, PG, B>::PreprocessConstructionParams(const AccelGroupConstructParams& p)
{
    assert(&pg == p.primGroup);
    // Instance Types (determined by transform type)
    AccelPartitionResult partitions = PartitionParamsForWork(p);
    // Total instance count (equavilently total surface count)
    auto linSurfData = LinearizeSurfaceData(p, partitions, pg);
    // Find out the concrete accel count and offsets
    auto leafResult = DetermineConcreteAccelCount(std::move(linSurfData.instancePrimBatches),
                                                  linSurfData.primRanges);

    concreteIndicesOfInstances = std::move(leafResult.concreteIndicesOfInstances);
    instanceLeafRanges = std::move(leafResult.instanceLeafRanges);
    concreteLeafRanges = std::move(leafResult.concreteLeafRanges);

    // Instantiate Works
    workInstanceOffsets = std::move(partitions.workPartitionOffsets);
    uint32_t i = 0;
    for(const auto& indices : partitions.packedIndices)
    {
        auto tGroupOpt = p.transformGroups->at(indices.tId);
        if(!tGroupOpt)
        {
            throw MRayError("{:s}:{:d}: Unable to find transform {:d}",
                            C::TypeName(), accelGroupId,
                            static_cast<uint32_t>(indices.tId));
        }
        const GenericGroupTransformT& tGroup = *tGroupOpt.value().get().get();

        using namespace TypeNameGen::CompTime;
        std::string workTypeName = AccelWorkTypeName(C::TypeName(),
                                                     tGroup.Name());
        auto workGenOpt = accelWorkGenerators.at(workTypeName);
        if(!workGenOpt)
        {
            throw MRayError("{:s}:{:d}: Unable to find generator for work \"{:s}\"",
                            C::TypeName(),
                            accelGroupId, workTypeName);
        }
        const auto& workGen = workGenOpt.value().get();
        workInstances.try_emplace(i, workGen(*this, tGroup));
        i++;
    }
    return PreprocessResult
    {
        .surfData = std::move(linSurfData),
        .concretePrimRanges = std::move(leafResult.concretePrimRanges)
    };
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
AcceleratorGroupT<C, PG, B>::AcceleratorGroupT(uint32_t groupId,
                                               BS::thread_pool& tp, const GPUSystem& sys,
                                               const GenericGroupPrimitiveT& pgIn,
                                               const AccelWorkGenMap& workGenMap)
    : threadPool(tp)
    , gpuSystem(sys)
    , accelGroupId(groupId)
    , pg(static_cast<const PG&>(pgIn))
    , accelWorkGenerators(workGenMap)
{}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
void AcceleratorGroupT<C, PG, B>::WriteInstanceKeysAndAABBsInternal(Span<AABB3> aabbWriteRegion,
                                                                    Span<AcceleratorKey> keyWriteRegion,
                                                                    // Input
                                                                    Span<const PrimitiveKey> dAllLeafs,
                                                                    Span<const TransformKey> dTransformKeys,
                                                                    // Constants
                                                                    const GPUQueue& queue) const
{
    // Sanity Checks
    assert(aabbWriteRegion.size() == concreteIndicesOfInstances.size());
    assert(keyWriteRegion.size() == concreteIndicesOfInstances.size());

    size_t totalInstanceCount = concreteIndicesOfInstances.size();
    size_t concreteAccelCount = concreteLeafRanges.size();

    // We will use a temp memory here
    // TODO: Add stream ordered memory allocator stuff to the
    // Device abstraction side maybe?
    DeviceLocalMemory tempMem(*queue.Device());

    using enum PrimTransformType;
    if constexpr(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
    {
        using namespace DeviceAlgorithms;
        size_t tmSize = SegmentedTransformReduceTMSize<AABB3, PrimitiveKey>(concreteLeafRanges.size());
        Span<uint32_t> dConcreteIndicesOfInstances;
        Span<AABB3> dConcreteAABBs;
        Span<uint32_t> dConcreteLeafOffsets;
        Span<Byte> dTransformSegReduceTM;
        MemAlloc::AllocateMultiData(std::tie(dConcreteIndicesOfInstances,
                                             dConcreteAABBs,
                                             dConcreteLeafOffsets,
                                             dTransformSegReduceTM),
                                    tempMem,
                                    {totalInstanceCount, concreteAccelCount,
                                     concreteAccelCount + 1,
                                     tmSize});
        Span<const uint32_t> hConcreteIndicesOfInstances(concreteIndicesOfInstances.data(),
                                                         concreteIndicesOfInstances.size());
        Span<const Vector2ui> hConcreteLeafRanges(concreteLeafRanges.data(),
                                                  concreteLeafRanges.size());
        // Normal copy to GPU
        queue.MemcpyAsync(dConcreteIndicesOfInstances, hConcreteIndicesOfInstances);

        // We need to copy the Vector2ui [(0, n_0), [n_0, n_1), ..., [n_{m-1}, n_m)]
        // As [n_0, n_1, ..., n_{m-1}, n_m]
        // This is technically UB maybe?
        // But it is hard to recognize by the compiler maybe? Dunno
        // Do a sanity check at least...
        static_assert(sizeof(Vector2ui) == 2 * sizeof(typename Vector2ui::InnerType));
        Span<const uint32_t> hConcreteLeafRangesInt(hConcreteLeafRanges.data()->AsArray().data(),
                                                    hConcreteLeafRanges.size() * Vector2ui::Dims);

        // Memset the first element to zero
        queue.MemsetAsync(dConcreteLeafOffsets.subspan(0, 1), 0x00);
        queue.MemcpyAsyncStrided(dConcreteLeafOffsets.subspan(1), 0,
                                 hConcreteLeafRangesInt.subspan(1), sizeof(Vector2ui));

        SegmentedTransformReduce<AABB3, PrimitiveKey>
        (
            dConcreteAABBs,
            dTransformSegReduceTM,
            ToConstSpan(dAllLeafs),
            dConcreteLeafOffsets,
            AABB3::Negative(),
            queue,
            UnionAABB3Functor(),
            AABBFetchFunctor<PG>(pg.SoA())
        );
        // Now, copy (and transform) concreteAABBs (which are on local space)
        // to actual accelerator instance aabb's (after transform these will be
        // in world space)
        for(const auto& kv : workInstances)
        {
            uint32_t index = kv.first;
            const AccelWorkPtr& workPtr = kv.second;
            size_t size = (workInstanceOffsets[index + 1] -
                           workInstanceOffsets[index]);
            Span<const uint32_t> dLocalIndices = dConcreteIndicesOfInstances.subspan(workInstanceOffsets[index], size);
            Span<const TransformKey> dLocalTKeys = dTransformKeys.subspan(workInstanceOffsets[index], size);
            Span<AABB3> dLocalAABBWriteRegion = aabbWriteRegion.subspan(workInstanceOffsets[index], size);

            workPtr->TransformLocallyConstantAABBs(dLocalAABBWriteRegion,
                                                   dConcreteAABBs,
                                                   dLocalIndices,
                                                   dLocalTKeys,
                                                   queue);
        }

        // TODO: This is actually common part, but compiler gives unreachable code error
        // due to below part is not yet implemented. So it is moved here untill that portion is
        // implemented.
        //
        // Now, copy (and transform) concreteAABBs (which are on local space)
        // to actual accelerator instance aabb's (after transform these will be
        // in world space)
        for(const auto& kv : workInstances)
        {
            uint32_t index = kv.first;
            size_t wIOffset = workInstanceOffsets[index];
            size_t size = (workInstanceOffsets[index + 1] - wIOffset);
            // Copy the keys as well
            Span<AcceleratorKey> dLocalKeyWriteRegion = keyWriteRegion.subspan(wIOffset, size);
            uint32_t accelBatchId = globalWorkIdToLocalOffset + index;
            using namespace std::string_literals;
            static const auto KernelName = "KCCopyLocalAccelKeys-"s + std::string(C::TypeName());

            queue.IssueSaturatingLambda
            (
                KernelName,
                KernelIssueParams{.workCount = static_cast<uint32_t>(size)},
                KeyGeneratorFunctor(accelBatchId, static_cast<uint32_t>(wIOffset),
                                    dLocalKeyWriteRegion)
            );
        }
        //  Don't forget to wait for temp memory!
        queue.Barrier().Wait();
    }
    else
    {
        throw MRayError("{}: PER_PRIM_TRANSFORM Accel Construct not yet implemented",
                        C::TypeName());
    }
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
void AcceleratorGroupT<C, PG, B>::PreConstruct(const BaseAcceleratorI*)
{}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
size_t AcceleratorGroupT<C, PG, B>::InstanceCount() const
{
    return instanceLeafRanges.size();
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
uint32_t AcceleratorGroupT<C, PG, B>::InstanceTypeCount() const
{
    return static_cast<uint32_t>(workInstances.size());
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
uint32_t AcceleratorGroupT<C, PG, B>::AcceleratorCount() const
{
    return static_cast<uint32_t>(concreteLeafRanges.size());
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
uint32_t AcceleratorGroupT<C, PG, B>::UsedIdBitsInKey() const
{
    using namespace Bit;
    size_t bitCount = RequiredBitsToRepresent(InstanceCount());
    return static_cast<uint32_t>(bitCount);
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
void AcceleratorGroupT<C, PG, B>::SetKeyOffset(uint32_t offset)
{
    globalWorkIdToLocalOffset = offset;
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
std::string_view AcceleratorGroupT<C, PG, B>::Name() const
{
    return C::TypeName();
}

template<class C, PrimitiveGroupC PG, std::derived_from<AcceleratorGroupI> B>
const GenericGroupPrimitiveT& AcceleratorGroupT<C, PG, B>::PrimGroup() const
{
    return pg;
}

template <class C>
BaseAcceleratorT<C>::BaseAcceleratorT(BS::thread_pool& tp,
                                      const GPUSystem& system,
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
    assert(std::is_sorted(surfList.begin(), surfList.end(), SurfaceLessThan));

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
            return value < batchPortion;
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
            auto innerEnd = std::upper_bound(innerStart, end, tGroupId,
            [](uint32_t value, const SurfParam& surf)
            {
                auto tId = surf.second.transformId;
                return value < TransGroupIdFetcher()(tId);
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
                                                       const Map<LightGroupId, LightGroupPtr>& lightGroups,
                                                       const Map<TransGroupId, TransformGroupPtr>& transformGroups)
{
    using LightSurfP = typename BaseAccelConstructParams::LightSurfPair;
    assert(std::is_sorted(lSurfList.begin(), lSurfList.end(), LightSurfaceLessThan));

    // Now partition
    auto start = lSurfList.begin();
    while(start != lSurfList.end())
    {
        uint32_t lGroupId = LightGroupIdFetcher()(start->second.lightId);
        auto end = std::upper_bound(start, lSurfList.end(), lGroupId,
        [](const uint32_t& value, const LightSurfP& surf)
        {
            uint32_t batchPortion = LightGroupIdFetcher()(surf.second.lightId);
            return value < batchPortion;
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
                .transformGroups = &transformGroups,
                .primGroup = pGroup,
                .lightGroup = lGroup,
                .tGroupSurfs = {},
                .tGroupLightSurfs = {}
            });
            slot = partitions.end() - 1;
        }
        else if(slot->lightGroup == nullptr) slot->lightGroup = lGroup;

        // Sub-partition wrt. transform
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            uint32_t tGroupId = TransGroupIdFetcher()(tId);
            auto innerEnd = std::upper_bound(innerStart, end, tGroupId,
            [](uint32_t value, const LightSurfP& surf) -> bool
            {
                auto tId = surf.second.transformId;
                return (value < TransGroupIdFetcher()(tId));
            });
            size_t elemCount = static_cast<size_t>(std::distance(innerStart, innerEnd));
            size_t startDistance = static_cast<size_t>(std::distance(lSurfList.begin(), innerStart));
            slot->tGroupLightSurfs.emplace_back(TransGroupId(tGroupId),
                                                lSurfList.subspan(startDistance, elemCount));
            innerStart = innerEnd;
        }
        start = end;
    }
}

template <class C>
void BaseAcceleratorT<C>::Construct(BaseAccelConstructParams p)
{
    static const auto annotation = gpuSystem.CreateAnnotation("Accelerator Construct");
    const auto _ = annotation.AnnotateScope();

    std::vector<AccelGroupConstructParams> partitions;
    PartitionSurfaces(partitions, p.mSurfList, p.primGroups,
                      p.transformGroups, p.texViewMap);
    // Add primitive-backed lights surfaces as well
    AddLightSurfacesToPartitions(partitions, p.lSurfList, p.lightGroups,
                                 p.transformGroups);

    // Generate the accelerators
    GPUQueueIteratorRoundRobin qIt(gpuSystem);
    for(auto&& partition : partitions)
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
        auto GenerateAccelGroup = accelGenerator.value().get();
        auto accelPtr = GenerateAccelGroup(std::move(aGroupId),
                                           threadPool, gpuSystem,
                                           *partition.primGroup,
                                           workGenGlobalMap);
        auto loc = generatedAccels.emplace(aGroupId, std::move(accelPtr));
        AcceleratorGroupI* acc = loc.first->second.get();
        acc->PreConstruct(this);
        acc->Construct(std::move(partition), qIt.Queue());
        qIt.Next();
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
        AcceleratorGroupI* aGroup = group.second.get();
        aGroup->SetKeyOffset(keyOffsets[i]);
        for(uint32_t key = keyOffsets[i]; key < keyOffsets[i + 1]; key++)
            accelInstances.emplace(key, aGroup);

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
    using namespace Bit;
    maxBitsUsedOnKey = Vector2ui(RequiredBitsToRepresent(keyBatchPortionMax),
                                 RequiredBitsToRepresent(keyIdPortionMax));
    // Calidate
    if(maxBitsUsedOnKey[0] > AcceleratorKey::BatchBits ||
       maxBitsUsedOnKey[1] > AcceleratorKey::IdBits)
    {
        throw MRayError("[{}]: Too many bits on accelerator [{}|{}], AcceleratorKey can hold "
                        "[{}|{}] amount of bits",
                        C::TypeName(), maxBitsUsedOnKey[0], maxBitsUsedOnKey[1],
                        AcceleratorKey::BatchBits, AcceleratorKey::IdBits);
    }

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

template <class C>
size_t BaseAcceleratorT<C>::TotalAccelCount() const
{
    size_t result = 0;
    for(const auto& kv : generatedAccels)
        result += kv.second->AcceleratorCount();

    return result;
}

template <class C>
size_t BaseAcceleratorT<C>::TotalInstanceCount() const
{
    size_t result = 0;
    for(const auto& kv : generatedAccels)
        result += kv.second->InstanceCount();

    return result;
}

// Generic PrimitiveKey copy kernel
// Extern it here, define it in a .cu file
// TODO: If I remember correctly extern __global__ was not allowed?
// Is this changed recently? Check this if it is undefined behaviour
//
// This kernel is shared by the all accelerators, but the body
// is in "AcceleratorLinear.cu" file
extern MRAY_KERNEL
void KCGeneratePrimitiveKeys(MRAY_GRID_CONSTANT const Span<PrimitiveKey> dAllLeafs,
                             //
                             MRAY_GRID_CONSTANT const Span<const PrimRangeArray> dConcretePrimRanges,
                             MRAY_GRID_CONSTANT const Span<const Vector2ui> dConcreteLeafRanges,
                             MRAY_GRID_CONSTANT const uint32_t groupId);