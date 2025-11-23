#pragma once

#include <concepts>

#include "TracerTypes.h"

#include "Core/Types.h"
#include "Core/TypeGenFunction.h"

#include "Device/GPUSystemForward.h"
#include "Device/GPUAlgForward.h"

#include "Random.h"

#include "Bitspan.h"
#include "PrimitiveC.h"
#include "LightC.h"
#include "TextureView.h"
#include "SurfaceComparators.h"
#include "AcceleratorWorkI.h"

using AccelWorkPtr = std::unique_ptr<AcceleratorWorkI>;

class TheadPool;

template<class HitType>
struct HitResultT
{
    HitType         hit;
    Float           t;
    PrimitiveKey    primitiveKey;
    LightOrMatKey   lmKey;
    VolumeIndex     volumeIndex;
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
    using SurfPair      = Pair<SurfaceId, SurfaceParams>;
    using LightSurfPair = Pair<LightSurfaceId, LightSurfaceParams>;
    using VolumeList    = std::vector<Pair<VolumeId, VolumeKeyPack>>;

    const VolumeList&                           globalVolumeList;
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
    using VolumeList            = std::vector<Pair<VolumeId, VolumeKeyPack>>;

    const Map<TransGroupId, TransformGroupPtr>* transformGroups;
    const TextureViewMap*                       textureViews;
    const VolumeList*               globalVolumeList;
    const GenericGroupPrimitiveT*   primGroup;
    const GenericGroupLightT*       lightGroup;
    TGroupedSurfaces                tGroupSurfs;
    TGroupedLightSurfaces           tGroupLightSurfs;


};

class AcceleratorGroupI
{
    public:
    virtual         ~AcceleratorGroupI() = default;

    virtual void        CastLocalRays(// Output
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
                                      CommonKey instanceId,
                                      bool resolveMedia,
                                      const GPUQueue& queue) = 0;
    virtual void        CastVisibilityRays(// Output
                                           Bitspan<uint32_t> dIsVisibleBuffer,
                                           // I-O
                                           Span<BackupRNGState> dRNGStates,
                                           // Input
                                           Span<const RayGMem> dRays,
                                           Span<const RayIndex> dRayIndices,
                                           Span<const CommonKey> dAccelKeys,
                                           // Constants
                                           CommonKey workId,
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
// is 8 in the current impl.) This should not be a memory concern until
// this codebase becomes production level renderer (doubt)
using CullFaceFlagArray     = Bitset<TracerConstants::MaxPrimBatchPerSurface>;
using AlphaMapArray         = std::array<Optional<AlphaMap>, TracerConstants::MaxPrimBatchPerSurface>;
using VolumeIndexArray      = std::array<VolumeIndex, TracerConstants::MaxPrimBatchPerSurface>;

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
    std::vector<VolumeIndexArray>    volumeIndices;
    std::vector<PrimRangeArray>      primRanges;
    std::vector<LightOrMatKeyArray>  lightOrMatKeys;
    std::vector<AlphaMapArray>       alphaMaps;
    std::vector<CullFaceFlagArray>   cullFaceFlags;
    std::vector<TransformKey>        transformKeys;
    std::vector<SurfacePrimList>     instancePrimBatches;
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
                                              ThreadPool&,
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
MR_PF_DECL
uint32_t FindPrimBatchIndex(const PrimRangeArray& primRanges, PrimitiveKey k) noexcept
{
    constexpr uint32_t N = static_cast<uint32_t>(LightOrMatKeyArray().size());

    // Linear search over the index
    // List has few elements so linear search should suffice
    CommonKey primIndex = k.FetchIndexPortion();
    MRAY_UNROLL_LOOP
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

class AcceleratorGroup : public AcceleratorGroupI
{
    private:
    // Common functionality for inheriting types
    static AccelPartitionResult     PartitionParamsForWork(const AccelGroupConstructParams& p);
    static AccelLeafResult          DetermineConcreteAccelCount(std::vector<SurfacePrimList> instancePrimBatches,
                                                                const std::vector<PrimRangeArray>& instancePrimRanges,
                                                                PrimTransformType primTransformType);
    static LinearizedSurfaceData    LinearizeSurfaceData(const AccelGroupConstructParams& p,
                                                         const AccelPartitionResult& partitions,
                                                         const GenericGroupPrimitiveT& pg,
                                                         std::string_view typeName);

    protected:
    ThreadPool&                     threadPool;
    const GPUSystem&                gpuSystem;
    const GenericGroupPrimitiveT&   pg;
    uint32_t                        accelGroupId = 0;

    // Each instance is backed with a concrete accelerator
    // This indirection represents it
    std::vector<uint32_t>       concreteIndicesOfInstances;
    // Offset list of instance/concrete leafs
    std::vector<Vector2ui>      concreteLeafRanges;
    std::vector<Vector2ui>      instanceLeafRanges;
    std::vector<uint32_t>       workInstanceOffsets;

    // Type Related
    uint32_t                        globalWorkIdToLocalOffset = std::numeric_limits<uint32_t>::max();
    const AccelWorkGenMap&          accelWorkGenerators;
    Map<CommonKey, AccelWorkPtr>    workInstances;
    //
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
                        AcceleratorGroup(uint32_t accelGroupId,
                                         ThreadPool&,
                                         const GPUSystem&,
                                         const GenericGroupPrimitiveT& pg,
                                         const AccelWorkGenMap&);

    virtual void        PreConstruct(const BaseAcceleratorI*) override;
    size_t              InstanceCount() const override;
    uint32_t            InstanceTypeCount() const override;
    uint32_t            AcceleratorCount() const override;
    uint32_t            UsedIdBitsInKey() const override;
    void                SetKeyOffset(uint32_t) override;
    //
    const GenericGroupPrimitiveT& PrimGroup() const override;
};

template <class Child, class ExtraInterface = EmptyType>
requires(!std::derived_from<ExtraInterface, AcceleratorGroupI>)
class AcceleratorGroupT : public AcceleratorGroup
                        , public ExtraInterface
{
    using AcceleratorGroup::AcceleratorGroup;

    std::string_view Name() const override { return Child::TypeName(); }
};

class BaseAcceleratorI
{
    public:
    virtual         ~BaseAcceleratorI() = default;

    // Interface
    // Fully cast rays to entire scene
    virtual void    CastRays(// Output
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
                             const GPUQueue& queue) = 0;
    // Fully cast rays to entire scene return true/false
    // If it hits to a surface, (this should be faster
    // since any hit will terminate the operation)
    virtual void    CastVisibilityRays(// Output
                                       Bitspan<uint32_t> dIsVisibleBuffer,
                                       // I-O
                                       Span<BackupRNGState> dRNGStates,
                                       // Input
                                       Span<const RayGMem> dRays,
                                       Span<const RayIndex> dRayIndices,
                                       const GPUQueue& queue) = 0;
    // Locally cast rays to a accelerator instances
    // This is multi-ray multi-accelerator instance
    virtual void    CastLocalRays(// Output
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
                                  const GPUQueue& queue) = 0;

    // Construction
    virtual void                Construct(BaseAccelConstructParams) = 0;
    virtual void                AllocateForTraversal(size_t maxRayCount) = 0;
    virtual size_t              GPUMemoryUsage() const = 0;
    virtual AABB3               SceneAABB() const = 0;
    virtual size_t              TotalAccelCount() const = 0;
    virtual size_t              TotalInstanceCount() const = 0;
    virtual std::string_view    Name() const = 0;
};

using AcceleratorPtr = std::unique_ptr<BaseAcceleratorI>;

class BaseAccelerator : public BaseAcceleratorI
{
    private:
    void PartitionSurfaces(std::vector<AccelGroupConstructParams>&,
                           const BaseAccelConstructParams& params);
    void AddLightSurfacesToPartitions(std::vector<AccelGroupConstructParams>& partitions,
                                      const BaseAccelConstructParams& params);

    protected:
    ThreadPool&         threadPool;
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
                        BaseAccelerator(ThreadPool&,
                                        const GPUSystem&,
                                        const AccelGroupGenMap&,
                                        const AccelWorkGenMap&);
    //
    void                Construct(BaseAccelConstructParams) override;
    AABB3               SceneAABB() const override;
    size_t              TotalAccelCount() const override;
    size_t              TotalInstanceCount() const override;
};

template <class Child>
class BaseAcceleratorT : public BaseAccelerator
{
    using BaseAccelerator::BaseAccelerator;

    std::string_view Name() const override { return Child::TypeName(); }
};

template<class T>
std::vector<Span<T>> AcceleratorGroup::CreateInstanceSubspans(Span<T> fullRange,
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

inline
AcceleratorGroup::AcceleratorGroup(uint32_t groupId,
                                   ThreadPool& tp, const GPUSystem& sys,
                                   const GenericGroupPrimitiveT& pgIn,
                                   const AccelWorkGenMap& workGenMap)
    : threadPool(tp)
    , gpuSystem(sys)
    , pg(pgIn)
    , accelGroupId(groupId)
    , accelWorkGenerators(workGenMap)
{}


inline void AcceleratorGroup::PreConstruct(const BaseAcceleratorI*)
{}

inline size_t AcceleratorGroup::InstanceCount() const
{
    return instanceLeafRanges.size();
}

inline uint32_t AcceleratorGroup::InstanceTypeCount() const
{
    return static_cast<uint32_t>(workInstances.size());
}

inline uint32_t AcceleratorGroup::AcceleratorCount() const
{
    return static_cast<uint32_t>(concreteLeafRanges.size());
}

inline uint32_t AcceleratorGroup::UsedIdBitsInKey() const
{
    using namespace Bit;
    size_t bitCount = RequiredBitsToRepresent(InstanceCount());
    return static_cast<uint32_t>(bitCount);
}

inline void AcceleratorGroup::SetKeyOffset(uint32_t offset)
{
    globalWorkIdToLocalOffset = offset;
}

inline const GenericGroupPrimitiveT&
AcceleratorGroup::PrimGroup() const
{
    return pg;
}

inline
BaseAccelerator::BaseAccelerator(ThreadPool& tp,
                                 const GPUSystem& system,
                                 const AccelGroupGenMap& aGen,
                                 const AccelWorkGenMap& globalWorkMap)
    : threadPool(tp)
    , gpuSystem(system)
    , accelGenerators(aGen)
    , workGenGlobalMap(globalWorkMap)
{}

inline
AABB3 BaseAccelerator::SceneAABB() const
{
    return sceneAABB;
}

inline
size_t BaseAccelerator::TotalAccelCount() const
{
    size_t result = 0;
    for(const auto& kv : generatedAccels)
        result += kv.second->AcceleratorCount();

    return result;
}

inline
size_t BaseAccelerator::TotalInstanceCount() const
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
MRAY_KERNEL
void KCGeneratePrimitiveKeys(MRAY_GRID_CONSTANT const Span<PrimitiveKey> dAllLeafs,
                             //
                             MRAY_GRID_CONSTANT const Span<const PrimRangeArray> dConcretePrimRanges,
                             MRAY_GRID_CONSTANT const Span<const Vector2ui> dConcreteLeafRanges,
                             MRAY_GRID_CONSTANT const CommonKey groupId);

MRAY_KERNEL
void KCSetIsVisibleIndirect(MRAY_GRID_CONSTANT const Bitspan<uint32_t> dIsVisibleBuffer,
                            //
                            MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices);