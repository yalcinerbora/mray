#pragma once


#include "TracerTypes.h"
#include "RenderImage.h"
#include "GenericGroup.h"
#include "TextureMemory.h"

#include "MaterialC.h"
#include "PrimitiveC.h"
#include "CameraC.h"
#include "TransformC.h"
#include "LightC.h"
#include "TracerTypes.h"
#include "AcceleratorC.h"

#include "Core/TracerI.h"
#include "Core/DataStructures.h"
#include "Core/Algorithm.h"

#include "TransientPool/TransientPool.h"

#include "Common/RenderImageStructs.h"

namespace BS { class thread_pool; }

// A nasty forward declaration
class GenericGroupPrimitiveT;
class GenericGroupCameraT;
class GenericGroupLightT;
class GenericGroupMaterialT;
template<class, class> class GenericGroupT;
template<class, class> class GenericTexturedGroupT;
using GenericGroupTransformT    = GenericGroupT<TransformKey, TransAttributeInfo>;
using GenericGroupMediumT       = GenericTexturedGroupT<MediumKey, MediumAttributeInfo>;

struct FlatSurfParams
{
    // For lookup maybe?
    SurfaceId   surfId;
    MaterialId  mId;
    TransformId tId;
    PrimBatchId pId;
    bool        operator<(const FlatSurfParams& right) const;
};

struct TracerView
{
    template<class K, class V>
    using IdPtrMap = Map<K, std::unique_ptr<V>>;

    // Base accelerator may change its state due to ray casting
    BaseAcceleratorI& baseAccelerator;

    const IdPtrMap<PrimGroupId, GenericGroupPrimitiveT>&    primGroups;
    const IdPtrMap<CameraGroupId, GenericGroupCameraT>&     camGroups;
    const IdPtrMap<MediumGroupId, GenericGroupMediumT>&     mediumGroups;
    const IdPtrMap<MatGroupId, GenericGroupMaterialT>&      matGroups;
    const IdPtrMap<TransGroupId, GenericGroupTransformT>&   transGroups;
    const IdPtrMap<LightGroupId, GenericGroupLightT>&       lightGroups;

    const TextureMap&               textures;
    const TextureViewMap&           textureViews;
    const TracerParameters&         tracerParams;
    const FilterGeneratorMap&       filterGenerators;
    const RNGGeneratorMap&          rngGenerators;

    const LightSurfaceParams&                                       boundarySurface;
    const std::vector<Pair<SurfaceId, SurfaceParams>>&              surfs;
    const std::vector<Pair<LightSurfaceId, LightSurfaceParams>>&    lightSurfs;
    const std::vector<Pair<CamSurfaceId, CameraSurfaceParams>>&     camSurfs;
    const std::vector<FlatSurfParams>&                              flattenedSurfaces;
};

class RenderWorkI
{
    public:
    virtual ~RenderWorkI() = default;

    virtual std::string_view Name() const = 0;
};

class RenderCameraWorkI
{
    public:
    virtual ~RenderCameraWorkI() = default;

    virtual std::string_view    Name() const = 0;
    virtual uint32_t            SampleRayRNCount() const = 0;
};

class RenderLightWorkI
{
    public:
    virtual ~RenderLightWorkI() = default;

    virtual std::string_view Name() const = 0;
};

using RenderImagePtr = std::unique_ptr<RenderImage>;
using RenderWorkGenerator = GeneratorFuncType<RenderWorkI,
                                              const GenericGroupMaterialT&,
                                              const GenericGroupPrimitiveT&,
                                              const GenericGroupTransformT&,
                                              const GPUSystem&>;
using RenderLightWorkGenerator = GeneratorFuncType<RenderLightWorkI,
                                                   const GenericGroupLightT&,
                                                   const GenericGroupTransformT&,
                                                   const GPUSystem&>;
using RenderCameraWorkGenerator = GeneratorFuncType<RenderCameraWorkI,
                                                    const GenericGroupCameraT&,
                                                    const GenericGroupTransformT&,
                                                    const GPUSystem&>;

// Work generator related
using RenderWorkGenMap = Map<std::string_view, RenderWorkGenerator>;
using RenderLightWorkGenMap = Map<std::string_view, RenderLightWorkGenerator>;
using RenderCamWorkGenMap = Map<std::string_view, RenderCameraWorkGenerator>;
using RenderWorkPack = Tuple<RenderWorkGenMap, RenderLightWorkGenMap, RenderCamWorkGenMap>;

template<class RendererType>
concept RendererC = requires(RendererType rt,
                             const TracerView& tv,
                             const typename RendererType::RayPayload& rPayload,
                             TransientData input,
                             BS::thread_pool& tp,
                             const GPUSystem& gpuSystem,
                             const GPUQueue& q)
{
    // Global State
    // These parameters are work agnostic.
    typename RendererType::GlobalState;
    // These parameters are work related.
    // (Not used but exposed for future use maybe?)
    typename RendererType::RayState;
    typename RendererType::RayPayload;
    typename RendererType::SpectrumConverterContext;
    typename RendererType::MetaHit;
    // Host side things
    typename RendererType::AttribInfoList;

    // CPU Side
    RendererType(RenderImagePtr{}, tv, tp, gpuSystem, RenderWorkPack{});
    {rt.AttributeInfo()
    } -> std::same_as<typename RendererType::AttribInfoList>;
    {rt.PushAttribute(uint32_t{}, std::move(input), q)
    } -> std::same_as<void>;
    {rt.StartRender(RenderImageParams{}, CamSurfaceId{},
                    Optional<CameraTransform>{}, uint32_t{}, uint32_t{})
    } ->std::same_as<RenderBufferInfo>;
    {rt.StopRender()} -> std::same_as<void>;
    {rt.DoRender()} -> std::same_as<RendererOutput>;
    {rt.GPUMemoryUsage()} -> std::same_as<size_t>;

    // Can query the type
    {rt.Name()} -> std::same_as<std::string_view>;
    {RendererType::TypeName()} -> std::same_as<std::string_view>;
};

using RenderWorkPtr = std::unique_ptr<RenderWorkI>;
using RenderCameraWorkPtr = std::unique_ptr<RenderCameraWorkI>;
using RenderLightWorkPtr = std::unique_ptr<RenderLightWorkI>;

class RendererI
{
    public:
    using AttribInfoList = RendererAttributeInfoList;

    public:
    virtual     ~RendererI() = default;

    // Interface
    virtual AttribInfoList  AttributeInfo() const = 0;
    virtual void            PushAttribute(uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& q) = 0;
    virtual RendererOptionPack  CurrentAttributes() const = 0;
    // ...
    virtual RenderBufferInfo    StartRender(const RenderImageParams&,
                                            CamSurfaceId camSurfId,
                                            Optional<CameraTransform>,
                                            uint32_t customLogicIndex0 = 0,
                                            uint32_t customLogicIndex1 = 0) = 0;
    virtual RendererOutput      DoRender() = 0;
    virtual void                StopRender() = 0;

    virtual std::string_view    Name() const = 0;
    virtual size_t              GPUMemoryUsage() const = 0;
};

using RendererPtr = std::unique_ptr<RendererI>;

// Some renderers will require multiple "work"
// per tuple/pair. Renderer itself (orchestrator)
// will not know the types but only the ids (hashes)
// of these tuples. Here we create up to N virtual functions
// DoWork_0, ... / DoBoundaryWork_0 ... for all camera
// light and material surfaces. Currently N is one
// this will be increased in future.
// Since we cannot have a templated virtual functions,
// RendererWork instances will implement these functions.
// These macros are here to reduce boilerplate since
// function argument count is quite large
#define MRAY_RENDER_DO_WORK_DECL(tag)           \
void DoWork_##tag                               \
(                                               \
    Span<RayDiff> dRayDiffsOut,                 \
    Span<RayGMem> dRaysOut,                     \
    const typename R::RayPayload& dPayloadsOut, \
    const typename R::RayState& dRayStates,     \
    Span<const RayIndex> dRayIndicesIn,         \
    Span<const RandomNumber> dRandomNumbers,    \
    Span<const RayDiff> dRayDiffsIn,            \
    Span<const RayGMem> dRaysIn,                \
    Span<const MetaHit> dHitsIn,                \
    Span<const HitKeyPack> dKeysIn,             \
    const typename R::RayPayload& dPayloadsIn,  \
    const typename R::GlobalState& globalState, \
    const GPUQueue& queue                       \
) const

#define MRAY_RENDER_DO_WORK_DEF(tag)            \
void DoWork_##tag                               \
(                                               \
    Span<RayDiff> a,                            \
    Span<RayGMem> b,                            \
    const typename R::RayPayload& c,            \
    const typename R::RayState& d,              \
    Span<const RayIndex> e,                     \
    Span<const RandomNumber> f,                 \
    Span<const RayDiff> g,                      \
    Span<const RayGMem> h,                      \
    Span<const MetaHit> i,                      \
    Span<const HitKeyPack> j,                   \
    const typename R::RayPayload& k,            \
    const typename R::GlobalState& l,           \
    const GPUQueue& m                           \
) const override                                \
{                                               \
    DoWorkInternal<tag>(a, b, c, d, e, f, g,    \
                        h, i, j, k, l, m);      \
}                                               \

#define MRAY_RENDER_DO_LIGHT_WORK_DECL(tag)     \
void DoBoundaryWork_##tag                       \
(                                               \
    const typename R::RayState& dRayStates,     \
    Span<const RayIndex> dRayIndicesIn,         \
    Span<const uint32_t> dRandomNumbers,        \
    Span<const RayDiff> dRayDiffsIn,            \
    Span<const RayGMem> dRaysIn,                \
    Span<const MetaHit> dHitsIn,                \
    Span<const HitKeyPack> dKeysIn,             \
    const typename R::RayPayload& dPayloadsIn,  \
    const typename R::GlobalState& globalState, \
    const GPUQueue& queue                       \
) const

#define MRAY_RENDER_DO_LIGHT_WORK_DEF(tag)      \
void DoBoundaryWork_##tag                       \
(                                               \
    const typename R::RayState& a,              \
    Span<const RayIndex> b,                     \
    Span<const uint32_t> c,                     \
    Span<const RayDiff> d,                      \
    Span<const RayGMem> e,                      \
    Span<const MetaHit> f,                      \
    Span<const HitKeyPack> g,                   \
    const typename R::RayPayload& h,            \
    const typename R::GlobalState& i,           \
    const GPUQueue& j                           \
) const override                                \
{                                               \
    DoBoundaryWorkInternal<tag>(a, b, c, d, e,  \
                                f, g, h, i, j); \
}

template<class R>
class RenderWorkT : public RenderWorkI
{
    public:
    virtual MRAY_RENDER_DO_WORK_DECL(0) = 0;
};

template<class R>
class RenderLightWorkT : public RenderLightWorkI
{
    public:
    virtual MRAY_RENDER_DO_LIGHT_WORK_DECL(0) = 0;
};

template<class R>
class RenderCameraWorkT : public RenderCameraWorkI
{
    // Camera may need the macro treatment
    // later, but current one generate rays routine
    // will suffice
    public:
    virtual void GenerateSubCamera(// Output
                                   Span<Byte> dCamBuffer,
                                   // Constants
                                   CameraKey camKey,
                                   Optional<CameraTransform> camTransform,
                                   Vector2ui stratumIndex,
                                   Vector2ui stratumCount,
                                   const GPUQueue& queue) const = 0;
    virtual void GenerateRays(// Output
                              const Span<RayDiff>& dRayDiffsOut,
                              const Span<RayGMem>& dRaysOut,
                              const typename R::RayPayload& dPayloadsOut,
                              const typename R::RayState& dStatesOut,
                              // Input
                              const Span<const uint32_t>& dRayIndices,
                              const Span<const uint32_t>& dRandomNums,
                              // Type erased buffer
                              Span<const Byte> dCamBuffer,
                              TransformKey transKey,
                              // Constants
                              uint64_t globalPixelIndex,
                              const Vector2ui regionCount,
                              const GPUQueue& queue) const = 0;
};

// Renderer holds its work in a linear array.
// by defition (by the key batch width) there should be small amount of groups
// so we will do a linear search (maybe in future binary search) over these
// values. The reason we do not use hash table or map is that we may move the
// multi-kernel call to the GPU side in future.
template<class R>
struct RenderWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderWorkT<R>>;
    using IdTuple = Tuple<MatGroupId, PrimGroupId, TransGroupId>;

    IdTuple     idPack;
    CommonKey   workGroupId;
    WorkPtr     workPtr;
    // For sorting
    auto operator<=>(const RenderWorkStruct& right) const;
};

template<class R>
struct RenderLightWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderLightWorkT<R>>;
    using IdPair = Pair<LightGroupId, TransGroupId>;

    IdPair      idPack;
    CommonKey   workGroupId;
    WorkPtr     workPtr;
    // For sorting
    auto operator<=>(const RenderLightWorkStruct& right) const;
};

template<class R>
struct RenderCameraWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderCameraWorkT<R>>;
    using IdPair = Pair<CameraGroupId, TransGroupId>;

    IdPair      idPack;
    CommonKey   workGroupId;
    WorkPtr     workPtr;
    // For sorting
    auto operator<=>(const RenderCameraWorkStruct& right) const;
};

template<class R>
using RenderWorkList = std::vector<RenderWorkStruct<R>>;

template<class R>
using RenderLightWorkList = std::vector<RenderLightWorkStruct<R>>;

template<class R>
using RenderCameraWorkList = std::vector<RenderCameraWorkStruct<R>>;

// Runtime hash of the current groups available on the tracer
// This will be used to partition types.
// Unlike other group keys (PrimitiveKey, TransformKey etc.)
// This has dynamic batch/id widths. This class will determine
// enough bits to uniquely represent:
//  - Mat/Prim/Transform triplets
//  - Light/Transform pairs
//  - Camera/Transform pairs
//
// For example when a scene has 1 camera surface, 6 material surface
// 3 light surface; material surfaces are partitioned to 4 groups
// (i.e. Triangle/Lambert/Identity, Triangle/Lambert/Single,
//       Triangle/Disney/Single, Sphere/Lambert/Identity)
// light surfaces partitioned to 2 groups
// (i.e. PrimLight<Triangle>/Single, Skyphere/Identity),
// Work identifiers will be in range [0,7).
//
// Then, this class will dedicate 3 bits to distinquish the partitions
// and the rest of the bits (Given 32-bit key 29 bits) will be utilized
// for data coherency. Not all of the remaning bits will be utilized.
// Similary renderer will ask each group how many actual materials/primitives
// are present and utilize that information to minimize the usage.
// Additional heuristics may apply to reduce the partitioning time.
//
// These batch identifiers will be on an array, accelerator hit result
// (Material/Light/Camera)/Transform/Primitive will be hashed and used
// to lookup this array (either linear search or binary search).
//
class RenderWorkHasher
{
    private:
    Span<uint32_t>  dWorkBatchHashes;
    Span<CommonKey> dWorkBatchIds;
    uint32_t batchBits  = 0;
    uint32_t dataBits   = 0;
    // Maximum identifier count that is used by a single
    // group.
    uint32_t maxMatOrLightIdBits    = 0;
    uint32_t maxPrimIdBits          = 0;
    uint32_t maxTransIdBits         = 0;

    protected:
    public:
                RenderWorkHasher() = default;
    MRAY_HOST   RenderWorkHasher(Span<uint32_t> dWorkBatchHashes,
                                 Span<CommonKey> dBatchIds);

    template<class R>
    MRAY_HOST
    void PopulateHashesAndKeys(const TracerView& tracerView,
                               const RenderWorkList<R>& curWorks,
                               const RenderLightWorkList<R>& curLightWorks,
                               const RenderCameraWorkList<R>& curCamWorks,
                               const GPUQueue& queue);

    MRAY_HYBRID
    Vector2ui WorkBatchDataRange() const;
    MRAY_HYBRID
    Vector2ui WorkBatchBitRange() const;
    MRAY_HYBRID
    uint32_t BisectBatchPortion(CommonKey key) const;
    MRAY_HYBRID
    uint32_t HashWorkBatchPortion(HitKeyPack p) const;
    MRAY_HYBRID
    uint32_t HashWorkDataPortion(HitKeyPack p) const;
    MRAY_HYBRID
    CommonKey GenerateWorkKeyGPU(HitKeyPack p) const;
};

template <class Child>
class RendererT : public RendererI
{
    public:
    using AttribInfoList    = typename RendererI::AttribInfoList;
    using MetaHit           = MetaHitT<TracerConstants::MaxHitFloatCount>;
    using WorkList          = RenderWorkList<Child>;
    using LightWorkList     = RenderLightWorkList<Child>;
    using CameraWorkList    = RenderCameraWorkList<Child>;

    static constexpr size_t SUB_CAMERA_BUFFER_SIZE = 1024;

    private:
    uint32_t                workCounter = 0;
    RenderWorkPack          workPack;

    uint32_t                GenerateWorkMappings(uint32_t workIdStart);
    uint32_t                GenerateLightWorkMappings(uint32_t workIdStart);
    uint32_t                GenerateCameraWorkMappings(uint32_t workIdStart);

    protected:
    BS::thread_pool&        globalThreadPool;
    const GPUSystem&        gpuSystem;
    TracerView              tracerView;
    const RenderImagePtr&   renderBuffer;

    WorkList                currentWorks;
    LightWorkList           currentLightWorks;
    CameraWorkList          currentCameraWorks;
    HitKeyPack              boundaryMatKeyPack;
    // Current Canvas info
    MRayColorSpaceEnum      curColorSpace;
    ImageTiler              imageTiler;
    uint64_t                totalIterationCount;

    uint32_t                GenerateWorks();
    RenderWorkHasher        InitializeHashes(Span<uint32_t> dHashes,
                                             Span<CommonKey> dWorkIds,
                                             const GPUQueue& queue);

    public:
                        RendererT(const RenderImagePtr&,
                                  const RenderWorkPack& workPacks,
                                  TracerView, const GPUSystem&,
                                  BS::thread_pool&);
    std::string_view    Name() const override;
};

inline bool FlatSurfParams::operator<(const FlatSurfParams& right) const
{
    auto GetMG = [](MaterialId id) -> CommonKey
    {
        return MaterialKey(static_cast<uint32_t>(id)).FetchBatchPortion();
    };
    auto GetPG = [](PrimBatchId id) -> CommonKey
    {
        return PrimBatchKey(static_cast<uint32_t>(id)).FetchBatchPortion();
    };
    auto GetTG = [](TransformId id) -> CommonKey
    {
        return TransformKey(static_cast<uint32_t>(id)).FetchBatchPortion();
    };

    return (Tuple(GetMG(mId), GetTG(tId), GetPG(pId)) <
            Tuple(GetMG(right.mId), GetTG(right.tId), GetPG(right.pId)));
}

template<class R>
auto RenderWorkStruct<R>::operator<=>(const RenderWorkStruct& right) const
{
    return workGroupId <=> right.workGroupId;
}

template<class R>
auto RenderLightWorkStruct<R>::operator<=>(const RenderLightWorkStruct& right) const
{
    return workGroupId <=> right.workGroupId;
}

template<class R>
auto RenderCameraWorkStruct<R>::operator<=>(const RenderCameraWorkStruct& right) const
{
    return workGroupId <=> right.workGroupId;
}

MRAY_HOST inline
RenderWorkHasher::RenderWorkHasher(Span<uint32_t> dWBHashes,
                                   Span<CommonKey> dWBIds)
    : dWorkBatchHashes(dWBHashes)
    , dWorkBatchIds(dWBIds)
    , batchBits(Bit::RequiredBitsToRepresent(static_cast<uint32_t>(dWBHashes.size())))
    , dataBits(sizeof(CommonKey) * CHAR_BIT - batchBits)
    , maxMatOrLightIdBits(0)
    , maxPrimIdBits(0)
    , maxTransIdBits(0)
{
    assert(dWBHashes.size() == dWBIds.size());
}

template<class R>
MRAY_HOST inline
void RenderWorkHasher::PopulateHashesAndKeys(const TracerView& tracerView,
                                             const RenderWorkList<R>& curWorks,
                                             const RenderLightWorkList<R>& curLightWorks,
                                             const RenderCameraWorkList<R>& curCamWorks,
                                             const GPUQueue& queue)
{
    size_t totalWorkBatchCount = (curWorks.size() + curLightWorks.size() +
                                  curCamWorks.size());
    std::vector<uint32_t> hHashes;
    std::vector<uint32_t> hBatchIds;
    hHashes.reserve(totalWorkBatchCount);
    hBatchIds.reserve(totalWorkBatchCount);
    uint32_t primMaxCount = 0;
    uint32_t lmMaxCount = 0;
    uint32_t transMaxCount = 0;

    for(const auto& work : curWorks)
    {
        MatGroupId matGroupId = std::get<0>(work.idPack);
        PrimGroupId primGroupId = std::get<1>(work.idPack);
        TransGroupId transGroupId = std::get<2>(work.idPack);

        auto pK = PrimitiveKey::CombinedKey(static_cast<CommonKey>(primGroupId), 0u);
        auto mK = LightOrMatKey::CombinedKey(IS_MAT_KEY_FLAG, static_cast<CommonKey>(matGroupId), 0u);
        auto tK = TransformKey::CombinedKey(static_cast<CommonKey>(transGroupId), 0u);
        HitKeyPack kp =
        {
            .primKey = pK,
            .lightOrMatKey = mK,
            .transKey = tK,
            .accelKey = AcceleratorKey::InvalidKey()
        };
        hHashes.emplace_back(HashWorkBatchPortion(kp));
        hBatchIds.push_back(work.workGroupId);

        // Might aswell check the data amount here
        uint32_t primCount = uint32_t(tracerView.primGroups.at(primGroupId)->get()->TotalPrimCount());
        uint32_t matCount = uint32_t(tracerView.matGroups.at(matGroupId)->get()->TotalItemCount());
        uint32_t transformCount = uint32_t(tracerView.transGroups.at(transGroupId)->get()->TotalItemCount());
        primMaxCount = std::max(primMaxCount, primCount);
        lmMaxCount = std::max(lmMaxCount, matCount);
        transMaxCount = std::max(transMaxCount, transformCount);
    }
    // Push light hashes
    for(const auto& work : curLightWorks)
    {
        LightGroupId lightGroupId = std::get<0>(work.idPack);
        TransGroupId transGroupId = std::get<1>(work.idPack);
        const auto& lightGroup = tracerView.lightGroups.at(lightGroupId)->get();
        const auto& transformGroup = tracerView.transGroups.at(transGroupId)->get();

        auto lK = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG, static_cast<CommonKey>(lightGroupId), 0u);
        auto tK = TransformKey::CombinedKey(static_cast<CommonKey>(transGroupId), 0u);

        HitKeyPack kp =
        {
            .primKey = PrimitiveKey::InvalidKey(),
            .lightOrMatKey = lK,
            .transKey = tK,
            .accelKey = AcceleratorKey::InvalidKey()
        };
        hHashes.emplace_back(HashWorkBatchPortion(kp));
        hBatchIds.push_back(work.workGroupId);

        uint32_t lightCount = uint32_t(lightGroup->TotalItemCount());
        uint32_t transformCount = uint32_t(transformGroup->TotalItemCount());
        lmMaxCount = std::max(lmMaxCount, lightCount);
        transMaxCount = std::max(transMaxCount, transformCount);
    }
    // TODO: Add camera hashes as well, it may require some redesign
    // Currently we "FF..F" these since we do not support light tracing yet
    for(const auto& work : curCamWorks)
    {
        hHashes.emplace_back(std::numeric_limits<uint32_t>::max());
        hBatchIds.push_back(work.workGroupId);
    }

    // Find bit count
    maxMatOrLightIdBits = Bit::RequiredBitsToRepresent(lmMaxCount);
    maxPrimIdBits = Bit::RequiredBitsToRepresent(primMaxCount);
    maxTransIdBits = Bit::RequiredBitsToRepresent(transMaxCount);

    queue.MemcpyAsync(dWorkBatchHashes, Span<const uint32_t>(hHashes));
    queue.MemcpyAsync(dWorkBatchIds, Span<const CommonKey>(hBatchIds));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2ui RenderWorkHasher::WorkBatchDataRange() const
{
    return Vector2ui(0, dataBits);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2ui RenderWorkHasher::WorkBatchBitRange() const
{
    return Vector2ui(dataBits, dataBits + batchBits);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t RenderWorkHasher::BisectBatchPortion(CommonKey key) const
{
    return Bit::FetchSubPortion(key, {dataBits, dataBits + batchBits});
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t RenderWorkHasher::HashWorkBatchPortion(HitKeyPack p) const
{
    static_assert(PrimitiveKey::BatchBits + LightOrMatKey::BatchBits +
                  LightOrMatKey::FlagBits +
                  TransformKey::BatchBits <= sizeof(uint32_t) * CHAR_BIT,
                  "Unable to pack batch bits for hasing!");
    // In common case, compose the batch identifiers
    bool isLight = (p.lightOrMatKey.FetchFlagPortion() == IS_LIGHT_KEY_FLAG);
    uint32_t isLightInt = (isLight) ? 1 : 0;
    uint32_t r = Bit::Compose<TransformKey::BatchBits, PrimitiveKey::BatchBits,
                              LightOrMatKey::BatchBits, LightOrMatKey::FlagBits>
    (
        p.transKey.FetchBatchPortion(),
        p.primKey.FetchBatchPortion(),
        p.lightOrMatKey.FetchBatchPortion(),
        isLightInt
    );
    return r;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t RenderWorkHasher::HashWorkDataPortion(HitKeyPack p) const
{
    // Get the Id portions
    uint32_t mlIndex = p.lightOrMatKey.FetchIndexPortion();
    uint32_t pIndex = p.primKey.FetchIndexPortion();
    uint32_t tIndex = p.transKey.FetchIndexPortion();

    // Heuristic: Most important bits are the material/light,
    // then primitive, then transform.
    //
    // This is debateable, hence it is a heuristic.
    // For example skinned meshes
    // will transform cost may be higher, a basic lambert
    // material does not have much data to be fetched from memory,
    // sometimes even UV coherency may be more important (when
    // there are many texture access due to complex material).
    //
    // So this will probably be optimized by experimentation.
    // We will see. (TODO: ...)
    uint32_t remainingBits = dataBits;
    uint32_t currentBit = 0;
    uint32_t result = 0;
    // Materials
    uint32_t bitsForMat = std::min(maxMatOrLightIdBits, remainingBits);
    result = Bit::SetSubPortion(result, mlIndex,
                                {currentBit, currentBit + bitsForMat});
    remainingBits -= bitsForMat;
    currentBit += bitsForMat;
    // Primitives
    uint32_t bitsForPrim = std::min(maxPrimIdBits, remainingBits);
    if(bitsForPrim == 0) return result;
    result = Bit::SetSubPortion(result, pIndex,
                                {currentBit, currentBit + bitsForPrim});
    remainingBits -= bitsForPrim;
    currentBit += bitsForPrim;
    // Transforms
    uint32_t bitsForTrans = std::min(maxTransIdBits, remainingBits);
    if(bitsForTrans == 0) return result;
    result = Bit::SetSubPortion(result, tIndex,
                                {currentBit, currentBit + bitsForTrans});
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
CommonKey RenderWorkHasher::GenerateWorkKeyGPU(HitKeyPack p) const
{
    CommonKey batchHash = HashWorkBatchPortion(p);
    // Find the batch portion (Linear search)
    uint32_t i = 0;
    CommonKey batchId = std::numeric_limits<CommonKey>::max();
    for(uint32_t checkHash : dWorkBatchHashes)
    {
        if(checkHash == batchHash)
        {
            batchId = dWorkBatchIds[i];
            break;
        }
        i++;
    }
    // Compose the sort key
    CommonKey hashLower = HashWorkDataPortion(p);
    CommonKey result = Bit::SetSubPortion(hashLower, batchId,
                                          {dataBits, dataBits + batchBits});
    return result;
}

template <class C>
uint32_t RendererT<C>::GenerateWorkMappings(uint32_t workStart)
{
    using Algo::PartitionRange;
    const auto& flatSurfs = tracerView.flattenedSurfaces;
    assert(std::is_sorted(tracerView.flattenedSurfaces.cbegin(),
                          tracerView.flattenedSurfaces.cend()));
    auto partitions = Algo::PartitionRange(flatSurfs.cbegin(),
                                           flatSurfs.cend());
    for(const auto& p : partitions)
    {
        size_t i = p[0];
        MatGroupId mgId{MaterialKey(uint32_t(flatSurfs[i].mId)).FetchBatchPortion()};
        PrimGroupId pgId{PrimBatchKey(uint32_t(flatSurfs[i].pId)).FetchBatchPortion()};
        TransGroupId tgId{TransformKey(uint32_t(flatSurfs[i].tId)).FetchBatchPortion()};
        // These should be checked beforehand, while actually creating
        // the surface
        const MaterialGroupPtr& mg = tracerView.matGroups.at(mgId).value();
        const PrimGroupPtr& pg = tracerView.primGroups.at(pgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view mgName = mg->Name();
        std::string_view pgName = pg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderWorkType;
        std::string workName = CreateRenderWorkType(mgName, pgName, tgName);

        auto loc = std::get<0>(workPack).at(workName);
        if(!loc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Mat/Prim/Transform "
                            "triplet of \"{}/{}/{}\"",
                            C::TypeName(), mgName, pgName, tgName);
        }
        RenderWorkGenerator generator = loc->get();
        RenderWorkPtr ptr = generator(*mg.get(), *pg.get(), *tg.get(), gpuSystem);

        using Ptr = typename RenderWorkStruct<C>::WorkPtr;
        Ptr renderTypedPtr = Ptr(static_cast<RenderWorkT<C>*>(ptr.release()));
        // Put this ptr somewhere... safe
        currentWorks.emplace_back
        (
            RenderWorkStruct
            {
                .idPack = Tuple(mgId, pgId, tgId),
                .workGroupId = workStart++,
                .workPtr = std::move(renderTypedPtr)
            }
        );
    }
    return workStart;
}

template <class C>
uint32_t RendererT<C>::GenerateLightWorkMappings(uint32_t workStart)
{
    using Algo::PartitionRange;
    const auto& lightSurfs = tracerView.lightSurfs;
    using LightSurfP = Pair<LightSurfaceId, LightSurfaceParams>;
    auto LightSurfIsLess = [](const LightSurfP& left, const LightSurfP& right)
    {
        auto GetLG = [](LightId id) -> CommonKey
        {
            return LightKey(static_cast<uint32_t>(id)).FetchBatchPortion();
        };
        auto GetTG = [](TransformId id) -> CommonKey
        {
            return TransformKey(static_cast<uint32_t>(id)).FetchBatchPortion();
        };

        return (Tuple(GetLG(left.second.lightId), GetTG(left.second.transformId)) <
                Tuple(GetLG(right.second.lightId), GetTG(right.second.transformId)));
    };
    assert(std::is_sorted(lightSurfs.cbegin(), lightSurfs.cend(),
                          LightSurfIsLess));

    auto partitions = Algo::PartitionRange(lightSurfs.cbegin(), lightSurfs.cend(),
                                           LightSurfIsLess);

    auto AddWork = [&, this](const LightSurfaceParams& lSurf,
                             bool checkBoundarySuitability)
    {
        LightGroupId lgId{LightKey(uint32_t(lSurf.lightId)).FetchBatchPortion()};
        TransGroupId tgId{TransformKey(uint32_t(lSurf.transformId)).FetchBatchPortion()};
        // These should be checked beforehand, while actually creating
        // the surface
        const LightGroupPtr& lg = tracerView.lightGroups.at(lgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view lgName = lg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderLightWorkType;
        std::string workName = CreateRenderLightWorkType(lgName, tgName);

        auto loc = std::get<1>(workPack).at(workName);
        if(!loc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Light/Transform "
                            "pair of \"{}/{}\"",
                            C::TypeName(), lgName, tgName);
        }
        if(checkBoundarySuitability)
        {
            bool isPrimBacked = lg.get()->IsPrimitiveBacked();
            if(isPrimBacked)
            {
                throw MRayError("[{}]: Primitive-backed light ({}) is requested "
                                "to be used as a boundary material!",
                                C::TypeName(), lgName);
            }
            auto duplicateLoc = std::find_if(currentLightWorks.cbegin(),
                                             currentLightWorks.cend(),
                                             [&](const auto& workInfo)
            {
                return workInfo.idPack == Pair(lgId, tgId);
            });
            if(duplicateLoc != currentLightWorks.cend())
            {
                // TODO: This restriction does not make sense too much.
                // Probably change later maybe?
                throw MRayError("[{}]: Light/Transform group \"{}\"({}) / \"{}\"({}), is used "
                                "as a non-boundary material. Boundary material should have its "
                                "own unique group!", C::TypeName(),
                                lgName, uint32_t(lgId), lgName, uint32_t(tgId));
            }
        }

        RenderLightWorkGenerator generator = loc->get();
        RenderLightWorkPtr ptr = generator(*lg.get(), *tg.get(), gpuSystem);

        using Ptr = typename RenderLightWorkStruct<C>::WorkPtr;
        Ptr renderTypedPtr = Ptr(static_cast<RenderLightWorkT<C>*>(ptr.release()));
        // Put this ptr somewhere... safe
        currentLightWorks.emplace_back
        (
            RenderLightWorkStruct
            {
                .idPack = Pair(lgId, tgId),
                .workGroupId = workStart++,
                .workPtr = std::move(renderTypedPtr)
            }
        );
    };

    for(const Vector2ul& p : partitions)
    {
        const auto& lSurf = lightSurfs[p[0]].second;
        AddWork(lSurf, false);
    }

    AddWork(tracerView.boundarySurface, true);

    // Set the boundary key pack here as well
    auto lK = std::bit_cast<LightKey>(tracerView.boundarySurface.lightId);
    auto lmK = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG,
                                          lK.FetchBatchPortion(),
                                          lK.FetchIndexPortion());
    auto tK = std::bit_cast<TransformKey>(tracerView.boundarySurface.transformId);
    boundaryMatKeyPack = HitKeyPack
    {
        .primKey = PrimitiveKey::InvalidKey(),
        .lightOrMatKey = lmK,
        .transKey = tK,
        .accelKey = AcceleratorKey::InvalidKey()
    };
    return workStart;
}

template <class C>
uint32_t RendererT<C>::GenerateCameraWorkMappings(uint32_t workStart)
{
    const auto& camSurfs = tracerView.camSurfs;
    using CamSurfP = Pair<CamSurfaceId, CameraSurfaceParams>;
    auto CamSurfIsLess = [](const CamSurfP& left, const CamSurfP& right)
    {
        auto GetCG = [](CameraId id) -> CommonKey
        {
            return CameraKey(static_cast<uint32_t>(id)).FetchBatchPortion();
        };
        auto GetTG = [](TransformId id) -> CommonKey
        {
            return TransformKey(static_cast<uint32_t>(id)).FetchBatchPortion();
        };
        return (Tuple(GetCG(left.second.cameraId), GetTG(left.second.transformId)) <
                Tuple(GetCG(right.second.cameraId), GetTG(right.second.transformId)));
    };
    assert(std::is_sorted(camSurfs.cbegin(), camSurfs.cend(),
                          CamSurfIsLess));

    auto partitions = Algo::PartitionRange(camSurfs.cbegin(), camSurfs.cend(),
                                           CamSurfIsLess);
    for(const auto& p : partitions)
    {
        size_t i = p[0];
        CameraGroupId cgId{CameraKey(uint32_t(camSurfs[i].second.cameraId)).FetchBatchPortion()};
        TransGroupId tgId{TransformKey(uint32_t(camSurfs[i].second.transformId)).FetchBatchPortion()};
        // These should be checked beforehand, while actually creating
        // the surface
        const CameraGroupPtr& cg = tracerView.camGroups.at(cgId).value();
        const TransformGroupPtr& tg = tracerView.transGroups.at(tgId).value();
        std::string_view cgName = cg->Name();
        std::string_view tgName = tg->Name();

        using TypeNameGen::Runtime::CreateRenderCameraWorkType;
        std::string workName = CreateRenderCameraWorkType(cgName, tgName);

        auto loc = std::get<2>(workPack).at(workName);
        if(!loc.has_value())
        {
            throw MRayError("[{}]: Could not find a renderer \"work\" for Camera/Transform "
                            "pair of \"{}/{}\"",
                            C::TypeName(), cgName, tgName);
        }
        RenderCameraWorkGenerator generator = loc->get();
        RenderCameraWorkPtr ptr = generator(*cg.get(), *tg.get(), gpuSystem);

        using Ptr = typename RenderCameraWorkStruct<C>::WorkPtr;
        Ptr renderTypedPtr = Ptr(static_cast<RenderCameraWorkT<C>*>(ptr.release()));
        // Put this ptr somewhere... safe
        currentCameraWorks.emplace_back
        (
            RenderCameraWorkStruct
            {
                .idPack = Pair(cgId, tgId),
                .workGroupId = workStart++,
                .workPtr = std::move(renderTypedPtr)
            }
        );
    }
    return workStart;
}

template <class C>
uint32_t RendererT<C>::GenerateWorks()
{
    // Generate works per
    // Material1/Primitive/Transform triplet,
    // Light/Transform pair,
    // Camera/Transform pair
    workCounter = 0;
    workCounter = GenerateWorkMappings(workCounter);
    workCounter = GenerateLightWorkMappings(workCounter);
    workCounter = GenerateCameraWorkMappings(workCounter);
    return workCounter;
}
template <class C>
RenderWorkHasher RendererT<C>::InitializeHashes(Span<uint32_t> dHashes,
                                                Span<CommonKey> dWorkIds,
                                                const GPUQueue& queue)
{
    RenderWorkHasher result(dHashes, dWorkIds);
    result.PopulateHashesAndKeys<C>(tracerView,
                                    currentWorks,
                                    currentLightWorks,
                                    currentCameraWorks,
                                    queue);
    return result;
}

template <class C>
RendererT<C>::RendererT(const RenderImagePtr& rb,
                        const RenderWorkPack& wp,
                        TracerView tv, const GPUSystem& s,
                        BS::thread_pool& tp)
    : globalThreadPool(tp)
    , gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)
    , workPack(wp)
{}

template <class C>
std::string_view RendererT<C>::Name() const
{
    return C::TypeName();
}

//=======================================//
// Helper functions to instantiate works //
//=======================================//
// TODO: Maybe add these on a namespace
template<class R, class Works, class LWorks, class CWorks>
struct RenderWorkTypePack
{
    using RendererType      = R;
    using WorkTypes         = Works;
    using LightWorkTypes    = LWorks;
    using CameraWorkTypes   = CWorks;
};

template <class RenderWorkTypePackT>
inline void AddSingleRenderWork(Map<std::string_view, RenderWorkPack>& workMap,
                                RenderWorkTypePackT*)
{
    using RendererType      = typename RenderWorkTypePackT::RendererType;
    using WorkTypes         = typename RenderWorkTypePackT::WorkTypes;
    using LightWorkTypes    = typename RenderWorkTypePackT::LightWorkTypes;
    using CameraWorkTypes   = typename RenderWorkTypePackT::CameraWorkTypes;

    RenderWorkPack& workPack = workMap.emplace(RendererType::TypeName(),
                                               RenderWorkPack()).first->second;

    //================//
    // Material Works //
    //================//
    using WorkGenArgs = Tuple<const GenericGroupMaterialT&,
                              const GenericGroupPrimitiveT&,
                              const GenericGroupTransformT&,
                              const GPUSystem&>;
    WorkGenArgs* workArgsResolver = nullptr;
    WorkTypes* workTypesResolver = nullptr;
    GenerateMapping<RenderWorkGenerator, RenderWorkI>
    (
        std::get<0>(workPack),
        workArgsResolver,
        workTypesResolver
    );
    //================//
    //   Light Works  //
    //================//
    using LightWorkGenArgs = Tuple<const GenericGroupLightT&,
                                   const GenericGroupTransformT&,
                                   const GPUSystem&>;
    LightWorkGenArgs* lightWorkArgsResolver = nullptr;
    LightWorkTypes* lightWorkTypesResolver = nullptr;
    GenerateMapping<RenderLightWorkGenerator, RenderLightWorkI>
    (
        std::get<1>(workPack),
        lightWorkArgsResolver,
        lightWorkTypesResolver
    );
    //================//
    //  Camera Works  //
    //================//
    using CameraWorkGenArgs = Tuple<const GenericGroupCameraT&,
                                    const GenericGroupTransformT&,
                                    const GPUSystem&>;
    CameraWorkGenArgs* cameraWorkArgsResolver = nullptr;
    CameraWorkTypes* cameraWorkTypesResolver = nullptr;
    GenerateMapping<RenderCameraWorkGenerator, RenderCameraWorkI>
    (
        std::get<2>(workPack),
        cameraWorkArgsResolver,
        cameraWorkTypesResolver
    );
}

template <class... Args>
void AddRenderWorks(Map<std::string_view, RenderWorkPack>& workMap,
                    Tuple<Args...>* list)
{
    auto AddRenderWorksInternal =
    []<class... Args, size_t... Is>(Map<std::string_view, RenderWorkPack>&workMap,
                                    Tuple<Args...>* list,
                                    std::index_sequence<Is...>)
    {
        // Param pack expansion over the index sequence
        (
            (AddSingleRenderWork(workMap, &std::get<Is>(*list))),
            ...
        );
    };

    AddRenderWorksInternal(workMap, list,
                           std::index_sequence_for<Args...>{});
}