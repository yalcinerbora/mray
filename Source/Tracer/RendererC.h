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

#include "Device/GPUSystemForward.h"

#include "Common/RenderImageStructs.h"

class ThreadPool;
//
struct MultiPartitionOutput;
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

    virtual uint32_t         SampleRNCount(uint32_t workIndex) const = 0;
    virtual std::string_view Name() const = 0;
};

class RenderLightWorkI
{
    public:
    virtual ~RenderLightWorkI() = default;

    virtual uint32_t         SampleRNCount(uint32_t workIndex) const = 0;
    virtual std::string_view Name() const = 0;
};

class RenderCameraWorkI
{
    public:
    virtual ~RenderCameraWorkI() = default;

    virtual std::string_view    Name() const = 0;
    virtual uint32_t            SampleRayRNCount() const = 0;
    virtual uint32_t            StochasticFilterSampleRayRNCount() const = 0;
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
using RenderWorkPack = std::tuple<RenderWorkGenMap, RenderLightWorkGenMap, RenderCamWorkGenMap>;

template<class RendererType>
concept RendererC = requires(RendererType rt,
                             const TracerView& tv,
                             TransientData input,
                             ThreadPool& tp,
                             const GPUSystem& gpuSystem,
                             const GPUQueue& q)
{
    // Global State
    // These parameters are work agnostic.
    typename RendererType::GlobalStateList;
    // These parameters are work related.
    // (Not used but exposed for future use maybe?)
    typename RendererType::RayStateList;
    typename RendererType::SpectrumConverterContext;
    // Host side things
    typename RendererType::AttribInfoList;

    // CPU Side
    RendererType(RenderImagePtr{}, tv, tp, gpuSystem, RenderWorkPack{});
    {rt.AttributeInfo()
    } -> std::same_as<typename RendererType::AttribInfoList>;
    {rt.PushAttribute(uint32_t{}, std::move(input), q)
    } -> std::same_as<void>;
    {rt.StartRender(RenderImageParams{}, CamSurfaceId{},
                    uint32_t{}, uint32_t{})
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
                                            uint32_t customLogicIndex0 = 0,
                                            uint32_t customLogicIndex1 = 0) = 0;
    virtual void                SetCameraTransform(const CameraTransform&) = 0;
    virtual RendererOutput      DoRender() = 0;
    virtual void                StopRender() = 0;

    virtual std::string_view    Name() const = 0;
    virtual size_t              GPUMemoryUsage() const = 0;
};

using RendererPtr = std::unique_ptr<RendererI>;

namespace RendererDetail
{
    // https://stackoverflow.com/questions/28432977/generic-way-of-lazily-evaluating-short-circuiting-with-stdconditional-t
    // Short circuit conditional
    template<RendererC R, uint32_t I, bool B>
    struct RenderGlobalState;

    template<RendererC R, uint32_t I>
    struct RenderGlobalState<R, I, false> { using type = EmptyType;};

    template<RendererC R, uint32_t I>
    struct RenderGlobalState<R, I, true>
    { using type = std::tuple_element_t<I, typename R::GlobalStateList>; };

    template<RendererC R, uint32_t I, bool B>
    struct RenderRayState;

    template<RendererC R, uint32_t I>
    struct RenderRayState<R, I, false> { using type = EmptyType;};

    template<RendererC R, uint32_t I>
    struct RenderRayState<R, I, true>
    { using type = std::tuple_element_t<I, typename R::RayStateList>; };

}

template<RendererC R, uint32_t I>
using RenderGlobalState = RendererDetail::RenderGlobalState<R, I, I < std::tuple_size_v<typename R::GlobalStateList>>::type;
template<RendererC R, uint32_t I>
using RenderRayState = RendererDetail::RenderRayState<R, I, I < std::tuple_size_v<typename R::RayStateList>>::type;

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
#define MRAY_RENDER_DO_WORK_DECL(tag)               \
void DoWork_##tag                                   \
(                                                   \
    const RenderRayState<R, tag>& dRayStates,       \
    Span<const RayIndex> dRayIndicesIn,             \
    Span<const RandomNumber> dRandomNumbers,        \
    Span<const RayCone> dRayDiffsIn,                \
    Span<const RayGMem> dRaysIn,                    \
    Span<const MetaHit> dHitsIn,                    \
    Span<const HitKeyPack> dKeysIn,                 \
    const RenderGlobalState<R, tag>& globalState,   \
    const GPUQueue& queue                           \
) const

#define MRAY_RENDER_DO_WORK_DEF(tag)    \
void DoWork_##tag                       \
(                                       \
    const RenderRayState<R, tag>& a,    \
    Span<const RayIndex> b,             \
    Span<const RandomNumber> c,         \
    Span<const RayCone> d,              \
    Span<const RayGMem> e,              \
    Span<const MetaHit> f,              \
    Span<const HitKeyPack> g,           \
    const RenderGlobalState<R, tag>& h, \
    const GPUQueue& i                   \
) const override                        \
{                                       \
    DoWorkInternal<tag>(a, b, c, d,     \
                        e, f, g, h, i); \
}

#define MRAY_RENDER_DO_LIGHT_WORK_DECL(tag)         \
void DoBoundaryWork_##tag                           \
(                                                   \
    const RenderRayState<R, tag>& dRayStates,       \
    Span<const RayIndex> dRayIndicesIn,             \
    Span<const uint32_t> dRandomNumbers,            \
    Span<const RayCone> dRayDiffsIn,                \
    Span<const RayGMem> dRaysIn,                    \
    Span<const MetaHit> dHitsIn,                    \
    Span<const HitKeyPack> dKeysIn,                 \
    const RenderGlobalState<R, tag>& globalState,   \
    const GPUQueue& queue                           \
) const

#define MRAY_RENDER_DO_LIGHT_WORK_DEF(tag)  \
void DoBoundaryWork_##tag                   \
(                                           \
    const RenderRayState<R, tag>& a,        \
    Span<const RayIndex> b,                 \
    Span<const uint32_t> c,                 \
    Span<const RayCone> d,                  \
    Span<const RayGMem> e,                  \
    Span<const MetaHit> f,                  \
    Span<const HitKeyPack> g,               \
    const RenderGlobalState<R, tag>& h,     \
    const GPUQueue& i                       \
) const override                            \
{                                           \
    DoBoundaryWorkInternal<tag>(a, b, c,    \
                                d, e, f,    \
                                g, h, i);   \
}

template<class R>
class RenderWorkT : public RenderWorkI
{
    public:
    virtual MRAY_RENDER_DO_WORK_DECL(0) = 0;
    virtual MRAY_RENDER_DO_WORK_DECL(1) = 0;
};

template<class R>
class RenderLightWorkT : public RenderLightWorkI
{
    public:
    virtual MRAY_RENDER_DO_LIGHT_WORK_DECL(0) = 0;
    virtual MRAY_RENDER_DO_LIGHT_WORK_DECL(1) = 0;
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
                              const Span<RayCone>& dRayDiffsOut,
                              const Span<RayGMem>& dRaysOut,
                              const Span<ImageCoordinate>& dImageCoordsOut,
                              const Span<Float>& dSampleWeightsOut,
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
    virtual void GenRaysStochasticFilter(// Output
                                         const Span<RayCone>& dRayDiffsOut,
                                         const Span<RayGMem>& dRaysOut,
                                         const Span<ImageCoordinate>& dImageCoordsOut,
                                         const Span<Float>& dSampleWeightsOut,
                                         // Input
                                         const Span<const uint32_t>& dRayIndices,
                                         const Span<const uint32_t>& dRandomNums,
                                         // Type erased buffer
                                         Span<const Byte> dCamBuffer,
                                         TransformKey transKey,
                                         // Constants
                                         uint64_t globalPixelIndex,
                                         const Vector2ui regionCount,
                                         FilterType filterType,
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
    using IdTuple = std::tuple<MatGroupId, PrimGroupId, TransGroupId>;

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
    Span<CommonKey> dWorkBatchHashes;
    Span<CommonKey> dWorkBatchIds;
    uint32_t batchBits  = 0;
    uint32_t dataBits   = 0;
    // Maximum identifier count that is used by a single
    // group.
    uint32_t maxMatOrLightIdBits    = 0;
    uint32_t maxPrimIdBits          = 0;
    uint32_t maxTransIdBits         = 0;
    uint32_t maxIndexBits           = 0;

    protected:
    public:
                RenderWorkHasher() = default;
    MRAY_HOST   RenderWorkHasher(Span<CommonKey> dWorkBatchHashes,
                                 Span<CommonKey> dBatchIds);

    template<class R>
    MRAY_HOST
    void PopulateHashesAndKeys(const TracerView& tracerView,
                               const RenderWorkList<R>& curWorks,
                               const RenderLightWorkList<R>& curLightWorks,
                               const RenderCameraWorkList<R>& curCamWorks,
                               uint32_t maxRayCount,
                               const GPUQueue& queue);

    MRAY_HYBRID
    Vector2ui WorkBatchDataRange() const;
    MRAY_HYBRID
    Vector2ui WorkBatchBitRange() const;
    MRAY_HYBRID
    CommonKey BisectBatchPortion(CommonKey key) const;
    MRAY_HYBRID
    CommonKey HashWorkBatchPortion(HitKeyPack p) const;
    MRAY_HYBRID
    CommonKey HashWorkDataPortion(HitKeyPack p, RayIndex i) const;
    MRAY_HYBRID
    CommonKey GenerateWorkKeyGPU(HitKeyPack p, RayIndex i) const;
};

template <class Child>
class RendererT : public RendererI
{
    public:
    using AttribInfoList    = typename RendererI::AttribInfoList;
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
    ThreadPool&                 globalThreadPool;
    const GPUSystem&            gpuSystem;
    TracerView                  tracerView;
    const RenderImagePtr&       renderBuffer;
    Optional<CameraTransform>   cameraTransform = {};

    WorkList              currentWorks;
    LightWorkList         currentLightWorks;
    CameraWorkList        currentCameraWorks;
    HitKeyPack            boundaryLightKeyPack;
    // Current Canvas info
    MRayColorSpaceEnum    curColorSpace;
    ImageTiler            imageTiler;
    uint64_t              totalIterationCount;

    uint32_t              GenerateWorks();
    void                  ClearAllWorkMappings();
    RenderWorkHasher      InitializeHashes(Span<CommonKey> dHashes,
                                           Span<CommonKey> dWorkIds,
                                           uint32_t maxRayCount,
                                           const GPUQueue& queue);

    // Some common functions between renderer
    template<class WorkF, class LightWorkF = EmptyFunctor, class CamWorkF = EmptyFunctor>
    void    IssueWorkKernelsToPartitions(const RenderWorkHasher&,
                                         const MultiPartitionOutput&,
                                         WorkF&&,
                                         LightWorkF && = LightWorkF(),
                                         CamWorkF && = CamWorkF()) const;

    public:
                        RendererT(const RenderImagePtr&,
                                  const RenderWorkPack& workPacks,
                                  TracerView, const GPUSystem&,
                                  ThreadPool&);
    void                SetCameraTransform(const CameraTransform&) override;
    std::string_view    Name() const override;
};

inline bool FlatSurfParams::operator<(const FlatSurfParams& right) const
{
    auto GetMG = [](MaterialId id) -> CommonKey
    {
        return std::bit_cast<MaterialKey>(id).FetchBatchPortion();
    };
    auto GetPG = [](PrimBatchId id) -> CommonKey
    {
        return std::bit_cast<PrimBatchKey>(id).FetchBatchPortion();
    };
    auto GetTG = [](TransformId id) -> CommonKey
    {
        return std::bit_cast<TransformKey>(id).FetchBatchPortion();
    };

    using T = std::tuple<CommonKey, CommonKey, CommonKey>;
    return (T(GetMG(mId), GetTG(tId), GetPG(pId)) <
            T(GetMG(right.mId), GetTG(right.tId), GetPG(right.pId)));
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
RenderWorkHasher::RenderWorkHasher(Span<CommonKey> dWBHashes,
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
                                             uint32_t maxRayCount,
                                             const GPUQueue& queue)
{
    size_t totalWorkBatchCount = (curWorks.size() + curLightWorks.size() +
                                  curCamWorks.size());
    std::vector<CommonKey> hHashes;
    std::vector<CommonKey> hBatchIds;
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

        auto pK = PrimitiveKey::CombinedKey(std::bit_cast<CommonKey>(primGroupId), 0u);
        auto mK = LightOrMatKey::CombinedKey(IS_MAT_KEY_FLAG, std::bit_cast<CommonKey>(matGroupId), 0u);
        auto tK = TransformKey::CombinedKey(std::bit_cast<CommonKey>(transGroupId), 0u);
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
        CommonKey primGroupId = lightGroup->GenericPrimGroup().GroupId();

        auto lK = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG, static_cast<CommonKey>(lightGroupId), 0u);
        auto tK = TransformKey::CombinedKey(std::bit_cast<CommonKey>(transGroupId), 0u);
        auto pK = PrimitiveKey::CombinedKey(primGroupId, 0u);
        HitKeyPack kp =
        {
            .primKey = pK,
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
    maxIndexBits = Bit::RequiredBitsToRepresent(maxRayCount);

    queue.MemcpyAsync(dWorkBatchHashes, Span<const CommonKey>(hHashes));
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
CommonKey RenderWorkHasher::BisectBatchPortion(CommonKey key) const
{
    return Bit::FetchSubPortion(key, {dataBits, dataBits + batchBits});
}

MRAY_HYBRID MRAY_CGPU_INLINE
CommonKey RenderWorkHasher::HashWorkBatchPortion(HitKeyPack p) const
{
    static_assert(PrimitiveKey::BatchBits + LightOrMatKey::BatchBits +
                  LightOrMatKey::FlagBits +
                  TransformKey::BatchBits <= sizeof(CommonKey) * CHAR_BIT,
                  "Unable to pack batch bits for hasing!");
    // In common case, compose the batch identifiers
    bool isLight = (p.lightOrMatKey.FetchFlagPortion() == IS_LIGHT_KEY_FLAG);
    CommonKey isLightInt = (isLight) ? 1u : 0u;
    CommonKey r = Bit::Compose<TransformKey::BatchBits, PrimitiveKey::BatchBits,
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
CommonKey RenderWorkHasher::HashWorkDataPortion(HitKeyPack p, RayIndex i) const
{
    // Get the Id portions
    CommonKey mlIndex = p.lightOrMatKey.FetchIndexPortion();
    CommonKey pIndex = p.primKey.FetchIndexPortion();
    CommonKey tIndex = p.transKey.FetchIndexPortion();

    // Heuristic: Most important bits are the material/light,
    // then primitive, then transform.
    // ---
    // This heuristic was incomplete.
    // Adding actual ray index to the heuristic
    // (which indirectly relates to ray payload)
    // since according to the measurements
    // it dominates due to writes.
    // ---
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
    CommonKey result = 0;
    auto WriteKey = [&](uint32_t maxItemBitCount, CommonKey item) -> bool
    {
        uint32_t bitsForItem = std::min(maxItemBitCount, remainingBits);
        if(bitsForItem == 0) return false;
        uint32_t offset = dataBits - bitsForItem + currentBit;
        std::array range = {CommonKey(offset), CommonKey(offset + bitsForItem)};
        result = Bit::SetSubPortion(result, item, range);
        remainingBits -= bitsForItem;
        currentBit += bitsForItem;
        return true;
    };
    // Actual Payload index
    if(!WriteKey(maxIndexBits, i)) return result;
    // Transforms
    if(!WriteKey(maxTransIdBits, tIndex)) return result;
    // Primitives
    if(!WriteKey(maxPrimIdBits, pIndex)) return result;
    // Materials
    if(!WriteKey(maxMatOrLightIdBits, mlIndex)) return result;

    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
CommonKey RenderWorkHasher::GenerateWorkKeyGPU(HitKeyPack p, RayIndex rayIndex) const
{
    CommonKey batchHash = HashWorkBatchPortion(p);
    // Find the batch portion (Linear search)
    uint32_t i = 0;
    CommonKey batchId = std::numeric_limits<CommonKey>::max();
    for(CommonKey checkHash : dWorkBatchHashes)
    {
        if(checkHash == batchHash)
        {
            batchId = dWorkBatchIds[i];
            break;
        }
        i++;
    }
    // Compose the sort key
    CommonKey hashLower = HashWorkDataPortion(p, rayIndex);
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
        MatGroupId mgId{std::bit_cast<MaterialKey>(flatSurfs[i].mId).FetchBatchPortion()};
        PrimGroupId pgId{std::bit_cast<PrimBatchKey>(flatSurfs[i].pId).FetchBatchPortion()};
        TransGroupId tgId{std::bit_cast<TransformKey>(flatSurfs[i].tId).FetchBatchPortion()};
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
                .idPack = std::tuple<MatGroupId, PrimGroupId, TransGroupId>
                (
                    mgId, pgId, tgId
                ),
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
            return std::bit_cast<LightKey>(id).FetchBatchPortion();
        };
        auto GetTG = [](TransformId id) -> CommonKey
        {
            return std::bit_cast<TransformKey>(id).FetchBatchPortion();
        };
        return (std::tuple(GetLG(left.second.lightId), GetTG(left.second.transformId)) <
                std::tuple(GetLG(right.second.lightId), GetTG(right.second.transformId)));
    };
    assert(std::is_sorted(lightSurfs.cbegin(), lightSurfs.cend(),
                          LightSurfIsLess));

    auto partitions = Algo::PartitionRange(lightSurfs.cbegin(), lightSurfs.cend(),
                                           LightSurfIsLess);

    auto AddWork = [&, this](const LightSurfaceParams& lSurf,
                             bool isBoundaryLight)
    {
        LightGroupId lgId{std::bit_cast<LightKey>(lSurf.lightId).FetchBatchPortion()};
        TransGroupId tgId{std::bit_cast<TransformKey>(lSurf.transformId).FetchBatchPortion()};
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
        if(isBoundaryLight)
        {
            bool isPrimBacked = lg.get()->IsPrimitiveBacked();
            if(isPrimBacked)
            {
                throw MRayError("[{}]: Primitive-backed light ({}) is requested "
                                "as a boundary material!",
                                C::TypeName(), lgName);
            }
            // Boundary material is not primitive-backed by definition
            // we can set the index portion as zero
            auto lK = std::bit_cast<LightKey>(lSurf.lightId);
            auto lmK = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG,
                                                  lK.FetchBatchPortion(),
                                                  lK.FetchIndexPortion());
            auto tK = std::bit_cast<TransformKey>(lSurf.transformId);
            CommonKey primGroupId = lg->GenericPrimGroup().GroupId();
            auto pK = PrimitiveKey::CombinedKey(primGroupId, 0u);
            boundaryLightKeyPack = HitKeyPack
            {
                .primKey = pK,
                .lightOrMatKey = lmK,
                .transKey = tK,
                .accelKey = AcceleratorKey::InvalidKey()
            };
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
                .idPack = Pair<LightGroupId, TransGroupId>(lgId, tgId),
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
            return std::bit_cast<CameraKey>(id).FetchBatchPortion();
        };
        auto GetTG = [](TransformId id) -> CommonKey
        {
            return std::bit_cast<TransformKey>(id).FetchBatchPortion();
        };
        return (std::tuple(GetCG(left.second.cameraId), GetTG(left.second.transformId)) <
                std::tuple(GetCG(right.second.cameraId), GetTG(right.second.transformId)));
    };
    assert(std::is_sorted(camSurfs.cbegin(), camSurfs.cend(),
                          CamSurfIsLess));

    auto partitions = Algo::PartitionRange(camSurfs.cbegin(), camSurfs.cend(),
                                           CamSurfIsLess);
    for(const auto& p : partitions)
    {
        size_t i = p[0];
        CameraGroupId cgId{std::bit_cast<CameraKey>(camSurfs[i].second.cameraId).FetchBatchPortion()};
        TransGroupId tgId{std::bit_cast<TransformKey>(camSurfs[i].second.transformId).FetchBatchPortion()};
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
                .idPack = Pair<CameraGroupId, TransGroupId>(cgId, tgId),
                .workGroupId = workStart++,
                .workPtr = std::move(renderTypedPtr)
            }
        );
    }
    return workStart;
}

template <class C>
void RendererT<C>::ClearAllWorkMappings()
{
    currentCameraWorks.clear();
    currentWorks.clear();
    currentLightWorks.clear();
    workCounter = 0;
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
RenderWorkHasher RendererT<C>::InitializeHashes(Span<CommonKey> dHashes,
                                                Span<CommonKey> dWorkIds,
                                                uint32_t maxRayCount,
                                                const GPUQueue& queue)
{
    RenderWorkHasher result(dHashes, dWorkIds);
    result.PopulateHashesAndKeys<C>(tracerView,
                                    currentWorks,
                                    currentLightWorks,
                                    currentCameraWorks,
                                    maxRayCount,
                                    queue);
    return result;
}

template <class C>
RendererT<C>::RendererT(const RenderImagePtr& rb,
                        const RenderWorkPack& wp,
                        TracerView tv, const GPUSystem& s,
                        ThreadPool& tp)
    : workPack(wp)
    , globalThreadPool(tp)
    , gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)

{}

template <class C>
void RendererT<C>::SetCameraTransform(const CameraTransform& ct)
{
    cameraTransform = ct;
}

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
    using WorkGenArgs = PackedTypes<const GenericGroupMaterialT&,
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
    using LightWorkGenArgs = PackedTypes<const GenericGroupLightT&,
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
    using CameraWorkGenArgs = PackedTypes<const GenericGroupCameraT&,
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
                    PackedTypes<Args...>*)
{
    auto AddRenderWorksInternal =
    []<size_t... Is>(Map<std::string_view, RenderWorkPack>& workMap,
                     std::tuple<Args...>* list,
                     std::index_sequence<Is...>)
    {
        // Param pack expansion over the index sequence
        (
            (AddSingleRenderWork(workMap, &std::get<Is>(*list))),
            ...
        );
    };

    std::tuple<Args...>* list = nullptr;
    AddRenderWorksInternal(workMap, list,
                           std::index_sequence_for<Args...>{});
}
