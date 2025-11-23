#pragma once

#include "TracerTypes.h"
#include "RenderImage.h"
#include "GenericGroup.h"
#include "MaterialC.h"
#include "PrimitiveC.h"
#include "CameraC.h"
#include "TransformC.h"
#include "LightC.h"
#include "TracerTypes.h"
#include "AcceleratorC.h"
#include "Texture.h"
#include "TextureFilter.h"
#include "MediaTracker.h"

#include "Core/Algorithm.h"
#include "Core/TypePack.h"

#include "TransientPool/TransientPool.h"

#include "Device/GPUSystemForward.h"

#include "Common/RenderImageStructs.h"

#include <climits>

class ThreadPool;
//
struct MultiPartitionOutput;
// A nasty forward declaration
class GenericGroupPrimitiveT;
class GenericGroupCameraT;
class GenericGroupLightT;
template<class, class> class GenericGroupT;
template<class, class> class GenericTexturedGroupT;
//
using GenericGroupTransformT = GenericGroupT<TransformKey, TransAttributeInfo>;
using GenericGroupMediumT    = GenericTexturedGroupT<MediumKey, MediumAttributeInfo>;
using GenericGroupMaterialT  = GenericTexturedGroupT<MaterialKey, MatAttributeInfo>;

class MediaTrackerView;

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
    const std::vector<Pair<VolumeId, VolumeKeyPack>>&               globalVolumeList;
};

class RenderWorkI
{
    public:
    virtual ~RenderWorkI() = default;

    virtual RNRequestList    SampleRNList(uint32_t workIndex) const = 0;
    virtual std::string_view Name() const = 0;
};

class RenderLightWorkI
{
    public:
    virtual ~RenderLightWorkI() = default;

    virtual RNRequestList    SampleRNList(uint32_t workIndex) const = 0;
    virtual std::string_view Name() const = 0;
};

class RenderCameraWorkI
{
    public:
    virtual ~RenderCameraWorkI() = default;

    virtual std::string_view    Name() const = 0;
    virtual RNRequestList       SampleRayRNList() const = 0;
    virtual RNRequestList       StochasticFilterSampleRayRNList() const = 0;

    // These are common for every camera and renderer and camera type independent
    virtual void GenerateSubCamera(// Output
                                   Span<Byte> dCamBuffer,
                                   // Constants
                                   CameraKey camKey,
                                   Optional<CameraTransform> camTransform,
                                   Vector2ui stratumIndex,
                                   Vector2ui stratumCount,
                                   const GPUQueue& queue) const = 0;
    virtual void GenCameraPosition(// Output
                                   const Span<Vector3, 1>& dCamPosOut,
                                   // Input
                                   Span<const Byte> dCamBuffer,
                                   TransformKey transKey,
                                   // Constants
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
    virtual void ReconstructCameraRays(// Output
                                       const Span<RayCone>& dRayDiffsOut,
                                       const Span<RayGMem>& dRaysOut,
                                       // Input
                                       const Span<const ImageCoordinate>& dImageCoordsOut,
                                       const Span<const uint32_t>& dRayIndices,
                                       // The actual pair to be used
                                       Span<const Byte> dCamBuffer,
                                       TransformKey transKey,
                                       // Constants
                                       const Vector2ui regionCount,
                                       const GPUQueue& queue) const = 0;
};

class RenderMediumWorkI
{
    public:
    virtual ~RenderMediumWorkI() = default;

    virtual RNRequestList    SampleRNList(uint32_t workIndex) const = 0;
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
using RenderMediumWorkGenerator = GeneratorFuncType<RenderMediumWorkI,
                                                    const GenericGroupMediumT&,
                                                    const GenericGroupTransformT&,
                                                    const GPUSystem&>;

// Work generator related
using RenderWorkGenMap = Map<std::string_view, RenderWorkGenerator>;
using RenderLightWorkGenMap = Map<std::string_view, RenderLightWorkGenerator>;
using RenderCamWorkGenMap = Map<std::string_view, RenderCameraWorkGenerator>;
using RenderMediumWorkGenMap = Map<std::string_view, RenderMediumWorkGenerator>;

struct RenderWorkPack
{
    RenderWorkGenMap        workMap;
    RenderLightWorkGenMap   lightWorkMap;
    RenderCamWorkGenMap     camWorkMap;
    RenderMediumWorkGenMap  mediumWorkMap;
};

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
    typename RendererType::SpectrumContext;
    // Host side things
    typename RendererType::AttribInfoList;

    // CPU Side
    RendererType(RenderImagePtr{}, tv, tp, gpuSystem, RenderWorkPack{});
    {rt.AttributeInfo()
    } -> std::same_as<typename RendererType::AttribInfoList>;
    // It is hard to instantiate renderer
    // due to it requires multiple references
    {RendererType::StaticAttributeInfo()
    } -> std::same_as<typename RendererType::AttribInfoList>;
    //
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
using RenderMediumWorkPtr = std::unique_ptr<RenderMediumWorkI>;

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
    { using type = TypePackElement<I, typename R::GlobalStateList>; };

    template<RendererC R, uint32_t I, bool B>
    struct RenderRayState;

    template<RendererC R, uint32_t I>
    struct RenderRayState<R, I, false> { using type = EmptyType;};

    template<RendererC R, uint32_t I>
    struct RenderRayState<R, I, true>
    { using type = TypePackElement<I, typename R::RayStateList>; };

}

template<RendererC R, uint32_t I>
using RenderGlobalState = RendererDetail::RenderGlobalState<R, I, I < TypePackSize<typename R::GlobalStateList>>::type;
template<RendererC R, uint32_t I>
using RenderRayState = RendererDetail::RenderRayState<R, I, I < TypePackSize<typename R::RayStateList>>::type;

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
#define MRAY_RENDER_DO_WORK_DECL(tag)             \
void DoWork_##tag                                 \
(                                                 \
    const RenderRayState<R, tag>& dRayStates,     \
    Span<RayGMem> dRaysIn,                        \
    Span<RayCone> dRayDiffsIn,                    \
    Span<const RayIndex> dRayIndicesIn,           \
    Span<const RandomNumber> dRandomNumbers,      \
    Span<const MetaHit> dHitsIn,                  \
    Span<const HitKeyPack> dKeysIn,               \
    const RenderGlobalState<R, tag>& globalState, \
    const GPUQueue& queue                         \
) const

#define MRAY_RENDER_DO_WORK_DEF(tag)    \
void DoWork_##tag                       \
(                                       \
    const RenderRayState<R, tag>& a,    \
    Span<RayGMem> b,                    \
    Span<RayCone> c,                    \
    Span<const RayIndex> d,             \
    Span<const RandomNumber> e,         \
    Span<const MetaHit> f,              \
    Span<const HitKeyPack> g,           \
    const RenderGlobalState<R, tag>& h, \
    const GPUQueue& i                   \
) const override                        \
{                                       \
    DoWorkInternal<tag>(a, b, c, d,     \
                        e, f, g, h, i); \
}

#define MRAY_RENDER_DO_LIGHT_WORK_DECL(tag)       \
void DoBoundaryWork_##tag                         \
(                                                 \
    const RenderRayState<R, tag>& dRayStates,     \
    Span<RayGMem> dRaysIO,                        \
    Span<RayCone> dRayDiffsIO,                    \
    Span<const RayIndex> dRayIndicesIn,           \
    Span<const uint32_t> dRandomNumbers,          \
    Span<const MetaHit> dHitsIn,                  \
    Span<const HitKeyPack> dKeysIn,               \
    const RenderGlobalState<R, tag>& globalState, \
    const GPUQueue& queue                         \
) const

#define MRAY_RENDER_DO_LIGHT_WORK_DEF(tag)  \
void DoBoundaryWork_##tag                   \
(                                           \
    const RenderRayState<R, tag>& a,        \
    Span<RayGMem> b,                        \
    Span<RayCone> c,                        \
    Span<const RayIndex> d,                 \
    Span<const uint32_t> e,                 \
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

#define MRAY_RENDER_MEDIUM_DO_WORK_DECL(tag)            \
void DoWork_##tag                                       \
(                                                       \
    const RenderRayState<R, tag>& dRayStates,           \
    Span<RayGMem> dRaysIO,                              \
    Span<RayCone> dRayDiffsIO,                          \
    Span<const RayIndex> dRayIndices,                   \
    Span<const RayMediaListPack> dRayMediaPacks,        \
    Span<const RandomNumber> dRandomNumbers,            \
    const MediaTrackerView& mediaTracker,               \
    const RenderGlobalState<R, tag>& globalState,       \
    const GPUQueue& queue                               \
) const

#define MRAY_RENDER_MEDIUM_DO_WORK_DEF(tag) \
void DoWork_##tag                           \
(                                           \
    const RenderRayState<R, tag>& a,        \
    Span<RayGMem> b,                        \
    Span<RayCone> c,                        \
    Span<const RayIndex> d,                 \
    Span<const RayMediaListPack> e,         \
    Span<const RandomNumber> f,             \
    const MediaTrackerView& g,              \
    const RenderGlobalState<R, tag>& h,     \
    const GPUQueue& i                       \
) const override                            \
{                                           \
    DoWorkInternal<tag>(a, b, c, d,         \
                        e, f, g, h, i);     \
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
    // later, but currently all of its functionality is
    // on the actual interface. So this is empty
};

template<class R>
class RenderMediumWorkT : public RenderMediumWorkI
{
    public:
    virtual MRAY_RENDER_MEDIUM_DO_WORK_DECL(0) = 0;
    virtual MRAY_RENDER_MEDIUM_DO_WORK_DECL(1) = 0;
};

// Renderer holds its work in a linear array.
// by definition (by the key batch width) there should be small amount of groups
// so we will do a linear search (maybe in future binary search) over these
// values. The reason we do not use hash table or map is that we may move the
// multi-kernel call to the GPU side in future.
struct RenderWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderWorkI>;
    //
    MatGroupId   mgId;
    PrimGroupId  pgId;
    TransGroupId tgId;
    CommonKey    workGroupId;
    WorkPtr      workPtr;

    auto operator<=>(const RenderWorkStruct&) const noexcept;
};

struct RenderLightWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderLightWorkI>;
    //
    LightGroupId lgId;
    TransGroupId tgId;
    CommonKey    workGroupId;
    WorkPtr      workPtr;

    auto operator<=>(const RenderLightWorkStruct&) const noexcept;
};

struct RenderCameraWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderCameraWorkI>;
    //
    CameraGroupId cgId;
    TransGroupId  tgId;
    CommonKey     workGroupId;
    WorkPtr       workPtr;

    auto operator<=>(const RenderCameraWorkStruct&) const noexcept;
};

struct RenderMediumWorkStruct
{
    using WorkPtr = std::unique_ptr<RenderMediumWorkI>;
    //
    MediumGroupId mgId;
    TransGroupId  tgId;
    CommonKey     workGroupId;
    WorkPtr       workPtr;

    auto operator<=>(const RenderMediumWorkStruct&) const noexcept;
};

using RenderWorkList        = std::vector<RenderWorkStruct>;
using RenderLightWorkList   = std::vector<RenderLightWorkStruct>;
using RenderCameraWorkList  = std::vector<RenderCameraWorkStruct>;
using RenderMediumWorkList  = std::vector<RenderMediumWorkStruct>;

template<class Renderer>
const RenderWorkT<Renderer>&
UpcastRenderWork(const std::unique_ptr<RenderWorkI>& ptr)
{
    using ResultT = RenderWorkT<Renderer>;
    return *static_cast<const ResultT*>(ptr.get());
}

template<class Renderer>
const RenderLightWorkT<Renderer>&
UpcastRenderLightWork(const std::unique_ptr<RenderLightWorkI>& ptr)
{
    using ResultT = RenderLightWorkT<Renderer>;
    return *static_cast<const ResultT*>(ptr.get());
}

template<class Renderer>
const RenderCameraWorkT<Renderer>&
UpcastRenderCameraWork(const std::unique_ptr<RenderCameraWorkI>& ptr)
{
    using ResultT = RenderCameraWorkT<Renderer>*;
    return *static_cast<const ResultT*>(ptr.get());
}

template<class Renderer>
const RenderMediumWorkT<Renderer>&
UpcastRenderMediumWork(const std::unique_ptr<RenderMediumWorkI>& ptr)
{
    using ResultT = RenderMediumWorkT<Renderer>*;
    return *static_cast<const ResultT*>(ptr.get());
}

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
// (i.e. PrimLight<Triangle>/Single, Skysphere/Identity),
// Work identifiers will be in range [0,7).
//
// Then, this class will dedicate 3 bits to distinguish the partitions
// and the rest of the bits (Given 32-bit key 29 bits) will be utilized
// for data coherency. Not all of the remaining bits will be utilized.
// Similary renderer will ask each group how many actual materials/primitives
// are present and utilize that information to minimize the usage.
// Additional heuristics may apply to reduce the partitioning time.
//
// These batch identifiers will be on an array, accelerator hit result
// (Material/Light/Camera)/Transform/Primitive will be hashed and used
// to lookup this array (either linear search or binary search).
//
class RenderSurfaceWorkHasher
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
                RenderSurfaceWorkHasher() = default;
    MRAY_HOST   RenderSurfaceWorkHasher(Span<CommonKey> dWorkBatchHashes,
                                        Span<CommonKey> dBatchIds);

    MRAY_HOST
    void PopulateHashesAndKeys(const TracerView& tracerView,
                               const RenderWorkList& curWorks,
                               const RenderLightWorkList& curLightWorks,
                               const RenderCameraWorkList& curCamWorks,
                               uint32_t maxRayCount,
                               const GPUQueue& queue);

    MR_HF_DECL
    Vector2ui WorkBatchDataRange() const;
    MR_HF_DECL
    Vector2ui WorkBatchBitRange() const;
    MR_HF_DECL
    CommonKey BisectBatchPortion(CommonKey key) const;
    MR_HF_DECL
    CommonKey HashWorkBatchPortion(HitKeyPack p) const;
    MR_HF_DECL
    CommonKey HashWorkDataPortion(HitKeyPack p, RayIndex i) const;
    MR_HF_DECL
    CommonKey GenerateWorkKeyGPU(HitKeyPack p, RayIndex i) const;
};

// Same as above but for media
class RenderMediumWorkHasher
{
    private:
    Span<CommonKey> dWorkBatchHashes;
    Span<CommonKey> dWorkBatchIds;
    uint32_t batchBits  = 0;
    uint32_t dataBits   = 0;
    // Maximum identifier count that is used by a single
    // group.
    uint32_t maxMediumIdBits = 0;
    uint32_t maxTransIdBits  = 0;
    uint32_t maxIndexBits    = 0;

    protected:
    public:
                RenderMediumWorkHasher() = default;
    MRAY_HOST   RenderMediumWorkHasher(Span<CommonKey> dWorkBatchHashes,
                                       Span<CommonKey> dBatchIds);

    MRAY_HOST
    void PopulateHashesAndKeys(const TracerView& tracerView,
                               const RenderMediumWorkList& curWorks,
                               uint32_t maxRayCount,
                               const GPUQueue& queue);

    MR_HF_DECL
    Vector2ui WorkBatchDataRange() const;
    MR_HF_DECL
    Vector2ui WorkBatchBitRange() const;
    MR_HF_DECL
    CommonKey BisectBatchPortion(CommonKey key) const;
    MR_HF_DECL
    CommonKey HashWorkBatchPortion(VolumeKeyPack p) const;
    MR_HF_DECL
    CommonKey HashWorkDataPortion(VolumeKeyPack p, RayIndex i) const;
    MR_HF_DECL
    CommonKey GenerateWorkKeyGPU(VolumeKeyPack p, RayIndex i) const;
};

class RendererBase : public RendererI
{
    public:
    using AttribInfoList    = typename RendererI::AttribInfoList;
    using WorkList          = RenderWorkList;
    using LightWorkList     = RenderLightWorkList;
    using CameraWorkList    = RenderCameraWorkList;

    static constexpr size_t SUB_CAMERA_BUFFER_SIZE = 1024;

    private:
    uint32_t         workCounter = 0;
    uint32_t         mediumWorkCounter = 0;
    RenderWorkPack   workPack;

    uint32_t        GenerateWorkMappings(uint32_t workIdStart);
    uint32_t        GenerateLightWorkMappings(uint32_t workIdStart);
    uint32_t        GenerateCameraWorkMappings(uint32_t workIdStart);
    //
    uint32_t        GenerateMediumWorkMappings(uint32_t workIdStart);

    protected:
    std::string_view            rendererName;
    ThreadPool&                 globalThreadPool;
    const GPUSystem&            gpuSystem;
    TracerView                  tracerView;
    const RenderImagePtr&       renderBuffer;
    Optional<CameraTransform>   cameraTransform = {};

    RenderWorkList          currentWorks;
    RenderLightWorkList     currentLightWorks;
    RenderCameraWorkList    currentCameraWorks;
    HitKeyPack              boundaryLightKeyPack;
    //
    RenderMediumWorkList    currentMediumWorks;
    // Current Canvas info
    ImageTiler              imageTiler;
    uint64_t                totalIterationCount;

    uint32_t                GenerateWorks();
    void                    ClearAllWorkMappings();
    RenderSurfaceWorkHasher InitializeSurfaceHashes(Span<CommonKey> dHashes,
                                                    Span<CommonKey> dWorkIds,
                                                    uint32_t maxRayCount,
                                                    const GPUQueue& queue);
    RenderMediumWorkHasher  InitializeMediumHashes(Span<CommonKey> dHashes,
                                                   Span<CommonKey> dWorkIds,
                                                   uint32_t maxRayCount,
                                                   const GPUQueue& queue);

    // Some common functions between renderer
    template<class Renderer, class WorkF, class LightWorkF = EmptyFunctor, class CamWorkF = EmptyFunctor>
    void IssueSurfaceWorkKernelsToPartitions(const RenderSurfaceWorkHasher&,
                                             const MultiPartitionOutput&,
                                             WorkF&&,
                                             LightWorkF&& = LightWorkF(),
                                             CamWorkF&&   = CamWorkF()) const;

    template<class Renderer, class WorkF>
    void IssueMediumWorkKernelsToPartitions(const RenderMediumWorkHasher&,
                                            const MultiPartitionOutput&,
                                            WorkF&&) const;

    public:
                        RendererBase(const RenderImagePtr&,
                                     const RenderWorkPack& workPacks,
                                     TracerView, const GPUSystem&,
                                     ThreadPool&, std::string_view rendererName);
    void                SetCameraTransform(const CameraTransform&) override;
    std::string_view    Name() const override;
};

inline
bool FlatSurfParams::operator<(const FlatSurfParams& right) const
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

    using T = Tuple<CommonKey, CommonKey, CommonKey>;
    return (T(GetMG(mId), GetTG(tId), GetPG(pId)) <
            T(GetMG(right.mId), GetTG(right.tId), GetPG(right.pId)));
}

inline
auto RenderWorkStruct::operator<=>(const RenderWorkStruct& right) const noexcept
{
    return workGroupId <=> right.workGroupId;
}

inline
auto RenderLightWorkStruct::operator<=>(const RenderLightWorkStruct& right) const noexcept
{
    return workGroupId <=> right.workGroupId;
}

inline
auto RenderCameraWorkStruct::operator<=>(const RenderCameraWorkStruct& right) const noexcept
{
    return workGroupId <=> right.workGroupId;
}

inline
auto RenderMediumWorkStruct::operator<=>(const RenderMediumWorkStruct& right) const noexcept
{
    return workGroupId <=> right.workGroupId;
}

MRAY_HOST inline
RenderSurfaceWorkHasher::RenderSurfaceWorkHasher(Span<CommonKey> dWBHashes,
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

MR_HF_DEF
Vector2ui RenderSurfaceWorkHasher::WorkBatchDataRange() const
{
    return Vector2ui(0, dataBits);
}

MR_HF_DEF
Vector2ui RenderSurfaceWorkHasher::WorkBatchBitRange() const
{
    return Vector2ui(dataBits, dataBits + batchBits);
}

MR_HF_DEF
CommonKey RenderSurfaceWorkHasher::BisectBatchPortion(CommonKey key) const
{
    return Bit::FetchSubPortion(key, {dataBits, dataBits + batchBits});
}

MR_HF_DEF
CommonKey RenderSurfaceWorkHasher::HashWorkBatchPortion(HitKeyPack p) const
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

MR_HF_DEF
CommonKey RenderSurfaceWorkHasher::HashWorkDataPortion(HitKeyPack p, RayIndex i) const
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
    // This is debatable, hence it is a heuristic.
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
        uint32_t bitsForItem = Math::Min(maxItemBitCount, remainingBits);
        if(bitsForItem == 0) return false;

        uint32_t start = currentBit;
        uint32_t end = currentBit + bitsForItem;
        assert(end <= sizeof(CommonKey) * CHAR_BIT);
        std::array range = {CommonKey(start), CommonKey(end)};
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

MR_HF_DEF
CommonKey RenderSurfaceWorkHasher::GenerateWorkKeyGPU(HitKeyPack p, RayIndex rayIndex) const
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


MRAY_HOST inline
RenderMediumWorkHasher::RenderMediumWorkHasher(Span<CommonKey> dWBHashes,
                                               Span<CommonKey> dWBIds)
    : dWorkBatchHashes(dWBHashes)
    , dWorkBatchIds(dWBIds)
    , batchBits(Bit::RequiredBitsToRepresent(static_cast<uint32_t>(dWBHashes.size())))
    , dataBits(sizeof(CommonKey) * CHAR_BIT - batchBits)
    , maxMediumIdBits(0)
    , maxTransIdBits(0)
{
    assert(dWBHashes.size() == dWBIds.size());
}

MR_HF_DEF
Vector2ui RenderMediumWorkHasher::WorkBatchDataRange() const
{
    return Vector2ui(0, dataBits);
}

MR_HF_DEF
Vector2ui RenderMediumWorkHasher::WorkBatchBitRange() const
{
    return Vector2ui(dataBits, dataBits + batchBits);
}

MR_HF_DEF
CommonKey RenderMediumWorkHasher::BisectBatchPortion(CommonKey key) const
{
    return Bit::FetchSubPortion(key, {dataBits, dataBits + batchBits});
}

MR_HF_DEF
CommonKey RenderMediumWorkHasher::HashWorkBatchPortion(VolumeKeyPack p) const
{
    static_assert(MediumKey::BatchBits + TransformKey::BatchBits
                  <= sizeof(CommonKey) * CHAR_BIT,
                  "Unable to pack batch bits for hasing!");
    // In common case, compose the batch identifiers
    CommonKey r = Bit::Compose<MediumKey::BatchBits, TransformKey::BatchBits>
    (
        p.medKey.FetchBatchPortion(),
        p.transKey.FetchBatchPortion()
    );
    return r;
}

MR_HF_DEF
CommonKey
RenderMediumWorkHasher::HashWorkDataPortion(VolumeKeyPack p, RayIndex i) const
{
    // Get the Id portions
    CommonKey mIndex = p.medKey.FetchIndexPortion();
    CommonKey tIndex = p.transKey.FetchIndexPortion();

    // Heuristic: Most important bits are the rayIndex,
    // then medium, then transform.
    //
    // This heruistic comes due to fact that writes dominate
    // the performance of surface sampling (see "RenderSurfaceWorkHasher")
    //
    // Since media sampling requires many/many read accesses
    // to the underlying data structure, we may implement
    // a bouqette sorting of the rays (spatio directional sorting)
    //
    // Since these keys are going to be used in radix sort, and this
    // part (data part) is just used for performance, we can
    // dynamically adjust this for highly heterogeneous media
    // etc. (We just need to transfer that heterogeneousity
    // here)
    //
    // We will see. (TODO: ...)
    uint32_t remainingBits = dataBits;
    uint32_t currentBit = 0;
    CommonKey result = 0;
    auto WriteKey = [&](uint32_t maxItemBitCount, CommonKey item) -> bool
    {
        uint32_t bitsForItem = Math::Min(maxItemBitCount, remainingBits);
        if(bitsForItem == 0) return false;

        uint32_t start = currentBit;
        uint32_t end = currentBit + bitsForItem;
        assert(end <= sizeof(CommonKey) * CHAR_BIT);
        std::array range = {CommonKey(start), CommonKey(end)};
        result = Bit::SetSubPortion(result, item, range);
        remainingBits -= bitsForItem;
        currentBit += bitsForItem;
        return true;
    };
    // Actual Payload index
    if(!WriteKey(maxIndexBits, i)) return result;
    // Materials
    if(!WriteKey(maxMediumIdBits, mIndex)) return result;
    // Transforms
    if(!WriteKey(maxTransIdBits, tIndex)) return result;

    return result;
}

MR_HF_DEF
CommonKey
RenderMediumWorkHasher::GenerateWorkKeyGPU(VolumeKeyPack p, RayIndex rayIndex) const
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

inline
void RendererBase::ClearAllWorkMappings()
{
    currentWorks.clear();
    currentLightWorks.clear();
    currentCameraWorks.clear();
    workCounter = 0;
}

inline
uint32_t RendererBase::GenerateWorks()
{
    assert(currentWorks.empty());
    assert(currentLightWorks.empty());
    assert(currentCameraWorks.empty());
    // Generate works per
    // Material1/Primitive/Transform triplet,
    // Light/Transform pair,
    // Camera/Transform pair
    workCounter = 0;
    workCounter = GenerateWorkMappings(workCounter);
    workCounter = GenerateLightWorkMappings(workCounter);
    workCounter = GenerateCameraWorkMappings(workCounter);
    workCounter = GenerateMediumWorkMappings(workCounter);
    return workCounter;
}

inline
RenderSurfaceWorkHasher
RendererBase::InitializeSurfaceHashes(Span<CommonKey> dHashes,
                                      Span<CommonKey> dWorkIds,
                                      uint32_t maxRayCount,
                                      const GPUQueue& queue)
{
    RenderSurfaceWorkHasher result(dHashes, dWorkIds);
    result.PopulateHashesAndKeys(tracerView,
                                 currentWorks,
                                 currentLightWorks,
                                 currentCameraWorks,
                                 maxRayCount,
                                 queue);
    return result;
}

inline
RenderMediumWorkHasher
RendererBase::InitializeMediumHashes(Span<CommonKey> dHashes,
                                     Span<CommonKey> dWorkIds,
                                     uint32_t maxRayCount,
                                     const GPUQueue& queue)
{
    RenderMediumWorkHasher result(dHashes, dWorkIds);
    result.PopulateHashesAndKeys(tracerView,
                                 currentMediumWorks,
                                 maxRayCount,
                                 queue);
    return result;
}

inline
RendererBase::RendererBase(const RenderImagePtr& rb,
                           const RenderWorkPack& wp,
                           TracerView tv, const GPUSystem& s,
                           ThreadPool& tp, std::string_view rendererName)
    : rendererName(rendererName)
    , workPack(wp)
    , globalThreadPool(tp)
    , gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)

{}

inline
void RendererBase::SetCameraTransform(const CameraTransform& ct)
{
    cameraTransform = ct;
}

inline
std::string_view RendererBase::Name() const
{
    return rendererName;
}

//=======================================//
// Helper functions to instantiate works //
//=======================================//
// TODO: Maybe add these on a namespace
template<class R, class Works, class LWorks,
         class CWorks, class MWorks>
struct RenderWorkTypePack
{
    using RendererType      = R;
    using WorkTypes         = Works;
    using LightWorkTypes    = LWorks;
    using CameraWorkTypes   = CWorks;
    using MediumWorkTypes   = MWorks;
};

template <class RenderWorkTypePackT>
inline void AddSingleRenderWork(Map<std::string_view, RenderWorkPack>& workMap,
                                RenderWorkTypePackT*)
{
    using RendererType      = typename RenderWorkTypePackT::RendererType;
    using WorkTypes         = typename RenderWorkTypePackT::WorkTypes;
    using LightWorkTypes    = typename RenderWorkTypePackT::LightWorkTypes;
    using CameraWorkTypes   = typename RenderWorkTypePackT::CameraWorkTypes;
    using MediumWorkTypes   = typename RenderWorkTypePackT::MediumWorkTypes;

    RenderWorkPack& workPack = workMap.emplace(RendererType::TypeName(),
                                               RenderWorkPack()).first->second;

    //================//
    // Material Works //
    //================//
    using WorkGenArgs = TypePack<const GenericGroupMaterialT&,
                                 const GenericGroupPrimitiveT&,
                                 const GenericGroupTransformT&,
                                 const GPUSystem&>;
    WorkGenArgs* workArgsResolver = nullptr;
    WorkTypes* workTypesResolver = nullptr;
    GenerateMapping<RenderWorkGenerator, RenderWorkI>
    (
        workPack.workMap,
        workArgsResolver,
        workTypesResolver
    );
    //================//
    //   Light Works  //
    //================//
    using LightWorkGenArgs = TypePack<const GenericGroupLightT&,
                                      const GenericGroupTransformT&,
                                      const GPUSystem&>;
    LightWorkGenArgs* lightWorkArgsResolver = nullptr;
    LightWorkTypes* lightWorkTypesResolver = nullptr;
    GenerateMapping<RenderLightWorkGenerator, RenderLightWorkI>
    (
        workPack.lightWorkMap,
        lightWorkArgsResolver,
        lightWorkTypesResolver
    );
    //================//
    //  Camera Works  //
    //================//
    using CameraWorkGenArgs = TypePack<const GenericGroupCameraT&,
                                       const GenericGroupTransformT&,
                                       const GPUSystem&>;
    CameraWorkGenArgs* cameraWorkArgsResolver = nullptr;
    CameraWorkTypes* cameraWorkTypesResolver = nullptr;
    GenerateMapping<RenderCameraWorkGenerator, RenderCameraWorkI>
    (
        workPack.camWorkMap,
        cameraWorkArgsResolver,
        cameraWorkTypesResolver
    );
    //================//
    //  Medium Works  //
    //================//
    using MediumWorkGenArgs = TypePack<const GenericGroupMediumT&,
                                       const GenericGroupTransformT&,
                                       const GPUSystem&>;
    MediumWorkGenArgs* medWorkArgsResolver = nullptr;
    MediumWorkTypes* medWorkTypesResolver = nullptr;
    GenerateMapping<RenderMediumWorkGenerator, RenderMediumWorkI>
    (
        workPack.mediumWorkMap,
        medWorkArgsResolver,
        medWorkTypesResolver
    );
}

template <class... Args>
void AddRenderWorks(Map<std::string_view, RenderWorkPack>& workMap,
                    TypePack<Args...>*)
{
    auto AddRenderWorksInternal =
    []<class TupleT, size_t... Is>(Map<std::string_view, RenderWorkPack>& workMap,
                                   TupleT*,
                                   std::index_sequence<Is...>)
    {
        // Param pack expansion over the index sequence
        (
            (AddSingleRenderWork<TypePackElement<Is, TupleT>>(workMap, nullptr)),
            ...
        );
    };

    TypePack<Args...>* list = nullptr;
    AddRenderWorksInternal(workMap, list,
                           std::index_sequence_for<Args...>{});
}
