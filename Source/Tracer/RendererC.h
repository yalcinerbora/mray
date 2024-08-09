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
    const TextureMap&                                       textures;
    const TextureViewMap&                                   textureViews;
    const TracerParameters&                                 tracerParams;
    const std::vector<Pair<SurfaceId, SurfaceParams>>&              surfs;
    const std::vector<Pair<LightSurfaceId, LightSurfaceParams>>&    lightSurfs;
    const std::vector<Pair<CamSurfaceId, CameraSurfaceParams>>&     camSurfs;
    const std::vector<FlatSurfParams>&                              flattenedSurfaces;
};

class RenderWorkI
{
    public:
    virtual ~RenderWorkI() = default;

    virtual std::string_view Name() = 0;
};

class RenderCameraWorkI
{
    public:
    virtual ~RenderCameraWorkI() = default;

    virtual std::string_view Name() = 0;
};

class RenderLightWorkI
{
    public:
    virtual ~RenderLightWorkI() = default;

    virtual std::string_view Name() = 0;
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
    RendererType(RenderImagePtr{}, RenderWorkPack{}, tv, gpuSystem);
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

using RenderWorkMap = std::map<Tuple<MatGroupId, PrimGroupId, TransGroupId>, RenderWorkPtr>;
using RenderLightWorkMap = std::map<Pair<LightGroupId, TransGroupId>, RenderLightWorkPtr>;
using RenderCameraWorkMap = std::map<Pair<CameraGroupId, TransGroupId>, RenderCameraWorkPtr>;

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

template <class Child>
class RendererT : public RendererI
{
    public:
    using AttribInfoList    = typename RendererI::AttribInfoList;
    using MetaHit           = MetaHitT<TracerConstants::MaxHitFloatCount>;

    private:
    protected:
    const GPUSystem&        gpuSystem;
    TracerView              tracerView;
    const RenderImagePtr&   renderBuffer;
    bool                    rendering = false;
    RenderWorkPack          workPack;
    //
    RenderWorkMap           currentWorks;
    RenderLightWorkMap      currentLightWorks;
    RenderCameraWorkMap     currentCameraWorks;

    // Current Canvas info
    MRayColorSpaceEnum      curColorSpace;
    Vector2ui               curFramebufferSize;
    Vector2ui               curFBMin;
    Vector2ui               curFBMax;

    void                    GenerateWorkMappings();
    void                    GenerateLightWorkMappings();
    void                    GenerateCameraWorkMappings();

    public:
                        RendererT(const RenderImagePtr&,
                                  const RenderWorkPack& workPacks,
                                  TracerView, const GPUSystem&);
    std::string_view    Name() const override;
};

inline bool FlatSurfParams::operator<(const FlatSurfParams& right) const
{
    return (Tuple(mId, tId, pId) <
            Tuple(right.mId, right.tId, right.pId));
}

template <class C>
void RendererT<C>::GenerateWorkMappings()
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
        PrimGroupId pgId{PrimBatchKey(uint32_t(flatSurfs[i].mId)).FetchBatchPortion()};
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
        // Put this ptr somewhere... safe
        currentWorks.try_emplace(Tuple(mgId, pgId, tgId), std::move(ptr));
    }
}

template <class C>
void RendererT<C>::GenerateLightWorkMappings()
{
    using Algo::PartitionRange;
    const auto& lightSurfs = tracerView.lightSurfs;
    using LightSurfP = Pair<LightSurfaceId, LightSurfaceParams>;
    auto LightSurfIsLess = [](const LightSurfP& left, const LightSurfP& right)
    {
        return (Tuple(left.second.lightId, left.second.transformId) <
                Tuple(right.second.lightId, right.second.transformId));
    };
    assert(std::is_sorted(lightSurfs.cbegin(), lightSurfs.cend(),
                          LightSurfIsLess));

    auto partitions = Algo::PartitionRange(lightSurfs.cbegin(), lightSurfs.cend(),
                                           LightSurfIsLess);
    for(const auto& p : partitions)
    {
        size_t i = p[0];
        LightGroupId lgId{LightKey(uint32_t(lightSurfs[i].second.lightId)).FetchBatchPortion()};
        TransGroupId tgId{TransformKey(uint32_t(lightSurfs[i].second.transformId)).FetchBatchPortion()};
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
        RenderLightWorkGenerator generator = loc->get();
        RenderLightWorkPtr ptr = generator(*lg.get(), *tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentLightWorks.try_emplace(Pair(lgId, tgId), std::move(ptr));
    }
}

template <class C>
void RendererT<C>::GenerateCameraWorkMappings()
{
    const auto& camSurfs = tracerView.camSurfs;
    using CamSurfP = Pair<CamSurfaceId, CameraSurfaceParams>;
    auto CamSurfIsLess = [](const CamSurfP& left, const CamSurfP& right)
    {
        return (Tuple(left.second.cameraId, left.second.transformId) <
                Tuple(right.second.cameraId, right.second.transformId));
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
        // Put this ptr somewhere... safe
        currentCameraWorks.try_emplace(Pair(cgId, tgId), std::move(ptr));

    }
}

template <class C>
RendererT<C>::RendererT(const RenderImagePtr& rb,
                        const RenderWorkPack& wp,
                        TracerView tv, const GPUSystem& s)
    : gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)
    , workPack(wp)
{}

template <class C>
std::string_view RendererT<C>::Name() const
{
    return C::TypeName();
}

inline Vector2ui FindOptimumTile(Vector2ui fbSize,
                                 uint32_t parallelizationHint)
{
    using namespace MathFunctions;
    // Start with an ~ aspect ratio tile
    // and adjust it
    Float aspectRatio = Float(fbSize[0]) / Float(fbSize[1]);
    Float factor = std::sqrt(Float(parallelizationHint) / aspectRatio);
    Vector2ui tileHint(std::round(aspectRatio * factor), std::roundf(factor));

    // Find optimal tile size that evenly divides the image
    // This may not happen (i.e., width or height is prime)
    // then expand the tile size to pass the edge barely.
    auto Adjust = [&](uint32_t i)
    {
        // If w/h is small use the full fb w/h
        if(fbSize[i] < tileHint[i]) return fbSize[i];

        // Divide down to get an agressive (lower) count,
        // but on second pass do a conservative divide
        Float tileCountF = Float(fbSize[i]) / Float(tileHint[i]);
        uint32_t tileCount = uint32_t(std::round(tileCountF));
        // Try to minimize residuals so that
        // GPU does consistent work
        uint32_t result = fbSize[i] / tileCount;
        uint32_t residual = fbSize[i] % tileCount;
        residual = DivideUp(residual, tileCount);
        result += residual;
        return result;
    };

    Vector2ui result = Vector2ui(Adjust(0), Adjust(1));
    return result;
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