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

using RenderImagePtr = std::unique_ptr<RenderImage>;

template<class RendererType>
concept RendererC = requires(RendererType rt,
                             TransientData input,
                             const GPUQueue& q)
{
    // Global State
    // These parameters are work agnostic.
    typename RendererType::GlobalState;
    // These parameters are work related.
    // (Not used but exposed for future use maybe?)
    typename RendererType::LocalState;
    typename RendererType::RayState;
    typename RendererType::RayPayload;

    // Host side things
    typename RendererType::Id;
    typename RendererType::IdInt;
    typename RendererType::AttribInfoList;

    // GPU Side
    { RendererType::InitRayState(RaySample{})
    } -> std::same_as<typename RendererType::RayState>;

    // CPU Side
    {rt.Commit()} -> std::same_as<void>;
    {rt.IsInCommitState()} -> std::same_as<bool>;
    {rt.AttributeInfo()
    } -> std::same_as<typename RendererType::AttribInfoList>;
    {rt.PushAttribute(uint32_t{}, std::move(input), q)
    } -> std::same_as<void>;
    {rt.StartRender(CamSurfaceId{}, RenderImageParams{})
    } ->std::same_as<void>;
    {rt.StopRender()} ->std::same_as<void>;

    // Can query the type
    {rt.Name()} -> std::same_as<std::string_view>;
    {RendererType::TypeName()} -> std::same_as<std::string_view>;
};

class RenderWorkI
{};

class RenderCameraWorkI
{};

class RenderLightWorkI
{};

using RenderWorkPtr = std::unique_ptr<RenderWorkI>;
using RenderCameraWorkPtr = std::unique_ptr<RenderCameraWorkI>;
using RenderLightWorkPtr = std::unique_ptr<RenderLightWorkI>;

// Render work of camera
//template <class CamGroup>
//class RenderBoundaryWork
//{};

// Boundary work of light group
//template <class LightGroup>
//class RenderBoundaryWork
//{};

//// Render work of surfaces
//template <class Renderer, class PrimGroup,
//          class MatGroup, class TransGroup,
//          auto WorkFunction>
//class RendererWork
//{
//    using RayPayload    = typename Renderer::RayPayload;
//    using GlobalState   = typename Renderer::GlobalState;
//    using LocalState    = typename Renderer::LocalState;
//    using RayState      = typename Renderer::RayState;
//
//    using PrimSoA       = typename PrimGroup::DataSoA;
//    using MatSoA        = typename MatGroup::DataSoA;
//    using TransSoA      = typename TransGroup::DataSoA;
//
//    using SpectrumConv  = typename Renderer::SpectrumConverterContext;
//
//    struct Outputs
//    {
//        WorkId*         gWorkIds;
//        RayGMem*        gRays;
//        RayPayload      gPayload;
//        uint32_t        outSize;
//        //
//
//    };
//
//    struct Inputs
//    {
//        // Sorted
//        const RayIndex*     gRayInIndices;
//        // Unsorte;
//        const RayGMem*      gRaysIn;
//        const HitIdPack*    gRayIds;
//        const MetaHitPtr    gHitParams;
//        const RayPayload&   payloadIn;
//        uint32_t            inSize;
//    };
//
//    private:
//    Span<Outputs, 1>    gInputSoA;
//    Span<Inputs, 1>     gOutputSoA;
//
//    // Host
//    const RNGDispenser* gRNGDispenser;
//    LocalState          perWorkLocalState;
//    GlobalState         renererGlobalState;
//
//    MRAY_HYBRID
//    void operator()(KernelCallParams kcParams) const
//    {
//        // Compile-time find the transform generator function and return type
//        constexpr auto TContextGen = AcquireTransformContextGenerator<PrimGroup, TransGroup>();
//        constexpr auto TGenFunc = decltype(TContextGen)::Function;
//        // Define the types
//        // First, this kernel uses a transform context
//        // that this primitive group provides to generate a surface
//        using TContextType = typename decltype(TContextGen)::ReturnType;
//        // And this material that converts the spectrum type to the renderer
//        // required spectrum type
//        using Material = typename MatGroup:: template Material<SpectrumConv>;
//        // And this primitive, that accepts the generated transform context
//        // to generate a surface in tangent space
//        using Primitive = typename PrimGroup:: template Primitive<TContextType>;
//        // And finally, the material acts on this surface
//        using Surface = Material::Surface;
//        // Compile-time check that this primitive supports this surface
//        static_assert(PrimWithSurfaceC<Primitive, PrimGroup, Surface>,
//                      "This primitive does not support the surface required by a material");
//        // The hit type of this specific primitive
//        using Hit = typename Primitive::Hit;
//        // Get instantiation of converter
//        typename SpectrumConv::Converter specConverter;
//
//        // Actual mock work
//        // Assume i is different per thread
//        uint32_t i = kcParams.GlobalID();
//
//        // Actually load per-work data
//        // Indices (Ids)
//        PrimitiveKey pId         = std::get<0>(gRayIds[i]);
//        MaterialKey  matId       = std::get<1>(gRayIds[i]);
//        TransformKey tId         = std::get<2>(gRayIds[i]);
//        MediumKey    mediumId    = std::get<3>(gRayIds[i]);
//        // Ray & "Ray Hit"
//        RayReg  rayReg(gRaysIn, gRayInIndices[i]);
//        Hit hit = gHitParams.Ref<Hit>(i);
//        // The transform context
//        TContextType transformContext = TGenFunc(transformSoA, primSoA, tId, pId);
//        // Convert ray to tangent space
//        rayReg.r = transformContext.InvApply(rayReg.r);
//        // Construct Primitive
//        Primitive prim(transformContext, primSoA, pId);
//        // Generate the surface
//        Surface s;
//        prim.GenerateSurface(s, hit, rayReg.r, DiffRay{});
//        // Generate the material
//        Material m(specConverter, matSoA, matId);
//
//        // Call the function
//        Work(prim, m, input, output, rng,);
//    }
//};


using RenderWorkGenerator = GeneratorFuncType<RenderWorkI,
                                              GenericGroupMaterialT*,
                                              GenericGroupPrimitiveT*,
                                              GenericGroupTransformT*,
                                              const GPUSystem&>;
using RenderLightWorkGenerator = GeneratorFuncType<RenderLightWorkI,
                                                   GenericGroupLightT*,
                                                   GenericGroupTransformT*,
                                                   const GPUSystem&>;
using RenderCameraWorkGenerator = GeneratorFuncType<RenderCameraWorkI,
                                                    GenericGroupCameraT*,
                                                    GenericGroupTransformT*,
                                                    const GPUSystem&>;

using WorkGenMap = Map<std::string_view, RenderWorkGenerator>;
using LightWorkGenMap = Map<std::string_view, RenderLightWorkGenerator>;
using CamWorkGenMap = Map<std::string_view, RenderCameraWorkGenerator>;

using WorkPack = Tuple<WorkGenMap, LightWorkGenMap, CamWorkGenMap>;
using WorkMap = std::map<Tuple<MatGroupId, PrimGroupId, TransGroupId>, RenderWorkPtr>;
using LightWorkMap = std::map<Pair<LightGroupId, TransGroupId>, RenderLightWorkPtr>;
using CameraWorkMap = std::map<Pair<CameraGroupId, TransGroupId>, RenderCameraWorkPtr>;

class RendererI
{
    public:
    using AttribInfoList = RendererAttributeInfoList;

    public:
    virtual     ~RendererI() = default;

    // Interface
    virtual MRayError       Commit() = 0;
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
    static constexpr uint32_t MaxWorkPacks = 16;

    using AttribInfoList = typename RendererI::AttribInfoList;
    using WorkPacks = StaticVector<WorkPack, MaxWorkPacks>;

    private:
    protected:
    const GPUSystem&        gpuSystem;
    TracerView              tracerView;
    const RenderImagePtr&   renderBuffer;
    bool                    rendering = false;
    WorkPacks               workPacks;
    //
    WorkMap                 currentWorks;
    LightWorkMap            currentLightWorks;
    CameraWorkMap           currentCameraWorks;

    // Current Canvas info
    MRayColorSpaceEnum      curColorSpace;
    Vector2ui               curFramebufferSize;
    Vector2ui               curFBMin;
    Vector2ui               curFBMax;

    void                    GenerateWorkMappings(uint32_t packIndex);
    void                    GenerateLightWorkMappings(uint32_t packIndex);
    void                    GenerateCameraWorkMappings(uint32_t packIndex);

    public:
                        RendererT(const RenderImagePtr&,
                                  TracerView, const GPUSystem&);
                        RendererT(const RenderImagePtr&,
                                  const WorkPacks& workPacks,
                                  TracerView, const GPUSystem&);
    std::string_view    Name() const override;
};

inline bool FlatSurfParams::operator<(const FlatSurfParams& right) const
{
    return (Tuple(mId, tId, pId) <
            Tuple(right.mId, right.tId, right.pId));
}

template <class C>
void RendererT<C>::GenerateWorkMappings(uint32_t packIndex)
{
    WorkPack workPack = workPacks[packIndex];
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
        RenderWorkPtr ptr = generator(mg.get(), pg.get(), tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentWorks.try_emplace(Tuple(mgId, pgId, tgId), std::move(ptr));
    }
}

template <class C>
void RendererT<C>::GenerateLightWorkMappings(uint32_t packIndex)
{
    WorkPack workPack = workPacks[packIndex];

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
        RenderLightWorkPtr ptr = generator(lg.get(), tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentLightWorks.try_emplace(Pair(lgId, tgId), std::move(ptr));
    }
}

template <class C>
void RendererT<C>::GenerateCameraWorkMappings(uint32_t packIndex)
{
    WorkPack workPack = workPacks[packIndex];
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
        RenderCameraWorkPtr ptr = generator(cg.get(), tg.get(), gpuSystem);
        // Put this ptr somewhere... safe
        currentCameraWorks.try_emplace(Pair(cgId, tgId), std::move(ptr));

    }
}

template <class C>
RendererT<C>::RendererT(const RenderImagePtr& rb,
                        TracerView tv, const GPUSystem& s)
    : gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)
{}

template <class C>
RendererT<C>::RendererT(const RenderImagePtr& rb,
                        const WorkPacks& wp,
                        TracerView tv, const GPUSystem& s)
    : gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)
    , workPacks(wp)
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
