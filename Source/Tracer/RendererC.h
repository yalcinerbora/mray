#pragma once


#include "TracerTypes.h"
#include "RenderImageBuffer.h"
#include "GenericGroup.h"

#include "Core/TracerI.h"
#include "Core/DataStructures.h"

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
    const TextureViewMap&                                   textureViews;
    const TracerParameters&                                 tracerParams;
    const std::vector<Pair<SurfaceId, SurfaceParams>>&              surfs;
    const std::vector<Pair<LightSurfaceId, LightSurfaceParams>>&    lightSurfs;
    const std::vector<Pair<CamSurfaceId, CameraSurfaceParams>>&     camSurfs;
};

using RenderImagePtr = std::shared_ptr<RenderImage>;

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

    //
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


// Render work of camera
template <class CamGroup>
class RenderBoundaryWork
{};

//// Boundary work of light group
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
//

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
                                            const CameraKey&) = 0;
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
    using AttribInfoList = typename RendererI::AttribInfoList;
    private:
    protected:
    const GPUSystem&    gpuSystem;
    TracerView          tracerView;
    RenderImagePtr      renderBuffer;
    bool                rendering = false;

    public:
                        RendererT(RenderImagePtr, TracerView,
                                  const GPUSystem&);
    std::string_view    Name() const override;
};

template <class C>
RendererT<C>::RendererT(RenderImagePtr rb,
                        TracerView tv,
                        const GPUSystem& s)
    : gpuSystem(s)
    , tracerView(tv)
    , renderBuffer(rb)
{}

template <class C>
std::string_view RendererT<C>::Name() const
{
    return C::TypeName();
}