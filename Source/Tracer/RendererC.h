#pragma once


#include "TracerTypes.h"

template<class Renderer>
concept RendererC = requires()
{
    // Global State
    // These parameters are work agnostic.
    typename Renderer::GlobalState;
    // These parameters are work related.
    // (Not used but exposed for future use maybe?)
    typename Renderer::LocalState;
    //
    typename Renderer::RayState;
    //
    typename Renderer::RayPayload;
};

// Render work of camera
template <class CamGroup>
class RenderBoundaryWork
{};

// Boundary work of light group
template <class LightGroup>
class RenderBoundaryWork
{};

// Render work of surfaces
template <class Renderer, class PrimGroup,
          class MatGroup, class TransGroup,
          auto WorkFunction>
class RendererWork
{
    using RayPayload    = typename Renderer::RayPayload;
    using GlobalState   = typename Renderer::GlobalState;
    using LocalState    = typename Renderer::LocalState;
    using RayState      = typename Renderer::RayState;

    using PrimSoA       = typename PrimGroup::DataSoA;
    using MatSoA        = typename MatGroup::DataSoA;
    using TransSoA      = typename TransGroup::DataSoA;

    using SpectrumConv  = typename Renderer::SpectrumConverterContext;

    struct Outputs
    {
        WorkId*         gWorkIds;
        RayGMem*        gRays;
        RayPayload      gPayload;
        uint32_t        outSize;
        //

    };

    struct Inputs
    {
        // Sorted
        const RayIndex*     gRayInIndices;
        // Unsorte;
        const RayGMem*      gRaysIn;
        const HitIdPack*    gRayIds;
        const MetaHitPtr    gHitParams;
        const RayPayload&   payloadIn;
        uint32_t            inSize;
    };

    private:
    Span<Outputs, 1>    gInputSoA;
    Span<Inputs, 1>     gOutputSoA;

    // Host
    const RNGDispenser* gRNGDispenser;
    LocalState          perWorkLocalState;
    GlobalState         renererGlobalState;

    MRAY_HYBRID
    void operator()(KernelCallParams kcParams) const
    {
        // Compile-time find the transform generator function and return type
        constexpr auto TContextGen = AcquireTransformContextGenerator<PrimGroup, TransGroup>();
        constexpr auto TGenFunc = decltype(TContextGen)::Function;
        // Define the types
        // First, this kernel uses a transform context
        // that this primitive group provides to generate a surface
        using TContextType = typename decltype(TContextGen)::ReturnType;
        // And this material that converts the spectrum type to the renderer
        // required spectrum type
        using Material = typename MatGroup:: template Material<SpectrumConv>;
        // And this primitive, that accepts the generated transform context
        // to generate a surface in tangent space
        using Primitive = typename PrimGroup:: template Primitive<TContextType>;
        // And finally, the material acts on this surface
        using Surface = Material::Surface;
        // Compile-time check that this primitive supports this surface
        static_assert(PrimWithSurfaceC<Primitive, PrimGroup, Surface>,
                      "This primitive does not support the surface required by a material");
        // The hit type of this specific primitive
        using Hit = typename Primitive::Hit;
        // Get instantiation of converter
        typename SpectrumConv::Converter specConverter;

        // Actual mock work
        // Assume i is different per thread
        uint32_t i = kcParams.GlobalID();

        // Actually load per-work data
        // Indices (Ids)
        PrimitiveKey pId         = std::get<0>(gRayIds[i]);
        MaterialKey  matId       = std::get<1>(gRayIds[i]);
        TransformKey tId         = std::get<2>(gRayIds[i]);
        MediumKey    mediumId    = std::get<3>(gRayIds[i]);
        // Ray & "Ray Hit"
        RayReg  rayReg(gRaysIn, gRayInIndices[i]);
        Hit hit = gHitParams.Ref<Hit>(i);
        // The transform context
        TContextType transformContext = TGenFunc(transformSoA, primSoA, tId, pId);
        // Convert ray to tangent space
        rayReg.r = transformContext.InvApply(rayReg.r);
        // Construct Primitive
        Primitive prim(transformContext, primSoA, pId);
        // Generate the surface
        Surface s;
        prim.GenerateSurface(s, hit, rayReg.r, DiffRay{});
        // Generate the material
        Material m(specConverter, matSoA, matId);

        // Call the function
        Work(prim, m, input, output, rng,);
    }
};

// Render work of materials
template <class MediumGroup>
class RenderWork
{};