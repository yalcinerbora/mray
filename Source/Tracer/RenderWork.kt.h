#pragma once

#include "RenderWork.h"

template<RendererC R, uint32_t I, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG,
         auto WorkFunction, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderWork(MRAY_GRID_CONSTANT const RenderWorkParamsR<R, I, PG, MG, TG> params)
{
    using SpectrumConv  = typename R::SpectrumConverterContext;
    // Define the types
    // First, this kernel uses a transform context.
    // This will be used to transform the primitive.
    // We may not be able act on the primitive without the primitive's data
    // (For example skinned meshes, we need weights and transform indices)
    // "TransformContext" abstraction handles these
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    // Similarly Renderer may not be RGB, but Material is
    // So it sends the spectrum converter context, which is either identity
    // or a function that is constructed via wavelengths etc.
    using Material = typename MG:: template Material<SpectrumConv>;
    // Primitive is straightforward but we get a type with the proper
    // transform context
    using Primitive = typename PG:: template Primitive<TransContext>;
    // And finally, the the definition of the surface that is dictated
    // by the material group
    using Surface = typename Material::Surface;
    // We need to compile-time check that this primitive supports such a surface
    static_assert(PrimitiveWithSurfaceC<Primitive, PG, Surface>,
                  "This primitive does not support the surface "
                  "required by a material");
    // The hit type of this specific primitive
    using Hit = typename Primitive::Hit;

    // Now finally we can start the runtime stuff
    uint32_t rayCount = static_cast<uint32_t>(params.in.dRayIndices.size());
    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId();
        globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(params.in.dRandomNumbers, globalId, rayCount);
        // Here is the potential drawback of sorting based
        // partitioning, all of the parameters are loaded
        // via scattered reads.
        //
        // Queue based partitioning does scattered writes
        // so we transferred the non-contigious memory I/O to
        // here
        // Ray
        RayIndex rIndex = params.in.dRayIndices[globalId];
        auto [ray, tMM] = RayFromGMem(params.in.dRays, rIndex);
        // Keys
        HitKeyPack keys = params.in.dKeys[rIndex];
        // Hit (TODO: What about single parameters?)
        Hit hit = params.in.dHits[rIndex].template AsVector<Hit::Dims>();
        // Advance the differential to the hit location
        RayCone rayCone = params.in.dRayCones[rIndex].Advance(tMM[1]);

        // Get instantiation of converter
        // TODO: Add spectrum related stuff, this should not be
        // default constructed
        typename SpectrumConv::Converter specConverter;
        // Create transform context
        TransContext tContext = GenerateTransformContext(params.transSoA,
                                                         params.primSoA,
                                                         keys.transKey,
                                                         keys.primKey);
        // Construct Primitive
        auto primitive = Primitive(tContext,
                                   params.primSoA,
                                   keys.primKey);
        // Generate the surface
        Surface surface;
        RayConeSurface rayConeSurface;
        primitive.GenerateSurface(surface, rayConeSurface,
                                  hit, ray, rayCone);
        // Generate the material
        auto material = Material(specConverter, surface, params.matSoA,
                                 std::bit_cast<MaterialKey>(keys.lightOrMatKey));
        // Call the function
        WorkFunction(primitive, material, surface, rayConeSurface,
                     tContext, rng, params, rIndex);
        // All Done!
    }
}

template<RendererC R, uint32_t I, LightGroupC LG, TransformGroupC TG,
         auto WorkFunction, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderLightWork(MRAY_GRID_CONSTANT const RenderLightWorkParamsR<R, I, LG, TG> params)
{
    using SpectrumConv = typename R::SpectrumConverterContext;
    //
    using PG = typename LG::PrimGroup;
    // Define the types
    // First, this kernel uses a transform context.
    // This will be used to transform the primitive.
    // We may not be able act on the primitive without the primitive's data
    // (For example skinned meshes, we need weights and transform indices)
    // "TransformContext" abstraction handles these
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    // Primitive is straightforward but we get a type with the proper
    // transform context
    using Primitive = typename PG:: template Primitive<TransContext>;
    // Light
    using Light = typename LG:: template Light<TransContext, SpectrumConv>;
    // Now finally we can start the runtime stuff
    uint32_t rayCount = static_cast<uint32_t>(params.in.dRayIndices.size());
    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId();
        globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(params.in.dRandomNumbers, globalId, rayCount);
        RayIndex rIndex = params.in.dRayIndices[globalId];
        // Keys
        HitKeyPack keys = params.in.dKeys[rIndex];
        LightKey lKey = LightKey::CombinedKey(keys.lightOrMatKey.FetchBatchPortion(),
                                              keys.lightOrMatKey.FetchIndexPortion());
        // Create transform context
        TransContext tContext = GenerateTransformContext(params.transSoA,
                                                         params.primSoA,
                                                         keys.transKey,
                                                         keys.primKey);

        // Get instantiation of converter
        // TODO: Add spectrum related stuff, this should not be
        // default constructed
        typename SpectrumConv::Converter specConverter;
        // Construct Primitive
        auto primitive = Primitive(tContext,
                                   params.primSoA,
                                   keys.primKey);
        // Construct light
        auto light = Light(specConverter, primitive, params.lightSoA, lKey);

        // Call the function
        WorkFunction(light, rng, params, rIndex);
        // All Done!
    }
}

template<RendererC R, uint32_t I, CameraGroupC CG, TransformGroupC TG,
         auto WorkFunction>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderCameraWork(MRAY_GRID_CONSTANT const RenderCameraWorkParamsR<R, I, CG, TG>)
{
    // TODO: This will be needed for light tracers
    // (Light -> Camera)
}