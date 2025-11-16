#pragma once

#include "RenderWork.h"

template<WorkFuncC WorkFunction,
         auto GenSpectrumConverter, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderWork(MRAY_GRID_CONSTANT const typename WorkFunction::Params params)
{
    using SpectrumConv = typename WorkFunction::SpectrumConv;
    // Define the types
    // First, this kernel uses a transform context.
    // This will be used to transform the primitive.
    // We may not be able act on the primitive without the primitive's data
    // (For example skinned meshes, we need weights and transform indices)
    // "TransformContext" abstraction handles these
    using TransContext = typename WorkFunction::TContext;
    // Similarly Renderer may not be RGB, but Material is
    // So it sends the spectrum converter context, which is either identity
    // or a function that is constructed via wavelengths etc.
    using Material = typename WorkFunction::Material;
    // Primitive is straightforward but we get a type with the proper
    // transform context
    using PG = typename WorkFunction::PG;
    using Primitive = typename WorkFunction::Primitive;
    // And finally, the the definition of the surface that is dictated
    // by the material group
    using Surface = typename WorkFunction::Surface;
    // We need to compile-time check that this primitive supports such a surface
    static_assert(PrimitiveWithSurfaceC<Primitive, PG, Surface>,
                  "This primitive does not support the surface "
                  "required by a material");
    // The hit type of this specific primitive
    using Hit = typename Primitive::Hit;

    // Now finally we can start the runtime stuff
    // Statically dedicate multiple threads per work
    // If requested by the
    KernelCallParams kp;
    static constexpr auto THREAD_PER_WARP = WorkFunction::THREAD_PER_WORK;
    uint32_t laneId          = kp.threadId    % THREAD_PER_WARP;
    uint32_t globalWarpId    = kp.GlobalId()  / THREAD_PER_WARP;
    uint32_t globalWarpCount = kp.TotalSize() / THREAD_PER_WARP;

    uint32_t rayCount = uint32_t(params.common.dRayIndices.size());
    for(uint32_t globalId = globalWarpId;
        globalId < rayCount; globalId += globalWarpCount)
    {
        RNGDispenser rng(params.common.dRandomNumbers, globalId, rayCount);
        // Here is the potential drawback of sorting based
        // partitioning, all of the parameters are loaded
        // via scattered reads.
        //
        // Queue based partitioning does scattered writes
        // so we transferred the non-contigious memory I/O to
        // here
        // Ray
        RayIndex rIndex = params.common.dRayIndices[globalId];
        auto [ray, tMM] = RayFromGMem(params.common.dRays, rIndex);
        // Keys
        HitKeyPack keys = params.common.dKeys[rIndex];
        // Hit (TODO: What about single parameters?)
        Hit hit = params.common.dHits[rIndex].template AsVector<Hit::Dims>();
        // Advance the differential to the hit location
        RayCone rayCone = params.common.dRayCones[rIndex].Advance(tMM[1]);

        // Get instantiation of converter
        // Converter is per ray, since it needs wavelengths of
        // the current trace, which is accessed by rIndex.
        SpectrumConv specConverter = GenSpectrumConverter(params, rIndex);

        // Create transform context
        TransContext tContext = GenerateTransformContext(params.transSoA,
                                                         params.primSoA,
                                                         keys.transKey,
                                                         keys.primKey);
        // Construct Primitive
        auto primitive = Primitive(tContext, params.primSoA, keys.primKey);
        auto matKey = Bit::BitCast<MaterialKey>(keys.lightOrMatKey);
        NormalMap normalMap = Material::GetNormalMap(params.matSoA, matKey);
        // Generate the surface
        Surface surface;
        RayConeSurface rayConeSurface;
        primitive.GenerateSurface(surface, rayConeSurface,
                                  normalMap, hit, ray, rayCone);
        // Generate the material
        auto material = Material(specConverter, surface,
                                 params.matSoA, matKey);
        // Call the function
        WorkFunction::Call(primitive, material, surface, rayConeSurface,
                           tContext, specConverter, rng, params, rIndex,
                           laneId);
        // All Done!
    }
}

template<LightWorkFuncC WorkFunction,
         auto GenSpectrumConverter, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderLightWork(MRAY_GRID_CONSTANT const typename WorkFunction::Params params)
{
    using SpectrumConv = typename WorkFunction::SpectrumConv;
    //
    using PG = typename WorkFunction::PG;
    // Define the types
    // First, this kernel uses a transform context.
    // This will be used to transform the primitive.
    // We may not be able act on the primitive without the primitive's data
    // (For example skinned meshes, we need weights and transform indices)
    // "TransformContext" abstraction handles these
    using TransContext = typename WorkFunction::TContext;
    // Primitive is straightforward but we get a type with the proper
    // transform context
    using Primitive = typename WorkFunction::Primitive;
    // Light
    using Light = typename WorkFunction::Light;

    // Statically dedicate multiple threads per work
    // If requested by the
    KernelCallParams kp;
    static constexpr auto THREAD_PER_WARP = WorkFunction::THREAD_PER_WORK;
    uint32_t laneId          = kp.threadId    % THREAD_PER_WARP;
    uint32_t globalWarpId    = kp.GlobalId()  / THREAD_PER_WARP;
    uint32_t globalWarpCount = kp.TotalSize() / THREAD_PER_WARP;

    uint32_t rayCount = uint32_t(params.common.dRayIndices.size());
    for(uint32_t globalId = globalWarpId;
        globalId < rayCount; globalId += globalWarpCount)
    {
        RNGDispenser rng(params.common.dRandomNumbers, globalId, rayCount);
        RayIndex rIndex = params.common.dRayIndices[globalId];
        // Keys
        HitKeyPack keys = params.common.dKeys[rIndex];
        LightKey lKey = LightKey::CombinedKey(keys.lightOrMatKey.FetchBatchPortion(),
                                              keys.lightOrMatKey.FetchIndexPortion());
        // Create transform context
        TransContext tContext = GenerateTransformContext(params.transSoA,
                                                         params.primSoA,
                                                         keys.transKey,
                                                         keys.primKey);

        // Get instantiation of converter
        // Converter is per ray, since it needs wavelengths of
        // the current trace, which is accessed by rIndex.
        SpectrumConv specConverter = GenSpectrumConverter(params, rIndex);

        // Construct Primitive
        auto primitive = Primitive(tContext,
                                   params.primSoA,
                                   keys.primKey);
        // Construct light
        auto light = Light(specConverter, primitive, params.lightSoA, lKey);

        // Call the function
        WorkFunction::Call(light, rng, specConverter, params, rIndex, laneId);
        // All Done!
    }
}

template<CamWorkFuncC WorkFunction>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderCameraWork(MRAY_GRID_CONSTANT const typename WorkFunction::Params)
{
    // TODO: This will be needed for light tracers
    // (Light -> Camera)
}

template<class MediumWorkFunction,
         auto GenSpectrumConverter, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCMediumWork(MRAY_GRID_CONSTANT const typename MediumWorkFunction::Params params)
{
    //using TransContext = typename MediumWorkFunction::TContext;
    //using Medium       = typename MediumWorkFunction::Medium;
    //using SpectrumConv = typename MediumWorkFunction::SpectrumConv;

    //// Work dedication
    //KernelCallParams kp;
    //static constexpr auto THREAD_PER_WARP = MediumWorkFunction::THREAD_PER_WORK;
    //uint32_t laneId          = kp.threadId % THREAD_PER_WARP;
    //uint32_t globalWarpId    = kp.GlobalId() / THREAD_PER_WARP;
    //uint32_t globalWarpCount = kp.TotalSize() / THREAD_PER_WARP;

    //uint32_t rayCount = uint32_t(params.renderState.dRayIndices.size());
    //for(uint32_t globalId = globalWarpId;
    //    globalId < rayCount; globalId += globalWarpCount)
    //{
    //    RNGDispenser rng(params.common.dRandomNumbers, globalId, rayCount);
    //    RayIndex rIndex = params.common.dRayIndices[globalId];
    //    InterfaceKeyPack keys = params.common.dKeys[rIndex];

    //    // Create transform context
    //    TransContext tContext = GenerateTransformContext(params.transSoA,
    //                                                     EmptyType{},
    //                                                     keys.transKey,
    //                                                     PrimitiveKey(0));
    //    SpectrumConv specConverter = GenSpectrumConverter(params, rIndex);
    //    Medium medium = Medium(specConverter, params.medSoA, keys.medKey);

    //    MediumWorkFunction::Call(medium, tContext, specConverter, rng,
    //                             params, rIndex, laneId);
    //}
}