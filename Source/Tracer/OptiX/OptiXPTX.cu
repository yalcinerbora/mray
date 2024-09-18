#include "OptiXPTX.h"

#include "Tracer/PrimitiveDefaultTriangle.h"

#include "Core/BitFunctions.h"

// ExternCWrapper Macro
#define WRAP_FUCTION(NAME, FUNCTION) \
    extern "C" __global__ void NAME(){FUNCTION();}

extern "C" __constant__ ArgumentPackOpitX params;

template <VectorC Hit>
requires std::is_floating_point_v<typename Hit::InnerType>
MRAY_GPU MRAY_GPU_INLINE
MetaHit ReadHitFromAttributes()
{
    static_assert(Hit::Dims >= MetaHit::MaxDim,
                  "Could not fit hit into meta hit!");
    Hit h = Hit::Zero();
    if constexpr(4 <= Hit::Dims)
        h[3] = std::bit_cast<float>(optixGetAttribute_3());
    if constexpr(3 <= Hit::Dims)
        h[2] = std::bit_cast<float>(optixGetAttribute_2());
    if constexpr(2 <= Hit::Dims)
        h[1] = std::bit_cast<float>(optixGetAttribute_1());
    if constexpr(1 <= Hit::Dims)
        h[0] = std::bit_cast<float>(optixGetAttribute_0());
    return MetaHit(h);
}

template <VectorC Hit>
requires std::is_floating_point_v<typename Hit::InnerType>
MRAY_GPU MRAY_GPU_INLINE
void ReportIntersection(float newT, unsigned int kind, Hit h)
{
    // Pre-check the Empty (C++ sizeof empty struct is 1
    // so this should never be branched)
    // But on device maybe it is different ??
    if constexpr(1 == Hit::Dims)
        optixReportIntersection(newT, kind,
                                std::bit_cast<uint32_t>(float(h[0])));
    else if constexpr(2 == Hit::Dims)
        optixReportIntersection(newT, kind,
                                std::bit_cast<uint32_t>(float(h[0])),
                                std::bit_cast<uint32_t>(float(h[1])));
    else if constexpr(3 == Hit::Dims)
        optixReportIntersection(newT, kind,
                                std::bit_cast<uint32_t>(float(h[0])),
                                std::bit_cast<uint32_t>(float(h[1])),
                                std::bit_cast<uint32_t>(float(h[2])));
    else if constexpr(4 == Hit::Dims)
        optixReportIntersection(newT, kind,
                                std::bit_cast<uint32_t>(float(h[0])),
                                std::bit_cast<uint32_t>(float(h[1])),
                                std::bit_cast<uint32_t>(float(h[2])),
                                std::bit_cast<uint32_t>(float(h[3])));
}

MRAY_GPU MRAY_GPU_INLINE
void SetRayIndexAsPayload(RayIndex rIndex)
{
    // Sanity check
    static_assert(std::is_same_v<RayIndex, unsigned int>);
    optixSetPayload_0(rIndex);
}

MRAY_GPU MRAY_GPU_INLINE
void SetRNGStateAsPayload(BackupRNGState s)
{
    // TODO: We use PCG32 as a backup generator,
    // it has 32-bit state. So directly writing it as payload
    static_assert(std::is_same_v<BackupRNGState, unsigned int>);
    optixSetPayload_1(s);
}

MRAY_GPU MRAY_GPU_INLINE
RayIndex GetRayIndexFromPayload()
{
    return RayIndex(optixGetPayload_0());
}

MRAY_GPU MRAY_GPU_INLINE
BackupRNGState GetRNGStateFromPayload()
{
    // TODO: We use PCG32 as a backup generator,
    // it has 32-bit state. So directly writing it as payload
    return BackupRNGState(optixGetPayload_1());
}

// Meta Closest Hit Shader
template<PrimitiveGroupC PGroup>
MRAY_GPU MRAY_GPU_INLINE
void KCClosestHit()
{
    using Hit = typename PGroup::Hit;
    using HitRecord = Record<typename PGroup::DataSoA>;
    const void* hrRaw = std::bit_cast<void*>(optixGetSbtDataPointer());
    const HitRecord& record = *reinterpret_cast<const HitRecord*>(hrRaw);

    const uint32_t leafId = optixGetPrimitiveIndex();
    const uint32_t rayId = optixGetLaunchIndex().x;

    // Fetch the workKey, transformId, primitiveId from table
    PrimitiveKey pKey = record.dPrimKeys[leafId];
    TransformKey tKey = record.transformKey;
    LightOrMatKey lmKey = record.materialKey;
    AcceleratorKey aKey = record.acceleratorKey;
    MetaHit hit = ReadHitFromAttributes<Hit>();

    // Write to the global memory
    RayIndex rIndex = params.dRayIndices[rayId];
    params.dHitKeys[rIndex] = HitKeyPack
    {
        .primKey = pKey,
        .lightOrMatKey = lmKey,
        .transKey = tKey,
        .accelKey = aKey
    };
    params.dHits[rIndex] = hit;
    params.dRays[rIndex].tMax = optixGetRayTmax();
}

// Meta Any Hit Shader
template<PrimitiveGroupC PGroup>
MRAY_GPU MRAY_GPU_INLINE
void KCAnyHit()
{
    using Primitive = typename PGroup:: template Primitive<>;
    using Hit = typename PGroup::Hit;
    using HitRecord = Record<typename PGroup::DataSoA>;
    const void* hrRaw = std::bit_cast<void*>(optixGetSbtDataPointer());
    const HitRecord& record = *reinterpret_cast<const HitRecord*>(hrRaw);

    if(record.alphaMap)
    {
        // This has alpha map check it
        const auto& alphaMap = record.alphaMap.value();
        // Get the current hit
        Hit hit = ReadHitFromAttributes<Hit>().template AsVector<Hit::Dims>();
        // Create primitive
        const uint32_t leafId = optixGetPrimitiveIndex();
        Primitive prim(TransformContextIdentity{},
                       * record.primSoA, record.dPrimKeys[leafId]);
        // Finally get uv form hit and get alpha
        Vector2 uv = prim.SurfaceParametrization(hit);
        Float alpha = alphaMap(uv).value();
        // Stochastic alpha culling
        BackupRNGState s = GetRNGStateFromPayload();
        Float xi = BackupRNG(s).NextFloat();
        if(xi > alpha)
        {
            SetRNGStateAsPayload(s);
            // This is somewhat like "return"
            // So save the changed state as payload
            // before calling
            optixIgnoreIntersection();
        }
        SetRNGStateAsPayload(s);
    }
}

// Meta Intersect Shader
template<PrimitiveGroupC PGroup, TransformGroupC TGroup>
MRAY_GPU MRAY_GPU_INLINE
void KCIntersect()
{
    ////GPUTransformIdentity ID_TRANSFORM;

    //using LeafStruct = typename PGroup::LeafData;
    //using HitStruct  = typename PGroup::HitData;
    //using HitRecord  = Record<typename PGroup::PrimitiveData,
    //                          typename PGroup::LeafData>;
    //// Construct a ray
    //float3 rP = optixGetWorldRayOrigin();
    //float3 rD = optixGetWorldRayDirection();
    //const float  tMin = optixGetRayTmin();
    //const float  tMax = optixGetRayTmax();

    //// Record Fetch
    //const HitRecord* r = (const HitRecord*)optixGetSbtDataPointer();
    //const int leafId = optixGetPrimitiveIndex();
    //// Fetch Leaf
    //const LeafStruct& gLeaf = r->gLeafs[leafId];

    //// Outputs
    //float newT;
    //HitStruct hitData;
    //bool intersects = false;
    //// Our intersects function requires a transform
    //// Why?
    ////
    //// For skinned meshes (or any primitive with a transform that cannot be applied
    //// to a ray inversely) each accelerator is fully constructed using transformed aabbs,
    ////
    //// However primitives are not transformed, thus for each shader we need the transform
    //if constexpr(PGroup::TransType == PrimTransformType::CONSTANT_LOCAL_TRANSFORM)
    //{
    //    // Use optix transform the ray to local (object) space of the primitive
    //    // and use an identity transform on the Intersection function
    //    rD = optixTransformVectorFromWorldToObjectSpace(rD);
    //    rP = optixTransformPointFromWorldToObjectSpace(rP);
    //}
    //else if constexpr(PGroup::TransType == PrimTransformType::PER_PRIMITIVE_TRANSFORM)
    //{
    //    // nvcc with clang compiles this when assert is false, so putting an impossible
    //    // statement
    //    static_assert(PGroup::TransType != PrimTransformType::PER_PRIMITIVE_TRANSFORM,
    //                 "Per primitive transform is not supported on OptiX yet");
    //    // Optix does not support virtual function calls
    //    // Just leave it as is for now
    //}
    //// nvcc with clang compiles this when assert is false, so putting an impossible
    //// statement
    //else static_assert(PGroup::TransType == PrimTransformType::PER_PRIMITIVE_TRANSFORM,
    //                   "Primitive does not have proper transform type");

    //// Construct the Register (after transformation)
    //RayReg rayRegister = RayReg(RayF(Vector3f(rD.x, rD.y, rD.z),
    //                                 Vector3f(rP.x, rP.y, rP.z)),
    //                            tMin, tMax);

    //// Since OptiX does not support virtual(indirect) function calls
    //// Call with an empty transform (this is not virtual and does nothing)
    //// <GPUTransformEmpty>
    //intersects = PGroup::IntersectsT(// Output
    //                                 newT,
    //                                 hitData,
    //                                 // I-O
    //                                 rayRegister,
    //                                 // Input
    //                                 GPUTransformEmpty(),
    //                                 gLeaf,
    //                                 (*r->gPrimData));
    //// Report the intersection
    //if(intersects) ReportIntersection<HitStruct>(newT, 0, hitData);
}

MRAY_GPU MRAY_GPU_INLINE
void KCMissOptiX()
{
    // Do Nothing
}

MRAY_GPU MRAY_GPU_INLINE
void KCRayGenOptix()
{
    // We Launch linearly
    const uint32_t launchDim = optixGetLaunchDimensions().x;
    const uint32_t launchIndex = optixGetLaunchIndex().x;
    // Should we check this ??
    if(launchIndex >= launchDim) return;

    RayIndex rIndex = params.dRayIndices[launchIndex];
    auto [ray, tMM] = RayFromGMem(params.dRays, rIndex);

    // Set the ray index (indirection) as payload
    // so we do not hit GMem for this
    SetRayIndexAsPayload(rIndex);
    // Set the RNG states as payload, any hit shaders will
    // do stochastic any hit invocation
    SetRNGStateAsPayload(params.dRNGStates[rIndex]);

    // Trace!
    optixTrace(// Accelerator
               params.baseAccelerator,
               // Ray Input
               make_float3(ray.Pos()[0], ray.Pos()[1], ray.Pos()[2]),
               make_float3(ray.Dir()[0], ray.Dir()[1], ray.Dir()[2]),
               tMM[0], tMM[1],
               0.0f,
               //
               OptixVisibilityMask(255),
               // Flags
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               // SBT
               0, 1, 0);

    // Save the state back
    params.dRNGStates[rIndex] = GetRNGStateFromPayload();
}

// Actual Definitions
WRAP_FUCTION(__raygen__OptiX, KCRayGenOptix);
WRAP_FUCTION(__miss__OptiX, KCMissOptiX);
// These function names must be equavlient to the return type "TypeName()"
// functions. We can't do language magic here unfortunately
// this needs to be maintained when a new primitive type introduced
// to the system.
//
// Default Triange
//
// Triange types do not need KCIntersect functions, since all these
// are handled by HW (Including the PER_PRIM and
// CONSTANT_LOCAL transformations)
//
WRAP_FUCTION(__closesthit__Triangle, KCClosestHit<PrimGroupTriangle>)
WRAP_FUCTION(__anyhit__Triangle, KCAnyHit<PrimGroupTriangle>)
// TODO: Add more...

WRAP_FUCTION(__closesthit__TriangleSkinned, KCClosestHit<PrimGroupSkinnedTriangle>)
WRAP_FUCTION(__anyhit__TriangleSkinned, KCAnyHit<PrimGroupSkinnedTriangle>)
